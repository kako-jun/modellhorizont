"""
Issue #5 follow-up: Gradio live-tuning tool for the layered far-replacement.

Loads ONE image, runs Depth Anything V2 ONCE on startup, caches the depth
tensor in memory, and exposes sliders so kako-jun can tune parameters by feel
without re-running depth on every change.

Usage:
  uv run python3 poc/gradio_tune.py path/to/image.jpg
  uv run python3 poc/gradio_tune.py path/to/image.jpg --port 7861

Sliders update three image panels (input, far mask, composite). Parameter
snapshots are printed to stdout when sliders move so good values can be
captured by hand.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from transformers import pipeline

from layered_replace import (
    composite,
    hard_far_alpha,
    load_bg,
    make_auto_bg,
)
from mono_farmask import edge_unknown_band, run_depth


# ---------- background presets ----------


def make_warm_sky(h: int, w: int) -> np.ndarray:
    horizon = np.array([180, 200, 230], dtype=np.float32)  # warm cream
    zenith = np.array([170, 130, 90], dtype=np.float32)  # cool blue
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    col = (1.0 - t) * zenith + t * horizon
    bg = np.broadcast_to(col[:, None, :], (h, w, 3)).copy()
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 3.0, size=(h, w, 1)).astype(np.float32)
    return np.clip(bg + noise, 0, 255).astype(np.uint8)


def make_cool_sky(h: int, w: int) -> np.ndarray:
    horizon = np.array([220, 200, 170], dtype=np.float32)  # pale haze (BGR)
    zenith = np.array([190, 110, 60], dtype=np.float32)  # deep blue
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    col = (1.0 - t) * zenith + t * horizon
    bg = np.broadcast_to(col[:, None, :], (h, w, 3)).copy()
    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, 2.5, size=(h, w, 1)).astype(np.float32)
    return np.clip(bg + noise, 0, 255).astype(np.uint8)


def make_solid_blue(h: int, w: int) -> np.ndarray:
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[:] = (200, 130, 70)  # BGR
    return bg


BG_PRESETS = {
    "auto-warm-sky": make_warm_sky,
    "auto-cool-sky": make_cool_sky,
    "solid blue": make_solid_blue,
}


# ---------- cached state ----------


class State:
    rgb: np.ndarray  # BGR uint8 H x W x 3 (input)
    inv: np.ndarray  # depth, float32 H x W
    h: int
    w: int


STATE = State()


# ---------- live update ----------


def to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def update(
    far_pct: float,
    edge_pct: float,
    dilate_width: int,
    antialias_sigma: float,
    unknown_policy: str,
    bg_choice: str,
    bg_upload,
):
    inv = STATE.inv
    rgb = STATE.rgb
    h, w = STATE.h, STATE.w

    far_thr = float(np.percentile(inv, far_pct))
    far = inv <= far_thr
    unknown = (
        edge_unknown_band(inv, width=int(dilate_width), edge_pct=edge_pct) & ~far
    )

    far_alpha = hard_far_alpha(
        far=far,
        unknown=unknown,
        unknown_policy=unknown_policy,
        antialias_sigma=float(antialias_sigma),
    )

    if bg_upload is not None:
        bg = load_bg(Path(bg_upload), h, w)
    elif bg_choice in BG_PRESETS:
        bg = BG_PRESETS[bg_choice](h, w)
    else:
        bg = make_warm_sky(h, w)

    comp = composite(rgb, bg, far_alpha)
    far_mask_img = (np.clip(far_alpha, 0, 1) * 255).astype(np.uint8)
    far_mask_rgb = cv2.cvtColor(far_mask_img, cv2.COLOR_GRAY2RGB)

    snap = (
        f"[snap] far_pct={far_pct:.1f} edge_pct={edge_pct:.1f} "
        f"dilate={int(dilate_width)} antialias={antialias_sigma:.2f} "
        f"unknown={unknown_policy} bg={bg_choice if bg_upload is None else 'upload'} "
        f"-> far%={far.mean() * 100:.2f} unk%={unknown.mean() * 100:.2f} "
        f"alpha_mean={far_alpha.mean():.3f}"
    )
    print(snap, flush=True)

    return to_rgb(rgb), far_mask_rgb, to_rgb(comp), snap


def build_ui():
    with gr.Blocks(title="modellhorizont live tune") as demo:
        gr.Markdown("# modellhorizont — live tune (Issue #5)")
        with gr.Row():
            with gr.Column(scale=1):
                far_pct = gr.Slider(0, 50, value=15, step=0.5, label="far_pct")
                edge_pct = gr.Slider(80, 99, value=95, step=0.5, label="edge_pct")
                dilate_width = gr.Slider(
                    1, 15, value=5, step=2, label="dilate_width"
                )
                antialias_sigma = gr.Slider(
                    0, 5, value=1.0, step=0.1, label="antialias_sigma"
                )
                unknown_policy = gr.Radio(
                    ["far", "near", "split"],
                    value="far",
                    label="unknown_policy",
                )
                bg_choice = gr.Dropdown(
                    list(BG_PRESETS.keys()) + ["upload"],
                    value="auto-warm-sky",
                    label="background",
                )
                bg_upload = gr.File(
                    label="background upload (optional, used when set)",
                    file_types=["image"],
                    type="filepath",
                )
                snap_text = gr.Textbox(label="last snapshot", interactive=False)
            with gr.Column(scale=2):
                input_view = gr.Image(label="input", interactive=False)
                far_view = gr.Image(label="far_alpha (≈ binary)", interactive=False)
                comp_view = gr.Image(label="composite", interactive=False)

        inputs = [
            far_pct,
            edge_pct,
            dilate_width,
            antialias_sigma,
            unknown_policy,
            bg_choice,
            bg_upload,
        ]
        outputs = [input_view, far_view, comp_view, snap_text]

        for ctrl in inputs:
            ctrl.change(update, inputs=inputs, outputs=outputs)

        demo.load(update, inputs=inputs, outputs=outputs)

    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    device = 0 if torch.cuda.is_available() else -1
    print(f"[init] model={args.model} device={'cuda' if device == 0 else 'cpu'}")
    pipe = pipeline(task="depth-estimation", model=args.model, device=device)

    print(f"[init] running depth on {args.image} (one-shot)…")
    rgb, depth = run_depth(pipe, args.image)
    STATE.rgb = rgb
    STATE.inv = depth.astype(np.float32)
    STATE.h, STATE.w = STATE.inv.shape
    print(f"[init] depth cached: shape={STATE.inv.shape}")

    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=args.port, inbrowser=False)


if __name__ == "__main__":
    main()
