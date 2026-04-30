"""
Issue #4 PoC: monocular far-mask via Depth Anything v2.

Input:  one (or two) hand-held shots of a static subject.
Output: depth heatmap, far mask via depth percentile, edge-derived unknown band.

Strategy (MVP):
  1. Run Depth Anything V2 (Small first) -> relative inverse depth.
  2. far  = pixels above high inverse-depth percentile? -> NO. Inverse depth is
           large for *near*. So far = inverse depth below low percentile.
  3. unknown = thick band along strong depth gradients (boundary uncertainty),
               consistent with #8.
  4. If two images are given, run both and report the depth percentile
     stability (sanity check that monocular is not flipping near<->far between
     shots).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline


def run_depth(pipe, path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = Image.open(path).convert("RGB")
    out = pipe(img)
    depth = np.array(out["depth"], dtype=np.float32)  # uint8 image cast to float
    rgb = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return rgb, depth


def normalize(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, [2, 98])
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)


def colorize(depth01: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap((depth01 * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def edge_unknown_band(
    depth: np.ndarray, width: int = 5, edge_pct: float = 95.0
) -> np.ndarray:
    # #4 follow-up: tightened from edge-pct 90 -> 95, dilate width 9 -> 5
    # so the "unknown" band along depth discontinuities is thinner.
    d = cv2.GaussianBlur(depth, (0, 0), 1.5)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    thr = np.percentile(mag, edge_pct)
    edges = (mag >= thr).astype(np.uint8)
    band = cv2.dilate(edges, np.ones((width, width), np.uint8))
    return band.astype(bool)


def process(
    pipe,
    path: Path,
    out_dir: Path,
    tag: str,
    far_pct: float,
    edge_pct: float = 95.0,
    dilate_width: int = 5,
):
    rgb, depth = run_depth(pipe, path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Depth Anything v2 returns *inverse* depth via the HF pipeline:
    # large value = near, small value = far. Confirmed by colormap convention.
    inv = depth.astype(np.float32)
    cv2.imwrite(str(out_dir / f"{tag}_00_input.png"), rgb)
    cv2.imwrite(str(out_dir / f"{tag}_01_depth.png"), colorize(normalize(inv)))

    # far = lowest inv-depth (most distant)
    thr = np.percentile(inv, far_pct)
    far = inv <= thr

    unknown = edge_unknown_band(inv, width=dilate_width, edge_pct=edge_pct) & ~far
    near = ~far & ~unknown

    print(
        f"[{tag}] depth shape={inv.shape} "
        f"far<={thr:.1f} -> {far.mean():.2%} "
        f"unknown={unknown.mean():.2%} near={near.mean():.2%}"
    )

    overlay = rgb.copy()
    overlay[far] = (0.4 * overlay[far] + 0.6 * np.array([255, 80, 80])).astype(np.uint8)
    overlay[unknown] = (
        0.5 * overlay[unknown] + 0.5 * np.array([80, 80, 80])
    ).astype(np.uint8)
    cv2.imwrite(str(out_dir / f"{tag}_02_far_overlay.png"), overlay)
    cv2.imwrite(str(out_dir / f"{tag}_03_far_mask.png"), (far.astype(np.uint8) * 255))
    cv2.imwrite(str(out_dir / f"{tag}_03_unknown_mask.png"), (unknown.astype(np.uint8) * 255))

    return inv, far


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--far-pct", type=float, default=15.0)
    ap.add_argument("--edge-pct", type=float, default=95.0)
    ap.add_argument("--dilate-width", type=int, default=5)
    args = ap.parse_args()

    device = 0 if torch.cuda.is_available() else -1
    print(f"[init] model={args.model} device={'cuda' if device == 0 else 'cpu'}")
    pipe = pipeline(task="depth-estimation", model=args.model, device=device)

    invs = []
    for i, p in enumerate(args.images):
        inv, _ = process(
            pipe,
            p,
            args.out,
            tag=f"img{i}",
            far_pct=args.far_pct,
            edge_pct=args.edge_pct,
            dilate_width=args.dilate_width,
        )
        invs.append(inv)

    if len(invs) == 2 and invs[0].shape == invs[1].shape:
        diff = np.abs(invs[0] - invs[1])
        print(f"[stability] mean abs depth diff between shots: {diff.mean():.2f}")
        cv2.imwrite(str(args.out / "stability_diff.png"),
                    colorize(normalize(diff)))


if __name__ == "__main__":
    main()
