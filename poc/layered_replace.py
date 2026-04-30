"""
Issue #5 PoC: layered far-background replacement with soft alpha.

Builds on Issue #4 (mono_farmask.py): runs Depth Anything V2 to get a far mask
and an "unknown" band along depth discontinuities, then composites the original
foreground over a synthetic (or supplied) sky/horizon plate.

Key changes vs. #4:
  - tightened defaults (edge_pct 90 -> 95, dilate width 9 -> 5)  [#4 follow-up]
  - the binary unknown band becomes a *soft* alpha ramp           [#8]
  - actually composites: out = far_alpha * bg + (1 - far_alpha) * fg

Usage:
  uv run python3 poc/layered_replace.py path/to/img1.jpg path/to/img2.jpg \
    --out out_layered

Background source:
  --bg auto         (default) procedural vertical-gradient sky
  --bg path/to.jpg  external image, center-cropped to input size
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import pipeline

from mono_farmask import (
    colorize,
    edge_unknown_band,
    normalize,
    run_depth,
)


def soft_far_alpha(
    inv: np.ndarray,
    far_thr: float,
    feather_sigma: float,
) -> np.ndarray:
    """Continuous far_alpha in [0,1].

    Strategy: take a signed-distance-like field around the far/near boundary
    by gaussian-blurring a centered "above/below threshold" indicator, then
    squash. This avoids needing scipy.ndimage.distance_transform.
    """
    # Centered indicator: +1 deep in far, -1 deep in near, 0 at boundary.
    # We use the signed margin from the threshold scaled by local depth std
    # for a soft transition that responds to actual depth contrast.
    margin = (far_thr - inv) / max(inv.std(), 1.0)
    # Smooth so the alpha doesn't follow every depth speckle.
    smooth = cv2.GaussianBlur(margin, (0, 0), max(feather_sigma, 0.5))
    # Sigmoid -> [0, 1].
    alpha = 1.0 / (1.0 + np.exp(-smooth * 4.0))
    return alpha.astype(np.float32)


def make_auto_bg(h: int, w: int) -> np.ndarray:
    """Vertical-gradient sky plate, BGR uint8."""
    # warm horizon (bottom) -> cool zenith (top), in BGR
    horizon = np.array([180, 200, 230], dtype=np.float32)  # warm cream
    zenith = np.array([170, 130, 90], dtype=np.float32)  # cool blue
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]  # 0=top, 1=bottom
    col = (1.0 - t) * zenith + t * horizon  # h x 3
    bg = np.broadcast_to(col[:, None, :], (h, w, 3)).copy()
    # subtle noise for non-flat feel
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 3.0, size=(h, w, 1)).astype(np.float32)
    bg = np.clip(bg + noise, 0, 255).astype(np.uint8)
    return bg


def load_bg(path: Path, h: int, w: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"could not read background image: {path}")
    ih, iw = img.shape[:2]
    # scale so it covers (h,w), then center-crop
    s = max(h / ih, w / iw)
    nh, nw = int(round(ih * s)), int(round(iw * s))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    y0 = (nh - h) // 2
    x0 = (nw - w) // 2
    return img[y0 : y0 + h, x0 : x0 + w].copy()


def composite(fg: np.ndarray, bg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None]
    out = a * bg.astype(np.float32) + (1.0 - a) * fg.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def process(
    pipe,
    path: Path,
    out_dir: Path,
    tag: str,
    far_pct: float,
    edge_pct: float,
    dilate_width: int,
    feather_sigma: float,
    bg_arg: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb, depth = run_depth(pipe, path)
    inv = depth.astype(np.float32)
    h, w = inv.shape

    # Hard masks (for diagnostics, parity with #4 outputs)
    far_thr = float(np.percentile(inv, far_pct))
    far = inv <= far_thr
    unknown = edge_unknown_band(inv, width=dilate_width, edge_pct=edge_pct) & ~far
    near = ~far & ~unknown

    # Soft alpha (Issue #8)
    far_alpha = soft_far_alpha(inv, far_thr=far_thr, feather_sigma=feather_sigma)

    # Background
    if bg_arg == "auto":
        bg = make_auto_bg(h, w)
    else:
        bg = load_bg(Path(bg_arg), h, w)

    comp = composite(rgb, bg, far_alpha)

    # Side by side
    sxs = np.concatenate([rgb, comp], axis=1)

    # Stats
    stats = {
        "far_pct": float(far.mean() * 100),
        "unknown_pct": float(unknown.mean() * 100),
        "near_pct": float(near.mean() * 100),
        "alpha_mean": float(far_alpha.mean()),
        "alpha_std": float(far_alpha.std()),
    }
    print(
        f"[{tag}] shape={inv.shape} far_thr={far_thr:.1f} "
        f"far={stats['far_pct']:.2f}% "
        f"unknown={stats['unknown_pct']:.2f}% "
        f"near={stats['near_pct']:.2f}% "
        f"alpha_mean={stats['alpha_mean']:.3f} "
        f"alpha_std={stats['alpha_std']:.3f}"
    )

    cv2.imwrite(str(out_dir / f"{tag}_00_input.png"), rgb)
    cv2.imwrite(str(out_dir / f"{tag}_01_depth.png"), colorize(normalize(inv)))
    cv2.imwrite(str(out_dir / f"{tag}_02_far_mask.png"), far.astype(np.uint8) * 255)
    cv2.imwrite(
        str(out_dir / f"{tag}_03_unknown_mask.png"), unknown.astype(np.uint8) * 255
    )
    cv2.imwrite(
        str(out_dir / f"{tag}_04_far_alpha.png"),
        (np.clip(far_alpha, 0, 1) * 255).astype(np.uint8),
    )
    cv2.imwrite(str(out_dir / f"{tag}_05_background.png"), bg)
    cv2.imwrite(str(out_dir / f"{tag}_06_composite.png"), comp)
    cv2.imwrite(str(out_dir / f"{tag}_07_side_by_side.png"), sxs)

    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--far-pct", type=float, default=15.0)
    ap.add_argument("--edge-pct", type=float, default=95.0)
    ap.add_argument("--dilate-width", type=int, default=5)
    ap.add_argument("--feather-sigma", type=float, default=12.0)
    ap.add_argument("--bg", default="auto", help="'auto' or path to background image")
    args = ap.parse_args()

    device = 0 if torch.cuda.is_available() else -1
    print(f"[init] model={args.model} device={'cuda' if device == 0 else 'cpu'}")
    pipe = pipeline(task="depth-estimation", model=args.model, device=device)

    for i, p in enumerate(args.images):
        process(
            pipe,
            p,
            args.out,
            tag=f"img{i}",
            far_pct=args.far_pct,
            edge_pct=args.edge_pct,
            dilate_width=args.dilate_width,
            feather_sigma=args.feather_sigma,
            bg_arg=args.bg,
        )


if __name__ == "__main__":
    main()
