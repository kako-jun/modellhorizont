"""
Issue #5 PoC (revision 2): hard far-background replacement.

Builds on Issue #4 (mono_farmask.py): runs Depth Anything V2 to get a far mask,
then composites the original foreground over a synthetic (or supplied) sky/
horizon plate using a HARD binary mask (with a thin antialias edge only).

Design ruling (overrides Issue #8 soft-transition):
  The miniature illusion fails when the far layer is semi-transparent — you
  can still see the original wall through the sky, which kills the effect.
  So:
    far     = full sky (alpha = 1.0)
    near    = original (alpha = 0.0)
    unknown = absorbed into far by default (configurable), and the only
              softness is a 1.0-1.5 px antialias ramp at the edge.

Usage:
  uv run python3 poc/layered_replace.py path/to/img1.jpg path/to/img2.jpg \\
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


def hard_far_alpha(
    far: np.ndarray,
    unknown: np.ndarray,
    unknown_policy: str,
    antialias_sigma: float,
) -> np.ndarray:
    """Build a near-binary far_alpha in [0, 1].

    The interior of the far region is exactly 1.0; the interior of the near
    region is exactly 0.0; only a thin band at the boundary (controlled by
    ``antialias_sigma``) is intermediate, to suppress jaggies.

    ``unknown_policy``:
      - 'far'   : unknown pixels become far (default; matches the
                   miniature-illusion philosophy).
      - 'near'  : unknown pixels stay near.
      - 'split' : unknown pixels get alpha = 0.5 (legacy behavior, mostly for
                   diagnostics).
    """
    h, w = far.shape
    if unknown_policy == "far":
        binary = (far | unknown).astype(np.float32)
    elif unknown_policy == "near":
        binary = far.astype(np.float32)
    elif unknown_policy == "split":
        binary = far.astype(np.float32)
        binary[unknown & ~far] = 0.5
    else:
        raise ValueError(f"unknown unknown_policy: {unknown_policy}")

    if antialias_sigma > 0.0:
        # Tiny gaussian blur on the binary mask -> antialiased edges only.
        # Interior pixels remain 1.0 (or 0.0) because they're far from the
        # boundary; only pixels within ~3*sigma of the edge get blended.
        binary = cv2.GaussianBlur(binary, (0, 0), antialias_sigma)
    return np.clip(binary, 0.0, 1.0).astype(np.float32)


def make_auto_bg(h: int, w: int) -> np.ndarray:
    """Vertical-gradient sky plate, BGR uint8."""
    horizon = np.array([180, 200, 230], dtype=np.float32)  # warm cream
    zenith = np.array([170, 130, 90], dtype=np.float32)  # cool blue
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    col = (1.0 - t) * zenith + t * horizon
    bg = np.broadcast_to(col[:, None, :], (h, w, 3)).copy()
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 3.0, size=(h, w, 1)).astype(np.float32)
    bg = np.clip(bg + noise, 0, 255).astype(np.uint8)
    return bg


def load_bg(path: Path, h: int, w: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"could not read background image: {path}")
    ih, iw = img.shape[:2]
    s = max(h / ih, w / iw)
    nh, nw = int(round(ih * s)), int(round(iw * s))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    y0 = (nh - h) // 2
    x0 = (nw - w) // 2
    return img[y0 : y0 + h, x0 : x0 + w].copy()


def composite(fg: np.ndarray, bg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """out = alpha * bg + (1 - alpha) * fg (alpha is now ~binary)."""
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
    unknown_policy: str,
    bg_arg: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb, depth = run_depth(pipe, path)
    inv = depth.astype(np.float32)
    h, w = inv.shape

    far_thr = float(np.percentile(inv, far_pct))
    far = inv <= far_thr
    unknown = edge_unknown_band(inv, width=dilate_width, edge_pct=edge_pct) & ~far
    near = ~far & ~unknown

    far_alpha = hard_far_alpha(
        far=far,
        unknown=unknown,
        unknown_policy=unknown_policy,
        antialias_sigma=feather_sigma,
    )

    if bg_arg == "auto":
        bg = make_auto_bg(h, w)
    else:
        bg = load_bg(Path(bg_arg), h, w)

    comp = composite(rgb, bg, far_alpha)
    sxs = np.concatenate([rgb, comp], axis=1)

    stats = {
        "far_pct": float(far.mean() * 100),
        "unknown_pct": float(unknown.mean() * 100),
        "near_pct": float(near.mean() * 100),
        "alpha_mean": float(far_alpha.mean()),
        "alpha_std": float(far_alpha.std()),
    }
    print(
        f"[{tag}] shape={inv.shape} far_thr={far_thr:.1f} "
        f"unknown_policy={unknown_policy} "
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
    ap.add_argument(
        "--feather-sigma",
        type=float,
        default=1.0,
        help="antialias-edge gaussian sigma (px); 0 = no antialias, hard step",
    )
    ap.add_argument(
        "--unknown-policy",
        choices=["far", "near", "split"],
        default="near",
        help="how to treat the depth-discontinuity unknown band",
    )
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
            unknown_policy=args.unknown_policy,
            bg_arg=args.bg,
        )


if __name__ == "__main__":
    main()
