"""
Issue #13 PoC: SAM2-snapped far mask.

Pipeline:
  1. Depth Anything V2 (Small) -> relative inverse depth (large = near).
  2. SAM2 (HF transformers) auto-masks via a regular grid of point prompts.
  3. For each segment, compute median inv-depth.
  4. Far = bottom-N% segments by depth median (i.e. smallest inv-depth = furthest).
  5. Absorb tiny segments (< area_pct) into the dominant neighbour.
  6. Emit comparison images: depth-only vs SAM2-snapped.

CPU-only (Mac M-series / Intel). Uses facebook/sam2.1-hiera-tiny.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor, pipeline


# ---------- Depth ----------

def run_depth(pipe, img_pil: Image.Image) -> np.ndarray:
    out = pipe(img_pil)
    return np.array(out["depth"], dtype=np.float32)


def normalize(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, [2, 98])
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)


def colorize(depth01: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap((depth01 * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


# ---------- SAM2 auto-mask via point grid ----------

def _make_point_grid(h: int, w: int, n_per_side: int) -> list[tuple[int, int]]:
    ys = np.linspace(0, h - 1, n_per_side + 2)[1:-1]
    xs = np.linspace(0, w - 1, n_per_side + 2)[1:-1]
    pts = []
    for y in ys:
        for x in xs:
            pts.append((int(round(x)), int(round(y))))
    return pts


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def sam2_auto_masks(
    model: Sam2Model,
    processor: Sam2Processor,
    img_pil: Image.Image,
    n_per_side: int = 16,
    batch_points: int = 32,
    pred_iou_thresh: float = 0.7,
    stability_score_thresh: float = 0.85,
    iou_dedup_thresh: float = 0.7,
    device: str = "cpu",
) -> list[np.ndarray]:
    """Run SAM2 with a regular grid of single-point prompts.

    Returns a deduped list of binary masks (HxW bool).
    """
    w, h = img_pil.size
    grid = _make_point_grid(h, w, n_per_side)
    print(f"[sam2] grid {n_per_side}x{n_per_side} -> {len(grid)} points, "
          f"image {w}x{h}")

    # Pre-encode the image once (returns list[Tensor] of multi-scale feats).
    base_inputs = processor(images=img_pil, return_tensors="pt").to(device)
    with torch.inference_mode():
        image_embeddings = model.get_image_embeddings(base_inputs["pixel_values"])
    print(f"[sam2] cached image_embeddings: "
          f"{type(image_embeddings).__name__} "
          f"len={len(image_embeddings) if hasattr(image_embeddings, '__len__') else '?'}")

    all_masks: list[tuple[float, np.ndarray]] = []  # (score, mask)

    t0 = time.time()
    for batch_start in range(0, len(grid), batch_points):
        batch = grid[batch_start: batch_start + batch_points]
        # Build input_points: list[batch=1][num_points][points_per_pred=1][xy]
        input_points = [[[[float(x), float(y)]] for (x, y) in batch]]
        input_labels = [[[1]] * len(batch)]

        proc = processor(
            images=img_pil,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            out = model(
                image_embeddings=image_embeddings,
                input_points=proc["input_points"],
                input_labels=proc["input_labels"],
                multimask_output=True,
            )

        # out.pred_masks: (batch=1, num_points, num_predictions=3, H', W')
        # out.iou_scores:  (batch=1, num_points, num_predictions=3)
        pred_masks = out.pred_masks  # logits
        iou_scores = out.iou_scores

        # Upsample masks back to original image size & binarize.
        masks_list = processor.post_process_masks(
            pred_masks.cpu(),
            original_sizes=proc["original_sizes"].cpu(),
            binarize=True,
        )
        # masks_list: list of length batch=1, each tensor (num_points, 3, H, W)
        masks_b = masks_list[0]
        scores_b = iou_scores[0].cpu().numpy()  # (num_points, 3)

        for pi in range(masks_b.shape[0]):
            for mi in range(masks_b.shape[1]):
                s = float(scores_b[pi, mi])
                if s < pred_iou_thresh:
                    continue
                m = masks_b[pi, mi].numpy().astype(bool)
                area = m.sum()
                if area < 50:
                    continue
                # crude stability: ratio between erode and dilate area.
                m_u8 = m.astype(np.uint8)
                eroded = cv2.erode(m_u8, np.ones((3, 3), np.uint8))
                dilated = cv2.dilate(m_u8, np.ones((3, 3), np.uint8))
                ed = eroded.sum()
                dd = dilated.sum()
                stab = ed / dd if dd > 0 else 0.0
                if stab < stability_score_thresh:
                    continue
                all_masks.append((s, m))

        done = min(batch_start + batch_points, len(grid))
        elapsed = time.time() - t0
        rate = done / max(elapsed, 1e-3)
        remaining = (len(grid) - done) / max(rate, 1e-3)
        print(f"[sam2] {done}/{len(grid)} pts, kept={len(all_masks)} "
              f"elapsed={elapsed:.1f}s eta={remaining:.1f}s")

    # Dedup by IoU, keeping highest-score first.
    all_masks.sort(key=lambda t: -t[0])
    kept: list[np.ndarray] = []
    for _, m in all_masks:
        dup = False
        for k in kept:
            if _mask_iou(m, k) > iou_dedup_thresh:
                dup = True
                break
        if not dup:
            kept.append(m)
    print(f"[sam2] deduped {len(all_masks)} -> {len(kept)}")
    return kept


# ---------- Segmentation visualization & far classification ----------

def visualize_segments(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    rng = np.random.default_rng(0)
    # Paint largest first so small segments overwrite.
    order = sorted(range(len(masks)), key=lambda i: -masks[i].sum())
    for i in order:
        color = rng.integers(40, 256, size=3, dtype=np.int32).astype(np.uint8)
        canvas[masks[i]] = color
    return canvas


def build_segment_map(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Make an int label map (-1 for background). Larger segs first; later
    masks paint on top so small segs win at overlaps."""
    h, w = shape
    label = -np.ones((h, w), dtype=np.int32)
    order = sorted(range(len(masks)), key=lambda i: -masks[i].sum())
    new_id = 0
    id_remap = {}
    for i in order:
        id_remap[i] = new_id
        new_id += 1
    # paint in reverse (smallest last so smaller wins)
    for i in order:
        label[masks[i]] = id_remap[i]
    return label


def absorb_tiny_segments(
    label: np.ndarray, min_area_frac: float = 0.001
) -> np.ndarray:
    h, w = label.shape
    min_area = max(1, int(min_area_frac * h * w))
    out = label.copy()
    changed = True
    iters = 0
    while changed and iters < 5:
        changed = False
        iters += 1
        ids, counts = np.unique(out, return_counts=True)
        small_ids = [i for i, c in zip(ids, counts) if i >= 0 and c < min_area]
        if not small_ids:
            break
        for sid in small_ids:
            ys, xs = np.where(out == sid)
            if len(ys) == 0:
                continue
            # gather neighbour ids via dilation of the small region
            mask = (out == sid).astype(np.uint8)
            dil = cv2.dilate(mask, np.ones((3, 3), np.uint8))
            border = (dil == 1) & (mask == 0)
            neighbour_ids = out[border]
            neighbour_ids = neighbour_ids[neighbour_ids != sid]
            if len(neighbour_ids) == 0:
                continue
            vals, cnts = np.unique(neighbour_ids, return_counts=True)
            target = vals[np.argmax(cnts)]
            out[out == sid] = target
            changed = True
    # also fill background pixels (label == -1) with nearest segment
    if (out == -1).any():
        # nearest-neighbour fill via distance transform
        bg = (out == -1).astype(np.uint8)
        # For each bg pixel, find nearest non-bg.
        _, labels = cv2.distanceTransformWithLabels(
            bg, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
        )
        # Build lookup: index -> label value of nearest non-bg pixel
        nonbg_ys, nonbg_xs = np.where(out != -1)
        # `labels` returns CCL labels of the zero-cells (i.e. non-bg). Each
        # zero-cell becomes its own component (PIXEL mode). The label at a bg
        # pixel is the component id of the nearest zero-cell.
        # Build per-component representative (y,x):
        zero_label_at = labels[out != -1]  # label id per non-bg pixel
        # Group: for each unique label id, take the first non-bg pixel.
        rep = {}
        for lid, y, x in zip(zero_label_at, nonbg_ys, nonbg_xs):
            if int(lid) not in rep:
                rep[int(lid)] = (y, x)
        # Assign bg pixels.
        bg_ys, bg_xs = np.where(out == -1)
        for y, x in zip(bg_ys, bg_xs):
            lid = int(labels[y, x])
            if lid in rep:
                ry, rx = rep[lid]
                out[y, x] = out[ry, rx]
    return out


def far_mask_from_segments(
    label: np.ndarray, inv_depth: np.ndarray, far_pct: float
) -> np.ndarray:
    """Pick segments whose median inv-depth is below the per-segment-median
    distribution's far_pct percentile. (i.e. furthest far_pct% of segments.)"""
    ids = np.unique(label)
    ids = ids[ids >= 0]
    medians = []
    sizes = []
    for sid in ids:
        m = label == sid
        medians.append(float(np.median(inv_depth[m])))
        sizes.append(int(m.sum()))
    medians = np.array(medians, dtype=np.float32)
    sizes = np.array(sizes, dtype=np.int64)

    # Weight by area so a single huge segment doesn't get drowned out by
    # many tiny ones at the same depth.
    order = np.argsort(medians)  # ascending: smallest inv-depth (furthest) first
    cum = np.cumsum(sizes[order])
    total = cum[-1]
    cutoff = far_pct / 100.0 * total
    far_ids = set()
    for k, idx in enumerate(order):
        if cum[k] <= cutoff:
            far_ids.add(int(ids[idx]))
        else:
            far_ids.add(int(ids[idx]))
            break  # include the boundary segment too
    far = np.zeros_like(label, dtype=bool)
    for sid in far_ids:
        far |= label == sid
    return far


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--sam2-model", default="facebook/sam2.1-hiera-tiny")
    ap.add_argument("--far-pct", type=float, default=40.0)
    ap.add_argument("--n-per-side", type=int, default=16)
    ap.add_argument("--max-side", type=int, default=1024,
                    help="Resize image so that max(H,W) <= this (CPU speedup)")
    ap.add_argument("--min-area-frac", type=float, default=0.001)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    device = "cpu"
    print(f"[init] device={device}")

    # Load image, resize for speed
    img_pil = Image.open(args.image).convert("RGB")
    w0, h0 = img_pil.size
    scale = min(1.0, args.max_side / max(w0, h0))
    if scale < 1.0:
        nw, nh = int(round(w0 * scale)), int(round(h0 * scale))
        img_pil = img_pil.resize((nw, nh), Image.LANCZOS)
        print(f"[init] resized {w0}x{h0} -> {nw}x{nh} (scale={scale:.3f})")
    rgb = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    h, w = rgb.shape[:2]

    cv2.imwrite(str(args.out / "00_input.png"), rgb)

    # ---- Depth ----
    print(f"[depth] loading {args.depth_model}")
    depth_pipe = pipeline(task="depth-estimation", model=args.depth_model, device=-1)
    t0 = time.time()
    inv = run_depth(depth_pipe, img_pil)
    print(f"[depth] done in {time.time()-t0:.1f}s, shape={inv.shape}")
    if inv.shape[:2] != (h, w):
        inv = cv2.resize(inv, (w, h), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(args.out / "01_depth.png"), colorize(normalize(inv)))

    # Depth-only baseline far mask: bottom far_pct% of pixels by inv-depth
    thr = np.percentile(inv, args.far_pct)
    far_depth_only = (inv <= thr).astype(np.uint8) * 255
    cv2.imwrite(str(args.out / "03_far_mask_depth.png"), far_depth_only)

    # ---- SAM2 ----
    print(f"[sam2] loading {args.sam2_model}")
    t0 = time.time()
    processor = Sam2Processor.from_pretrained(args.sam2_model)
    model = Sam2Model.from_pretrained(args.sam2_model).to(device).eval()
    print(f"[sam2] loaded in {time.time()-t0:.1f}s")

    t0 = time.time()
    masks = sam2_auto_masks(
        model, processor, img_pil,
        n_per_side=args.n_per_side,
        batch_points=8,
        device=device,
    )
    sam2_secs = time.time() - t0
    print(f"[sam2] auto-mask {sam2_secs:.1f}s, {len(masks)} segments")

    seg_vis = visualize_segments(masks, (h, w))
    cv2.imwrite(str(args.out / "02_segments.png"), seg_vis)

    label = build_segment_map(masks, (h, w))
    label = absorb_tiny_segments(label, min_area_frac=args.min_area_frac)

    far_sam2 = far_mask_from_segments(label, inv, args.far_pct)
    far_sam2_u8 = (far_sam2.astype(np.uint8) * 255)
    cv2.imwrite(str(args.out / "04_far_mask_sam2.png"), far_sam2_u8)

    # Side-by-side: input | depth-only mask overlay | sam2 mask overlay
    def overlay(rgb_in: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
        out = rgb_in.copy()
        out[mask_bool] = (0.4 * out[mask_bool] + 0.6 * np.array([255, 80, 80])).astype(np.uint8)
        return out

    depth_only_bool = far_depth_only > 127
    panel = np.concatenate([
        rgb,
        overlay(rgb, depth_only_bool),
        overlay(rgb, far_sam2),
    ], axis=1)
    # add labels
    cv2.putText(panel, "input", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(panel, "depth-only far", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(panel, "SAM2 snapped far", (2*w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imwrite(str(args.out / "05_side_by_side.png"), panel)

    print(f"[done] outputs in {args.out}")
    print(f"[done] depth-only far={depth_only_bool.mean():.2%} "
          f"sam2 far={far_sam2.mean():.2%}")


if __name__ == "__main__":
    main()
