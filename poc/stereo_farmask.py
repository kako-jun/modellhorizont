"""
Issue #1 PoC: two-image stereo far-mask for static subjects.

Input:  two hand-held shots of the same static scene with small lateral shift.
Output: rectified L/R, disparity heatmap, far mask, unknown mask.

Strategy:
  1. Resize for tractable matching.
  2. ORB + RANSAC fundamental matrix -> uncalibrated stereoRectify.
  3. StereoSGBM left + right matchers -> WLS-filtered disparity.
  4. far  = disparity below low percentile of confident pixels.
     unknown = WLS confidence below threshold OR disparity invalid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def load_resized(path: Path, max_side: int) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def rectify_uncalibrated(imgL: np.ndarray, imgR: np.ndarray):
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=8000)
    kpL, desL = orb.detectAndCompute(grayL, None)
    kpR, desR = orb.detectAndCompute(grayR, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(desL, desR), key=lambda m: m.distance)[:1500]
    if len(matches) < 50:
        raise RuntimeError(f"too few matches: {len(matches)}")

    ptsL = np.float32([kpL[m.queryIdx].pt for m in matches])
    ptsR = np.float32([kpR[m.trainIdx].pt for m in matches])

    F, mask = cv2.findFundamentalMat(ptsL, ptsR, cv2.FM_RANSAC, 1.5, 0.999)
    inl = mask.ravel().astype(bool)
    ptsL_in, ptsR_in = ptsL[inl], ptsR[inl]
    print(f"[rectify] matches={len(matches)} inliers={inl.sum()}")

    h, w = grayL.shape
    ok, H1, H2 = cv2.stereoRectifyUncalibrated(
        ptsL_in.reshape(-1, 1, 2), ptsR_in.reshape(-1, 1, 2), F, (w, h)
    )
    if not ok:
        raise RuntimeError("stereoRectifyUncalibrated failed")

    rectL = cv2.warpPerspective(imgL, H1, (w, h))
    rectR = cv2.warpPerspective(imgR, H2, (w, h))
    return rectL, rectR


def compute_disparity(rectL: np.ndarray, rectR: np.ndarray):
    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    # Disparity range: a hand-held shift on a low-angle desk shot can give
    # large near-foreground disparity. Keep generous range, small block.
    min_disp = 0
    num_disp = 16 * 12  # 192
    block = 5

    left = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block,
        P1=8 * 3 * block * block,
        P2=32 * 3 * block * block,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    right = cv2.ximgproc.createRightMatcher(left)

    dispL = left.compute(grayL, grayR)
    dispR = right.compute(grayR, grayL)

    wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left)
    wls.setLambda(8000.0)
    wls.setSigmaColor(1.5)
    filtered = wls.filter(dispL, grayL, disparity_map_right=dispR)
    conf = wls.getConfidenceMap()  # 0..255

    return filtered.astype(np.float32) / 16.0, conf


def colorize_disp(disp: np.ndarray, valid: np.ndarray) -> np.ndarray:
    vis = np.zeros_like(disp)
    if valid.any():
        v = disp[valid]
        lo, hi = np.percentile(v, [2, 98])
        vis[valid] = np.clip((disp[valid] - lo) / max(hi - lo, 1e-6), 0, 1)
    vis8 = (vis * 255).astype(np.uint8)
    color = cv2.applyColorMap(vis8, cv2.COLORMAP_TURBO)
    color[~valid] = (0, 0, 0)
    return color


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("left", type=Path)
    ap.add_argument("right", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-side", type=int, default=1400)
    ap.add_argument("--far-pct", type=float, default=15.0,
                    help="pixels with confident disparity below this percentile = far")
    ap.add_argument("--conf-thr", type=int, default=128,
                    help="WLS confidence below this = unknown")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    imgL = load_resized(args.left, args.max_side)
    imgR = load_resized(args.right, args.max_side)
    cv2.imwrite(str(args.out / "00_left.png"), imgL)
    cv2.imwrite(str(args.out / "00_right.png"), imgR)

    rectL, rectR = rectify_uncalibrated(imgL, imgR)
    cv2.imwrite(str(args.out / "01_rect_left.png"), rectL)
    cv2.imwrite(str(args.out / "01_rect_right.png"), rectR)

    # epipolar overlay for sanity
    overlay = cv2.addWeighted(rectL, 0.5, rectR, 0.5, 0)
    for y in range(0, overlay.shape[0], 40):
        cv2.line(overlay, (0, y), (overlay.shape[1], y), (0, 255, 0), 1)
    cv2.imwrite(str(args.out / "02_rect_overlay.png"), overlay)

    disp, conf = compute_disparity(rectL, rectR)

    valid = (disp > 0) & np.isfinite(disp)
    confident = valid & (conf >= args.conf_thr)
    print(f"[disp] valid={valid.mean():.2%} confident={confident.mean():.2%}")

    cv2.imwrite(str(args.out / "03_disparity.png"), colorize_disp(disp, confident))
    cv2.imwrite(str(args.out / "03_confidence.png"), conf.astype(np.uint8))

    # far = lowest disparity among confident pixels
    if confident.any():
        thr = np.percentile(disp[confident], args.far_pct)
        far = confident & (disp <= thr)
        print(f"[far] threshold disp<={thr:.2f}  -> {far.mean():.2%} of frame")
    else:
        far = np.zeros_like(valid)

    unknown = ~confident  # not confident enough to call near or far

    overlay = rectL.copy()
    overlay[far] = (0.4 * overlay[far] + 0.6 * np.array([255, 80, 80])).astype(np.uint8)
    overlay[unknown & ~far] = (
        0.6 * overlay[unknown & ~far] + 0.4 * np.array([80, 80, 80])
    ).astype(np.uint8)
    cv2.imwrite(str(args.out / "04_far_overlay.png"), overlay)

    far_mask = (far.astype(np.uint8) * 255)
    cv2.imwrite(str(args.out / "04_far_mask.png"), far_mask)
    cv2.imwrite(str(args.out / "04_unknown_mask.png"), (unknown.astype(np.uint8) * 255))

    print(f"[done] wrote debug images to {args.out}")


if __name__ == "__main__":
    main()
