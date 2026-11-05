"""Turn existing text / bubble detections into bubble masks with plain OpenCV.

Motivation: the 768 detection-only RF-DETR build returns boxes but no masks.
Once a text box or a bubble box is known, the bubble outline supplies the rest,
so classic CV should recover the bubble mask without a segmentation head.

Two strategies, chosen by the seed's class:

  text box  -> find the *hole* contour of the ink layer that encloses the text
               cluster.  A bubble outline is a closed black loop, so its inner
               contour is exactly the bubble interior (text included).  This is
               immune to white *outside* the bubble leaking into the ROI.  A
               small closing pass first seals the spot where a scalloped
               bubble edge meets nearby art, which would otherwise merge the
               bubble with the white area outside it.
  bubble box -> prefer the ink hole whose bounding box matches the detection
               box, which follows the drawn outline and never spills into the
               white page ground.  Only when no outline closes into a matching
               hole (e.g. a bubble tangent to a panel border) fall back to the
               white component at the box centre, clipped by the box.

The hole search runs on several ink-closing levels because outlines with gaps
only close up with a larger kernel; each seed picks its best match across them.

Box sources for this demo are the saved model masks (each blob -> a box), which
stand in for the detection-only model's output.  --boxes-json replaces them with
real detections: [{"box":[x0,y0,x1,y1],"kind":"text|bubble"}, ...]
"""
import argparse
import json

import cv2
import numpy as np


def boxes_from_mask(mask_path, min_area, cluster_px=0):
    """One box per blob; cluster_px>0 merges glyphs into text lines/blocks.

    With clustering, min_area applies to the *original* pixels in a cluster,
    not the dilated blob, so a single speck cannot become a text seed.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    raw = (mask > 127).astype(np.uint8)
    m = raw
    if cluster_px > 0:
        m = cv2.dilate(raw, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * cluster_px + 1, 2 * cluster_px + 1)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i, (x, y, bw, bh, area) in enumerate(stats[1:], start=1):
        if cluster_px > 0:
            area = int(np.count_nonzero(raw[y:y + bh, x:x + bw] &
                                        (labels[y:y + bh, x:x + bw] == i)))
        if area >= min_area:
            out.append((x, y, x + bw, y + bh))
    return out


def fill_holes(mask):
    """Fill enclosed black regions (glyph strokes) of a white component."""
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = padded.copy()
    cv2.floodFill(ff, np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8),
                  (0, 0), 255)
    holes = cv2.bitwise_not(ff)[1:-1, 1:-1]
    return cv2.bitwise_or(mask, holes)


def find_bubble_holes(ink, min_area):
    """Hole contours of the ink layer, i.e. candidate bubble interiors."""
    contours, hier = cv2.findContours(ink, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    holes = []
    for ci, hi in enumerate(hier[0]):
        if hi[3] == -1:                     # -1 = outer boundary, not a hole
            continue
        area = cv2.contourArea(contours[ci])
        if area >= min_area:
            holes.append((ci, area, cv2.boundingRect(contours[ci])))
    return contours, holes


def smooth_shrink(mask, shrink):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if shrink > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * shrink + 1, 2 * shrink + 1))
        mask = cv2.erode(mask, k)
    return mask


def build_hole_levels(ink, close_sizes, min_area=1500):
    """findContours at several ink-closing levels.

    A bubble outline with a hairline gap only becomes a closed loop once the
    ink is closed by a wide enough kernel, so try a ladder of sizes.
    """
    levels = []
    for k in close_sizes:
        ik = ink
        if k > 0:
            ik = cv2.morphologyEx(
                ink, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        contours, holes = find_bubble_holes(ik, min_area)
        levels.append((k, contours, holes))
    return levels


def pick_text_hole(levels, box, max_area):
    """Innermost hole (smallest bbox) that fully contains a text box."""
    tx0, ty0, tx1, ty1 = box
    best = None
    for _, contours, holes in levels:
        for ci, area, (hx, hy, hw, hh) in holes:
            if hx <= tx0 and hy <= ty0 and hx + hw >= tx1 and hy + hh >= ty1:
                if area > max_area:
                    continue
                if best is None or hw * hh < best[0]:
                    best = (hw * hh, contours, ci)
    return (best[1], best[2]) if best else None


def pick_bubble_hole(levels, box, max_area):
    """Ink hole whose bounding box best matches a bubble detection box."""
    bx0, by0, bx1, by1 = box
    barea = max(1, (bx1 - bx0) * (by1 - by0))
    best = None
    for _, contours, holes in levels:
        for ci, area, (hx, hy, hw, hh) in holes:
            harea = hw * hh
            if area > max_area or harea > 4 * barea:
                continue
            ix0, iy0 = max(bx0, hx), max(by0, hy)
            ix1, iy1 = min(bx1, hx + hw), min(by1, hy + hh)
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            if inter < 0.5 * barea:       # hole must cover most of the box
                continue
            score = inter / (barea + harea - inter)
            if best is None or score > best[0]:
                best = (score, contours, ci)
    return (best[1], best[2]) if best else None


def segment_seed(kind, box, gray, levels, shrink, max_area):
    """Bubble mask for one detection box, or None."""
    if kind == "text":
        found = pick_text_hole(levels, box, max_area)
    else:
        found = pick_bubble_hole(levels, box, max_area)
    if found:
        contours, ci = found
        m = np.zeros_like(gray)
        cv2.drawContours(m, contours, ci, 255, cv2.FILLED)
        m = smooth_shrink(m, shrink)
        # a text seed only makes sense on a light bubble ground; white text on
        # a black (inverted) balloon would otherwise fill the black area
        if kind == "text" and cv2.mean(gray, m)[0] < 150:
            return None
        return m
    if kind == "bubble":                  # outline never closed (panel-tangent
        return from_bubble_box(gray, box, shrink)   # bubble): clip by the box
    return None


def from_bubble_box(gray, box, shrink):
    """White component at the box center, holes filled -> bubble mask."""
    h, w = gray.shape
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    roi = gray[y0:y1, x0:x1]
    _, ri = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    white = 255 - ri
    n, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    if n <= 1:
        return None
    cx, cy = (x1 - x0) // 2, (y1 - y0) // 2
    pick = labels[cy, cx]
    if pick == 0:                           # center is ink: take the largest blob
        pick = 1 + int(np.argmax(stats[1:, 4]))
    comp = (labels == pick).astype(np.uint8) * 255
    comp = fill_holes(comp)
    comp = smooth_shrink(comp, shrink)
    full = np.zeros_like(gray)
    full[y0:y1, x0:x1] = comp
    return full


def iou(a, b):
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return inter / union if union else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="18_094.jpg")
    ap.add_argument("--text-mask", default="18_094_mask_text.png",
                    help="stand-in for text detection boxes (or pass --boxes-json)")
    ap.add_argument("--bubble-mask", default="18_094_mask_bubble.png",
                    help="stand-in for bubble detection boxes")
    ap.add_argument("--boxes-json", default="",
                    help='optional [{"box":[x0,y0,x1,y1],"kind":"text|bubble"}]')
    ap.add_argument("--seeds", default="both", choices=["text", "bubble", "both"])
    ap.add_argument("--cluster-px", type=int, default=6,
                    help="merge glyph boxes into one text block before search")
    ap.add_argument("--shrink", type=int, default=3,
                    help="erode px so the black bubble outline survives")
    ap.add_argument("--hole-levels", default="0,5,9,13,17",
                    help="ink closing sizes for the hole search; outlines with gaps "
                         "only close at larger sizes")
    ap.add_argument("--max-bubble-frac", type=float, default=0.15,
                    help="reject a hole larger than this fraction of the page")
    ap.add_argument("--out", default="18_094_cv_from_detections.jpg")
    ap.add_argument("--ref-mask", default="18_094_mask_bubble.png")
    args = ap.parse_args()

    bgr = cv2.imread(args.image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    levels = build_hole_levels(ink, [int(k) for k in args.hole_levels.split(",")])
    max_area = args.max_bubble_frac * h * w

    dets = []
    if args.boxes_json:
        with open(args.boxes_json, encoding="utf-8") as f:
            dets = [(tuple(d["box"]), d.get("kind", "bubble"))
                    for d in json.load(f)]
    else:
        # bubble boxes first: a tight box clips any hole that leaked into
        # nearby art, so its mask wins the dedupe over a text-derived one
        if args.seeds in ("bubble", "both"):
            dets += [(b, "bubble") for b in boxes_from_mask(args.bubble_mask, 100)]
        if args.seeds in ("text", "both"):
            dets += [(b, "text") for b in boxes_from_mask(
                args.text_mask, 100, args.cluster_px)]

    union = np.zeros_like(gray)
    dbg = bgr.copy()
    kept = []
    for box, kind in dets:
        m = segment_seed(kind, box, gray, levels, args.shrink, max_area)
        if m is None or np.count_nonzero(m) < 800:
            continue
        # text lines of one bubble all recover the same bubble: dedupe
        if any(iou(m > 0, k > 0) > 0.6 for k in kept):
            continue
        kept.append(m)
        union = cv2.bitwise_or(union, m)
        x0, y0, x1, y1 = box
        cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 128, 255), 1)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(dbg, cs, -1, (0, 200, 0), 2)
        print(f"  {kind:6s} box {x0:>4},{y0:>4},{x1:>4},{y1:>4} "
              f"-> mask {int(np.count_nonzero(m)):>6} px")

    painted = bgr.copy()
    painted[union > 0] = (255, 255, 255)
    cv2.imwrite(args.out, painted)
    cv2.imwrite(args.out.rsplit(".", 1)[0] + "_mask.png", union)
    cv2.imwrite(args.out.rsplit(".", 1)[0] + "_debug.png", dbg)

    ref = cv2.imread(args.ref_mask, cv2.IMREAD_GRAYSCALE)
    if ref is not None:
        ref = (ref > 127).astype(np.uint8)
        print(f"\nCV union {int(np.count_nonzero(union))} px, model "
              f"{int(np.count_nonzero(ref))} px, IoU {iou(union > 0, ref > 0):.3f}")
        n, _, stats, _ = cv2.connectedComponentsWithStats(ref, connectivity=8)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < 500:
                continue
            hit = np.count_nonzero(ref[y:y + bh, x:x + bw] &
                                   (union[y:y + bh, x:x + bw] > 0))
            print(f"  ref blob {i:>2} bbox {x:>4},{y:>4},{bw:>4},{bh:>4} "
                  f"area {area:>6} recall {hit / area:.2f}")
    print("saved:", args.out, ", *_mask.png and *_debug.png")


if __name__ == "__main__":
    main()
