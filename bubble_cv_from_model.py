"""Locate bubbles with the RF-DETR-Seg model, then rebuild their masks with CV.

This emulates the detection-only build: the model's dets/labels are used to
place boxes, the segmentation head's masks are only used as a reference for
scoring.  Masks are produced by bubble_cv_from_boxes.py's classic-CV pipeline.

  python bubble_cv_from_model.py --image 006.jpg --model koharu/..._int8.onnx
"""
import argparse

import cv2
import numpy as np
import onnxruntime as ort

import bubble_cv_from_boxes as cvseg

CLASS_NAMES = ["text", "onomatopoeia", "bubble", "panel"]
DEFAULT_THRESHOLDS = [0.25, 0.25, 0.50, 0.50]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def run_model(model_path, image, num_select):
    """Return detections as a list of dicts, plus the raw mask logits."""
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    in_h, in_w = int(inp.shape[2]), int(inp.shape[3])
    outputs = {o.name: o for o in sess.get_outputs()}
    dets_name = next(n for n in outputs if "det" in n.lower())
    labels_name = next(n for n in outputs if "label" in n.lower() or "logit" in n.lower())
    masks_name = next((n for n in outputs if "mask" in n.lower()), None)
    print(f"model {model_path} input {in_w}x{in_h}, outputs {list(outputs)}")

    org_h, org_w = image.shape[:2]
    resized = cv2.resize(image, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = ((rgb - MEAN) / STD).transpose(2, 0, 1)[None]

    names = [dets_name, labels_name] + ([masks_name] if masks_name else [])
    out = sess.run(names, {inp.name: blob})
    dets, labels = out[0][0], out[1][0]
    masks = out[2] if masks_name else None

    num_classes = labels.shape[1] - 1
    scores = 1.0 / (1.0 + np.exp(-np.clip(labels[:, :num_classes], -88.0, 88.0)))
    q, c = scores.shape
    order = np.argsort(-scores, axis=None)[:num_select]
    dets_out = []
    for flat in order:
        qi, ci = divmod(int(flat), c)
        score = float(scores[qi, ci])
        thr = DEFAULT_THRESHOLDS[ci] if ci < len(DEFAULT_THRESHOLDS) else 0.3
        if score < thr:
            continue
        cx, cy, w, h = dets[qi]
        box = (int(round(np.clip((cx - w / 2) * org_w, 0, org_w - 1))),
               int(round(np.clip((cy - h / 2) * org_h, 0, org_h - 1))),
               int(round(np.clip((cx + w / 2) * org_w, 0, org_w - 1))),
               int(round(np.clip((cy + h / 2) * org_h, 0, org_h - 1))))
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        dets_out.append(dict(qi=qi, cls=ci, score=score, box=box))
    return dets_out, masks


def model_mask(masks, qi, org_w, org_h, shrink):
    """Reproduce bubble_mask_preview.py's post-processing for one instance."""
    if masks is None:
        return None
    up = cv2.resize(masks[0, qi], (org_w, org_h), interpolation=cv2.INTER_CUBIC)
    up = cv2.GaussianBlur(up, (0, 0), 2.0)
    m = (up > 0).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.GaussianBlur(m, (0, 0), 1.5)
    m = ((m > 127) * 255).astype(np.uint8)
    if shrink > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * shrink + 1, 2 * shrink + 1))
        m = cv2.erode(m, k)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"koharu\koharu_layout_rfdetr_seg_2xl_1152_int8.onnx")
    ap.add_argument("--image", default="006.jpg")
    ap.add_argument("--seeds", default="both", choices=["text", "bubble", "both"])
    ap.add_argument("--num-select", type=int, default=160)
    ap.add_argument("--shrink", type=int, default=3)
    ap.add_argument("--hole-levels", default="0,5,9,13,17")
    ap.add_argument("--cluster-px", type=int, default=6)
    ap.add_argument("--max-bubble-frac", type=float, default=0.15)
    ap.add_argument("--out", default="006_cv_from_model.jpg")
    args = ap.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SystemExit(f"cannot read {args.image}")
    org_h, org_w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    print(f"image {org_w}x{org_h}")

    dets, masks = run_model(args.model, bgr, args.num_select)
    bubbles = [d for d in dets if d["cls"] == 2]
    texts = [d for d in dets if d["cls"] == 0]
    print(f"detections: {len(texts)} text, {len(bubbles)} bubble")
    for d in bubbles:
        print(f"  bubble {d['score']:.3f} @ {d['box']}")
    for d in texts:
        print(f"  text   {d['score']:.3f} @ {d['box']}")

    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    levels = cvseg.build_hole_levels(
        ink, [int(k) for k in args.hole_levels.split(",")])
    max_area = args.max_bubble_frac * org_h * org_w

    seeds = []
    if args.seeds in ("bubble", "both"):
        seeds += [(d["box"], "bubble", d) for d in bubbles]
    if args.seeds in ("text", "both"):
        # model text instances are per-line; cluster neighbouring ones so a
        # single hole search serves a whole text block
        if texts:
            tmask = np.zeros((org_h, org_w), np.uint8)
            for d in texts:
                x0, y0, x1, y1 = d["box"]
                tmask[y0:y1 + 1, x0:x1 + 1] = 255
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * args.cluster_px + 1, 2 * args.cluster_px + 1))
            n, _, stats, _ = cv2.connectedComponentsWithStats(
                cv2.dilate(tmask, k), connectivity=8)
            for x, y, bw, bh, _ in stats[1:]:
                seeds.append(((x, y, x + bw, y + bh), "text", None))

    union = np.zeros((org_h, org_w), np.uint8)
    dbg = bgr.copy()
    kept = []
    for box, kind, det in seeds:
        m = cvseg.segment_seed(kind, box, gray, levels, args.shrink, max_area)
        if m is None or np.count_nonzero(m) < 800:
            continue
        if any(cvseg.iou(m > 0, k2 > 0) > 0.6 for k2 in kept):
            continue
        kept.append(m)
        union = cv2.bitwise_or(union, m)
        x0, y0, x1, y1 = box
        cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 128, 255), 2)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(dbg, cs, -1, (0, 200, 0), 3)
        print(f"  {kind:6s} box {x0:>5},{y0:>5},{x1:>5},{y1:>5} "
              f"-> mask {int(np.count_nonzero(m)):>7} px")

    # reference: union of the model's own bubble masks
    ref = np.zeros((org_h, org_w), np.uint8)
    for d in bubbles:
        mm = model_mask(masks, d["qi"], org_w, org_h, args.shrink)
        if mm is not None:
            ref[mm > 0] = 255
    if masks is not None:
        print(f"\nCV union {int(np.count_nonzero(union))} px, model "
              f"{int(np.count_nonzero(ref))} px, IoU {cvseg.iou(union > 0, ref > 0):.3f}")
        for d in bubbles:
            mm = model_mask(masks, d["qi"], org_w, org_h, args.shrink)
            if mm is None:
                continue
            x0, y0, x1, y1 = d["box"]
            area = int(np.count_nonzero(mm[y0:y1 + 1, x0:x1 + 1]))
            hit = int(np.count_nonzero(mm[y0:y1 + 1, x0:x1 + 1] &
                                       (union[y0:y1 + 1, x0:x1 + 1] > 0)))
            print(f"  bubble {d['score']:.2f} @ {d['box']} "
                  f"recall {hit / area if area else 0:.2f}")

    painted = bgr.copy()
    painted[union > 0] = (255, 255, 255)
    cv2.imwrite(args.out, painted)
    cv2.imwrite(args.out.rsplit(".", 1)[0] + "_mask.png", union)
    cv2.imwrite(args.out.rsplit(".", 1)[0] + "_debug.png", dbg)
    print("saved:", args.out, ", *_mask.png and *_debug.png")


if __name__ == "__main__":
    main()
