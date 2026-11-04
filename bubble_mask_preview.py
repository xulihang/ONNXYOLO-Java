"""Preview: paint Koharu RF-DETR-Seg masks white on the original image.

Mirrors the post-processing in ONNXRFDETRSeg.java:
  sigmoid on class logits -> top-k (num_select) -> per-class threshold
  -> normalized cxcywh to pixel xyxy -> mask logits upscaled, >0 = foreground
"""
import argparse

import cv2
import numpy as np
import onnxruntime as ort

CLASS_NAMES = ["text", "onomatopoeia", "bubble", "panel"]
DEFAULT_THRESHOLDS = [0.25, 0.25, 0.50, 0.50]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"koharu\koharu_layout_rfdetr_seg_2xl_1152_int8.onnx")
    ap.add_argument("--image", default="18_094.jpg")
    ap.add_argument("--classes", default="bubble", help="comma separated: text,onomatopoeia,bubble,panel")
    ap.add_argument("--num-select", type=int, default=160)
    ap.add_argument("--shrink", type=int, default=3,
                    help="erode the mask inward by N px so the bubble outline survives (0 = old behavior)")
    ap.add_argument("--no-smooth", action="store_true", help="disable edge smoothing")
    ap.add_argument("--out", default="18_094_bubble_white.jpg")
    args = ap.parse_args()

    wanted = {c.strip() for c in args.classes.split(",") if c.strip()}
    class_ids = [i for i, n in enumerate(CLASS_NAMES) if n in wanted]
    if not class_ids:
        raise SystemExit(f"unknown classes: {args.classes}")

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    inp_name = inp.name
    in_h, in_w = int(inp.shape[2]), int(inp.shape[3])
    outputs = {o.name: o.shape for o in sess.get_outputs()}
    print("model:", args.model, "input:", inp.shape)
    print("outputs:", outputs)

    dets_name = next(n for n in outputs if "det" in n.lower())
    labels_name = next(n for n in outputs if "label" in n.lower() or "logit" in n.lower())
    masks_name = next((n for n in outputs if "mask" in n.lower()), None)

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"cannot read {args.image}")
    org_h, org_w = image.shape[:2]
    print(f"image: {org_w}x{org_h}")

    # preprocess: stretch resize, BGR->RGB, (x/255-mean)/std, CHW
    resized = cv2.resize(image, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = ((rgb - MEAN) / STD).transpose(2, 0, 1)[None]

    dets, labels, masks = sess.run([dets_name, labels_name, masks_name] if masks_name else [dets_name, labels_name],
                                   {inp_name: blob})[:3]

    dets, labels = dets[0], labels[0]
    num_classes = labels.shape[1] - 1
    scores = 1.0 / (1.0 + np.exp(-np.clip(labels[:, :num_classes], -88.0, 88.0)))

    q, c = scores.shape
    order = np.argsort(-scores, axis=None)[: args.num_select]
    picks = []
    for flat in order:
        qi, ci = divmod(int(flat), c)
        thr = DEFAULT_THRESHOLDS[ci] if ci < len(DEFAULT_THRESHOLDS) else 0.3
        if scores[qi, ci] < thr:
            continue
        cx, cy, w, h = dets[qi]
        x1 = np.clip((cx - w / 2) * org_w, 0, org_w - 1)
        y1 = np.clip((cy - h / 2) * org_h, 0, org_h - 1)
        x2 = np.clip((cx + w / 2) * org_w, 0, org_w - 1)
        y2 = np.clip((cy + h / 2) * org_h, 0, org_h - 1)
        if x2 <= x1 or y2 <= y1:
            continue
        picks.append((qi, ci, float(scores[qi, ci]), (x1, y1, x2, y2)))

    print(f"detections above threshold: {len(picks)}")
    for qi, ci, score, box in picks:
        print(f"  {CLASS_NAMES[ci]:14s} {score:.3f} @ [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")

    union = np.zeros((org_h, org_w), np.uint8)
    chosen = [(qi, ci, s, b) for qi, ci, s, b in picks if ci in class_ids]
    if masks is not None:
        # upsample the low-res logits (288x288) with cubic + a light blur, so the
        # hard threshold below does not leave staircase edges at 4x magnification
        for qi, ci, score, box in chosen:
            up = cv2.resize(masks[0, qi], (org_w, org_h), interpolation=cv2.INTER_CUBIC)
            if not args.no_smooth:
                up = cv2.GaussianBlur(up, (0, 0), 2.0)
            union[up > 0] = 255
        if not args.no_smooth:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, k)
            union = cv2.morphologyEx(union, cv2.MORPH_OPEN, k)
            union = cv2.GaussianBlur(union, (0, 0), 1.5)
            union = ((union > 127) * 255).astype(np.uint8)
        # shrink inward: keep the bubble's black outline and avoid spilling outside it
        if args.shrink > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * args.shrink + 1, 2 * args.shrink + 1))
            union = cv2.erode(union, k)
        print(f"chosen {CLASS_NAMES[class_ids[0]] if len(class_ids)==1 else class_ids} masks: {len(chosen)}"
              f" (shrink={args.shrink}px)")
    else:
        print("no mask output in model; painting boxes instead")
        for qi, ci, score, box in chosen:
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            union[y1:y2 + 1, x1:x2 + 1] = 255

    painted = image.copy()
    painted[union > 0] = (255, 255, 255)
    cv2.imwrite(args.out, painted)
    mask_path = args.out.rsplit(".", 1)[0] + "_mask.png"
    cv2.imwrite(mask_path, union)
    print("saved:", args.out, "and", mask_path)


if __name__ == "__main__":
    main()
