"""Export koharu/model.safetensors (RF-DETR Seg 2XL, 1152px) to ONNX.

The Java class com.xulihang.ONNXRFDETRSeg consumes the generated file.

Install the export dependencies first (Tsinghua mirror):

    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "rfdetr==1.7.0" safetensors onnx onnxruntime

Usage:

    python koharu/export_onnx.py

Output:

    koharu/koharu_layout_rfdetr_seg_2xl_1152.onnx
      input  [1, 3, 1152, 1152]
      dets   [1, 300, 4]        normalized cxcywh
      labels [1, 300, 5]        class logits, last column is no-object
      masks  [1, 300, 288, 288] instance mask logits (input / 4)
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path

from rfdetr import RFDETRSeg2XLarge
from rfdetr.config import PretrainWeightsCompatibilityWarning
from safetensors.torch import load_file

CLASS_NAMES = ["text", "onomatopoeia", "bubble", "panel"]
RESOLUTION = 1152

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "model.safetensors"
OUTPUT = HERE / "koharu_layout_rfdetr_seg_2xl_1152.onnx"


def main() -> None:
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"weights not found: {WEIGHTS}")

    with warnings.catch_warnings():
        # Building from scratch is intentional: the checkpoint is loaded strictly below.
        warnings.simplefilter("ignore", PretrainWeightsCompatibilityWarning)
        model = RFDETRSeg2XLarge(
            pretrain_weights=None,
            resolution=RESOLUTION,
            num_select=160,
            num_classes=len(CLASS_NAMES),
        )

    incompatible = model.model.model.load_state_dict(load_file(str(WEIGHTS), device="cpu"), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"incompatible weights: {incompatible}")

    model.model.class_names = CLASS_NAMES.copy()
    model.model.model.eval()

    exported = model.export(
        output_dir=str(HERE),
        shape=(RESOLUTION, RESOLUTION),
        batch_size=1,
        opset_version=17,
        verbose=False,
        notes={
            "source": "koharu-layout-rfdetr-seg-2xl-1152",
            "classes": CLASS_NAMES,
        },
    )

    exported = Path(exported)
    if exported.resolve() != OUTPUT.resolve():
        if OUTPUT.exists():
            OUTPUT.unlink()
        shutil.move(str(exported), str(OUTPUT))

    print(f"exported: {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
