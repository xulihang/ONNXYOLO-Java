"""Weight-only INT8 quantization for the KoharuLayout RF-DETR Seg 2XL ONNX model.

Only MatMul/Gemm weights inside the DINOv2 backbone are quantized to INT8.
The transformer decoder and the detection/segmentation heads stay FP32:
quantizing them (static or dynamic) collapses the class logits and produces
near-zero detections, while backbone-only quantization keeps boxes and masks
almost identical to FP32.

Usage:

    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple onnx onnxruntime
    python koharu/quantize_onnx.py

Input:

    koharu/koharu_layout_rfdetr_seg_2xl_1152.onnx        (FP32, ~148 MB)

Output:

    koharu/koharu_layout_rfdetr_seg_2xl_1152_int8.onnx   (INT8 weights, ~85 MB)
"""

from __future__ import annotations

from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "koharu_layout_rfdetr_seg_2xl_1152.onnx"
OUTPUT = HERE / "koharu_layout_rfdetr_seg_2xl_1152_int8.onnx"


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"FP32 ONNX not found: {SOURCE} (run export_onnx.py first)")

    model = onnx.load(str(SOURCE), load_external_data=False)
    backbone_nodes = [
        node.name
        for node in model.graph.node
        if "/backbone/" in node.name and node.op_type in ("MatMul", "Gemm")
    ]
    if not backbone_nodes:
        raise RuntimeError("no backbone MatMul/Gemm nodes found")

    quantize_dynamic(
        str(SOURCE),
        str(OUTPUT),
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul", "Gemm", "Attention"],
        nodes_to_quantize=backbone_nodes,
        # Only quantize MatMul when the weight is a constant (linear layers),
        # leaving attention QK^T / attn*V and the decoder untouched.
        extra_options={"MatMulConstBOnly": True},
    )

    print(f"quantized {len(backbone_nodes)} backbone nodes")
    print(f"exported: {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
