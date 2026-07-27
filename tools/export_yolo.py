#!/usr/bin/env python3
"""Export YOLO11n / YOLO11s to ONNX (opset 12, 640x640, simplified).

Weights are auto-downloaded by ultralytics on first run; set an HTTPS proxy in
the environment if this machine has no direct internet, e.g.:

    export https_proxy=http://127.0.0.1:10826 http_proxy=http://127.0.0.1:10826
    python tools/export_yolo.py

Outputs <name>.onnx next to the downloaded <name>.pt (ultralytics default).
"""
import sys
from ultralytics import YOLO

MODELS = sys.argv[1:] or ["yolo11n.pt", "yolo11s.pt"]

for model_path in MODELS:
    print(f"[export] loading {model_path}")
    model = YOLO(model_path)
    onnx_path = model.export(
        format="onnx",
        opset=12,
        imgsz=640,
        simplify=True,
        dynamic=False,
    )
    print(f"[export] wrote {onnx_path}")
