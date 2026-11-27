"""
Step 1: Convert YOLOv8n (PyTorch) -> ONNX -> TensorRT engines (FP32, FP16, INT8).

Pipeline:
    yolov8n.pt  --(export)-->  yolov8n.onnx  --(TensorRT builder)-->  *.engine

Run:  python export_engines.py
Output files go to the ./models folder.
"""
import os
import shutil
import time

from ultralytics import YOLO

MODEL_NAME = "yolov8n.pt"   # smallest YOLOv8 model, ~3.2M parameters
IMG_SIZE = 640              # input image size (640 x 640)
OUT_DIR = "models"
CALIB_DATA = "coco8.yaml"   # small dataset used for INT8 calibration


def export(precision: str) -> str:
    """Build one TensorRT engine and move it to models/yolov8n_<precision>.engine"""
    model = YOLO(MODEL_NAME)  # fresh copy each time
    kwargs = dict(format="engine", imgsz=IMG_SIZE, device=0, batch=1, workspace=2)

    if precision == "fp16":
        kwargs["half"] = True
    elif precision == "int8":
        kwargs["int8"] = True
        kwargs["data"] = CALIB_DATA   # INT8 needs sample images to calibrate scales

    start = time.time()
    path = model.export(**kwargs)
    took = time.time() - start

    target = os.path.join(OUT_DIR, f"yolov8n_{precision}.engine")
    shutil.move(path, target)
    print(f"[OK] {precision.upper()} engine -> {target}  (built in {took:.0f}s)")
    return target


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ONNX: framework-neutral format, the "bridge" between PyTorch and TensorRT
    onnx_path = YOLO(MODEL_NAME).export(format="onnx", imgsz=IMG_SIZE, simplify=True)
    shutil.move(onnx_path, os.path.join(OUT_DIR, "yolov8n.onnx"))
    print(f"[OK] ONNX model -> {OUT_DIR}/yolov8n.onnx")

    # Keep a copy of the original PyTorch weights for the baseline
    if os.path.exists(MODEL_NAME):
        shutil.copy(MODEL_NAME, os.path.join(OUT_DIR, MODEL_NAME))

    for precision in ["fp32", "fp16", "int8"]:
        export(precision)

    print("\nAll engines built. Next: python benchmark.py")


if __name__ == "__main__":
    main()
