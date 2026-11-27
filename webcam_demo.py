"""
Step 3: Live webcam demo with an FPS counter.

Single model:
    python webcam_demo.py --model models/yolov8n_fp16.engine
Side-by-side comparison (left = PyTorch, right = TensorRT):
    python webcam_demo.py --compare

Press 'q' to quit.
"""
import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO


class FPSCounter:
    """Smoothed FPS so the number on screen doesn't jump around."""

    def __init__(self, alpha=0.9):
        self.alpha, self.fps = alpha, 0.0

    def update(self, seconds: float) -> float:
        current = 1.0 / max(seconds, 1e-6)
        self.fps = current if self.fps == 0 else self.alpha * self.fps + (1 - self.alpha) * current
        return self.fps


def run_model(model, frame, counter, label):
    start = time.perf_counter()
    result = model(frame, imgsz=640, verbose=False)[0]
    fps = counter.update(time.perf_counter() - start)
    out = result.plot()  # draws boxes + class names
    cv2.rectangle(out, (0, 0), (330, 70), (0, 0, 0), -1)
    cv2.putText(out, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(out, f"FPS: {fps:.0f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/yolov8n_fp16.engine")
    ap.add_argument("--compare", action="store_true", help="PyTorch vs TensorRT side by side")
    ap.add_argument("--baseline", default="models/yolov8n.pt")
    ap.add_argument("--source", type=int, default=0, help="webcam index")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam")

    trt_model = YOLO(args.model, task="detect")
    trt_counter = FPSCounter()
    label = "TensorRT " + ("FP16" if "fp16" in args.model else "INT8" if "int8" in args.model else "FP32")

    if args.compare:
        torch_model = YOLO(args.baseline, task="detect")
        torch_counter = FPSCounter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        right = run_model(trt_model, frame, trt_counter, label)
        if args.compare:
            left = run_model(torch_model, frame, torch_counter, "PyTorch FP32")
            view = np.hstack([left, right])
        else:
            view = right

        cv2.imshow("TensorRT Real-Time Detection (press q to quit)", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
