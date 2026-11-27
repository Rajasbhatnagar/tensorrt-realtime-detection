"""
Step 2: Compare PyTorch vs TensorRT (FP32 / FP16 / INT8) on speed, accuracy and size.

For each model we run validation on the COCO128 dataset (128 images) and record:
  - inference latency (ms per image)  -> FPS = 1000 / latency
  - accuracy (mAP@0.5 and mAP@0.5:0.95)
  - file size (MB)

Run:  python benchmark.py
Output: results/benchmark.csv and results/benchmark.png
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

MODELS = {
    "PyTorch FP32": "models/yolov8n.pt",
    "TensorRT FP32": "models/yolov8n_fp32.engine",
    "TensorRT FP16": "models/yolov8n_fp16.engine",
    "TensorRT INT8": "models/yolov8n_int8.engine",
}
DATA = "coco128.yaml"
OUT_DIR = "results"


def evaluate(name: str, path: str) -> dict:
    print(f"\n===== {name} =====")
    model = YOLO(path, task="detect")
    metrics = model.val(data=DATA, imgsz=640, batch=1, device=0, verbose=False, plots=False)
    latency = metrics.speed["inference"]          # ms per image (pure model time)
    return {
        "Model": name,
        "Latency (ms)": round(latency, 2),
        "FPS": round(1000 / latency, 1),
        "mAP@0.5": round(metrics.box.map50, 3),
        "mAP@0.5:0.95": round(metrics.box.map, 3),
        "Size (MB)": round(os.path.getsize(path) / 1024**2, 1),
    }


def plot(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(df["Model"], df["FPS"], color=["#888888", "#4C9BE8", "#2E7D32", "#F9A825"])
    axes[0].set_title("Speed (higher is better)")
    axes[0].set_ylabel("Frames per second")
    for i, v in enumerate(df["FPS"]):
        axes[0].text(i, v, f"{v:.0f}", ha="center", va="bottom")

    axes[1].bar(df["Model"], df["mAP@0.5"], color=["#888888", "#4C9BE8", "#2E7D32", "#F9A825"])
    axes[1].set_title("Accuracy (mAP@0.5)")
    axes[1].set_ylim(0, 1)
    for i, v in enumerate(df["mAP@0.5"]):
        axes[1].text(i, v, f"{v:.3f}", ha="center", va="bottom")

    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("YOLOv8n: PyTorch vs TensorRT")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "benchmark.png"), dpi=150)
    print(f"[OK] Chart saved -> {OUT_DIR}/benchmark.png")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = [evaluate(n, p) for n, p in MODELS.items() if os.path.exists(p)]
    df = pd.DataFrame(rows)

    base_fps = df.loc[df["Model"] == "PyTorch FP32", "FPS"]
    if not base_fps.empty:
        df["Speed-up"] = (df["FPS"] / base_fps.iloc[0]).round(2).astype(str) + "x"

    df.to_csv(os.path.join(OUT_DIR, "benchmark.csv"), index=False)
    print("\n" + df.to_string(index=False))
    plot(df)


if __name__ == "__main__":
    main()
