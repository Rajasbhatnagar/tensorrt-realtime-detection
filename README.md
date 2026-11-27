# Real-Time Object Detection Accelerated with TensorRT

Optimizing a YOLOv8 object detection model with **NVIDIA TensorRT** and measuring how much faster it runs on a laptop GPU (RTX 3050), compared to plain PyTorch.

## Idea in one line
Take a trained YOLOv8n detector → convert it to a TensorRT engine (FP32, FP16, INT8) → compare speed and accuracy → show it live on a webcam.

## Pipeline

```
yolov8n.pt (PyTorch)  ──export──▶  yolov8n.onnx  ──TensorRT builder──▶  yolov8n_fp32.engine
                                                                      ▶  yolov8n_fp16.engine
                                                                      ▶  yolov8n_int8.engine (calibrated)
                                                                                │
                                         benchmark.py  ◀────────────────────────┤
                                         webcam_demo.py ◀───────────────────────┘
```

## Project files

| File | What it does |
|---|---|
| `check_gpu.py` | Checks GPU, driver, CUDA, PyTorch and TensorRT are installed |
| `export_engines.py` | PyTorch → ONNX → TensorRT engines (FP32 / FP16 / INT8) |
| `build_engine_manual.py` | Same conversion written with the raw TensorRT API (to explain internals) |
| `benchmark.py` | Measures latency, FPS, mAP and size; saves a table + chart |
| `webcam_demo.py` | Live detection with FPS counter; `--compare` shows PyTorch vs TensorRT side by side |
| `docs/VIVA_NOTES.md` | Concepts and likely viva questions with answers |

## Setup (Windows, NVIDIA GPU)

1. Update the NVIDIA driver (check with `nvidia-smi`).
2. Create an environment:
   ```bash
   conda create -n trt python=3.11 -y
   conda activate trt
   ```
3. Install PyTorch with CUDA (pick the CUDA version shown on https://pytorch.org):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   ```
4. Install the rest:
   ```bash
   pip install -r requirements.txt
   ```
5. Verify:
   ```bash
   python check_gpu.py
   ```

## How to run

```bash
python export_engines.py                 # builds the TensorRT engines (a few minutes)
python benchmark.py                      # results/benchmark.csv + results/benchmark.png
python webcam_demo.py --compare          # live demo: PyTorch (left) vs TensorRT (right)
```

## Results (RTX 3050 Laptop GPU)

_Fill in after running `benchmark.py`._

| Model | Latency (ms) | FPS | mAP@0.5 | Size (MB) | Speed-up |
|---|---|---|---|---|---|
| PyTorch FP32 | | | | | 1.0x |
| TensorRT FP32 | | | | | |
| TensorRT FP16 | | | | | |
| TensorRT INT8 | | | | | |

![Benchmark](results/benchmark.png)

## Key takeaways
- TensorRT speeds up inference without retraining the model.
- FP16 gives a large speed-up with almost no accuracy loss.
- INT8 is the fastest but loses a little accuracy and needs calibration data.
- An engine is built for one specific GPU, so it has to be rebuilt on a different machine.
