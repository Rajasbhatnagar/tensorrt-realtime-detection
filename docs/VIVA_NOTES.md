# Viva Notes: TensorRT Real-Time Detection

## 30-second explanation
"Deep learning models are trained in frameworks like PyTorch, but those frameworks are not optimized for fast inference. I used NVIDIA TensorRT to optimize a YOLOv8 object detector for my RTX 3050. I converted the model PyTorch → ONNX → TensorRT engine in three precisions (FP32, FP16, INT8), then measured speed and accuracy. TensorRT made it significantly faster with almost the same accuracy, and I show it live on a webcam."

## Core concepts

**Object detection**: finding *what* objects are in an image and *where* they are (bounding boxes). YOLO = "You Only Look Once": it predicts all boxes in one pass of the network, so it's fast.

**Training vs inference**: training = learning the weights (slow, done once). Inference = using the trained model to predict (done millions of times, needs to be fast). TensorRT is only for inference.

**TensorRT**: NVIDIA's SDK that takes a trained model and produces an optimized "engine" for a specific NVIDIA GPU.

**ONNX (Open Neural Network Exchange)**: a common file format for neural networks. PyTorch exports to ONNX, and TensorRT reads ONNX. It's the bridge between them.

**Engine (.engine / .plan)**: the output of TensorRT: an optimized, compiled version of the model for one GPU + one TensorRT version.

## How TensorRT makes it faster
1. **Layer & tensor fusion**: merges several layers (e.g. Convolution + BatchNorm + Activation) into one GPU kernel, so data is read/written to memory fewer times.
2. **Precision reduction**: uses FP16 or INT8 instead of FP32. Smaller numbers = less memory and faster math (RTX GPUs have Tensor Cores built for FP16/INT8).
3. **Kernel auto-tuning**: tries several implementations of each layer on *your* GPU and keeps the fastest.
4. **Memory optimization**: reuses memory buffers between layers instead of allocating new ones.
5. **Removing unused parts**: drops operations only needed for training (e.g. dropout).

## Precisions
| Precision | Bits | Speed | Accuracy | Notes |
|---|---|---|---|---|
| FP32 | 32-bit float | Baseline | Full | Default in PyTorch |
| FP16 | 16-bit float | ~2x faster | Almost identical | Easy: just a flag |
| INT8 | 8-bit integer | Fastest | Small drop | Needs **calibration** |

**Calibration (INT8)**: INT8 can only represent 256 values. TensorRT runs a few sample images through the model to see the range of values in each layer, then picks a scale factor that maps those ranges to INT8 with the least information loss.

## Metrics
- **Latency**: time to process one image (ms). Lower is better.
- **FPS (frames per second)**: = 1000 / latency. Real-time ≈ 30 FPS or more.
- **mAP (mean Average Precision)**: standard accuracy metric for detection. mAP@0.5 counts a box as correct if it overlaps the true box by at least 50% (IoU ≥ 0.5).
- **IoU (Intersection over Union)**: overlap area / total combined area of predicted and real box.

## Likely questions and answers

**Q: Why not just use PyTorch?**
PyTorch is flexible and built for training. TensorRT is built only for inference, so it can make GPU-specific optimizations PyTorch can't.

**Q: Why did you choose YOLOv8n?**
It's the smallest YOLOv8 model (~3.2 million parameters), it runs in real time on a laptop GPU, and it's pretrained on COCO (80 everyday object classes), so no training was needed.

**Q: Can you copy the engine file to another computer?**
No. The engine is tuned for the exact GPU and TensorRT version. On another machine you rebuild it from the ONNX file.

**Q: Why does accuracy drop with INT8 but not much with FP16?**
FP16 still has a wide range and enough precision for these weights. INT8 has only 256 levels, so some rounding error is unavoidable; calibration keeps it small.

**Q: What is a workspace size?**
Temporary GPU memory TensorRT may use while trying different kernels during the build.

**Q: What dataset did you use?**
COCO128: the first 128 images of the COCO dataset, for benchmarking. COCO8 (8 images) for INT8 calibration.

**Q: Why do you run warm-up before timing?**
The first few runs are slow because of GPU initialization and memory allocation. Warm-up removes that from the measurement. (Ultralytics `val()` does warm-up internally.)

**Q: Where is TensorRT used in real life?**
Self-driving cars, CCTV analytics, robots and drones (NVIDIA Jetson), and serving AI models in data centers.

**Q: What are the limitations?**
Only works on NVIDIA GPUs, engines aren't portable, building takes time, and some custom layers need plugins.

**Q: What would you do next?**
Train on a custom dataset (e.g. helmet detection), try dynamic batch sizes, or deploy on an NVIDIA Jetson.
