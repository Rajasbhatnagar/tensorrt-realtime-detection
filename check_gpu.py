"""
Step 0: Check that the GPU, CUDA, PyTorch and TensorRT are all ready.

Run:  python check_gpu.py
"""
import shutil
import subprocess


def main():
    print("=" * 60)
    print("GPU / CUDA / TensorRT environment check")
    print("=" * 60)

    # 1. nvidia-smi (driver level)
    if shutil.which("nvidia-smi"):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        print(f"[nvidia-smi] GPU, VRAM, driver : {out.stdout.strip()}")
    else:
        print("[nvidia-smi] NOT FOUND - install/update the NVIDIA driver")

    # 2. PyTorch + CUDA
    try:
        import torch
        print(f"[PyTorch]    version          : {torch.__version__}")
        print(f"[PyTorch]    CUDA available   : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"[PyTorch]    GPU              : {props.name}")
            print(f"[PyTorch]    VRAM             : {props.total_memory / 1024**3:.1f} GB")
            print(f"[PyTorch]    Compute capability: {props.major}.{props.minor}")
            print(f"[PyTorch]    CUDA version     : {torch.version.cuda}")
        else:
            print("  -> Install the CUDA build of PyTorch (see README).")
    except ImportError:
        print("[PyTorch]    NOT INSTALLED")

    # 3. TensorRT
    try:
        import tensorrt as trt
        print(f"[TensorRT]   version          : {trt.__version__}")
    except ImportError:
        print("[TensorRT]   NOT INSTALLED - run: pip install tensorrt")

    # 4. Ultralytics
    try:
        import ultralytics
        print(f"[Ultralytics] version         : {ultralytics.__version__}")
    except ImportError:
        print("[Ultralytics] NOT INSTALLED - run: pip install ultralytics")

    print("=" * 60)


if __name__ == "__main__":
    main()
