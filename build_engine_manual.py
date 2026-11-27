"""
(Optional, for understanding) Build a TensorRT engine from ONNX using the raw TensorRT Python API.

export_engines.py does this for us through Ultralytics. This file shows what happens
underneath, step by step, so it is easy to explain in the viva.

Run:  python build_engine_manual.py --onnx models/yolov8n.onnx --fp16
"""
import argparse

import tensorrt as trt

LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(onnx_path: str, engine_path: str, fp16: bool):
    # 1. Builder: the object that optimizes the network
    builder = trt.Builder(LOGGER)

    # 2. Network definition: an empty graph that TensorRT will fill
    network = builder.create_network(0)

    # 3. ONNX parser: reads the ONNX file and fills the network graph
    parser = trt.OnnxParser(network, LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise SystemExit("Failed to parse ONNX file")

    # 4. Builder config: optimization settings
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2 GB scratch memory
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)   # allow half-precision kernels

    # 5. Build: TensorRT fuses layers, picks the fastest GPU kernels (auto-tuning)
    #    and returns a serialized engine (a binary plan for THIS GPU only)
    print("Building engine... (this can take 1-3 minutes)")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("Engine build failed")

    # 6. Save the engine to disk
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"[OK] Saved engine -> {engine_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="models/yolov8n.onnx")
    ap.add_argument("--out", default="models/yolov8n_manual.engine")
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()
    build_engine(args.onnx, args.out, args.fp16)
