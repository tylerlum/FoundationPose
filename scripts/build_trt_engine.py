"""Build a TensorRT engine from an ONNX file.

Used to build engines for the FoundationPose refiner / scorer ONNXes
exported via `export_refine_onnx.py`. Sets up a single optimization
profile with a dynamic batch range so the same engine can serve
track_one (batch=1) and register (batch=~252).

Usage:
    python third_party/FoundationPose/scripts/build_trt_engine.py \
        --onnx third_party/FoundationPose/weights/2023-10-28-18-33-37/refine_net.onnx \
        --plan third_party/FoundationPose/weights/2023-10-28-18-33-37/refine_net_fp16.plan \
        --min_batch 1 --opt_batch 1 --max_batch 256 --fp16
"""
import argparse
import os
import time
from pathlib import Path

import tensorrt as trt


def build(onnx_path: str, plan_path: str,
          min_batch: int, opt_batch: int, max_batch: int,
          fp16: bool, workspace_gb: int = 8):
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(onnx_path)
    os.makedirs(os.path.dirname(os.path.abspath(plan_path)), exist_ok=True)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError(f"failed to parse ONNX {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                 workspace_gb * (1 << 30))
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[trt] FP16 enabled")
    else:
        print("[trt] FP16 disabled")

    # One profile covering min..max batch on every input.
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        shape = list(tensor.shape)
        # Assume dim 0 is the batch dim and any other dynamic dims are not
        # allowed (true for refiner: shape=[batch, 6, 160, 160]).
        if shape[0] != -1:
            print(f"[trt] WARNING: input {tensor.name} has fixed batch={shape[0]}; "
                  f"profile will pin it.")
            mn = op = mx = shape[0]
            tail = shape[1:]
            concrete_mn = tuple([mn] + tail)
            concrete_op = tuple([op] + tail)
            concrete_mx = tuple([mx] + tail)
        else:
            tail = shape[1:]
            if any(d == -1 for d in tail):
                raise RuntimeError(
                    f"input {tensor.name} has non-batch dynamic dims: {shape}")
            concrete_mn = tuple([min_batch] + tail)
            concrete_op = tuple([opt_batch] + tail)
            concrete_mx = tuple([max_batch] + tail)
        profile.set_shape(tensor.name, concrete_mn, concrete_op, concrete_mx)
        print(f"[trt] input {tensor.name}: min={concrete_mn} "
              f"opt={concrete_op} max={concrete_mx}")
    config.add_optimization_profile(profile)

    print(f"[trt] building from {onnx_path} (this can take several minutes)...")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None")
    print(f"[trt] built in {(time.time()-t0)/60:.1f} min")

    with open(plan_path, 'wb') as f:
        f.write(serialized)
    print(f"[trt] saved -> {plan_path}  ({os.path.getsize(plan_path)/1e6:.1f} MB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--onnx', required=True)
    p.add_argument('--plan', required=True)
    p.add_argument('--min_batch', type=int, default=1)
    p.add_argument('--opt_batch', type=int, default=1,
                   help='Batch size the engine is optimized for (track_one is 1).')
    p.add_argument('--max_batch', type=int, default=256,
                   help='Upper bound — needs to cover register, which uses ~252 pose hypotheses.')
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--workspace_gb', type=int, default=8)
    args = p.parse_args()
    build(args.onnx, args.plan, args.min_batch, args.opt_batch,
          args.max_batch, args.fp16, args.workspace_gb)


if __name__ == '__main__':
    main()
