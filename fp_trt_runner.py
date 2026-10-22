"""TensorRT inference for FoundationPose's RefineNet (and ScoreNet, TBD).

Drop-in replacement for `RefineNet.forward(A, B) -> {'trans': ..., 'rot': ...}`.
Accepts torch CUDA tensors (zero-copy via `data_ptr()`), runs the engine
on the same CUDA stream as the rest of the pipeline.

Engines must be built with a single optimization profile whose batch dim
covers the values needed at call time (1 for track_one, ~252 for register).
See `scripts/build_trt_engine.py`.
"""
import threading

import numpy as np
import torch
import torch.nn as nn

try:
    import tensorrt as trt
    from cuda import cudart
except ImportError as e:
    raise ImportError(
        f"TRT runner requires `tensorrt` + `cuda-python`; got: {e}")

_TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def _check_cuda(err):
    if hasattr(err, '__iter__'):
        err = err[0] if err else cudart.cudaError_t.cudaSuccess
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")


_TRT_DTYPE_TO_TORCH = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.int32: torch.int32,
    trt.int8: torch.int8,
}


class TrtRefinerRunner(nn.Module):
    """RefineNet TRT runner.

    Signature mirrors RefineNet.forward exactly:
        forward(A, B)  ->  {'trans': (B, 3) cuda fp32,
                            'rot':   (B, K) cuda fp32}

    K is 3 (axis_angle) or 6 (6d) — read from the engine's rot output
    shape at init.

    The runner is an `nn.Module` only so `predictor.model = runner` keeps
    the surrounding code happy (e.g. `.eval()`, `.train()`). It owns no
    learnable parameters.
    """

    def __init__(self, engine_path: str):
        super().__init__()
        with open(engine_path, 'rb') as f:
            engine_bytes = f.read()
        self._runtime = trt.Runtime(_TRT_LOGGER)
        self.engine = self._runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()

        # Tensor metadata. Inputs: A, B  (-1, c_in, H, W). Outputs: trans, rot.
        self._tensors = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self._tensors[name] = {
                'name': name,
                'mode': self.engine.get_tensor_mode(name),
                'shape': tuple(self.engine.get_tensor_shape(name)),
                'dtype': self.engine.get_tensor_dtype(name),
            }
        self.input_names = [n for n, t in self._tensors.items()
                            if t['mode'] == trt.TensorIOMode.INPUT]
        self.output_names = [n for n, t in self._tensors.items()
                             if t['mode'] == trt.TensorIOMode.OUTPUT]
        if set(self.input_names) != {'A', 'B'}:
            raise RuntimeError(
                f"refiner engine should have inputs ['A','B'], got {self.input_names}")
        if 'trans' not in self.output_names or 'rot' not in self.output_names:
            raise RuntimeError(
                f"refiner engine should have outputs 'trans' and 'rot', "
                f"got {self.output_names}")

        # Read the profile's max-batch so we can pre-allocate buffers.
        # An engine without dynamic batch will return shape with bs already set.
        # `get_profile_shape` returns (min, opt, max).
        a_min, a_opt, a_max = self.engine.get_tensor_profile_shape('A', 0)
        self._max_batch = int(a_max[0])
        self._c_in = int(a_max[1])
        self._H = int(a_max[2])
        self._W = int(a_max[3])

        # Output K dim (3 for axis_angle, 6 for 6d). The engine's rot output
        # shape is (-1, K); pull K from the static dims.
        rot_shape = self._tensors['rot']['shape']
        if len(rot_shape) != 2:
            raise RuntimeError(f"unexpected rot output shape {rot_shape}")
        self._rot_k = int(rot_shape[1])

        # Pre-allocate worst-case buffers as torch tensors so they can be
        # used zero-copy. We always run on the default CUDA stream.
        torch_dt_in = _TRT_DTYPE_TO_TORCH[self._tensors['A']['dtype']]
        torch_dt_out = _TRT_DTYPE_TO_TORCH[self._tensors['trans']['dtype']]
        self._A_buf = torch.empty(
            (self._max_batch, self._c_in, self._H, self._W),
            dtype=torch_dt_in, device='cuda').contiguous()
        self._B_buf = torch.empty_like(self._A_buf)
        self._trans_buf = torch.empty((self._max_batch, 3),
                                      dtype=torch_dt_out, device='cuda')
        self._rot_buf = torch.empty((self._max_batch, self._rot_k),
                                    dtype=torch_dt_out, device='cuda')

        # Permanent bindings — TRT 10 needs addresses set before each
        # execute_async_v3, but the underlying buffer is stable.
        self.context.set_tensor_address('A',     int(self._A_buf.data_ptr()))
        self.context.set_tensor_address('B',     int(self._B_buf.data_ptr()))
        self.context.set_tensor_address('trans', int(self._trans_buf.data_ptr()))
        self.context.set_tensor_address('rot',   int(self._rot_buf.data_ptr()))

        # Use a dedicated CUDA stream so we can synchronize on output
        # without blocking unrelated work. (PyTorch's default stream is
        # also fine; this just keeps lifetime explicit.)
        err, self._stream = cudart.cudaStreamCreate()
        _check_cuda(err)

        self._lock = threading.Lock()

    def forward(self, A, B):
        """A, B: torch (bs, 6, 160, 160) cuda fp32. Returns dict of cuda tensors."""
        if not (A.is_cuda and B.is_cuda):
            raise RuntimeError("TrtRefinerRunner expects CUDA tensors")
        if A.shape != B.shape:
            raise RuntimeError(f"A/B shape mismatch: {A.shape} vs {B.shape}")
        bs = int(A.shape[0])
        if bs > self._max_batch:
            raise RuntimeError(
                f"batch {bs} exceeds engine max {self._max_batch}")

        # Cast and copy into pre-allocated buffer slots. We must keep the
        # bound device pointer stable across calls; copy_ guarantees that
        # without re-pointing the binding.
        target_in_dtype = self._A_buf.dtype
        a_in = A if A.dtype == target_in_dtype else A.to(target_in_dtype)
        b_in = B if B.dtype == target_in_dtype else B.to(target_in_dtype)
        with self._lock:
            self._A_buf[:bs].copy_(a_in.contiguous())
            self._B_buf[:bs].copy_(b_in.contiguous())

            ok = self.context.set_input_shape('A', (bs, self._c_in, self._H, self._W))
            if not ok:
                raise RuntimeError(f"set_input_shape A bs={bs} failed")
            ok = self.context.set_input_shape('B', (bs, self._c_in, self._H, self._W))
            if not ok:
                raise RuntimeError(f"set_input_shape B bs={bs} failed")

            ok = self.context.execute_async_v3(stream_handle=self._stream)
            if not ok:
                raise RuntimeError("TRT execute_async_v3 failed")
            err, = cudart.cudaStreamSynchronize(self._stream)
            _check_cuda(err)

            # Slice to the active batch range. Cast to fp32 so callers
            # downstream of the predictor (`output[k].float()`) don't see
            # a regression in dtype.
            trans = self._trans_buf[:bs].clone().float()
            rot = self._rot_buf[:bs].clone().float()
        return {'trans': trans, 'rot': rot}

    @property
    def max_batch(self):
        return self._max_batch

    @property
    def rot_dim(self):
        return self._rot_k

    def __del__(self):
        try:
            cudart.cudaStreamDestroy(self._stream)
        except Exception:
            pass
