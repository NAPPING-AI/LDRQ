"""Inference back ends with one interface: `infer(x) -> np.ndarray`.

  TrtRuntime   the robot: a TensorRT plan built on the Orin by r1_build_engine (the
               same tool, the same parity gate, as the walking policy). CUDA memory goes
               through ctypes on libcudart, because the robot's apt is broken and
               pycuda / cuda-python are not there. Both TensorRT Python APIs are handled:
               8.5 (JetPack 5.1.1: bindings + execute_async_v2) and 10.x (the dev box:
               named tensors + execute_async_v3). Only the 10.x path has run on the dev
               box; the 8.5 path is checked on the robot by `engine-check` before use.
  OrtRuntime   onnxruntime on the CPU: the reference the fixture is made with, and
               the dev box's runtime for replay.

One input, one output, static shapes, FP32 at the boundary -- the same contract
r1_policy_runner::TrtPolicy enforces.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import time

import numpy as np

_CUDART_CANDIDATES = (
    "libcudart.so", "libcudart.so.11.0", "libcudart.so.12", "libcudart.so.13",
    "/usr/local/cuda/lib64/libcudart.so", "/usr/local/cuda/targets/aarch64-linux/lib/libcudart.so",
    "/usr/local/cuda-11.4/lib64/libcudart.so.11.0")
H2D, D2H = 1, 2


def load_cudart():
    tried = []
    found = ctypes.util.find_library("cudart")
    for name in ([os.environ["R1_CUDART"]] if os.environ.get("R1_CUDART") else []) + \
            ([found] if found else []) + list(_CUDART_CANDIDATES):
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            tried.append(name)
            continue
        lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        lib.cudaFree.argtypes = [ctypes.c_void_p]
        lib.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                        ctypes.c_int, ctypes.c_void_p]
        lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        lib.cudaGetErrorString.restype = ctypes.c_char_p
        return lib, name
    raise OSError("no CUDA runtime library found (tried {}). Set R1_CUDART to its path."
                  .format(", ".join(tried)))


class TrtRuntime(object):
    def __init__(self, plan_path):
        import tensorrt as trt
        self.trt = trt
        self.cuda, self.cudart_name = load_cudart()
        self.name = "tensorrt {} {}".format(trt.__version__, os.path.basename(plan_path))
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._rt = trt.Runtime(self._logger)
        with open(plan_path, "rb") as f:
            self.engine = self._rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("TensorRT could not load {} (built by another TensorRT "
                               "version, or for another GPU?)".format(plan_path))
        self.ctx = self.engine.create_execution_context()
        self.v10 = hasattr(self.engine, "num_io_tensors")
        ins, outs = [], []
        if self.v10:
            for i in range(self.engine.num_io_tensors):
                n = self.engine.get_tensor_name(i)
                shape = tuple(self.engine.get_tensor_shape(n))
                dt = self.engine.get_tensor_dtype(n)
                mode = self.engine.get_tensor_mode(n)
                (ins if mode == trt.TensorIOMode.INPUT else outs).append((n, shape, dt))
        else:
            for i in range(self.engine.num_bindings):
                n = self.engine.get_binding_name(i)
                shape = tuple(self.engine.get_binding_shape(i))
                dt = self.engine.get_binding_dtype(i)
                (ins if self.engine.binding_is_input(i) else outs).append((n, shape, dt))
        if len(ins) != 1 or len(outs) != 1:
            raise RuntimeError("expected 1 input and 1 output, got {} and {}".format(
                len(ins), len(outs)))
        for n, shape, dt in ins + outs:
            if dt != trt.DataType.FLOAT:
                raise RuntimeError("tensor {} is not FP32 at the boundary".format(n))
            if any(int(s) < 0 for s in shape):
                raise RuntimeError("dynamic shapes are not supported ({} {})".format(n, shape))
        (self.in_name, self.in_shape, _), = ins
        (self.out_name, self.out_shape, _), = outs
        self.in_shape = tuple(int(s) for s in self.in_shape)
        self.out_shape = tuple(int(s) for s in self.out_shape)
        self._in_bytes = int(np.prod(self.in_shape)) * 4
        self._out_bytes = int(np.prod(self.out_shape)) * 4
        self.d_in, self.d_out, self.stream = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
        self._ck(self.cuda.cudaMalloc(ctypes.byref(self.d_in), self._in_bytes), "cudaMalloc in")
        self._ck(self.cuda.cudaMalloc(ctypes.byref(self.d_out), self._out_bytes), "cudaMalloc out")
        self._ck(self.cuda.cudaStreamCreate(ctypes.byref(self.stream)), "cudaStreamCreate")
        self.h_out = np.empty(self.out_shape, dtype=np.float32)
        if self.v10:
            self.ctx.set_tensor_address(self.in_name, self.d_in.value)
            self.ctx.set_tensor_address(self.out_name, self.d_out.value)
        else:
            self._bindings = [0] * self.engine.num_bindings
            self._bindings[self.engine.get_binding_index(self.in_name)] = self.d_in.value
            self._bindings[self.engine.get_binding_index(self.out_name)] = self.d_out.value

    def _ck(self, rc, what):
        if rc != 0:
            raise RuntimeError("{} failed: {} ({})".format(
                what, self.cuda.cudaGetErrorString(rc).decode(), rc))

    def infer(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.size * 4 != self._in_bytes:
            raise ValueError("input has {} floats, the engine wants {}".format(
                x.size, self._in_bytes // 4))
        s = self.stream
        self._ck(self.cuda.cudaMemcpyAsync(self.d_in, x.ctypes.data, self._in_bytes, H2D, s),
                 "copy in")
        ok = (self.ctx.execute_async_v3(s.value) if self.v10
              else self.ctx.execute_async_v2(self._bindings, s.value))
        if not ok:
            raise RuntimeError("TensorRT execute failed")
        self._ck(self.cuda.cudaMemcpyAsync(self.h_out.ctypes.data, self.d_out,
                                           self._out_bytes, D2H, s), "copy out")
        self._ck(self.cuda.cudaStreamSynchronize(s), "synchronize")
        return self.h_out.copy()

    def close(self):
        for p in (self.d_in, self.d_out):
            if p.value:
                self.cuda.cudaFree(p)
        if self.stream.value:
            self.cuda.cudaStreamDestroy(self.stream)
        self.d_in = self.d_out = self.stream = ctypes.c_void_p()


class OrtRuntime(object):
    def __init__(self, onnx_path, threads=2):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads)
        self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        i, o = self.sess.get_inputs()[0], self.sess.get_outputs()[0]
        self.in_name, self.out_name = i.name, o.name
        self.in_shape, self.out_shape = tuple(i.shape), tuple(o.shape)
        self.name = "onnxruntime {} {}".format(ort.__version__, os.path.basename(onnx_path))

    def infer(self, x):
        return self.sess.run([self.out_name], {self.in_name: np.asarray(x, np.float32)})[0]

    def close(self):
        pass


def make(path, threads=2):
    if str(path).endswith(".onnx"):
        return OrtRuntime(path, threads)
    return TrtRuntime(path)


def timed(rt, x, n=200, period_s=0.0):
    """Latency of rt.infer, in microseconds, n calls. `period_s` > 0 paces the calls
    like the real node (12.5 Hz = 0.08 s): on a Jetson a compact loop holds the GPU
    clock high, and the paced number is the one that describes the robot."""
    lat = []
    for _ in range(n):
        t0 = time.perf_counter()
        rt.infer(x)
        dt = time.perf_counter() - t0
        lat.append(dt * 1e6)
        if period_s > 0:
            time.sleep(max(0.0, period_s - dt))
    lat.sort()

    def pct(p):
        return lat[min(len(lat) - 1, int(p * (len(lat) - 1)))]
    return {"n": n, "p50": pct(0.5), "p95": pct(0.95), "p99": pct(0.99), "max": lat[-1]}
