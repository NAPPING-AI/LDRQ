#!/usr/bin/env python3
"""Parity fixture for the YOLOX engine, in the walking policy's R1FX format, so the
SAME r1_parity_check gates both networks.

Dev box (it needs onnxruntime, which the robot does not have):

  python3 perception/tools/make_fixture.py --onnx perception/models/yolox_nano.onnx \\
      --images outputs/stageE/frames/*.jpg --out perception/models/yolox_fixture.bin

Each image goes through the deployed preprocessing (rectify if --cfg is given, then
the letterbox), and onnxruntime on the CPU writes the reference output. Use frames
from the robot's own camera: the fixture is the definition of "the engine computes
what the ONNX computes" on the inputs it will actually see.

A sample is 519168 + 301665 floats (3.3 MB), so the default 8 samples is 26 MB: the
fixture is a build artifact (gitignored), its sha256 goes next to the model's.

  R1FX | u32 version=1 | u32 n | u32 in_dim | u32 out_dim | f32 in[n*in] | f32 out[n*out]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import pathlib
import struct
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from r1_perception import pipeline as PL  # noqa: E402
from r1_perception import runtime as RT  # noqa: E402
from r1_perception import yolox as Y  # noqa: E402


def load_image(path):
    p = str(path)
    if p.endswith(".npy"):
        return np.load(p)
    try:
        import cv2
    except ImportError:
        sys.exit("reading {} needs OpenCV (or pass .npy BGR arrays)".format(p))
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit("cannot read " + p)
    return img


def write_fixture(path, inputs, outputs):
    inputs = np.ascontiguousarray(inputs, dtype=np.float32)
    outputs = np.ascontiguousarray(outputs, dtype=np.float32)
    n = inputs.shape[0]
    with open(str(path), "wb") as f:
        f.write(b"R1FX")
        f.write(struct.pack("<IIII", 1, n, inputs[0].size, outputs[0].size))
        f.write(inputs.tobytes())
        f.write(outputs.tobytes())


def read_fixture(path):
    with open(str(path), "rb") as f:
        if f.read(4) != b"R1FX":
            raise ValueError("bad magic in {}".format(path))
        ver, n, di, do = struct.unpack("<IIII", f.read(16))
        if ver != 1:
            raise ValueError("unsupported fixture version {}".format(ver))
        x = np.frombuffer(f.read(n * di * 4), dtype=np.float32).reshape(n, di)
        y = np.frombuffer(f.read(n * do * 4), dtype=np.float32).reshape(n, do)
    return x, y


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--cfg", default=None, help="perception.yaml: rectify like the robot")
    ap.add_argument("--max", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    paths = []
    for p in a.images:
        paths += sorted(glob.glob(p)) or [p]
    paths = paths[:a.max]
    rt = RT.OrtRuntime(a.onnx)
    pipe = PL.Pipeline(PL.read_flat_yaml(a.cfg), rt) if a.cfg else None
    xs, ys = [], []
    for p in paths:
        img = load_image(p)
        if pipe is not None:
            img, _ = pipe.rectify(img)
        x, ratio = Y.letterbox(img)
        y = rt.infer(x)
        n = len(Y.postprocess(y, ratio, 0.3))
        print("{}: {} detections at 0.3".format(p, n))
        xs.append(x.reshape(-1))
        ys.append(y.reshape(-1))
    write_fixture(a.out, np.stack(xs), np.stack(ys))
    h = hashlib.sha256(pathlib.Path(a.out).read_bytes()).hexdigest()
    print("wrote {}: {} samples, in {} out {}, sha256 {}".format(
        a.out, len(xs), xs[0].size, ys[0].size, h))
    return 0


if __name__ == "__main__":
    sys.exit(main())
