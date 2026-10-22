"""YOLOX-Nano around the network: the letterbox in front of it and the decode + NMS
after it. numpy only (cv2 is used for the resize when it is there), Python 3.8 clean.

The released ONNX (Megvii YOLOX 0.1.1rc0, Apache-2.0) is static: images [1,3,416,416]
-> output [1,3549,85]. 3549 = 52^2 + 26^2 + 13^2 anchors at strides 8, 16, 32. Each row
is (dx, dy, log w, log h, objectness, 80 class scores); objectness and class scores are
already sigmoid, the box is NOT decoded (the export had decode_in_inference off: the
graph has Sigmoid and no Exp). So the grid decode below is part of the model's
definition, not a tuning choice, and it matches YOLOX's demo_postprocess exactly.

Input convention, also the model's: BGR, 0..255 as float, no mean/std, the image
resized by one ratio and pasted at the top-left of a 114-gray canvas.
"""
from __future__ import annotations

import numpy as np

INPUT_HW = (416, 416)
STRIDES = (8, 16, 32)
PAD_VALUE = 114.0

COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush")
PERSON = 0

try:  # the robot's JetPack OpenCV; the dev box tests run without it
    import cv2  # noqa: F401
    _HAVE_CV2 = True
except ImportError:  # pragma: no cover - depends on the host
    _HAVE_CV2 = False


def _resize(img, w, h):
    """Bilinear resize. cv2 when present (what the robot runs); otherwise a numpy
    bilinear with the same pixel-center convention, close enough for tests -- the
    parity gate compares engine against ONNX on the SAME tensor, so the resize never
    enters it."""
    if _HAVE_CV2:
        import cv2
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    H, W = img.shape[:2]
    ys = (np.arange(h) + 0.5) * (H / float(h)) - 0.5
    xs = (np.arange(w) + 0.5) * (W / float(w)) - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, H - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)
    wy = np.clip(ys - y0, 0.0, 1.0)[:, None, None]
    wx = np.clip(xs - x0, 0.0, 1.0)[None, :, None]
    f = img.astype(np.float32)
    top = f[y0][:, x0] * (1 - wx) + f[y0][:, x1] * wx
    bot = f[y1][:, x0] * (1 - wx) + f[y1][:, x1] * wx
    return np.clip(top * (1 - wy) + bot * wy + 0.5, 0, 255).astype(np.uint8)


def letterbox(bgr, input_hw=INPUT_HW):
    """HxWx3 uint8 BGR -> (1x3xHxW float32 tensor, ratio). Boxes in the network's
    frame divide by `ratio` to get back to the image."""
    ih, iw = input_hw
    h, w = bgr.shape[:2]
    r = min(ih / float(h), iw / float(w))
    nh, nw = int(h * r), int(w * r)
    canvas = np.full((ih, iw, 3), PAD_VALUE, dtype=np.float32)
    canvas[:nh, :nw] = _resize(bgr, nw, nh).astype(np.float32)
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None]), r


def _grids(input_hw=INPUT_HW, strides=STRIDES):
    ih, iw = input_hw
    grids, scale = [], []
    for s in strides:
        hs, ws = ih // s, iw // s
        xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        scale.append(np.full((hs * ws, 1), s))
    return (np.concatenate(grids).astype(np.float32),
            np.concatenate(scale).astype(np.float32))


_GRID_CACHE = {}


def decode(raw, input_hw=INPUT_HW):
    """[1,N,85] or [N,85] raw output -> [N,85] with boxes as (cx, cy, w, h) in the
    network's input pixels. Does not modify `raw`."""
    out = np.array(raw, dtype=np.float32).reshape(-1, 5 + len(COCO_CLASSES))
    key = tuple(input_hw)
    if key not in _GRID_CACHE:
        _GRID_CACHE[key] = _grids(input_hw)
    grid, stride = _GRID_CACHE[key]
    if out.shape[0] != grid.shape[0]:
        raise ValueError("output has {} rows, the {} input expects {}".format(
            out.shape[0], input_hw, grid.shape[0]))
    out[:, :2] = (out[:, :2] + grid) * stride
    out[:, 2:4] = np.exp(out[:, 2:4]) * stride
    return out


def nms(boxes, scores, iou_thr):
    """Greedy NMS on xyxy boxes; returns kept indices, best first."""
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = np.maximum(0.0, x1 - x0) * np.maximum(0.0, y1 - y0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx0 = np.maximum(x0[i], x0[order[1:]])
        yy0 = np.maximum(y0[i], y0[order[1:]])
        xx1 = np.minimum(x1[i], x1[order[1:]])
        yy1 = np.minimum(y1[i], y1[order[1:]])
        inter = np.maximum(0.0, xx1 - xx0) * np.maximum(0.0, yy1 - yy0)
        iou = inter / np.maximum(1e-9, area[i] + area[order[1:]] - inter)
        order = order[1:][iou <= iou_thr]
    return keep


def postprocess(raw, ratio, score_thr=0.25, nms_thr=0.45, input_hw=INPUT_HW,
                image_wh=None):
    """Raw network output -> list of (x0, y0, x1, y1, score, class_id) in the ORIGINAL
    image's pixels, best first. Class-aware NMS (YOLOX's multiclass_nms, class-aware
    branch): a chair and a person may overlap, two chairs may not."""
    pred = decode(raw, input_hw)
    cls = pred[:, 5:]
    cid = cls.argmax(1)
    score = pred[:, 4] * cls[np.arange(cls.shape[0]), cid]
    m = score >= score_thr
    if not m.any():
        return []
    p, cid, score = pred[m], cid[m], score[m]
    xyxy = np.stack([p[:, 0] - p[:, 2] / 2, p[:, 1] - p[:, 3] / 2,
                     p[:, 0] + p[:, 2] / 2, p[:, 1] + p[:, 3] / 2], 1) / ratio
    if image_wh is not None:
        w, h = image_wh
        xyxy[:, 0::2] = np.clip(xyxy[:, 0::2], 0, w - 1)
        xyxy[:, 1::2] = np.clip(xyxy[:, 1::2], 0, h - 1)
    out = []
    for c in np.unique(cid):
        idx = np.where(cid == c)[0]
        for k in nms(xyxy[idx], score[idx], nms_thr):
            j = idx[k]
            out.append((float(xyxy[j, 0]), float(xyxy[j, 1]), float(xyxy[j, 2]),
                        float(xyxy[j, 3]), float(score[j]), int(c)))
    out.sort(key=lambda d: -d[4])
    return out
