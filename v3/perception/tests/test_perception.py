#!/usr/bin/env python3
"""r1_perception without a robot, a camera or a GPU: numpy only. The real-model check at
the end runs only where the model and onnxruntime are (the dev box after
models/fetch_yolox.sh); elsewhere it says it was skipped.

  python3 perception/tests/test_perception.py
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r1_perception import camera_model as C  # noqa: E402
from r1_perception import depth as D  # noqa: E402
from r1_perception import pipeline as PL  # noqa: E402
from r1_perception import sources as SRC  # noqa: E402
from r1_perception import tracker as T  # noqa: E402
from r1_perception import yolox as Y  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  ok    " + name)
    else:
        FAIL += 1
        print("  FAIL  {}  <- {}".format(name, detail))


print("== letterbox ==")
img = np.zeros((480, 640, 3), np.uint8)
x, r = Y.letterbox(img)
check("shape 1x3x416x416 float32", x.shape == (1, 3, 416, 416) and x.dtype == np.float32, x.shape)
check("ratio = 416/640", abs(r - 0.65) < 1e-9, r)
check("pasted top-left, padded with 114 below",
      x[0, 0, 0, 0] == 0.0 and x[0, 0, 415, 0] == 114.0 and x[0, 0, int(480 * r) - 1, 0] == 0.0,
      (x[0, 0, 0, 0], x[0, 0, 415, 0]))
x2, r2 = Y.letterbox(np.full((200, 100, 3), 7, np.uint8))
check("tall image: ratio from height", abs(r2 - 416 / 200.0) < 1e-9 and x2[0, 0, 0, 0] == 7.0, r2)

print("== decode + NMS ==")
raw = np.zeros((1, 3549, 85), np.float32)
raw[0, :, 4] = 0.0
# anchor 10 is grid (10, 0) at stride 8; offsets (0.5, 0.25), log size log(4) -> 32 px
raw[0, 10, :4] = [0.5, 0.25, math.log(4.0), math.log(2.0)]
raw[0, 10, 4] = 0.9
raw[0, 10, 5 + 56] = 0.8                           # chair
dec = Y.decode(raw)
check("grid decode: cx = (10 + 0.5) * 8", abs(dec[10, 0] - 84.0) < 1e-4, dec[10, :4])
check("grid decode: w = 4 * 8, h = 2 * 8", abs(dec[10, 2] - 32.0) < 1e-4 and abs(dec[10, 3] - 16.0) < 1e-4,
      dec[10, :4])
check("decode leaves the input alone", raw[0, 10, 0] == 0.5)
first13 = 52 * 52 + 26 * 26                        # first anchor at stride 32
raw[0, first13, :4] = [0.0, 0.0, 0.0, 0.0]
check("stride-32 block starts at row {}".format(first13),
      abs(Y.decode(raw)[first13, 2] - 32.0) < 1e-4)
det = Y.postprocess(raw, 0.5, 0.3)
check("score = objectness x class, boxes / ratio", len(det) == 1 and abs(det[0][4] - 0.72) < 1e-6
      and abs(det[0][0] - (84 - 16) / 0.5) < 1e-3 and det[0][5] == 56, det)
boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
keep = Y.nms(boxes, np.array([0.9, 0.8, 0.7], np.float32), 0.45)
check("NMS drops the overlapping weaker box", keep == [0, 2], keep)
raw2 = np.zeros((1, 3549, 85), np.float32)
for i, c in ((20, 0), (21, 56)):                   # neighbours, same place, two classes
    raw2[0, i, :4] = [0.0, 0.0, math.log(4.0), math.log(4.0)]
    raw2[0, i, 4] = 0.9
    raw2[0, i, 5 + c] = 0.9
check("class-aware NMS keeps a person and a chair that overlap",
      sorted(d[5] for d in Y.postprocess(raw2, 1.0, 0.3)) == [0, 56])
check("80 COCO classes, person first, chair 56", len(Y.COCO_CLASSES) == 80 and
      Y.COCO_CLASSES[0] == "person" and Y.COCO_CLASSES[56] == "chair")

print("== camera model ==")
fish = C.Fisheye(330.0, 330.0, 639.5, 359.5, (0.05, -0.01, 0.002, -0.0005), (1280, 720))
xs, ys = np.array([0.0, 0.3, -1.2, 2.5]), np.array([0.0, -0.4, 0.8, 1.1])
u, v = fish.project(xs, ys)
xb, yb = fish.unproject(u, v)
check("fisheye project/unproject round trip (to 68 deg off axis)",
      np.allclose(xb, xs, atol=1e-6) and np.allclose(yb, ys, atol=1e-6), (xb, yb))
th = math.radians(75.0)
uu, _ = C.Fisheye(330, 330, 639.5, 359.5).project(math.tan(th), 0.0)
check("equidistant: 75 deg lands at f * theta from the center",
      abs((uu - 639.5) - 330 * th) < 1e-6, uu)
rect = C.Rectified(640, 480, 100.0)
check("rectified focal from the field of view", abs(rect.f - 320.0 / math.tan(math.radians(50))) < 1e-9,
      rect.f)
mx, my = rect.maps(fish)
check("the rectified center samples the fisheye center",
      abs(mx[240, 320] - 639.5) < 2.0 and abs(my[240, 320] - 359.5) < 2.0, (mx[240, 320], my[240, 320]))
level = C.Mount(1.2, 0.0, 0.0)
b_r, _ = level.bearing_elev_deg(*rect.ray(639, 239.5))
b_l, _ = level.bearing_elev_deg(*rect.ray(0, 239.5))
check("bearing is LEFT-positive: right edge -50, left edge +50 (100 deg view)",
      abs(b_r + 50.0) < 0.2 and abs(b_l - 50.0) < 0.2, (b_l, b_r))
_, e_up = level.bearing_elev_deg(*rect.ray(319.5, 0))
check("elevation is up-positive", e_up > 30.0, e_up)
down = C.Mount(1.2, 20.0, 0.0)
_, e0 = down.bearing_elev_deg(0.0, 0.0)
check("a camera pitched 20 deg down sees its optical axis at -20 deg", abs(e0 + 20.0) < 1e-9, e0)
b_p, _ = down.bearing_elev_deg(math.tan(math.radians(-30)), 0.0)
check("pitch does not change the bearing of a point on the image's center row much",
      abs(b_p - 30.0) < 2.5, b_p)
yaw = C.Mount(1.2, 0.0, 10.0)
check("a head turned 10 deg left adds 10 deg", abs(yaw.bearing_elev_deg(0, 0)[0] - 10.0) < 1e-9)
check("range at 2 m depth straight ahead is 2 m", abs(level.range_m(0.0, 0.0, 2.0) - 2.0) < 1e-12)
check("range off axis is horizontal, not optical depth",
      abs(level.range_m(1.0, 0.0, 2.0) - math.hypot(2, 2)) < 1e-9)

print("== depth ==")
dimg = np.full((480, 640), 3.0, np.float32)
dimg[100:300, 200:400] = 1.5
dimg[100:300, 200:230] = 0.0                      # no return on the left edge of the object
check("box median uses the central half", abs(D.box_depth(dimg, (200, 100, 400, 300)) - 1.5) < 1e-9)
check("background at the corners does not count",
      abs(D.box_depth(dimg, (150, 50, 450, 350)) - 1.5) < 1e-9)
empty = np.zeros((480, 640), np.float32)
check("no returns -> None, not 0", D.box_depth(empty, (200, 100, 400, 300)) is None)
# Corridor: a camera 1.2 m up looking 25 deg down at a flat floor sees floor everywhere
# below the horizon. None of it may count as an obstacle.
mount = C.Mount(1.2, 25.0, 0.0)
vv, uu2 = np.mgrid[0:480, 0:640]
xn, yn = rect.ray(uu2, vv)
fx, fl, fu = mount.to_body(xn, yn, np.ones_like(xn))
dz = np.where(fu - mount.height_m < -1e-6, mount.height_m / np.maximum(1e-9, -(fu - mount.height_m)),
              np.inf)
floor = np.where(np.isfinite(dz), dz, 0.0).astype(np.float32)
obs, near_m, n = D.corridor(floor, rect, mount, 0.6)
check("a bare floor is not an obstacle at any pitch", not obs and n == 0, (obs, near_m, n))
box = floor.copy()
wall_z = 0.5 / np.maximum(1e-9, fx)               # a wall 0.5 m ahead, everywhere above the floor
wall = (fx > 0) & (wall_z < np.where(floor > 0, floor, np.inf))
box[wall] = wall_z[wall]
obs, near_m, n = D.corridor(box, rect, mount, 0.6)
check("a wall 0.5 m ahead is an obstacle at 0.5 m", obs and abs(near_m - 0.5) < 0.05, (obs, near_m, n))

print("== tracker ==")
tr = T.Tracker(confirm_frames=3)
a = (100, 100, 200, 300, 0.9, 56)
hits = [tr.update([a])[0]["hits"] for _ in range(3)]
check("hits count consecutive frames", hits == [1, 2, 3], hits)
check("confirmed at 3", tr.update([a])[0]["confirmed"])
tr.update([])
check("one missed frame resets the count", tr.update([a])[0]["hits"] == 1)
two = tr.update([a, (400, 100, 500, 300, 0.8, 56)])
check("a second chair is a second track", len(set(d["id"] for d in two)) == 2, two)
moved = tr.update([(110, 105, 210, 305, 0.9, 56)])[0]
check("a box that moved a little keeps its id", moved["id"] == two[0]["id"], (moved, two))
other = tr.update([(110, 105, 210, 305, 0.9, 0)])[0]
check("a different class is never the same track", other["id"] != two[0]["id"])

print("== pipeline (fake engine) ==")


class FakeRT(object):
    name = "fake"

    def infer(self, x):
        out = np.zeros((1, 3549, 85), np.float32)
        # grid (5, 30) at stride 8: centered at (40, 240) in the 416 input = (62, 369) in
        # the 640 x 480 image -- well left of center
        i = 30 * 52 + 5
        out[0, i, :4] = [0.0, 0.0, math.log(10.0), math.log(20.0)]    # 80 x 160 input px
        out[0, i, 4] = 0.95
        out[0, i, 5] = 0.9                                           # a person
        return out


cfg = {"cam_fx": 320 / math.tan(math.radians(50)), "cam_fy": 320 / math.tan(math.radians(50)),
       "cam_cx": 319.5, "cam_cy": 239.5, "cam_width": 640, "cam_height": 480, "rectify": 0,
       "cam_height_m": 1.2, "cam_pitch_deg": 0.0, "person_stop_m": 1.2}
pipe = PL.Pipeline(cfg, FakeRT())
dep = np.full((480, 640), 1.5, np.float32)
msg, _, _ = pipe.process(np.zeros((480, 640, 3), np.uint8), dep, t_cap=123.0)
d0 = msg["dets"][0]
check("message: one person, left of center, range from depth",
      d0["cls"] == "person" and d0["bearing_deg"] > 30 and d0["range_m"] > 1.5, d0)
check("guard: a person ~2 m away is not inside 1.2 m, a wall 1.5 m ahead is not inside 0.6",
      not msg["guard"]["person_near"] and not msg["guard"]["obstacle"], msg["guard"])
check("t_cap and version carried", msg["t_cap"] == 123.0 and msg["v"] == PL.MSG_VERSION)
msg2, _, _ = pipe.process(np.zeros((480, 640, 3), np.uint8), np.full((480, 640), 0.7, np.float32))
check("guard: the same person at 0.7 m depth is near", msg2["guard"]["person_near"], msg2["guard"])
msg3, _, _ = pipe.process(np.zeros((480, 640, 3), np.uint8), None)
check("no depth: ranges are None, the guard says so", msg3["dets"][0]["range_m"] is None
      and msg3["guard"]["depth_ok"] is False, msg3)
calls = []
real_infer, pipe.rt.infer = pipe.rt.infer, lambda x: calls.append(1) or real_infer(x)
prep = pipe.prepare(np.zeros((480, 640, 3), np.uint8), dep, t_cap=124.0)
n_after_prepare = len(calls)
msg4, _, _ = pipe.finish(prep)
check("prepare is CPU only (the node runs it a step early); finish = process",
      n_after_prepare == 0 and len(calls) == 1 and msg4["t_cap"] == 124.0
      and msg4["dets"][0]["cls"] == d0["cls"] and set(msg4["timing"]) == set(msg["timing"]), (n_after_prepare, len(calls)))

print("== V3 proximity ==")


def wall_at(dist_m, base=floor):
    """The floor scene with a wall `dist_m` ahead, everywhere above the floor."""
    out = base.copy()
    wz = dist_m / np.maximum(1e-9, fx)
    hit = (fx > 0) & (wz < np.where(base > 0, base, np.inf))
    out[hit] = wz[hit]
    return out


for dist in (0.4, 0.6, 1.0):
    fwd, valid = D.corridor_points(wall_at(dist), rect, mount)
    k = D.kth_nearest(fwd, 30)
    check("wall {:.1f} m ahead: obstacle_m {:.1f} m".format(dist, dist),
          k is not None and abs(k - dist) < 0.03, (k, fwd.size))
fwd, valid = D.corridor_points(floor, rect, mount)
check("bare floor: no corridor points, no obstacle_m, depth valid below the horizon",
      fwd.size == 0 and D.kth_nearest(fwd, 30) is None and 0.3 < valid < 1.0, (fwd.size, valid))
far = wall_at(2.0)
speck = far.copy()
speck[236:240, 316:320] = 0.3                      # 16 pixels -> 1 sampled point at 0.3 m
fwd, _ = D.corridor_points(speck, rect, mount)
check("a speck nearer than k points does not make an obstacle_m: the wall behind it does",
      abs(D.kth_nearest(fwd, 30) - 2.0) < 0.03, D.kth_nearest(fwd, 30))
small = far.copy()
small[224:240, 304:336] = 0.5                     # 4 x 8 = 32 sampled points, 0.5 m depth
fwd, _ = D.corridor_points(small, rect, mount)
_, pct5, _ = D.corridor(small, rect, mount, 0.6)
check("a 32-point object in front of a 2 m wall: obstacle_m is the object (k = 30), "
      "where the 5th percentile of the corridor reads the wall",
      D.kth_nearest(fwd, 30) < 0.6 and pct5 > 1.5, (D.kth_nearest(fwd, 30), pct5, fwd.size))
check("guard's corridor() is unchanged by the refactor: wall 0.5 m -> obstacle at 0.5",
      D.corridor(wall_at(0.5), rect, mount, 0.6)[0]
      and abs(D.corridor(wall_at(0.5), rect, mount, 0.6)[1] - 0.5) < 0.05)
hole = far.copy()
hole[120:480, 160:480] = 0.0                       # too near for stereo: no return
check("center_valid sees a hole in the middle that depth_valid dilutes",
      D.valid_fraction(hole, (0.25, 0.25, 0.75, 1.0)) < 0.05 < 0.5 < D.valid_fraction(hole),
      (D.valid_fraction(hole, (0.25, 0.25, 0.75, 1.0)), D.valid_fraction(hole)))
check("no returns at all: valid fraction 0", D.valid_fraction(empty) == 0.0)

m5, _, _ = pipe.process(np.zeros((480, 640, 3), np.uint8), np.full((480, 640), 0.55, np.float32))
pr = m5["proximity"]
check("proximity: the person's range, one person, a flat 0.55 m depth is an obstacle at 0.55",
      pr["person_m"] == m5["dets"][0]["range_m"] and pr["persons"] == 1
      and abs(pr["obstacle_m"] - 0.55) < 0.03 and pr["depth_ok"] and pr["v"] == PL.PROX_VERSION, pr)
m6, _, _ = pipe.process(np.zeros((480, 640, 3), np.uint8), None)
check("proximity without depth: no distances, depth_ok false, the person still counted",
      m6["proximity"]["person_m"] is None and m6["proximity"]["obstacle_m"] is None
      and not m6["proximity"]["depth_ok"] and m6["proximity"]["persons"] == 1, m6["proximity"])
check("a 234 px person box (of 480) is not near at person_box_frac 0.8",
      not m6["proximity"]["person_near_box"])
tall = PL.Pipeline(dict(cfg, person_box_frac=0.45), FakeRT())   # the box is 234 px, clipped
m7, _, _ = tall.process(np.zeros((480, 640, 3), np.uint8), None)
check("...and is at person_box_frac 0.45: near, distance unknown",
      m7["proximity"]["person_near_box"] and m7["proximity"]["person_m"] is None, m7["proximity"])
arr = PL.proximity_array(m6["proximity"], 9, 100.25, 100.5)
check("proximity array: the layout's length, version first, NaN for no reading",
      len(arr) == len(PL.PROX_LAYOUT) and arr[0] == PL.PROX_VERSION
      and math.isnan(arr[PL.PROX_LAYOUT.index("obstacle_m")]), arr)
back = PL.parse_proximity(arr)
check("proximity array round trip", back["seq"] == 9 and back["t_cap_mono"] == 100.25
      and back["person_m"] is None and back["depth_ok"] is False and back["persons"] == 1, back)
arr5 = PL.proximity_array(pr, 10, 1.0, 2.0)
check("...with readings", abs(PL.parse_proximity(arr5)["obstacle_m"] - pr["obstacle_m"]) < 1e-12)
try:
    PL.parse_proximity([2.0] + arr[1:])
    check("a wrong version is refused", False)
except ValueError:
    check("a wrong version is refused", True)
try:
    PL.parse_proximity(arr[:-1])
    check("a short array is refused", False)
except ValueError:
    check("a short array is refused", True)
cfg_file = PL.read_flat_yaml(ROOT / "config" / "perception.yaml")
check("the shipped config's center region parses",
      tuple(float(x) for x in str(cfg_file["proximity_center_region"]).split(",")) == (0.25, 0.25, 0.75, 1.0),
      cfg_file.get("proximity_center_region"))

print("== stereo source (fake decoder, fake depth device) ==")
import os  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

pin = C.Rectified(544, 448, fx=300.0, fy=310.0, cx=270.0, cy=220.0)
xn, yn = pin.ray(570.0, 530.0)
check("pinhole intrinsics: rays use fx, fy, cx, cy", abs(xn - 1.0) < 1e-9 and abs(yn - 1.0) < 1e-9,
      (xn, yn))
check("pinhole hfov from fx and cx", abs(pin.hfov_deg - math.degrees(
    math.atan(270 / 300.0) + math.atan(273 / 300.0))) < 1e-9, pin.hfov_deg)
raw = np.array([[0, 1500, 65535, 800]], np.uint16)
check("depth uint16 mm -> m, 0 and 65535 are no return",
      SRC.depth_from_u16mm(raw).tolist() == [[0.0, 1.5, 0.0, 0.800000011920929]],
      SRC.depth_from_u16mm(raw))
argv, how = SRC.StereoSource.gst_command({"stereo_rgb_port": 5002, "cam_width": 544,
                                          "cam_height": 448, "stereo_decoder": "avdec_h264"})
txt = " ".join(argv)
check("gst pipeline: port, H264, decoder, BGR at the configured size, raw to stdout",
      argv[:2] == ["gst-launch-1.0", "-q"] and "port=5002" in txt and "encoding-name=H264" in txt
      and "avdec_h264" in txt and "width=544,height=448" in txt and "fdsink fd=1" in txt, txt)
argv_j, how_j = SRC.StereoSource.gst_command({"stereo_codec": "jpeg", "stereo_rgb_port": 5004})
check("gst pipeline: the raw-image (JPEG) ports", how_j == "jpeg" and "rtpjpegdepay" in " ".join(argv_j))

tmp = tempfile.mkdtemp()
dev = os.path.join(tmp, "video-dep")
dframe = np.full((4, 6), 1200, np.uint16)
dframe[0, 0] = 0
dframe.tofile(dev)                       # one whole frame, then EOF (= no new frame)
fake = [sys.executable, "-c",
        "import sys, time\n"
        "for i in range(3):\n"
        "    sys.stdout.buffer.write(bytes([i]) * (6 * 4 * 3)); sys.stdout.flush(); time.sleep(0.05)\n"
        "time.sleep(1)\n"]
st = SRC.StereoSource({"cam_width": 6, "cam_height": 4, "stereo_depth_device": dev,
                       "stereo_depth_width": 6, "stereo_depth_height": 4,
                       "stereo_depth_max_age_s": 5.0}, rgb_cmd=fake)
f, t_end = None, time.time() + 3
while time.time() < t_end and (st.latest.count < 3 or st.depth_frames < 1):
    time.sleep(0.02)
f = st.take()
check("stereo: whole BGR frames from the decoder pipe, the newest one",
      f is not None and f[0].shape == (4, 6, 3) and int(f[0][0, 0, 0]) == 2, st.status())
check("stereo: depth paired, in metres, no-return pixels 0",
      f is not None and f[1] is not None and abs(float(f[1][1, 1]) - 1.2) < 1e-6
      and float(f[1][0, 0]) == 0.0, None if f is None else f[1])
check("stereo: a frame is never handed out twice", st.take() is None)
st._t_depth -= 10.0
st.latest.put_rgb(np.zeros((4, 6, 3), np.uint8), time.time())
f2 = st.take()
check("stereo: stale depth is not paired (guard says depth_ok false)", f2 is not None and f2[1] is None)
st.close()
bad = SRC.StereoSource({"cam_width": 6, "cam_height": 4, "stereo_depth_device": ""},
                       rgb_cmd=[sys.executable, "-c", "import sys; sys.stderr.write('no element avdec_h264\\n')"])
time.sleep(0.5)
try:
    bad.take()
    said = ""
except RuntimeError as exc:
    said = str(exc)
check("stereo: a decoder that dies is an error with its own message", "avdec_h264" in said, said)
bad.close()

print("== the real model (if present) ==")
onnx = ROOT / "models" / "yolox_nano.onnx"
dog = ROOT / "models" / "dog.jpg"
try:
    import onnxruntime  # noqa: F401
    import cv2
    have = True
except ImportError:
    have = False
if onnx.is_file() and dog.is_file() and have:
    from r1_perception import runtime as RT
    rt = RT.OrtRuntime(str(onnx))
    im = cv2.imread(str(dog))
    xx, rr = Y.letterbox(im)
    got = [(Y.COCO_CLASSES[d[5]], round(d[4], 2)) for d in
           Y.postprocess(rt.infer(xx), rr, 0.5, image_wh=(im.shape[1], im.shape[0]))]
    check("YOLOX-Nano on YOLOX's own sample: dog, bicycle, car",
          sorted(c for c, _ in got) == ["bicycle", "car", "dog"], got)
else:
    print("  skip  the real model: needs models/yolox_nano.onnx + models/dog.jpg "
          "(models/fetch_yolox.sh), onnxruntime and OpenCV")

print("-" * 50)
print("pass {}  fail {}".format(PASS, FAIL))
sys.exit(1 if FAIL else 0)
