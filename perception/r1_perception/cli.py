"""r1_perception_cli.py: everything stage E does with the camera and the engine.

  probe          FIRST robot step: what camera is there, how it is exposed, which
                 Python pieces exist. Moves nothing, needs no stack.
  run            the node (ROS 2): /r1_perception/detections at ~12.5 Hz
  engine-check   the Python runtime -- the path the node runs -- against the parity
                 fixture: the same gate r1_parity_check applies to the C++ path
  compare        FP32 vs FP16 engine on real frames: latency (paced and compact) and
                 how the detections differ
  capture        save frames: raw (for calibration) and rectified + depth + detections
                 (for the detection and AprilTag evaluations)
  look           one-shot: print what the detector sees now (via the running node)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

import numpy as np

from . import pipeline as PL
from . import runtime as RT
from . import yolox as Y

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_CFG = ROOT / "config" / "perception.yaml"
MODELS = ROOT / "models"


def load_cfg(path):
    return PL.read_flat_yaml(path) if pathlib.Path(path).is_file() else {}


def _engine_default():
    # FP16 first: on the robot (10-14) it is the engine that passed its gate and the
    # coexistence test; FP32 missed the raw gate on TensorRT 8.5 and is 1.6x slower.
    for name in ("yolox_fp16.plan", "yolox_fp32.plan", "yolox_nano.onnx"):
        p = MODELS / name
        if p.is_file():
            return str(p)
    return str(MODELS / "yolox_fp16.plan")


# ------------------------------------------------------------------ probe
def probe(args):
    """Report, do not guess. Each line is a fact with where it came from."""
    out = pathlib.Path(os.path.expanduser(args.out))
    out.mkdir(parents=True, exist_ok=True)
    rep = {"when": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    # Robot, 10-14: with stdout piped into tee, a hang + Ctrl-C lost every line (the
    # buffer died with the process). Line-buffer, and save probe.json after each part.
    sys.stdout.reconfigure(line_buffering=True)

    print("-- python pieces")
    for mod in ("numpy", "cv2", "tensorrt", "rclpy", "sensor_msgs.msg"):
        try:
            m = __import__(mod, fromlist=["x"])
            v = getattr(m, "__version__", "") or ""
            print("  ok    {} {}".format(mod, v))
            rep[mod] = v or True
        except Exception as exc:  # noqa: BLE001
            print("  MISSING {} ({})".format(mod, exc))
            rep[mod] = None
    try:
        _, name = RT.load_cudart()
        print("  ok    libcudart: {}".format(name))
        rep["cudart"] = name
    except OSError as exc:
        print("  MISSING libcudart ({})".format(exc))
        rep["cudart"] = None

    print("-- V4L2 devices")
    vids = sorted(glob.glob("/dev/video*"))
    rep["video_devices"] = vids
    print("  " + (", ".join(vids) if vids else "none"))
    capture = {}           # device -> can it capture (None = v4l2-ctl could not say)
    for v in vids:
        if not shutil.which("v4l2-ctl"):
            break
        print("  {}: v4l2-ctl ...".format(v))
        txt = _run_bounded(["v4l2-ctl", "-d", v, "--info", "--list-formats-ext"], 10)
        f = out / ("v4l2_" + os.path.basename(v) + ".txt")
        f.write_text(txt)
        card = [ln.strip() for ln in txt.splitlines() if "Card type" in ln]
        dcaps = txt.split("Device Caps", 1)[1].split("ioctl", 1)[0] if "Device Caps" in txt else ""
        capture[v] = ("Video Capture" in dcaps) if dcaps else None
        print("  {}: {} | capture {} (in {})".format(
            v, card[0] if card else "?", {True: "yes", False: "NO", None: "?"}[capture[v]], f))
    rep["v4l2_capture"] = capture
    _save(out, rep)
    if vids and rep.get("cv2"):
        # Robot, 10-14: OpenCV's read() on a v4l2loopback with no writer never returned,
        # so each device is read in a child that is killed after 8 s.
        for v in vids:
            if capture.get(v) is False:
                print("  {}: skipped by OpenCV (an output-only device)".format(v))
                continue
            p = out / ("frame_" + os.path.basename(v) + ".jpg")
            print("  {}: OpenCV read (8 s at most) ...".format(v))
            txt = _run_bounded([sys.executable, "-c", _CV_READ, v, str(p)], 8)
            print("  {}: {}".format(v, txt.strip() or "?"))
            rep.setdefault("v4l2_opencv", {})[v] = txt.strip()
    _save(out, rep)

    print("-- USB")
    if shutil.which("lsusb"):
        r = subprocess.run(["lsusb"], stdout=subprocess.PIPE, universal_newlines=True)
        rep["lsusb"] = r.stdout.splitlines()
        for ln in r.stdout.splitlines():
            print("  " + ln)

    _save(out, rep)
    rep["stereo"] = _probe_stereo(args, out)
    _save(out, rep)
    print("-- ROS image topics ({} s of discovery)".format(args.ros_wait))
    if rep.get("rclpy") and rep.get("sensor_msgs.msg"):
        rep["ros"] = _probe_ros(args, out)
    else:
        print("  skipped: rclpy / sensor_msgs not importable (source env.sh first)")
    _save(out, rep)
    print("\nwrote {} -- bring the whole folder back to the dev box".format(out))
    return 0


_CV_READ = """
import sys, cv2
cap = cv2.VideoCapture(sys.argv[1])
ok, img = cap.read() if cap.isOpened() else (False, None)
if ok:
    cv2.imwrite(sys.argv[2], img)
    print("OpenCV reads {}x{} -> {}".format(img.shape[1], img.shape[0], sys.argv[2]))
else:
    print("OpenCV: " + ("opened, no frame" if cap.isOpened() else "cannot open"))
"""


def _save(out, rep):
    (out / "probe.json").write_text(json.dumps(rep, indent=1, default=str))


def _run_bounded(cmd, timeout_s):
    """stdout+stderr of cmd, or what it printed plus a TIMEOUT line. Never waits on a
    child stuck in the kernel for more than 2 s after killing it."""
    pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True)
    try:
        return pr.communicate(timeout=timeout_s)[0]
    except subprocess.TimeoutExpired:
        pr.kill()
        try:
            txt = pr.communicate(timeout=2)[0] or ""
        except subprocess.TimeoutExpired:
            txt = ""
        return txt + "TIMEOUT after {} s (killed)\n".format(timeout_s)


def _probe_stereo(args, out):
    """The head's stereo module (Unitree's video stream service page): what this
    computer can decode, what arrives on each UDP port, and whether depth is flowing.
    Needs APP video_hub OFF + Stereo patch PC1 ON, and `sudo unitree_depth_start`."""
    from . import sources as SRC
    res = {}
    print("-- stereo stream (UDP RTP from the head; depth in {})".format(args.depth_device))
    if not shutil.which("gst-launch-1.0"):
        print("  MISSING gst-launch-1.0: the stereo source cannot decode anything")
        return {"gst": None}
    els = {e: SRC.gst_has(e) for e in ("rtph264depay", "h264parse", "avdec_h264", "nvv4l2decoder",
                                       "nvvidconv", "rtpjpegdepay", "jpegdec", "jpegenc")}
    res["gst_elements"] = els
    print("  gstreamer: " + ", ".join("{} {}".format(k, "ok" if v else "MISSING")
                                      for k, v in els.items()))
    try:
        import cv2
        gs = [ln.strip() for ln in cv2.getBuildInformation().splitlines() if "GStreamer" in ln]
        res["cv2_gstreamer"] = gs[0] if gs else "?"
        print("  OpenCV: " + res["cv2_gstreamer"])
    except Exception as exc:  # noqa: BLE001
        res["cv2_gstreamer"] = "cv2 unavailable: {}".format(exc)
    dec = "nvv4l2decoder" if els["nvv4l2decoder"] and els["nvvidconv"] else "avdec_h264"
    res["decoder"] = dec
    for port, name in ((5001, "img"), (5002, "left"), (5003, "right")):
        loc = str(out / "stereo_{}_%05d.jpg".format(port))
        pipe = ("udpsrc port={} caps=application/x-rtp,media=video,clock-rate=90000,"
                "encoding-name=H264,payload=96 ! rtph264depay ! {} ! videoconvert ! "
                "jpegenc ! multifilesink location={} max-files=2").format(
                    port, SRC.StereoSource.DECODERS[dec], loc)
        print("  udp {} ({}): listening {:.0f} s ...".format(port, name, args.stereo_wait))
        for old in glob.glob(str(out / "stereo_{}_*.jpg".format(port))):
            os.remove(old)
        txt = _run_bounded(["gst-launch-1.0", "-v"] + pipe.split(), args.stereo_wait)
        files = sorted(glob.glob(str(out / "stereo_{}_*.jpg".format(port))))
        n = int(files[-1][-9:-4]) + 1 if files else 0
        size = re.findall(r"width=\(int\)(\d+), height=\(int\)(\d+)", txt)
        errs = [ln.strip() for ln in txt.splitlines() if ln.startswith(("ERROR", "WARNING"))]
        r = {"frames": n, "hz": round(n / args.stereo_wait, 1),
             "size": "{}x{}".format(*size[-1]) if size else None, "errors": errs[:3],
             "frame": files[-1] if files else None}
        res["udp_{}".format(port)] = r
        print("  udp {} ({}): {} frames ({} Hz), {}{}".format(
            port, name, n, r["hz"], r["size"] or "no size", " | " + errs[0] if errs else ""))
    res["depth"] = _probe_depth(args.depth_device, out)
    return res


def _probe_depth(dev, out, w=544, h=448, wait_s=3.0):
    from . import sources as SRC
    if not os.path.exists(dev):
        print("  depth: {} does not exist".format(dev))
        return {"exists": False}
    fd = None
    try:
        fd = os.open(dev, os.O_RDONLY | os.O_NONBLOCK)
    except OSError as exc:
        print("  depth: cannot open {}: {}".format(dev, exc))
        return {"exists": True, "open": str(exc)}
    n, got, t_end, frames, t0 = w * h * 2, None, time.time() + wait_s, 0, time.time()
    try:
        while time.time() < t_end:
            try:
                data = os.read(fd, n)
            except BlockingIOError:
                time.sleep(0.005)
                continue
            if len(data) == n:
                got = np.frombuffer(data, "<u2").reshape(h, w)
                frames += 1
            else:
                time.sleep(0.005)
    finally:
        os.close(fd)
    if got is None:
        print("  depth: no frame in {:.0f} s (is `sudo unitree_depth_start` running?)".format(wait_s))
        return {"exists": True, "frames": 0}
    d = SRC.depth_from_u16mm(got)
    ok = d > 0
    q = np.percentile(d[ok], [5, 50, 95]).tolist() if ok.any() else [None] * 3
    np.save(str(out / "depth_frame.npy"), got)
    r = {"exists": True, "frames": frames, "hz": round(frames / (time.time() - t0), 1),
         "valid_frac": round(float(ok.mean()), 3),
         "m_p5_p50_p95": [None if v is None else round(v, 3) for v in q]}
    print("  depth: {} frames ({} Hz), {:.0%} of pixels valid, p5/p50/p95 {} m -> {}".format(
        frames, r["hz"], r["valid_frac"], r["m_p5_p50_p95"], out / "depth_frame.npy"))
    return r


def _probe_ros(args, out):
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, CompressedImage, Image

    from . import sources as SRC
    rclpy.init()
    node = rclpy.create_node("r1_perception_probe")
    t_end = time.time() + args.ros_wait
    while time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.2)
    topics = node.get_topic_names_and_types()
    found = {}
    kinds = {"sensor_msgs/msg/Image": Image, "sensor_msgs/msg/CompressedImage": CompressedImage,
             "sensor_msgs/msg/CameraInfo": CameraInfo}
    for name, types in topics:
        for t in types:
            if t in kinds:
                found[name] = t
    if not found:
        print("  no Image / CompressedImage / CameraInfo topics (is a camera driver running?)")
    stats = {}
    for name, t in sorted(found.items()):
        stats[name] = {"type": t, "n": 0}

        def cb(m, name=name, t=t):
            s = stats[name]
            s["n"] += 1
            s.setdefault("t0", time.time())
            s["t1"] = time.time()
            if s["n"] == 1:
                if t.endswith("CameraInfo"):
                    s.update({"width": m.width, "height": m.height,
                              "distortion_model": m.distortion_model,
                              "d": list(m.d), "k": list(m.k)})
                    return
                try:
                    img = (SRC.decode_compressed(m) if t.endswith("CompressedImage")
                           else SRC.decode_image(m))
                    s["shape"] = list(img.shape)
                    s["dtype"] = str(img.dtype)
                    if not t.endswith("CompressedImage"):
                        s["encoding"] = m.encoding
                    base = name.strip("/").replace("/", "_")
                    if img.dtype == np.uint8:
                        import cv2
                        cv2.imwrite(str(out / (base + ".jpg")), img)
                    else:
                        np.save(str(out / (base + ".npy")), img)
                        ok = np.isfinite(img) & (img > 0)
                        if ok.any():
                            s["depth_m_p5_p50_p95"] = [round(float(np.percentile(img[ok], q)), 3)
                                                       for q in (5, 50, 95)]
                            s["valid_frac"] = round(float(ok.mean()), 3)
                except Exception as exc:  # noqa: BLE001
                    s["decode_error"] = str(exc)
        node.create_subscription(kinds[t], name, cb, qos_profile_sensor_data)
    t_end = time.time() + 3.0
    while time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.05)
    for name, s in sorted(stats.items()):
        hz = (s["n"] - 1) / (s["t1"] - s["t0"]) if s["n"] > 1 else 0.0
        s["hz"] = round(hz, 1)
        extra = {k: v for k, v in s.items() if k not in ("t0", "t1", "n", "type")}
        print("  {} [{}] {} msgs in 3 s ({:.1f} Hz) {}".format(
            name, s["type"].split("/")[-1], s["n"], hz, json.dumps(extra)))
    node.destroy_node()
    rclpy.shutdown()
    return {n: {k: v for k, v in s.items() if k not in ("t0", "t1")} for n, s in stats.items()}


# ---------------------------------------------------------- engine-check
def engine_check(args):
    sys.path.insert(0, str(ROOT / "tools"))
    from make_fixture import read_fixture
    x, y = read_fixture(args.fixture)
    rt = RT.make(args.engine)
    print("engine {} | fixture {} samples, in {} out {}".format(rt.name, len(x), x.shape[1],
                                                                 y.shape[1]))
    worst, worst_at, det_diff, n_ref, iou_min, conf_max = 0.0, None, 0, 0, 1.0, 0.0
    for i in range(len(x)):
        o = rt.infer(x[i].reshape(rt.in_shape)).reshape(-1)
        d = float(np.abs(o - y[i]).max())
        if d > worst:
            worst, worst_at = d, (i, int(np.abs(o - y[i]).argmax()))
        a = Y.postprocess(o.reshape(1, -1, 85), 1.0, args.thr)
        b = Y.postprocess(y[i].reshape(1, -1, 85), 1.0, args.thr)
        pairs = _match(a, b, 0.5)
        n_ref += len(b)
        for pi, pj, o_ in pairs:
            iou_min = min(iou_min, o_)
            conf_max = max(conf_max, abs(a[pi][4] - b[pj][4]))
        if len(pairs) != len(a) or len(pairs) != len(b):
            det_diff += 1
    lat = RT.timed(rt, x[0].reshape(rt.in_shape), n=args.n)
    paced = RT.timed(rt, x[0].reshape(rt.in_shape), n=max(20, args.n // 5), period_s=0.08)
    print("parity: max_abs={:.3e} (worst at sample {}, element {})".format(
        worst, *(worst_at or (None, None))))
    print("detections at {:.2f}: {} in the reference; {} of {} samples have one that does not "
          "match (class + IoU >= 0.5); matched: smallest IoU {:.3f}, largest confidence "
          "difference {:.4f}".format(args.thr, n_ref, det_diff, len(x), iou_min, conf_max))
    print("latency compact (us over {}): p50={:.0f} p95={:.0f} p99={:.0f} max={:.0f}".format(
        lat["n"], lat["p50"], lat["p95"], lat["p99"], lat["max"]))
    print("latency paced 12.5 Hz (us over {}): p50={:.0f} p95={:.0f} p99={:.0f} max={:.0f}".format(
        paced["n"], paced["p50"], paced["p95"], paced["p99"], paced["max"]))
    # raw: every output element within tol -- the FP32 gate, the policy's own rule.
    # detections: what reaches the mission layer is the same -- the FP16 gate. FP16 moves
    # raw values of background anchors by ~1 (dev box: 1.29) without changing one box, so a
    # raw tolerance would only say "FP16 is not FP32", which is known.
    if args.gate == "raw":
        ok = worst <= args.tol
        what = "max_abs {:.1e}".format(args.tol)
    else:
        ok = det_diff == 0 and iou_min >= args.iou and conf_max <= args.conf
        what = "every detection matched, IoU >= {:.2f}, confidence within {:.3f}".format(
            args.iou, args.conf)
    print("PARITY {} ({} gate: {}) -- {}".format(
        "PASS" if ok else "FAIL", args.gate, what,
        "the Python runtime computes what the ONNX computes" if ok else
        "do not run this engine"))
    rt.close()
    return 0 if ok else 1


# --------------------------------------------------------------- compare
def _match(a, b, iou_thr=0.5):
    from .tracker import iou
    used, pairs = set(), []
    for i, p in enumerate(a):
        best, bj = 0.0, None
        for j, q in enumerate(b):
            if j in used or q[5] != p[5]:
                continue
            o = iou(p[:4], q[:4])
            if o > best:
                best, bj = o, j
        if bj is not None and best >= iou_thr:
            used.add(bj)
            pairs.append((i, bj, best))
    return pairs


def compare(args):
    """Two engines (FP32 and FP16), the same frames: latency and detections."""
    ra, rb = RT.make(args.a), RT.make(args.b)
    files = sorted(f for f in glob.glob(os.path.join(os.path.expanduser(args.frames), "*.jpg"))
                   if not f.endswith(("_raw.jpg", "_vis.jpg")))[:args.max]
    if not files:
        print("no frames in {} (capture some first)".format(args.frames))
        return 2
    import cv2
    rows, raw_max = [], 0.0
    for f in files:
        img = cv2.imread(f)      # capture's NNNNNN.jpg: already the rectified view
        x, r = Y.letterbox(img)
        oa, ob = ra.infer(x), rb.infer(x)
        raw_max = max(raw_max, float(np.abs(oa - ob).max()))
        da = Y.postprocess(oa, r, args.thr)
        db = Y.postprocess(ob, r, args.thr)
        pairs = _match(da, db)
        rows.append({"frame": os.path.basename(f), "a": len(da), "b": len(db),
                     "matched": len(pairs),
                     "iou_min": min([p[2] for p in pairs]) if pairs else None,
                     "conf_diff_max": max([abs(da[i][4] - db[j][4]) for i, j, _ in pairs])
                     if pairs else None})
    x = Y.letterbox(cv2.imread(files[0]))[0]
    lat = {}
    for name, rt in (("a", ra), ("b", rb)):
        lat[name] = {"compact": RT.timed(rt, x, n=args.n),
                     "paced": RT.timed(rt, x, n=max(20, args.n // 5), period_s=0.08)}
    tot_a = sum(r["a"] for r in rows)
    tot_b = sum(r["b"] for r in rows)
    tot_m = sum(r["matched"] for r in rows)
    ious = [r["iou_min"] for r in rows if r["iou_min"] is not None]
    cds = [r["conf_diff_max"] for r in rows if r["conf_diff_max"] is not None]
    res = {"a": ra.name, "b": rb.name, "frames": len(rows), "thr": args.thr,
           "raw_max_abs": raw_max, "dets_a": tot_a, "dets_b": tot_b, "matched": tot_m,
           "iou_min": min(ious) if ious else None,
           "conf_diff_max": max(cds) if cds else None, "latency_us": lat, "rows": rows}
    print("A = {}\nB = {}".format(ra.name, rb.name))
    for name in ("a", "b"):
        c, p = lat[name]["compact"], lat[name]["paced"]
        print("  {} latency us: compact p50 {:.0f} p99 {:.0f} | paced 12.5 Hz p50 {:.0f} p99 {:.0f}"
              .format(name.upper(), c["p50"], c["p99"], p["p50"], p["p99"]))
    print("detections at {:.2f} over {} frames: A {} | B {} | matched {} (class + IoU >= 0.5) | "
          "only A {} | only B {}".format(args.thr, len(rows), tot_a, tot_b, tot_m,
                                         tot_a - tot_m, tot_b - tot_m))
    print("matched pairs: smallest IoU {} | largest confidence difference {}".format(
        "{:.3f}".format(res["iou_min"]) if ious else "-",
        "{:.3f}".format(res["conf_diff_max"]) if cds else "-"))
    print("raw output max_abs A vs B: {:.3e}".format(raw_max))
    if args.out:
        pathlib.Path(os.path.expanduser(args.out)).write_text(json.dumps(res, indent=1))
        print("wrote " + args.out)
    return 0


# ---------------------------------------------------------------- capture
def capture(args):
    """Frames for the evaluations. Each kept frame writes
         NNNNNN_raw.jpg    the camera's own image (calibration needs this one)
         NNNNNN.jpg        the rectified image YOLOX sees
         NNNNNN_depth.npy  rectified depth, meters, float16 (if there is depth)
         NNNNNN.json       the detections message for that exact frame
    and meta.json holds the rectified intrinsics and the mount, for eval_tag.py."""
    cfg = load_cfg(args.config)
    out = pathlib.Path(os.path.expanduser(args.out))
    out.mkdir(parents=True, exist_ok=True)
    rt = RT.make(args.engine)
    pipe = PL.Pipeline(cfg, rt)
    meta = {"rect": {"w": pipe.rect.w, "h": pipe.rect.h, "f": pipe.rect.f, "fy": pipe.rect.fy,
                     "cx": pipe.rect.cx,
                     "cy": pipe.rect.cy, "hfov_deg": pipe.rect.hfov_deg},
            "mount": {"height_m": pipe.mount.height_m, "pitch_deg": cfg.get("cam_pitch_deg", 0.0),
                      "yaw_deg": cfg.get("cam_yaw_deg", 0.0)},
            "engine": rt.name, "rectified": True, "config": cfg}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    import cv2
    import rclpy
    rclpy.init()
    node = rclpy.create_node("r1_perception_capture")
    from . import sources as SRC
    src = SRC.make(cfg, node) if cfg.get("source", "ros") == "ros" else SRC.make(cfg)
    print("source: {}\nwriting to {}".format(src.describe, out))
    print("Enter = keep the next frame, q + Enter = quit" if not args.every
          else "one frame every {:.1f} s, Ctrl-C to stop".format(args.every))
    n = len(glob.glob(str(out / "*_raw.jpg")))
    kept = 0
    try:
        while kept < args.count:
            if not args.every:
                if input().strip().lower() == "q":
                    break
            else:
                time.sleep(args.every)
            f = None
            t_end = time.time() + 2.0
            while f is None and time.time() < t_end:
                rclpy.spin_once(node, timeout_sec=0.05)
                f = src.take()
            if f is None:
                print("no frame in 2 s")
                continue
            bgr, depth, t_cap = f
            msg, rect, rdepth = pipe.process(bgr, depth, t_cap)
            base = out / "{:06d}".format(n)
            cv2.imwrite(str(base) + "_raw.jpg", bgr)
            cv2.imwrite(str(base) + ".jpg", rect)
            if rdepth is not None:
                np.save(str(base) + "_depth.npy", rdepth.astype(np.float16))
            (pathlib.Path(str(base) + ".json")).write_text(json.dumps(msg))
            if args.vis:
                cv2.imwrite(str(base) + "_vis.jpg", PL.draw(rect, msg))
                if rdepth is not None:
                    cv2.imwrite(str(base) + "_overlay.jpg", PL.depth_overlay(rect, rdepth))
            seen = ", ".join("{} {:.2f}".format(d["cls"], d["conf"]) for d in msg["dets"][:6])
            print("{:06d}: {}".format(n, seen or "nothing at the safety threshold"))
            n += 1
            kept += 1
    except KeyboardInterrupt:
        pass
    src.close()
    node.destroy_node()
    rclpy.shutdown()
    print("kept {} frames in {}".format(kept, out))
    return 0


# ------------------------------------------------------------------- look
def look(args):
    """What the running node sees now: the next few messages, summarized."""
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String
    rclpy.init()
    node = rclpy.create_node("r1_perception_look")
    got = []
    node.create_subscription(String, "/r1_perception/detections",
                             lambda m: got.append(json.loads(m.data)),
                             QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
    t_end = time.time() + args.seconds
    while time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_node()
    rclpy.shutdown()
    if not got:
        print("no /r1_perception/detections in {:.0f} s -- is the node running?".format(
            args.seconds))
        return 3
    m = got[-1]
    print("{} messages in {:.0f} s; latest seq {} ({}), guard {}".format(
        len(got), args.seconds, m["seq"], m.get("engine"), json.dumps(m["guard"])))
    for d in m["dets"]:
        print("  {:>3} {:<14} conf {:.2f} {:<4} {:>18}  depth {}  range {}".format(
            d.get("id", "-"), d["cls"], d["conf"], "ok" if d.get("confirmed") else "",
            PL.angle_sign_text(d["bearing_deg"]), PL.fmt_m(d.get("depth_m")),
            PL.fmt_m(d.get("range_m"))))
    return 0


# ------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(prog="r1_perception", description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("probe", help="what camera is there (first robot step; moves nothing)")
    p.add_argument("--out", default="~/orin_commissioning/camera_probe")
    p.add_argument("--ros-wait", type=float, default=3.0)
    p.add_argument("--stereo-wait", type=float, default=6.0, help="seconds per UDP port")
    p.add_argument("--depth-device", default="/dev/video-dep")

    r = sub.add_parser("run", help="the ROS node")
    r.add_argument("--engine", default=None, help="a .plan (TensorRT) or .onnx (onnxruntime)")
    r.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a perception.yaml value")

    e = sub.add_parser("engine-check", help="the Python runtime against the parity fixture")
    e.add_argument("--engine", default=None)
    e.add_argument("--fixture", default=str(MODELS / "yolox_fixture.bin"))
    e.add_argument("--gate", choices=["raw", "detections"], default="raw",
                   help="raw = every output within --tol (FP32); detections = the same "
                        "boxes after decode + NMS (FP16)")
    e.add_argument("--tol", type=float, default=1e-3)
    e.add_argument("--thr", type=float, default=0.3)
    e.add_argument("--iou", type=float, default=0.95)
    e.add_argument("--conf", type=float, default=0.02)
    e.add_argument("--n", type=int, default=200)

    c = sub.add_parser("compare", help="FP32 vs FP16 on real frames")
    c.add_argument("--a", default=str(MODELS / "yolox_fp32.plan"))
    c.add_argument("--b", default=str(MODELS / "yolox_fp16.plan"))
    c.add_argument("--frames", required=True, help="a capture folder")
    c.add_argument("--thr", type=float, default=0.3)
    c.add_argument("--max", type=int, default=200)
    c.add_argument("--n", type=int, default=500)
    c.add_argument("--out", default=None)

    k = sub.add_parser("capture", help="save frames for calibration / evaluation")
    k.add_argument("--engine", default=None)
    k.add_argument("--out", default="~/r1_captures/" + time.strftime("%m%d_%H%M%S"))
    k.add_argument("--every", type=float, default=0.0, help="seconds between frames; 0 = on Enter")
    k.add_argument("--count", type=int, default=1000)
    k.add_argument("--vis", action="store_true",
                   help="also write NNNNNN_vis.jpg with boxes, NNNNNN_overlay.jpg with depth")

    lk = sub.add_parser("look", help="what the running node sees now")
    lk.add_argument("--seconds", type=float, default=1.0)

    a = ap.parse_args(argv)
    if not a.cmd:
        ap.print_help()
        return 2
    if getattr(a, "engine", "x") is None:
        a.engine = _engine_default()
    if a.cmd == "probe":
        return probe(a)
    if a.cmd == "engine-check":
        return engine_check(a)
    if a.cmd == "compare":
        return compare(a)
    if a.cmd == "capture":
        return capture(a)
    if a.cmd == "look":
        return look(a)
    cfg = load_cfg(a.config)
    for kv in a.set:
        key, _, val = kv.partition("=")
        try:
            cfg[key] = float(val) if ("." in val or "e" in val.lower()) else int(val)
        except ValueError:
            cfg[key] = val
    from . import node
    return node.run(cfg, a.engine)
