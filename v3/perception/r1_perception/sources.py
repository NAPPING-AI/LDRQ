"""Where frames come from.

  stereo   the R1 head's stereo module (the robot's camera; see StereoSource): the
           left camera's RGB over UDP RTP + its depth from /dev/video-dep
  ros      sensor_msgs/Image (or CompressedImage) on `rgb_topic`, optional depth on
           `depth_topic` -- if a driver is already publishing
  v4l2     cv2.VideoCapture(`v4l2_device`): RGB only, no depth
  replay   a directory written by `capture`: NNNNNN.jpg (+ NNNNNN_depth.npy)

Every source hands out the latest frame only: (bgr uint8 HxWx3, depth meters HxW or
None, t_cap = time.time() when it arrived). A frame is never processed twice.

Depth is used only if `depth_aligned: 1` says the driver registers it to the RGB
image. Unregistered depth would put the median of the wall behind a chair into the
chair's box, so it is dropped rather than used (walk_to then refuses: no depth).
"""
from __future__ import annotations

import glob
import os
import select
import shutil
import subprocess
import threading
import time

import numpy as np


class Latest(object):
    def __init__(self):
        self._lock = threading.Lock()
        self._rgb = None
        self._depth = None
        self._t = None
        self.count = 0
        self._taken = 0

    def put_rgb(self, bgr, t):
        with self._lock:
            self._rgb, self._t = bgr, t
            self.count += 1

    def put_depth(self, depth):
        with self._lock:
            self._depth = depth

    def take(self):
        """The newest frame if it has not been taken yet, else None."""
        with self._lock:
            if self._rgb is None or self._taken == self.count:
                return None
            self._taken = self.count
            return self._rgb, self._depth, self._t


# --------------------------------------------------------------- decoding
def decode_image(msg):
    """sensor_msgs/Image -> BGR uint8 or depth float (meters)."""
    enc = msg.encoding.lower()
    h, w = msg.height, msg.width
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if enc in ("rgb8", "bgr8"):
        img = buf.reshape(h, msg.step)[:, :w * 3].reshape(h, w, 3)
        return img[:, :, ::-1].copy() if enc == "rgb8" else img.copy()
    if enc in ("rgba8", "bgra8"):
        img = buf.reshape(h, msg.step)[:, :w * 4].reshape(h, w, 4)
        return img[:, :, [2, 1, 0]].copy() if enc == "rgba8" else img[:, :, :3].copy()
    if enc in ("mono8", "8uc1"):
        g = buf.reshape(h, msg.step)[:, :w]
        return np.repeat(g[:, :, None], 3, axis=2)
    if enc in ("yuv422", "yuyv", "yuv422_yuy2"):
        import cv2
        img = buf.reshape(h, msg.step)[:, :w * 2].reshape(h, w, 2)
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUYV)
    if enc in ("16uc1", "mono16"):
        big = bool(getattr(msg, "is_bigendian", 0))
        d = np.frombuffer(bytes(msg.data), dtype=">u2" if big else "<u2")
        return d.reshape(h, msg.step // 2)[:, :w].astype(np.float32) / 1000.0  # mm
    if enc == "32fc1":
        d = np.frombuffer(bytes(msg.data), dtype=np.float32)
        return d.reshape(h, msg.step // 4)[:, :w].copy()                          # m
    raise ValueError("unsupported image encoding {!r}".format(msg.encoding))


def decode_compressed(msg):
    import cv2
    return cv2.imdecode(np.frombuffer(bytes(msg.data), dtype=np.uint8), cv2.IMREAD_COLOR)


def fit_depth(depth, shape_hw):
    """Depth at the RGB image's size (nearest). Same field of view is the driver's
    job (`depth_aligned`); this only fixes a resolution difference."""
    if depth is None or depth.shape[:2] == tuple(shape_hw):
        return depth
    h, w = shape_hw
    ys = (np.arange(h) * depth.shape[0] / float(h)).astype(int)
    xs = (np.arange(w) * depth.shape[1] / float(w)).astype(int)
    return depth[ys][:, xs]


# ---------------------------------------------------------------- sources
class RosSource(object):
    """Subscribes on an existing rclpy node."""

    def __init__(self, node, cfg):
        from rclpy.qos import qos_profile_sensor_data
        self.latest = Latest()
        self.use_depth = int(cfg.get("depth_aligned", 0)) == 1 and bool(cfg.get("depth_topic"))
        rgb_topic = str(cfg.get("rgb_topic", "/camera/color/image_raw"))
        if rgb_topic.endswith("/compressed"):
            from sensor_msgs.msg import CompressedImage
            node.create_subscription(CompressedImage, rgb_topic, self._on_cimg,
                                     qos_profile_sensor_data)
        else:
            from sensor_msgs.msg import Image
            node.create_subscription(Image, rgb_topic, self._on_img, qos_profile_sensor_data)
        if self.use_depth:
            from sensor_msgs.msg import Image
            node.create_subscription(Image, str(cfg["depth_topic"]), self._on_depth,
                                     qos_profile_sensor_data)
        self.describe = "ros {}{}".format(rgb_topic, " + depth " + str(cfg["depth_topic"])
                                          if self.use_depth else " (no depth)")

    def _on_img(self, m):
        self.latest.put_rgb(decode_image(m), time.time())

    def _on_cimg(self, m):
        self.latest.put_rgb(decode_compressed(m), time.time())

    def _on_depth(self, m):
        self.latest.put_depth(decode_image(m))

    def take(self):
        f = self.latest.take()
        if f is None:
            return None
        bgr, d, t = f
        return bgr, fit_depth(d, bgr.shape[:2]) if self.use_depth else None, t

    def close(self):
        pass


class V4l2Source(object):
    def __init__(self, cfg):
        import cv2
        dev = cfg.get("v4l2_device", 0)
        self.cap = cv2.VideoCapture(int(dev) if str(dev).isdigit() else str(dev))
        if cfg.get("cam_width"):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg["cam_width"]))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg["cam_height"]))
        if cfg.get("v4l2_fps"):
            self.cap.set(cv2.CAP_PROP_FPS, float(cfg["v4l2_fps"]))
        if not self.cap.isOpened():
            raise RuntimeError("cannot open V4L2 device {!r}".format(dev))
        self.latest = Latest()
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()
        self.describe = "v4l2 {} (no depth)".format(dev)

    def _loop(self):
        while self._run:
            ok, img = self.cap.read()
            if ok:
                self.latest.put_rgb(img, time.time())
            else:
                time.sleep(0.01)

    def take(self):
        f = self.latest.take()
        return None if f is None else (f[0], None, f[2])

    def close(self):
        self._run = False
        self.cap.release()


class StereoSource(object):
    """The R1 head's stereo module, as Unitree's "video stream service" page describes
    it (R1 SDK guide; the robot facts are in docs/stageE_perception.md):

      RGB    the LEFT camera, 544x448, RTP/H264 on UDP `stereo_rgb_port` (5002), pushed
             to this computer once the APP has video_hub OFF and "Stereo patch PC1" ON
      depth  544x448 uint16 millimetres (0 and 65535 = no return) in the v4l2loopback
             `stereo_depth_device` (/dev/video-dep), written once
             `sudo unitree_depth_start` has run

    The RGB is decoded by a gst-launch-1.0 child that writes raw BGR frames to a pipe:
    no dependence on how the robot's OpenCV was built, and a stuck decoder is a process
    to kill, not a thread to lose. Depth is read the way Unitree's own example reads it:
    whole frames from a non-blocking descriptor.

    The two streams carry no common timestamp, so a frame is paired with the newest
    depth image if that is younger than `stereo_depth_max_age_s`, else with none (the
    guard then reports depth_ok false and walk_to refuses). Same size as the RGB, so
    `depth_aligned: 1` is assumed -- session 3 checks it on an overlay."""

    DECODERS = {
        # name -> the part between the depayloader and videoconvert
        "nvv4l2decoder": "h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx",
        "avdec_h264": "avdec_h264",
    }

    def __init__(self, cfg, rgb_cmd=None):
        self.w = int(cfg.get("cam_width", 544))
        self.h = int(cfg.get("cam_height", 448))
        self.port = int(cfg.get("stereo_rgb_port", 5002))
        self.max_age = float(cfg.get("stereo_depth_max_age_s", 0.3))
        self.latest = Latest()
        self._run = True
        self._err = []
        if rgb_cmd is None:
            rgb_cmd, how = self.gst_command(cfg)
        else:
            how = "custom"
        self.rgb_cmd = rgb_cmd
        self.proc = subprocess.Popen(rgb_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     bufsize=0)
        threading.Thread(target=self._rgb_loop, daemon=True).start()
        threading.Thread(target=self._err_loop, daemon=True).start()
        self.depth_dev = str(cfg.get("stereo_depth_device", "/dev/video-dep"))
        self.use_depth = int(cfg.get("depth_aligned", 1)) == 1 and bool(self.depth_dev)
        self._depth, self._t_depth, self.depth_frames = None, 0.0, 0
        self.dw = int(cfg.get("stereo_depth_width", self.w))
        self.dh = int(cfg.get("stereo_depth_height", self.h))
        if self.use_depth:
            threading.Thread(target=self._depth_loop, daemon=True).start()
        self.describe = "stereo left RGB udp:{} {}x{} ({}) + depth {}".format(
            self.port, self.w, self.h, how, self.depth_dev if self.use_depth else "off")

    @classmethod
    def gst_command(cls, cfg):
        """The receiving pipeline as an argv list, and the decoder it uses."""
        port, w, h = int(cfg.get("stereo_rgb_port", 5002)), int(cfg.get("cam_width", 544)), \
            int(cfg.get("cam_height", 448))
        codec = str(cfg.get("stereo_codec", "h264")).lower()
        if codec == "jpeg":       # the page's raw-image ports (5004 / 5005)
            caps = "encoding-name=JPEG"
            body, how = "rtpjpegdepay ! jpegdec", "jpeg"
        else:
            caps = "encoding-name=H264"
            dec = str(cfg.get("stereo_decoder", "auto"))
            if dec == "auto":
                dec = "nvv4l2decoder" if gst_has("nvv4l2decoder") and gst_has("nvvidconv") \
                    else "avdec_h264"     # Jetson's hardware decoder, else libav on the CPU
            body, how = "rtph264depay ! " + cls.DECODERS.get(dec, dec), dec
        pipe = ("udpsrc port={p} buffer-size=4194304 caps=application/x-rtp,media=video,"
                "clock-rate=90000,{caps},payload=96 ! {body} ! videoconvert ! "
                "video/x-raw,format=BGR,width={w},height={h} ! fdsink fd=1 sync=false").format(
                    p=port, caps=caps, body=body, w=w, h=h)
        # width/height in the last caps are a check, not a scale: a stream of another
        # size fails to negotiate and says so on stderr instead of being stretched.
        return ["gst-launch-1.0", "-q"] + pipe.split(), how

    def _rgb_loop(self):
        n = self.w * self.h * 3
        f = self.proc.stdout
        while self._run:
            buf = bytearray()
            while len(buf) < n:
                chunk = f.read(n - len(buf))
                if not chunk:
                    return              # the child exited; take() reports it
                buf += chunk
            img = np.frombuffer(bytes(buf), np.uint8).reshape(self.h, self.w, 3)
            self.latest.put_rgb(img, time.time())

    def _err_loop(self):
        for line in iter(self.proc.stderr.readline, b""):
            self._err = (self._err + [line.decode("utf-8", "replace").rstrip()])[-20:]

    def _depth_loop(self):
        n = self.dw * self.dh * 2
        fd = None
        while self._run:
            if fd is None:
                try:
                    fd = os.open(self.depth_dev, os.O_RDONLY | os.O_NONBLOCK)
                except OSError:
                    time.sleep(1.0)     # not there yet: `sudo unitree_depth_start`
                    continue
            try:
                data = os.read(fd, n)
            except BlockingIOError:
                select.select([fd], [], [], 0.05)
                time.sleep(0.002)       # a loopback that polls readable but is empty
                continue
            except OSError:
                os.close(fd)
                fd = None
                continue
            if len(data) != n:
                time.sleep(0.002)
                continue
            self._depth = depth_from_u16mm(np.frombuffer(data, "<u2").reshape(self.dh, self.dw))
            self._t_depth = time.time()
            self.depth_frames += 1
        if fd is not None:
            os.close(fd)

    def status(self):
        """One line for logs: is each half delivering?"""
        alive = self.proc.poll() is None
        return "rgb frames {}{} | depth frames {} (newest {:.2f} s old)".format(
            self.latest.count, "" if alive else " (decoder EXITED: {})".format(
                " / ".join(self._err[-3:]) or "no message"),
            self.depth_frames, time.time() - self._t_depth if self.depth_frames else -1.0)

    def take(self):
        if self.proc.poll() is not None and self.latest.count == 0:
            raise RuntimeError("stereo RGB decoder exited: " + (" / ".join(self._err[-5:])
                                                                or "no message"))
        f = self.latest.take()
        if f is None:
            return None
        bgr, _, t = f
        d = self._depth
        fresh = d is not None and (t - self._t_depth) <= self.max_age
        return bgr, fit_depth(d, bgr.shape[:2]) if (self.use_depth and fresh) else None, t

    def close(self):
        self._run = False
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def depth_from_u16mm(raw):
    """uint16 millimetres -> float32 metres; 0 and 65535 (no return) -> 0."""
    d = raw.astype(np.float32) / 1000.0
    d[(raw == 0) | (raw == 0xFFFF)] = 0.0
    return d


def gst_has(element):
    if not shutil.which("gst-inspect-1.0"):
        return False
    try:
        return subprocess.run(["gst-inspect-1.0", element], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=10).returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


class ReplaySource(object):
    """Frames from a `capture` directory, one per take() (no pacing: the caller paces)."""

    def __init__(self, cfg, loop=True):
        d = os.path.expanduser(str(cfg["replay_dir"]))
        self.files = sorted(f for f in glob.glob(os.path.join(d, "*.jpg"))
                            if not f.endswith(("_vis.jpg", "_raw.jpg", "_overlay.jpg")))
        if not self.files:
            raise RuntimeError("no .jpg frames in " + d)
        self.i, self.loop = 0, loop
        self.describe = "replay {} ({} frames)".format(d, len(self.files))

    def take(self):
        import cv2
        if self.i >= len(self.files):
            if not self.loop:
                return None
            self.i = 0
        f = self.files[self.i]
        self.i += 1
        bgr = cv2.imread(f, cv2.IMREAD_COLOR)
        dp = f[:-4] + "_depth.npy"
        depth = np.load(dp) if os.path.exists(dp) else None
        return bgr, depth, time.time()

    def close(self):
        pass


def make(cfg, node=None):
    kind = str(cfg.get("source", "ros"))
    if kind == "stereo":
        return StereoSource(cfg)
    if kind == "ros":
        if node is None:
            raise ValueError("source ros needs a ROS node")
        return RosSource(node, cfg)
    if kind == "v4l2":
        return V4l2Source(cfg)
    if kind == "replay":
        return ReplaySource(cfg)
    raise ValueError("unknown source {!r} (stereo | ros | v4l2 | replay)".format(kind))
