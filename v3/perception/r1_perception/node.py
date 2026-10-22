"""The ROS 2 node: /r1_perception. foxy / Python 3.8, std_msgs + sensor_msgs only.

When it runs inference is the design, not a detail. The walking policy runs
TensorRT on the same GPU every 20 ms for ~0.45 ms, and in stage C a model decoding on
that GPU pushed the policy's inference to ~5 ms. So this node does not run on a free
timer: it listens to /r1_policy_node/action, which the policy publishes right after
each inference, and runs on every `policy_ticks_per_frame`-th one (4 -> 12.5 Hz). YOLO
then has ~19 ms of an otherwise idle GPU before the policy's next step. The CPU half
(frame, rectify, letterbox) runs on the step before, so the GPU starts the moment the
slot opens: on the robot (10-14) doing it inside the slot cost 6-9 ms of CPU first, and
with the LLM on the CPU the FP32 engine then ran into the policy (p99 5.2 ms). If no action
message arrives for a second (no stack, the dev box), a timer takes over at the same
rate and the stats line says so.

Topics (all under /r1_perception):
  ~/proximity         std_msgs/Float64MultiArray (V3, pipeline.PROX_LAYOUT), one per
                      processed frame, published first: what the bridge's reflex reads
  ~/detections        std_msgs/String, JSON (pipeline.py), one per processed frame
  ~/snapshot_request  std_msgs/String, JSON {"path": ..., "highlight": <det id>}
  ~/snapshot_done     std_msgs/String, JSON {"path": ..., "ok": bool, "error": ...}
Stats line every 5 s, parsed by llm/coexist.py:
  PERCEPTION | 12.5 Hz | frames 62 | infer ms p50=.. p95=.. p99=.. max=.. | total ms
  p50=.. p99=.. | trigger policy | <engine>
"""
from __future__ import annotations

import json
import os
import threading
import time

from . import pipeline as PL
from . import runtime as RT
from . import sources as SRC

POLICY_ACTION = "/r1_policy_node/action"


def _pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(p * (len(s) - 1)))]


def run(cfg, engine_path, stats_period_s=5.0):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from std_msgs.msg import Float32MultiArray, Float64MultiArray, String

    rt = RT.make(engine_path)
    pipe = PL.Pipeline(cfg, rt)
    k = max(1, int(cfg.get("policy_ticks_per_frame", 4)))
    fallback_hz = float(cfg.get("fallback_hz", 12.5))

    class Perception(Node):
        def __init__(self):
            Node.__init__(self, "r1_perception")
            self.src = SRC.make(cfg, self)
            # RELIABLE, newest only: a BEST_EFFORT subscriber (the bridge) can read a RELIABLE
            # publisher, not the other way round. Freshness is the bridge's check, on t_cap.
            self.prox_pub = self.create_publisher(
                Float64MultiArray, "~/proximity",
                QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
            self.pub = self.create_publisher(
                String, "~/detections", QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
            self.snap_pub = self.create_publisher(
                String, "~/snapshot_done", QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE))
            self.create_subscription(String, "~/snapshot_request", self.on_snapshot,
                                     QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE))
            self.create_subscription(Float32MultiArray, POLICY_ACTION, self.on_action,
                                     qos_profile_sensor_data)
            self.create_timer(1.0 / fallback_hz, self.on_timer)
            self.create_timer(stats_period_s, self.on_stats)
            self.ticks = 0
            self.prep = None
            self.last_action = None
            self.trigger = "timer"
            self.infer_ms, self.total_ms = [], []
            self.frames = 0
            self.last_msg = None
            self.get_logger().info("source: {} | engine: {} | {} Hz from every {}th policy "
                                   "step".format(self.src.describe, rt.name, 50.0 / k, k))

        def on_action(self, _m):
            self.last_action = time.monotonic()
            self.ticks += 1
            if self.ticks % k == 0:
                self.trigger = "policy"
                if k == 1 or self.prep is None:
                    self.prepare()
                self.finish()
            elif self.ticks % k == k - 1:
                self.prepare()

        def on_timer(self):
            if self.last_action is None or time.monotonic() - self.last_action > 1.0:
                self.trigger = "timer"
                self.prepare()
                self.finish()

        def prepare(self):
            f = self.src.take()
            if f is not None:
                self.prep = pipe.prepare(*f)

        def finish(self):
            if self.prep is None:
                return
            t_finish = time.perf_counter()
            msg, _, _ = pipe.finish(self.prep)
            self.prep = None
            self.last_msg = msg
            # t_cap is time.time() at arrival (sources.py); the bridge works in
            # CLOCK_MONOTONIC (steady_clock), so convert with the offset of this instant
            now_mono = time.monotonic()
            prox = Float64MultiArray()
            prox.data = PL.proximity_array(msg["proximity"], msg["seq"],
                                           now_mono - (time.time() - msg["t_cap"]), now_mono)
            self.prox_pub.publish(prox)
            out = String()
            out.data = json.dumps(msg, separators=(",", ":"))
            self.pub.publish(out)
            self.frames += 1
            self.infer_ms.append(msg["timing"]["infer_ms"])
            tm = msg["timing"]
            self.total_ms.append(tm["rect_ms"] + tm["pre_ms"]
                                 + (time.perf_counter() - t_finish) * 1e3)

        def on_snapshot(self, m):
            reply = {"ok": False}
            try:
                req = json.loads(m.data)
                reply["path"] = path = os.path.expanduser(str(req["path"]))
                if pipe.last_rect is None or self.last_msg is None:
                    raise RuntimeError("no frame processed yet")
                import cv2
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                img = PL.draw(pipe.last_rect, self.last_msg, req.get("highlight"),
                              req.get("target_conf"))
                if not cv2.imwrite(path, img):
                    raise RuntimeError("cv2.imwrite failed")
                reply["ok"] = True
                reply["seq"] = self.last_msg["seq"]
            except Exception as exc:  # noqa: BLE001 -- reported to the requester
                reply["error"] = str(exc)
            out = String()
            out.data = json.dumps(reply)
            self.snap_pub.publish(out)

        def src_status(self):
            st = getattr(self.src, "status", None)
            return " | " + st() if st else ""

        def on_stats(self):
            n = len(self.infer_ms)
            if n:
                self.get_logger().info(
                    "PERCEPTION | {:.1f} Hz | frames {} | infer ms p50={:.2f} p95={:.2f} "
                    "p99={:.2f} max={:.2f} | total ms p50={:.2f} p99={:.2f} | trigger {} | {}"
                    .format(n / stats_period_s, self.frames, _pct(self.infer_ms, .5),
                            _pct(self.infer_ms, .95), _pct(self.infer_ms, .99),
                            max(self.infer_ms), _pct(self.total_ms, .5),
                            _pct(self.total_ms, .99), self.trigger, rt.name)
                    + self.src_status())
            else:
                self.get_logger().warn("PERCEPTION | 0.0 Hz | no frames from {} in the last "
                                       "{:.0f} s".format(self.src.describe, stats_period_s)
                                       + self.src_status())
            self.infer_ms, self.total_ms = [], []

    rclpy.init()
    node = Perception()
    port = int(cfg.get("snapshot_http_port", 0))
    if port:
        _serve_latest(port, os.path.expanduser(str(cfg.get("snapshot_dir", "~/r1_snapshots"))),
                      node.get_logger())
    try:
        from rclpy.executors import ExternalShutdownException   # galactic and later
    except ImportError:                                          # foxy (the robot)
        ExternalShutdownException = KeyboardInterrupt
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.src.close()
        rt.close()
        node.destroy_node()
        if rclpy.ok():           # humble's own SIGINT handler may have shut it down already
            rclpy.shutdown()
    return 0


def _serve_latest(port, folder, log):
    """Read-only: GET / returns the newest .jpg in `folder`. Off unless
    snapshot_http_port is set; for an operator laptop on the robot's own network."""
    import http.server
    import glob

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 -- the stdlib's name
            files = sorted(glob.glob(os.path.join(folder, "*.jpg")), key=os.path.getmtime)
            if not files:
                self.send_error(404, "no snapshot yet")
                return
            data = open(files[-1], "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("0.0.0.0", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log.info("latest snapshot at http://<robot>:{}/".format(port))
