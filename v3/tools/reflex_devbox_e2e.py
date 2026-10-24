#!/usr/bin/env python3
"""V3 reflex end to end on a dev box: the real bridge node, built against the fake
robot (-DR1_FAKE_ROBOT_IO=ON), driven over real ROS 2 topics.

This script plays everything around the bridge:
  * the policy: answers every /r1_hw_bridge/obs with a joint target, so the bridge
    reaches RUNNING;
  * perception: /r1_perception/proximity at 12.5 Hz, from a scripted scene, stamped
    0.1 s before publication (a camera's delay);
  * mission_ctl: /r1_hw_bridge/cmd_vel at 10 Hz, and one /r1_hw_bridge/reflex_reset.
It records the velocity command the bridge writes into each observation and the
bridge's /r1_hw_bridge/reflex messages, then checks them against the scene:

  0-3 s   clear, walking at 0.3 m/s     -> the observation carries 0.3
  3-5 s   obstacle at 0.8 m             -> forward blocked (stop distance 0.87 m)
  5-7 s   obstacle at 0.4 m             -> RETREAT: -0.3 for 1.5 s, then STAND (0)
  7-9 s   clear again, still asked 0.3  -> STAND holds: 0
  9 s     reset                         -> ARMED, 0.3 again

    source /opt/ros/humble/setup.bash      # or foxy
    export ROS_DOMAIN_ID=88 ROS_LOCALHOST_ONLY=1
    <ws>/install/r1_hw_bridge/lib/r1_hw_bridge/r1_hw_bridge_node --ros-args \\
        --params-file v3/ros2_ws/src/r1_hw_bridge/config/bridge.yaml \\
        -p reflex_enable:=true -p reflex_log:=/tmp/reflex.csv &
    python3 v3/tools/reflex_devbox_e2e.py [--dump reflex_messages.jsonl]

--dump writes every /r1_hw_bridge/reflex message as received, one per line: the
fixture v3/mission_ctl/tests/reflex_messages.jsonl is such a dump, so mission_ctl's
parser is tested on what the C++ bridge actually publishes.
"""
from __future__ import annotations

import json
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty, Float32MultiArray, Float64MultiArray, String

CMD_IDX = slice(6, 9)          # the observation frame: gyro 3, gravity 3, command 3, ...
DURATION = 11.0
RESET_AT = 9.0


def scene(t):
    """(obstacle_m, cmd_vx) at script time t."""
    if t < 3.0:
        return 3.0, 0.3
    if t < 5.0:
        return 0.8, 0.3
    if t < 7.0:
        return 0.4, 0.3
    return 2.0, 0.3


class Driver(Node):
    def __init__(self):
        super().__init__("reflex_e2e")
        be = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        rel = QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE)
        self.target_pub = self.create_publisher(Float32MultiArray, "/r1_policy_node/joint_target", be)
        self.action_pub = self.create_publisher(Float32MultiArray, "/r1_policy_node/action", be)
        self.prox_pub = self.create_publisher(Float64MultiArray, "/r1_perception/proximity",
                                              QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
        self.cmd_pub = self.create_publisher(Float32MultiArray, "/r1_hw_bridge/cmd_vel", be)
        self.reset_pub = self.create_publisher(Empty, "/r1_hw_bridge/reflex_reset",
                                               QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE))
        self.create_subscription(Float32MultiArray, "/r1_hw_bridge/obs", self.on_obs, be)
        self.create_subscription(String, "/r1_hw_bridge/reflex", self.on_reflex, rel)
        self.create_subscription(String, "/r1_hw_bridge/status", self.on_status,
                                 QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
        self.t0 = None
        self.seq = 0
        self.obs = []           # (t, vx, vy, wz)
        self.reflex = []        # (t, dict)
        self.raw = []           # the messages as received
        self.prox_log = []      # (t_pub, t_cap, obstacle_m)
        self.status = ""
        self.reset_sent = False
        self.create_timer(0.08, self.on_prox)
        self.create_timer(0.1, self.on_cmd)

    def now(self):
        return time.monotonic() - self.t0 if self.t0 is not None else -1.0

    def on_status(self, m):
        self.status = m.data
        if self.t0 is None and m.data.startswith("RUNNING"):
            self.t0 = time.monotonic()
            self.get_logger().info("bridge RUNNING: script starts")

    def on_obs(self, m):
        tgt = Float32MultiArray()
        tgt.data = [0.0] * 24
        self.target_pub.publish(tgt)
        self.action_pub.publish(tgt)
        if self.t0 is not None:
            self.obs.append((self.now(),) + tuple(m.data[CMD_IDX]))

    def on_reflex(self, m):
        if self.t0 is not None:
            self.reflex.append((self.now(), json.loads(m.data)))
            self.raw.append(m.data)

    def on_prox(self):
        t = self.now()
        obst = scene(max(t, 0.0))[0]
        now = time.monotonic()
        self.seq += 1
        msg = Float64MultiArray()
        msg.data = [1.0, float(self.seq), now - 0.1, now, float("nan"), 0.0, 0.0, obst, 500.0,
                    0.95, 0.95, 1.0]
        self.prox_pub.publish(msg)
        if self.t0 is not None:
            self.prox_log.append((t, now - 0.1 - self.t0, obst))

    def on_cmd(self):
        t = self.now()
        m = Float32MultiArray()
        m.data = [scene(max(t, 0.0))[1], 0.0, 0.0]
        self.cmd_pub.publish(m)
        if t >= RESET_AT and not self.reset_sent:
            self.reset_pub.publish(Empty())
            self.reset_sent = True


def main():
    rclpy.init()
    d = Driver()
    t_end = time.monotonic() + 30.0
    while rclpy.ok() and time.monotonic() < t_end:
        rclpy.spin_once(d, timeout_sec=0.01)
        if d.t0 is not None and d.now() > DURATION:
            break
    if d.t0 is None:
        print("FAIL: the bridge never reached RUNNING (status: {!r})".format(d.status))
        return 1

    def vx_at(t0, t1):
        return sorted({round(v[1], 3) for v in d.obs if t0 <= v[0] < t1})

    def states_at(t0, t1):
        return [r["state"] for (t, r) in d.reflex if t0 <= t < t1]

    fails, n = [], [0]

    def expect(name, cond, detail):
        n[0] += 1
        print("  {:5} {}  ({})".format("ok" if cond else "FAIL", name, detail))
        if not cond:
            fails.append(name)

    expect("0-3 s clear: the observation carries 0.3", vx_at(0.5, 2.9) == [0.3], vx_at(0.5, 2.9))
    expect("3-5 s obstacle 0.8 m: forward blocked (0)", vx_at(3.3, 4.9) == [0.0], vx_at(3.3, 4.9))
    retreat = [v[0] for v in d.obs if v[1] < -0.29]
    expect("5-7 s obstacle 0.4 m: a retreat at -0.3", bool(retreat),
           "{} obs frames at -0.3".format(len(retreat)))
    if retreat:
        first_near_cap = min(c for (_, c, o) in d.prox_log if o == 0.4)
        lat = retreat[0] - first_near_cap
        dur = retreat[-1] - retreat[0]
        expect("retreat lasts 1.5 s", abs(dur - 1.48) < 0.06, "{:.3f} s of -0.3 frames".format(dur))
        # two frames at 12.5 Hz after the first near capture, plus the camera's 0.1 s
        expect("capture -> retreat in the observation within 0.25 s", lat < 0.25,
               "{:.0f} ms (camera 100 ms + 2 frames 80 ms + one step)".format(lat * 1e3))
        expect("then STAND: 0 while the scene says walk", vx_at(retreat[-1] + 0.1, RESET_AT - 0.05) == [0.0],
               vx_at(retreat[-1] + 0.1, RESET_AT - 0.05))
    expect("the reflex topic said RETREAT then STAND", "RETREAT" in states_at(5.0, 7.0)
           and "STAND" in states_at(6.0, 9.0), sorted(set(states_at(5.0, 9.0))))
    expect("after the reset at 9 s: ARMED, 0.3 again", vx_at(9.3, DURATION) == [0.3]
           and "ARMED" in states_at(9.0, DURATION), (vx_at(9.3, DURATION), states_at(9.0, DURATION)))
    print("{} checks, {} failed".format(n[0], len(fails)))
    if "--dump" in sys.argv:
        path = sys.argv[sys.argv.index("--dump") + 1]
        with open(path, "w") as fh:
            fh.write("\n".join(d.raw) + "\n")
        print("wrote {} messages to {}".format(len(d.raw), path))
    d.destroy_node()
    rclpy.shutdown()
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
