#!/usr/bin/env python3
"""Plays the policy and the camera for a V3 bridge on a dev box, so mission_ctl can be
run live against it (the real `r1_mission_cli.py run`, the real bridge).

  * the policy: answers every /r1_hw_bridge/obs with a joint target;
  * the camera: /r1_perception/proximity at 12.5 Hz, stamped 0.1 s before publication,
    with the obstacle distance taken from a timeline.

    python3 v3/tools/reflex_devbox_world.py --timeline "0:3.0,4:0.4,14:2.0" --seconds 25

The timeline is "t:obstacle_m,..." in seconds from the start: 3.0 m until 4 s, then
0.4 m until 14 s, then 2.0 m. See reflex_devbox_e2e.py for the bridge's command line.
"""
from __future__ import annotations

import argparse
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray, Float64MultiArray


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--timeline", default="0:3.0")
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()
    timeline = sorted((float(a), float(b)) for a, b in
                      (item.split(":") for item in args.timeline.split(",")))

    rclpy.init()
    node = Node("reflex_world")
    be = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    tgt_pub = node.create_publisher(Float32MultiArray, "/r1_policy_node/joint_target", be)
    act_pub = node.create_publisher(Float32MultiArray, "/r1_policy_node/action", be)
    prox_pub = node.create_publisher(Float64MultiArray, "/r1_perception/proximity",
                                     QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
    zero = Float32MultiArray()
    zero.data = [0.0] * 24

    def on_obs(_m):
        tgt_pub.publish(zero)
        act_pub.publish(zero)

    node.create_subscription(Float32MultiArray, "/r1_hw_bridge/obs", on_obs, be)
    t0 = time.monotonic()
    state = {"seq": 0, "last": None}

    def on_frame():
        t = time.monotonic() - t0
        obst = timeline[0][1]
        for ti, d in timeline:
            if t >= ti:
                obst = d
        if obst != state["last"]:
            print("{:6.2f} s  obstacle {:.2f} m".format(t, obst), flush=True)
            state["last"] = obst
        state["seq"] += 1
        now = time.monotonic()
        m = Float64MultiArray()
        m.data = [1.0, float(state["seq"]), now - 0.1, now, float("nan"), 0.0, 0.0, obst, 500.0,
                  0.95, 0.95, 1.0]
        prox_pub.publish(m)

    node.create_timer(0.08, on_frame)
    while rclpy.ok() and time.monotonic() - t0 < args.seconds:
        rclpy.spin_once(node, timeout_sec=0.01)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
