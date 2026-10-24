#!/usr/bin/env python3
"""Plays the camera for a V3 dance run on a dev box: /r1_perception/proximity at 12.5 Hz,
stamped 0.1 s before publication, with the PERSON distance from a timeline (perception's
person_box mode: no obstacle readings). Everything else is real: the bridge (fake robot
I/O), the runner fork with both engines, mission_ctl live.

    python3 v3/tools/dance_devbox_camera.py --timeline "0:3.0,9:0.7,15:2.0" --seconds 30

The timeline is "t:person_m,..." in seconds from the start; "none" = nobody in view.
"""
from __future__ import annotations

import argparse
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--timeline", default="0:none")
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()
    timeline = sorted((float(a), None if b == "none" else float(b)) for a, b in
                      (item.split(":") for item in args.timeline.split(",")))

    rclpy.init()
    node = Node("dance_camera")
    pub = node.create_publisher(Float64MultiArray, "/r1_perception/proximity",
                                QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
    t0 = time.monotonic()
    state = {"seq": 0, "last": "unset"}

    def on_frame():
        t = time.monotonic() - t0
        person = timeline[0][1]
        for ti, d in timeline:
            if t >= ti:
                person = d
        if person != state["last"]:
            print("{:6.2f} s  person {}".format(t, "none" if person is None else "%.2f m" % person),
                  flush=True)
            state["last"] = person
        state["seq"] += 1
        now = time.monotonic()
        m = Float64MultiArray()
        # PROX_LAYOUT: version, seq, t_cap, t_pub, person_m, near_box, persons, obstacle_m,
        # corridor_points, depth_valid, center_valid, depth_ok
        m.data = [1.0, float(state["seq"]), now - 0.1, now,
                  float("nan") if person is None else person, 0.0, 0.0 if person is None else 1.0,
                  float("nan"), 0.0, 1.0, 1.0, 1.0]
        pub.publish(m)

    node.create_timer(0.08, on_frame)
    while rclpy.ok() and time.monotonic() - t0 < args.seconds:
        rclpy.spin_once(node, timeout_sec=0.01)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
