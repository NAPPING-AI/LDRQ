#!/usr/bin/env python3
"""Record the head cameras' raw RTP packets, for building our own stereo depth (V3 B0).

    python3 perception/tools/stereo_capture.py --seconds 120 --label turning --out ~/o/b0
    python3 perception/tools/stereo_capture.py --probe            # 3 s, then the summary only

Nothing is decoded here. Every UDP datagram that arrives on --ports (left 5002 and right
5003 by default) is written with its port and CLOCK_MONOTONIC arrival time to
<out>/<label>.rtpdump; the dev box takes the frames apart afterwards
(stereo_unpack.py). Keeping the packets keeps each frame's own RTP timestamp, which is
what pairs a left frame with the right frame taken at the same moment.

At the end it prints, per port: packets, frames (marker bits), frames per second, how
many frames were incomplete (lost packets), the RTP clock step between frames, and the
SSRC. With two ports it also compares the two streams' timestamps (same clock or not).

The ports must be free: stop r1_perception first (it listens on 5002). The APP must
have video_hub OFF and Stereo patch PC1 ON, as for the perception node. Python 3.8,
standard library only.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import selectors
import socket
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r1_perception import rtp  # noqa: E402


def open_ports(ports):
    sel = selectors.DefaultSelector()
    socks = []
    for port in ports:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
        try:
            s.bind(("0.0.0.0", port))
        except OSError as e:
            raise SystemExit("port {} is busy ({}): stop r1_perception (or anything else "
                             "receiving the stream) first".format(port, e))
        s.setblocking(False)
        sel.register(s, selectors.EVENT_READ, port)
        socks.append(s)
    return sel, socks


def summarize(per_port):
    """Lines describing each stream, and the two streams against each other."""
    lines = []
    first_ts = {}
    for port, info in sorted(per_port.items()):
        frames = info["frames"]
        if not frames:
            lines.append("port {}: {} packets, no complete frame".format(port, info["packets"]))
            continue
        span = frames[-1].t_last - frames[0].t_first
        fps = (len(frames) - 1) / span if span > 0 else 0.0
        steps = [b.ts - a.ts for a, b in zip(frames, frames[1:])]
        step = statistics.median(steps) if steps else 0
        bad = sum(1 for f in frames if not f.complete)
        lines.append("port {}: {} packets, {} frames, {:.2f} fps, {} incomplete, RTP step {} "
                     "({:.1f} ms at 90 kHz), payload type {}, SSRC {:08x}".format(
                         port, info["packets"], len(frames), fps, bad, step, step / 90.0,
                         info["pt"], info["ssrc"]))
        first_ts[port] = frames
    if len(first_ts) == 2:
        (pa, fa), (pb, fb) = sorted(first_ts.items())
        # same clock: the arrival-matched frames differ by a constant RTP offset
        offs = []
        j = 0
        for f in fa:
            while j + 1 < len(fb) and abs(fb[j + 1].t_last - f.t_last) <= abs(fb[j].t_last - f.t_last):
                j += 1
            offs.append((fb[j].ts - f.ts) & 0xFFFFFFFF)
        offs = [o - (1 << 32) if o >= (1 << 31) else o for o in offs]
        lines.append("ports {} vs {}: RTP offset of the nearest-arriving frames: median {}, "
                     "spread (p10..p90) {}..{} -- a constant offset means one clock".format(
                         pa, pb, int(statistics.median(offs)),
                         int(sorted(offs)[len(offs) // 10]), int(sorted(offs)[(len(offs) * 9) // 10])))
        dt = [abs(fb[min(range(len(fb)), key=lambda k: abs(fb[k].t_last - f.t_last))].t_last - f.t_last)
              for f in fa[:200]]
        lines.append("ports {} vs {}: arrival gap to the nearest other-camera frame: median {:.1f} ms, "
                     "max {:.1f} ms".format(pa, pb, 1000 * statistics.median(dt), 1000 * max(dt)))
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ports", default="5002,5003")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--label", default="capture")
    ap.add_argument("--out", default="~/orin_commissioning/b0")
    ap.add_argument("--probe", action="store_true", help="3 s, no file: just the summary")
    args = ap.parse_args()
    ports = [int(p) for p in args.ports.split(",") if p]
    seconds = 3.0 if args.probe else args.seconds

    writer = None
    if not args.probe:
        out = pathlib.Path(os.path.expanduser(args.out))
        out.mkdir(parents=True, exist_ok=True)
        path = out / "{}.rtpdump".format(args.label)
        writer = rtp.DumpWriter(str(path))
    sel, socks = open_ports(ports)
    per_port = {p: {"packets": 0, "frames": [], "asm": rtp.H264Assembler(), "pt": None, "ssrc": 0}
                for p in ports}
    t_end = time.monotonic() + seconds
    print("recording {} for {:g} s{}".format(ports, seconds, "" if writer is None else " -> " + str(path)),
          flush=True)
    next_note = time.monotonic() + 10.0
    try:
        while time.monotonic() < t_end:
            for key, _ in sel.select(timeout=0.2):
                while True:
                    try:
                        data = key.fileobj.recv(65536)
                    except BlockingIOError:
                        break
                    t = time.monotonic()
                    port = key.data
                    if writer is not None:
                        writer.write(port, t, data)
                    info = per_port[port]
                    info["packets"] += 1
                    pkt = rtp.parse(data)
                    if pkt is None:
                        continue
                    info["pt"], info["ssrc"] = pkt.pt, pkt.ssrc
                    for f in info["asm"].push(pkt, t):
                        info["frames"].append(f._replace(annexb=b""))   # stats only
            if time.monotonic() >= next_note:
                next_note += 10.0
                print("  {:.0f} s left: {}".format(t_end - time.monotonic(), ", ".join(
                    "{} {} frames".format(p, len(i["frames"])) for p, i in per_port.items())), flush=True)
    except KeyboardInterrupt:
        print("stopped early", flush=True)
    finally:
        for s in socks:
            s.close()
        if writer is not None:
            writer.close()
    for line in summarize(per_port):
        print(line)
    if writer is not None:
        print("wrote {} packets to {} ({:.1f} MB)".format(writer.count, path, path.stat().st_size / 1e6))
    return 0 if all(per_port[p]["frames"] for p in ports) else 1


if __name__ == "__main__":
    sys.exit(main())
