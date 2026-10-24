#!/usr/bin/env python3
"""Take a stereo_capture.py dump apart into decoded frames (dev box, V3 B0/B1).

    python3 perception/tools/stereo_unpack.py ~/o/b0/turning.rtpdump --out ~/o/b0/turning

For each port it rebuilds the frames from the RTP packets (rtp.H264Assembler), writes
<out>/<port>.h264 (Annex B) and <out>/<port>_frames.csv (one row per frame: index, RTP
timestamp, first and last packet arrival, packet count, complete), and decodes the H264
with ffmpeg into <out>/<port>.npy (frames x H x W uint8 grey; the stereo matcher only needs
grey). Row i of the CSV is image i of the npy, and the counts are checked. That only holds
if every frame handed to the decoder decodes, so a frame is written only when it can be:
from the first keyframe (IDR) on, and after a frame with lost packets, not again until the
next keyframe (the frames in between reference a broken one). Skipped frames are counted.

Needs an ffmpeg binary: --ffmpeg, else $FFMPEG, else `ffmpeg` on PATH, else the one that
imageio-ffmpeg ships in the env_isaaclab conda env.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r1_perception import rtp  # noqa: E402


def find_ffmpeg(explicit=""):
    for cand in (explicit, os.environ.get("FFMPEG", ""), shutil.which("ffmpeg") or ""):
        if cand and os.path.exists(cand):
            return cand
    found = sorted(glob.glob(os.path.expanduser(
        "~/miniconda3/envs/env_isaaclab/lib/python3*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*")))
    if found:
        return found[0]
    raise SystemExit("no ffmpeg: pass --ffmpeg")


def frames_by_port(dump):
    asm, out = {}, {}
    for port, t, data in rtp.read_dump(dump):
        pkt = rtp.parse(data)
        if pkt is None:
            continue
        a = asm.setdefault(port, rtp.H264Assembler())
        out.setdefault(port, []).extend(a.push(pkt, t))
    for port, a in asm.items():
        out[port].extend(a.flush())
    return out


def nal_types(annexb):
    return {chunk[0] & 0x1F for chunk in annexb.split(b"\x00\x00\x00\x01") if chunk}


def decodable(frames):
    """The frames a decoder can turn into images, in order: from a keyframe on, and never
    between an incomplete frame and the next keyframe."""
    keep, ok = [], False
    for fr in frames:
        if 5 in nal_types(fr.annexb):         # IDR slice
            ok = True
        if not fr.complete:
            ok = False
        if ok:
            keep.append(fr)
    return keep


def decode(ffmpeg, h264_path, w, h):
    """All frames of an Annex-B file as (N, h, w) uint8 grey, in stream order."""
    cmd = [ffmpeg, "-loglevel", "error", "-f", "h264", "-i", str(h264_path), "-vsync", "passthrough",
           "-f", "rawvideo", "-pix_fmt", "gray", "-s", "{}x{}".format(w, h), "-"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dump")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=544)
    ap.add_argument("--height", type=int, default=448)
    ap.add_argument("--ffmpeg", default="")
    args = ap.parse_args()
    out = pathlib.Path(os.path.expanduser(args.out))
    out.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg(args.ffmpeg)
    status = 0
    for port, received in sorted(frames_by_port(os.path.expanduser(args.dump)).items()):
        frames = decodable(received)
        h264 = out / "{}.h264".format(port)
        with open(h264, "wb") as f:
            for fr in frames:
                f.write(fr.annexb)
        with open(out / "{}_frames.csv".format(port), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["index", "rtp_ts", "t_first", "t_last", "packets", "complete"])
            for i, fr in enumerate(frames):
                w.writerow([i, fr.ts, "{:.6f}".format(fr.t_first), "{:.6f}".format(fr.t_last), fr.packets,
                            int(fr.complete)])
        imgs = decode(ffmpeg, h264, args.width, args.height)
        np.save(str(out / "{}.npy".format(port)), imgs)
        same = len(imgs) == len(frames)
        status |= 0 if same else 1
        print("port {}: {} frames received ({} incomplete), {} decodable, {} decoded{}".format(
            port, len(received), sum(1 for f in received if not f.complete), len(frames), len(imgs),
            "" if same else "  <-- COUNTS DIFFER: the CSV rows no longer match the images"))
    return status


if __name__ == "__main__":
    sys.exit(main())
