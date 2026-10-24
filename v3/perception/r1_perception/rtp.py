"""RTP packets of the head cameras: parse, record, and put frames back together.

The stereo module sends each camera as its own RTP stream (Unitree's video stream service:
left 5002, right 5003, H264). The two streams share no timestamp that gst-launch would
pass on, but every RTP packet carries the 90 kHz timestamp of the frame it belongs to.
Pairing a left frame with the right frame taken at the same moment needs those
timestamps, so V3 reads the packets itself (plan 01 §1.8).

  parse(datagram)          -> Packet (RFC 3550 fixed header, CSRCs and extension skipped)
  DumpWriter / read_dump   the raw-packet file stereo_capture.py writes on the robot
  H264Assembler            packets -> access units (one per frame) as Annex-B bytes,
                           with the frame's RTP timestamp and arrival times (RFC 6184:
                           single NAL units, STAP-A, FU-A)

Python 3.8, standard library only: it runs on the robot.
"""
from __future__ import annotations

import collections
import struct

Packet = collections.namedtuple("Packet", "pt marker seq ts ssrc payload")

DUMP_MAGIC = b"R1RTPDUMP1\n"
_REC = struct.Struct("<HdI")        # port, arrival (CLOCK_MONOTONIC s), datagram length


def parse(data):
    """Packet, or None if `data` is not an RTP version 2 packet."""
    if len(data) < 12 or data[0] >> 6 != 2:
        return None
    cc = data[0] & 0x0F
    has_ext = bool(data[0] & 0x10)
    has_pad = bool(data[0] & 0x20)
    marker = bool(data[1] & 0x80)
    pt = data[1] & 0x7F
    seq, ts, ssrc = struct.unpack("!HII", data[2:12])
    off = 12 + 4 * cc
    if has_ext:
        if len(data) < off + 4:
            return None
        ext_words = struct.unpack("!H", data[off + 2:off + 4])[0]
        off += 4 + 4 * ext_words
    end = len(data)
    if has_pad and end > off:
        end -= data[-1]
    if off > end:
        return None
    return Packet(pt, marker, seq, ts, ssrc, bytes(data[off:end]))


class DumpWriter(object):
    def __init__(self, path):
        self.f = open(path, "wb")
        self.f.write(DUMP_MAGIC)
        self.count = 0

    def write(self, port, t, data):
        self.f.write(_REC.pack(port, t, len(data)))
        self.f.write(data)
        self.count += 1

    def close(self):
        self.f.close()


def read_dump(path):
    """Yield (port, arrival s, datagram) from a stereo_capture.py dump."""
    with open(path, "rb") as f:
        if f.read(len(DUMP_MAGIC)) != DUMP_MAGIC:
            raise ValueError("{} is not a stereo_capture dump".format(path))
        while True:
            head = f.read(_REC.size)
            if len(head) < _REC.size:
                return
            port, t, n = _REC.unpack(head)
            data = f.read(n)
            if len(data) < n:
                return
            yield port, t, data


Frame = collections.namedtuple("Frame", "ts t_first t_last packets complete annexb")

_START = b"\x00\x00\x00\x01"


class H264Assembler(object):
    """Feed packets of ONE stream in arrival order; complete frames come out of push().

    A frame is the packets sharing an RTP timestamp; it ends at the marker bit (or when a
    packet of a new timestamp arrives). `complete` is False if a sequence number was
    skipped inside it or an FU-A was cut short: such a frame must not be decoded as if
    whole, and must not be paired."""

    def __init__(self):
        self._ts = None
        self._nals = []
        self._fu = None
        self._t0 = self._t1 = 0.0
        self._n = 0
        self._ok = True
        self._last_seq = None

    def _close(self):
        if self._ts is None:
            return None
        if self._fu is not None:          # an FU-A without its end fragment
            self._ok = False
        frame = Frame(self._ts, self._t0, self._t1, self._n, self._ok and bool(self._nals),
                      b"".join(_START + n for n in self._nals))
        self._ts, self._nals, self._fu, self._n, self._ok = None, [], None, 0, True
        return frame

    def push(self, pkt, t):
        """Add one packet (from parse()) that arrived at t; returns finished frames."""
        out = []
        if self._ts is not None and pkt.ts != self._ts:
            out.append(self._close())
        gap = self._last_seq is not None and pkt.seq != (self._last_seq + 1) & 0xFFFF
        self._last_seq = pkt.seq
        if self._ts is None:
            self._ts, self._t0, self._ok = pkt.ts, t, not gap
        elif gap:
            self._ok = False
        self._t1 = t
        self._n += 1
        p = pkt.payload
        if p:
            kind = p[0] & 0x1F
            if 1 <= kind <= 23:                       # a single NAL unit
                self._nals.append(p)
            elif kind == 24:                          # STAP-A: several NAL units
                i = 1
                while i + 2 <= len(p):
                    size = struct.unpack("!H", p[i:i + 2])[0]
                    self._nals.append(p[i + 2:i + 2 + size])
                    i += 2 + size
            elif kind == 28 and len(p) >= 2:          # FU-A: a fragment of one NAL unit
                start, end = bool(p[1] & 0x80), bool(p[1] & 0x40)
                if start:
                    self._fu = bytearray([(p[0] & 0xE0) | (p[1] & 0x1F)]) + p[2:]
                elif self._fu is not None:
                    self._fu += p[2:]
                else:
                    self._ok = False                  # a middle fragment without a start
                if end and self._fu is not None:
                    self._nals.append(bytes(self._fu))
                    self._fu = None
        if pkt.marker:
            out.append(self._close())
        return [f for f in out if f is not None]

    def flush(self):
        f = self._close()
        return [f] if f is not None else []
