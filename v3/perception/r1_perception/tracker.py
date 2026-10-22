"""Frame-to-frame consistency: a detection counts as a hit for the mission layer only
after `confirm_frames` consecutive frames agree (same class, overlapping box).

This is for TARGETS. The safety check (a person too close) deliberately does not wait
for it: it acts on every single frame, at a lower threshold, because a missed person
costs more than a spurious stop. The two thresholds are calibrated separately
(tools/eval_detect.py).
"""
from __future__ import annotations


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class Tracker(object):
    """Greedy IoU association per class. `hits` counts consecutive frames; one missed
    frame resets it to 0 and a track is dropped after `max_misses` misses in a row."""

    def __init__(self, confirm_frames=3, iou_thr=0.3, max_misses=2):
        self.n = int(confirm_frames)
        self.iou_thr = float(iou_thr)
        self.max_misses = int(max_misses)
        self.tracks = []
        self._next = 1

    def update(self, dets):
        """dets: list of (x0, y0, x1, y1, score, class_id). Returns one dict per
        detection: id, hits, confirmed -- in the order given."""
        pairs = []
        for i, d in enumerate(dets):
            for j, t in enumerate(self.tracks):
                if t["cls"] == d[5]:
                    o = iou(d[:4], t["box"])
                    if o >= self.iou_thr:
                        pairs.append((o, i, j))
        pairs.sort(reverse=True)
        used_d, used_t, match = set(), set(), {}
        for o, i, j in pairs:
            if i in used_d or j in used_t:
                continue
            used_d.add(i)
            used_t.add(j)
            match[i] = j
        out = []
        for i, d in enumerate(dets):
            if i in match:
                t = self.tracks[match[i]]
                t["hits"] += 1
                t["misses"] = 0
                t["box"] = d[:4]
            else:
                t = {"id": self._next, "cls": d[5], "box": d[:4], "hits": 1, "misses": 0}
                self._next += 1
                self.tracks.append(t)
                used_t.add(len(self.tracks) - 1)
            out.append({"id": t["id"], "hits": t["hits"], "confirmed": t["hits"] >= self.n})
        keep = []
        for j, t in enumerate(self.tracks):
            if j not in used_t:
                t["misses"] += 1
                t["hits"] = 0
            if t["misses"] <= self.max_misses:
                keep.append(t)
        self.tracks = keep
        return out
