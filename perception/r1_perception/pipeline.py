"""One camera frame -> one detections message. No ROS: the node, `capture`, `compare`
and the tests all call this, so what is measured is what runs.

The message (JSON on /r1_perception/detections, std_msgs/String) -- mission_ctl reads
it with nothing but `json`:

  {"v": 1, "seq": 17, "t_cap": <time.time() of the frame>, "img": [w, h],
   "hfov_deg": 100.0, "engine": "...",
   "dets": [{"id": 3, "cls": "chair", "cls_id": 56, "conf": 0.81,
             "bbox": [x0, y0, x1, y1],            rectified pixels
             "bearing_deg": 12.3,                  LEFT positive, body frame
             "elev_deg": -8.1,
             "depth_m": 2.41 | null,               median optical depth in the box
             "range_m": 2.38 | null,               horizontal distance, camera to object
             "hits": 4, "confirmed": true}],       consecutive frames (tracker.py)
   "guard": {"person_near": false, "person_m": null, "obstacle": false,
             "obstacle_m": 3.2 | null, "depth_ok": true},
   "timing": {"rect_ms": .., "pre_ms": .., "infer_ms": .., "post_ms": ..}}

`dets` holds everything at or above the SAFETY threshold, confirmed or not; the
mission layer applies the higher TARGET threshold and the confirmation itself. The
guard is computed here, on this frame alone.
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import camera_model as C
from . import depth as D
from . import tracker as T
from . import yolox as Y

MSG_VERSION = 1


def read_flat_yaml(path):
    """`key: value` per line, # comments. Same reader as mission_ctl's: no PyYAML on the
    robot."""
    out = {}
    for raw in open(str(path)).read().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip()
        if not v:
            continue
        try:
            out[k.strip()] = float(v) if ("." in v or "e" in v.lower()) else int(v)
        except ValueError:
            out[k.strip()] = v.strip("\"'")
    return out


class Pipeline(object):
    def __init__(self, cfg, runtime):
        self.cfg = cfg
        self.rt = runtime
        self.mount = C.Mount(cfg.get("cam_height_m", 1.2), cfg.get("cam_pitch_deg", 0.0),
                             cfg.get("cam_yaw_deg", 0.0))
        self.passthrough = int(cfg.get("rectify", 1)) == 0
        if self.passthrough:
            # The source is already a pinhole image (the stereo module's left camera):
            # its own intrinsics are the geometry, no remap.
            self.rect = C.Rectified(cfg["cam_width"], cfg["cam_height"], fx=cfg["cam_fx"],
                                    fy=cfg.get("cam_fy", cfg["cam_fx"]),
                                    cx=cfg.get("cam_cx"), cy=cfg.get("cam_cy"))
            self.fish, self.map_x, self.map_y = None, None, None
        else:
            self.fish = C.Fisheye.from_cfg(cfg)
            self.rect = C.Rectified(cfg.get("rect_width", 640), cfg.get("rect_height", 480),
                                    cfg.get("rect_hfov_deg", 100.0))
            self.map_x, self.map_y = self.rect.maps(self.fish)
        self.tracker = T.Tracker(cfg.get("confirm_frames", 3), cfg.get("track_iou", 0.3),
                                 cfg.get("track_max_misses", 2))
        self.safety_conf = float(cfg.get("safety_conf", 0.30))
        self.nms_thr = float(cfg.get("nms_iou", 0.45))
        self.seq = 0
        self.last_rect = None

    # ------------------------------------------------------------------ steps
    def rectify(self, bgr, depth=None):
        if self.passthrough:
            return bgr, depth
        r = C.remap(bgr, self.map_x, self.map_y)
        d = None
        if depth is not None:
            d = C.remap(depth, self.map_x, self.map_y, nearest=True)
        return r, d

    def _infer(self, x, ratio, hw):
        t1 = time.perf_counter()
        raw = self.rt.infer(x)
        t2 = time.perf_counter()
        dets = Y.postprocess(raw, ratio, self.safety_conf, self.nms_thr, image_wh=(hw[1], hw[0]))
        t3 = time.perf_counter()
        return dets, {"infer_ms": (t2 - t1) * 1e3, "post_ms": (t3 - t2) * 1e3}

    def describe(self, dets, depth):
        """Detections -> message entries with angles, depth and range."""
        out = []
        for (x0, y0, x1, y1, s, c) in dets:
            u, v = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            xn, yn = self.rect.ray(u, v)
            b, e = self.mount.bearing_elev_deg(xn, yn)
            dz = None
            if depth is not None:
                dz = D.box_depth(depth, (x0, y0, x1, y1), self.cfg.get("box_shrink", 0.5),
                                 self.cfg.get("depth_min_m", 0.15),
                                 self.cfg.get("depth_max_m", 10.0),
                                 self.cfg.get("depth_min_valid", 0.1))
            rng = float(self.mount.range_m(xn, yn, dz)) if dz is not None else None
            out.append({"cls": Y.COCO_CLASSES[c], "cls_id": int(c), "conf": round(float(s), 4),
                        "bbox": [round(float(a), 1) for a in (x0, y0, x1, y1)],
                        "bearing_deg": round(float(b), 2), "elev_deg": round(float(e), 2),
                        "depth_m": None if dz is None else round(dz, 3),
                        "range_m": None if rng is None else round(rng, 3)})
        return out

    def guard(self, entries, depth):
        """Per-frame safety facts for the mission layer: a person inside person_stop_m
        (from depth, or -- with no depth for that box -- a box taller than
        person_box_frac of the image, which at this field of view is a person about an
        arm's length away), and returns inside the walking corridor."""
        stop_p = float(self.cfg.get("person_stop_m", 1.2))
        frac = float(self.cfg.get("person_box_frac", 0.8))
        person_m, near = None, False
        for e in entries:
            if e["cls_id"] != Y.PERSON or e["conf"] < self.safety_conf:
                continue
            if e["range_m"] is not None:
                person_m = e["range_m"] if person_m is None else min(person_m, e["range_m"])
                near = near or e["range_m"] < stop_p
            elif (e["bbox"][3] - e["bbox"][1]) >= frac * self.rect.h:
                near = True
        g = {"person_near": bool(near), "person_m": None if person_m is None else round(person_m, 3),
             "obstacle": False, "obstacle_m": None, "depth_ok": depth is not None}
        if depth is not None:
            obs, nearest, _ = D.corridor(
                depth, self.rect, self.mount, float(self.cfg.get("obstacle_stop_m", 0.6)),
                float(self.cfg.get("corridor_half_width_m", 0.35)),
                float(self.cfg.get("corridor_min_height_m", 0.10)),
                float(self.cfg.get("corridor_max_height_m", 1.9)),
                int(self.cfg.get("corridor_step", 4)),
                int(self.cfg.get("corridor_min_points", 30)))
            g["obstacle"] = bool(obs)
            g["obstacle_m"] = None if nearest is None else round(nearest, 3)
        return g

    # ------------------------------------------------------------------- all
    def process(self, bgr, depth=None, t_cap=None):
        """Raw camera frame (+ raw-aligned depth in meters) -> (message, rectified BGR,
        rectified depth)."""
        return self.finish(self.prepare(bgr, depth, t_cap))

    def prepare(self, bgr, depth=None, t_cap=None):
        """The CPU half: rectify + letterbox. The node runs it one policy step ahead, so
        that when its slot comes the GPU starts at once (robot, 10-14: done inside the
        slot it took 6-9 ms of CPU first, and with the LLM busy on the CPU the GPU work
        ran into the policy's next step)."""
        t0 = time.perf_counter()
        rect, rdepth = self.rectify(bgr, depth)
        t1 = time.perf_counter()
        x, ratio = Y.letterbox(rect)
        t2 = time.perf_counter()
        return {"rect": rect, "depth": rdepth, "x": x, "ratio": ratio, "t_cap": t_cap,
                "rect_ms": (t1 - t0) * 1e3, "pre_ms": (t2 - t1) * 1e3}

    def finish(self, prep):
        """The GPU half and what follows it: infer, decode, describe, track, guard."""
        rect, rdepth, t_cap = prep["rect"], prep["depth"], prep["t_cap"]
        dets, timing = self._infer(prep["x"], prep["ratio"], rect.shape[:2])
        entries = self.describe(dets, rdepth)
        for e, tr in zip(entries, self.tracker.update(dets)):
            e.update(tr)
        g = self.guard(entries, rdepth)
        timing["rect_ms"], timing["pre_ms"] = prep["rect_ms"], prep["pre_ms"]
        self.seq += 1
        msg = {"v": MSG_VERSION, "seq": self.seq,
               "t_cap": float(t_cap if t_cap is not None else time.time()),
               "img": [self.rect.w, self.rect.h], "hfov_deg": self.rect.hfov_deg,
               "engine": getattr(self.rt, "name", "?"), "dets": entries, "guard": g,
               "timing": {k: round(v, 3) for k, v in timing.items()}}
        self.last_rect = rect
        return msg, rect, rdepth


def depth_overlay(rect_bgr, depth, dmax=5.0):
    """The image with its depth on top (colour = distance, nothing = no return): if the
    two are registered, object edges in colour sit on object edges in the picture."""
    import cv2
    d8 = (np.clip(depth / dmax, 0.0, 1.0) * 255).astype(np.uint8)
    col = cv2.applyColorMap(255 - d8, cv2.COLORMAP_JET)
    out = rect_bgr.copy()
    ok = depth > 0
    out[ok] = (0.45 * rect_bgr[ok] + 0.55 * col[ok]).astype(np.uint8)
    return out


def draw(rect_bgr, msg, highlight=None, target_conf=None):
    """The operator's confirmation picture: every box, the target thick and labeled
    with what the robot will act on (class, confidence, bearing, range)."""
    import cv2
    img = rect_bgr.copy()
    for d in msg.get("dets", []):
        x0, y0, x1, y1 = (int(round(v)) for v in d["bbox"])
        hl = highlight is not None and d.get("id") == highlight
        weak = target_conf is not None and d["conf"] < target_conf
        color = (0, 0, 255) if hl else ((150, 150, 150) if weak else (0, 200, 0))
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 4 if hl else 1)
        side = "L" if d["bearing_deg"] >= 0 else "R"
        label = "{} {:.2f} {:.0f}{}".format(d["cls"], d["conf"], abs(d["bearing_deg"]), side)
        if d.get("range_m") is not None:
            label += " {:.2f}m".format(d["range_m"])
        if hl:
            label = "TARGET " + label
        cv2.putText(img, label, (x0 + 2, max(14, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 if not hl else 0.6, color, 2 if hl else 1, cv2.LINE_AA)
    cx = int(round((img.shape[1] - 1) / 2.0))
    cv2.line(img, (cx, 0), (cx, img.shape[0] - 1), (255, 255, 0), 1)
    return img


def angle_sign_text(deg):
    return "{:.1f} deg {}".format(abs(deg), "left" if deg >= 0 else "right") \
        if abs(deg) >= 0.05 else "straight ahead"


def fmt_m(v):
    return "-" if v is None or not math.isfinite(v) else "{:.2f} m".format(v)
