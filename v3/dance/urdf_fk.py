"""Forward kinematics of R1 from its URDF, numpy only.

The dance checks (check_routine.py) and the motion export (routine.py) need link
positions for a joint configuration: where the hands are relative to the camera, and
where the pelvis has to be so that the soles stay on the floor while the knees bend.
Isaac Lab could answer that too, but these run under plain Python (3.8 on the robot's
test box) and in run_tests.sh, so they read assets/r1/R1.urdf themselves.

Only what R1.urdf uses is supported: fixed and revolute joints, origin xyz/rpy and a
joint axis. Fixed joints are kept as links (the ankle rods are leaves and do not matter).
"""
from __future__ import annotations

import math
import pathlib
import xml.etree.ElementTree as ET

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
URDF = REPO / "assets" / "r1" / "R1.urdf"


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_angle_matrix(axis, angle):
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def read_stl(path):
    """(N, 3) vertices of an ASCII or binary STL."""
    raw = pathlib.Path(path).read_bytes()
    if raw[:5] == b"solid" and b"facet" in raw[:512]:
        return np.array([[float(x) for x in line.split()[1:4]] for line in raw.decode().splitlines()
                         if line.strip().startswith("vertex")])
    n = int.from_bytes(raw[80:84], "little")
    rec = np.frombuffer(raw[84:84 + 50 * n], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return rec["v"].reshape(-1, 3).astype(float)


def _floats(text, default):
    return [float(v) for v in text.split()] if text else list(default)


class Robot:
    """The kinematic tree of a URDF, rooted at the first link that is no joint's child."""

    def __init__(self, path=URDF):
        root = ET.parse(str(path)).getroot()
        self.joints = {}            # child link -> (name, type, parent, R0, p0, axis, (lo, hi))
        self.limits = {}            # joint name -> (lower, upper)
        self.velocity_limits = {}   # joint name -> rad/s
        children = set()
        for j in root.findall("joint"):
            origin = j.find("origin")
            xyz = _floats(origin.get("xyz") if origin is not None else None, (0, 0, 0))
            rpy = _floats(origin.get("rpy") if origin is not None else None, (0, 0, 0))
            axis_el = j.find("axis")
            axis = np.array(_floats(axis_el.get("xyz") if axis_el is not None else None, (1, 0, 0)))
            lim = j.find("limit")
            rng = (float(lim.get("lower")), float(lim.get("upper"))) if lim is not None and lim.get("lower") else None
            parent, child = j.find("parent").get("link"), j.find("child").get("link")
            self.joints[child] = (j.get("name"), j.get("type"), parent, rpy_matrix(*rpy), np.array(xyz),
                                  axis / (np.linalg.norm(axis) or 1.0), rng)
            if j.get("type") in ("revolute", "continuous") and rng:
                self.limits[j.get("name")] = rng
                self.velocity_limits[j.get("name")] = float(lim.get("velocity", "inf"))
            children.add(child)
        self.meshes = {}            # link -> (mesh path, R, p) of its collision mesh (else visual)
        for l in root.findall("link"):
            for kind in ("collision", "visual"):
                mesh = l.find(kind + "/geometry/mesh")
                if mesh is None:
                    continue
                origin = l.find(kind + "/origin")
                xyz = _floats(origin.get("xyz") if origin is not None else None, (0, 0, 0))
                rpy = _floats(origin.get("rpy") if origin is not None else None, (0, 0, 0))
                self.meshes[l.get("name")] = (pathlib.Path(path).parent / mesh.get("filename"),
                                              rpy_matrix(*rpy), np.array(xyz))
                break
        links = [l.get("name") for l in root.findall("link")]
        roots = [l for l in links if l not in children]
        # R1.urdf starts with a "world" link joined to the pelvis by a fixed joint; FK is
        # wanted relative to the pelvis, which is what the floating base is in simulation.
        self.root = "pelvis_link" if "pelvis_link" in links else roots[0]
        self.links = links

    def fk(self, q, base_R=None, base_p=None):
        """{link: (R, p)} in the world, for joint angles q (dict name -> rad, missing = 0)."""
        base_R = np.eye(3) if base_R is None else np.asarray(base_R)
        base_p = np.zeros(3) if base_p is None else np.asarray(base_p, dtype=float)
        out = {self.root: (base_R, base_p)}

        def pose(link):
            if link in out:
                return out[link]
            name, jtype, parent, R0, p0, axis, _ = self.joints[link]
            Rp, pp = pose(parent)
            R = Rp @ R0
            p = pp + Rp @ p0
            if jtype in ("revolute", "continuous"):
                R = R @ axis_angle_matrix(axis, float(q.get(name, 0.0)))
            out[link] = (R, p)
            return out[link]

        for link in self.links:
            if link == self.root or link not in self.joints:
                continue
            if self._descends_from_root(link):
                pose(link)
        return out

    def mesh_points(self, link, max_points=400):
        """Vertices of a link's mesh in the link frame (deduplicated, evenly subsampled)."""
        path, R, p = self.meshes[link]
        v = np.unique(np.round(read_stl(path), 5), axis=0)
        if len(v) > max_points:
            v = v[np.linspace(0, len(v) - 1, max_points).astype(int)]
        return v @ R.T + p

    def _descends_from_root(self, link):
        while link in self.joints:
            parent = self.joints[link][2]
            if parent == self.root:
                return True
            link = parent
        return link == self.root
