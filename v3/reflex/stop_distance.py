"""V3 stop distance: how far from a person or an obstacle forward walking must stop.

The retreat triggers at a fixed distance (person 0.6 m, obstacle 0.45 m). Walking
toward something must stop early enough that the robot comes to rest outside that
distance, or every approach ends in a retreat. The stop distance therefore grows with
speed, and every term carries a margin:

    stop(v) = max(floor, trigger + margin + k_lat * v * latency + k_brake * brake(v))

- floor: the fixed distances used so far (perception.yaml person_stop_m 1.2,
  obstacle_stop_m 0.6). The formula only ever raises them.
- margin 0.10 m: between where the robot comes to rest and the trigger.
- latency: from a camera frame to the command in the policy's observation. 0.25 s
  is an estimate until it is measured on the robot; then it is the measured p99,
  not the mean.
  k_lat 1.5 on top.
- brake(v): the distance the v2 policy travels after its command drops to zero,
  the WORST env of 5 seeds x 256 envs under the training randomization (2026-10-22:
  v3/sim/eval_retreat.py --retreat_vx 0 on Isaac-Velocity-Flat-R1-v2, seeds 1-5,
  combined by v3/sim/summarize_retreat.py; table in v3/README.md), linearly
  interpolated. k_brake 1.3 on top.
- v: the largest forward command of the last `window_s` (1 s), not the current
  one, so the distance does not shrink the moment the command does while the body
  is still moving.

Run it to print the table:  python3 v3/reflex/stop_distance.py
"""

# (forward command m/s, worst braking distance m) -- training task, seeds 1-5, 256 envs each
BRAKE_WORST = ((0.0, 0.068), (0.3, 0.157), (0.5, 0.228), (1.0, 0.470))

DEFAULTS = {
    "person": {"trigger_m": 0.6, "floor_m": 1.2},
    "obstacle": {"trigger_m": 0.45, "floor_m": 0.6},
    "margin_m": 0.10,
    "latency_s": 0.25,
    "k_lat": 1.5,
    "k_brake": 1.3,
    "window_s": 1.0,
}


def brake(v, table=BRAKE_WORST):
    """Worst braking distance at forward speed v (linear; extrapolated past the ends)."""
    v = max(0.0, float(v))
    for (v0, d0), (v1, d1) in zip(table, table[1:]):
        if v <= v1:
            return d0 + (d1 - d0) * (v - v0) / (v1 - v0)
    (v0, d0), (v1, d1) = table[-2], table[-1]
    return d1 + (d1 - d0) * (v - v1) / (v1 - v0)


def stop_distance(kind, v, cfg=DEFAULTS):
    """Stop distance for `kind` ("person" or "obstacle") at forward speed v (m/s)."""
    k = cfg[kind]
    v = max(0.0, float(v))
    need = (k["trigger_m"] + cfg["margin_m"] + cfg["k_lat"] * v * cfg["latency_s"]
            + cfg["k_brake"] * brake(v))
    return max(k["floor_m"], need)


def table(speeds=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0), cfg=DEFAULTS):
    rows = ["| forward command m/s | worst braking m | person stop m | obstacle stop m |",
            "|---|---|---|---|"]
    for v in speeds:
        rows.append("| {:.1f} | {:.3f} | {:.2f} | {:.2f} |".format(
            v, brake(v), stop_distance("person", v, cfg), stop_distance("obstacle", v, cfg)))
    return "\n".join(rows)


if __name__ == "__main__":
    print(table())
