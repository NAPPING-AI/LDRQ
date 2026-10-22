"""Combine eval_retreat.py runs (one JSON per seed) into one table: falls summed, means
averaged, worst taken over every seed. No Isaac needed.

    python3 v3/sim/summarize_retreat.py outputs/v3/sim/retreat_dr_r15_s*.json
"""

import json
import sys


def main(paths):
    runs = [json.load(open(p)) for p in sorted(paths)]
    starts = [tuple(r["start"]) for r in runs[0]["rows"]]
    cfg = runs[0]["config"]
    print(f"{len(runs)} runs ({', '.join(str(r['config'].get('seed')) for r in runs)}),"
          f" task {cfg['task']}, retreat {float(cfg['retreat_vx']):+.2f} m/s for {float(cfg['retreat_s']):.2f} s,"
          f" ramp {float(cfg['ramp_s']):.2f} s")
    print()
    print("| start (vx, vy, wz) | falls | overshoot mean / max | reverse s max | net x mean / least back |"
          " back mean / min | stand speed last 1 s max | tilt max deg |")
    print("|---|---|---|---|---|---|---|---|")
    for i, s in enumerate(starts):
        rows = [r["rows"][i] for r in runs]
        assert all(tuple(r["start"]) == s for r in rows)
        falls = sum(r["falls"] for r in rows)
        envs = sum(r["envs"] for r in rows)

        def mean(k):
            return sum(r[k]["mean"] for r in rows) / len(rows)

        def worst(k, which):
            vals = [r[k][which] for r in rows if r[k][which] == r[k][which]]
            if not vals:
                return float("nan")
            return max(vals) if which == "max" else min(vals)

        rev = worst("reverse_s", "max")
        rev_s = "never" if rev != rev else f"{rev:.2f}"
        print(f"| {s[0]:+.2f}, {s[1]:+.2f}, {s[2]:+.2f} | {falls}/{envs} |"
              f" {mean('overshoot'):.3f} / {worst('overshoot', 'max'):.3f} | {rev_s} |"
              f" {mean('net_x'):+.3f} / {worst('net_x', 'max'):+.3f} |"
              f" {mean('back'):.3f} / {worst('back', 'min'):.3f} |"
              f" {worst('stand_speed_last_1s', 'max'):.3f} | {worst('tilt_max_deg', 'max'):.1f} |")


if __name__ == "__main__":
    main(sys.argv[1:])
