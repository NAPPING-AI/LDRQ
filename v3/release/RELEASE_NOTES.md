# r1-robot v@VERSION@ (V3): a dance, and stepping back from people

**Final release of the project.** V3 ran on the robot in a gantry, with the output off and
with the motors on (2026-10-24); that completed the project's engineering (2026-10-26). It
was validated in the gantry only and has never run untethered.

- **A dance policy** (`dance_wave`, run `2026-10-23_16-51-44_wave`, trained with BeyondMimic's
  motion tracking): a 10 s routine on the spot, both feet on the floor.
  `ask "dance for 20 seconds"` starts it only if nobody is within 1 m. A dance ends at the
  end of a bar, on Ctrl-C at the next bar, and lasts at most 60 s.
- **Two policies in one runner.** v2 (the walking policy, unchanged) and the dance policy
  are inferred every control step: at most 2.7 ms together at p99 on the Orin. A state
  machine in the bridge (the conductor) chooses which one drives the motors.
- **A person reflex in the bridge.** Within 0.6 m of a person (0.9 m while dancing), v2
  takes over on the same control step and steps back for 1.5 s (about 0.33 m). The robot
  then stands until the operator confirms a new plan. The language model is not in this
  loop.
- **Vision:** the head camera's left image over the robot's network, and YOLOX-Nano as a
  TensorRT FP16 engine built on the robot. A person's distance comes from the height of
  their box (`person_box_k`, calibrated without tape).
- **One install script.** `install.sh` installs V3 into its own tree, `r1-robot-v@VERSION@/v3/`.
  It does this the way the robot was installed on 2026-10-24: the v2 bundle into `base/`,
  the vision engine, then `install_v3.sh`. Every engine is built on the robot and gated by
  parity. `base/` stays a working v2 tree for rollback.

**On the robot, 2026-10-24 (gantry):**
- the bridge stayed RUNNING throughout;
- the robot danced and handed back to v2 at the end of a bar, and stopped at the next bar
  on Ctrl-C;
- it stepped back from the dance when a person came within 0.79–0.88 m;
- a dance asked for with someone too close was refused.

**In simulation:** 0 falls in 2017 normal and 2044 emergency handovers.

**Known limits:**
- validated in the gantry only, never untethered: keep it in the gantry;
- the reflex acts on people only: obstacles would need depth from the two head images, which was not finished;
- the camera looks forward only, so the robot backs off into space it cannot see;
- one `DEGRADED: joint_target stale` was seen once with the output off; a restart cleared it.

## Assets

| file | | |
|---|---|---|
| `release_v@VERSION@.tar.gz` | about 0.5 GB | code, both walking bundles (v2, v1), the dance policy, YOLOX-Nano, the Ollama runtime for JetPack 5, install.sh, READMEs (English, 中文) |
| `qwen3-1.7b_r1-robot-v@VERSION@.tar.gz` | 1.3 GB | the language model; it extracts into the same directory |
| `SHA256SUMS` | | `sha256sum -c SHA256SUMS` |

Both archives extract to `~/r1-robot-v@VERSION@`. Earlier trees (`~/r1-robot-v1.1.0`,
`~/r1-robot-v2.0.0`, `~/r1-robot-v3`) are left alone. Install, use and safety: `README.md` /
`README_zh.md` in the archive. Built from git `@GIT@` (branch `v3`); the detailed account is
`v3/README.md` there.
