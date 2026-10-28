# r1-robot v@VERSION@ (V3): the head stays where it is

**A patch to v3.0.1:** on the robot, the head no longer turns to the right when the stack
starts. Nothing else changed. V3 is validated in the gantry only and has never run
untethered.

- **The fix.** The policy does not drive the two head joints, so the bridge holds them. Up
  to v3.0.1 it held them at the URDF zero. On the robot, that turned the head about 43
  degrees to the right every time driving started, along with the first steps in place.
  The cause: with the head facing straight ahead, the head encoder does not read 0, and the
  reading changes between sessions (+0.75 rad in one, +2.11 in the next, the same day).
  The bridge now holds the head where it is when driving starts, and logs that pose
  (`head held where it is: ...`). It feeds the policy the head's offset from that pose, so
  the policy sees the head at 0, as in training.
- **What the operator does:** face the head straight ahead before starting the stack. In
  developer mode the head has no stiffness, and its pitch can sag: set it by hand.
- **On the robot, 2026-10-28 (gantry, motors on):** the head stayed still as driving
  started and while the robot stepped in place.

Everything else is as in v3.0.1: the dance policy, two policies in one runner, the person
reflex, vision, and the one install script. See `README.md` in the archive, and the
[v3.0.0 release](https://github.com/NAPPING-AI/LDRQ/releases/tag/v3.0.0) notes for the
V3 features.

**Known limits** (unchanged):
- validated in the gantry only, never untethered: keep it in the gantry;
- the reflex acts on people only: obstacles would need depth from the two head images, which was not finished;
- the camera looks forward only, so the robot backs off into space it cannot see;
- one `DEGRADED: joint_target stale` was seen once with the output off; a restart cleared it.

## Assets

| file | | |
|---|---|---|
| `release_v@VERSION@.tar.gz` | about 0.5 GB | code, both walking bundles (v2, v1), the dance policy, YOLOX-Nano, the Ollama runtime for JetPack 5, install.sh, READMEs (English, 中文) |
| `SHA256SUMS` | | `sha256sum -c SHA256SUMS` |

**The language model is not attached.** Use `qwen3-1.7b_r1-robot-v3.0.0.tar.gz` from the
[v3.0.0 release](https://github.com/NAPPING-AI/LDRQ/releases/tag/v3.0.0), the same
qwen3:1.7b (sha256 `7713eadc2c0c8aee9565322994a1d21355fe6ed44a5fb7d39a33d30334780f0a`).
Extract it into this release's directory:

```bash
tar xzf release_v@VERSION@.tar.gz
tar xzf qwen3-1.7b_r1-robot-v3.0.0.tar.gz --strip-components=1 -C r1-robot-v@VERSION@
```

Everything extracts to `~/r1-robot-v@VERSION@`. Earlier trees (`~/r1-robot-v3`,
`~/r1-robot-v3.0.0`, `~/r1-robot-v3.0.1`, ...) are left alone. Built from git `@GIT@`
(branch `v3`); the detailed account is `v3/README.md` there.
