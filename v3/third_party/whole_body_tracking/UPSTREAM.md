# whole_body_tracking (BeyondMimic), vendored

- Upstream: https://github.com/HybridRobotics/whole_body_tracking (archived, read-only)
- Commit: `9db915838b17785c6fec90c91fe80b95b147e1bd` (2026-10-15), copied 2026-10-23
- License: MIT, see `LICENCE` (kept unchanged)
- Requires Isaac Lab 2.1.0 and rsl-rl-lib 2.3.1, the versions this project trains with.

**This directory is the upstream tree unchanged**, except that its dotfiles (`.gitattributes` with Git LFS filters, `.gitignore`, `.flake8`, `.pre-commit-config.yaml`) are left out so they cannot change how this repository treats files. V3's R1 dance task, the motion
conversion and the training entry point live in `v3/dance/`. They import this package
and override what R1 needs (robot, bodies, gains, no WandB registry) in their own files,
so `diff -r` against the upstream commit stays empty.
