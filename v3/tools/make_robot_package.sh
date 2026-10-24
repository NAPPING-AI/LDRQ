#!/usr/bin/env bash
# Dev box: the V3 robot package -- a tar.gz that unpacks to r1_v3_<n>_<date>/ with
# install_v3.sh, the V3 code in the robot tree's layout (tree/), and two hash lists:
#   MANIFEST.sha256          every file of the package (install_v3.sh checks it first)
#   PRISTINE_v2.0.0.sha256   the files V3 replaces, as release v2.0.0 has them, so the
#                            installer can show what on the robot differs from that
#
#   bash v3/tools/make_robot_package.sh <n> [--model v3/models/dance_<run>]... [out_dir]
# Packs the WORKING TREE (tracked and new files under v3/, ignored ones left out); the
# VERSION file records the commit and whether anything was uncommitted.
set -euo pipefail
N="${1:?package number}"; shift
MODELS=(); OUT="$HOME"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODELS+=("$2"); shift 2 ;;
    *) OUT="$1"; shift ;;
  esac
done
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NAME="r1_v3_${N}_$(date +%F)"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
S="$WORK/$NAME"; T="$S/tree"
mkdir -p "$T/deploy/ros2_ws/src"
cd "$REPO"

copy() {  # copy <repo dir> <dest dir> [exclude regex]: tracked + new files, not ignored
  local src="$1" dst="$2" ex="${3:-^$}"
  git ls-files -co --exclude-standard "$src" | grep -vE "$ex" | while read -r f; do
    mkdir -p "$dst/$(dirname "${f#$src/}")"
    cp -p "$f" "$dst/${f#$src/}"
  done
}
copy v3/mission_ctl "$T/mission_ctl"
copy v3/perception "$T/perception" "^v3/perception/models/"
copy v3/reflex "$T/reflex"
copy v3/ros2_ws/src/r1_hw_bridge "$T/deploy/ros2_ws/src/r1_hw_bridge"
copy v3/ros2_ws/src/r1_policy_runner "$T/deploy/ros2_ws/src/r1_policy_runner"
copy v3/dance "$T/dance"
mkdir -p "$T/v3_tools" && cp -p v3/tools/dance_devbox_camera.py "$T/v3_tools/"   # R3: a scripted "person" for the bridge
cp -p v3/run_tests.sh "$T/v3_tests.sh"
cp -p v3/README.md "$T/V3_README.md"
cp -p v3/tools/install_v3.sh "$S/"
for M in "${MODELS[@]}"; do
  [[ -f "$M/policy.onnx" && -f "$M/SHA256SUMS" && -f "$M/parity_fixture.bin" ]] \
    || { echo "$M: needs policy.onnx, parity_fixture.bin and SHA256SUMS"; exit 1; }
  mkdir -p "$T/v3_models/$(basename "$M")"
  cp -p "$M"/* "$T/v3_models/$(basename "$M")/"
done

dirty="$(git status --porcelain v3 | wc -l)"
{
  echo "V3 robot package $N, $(date '+%F %T')"
  echo "branch $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)$([[ $dirty -gt 0 ]] && echo ", plus $dirty uncommitted path(s) under v3/")"
  echo "base: release v2.0.0 (28fb7c1)"
  echo "models: ${MODELS[*]:-none}"
} > "$S/VERSION"

# what release v2.0.0 has where V3 lays its code, paths relative to the robot tree
P="$WORK/pristine"; mkdir -p "$P"
git archive 28fb7c1 mission_ctl perception deploy/ros2_ws/src/r1_hw_bridge \
  deploy/ros2_ws/src/r1_policy_runner | tar -x -C "$P"
( cd "$P" && find mission_ctl perception deploy -type f ! -path 'perception/models/*' | sort \
    | xargs sha256sum ) > "$S/PRISTINE_v2.0.0.sha256"

( cd "$S" && find . -type f ! -name MANIFEST.sha256 -printf '%P\n' | sort | xargs sha256sum ) > "$S/MANIFEST.sha256"
tar -C "$WORK" --owner=0 --group=0 --numeric-owner -czf "$OUT/$NAME.tar.gz" "$NAME"
echo "$OUT/$NAME.tar.gz  $(stat -c %s "$OUT/$NAME.tar.gz") bytes  md5 $(md5sum < "$OUT/$NAME.tar.gz" | cut -c1-32)"
echo "  $(wc -l < "$S/MANIFEST.sha256") files; $(cat "$S/VERSION" | sed -n 2p)"
