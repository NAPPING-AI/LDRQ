#!/usr/bin/env bash
# V3 on the robot: a NEW tree, ~/r1-robot-v3, made from the one in use (~/r1-robot-v1.1.0,
# with the stage E and v2 packages on it), with V3's code laid over it and the ROS
# workspace rebuilt. The tree in use is only read: going back to v2 is sourcing its
# env.sh again.
#
#   bash install_v3.sh [--src DIR] [--dst DIR]          # run from the unpacked package
#   bash install_v3.sh --rehearsal --src ... --dst ...  # dev box only, see below
#
# Steps, each printed as [n/8]:
#   1 preflight     source tree there, target not, stack stopped, disk space
#   2 compare       the source tree's code against release v2.0.0 (informational)
#   3 copy          the source tree, without builds and backups; the Ollama runtime
#                   and model store (llm/ollama, llm/models: about 2 GB, read only)
#                   become links to the source's
#   4 overlay       mission_ctl/, perception/ (models/ kept), the bridge's and the policy
#                   runner's source (both V3 forks), reflex/, dance/; joint_map.hpp must be
#                   identical or nothing is laid over; the dance model bundle(s) ->
#                   v3_models/<name>/
#   5 configs       the diffs of mission.yaml, perception.yaml, bridge.yaml
#   6 build         colcon build in the new tree (both packages)
#   7 check         ldd: CycloneDDS from /usr/local/lib; V3's tests (v3_tests.sh)
#   8 engines       the v2 engine copied with the tree, and each dance model's engine
#                   (tree/v3_models/, from r1_v3_4 on) built HERE (fp32, TF32 off); all
#                   gated by r1_parity_check against their fixtures
#
# --rehearsal: for a dev box with no Unitree SDK and no TensorRT runtime here. Builds
# the bridge against the fake robot (-DR1_FAKE_ROBOT_IO=ON) and nothing else, and
# skips steps 7's ldd and 8. Never on the robot.
set -uo pipefail
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HOME/r1-robot-v1.1.0"
DST="$HOME/r1-robot-v3"
REHEARSAL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --src) SRC="$2"; shift 2 ;;
    --dst) DST="$2"; shift 2 ;;
    --rehearsal) REHEARSAL=1; shift ;;
    *) echo "unknown argument $1"; exit 2 ;;
  esac
done
O="${R1_V3_LOG_DIR:-$HOME/orin_commissioning}"; mkdir -p "$O"
LOG="$O/v3_install_$(date +%m%d%H%M).log"
exec > >(tee "$LOG") 2>&1
step() { echo; echo "[$1/8] $2"; }
CREATED=0
die() {
  echo; echo "STOPPED at this step: $*"; echo "Nothing in $SRC was changed."
  if [[ $CREATED -eq 1 && -e "$DST" ]]; then
    echo "The partial $DST is left for a look; remove it before trying again: rm -rf $DST"
  fi
  exit 1
}
echo "install_v3.sh $(cat "$PKG/VERSION" 2>/dev/null) | $(date '+%F %T') | src $SRC -> dst $DST"
[[ $REHEARSAL -eq 1 ]] && echo "REHEARSAL: fake robot, bridge only, no ldd / parity -- not for the robot"

# ------------------------------------------------------------------ 1 preflight
step 1 "preflight"
[[ -d "$SRC/deploy" && -f "$SRC/deploy/env.sh" ]] || die "$SRC is not a robot tree (no deploy/env.sh)"
[[ -d "$SRC/perception" ]] || die "$SRC has no perception/ (the stage E packages are not on it)"
[[ -e "$DST" ]] && die "$DST exists. To install again: mv $DST $DST.old_\$(date +%m%d%H%M) first"
# A shell that sourced the old tree's env.sh carries its workspace on every ROS path;
# building and testing the new tree from there mixes the two.
if [[ ":${AMENT_PREFIX_PATH:-}:${CMAKE_PREFIX_PATH:-}:" == *"/ros2_ws/install/"* ]]; then
  die "this terminal has sourced a tree's env.sh (AMENT_PREFIX_PATH has a ros2_ws/install). Open a new terminal and run the installer there."
fi
if pgrep -x r1_hw_bridge_no >/dev/null || pgrep -x r1_policy_node >/dev/null; then
  die "the stack is running -- stop it (Ctrl-C in its terminal) first"
fi
need_kb=$(( $(du -sk --exclude=./llm/models --exclude=./llm/ollama --exclude='./deploy/ros2_ws/build' \
  --exclude='./deploy/ros2_ws/install' --exclude='./deploy/ros2_ws/log' "$SRC" | cut -f1) + 600000 ))
free_kb=$(df -Pk "$(dirname "$DST")" | awk 'NR==2{print $4}')
echo "  need about $((need_kb / 1024)) MB, free $((free_kb / 1024)) MB"
[[ $free_kb -gt $need_kb ]] || die "not enough disk space"
( cd "$PKG" && sha256sum --quiet -c MANIFEST.sha256 ) || die "the package does not match its MANIFEST.sha256"
echo "  package files match MANIFEST.sha256"
# The source tree must come out of this untouched: its fingerprint (every path, size
# and modification time) is compared at the end.
fingerprint() { ( cd "$SRC" && find . -printf '%P %s %T@\n' | sort | md5sum | cut -c1-32 ); }
SRC_FP="$(fingerprint)"

# ------------------------------------------------------------------ 2 compare
step 2 "the source tree's code against release v2.0.0 (informational)"
n_diff=$( cd "$SRC" && sha256sum -c "$PKG/PRISTINE_v2.0.0.sha256" 2>/dev/null | grep -v ': OK$' | tee "$O/v3_src_vs_v2.0.0.txt" | wc -l )
echo "  $n_diff of $(wc -l < "$PKG/PRISTINE_v2.0.0.sha256") files differ from v2.0.0 (list: $O/v3_src_vs_v2.0.0.txt)"
echo "  expected: bridge.yaml (v2's ranges, written by the installer) and little else; V3 replaces"
echo "  these files, so a difference here is only lost if it was a local change worth keeping"
sed 's/^/    /' "$O/v3_src_vs_v2.0.0.txt" | head -20

# ------------------------------------------------------------------ 3 copy
step 3 "copy $SRC -> $DST"
mkdir -p "$DST"; CREATED=1
( cd "$SRC" && tar --exclude=./llm/models --exclude=./llm/ollama --exclude=./deploy/ros2_ws/build \
    --exclude=./deploy/ros2_ws/install --exclude=./deploy/ros2_ws/log \
    --exclude='./*.bak_*' --exclude='./*.failed_*' --exclude='./*/*.bak_*' --exclude='./*/*.failed_*' \
    -cf - . ) | ( cd "$DST" && tar -xf - ) || die "copy failed"
for d in llm/ollama llm/models; do
  if [[ -e "$SRC/$d" ]]; then
    ln -s "$(readlink -f "$SRC/$d")" "$DST/$d"
    echo "  $d -> $(readlink -f "$SRC/$d") (shared, read only)"
  fi
done
echo "  copied: $(du -sh "$DST" | cut -f1)"

# ------------------------------------------------------------------ 4 overlay
step 4 "lay V3 over the copy"
T="$PKG/tree"
BR=deploy/ros2_ws/src/r1_hw_bridge
if ! cmp -s "$DST/$BR/include/r1_hw_bridge/joint_map.hpp" "$T/$BR/include/r1_hw_bridge/joint_map.hpp"; then
  diff "$DST/$BR/include/r1_hw_bridge/joint_map.hpp" "$T/$BR/include/r1_hw_bridge/joint_map.hpp" | head -20
  rm -rf "$DST"
  die "joint_map.hpp on the robot differs from V3's: the installed policy's gains are not v2's. $DST removed."
fi
echo "  joint_map.hpp identical (v2's gains)"
mv "$DST/mission_ctl" "$DST/mission_ctl.v2" && cp -a "$T/mission_ctl" "$DST/mission_ctl" \
  || die "overlay of mission_ctl failed"
# The old bridge source goes OUT of ros2_ws/src: two packages of one name there and
# colcon refuses to build.
mv "$DST/$BR" "$DST/deploy/r1_hw_bridge.v2" && cp -a "$T/$BR" "$DST/$BR" \
  || die "overlay of the bridge failed"
RN=deploy/ros2_ws/src/r1_policy_runner
mv "$DST/$RN" "$DST/deploy/r1_policy_runner.v2" && cp -a "$T/$RN" "$DST/$RN" \
  || die "overlay of the policy runner failed"
mv "$DST/perception" "$DST/perception.v2"
cp -a "$T/perception" "$DST/perception"
cp -a "$DST/perception.v2/models" "$DST/perception/models" 2>/dev/null || echo "  (no perception/models to keep)"
cp -a "$T/reflex" "$DST/reflex"
cp -a "$T/dance" "$DST/dance"
if [[ -d "$T/v3_models" ]]; then
  cp -a "$T/v3_models" "$DST/v3_models" || die "copy of v3_models failed"
  echo "  v3_models/: $(ls "$T/v3_models" | tr '\n' ' ')"
fi
[[ -d "$T/v3_tools" ]] && cp -a "$T/v3_tools" "$DST/v3_tools"
cp "$T/v3_tests.sh" "$T/V3_README.md" "$PKG/VERSION" "$DST/"
echo "  mission_ctl/, perception/ (models kept), $BR/, $RN/, reflex/, dance/; the replaced ones"
echo "  are kept as mission_ctl.v2/, perception.v2/, deploy/r1_hw_bridge.v2/, deploy/r1_policy_runner.v2/"

# ------------------------------------------------------------------ 5 configs
step 5 "configuration diffs (v2 -> V3)"
for pair in "mission_ctl.v2/config/mission.yaml mission_ctl/config/mission.yaml" \
            "perception.v2/config/perception.yaml perception/config/perception.yaml" \
            "deploy/r1_hw_bridge.v2/config/bridge.yaml $BR/config/bridge.yaml"; do
  set -- $pair
  echo "  --- $2"
  diff "$DST/$1" "$DST/$2" | sed 's/^/    /' | head -60
done
echo "  expected: mission.yaml gains reflex: on and obj_stop_min_m 0.9; perception.yaml gains"
echo "  proximity_center_region; bridge.yaml gains the reflex_* block (reflex_enable: false)."
echo "  Any '<' line that is not one of those was a local value: write it into the new file."

# ------------------------------------------------------------------ 6 build
step 6 "colcon build in $DST"
(
  set +u
  source "$DST/deploy/env.sh" >/dev/null
  cd "$DST/deploy/ros2_ws"
  if [[ $REHEARSAL -eq 1 ]]; then
    colcon build --packages-select r1_hw_bridge --cmake-args -DR1_FAKE_ROBOT_IO=ON
  else
    colcon build --packages-select r1_hw_bridge r1_policy_runner
  fi
) 2>&1 | tail -6
[[ ${PIPESTATUS[0]} -eq 0 ]] || die "colcon build failed (the full output is in $LOG)"
BIN="$DST/deploy/ros2_ws/install/r1_hw_bridge/lib/r1_hw_bridge/r1_hw_bridge_node"
[[ -x "$BIN" ]] || die "no bridge binary after the build"

# ------------------------------------------------------------------ 7 check
step 7 "checks"
if [[ $REHEARSAL -eq 0 ]]; then
  ldd "$BIN" | grep -E "ddsc" | sed 's/^/  /'
  if ldd "$BIN" | grep -E "libddsc" | grep -qv "/usr/local/lib"; then
    die "the bridge links a CycloneDDS outside /usr/local/lib (the DDS version split)"
  fi
  echo "  CycloneDDS from /usr/local/lib: OK"
fi
( cd "$DST" && bash v3_tests.sh ) | sed 's/^/  /'
[[ ${PIPESTATUS[0]} -eq 0 ]] || die "V3's tests failed in the new tree"

# ------------------------------------------------------------------ 8 engine
step 8 "engines"
if [[ $REHEARSAL -eq 0 ]]; then
  (
    set +u
    source "$DST/deploy/env.sh" >/dev/null
    A="$R1_DEPLOY_ROOT/artifacts"
    BINS="$R1_DEPLOY_ROOT/ros2_ws/install/r1_policy_runner/lib/r1_policy_runner"
    python3 -c "import json;d=json.load(open('$A/installed.json'));print('  installed policy:',d.get('run_id'))"
    "$BINS/r1_parity_check" --plan "$A/policy_fp32.plan" --fixture "$A/parity_fixture.bin" --tol 1e-3 \
      | grep -E "max_abs|PARITY" | sed 's/^/    /'
    [[ ${PIPESTATUS[0]} -eq 0 ]] || exit 1
    for M in "$DST"/v3_models/*/; do
      [[ -f "$M/policy.onnx" ]] || continue
      echo "  $(basename "$M"):"
      ( cd "$M" && sha256sum --quiet -c SHA256SUMS ) || { echo "    its SHA256SUMS do not match"; exit 1; }
      "$BINS/r1_build_engine" --onnx "$M/policy.onnx" --plan "$M/policy_fp32.plan" | tail -1 | sed 's/^/    /'
      "$BINS/r1_parity_check" --plan "$M/policy_fp32.plan" --fixture "$M/parity_fixture.bin" --tol 1e-3 \
        | grep -E "max_abs|PARITY" | sed 's/^/    /'
      [[ ${PIPESTATUS[0]} -eq 0 ]] || exit 1
    done
  ) || die "an engine failed its parity check"
fi

[[ "$(fingerprint)" == "$SRC_FP" ]] || die "the source tree CHANGED during the install -- bring $LOG back"
echo
echo "source tree unchanged: OK (fingerprint $SRC_FP)"
echo "V3 installed: $DST   (reflex_enable and dance_enable are false until you turn them on)"
echo "  source $DST/deploy/env.sh   in every terminal from now on"
echo "  going back to v2: source $SRC/deploy/env.sh  (nothing there was changed)"
echo "  log: $LOG"
