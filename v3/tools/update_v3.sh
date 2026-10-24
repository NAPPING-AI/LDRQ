#!/usr/bin/env bash
# V3 update on the robot, in place: ~/r1-robot-v3 (made by install_v3.sh) gets this
# package's code and, from r1_v3_4 on, the dance policy. Everything it replaces is kept
# in ~/r1-robot-v3/backup_<from>_<date>/ first, so going back is a copy and a rebuild
# (printed at the end).
#
#   bash update_v3.sh [--dst DIR]                 # run from the unpacked package
#   bash update_v3.sh --rehearsal --dst DIR       # dev box: fake robot, no ldd / engines
#
# Steps, each printed as [n/7]:
#   1 preflight   the tree is a V3 tree, the stack is stopped, this terminal has not
#                 sourced any env.sh, the package matches MANIFEST.sha256
#   2 backup      mission_ctl/, perception/ (without models/), reflex/, the bridge's and
#                 the runner's source -> backup_<from>_<date>/ (outside ros2_ws/src: two
#                 packages of one name there and colcon refuses to build)
#   3 overlay     the same from the package; perception/models kept; the dance model
#                 bundle -> v3_models/<name>/
#   4 configs     the diffs of mission.yaml, perception.yaml, bridge.yaml
#   5 build       colcon build: the bridge and the runner (both forks)
#   6 check       ldd: CycloneDDS from /usr/local/lib; V3's tests (v3_tests.sh)
#   7 engines     the dance policy's TensorRT engine built HERE (fp32, TF32 off) and both
#                 engines gated by r1_parity_check against their fixtures
set -uo pipefail
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="$HOME/r1-robot-v3"
REHEARSAL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dst) DST="$2"; shift 2 ;;
    --rehearsal) REHEARSAL=1; shift ;;
    *) echo "unknown argument $1"; exit 2 ;;
  esac
done
O="${R1_V3_LOG_DIR:-$HOME/orin_commissioning}"; mkdir -p "$O"
NAME="$(head -1 "$PKG/VERSION" | awk '{print $1}')"
LOG="$O/v3_update_${NAME}.log"
exec > >(tee "$LOG") 2>&1
step() { echo; echo "[$1/7] $2"; }
die() {
  echo; echo "STOPPED at this step: $*"
  [[ -n "${BK:-}" && -d "${BK:-}" ]] && echo "The replaced code is in $BK; the rollback below puts it back."
  rollback_help
  exit 1
}
rollback_help() {
  [[ -n "${BK:-}" && -d "${BK:-}" ]] || return 0
  echo "Rollback:"
  echo "  cd $DST && for d in mission_ctl perception reflex dance; do [ -d $BK/\$d ] || continue; rm -rf \$d.new && mv \$d \$d.new && cp -a $BK/\$d \$d; done"
  echo "  cp -a perception.new/models perception/ 2>/dev/null; rm -rf mission_ctl.new perception.new reflex.new dance.new"
  echo "  for p in r1_hw_bridge r1_policy_runner; do mv deploy/ros2_ws/src/\$p $BK/\$p.new && cp -a $BK/\$p deploy/ros2_ws/src/\$p; done"
  echo "  (new terminal) source deploy/env.sh && cd deploy/ros2_ws && colcon build --packages-select r1_hw_bridge r1_policy_runner"
}
echo "update_v3.sh $(cat "$PKG/VERSION" | head -2 | tr '\n' ' ') | $(date '+%F %T') | dst $DST"
[[ $REHEARSAL -eq 1 ]] && echo "REHEARSAL: fake robot, no ldd / engines -- not for the robot"

# ------------------------------------------------------------------ 1 preflight
step 1 "preflight"
missing=""
for f in deploy/env.sh v3_tests.sh reflex; do [[ -e "$DST/$f" ]] || missing+=" $f"; done
[[ -z "$missing" ]] || die "$DST is not a V3 tree (missing:$missing). If there is no $DST at all,
  this robot needs a fresh install: install_v3.sh, from a robot package (r1_v3_<n>i)"
if [[ ":${AMENT_PREFIX_PATH:-}:${CMAKE_PREFIX_PATH:-}:" == *"/ros2_ws/install/"* ]]; then
  die "this terminal has sourced a tree's env.sh (AMENT_PREFIX_PATH has a ros2_ws/install). Open a new terminal and run the update there."
fi
if pgrep -x r1_hw_bridge_no >/dev/null || pgrep -x r1_policy_node >/dev/null; then
  die "the stack is running -- stop it (Ctrl-C in its terminal) first"
fi
( cd "$PKG" && sha256sum --quiet -c MANIFEST.sha256 ) || die "the package does not match its MANIFEST.sha256"
echo "  package files match MANIFEST.sha256"
FROM="$(head -1 "$DST/VERSION" 2>/dev/null | awk '{print $1}')"
[[ "$FROM" == r1_v3_* ]] || FROM="v3_installed"     # install_v3.sh's VERSION line
echo "  updating $FROM -> $NAME"
free_kb=$(df -Pk "$DST" | awk 'NR==2{print $4}')
[[ $free_kb -gt 400000 ]] || die "less than 400 MB free"

# ------------------------------------------------------------------ 2 backup
step 2 "backup of what is replaced"
BK="$DST/backup_${FROM}_$(date +%m%d%H%M)"
mkdir -p "$BK" || die "cannot make $BK"
for d in mission_ctl reflex dance; do
  [[ -d "$DST/$d" ]] || continue
  cp -a "$DST/$d" "$BK/$d" || die "backup of $d failed"
done
( cd "$DST" && tar --exclude=./perception/models -cf - perception ) | ( cd "$BK" && tar -xf - ) \
  || die "backup of perception failed"
for p in r1_hw_bridge r1_policy_runner; do
  cp -a "$DST/deploy/ros2_ws/src/$p" "$BK/$p" || die "backup of $p failed"
done
cp "$DST/VERSION" "$BK/" 2>/dev/null
echo "  $BK: $(du -sh "$BK" | cut -f1)"

# ------------------------------------------------------------------ 3 overlay
step 3 "overlay"
T="$PKG/tree"
BR=deploy/ros2_ws/src/r1_hw_bridge
RN=deploy/ros2_ws/src/r1_policy_runner
cmp -s "$DST/$BR/include/r1_hw_bridge/joint_map.hpp" "$T/$BR/include/r1_hw_bridge/joint_map.hpp" \
  || die "joint_map.hpp on the robot differs from the package's: the gains are not v2's"
for d in mission_ctl reflex dance; do rm -rf "$DST/$d" && cp -a "$T/$d" "$DST/$d" || die "overlay of $d failed"; done
mv "$DST/perception/models" "$BK/perception_models_kept" 2>/dev/null
rm -rf "$DST/perception" && cp -a "$T/perception" "$DST/perception" || die "overlay of perception failed"
mv "$BK/perception_models_kept" "$DST/perception/models" 2>/dev/null || echo "  (no perception/models to keep)"
for p in "$BR" "$RN"; do rm -rf "$DST/$p" && cp -a "$T/$p" "$DST/$p" || die "overlay of $p failed"; done
if [[ -d "$T/v3_models" ]]; then
  mkdir -p "$DST/v3_models" && cp -a "$T/v3_models/." "$DST/v3_models/" || die "copy of v3_models failed"
  echo "  v3_models/: $(ls "$T/v3_models" | tr '\n' ' ')"
fi
[[ -d "$T/v3_tools" ]] && mkdir -p "$DST/v3_tools" && cp -a "$T/v3_tools/." "$DST/v3_tools/"
cp "$T/v3_tests.sh" "$T/V3_README.md" "$PKG/VERSION" "$DST/"
echo "  mission_ctl/, perception/ (models kept), reflex/, dance/, $BR/, $RN/"

# ------------------------------------------------------------------ 4 configs
step 4 "configuration diffs ($FROM -> $NAME)"
for pair in "mission_ctl/config/mission.yaml" "perception/config/perception.yaml" \
            "r1_hw_bridge/config/bridge.yaml"; do
  old="$BK/$pair"; new="$DST/${pair/r1_hw_bridge/$BR}"
  echo "  --- $new"
  diff "$old" "$new" | sed 's/^/    /' | head -60
done
echo "  Any '<' line is a value the robot had that the package does not: if it was a local"
echo "  setting, write it into the new file."

# ------------------------------------------------------------------ 5 build
step 5 "colcon build"
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

# ------------------------------------------------------------------ 6 check
step 6 "checks"
if [[ $REHEARSAL -eq 0 ]]; then
  if ldd "$BIN" | grep -E "libddsc" | grep -qv "/usr/local/lib"; then
    die "the bridge links a CycloneDDS outside /usr/local/lib (the DDS version split)"
  fi
  echo "  CycloneDDS from /usr/local/lib: OK"
fi
( cd "$DST" && bash v3_tests.sh ) | sed 's/^/  /'
[[ ${PIPESTATUS[0]} -eq 0 ]] || die "V3's tests failed in the updated tree"

# ------------------------------------------------------------------ 7 engines
step 7 "engines"
if [[ $REHEARSAL -eq 0 ]]; then
  (
    set +u
    source "$DST/deploy/env.sh" >/dev/null
    BINS="$R1_DEPLOY_ROOT/ros2_ws/install/r1_policy_runner/lib/r1_policy_runner"
    A="$R1_DEPLOY_ROOT/artifacts"
    echo "  v2 (installed):"
    "$BINS/r1_parity_check" --plan "$A/policy_fp32.plan" --fixture "$A/parity_fixture.bin" --tol 1e-3 \
      | grep -E "max_abs|PARITY" | sed 's/^/    /' || exit 1
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

echo
echo "updated to $NAME: $DST   (reflex_enable and dance_enable are false until you turn them on)"
echo "  the replaced code: $BK"
rollback_help
echo "  log: $LOG"
