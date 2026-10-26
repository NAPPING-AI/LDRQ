#!/usr/bin/env bash
# Install V3 from the release directory, on the robot (Unitree R1 EDU, Orin NX, JetPack 5.1.1).
#
#   bash install.sh [--dst DIR]          # default DIR: v3/ next to this script
#
# The same three tools that installed V3 on the robot on 2026-10-24, in order:
#   [1/5] checks       a clean terminal, no stack running, selftest.sh
#   [2/5] v2 policy    base/policy_pack/install_bundle.sh with the v2 bundle: builds the
#                      bridge and the policy node in base/, the engine on THIS robot, parity
#   [3/5] vision       base/perception/models/yolox_fp16.plan (r1_build_engine --fp16) and
#                      engine-check against the bundled fixture
#   [4/5] V3           v3_package/install_v3.sh --src base --dst DIR: copies base, lays V3
#                      over it, builds, runs V3's tests, builds the dance engine, parity
#   [5/5] summary      what to note down, how to start
# base/ ends up as an installed v2 tree and is never changed by step 4: going back to v2 is
# sourcing base/deploy/env.sh. Steps 2 and 3 are skipped when already done; step 4 refuses
# an existing DIR. Everything is logged to ~/orin_commissioning/v3.0.0_install_<time>.log.
set -uo pipefail
REL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="$REL/v3"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dst) DST="$2"; shift 2 ;;
    -h|--help) sed -n 2,17p "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument $1"; exit 2 ;;
  esac
done
BASE="$REL/base"
V2_RUN=2026-10-16_17-57-14_v2_drift_ft
O="${R1_V3_LOG_DIR:-$HOME/orin_commissioning}"; mkdir -p "$O"
LOG="$O/v3.0.0_install_$(date +%m%d%H%M).log"
exec > >(tee "$LOG") 2>&1
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-99}" ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
step() { echo; echo "[$1/5] $2"; }
die() { echo; echo "STOPPED: $*"; echo "log: $LOG"; exit 1; }
in_base() {  # in_base <command...>: run with base/'s environment, in a subshell
  ( set +u; source "$BASE/deploy/env.sh" >/dev/null; set -u; "$@" )
}
echo "install.sh $(head -1 "$REL/VERSION" 2>/dev/null) | $(date '+%F %T') | $REL -> $DST"

# ------------------------------------------------------------------ 1 checks
step 1 "checks"
if [[ ":${AMENT_PREFIX_PATH:-}:${CMAKE_PREFIX_PATH:-}:" == *"/ros2_ws/install/"* ]]; then
  die "this terminal has sourced a tree's env.sh. Open a new terminal and run install.sh there."
fi
if pgrep -x r1_hw_bridge_no >/dev/null || pgrep -x r1_policy_node >/dev/null; then
  die "a stack is running: stop it (Ctrl-C in its terminal) first"
fi
[[ -e "$DST" ]] && die "$DST exists. To install again: mv $DST $DST.old_\$(date +%m%d%H%M) first"
bash "$REL/selftest.sh" || die "selftest.sh failed (above)"

# ------------------------------------------------------------------ 2 v2 policy
step 2 "the v2 policy in base/ (engine built and checked on this robot)"
inst="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('run_id',''))" \
        "$BASE/deploy/artifacts/installed.json" 2>/dev/null)"
if [[ "$inst" == "$V2_RUN" && -f "$BASE/deploy/artifacts/policy_fp32.plan" ]]; then
  echo "  already installed: $inst (base/deploy/artifacts/installed.json); skipping"
else
  in_base bash "$BASE/policy_pack/install_bundle.sh" "$BASE/bundles/$V2_RUN" \
    || die "install_bundle.sh failed: nothing after it was done"
fi

# ------------------------------------------------------------------ 3 vision
step 3 "the vision engine (YOLOX-Nano, TensorRT FP16; about 7 minutes the first time)"
PM="$BASE/perception/models"
( cd "$PM" && sha256sum --quiet -c SHA256SUMS && sha256sum --quiet -c yolox_fixture.sha256 ) \
  || die "perception/models does not match its SHA256SUMS"
if [[ -f "$PM/yolox_fp16.plan" ]]; then
  echo "  yolox_fp16.plan exists; not rebuilt"
else
  in_base bash -c '"$R1_DEPLOY_ROOT/ros2_ws/install/r1_policy_runner/lib/r1_policy_runner/r1_build_engine" \
      --onnx "$0/yolox_nano.onnx" --plan "$0/yolox_fp16.plan" --fp16 | tail -2' "$PM" \
    || { rm -f "$PM/yolox_fp16.plan"; die "the YOLOX engine did not build"; }
fi
in_base python3 "$BASE/perception/r1_perception_cli.py" engine-check \
    --engine "$PM/yolox_fp16.plan" --gate detections | tail -1 | tee "$O/.v3_engine_check"
grep -q "PARITY PASS" "$O/.v3_engine_check" || { rm -f "$PM/yolox_fp16.plan"; die "the YOLOX engine failed engine-check: removed"; }
rm -f "$O/.v3_engine_check"

# ------------------------------------------------------------------ 4 V3
step 4 "V3: install_v3.sh --src base --dst $DST"
bash "$REL/v3_package/install_v3.sh" --src "$BASE" --dst "$DST" || die "install_v3.sh stopped (its own message above)"

# ------------------------------------------------------------------ 5 summary
step 5 "done"
D="$DST/v3_models/dance_wave"
echo "  V3 installed in   $DST"
echo "  v2 (unchanged) in $BASE"
grep -hoE "fnv1a=0x[0-9a-f]+" "$LOG" | tail -1 | sed 's/^/  dance engine      /;s/$/   <- note it: the stack prints it at every start/'
echo
echo "  Every terminal, from now on:"
echo "    export ROS_DOMAIN_ID=99 ROS_LOCALHOST_ONLY=1"
echo "    source $DST/deploy/env.sh"
echo "  Then README.md, 'Every session'. reflex_enable and dance_enable are off until you"
echo "  pass them to the launch; the first runs are with enable_output:=false."
echo "  log: $LOG"
