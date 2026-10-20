#!/usr/bin/env bash
# Every check that needs no GPU, no ROS and no robot: Python 3.8+ standard
# library and coreutils, plus numpy for the perception suite. The same script runs in CI
# (.github/workflows/tests.yml), so the test counts it prints are the only
# ones to trust -- the documentation deliberately does not copy them.
#
#   ./run_tests.sh        # summary only; exits non-zero if anything fails
#   ./run_tests.sh -v     # also print each suite's full output
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY="${PYTHON:-python3}"
VERBOSE=0; [[ "${1:-}" == "-v" ]] && VERBOSE=1
failed=0

run() {  # run <name> <command...>; prints the command's last line
  local name="$1"; shift
  local out rc
  out="$("$@" 2>&1)"; rc=$?
  [[ $VERBOSE -eq 1 ]] && printf '%s\n' "$out"
  printf '%-34s %-6s %s\n' "$name" "$([[ $rc -eq 0 ]] && echo ok || echo FAIL)" "$(printf '%s\n' "$out" | tail -1)"
  if [[ $rc -ne 0 ]]; then
    failed=1
    [[ $VERBOSE -eq 0 ]] && printf '%s\n' "$out" | tail -20 | sed 's/^/    | /'
  fi
}

echo "python: $("$PY" --version 2>&1)"
run "mission_ctl core"            "$PY" mission_ctl/tests/test_core.py
run "mission_ctl English (ask)"   "$PY" mission_ctl/tests/test_nl.py
run "mission_ctl objects (stage E)" "$PY" mission_ctl/tests/test_objects.py
run "perception (numpy)"          "$PY" perception/tests/test_perception.py
run "eval sets vs shipped config" "$PY" mission_ctl/eval/run_nl_eval.py --check --set all
run "llm coexist selftest"        "$PY" llm/coexist.py --selftest
run "policy_pack refusals"        env PATH="$(dirname "$(command -v "$PY")"):$PATH" bash policy_pack/tests/run_tests.sh
run "policy_pack negative installs" bash -c '
  set -e; t="$(mktemp -d)"; trap "rm -rf \"\$t\"" EXIT
  tar -C . --exclude=deploy/.venv --exclude=deploy/third_party --exclude="deploy/ros2_ws/build" \
      --exclude="deploy/ros2_ws/install" --exclude="deploy/ros2_ws/log" -cf - deploy | tar -C "$t" -xf -
  python3 policy_pack/make_bundle.py --deploy "$t/deploy" --out "$t/b" >/dev/null
  R1_DEPLOY_ROOT="$t/deploy" bash policy_pack/tests/negative_installs.sh "$(ls -d "$t"/b/*)"'
run "deployed policy SHA256"      bash -c 'cd models/week04_nohead && sha256sum --quiet -c SHA256SUMS && echo "$(wc -l < SHA256SUMS) files match SHA256SUMS"'
run "v2 policy SHA256"            bash -c 'cd models/v2_drift_ft && sha256sum --quiet -c SHA256SUMS && echo "$(wc -l < SHA256SUMS) files match SHA256SUMS"'
run "v2 bundle builds and verifies" bash -c '
  set -e; t="$(mktemp -d)"; trap "rm -rf \"\$t\"" EXIT
  "$0" policy_pack/make_bundle.py --out "$t" --task-cfg tasks/r1_flat/flat_v2_env_cfg.py \
      --interface models/v2_drift_ft/interface --artifacts models/v2_drift_ft | tail -1
  grep -q "\-0.6" "$t"/*/command_envelope.json' "$PY"
run "Markdown links"              "$PY" scripts/check_doc_links.py
run "mission dry run"             bash -c '"$0" mission_ctl/r1_mission_cli.py run "walk 3s@0.3; turn left 90" --dry-run | grep -o "\"final_state\": \"DONE\""' "$PY"
run "v2 dry run (backward, sideways)" bash -c 't="$(mktemp -d)"; trap "rm -rf \"$t\"" EXIT
  printf "{\"vx\": [-0.6, 1.0], \"vy\": [-0.5, 0.5], \"wz\": [-1.57, 1.57]}" > "$t/env.json"
  "$0" mission_ctl/r1_mission_cli.py run "walk backward 1m; walk left 2s; turn right 180@1.0" --dry-run --envelope "$t/env.json" | grep -o "\"final_state\": \"DONE\""' "$PY"
run "walk_to dry run (sim room)"  bash -c '"$0" mission_ctl/r1_mission_cli.py run "walk_to the chair stop_at 1.2; turn left 90" --dry-run --world mission_ctl/tests/worlds/room.json | grep -o "\"final_state\": \"DONE\""' "$PY"

exit $failed
