#!/usr/bin/env bash
# V3's checks that need no GPU, no ROS and no robot: the forks (perception, mission_ctl)
# and the reflex. Same conventions as ../run_tests.sh, which covers v1 and v2 and does
# not look in here.
#
#   v3/run_tests.sh        # summary only; exits non-zero if anything fails
#   v3/run_tests.sh -v     # also print each suite's full output
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
run "V3 reflex (state machine, golden)" "$PY" reflex/test_reflex.py
run "V3 perception (proximity)"   "$PY" perception/tests/test_perception.py
run "V3 mission_ctl core"         "$PY" mission_ctl/tests/test_core.py
run "V3 mission_ctl English (ask)" "$PY" mission_ctl/tests/test_nl.py
run "V3 mission_ctl objects"      "$PY" mission_ctl/tests/test_objects.py
run "V3 eval sets vs V3 config"   "$PY" mission_ctl/eval/run_nl_eval.py --check --set all

exit $failed
