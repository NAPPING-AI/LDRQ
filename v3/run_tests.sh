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

# The bridge: v3/ros2_ws/... in the repo, deploy/ros2_ws/... in an installed tree
BR=ros2_ws/src/r1_hw_bridge; [[ -d deploy/$BR ]] && BR=deploy/$BR
export BR
echo "python: $("$PY" --version 2>&1)"
run "V3 reflex (state machine, golden)" "$PY" reflex/test_reflex.py
run "V3 bridge reflex (C++, golden)" bash -c '
  set -e; t="$(mktemp -d)"; trap "rm -rf \"\$t\"" EXIT
  g++ -std=c++17 -O2 -Wall -Wextra -Werror -I "$BR/include" "$BR/test/test_reflex_fsm.cpp" -o "$t/t"
  "$t/t" reflex/golden.csv'
# The runner fork, the same way
RN=ros2_ws/src/r1_policy_runner; [[ -d deploy/$RN ]] && RN=deploy/$RN
export RN
run "V3 runner dance obs (C++)" bash -c '
  set -e; t="$(mktemp -d)"; trap "rm -rf \"\$t\"" EXIT
  g++ -std=c++17 -O2 -Wall -Wextra -Werror -I "$RN/include" "$RN/test/test_dance_obs.cpp" -o "$t/t"
  "$t/t" "$RN/test/data/dance_ref.bin" "$RN/test/data/dance_obs_fixture.bin"'
[[ -d dance ]] && run "V3 dance (routine, observation)" "$PY" dance/test_dance.py
run "V3 perception (proximity)"   "$PY" perception/tests/test_perception.py
run "V3 perception (own stereo depth)" "$PY" perception/tests/test_stereo.py
run "V3 mission_ctl core"         "$PY" mission_ctl/tests/test_core.py
run "V3 mission_ctl English (ask)" "$PY" mission_ctl/tests/test_nl.py
run "V3 mission_ctl objects"      "$PY" mission_ctl/tests/test_objects.py
run "V3 mission_ctl reflex"       "$PY" mission_ctl/tests/test_reflex.py
run "V3 eval sets vs V3 config"   "$PY" mission_ctl/eval/run_nl_eval.py --check --set all
run "V3 dry run (person walks up)" bash -c '"$0" mission_ctl/r1_mission_cli.py run "stand 9s" --dry-run \
  --world mission_ctl/tests/worlds/approach.json | grep -o "ABORT: stepped back because of a person"' "$PY"
run "V3 dry run (dancing, person walks up)" bash -c '"$0" mission_ctl/r1_mission_cli.py run "dance 12s" --dry-run \
  --world mission_ctl/tests/worlds/approach.json | grep -o "ABORT: stepped back because of a person"' "$PY"
run "V3 dry run (walk_to, reflex on)" bash -c '"$0" mission_ctl/r1_mission_cli.py run "walk_to the chair stop_at 1.2; turn left 90" \
  --dry-run --world mission_ctl/tests/worlds/room.json | grep -o "\"final_state\": \"DONE\""' "$PY"

exit $failed
