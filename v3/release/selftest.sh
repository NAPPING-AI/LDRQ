#!/usr/bin/env bash
# Checks for a freshly extracted V3 release, on the robot or a dev box. Moves nothing,
# starts nothing, needs no ROS stack:   bash selftest.sh
#   - every file of the release against FILES.sha256;
#   - base/selftest.sh: the v2 tree, the language model, its runtime, the robot system;
#   - v3_package/: its files against MANIFEST.sha256, and V3's own tests (v3_tests.sh:
#     the reflex and the conductor against their golden files, C++ included, the dance
#     observation, perception, mission_ctl and three dry runs).
# Lines marked "warn" are about the robot's system and do not fail the test.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
fail=0
ok()  { printf '  ok    %s\n' "$*"; }
bad() { printf '  FAIL  %s\n' "$*"; fail=$((fail+1)); }

echo "$(head -1 VERSION 2>/dev/null) ($(sed -n 2p VERSION 2>/dev/null | cut -c1-16)) in $PWD"
echo "-- release files"
if sha256sum --quiet -c FILES.sha256 2>/dev/null; then ok "release files match FILES.sha256: $(wc -l < FILES.sha256) files"
else bad "release files differ from FILES.sha256 (copied as zip, or edited?)"; fi

echo "== base/ (the v2 tree)"
bash base/selftest.sh | sed -n '2,$p' | grep -v '^selftest:'
[[ ${PIPESTATUS[0]} -eq 0 ]] || fail=$((fail+1))

echo "== v3_package/"
if ( cd v3_package && sha256sum --quiet -c MANIFEST.sha256 ); then ok "package files match MANIFEST.sha256: $(wc -l < v3_package/MANIFEST.sha256) files"
else bad "v3_package differs from its MANIFEST.sha256"; fi
command -v g++ >/dev/null || bad "no g++: V3's C++ checks (and the build) need it"
out="$(cd v3_package/tree && bash v3_tests.sh 2>&1)"; rc=$?
printf '%s\n' "$out" | sed -n '2,$p' | sed 's/^/  /'
[[ $rc -eq 0 ]] || bad "V3's tests (above)"

echo
[[ $fail -eq 0 ]] && echo "selftest: OK" || echo "selftest: $fail FAILED"
exit $fail
