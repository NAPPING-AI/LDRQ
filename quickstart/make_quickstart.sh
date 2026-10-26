#!/usr/bin/env bash
# Dev box: the v2 quick-start archive, from the COMMITTED tree.
#
#   bash quickstart/make_quickstart.sh [--out <dir>]        # default: quickstart/
#
# Writes <out>/ldrq-v2-quickstart.tar.gz (and its .sha256). It unpacks to ldrq-v2/:
#   - what release v2.0.0 puts on the robot: the bridge and policy node (source, built on
#     the robot), policy_pack, mission_ctl, llm/ollama_ctl.sh, and two policy bundles
#     (v2, and v1 for rollback);
#   - the stage E vision node, perception/, with YOLOX-Nano and a parity fixture
#     (perception stays off in mission.yaml, as in the release);
#   - ROBOT_GUIDE.md / ROBOT_GUIDE_zh.md and selftest.sh.
# Not in it: the language model's runtime and weights (about 1.8 GB, over GitHub's 100 MB
# file limit). ROBOT_GUIDE.md says where they come from; without them `run` works and
# `ask` does not.
#
# Python: PYTHON (default python3) runs the bundle tools; FIXTURE_PYTHON (default
# deploy/.venv/bin/python) needs numpy, onnxruntime and OpenCV, for the YOLOX fixture.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO/quickstart"
PY="${PYTHON:-python3}"
FPY="${FIXTURE_PYTHON:-$REPO/deploy/.venv/bin/python}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    *) echo "unknown argument: $1"; exit 2 ;;
  esac
done
cd "$REPO"
git diff --quiet HEAD -- . ':!quickstart/*.tar.gz*' \
  || { echo "uncommitted changes: commit first, the archive is built from HEAD"; exit 2; }
NAME=ldrq-v2
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
S="$WORK/$NAME"; mkdir -p "$S" "$OUT"
TAROPT=(--sort=name --owner=0 --group=0 --numeric-owner --mtime="@$(git log -1 --format=%ct)")

echo "== code from git HEAD $(git rev-parse --short=12 HEAD)"
git archive HEAD deploy mission_ctl llm policy_pack perception LICENSE NOTICE | tar -x -C "$S"
# dev-box-only or superseded files (as release/make_release.sh)
rm -f "$S/deploy/setup.sh" "$S/deploy/tools/fetch_orin_snapshot.sh" "$S/llm/make_runtime.sh"
rm -rf "$S/deploy/docs"
cp release/THIRD_PARTY_NOTICES.md quickstart/ROBOT_GUIDE.md quickstart/ROBOT_GUIDE_zh.md "$S/"
# The release's selftest, except that a missing language model is a warning here:
# the guide adds it as a separate step.
sed -e 's/^  bad "no model: /  warn "no model yet (ROBOT_GUIDE.md step 3; run works without it): /' \
    -e 's/|| bad "llm\/ollama\/bin\/ollama missing or not executable (copied as zip?)"/|| warn "no llm\/ollama runtime yet (ROBOT_GUIDE.md step 3)"/' \
    release/selftest.sh > "$S/selftest.sh"
chmod 755 "$S/selftest.sh"
grep -q 'warn "no model yet' "$S/selftest.sh" && grep -q 'warn "no llm/ollama runtime yet' "$S/selftest.sh" \
  || { echo "could not adapt selftest.sh"; exit 1; }
# turn_to / walk_to have not run on the robot yet: off, as in release v2.0.0
sed -i 's/^perception: on /perception: off/' "$S/mission_ctl/config/mission.yaml"
grep -q '^perception: off' "$S/mission_ctl/config/mission.yaml" || { echo "could not switch perception off"; exit 1; }

echo "== policy bundles"
"$PY" policy_pack/make_bundle.py --out "$S/bundles" >/dev/null           # v1, for rollback
"$PY" policy_pack/make_bundle.py --out "$S/bundles" --task-cfg tasks/r1_flat/flat_v2_env_cfg.py \
    --interface models/v2_drift_ft/interface --artifacts models/v2_drift_ft >/dev/null
for b in "$S"/bundles/*/; do "$PY" policy_pack/verify_bundle.py "$b" | tail -1; done

echo "== YOLOX-Nano and its parity fixture"
bash perception/models/fetch_yolox.sh >/dev/null           # yolox_nano.onnx + dog.jpg, sha256-checked
PM="$S/perception/models"
cp perception/models/yolox_nano.onnx perception/models/dog.jpg "$PM/"
# The fixture as on the robot in stage E: YOLOX's sample image, and the same mirrored.
"$FPY" - "$PM/dog.jpg" "$WORK" <<'PY'
import sys, cv2, numpy as np
img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
np.save(sys.argv[2] + "/s0.npy", img)
np.save(sys.argv[2] + "/s1.npy", np.ascontiguousarray(img[:, ::-1]))
PY
"$FPY" perception/tools/make_fixture.py --onnx "$PM/yolox_nano.onnx" \
    --images "$WORK/s0.npy" "$WORK/s1.npy" --out "$PM/yolox_fixture.bin" | tail -1
( cd "$PM" && sha256sum yolox_fixture.bin > yolox_fixture.sha256 )

printf 'ldrq v2 quick start\ngit %s\nbuilt %s\npolicy 2026-10-16_17-57-14_v2_drift_ft (rollback: 2026-08-19_11-03-32_week04_nohead)\nmodel qwen3:1.7b (not included)\n' \
  "$(git rev-parse HEAD)" "$(date -Iseconds)" > "$S/VERSION"
( cd "$S" && find . -type f ! -name FILES.sha256 -printf '%P\n' | sort \
    | xargs -d '\n' sha256sum > FILES.sha256 )
echo "   $(wc -l < "$S/FILES.sha256") files"

echo "== archive"
A="$OUT/$NAME-quickstart.tar.gz"
tar -C "$WORK" "${TAROPT[@]}" -cf - "$NAME" | gzip -n -9 > "$A"
( cd "$OUT" && sha256sum "$(basename "$A")" > "$(basename "$A").sha256" )
echo "$A  $(du -h "$A" | cut -f1)  sha256 $(cut -c1-16 < "$A.sha256")"
