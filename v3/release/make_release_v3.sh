#!/usr/bin/env bash
# Dev box: the V3 robot release, from the COMMITTED tree.
#
#   bash v3/release/make_release_v3.sh --runtime <dir> [--version 3.0.1] [--out <dir>]
#
# <dir> holds llm/ollama/ (Ollama v0.34.4 arm64 + jetpack5, unpacked) and llm/models/ (an
# Ollama model store with qwen3:1.7b), as release/make_release.sh takes it; the v2.0.0
# release's two archives, extracted, are one.
#
# Writes to <out> (default ./dist/v<V>):
#   release_v<V>.tar.gz                 everything but the model, unpacking to r1-robot-v<V>/
#   qwen3-1.7b_r1-robot-v<V>.tar.gz     the model, into r1-robot-v<V>/base/llm/models/
#   SHA256SUMS, RELEASE_NOTES.md
# r1-robot-v<V>/ holds:
#   README.md, README_zh.md, install.sh, selftest.sh, VERSION, licenses, FILES.sha256
#   base/        the v2 robot tree (release v2.0.0's layout: the bridge and policy node as
#                source, policy_pack, mission_ctl, llm/ with the Ollama runtime, the v2 and v1
#                bundles) plus perception/ with YOLOX-Nano and its parity fixture
#   v3_package/  V3 as v3/tools/make_robot_package.sh packs it, with dance_wave
# install.sh installs V3 into r1-robot-v<V>/v3/ the way the robot was installed on
# 2026-10-24: the v2 bundle into base/, the vision engine, then install_v3.sh.
#
# Python: PYTHON (default python3) runs the bundle tools; FIXTURE_PYTHON (default
# deploy/.venv/bin/python) needs numpy, onnxruntime and OpenCV, for the YOLOX fixture.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
V=3.0.1; RUNTIME=""; OUT=""; MODEL=qwen3; TAGM=1.7b
PY="${PYTHON:-python3}"
FPY="${FIXTURE_PYTHON:-$REPO/deploy/.venv/bin/python}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --runtime) RUNTIME="$2"; shift 2 ;;
    --version) V="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "unknown argument: $1"; exit 2 ;;
  esac
done
OUT="${OUT:-$REPO/dist/v$V}"
[[ -x "$RUNTIME/llm/ollama/bin/ollama" ]] || { echo "--runtime: no llm/ollama/bin/ollama under '$RUNTIME'"; exit 2; }
MAN="$RUNTIME/llm/models/manifests/registry.ollama.ai/library/$MODEL/$TAGM"
[[ -f "$MAN" ]] || { echo "--runtime: no $MODEL:$TAGM manifest under $RUNTIME/llm/models"; exit 2; }
cd "$REPO"
git diff --quiet HEAD -- . ':!dist' ':!quickstart/*.tar.gz*' \
  || { echo "uncommitted changes: commit first, the release is built from HEAD"; exit 2; }
[[ -z "$(git status --porcelain v3)" ]] || { echo "untracked files under v3/: the V3 package would take them"; exit 2; }
SHA="$(git rev-parse HEAD)"
NAME="r1-robot-v$V"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
R="$WORK/main/$NAME"; S="$R/base"; mkdir -p "$S" "$OUT"
TAROPT=(--sort=name --owner=0 --group=0 --numeric-owner --mtime="@$(git log -1 --format=%ct)")

echo "== base/: the v2 tree, code from git HEAD ${SHA:0:12}"
git archive HEAD deploy mission_ctl llm policy_pack perception LICENSE NOTICE | tar -x -C "$S"
rm -f "$S/deploy/setup.sh" "$S/deploy/tools/fetch_orin_snapshot.sh" "$S/llm/make_runtime.sh"
rm -rf "$S/deploy/docs"
install -m 755 release/selftest.sh "$S/selftest.sh"
"$PY" policy_pack/make_bundle.py --out "$S/bundles" >/dev/null           # v1, for rollback
"$PY" policy_pack/make_bundle.py --out "$S/bundles" --task-cfg tasks/r1_flat/flat_v2_env_cfg.py \
    --interface models/v2_drift_ft/interface --artifacts models/v2_drift_ft >/dev/null
for b in "$S"/bundles/*/; do "$PY" policy_pack/verify_bundle.py "$b" | tail -1; done

echo "== base/perception/models: YOLOX-Nano and its parity fixture"
bash perception/models/fetch_yolox.sh >/dev/null
PM="$S/perception/models"
cp perception/models/yolox_nano.onnx perception/models/dog.jpg "$PM/"
"$FPY" - "$PM/dog.jpg" "$WORK" <<'PY'
import sys, cv2, numpy as np
img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
np.save(sys.argv[2] + "/s0.npy", img)
np.save(sys.argv[2] + "/s1.npy", np.ascontiguousarray(img[:, ::-1]))
PY
"$FPY" perception/tools/make_fixture.py --onnx "$PM/yolox_nano.onnx" \
    --images "$WORK/s0.npy" "$WORK/s1.npy" --out "$PM/yolox_fixture.bin" | tail -1
( cd "$PM" && sha256sum yolox_fixture.bin > yolox_fixture.sha256 )

echo "== base/llm/ollama: the Ollama runtime"
cp -a "$RUNTIME/llm/ollama" "$S/llm/ollama"
cp release/third_party/OLLAMA_LICENSE "$S/llm/ollama/LICENSE"
mkdir -p "$S/llm/models"          # the model archive fills this
printf 'r1-robot v%s base (v2 tree)\ngit %s\nbuilt %s\npolicy 2026-10-16_17-57-14_v2_drift_ft (rollback: 2026-08-19_11-03-32_week04_nohead)\nmodel %s:%s\n' \
  "$V" "$SHA" "$(date -Iseconds)" "$MODEL" "$TAGM" > "$S/VERSION"
# Left out of base/FILES.sha256: the files install_bundle.sh writes from the bundle (which
# selftest verifies on its own), so that selftest.sh, and install.sh after a failed step 4,
# still pass once the v2 bundle is installed in base/.
WRITTEN='^(deploy/interface/(actuator_gains|policy_interface)\.json|deploy/ros2_ws/src/r1_hw_bridge/config/bridge\.yaml|deploy/ros2_ws/src/r1_hw_bridge/include/r1_hw_bridge/joint_map\.hpp|deploy/ros2_ws/src/r1_policy_runner/config/policy_interface\.yaml)$'
( cd "$S" && find . -type f ! -name FILES.sha256 ! -path './llm/models/*' -printf '%P\n' | sort \
    | grep -vE "$WRITTEN" | xargs -d '\n' sha256sum > FILES.sha256 )
echo "   base: $(wc -l < "$S/FILES.sha256") files (and 5 the bundle install writes)"

echo "== v3_package/"
bash v3/tools/make_robot_package.sh rel --model v3/models/dance_wave "$WORK" | tail -1
tar -xzf "$WORK"/r1_v3_rel_*.tar.gz -C "$WORK"
mv "$WORK"/r1_v3_rel_*/ "$R/v3_package"

echo "== top level"
cp v3/release/README.md v3/release/README_zh.md v3/release/THIRD_PARTY_NOTICES.md LICENSE NOTICE "$R/"
install -m 755 v3/release/install.sh "$R/install.sh"
install -m 755 v3/release/selftest.sh "$R/selftest.sh"
printf 'r1-robot v%s (V3)\ngit %s\nbuilt %s\nwalking policy 2026-10-16_17-57-14_v2_drift_ft (v2); dance policy 2026-10-23_16-51-44_wave\nmodel %s:%s\n' \
  "$V" "$SHA" "$(date -Iseconds)" "$MODEL" "$TAGM" > "$R/VERSION"
( cd "$R" && find . -maxdepth 1 -type f ! -name FILES.sha256 -printf '%P\n' | sort \
    | xargs -d '\n' sha256sum > FILES.sha256 )

echo "== model $MODEL:$TAGM"
M="$WORK/model/$NAME/base/llm/models"
mkdir -p "$M/blobs" "$(dirname "$M/manifests/registry.ollama.ai/library/$MODEL/$TAGM")"
cp "$MAN" "$M/manifests/registry.ollama.ai/library/$MODEL/$TAGM"
"$PY" - "$MAN" <<'PY' | while read -r d; do cp "$RUNTIME/llm/models/blobs/$d" "$M/blobs/$d"; done
import json, sys
m = json.load(open(sys.argv[1]))
for x in [m["config"]] + m["layers"]:
    print(x["digest"].replace(":", "-"))
PY
( cd "$M" && find . -type f ! -name MODEL.sha256 -printf '%P\n' | sort | xargs -d '\n' sha256sum > MODEL.sha256 )

echo "== archives"
MAIN="release_v$V.tar.gz"; MOD="$MODEL-${TAGM}_r1-robot-v$V.tar.gz"
tar -C "$WORK/main" "${TAROPT[@]}" -cf - "$NAME" | gzip -n -6 > "$OUT/$MAIN"
tar -C "$WORK/model" "${TAROPT[@]}" -cf - "$NAME" | gzip -n -6 > "$OUT/$MOD"
sed -e "s/@VERSION@/$V/g" -e "s/@GIT@/${SHA:0:12}/g" v3/release/RELEASE_NOTES.md > "$OUT/RELEASE_NOTES.md"
( cd "$OUT" && sha256sum "$MAIN" "$MOD" > SHA256SUMS && ls -l "$MAIN" "$MOD" && cat SHA256SUMS )
LIMIT=$((2*1024*1024*1024))
for f in "$OUT/$MAIN" "$OUT/$MOD"; do
  [[ $(stat -c %s "$f") -lt $LIMIT ]] || { echo "$f is over GitHub's 2 GiB asset limit"; exit 1; }
done
echo "done: $OUT"
