#!/usr/bin/env bash
# Dev box: the v1 quick-start archive, from the COMMITTED tree.
#
#   bash quickstart/make_quickstart.sh [--out <dir>]        # default: quickstart/
#
# Writes <out>/ldrq-v1-quickstart.tar.gz (and its .sha256). It unpacks to ldrq-v1/: what
# release v1.1.0 puts on the robot -- the bridge and policy node (source, built on the
# robot), policy_pack, mission_ctl, llm/ollama_ctl.sh and the v1 policy bundle -- with
# ROBOT_GUIDE.md / ROBOT_GUIDE_zh.md and selftest.sh.
# Not in it: the language model's runtime and weights (about 1.8 GB, over GitHub's 100 MB
# file limit). ROBOT_GUIDE.md says where they come from; without them `run` works and
# `ask` does not.
#
# PYTHON (default python3) runs the bundle tools.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO/quickstart"
PY="${PYTHON:-python3}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    *) echo "unknown argument: $1"; exit 2 ;;
  esac
done
cd "$REPO"
git diff --quiet HEAD -- . ':!quickstart/*.tar.gz*' \
  || { echo "uncommitted changes: commit first, the archive is built from HEAD"; exit 2; }
NAME=ldrq-v1
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
S="$WORK/$NAME"; mkdir -p "$S" "$OUT"
TAROPT=(--sort=name --owner=0 --group=0 --numeric-owner --mtime="@$(git log -1 --format=%ct)")

echo "== code from git HEAD $(git rev-parse --short=12 HEAD)"
git archive HEAD deploy mission_ctl llm policy_pack LICENSE NOTICE | tar -x -C "$S"
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

echo "== policy bundle"
"$PY" policy_pack/make_bundle.py --out "$S/bundles" >/dev/null
"$PY" policy_pack/verify_bundle.py "$S"/bundles/*/ | tail -1

printf 'ldrq v1 quick start\ngit %s\nbuilt %s\npolicy 2026-08-19_11-03-32_week04_nohead\nmodel qwen3:1.7b (not included)\n' \
  "$(git rev-parse HEAD)" "$(date -Iseconds)" > "$S/VERSION"
( cd "$S" && find . -type f ! -name FILES.sha256 -printf '%P\n' | sort \
    | xargs -d '\n' sha256sum > FILES.sha256 )
echo "   $(wc -l < "$S/FILES.sha256") files"

echo "== archive"
A="$OUT/$NAME-quickstart.tar.gz"
tar -C "$WORK" "${TAROPT[@]}" -cf - "$NAME" | gzip -n -9 > "$A"
( cd "$OUT" && sha256sum "$(basename "$A")" > "$(basename "$A").sha256" )
echo "$A  $(du -h "$A" | cut -f1)  sha256 $(cut -c1-16 < "$A.sha256")"
