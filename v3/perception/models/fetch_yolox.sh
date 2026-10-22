#!/usr/bin/env bash
# YOLOX-Nano (Megvii, Apache-2.0, release 0.1.1rc0) and YOLOX's own sample image, pinned
# by sha256. Not in git: they are downloaded, checked, and left here.
#
#   bash perception/models/fetch_yolox.sh
#
# The robot has no internet: run this on the dev box and copy models/ with the code.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
BASE=https://github.com/Megvii-BaseDetection/YOLOX
get() {  # get <file> <url>
  if [[ -f "$1" ]] && grep " $1\$" SHA256SUMS | sha256sum --quiet -c - 2>/dev/null; then
    echo "have $1"; return
  fi
  curl -fsSL -o "$1.part" "$2" && mv "$1.part" "$1"
  echo "got  $1"
}
get yolox_nano.onnx "$BASE/releases/download/0.1.1rc0/yolox_nano.onnx"
get dog.jpg "https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/main/assets/dog.jpg"
sha256sum -c SHA256SUMS
