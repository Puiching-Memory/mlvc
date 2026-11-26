#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p vendor logs
python -m pip install --disable-pip-version-check meson==1.7.0 ninja==1.11.1.3
if ! command -v nasm >/dev/null || ! command -v xxd >/dev/null; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nasm xxd build-essential pkg-config
fi
if [ ! -d vendor/vmaf/.git ]; then
  timeout 240 git clone --depth 1 --branch v3.0.0 https://github.com/Netflix/vmaf.git vendor/vmaf
fi
git -C vendor/vmaf rev-parse HEAD > logs/vmaf-source-revision.txt
git -C vendor/vmaf diff --exit-code
if [ ! -f vendor/vmaf/libvmaf/build/build.ninja ]; then
  meson setup vendor/vmaf/libvmaf/build vendor/vmaf/libvmaf --buildtype release -Ddefault_library=static -Denable_tests=false -Denable_docs=false
fi
ninja -C vendor/vmaf/libvmaf/build -j 8
vendor/vmaf/libvmaf/build/tools/vmaf --version
sha256sum vendor/vmaf/libvmaf/build/tools/vmaf vendor/vmaf/model/vmaf_v0.6.1.json > logs/vmaf-digests.txt
