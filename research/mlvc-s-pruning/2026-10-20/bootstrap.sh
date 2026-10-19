#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p logs
printf '%s\n' "$$" > logs/bootstrap.pid
trap 'task_exit=$?; printf "%s\n" "$task_exit" > logs/bootstrap.exitcode' EXIT
python prepare_source.py
python bootstrap_remote.py --output logs/preflight-before.json
python -c 'import torch; assert torch.__version__.split("+")[0] == "2.8.0"; assert torch.version.cuda == "12.8"; assert torch.cuda.is_available(); print("Pinned image runtime:", torch.__version__, torch.version.cuda)'
python -m pip install --disable-pip-version-check -r requirements-runtime.txt
python -m pip install --disable-pip-version-check ./vendor/mlvc/packages/msrtc_rans
bash build_vmaf.sh
python -m pip freeze > logs/environment-installed.txt
python verify_pruning.py --source vendor/mlvc --checkpoint mlvc-s-psnr-v1.ckpt --output logs/pruning-verification.json
python verify_quality_gate.py
python bootstrap_remote.py --output logs/preflight-after.json
