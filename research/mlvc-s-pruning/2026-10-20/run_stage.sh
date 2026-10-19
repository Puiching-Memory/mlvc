#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "$0")"
stage="$1"
shift
[[ "$stage" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
mkdir -p logs
printf '%s\n' "$$" > "logs/$stage.pid"
trap 'stage_exit=$?; printf "%s\n" "$stage_exit" > "logs/$stage.exitcode"' EXIT
printf '%q ' "$@" > "logs/$stage.command"
printf '\n' >> "logs/$stage.command"
"$@"
