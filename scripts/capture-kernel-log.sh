#!/bin/sh
set -eu

case_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
helper="$case_dir/kernel-log-capture"

if [ ! -x "$helper" ]; then
    echo "missing executable kernel log helper: $helper" >&2
    exit 1
fi

exec "$helper" \
    "${1:-/root/kernel-live-v2.log}" \
    "${2:-0.2}" \
    "${3:-2.0}"
