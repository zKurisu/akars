#!/bin/sh
set -eu

case_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ball_count=${1:-3}

case "$ball_count" in
  ''|*[!0-9]*|0)
    echo "usage: $0 POSITIVE_BALL_COUNT [extra akars options ...]" >&2
    exit 2
    ;;
esac
shift

exec "$case_dir/run-robot.sh" --max-deposits "$ball_count" "$@"
