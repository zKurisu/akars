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

model="${AKARS_MODEL:-$case_dir/model/yolov8n_tennis_v2.cvimodel}"
export LD_LIBRARY_PATH="$case_dir/lib:${LD_LIBRARY_PATH:-}"

exec "$case_dir/akars" "$model" \
  --camera /dev/cvi-usb-camera0 \
  --vpss /dev/cvi-vpss0 \
  --motor /dev/ttyS1 \
  --arm /dev/ttyS2 \
  --classes 1 \
  --conf 0.5 \
  --iou 0.5 \
  --tpu-debug "${AKARS_TPU_DEBUG:-off}" \
  --max-deposits "$ball_count" \
  "$@"
