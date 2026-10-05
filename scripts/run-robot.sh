#!/bin/sh
set -eu

case_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
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
  "$@"
