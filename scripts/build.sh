#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/env.sh
source "$ROOT/scripts/env.sh"

if ! akars_toolchain_ready; then
  echo "error: Xuantie toolchain is missing: $AKARS_TOOLCHAIN_DIR" >&2
  echo "       run scripts/setup.sh or set AKARS_TOOLCHAIN_DIR" >&2
  exit 1
fi

if ! akars_tpu_sdk_ready; then
  echo "error: TPU SDK submodule is missing or incomplete: $AKARS_TPU_SDK_DIR" >&2
  echo "       run scripts/setup.sh or set AKARS_TPU_SDK_DIR" >&2
  exit 1
fi

export AKARS_TOOLCHAIN_DIR
export AKARS_TPU_SDK_DIR
export AKARS_OPENCV_DIR
export PATH="$AKARS_TOOLCHAIN_DIR/bin:$PATH"
export CC="${CC:-$AKARS_CC}"
export CXX="${CXX:-$AKARS_CXX}"
export AR="${AR:-$AKARS_AR}"

cd "$AKARS_ROOT"
exec cargo build --release --target "$AKARS_TARGET" "$@"
