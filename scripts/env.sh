#!/usr/bin/env bash
# Shared build settings for the SG2002 target.

AKARS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
AKARS_TARGET="riscv64gc-unknown-linux-musl"
AKARS_DYNAMIC_LINKER="/lib/ld-musl-riscv64v0p7_xthead.so.1"

AKARS_TOOLCHAIN_URL="https://occ-oss-prod.oss-cn-hangzhou.aliyuncs.com/resource//1777015046405/Xuantie-900-gcc-linux-6.6.36-musl64-x86_64-V3.4.0-20260323.tar.gz"
AKARS_TOOLCHAIN_SHA256="10306ce30f98c8168d47f59487da83ba869d5d191193654a27032835d9bb16f8"
AKARS_TOOLCHAIN_ARCHIVE="Xuantie-900-gcc-linux-6.6.36-musl64-x86_64-V3.4.0-20260323.tar.gz"
AKARS_TOOLCHAIN_EXTRACTED="Xuantie-900-gcc-linux-6.6.36-musl64-x86_64-V3.4.0"
AKARS_TOOLCHAIN_NAME="xuantie-v3.4.0"

AKARS_TOOLCHAINS_DIR="$AKARS_ROOT/toolchains"
AKARS_TOOLCHAIN_CACHE="$AKARS_TOOLCHAINS_DIR/.cache"
AKARS_TOOLCHAIN_DIR="${AKARS_TOOLCHAIN_DIR:-$AKARS_TOOLCHAINS_DIR/$AKARS_TOOLCHAIN_NAME}"
AKARS_TPU_SDK_SUBMODULE="toolchains/tpu-sdk-sg200x"
AKARS_TPU_SDK_DIR="${AKARS_TPU_SDK_DIR:-$AKARS_ROOT/$AKARS_TPU_SDK_SUBMODULE}"

AKARS_CC="$AKARS_TOOLCHAIN_DIR/bin/riscv64-unknown-linux-musl-gcc"
AKARS_CXX="$AKARS_TOOLCHAIN_DIR/bin/riscv64-unknown-linux-musl-g++"
AKARS_AR="$AKARS_TOOLCHAIN_DIR/bin/riscv64-unknown-linux-musl-ar"

akars_toolchain_ready() {
  [[ -x "$AKARS_CC" && -x "$AKARS_CXX" && -x "$AKARS_AR" && -d "$AKARS_TOOLCHAIN_DIR/sysroot" ]]
}

akars_tpu_sdk_ready() {
  [[ -f "$AKARS_TPU_SDK_DIR/include/cviruntime.h" \
    && -f "$AKARS_TPU_SDK_DIR/lib/libcviruntime.so" \
    && -f "$AKARS_TPU_SDK_DIR/lib/libcvikernel.so" ]]
}
