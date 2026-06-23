#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/env.sh
source "$ROOT/scripts/env.sh"

force=0
check_only=0
archive_override=""

usage() {
  cat <<'USAGE'
Usage: scripts/setup.sh [--force] [--check] [--archive PATH]

Prepare the SG2002 build environment:
  - install the Rust riscv64gc-unknown-linux-musl target
  - initialize the TPU SDK submodule
  - download, verify, and extract the Xuantie V3.4.0 toolchain

Options:
  --force         reinstall the toolchain
  --check         check prerequisites without changing them
  --archive PATH  use a local toolchain archive instead of downloading
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      force=1
      ;;
    --check)
      check_only=1
      ;;
    --archive)
      [[ $# -ge 2 ]] || die "--archive needs a path"
      archive_override="$2"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
  shift
done

rust_target_ready() {
  rustup target list --installed | grep -qx "$AKARS_TARGET"
}

check_ready() {
  local ok=1

  if rust_target_ready; then
    echo "rust target: $AKARS_TARGET"
  else
    echo "missing rust target: $AKARS_TARGET" >&2
    ok=0
  fi

  if akars_tpu_sdk_ready; then
    echo "TPU SDK: $AKARS_TPU_SDK_DIR"
  else
    echo "missing TPU SDK submodule contents: $AKARS_TPU_SDK_DIR" >&2
    ok=0
  fi

  if akars_toolchain_ready; then
    echo "toolchain: $AKARS_TOOLCHAIN_DIR"
  else
    echo "missing Xuantie toolchain: $AKARS_TOOLCHAIN_DIR" >&2
    ok=0
  fi

  [[ "$ok" -eq 1 ]]
}

verify_archive() {
  local archive="$1"
  [[ -f "$archive" ]] || die "archive not found: $archive"
  printf '%s  %s\n' "$AKARS_TOOLCHAIN_SHA256" "$archive" | sha256sum -c - >&2
}

fetch_archive() {
  mkdir -p "$AKARS_TOOLCHAIN_CACHE"
  local archive="$AKARS_TOOLCHAIN_CACHE/$AKARS_TOOLCHAIN_ARCHIVE"

  if [[ -n "$archive_override" ]]; then
    [[ -f "$archive_override" ]] || die "archive not found: $archive_override"
    cp "$archive_override" "$archive"
  elif [[ "$force" -eq 1 || ! -f "$archive" ]]; then
    local partial="$archive.download"
    rm -f "$partial"
    if command -v curl >/dev/null 2>&1; then
      curl -fL --retry 3 --output "$partial" "$AKARS_TOOLCHAIN_URL" || {
        rm -f "$partial"
        die "failed to download toolchain from $AKARS_TOOLCHAIN_URL"
      }
    elif command -v wget >/dev/null 2>&1; then
      wget -O "$partial" "$AKARS_TOOLCHAIN_URL" || {
        rm -f "$partial"
        die "failed to download toolchain from $AKARS_TOOLCHAIN_URL"
      }
    else
      die "curl or wget is required to download the toolchain"
    fi
    [[ -s "$partial" ]] || die "downloaded toolchain archive is empty: $partial"
    mv "$partial" "$archive"
  fi

  verify_archive "$archive"
  printf '%s\n' "$archive"
}

install_toolchain() {
  if [[ "$force" -eq 0 ]] && akars_toolchain_ready; then
    echo "toolchain already installed: $AKARS_TOOLCHAIN_DIR"
    return
  fi

  local archive="$1"
  local tmp_dir="$AKARS_TOOLCHAINS_DIR/.extract.$$"
  local tmp_dst="$AKARS_TOOLCHAIN_DIR.tmp.$$"
  rm -rf "$tmp_dir" "$tmp_dst"
  mkdir -p "$tmp_dir" "$(dirname -- "$AKARS_TOOLCHAIN_DIR")"

  tar -xzf "$archive" -C "$tmp_dir"
  local extracted="$tmp_dir/$AKARS_TOOLCHAIN_EXTRACTED"
  [[ -d "$extracted" ]] || die "archive did not contain $AKARS_TOOLCHAIN_EXTRACTED"

  mv "$extracted" "$tmp_dst"
  rm -rf "$AKARS_TOOLCHAIN_DIR"
  mv "$tmp_dst" "$AKARS_TOOLCHAIN_DIR"
  rm -rf "$tmp_dir"

  akars_toolchain_ready || die "installed toolchain is incomplete: $AKARS_TOOLCHAIN_DIR"
  echo "installed toolchain: $AKARS_TOOLCHAIN_DIR"
}

if [[ "$check_only" -eq 1 ]]; then
  check_ready
  exit $?
fi

if ! rust_target_ready; then
  rustup target add "$AKARS_TARGET"
fi

if ! akars_tpu_sdk_ready; then
  echo "initializing TPU SDK submodule: $AKARS_TPU_SDK_SUBMODULE"
  git -C "$AKARS_ROOT" submodule update --init --recursive "$AKARS_TPU_SDK_SUBMODULE"
fi
akars_tpu_sdk_ready || die "TPU SDK is incomplete: $AKARS_TPU_SDK_DIR"

archive="$(fetch_archive)"
install_toolchain "$archive"

"$AKARS_CC" --version
"$AKARS_CC" --print-sysroot

echo "setup complete"
