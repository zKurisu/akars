#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY="$ROOT/target/riscv64gc-unknown-linux-musl/release/akars"
DEST="root/akars"
ORIGINAL_ARGS=("$@")

usage() {
  cat <<'USAGE'
Usage: scripts/upload.sh DEVICE

Mount DEVICE at a temporary directory and copy the SG2002 akars binary to
/root/akars on that filesystem.

Example:
  scripts/upload.sh /dev/sda2
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

warn() {
  echo "warning: $*" >&2
}

if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  usage
  exit 0
fi

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

DEVICE="$1"

[[ -f "$BINARY" ]] || die "binary not found: $BINARY; run scripts/build.sh first"
[[ -e "$DEVICE" ]] || die "device not found: $DEVICE"

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  command -v sudo >/dev/null 2>&1 || die "sudo is required to mount $DEVICE"
  exec sudo -- "$0" "${ORIGINAL_ARGS[@]}"
fi

MOUNT_DIR=""
MOUNTED=0

cleanup() {
  local status=$?
  if [[ "$MOUNTED" -eq 1 ]]; then
    umount "$MOUNT_DIR" || warn "failed to unmount $MOUNT_DIR during cleanup"
  fi
  if [[ -n "$MOUNT_DIR" ]]; then
    rmdir "$MOUNT_DIR" || warn "failed to remove temporary mount directory: $MOUNT_DIR"
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

MOUNT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/akars-upload.XXXXXX")" || die "failed to create temporary mount directory"

mount "$DEVICE" "$MOUNT_DIR" || die "failed to mount $DEVICE at $MOUNT_DIR"
MOUNTED=1

install -D -m 0755 "$BINARY" "$MOUNT_DIR/$DEST" || die "failed to copy $BINARY to $MOUNT_DIR/$DEST"
sync "$MOUNT_DIR" || sync || warn "sync failed"

umount "$MOUNT_DIR" || die "failed to unmount $MOUNT_DIR"
MOUNTED=0
rmdir "$MOUNT_DIR" || warn "failed to remove temporary mount directory: $MOUNT_DIR"
MOUNT_DIR=""
trap - EXIT INT TERM

echo "uploaded $BINARY to $DEVICE:/$DEST"
