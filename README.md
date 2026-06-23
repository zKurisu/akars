# akars

Rust rewrite entry point for AKA-00 hardware control on SG2002.

Implemented scope:

- USB camera capture from `/dev/cvi-usb-camera0` using the StarryOS ioctl protocol.
- MJPEG decode and letterbox preprocessing through the TPU SDK's OpenCV libraries.
- YOLOv8 CVI runtime inference and postprocess from `aka-sg2002/detect.*`.
- UART motor protocol from `aka-rk3588/motor/uart_motor_driver.*`.
- ZP10D arm UART protocol from `aka-rk3588/arm`.
- Tennis chasing and grab state machine from `aka-sg2002/tennis.cpp`.

`AKA-00/demo` and shell scripts are intentionally not rewritten here.

Host builds compile the pure Rust pieces and use a TPU stub. SG2002 builds link
the TPU SDK and its bundled OpenCV libraries for `riscv64gc-unknown-linux-musl`.

Build for SG2002:

```bash
cd akars
scripts/setup.sh
scripts/build.sh
```

`scripts/setup.sh` installs the Rust target, initializes the TPU SDK submodule,
downloads the Xuantie V3.4.0 musl toolchain, verifies its SHA-256, and extracts
it to `toolchains/xuantie-v3.4.0`. The TPU SDK lives at
`toolchains/tpu-sdk-sg200x` as a git submodule.

For an offline setup, provide the toolchain archive explicitly:

```bash
scripts/setup.sh --archive /path/to/Xuantie-900-gcc-linux-6.6.36-musl64-x86_64-V3.4.0-20260323.tar.gz
```

Override local paths when needed:

```bash
AKARS_TOOLCHAIN_DIR=/path/to/xuantie-v3.4.0 \
AKARS_TPU_SDK_DIR=/path/to/tpu-sdk-sg200x \
scripts/build.sh
```

Build facts:

- Target: `riscv64gc-unknown-linux-musl`. The AKA-00 board runs a musl rootfs,
  not glibc and not bare-metal `-none-elf`.
- Rust uses the prebuilt target `std`; nightly and `-Zbuild-std` are not needed.
- `.cargo/config.toml` disables static CRT and requests the board loader:
  `/lib/ld-musl-riscv64v0p7_xthead.so.1`.
- `scripts/linker.sh` uses the Xuantie V3.4.0 GCC driver directly; its GNU ld
  links current Rust output successfully.

The output binary is `target/riscv64gc-unknown-linux-musl/release/akars`. On the
device it needs `libcviruntime.so`, `libcvikernel.so`, `libopencv_*.so.3.2`, and
the matching `libstdc++.so.6` reachable via `LD_LIBRARY_PATH`.

Upload the binary to a rootfs partition device, for example an SD card second
partition:

```bash
scripts/upload.sh /dev/sda2
```

Run on device:

```bash
./akars /path/to/yolo_model.cvimodel \
  --camera /dev/cvi-usb-camera0 \
  --motor /dev/ttyS3 \
  --arm /dev/ttyS2
```

Useful options:

- `--frames N`: stop after N frames for smoke tests.
- `--conf X`: confidence threshold, default `0.5`.
- `--iou X`: NMS IOU threshold, default `0.5`.
- `--classes N`: class count, default `1`.

Test TPU inference on a single image:

```bash
./akars detect /path/to/yolo_model.cvimodel input.jpg --out result.jpg
```

Runs the model on `input.jpg`, prints each detection (class, score, box), and
writes `result.jpg` with the boxes drawn. Same `--classes` / `--conf` / `--iou`
options as above; `--out` defaults to `detect_out.jpg`. Useful for verifying the
model and runtime independently of the camera and motors.

Web control:

```bash
./akars serve \
  --listen 0.0.0.0:8080 \
  --motor /dev/ttyS3 \
  --arm /dev/ttyS2
```

Open `http://<device-ip>:8080/` to control the base and arm from the built-in frontend.
Use `--mock` to run the same frontend/backend on a development machine without AKA hardware.
