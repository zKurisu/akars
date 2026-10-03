# akars

Rust rewrite entry point for AKA-00 hardware control on SG2002.

Implemented scope:

- USB camera capture from `/dev/cvi-usb-camera0` using the StarryOS ioctl protocol.
- MJPEG decode and letterbox preprocessing through a pure Rust image pipeline.
- YOLOv8 CVI runtime inference and postprocess from `aka-sg2002/detect.*`.
- UART motor protocol from `aka-rk3588/motor/uart_motor_driver.*`.
- ZP10D arm UART protocol from `aka-rk3588/arm`.
- Tennis chasing, grab, red-container return, and release state machine based
  on the existing tennis controller and calibrated arm operations.

`AKA-00/demo` and shell scripts are intentionally not rewritten here.

Host builds compile the pure Rust pieces and use a TPU stub. SG2002 builds link
only the TPU SDK runtime libraries for `riscv64gc-unknown-linux-musl`.

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
device it needs `libcviruntime.so`, `libcvikernel.so`, and the matching
`libstdc++.so.6` reachable via `LD_LIBRARY_PATH`.

Upload the binary to a rootfs partition device, for example an SD card second
partition:

```bash
scripts/upload.sh /dev/sda2
```

Run on device:

```bash
./akars /akars_tennis/model/yolov8n_tennis_p2_384_aligned_int8.cvimodel \
  --camera /dev/cvi-usb-camera0 \
  --vpss /dev/cvi-vpss0 \
  --motor /dev/ttyS1 \
  --arm /dev/ttyS2
```

The autonomous hunt path requires an aligned, fused-preprocess UINT8 RGB-planar model. Camera MJPEG is decoded by JPU into an ION YUV422P buffer, VPSS performs letterbox and colour conversion into an RGB-planar ION buffer, and the TPU runtime binds that physical buffer directly. The older non-aligned yolov8n_tennis_v2.cvimodel remains usable by detect, but hunt deliberately rejects it so the controller cannot silently fall back to CPU preprocessing.

The autonomous mission is cyclic: chase and grab a tennis ball, search for the
red container only after the grab completes, approach until its bounding box
covers 100% of the frame width and 98% of its height for three frames, release
the ball directly from the carrying pose, rotate in place to search behind it,
and resume tennis-ball chasing. Red detections are ignored while the gripper is
empty. The current ZP10 arm interface has no ball-presence sensor, so
`holding_ball` records completion of the existing blocking `Arm::grab()`
sequence rather than independent physical feedback.

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

Approach the only red object in a clean test scene:

```bash
./red_cloth_follow \
  --camera /dev/cvi-usb-camera0 \
  --motor /dev/ttyS1
```

This command reuses the existing YUV422P camera capture and motor UART code. It
centres the red object, reduces forward speed as the red coverage grows, brakes
after three consecutive full-view observations, and has a 120-second safety
timeout. Forward speed, pulsed steering, and continuous search reuse the same helpers as the existing tennis-chase controller. Always validate colour thresholds and steering without motor output first:

```bash
./red_cloth_follow --dry-run --max-frames 300
```

The default full-view condition is either 98% red pixels, or at least 55% red
pixels whose bounding box covers 100% of the frame width and 98% of its height.
The condition must hold for three consecutive frames. Use `--help` for
threshold and safety-limit options.
