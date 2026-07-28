//! Red-cloth following test.
//!
//! Detects a red cloth on the floor using YUV colour-space thresholding
//! (no OpenCV needed — the camera already outputs YUV422) and drives the
//! robot toward the largest red blob.
//!
//! Usage:
//!   ./red_cloth_follow
//!   ./red_cloth_follow --camera /dev/cvi-usb-camera0 --motor /dev/ttyS1

use akars::camera::UsbCamera;
use akars::motor::{Motor, MotorConfig};
use std::env;
use std::thread;
use std::time::{Duration, Instant};

// ── CLI defaults ──────────────────────────────────────────────────────
const DEFAULT_CAMERA: &str = "/dev/cvi-usb-camera0";
const DEFAULT_MOTOR: &str = "/dev/ttyS1";

// ── Red detection thresholds (YUV422, 8-bit) ─────────────────────────
// YUV values are centred at 128 for U/V.  Red has high V (Cr), low U (Cb).
const V_RED_MIN: u8 = 155;   // V (Cr) must be above this  (red chroma)
const U_RED_MAX: u8 = 115;   // U (Cb) must be below this  (not blue/green)

// ── Navigation ────────────────────────────────────────────────────────
const FORWARD_SPEED: i32 = 40;
const TURN_SPEED: i32 = 10;
const SEARCH_SPEED: i32 = 8;
const CENTER_MARGIN: i32 = 60; // pixels tolerance for "centred"
const MIN_BLOB_PIXELS: usize = 200; // ignore tiny red speckles

// ── YUV422 frame layout ───────────────────────────────────────────────
const FRAME_W: usize = 640;
const FRAME_H: usize = 480;
const Y_PLANE_SIZE: usize = FRAME_W * FRAME_H;       // 307200
const UV_PLANE_SIZE: usize = (FRAME_W / 2) * FRAME_H; // 153600

fn sleep_ms(ms: u64) {
    thread::sleep(Duration::from_millis(ms));
}

fn sleep_us(us: u64) {
    thread::sleep(Duration::from_micros(us));
}

fn main() {
    let mut camera_path = DEFAULT_CAMERA.to_string();
    let mut motor_path = DEFAULT_MOTOR.to_string();
    let mut args = env::args().skip(1).peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--camera" => camera_path = args.next().unwrap_or_else(|| DEFAULT_CAMERA.into()),
            "--motor" => motor_path = args.next().unwrap_or_else(|| DEFAULT_MOTOR.into()),
            other => eprintln!("warning: unknown flag {other}"),
        }
    }

    println!("=== Red Cloth Follow Test ===");
    println!("Camera : {camera_path}");
    println!("Motor  : {motor_path}");
    println!("V_min  : {V_RED_MIN}  U_max  : {U_RED_MAX}");
    println!();

    // ── Open camera ──
    let mut camera = UsbCamera::open(&camera_path).expect("open camera");
    let info = camera.info();
    println!("Camera opened: {}x{} format={}", info.width, info.height, info.format);

    // ── Open motor ──
    let motor_config = MotorConfig {
        device: motor_path.clone(),
        ..MotorConfig::default()
    };
    let mut motor = Motor::open(&motor_config).expect("open motor");
    println!("Motor opened: {motor_path}");

    println!("\nStarting red-cloth following loop (Ctrl-C to stop)...\n");

    loop {
        let frame_start = Instant::now();

        // ── Capture frame ──
        let capture_start = Instant::now();
        let frame = match camera.get_frame() {
            Ok(f) => f,
            Err(e) => {
                eprintln!("[camera] get_frame failed: {e}");
                sleep_ms(100);
                continue;
            }
        };
        let cap_ms = capture_start.elapsed().as_millis();

        // ── Detect red blob in YUV422 ──
        let detect_start = Instant::now();
        let (cx, area) = detect_red_yuv(&frame.pixels);
        let det_ms = detect_start.elapsed().as_millis();

        let total_ms = frame_start.elapsed().as_millis();
        let image_cx = (FRAME_W / 2) as i32;
        let offset = cx - image_cx;
        let centered = offset.abs() <= CENTER_MARGIN;

        if area >= MIN_BLOB_PIXELS {
            let area_pct = area as f32 / (FRAME_W * FRAME_H) as f32 * 100.0;
            eprintln!(
                "[red] area={area}px ({area_pct:.1}%) cx={cx} offset={offset} \
                 centered={centered} cap={cap_ms}ms det={det_ms}ms total={total_ms}ms"
            );

            if centered {
                // Red cloth is ahead — drive forward.
                println!("  → forward (centred)");
                motor.forward(FORWARD_SPEED);
            } else if offset < 0 {
                // Red is to the left — turn right.
                println!("  → turn right  (offset={offset})");
                motor.drive(TURN_SPEED, -TURN_SPEED);
                sleep_us(60_000);
                motor.standby();
            } else {
                // Red is to the right — turn left.
                println!("  → turn left   (offset={offset})");
                motor.drive(-TURN_SPEED, TURN_SPEED);
                sleep_us(60_000);
                motor.standby();
            }
        } else {
            eprintln!(
                "[red] no red found  cap={cap_ms}ms det={det_ms}ms total={total_ms}ms"
            );
            // Search: slow turn to look for the red cloth.
            motor.drive(SEARCH_SPEED, -SEARCH_SPEED);
            sleep_ms(200);
            motor.standby();
        }
    }
}

/// Scan the YUV422 frame for red pixels and return (centre-x, pixel count)
/// of the largest contiguous red region.
///
/// YUV422 planar layout:
///   [Y plane: 640×480] [U plane: 320×480] [V plane: 320×480]
///
/// For each pixel at (x, y) in the full-resolution grid, the chroma sample
/// is at (x/2, y) in the U/V planes.
fn detect_red_yuv(yuv: &[u8]) -> (i32, usize) {
    if yuv.len() < Y_PLANE_SIZE + 2 * UV_PLANE_SIZE {
        return (0, 0);
    }

    let u_plane = &yuv[Y_PLANE_SIZE..Y_PLANE_SIZE + UV_PLANE_SIZE];
    let v_plane = &yuv[Y_PLANE_SIZE + UV_PLANE_SIZE..];

    let mut sum_x: u64 = 0;
    let mut count: usize = 0;

    for row in 0..FRAME_H {
        let uv_row_off = row * (FRAME_W / 2);

        for col in 0..FRAME_W {
            let chroma_idx = uv_row_off + col / 2;
            let v = v_plane[chroma_idx];
            let u = u_plane[chroma_idx];

            // Red has high V (Cr) and low U (Cb).
            if v >= V_RED_MIN && u <= U_RED_MAX {
                sum_x += col as u64;
                count += 1;
            }
        }
    }

    if count == 0 {
        return (0, 0);
    }

    let avg_cx = (sum_x / count as u64) as i32;
    (avg_cx, count)
}
