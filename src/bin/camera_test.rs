//! Simple camera capture test program.
//!
//! Continuously grabs frames and reports timing. Run with:
//!
//!   ./camera_test                           # uses /dev/cvi-usb-camera0
//!   ./camera_test --camera /dev/video0       # V4L2 device
//!   ./camera_test --frames 50                # stop after 50 frames
//!   ./camera_test --save /tmp/frame          # save YUV→RGB as frame-N.jpg

use akars::camera::UsbCamera;
use std::env;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let mut camera_dev = "/dev/cvi-usb-camera0".to_string();
    let mut max_frames: Option<u64> = None;
    let mut save_prefix: Option<String> = None;

    let mut args = env::args().skip(1).peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--camera" => camera_dev = args.next().unwrap_or_else(|| "/dev/cvi-usb-camera0".into()),
            "--frames" => {
                max_frames = Some(args.next().and_then(|s| s.parse().ok()).unwrap_or(0));
            }
            "--save" => save_prefix = args.next().or(Some("/tmp/frame".into())),
            other => eprintln!("warning: unknown flag {other}"),
        }
    }

    println!("=== Camera Test ===");
    println!("Device: {camera_dev}");
    if let Some(n) = max_frames {
        println!("Frames: {n}");
    }
    if save_prefix.is_some() {
        println!("Save prefix: {}", save_prefix.as_ref().unwrap());
    }
    println!();

    println!("Opening camera ...");
    let mut camera = match UsbCamera::open(&camera_dev) {
        Ok(cam) => {
            let info = cam.info();
            println!("  OK: {}x{} format={} connected={}",
                info.width, info.height, info.format, info.connected);
            cam
        }
        Err(err) => {
            eprintln!("  FAILED: {err}");
            std::process::exit(1);
        }
    };
    println!();

    println!("Frame | Capture   | Total     | FPS  | Status");
    println!("------+-----------+-----------+------+-------");

    let mut frame_idx = 1u64;
    let mut err_count = 0u32;
    let mut total_frames = 0u64;
    let mut total_time = std::time::Duration::ZERO;

    loop {
        if let Some(max) = max_frames {
            if frame_idx > max {
                break;
            }
        }

        let loop_start = Instant::now();
        let capture_start = Instant::now();

        let frame = match camera.get_frame() {
            Ok(frame) => frame,
            Err(err) => {
                err_count += 1;
                println!("{frame_idx:>5} | ERR      |          |      | {err} (#{err_count})");
                if err_count >= 10 {
                    eprintln!("\nToo many errors ({err_count}), stopping.");
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(200));
                frame_idx += 1;
                continue;
            }
        };

        let capture_us = capture_start.elapsed().as_micros() as u64;
        let total_us = loop_start.elapsed().as_micros() as u64;
        let fps = if total_us > 0 { 1_000_000.0 / total_us as f64 } else { 0.0 };

        total_frames += 1;
        total_time += loop_start.elapsed();

        println!(
            "{frame_idx:>5} | {capture_us:>5} µs | {total_us:>5} µs | {fps:>4.1} | OK ({})",
            frame.pixels.len()
        );

        // Optionally save as image
        if let Some(prefix) = &save_prefix {
            let out_path = format!("{prefix}-{frame_idx:03}.jpg");
            if let Err(err) = akars::image_bridge::save_yuv422p(
                &frame.pixels,
                frame.width as i32,
                frame.height as i32,
                PathBuf::from(&out_path).as_path(),
            ) {
                eprintln!("  warn: save failed: {err}");
            } else {
                println!("  -> saved {}", out_path);
            }
        }

        frame_idx += 1;
    }

    // Print summary
    let avg_fps = if total_time.as_secs_f32() > 0.0 {
        total_frames as f32 / total_time.as_secs_f32()
    } else {
        0.0
    };
    println!("\n=== Summary ===");
    println!("Frames captured: {total_frames}");
    println!("Errors: {err_count}");
    println!("Avg FPS: {avg_fps:.2}");
    println!("Total time: {:.1}s", total_time.as_secs_f32());
}
