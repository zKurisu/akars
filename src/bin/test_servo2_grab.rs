//! Test servo 2 grab angle (gripper — position during grab-down motion).
//!
//! Usage:
//!   ./test_servo2_grab                          # default angle 150
//!   ./test_servo2_grab --angle 160              # custom angle
//!   ./test_servo2_grab --arm /dev/ttyS2 --angle 150

use akars::arm::Arm;
use std::env;
use std::thread;
use std::time::Duration;

const DEFAULT_ANGLE: f32 = 150.0;

fn main() {
    let mut device = "/dev/ttyS2".to_string();
    let mut angle = DEFAULT_ANGLE;
    let mut args = env::args().skip(1).peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--arm" => device = args.next().unwrap_or_else(|| "/dev/ttyS2".into()),
            "--angle" => {
                angle = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(DEFAULT_ANGLE);
            }
            other => eprintln!("warning: unknown flag {other}"),
        }
    }

    println!("=== test_servo2_grab ===");
    println!("Servo: 2 (gripper — during grab-down)");
    println!("Default angle: {DEFAULT_ANGLE}");
    println!("Testing angle: {angle}");
    println!();

    akars::pinmux::configure_for_device(&device);
    let mut arm = Arm::open(&device, 115200).expect("open arm");
    arm.restore_torque(2);
    thread::sleep(Duration::from_millis(300));

    arm.set_angle(2, angle, 1000);
    println!("Sent: servo2 → {angle}°");
    thread::sleep(Duration::from_millis(1500));

    println!("Done.");
}
