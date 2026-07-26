//! Test servo 1 prepare angle (shoulder — arm reaching down to grab).
//!
//! Usage:
//!   ./test_servo1_prepare                       # default angle 150
//!   ./test_servo1_prepare --angle 140           # custom angle
//!   ./test_servo1_prepare --arm /dev/ttyS2 --angle 150

use akars::arm::Arm;
use std::env;
use std::thread;
use std::time::Duration;

const DEFAULT_ANGLE: f32 = 100.0;

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

    println!("=== test_servo1_prepare ===");
    println!("Servo: 1 (shoulder lift)");
    println!("Default angle: {DEFAULT_ANGLE}");
    println!("Testing angle: {angle}");
    println!();

    akars::pinmux::configure_for_device(&device);
    let mut arm = Arm::open(&device, 115200).expect("open arm");
    arm.restore_torque(1);
    thread::sleep(Duration::from_millis(300));

    arm.set_angle(1, angle, 1000);
    println!("Sent: servo1 → {angle}°");
    thread::sleep(Duration::from_millis(1500));

    println!("Done. The servo should now be at {angle}°.");
}
