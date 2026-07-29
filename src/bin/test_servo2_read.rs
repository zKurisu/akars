//! Quick test: set servo2 angle then read actual position via PRAD.
//!
//! Usage:
//!   ./test_servo2_read --angle 89          # empty close, read
//!   ./test_servo2_read --angle 89 --ball   # with ball, read

use akars::arm::Arm;
use std::env;

fn main() {
    let mut device = "/dev/ttyS2".to_string();
    let mut angle = 89.0f32;
    let mut args = env::args().skip(1).peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--arm" => device = args.next().unwrap_or_else(|| "/dev/ttyS2".into()),
            "--angle" => {
                angle = args.next().and_then(|s| s.parse().ok()).unwrap_or(89.0);
            }
            _ => {}
        }
    }

    println!("=== servo2 PRAD test: target={angle}° ===");
    println!();

    akars::pinmux::configure_for_device(&device);
    let mut arm = Arm::open(&device, 115200).expect("open arm");
    arm.restore_torque(2);
    std::thread::sleep(std::time::Duration::from_millis(300));

    // Set angle and wait for move to finish.
    arm.set_angle(2, angle, 1000);
    std::thread::sleep(std::time::Duration::from_millis(1500));

    // Read actual position.
    let actual = arm.read_angle(2);
    println!("actual={actual:?}");
}
