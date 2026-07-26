//! Test servo 0 lift angle (base rotation — arm up / resting position).
//!
//! Usage:
//!   ./test_servo0_lift                          # default angle 150
//!   ./test_servo0_lift --angle 140              # custom angle
//!   ./test_servo0_lift --arm /dev/ttyS2 --angle 150

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

    println!("=== test_servo0_lift ===");
    println!("Servo: 0 (base rotation)");
    println!("Default angle: {DEFAULT_ANGLE}");
    println!("Testing angle: {angle}");
    println!();

    akars::pinmux::configure_for_device(&device);
    let mut arm = Arm::open(&device, 115200).expect("open arm");
    arm.restore_torque(0);
    thread::sleep(Duration::from_millis(300));

    arm.set_angle(0, angle, 1000);
    println!("Sent: servo0 → {angle}°");
    thread::sleep(Duration::from_millis(1500));

    println!("Done. The servo should now be at {angle}°.");
    println!("Tip: release torque to manually adjust:");
    println!("  curl -sX POST http://<ip>:8080/api/arm/action -H 'Content-Type: application/json' -d '{{\"action\":\"release_torque\",\"servo_id\":0}}'");
}
