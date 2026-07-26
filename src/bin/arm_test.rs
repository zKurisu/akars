//! Simple robotic arm (ZP10S servo) test program.
//!
//! Cycles through preset positions to verify each servo and the grab
//! sequence. Run with:
//!
//!   ./arm_test                              # uses /dev/ttyS2
//!   ./arm_test --arm /dev/ttyS2 --cycles 3  # run 3 cycles

use akars::arm::Arm;
use std::env;
use std::thread;
use std::time::Duration;

fn sleep_ms(ms: u64) {
    thread::sleep(Duration::from_millis(ms));
}

fn main() {
    let mut device = "/dev/ttyS2".to_string();
    let mut cycles: u32 = 1;
    let mut args = env::args().skip(1).peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--arm" => device = args.next().unwrap_or_else(|| "/dev/ttyS2".into()),
            "--cycles" => {
                cycles = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(1);
            }
            other => eprintln!("warning: unknown flag {other}"),
        }
    }

    println!("=== Arm Test ===");
    println!("Device: {device}");
    println!("Baud: 115200, Protocol: ZP10S ASCII (#NNNPTTTTT!)");
    println!("Cycles: {cycles}");
    println!();

    akars::pinmux::configure_for_device(&device);

    println!("1. Opening arm ...");
    let mut arm = match Arm::open(&device, 115200) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("FAILED to open arm: {e}");
            std::process::exit(1);
        }
    };
    println!("   Arm opened OK, restoring torque ...");
    sleep_ms(500);

    arm.restore_torque(0);
    arm.restore_torque(1);
    arm.restore_torque(2);
    println!("   Torque restored");
    sleep_ms(500);

    println!();
    println!("2. Running grab_pos (set servos to ready position) ...");
    arm.grab_pos();
    println!("   -> servo0=150deg, servo1=100deg, servo2=110deg(open)");
    sleep_ms(1500);

    for cycle in 1..=cycles {
        println!();
        println!("=== Cycle {cycle}/{cycles} ===");
        println!();

        println!("3. GRAB sequence (pick up) ...");
        arm.grab();
        println!("   -> servo0=225, servo1=60, open->close gripper, lift");
        sleep_ms(4500);

        println!("4. SHOW (present) ...");
        arm.show();
        println!("   -> servo0=150, servo1=100, servo2=50(closed, show ball)");
        sleep_ms(2000);

        println!("5. RELEASE (drop ball) ...");
        arm.release();
        println!("   -> servo2=110 (open gripper)");
        sleep_ms(1500);

        println!("6. grab_pos (ready for next pickup) ...");
        arm.grab_pos();
        println!("   -> servo0=150, servo1=100, servo2=110(open)");
        sleep_ms(1500);
    }

    println!();
    println!("=== Arm test complete ===");
    println!("If servos did not move:");
    println!("  - Check servo power supply (external 6-8.4V battery)");
    println!("  - Verify the UART servo control board has power (LED on board)");
    println!("  - Check physical wiring: A28=TX, A29=RX on LicheeRV board");
    println!("  - Verify baud rate matches servo controller (115200 for ZP10S)");
}
