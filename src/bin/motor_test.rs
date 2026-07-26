//! Simple motor (ESP32 TT PID) test program.
//!
//! Sends INIT → CONFIG → drive commands step by step, printing every
//! response. Run with:
//!
//!   ./motor_test                            # uses /dev/ttyS1 (JTAG pads, no WiFi conflict)
//!   ./motor_test --motor /dev/ttyS3         # UART3 (GPIOP pads, breaks WiFi)

use akars::motor::{Motor, MotorConfig};
use std::env;
use std::thread;
use std::time::Duration;

fn main() {
    let mut device = "/dev/ttyS1".to_string();
    let mut args = env::args().skip(1).peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--motor" => device = args.next().unwrap_or_else(|| "/dev/ttyS1".into()),
            other => eprintln!("warning: unknown flag {other}"),
        }
    }

    println!("=== Motor Test ===");
    println!("Device: {device}");
    println!("Baud: 115200, Protocol: binary TT PID (ESP32-C3)");
    println!();

    akars::pinmux::configure_for_device(&device);

    let config = MotorConfig {
        device,
        ..MotorConfig::default()
    };

    println!("1. Opening motor ...");
    let mut motor = match Motor::open(&config) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("FAILED to open motor: {e}");
            std::process::exit(1);
        }
    };
    println!("   Motor opened OK");
    println!();

    // Test 1: Forward
    println!("2. FORWARD (speed=30) for 2s ...");
    motor.forward(30);
    println!("   -> wrote left+right CMD_SET_SPEED");
    thread::sleep(Duration::from_millis(2000));

    // Test 2: Stop
    println!("3. STANDBY ...");
    motor.standby();
    println!("   -> wrote left+right CMD_STOP");
    thread::sleep(Duration::from_millis(500));

    // Test 3: Backward
    println!("4. BACKWARD (speed=20) for 1.5s ...");
    motor.backward(20);
    thread::sleep(Duration::from_millis(1500));

    // Test 4: Left turn
    println!("5. LEFT turn (speed=20) for 1s ...");
    motor.left(20);
    thread::sleep(Duration::from_millis(1000));

    // Test 5: Right turn
    println!("6. RIGHT turn (speed=20) for 1s ...");
    motor.right(20);
    thread::sleep(Duration::from_millis(1000));

    // Test 6: Brake
    println!("7. BRAKE ...");
    motor.brake();
    println!("   -> wrote left+right CMD_BRAKE");
    thread::sleep(Duration::from_millis(500));

    println!();
    println!("=== Motor test complete ===");
    println!("If the car did not move:");
    println!("  - Check ESP32 is powered (look for LED on ESP32 board)");
    println!("  - Verify baud rate (ESP32-C3 expects 115200)");
    println!("  - Try different UART: --motor /dev/ttyS1 (JTAG) or --motor /dev/ttyS3 (GPIOP)");
    println!("  - Check pinmux output above for routing errors");
    println!("  - Note: UART3 (/dev/ttyS3) shares pins with SDIO (WiFi)");
}
