use crate::serial::SerialPort;
use std::io;
use std::thread;
use std::time::Duration;

// ── Servo angle constants ───────────────────────────────────────────
// Servo 0: base swing     (decrease = up, increase = down)
// Servo 1: shoulder lift   (decrease = down, increase = up)
// Servo 2: gripper         (decrease = close, increase = open)
//
// 5-step grab sequence (see grab() below).

const ANGLE_MAX: f32 = 270.0;
const PULSE_MIN: i32 = 500;
const PULSE_MAX: i32 = 2500;

// ── Step 1: Initial/ready position ──
const S0_READY: f32 = 150.0;
const S1_READY: f32 = 140.0;
const S2_READY: f32 = 100.0;

// ── Step 2: Reach down, push ball toward body ──
const S0_REACH: f32 = 198.0; // servo0 swings down
const S1_PUSH: f32 = 115.0; // servo1 pushes arm toward ball

// ── Step 3: Open gripper wide + continue pushing ──
const S2_OPEN: f32 = 180.0; // gripper fully open (was 150, wider for visibility)
const S1_CONTINUE: f32 = 95.0; // servo1 keeps going down

// ── Step 4: Close gripper to grab ball ──
const S2_CLOSE: f32 = 80.0; // gripper clamped closed (was 100, tighter grip)

// ── Step 5: Lift arm back to ready ──
// Uses S0_READY + S1_READY; gripper stays closed (S2_CLOSE)

pub struct Arm {
    port: SerialPort,
}

impl Arm {
    pub fn open(device: &str, baudrate: i32) -> io::Result<Self> {
        crate::pinmux::configure_for_device(device);
        Ok(Self {
            port: SerialPort::open(device, baudrate, false)?,
        })
    }

    pub fn set_angle(&mut self, servo_id: i32, angle: f32, time_ms: i32) {
        let clamped = angle.clamp(0.0, ANGLE_MAX);
        let pulse = angle_to_pulse(clamped);
        let cmd = format!("#{servo_id:03}P{pulse:04}T{time_ms}!\r\n");
        eprintln!("[arm] set servo {servo_id} angle={clamped} pulse={pulse} time={time_ms}");
        self.send_command(&cmd);
    }

    pub fn release_torque(&mut self, servo_id: i32) {
        self.send_command(&format!("#{servo_id:03}PULK\r\n"));
    }

    pub fn restore_torque(&mut self, servo_id: i32) {
        eprintln!("[arm] restore_torque servo {servo_id}");
        self.send_command(&format!("#{servo_id:03}PULR\r\n"));
    }

    /// 5-step grab sequence:
    ///   1. (ready)  s0=150  s1=140  s2=120   — arm up, gripper ready
    ///   2. (reach)  s0=200  s1=100            — swing down, push ball toward body
    ///   3. (open)   s2=150  s1=90             — open gripper, keep pushing down
    ///   4. (grab)   s2=100                    — close gripper on ball
    ///   5. (lift)   s0=150  s1=140            — lift arm back to ready
    pub fn grab(&mut self) {
        eprintln!("[arm] === grab sequence start ===");

        // Step 2: servo0 swings down, servo1 pushes toward ball.
        eprintln!("[arm] step 2: reach down (s0={S0_REACH} s1={S1_PUSH})");
        self.set_angle(0, S0_REACH, 1000);
        sleep_ms(300);
        self.set_angle(1, S1_PUSH, 1000);
        sleep_ms(1500); // let both moves finish

        // Step 3: open gripper wide WHILE servo1 keeps going down.
        eprintln!("[arm] step 3: open gripper + push (s2={S2_OPEN} s1={S1_CONTINUE})");
        self.set_angle(2, S2_OPEN, 1000);
        sleep_ms(300);
        self.set_angle(1, S1_CONTINUE, 1000);
        sleep_ms(1500); // both moves are 1000ms, wait for completion

        // Step 4: close gripper — grab the ball.
        eprintln!("[arm] step 4: close gripper (s2={S2_CLOSE})");
        self.set_angle(2, S2_CLOSE, 1000);
        sleep_ms(1500);

        // Step 5: lift arm back to ready position.
        eprintln!("[arm] step 5: lift back (s0={S0_READY} s1={S1_READY})");
        self.set_angle(0, S0_READY, 1000);
        sleep_ms(300);
        self.set_angle(1, S1_READY, 1000);
        sleep_ms(1500);

        eprintln!("[arm] === grab sequence done ===");
    }

    /// Move arm to release position (arm down, gripper still holding).
    pub fn release_pos(&mut self) {
        self.set_angle(0, S0_REACH, 1000);
        sleep_ms(50);
        self.set_angle(1, S1_CONTINUE, 1000);
        sleep_ms(50);
        self.set_angle(2, S2_CLOSE, 1000);
    }

    /// Open the gripper to drop the ball.
    pub fn release(&mut self) {
        self.set_angle(2, S2_OPEN, 800);
    }

    /// Return arm to ready position (arm up, gripper open).
    pub fn grab_pos(&mut self) {
        self.set_angle(0, S0_READY, 1000);
        sleep_ms(50);
        self.set_angle(1, S1_READY, 1000);
        sleep_ms(50);
        self.set_angle(2, S2_READY, 1000);
    }

    /// Display pose: arm lifted, gripper closed (holding ball for show).
    pub fn show(&mut self) {
        self.set_angle(0, S0_READY, 1000);
        sleep_ms(50);
        self.set_angle(1, S1_READY, 1000);
        sleep_ms(50);
        self.set_angle(2, S2_CLOSE, 1000);
    }

    fn send_command(&mut self, command: &str) {
        if let Err(err) = self.port.write_all_drain(command.as_bytes()) {
            eprintln!("[arm] write failed: {err} (cmd={command})");
        }
    }
}

fn angle_to_pulse(angle: f32) -> i32 {
    ((500.0 + (angle / ANGLE_MAX) * 2000.0).round() as i32).clamp(PULSE_MIN, PULSE_MAX)
}

fn sleep_ms(ms: u64) {
    thread::sleep(Duration::from_millis(ms));
}

#[cfg(test)]
mod tests {
    use super::angle_to_pulse;

    #[test]
    fn converts_angles_to_pulses() {
        assert_eq!(angle_to_pulse(0.0), 500);
        assert_eq!(angle_to_pulse(270.0), 2500);
        assert_eq!(angle_to_pulse(135.0), 1500);
    }
}
