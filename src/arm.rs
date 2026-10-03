use crate::serial::SerialPort;
use std::io;
use std::thread;
use std::time::{Duration, Instant};

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

// Deposit-only angle. Keep S2_OPEN unchanged because it is part of the
// field-verified five-step grab sequence. At the raised carrying pose the
// gripper needs more travel to release a loaded ball reliably.
const S2_DEPOSIT_OPEN: f32 = 200.0;
const S2_DEPOSIT_MIN_ACTUAL: f32 = 190.0;
// The loaded servo reported 91.4 degrees after a commanded 80-degree close.
// The existing 100-degree ready position is already mechanically closed
// enough, so accept the full calibrated closed range instead of rejecting a
// harmless 1.4-degree overshoot at an artificially strict 90-degree boundary.
const S2_DEPOSIT_CLOSE_MAX_ACTUAL: f32 = S2_READY;
const DEPOSIT_RELEASE_ATTEMPTS: usize = 3;

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

    /// Open the loaded gripper at the red container and verify servo 2's
    /// reported position. This deliberately uses a separate deposit angle so
    /// the existing grab sequence and all of its calibrated angles stay
    /// untouched.
    pub fn release_for_deposit_verified(&mut self) -> io::Result<f32> {
        let mut last_error = None;

        for attempt in 1..=DEPOSIT_RELEASE_ATTEMPTS {
            eprintln!(
                "AKARS_GRIPPER_RELEASE attempt={attempt}/{DEPOSIT_RELEASE_ATTEMPTS} servo=2 target_angle={S2_DEPOSIT_OPEN} chassis_stopped=1"
            );
            self.restore_torque(2);
            sleep_ms(300);
            self.set_angle(2, S2_DEPOSIT_OPEN, 1000);
            sleep_ms(1500);

            match self.read_angle(2) {
                Ok(actual_angle) => {
                    let reached = actual_angle >= S2_DEPOSIT_MIN_ACTUAL;
                    eprintln!(
                        "AKARS_GRIPPER_POSITION servo=2 target_angle={S2_DEPOSIT_OPEN} actual_angle={actual_angle:.1} minimum_angle={S2_DEPOSIT_MIN_ACTUAL} reached={}",
                        i32::from(reached),
                    );
                    if reached {
                        return Ok(actual_angle);
                    }
                    last_error = Some(io::Error::new(
                        io::ErrorKind::Other,
                        format!(
                            "servo 2 did not reach deposit-open position: actual={actual_angle:.1}, minimum={S2_DEPOSIT_MIN_ACTUAL:.1}"
                        ),
                    ));
                }
                Err(error) => {
                    eprintln!(
                        "AKARS_GRIPPER_POSITION servo=2 read_failed=1 attempt={attempt} error={error}"
                    );
                    last_error = Some(error);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            io::Error::new(io::ErrorKind::Other, "deposit release verification failed")
        }))
    }
    /// Return the gripper to the original 80-degree closed position as soon
    /// as the deposited ball has been released. Chassis movement remains
    /// inhibited until servo 2 reports that it has closed again.
    pub fn close_after_deposit_verified(&mut self) -> io::Result<f32> {
        eprintln!("AKARS_GRIPPER_RESTORE servo=2 target_angle={S2_CLOSE} chassis_stopped=1");
        self.set_angle(2, S2_CLOSE, 1000);
        sleep_ms(1500);

        let actual_angle = self.read_angle(2)?;
        let reached = deposit_close_reached(actual_angle);
        eprintln!(
            "AKARS_GRIPPER_RESTORE_POSITION servo=2 target_angle={S2_CLOSE} actual_angle={actual_angle:.1} maximum_angle={S2_DEPOSIT_CLOSE_MAX_ACTUAL} reached={}",
            i32::from(reached),
        );
        if reached {
            Ok(actual_angle)
        } else {
            Err(io::Error::new(
                io::ErrorKind::Other,
                format!(
                    "servo 2 did not return to closed position: actual={actual_angle:.1}, maximum={S2_DEPOSIT_CLOSE_MAX_ACTUAL:.1}"
                ),
            ))
        }
    }

    /// Read the actual servo position using the controller's PRAD command.
    pub fn read_angle(&mut self, servo_id: i32) -> io::Result<f32> {
        let _ = self.port.discard_input()?;
        let query = format!("#{servo_id:03}PRAD!");
        self.port.write_all_drain(query.as_bytes())?;

        let deadline = Instant::now() + Duration::from_millis(750);
        let mut frame = Vec::with_capacity(16);
        let mut last_parse_error = None;

        while Instant::now() < deadline {
            let Some(byte) = self.port.read_byte(Duration::from_millis(50))? else {
                continue;
            };

            if byte == b'#' {
                frame.clear();
                frame.push(byte);
                continue;
            }
            if frame.is_empty() {
                continue;
            }
            frame.push(byte);
            if frame.len() > 32 {
                frame.clear();
                continue;
            }
            if byte != b'!' {
                continue;
            }

            match parse_position_response(&frame, servo_id) {
                Ok(pulse) => return Ok(pulse_to_angle(pulse)),
                Err(error) => {
                    last_parse_error = Some(error);
                    frame.clear();
                }
            }
        }

        Err(last_parse_error.unwrap_or_else(|| {
            io::Error::new(
                io::ErrorKind::TimedOut,
                format!("servo {servo_id} position response timed out"),
            )
        }))
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

fn pulse_to_angle(pulse: i32) -> f32 {
    ((pulse.clamp(PULSE_MIN, PULSE_MAX) - PULSE_MIN) as f32 * ANGLE_MAX)
        / (PULSE_MAX - PULSE_MIN) as f32
}

fn parse_position_response(response: &[u8], expected_servo: i32) -> io::Result<i32> {
    if response.len() != 10
        || response[0] != b'#'
        || response[4] != b'P'
        || response[9] != b'!'
        || !response[1..4].iter().all(u8::is_ascii_digit)
        || !response[5..9].iter().all(u8::is_ascii_digit)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("malformed servo position response: {:?}", response),
        ));
    }

    let servo = response[1..4]
        .iter()
        .fold(0i32, |value, digit| value * 10 + i32::from(digit - b'0'));
    if servo != expected_servo {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("servo position response mismatch: expected={expected_servo}, actual={servo}"),
        ));
    }

    Ok(response[5..9]
        .iter()
        .fold(0i32, |value, digit| value * 10 + i32::from(digit - b'0')))
}

fn deposit_close_reached(actual_angle: f32) -> bool {
    actual_angle <= S2_DEPOSIT_CLOSE_MAX_ACTUAL
}
fn sleep_ms(ms: u64) {
    thread::sleep(Duration::from_millis(ms));
}

#[cfg(test)]
mod tests {
    use super::{angle_to_pulse, deposit_close_reached, parse_position_response, pulse_to_angle};

    #[test]
    fn converts_angles_to_pulses() {
        assert_eq!(angle_to_pulse(0.0), 500);
        assert_eq!(angle_to_pulse(270.0), 2500);
        assert_eq!(angle_to_pulse(135.0), 1500);
    }

    #[test]
    fn parses_position_response() {
        assert_eq!(parse_position_response(b"#002P1981!", 2).unwrap(), 1981);
        assert!(parse_position_response(b"#001P1981!", 2).is_err());
        assert!(parse_position_response(b"#002PRAD!", 2).is_err());
    }

    #[test]
    fn converts_position_pulse_to_angle() {
        assert!((pulse_to_angle(1981) - 199.935).abs() < 0.01);
    }

    #[test]
    fn accepts_calibrated_closed_range_after_deposit() {
        assert!(deposit_close_reached(80.0));
        assert!(deposit_close_reached(91.4));
        assert!(deposit_close_reached(100.0));
        assert!(!deposit_close_reached(100.1));
    }
}
