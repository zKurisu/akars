use crate::serial::SerialPort;
use std::io;
use std::thread;
use std::time::Duration;

const ID2_ANGLE_OPEN: f32 = 140.0;
const ID2_ANGLE_CLOSE: f32 = 90.0;
const ANGLE_MAX: f32 = 270.0;
const PULSE_MIN: i32 = 500;
const PULSE_MAX: i32 = 2500;

const SERVO0_READY: f32 = 180.0;
const SERVO1_READY: f32 = 185.0;
const SERVO0_GRAB: f32 = 245.0;
const SERVO1_GRAB: f32 = 185.0;
const SERVO0_LIFT: f32 = 180.0;
const SERVO1_LIFT: f32 = 100.0;

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
        self.send_command(&format!("#{servo_id:03}P{pulse:04}T{time_ms}!"));
    }

    pub fn release_torque(&mut self, servo_id: i32) {
        self.send_command(&format!("#{servo_id:03}PULK"));
    }

    pub fn restore_torque(&mut self, servo_id: i32) {
        self.send_command(&format!("#{servo_id:03}PULR"));
    }

    pub fn grab(&mut self) {
        self.set_angle(0, SERVO0_GRAB, 1000);
        self.set_angle(1, SERVO1_GRAB, 1000);
        self.set_angle(2, ID2_ANGLE_OPEN, 1000);
        sleep_ms(1500);

        self.set_angle(2, ID2_ANGLE_CLOSE, 1000);
        sleep_ms(1000);

        self.set_angle(0, SERVO0_LIFT, 1000);
        self.set_angle(1, SERVO1_LIFT, 1000);
        sleep_ms(1200);
    }

    pub fn release_pos(&mut self) {
        self.set_angle(0, SERVO0_READY, 1000);
        self.set_angle(1, SERVO1_READY, 1000);
        self.set_angle(2, ID2_ANGLE_OPEN, 1000);
    }

    pub fn release(&mut self) {
        self.set_angle(2, ID2_ANGLE_OPEN, 1000);
    }

    pub fn grab_pos(&mut self) {
        self.set_angle(0, SERVO0_READY, 1000);
        self.set_angle(1, SERVO1_READY, 1000);
        self.set_angle(2, ID2_ANGLE_OPEN, 1000);
    }

    pub fn show(&mut self) {
        self.set_angle(0, SERVO0_LIFT, 1000);
        self.set_angle(1, SERVO1_LIFT, 1000);
        self.set_angle(2, ID2_ANGLE_CLOSE, 1000);
    }

    fn send_command(&mut self, command: &str) {
        if let Err(err) = self.port.write_all_drain(command.as_bytes()) {
            eprintln!("[arm] write failed: {err}");
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
