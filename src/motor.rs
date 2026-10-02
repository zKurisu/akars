use crate::serial::SerialPort;
use std::io;
use std::thread;
use std::time::Duration;

const FRAME_SOF0: u8 = 0xAA;
const FRAME_SOF1: u8 = 0x55;

const CMD_INIT: u8 = 0x01;
const CMD_CONFIG: u8 = 0x02;
const CMD_SET_SPEEDS: u8 = 0x13;
const CMD_STOP: u8 = 0x11;
const CMD_BRAKE: u8 = 0x12;
const CMD_GET_STATUS: u8 = 0x21;

const RSP_ACK: u8 = 0x80;
const RSP_NACK: u8 = 0x81;
const RSP_STATUS: u8 = 0x91;
const BOTH_MOTORS: u8 = 2;

#[derive(Debug, Clone)]
pub struct MotorConfig {
    pub device: String,
    pub speed_scale: i32,
    pub ppr: u16,
    pub pwm_freq: u16,
    pub min_speed: i32,
}

impl Default for MotorConfig {
    fn default() -> Self {
        Self {
            device: "/dev/ttyS1".to_string(),
            speed_scale: 150,
            ppr: 4680,
            pwm_freq: 20000,
            min_speed: 15,
        }
    }
}

pub struct Motor {
    port: Option<SerialPort>,
    speed_scale: i32,
    min_speed: i32,
}

impl Motor {
    pub fn open(config: &MotorConfig) -> io::Result<Self> {
        // The UART pads need muxing before the port will physically transmit;
        // the device tree enables UART1 but leaves its pins unrouted.
        crate::pinmux::configure_for_device(&config.device);
        let mut port = SerialPort::open(&config.device, 115200, true)?;
        eprintln!("[motor] port opened, waiting 500ms for ESP32");
        thread::sleep(Duration::from_millis(500));
        let discarded = port.discard_input()?;
        eprintln!("[motor] discarded {discarded} stale RX byte(s)");

        let mut motor = Self {
            port: Some(port),
            speed_scale: config.speed_scale,
            min_speed: config.min_speed,
        };
        motor.command_expect_ack(CMD_INIT, &[], "INIT")?;
        let mut payload = [0u8; 4];
        put_be16(&mut payload[0..2], config.ppr);
        put_be16(&mut payload[2..4], config.pwm_freq);
        motor.command_expect_ack(CMD_CONFIG, &payload, "CONFIG")?;
        eprintln!("[motor] handshake complete; controller ready");
        Ok(motor)
    }

    pub fn forward(&mut self, speed: i32) {
        self.drive(speed, speed);
    }

    pub fn backward(&mut self, speed: i32) {
        self.drive(-speed, -speed);
    }

    pub fn left(&mut self, speed: i32) {
        self.drive(-speed, speed);
    }

    pub fn right(&mut self, speed: i32) {
        self.drive(speed, -speed);
    }

    pub fn brake(&mut self) {
        let _ = self.send_frame(CMD_BRAKE, &[BOTH_MOTORS]);
    }

    pub fn standby(&mut self) {
        let _ = self.send_frame(CMD_STOP, &[BOTH_MOTORS]);
    }

    pub fn drive(&mut self, left_speed: i32, right_speed: i32) {
        let left = to_pwm(map_deadzone(left_speed, self.min_speed), self.speed_scale);
        let right = to_pwm(map_deadzone(right_speed, self.min_speed), self.speed_scale);
        let left_bytes = left.to_be_bytes();
        let right_bytes = right.to_be_bytes();
        let payload = [left_bytes[0], left_bytes[1], right_bytes[0], right_bytes[1]];
        eprintln!("[motor] set_speeds requested={left_speed},{right_speed} raw={left},{right}");
        let _ = self.send_frame(CMD_SET_SPEEDS, &payload);
    }

    /// Return `(controller_state, left_rpm, right_rpm)` from the ESP32.
    pub fn status(&mut self) -> io::Result<(u8, i16, i16)> {
        if let Some(port) = &mut self.port {
            port.discard_input()?;
        }
        self.send_frame(CMD_GET_STATUS, &[])?;
        match self.recv_frame(Duration::from_millis(500))? {
            Some((RSP_STATUS, payload)) if payload.len() == 5 => Ok((
                payload[0],
                i16::from_be_bytes([payload[1], payload[2]]),
                i16::from_be_bytes([payload[3], payload[4]]),
            )),
            Some((response, payload)) => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("GET_STATUS: response=0x{response:02X} payload={payload:02X?}"),
            )),
            None => Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "GET_STATUS: controller did not reply within 500ms",
            )),
        }
    }

    fn command_expect_ack(&mut self, cmd: u8, payload: &[u8], label: &str) -> io::Result<()> {
        if let Some(port) = &mut self.port {
            let discarded = port.discard_input()?;
            if discarded != 0 {
                eprintln!("[motor] {label}: discarded {discarded} stale RX byte(s)");
            }
        }
        self.send_frame(cmd, payload)?;
        match self.recv_frame(Duration::from_millis(500))? {
            Some((RSP_ACK, ack_payload)) if ack_payload.first() == Some(&cmd) => {
                eprintln!("[motor] {label}: ACK payload={ack_payload:02X?}");
                Ok(())
            }
            Some((RSP_ACK, ack_payload)) => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{label}: stale/mismatched ACK payload={ack_payload:02X?}, expected command 0x{cmd:02X}"
                ),
            )),
            Some((RSP_NACK, nack_payload)) => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{label}: controller NACK payload={nack_payload:02X?}"),
            )),
            Some((response, response_payload)) => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{label}: unexpected response 0x{response:02X} payload={response_payload:02X?}"
                ),
            )),
            None => Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!("{label}: controller did not reply within 500ms"),
            )),
        }
    }

    fn send_frame(&mut self, cmd: u8, payload: &[u8]) -> io::Result<()> {
        if payload.len() > u8::MAX as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "motor payload too large",
            ));
        }
        let len = payload.len() as u8;
        let mut frame = Vec::with_capacity(payload.len() + 5);
        frame.push(FRAME_SOF0);
        frame.push(FRAME_SOF1);
        frame.push(cmd);
        frame.push(len);
        frame.extend_from_slice(payload);
        frame.push(checksum(cmd, len, payload));

        let port = self
            .port
            .as_mut()
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotConnected, "motor port closed"))?;
        port.write_all_drain(&frame)?;
        eprintln!("[motor] wrote cmd=0x{cmd:02X} len={} ok", payload.len());
        Ok(())
    }

    fn recv_frame(&mut self, timeout: Duration) -> io::Result<Option<(u8, Vec<u8>)>> {
        let Some(port) = &mut self.port else {
            return Ok(None);
        };

        let deadline = std::time::Instant::now() + timeout;
        loop {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                return Ok(None);
            }
            if port.read_byte(remaining)? != Some(FRAME_SOF0) {
                continue;
            }
            if port.read_byte(Duration::from_millis(50))? != Some(FRAME_SOF1) {
                continue;
            }
            let Some(cmd) = port.read_byte(Duration::from_millis(50))? else {
                return Ok(None);
            };
            let Some(len) = port.read_byte(Duration::from_millis(50))? else {
                return Ok(None);
            };
            let mut payload = vec![0u8; len as usize];
            for byte in &mut payload {
                let Some(value) = port.read_byte(Duration::from_millis(50))? else {
                    return Ok(None);
                };
                *byte = value;
            }
            let Some(chk) = port.read_byte(Duration::from_millis(50))? else {
                return Ok(None);
            };
            if checksum(cmd, len, &payload) == chk {
                return Ok(Some((cmd, payload)));
            }
        }
    }
}

impl Drop for Motor {
    fn drop(&mut self) {
        let _ = self.send_frame(CMD_STOP, &[BOTH_MOTORS]);
    }
}

fn checksum(cmd: u8, len: u8, payload: &[u8]) -> u8 {
    payload.iter().fold(cmd ^ len, |acc, b| acc ^ *b)
}

fn put_be16(dst: &mut [u8], value: u16) {
    dst[0] = (value >> 8) as u8;
    dst[1] = value as u8;
}

fn map_deadzone(value: i32, min_speed: i32) -> i32 {
    if value == 0 {
        return 0;
    }
    let sign = value.signum();
    let mag = value.abs().min(100);
    let mapped = min_speed + (mag - 1) * (100 - min_speed) / 99;
    sign * mapped.min(100)
}

fn to_pwm(speed: i32, scale: i32) -> i16 {
    let clamped = speed.clamp(-100, 100);
    (clamped * scale / 100) as i16
}

#[cfg(test)]
mod tests {
    use super::{checksum, map_deadzone, to_pwm};

    #[test]
    fn checksum_matches_protocol() {
        assert_eq!(checksum(0x02, 4, &[0x12, 0x48, 0x4E, 0x20]), 0x32);
    }

    #[test]
    fn maps_deadzone_like_cpp_driver() {
        assert_eq!(map_deadzone(0, 15), 0);
        assert_eq!(map_deadzone(1, 15), 15);
        assert_eq!(map_deadzone(-1, 15), -15);
        assert_eq!(map_deadzone(100, 15), 100);
    }

    #[test]
    fn maps_speed_to_pwm() {
        assert_eq!(to_pwm(100, 150), 150);
        assert_eq!(to_pwm(-100, 150), -150);
        assert_eq!(to_pwm(50, 150), 75);
    }
}
