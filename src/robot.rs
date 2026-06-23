use crate::arm::Arm;
use crate::camera::UsbCamera;
use crate::detector::Detection;
use crate::motor::Motor;
use crate::tpu::{InferTiming, InferenceConfig, YoloModel};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

static STOP_REQUESTED: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Debug)]
pub struct RobotConfig {
    pub inference: InferenceConfig,
    pub max_frames: Option<u64>,
}

impl Default for RobotConfig {
    fn default() -> Self {
        Self {
            inference: InferenceConfig::default(),
            max_frames: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum RobotStatus {
    ChaseTennis,
    GrabTennis,
}

#[derive(Clone, Copy, Debug)]
struct RobotState {
    status: RobotStatus,
    area_ratio: f32,
    ball_cx: i32,
    grab_confirm_count: i32,
}

impl Default for RobotState {
    fn default() -> Self {
        Self {
            status: RobotStatus::ChaseTennis,
            area_ratio: 0.0,
            ball_cx: 0,
            grab_confirm_count: 0,
        }
    }
}

const REFERENCE_FRAME_WIDTH: f32 = 640.0;
const GRAB_AREA: f32 = 0.40;
const CENTER_MARGIN: i32 = 35;
const K_TURN_PULSE: f32 = 5000.0;
const TURN_PULSE_MAX_US: u64 = 500_000;
const GRAB_AREA_MAX: f32 = 0.55;
const CHASE_SPEED: i32 = 56;
const TURN_SPEED: i32 = 18;
const IDLE_SPEED: i32 = 18;
const TURN_PULSE_MIN_US: u64 = 75_000;
const GRAB_CONFIRM_THRESHOLD: i32 = 5;
const BACKWARD_SPEED: i32 = 18;
const BACKWARD_PULSE_US: u64 = 200_000;
const GRAB_LEFT_TURN_SPEED: i32 = 18;
const GRAB_LEFT_TURN_US: u64 = 150_000;
const GRAB_LEFT_TURN_COUNT: i32 = 4;

pub fn install_signal_handlers() {
    unsafe {
        crate::linux::signal(crate::linux::SIGINT, signal_handler);
        crate::linux::signal(crate::linux::SIGTERM, signal_handler);
    }
}

extern "C" fn signal_handler(_: i32) {
    STOP_REQUESTED.store(true, Ordering::SeqCst);
}

pub fn stop_requested() -> bool {
    STOP_REQUESTED.load(Ordering::SeqCst)
}

pub fn run_tennis_hunter(
    mut camera: UsbCamera,
    mut model: YoloModel,
    mut motor: Motor,
    mut arm: Arm,
    config: RobotConfig,
) {
    let mut robot = RobotState::default();
    let mut frame_idx = 0u64;
    let mut total_time = Duration::ZERO;
    let mut frame_count = 0u64;

    arm.grab_pos();

    while !stop_requested() {
        if let Some(max_frames) = config.max_frames {
            if frame_idx >= max_frames {
                break;
            }
        }

        let frame_start = Instant::now();
        frame_idx += 1;

        let capture_start = Instant::now();
        let frame = match camera.get_frame() {
            Ok(frame) => frame,
            Err(err) => {
                eprintln!("[camera] failed to get frame: {err}");
                sleep_us(100_000);
                continue;
            }
        };
        let capture_us = capture_start.elapsed().as_micros() as i64;

        let mut timing = InferTiming::default();
        let detections = match model.infer_timed(&frame, config.inference, Some(&mut timing)) {
            Ok(detections) => detections,
            Err(err) => {
                eprintln!("[detect] inference failed: {err}");
                sleep_us(100_000);
                continue;
            }
        };

        eprintln!(
            "[time] capture={:.1} pre={:.1}(dec={:.1} rsz={:.1}) fwd={:.1} post={:.1} ms",
            capture_us as f32 / 1000.0,
            timing.preprocess_us as f32 / 1000.0,
            timing.decode_us as f32 / 1000.0,
            timing.resize_us as f32 / 1000.0,
            timing.forward_us as f32 / 1000.0,
            timing.postprocess_us as f32 / 1000.0,
        );

        // handle_detections(
        //     &detections,
        //     frame.width as i32,
        //     frame.height as i32,
        //     &mut robot,
        //     &mut motor,
        //     &mut arm,
        // );

        let frame_time = frame_start.elapsed();
        total_time += frame_time;
        frame_count += 1;
        let fps = if frame_time.as_secs_f32() > 0.0 {
            1.0 / frame_time.as_secs_f32()
        } else {
            0.0
        };
        let avg_fps = if total_time.as_secs_f32() > 0.0 {
            frame_count as f32 / total_time.as_secs_f32()
        } else {
            0.0
        };
        println!(
            "[FPS] {:.2} avg: {:.2} ({:.1}ms)",
            fps,
            avg_fps,
            frame_time.as_secs_f32() * 1000.0
        );
    }

    motor.standby();
}

fn handle_detections(
    detections: &[Detection],
    image_w: i32,
    image_h: i32,
    robot: &mut RobotState,
    motor: &mut Motor,
    arm: &mut Arm,
) {
    if detections.is_empty() {
        eprintln!("[detect] no ball detected, searching");
        robot.grab_confirm_count = 0;
        robot.status = RobotStatus::ChaseTennis;
        motor.drive(IDLE_SPEED, -IDLE_SPEED);
        return;
    }

    let best = detections
        .iter()
        .max_by(|a, b| {
            let area_a = a.bbox.w * a.bbox.h;
            let area_b = b.bbox.w * b.bbox.h;
            area_a.total_cmp(&area_b)
        })
        .expect("non-empty detections");

    let image_area = (image_w.max(1) * image_h.max(1)) as f32;
    let area_ratio = (best.bbox.w * best.bbox.h) / image_area;
    let ball_cx = best.bbox.x as i32;
    let center = image_w / 2;
    let center_margin = scaled_center_margin(image_w);
    let offset = ball_cx - center;
    let centered = offset.abs() <= center_margin;
    let pulse_us = turn_pulse_us(area_ratio);

    robot.area_ratio = area_ratio;
    robot.ball_cx = ball_cx;

    eprintln!(
        "[detect] area={area_ratio:.3} cx={ball_cx} conf={:.3} centered={centered}",
        best.score
    );

    if area_ratio >= GRAB_AREA && centered {
        robot.status = RobotStatus::GrabTennis;
        robot.grab_confirm_count += 1;

        if area_ratio >= GRAB_AREA_MAX {
            eprintln!("[grab] too close, backing up");
            motor.backward(BACKWARD_SPEED);
            sleep_us(BACKWARD_PULSE_US);
            motor.standby();
        } else {
            motor.standby();
        }

        if robot.grab_confirm_count >= GRAB_CONFIRM_THRESHOLD {
            if area_ratio >= GRAB_AREA_MAX {
                motor.backward(BACKWARD_SPEED);
                sleep_us(BACKWARD_PULSE_US);
                motor.standby();
            }

            for _ in 0..GRAB_LEFT_TURN_COUNT {
                motor.drive(-GRAB_LEFT_TURN_SPEED, GRAB_LEFT_TURN_SPEED);
                sleep_us(GRAB_LEFT_TURN_US);
                motor.standby();
                sleep_us(100_000);
            }

            arm.grab();
            sleep_us(2_000_000);
            arm.release();
            sleep_us(1_000_000);
            arm.grab_pos();
            sleep_us(1_000_000);

            robot.grab_confirm_count = 0;
            robot.status = RobotStatus::ChaseTennis;
        }
    } else if area_ratio >= GRAB_AREA && !centered {
        robot.grab_confirm_count = 0;
        align(offset, pulse_us, motor);
    } else {
        robot.grab_confirm_count = 0;
        robot.status = RobotStatus::ChaseTennis;
        if centered {
            motor.forward(CHASE_SPEED);
        } else {
            align(offset, pulse_us, motor);
        }
    }
}

fn align(offset: i32, pulse_us: u64, motor: &mut Motor) {
    if offset < 0 {
        motor.drive(-TURN_SPEED, TURN_SPEED);
    } else {
        motor.drive(TURN_SPEED, -TURN_SPEED);
    }
    sleep_us(pulse_us);
    motor.standby();
}

fn scaled_center_margin(image_w: i32) -> i32 {
    ((CENTER_MARGIN as f32) * image_w as f32 / REFERENCE_FRAME_WIDTH)
        .round()
        .max(1.0) as i32
}

fn turn_pulse_us(area_ratio: f32) -> u64 {
    ((K_TURN_PULSE * area_ratio * 1000.0) as u64).clamp(TURN_PULSE_MIN_US, TURN_PULSE_MAX_US)
}

fn sleep_us(us: u64) {
    thread::sleep(Duration::from_micros(us));
}

#[cfg(test)]
mod tests {
    use super::{scaled_center_margin, turn_pulse_us, TURN_PULSE_MAX_US, TURN_PULSE_MIN_US};

    #[test]
    fn scales_center_margin() {
        assert_eq!(scaled_center_margin(640), 35);
        assert_eq!(scaled_center_margin(320), 18);
    }

    #[test]
    fn clamps_turn_pulse() {
        assert_eq!(turn_pulse_us(0.0), TURN_PULSE_MIN_US);
        assert_eq!(turn_pulse_us(100.0), TURN_PULSE_MAX_US);
    }
}
