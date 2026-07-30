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
/// Ball area ratio at which the robot should stop and grab.
/// ~0.55 = ball fills the screen top-to-bottom at 640×480.
const GRAB_AREA: f32 = 0.55;
const CENTER_MARGIN: i32 = 35;
/// Offset the "centred" target position to the right of image centre.
/// 0 = dead centre, positive = rightward.  Unit: pixels at 640×480.
const GRAB_CENTER_OFFSET: i32 = 25;
const K_TURN_PULSE: f32 = 2500.0;
const TURN_PULSE_MAX_US: u64 = 120_000;
/// Only back up if the ball literally fills nearly the whole frame.
const GRAB_AREA_MAX: f32 = 0.85;
const CHASE_SPEED: i32 = 45;
const TURN_SPEED: i32 = 10;
const IDLE_SPEED: i32 = 12;
const TURN_PULSE_MIN_US: u64 = 25_000;
const GRAB_CONFIRM_THRESHOLD: i32 = 2;
const BACKWARD_SPEED: i32 = 16;
const BACKWARD_PULSE_US: u64 = 80_000;

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
    let mut camera_errors = 0u32;
    // Recovery tier: 0=not tried yet, 1=re_init tried, 2=reopen tried,
    // 3=hard_reset tried.
    let mut camera_recoveries = 0u32;

    arm.restore_torque(0);
    sleep_us(50_000);
    arm.restore_torque(1);
    sleep_us(50_000);
    arm.restore_torque(2);
    sleep_us(50_000);
    arm.grab_pos();

    eprintln!(
        "[cfg] GRAB_AREA={:.3} GRAB_AREA_MAX={:.3} \
         CHASE_SPEED={CHASE_SPEED} CONFIRM={GRAB_CONFIRM_THRESHOLD} \
         IDLE={IDLE_SPEED} CENTER={CENTER_MARGIN}±{GRAB_CENTER_OFFSET} \
         TURN_PULSE=[{TURN_PULSE_MIN_US}..{TURN_PULSE_MAX_US}]us",
        GRAB_AREA, GRAB_AREA_MAX,
    );

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
            Ok(frame) => {
                camera_errors = 0;
                camera_recoveries = 0;
                frame
            }
            Err(err) => {
                camera_errors += 1;
                eprintln!("[camera] failed to get frame (#{camera_errors}): {err}");
                if camera_errors >= 3 {
                    camera_recoveries += 1;
                    if camera_recoveries == 1 {
                        // Tier 1: INIT resets session and re-initializes.
                        eprintln!(
                            "[camera] {} consecutive errors, re-initializing ...",
                            camera_errors
                        );
                        if let Err(e) = camera.re_init() {
                            eprintln!("[camera] re-init failed: {e}");
                            sleep_us(500_000);
                        } else {
                            eprintln!("[camera] re-init ok");
                            camera_errors = 0;
                        }
                    } else if camera_recoveries == 2 {
                        // Tier 2: close fd + reopen + INIT (close now clears session).
                        eprintln!("[camera] re-init didn't help, closing and reopening device ...");
                        match camera.reopen() {
                            Ok(()) => {
                                eprintln!("[camera] reopen ok");
                                camera_errors = 0;
                            }
                            Err(e) => {
                                eprintln!("[camera] reopen failed: {e}");
                                sleep_us(500_000);
                            }
                        }
                    } else {
                        // Tier 3: VBUS power-cycle (~2.5 s), the strongest recovery.
                        eprintln!("[camera] reopen didn't help, power-cycling camera VBUS ...");
                        match camera.hard_reset() {
                            Ok(()) => {
                                eprintln!("[camera] hard-reset ok");
                                camera_errors = 0;
                                camera_recoveries = 0;
                            }
                            Err(e) => {
                                eprintln!("[camera] hard-reset also failed: {e}");
                                sleep_us(1_000_000);
                            }
                        }
                    }
                }
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

        let handle_start = Instant::now();
        handle_detections(
            &detections,
            frame.width as i32,
            frame.height as i32,
            &mut robot,
            &mut motor,
            &mut arm,
            &mut camera,
        );
        let handle_us = handle_start.elapsed().as_micros() as i64;

        let frame_time = frame_start.elapsed();

        eprintln!(
            "[time] cap={:.1} pre={:.1}(dec={:.1} rsz={:.1}) fwd={:.1} post={:.1} handle={:.1} total={:.1} ms",
            capture_us as f32 / 1000.0,
            timing.preprocess_us as f32 / 1000.0,
            timing.decode_us as f32 / 1000.0,
            timing.resize_us as f32 / 1000.0,
            timing.forward_us as f32 / 1000.0,
            timing.postprocess_us as f32 / 1000.0,
            handle_us as f32 / 1000.0,
            frame_time.as_secs_f32() * 1000.0,
        );
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
        eprintln!(
            "[FPS] {:.2} avg: {:.2} ({:.1}ms) status={:?} area={:.3} cx={} confirm={}",
            fps,
            avg_fps,
            frame_time.as_secs_f32() * 1000.0,
            robot.status,
            robot.area_ratio,
            robot.ball_cx,
            robot.grab_confirm_count,
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
    camera: &mut UsbCamera,
) {
    if detections.is_empty() {
        eprintln!("[detect] no ball detected, searching");
        // Ball was recently close (area > 30%) and now disappeared —
        // likely blocking the camera.  Back up further to get a clear view.
        let was_close = robot.area_ratio >= 0.30 || robot.grab_confirm_count > 0;
        if was_close {
            eprintln!(
                "[detect] ball disappeared at area={:.3}, backing up to re-detect",
                robot.area_ratio
            );
            motor.backward(BACKWARD_SPEED);
            sleep_us(BACKWARD_PULSE_US * 8);
            motor.standby();
        }
        robot.grab_confirm_count = 0;
        robot.status = RobotStatus::ChaseTennis;
        // Continuous clockwise rotation while searching — no stop.
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
    let center = image_w / 2 + scaled_center_margin(image_w, GRAB_CENTER_OFFSET);
    let center_margin = scaled_center_margin(image_w, CENTER_MARGIN);
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
        // ── Ball fills the screen: stop and grab ──
        robot.status = RobotStatus::GrabTennis;
        robot.grab_confirm_count += 1;

        if area_ratio >= GRAB_AREA_MAX {
            // Extremely close — gentle nudge back, then grab.
            eprintln!("[grab] too close (area={area_ratio:.3}), nudging back");
            motor.backward(BACKWARD_SPEED);
            sleep_us(BACKWARD_PULSE_US / 2);
            motor.standby();
        } else {
            motor.standby();
        }

        if robot.grab_confirm_count >= GRAB_CONFIRM_THRESHOLD {
            if area_ratio >= GRAB_AREA_MAX {
                motor.backward(BACKWARD_SPEED);
                sleep_us(BACKWARD_PULSE_US / 2);
                motor.standby();
            }

            eprintln!("[grab] executing grab sequence");
            arm.grab();
            arm.release();
            arm.grab_pos();

            robot.grab_confirm_count = 0;
            robot.status = RobotStatus::ChaseTennis;

            // Grab took ~11 s — the camera pipeline likely timed out.
            // Reset it so the next capture doesn't get EIO.
            if let Err(e) = camera.re_init() {
                eprintln!("[camera] re-init after grab failed: {e}");
            }
        }
    } else if area_ratio >= GRAB_AREA && !centered {
        // Big but off-centre — align to centre the ball, keep progress.
        align(offset, pulse_us, motor);
    } else {
        // Ball is still far — chase.
        robot.grab_confirm_count = 0;
        robot.status = RobotStatus::ChaseTennis;
        if centered {
            motor.forward(chase_speed(area_ratio));
        } else {
            align(offset, pulse_us, motor);
        }
    }
}

/// Progressively reduce forward speed as the target gets larger in frame.
/// Uses a quadratic curve: far away stays at full CHASE_SPEED, but speed
/// drops FAST as the ball fills the screen, preventing overshoot.
fn chase_speed(area_ratio: f32) -> i32 {
    if area_ratio >= GRAB_AREA {
        return 0; // close enough — stop
    }
    // Quadratic falloff from 0 → GRAB_AREA.  Even at small area ratios
    // (e.g. 3 %) the speed is already reduced, preventing over-aggressive
    // charging when the ball is first spotted.
    //
    //  area=0.03 → 38   area=0.10 → 27   area=0.30 → 8   area=0.50 → 1
    let t = area_ratio / GRAB_AREA;
    let speed = CHASE_SPEED as f32 * (1.0 - t) * (1.0 - t);
    speed.max(6.0).round() as i32
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

fn scaled_center_margin(image_w: i32, base: i32) -> i32 {
    ((base as f32) * image_w as f32 / REFERENCE_FRAME_WIDTH)
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
        assert_eq!(scaled_center_margin(640, 35), 35);
        assert_eq!(scaled_center_margin(320, 35), 18);
    }

    #[test]
    fn clamps_turn_pulse() {
        assert_eq!(turn_pulse_us(0.0), TURN_PULSE_MIN_US);
        assert_eq!(turn_pulse_us(100.0), TURN_PULSE_MAX_US);
    }
}
