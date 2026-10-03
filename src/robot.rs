use crate::arm::{Arm, S2_DEPOSIT_RESTORE};
use crate::detector::Detection;
use crate::motor::Motor;
use crate::red_target::{
    detect_red_yuv422p, reaches_stop_geometry, RedObservation, RedThreshold,
    DEFAULT_RED_MINIMUM_PIXELS, DEFAULT_RED_STOP_AREA_RATIO, DEFAULT_RED_STOP_CONFIRM_FRAMES,
    DEFAULT_RED_STOP_HEIGHT_RATIO, DEFAULT_RED_STOP_MIN_AREA_RATIO, DEFAULT_RED_STOP_WIDTH_RATIO,
};
use crate::tpu::{
    AlignedPhysicalFrames, InferTiming, InferenceConfig, PhysicalPixelFormat, YoloModel,
};
use crate::vpss_pipeline::{VpssRgbPipeline, CAMERA_HEIGHT, CAMERA_WIDTH};
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RobotStatus {
    ChaseTennis,
    GrabTennis,
    FindRedContainer,
    ApproachRedContainer,
    ReleaseTennis,
}

#[derive(Clone, Copy, Debug)]
struct RobotState {
    status: RobotStatus,
    area_ratio: f32,
    ball_cx: i32,
    grab_confirm_count: i32,
    holding_ball: bool,
    red_confirm_count: u32,
}

impl Default for RobotState {
    fn default() -> Self {
        Self {
            status: RobotStatus::ChaseTennis,
            area_ratio: 0.0,
            ball_cx: 0,
            grab_confirm_count: 0,
            holding_ball: false,
            red_confirm_count: 0,
        }
    }
}

impl RobotState {
    fn mark_grab_complete(&mut self) {
        self.holding_ball = true;
        self.status = RobotStatus::FindRedContainer;
        self.area_ratio = 0.0;
        self.ball_cx = 0;
        self.grab_confirm_count = 0;
        self.red_confirm_count = 0;
    }

    fn mark_deposit_complete(&mut self) {
        self.holding_ball = false;
        self.status = RobotStatus::ChaseTennis;
        self.area_ratio = 0.0;
        self.ball_cx = 0;
        self.grab_confirm_count = 0;
        self.red_confirm_count = 0;
    }

    fn may_approach_red(&self) -> bool {
        self.holding_ball
    }
}

const REFERENCE_FRAME_WIDTH: f32 = 640.0;
/// Ball area ratio at which the robot should stop and grab.
/// ~0.55 = ball fills the screen top-to-bottom at 640×480.
const GRAB_AREA: f32 = 0.55;
// The original +/-35 px dead band combined with a 120 ms in-place turn made
// the ball jump across the target centre between frames. Accept a wider
// centred region and correct only the portion outside it with a short pulse.
const CENTER_MARGIN: i32 = 55;
/// Offset the "centred" target position to the right of image centre.
/// 0 = dead centre, positive = rightward.  Unit: pixels at 640×480.
const GRAB_CENTER_OFFSET: i32 = 25;
const BALL_TURN_PULSE_MIN_US: u64 = 15_000;
const BALL_TURN_PULSE_MAX_US: u64 = 45_000;
/// Only back up if the ball literally fills nearly the whole frame.
const GRAB_AREA_MAX: f32 = 0.85;
const CHASE_SPEED: i32 = 45;
const TURN_SPEED: i32 = 10;
const IDLE_SPEED: i32 = 12;
// After depositing a ball, rotate slowly enough that the camera can acquire
// the next tennis ball instead of sweeping past it between captured frames.
// Keep IDLE_SPEED unchanged for the existing chase/search behaviours.
const POST_DEPOSIT_SEARCH_SPEED: i32 = 6;
const GRAB_CONFIRM_THRESHOLD: i32 = 2;
const BACKWARD_SPEED: i32 = 16;
const BACKWARD_PULSE_US: u64 = 80_000;
const RED_CRAWL_SPEED: i32 = 6;
// A wide red container often already covers the optical centre even when its
// noisy bounding-box centre is offset. Give it a wider, bbox-aware dead band
// and use short corrections so consecutive frames do not command opposite
// 120 ms turns.
const RED_CENTER_MARGIN: i32 = 70;
const RED_TURN_PULSE_MIN_US: u64 = 25_000;
const RED_TURN_PULSE_MAX_US: u64 = 60_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RedMissionAction {
    Ignore,
    Search,
    Chase(ChaseMotion),
    HoldForConfirmation,
    Deposit,
}

/// Motor decision shared by the tennis hunter and any target follower that
/// must move exactly like the original tennis chase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChaseMotion {
    Forward(i32),
    TurnLeft(u64),
    TurnRight(u64),
}

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
    mut pipeline: VpssRgbPipeline,
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
         BALL_TURN_PULSE=[{BALL_TURN_PULSE_MIN_US}..{BALL_TURN_PULSE_MAX_US}]us \
         RED_STOP={:.0}%x{:.0}% RED_CONFIRM={DEFAULT_RED_STOP_CONFIRM_FRAMES}",
        GRAB_AREA,
        GRAB_AREA_MAX,
        DEFAULT_RED_STOP_WIDTH_RATIO * 100.0,
        DEFAULT_RED_STOP_HEIGHT_RATIO * 100.0,
    );

    while !stop_requested() {
        if let Some(max_frames) = config.max_frames {
            if frame_idx >= max_frames {
                break;
            }
        }

        let frame_start = Instant::now();
        frame_idx += 1;

        let frame = match pipeline.next(2_000) {
            Ok(frame) => {
                camera_errors = 0;
                camera_recoveries = 0;
                frame
            }
            Err(err) => {
                camera_errors += 1;
                eprintln!("[camera-vpss] failed to acquire frame (#{camera_errors}): {err}");
                if camera_errors >= 3 {
                    camera_recoveries += 1;
                    let recovery = if camera_recoveries <= 2 {
                        eprintln!("[camera-vpss] restarting asynchronous pipeline ...");
                        pipeline.restart()
                    } else {
                        eprintln!("[camera-vpss] power-cycling camera ...");
                        pipeline.hard_reset()
                    };
                    match recovery {
                        Ok(()) => {
                            eprintln!("[camera-vpss] recovery ok");
                            camera_errors = 0;
                            if camera_recoveries > 2 {
                                camera_recoveries = 0;
                            }
                        }
                        Err(error) => {
                            eprintln!("[camera-vpss] recovery failed: {error}");
                        }
                    }
                }
                sleep_us(100_000);
                continue;
            }
        };
        let capture_us = frame.camera_request_us as i64;
        let vpss_wall_us = frame.vpss_wall_us as i64;
        let vpss_hardware_us = frame.vpss_hardware_us as i64;
        let sequence = frame.meta.sequence;

        let mut timing = InferTiming::default();
        let handle_start = Instant::now();
        if robot.may_approach_red() {
            handle_red_container(
                frame.yuv,
                frame.yuv_width as usize,
                frame.yuv_height as usize,
                &mut robot,
                &mut motor,
                &mut arm,
            );
        } else {
            let frame_paddrs = [frame.physical_address];
            // SAFETY: VpssRgbPipeline owns the destination ION allocation and
            // keeps it live until this blocking inference and postprocess return.
            let detections = match unsafe {
                model.infer_aligned_physical_timed(
                    AlignedPhysicalFrames {
                        frame_paddrs: &frame_paddrs,
                        pixel_format: PhysicalPixelFormat::RgbPlanar,
                        source_width: CAMERA_WIDTH as i32,
                        source_height: CAMERA_HEIGHT as i32,
                    },
                    config.inference,
                    Some(&mut timing),
                )
            } {
                Ok(detections) => detections,
                Err(err) => {
                    eprintln!("[detect] aligned VPSS inference failed: {err}");
                    motor.standby();
                    sleep_us(100_000);
                    continue;
                }
            };
            handle_detections(
                &detections,
                CAMERA_WIDTH as i32,
                CAMERA_HEIGHT as i32,
                &mut robot,
                &mut motor,
                &mut arm,
            );
        }
        let handle_us = handle_start.elapsed().as_micros() as i64;

        let frame_time = frame_start.elapsed();

        eprintln!(
            "[time] seq={} cap={:.1} vpss={:.1}(hw={:.1}) pre={:.1}(dec={:.1} rsz={:.1}) fwd={:.1} post={:.1} handle={:.1} total={:.1} ms",
            sequence,
            capture_us as f32 / 1000.0,
            vpss_wall_us as f32 / 1000.0,
            vpss_hardware_us as f32 / 1000.0,
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
            "[FPS] {:.2} avg: {:.2} ({:.1}ms) status={:?} holding_ball={} area={:.3} cx={} grab_confirm={} red_confirm={}",
            fps,
            avg_fps,
            frame_time.as_secs_f32() * 1000.0,
            robot.status,
            robot.holding_ball,
            robot.area_ratio,
            robot.ball_cx,
            robot.grab_confirm_count,
            robot.red_confirm_count,
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
        search_for_target(motor);
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
    let chase_motion = chase_motion(ball_cx, area_ratio, image_w);
    let centered = matches!(chase_motion, ChaseMotion::Forward(_));

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
            // Keep Arm::grab() itself unchanged: its calibrated angles and
            // timings are the existing, verified tennis-grab settings. It
            // finishes with the arm lifted and the gripper closed.
            arm.grab();
            robot.mark_grab_complete();
            eprintln!(
                "AKARS_MISSION_TRANSITION from=GrabTennis to=FindRedContainer holding_ball=1"
            );
        }
    } else if area_ratio >= GRAB_AREA && !centered {
        // Big but off-centre — align to centre the ball, keep progress.
        execute_chase_motion(chase_motion, motor);
    } else {
        // Ball is still far — chase.
        robot.grab_confirm_count = 0;
        robot.status = RobotStatus::ChaseTennis;
        execute_chase_motion(chase_motion, motor);
    }
}

fn decide_red_action(
    robot: &mut RobotState,
    observation: Option<RedObservation>,
) -> RedMissionAction {
    // Keep this guard local even though the main loop only enters the red path
    // while carrying a ball: seeing red must never move an empty robot toward
    // the container.
    if !robot.may_approach_red() {
        robot.red_confirm_count = 0;
        return RedMissionAction::Ignore;
    }

    let Some(red) = observation.filter(|red| red.pixels >= DEFAULT_RED_MINIMUM_PIXELS) else {
        robot.status = RobotStatus::FindRedContainer;
        robot.area_ratio = 0.0;
        robot.ball_cx = 0;
        robot.red_confirm_count = 0;
        return RedMissionAction::Search;
    };

    robot.area_ratio = red.area_ratio();
    robot.ball_cx = red.center_x as i32;

    if reaches_stop_geometry(
        red,
        DEFAULT_RED_STOP_AREA_RATIO,
        DEFAULT_RED_STOP_MIN_AREA_RATIO,
        DEFAULT_RED_STOP_WIDTH_RATIO,
        DEFAULT_RED_STOP_HEIGHT_RATIO,
    ) {
        robot.status = RobotStatus::ApproachRedContainer;
        robot.red_confirm_count = robot.red_confirm_count.saturating_add(1);
        if robot.red_confirm_count >= DEFAULT_RED_STOP_CONFIRM_FRAMES {
            robot.status = RobotStatus::ReleaseTennis;
            return RedMissionAction::Deposit;
        }
        return RedMissionAction::HoldForConfirmation;
    }

    robot.status = RobotStatus::ApproachRedContainer;
    robot.red_confirm_count = 0;
    let motion = red_chase_motion(red);
    // Tennis motion stops at the grab distance. The red target must reach the
    // closer 100% x 98% condition, so keep crawling until that is confirmed.
    RedMissionAction::Chase(match motion {
        ChaseMotion::Forward(0) => ChaseMotion::Forward(RED_CRAWL_SPEED),
        motion => motion,
    })
}

fn handle_red_container(
    yuv422p: &[u8],
    width: usize,
    height: usize,
    robot: &mut RobotState,
    motor: &mut Motor,
    arm: &mut Arm,
) {
    let observation = detect_red_yuv422p(yuv422p, width, height, RedThreshold::default());
    let action = decide_red_action(robot, observation);

    if let Some(red) = observation.filter(|red| red.pixels >= DEFAULT_RED_MINIMUM_PIXELS) {
        eprintln!(
            "AKARS_RED_TARGET pixels={} area={:.3} bbox={:.3}x{:.3} cx={} action={:?} holding_ball={}",
            red.pixels,
            red.area_ratio(),
            red.bbox_width_ratio(),
            red.bbox_height_ratio(),
            red.center_x,
            action,
            robot.holding_ball,
        );
    } else {
        eprintln!(
            "AKARS_RED_TARGET pixels=0 action={:?} holding_ball={}",
            action, robot.holding_ball,
        );
    }

    match action {
        RedMissionAction::Ignore => motor.standby(),
        RedMissionAction::Search => search_for_target(motor),
        RedMissionAction::Chase(motion) => execute_chase_motion(motion, motor),
        RedMissionAction::HoldForConfirmation => {
            motor.brake();
            sleep_us(20_000);
            motor.standby();
        }
        RedMissionAction::Deposit => {
            motor.brake();
            sleep_us(20_000);
            motor.standby();
            eprintln!(
                "AKARS_MISSION_TRANSITION from=ApproachRedContainer to=ReleaseTennis holding_ball=1"
            );

            // The five-step grab sequence remains untouched. Deposit uses a
            // separate wider angle and must read back servo 2's actual
            // position before any chassis movement is allowed.
            let actual_angle = match arm.release_for_deposit_verified() {
                Ok(actual_angle) => actual_angle,
                Err(error) => {
                    eprintln!(
                        "AKARS_GRIPPER_RELEASE_FAULT holding_ball=1 chassis_stopped=1 error={error}"
                    );
                    eprintln!(
                        "AKARS_MISSION_HALT reason=gripper_release_unverified action=manual_stop_required"
                    );
                    motor.standby();
                    while !stop_requested() {
                        sleep_us(100_000);
                    }
                    return;
                }
            };
            eprintln!(
                "AKARS_GRIPPER_RELEASE_COMPLETE servo=2 target_angle=200 actual_angle={actual_angle:.1} verified=1 chassis_stopped=1"
            );
            // The ball is already released and verified. Send the relaxed
            // 98-degree restore command, then start searching immediately;
            // do not block chassis motion on a second position read. The
            // gripper and chassis have independent UART controllers, so the
            // restore and slow turn can safely run concurrently.
            arm.set_angle(2, S2_DEPOSIT_RESTORE, 1000);
            eprintln!(
                "AKARS_GRIPPER_RESTORE_COMMAND servo=2 target_angle={S2_DEPOSIT_RESTORE} immediate_turn=1"
            );

            robot.mark_deposit_complete();
            eprintln!("AKARS_MISSION_TRANSITION from=ReleaseTennis to=ChaseTennis holding_ball=0");

            // Servos 0/1 stay in the raised carrying pose. Servo 2 has already
            // returned directly from deposit-open 200 degrees to the relaxed
            // deposit-only 98-degree position. The original 80-degree clamp
            // remains reserved for grabbing or holding a tennis ball.
            eprintln!(
                "AKARS_MISSION_ACTION action=turn_around_and_search direction=right speed={POST_DEPOSIT_SEARCH_SPEED}"
            );
            search_for_target_after_deposit(motor);

            // The asynchronous camera worker kept capturing while the arm was
            // blocked. The next iteration requests the newest sequence, so no
            // synchronous camera re-initialization is needed here.
        }
    }
}

/// Reproduce the original tennis-chase steering decision from target geometry.
/// `area_ratio` is the target bounding-box area divided by the frame area.
pub fn chase_motion(target_cx: i32, area_ratio: f32, image_w: i32) -> ChaseMotion {
    let center = image_w / 2 + scaled_center_margin(image_w, GRAB_CENTER_OFFSET);
    let center_margin = scaled_center_margin(image_w, CENTER_MARGIN);
    let offset = target_cx - center;
    if offset.abs() <= center_margin {
        ChaseMotion::Forward(chase_speed(area_ratio))
    } else if offset < 0 {
        ChaseMotion::TurnLeft(ball_turn_pulse_us(offset, center_margin))
    } else {
        ChaseMotion::TurnRight(ball_turn_pulse_us(offset, center_margin))
    }
}

/// Steer toward the red container without the left/right oscillation caused
/// by applying the small-ball controller to a large, noisy red bounding box.
fn red_chase_motion(red: RedObservation) -> ChaseMotion {
    let image_w = red.frame_width.max(1) as i32;
    let target_cx = red.center_x as i32;
    let center = image_w / 2 + scaled_center_margin(image_w, GRAB_CENTER_OFFSET);
    let bbox_width = red.right.saturating_sub(red.left).saturating_add(1) as i32;
    let center_margin = scaled_center_margin(image_w, RED_CENTER_MARGIN)
        .max(bbox_width / 4)
        .min(image_w / 3);
    let offset = target_cx - center;

    if offset.abs() <= center_margin {
        ChaseMotion::Forward(RED_CRAWL_SPEED)
    } else {
        let excess = offset.abs().saturating_sub(center_margin) as u64;
        let pulse_us = (RED_TURN_PULSE_MIN_US + excess.saturating_mul(300))
            .clamp(RED_TURN_PULSE_MIN_US, RED_TURN_PULSE_MAX_US);
        if offset < 0 {
            ChaseMotion::TurnLeft(pulse_us)
        } else {
            ChaseMotion::TurnRight(pulse_us)
        }
    }
}

pub fn execute_chase_motion(motion: ChaseMotion, motor: &mut Motor) {
    match motion {
        ChaseMotion::Forward(speed) => motor.forward(speed),
        ChaseMotion::TurnLeft(pulse_us) => align_target(-1, pulse_us, motor),
        ChaseMotion::TurnRight(pulse_us) => align_target(1, pulse_us, motor),
    }
}

/// Progressively reduce forward speed as the target gets larger in frame.
/// Uses a quadratic curve: far away stays at full CHASE_SPEED, but speed
/// drops FAST as the ball fills the screen, preventing overshoot.
pub fn chase_speed(area_ratio: f32) -> i32 {
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

fn align_target(offset: i32, pulse_us: u64, motor: &mut Motor) {
    if offset < 0 {
        motor.drive(-TURN_SPEED, TURN_SPEED);
    } else {
        motor.drive(TURN_SPEED, -TURN_SPEED);
    }
    sleep_us(pulse_us);
    motor.standby();
}

/// Use the exact same continuous search motion as the tennis chase loop.
pub fn search_for_target(motor: &mut Motor) {
    motor.drive(IDLE_SPEED, -IDLE_SPEED);
}

/// Slow clockwise search used only immediately after depositing a ball.
fn search_for_target_after_deposit(motor: &mut Motor) {
    motor.drive(POST_DEPOSIT_SEARCH_SPEED, -POST_DEPOSIT_SEARCH_SPEED);
}

fn scaled_center_margin(image_w: i32, base: i32) -> i32 {
    ((base as f32) * image_w as f32 / REFERENCE_FRAME_WIDTH)
        .round()
        .max(1.0) as i32
}

fn ball_turn_pulse_us(offset: i32, center_margin: i32) -> u64 {
    let excess = offset.abs().saturating_sub(center_margin) as u64;
    (BALL_TURN_PULSE_MIN_US + excess.saturating_mul(180))
        .clamp(BALL_TURN_PULSE_MIN_US, BALL_TURN_PULSE_MAX_US)
}

fn sleep_us(us: u64) {
    thread::sleep(Duration::from_micros(us));
}

#[cfg(test)]
mod tests {
    use super::{
        ball_turn_pulse_us, chase_motion, decide_red_action, scaled_center_margin, ChaseMotion,
        RedMissionAction, RobotState, RobotStatus, BALL_TURN_PULSE_MAX_US, BALL_TURN_PULSE_MIN_US,
        DEFAULT_RED_STOP_CONFIRM_FRAMES,
    };
    use crate::red_target::RedObservation;

    fn red_observation(left: usize, top: usize, right: usize, bottom: usize) -> RedObservation {
        let width = 640;
        let height = 480;
        let bbox_width = right - left + 1;
        let bbox_height = bottom - top + 1;
        RedObservation {
            pixels: bbox_width * bbox_height,
            center_x: (left + right) / 2,
            center_y: (top + bottom) / 2,
            left,
            top,
            right,
            bottom,
            frame_width: width,
            frame_height: height,
        }
    }

    #[test]
    fn scales_center_margin() {
        assert_eq!(scaled_center_margin(640, 35), 35);
        assert_eq!(scaled_center_margin(320, 35), 18);
    }

    #[test]
    fn clamps_turn_pulse() {
        assert_eq!(ball_turn_pulse_us(55, 55), BALL_TURN_PULSE_MIN_US);
        assert_eq!(ball_turn_pulse_us(1_000, 55), BALL_TURN_PULSE_MAX_US);
    }

    #[test]
    fn logged_ball_centres_do_not_immediately_reverse() {
        let ChaseMotion::TurnRight(pulse_us) = chase_motion(418, 0.080, 640) else {
            panic!("cx=418 must make a short right correction");
        };
        assert!(pulse_us <= 20_000);
        assert_eq!(chase_motion(303, 0.076, 640), ChaseMotion::Forward(33));
    }

    #[test]
    fn red_is_ignored_when_gripper_is_empty() {
        let mut robot = RobotState::default();
        let full_frame = red_observation(0, 0, 639, 479);

        assert_eq!(
            decide_red_action(&mut robot, Some(full_frame)),
            RedMissionAction::Ignore
        );
        assert_eq!(robot.status, RobotStatus::ChaseTennis);
        assert!(!robot.holding_ball);
        assert_eq!(robot.red_confirm_count, 0);
    }

    #[test]
    fn successful_grab_enables_red_search_and_approach() {
        let mut robot = RobotState::default();
        robot.mark_grab_complete();
        assert!(robot.holding_ball);
        assert_eq!(robot.status, RobotStatus::FindRedContainer);

        assert_eq!(
            decide_red_action(&mut robot, None),
            RedMissionAction::Search
        );
        let far_centered = red_observation(220, 140, 419, 339);
        assert!(matches!(
            decide_red_action(&mut robot, Some(far_centered)),
            RedMissionAction::Chase(ChaseMotion::Forward(_))
        ));
        assert_eq!(robot.status, RobotStatus::ApproachRedContainer);
    }

    #[test]
    fn wide_red_bbox_does_not_oscillate_on_noisy_center() {
        let mut robot = RobotState::default();
        robot.mark_grab_complete();

        // This geometry matches the wide container boxes seen in the field
        // log. Although the bbox centre is right of the old narrow dead band,
        // the container already covers the optical centre and should advance.
        let wide_right = red_observation(160, 140, 639, 339);
        assert_eq!(
            decide_red_action(&mut robot, Some(wide_right)),
            RedMissionAction::Chase(ChaseMotion::Forward(super::RED_CRAWL_SPEED))
        );
    }

    #[test]
    fn narrow_off_axis_red_bbox_uses_short_correction() {
        let mut robot = RobotState::default();
        robot.mark_grab_complete();

        let far_right = red_observation(500, 140, 620, 339);
        let RedMissionAction::Chase(ChaseMotion::TurnRight(pulse_us)) =
            decide_red_action(&mut robot, Some(far_right))
        else {
            panic!("right-side red target must turn right");
        };
        assert!((super::RED_TURN_PULSE_MIN_US..=super::RED_TURN_PULSE_MAX_US).contains(&pulse_us));
    }

    #[test]
    fn close_red_requires_confirmation_before_deposit() {
        let mut robot = RobotState::default();
        robot.mark_grab_complete();
        let close = red_observation(0, 0, 639, 470);

        for expected in 1..DEFAULT_RED_STOP_CONFIRM_FRAMES {
            assert_eq!(
                decide_red_action(&mut robot, Some(close)),
                RedMissionAction::HoldForConfirmation
            );
            assert_eq!(robot.red_confirm_count, expected);
            assert!(robot.holding_ball);
        }
        assert_eq!(
            decide_red_action(&mut robot, Some(close)),
            RedMissionAction::Deposit
        );
        assert_eq!(robot.status, RobotStatus::ReleaseTennis);
        assert!(robot.holding_ball);
    }

    #[test]
    fn deposit_completion_returns_to_ball_chase_and_disables_red() {
        let mut robot = RobotState::default();
        robot.mark_grab_complete();
        robot.mark_deposit_complete();

        assert_eq!(robot.status, RobotStatus::ChaseTennis);
        assert!(!robot.holding_ball);
        assert_eq!(robot.red_confirm_count, 0);
        assert_eq!(
            decide_red_action(&mut robot, Some(red_observation(0, 0, 639, 479))),
            RedMissionAction::Ignore
        );
    }
}
