//! Approach the only red object in a clean scene and stop when it fills the view.
//!
//! This deliberately builds on the existing `UsbCamera` YUV422P capture and
//! `Motor` UART implementation. No second camera stack or motor protocol is
//! introduced.

use akars::camera::UsbCamera;
use akars::motor::{Motor, MotorConfig};
use akars::red_target::{detect_red_yuv422p, RedObservation, RedThreshold};
use akars::robot::{
    chase_motion, chase_speed, execute_chase_motion, install_signal_handlers, search_for_target,
    stop_requested, ChaseMotion,
};
use std::env;
use std::thread;
use std::time::{Duration, Instant};

const DEFAULT_CAMERA: &str = "/dev/cvi-usb-camera0";
const DEFAULT_MOTOR: &str = "/dev/ttyS1";
const DEFAULT_FRAME_WIDTH: usize = 640;
const DEFAULT_FRAME_HEIGHT: usize = 480;
const RED_CRAWL_SPEED: i32 = 6;

#[derive(Clone, Debug)]
struct Config {
    camera: String,
    motor: String,
    dry_run: bool,
    force_forward: bool,
    warmup_frames: u32,
    max_frames: Option<u64>,
    max_seconds: Option<u64>,
    threshold: RedThreshold,
    minimum_pixels: usize,
    stop_area_ratio: f32,
    stop_bbox_min_area_ratio: f32,
    stop_bbox_width_ratio: f32,
    stop_bbox_height_ratio: f32,
    stop_confirm_frames: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            camera: DEFAULT_CAMERA.to_owned(),
            motor: DEFAULT_MOTOR.to_owned(),
            dry_run: false,
            force_forward: false,
            warmup_frames: 10,
            max_frames: None,
            max_seconds: Some(120),
            threshold: RedThreshold::default(),
            minimum_pixels: 200,
            // The final approach should put the container across almost the
            // entire image and about 98% of its height. A 98% pixel-coverage
            // fallback cannot geometrically trigger with a bbox below 98%
            // height, so it cannot bypass the close-range geometry below.
            stop_area_ratio: 0.98,
            stop_bbox_min_area_ratio: 0.55,
            stop_bbox_width_ratio: 1.00,
            stop_bbox_height_ratio: 0.98,
            stop_confirm_frames: 3,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Motion {
    Search,
    Chase(ChaseMotion),
    HoldForConfirmation,
    Reached,
}

#[derive(Debug, Default)]
struct ApproachController {
    close_confirmations: u32,
    reached: bool,
}

impl ApproachController {
    fn decide(&mut self, observation: Option<RedObservation>, config: &Config) -> Motion {
        if self.reached {
            return Motion::Reached;
        }
        let Some(red) = observation.filter(|red| red.pixels >= config.minimum_pixels) else {
            self.close_confirmations = 0;
            return Motion::Search;
        };

        if fills_view(red, config) {
            self.close_confirmations = self.close_confirmations.saturating_add(1);
            if self.close_confirmations >= config.stop_confirm_frames {
                self.reached = true;
                return Motion::Reached;
            }
            return Motion::HoldForConfirmation;
        }
        self.close_confirmations = 0;

        let chase = if config.force_forward {
            // Diagnostic mode: isolate translation polarity from steering.
            // The normal controller never enters this path unless explicitly
            // requested on the command line.
            ChaseMotion::Forward(chase_speed(red.area_ratio()))
        } else {
            chase_motion(
                red.center_x as i32,
                // Red targets are often non-rectangular. Their bounding box
                // reaches the tennis grab threshold before the target really
                // fills the view, which would command speed zero and deadlock
                // below the red stop condition. Use measured red coverage as
                // the progress metric while preserving the original chase
                // steering and speed curve.
                red.area_ratio(),
                red.frame_width as i32,
            )
        };
        // The shared tennis curve intentionally returns zero at 55% target
        // area because that is the tennis grab distance. The red container's
        // requested stop distance is closer (roughly 95% x 90%), so continue
        // crawling until its own stop condition above is satisfied.
        Motion::Chase(match chase {
            ChaseMotion::Forward(0) => ChaseMotion::Forward(RED_CRAWL_SPEED),
            motion => motion,
        })
    }
}

fn fills_view(red: RedObservation, config: &Config) -> bool {
    red.area_ratio() >= config.stop_area_ratio
        || (red.area_ratio() >= config.stop_bbox_min_area_ratio
            && red.bbox_width_ratio() >= config.stop_bbox_width_ratio
            && red.bbox_height_ratio() >= config.stop_bbox_height_ratio)
}

fn main() {
    if let Err(error) = run() {
        eprintln!("red approach failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let config = parse_config(env::args().skip(1))?;
    install_signal_handlers();

    println!("=== Red Object Approach ===");
    println!("Camera : {}", config.camera);
    println!(
        "Motor  : {}{}",
        config.motor,
        if config.dry_run { " (dry-run)" } else { "" }
    );
    println!(
        "Red    : Y>={} U<={} V>={} min_pixels={}",
        config.threshold.y_min,
        config.threshold.u_max,
        config.threshold.v_min,
        config.minimum_pixels,
    );
    println!(
        "Stop   : area>={:.1}% OR (area>={:.1}% AND bbox>={:.1}%x{:.1}%) for {} frames",
        config.stop_area_ratio * 100.0,
        config.stop_bbox_min_area_ratio * 100.0,
        config.stop_bbox_width_ratio * 100.0,
        config.stop_bbox_height_ratio * 100.0,
        config.stop_confirm_frames,
    );

    let mut camera = UsbCamera::open(&config.camera)
        .map_err(|error| format!("open camera {}: {error}", config.camera))?;
    let info = camera.info();
    let width = usize::from(info.width);
    let height = usize::from(info.height);
    if width != DEFAULT_FRAME_WIDTH || height != DEFAULT_FRAME_HEIGHT {
        return Err(format!(
            "red approach currently requires {}x{}, camera reports {}x{}",
            DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT, width, height
        ));
    }
    println!(
        "Camera opened: {}x{} format={} connected={}",
        width, height, info.format, info.connected
    );

    let mut motor = if config.dry_run {
        None
    } else {
        let motor_config = MotorConfig {
            device: config.motor.clone(),
            ..MotorConfig::default()
        };
        let mut motor = Motor::open(&motor_config)
            .map_err(|error| format!("open motor {}: {error}", config.motor))?;
        motor.standby();
        Some(motor)
    };

    for index in 0..config.warmup_frames {
        if stop_requested() {
            stop_motor(motor.as_mut());
            return Ok(());
        }
        camera
            .get_frame()
            .map_err(|error| format!("warm-up frame {}: {error}", index + 1))?;
    }
    println!("Warm-up complete: {} frames", config.warmup_frames);

    let started = Instant::now();
    let mut controller = ApproachController::default();
    let mut frames = 0u64;
    let mut camera_errors = 0u32;
    let mut red_frames = 0u64;
    let mut detection_total = Duration::ZERO;
    let mut outcome = "stopped";

    while !stop_requested() {
        if config.max_frames.is_some_and(|limit| frames >= limit) {
            outcome = "frame_limit";
            break;
        }
        if config
            .max_seconds
            .is_some_and(|limit| started.elapsed() >= Duration::from_secs(limit))
        {
            outcome = "timeout";
            break;
        }

        let frame_start = Instant::now();
        let capture_start = Instant::now();
        let frame = match camera.get_frame() {
            Ok(frame) => {
                camera_errors = 0;
                frame
            }
            Err(error) => {
                camera_errors = camera_errors.saturating_add(1);
                stop_motor(motor.as_mut());
                eprintln!("[camera] frame failed ({camera_errors}/3): {error}");
                if camera_errors >= 3 {
                    camera
                        .re_init()
                        .map_err(|reset_error| format!("camera recovery failed: {reset_error}"))?;
                    camera_errors = 0;
                }
                thread::sleep(Duration::from_millis(100));
                continue;
            }
        };
        let capture_us = capture_start.elapsed().as_micros();
        frames += 1;

        let detection_start = Instant::now();
        let observation = detect_red_yuv422p(
            &frame.pixels,
            usize::from(frame.width),
            usize::from(frame.height),
            config.threshold,
        );
        let detection_time = detection_start.elapsed();
        detection_total += detection_time;
        let detection_us = detection_time.as_micros();
        if observation.is_some_and(|red| red.pixels >= config.minimum_pixels) {
            red_frames += 1;
        }

        let motion = controller.decide(observation, &config);
        let motion_start = Instant::now();
        execute_motion(motion, motor.as_mut());
        let motion_us = motion_start.elapsed().as_micros();
        let total_us = frame_start.elapsed().as_micros();

        if let Some(red) = observation {
            println!(
                "AKARS_RED_FRAME index={} pixels={} area_percent={:.2} center={},{} bbox={},{},{},{} bbox_percent={:.2}x{:.2} action={:?} confirm={}/{} capture_us={} detect_us={} motion_us={} total_us={}",
                frames,
                red.pixels,
                red.area_ratio() * 100.0,
                red.center_x,
                red.center_y,
                red.left,
                red.top,
                red.right,
                red.bottom,
                red.bbox_width_ratio() * 100.0,
                red.bbox_height_ratio() * 100.0,
                motion,
                controller.close_confirmations,
                config.stop_confirm_frames,
                capture_us,
                detection_us,
                motion_us,
                total_us,
            );
        } else {
            println!(
                "AKARS_RED_FRAME index={} pixels=0 action={:?} confirm=0/{} capture_us={} detect_us={} motion_us={} total_us={}",
                frames,
                motion,
                config.stop_confirm_frames,
                capture_us,
                detection_us,
                motion_us,
                total_us,
            );
        }

        if motion == Motion::Reached {
            outcome = "target_fills_view";
            break;
        }
    }

    stop_motor(motor.as_mut());
    let elapsed_us = started.elapsed().as_micros();
    let detection_avg_us = if frames == 0 {
        0
    } else {
        detection_total.as_micros() / u128::from(frames)
    };
    println!(
        "AKARS_RED_RESULT outcome={} frames={} red_frames={} elapsed_us={} detect_avg_us={} dry_run={}",
        outcome,
        frames,
        red_frames,
        elapsed_us,
        detection_avg_us,
        u8::from(config.dry_run),
    );
    Ok(())
}

fn execute_motion(motion: Motion, motor: Option<&mut Motor>) {
    let Some(motor) = motor else {
        return;
    };
    match motion {
        Motion::Chase(chase_motion) => execute_chase_motion(chase_motion, motor),
        Motion::Search => search_for_target(motor),
        Motion::HoldForConfirmation | Motion::Reached => stop_motor(Some(motor)),
    }
}

fn stop_motor(motor: Option<&mut Motor>) {
    if let Some(motor) = motor {
        motor.brake();
        thread::sleep(Duration::from_millis(20));
        motor.standby();
    }
}

fn parse_config(args: impl Iterator<Item = String>) -> Result<Config, String> {
    let mut config = Config::default();
    let mut args = args.peekable();
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            "--camera" => config.camera = take_value(&mut args, "--camera")?,
            "--motor" => config.motor = take_value(&mut args, "--motor")?,
            "--dry-run" => config.dry_run = true,
            "--force-forward" => config.force_forward = true,
            "--warmup" => config.warmup_frames = parse_value(&mut args, "--warmup")?,
            "--max-frames" => config.max_frames = Some(parse_value(&mut args, "--max-frames")?),
            "--max-seconds" => {
                let seconds: u64 = parse_value(&mut args, "--max-seconds")?;
                config.max_seconds = (seconds != 0).then_some(seconds);
            }
            "--y-min" => config.threshold.y_min = parse_value(&mut args, "--y-min")?,
            "--u-max" => config.threshold.u_max = parse_value(&mut args, "--u-max")?,
            "--v-min" => config.threshold.v_min = parse_value(&mut args, "--v-min")?,
            "--min-pixels" => config.minimum_pixels = parse_value(&mut args, "--min-pixels")?,
            "--stop-area-percent" => {
                config.stop_area_ratio = parse_percent(&mut args, "--stop-area-percent")?
            }
            "--stop-bbox-min-area-percent" => {
                config.stop_bbox_min_area_ratio =
                    parse_percent(&mut args, "--stop-bbox-min-area-percent")?
            }
            "--stop-width-percent" => {
                config.stop_bbox_width_ratio = parse_percent(&mut args, "--stop-width-percent")?
            }
            "--stop-height-percent" => {
                config.stop_bbox_height_ratio = parse_percent(&mut args, "--stop-height-percent")?
            }
            "--confirm-frames" => {
                config.stop_confirm_frames = parse_value(&mut args, "--confirm-frames")?
            }
            unknown => return Err(format!("unknown option: {unknown}")),
        }
    }

    if config.stop_confirm_frames == 0 {
        return Err("--confirm-frames must be positive".to_owned());
    }
    if config.minimum_pixels == 0 {
        return Err("--min-pixels must be positive".to_owned());
    }
    Ok(config)
}

fn take_value(
    args: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    option: &str,
) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{option} expects a value"))
}

fn parse_value<T: std::str::FromStr>(
    args: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    option: &str,
) -> Result<T, String> {
    take_value(args, option)?
        .parse()
        .map_err(|_| format!("{option} has an invalid value"))
}

fn parse_percent(
    args: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    option: &str,
) -> Result<f32, String> {
    let value: f32 = parse_value(args, option)?;
    if !value.is_finite() || !(0.0..=100.0).contains(&value) {
        return Err(format!("{option} must be between 0 and 100"));
    }
    Ok(value / 100.0)
}

fn print_usage() {
    eprintln!(
        "Usage: red_cloth_follow [OPTIONS]\n\
         \n\
         Approach the only red target and stop after it fills the camera view.\n\
         \n\
         Hardware:\n\
           --camera DEV               Camera device (default: /dev/cvi-usb-camera0)\n\
           --motor DEV                Motor UART (default: /dev/ttyS1)\n\
           --dry-run                  Detect and log without opening the motor\n\
           --force-forward            Diagnostic: disable steering and test translation only\n\
         Detection:\n\
           --y-min N                  Minimum luma (default: 24)\n\
           --u-max N                  Maximum Cb (default: 115)\n\
           --v-min N                  Minimum Cr (default: 155)\n\
           --min-pixels N             Ignore smaller red regions (default: 200)\n\
         Stop condition:\n\
           --stop-area-percent P      Red pixel coverage (default: 98)\n\
           --stop-bbox-min-area-percent P  Minimum red area for bbox stop (default: 55)\n\
           --stop-width-percent P     Red bbox width coverage (default: 100)\n\
           --stop-height-percent P    Red bbox height coverage (default: 98)\n\
           --confirm-frames N         Consecutive close frames (default: 3)\n\
         Limits:\n\
           --warmup N                 Discard initial frames (default: 10)\n\
           --max-frames N             Stop after N processed frames\n\
           --max-seconds N            Safety timeout (default: 120, 0 disables)"
    );
}

#[cfg(test)]
mod tests {
    use super::{ApproachController, Config, Motion};
    use akars::red_target::RedObservation;
    use akars::robot::ChaseMotion;

    fn observation(center_x: usize, area_ratio: f32, bbox_ratio: f32) -> RedObservation {
        observation_with_bbox(center_x, area_ratio, bbox_ratio, bbox_ratio)
    }

    fn observation_with_bbox(
        center_x: usize,
        area_ratio: f32,
        bbox_width_ratio: f32,
        bbox_height_ratio: f32,
    ) -> RedObservation {
        let width = 100;
        let height = 100;
        let bbox_width = (bbox_width_ratio * width as f32) as usize;
        let bbox_height = (bbox_height_ratio * height as f32) as usize;
        RedObservation {
            pixels: (area_ratio * (width * height) as f32) as usize,
            center_x,
            center_y: 50,
            left: 0,
            top: 0,
            right: bbox_width.saturating_sub(1),
            bottom: bbox_height.saturating_sub(1),
            frame_width: width,
            frame_height: height,
        }
    }

    #[test]
    fn approaches_a_centered_target_and_slows_down() {
        let mut config = Config::default();
        config.minimum_pixels = 1;
        let mut controller = ApproachController::default();
        let far = controller.decide(Some(observation(50, 0.05, 0.2)), &config);
        let near = controller.decide(Some(observation(50, 0.50, 0.8)), &config);
        let (
            Motion::Chase(ChaseMotion::Forward(far_speed)),
            Motion::Chase(ChaseMotion::Forward(near_speed)),
        ) = (far, near)
        else {
            panic!("centered targets should move forward");
        };
        assert!(far_speed > near_speed);
        assert_eq!(near_speed, 6);
    }

    #[test]
    fn large_bbox_does_not_stall_below_red_stop_threshold() {
        let mut config = Config::default();
        config.minimum_pixels = 1;
        let mut controller = ApproachController::default();
        // Matches the real stalled geometry: the rectangular extent is over
        // the tennis 55% grab threshold, but actual red coverage is only 43%
        // and the red-object stop condition is not yet satisfied.
        let motion = controller.decide(
            Some(observation_with_bbox(50, 0.43, 0.91, 0.61)),
            &config,
        );
        assert_eq!(motion, Motion::Chase(ChaseMotion::Forward(6)));
    }

    #[test]
    fn turns_toward_an_off_center_target() {
        let mut config = Config::default();
        config.minimum_pixels = 1;
        let mut controller = ApproachController::default();
        assert!(matches!(
            controller.decide(Some(observation(20, 0.05, 0.2)), &config),
            Motion::Chase(ChaseMotion::TurnLeft(_))
        ));
        assert!(matches!(
            controller.decide(Some(observation(80, 0.05, 0.2)), &config),
            Motion::Chase(ChaseMotion::TurnRight(_))
        ));
    }

    #[test]
    fn requires_consecutive_full_view_frames_and_latches_stop() {
        let mut config = Config::default();
        config.minimum_pixels = 1;
        config.stop_confirm_frames = 3;
        let close = observation_with_bbox(50, 0.75, 1.00, 0.99);
        let mut controller = ApproachController::default();
        assert_eq!(
            controller.decide(Some(close), &config),
            Motion::HoldForConfirmation
        );
        assert_eq!(
            controller.decide(Some(close), &config),
            Motion::HoldForConfirmation
        );
        assert_eq!(controller.decide(Some(close), &config), Motion::Reached);
        assert_eq!(controller.decide(None, &config), Motion::Reached);
    }

    #[test]
    fn sparse_red_outliers_cannot_trigger_bbox_stop() {
        let mut config = Config::default();
        config.minimum_pixels = 1;
        let sparse = observation(50, 0.01, 1.0);
        let mut controller = ApproachController::default();
        assert!(matches!(
            controller.decide(Some(sparse), &config),
            Motion::Chase(_)
        ));
        assert_eq!(controller.close_confirmations, 0);
    }

    #[test]
    fn actual_container_close_geometry_triggers_stop_but_distant_geometry_does_not() {
        let mut config = Config::default();
        config.minimum_pixels = 1;
        config.stop_confirm_frames = 1;
        let mut controller = ApproachController::default();

        let distant = observation_with_bbox(50, 0.164, 0.50, 0.40);
        assert!(matches!(
            controller.decide(Some(distant), &config),
            Motion::Chase(_)
        ));

        let close = observation_with_bbox(50, 0.85, 1.00, 0.98);
        assert_eq!(controller.decide(Some(close), &config), Motion::Reached);
    }

    #[test]
    fn old_seventy_five_percent_height_is_not_close_enough() {
        let mut config = Config::default();
        config.minimum_pixels = 1;
        config.stop_confirm_frames = 1;
        let mut controller = ApproachController::default();
        let old_stop_geometry = observation_with_bbox(50, 0.62, 1.00, 0.75);
        assert_eq!(
            controller.decide(Some(old_stop_geometry), &config),
            Motion::Chase(ChaseMotion::Forward(6))
        );
    }

    #[test]
    fn ninety_percent_height_is_not_close_enough() {
        let mut config = Config::default();
        config.minimum_pixels = 1;
        config.stop_confirm_frames = 1;
        let mut controller = ApproachController::default();
        let old_stop_geometry = observation_with_bbox(50, 0.78, 1.00, 0.90);
        assert_eq!(
            controller.decide(Some(old_stop_geometry), &config),
            Motion::Chase(ChaseMotion::Forward(6))
        );
    }

    #[test]
    fn ninety_five_percent_height_is_not_close_enough() {
        let mut config = Config::default();
        config.minimum_pixels = 1;
        config.stop_confirm_frames = 1;
        let mut controller = ApproachController::default();
        let old_stop_geometry = observation_with_bbox(50, 0.85, 1.00, 0.95);
        assert_eq!(
            controller.decide(Some(old_stop_geometry), &config),
            Motion::Chase(ChaseMotion::Forward(6))
        );
    }

    #[test]
    fn ninety_nine_percent_width_is_not_close_enough() {
        let mut config = Config::default();
        config.minimum_pixels = 1;
        config.stop_confirm_frames = 1;
        let mut controller = ApproachController::default();
        let almost_full_width = observation_with_bbox(50, 0.85, 0.99, 1.00);
        assert_eq!(
            controller.decide(Some(almost_full_width), &config),
            Motion::Chase(ChaseMotion::Forward(6))
        );
    }

    #[test]
    fn searches_when_no_valid_red_target_exists() {
        let config = Config::default();
        let mut controller = ApproachController::default();
        assert_eq!(controller.decide(None, &config), Motion::Search);
    }
}
