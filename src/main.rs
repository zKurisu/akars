use akars::arm::Arm;
use akars::camera::UsbCamera;
use akars::motor::{Motor, MotorConfig};
use akars::robot::{install_signal_handlers, run_tennis_hunter, RobotConfig};
use akars::tpu::{open_model, InferenceConfig, PhysicalPixelFormat};
use akars::vpss_pipeline::VpssRgbPipeline;
use akars::web::{serve, WebConfig};
use std::env;
use std::net::SocketAddr;
use std::path::PathBuf;

#[derive(Debug)]
struct Cli {
    model: PathBuf,
    camera: String,
    vpss: String,
    motor: String,
    arm: String,
    frames: Option<u64>,
    max_deposits: Option<u32>,
    classes: i32,
    conf: f32,
    iou: f32,
    tpu_debug: bool,
}

/// Arguments for the standalone TPU inference test: run a model on a single
/// image file and write an annotated result image.
#[derive(Debug)]
struct DetectCli {
    model: PathBuf,
    input: PathBuf,
    output: PathBuf,
    classes: i32,
    conf: f32,
    iou: f32,
}

/// Arguments for the camera capture test: grab a single frame and save it,
/// discarding the first few frames so the sensor's exposure/white-balance can
/// settle (warm up).
#[derive(Debug)]
struct CaptureCli {
    camera: String,
    output: PathBuf,
    warmup: u32,
}

#[derive(Debug)]
enum Command {
    Hunt(Cli),
    Serve(WebConfig),
    Detect(DetectCli),
    Capture(CaptureCli),
}

impl Default for Cli {
    fn default() -> Self {
        Self {
            model: PathBuf::new(),
            camera: "/dev/cvi-usb-camera0".to_string(),
            vpss: "/dev/cvi-vpss0".to_string(),
            motor: "/dev/ttyS1".to_string(),
            arm: "/dev/ttyS2".to_string(),
            frames: None,
            max_deposits: None,
            classes: 1,
            conf: 0.5,
            iou: 0.5,
            tpu_debug: false,
        }
    }
}

impl Default for DetectCli {
    fn default() -> Self {
        Self {
            model: PathBuf::new(),
            input: PathBuf::new(),
            output: PathBuf::from("detect_out.jpg"),
            classes: 1,
            conf: 0.5,
            iou: 0.5,
        }
    }
}

impl Default for CaptureCli {
    fn default() -> Self {
        Self {
            camera: "/dev/cvi-usb-camera0".to_string(),
            output: PathBuf::from("capture.jpg"),
            warmup: 30,
        }
    }
}

fn main() {
    let command = match parse_cli(env::args().skip(1)) {
        Ok(command) => command,
        Err(message) => {
            eprintln!("{message}");
            print_usage();
            std::process::exit(2);
        }
    };

    match command {
        Command::Hunt(cli) => {
            install_signal_handlers();
            run_hunt(cli);
        }
        Command::Serve(config) => {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("failed to build tokio runtime");
            if let Err(err) = runtime.block_on(serve(config)) {
                eprintln!("[web] server failed: {err}");
                std::process::exit(1);
            }
        }
        Command::Detect(cli) => run_detect(cli),
        Command::Capture(cli) => run_capture(cli),
    }
}

fn run_hunt(cli: Cli) {
    let model = match open_model(&cli.model) {
        Ok(model) => model,
        Err(err) => {
            eprintln!("[tpu] failed to open model {}: {err}", cli.model.display());
            std::process::exit(1);
        }
    };
    let contract = model.input_contract();
    eprintln!("{}", contract.summary());
    let contract_ok = contract.format == 7
        && contract.pixel_format == PhysicalPixelFormat::RgbPlanar as i32
        && (contract.qscale - 1.0).abs() <= 1.0e-6
        && contract.zero_point == 0
        && contract.mean.iter().all(|value| value.abs() <= 1.0e-6)
        && contract
            .scale
            .iter()
            .all(|value| (*value - 1.0).abs() <= 1.0e-6);
    if !contract_ok {
        eprintln!(
            "[tpu] model is incompatible with VPSS input; expected UINT8 RGB_PLANAR, qscale=1, zero_point=0, mean=0, scale=1"
        );
        std::process::exit(1);
    }
    let (input_w, input_h) = model.input_dimensions();
    let output_w = u32::try_from(input_w).unwrap_or_else(|_| {
        eprintln!("[tpu] invalid model input width: {input_w}");
        std::process::exit(1);
    });
    let output_h = u32::try_from(input_h).unwrap_or_else(|_| {
        eprintln!("[tpu] invalid model input height: {input_h}");
        std::process::exit(1);
    });
    let pipeline = match VpssRgbPipeline::open(&cli.camera, &cli.vpss, output_w, output_h) {
        Ok(pipeline) => pipeline,
        Err(err) => {
            eprintln!(
                "[camera-vpss] failed to open camera={} vpss={}: {err}",
                cli.camera, cli.vpss
            );
            std::process::exit(1);
        }
    };
    eprintln!(
        "[camera-vpss] opened camera={} vpss={} output={}x{}",
        cli.camera, cli.vpss, output_w, output_h
    );
    eprintln!(
        "[dbg] model and VPSS opened, opening motor {} ...",
        cli.motor
    );

    let motor_config = MotorConfig {
        device: cli.motor.clone(),
        ..MotorConfig::default()
    };
    let motor = match Motor::open(&motor_config) {
        Ok(motor) => motor,
        Err(err) => {
            eprintln!("[motor] failed to open {}: {err}", cli.motor);
            std::process::exit(1);
        }
    };
    eprintln!("[dbg] motor opened, opening arm {} ...", cli.arm);

    let arm = match Arm::open(&cli.arm, 115200) {
        Ok(arm) => arm,
        Err(err) => {
            eprintln!("[arm] failed to open {}: {err}", cli.arm);
            std::process::exit(1);
        }
    };
    eprintln!("[dbg] arm opened, entering tennis hunter ...");

    let config = RobotConfig {
        inference: InferenceConfig {
            classes_num: cli.classes,
            confidence_threshold: cli.conf,
            iou_threshold: cli.iou,
            debug_logging: cli.tpu_debug,
        },
        max_frames: cli.frames,
        max_deposits: cli.max_deposits,
    };

    run_tennis_hunter(pipeline, model, motor, arm, config);
}

fn run_detect(cli: DetectCli) {
    eprintln!("[detect] reading input image {} ...", cli.input.display());
    let image = match std::fs::read(&cli.input) {
        Ok(bytes) => {
            eprintln!("[detect] read {} bytes", bytes.len());
            bytes
        }
        Err(err) => {
            eprintln!("[detect] failed to read {}: {err}", cli.input.display());
            std::process::exit(1);
        }
    };

    eprintln!("[detect] opening model {} ...", cli.model.display());
    let mut model = match open_model(&cli.model) {
        Ok(model) => model,
        Err(err) => {
            eprintln!("[tpu] failed to open model {}: {err}", cli.model.display());
            std::process::exit(1);
        }
    };
    eprintln!("[detect] model opened successfully");

    let config = InferenceConfig {
        classes_num: cli.classes,
        confidence_threshold: cli.conf,
        iou_threshold: cli.iou,
        debug_logging: false,
    };

    eprintln!("[detect] starting detect_image ...");
    match model.detect_image(&image, &cli.output, config) {
        Ok(detections) => {
            eprintln!(
                "[detect] {} detection(s) on {}",
                detections.len(),
                cli.input.display()
            );
            for (i, d) in detections.iter().enumerate() {
                eprintln!(
                    "  #{i} class={} score={:.3} box=(cx={:.1}, cy={:.1}, w={:.1}, h={:.1})",
                    d.cls, d.score, d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h
                );
            }
            eprintln!(
                "[detect] annotated image written to {}",
                cli.output.display()
            );
        }
        Err(err) => {
            eprintln!("[detect] inference failed: {err}");
            std::process::exit(1);
        }
    }
}

fn run_capture(cli: CaptureCli) {
    let mut camera = match UsbCamera::open(&cli.camera) {
        Ok(camera) => {
            let info = camera.info();
            eprintln!(
                "[capture] opened {}: {}x{} format={} connected={}",
                cli.camera, info.width, info.height, info.format, info.connected
            );
            camera
        }
        Err(err) => {
            eprintln!("[capture] failed to open {}: {err}", cli.camera);
            std::process::exit(1);
        }
    };

    // Discard the first few frames so the sensor's auto exposure / white
    // balance can settle before we keep one.
    for i in 0..cli.warmup {
        if let Err(err) = camera.get_frame() {
            eprintln!("[capture] warm-up frame {i} failed: {err}");
            std::process::exit(1);
        }
    }
    if cli.warmup > 0 {
        eprintln!("[capture] discarded {} warm-up frame(s)", cli.warmup);
    }

    let frame = match camera.get_frame() {
        Ok(frame) => frame,
        Err(err) => {
            eprintln!("[capture] failed to capture frame: {err}");
            std::process::exit(1);
        }
    };

    if let Err(err) = akars::image_bridge::save_yuv422p(
        &frame.pixels,
        frame.width as i32,
        frame.height as i32,
        &cli.output,
    ) {
        eprintln!("[capture] failed to write {}: {err}", cli.output.display());
        std::process::exit(1);
    }

    eprintln!(
        "[capture] wrote {}x{} frame to {}",
        frame.width,
        frame.height,
        cli.output.display()
    );
}

fn parse_cli(args: impl Iterator<Item = String>) -> Result<Command, String> {
    let mut args = args.peekable();
    match args.peek().map(String::as_str) {
        Some("serve") => {
            args.next();
            parse_serve_cli(args).map(Command::Serve)
        }
        Some("detect") => {
            args.next();
            parse_detect_cli(args).map(Command::Detect)
        }
        Some("capture") => {
            args.next();
            parse_capture_cli(args).map(Command::Capture)
        }
        _ => parse_hunt_cli(args).map(Command::Hunt),
    }
}

fn parse_hunt_cli(args: impl Iterator<Item = String>) -> Result<Cli, String> {
    let mut cli = Cli::default();
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            "--camera" => cli.camera = take_value(&mut args, "--camera")?,
            "--vpss" => cli.vpss = take_value(&mut args, "--vpss")?,
            "--motor" => cli.motor = take_value(&mut args, "--motor")?,
            "--arm" => cli.arm = take_value(&mut args, "--arm")?,
            "--frames" => {
                cli.frames = Some(
                    take_value(&mut args, "--frames")?
                        .parse()
                        .map_err(|_| "--frames expects an integer".to_string())?,
                );
            }
            "--max-deposits" => {
                let value = take_value(&mut args, "--max-deposits")?
                    .parse::<u32>()
                    .map_err(|_| "--max-deposits expects a positive integer".to_string())?;
                if value == 0 {
                    return Err("--max-deposits expects a positive integer".to_string());
                }
                cli.max_deposits = Some(value);
            }
            "--classes" => {
                cli.classes = take_value(&mut args, "--classes")?
                    .parse()
                    .map_err(|_| "--classes expects an integer".to_string())?;
            }
            "--conf" => {
                cli.conf = take_value(&mut args, "--conf")?
                    .parse()
                    .map_err(|_| "--conf expects a float".to_string())?;
            }
            "--iou" => {
                cli.iou = take_value(&mut args, "--iou")?
                    .parse()
                    .map_err(|_| "--iou expects a float".to_string())?;
            }
            "--tpu-debug" => {
                cli.tpu_debug = parse_on_off(&take_value(&mut args, "--tpu-debug")?)?;
            }
            value if value.starts_with('-') => return Err(format!("unknown option: {value}")),
            value => {
                if cli.model.as_os_str().is_empty() {
                    cli.model = PathBuf::from(value);
                } else {
                    return Err(format!("unexpected positional argument: {value}"));
                }
            }
        }
    }

    if cli.model.as_os_str().is_empty() {
        return Err("missing cvimodel path".to_string());
    }
    Ok(cli)
}

fn parse_serve_cli(args: impl Iterator<Item = String>) -> Result<WebConfig, String> {
    let mut config = WebConfig::default();
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_web_usage();
                std::process::exit(0);
            }
            "--listen" => {
                config.listen = take_value(&mut args, "--listen")?
                    .parse::<SocketAddr>()
                    .map_err(|_| {
                        "--listen expects HOST:PORT, for example 0.0.0.0:8080".to_string()
                    })?;
            }
            "--motor" => config.motor_device = take_value(&mut args, "--motor")?,
            "--arm" => config.arm_device = take_value(&mut args, "--arm")?,
            "--camera" => config.camera_device = Some(take_value(&mut args, "--camera")?),
            "--mock" => config.mock = true,
            value if value.starts_with('-') => {
                return Err(format!("unknown serve option: {value}"))
            }
            value => return Err(format!("unexpected serve argument: {value}")),
        }
    }
    Ok(config)
}

fn parse_detect_cli(args: impl Iterator<Item = String>) -> Result<DetectCli, String> {
    let mut cli = DetectCli::default();
    let mut positional = 0;
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_detect_usage();
                std::process::exit(0);
            }
            "-o" | "--out" => cli.output = PathBuf::from(take_value(&mut args, "--out")?),
            "--classes" => {
                cli.classes = take_value(&mut args, "--classes")?
                    .parse()
                    .map_err(|_| "--classes expects an integer".to_string())?;
            }
            "--conf" => {
                cli.conf = take_value(&mut args, "--conf")?
                    .parse()
                    .map_err(|_| "--conf expects a float".to_string())?;
            }
            "--iou" => {
                cli.iou = take_value(&mut args, "--iou")?
                    .parse()
                    .map_err(|_| "--iou expects a float".to_string())?;
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown detect option: {value}"))
            }
            value => {
                match positional {
                    0 => cli.model = PathBuf::from(value),
                    1 => cli.input = PathBuf::from(value),
                    _ => return Err(format!("unexpected detect argument: {value}")),
                }
                positional += 1;
            }
        }
    }

    if cli.model.as_os_str().is_empty() {
        return Err("missing cvimodel path".to_string());
    }
    if cli.input.as_os_str().is_empty() {
        return Err("missing input image path".to_string());
    }
    Ok(cli)
}

fn parse_capture_cli(args: impl Iterator<Item = String>) -> Result<CaptureCli, String> {
    let mut cli = CaptureCli::default();
    let mut positional = 0;
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_capture_usage();
                std::process::exit(0);
            }
            "--camera" => cli.camera = take_value(&mut args, "--camera")?,
            "-o" | "--out" => cli.output = PathBuf::from(take_value(&mut args, "--out")?),
            "--warmup" => {
                cli.warmup = take_value(&mut args, "--warmup")?
                    .parse()
                    .map_err(|_| "--warmup expects a non-negative integer".to_string())?;
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown capture option: {value}"))
            }
            value => {
                match positional {
                    0 => cli.output = PathBuf::from(value),
                    _ => return Err(format!("unexpected capture argument: {value}")),
                }
                positional += 1;
            }
        }
    }
    Ok(cli)
}

fn take_value(
    args: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    option: &str,
) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{option} expects a value"))
}

fn parse_on_off(value: &str) -> Result<bool, String> {
    match value {
        "on" => Ok(true),
        "off" => Ok(false),
        _ => Err("--tpu-debug expects 'on' or 'off'".to_string()),
    }
}

fn print_usage() {
    eprintln!(
        "Usage:\n  akars <aligned-model.cvimodel> [--camera DEV] [--vpss DEV] [--motor DEV] [--arm DEV] [--frames N] [--max-deposits N] [--classes N] [--conf X] [--iou X] [--tpu-debug on|off]\n  akars serve [--listen HOST:PORT] [--motor DEV] [--arm DEV] [--mock]\n  akars detect <model.cvimodel> <image> [--out PATH] [--classes N] [--conf X] [--iou X]\n  akars capture [output.jpg] [--camera DEV] [--out PATH] [--warmup N]\n\nTPU debug defaults to off. With debug off, per-frame TPU details are suppressed,\nbut final per-stage single-frame averages are still printed. --max-deposits stops\nthe chassis safely after N verified deposit cycles.\n\nNote: motor defaults to /dev/ttyS1 (JTAG pads). Use --motor /dev/ttyS3 for\nGPIOP UART3, but this will disconnect WiFi (shared SDIO pins)."
    );
}

fn print_capture_usage() {
    eprintln!(
        "Usage: akars capture [output.jpg] [--camera DEV] [--out PATH] [--warmup N]\n\nGrabs a single JPEG frame from the camera, discarding the first N frames so the\nsensor's auto exposure / white balance can settle (warm up).\n\nDefaults:\n  --camera /dev/cvi-usb-camera0\n  --out capture.jpg\n  --warmup 30"
    );
}

fn print_detect_usage() {
    eprintln!(
        "Usage: akars detect <model.cvimodel> <image> [--out PATH] [--classes N] [--conf X] [--iou X]\n\nRuns the TPU model on a single image and writes a copy with detection boxes drawn.\n\nDefaults:\n  --out detect_out.jpg\n  --classes 1\n  --conf 0.5\n  --iou 0.5"
    );
}

fn print_web_usage() {
    eprintln!(
        "Usage: akars serve [--listen HOST:PORT] [--motor DEV] [--arm DEV] [--camera DEV] [--mock]\n\nDefaults:\n  --listen 0.0.0.0:8080\n  --motor /dev/ttyS1\n  --arm /dev/ttyS2\n  --camera (none — camera streaming disabled)"
    );
}

#[cfg(test)]
mod tests {
    use super::parse_on_off;

    #[test]
    fn parses_tpu_debug_switch() {
        assert_eq!(parse_on_off("on"), Ok(true));
        assert_eq!(parse_on_off("off"), Ok(false));
        assert!(parse_on_off("1").is_err());
    }
}
