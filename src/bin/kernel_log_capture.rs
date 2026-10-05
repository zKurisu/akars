#[cfg(target_arch = "riscv64")]
use std::arch::asm;
use std::env;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[cfg(target_arch = "riscv64")]
const SYSLOG_ACTION_READ_CLEAR: usize = 4;
#[cfg(target_arch = "riscv64")]
const RISCV64_NR_SYSLOG: usize = 116;
const SYSLOG_BUFFER_SIZE: usize = 65_536;

static STOP_REQUESTED: AtomicBool = AtomicBool::new(false);

unsafe extern "C" {
    fn signal(signal: i32, handler: usize) -> usize;
}

extern "C" fn request_stop(_signal: i32) {
    STOP_REQUESTED.store(true, Ordering::Release);
}

#[cfg(target_arch = "riscv64")]
fn read_and_clear_kernel_log(buffer: &mut [u8]) -> io::Result<usize> {
    let result: isize;
    unsafe {
        asm!(
            "ecall",
            in("a0") SYSLOG_ACTION_READ_CLEAR,
            in("a1") buffer.as_mut_ptr(),
            in("a2") buffer.len(),
            in("a7") RISCV64_NR_SYSLOG,
            lateout("a0") result,
            options(nostack),
        );
    }
    if result < 0 {
        Err(io::Error::from_raw_os_error((-result) as i32))
    } else {
        Ok(result as usize)
    }
}

#[cfg(not(target_arch = "riscv64"))]
fn read_and_clear_kernel_log(_buffer: &mut [u8]) -> io::Result<usize> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "kernel-log-capture requires riscv64 StarryOS",
    ))
}

fn parse_interval(value: Option<String>, default: f64, name: &str) -> Result<Duration, String> {
    let value = value.map_or(Ok(default), |raw| {
        raw.parse::<f64>()
            .map_err(|_| format!("{name} must be a positive number, got: {raw}"))
    })?;
    if !value.is_finite() || value <= 0.0 {
        return Err(format!("{name} must be greater than zero, got: {value}"));
    }
    Ok(Duration::from_secs_f64(value))
}

fn drain_log(file: &mut File, buffer: &mut [u8]) -> io::Result<usize> {
    let length = read_and_clear_kernel_log(buffer)?;
    if length > 0 {
        file.write_all(&buffer[..length])?;
    }
    Ok(length)
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let log_path = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "/root/kernel-live-v2.log".to_string()),
    );
    let poll_interval = parse_interval(args.next(), 0.2, "poll interval")?;
    // StarryOS file fsync can take much longer than Linux. Two seconds retains
    // crash-adjacent logs without continuously monopolising the ext4 path.
    let fsync_interval = parse_interval(args.next(), 2.0, "fsync interval")?;
    if let Some(extra) = args.next() {
        return Err(format!("unexpected argument: {extra}"));
    }

    unsafe {
        signal(2, request_stop as *const () as usize);
        signal(15, request_stop as *const () as usize);
    }

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|error| format!("open {}: {error}", log_path.display()))?;

    writeln!(
        file,
        "\n===== AKARS_KERNEL_LOG_START unix_s={} pid={} poll_s={:.3} fsync_s={:.3} mode=syslog-read-clear =====",
        unix_seconds(),
        std::process::id(),
        poll_interval.as_secs_f64(),
        fsync_interval.as_secs_f64(),
    )
    .map_err(|error| format!("write header: {error}"))?;

    let mut buffer = vec![0u8; SYSLOG_BUFFER_SIZE];
    drain_log(&mut file, &mut buffer).map_err(|error| format!("initial syslog read: {error}"))?;
    file.sync_all()
        .map_err(|error| format!("initial fsync: {error}"))?;

    eprintln!(
        "AKARS_KERNEL_LOG_RUNNING pid={} path={} poll_s={:.3} fsync_s={:.3} mode=syslog-read-clear",
        std::process::id(),
        log_path.display(),
        poll_interval.as_secs_f64(),
        fsync_interval.as_secs_f64(),
    );

    let mut last_fsync = Instant::now();
    while !STOP_REQUESTED.load(Ordering::Acquire) {
        thread::sleep(poll_interval);
        drain_log(&mut file, &mut buffer).map_err(|error| format!("syslog read: {error}"))?;
        if last_fsync.elapsed() >= fsync_interval {
            file.sync_data()
                .map_err(|error| format!("periodic fsync: {error}"))?;
            last_fsync = Instant::now();
        }
    }

    drain_log(&mut file, &mut buffer).map_err(|error| format!("final syslog read: {error}"))?;
    file.sync_all()
        .map_err(|error| format!("final fsync: {error}"))?;
    eprintln!("AKARS_KERNEL_LOG_STOPPED path={}", log_path.display());
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("AKARS_KERNEL_LOG_ERROR {error}");
        std::process::exit(1);
    }
}
