use crate::linux;
use std::fs::File;
use std::io::{self, Read, Write};
use std::mem::MaybeUninit;
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::thread;
use std::time::{Duration, Instant};

pub struct SerialPort {
    file: File,
}

impl SerialPort {
    pub fn open<P: AsRef<Path>>(path: P, baudrate: i32, nonblocking: bool) -> io::Result<Self> {
        let path_ref = path.as_ref();
        let path_str = path_ref.to_string_lossy().to_string();
        let mut open_opts = std::fs::OpenOptions::new();
        open_opts.read(true).write(true);
        if nonblocking {
            open_opts.custom_flags(linux::O_NONBLOCK);
        }
        eprintln!(
            "[serial] opening {} (baud={baudrate}, nonblocking={nonblocking}) ...",
            path_str
        );
        let file = open_opts.open(path_ref)?;
        eprintln!("[serial] {} opened, configuring ...", path_str);
        configure(file.as_raw_fd(), baudrate)?;
        eprintln!("[serial] {} configured ok", path_str);
        Ok(Self { file })
    }

    pub fn write_all_drain(&mut self, data: &[u8]) -> io::Result<()> {
        self.file.write_all(data)?;
        let ret = unsafe { linux::tcdrain(self.file.as_raw_fd()) };
        if ret < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    pub fn read_byte(&mut self, timeout: Duration) -> io::Result<Option<u8>> {
        let deadline = Instant::now() + timeout;
        let mut byte = [0u8; 1];
        loop {
            match self.file.read(&mut byte) {
                Ok(1) => return Ok(Some(byte[0])),
                Ok(_) => {}
                Err(err) if err.kind() == io::ErrorKind::Interrupted => {}
                Err(err) if err.kind() == io::ErrorKind::WouldBlock => {}
                Err(err) => return Err(err),
            }
            if Instant::now() >= deadline {
                return Ok(None);
            }
            thread::sleep(Duration::from_millis(2));
        }
    }

    pub fn flush(&self) {
        unsafe {
            linux::tcflush(self.file.as_raw_fd(), linux::TCIOFLUSH);
        }
    }

    /// Clear stale protocol replies even when the target OS does not fully
    /// implement `tcflush(2)` for this UART driver.
    pub fn discard_input(&mut self) -> io::Result<usize> {
        self.flush();
        let mut discarded = 0usize;
        let mut buffer = [0u8; 64];
        loop {
            match self.file.read(&mut buffer) {
                Ok(0) => break,
                Ok(count) => discarded += count,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => break,
                Err(error) => return Err(error),
            }
        }
        Ok(discarded)
    }

    // Drain RX until no byte has arrived for `quiet_period`. This handles
    // delayed replies that arrive just after a nonblocking read first reports
    // `WouldBlock`; `max_duration` prevents a noisy device from hanging open.
    pub fn discard_input_until_quiet(
        &mut self,
        quiet_period: Duration,
        max_duration: Duration,
    ) -> io::Result<usize> {
        self.flush();
        let started = Instant::now();
        let mut quiet_since = Instant::now();
        let mut discarded = 0usize;
        let mut buffer = [0u8; 64];

        loop {
            match self.file.read(&mut buffer) {
                Ok(0) => {}
                Ok(count) => {
                    discarded += count;
                    quiet_since = Instant::now();
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {}
                Err(error) => return Err(error),
            }

            if quiet_since.elapsed() >= quiet_period || started.elapsed() >= max_duration {
                return Ok(discarded);
            }
            thread::sleep(Duration::from_millis(2));
        }
    }
}

fn configure(fd: i32, baudrate: i32) -> io::Result<()> {
    let mut termios = MaybeUninit::<linux::Termios>::uninit();
    let ret = unsafe { linux::tcgetattr(fd, termios.as_mut_ptr()) };
    if ret < 0 {
        return Err(io::Error::last_os_error());
    }
    let mut tty = unsafe { termios.assume_init() };

    let speed = baud_to_speed(baudrate);
    if unsafe { linux::cfsetispeed(&mut tty, speed) } < 0 {
        return Err(io::Error::last_os_error());
    }
    if unsafe { linux::cfsetospeed(&mut tty, speed) } < 0 {
        return Err(io::Error::last_os_error());
    }

    tty.c_cflag &= !linux::PARENB;
    tty.c_cflag &= !linux::PARODD;
    tty.c_cflag &= !linux::CSTOPB;
    tty.c_cflag &= !linux::CSIZE;
    tty.c_cflag |= linux::CS8;
    tty.c_cflag &= !linux::CRTSCTS;
    tty.c_cflag |= linux::CLOCAL | linux::CREAD;

    tty.c_iflag &= !(linux::IXON
        | linux::IXOFF
        | linux::IXANY
        | linux::IGNBRK
        | linux::BRKINT
        | linux::PARMRK
        | linux::ISTRIP
        | linux::INLCR
        | linux::IGNCR
        | linux::ICRNL);
    tty.c_oflag &= !linux::OPOST;
    tty.c_lflag = 0;
    tty.c_cc[6] = 0;
    tty.c_cc[5] = 0;

    unsafe {
        linux::tcflush(fd, linux::TCIOFLUSH);
    }
    if unsafe { linux::tcsetattr(fd, linux::TCSANOW, &tty) } < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

fn baud_to_speed(baudrate: i32) -> linux::speed_t {
    match baudrate {
        9600 => linux::B9600,
        57600 => linux::B57600,
        230400 => linux::B230400,
        _ => linux::B115200,
    }
}
