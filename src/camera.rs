use crate::linux;
use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::time::Instant;

const CVI_CAMERA_IOCTL_INIT: u64 = 1;
const CVI_CAMERA_IOCTL_GET_INFO: u64 = 2;
/// cmd 3 — returns the raw MJPEG frame straight from the sensor (no JPU decode).
const CVI_CAMERA_IOCTL_GET_RAW: u64 = 3;
/// cmd 4 — kernel JPU-decodes MJPEG and returns YUV422 planar (I422).
const CVI_CAMERA_IOCTL_GET_FRAME: u64 = 4;
/// Power-cycle the camera VBUS and re-init from scratch.  This is the
/// strongest recovery — use when persistent EIO cannot be fixed by INIT or
/// reopen alone.
const CVI_CAMERA_IOCTL_HARD_RESET: u64 = 5;
const YUV_FRAME_WIDTH: usize = 640;
const YUV_FRAME_HEIGHT: usize = 480;
// I422: full-size Y plane + U and V planes each subsampled horizontally (w/2 × h).
const YUV_FRAME_SIZE: usize = YUV_FRAME_WIDTH * YUV_FRAME_HEIGHT * 2;
const JPEG_MARKER_START: [u8; 2] = [0xFF, 0xD8];
const JPEG_MARKER_END: [u8; 2] = [0xFF, 0xD9];

#[repr(C, packed)]
#[derive(Clone, Copy, Default)]
struct RawCameraInfo {
    width: u16,
    height: u16,
    format: u8,
    connected: u8,
}

#[derive(Clone, Copy, Debug)]
pub struct CameraInfo {
    pub width: u16,
    pub height: u16,
    pub format: u8,
    pub connected: bool,
}

impl RawCameraInfo {
    fn unpack(&self) -> CameraInfo {
        let width = unsafe { std::ptr::addr_of!(self.width).read_unaligned() };
        let height = unsafe { std::ptr::addr_of!(self.height).read_unaligned() };
        CameraInfo {
            width,
            height,
            format: self.format,
            connected: self.connected != 0,
        }
    }
}

#[derive(Debug)]
pub struct CameraFrame {
    /// Raw YUV422 planar (I422) pixels: Y plane, then U plane, then V plane.
    pub pixels: Vec<u8>,
    pub width: u16,
    pub height: u16,
}

pub struct UsbCamera {
    file: File,
    path: String,
    info: CameraInfo,
    /// Pre-allocated I/O buffer reused across get_frame to avoid per-frame
    /// 614 KB heap allocations that can exhaust memory on embedded systems.
    buffer: Vec<u8>,
}

impl UsbCamera {
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path_str = path.as_ref().to_string_lossy().into_owned();
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path.as_ref())?;

        ioctl_no_arg(file.as_raw_fd(), CVI_CAMERA_IOCTL_INIT)?;

        let mut raw = RawCameraInfo::default();
        ioctl_ptr(
            file.as_raw_fd(),
            CVI_CAMERA_IOCTL_GET_INFO,
            &mut raw as *mut RawCameraInfo,
        )?;
        let info = raw.unpack();
        if !info.connected {
            return Err(io::Error::new(
                io::ErrorKind::NotConnected,
                "camera reports disconnected",
            ));
        }
        if info.format != 1 {
            eprintln!(
                "[camera] warning: camera format {} is not MJPEG(1)",
                info.format
            );
        }

        // Pre-allocate frame buffer once.
        let buffer = vec![0u8; YUV_FRAME_SIZE];

        Ok(Self { file, path: path_str, info, buffer })
    }

    pub fn info(&self) -> CameraInfo {
        self.info
    }

    /// Re-initialize the capture pipeline.  With the fixed kernel driver this
    /// forces a full session reset and hardware re-init when called after a
    /// failed capture.
    pub fn re_init(&mut self) -> io::Result<()> {
        ioctl_no_arg(self.file.as_raw_fd(), CVI_CAMERA_IOCTL_INIT)
    }

    /// VBUS power-cycle: physically power-cycles the camera module via GPIO
    /// and re-initializes everything from scratch.  This takes ~2.5 s.
    /// Use as last-resort recovery when persistent EIO cannot be fixed by
    /// close/reopen alone.
    pub fn hard_reset(&mut self) -> io::Result<()> {
        ioctl_no_arg(self.file.as_raw_fd(), CVI_CAMERA_IOCTL_HARD_RESET)
    }

    /// Close the fd and reopen the device file, then re-initialize.  With the
    /// fixed kernel driver close() now clears the session, so the subsequent
    /// open+INIT forces a full hardware re-init.  If this still doesn't help,
    /// call hard_reset() next.
    pub fn reopen(&mut self) -> io::Result<()> {
        let path = self.path.clone();

        // Open a fresh fd (the old fd is dropped → close() clears session).
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)?;
        self.file = file;

        ioctl_no_arg(self.file.as_raw_fd(), CVI_CAMERA_IOCTL_INIT)?;

        let mut raw = RawCameraInfo::default();
        ioctl_ptr(
            self.file.as_raw_fd(),
            CVI_CAMERA_IOCTL_GET_INFO,
            &mut raw as *mut RawCameraInfo,
        )?;
        let info = raw.unpack();
        if !info.connected {
            return Err(io::Error::new(
                io::ErrorKind::NotConnected,
                "camera reports disconnected after reopen",
            ));
        }
        self.info = info;

        eprintln!("[camera] device reopened successfully");
        Ok(())
    }

    /// Diagnostic: capture one frame with cmd 3 (raw MJPEG) and one with
    /// cmd 4 (YUV via JPU), then report both timings.  This helps separate
    /// frame-wait latency from JPU decode overhead.
    /// **Burns 2 frames** — call once at startup.
    pub fn diagnose_timing(&mut self) {
        // Buffer large enough for a full raw MJPEG frame (640×480×2 worst case).
        let mut raw = vec![0u8; YUV_FRAME_SIZE];

        eprintln!("[camera] === timing diagnostic (cmd 3 vs cmd 4) ===");

        // --- cmd 3: raw MJPEG (no JPU decode) ---
        ioctl_no_arg(self.file.as_raw_fd(), CVI_CAMERA_IOCTL_INIT)
            .unwrap_or_else(|e| eprintln!("[camera] diag init: {e}"));
        let t0 = Instant::now();
        let ret3 = unsafe {
            linux::ioctl(
                self.file.as_raw_fd(),
                CVI_CAMERA_IOCTL_GET_RAW as _,
                raw.as_mut_ptr(),
            )
        };
        let raw_us = t0.elapsed().as_micros();
        let raw_size = if ret3 > 0 { ret3 as usize } else { 0 };
        ioctl_no_arg(self.file.as_raw_fd(), CVI_CAMERA_IOCTL_INIT)
            .unwrap_or_else(|e| eprintln!("[camera] diag re-init: {e}"));

        eprintln!(
            "[camera] cmd3 (raw MJPEG): {:.1} ms, {raw_size} bytes \
             (compression {:.0}%)",
            raw_us as f32 / 1000.0,
            (1.0 - raw_size as f32 / YUV_FRAME_SIZE as f32) * 100.0,
        );

        // --- cmd 4: YUV422 planar (with JPU decode) ---
        self.buffer.fill(0);
        let t1 = Instant::now();
        let ret4 = unsafe {
            linux::ioctl(
                self.file.as_raw_fd(),
                CVI_CAMERA_IOCTL_GET_FRAME as _,
                self.buffer.as_mut_ptr(),
            )
        };
        let yuv_us = t1.elapsed().as_micros();
        let yuv_size = if ret4 > 0 { ret4 as usize } else { 0 };
        ioctl_no_arg(self.file.as_raw_fd(), CVI_CAMERA_IOCTL_INIT)
            .unwrap_or_else(|e| eprintln!("[camera] diag re-init: {e}"));

        eprintln!(
            "[camera] cmd4 (YUV via JPU): {:.1} ms, {yuv_size} bytes",
            yuv_us as f32 / 1000.0,
        );

        if raw_us > 0 && yuv_us > 0 {
            let jpu_overhead = yuv_us.saturating_sub(raw_us);
            eprintln!(
                "[camera] JPU decode overhead ~{:.1} ms, \
                 frame-wait ~{:.1} ms",
                jpu_overhead as f32 / 1000.0,
                raw_us as f32 / 1000.0,
            );
        }
        eprintln!("[camera] === diagnostic done ===");
    }

    pub fn get_frame(&mut self) -> io::Result<CameraFrame> {
        // Do NOT call INIT between frames.  The stream was started once during
        // open() and each INIT→GET_FRAME cycle appears to reset the UVC
        // pipeline, forcing the sensor's AE/AWB to re-converge (~3 s).
        // Simply calling GET_FRAME on the already-running stream returns the
        // next frame at the sensor's native rate (~33 ms).
        self.buffer.fill(0);
        let ret = unsafe {
            linux::ioctl(
                self.file.as_raw_fd(),
                CVI_CAMERA_IOCTL_GET_FRAME as _,
                self.buffer.as_mut_ptr(),
            )
        };
        if ret < 0 {
            return Err(io::Error::last_os_error());
        }

        eprintln!("[camera] ioctl returned {ret} bytes");
        let size = ret as usize;

        if size != 0 && size < YUV_FRAME_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "camera returned a short YUV frame",
            ));
        }

        Ok(CameraFrame {
            pixels: self.buffer[..YUV_FRAME_SIZE].to_vec(),
            width: YUV_FRAME_WIDTH as u16,
            height: YUV_FRAME_HEIGHT as u16,
        })
    }
}

fn ioctl_no_arg(fd: i32, request: u64) -> io::Result<()> {
    let ret = unsafe { linux::ioctl(fd, request as _, 0usize) };
    if ret < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

fn ioctl_ptr<T>(fd: i32, request: u64, ptr: *mut T) -> io::Result<()> {
    let ret = unsafe { linux::ioctl(fd, request as _, ptr) };
    if ret < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

pub fn is_valid_jpeg(data: &[u8]) -> bool {
    data.len() >= 4 && data.starts_with(&JPEG_MARKER_START) && data.ends_with(&JPEG_MARKER_END)
}

#[cfg(test)]
mod tests {
    use super::is_valid_jpeg;

    #[test]
    fn checks_jpeg_markers() {
        assert!(is_valid_jpeg(&[0xFF, 0xD8, 1, 2, 0xFF, 0xD9]));
        assert!(!is_valid_jpeg(&[0x00, 0xD8, 1, 2, 0xFF, 0xD9]));
        assert!(!is_valid_jpeg(&[0xFF, 0xD8, 1]));
    }
}
