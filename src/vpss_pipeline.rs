use std::{
    ffi::{c_int, c_ulong, c_void},
    fs::{File, OpenOptions},
    io,
    mem::size_of,
    os::fd::{AsRawFd, FromRawFd},
    path::Path,
    ptr::NonNull,
    slice,
    time::Instant,
};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CameraCaptureProfile {
    pub capabilities: u32,
    pub attempts: u32,
    pub frame_total_us: u64,
    pub uvc_total_us: u64,
    pub wait_first_packet_us: u64,
    pub usb_transfer_us: u64,
    pub jpeg_assemble_us: u64,
    pub validate_us: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CameraCaptureStats {
    pub capture_calls: u64,
    pub transfer_attempts: u64,
    pub successful_frames: u64,
    pub failed_frames: u64,
    pub retry_attempts: u64,
    pub invalid_frames: u64,
    pub invalid_soi: u64,
    pub invalid_eoi: u64,
    pub invalid_too_small: u64,
    pub usb_errors: u64,
    pub published_frames: u64,
    pub overwritten_frames: u64,
    pub total_frame_us: u64,
    pub max_frame_us: u64,
    pub last_profile: CameraCaptureProfile,
}

#[derive(Clone, Copy, Debug)]
pub struct CameraFrameMeta {
    pub sequence: u64,
    pub timestamp_ns: u64,
    pub profile: CameraCaptureProfile,
}

const CAMERA_IOCTL_INIT: c_ulong = 1;
const CAMERA_IOCTL_HARD_RESET: c_ulong = 5;
const CAMERA_IOCTL_START_ASYNC: c_ulong = 6;
const CAMERA_IOCTL_STOP_ASYNC: c_ulong = 7;
const CAMERA_IOCTL_GET_CAPTURE_STATS: c_ulong = 9;
const CAMERA_IOCTL_RESET_CAPTURE_STATS: c_ulong = 10;
const CAMERA_IOCTL_GET_LATEST_YUV_ION: c_ulong = 13;
const CAMERA_ION_ABI_VERSION: u32 = 2;
const CAMERA_FORMAT_YUV422_PLANAR: u8 = 3;

const VPSS_ABI_VERSION: u32 = 1;
const VPSS_IOCTL_RUN_YUV422P_RGB: c_ulong = ioctl_read_write(b'V', 6, 248);
const ION_IOC_ALLOC: c_ulong = ioctl_read_write(b'I', 0, 64);
const ION_HEAP_DMA_COHERENT: u32 = 1;
const PROT_READ: c_int = 1;
const PROT_WRITE: c_int = 2;
const MAP_SHARED: c_int = 1;

const SOURCE_CAPACITY: usize = 2 * 1024 * 1024;
pub const CAMERA_WIDTH: u32 = 640;
pub const CAMERA_HEIGHT: u32 = 480;
const VPSS_STRIDE_ALIGNMENT: u32 = 64;

const fn ioctl_read_write(kind: u8, number: u8, size: usize) -> c_ulong {
    ((3_u64 << 30) | ((size as u64) << 16) | ((kind as u64) << 8) | number as u64) as c_ulong
}

unsafe extern "C" {
    fn ioctl(fd: c_int, request: c_ulong, ...) -> c_int;
    fn mmap(
        address: *mut c_void,
        length: usize,
        protection: c_int,
        flags: c_int,
        fd: c_int,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(address: *mut c_void, length: usize) -> c_int;
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct IonAllocData {
    len: u64,
    heap_id_mask: u32,
    flags: u32,
    fd: u32,
    unused: u32,
    paddr: u64,
    name: [u8; 32],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct CameraIonFrameRequest {
    abi_version: u32,
    flags: u32,
    ion_fd: i32,
    timeout_ms: u32,
    buffer_offset: u64,
    capacity: u64,
    last_sequence: u64,
    sequence: u64,
    timestamp_ns: u64,
    y_offset: u64,
    cb_offset: u64,
    cr_offset: u64,
    length: u32,
    stride_y: u32,
    stride_c: u32,
    width: u16,
    height: u16,
    format: u8,
    reserved: [u8; 3],
    jpu_decode_us: u64,
    profile: CameraCaptureProfile,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct VpssRunYuv422pRgb {
    abi_version: u32,
    flags: u32,
    source_fd: i32,
    destination_fd: i32,
    source_y_offset: u64,
    source_cb_offset: u64,
    source_cr_offset: u64,
    destination_r_offset: u64,
    destination_g_offset: u64,
    destination_b_offset: u64,
    sequence: u64,
    timestamp_ns: u64,
    source_width: u32,
    source_height: u32,
    source_y_stride: u32,
    source_c_stride: u32,
    crop_x: u32,
    crop_y: u32,
    crop_width: u32,
    crop_height: u32,
    content_x: u32,
    content_y: u32,
    content_width: u32,
    content_height: u32,
    destination_width: u32,
    destination_height: u32,
    destination_r_stride: u32,
    destination_gb_stride: u32,
    border_rgb: u32,
    timeout_ms: u32,
    status: i32,
    irq_status: u32,
    reserved0: u32,
    queue_enter_ns: u64,
    hardware_start_ns: u64,
    hardware_done_ns: u64,
    elapsed_ns: u64,
    output_sequence: u64,
    output_timestamp_ns: u64,
    img_debug: u32,
    img_axi_status: u32,
    scaler_status: u32,
    odma_debug: u32,
    reserved: [u32; 4],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct VpssTimingBreakdown {
    queue_us: u64,
    hardware_us: u64,
    wake_return_us: u64,
    timestamps_valid: bool,
}

fn split_vpss_timing(
    wall_us: u64,
    queue_enter_ns: u64,
    hardware_start_ns: u64,
    hardware_done_ns: u64,
    elapsed_ns: u64,
) -> VpssTimingBreakdown {
    let timestamps_valid = queue_enter_ns != 0
        && hardware_start_ns >= queue_enter_ns
        && hardware_done_ns >= hardware_start_ns;
    let (queue_us, hardware_us) = if timestamps_valid {
        (
            hardware_start_ns.saturating_sub(queue_enter_ns) / 1_000,
            hardware_done_ns.saturating_sub(hardware_start_ns) / 1_000,
        )
    } else {
        (0, elapsed_ns / 1_000)
    };
    // Subtract the kernel-observed portions from userspace ioctl wall time.
    // This boundary bucket includes syscall entry, IRQ wake-up/scheduling,
    // return, and the sub-microsecond rounding residue, so all three reported
    // stages add back to wall_us exactly.
    let wake_return_us = wall_us.saturating_sub(queue_us.saturating_add(hardware_us));
    VpssTimingBreakdown {
        queue_us,
        hardware_us,
        wake_return_us,
        timestamps_valid,
    }
}

const _: [(); 56] = [(); size_of::<CameraCaptureProfile>()];
const _: [(); 168] = [(); size_of::<CameraCaptureStats>()];
const _: [(); 64] = [(); size_of::<IonAllocData>()];
const _: [(); 168] = [(); size_of::<CameraIonFrameRequest>()];
const _: [(); 248] = [(); size_of::<VpssRunYuv422pRgb>()];

struct IonAllocation {
    file: File,
    size: usize,
    physical_address: u64,
    mapping: NonNull<u8>,
}

impl IonAllocation {
    fn allocate(ion: &File, requested_size: usize, name: &str) -> io::Result<Self> {
        let size = requested_size.div_ceil(4096) * 4096;
        let mut request = IonAllocData {
            len: size as u64,
            heap_id_mask: 1 << ION_HEAP_DMA_COHERENT,
            ..IonAllocData::default()
        };
        let name_bytes = name.as_bytes();
        let name_len = name_bytes.len().min(request.name.len() - 1);
        request.name[..name_len].copy_from_slice(&name_bytes[..name_len]);
        ioctl_ptr(ion, ION_IOC_ALLOC, &mut request)?;
        // SAFETY: a successful ION allocation returns ownership of a new fd.
        let file = unsafe { File::from_raw_fd(request.fd as i32) };
        // SAFETY: the ION fd owns at least `size` bytes and remains held by
        // this allocation until after the mapping is released in Drop.
        let mapping = unsafe {
            mmap(
                std::ptr::null_mut(),
                size,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if mapping as isize == -1 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            file,
            size,
            physical_address: request.paddr,
            mapping: NonNull::new(mapping.cast()).ok_or_else(io::Error::last_os_error)?,
        })
    }

    fn as_slice(&self) -> &[u8] {
        // SAFETY: `mapping` covers `size` bytes and is valid for the lifetime
        // of this allocation. DMA writes are complete before this is called.
        unsafe { slice::from_raw_parts(self.mapping.as_ptr(), self.size) }
    }
}

impl Drop for IonAllocation {
    fn drop(&mut self) {
        // SAFETY: the mapping was created by mmap with exactly this address
        // and length, and is released once before the fd field is dropped.
        unsafe {
            munmap(self.mapping.as_ptr().cast(), self.size);
        }
    }
}

pub struct VpssRgbFrame<'a> {
    pub vpss_executed: bool,
    pub physical_address: u64,
    pub rgb: &'a [u8],
    pub yuv: &'a [u8],
    pub yuv_width: u32,
    pub yuv_height: u32,
    pub yuv_format: u8,
    pub meta: CameraFrameMeta,
    pub camera_request_us: u64,
    pub jpu_decode_us: u64,
    pub vpss_wall_us: u64,
    /// Time from entering the kernel VPSS queue to the MMIO start trigger.
    pub vpss_queue_us: u64,
    pub vpss_hardware_us: u64,
    /// ioctl wall time not covered by queue/programming or hardware execution.
    /// This includes syscall entry plus IRQ wake-up, scheduling, and return.
    pub vpss_wake_return_us: u64,
    pub vpss_timestamps_valid: bool,
}

pub struct VpssRgbPipeline {
    camera: File,
    vpss: File,
    _ion: File,
    source: IonAllocation,
    destination: IonAllocation,
    output_width: u32,
    output_height: u32,
    output_plane_size: usize,
    output_size: usize,
    content_y: u32,
    content_height: u32,
    sequence: u64,
    started: bool,
}

impl VpssRgbPipeline {
    pub fn open(
        camera_path: impl AsRef<Path>,
        vpss_path: impl AsRef<Path>,
        output_width: u32,
        output_height: u32,
    ) -> io::Result<Self> {
        if output_width == 0
            || output_width != output_height
            || !output_width.is_multiple_of(VPSS_STRIDE_ALIGNMENT)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "VPSS RGB output must be a non-zero square with {}-byte aligned rows, got \
                     {}x{}",
                    VPSS_STRIDE_ALIGNMENT, output_width, output_height
                ),
            ));
        }
        let content_height = output_height
            .checked_mul(CAMERA_HEIGHT)
            .and_then(|value| value.checked_div(CAMERA_WIDTH))
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "VPSS content size overflow")
            })?;
        let content_y = (output_height - content_height) / 2;
        let output_plane_size = usize::try_from(output_width)
            .ok()
            .and_then(|width| {
                usize::try_from(output_height)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "VPSS output size overflow")
            })?;
        let output_size = output_plane_size
            .checked_mul(3)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "VPSS RGB size overflow"))?;
        let ion = OpenOptions::new().read(true).write(true).open("/dev/ion")?;
        let source = IonAllocation::allocate(&ion, SOURCE_CAPACITY, "akars-jpu")?;
        let destination = IonAllocation::allocate(&ion, output_size, "akars-vpss-rgb")?;
        let camera = OpenOptions::new().read(true).open(camera_path)?;
        let vpss = OpenOptions::new().read(true).write(true).open(vpss_path)?;
        ioctl_no_arg(&camera, CAMERA_IOCTL_INIT)?;
        ioctl_no_arg(&camera, CAMERA_IOCTL_RESET_CAPTURE_STATS)?;
        ioctl_no_arg(&camera, CAMERA_IOCTL_START_ASYNC)?;
        println!(
            "AKARS_VPSS_BUFFERS source_paddr=0x{:x} source_size={} destination_paddr=0x{:x} \
             destination_size={} layout=rgb_planar_3x{}x{} stride={} content_y={} \
             content_height={}",
            source.physical_address,
            source.size,
            destination.physical_address,
            destination.size,
            output_width,
            output_height,
            output_width,
            content_y,
            content_height,
        );
        Ok(Self {
            camera,
            vpss,
            _ion: ion,
            source,
            destination,
            output_width,
            output_height,
            output_plane_size,
            output_size,
            content_y,
            content_height,
            sequence: 0,
            started: true,
        })
    }

    pub fn next(&mut self, timeout_ms: u32) -> io::Result<VpssRgbFrame<'_>> {
        self.next_inner(timeout_ms, true)
    }

    /// Capture the latest JPU YUV422P frame without running VPSS.  The red
    /// container detector consumes YUV directly, so producing an RGB tensor
    /// in that mission phase only burns time and memory bandwidth.
    pub fn next_yuv(&mut self, timeout_ms: u32) -> io::Result<VpssRgbFrame<'_>> {
        self.next_inner(timeout_ms, false)
    }

    fn next_inner(
        &mut self,
        timeout_ms: u32,
        execute_vpss: bool,
    ) -> io::Result<VpssRgbFrame<'_>> {
        let mut camera = CameraIonFrameRequest {
            abi_version: CAMERA_ION_ABI_VERSION,
            ion_fd: self.source.file.as_raw_fd(),
            timeout_ms,
            capacity: self.source.size as u64,
            last_sequence: self.sequence,
            ..CameraIonFrameRequest::default()
        };
        let camera_start = Instant::now();
        ioctl_ptr(&self.camera, CAMERA_IOCTL_GET_LATEST_YUV_ION, &mut camera)?;
        let camera_request_us = camera_start.elapsed().as_micros() as u64;
        if camera.format != CAMERA_FORMAT_YUV422_PLANAR {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("JPU returned format {}, expected YUV422P", camera.format),
            ));
        }
        if u32::from(camera.width) != CAMERA_WIDTH || u32::from(camera.height) != CAMERA_HEIGHT {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "VPSS fast path requires 640x480 camera output, got {}x{}",
                    camera.width, camera.height
                ),
            ));
        }
        let yuv_start = usize::try_from(camera.y_offset).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "JPU Y offset exceeds usize")
        })?;
        let yuv_len = usize::try_from(camera.length).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "JPU frame length exceeds usize")
        })?;
        let yuv_end = yuv_start.checked_add(yuv_len).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "JPU frame range overflow")
        })?;
        if yuv_end > self.source.size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "JPU frame exceeds source ION mapping",
            ));
        }

        let mut run = VpssRunYuv422pRgb {
            abi_version: VPSS_ABI_VERSION,
            source_fd: self.source.file.as_raw_fd(),
            destination_fd: self.destination.file.as_raw_fd(),
            source_y_offset: camera.y_offset,
            source_cb_offset: camera.cb_offset,
            source_cr_offset: camera.cr_offset,
            destination_r_offset: 0,
            destination_g_offset: self.output_plane_size as u64,
            destination_b_offset: (2 * self.output_plane_size) as u64,
            sequence: camera.sequence,
            timestamp_ns: camera.timestamp_ns,
            source_width: u32::from(camera.width),
            source_height: u32::from(camera.height),
            source_y_stride: camera.stride_y,
            source_c_stride: camera.stride_c,
            crop_width: u32::from(camera.width),
            crop_height: u32::from(camera.height),
            content_y: self.content_y,
            content_width: self.output_width,
            content_height: self.content_height,
            destination_width: self.output_width,
            destination_height: self.output_height,
            destination_r_stride: self.output_width,
            destination_gb_stride: self.output_width,
            timeout_ms: 100,
            ..VpssRunYuv422pRgb::default()
        };
        let (vpss_wall_us, vpss_timing) = if execute_vpss {
            let vpss_start = Instant::now();
            ioctl_ptr(&self.vpss, VPSS_IOCTL_RUN_YUV422P_RGB, &mut run)?;
            let wall_us = vpss_start.elapsed().as_micros() as u64;
            if run.status != 0 {
                return Err(io::Error::other(format!(
                    "VPSS failed: status={} irq={:#x} img={:#x} axi={:#x} sc={:#x} odma={:#x}",
                    run.status,
                    run.irq_status,
                    run.img_debug,
                    run.img_axi_status,
                    run.scaler_status,
                    run.odma_debug
                )));
            }
            (
                wall_us,
                split_vpss_timing(
                    wall_us,
                    run.queue_enter_ns,
                    run.hardware_start_ns,
                    run.hardware_done_ns,
                    run.elapsed_ns,
                ),
            )
        } else {
            (0, VpssTimingBreakdown::default())
        };
        self.sequence = camera.sequence;
        Ok(VpssRgbFrame {
            vpss_executed: execute_vpss,
            physical_address: if execute_vpss {
                self.destination.physical_address
            } else {
                0
            },
            rgb: if execute_vpss {
                &self.destination.as_slice()[..self.output_size]
            } else {
                &[]
            },
            yuv: &self.source.as_slice()[yuv_start..yuv_end],
            yuv_width: u32::from(camera.width),
            yuv_height: u32::from(camera.height),
            yuv_format: camera.format,
            meta: CameraFrameMeta {
                sequence: camera.sequence,
                timestamp_ns: camera.timestamp_ns,
                profile: camera.profile,
            },
            camera_request_us,
            jpu_decode_us: camera.jpu_decode_us,
            vpss_wall_us,
            vpss_queue_us: vpss_timing.queue_us,
            vpss_hardware_us: vpss_timing.hardware_us,
            vpss_wake_return_us: vpss_timing.wake_return_us,
            vpss_timestamps_valid: vpss_timing.timestamps_valid,
        })
    }

    pub fn stats(&self) -> io::Result<CameraCaptureStats> {
        let mut stats = CameraCaptureStats::default();
        ioctl_ptr(&self.camera, CAMERA_IOCTL_GET_CAPTURE_STATS, &mut stats)?;
        Ok(stats)
    }

    /// Restart the asynchronous latest-frame producer after recoverable errors.
    pub fn restart(&mut self) -> io::Result<()> {
        self.stop()?;
        ioctl_no_arg(&self.camera, CAMERA_IOCTL_INIT)?;
        ioctl_no_arg(&self.camera, CAMERA_IOCTL_RESET_CAPTURE_STATS)?;
        ioctl_no_arg(&self.camera, CAMERA_IOCTL_START_ASYNC)?;
        self.sequence = 0;
        self.started = true;
        Ok(())
    }

    /// Power-cycle the camera and rebuild the asynchronous producer.
    pub fn hard_reset(&mut self) -> io::Result<()> {
        self.stop()?;
        ioctl_no_arg(&self.camera, CAMERA_IOCTL_HARD_RESET)?;
        ioctl_no_arg(&self.camera, CAMERA_IOCTL_RESET_CAPTURE_STATS)?;
        ioctl_no_arg(&self.camera, CAMERA_IOCTL_START_ASYNC)?;
        self.sequence = 0;
        self.started = true;
        Ok(())
    }

    pub fn stop(&mut self) -> io::Result<()> {
        if self.started {
            ioctl_no_arg(&self.camera, CAMERA_IOCTL_STOP_ASYNC)?;
            self.started = false;
        }
        Ok(())
    }
}

impl Drop for VpssRgbPipeline {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

fn ioctl_no_arg(file: &File, request: c_ulong) -> io::Result<()> {
    // SAFETY: this request has no pointer argument.
    let result = unsafe { ioctl(file.as_raw_fd(), request, 0usize) };
    ioctl_result(result)
}

fn ioctl_ptr<T>(file: &File, request: c_ulong, value: &mut T) -> io::Result<()> {
    // SAFETY: `value` is a live ABI structure for this blocking call.
    let result = unsafe { ioctl(file.as_raw_fd(), request, value as *mut T) };
    ioctl_result(result)
}

fn ioctl_result(result: c_int) -> io::Result<()> {
    if result < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{split_vpss_timing, VpssTimingBreakdown};

    #[test]
    fn splits_valid_kernel_timestamps_and_preserves_wall_total() {
        let timing = split_vpss_timing(50_000, 1_000_000, 3_500_000, 5_000_000, 0);
        assert_eq!(
            timing,
            VpssTimingBreakdown {
                queue_us: 2_500,
                hardware_us: 1_500,
                wake_return_us: 46_000,
                timestamps_valid: true,
            }
        );
        assert_eq!(
            timing.queue_us + timing.hardware_us + timing.wake_return_us,
            50_000
        );
    }

    #[test]
    fn falls_back_to_elapsed_when_kernel_timestamps_are_unavailable() {
        let timing = split_vpss_timing(9_000, 0, 0, 0, 2_250_000);
        assert_eq!(
            timing,
            VpssTimingBreakdown {
                queue_us: 0,
                hardware_us: 2_250,
                wake_return_us: 6_750,
                timestamps_valid: false,
            }
        );
    }
}
