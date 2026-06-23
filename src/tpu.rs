use crate::camera::CameraFrame;
use crate::detector::Detection;
use std::error::Error;
use std::fmt;
use std::path::Path;

#[derive(Clone, Copy, Debug)]
pub struct InferenceConfig {
    pub classes_num: i32,
    pub confidence_threshold: f32,
    pub iou_threshold: f32,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            classes_num: 1,
            confidence_threshold: 0.5,
            iou_threshold: 0.5,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct InferTiming {
    /// JPEG decode (libjpeg) microseconds.
    pub decode_us: i64,
    /// resize + pad microseconds.
    pub resize_us: i64,
    /// BGR2RGB + split + memcpy microseconds.
    pub pack_us: i64,
    /// total preprocess (the C bridge call) microseconds.
    pub preprocess_us: i64,
    /// CVI_NN_Forward microseconds.
    pub forward_us: i64,
    /// detection parse + dequant + NMS + box correction microseconds.
    pub postprocess_us: i64,
}

#[derive(Debug)]
pub struct TpuError(String);

impl TpuError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for TpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for TpuError {}

#[cfg(target_arch = "riscv64")]
mod imp {
    use super::{CameraFrame, Detection, InferTiming, InferenceConfig, TpuError};
    use crate::detector::{correct_yolo_boxes, nms, parse_yolov8_output};
    use std::ffi::{c_char, c_int, c_void, CString};
    use std::os::unix::ffi::OsStrExt;
    use std::path::Path;
    use std::ptr;
    use std::slice;
    use std::time::Instant;

    const CVI_FMT_FP32: i32 = 0;
    const CVI_FMT_BF16: i32 = 3;
    const CVI_FMT_INT16: i32 = 4;
    const CVI_FMT_INT8: i32 = 6;
    const CVI_FMT_UINT8: i32 = 7;
    const CVI_RC_SUCCESS: i32 = 0;
    const CVI_DIM_MAX: usize = 6;

    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    struct CviShape {
        dim: [i32; CVI_DIM_MAX],
        dim_size: usize,
    }

    #[repr(C)]
    #[derive(Debug)]
    struct CviTensor {
        name: *mut c_char,
        shape: CviShape,
        fmt: i32,
        count: usize,
        mem_size: usize,
        sys_mem: *mut u8,
        paddr: u64,
        mem_type: i32,
        qscale: f32,
        zero_point: c_int,
        pixel_format: i32,
        aligned: bool,
        mean: [f32; 3],
        scale: [f32; 3],
        owner: *mut c_void,
        reserved: [c_char; 32],
    }

    type CviModelHandle = *mut c_void;

    unsafe extern "C" {
        fn CVI_NN_RegisterModel(model_file: *const c_char, model: *mut CviModelHandle) -> i32;
        fn CVI_NN_GetInputOutputTensors(
            model: CviModelHandle,
            inputs: *mut *mut CviTensor,
            input_num: *mut i32,
            outputs: *mut *mut CviTensor,
            output_num: *mut i32,
        ) -> i32;
        fn CVI_NN_GetTensorByName(
            name: *const c_char,
            tensors: *mut CviTensor,
            num: i32,
        ) -> *mut CviTensor;
        fn CVI_NN_TensorPtr(tensor: *mut CviTensor) -> *mut c_void;
        fn CVI_NN_TensorShape(tensor: *mut CviTensor) -> CviShape;
        fn CVI_NN_Forward(
            model: CviModelHandle,
            inputs: *mut CviTensor,
            input_num: i32,
            outputs: *mut CviTensor,
            output_num: i32,
        ) -> i32;
        fn CVI_NN_CleanupModel(model: CviModelHandle) -> i32;

        fn akars_mjpeg_to_rgb_planar(
            jpeg: *const u8,
            jpeg_len: usize,
            dst: *mut u8,
            dst_w: i32,
            dst_h: i32,
            src_w: *mut i32,
            src_h: *mut i32,
            decode_us: *mut i64,
            resize_us: *mut i64,
            pack_us: *mut i64,
        ) -> i32;

        fn akars_draw_detections(
            image: *const u8,
            image_len: usize,
            boxes: *const f32,
            classes: *const c_int,
            scores: *const f32,
            count: c_int,
            out_path: *const c_char,
        ) -> i32;
    }

    pub struct YoloModel {
        model: CviModelHandle,
        inputs: *mut CviTensor,
        input_num: i32,
        outputs: *mut CviTensor,
        output_num: i32,
        input: *mut CviTensor,
        input_h: i32,
        input_w: i32,
        output_shapes: Vec<CviShape>,
    }

    impl YoloModel {
        pub fn open(path: &Path) -> Result<Self, TpuError> {
            let c_path = CString::new(path.as_os_str().as_bytes())
                .map_err(|_| TpuError::new("model path contains NUL byte"))?;
            let mut model: CviModelHandle = ptr::null_mut();
            let rc = unsafe { CVI_NN_RegisterModel(c_path.as_ptr(), &mut model) };
            if rc != CVI_RC_SUCCESS {
                return Err(TpuError::new(format!("CVI_NN_RegisterModel failed: {rc}")));
            }

            let mut inputs = ptr::null_mut();
            let mut outputs = ptr::null_mut();
            let mut input_num = 0;
            let mut output_num = 0;
            let rc = unsafe {
                CVI_NN_GetInputOutputTensors(
                    model,
                    &mut inputs,
                    &mut input_num,
                    &mut outputs,
                    &mut output_num,
                )
            };
            if rc != CVI_RC_SUCCESS {
                unsafe {
                    CVI_NN_CleanupModel(model);
                }
                return Err(TpuError::new(format!(
                    "CVI_NN_GetInputOutputTensors failed: {rc}"
                )));
            }

            let input = unsafe { CVI_NN_GetTensorByName(ptr::null(), inputs, input_num) };
            if input.is_null() {
                unsafe {
                    CVI_NN_CleanupModel(model);
                }
                return Err(TpuError::new("default input tensor not found"));
            }

            let input_shape = unsafe { CVI_NN_TensorShape(input) };
            let input_h = input_shape.dim[2];
            let input_w = input_shape.dim[3];
            let outputs_slice = unsafe { slice::from_raw_parts_mut(outputs, output_num as usize) };
            let output_shapes = outputs_slice
                .iter_mut()
                .map(|tensor| unsafe { CVI_NN_TensorShape(tensor as *mut CviTensor) })
                .collect();

            Ok(Self {
                model,
                inputs,
                input_num,
                outputs,
                output_num,
                input,
                input_h,
                input_w,
                output_shapes,
            })
        }

        pub fn infer(
            &mut self,
            frame: &CameraFrame,
            config: InferenceConfig,
        ) -> Result<Vec<Detection>, TpuError> {
            self.infer_timed(frame, config, None)
        }

        pub fn infer_timed(
            &mut self,
            frame: &CameraFrame,
            config: InferenceConfig,
            mut timing: Option<&mut InferTiming>,
        ) -> Result<Vec<Detection>, TpuError> {
            let input_ptr = unsafe { CVI_NN_TensorPtr(self.input) as *mut u8 };
            if input_ptr.is_null() {
                return Err(TpuError::new("input tensor pointer is null"));
            }

            let mut decoded_w = 0;
            let mut decoded_h = 0;
            let mut decode_us = 0i64;
            let mut resize_us = 0i64;
            let mut pack_us = 0i64;
            let pre_start = Instant::now();
            let rc = unsafe {
                akars_mjpeg_to_rgb_planar(
                    frame.jpeg.as_ptr(),
                    frame.jpeg.len(),
                    input_ptr,
                    self.input_w,
                    self.input_h,
                    &mut decoded_w,
                    &mut decoded_h,
                    &mut decode_us,
                    &mut resize_us,
                    &mut pack_us,
                )
            };
            let preprocess_us = pre_start.elapsed().as_micros() as i64;
            if rc != 0 {
                return Err(TpuError::new(format!(
                    "MJPEG decode/preprocess failed: {rc}"
                )));
            }

            let fwd_start = Instant::now();
            let rc = unsafe {
                CVI_NN_Forward(
                    self.model,
                    self.inputs,
                    self.input_num,
                    self.outputs,
                    self.output_num,
                )
            };
            let forward_us = fwd_start.elapsed().as_micros() as i64;
            if rc != CVI_RC_SUCCESS {
                return Err(TpuError::new(format!("CVI_NN_Forward failed: {rc}")));
            }

            let post_start = Instant::now();
            let mut detections = self.get_detections(config)?;
            nms(&mut detections, config.iou_threshold);

            let image_w = if decoded_w > 0 {
                decoded_w
            } else {
                frame.width as i32
            };
            let image_h = if decoded_h > 0 {
                decoded_h
            } else {
                frame.height as i32
            };
            correct_yolo_boxes(
                &mut detections,
                image_h,
                image_w,
                self.input_h,
                self.input_w,
            );
            let postprocess_us = post_start.elapsed().as_micros() as i64;

            if let Some(t) = timing.as_deref_mut() {
                *t = InferTiming {
                    decode_us,
                    resize_us,
                    pack_us,
                    preprocess_us,
                    forward_us,
                    postprocess_us,
                };
            }
            Ok(detections)
        }

        /// Run inference on a standalone image (JPEG/PNG/... anything OpenCV can
        /// decode) and write a copy with the detection boxes drawn to out_path.
        pub fn detect_image(
            &mut self,
            image: &[u8],
            out_path: &Path,
            config: InferenceConfig,
        ) -> Result<Vec<Detection>, TpuError> {
            let frame = CameraFrame {
                jpeg: image.to_vec(),
                width: 0,
                height: 0,
            };
            let detections = self.infer(&frame, config)?;

            let boxes: Vec<f32> = detections
                .iter()
                .flat_map(|d| [d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h])
                .collect();
            let classes: Vec<c_int> = detections.iter().map(|d| d.cls as c_int).collect();
            let scores: Vec<f32> = detections.iter().map(|d| d.score).collect();

            let c_out = CString::new(out_path.as_os_str().as_bytes())
                .map_err(|_| TpuError::new("output path contains NUL byte"))?;
            let rc = unsafe {
                akars_draw_detections(
                    image.as_ptr(),
                    image.len(),
                    boxes.as_ptr(),
                    classes.as_ptr(),
                    scores.as_ptr(),
                    detections.len() as c_int,
                    c_out.as_ptr(),
                )
            };
            if rc != 0 {
                return Err(TpuError::new(format!(
                    "failed to write annotated image: {rc}"
                )));
            }
            Ok(detections)
        }

        fn get_detections(&mut self, config: InferenceConfig) -> Result<Vec<Detection>, TpuError> {
            if self.output_num < 1 || self.output_shapes.is_empty() {
                return Err(TpuError::new("model has no output tensor"));
            }
            let output = unsafe { &mut *self.outputs };
            let shape = self.output_shapes[0];
            let count = output.count;
            let ptr = unsafe { CVI_NN_TensorPtr(output as *mut CviTensor) };
            if ptr.is_null() {
                return Err(TpuError::new("output tensor pointer is null"));
            }

            let data = tensor_to_f32(output, ptr, count)?;
            Ok(parse_yolov8_output(
                &data,
                [shape.dim[0], shape.dim[1], shape.dim[2], shape.dim[3]],
                config.classes_num,
                config.confidence_threshold,
            ))
        }
    }

    impl Drop for YoloModel {
        fn drop(&mut self) {
            if !self.model.is_null() {
                unsafe {
                    CVI_NN_CleanupModel(self.model);
                }
            }
        }
    }

    fn tensor_to_f32(
        tensor: &CviTensor,
        ptr: *mut c_void,
        count: usize,
    ) -> Result<Vec<f32>, TpuError> {
        match tensor.fmt {
            CVI_FMT_FP32 => {
                let src = unsafe { slice::from_raw_parts(ptr as *const f32, count) };
                Ok(src.to_vec())
            }
            CVI_FMT_INT8 => {
                let src = unsafe { slice::from_raw_parts(ptr as *const i8, count) };
                Ok(src.iter().map(|v| *v as f32 * tensor.qscale).collect())
            }
            CVI_FMT_UINT8 => {
                let src = unsafe { slice::from_raw_parts(ptr as *const u8, count) };
                Ok(src
                    .iter()
                    .map(|v| (*v as i32 - tensor.zero_point) as f32 * tensor.qscale)
                    .collect())
            }
            CVI_FMT_BF16 => {
                let src = unsafe { slice::from_raw_parts(ptr as *const u16, count) };
                Ok(src
                    .iter()
                    .map(|v| f32::from_bits((*v as u32) << 16))
                    .collect())
            }
            CVI_FMT_INT16 => {
                let src = unsafe { slice::from_raw_parts(ptr as *const i16, count) };
                Ok(src.iter().map(|v| *v as f32 * tensor.qscale).collect())
            }
            other => Err(TpuError::new(format!(
                "unsupported output tensor format: {other}"
            ))),
        }
    }
}

#[cfg(not(target_arch = "riscv64"))]
mod imp {
    use super::{CameraFrame, Detection, InferTiming, InferenceConfig, TpuError};
    use std::path::Path;

    pub struct YoloModel;

    impl YoloModel {
        pub fn open(_path: &Path) -> Result<Self, TpuError> {
            Err(TpuError::new(
                "akars was built without SG2002 TPU/OpenCV runtime support",
            ))
        }

        pub fn infer(
            &mut self,
            _frame: &CameraFrame,
            _config: InferenceConfig,
        ) -> Result<Vec<Detection>, TpuError> {
            Err(TpuError::new(
                "akars was built without SG2002 TPU/OpenCV runtime support",
            ))
        }

        pub fn infer_timed(
            &mut self,
            _frame: &CameraFrame,
            _config: InferenceConfig,
            _timing: Option<&mut InferTiming>,
        ) -> Result<Vec<Detection>, TpuError> {
            Err(TpuError::new(
                "akars was built without SG2002 TPU/OpenCV runtime support",
            ))
        }

        pub fn detect_image(
            &mut self,
            _image: &[u8],
            _out_path: &Path,
            _config: InferenceConfig,
        ) -> Result<Vec<Detection>, TpuError> {
            Err(TpuError::new(
                "akars was built without SG2002 TPU/OpenCV runtime support",
            ))
        }
    }
}

pub use imp::YoloModel;

pub fn open_model(path: impl AsRef<Path>) -> Result<YoloModel, TpuError> {
    YoloModel::open(path.as_ref())
}
