//! Lightweight red-target detection directly on planar YUV422 camera frames.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RedThreshold {
    /// Minimum luma accepted as a visible target pixel.
    pub y_min: u8,
    /// Maximum Cb value accepted as red.
    pub u_max: u8,
    /// Minimum Cr value accepted as red.
    pub v_min: u8,
}

impl Default for RedThreshold {
    fn default() -> Self {
        Self {
            y_min: 24,
            u_max: 115,
            v_min: 155,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RedObservation {
    pub pixels: usize,
    pub center_x: usize,
    pub center_y: usize,
    pub left: usize,
    pub top: usize,
    pub right: usize,
    pub bottom: usize,
    pub frame_width: usize,
    pub frame_height: usize,
}

impl RedObservation {
    pub fn area_ratio(self) -> f32 {
        self.pixels as f32 / (self.frame_width * self.frame_height) as f32
    }

    pub fn bbox_width_ratio(self) -> f32 {
        (self.right - self.left + 1) as f32 / self.frame_width as f32
    }

    pub fn bbox_height_ratio(self) -> f32 {
        (self.bottom - self.top + 1) as f32 / self.frame_height as f32
    }

    pub fn bbox_area_ratio(self) -> f32 {
        self.bbox_width_ratio() * self.bbox_height_ratio()
    }
}

/// Detect the only red target in a planar YUV422 (I422/YU16) camera frame.
///
/// The clean test environment guarantees that red samples belong to one
/// physical target, so accumulating all qualifying pixels is both cheaper and
/// more stable than allocating a full mask and running connected components.
/// Luma is checked per pixel while each U/V sample is shared by two horizontal
/// pixels, matching the I422 layout used by the StarryOS camera device.
pub fn detect_red_yuv422p(
    yuv: &[u8],
    width: usize,
    height: usize,
    threshold: RedThreshold,
) -> Option<RedObservation> {
    if width == 0 || height == 0 || !width.is_multiple_of(2) {
        return None;
    }
    let y_size = width.checked_mul(height)?;
    let chroma_width = width / 2;
    let chroma_size = chroma_width.checked_mul(height)?;
    let frame_size = y_size.checked_add(chroma_size.checked_mul(2)?)?;
    if yuv.len() < frame_size {
        return None;
    }

    let y_plane = &yuv[..y_size];
    let u_plane = &yuv[y_size..y_size + chroma_size];
    let v_plane = &yuv[y_size + chroma_size..frame_size];
    let mut pixels = 0usize;
    let mut sum_x = 0u64;
    let mut sum_y = 0u64;
    let mut left = width;
    let mut top = height;
    let mut right = 0usize;
    let mut bottom = 0usize;

    for row in 0..height {
        let y_row = row * width;
        let chroma_row = row * chroma_width;
        for pair in 0..chroma_width {
            let chroma_index = chroma_row + pair;
            if u_plane[chroma_index] > threshold.u_max || v_plane[chroma_index] < threshold.v_min {
                continue;
            }

            let first_x = pair * 2;
            for x in first_x..first_x + 2 {
                if y_plane[y_row + x] < threshold.y_min {
                    continue;
                }
                pixels += 1;
                sum_x += x as u64;
                sum_y += row as u64;
                left = left.min(x);
                top = top.min(row);
                right = right.max(x);
                bottom = bottom.max(row);
            }
        }
    }

    (pixels != 0).then(|| RedObservation {
        pixels,
        center_x: (sum_x / pixels as u64) as usize,
        center_y: (sum_y / pixels as u64) as usize,
        left,
        top,
        right,
        bottom,
        frame_width: width,
        frame_height: height,
    })
}

#[cfg(test)]
mod tests {
    use super::{detect_red_yuv422p, RedThreshold};

    fn yuv422_frame(width: usize, height: usize, y: u8, u: u8, v: u8) -> Vec<u8> {
        let y_size = width * height;
        let chroma_size = width / 2 * height;
        let mut frame = vec![y; y_size];
        frame.extend(vec![u; chroma_size]);
        frame.extend(vec![v; chroma_size]);
        frame
    }

    #[test]
    fn rejects_invalid_or_short_frames() {
        assert!(detect_red_yuv422p(&[], 640, 480, RedThreshold::default()).is_none());
        assert!(detect_red_yuv422p(&[0; 16], 3, 2, RedThreshold::default()).is_none());
    }

    #[test]
    fn detects_a_full_red_frame() {
        let frame = yuv422_frame(8, 4, 96, 90, 200);
        let red = detect_red_yuv422p(&frame, 8, 4, RedThreshold::default()).unwrap();
        assert_eq!(red.pixels, 32);
        assert_eq!((red.left, red.top, red.right, red.bottom), (0, 0, 7, 3));
        assert_eq!((red.center_x, red.center_y), (3, 1));
        assert_eq!(red.area_ratio(), 1.0);
        assert_eq!(red.bbox_width_ratio(), 1.0);
        assert_eq!(red.bbox_height_ratio(), 1.0);
    }

    #[test]
    fn reports_red_rectangle_geometry() {
        let width = 8;
        let height = 4;
        let y_size = width * height;
        let chroma_size = width / 2 * height;
        let mut frame = yuv422_frame(width, height, 96, 128, 128);
        for row in 1..3 {
            for pair in 1..3 {
                let index = row * (width / 2) + pair;
                frame[y_size + index] = 90;
                frame[y_size + chroma_size + index] = 200;
            }
        }

        let red = detect_red_yuv422p(&frame, width, height, RedThreshold::default()).unwrap();
        assert_eq!(red.pixels, 8);
        assert_eq!((red.left, red.top, red.right, red.bottom), (2, 1, 5, 2));
        assert_eq!((red.center_x, red.center_y), (3, 1));
        assert_eq!(red.area_ratio(), 0.25);
        assert_eq!(red.bbox_width_ratio(), 0.5);
        assert_eq!(red.bbox_height_ratio(), 0.5);
    }

    #[test]
    fn rejects_dark_red_chroma_noise() {
        let frame = yuv422_frame(8, 4, 10, 90, 200);
        assert!(detect_red_yuv422p(&frame, 8, 4, RedThreshold::default()).is_none());
    }
}
