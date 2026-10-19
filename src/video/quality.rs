//! Frame quality checks that decide whether a decoded frame is a usable poster.

use image::RgbImage;

/// Luma statistics of a frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameStats {
    pub mean: f32,
    pub std_dev: f32,
}

/// Mean and standard deviation of the luma over every pixel.
pub fn frame_stats(img: &RgbImage) -> FrameStats {
    let n = u64::from(img.width()) * u64::from(img.height());
    if n == 0 {
        return FrameStats { mean: 0.0, std_dev: 0.0 };
    }
    let (mut sum, mut sum_sq) = (0u64, 0u64);
    for p in img.pixels() {
        let [r, g, b] = p.0;
        let y = (u32::from(r) * 77 + u32::from(g) * 150 + u32::from(b) * 29) >> 8;
        sum += u64::from(y);
        sum_sq += u64::from(y * y);
    }
    let mean = sum as f64 / n as f64;
    let var = (sum_sq as f64 / n as f64 - mean * mean).max(0.0);
    FrameStats { mean: mean as f32, std_dev: var.sqrt() as f32 }
}

/// Near-uniform or near-black frames, such as fade-ins and slates.
pub fn is_blank(s: &FrameStats) -> bool {
    s.std_dev < 3.0 || (s.mean < 16.0 && s.std_dev < 12.0)
}

/// Higher is better when ranking blank candidates.
pub fn score(s: &FrameStats) -> f32 {
    s.std_dev + 0.1 * s.mean
}

/// Rotates a frame clockwise by 0, 90, 180 or 270 degrees.
pub fn rotate_upright(img: RgbImage, rotation: u32) -> RgbImage {
    match rotation % 360 {
        90 => image::imageops::rotate90(&img),
        180 => image::imageops::rotate180(&img),
        270 => image::imageops::rotate270(&img),
        _ => img,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    fn flat(w: u32, h: u32, v: u8) -> RgbImage {
        RgbImage::from_pixel(w, h, Rgb([v, v, v]))
    }

    #[test]
    fn uniform_frames_are_blank_at_any_brightness() {
        for v in [0u8, 10, 128, 255] {
            assert!(is_blank(&frame_stats(&flat(32, 18, v))), "{v}");
        }
    }

    #[test]
    fn detailed_frames_are_not_blank() {
        let img = RgbImage::from_fn(64, 36, |x, y| if (x / 4 + y / 4) % 2 == 0 { Rgb([200, 200, 200]) } else { Rgb([40, 60, 80]) });
        let s = frame_stats(&img);
        assert!(!is_blank(&s), "{s:?}");
        assert!(s.std_dev > 50.0);
    }

    #[test]
    fn dark_scenes_with_content_are_kept() {
        let img = RgbImage::from_fn(64, 36, |x, _| if x < 32 { Rgb([10, 10, 10]) } else { Rgb([90, 90, 90]) });
        assert!(!is_blank(&frame_stats(&img)));
    }

    #[test]
    fn faint_noise_on_black_is_blank() {
        let img = RgbImage::from_fn(64, 36, |x, y| {
            let v = ((x * 7 + y * 13) % 5) as u8;
            Rgb([v, v, v])
        });
        assert!(is_blank(&frame_stats(&img)));
    }

    #[test]
    fn empty_images_have_zero_stats() {
        let s = frame_stats(&RgbImage::new(0, 0));
        assert_eq!((s.mean, s.std_dev), (0.0, 0.0));
    }

    #[test]
    fn score_prefers_contrast_then_brightness() {
        let flat_bright = frame_stats(&flat(8, 8, 200));
        let contrasty = FrameStats { mean: 50.0, std_dev: 60.0 };
        assert!(score(&contrasty) > score(&flat_bright));
        assert!(score(&FrameStats { mean: 90.0, std_dev: 5.0 }) > score(&FrameStats { mean: 10.0, std_dev: 5.0 }));
    }

    #[test]
    fn rotation_turns_pixels_clockwise() {
        let mut img = RgbImage::new(4, 2);
        img.put_pixel(0, 0, Rgb([255, 0, 0]));
        let r90 = rotate_upright(img.clone(), 90);
        assert_eq!(r90.dimensions(), (2, 4));
        assert_eq!(r90.get_pixel(1, 0), &Rgb([255, 0, 0]));
        let r180 = rotate_upright(img.clone(), 180);
        assert_eq!(r180.get_pixel(3, 1), &Rgb([255, 0, 0]));
        let r270 = rotate_upright(img.clone(), 270);
        assert_eq!(r270.get_pixel(0, 3), &Rgb([255, 0, 0]));
        assert_eq!(rotate_upright(img.clone(), 0), img);
    }
}
