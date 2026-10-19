//! Probe results and the pure geometry, rotation and timing maths shared by the backends.

/// One audio stream of a media file.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AudioTrack {
    pub index: usize,
    pub codec: Option<String>,
    pub channels: Option<u32>,
    pub sample_rate: Option<u32>,
    pub bitrate: Option<u64>,
    pub language: Option<String>,
}

/// Container and stream facts of a media file.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VideoInfo {
    pub container: Option<String>,
    pub size_bytes: Option<u64>,
    pub duration_secs: Option<f64>,
    pub bitrate: Option<u64>,
    pub creation_time: Option<String>,
    pub has_video: bool,
    pub video_index: Option<usize>,
    pub video_codec: Option<String>,
    pub pixel_format: Option<String>,
    /// Coded frame size before any rotation or pixel aspect correction.
    pub width: u32,
    pub height: u32,
    /// Sample aspect ratio, `None` for square pixels.
    pub sar: Option<(u32, u32)>,
    /// Degrees clockwise the stored frame is rotated for upright display: 0, 90, 180 or 270.
    pub rotation: u32,
    pub fps: Option<f64>,
    pub video_bitrate: Option<u64>,
    pub audio: Vec<AudioTrack>,
    pub subtitle_tracks: usize,
}

impl VideoInfo {
    /// Frame size as displayed: square pixels, after rotation.
    pub fn display_size(&self) -> (u32, u32) {
        let (w, h) = square_pixel_size(self.width, self.height, self.sar);
        rotated_size(w, h, self.rotation)
    }

    pub fn is_audio_only(&self) -> bool {
        !self.has_video && !self.audio.is_empty()
    }

    /// One-line summary for logs and the debug window.
    pub fn summary(&self) -> String {
        let (dw, dh) = self.display_size();
        let video = if self.has_video {
            format!("{} {}x{} rot {} {}", self.video_codec.as_deref().unwrap_or("?"), dw, dh, self.rotation, self.fps.map(|f| format!("{f:.2} fps")).unwrap_or_else(|| "? fps".into()))
        } else {
            "no video".to_string()
        };
        let dur = self.duration_secs.map(|d| format!("{d:.2}s")).unwrap_or_else(|| "?".into());
        format!("{video}, {dur}, {} audio", self.audio.len())
    }
}

/// Counterclockwise rotation in degrees stored in a display matrix, `None` for a degenerate matrix.
pub fn rotation_from_display_matrix(m: &[i32; 9]) -> Option<f64> {
    let fp = |v: i32| f64::from(v) / 65536.0;
    let s0 = fp(m[0]).hypot(fp(m[3]));
    let s1 = fp(m[1]).hypot(fp(m[4]));
    if s0 == 0.0 || s1 == 0.0 {
        return None;
    }
    let rotation = (fp(m[1]) / s1).atan2(fp(m[0]) / s0).to_degrees();
    Some(-rotation)
}

/// Nine native-endian i32 values of a display matrix side-data payload.
pub fn display_matrix_from_bytes(bytes: &[u8]) -> Option<[i32; 9]> {
    if bytes.len() < 36 {
        return None;
    }
    let mut m = [0i32; 9];
    for (i, slot) in m.iter_mut().enumerate() {
        *slot = i32::from_ne_bytes(bytes[i * 4..i * 4 + 4].try_into().ok()?);
    }
    Some(m)
}

/// Clockwise display rotation (0, 90, 180, 270) for a counterclockwise angle as ffmpeg reports it.
pub fn clockwise_rotation(ccw_degrees: f64) -> u32 {
    if !ccw_degrees.is_finite() {
        return 0;
    }
    let cw = (-ccw_degrees).rem_euclid(360.0);
    (((cw / 90.0).round() as i64).rem_euclid(4) * 90) as u32
}

/// Clockwise rotation from the legacy `rotate` stream tag, which is already clockwise.
pub fn rotation_from_tag(tag: &str) -> Option<u32> {
    let degrees: f64 = tag.trim().parse().ok()?;
    degrees.is_finite().then(|| (((degrees.rem_euclid(360.0) / 90.0).round() as i64).rem_euclid(4) * 90) as u32)
}

/// Sample aspect ratio as a reduced pair, `None` when absent, invalid or square.
pub fn normalize_sar(num: i64, den: i64) -> Option<(u32, u32)> {
    if num <= 0 || den <= 0 || num == den || num > i64::from(u32::MAX) || den > i64::from(u32::MAX) {
        return None;
    }
    let g = gcd(num as u64, den as u64);
    Some(((num as u64 / g) as u32, (den as u64 / g) as u32))
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a.max(1) } else { gcd(b, a % b) }
}

/// Parses `a:b` or `a/b` into a normalized sample aspect ratio.
pub fn parse_sar(text: &str) -> Option<(u32, u32)> {
    let (n, d) = text.split_once([':', '/'])?;
    normalize_sar(n.trim().parse().ok()?, d.trim().parse().ok()?)
}

/// Parses `30000/1001` or `25` into a positive rate.
pub fn parse_rate(text: &str) -> Option<f64> {
    let rate = match text.split_once('/') {
        Some((n, d)) => {
            let (n, d): (f64, f64) = (n.trim().parse().ok()?, d.trim().parse().ok()?);
            if d == 0.0 { return None; }
            n / d
        }
        None => text.trim().parse().ok()?,
    };
    (rate.is_finite() && rate > 0.0).then_some(rate)
}

/// Frame size with the pixel aspect ratio applied by widening or narrowing the width.
pub fn square_pixel_size(w: u32, h: u32, sar: Option<(u32, u32)>) -> (u32, u32) {
    match sar {
        Some((n, d)) if w > 0 && h > 0 => (((u64::from(w) * u64::from(n) + u64::from(d) / 2) / u64::from(d)).max(1) as u32, h),
        _ => (w, h),
    }
}

pub fn rotated_size(w: u32, h: u32, rotation: u32) -> (u32, u32) {
    if rotation % 180 == 90 { (h, w) } else { (w, h) }
}

/// Output size of a poster: what the scaler produces before rotation and what the image has after it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub scaled_w: u32,
    pub scaled_h: u32,
    pub out_w: u32,
    pub out_h: u32,
    pub rotation: u32,
}

/// Geometry that fits a frame, after pixel aspect and rotation, inside `max_dim` without upscaling.
pub fn thumb_geometry(w: u32, h: u32, sar: Option<(u32, u32)>, rotation: u32, max_dim: u32) -> Option<Geometry> {
    if w == 0 || h == 0 || max_dim == 0 {
        return None;
    }
    let (sw, sh) = square_pixel_size(w, h, sar);
    let (dw, dh) = rotated_size(sw, sh, rotation);
    let scale = (f64::from(max_dim) / f64::from(dw.max(dh))).min(1.0);
    let fit = |v: u32| ((f64::from(v) * scale).round() as u32).max(1);
    let (scaled_w, scaled_h) = (fit(sw), fit(sh));
    let (out_w, out_h) = rotated_size(scaled_w, scaled_h, rotation);
    Some(Geometry { scaled_w, scaled_h, out_w, out_h, rotation })
}

/// Seek positions as fractions of the duration: the preferred one first, then spread-out alternatives.
pub fn attempt_fractions(first: f64, attempts: usize) -> Vec<f64> {
    const FALLBACKS: [f64; 6] = [0.35, 0.6, 0.03, 0.8, 0.5, 0.2];
    let first = if first.is_finite() { first.clamp(0.0, 0.95) } else { 0.1 };
    let mut out = vec![first];
    for f in FALLBACKS {
        if out.len() >= attempts.max(1) {
            break;
        }
        if out.iter().all(|e| (e - f).abs() > 0.04) {
            out.push(f);
        }
    }
    out.truncate(attempts.max(1));
    out
}

/// Seek target in microseconds for a fraction of a duration given in microseconds.
pub fn seek_target_micros(duration_us: i64, fraction: f64) -> i64 {
    if duration_us <= 0 {
        return 0;
    }
    (duration_us as f64 * fraction.clamp(0.0, 0.95)) as i64
}

/// Parses an ffmpeg `HH:MM:SS.xx` timestamp into seconds.
pub fn parse_clock(text: &str) -> Option<f64> {
    let mut parts = text.trim().split(':');
    let (h, m, s) = (parts.next()?, parts.next()?, parts.next()?);
    if parts.next().is_some() {
        return None;
    }
    let secs = h.parse::<f64>().ok()? * 3600.0 + m.parse::<f64>().ok()? * 60.0 + s.parse::<f64>().ok()?;
    secs.is_finite().then_some(secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(angle_ccw: f64) -> [i32; 9] {
        let (s, c) = angle_ccw.to_radians().sin_cos();
        let fx = |v: f64| (v * 65536.0).round() as i32;
        [fx(c), fx(-s), 0, fx(s), fx(c), 0, 0, 0, 1 << 30]
    }

    #[test]
    fn display_matrix_rotation_matches_ffmpeg_sign_convention() {
        let identity = matrix(0.0);
        assert!(rotation_from_display_matrix(&identity).unwrap().abs() < 1e-6);
        assert!((rotation_from_display_matrix(&matrix(90.0)).unwrap() - 90.0).abs() < 0.01);
        assert!((rotation_from_display_matrix(&matrix(-90.0)).unwrap() + 90.0).abs() < 0.01);
        assert!((rotation_from_display_matrix(&matrix(180.0)).unwrap().abs() - 180.0).abs() < 0.01);
        assert_eq!(rotation_from_display_matrix(&[0; 9]), None);
    }

    #[test]
    fn clockwise_rotation_snaps_and_flips_the_sign() {
        assert_eq!(clockwise_rotation(0.0), 0);
        assert_eq!(clockwise_rotation(-90.0), 90);
        assert_eq!(clockwise_rotation(90.0), 270);
        assert_eq!(clockwise_rotation(180.0), 180);
        assert_eq!(clockwise_rotation(-180.0), 180);
        assert_eq!(clockwise_rotation(-89.7), 90);
        assert_eq!(clockwise_rotation(f64::NAN), 0);
        assert_eq!(clockwise_rotation(-270.0), 270);
    }

    #[test]
    fn legacy_rotate_tag_is_clockwise() {
        assert_eq!(rotation_from_tag("90"), Some(90));
        assert_eq!(rotation_from_tag("270\n"), Some(270));
        assert_eq!(rotation_from_tag("-90"), Some(270));
        assert_eq!(rotation_from_tag("abc"), None);
    }

    #[test]
    fn display_matrix_bytes_need_nine_words() {
        let m = matrix(90.0);
        let bytes: Vec<u8> = m.iter().flat_map(|v| v.to_ne_bytes()).collect();
        assert_eq!(display_matrix_from_bytes(&bytes), Some(m));
        assert_eq!(display_matrix_from_bytes(&bytes[..35]), None);
    }

    #[test]
    fn sar_is_reduced_and_square_is_none() {
        assert_eq!(normalize_sar(1, 1), None);
        assert_eq!(normalize_sar(0, 1), None);
        assert_eq!(normalize_sar(4, -3), None);
        assert_eq!(normalize_sar(32, 27), Some((32, 27)));
        assert_eq!(normalize_sar(64, 54), Some((32, 27)));
        assert_eq!(parse_sar("16:11"), Some((16, 11)));
        assert_eq!(parse_sar("1:1"), None);
        assert_eq!(parse_sar("0:1"), None);
        assert_eq!(parse_sar("N/A"), None);
    }

    #[test]
    fn rates_parse_from_ratios_and_plain_numbers() {
        assert!((parse_rate("30000/1001").unwrap() - 29.97).abs() < 0.01);
        assert_eq!(parse_rate("25"), Some(25.0));
        assert_eq!(parse_rate("0/0"), None);
        assert_eq!(parse_rate("0/1"), None);
        assert_eq!(parse_rate("x"), None);
    }

    #[test]
    fn square_pixel_size_widens_by_the_aspect_ratio() {
        assert_eq!(square_pixel_size(720, 576, None), (720, 576));
        assert_eq!(square_pixel_size(720, 576, Some((16, 15))), (768, 576));
        assert_eq!(square_pixel_size(1440, 1080, Some((4, 3))), (1920, 1080));
        assert_eq!(square_pixel_size(0, 10, Some((4, 3))), (0, 10));
    }

    #[test]
    fn geometry_fits_rotated_anamorphic_frames_without_upscaling() {
        let g = thumb_geometry(1920, 1080, None, 0, 256).unwrap();
        assert_eq!((g.scaled_w, g.scaled_h, g.out_w, g.out_h), (256, 144, 256, 144));
        let g = thumb_geometry(1920, 1080, None, 90, 256).unwrap();
        assert_eq!((g.scaled_w, g.scaled_h, g.out_w, g.out_h), (256, 144, 144, 256));
        let g = thumb_geometry(1440, 1080, Some((4, 3)), 0, 256).unwrap();
        assert_eq!((g.out_w, g.out_h), (256, 144));
        let g = thumb_geometry(160, 90, None, 0, 256).unwrap();
        assert_eq!((g.out_w, g.out_h), (160, 90));
        let g = thumb_geometry(3, 5000, None, 0, 256).unwrap();
        assert_eq!((g.out_w, g.out_h), (1, 256));
        assert_eq!(thumb_geometry(0, 10, None, 0, 256), None);
        assert_eq!(thumb_geometry(10, 10, None, 0, 0), None);
    }

    #[test]
    fn attempt_fractions_start_with_the_preference_and_stay_distinct() {
        assert_eq!(attempt_fractions(0.1, 1), vec![0.1]);
        let f = attempt_fractions(0.1, 4);
        assert_eq!(f.len(), 4);
        assert_eq!(f[0], 0.1);
        for (i, a) in f.iter().enumerate() {
            assert!((0.0..=0.95).contains(a));
            assert!(f[i + 1..].iter().all(|b| (a - b).abs() > 0.03));
        }
        assert_eq!(attempt_fractions(f64::NAN, 2)[0], 0.1);
        assert_eq!(attempt_fractions(2.0, 1), vec![0.95]);
        assert_eq!(attempt_fractions(0.35, 2)[1], 0.6);
        assert_eq!(attempt_fractions(0.5, 0).len(), 1);
    }

    #[test]
    fn seek_target_clamps_to_the_duration() {
        assert_eq!(seek_target_micros(10_000_000, 0.1), 1_000_000);
        assert_eq!(seek_target_micros(10_000_000, 5.0), 9_500_000);
        assert_eq!(seek_target_micros(0, 0.5), 0);
        assert_eq!(seek_target_micros(-1, 0.5), 0);
    }

    #[test]
    fn clock_timestamps_parse() {
        assert_eq!(parse_clock("00:01:30.50"), Some(90.5));
        assert_eq!(parse_clock("01:00:00.00"), Some(3600.0));
        assert_eq!(parse_clock("N/A"), None);
        assert_eq!(parse_clock("1:2"), None);
    }

    #[test]
    fn display_size_applies_aspect_then_rotation() {
        let info = VideoInfo { has_video: true, width: 720, height: 576, sar: Some((16, 15)), rotation: 90, ..Default::default() };
        assert_eq!(info.display_size(), (576, 768));
        assert!(info.summary().contains("576x768"));
        let audio = VideoInfo { audio: vec![AudioTrack::default()], ..Default::default() };
        assert!(audio.is_audio_only());
    }
}
