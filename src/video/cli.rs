//! Poster frames through the ffmpeg command: one process per attempted position.

use super::info;
use super::poster::{Candidate, PosterOptions, Selector};
use super::{Backend, Deadline, Poster, VideoError, ffmpeg_path, proc, probe_cli};
use image::RgbImage;
use std::path::Path;
use std::time::Instant;

const MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

/// Video filter that applies pixel aspect, then fits the frame inside `max_dim` without upscaling.
pub fn scale_filter(max_dim: u32) -> String {
    format!("scale=iw*sar:ih,setsar=1,scale=w=min({max_dim}\\,iw):h=min({max_dim}\\,ih):force_original_aspect_ratio=decrease:flags=area")
}

/// Arguments that decode one frame at `seek_secs` and write it as a binary PPM on stdout.
pub fn frame_args(path: &Path, seek_secs: f64, stream_index: Option<usize>, max_dim: u32) -> Vec<String> {
    let mut args: Vec<String> = ["-nostdin", "-hide_banner", "-loglevel", "error", "-noaccurate_seek"].iter().map(|s| s.to_string()).collect();
    if seek_secs > 0.0 {
        args.extend(["-ss".into(), format!("{seek_secs:.3}")]);
    }
    args.extend(["-i".into(), proc::input_arg(path)]);
    if let Some(i) = stream_index {
        args.extend(["-map".into(), format!("0:{i}")]);
    }
    args.extend(["-an", "-sn", "-dn", "-frames:v", "1", "-vf"].iter().map(|s| s.to_string()));
    args.push(scale_filter(max_dim));
    args.extend(["-pix_fmt", "rgb24", "-f", "image2pipe", "-c:v", "ppm", "-"].iter().map(|s| s.to_string()));
    args
}

/// Decodes a binary PPM (P6, 8 bit) into an RGB image.
pub fn parse_ppm(bytes: &[u8]) -> Result<RgbImage, VideoError> {
    let bad = |why: &str| VideoError::Corrupt(format!("ppm: {why}"));
    let mut pos = 0usize;
    let mut token = || -> Option<&str> {
        while bytes.get(pos).is_some_and(|b| b.is_ascii_whitespace()) {
            pos += 1;
        }
        let start = pos;
        while bytes.get(pos).is_some_and(|b| !b.is_ascii_whitespace()) {
            pos += 1;
        }
        std::str::from_utf8(&bytes[start..pos]).ok().filter(|t| !t.is_empty())
    };
    if token() != Some("P6") {
        return Err(bad("not a P6 image"));
    }
    let w: u32 = token().and_then(|t| t.parse().ok()).ok_or_else(|| bad("bad width"))?;
    let h: u32 = token().and_then(|t| t.parse().ok()).ok_or_else(|| bad("bad height"))?;
    if token() != Some("255") {
        return Err(bad("unsupported depth"));
    }
    let data = bytes.get(pos + 1..).ok_or_else(|| bad("missing pixels"))?;
    let need = w as usize * h as usize * 3;
    if w == 0 || h == 0 || data.len() < need {
        return Err(bad("truncated pixels"));
    }
    RgbImage::from_raw(w, h, data[..need].to_vec()).ok_or_else(|| bad("size mismatch"))
}

fn grab_frame(ffmpeg: &Path, path: &Path, seek: f64, index: Option<usize>, max_dim: u32, deadline: &Deadline) -> Result<RgbImage, VideoError> {
    let mut cmd = proc::command(ffmpeg);
    cmd.args(frame_args(path, seek, index, max_dim));
    let out = proc::run_capture(cmd, deadline, MAX_FRAME_BYTES)?;
    if !out.success {
        return Err(VideoError::Corrupt(out.stderr.lines().next().unwrap_or("ffmpeg failed").to_string()));
    }
    parse_ppm(&out.stdout)
}

/// Decodes one poster with the ffmpeg command.
pub fn poster(path: &Path, size: u64, opts: &PosterOptions, deadline: &Deadline) -> Result<Poster, VideoError> {
    let started = Instant::now();
    let ffmpeg = ffmpeg_path().ok_or(VideoError::NoBackend)?;
    let info = probe_cli::probe(path, size, deadline)?;
    if !info.has_video {
        return Err(VideoError::NoVideoStream);
    }
    let duration = info.duration_secs.filter(|d| *d > 0.0);
    let mut selector = Selector::new(opts.attempts);
    let mut last_error = None;
    let mut offer = |selector: &mut Selector, seek: f64| -> Result<bool, VideoError> {
        match grab_frame(ffmpeg, path, seek, info.video_index, opts.max_dim, deadline) {
            Ok(image) => Ok(selector.offer(Candidate { image, position_secs: duration.map(|_| seek) })),
            Err(e @ (VideoError::Timeout | VideoError::Cancelled)) => Err(e),
            Err(e) => {
                last_error = Some(e);
                Ok(false)
            }
        }
    };
    for (i, fraction) in info::attempt_fractions(opts.at_fraction, opts.attempts as usize).into_iter().enumerate() {
        deadline.check()?;
        if duration.is_none() && i > 0 {
            break;
        }
        if offer(&mut selector, duration.map_or(0.0, |d| d * fraction))? {
            break;
        }
    }
    if selector.attempts() == 0 && duration.is_some() {
        offer(&mut selector, 0.0)?;
    }
    selector.finish(&info, Backend::Cli, started.elapsed()).ok_or_else(|| last_error.unwrap_or_else(|| VideoError::Corrupt("no frame decoded".into())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_escapes_commas_for_the_filtergraph() {
        let f = scale_filter(256);
        assert!(f.contains("min(256\\,iw)") && f.contains("min(256\\,ih)"));
        assert!(f.starts_with("scale=iw*sar:ih,setsar=1,"));
    }

    #[test]
    fn frame_args_seek_only_when_positive_and_map_the_stream() {
        let a = frame_args(Path::new("/v/a.mp4"), 0.0, None, 256);
        assert!(!a.contains(&"-ss".to_string()) && !a.contains(&"-map".to_string()));
        assert!(a.contains(&"-noaccurate_seek".to_string()) && a.contains(&"file:/v/a.mp4".to_string()));
        let b = frame_args(Path::new("/v/a.mp4"), 1.5, Some(2), 128);
        let ss = b.iter().position(|x| x == "-ss").unwrap();
        assert_eq!(b[ss + 1], "1.500");
        assert!(ss < b.iter().position(|x| x == "-i").unwrap());
        assert_eq!(b[b.iter().position(|x| x == "-map").unwrap() + 1], "0:2");
        assert_eq!(b.last().map(String::as_str), Some("-"));
    }

    #[test]
    fn ppm_round_trips() {
        let mut bytes = b"P6\n2 1\n255\n".to_vec();
        bytes.extend_from_slice(&[1, 2, 3, 4, 5, 6]);
        let img = parse_ppm(&bytes).unwrap();
        assert_eq!(img.dimensions(), (2, 1));
        assert_eq!(img.get_pixel(1, 0).0, [4, 5, 6]);
    }

    #[test]
    fn ppm_errors_are_typed() {
        assert!(matches!(parse_ppm(b""), Err(VideoError::Corrupt(_))));
        assert!(matches!(parse_ppm(b"P5\n1 1\n255\nx"), Err(VideoError::Corrupt(_))));
        assert!(matches!(parse_ppm(b"P6\n2 2\n255\n123"), Err(VideoError::Corrupt(_))));
        assert!(matches!(parse_ppm(b"P6\n0 2\n255\n"), Err(VideoError::Corrupt(_))));
        assert!(matches!(parse_ppm(b"P6\n1 1\n65535\nabcdef"), Err(VideoError::Corrupt(_))));
    }
}
