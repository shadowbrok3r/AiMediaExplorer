//! Test media generated with the ffmpeg command; every maker returns `None` when the tool or an encoder is missing.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Whether both ffmpeg and ffprobe commands exist.
pub fn available() -> bool {
    super::ffmpeg_path().is_some() && super::ffprobe_path().is_some()
}

/// Prints the skip notice tests use when fixtures cannot be made.
pub fn skip(test: &str) {
    eprintln!("skipping {test}: ffmpeg command line tools or an encoder are not available");
}

/// Runs the ffmpeg command with quiet logging; true when it succeeded.
pub fn ffmpeg(args: &[&str]) -> bool {
    let Some(tool) = super::ffmpeg_path() else { return false };
    Command::new(tool)
        .args(["-y", "-nostdin", "-hide_banner", "-loglevel", "error"])
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

fn path_str(p: &Path) -> &str {
    p.to_str().expect("fixture paths are utf-8")
}

fn make(dir: &Path, name: &str, args: &[&str]) -> Option<PathBuf> {
    let out = dir.join(name);
    let mut full: Vec<&str> = args.to_vec();
    full.push(path_str(&out));
    ffmpeg(&full).then_some(out).filter(|p| p.metadata().is_ok_and(|m| m.len() > 0))
}

fn cat<'a>(parts: &[&[&'a str]]) -> Vec<&'a str> {
    parts.iter().flat_map(|p| p.iter().copied()).collect()
}

const X264: [&str; 6] = ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"];

/// 320x180 test pattern with a sine track, `secs` long, in an mp4 with the index up front.
pub fn h264_mp4(dir: &Path, name: &str, secs: u32) -> Option<PathBuf> {
    let src = format!("testsrc=size=320x180:rate=25:duration={secs}");
    let tone = format!("sine=frequency=440:duration={secs}");
    make(dir, name, &cat(&[&["-f", "lavfi", "-i", &src, "-f", "lavfi", "-i", &tone], &X264, &["-g", "25", "-c:a", "aac", "-shortest", "-movflags", "+faststart"]]))
}

/// A black frame with a red square in its top-left corner, used to check orientation.
pub fn marker_mp4(dir: &Path, name: &str) -> Option<PathBuf> {
    let src = "color=c=black:s=320x180:r=25:d=2,drawbox=x=0:y=0:w=80:h=80:color=red:t=fill";
    make(dir, name, &cat(&[&["-f", "lavfi", "-i", src], &X264, &["-movflags", "+faststart"]]))
}

/// The marker clip with a display matrix of `ccw_degrees` (ffmpeg's `-display_rotation`).
pub fn rotated(dir: &Path, name: &str, ccw_degrees: i32) -> Option<PathBuf> {
    let base = marker_mp4(dir, &format!("base_for_{name}"))?;
    let angle = ccw_degrees.to_string();
    make(dir, name, &["-display_rotation", &angle, "-i", path_str(&base), "-c", "copy"])
}

/// 320x180 coded frames flagged with a 4:3 sample aspect ratio.
pub fn anamorphic(dir: &Path, name: &str) -> Option<PathBuf> {
    make(dir, name, &cat(&[&["-f", "lavfi", "-i", "testsrc=size=320x180:rate=25:duration=2", "-vf", "setsar=4/3"], &X264]))
}

/// Three seconds of black followed by seven seconds of test pattern.
pub fn dark_intro(dir: &Path, name: &str) -> Option<PathBuf> {
    make(
        dir,
        name,
        &cat(&[
            &["-f", "lavfi", "-i", "color=c=black:s=320x180:r=25:d=3", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=25:duration=7"],
            &["-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]"],
            &X264,
            &["-g", "25"],
        ]),
    )
}

pub fn all_black(dir: &Path, name: &str) -> Option<PathBuf> {
    make(dir, name, &cat(&[&["-f", "lavfi", "-i", "color=c=black:s=320x180:r=25:d=3"], &X264]))
}

/// Frames dropped irregularly so the stream has variable frame timing.
pub fn variable_rate(dir: &Path, name: &str) -> Option<PathBuf> {
    make(dir, name, &cat(&[&["-f", "lavfi", "-i", "testsrc=size=320x180:rate=30:duration=6", "-vf", "select='not(mod(n,3))+gte(n,100)'", "-fps_mode", "vfr"], &X264]))
}

/// One hour of 160x90 frames at 1 fps with a keyframe every ten frames.
pub fn one_hour(dir: &Path, name: &str) -> Option<PathBuf> {
    make(dir, name, &cat(&[&["-f", "lavfi", "-i", "testsrc=size=160x90:rate=1:duration=3600"], &X264, &["-g", "10", "-movflags", "+faststart"]]))
}

/// Two 8K frames.
pub fn eight_k(dir: &Path, name: &str) -> Option<PathBuf> {
    let src = "color=c=0x305070:s=7680x4320:r=1:d=2,drawbox=x=400:y=300:w=3000:h=2000:color=orange:t=fill";
    make(dir, name, &cat(&[&["-f", "lavfi", "-i", src], &X264, &["-level", "6.2"]]))
}

/// A test pattern of `size` (for example `3840x2160`) lasting `secs` seconds.
pub fn sized(dir: &Path, name: &str, size: &str, secs: u32) -> Option<PathBuf> {
    let src = format!("testsrc2=size={size}:rate=25:duration={secs}");
    make(dir, name, &cat(&[&["-f", "lavfi", "-i", &src], &X264, &["-g", "25"]]))
}

pub fn audio_only(dir: &Path, name: &str) -> Option<PathBuf> {
    make(dir, name, &["-f", "lavfi", "-i", "sine=frequency=330:duration=2", "-c:a", "aac"])
}

/// Audio with a still cover picture attached as a video stream.
pub fn audio_with_cover(dir: &Path, name: &str) -> Option<PathBuf> {
    let audio = audio_only(dir, &format!("audio_for_{name}.m4a"))?;
    let cover = dir.join(format!("cover_for_{name}.png"));
    if !ffmpeg(&["-f", "lavfi", "-i", "testsrc=size=200x200:rate=1:duration=1", "-frames:v", "1", path_str(&cover)]) {
        return None;
    }
    make(dir, name, &["-i", path_str(&audio), "-i", path_str(&cover), "-map", "0:a", "-map", "1:v", "-c:a", "copy", "-c:v", "mjpeg", "-disposition:v:0", "attached_pic"])
}

/// One short clip per container and codec pairing worth covering: `(file name, ffmpeg codec arguments)`.
pub const CODEC_MATRIX: &[(&str, &[&str])] = &[
    ("c_h264.mp4", &["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]),
    ("c_h264.mkv", &["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]),
    ("c_vp8.webm", &["-c:v", "libvpx", "-deadline", "realtime", "-cpu-used", "8", "-b:v", "500k"]),
    ("c_vp9.webm", &["-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8", "-b:v", "500k"]),
    ("c_mpeg4.mov", &["-c:v", "mpeg4", "-q:v", "5"]),
    ("c_prores.mov", &["-c:v", "prores_ks", "-profile:v", "0", "-pix_fmt", "yuv422p10le"]),
    ("c_mjpeg.avi", &["-c:v", "mjpeg", "-q:v", "6", "-pix_fmt", "yuvj420p"]),
    ("c_mpeg2.mpg", &["-c:v", "mpeg2video", "-q:v", "5"]),
    ("c_flv.flv", &["-c:v", "flv", "-q:v", "5"]),
    ("c_wmv2.wmv", &["-c:v", "wmv2", "-q:v", "5"]),
    ("c_hevc.mkv", &["-c:v", "libx265", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-x265-params", "log-level=none"]),
    ("c_h264.ts", &["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-f", "mpegts"]),
    ("c_gif.gif", &["-c:v", "gif"]),
];

/// A two second 320x180 test pattern encoded as listed in [`CODEC_MATRIX`].
pub fn coded(dir: &Path, name: &str, codec_args: &[&str]) -> Option<PathBuf> {
    make(dir, name, &cat(&[&["-f", "lavfi", "-i", "testsrc=size=320x180:rate=25:duration=2"], codec_args]))
}

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *seed >> 33
}

pub fn garbage(dir: &Path, name: &str, len: usize) -> PathBuf {
    let mut seed = 7u64;
    let bytes: Vec<u8> = (0..len).map(|_| lcg(&mut seed) as u8).collect();
    let out = dir.join(name);
    std::fs::write(&out, bytes).expect("write garbage fixture");
    out
}

/// The first `fraction` of a file's bytes.
pub fn truncated(dir: &Path, name: &str, source: &Path, fraction: f64) -> PathBuf {
    let bytes = std::fs::read(source).expect("read source fixture");
    let keep = ((bytes.len() as f64 * fraction) as usize).clamp(1, bytes.len());
    let out = dir.join(name);
    std::fs::write(&out, &bytes[..keep]).expect("write truncated fixture");
    out
}

/// A copy with `flips` bytes overwritten after the first `skip` bytes.
pub fn corrupted(dir: &Path, name: &str, source: &Path, skip: usize, flips: usize, seed: u64) -> PathBuf {
    let mut bytes = std::fs::read(source).expect("read source fixture");
    let mut s = seed;
    let span = bytes.len().saturating_sub(skip).max(1);
    for _ in 0..flips {
        let at = skip + (lcg(&mut s) as usize % span);
        if let Some(b) = bytes.get_mut(at) {
            *b = lcg(&mut s) as u8;
        }
    }
    let out = dir.join(name);
    std::fs::write(&out, bytes).expect("write corrupted fixture");
    out
}
