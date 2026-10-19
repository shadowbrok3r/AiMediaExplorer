//! Media probing through the ffprobe command, with a banner-parsing fallback when only ffmpeg is installed.

use super::info::{self, AudioTrack, VideoInfo};
use super::{Deadline, VideoError, ffmpeg_path, ffprobe_path, proc};
use serde_json::Value;
use std::path::Path;

const MAX_PROBE_OUTPUT: usize = 8 * 1024 * 1024;

fn text(v: &Value, key: &str) -> Option<String> {
    v.get(key).and_then(Value::as_str).filter(|s| !s.is_empty() && *s != "N/A").map(str::to_string)
}

fn number(v: &Value, key: &str) -> Option<f64> {
    match v.get(key)? {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

fn unsigned(v: &Value, key: &str) -> Option<u64> {
    number(v, key).filter(|n| *n >= 0.0 && n.is_finite()).map(|n| n as u64)
}

fn tag(v: &Value, key: &str) -> Option<String> {
    let tags = v.get("tags")?;
    tags.as_object()?.iter().find(|(k, _)| k.eq_ignore_ascii_case(key)).and_then(|(_, val)| val.as_str()).filter(|s| !s.is_empty()).map(str::to_string)
}

fn flag(v: &Value, section: &str, key: &str) -> bool {
    v.get(section).and_then(|d| d.get(key)).and_then(Value::as_i64).is_some_and(|n| n != 0)
}

fn stream_rotation(stream: &Value) -> u32 {
    let from_side_data = stream
        .get("side_data_list")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .find_map(|sd| sd.get("rotation").and_then(|r| r.as_f64().or_else(|| r.as_str().and_then(|s| s.parse().ok()))))
        .map(info::clockwise_rotation);
    from_side_data.or_else(|| tag(stream, "rotate").and_then(|t| info::rotation_from_tag(&t))).unwrap_or(0)
}

/// Stream facts from `ffprobe -print_format json -show_format -show_streams`.
pub fn parse_ffprobe_json(json: &str, size_bytes: Option<u64>) -> Result<VideoInfo, VideoError> {
    let root: Value = serde_json::from_str(json).map_err(|e| VideoError::Backend(format!("ffprobe output: {e}")))?;
    let streams = root.get("streams").and_then(Value::as_array).cloned().unwrap_or_default();
    let format = root.get("format").cloned().unwrap_or(Value::Null);
    if streams.is_empty() && format.is_null() {
        return Err(VideoError::Corrupt("no streams found".into()));
    }
    let mut out = VideoInfo {
        container: text(&format, "format_name"),
        size_bytes: size_bytes.or_else(|| unsigned(&format, "size")),
        duration_secs: number(&format, "duration").filter(|d| d.is_finite() && *d >= 0.0),
        bitrate: unsigned(&format, "bit_rate"),
        creation_time: tag(&format, "creation_time"),
        ..VideoInfo::default()
    };
    let mut best_video: Option<(i64, &Value)> = None;
    for s in &streams {
        match s.get("codec_type").and_then(Value::as_str) {
            Some("video") if !flag(s, "disposition", "attached_pic") => {
                let area = unsigned(s, "width").unwrap_or(0) as i64 * unsigned(s, "height").unwrap_or(0) as i64;
                let rank = area + if flag(s, "disposition", "default") { 1 << 40 } else { 0 };
                if best_video.is_none_or(|(r, _)| rank > r) {
                    best_video = Some((rank, s));
                }
            }
            Some("audio") => out.audio.push(AudioTrack {
                index: unsigned(s, "index").unwrap_or(0) as usize,
                codec: text(s, "codec_name"),
                channels: unsigned(s, "channels").map(|c| c as u32),
                sample_rate: unsigned(s, "sample_rate").map(|r| r as u32),
                bitrate: unsigned(s, "bit_rate"),
                language: tag(s, "language").filter(|l| l != "und"),
            }),
            Some("subtitle") => out.subtitle_tracks += 1,
            _ => {}
        }
    }
    if let Some((_, v)) = best_video {
        out.has_video = true;
        out.video_index = unsigned(v, "index").map(|i| i as usize);
        out.video_codec = text(v, "codec_name");
        out.pixel_format = text(v, "pix_fmt");
        out.width = unsigned(v, "width").unwrap_or(0) as u32;
        out.height = unsigned(v, "height").unwrap_or(0) as u32;
        out.sar = text(v, "sample_aspect_ratio").and_then(|s| info::parse_sar(&s));
        out.rotation = stream_rotation(v);
        out.fps = text(v, "avg_frame_rate").and_then(|r| info::parse_rate(&r)).or_else(|| text(v, "r_frame_rate").and_then(|r| info::parse_rate(&r)));
        out.video_bitrate = unsigned(v, "bit_rate");
        out.creation_time = out.creation_time.or_else(|| tag(v, "creation_time"));
        if out.duration_secs.is_none() {
            out.duration_secs = number(v, "duration").filter(|d| d.is_finite() && *d >= 0.0);
        }
    }
    Ok(out)
}

/// Duration and a video marker from the banner `ffmpeg -i` prints on stderr.
pub fn parse_ffmpeg_banner(stderr: &str) -> VideoInfo {
    let mut out = VideoInfo::default();
    for line in stderr.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("Duration:") {
            let clock = rest.split(',').next().unwrap_or("").trim();
            out.duration_secs = info::parse_clock(clock);
        } else if line.starts_with("Stream #") && line.contains(" Video:") && !line.contains("attached pic") {
            out.has_video = true;
        } else if line.starts_with("Stream #") && line.contains(" Audio:") {
            out.audio.push(AudioTrack::default());
        }
    }
    out
}

/// Probes with ffprobe, or with the ffmpeg banner when ffprobe is missing.
pub fn probe(path: &Path, size_bytes: u64, deadline: &Deadline) -> Result<VideoInfo, VideoError> {
    if let Some(ffprobe) = ffprobe_path() {
        let mut cmd = proc::command(ffprobe);
        cmd.args(["-v", "error", "-print_format", "json", "-show_format", "-show_streams", "-i"]).arg(proc::input_arg(path));
        let out = proc::run_capture(cmd, deadline, MAX_PROBE_OUTPUT)?;
        if !out.success {
            return Err(VideoError::Corrupt(first_line(&out.stderr)));
        }
        return parse_ffprobe_json(&String::from_utf8_lossy(&out.stdout), Some(size_bytes));
    }
    let ffmpeg = ffmpeg_path().ok_or(VideoError::NoBackend)?;
    let mut cmd = proc::command(ffmpeg);
    cmd.args(["-hide_banner", "-nostdin", "-i"]).arg(proc::input_arg(path));
    let out = proc::run_capture(cmd, deadline, MAX_PROBE_OUTPUT)?;
    let mut info = parse_ffmpeg_banner(&out.stderr);
    if !info.has_video && info.audio.is_empty() && info.duration_secs.is_none() {
        return Err(VideoError::Corrupt(first_line(&out.stderr)));
    }
    info.size_bytes = Some(size_bytes);
    Ok(info)
}

fn first_line(text: &str) -> String {
    text.lines().find(|l| !l.trim().is_empty()).unwrap_or("probe failed").trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    const PHONE: &str = r#"{
      "streams": [
        {"index":0,"codec_name":"h264","codec_type":"video","width":1920,"height":1080,"pix_fmt":"yuv420p",
         "sample_aspect_ratio":"1:1","r_frame_rate":"30/1","avg_frame_rate":"30000/1001","bit_rate":"8000000",
         "disposition":{"default":1,"attached_pic":0},
         "tags":{"creation_time":"2024-05-01T10:00:00.000000Z","language":"und"},
         "side_data_list":[{"side_data_type":"Display Matrix","displaymatrix":"...","rotation":-90}]},
        {"index":1,"codec_name":"aac","codec_type":"audio","sample_rate":"48000","channels":2,"bit_rate":"128000","tags":{"language":"eng"}},
        {"index":2,"codec_name":"mov_text","codec_type":"subtitle"}
      ],
      "format":{"format_name":"mov,mp4,m4a,3gp,3g2,mj2","duration":"12.480000","size":"13000000","bit_rate":"8300000",
                "tags":{"creation_time":"2024-05-01T10:00:00.000000Z"}}}"#;

    #[test]
    fn phone_clip_probe_reads_rotation_audio_and_times() {
        let info = parse_ffprobe_json(PHONE, None).unwrap();
        assert!(info.has_video);
        assert_eq!((info.width, info.height), (1920, 1080));
        assert_eq!(info.rotation, 90);
        assert_eq!(info.display_size(), (1080, 1920));
        assert_eq!(info.video_codec.as_deref(), Some("h264"));
        assert_eq!(info.pixel_format.as_deref(), Some("yuv420p"));
        assert!((info.fps.unwrap() - 29.97).abs() < 0.01);
        assert_eq!(info.duration_secs, Some(12.48));
        assert_eq!(info.bitrate, Some(8_300_000));
        assert_eq!(info.video_bitrate, Some(8_000_000));
        assert_eq!(info.size_bytes, Some(13_000_000));
        assert_eq!(info.creation_time.as_deref(), Some("2024-05-01T10:00:00.000000Z"));
        assert_eq!(info.sar, None);
        assert_eq!(info.audio.len(), 1);
        assert_eq!(info.audio[0], AudioTrack { index: 1, codec: Some("aac".into()), channels: Some(2), sample_rate: Some(48000), bitrate: Some(128000), language: Some("eng".into()) });
        assert_eq!(info.subtitle_tracks, 1);
        assert!(info.container.unwrap().starts_with("mov"));
    }

    #[test]
    fn anamorphic_and_legacy_rotate_tag_are_read() {
        let json = r#"{"streams":[{"index":0,"codec_type":"video","codec_name":"mpeg2video","width":720,"height":576,
            "sample_aspect_ratio":"16:15","avg_frame_rate":"0/0","r_frame_rate":"25/1","tags":{"rotate":"270"}}],
            "format":{"duration":"N/A"}}"#;
        let info = parse_ffprobe_json(json, Some(5)).unwrap();
        assert_eq!(info.sar, Some((16, 15)));
        assert_eq!(info.rotation, 270);
        assert_eq!(info.fps, Some(25.0));
        assert_eq!(info.duration_secs, None);
        assert_eq!(info.size_bytes, Some(5));
        assert_eq!(info.display_size(), (576, 768));
    }

    #[test]
    fn audio_only_files_have_no_video() {
        let json = r#"{"streams":[{"index":0,"codec_type":"audio","codec_name":"mp3","channels":2,"sample_rate":"44100"}],"format":{"duration":"3.5"}}"#;
        let info = parse_ffprobe_json(json, None).unwrap();
        assert!(!info.has_video && info.is_audio_only());
        assert_eq!(info.duration_secs, Some(3.5));
    }

    #[test]
    fn cover_art_is_not_a_video_stream() {
        let json = r#"{"streams":[
            {"index":0,"codec_type":"audio","codec_name":"mp3"},
            {"index":1,"codec_type":"video","codec_name":"mjpeg","width":500,"height":500,"disposition":{"attached_pic":1}}],
            "format":{}}"#;
        assert!(!parse_ffprobe_json(json, None).unwrap().has_video);
    }

    #[test]
    fn the_default_or_largest_video_stream_wins() {
        let json = r#"{"streams":[
            {"index":0,"codec_type":"video","codec_name":"h264","width":640,"height":360},
            {"index":1,"codec_type":"video","codec_name":"hevc","width":3840,"height":2160}],"format":{}}"#;
        let info = parse_ffprobe_json(json, None).unwrap();
        assert_eq!((info.video_index, info.width), (Some(1), 3840));
        let json = r#"{"streams":[
            {"index":0,"codec_type":"video","codec_name":"h264","width":640,"height":360,"disposition":{"default":1}},
            {"index":1,"codec_type":"video","codec_name":"hevc","width":3840,"height":2160,"disposition":{"default":0}}],"format":{}}"#;
        assert_eq!(parse_ffprobe_json(json, None).unwrap().video_index, Some(0));
    }

    #[test]
    fn garbage_and_empty_documents_are_errors() {
        assert!(matches!(parse_ffprobe_json("not json", None), Err(VideoError::Backend(_))));
        assert!(matches!(parse_ffprobe_json("{}", None), Err(VideoError::Corrupt(_))));
        assert!(parse_ffprobe_json(r#"{"streams":[],"format":{}}"#, None).is_ok());
    }

    #[test]
    fn banner_parsing_finds_duration_and_streams() {
        let banner = "Input #0, mov,mp4, from 'a.mp4':\n  Duration: 00:00:12.48, start: 0.000000, bitrate: 8300 kb/s\n  Stream #0:0[0x1](und): Video: h264 (High), yuv420p, 1920x1080, 30 fps\n  Stream #0:1[0x2](eng): Audio: aac, 48000 Hz, stereo\n";
        let info = parse_ffmpeg_banner(banner);
        assert_eq!(info.duration_secs, Some(12.48));
        assert!(info.has_video);
        assert_eq!(info.audio.len(), 1);
        let cover = "  Stream #0:1: Video: mjpeg, yuvj420p, 500x500 (attached pic)\n";
        assert!(!parse_ffmpeg_banner(cover).has_video);
    }
}
