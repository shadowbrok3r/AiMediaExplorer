//! Integration tests over media generated with the ffmpeg command.

use super::fixtures as fx;
use super::quality::{frame_stats, is_blank};
use super::*;
use image::RgbImage;
use std::path::Path;
use std::time::Duration;

/// Backends that can run in this build and environment.
fn backends() -> Vec<BackendChoice> {
    let mut out = Vec::new();
    if native_available() {
        out.push(BackendChoice::Native);
    }
    if ffmpeg_path().is_some() {
        out.push(BackendChoice::Cli);
    }
    out
}

fn opts(backend: BackendChoice) -> PosterOptions {
    PosterOptions { backend, timeout: Duration::from_secs(30), ..Default::default() }
}

fn poster_of(path: &Path, backend: BackendChoice) -> Result<Poster, VideoError> {
    poster_frame(path, &opts(backend))
}

/// Centroid of strongly red pixels as fractions of the image size.
fn red_centroid(img: &RgbImage) -> Option<(f32, f32)> {
    let (mut sx, mut sy, mut n) = (0f64, 0f64, 0u64);
    for (x, y, p) in img.enumerate_pixels() {
        if p.0[0] > 170 && p.0[1] < 90 && p.0[2] < 90 {
            sx += f64::from(x);
            sy += f64::from(y);
            n += 1;
        }
    }
    (n > 20).then(|| ((sx / n as f64) as f32 / img.width() as f32, (sy / n as f64) as f32 / img.height() as f32))
}

fn mean_abs_diff(a: &RgbImage, b: &RgbImage) -> f64 {
    let b = image::imageops::resize(b, a.width(), a.height(), image::imageops::FilterType::Triangle);
    let total: u64 = a.as_raw().iter().zip(b.as_raw()).map(|(x, y)| u64::from(x.abs_diff(*y))).sum();
    total as f64 / a.as_raw().len() as f64
}

macro_rules! need_tools {
    ($name:expr) => {
        if !fx::available() {
            return fx::skip($name);
        }
    };
}

macro_rules! need {
    ($name:expr, $maker:expr) => {
        match $maker {
            Some(p) => p,
            None => return fx::skip($name),
        }
    };
}

#[test]
fn probe_backends_agree_for_every_container_and_codec() {
    need_tools!("probe_backends_agree");
    let dir = tempfile::tempdir().unwrap();
    let mut files = vec![need!("h264 clip", fx::h264_mp4(dir.path(), "main.mp4", 3))];
    files.extend(fx::CODEC_MATRIX.iter().filter_map(|(name, args)| fx::coded(dir.path(), name, args)));
    assert!(files.len() >= 8, "only {} fixtures could be generated", files.len());
    for file in &files {
        let cli = probe_with(file, &ProbeOptions { backend: BackendChoice::Cli, ..Default::default() }).unwrap_or_else(|e| panic!("{}: {e}", file.display()));
        assert!(cli.has_video, "{}", file.display());
        assert_eq!((cli.width, cli.height), (320, 180), "{}", file.display());
        assert!(cli.duration_secs.is_some_and(|d| (1.5..=3.6).contains(&d)), "{} {:?}", file.display(), cli.duration_secs);
        if native_available() {
            let native = probe_with(file, &ProbeOptions { backend: BackendChoice::Native, ..Default::default() }).unwrap_or_else(|e| panic!("{}: {e}", file.display()));
            assert_eq!(native.has_video, cli.has_video, "{}", file.display());
            assert_eq!(native.video_codec, cli.video_codec, "{}", file.display());
            assert_eq!((native.width, native.height, native.rotation, native.sar), (cli.width, cli.height, cli.rotation, cli.sar), "{}", file.display());
            assert_eq!(native.audio.len(), cli.audio.len(), "{}", file.display());
            assert_eq!(native.subtitle_tracks, cli.subtitle_tracks, "{}", file.display());
            assert_eq!(native.pixel_format, cli.pixel_format, "{}", file.display());
            let (nd, cd) = (native.duration_secs.unwrap(), cli.duration_secs.unwrap());
            assert!((nd - cd).abs() < 0.2, "{} {nd} {cd}", file.display());
            if let (Some(nf), Some(cf)) = (native.fps, cli.fps) {
                assert!((nf - cf).abs() < 0.5, "{} {nf} {cf}", file.display());
            }
        }
    }
}

#[test]
fn probe_reads_audio_tracks_and_times_of_the_main_clip() {
    need_tools!("probe_reads_audio_tracks");
    let dir = tempfile::tempdir().unwrap();
    let clip = need!("main clip", fx::h264_mp4(dir.path(), "main.mp4", 3));
    for backend in backends() {
        let info = probe_with(&clip, &ProbeOptions { backend, ..Default::default() }).unwrap();
        assert_eq!(info.video_codec.as_deref(), Some("h264"), "{backend:?}");
        assert_eq!(info.audio.len(), 1, "{backend:?}");
        assert_eq!(info.audio[0].codec.as_deref(), Some("aac"), "{backend:?}");
        assert_eq!(info.audio[0].channels, Some(1), "{backend:?}");
        assert!((info.duration_secs.unwrap() - 3.0).abs() < 0.2, "{backend:?}");
        assert!((info.fps.unwrap() - 25.0).abs() < 0.1, "{backend:?}");
        assert!(info.bitrate.is_some_and(|b| b > 0), "{backend:?}");
        assert_eq!(info.size_bytes, Some(std::fs::metadata(&clip).unwrap().len()), "{backend:?}");
        assert_eq!(info.pixel_format.as_deref(), Some("yuv420p"), "{backend:?}");
        assert!(info.container.as_deref().is_some_and(|c| c.contains("mp4")), "{backend:?}");
        assert!(info.summary().contains("h264"));
    }
}

#[test]
fn posters_work_for_every_container_and_codec() {
    need_tools!("posters_work_for_every_container");
    let dir = tempfile::tempdir().unwrap();
    let mut files = vec![need!("h264 clip", fx::h264_mp4(dir.path(), "main.mp4", 3))];
    files.extend(fx::CODEC_MATRIX.iter().filter_map(|(name, args)| fx::coded(dir.path(), name, args)));
    assert!(files.len() >= 8, "only {} fixtures could be generated", files.len());
    for backend in backends() {
        for file in &files {
            let p = poster_of(file, backend).unwrap_or_else(|e| panic!("{backend:?} {}: {e}", file.display()));
            assert_eq!(p.image.dimensions(), (256, 144), "{backend:?} {}", file.display());
            assert!(!p.blank && !is_blank(&frame_stats(&p.image)), "{backend:?} {}", file.display());
            assert_eq!(p.source, (320, 180));
        }
    }
}

#[test]
fn native_and_command_posters_look_the_same() {
    need_tools!("native_and_command_posters_look_the_same");
    if !native_available() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let clip = need!("main clip", fx::h264_mp4(dir.path(), "main.mp4", 3));
    let native = poster_of(&clip, BackendChoice::Native).unwrap();
    let cli = poster_of(&clip, BackendChoice::Cli).unwrap();
    assert_eq!(native.image.dimensions(), cli.image.dimensions());
    let diff = mean_abs_diff(&native.image, &cli.image);
    assert!(diff < 6.0, "mean channel difference {diff}");
}

#[test]
fn colours_of_hd_content_use_the_stream_matrix() {
    need_tools!("colours_of_hd_content");
    if !native_available() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let src = "color=c=0xCC3322:s=1280x720:r=5:d=1";
    let clip = dir.path().join("hd.mp4");
    let made = fx::ffmpeg(&["-f", "lavfi", "-i", src, "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", clip.to_str().unwrap()]);
    if !made {
        return fx::skip("hd colour clip");
    }
    for backend in backends() {
        let p = poster_of(&clip, backend).unwrap();
        let px = p.image.get_pixel(p.image.width() / 2, p.image.height() / 2).0;
        let expected = [0xCC, 0x33, 0x22];
        for c in 0..3 {
            assert!(px[c].abs_diff(expected[c]) <= 10, "{backend:?} got {px:?} expected {expected:?}");
        }
    }
}

#[test]
fn rotated_clips_come_out_upright() {
    need_tools!("rotated_clips_come_out_upright");
    let dir = tempfile::tempdir().unwrap();
    let plain = need!("marker", fx::marker_mp4(dir.path(), "plain.mp4"));
    let ccw90 = need!("rotated ccw 90", fx::rotated(dir.path(), "ccw90.mp4", 90));
    let cw90 = need!("rotated cw 90", fx::rotated(dir.path(), "cw90.mp4", -90));
    let half = need!("rotated 180", fx::rotated(dir.path(), "r180.mp4", 180));
    for backend in backends() {
        let p0 = poster_of(&plain, backend).unwrap();
        assert_eq!(p0.image.dimensions(), (256, 144));
        let (x, y) = red_centroid(&p0.image).expect("marker visible");
        assert!(x < 0.3 && y < 0.3, "{backend:?} unrotated marker at {x},{y}");

        let p = poster_of(&ccw90, backend).unwrap();
        assert_eq!(p.image.dimensions(), (144, 256), "{backend:?}");
        assert_eq!(p.rotation, 270, "{backend:?}");
        let (x, y) = red_centroid(&p.image).expect("marker visible");
        assert!(x < 0.5 && y > 0.5, "{backend:?} ccw 90 marker at {x},{y}");

        let p = poster_of(&cw90, backend).unwrap();
        assert_eq!(p.image.dimensions(), (144, 256), "{backend:?}");
        assert_eq!(p.rotation, 90, "{backend:?}");
        let (x, y) = red_centroid(&p.image).expect("marker visible");
        assert!(x > 0.5 && y < 0.5, "{backend:?} cw 90 marker at {x},{y}");

        let p = poster_of(&half, backend).unwrap();
        assert_eq!(p.image.dimensions(), (256, 144), "{backend:?}");
        let (x, y) = red_centroid(&p.image).expect("marker visible");
        assert!(x > 0.5 && y > 0.5, "{backend:?} 180 marker at {x},{y}");
    }
    for backend in backends() {
        let info = probe_with(&cw90, &ProbeOptions { backend, ..Default::default() }).unwrap();
        assert_eq!((info.rotation, info.display_size()), (90, (180, 320)), "{backend:?}");
        let info = probe_with(&ccw90, &ProbeOptions { backend, ..Default::default() }).unwrap();
        assert_eq!((info.rotation, info.display_size()), (270, (180, 320)), "{backend:?}");
    }
}

#[test]
fn anamorphic_clips_are_widened_to_square_pixels() {
    need_tools!("anamorphic_clips_are_widened");
    let dir = tempfile::tempdir().unwrap();
    let clip = need!("anamorphic", fx::anamorphic(dir.path(), "wide.mp4"));
    for backend in backends() {
        let info = probe_with(&clip, &ProbeOptions { backend, ..Default::default() }).unwrap();
        assert_eq!(info.sar, Some((4, 3)), "{backend:?}");
        assert_eq!(info.display_size(), (427, 180), "{backend:?}");
        let p = poster_of(&clip, backend).unwrap();
        let (w, h) = p.image.dimensions();
        assert_eq!(w, 256, "{backend:?}");
        assert!(h.abs_diff(108) <= 1, "{backend:?} height {h}");
    }
}

#[test]
fn black_intros_are_skipped_and_all_black_clips_are_flagged() {
    need_tools!("black_intros_are_skipped");
    let dir = tempfile::tempdir().unwrap();
    let intro = need!("dark intro", fx::dark_intro(dir.path(), "intro.mp4"));
    let black = need!("black", fx::all_black(dir.path(), "black.mp4"));
    for backend in backends() {
        let p = poster_of(&intro, backend).unwrap();
        assert!(!p.blank, "{backend:?}");
        assert!(p.attempts >= 2, "{backend:?} attempts {}", p.attempts);
        assert!(p.position_secs.is_none_or(|s| s > 3.0), "{backend:?} {:?}", p.position_secs);
        let b = poster_of(&black, backend).unwrap();
        assert!(b.blank, "{backend:?}");
        assert_eq!(b.attempts, 4, "{backend:?}");
    }
}

#[test]
fn one_hour_clip_is_sampled_by_seeking() {
    need_tools!("one_hour_clip_is_sampled_by_seeking");
    let dir = tempfile::tempdir().unwrap();
    let clip = need!("one hour clip", fx::one_hour(dir.path(), "long.mp4"));
    for backend in backends() {
        let started = std::time::Instant::now();
        let p = poster_of(&clip, backend).unwrap();
        let took = started.elapsed();
        assert!(took < Duration::from_secs(5), "{backend:?} took {took:?}");
        let pos = p.position_secs.expect("position known");
        assert!((300.0..=420.0).contains(&pos), "{backend:?} position {pos}");
        assert_eq!(p.image.dimensions(), (160, 90));
    }
}

#[test]
fn eight_k_frames_are_scaled_down_within_the_deadline() {
    need_tools!("eight_k_frames");
    let dir = tempfile::tempdir().unwrap();
    let clip = need!("8k clip", fx::eight_k(dir.path(), "huge.mp4"));
    for backend in backends() {
        let started = std::time::Instant::now();
        let p = poster_of(&clip, backend).unwrap();
        assert_eq!(p.image.dimensions(), (256, 144), "{backend:?}");
        assert_eq!(p.source, (7680, 4320));
        assert!(started.elapsed() < Duration::from_secs(20), "{backend:?}");
    }
}

#[test]
fn variable_frame_rate_clips_work() {
    need_tools!("variable_frame_rate_clips");
    let dir = tempfile::tempdir().unwrap();
    let clip = need!("vfr", fx::variable_rate(dir.path(), "vfr.mp4"));
    for backend in backends() {
        let p = poster_of(&clip, backend).unwrap();
        assert!(!p.blank, "{backend:?}");
        let info = probe_with(&clip, &ProbeOptions { backend, ..Default::default() }).unwrap();
        assert!(info.fps.is_some_and(|f| f > 1.0), "{backend:?}");
    }
}

#[test]
fn audio_only_files_have_no_video_stream() {
    need_tools!("audio_only_files");
    let dir = tempfile::tempdir().unwrap();
    let audio = need!("audio", fx::audio_only(dir.path(), "tone.mp4"));
    let cover = need!("cover art", fx::audio_with_cover(dir.path(), "covered.mp4"));
    for backend in backends() {
        for file in [&audio, &cover] {
            assert_eq!(poster_of(file, backend).err(), Some(VideoError::NoVideoStream), "{backend:?} {}", file.display());
            let info = probe_with(file, &ProbeOptions { backend, ..Default::default() }).unwrap();
            assert!(!info.has_video && info.audio.len() == 1, "{backend:?} {}", file.display());
        }
    }
    if !backends().is_empty() {
        match thumb_image(&audio, &PosterOptions::default()).unwrap() {
            ThumbImage::Placeholder { kind, reason } => assert_eq!((kind, reason), (placeholder::PlaceholderKind::Audio, VideoError::NoVideoStream)),
            ThumbImage::Frame(_) => panic!("audio files cannot have a frame"),
        }
    }
}

#[test]
fn damaged_files_fail_cleanly_and_quickly() {
    need_tools!("damaged_files_fail_cleanly");
    let dir = tempfile::tempdir().unwrap();
    let good = need!("main clip", fx::h264_mp4(dir.path(), "good.mp4", 3));
    let mut damaged = vec![
        fx::garbage(dir.path(), "garbage.mp4", 200_000),
        fx::truncated(dir.path(), "header_only.mp4", &good, 0.0005),
        fx::truncated(dir.path(), "half.mp4", &good, 0.5),
        fx::truncated(dir.path(), "most.mp4", &good, 0.9),
    ];
    std::fs::write(dir.path().join("text.mp4"), "this is not a video, just words\n").unwrap();
    damaged.push(dir.path().join("text.mp4"));
    std::fs::write(dir.path().join("zeros.mp4"), vec![0u8; 1 << 20]).unwrap();
    damaged.push(dir.path().join("zeros.mp4"));
    for seed in 0..24u64 {
        damaged.push(fx::corrupted(dir.path(), &format!("flip_{seed}.mp4"), &good, 2000, 30 + 10 * seed as usize, seed + 1));
    }
    for backend in backends() {
        for file in &damaged {
            let started = std::time::Instant::now();
            let o = PosterOptions { backend, timeout: Duration::from_secs(10), ..Default::default() };
            let result = poster_frame(file, &o);
            assert!(started.elapsed() < Duration::from_secs(10), "{backend:?} {} took {:?}", file.display(), started.elapsed());
            match result {
                Ok(p) => assert!(p.image.width() > 0 && p.image.width() <= 256),
                Err(VideoError::Timeout) => panic!("{backend:?} {} hung", file.display()),
                Err(e) => assert!(matches!(e, VideoError::Corrupt(_) | VideoError::NoVideoStream | VideoError::Backend(_)), "{backend:?} {}: {e}", file.display()),
            }
            let _ = probe_with(file, &ProbeOptions { backend, timeout: Duration::from_secs(10), ..Default::default() });
        }
    }
}

#[test]
fn empty_missing_and_directory_paths_are_typed_errors() {
    let dir = tempfile::tempdir().unwrap();
    let empty = dir.path().join("empty.mp4");
    std::fs::write(&empty, b"").unwrap();
    let folder = dir.path().join("folder.mp4");
    std::fs::create_dir(&folder).unwrap();
    for backend in [BackendChoice::Auto, BackendChoice::Disabled] {
        assert_eq!(poster_of(&empty, backend).err(), Some(VideoError::Empty));
        assert!(matches!(poster_of(&dir.path().join("none.mp4"), backend), Err(VideoError::Missing(_))));
        assert!(matches!(poster_of(&folder, backend), Err(VideoError::Missing(_))));
        assert_eq!(probe_with(&empty, &ProbeOptions { backend, ..Default::default() }).err(), Some(VideoError::Empty));
    }
}

#[test]
fn expired_deadlines_and_cancellation_stop_the_work() {
    need_tools!("expired_deadlines_and_cancellation");
    let dir = tempfile::tempdir().unwrap();
    let clip = need!("main clip", fx::h264_mp4(dir.path(), "main.mp4", 3));
    for backend in backends() {
        let o = PosterOptions { backend, timeout: Duration::ZERO, ..Default::default() };
        assert_eq!(poster_frame(&clip, &o).err(), Some(VideoError::Timeout), "{backend:?}");
        let cancel = Cancel::new();
        cancel.cancel();
        let o = PosterOptions { backend, cancel: cancel.clone(), ..Default::default() };
        assert_eq!(poster_frame(&clip, &o).err(), Some(VideoError::Cancelled), "{backend:?}");
        let p = ProbeOptions { backend, timeout: Duration::ZERO, ..Default::default() };
        assert_eq!(probe_with(&clip, &p).err(), Some(VideoError::Timeout), "{backend:?}");
        let p = ProbeOptions { backend, cancel, ..Default::default() };
        assert_eq!(probe_with(&clip, &p).err(), Some(VideoError::Cancelled), "{backend:?}");
    }
}

#[test]
fn many_concurrent_posters_never_exceed_the_decode_slots() {
    need_tools!("many_concurrent_posters");
    let dir = tempfile::tempdir().unwrap();
    let clip = need!("main clip", fx::h264_mp4(dir.path(), "main.mp4", 3));
    if backends().is_empty() {
        return;
    }
    let gate = gate::global();
    let limit = gate.limit();
    let running = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let peak = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let sampler = {
        let (running, peak) = (running.clone(), peak.clone());
        std::thread::spawn(move || {
            while running.load(std::sync::atomic::Ordering::SeqCst) {
                peak.fetch_max(gate::global().in_flight(), std::sync::atomic::Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(1));
            }
        })
    };
    let workers: Vec<_> = (0..16)
        .map(|_| {
            let clip = clip.clone();
            std::thread::spawn(move || poster_frame(&clip, &PosterOptions::default()).map(|p| p.image.width()))
        })
        .collect();
    for w in workers {
        assert_eq!(w.join().unwrap().unwrap(), 256);
    }
    running.store(false, std::sync::atomic::Ordering::SeqCst);
    sampler.join().unwrap();
    assert!(peak.load(std::sync::atomic::Ordering::SeqCst) <= gate.limit().max(limit), "peak {} limit {limit}", peak.load(std::sync::atomic::Ordering::SeqCst));
}

#[tokio::test(flavor = "multi_thread")]
async fn async_wrappers_run_on_the_blocking_pool() {
    need_tools!("async_wrappers_run_on_the_blocking_pool");
    let dir = tempfile::tempdir().unwrap();
    let clip = need!("main clip", fx::h264_mp4(dir.path(), "main.mp4", 3));
    if backends().is_empty() {
        return;
    }
    let (poster, info) = tokio::join!(poster_frame_async(clip.clone(), PosterOptions::default()), probe_async(clip.clone(), ProbeOptions::default()));
    assert_eq!(poster.unwrap().image.width(), 256);
    assert!(info.unwrap().has_video);
}

#[test]
fn thumbnail_generation_encodes_jpegs_and_typed_placeholders() {
    need_tools!("thumbnail_generation_encodes");
    let dir = tempfile::tempdir().unwrap();
    let clip = need!("main clip", fx::h264_mp4(dir.path(), "main.mp4", 3));
    let audio = need!("audio", fx::audio_only(dir.path(), "tone.mp4"));
    let thumb = crate::utilities::thumbs::generate_video_thumb(&clip).unwrap();
    assert_eq!(thumb.format, crate::utilities::thumbs::ThumbFormat::Jpeg);
    assert_eq!((thumb.width, thumb.height), (256, 144));
    let reparsed = crate::utilities::thumbs::EncodedThumb::from_bytes(thumb.bytes.clone()).unwrap();
    assert_eq!((reparsed.width, reparsed.height), (256, 144));
    let legacy = crate::utilities::thumbs::generate_video_thumb_data(&clip).unwrap();
    assert!(crate::utilities::thumbs::EncodedThumb::from_legacy_string(&legacy).is_ok());

    let none = PosterOptions { backend: BackendChoice::Disabled, ..Default::default() };
    let (placeholder, report) = crate::utilities::thumbs::generate_video_thumb_with(&clip, &none).unwrap();
    assert_eq!(report.placeholder, Some(placeholder::PlaceholderKind::Video));
    assert_eq!((placeholder.width, placeholder.height), (placeholder::PLACEHOLDER_W, placeholder::PLACEHOLDER_H));
    let (audio_tile, report) = crate::utilities::thumbs::generate_video_thumb_with(&audio, &PosterOptions::default()).unwrap();
    assert_eq!(report.placeholder, Some(placeholder::PlaceholderKind::Audio));
    assert_ne!(audio_tile.bytes, placeholder.bytes);
    assert!(crate::utilities::thumbs::generate_video_thumb(&dir.path().join("none.mp4")).unwrap_err().contains("video:"));
}

#[tokio::test(flavor = "multi_thread")]
async fn videos_flow_through_the_real_thumbnail_service() {
    need_tools!("videos_flow_through_the_real_thumbnail_service");
    if backends().is_empty() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let clip = need!("main clip", fx::h264_mp4(dir.path(), "flow.mp4", 3));
    let broken = fx::garbage(dir.path(), "broken.mp4", 4096);
    let svc = crate::ui::file_table::thumbs::service();
    let ctx = egui::Context::default();
    let scope = crate::ui::file_table::thumbs::alloc_scope();
    let (clip_path, broken_path) = (clip.display().to_string(), broken.display().to_string());
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    let mut ready = false;
    while std::time::Instant::now() < deadline {
        let view = svc.request(scope, &clip_path, crate::ui::file_table::thumbs::Tier::Thumb, 0.0);
        let _ = svc.request(scope, &broken_path, crate::ui::file_table::thumbs::Tier::Thumb, 0.0);
        svc.poll_now(&ctx);
        if let crate::ui::file_table::thumbs::ThumbView::Ready { size, .. } = view {
            assert_eq!((size.x as u32, size.y as u32), (256, 144));
            ready = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(ready, "video thumbnail never became ready");
    assert!(svc.has_thumb(&clip_path));
}

#[test]
#[ignore = "timing report"]
fn bench_posters() {
    need_tools!("bench_posters");
    let dir = tempfile::tempdir().unwrap();
    let hd = need!("1080p", fx::sized(dir.path(), "hd.mp4", "1920x1080", 4));
    let uhd = need!("4k", fx::sized(dir.path(), "uhd.mp4", "3840x2160", 4));
    for backend in backends() {
        for (name, file) in [("1080p", &hd), ("4k", &uhd)] {
            let _ = poster_of(file, backend).unwrap();
            let runs = 20;
            let started = std::time::Instant::now();
            for _ in 0..runs {
                poster_of(file, backend).unwrap();
            }
            eprintln!("poster {backend:?} {name}: {:.1} ms", started.elapsed().as_secs_f64() * 1000.0 / f64::from(runs));
        }
    }
}
