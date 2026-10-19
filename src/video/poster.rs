//! Poster frames: the request options, candidate selection and the entry points that gate and dispatch to a backend.

use super::info::VideoInfo;
use super::placeholder::{PlaceholderKind, placeholder};
use super::quality::{self, FrameStats};
use super::{Backend, BackendChoice, Cancel, Deadline, VideoError, backends_for, check_file, gate};
use image::RgbImage;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct PosterOptions {
    /// Longest side of the poster in pixels.
    pub max_dim: u32,
    /// Preferred position as a fraction of the duration.
    pub at_fraction: f64,
    /// Positions tried before settling for the best blank frame.
    pub attempts: u32,
    /// Time allowed for one poster once it holds a decode slot.
    pub timeout: Duration,
    pub cancel: Cancel,
    pub backend: BackendChoice,
}

impl Default for PosterOptions {
    fn default() -> Self {
        Self { max_dim: 256, at_fraction: 0.10, attempts: 4, timeout: Duration::from_secs(20), cancel: Cancel::new(), backend: BackendChoice::Auto }
    }
}

/// An upright, square-pixel poster frame.
#[derive(Clone, Debug)]
pub struct Poster {
    pub image: RgbImage,
    pub position_secs: Option<f64>,
    pub attempts: u32,
    /// Every attempted frame was blank; this is the best of them.
    pub blank: bool,
    pub backend: Backend,
    pub elapsed: Duration,
    /// Coded size of the video stream.
    pub source: (u32, u32),
    pub rotation: u32,
}

pub struct Candidate {
    pub image: RgbImage,
    pub position_secs: Option<f64>,
}

struct Best {
    candidate: Candidate,
    stats: FrameStats,
}

/// Picks the first non-blank frame, or the best blank one when the attempts run out.
pub struct Selector {
    max_attempts: u32,
    attempts: u32,
    best: Option<Best>,
}

impl Selector {
    pub fn new(max_attempts: u32) -> Self {
        Self { max_attempts: max_attempts.max(1), attempts: 0, best: None }
    }

    pub fn attempts(&self) -> u32 {
        self.attempts
    }

    /// Records a candidate; true when it is good enough to stop looking.
    pub fn offer(&mut self, candidate: Candidate) -> bool {
        self.attempts += 1;
        let stats = quality::frame_stats(&candidate.image);
        let blank = quality::is_blank(&stats);
        if blank {
            gate::counters().blank_retries.fetch_add(1, Ordering::Relaxed);
        }
        let better = self.best.as_ref().is_none_or(|b| quality::is_blank(&b.stats) && (!blank || quality::score(&stats) > quality::score(&b.stats)));
        if better {
            self.best = Some(Best { candidate, stats });
        }
        !blank || self.attempts >= self.max_attempts
    }

    pub fn finish(self, info: &VideoInfo, backend: Backend, elapsed: Duration) -> Option<Poster> {
        let attempts = self.attempts;
        self.best.map(|b| Poster {
            blank: quality::is_blank(&b.stats),
            image: b.candidate.image,
            position_secs: b.candidate.position_secs,
            attempts,
            backend,
            elapsed,
            source: (info.width, info.height),
            rotation: info.rotation,
        })
    }
}

/// Decodes a poster frame; blocks the calling thread and waits for a decode slot first.
pub fn poster_frame(path: &Path, opts: &PosterOptions) -> Result<Poster, VideoError> {
    let started = Instant::now();
    let result = run_poster(path, opts);
    let counters = gate::counters();
    match &result {
        Ok(_) => {
            counters.ok.fetch_add(1, Ordering::Relaxed);
            counters.record_time(started.elapsed());
        }
        Err(VideoError::Timeout) => {
            counters.timeouts.fetch_add(1, Ordering::Relaxed);
            counters.failed.fetch_add(1, Ordering::Relaxed);
        }
        Err(VideoError::Cancelled) => {}
        Err(_) => {
            counters.failed.fetch_add(1, Ordering::Relaxed);
        }
    }
    result
}

fn run_poster(path: &Path, opts: &PosterOptions) -> Result<Poster, VideoError> {
    let size = check_file(path)?;
    let backends = backends_for(opts.backend);
    if backends.is_empty() {
        return Err(VideoError::NoBackend);
    }
    let _slot = gate::global().acquire(&opts.cancel)?;
    let deadline = Deadline::new(opts.timeout, opts.cancel.clone());
    let mut last = VideoError::NoBackend;
    for backend in backends {
        let result = match backend {
            #[cfg(ame_ffmpeg)]
            Backend::Native => super::native::poster(path, size, opts, &deadline),
            #[cfg(not(ame_ffmpeg))]
            Backend::Native => Err(VideoError::NoBackend),
            Backend::Cli => super::cli::poster(path, size, opts, &deadline),
        };
        match result {
            Err(VideoError::Backend(why)) => last = VideoError::Backend(why),
            other => return other,
        }
    }
    Err(last)
}

/// Async wrapper that runs the decode on the blocking pool.
pub async fn poster_frame_async(path: PathBuf, opts: PosterOptions) -> Result<Poster, VideoError> {
    tokio::task::spawn_blocking(move || poster_frame(&path, &opts)).await.unwrap_or_else(|e| Err(VideoError::Backend(format!("poster task failed: {e}"))))
}

/// What a thumbnail is built from.
pub enum ThumbImage {
    Frame(Poster),
    Placeholder { kind: PlaceholderKind, reason: VideoError },
}

impl ThumbImage {
    pub fn into_image(self) -> RgbImage {
        match self {
            ThumbImage::Frame(p) => p.image,
            ThumbImage::Placeholder { kind, .. } => placeholder(kind),
        }
    }

    pub fn is_placeholder(&self) -> bool {
        matches!(self, ThumbImage::Placeholder { .. })
    }
}

/// Thumbnail source for a video file: a decoded frame, or a placeholder when there is nothing to decode.
pub fn thumb_image(path: &Path, opts: &PosterOptions) -> Result<ThumbImage, VideoError> {
    match poster_frame(path, opts) {
        Ok(p) => Ok(ThumbImage::Frame(p)),
        Err(reason) if reason.wants_placeholder() => {
            gate::counters().placeholders.fetch_add(1, Ordering::Relaxed);
            let kind = if reason == VideoError::NoVideoStream { PlaceholderKind::Audio } else { PlaceholderKind::Video };
            Ok(ThumbImage::Placeholder { kind, reason })
        }
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    fn flat(v: u8) -> Candidate {
        Candidate { image: RgbImage::from_pixel(16, 9, Rgb([v, v, v])), position_secs: Some(f64::from(v)) }
    }

    fn busy(pos: f64) -> Candidate {
        Candidate { image: RgbImage::from_fn(16, 9, |x, y| if (x + y) % 2 == 0 { Rgb([240, 240, 240]) } else { Rgb([10, 30, 60]) }), position_secs: Some(pos) }
    }

    fn info() -> VideoInfo {
        VideoInfo { has_video: true, width: 1280, height: 720, rotation: 90, ..Default::default() }
    }

    #[test]
    fn a_good_first_frame_stops_the_search() {
        let mut s = Selector::new(4);
        assert!(s.offer(busy(1.0)));
        let p = s.finish(&info(), Backend::Cli, Duration::ZERO).unwrap();
        assert!(!p.blank);
        assert_eq!((p.attempts, p.position_secs, p.source, p.rotation), (1, Some(1.0), (1280, 720), 90));
    }

    #[test]
    fn blank_frames_are_skipped_for_a_later_good_one() {
        let mut s = Selector::new(4);
        assert!(!s.offer(flat(0)));
        assert!(!s.offer(flat(0)));
        assert!(s.offer(busy(7.0)));
        let p = s.finish(&info(), Backend::Native, Duration::ZERO).unwrap();
        assert!(!p.blank);
        assert_eq!((p.attempts, p.position_secs), (3, Some(7.0)));
    }

    #[test]
    fn all_blank_attempts_keep_the_brightest_after_the_limit() {
        let mut s = Selector::new(3);
        assert!(!s.offer(flat(0)));
        assert!(!s.offer(flat(200)));
        assert!(s.offer(flat(30)));
        let p = s.finish(&info(), Backend::Cli, Duration::ZERO).unwrap();
        assert!(p.blank);
        assert_eq!(p.position_secs, Some(200.0));
        assert_eq!(p.attempts, 3);
    }

    #[test]
    fn a_later_blank_never_replaces_a_good_frame() {
        let mut s = Selector::new(4);
        assert!(s.offer(busy(2.0)));
        s.offer(flat(255));
        assert_eq!(s.finish(&info(), Backend::Cli, Duration::ZERO).unwrap().position_secs, Some(2.0));
    }

    #[test]
    fn nothing_offered_means_no_poster() {
        assert!(Selector::new(2).finish(&info(), Backend::Cli, Duration::ZERO).is_none());
    }

    #[test]
    fn disabled_backend_reports_no_backend_and_yields_a_video_placeholder() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.mp4");
        std::fs::write(&file, b"not really a video").unwrap();
        let opts = PosterOptions { backend: BackendChoice::Disabled, ..Default::default() };
        assert_eq!(poster_frame(&file, &opts).err(), Some(VideoError::NoBackend));
        match thumb_image(&file, &opts).unwrap() {
            ThumbImage::Placeholder { kind, reason } => assert_eq!((kind, reason), (PlaceholderKind::Video, VideoError::NoBackend)),
            ThumbImage::Frame(_) => panic!("expected a placeholder"),
        }
    }

    #[test]
    fn missing_and_empty_files_are_errors_not_placeholders() {
        let dir = tempfile::tempdir().unwrap();
        let opts = PosterOptions { backend: BackendChoice::Disabled, ..Default::default() };
        assert!(matches!(thumb_image(&dir.path().join("none.mp4"), &opts), Err(VideoError::Missing(_))));
        let empty = dir.path().join("empty.mp4");
        std::fs::write(&empty, b"").unwrap();
        assert!(matches!(thumb_image(&empty, &opts), Err(VideoError::Empty)));
        assert!(matches!(thumb_image(dir.path(), &opts), Err(VideoError::Missing(_))));
    }
}
