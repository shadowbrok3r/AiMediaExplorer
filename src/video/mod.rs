//! Video support: probing, poster frames, thumbnails and preview playback behind one backend-neutral API.

pub mod cli;
pub mod gate;
pub mod info;
#[cfg(ame_ffmpeg)]
pub mod native;
pub mod placeholder;
#[cfg(ame_ffmpeg)]
pub mod player;
pub mod poster;
pub mod probe_cli;
pub mod proc;
pub mod quality;
#[cfg(ame_ffmpeg)]
pub mod view;
#[cfg(test)]
pub(crate) mod fixtures;
#[cfg(test)]
mod tests;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

pub use info::{AudioTrack, VideoInfo};
pub use poster::{Poster, PosterOptions, ThumbImage, poster_frame, poster_frame_async, thumb_image};

/// Why a video operation produced nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VideoError {
    Missing(String),
    Empty,
    NoVideoStream,
    Corrupt(String),
    Timeout,
    Cancelled,
    NoBackend,
    Backend(String),
}

impl std::fmt::Display for VideoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VideoError::Missing(why) => write!(f, "file unavailable: {why}"),
            VideoError::Empty => f.write_str("file is empty"),
            VideoError::NoVideoStream => f.write_str("no video stream"),
            VideoError::Corrupt(why) => write!(f, "unreadable media: {why}"),
            VideoError::Timeout => f.write_str("timed out"),
            VideoError::Cancelled => f.write_str("cancelled"),
            VideoError::NoBackend => f.write_str("no video backend: build with ffmpeg or install the ffmpeg command"),
            VideoError::Backend(why) => write!(f, "video backend error: {why}"),
        }
    }
}

impl std::error::Error for VideoError {}

impl VideoError {
    /// Errors that a generated placeholder thumbnail stands in for.
    pub fn wants_placeholder(&self) -> bool {
        matches!(self, VideoError::NoVideoStream | VideoError::NoBackend)
    }
}

/// Shared flag that ends a running video job.
#[derive(Clone, Debug, Default)]
pub struct Cancel(Arc<AtomicBool>);

impl Cancel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// Time limit plus cancellation flag checked by every backend loop.
#[derive(Clone, Debug)]
pub struct Deadline {
    at: Option<Instant>,
    cancel: Cancel,
}

impl Deadline {
    pub fn new(timeout: Duration, cancel: Cancel) -> Self {
        Self { at: Instant::now().checked_add(timeout), cancel }
    }

    pub fn unlimited(cancel: Cancel) -> Self {
        Self { at: None, cancel }
    }

    /// The error that applies now, if the job has to stop.
    pub fn reason(&self) -> Option<VideoError> {
        if self.cancel.is_cancelled() {
            Some(VideoError::Cancelled)
        } else if self.at.is_some_and(|at| Instant::now() >= at) {
            Some(VideoError::Timeout)
        } else {
            None
        }
    }

    pub fn check(&self) -> Result<(), VideoError> {
        self.reason().map_or(Ok(()), Err)
    }
}

/// Which implementation runs a request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BackendChoice {
    /// In-process ffmpeg when compiled in, otherwise the ffmpeg command.
    #[default]
    Auto,
    Native,
    Cli,
    /// No backend at all, for exercising the placeholder path.
    Disabled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Native,
    Cli,
}

impl Backend {
    pub fn label(self) -> &'static str {
        match self {
            Backend::Native => "ffmpeg (in-process)",
            Backend::Cli => "ffmpeg (command)",
        }
    }
}

pub const fn native_available() -> bool {
    cfg!(ame_ffmpeg)
}

fn find_tool(name: &str, env_key: &str) -> Option<PathBuf> {
    if let Some(custom) = std::env::var_os(env_key).map(PathBuf::from).filter(|p| p.is_file()) {
        return Some(custom);
    }
    let exe = format!("{name}{}", std::env::consts::EXE_SUFFIX);
    std::env::split_paths(&std::env::var_os("PATH")?).map(|dir| dir.join(&exe)).find(|p| p.is_file())
}

/// The ffmpeg command: `AME_FFMPEG` or the first match on `PATH`.
pub fn ffmpeg_path() -> Option<&'static Path> {
    static PATH: OnceLock<Option<PathBuf>> = OnceLock::new();
    PATH.get_or_init(|| find_tool("ffmpeg", "AME_FFMPEG")).as_deref()
}

/// The ffprobe command: `AME_FFPROBE` or the first match on `PATH`.
pub fn ffprobe_path() -> Option<&'static Path> {
    static PATH: OnceLock<Option<PathBuf>> = OnceLock::new();
    PATH.get_or_init(|| find_tool("ffprobe", "AME_FFPROBE")).as_deref()
}

/// Backends that can serve a request for `choice`, in the order they are tried.
pub fn backends_for(choice: BackendChoice) -> Vec<Backend> {
    let native = native_available().then_some(Backend::Native);
    let cli = ffmpeg_path().is_some().then_some(Backend::Cli);
    match choice {
        BackendChoice::Auto => native.into_iter().chain(cli).collect(),
        BackendChoice::Native => native.into_iter().collect(),
        BackendChoice::Cli => cli.into_iter().collect(),
        BackendChoice::Disabled => Vec::new(),
    }
}

/// One-line description of the active video backends.
pub fn backend_summary() -> String {
    let native = if native_available() { "in-process ffmpeg: yes" } else { "in-process ffmpeg: no" };
    let cli = match ffmpeg_path() {
        Some(p) => format!("ffmpeg command: {}", p.display()),
        None => "ffmpeg command: not found".to_string(),
    };
    let probe = match ffprobe_path() {
        Some(p) => format!("ffprobe: {}", p.display()),
        None => "ffprobe: not found".to_string(),
    };
    format!("{native}; {cli}; {probe}")
}

/// Size of a readable, non-empty regular file.
pub fn check_file(path: &Path) -> Result<u64, VideoError> {
    let meta = std::fs::metadata(path).map_err(|e| VideoError::Missing(e.to_string()))?;
    if !meta.is_file() {
        return Err(VideoError::Missing("not a regular file".into()));
    }
    if meta.len() == 0 {
        return Err(VideoError::Empty);
    }
    Ok(meta.len())
}

#[derive(Clone, Debug)]
pub struct ProbeOptions {
    pub timeout: Duration,
    pub cancel: Cancel,
    pub backend: BackendChoice,
}

impl Default for ProbeOptions {
    fn default() -> Self {
        Self { timeout: Duration::from_secs(15), cancel: Cancel::new(), backend: BackendChoice::Auto }
    }
}

/// Container and stream facts of a media file.
pub fn probe(path: &Path) -> Result<VideoInfo, VideoError> {
    probe_with(path, &ProbeOptions::default())
}

pub fn probe_with(path: &Path, opts: &ProbeOptions) -> Result<VideoInfo, VideoError> {
    let size = check_file(path)?;
    let deadline = Deadline::new(opts.timeout, opts.cancel.clone());
    let mut last = VideoError::NoBackend;
    for backend in backends_for(opts.backend) {
        let result = match backend {
            #[cfg(ame_ffmpeg)]
            Backend::Native => native::probe(path, size, &deadline),
            #[cfg(not(ame_ffmpeg))]
            Backend::Native => Err(VideoError::NoBackend),
            Backend::Cli => probe_cli::probe(path, size, &deadline),
        };
        match result {
            Err(VideoError::Backend(why)) => last = VideoError::Backend(why),
            other => return other,
        }
    }
    Err(last)
}

/// Async wrapper that runs the probe on the blocking pool.
pub async fn probe_async(path: PathBuf, opts: ProbeOptions) -> Result<VideoInfo, VideoError> {
    tokio::task::spawn_blocking(move || probe_with(&path, &opts)).await.unwrap_or_else(|e| Err(VideoError::Backend(format!("probe task failed: {e}"))))
}
