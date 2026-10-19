//! The data of the Image Edit tab without any drawing: parameters, presets, the job phases, error advice and output file names.

use std::path::{Path, PathBuf};

/// Prompt presets: a short label and the instruction it fills in.
pub const PRESETS: [(&str, &str); 10] = [
    ("Enhance light", "Improve the lighting and exposure, keep the colors natural"),
    ("Black and white", "Convert to a high contrast black and white photograph"),
    ("Remove background", "Remove the background and replace it with plain white"),
    ("Sharpen", "Sharpen the details and reduce noise"),
    ("Golden hour", "Relight the scene with warm golden hour sunlight"),
    ("Night", "Turn the scene into a night shot with soft artificial lights"),
    ("Vintage film", "Give it the look of a faded vintage film photograph with grain"),
    ("Watercolor", "Repaint the image as a soft watercolor illustration"),
    ("Remove people", "Remove the people and fill in the background naturally"),
    ("Clean up", "Remove small distractions and blemishes, keep everything else unchanged"),
];

/// Settings of one edit.
#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    pub prompt: String,
    pub negative_prompt: String,
    pub guidance_scale: f32,
    pub strength: f32,
    pub steps: usize,
    /// A live preview every this many steps; zero turns it off.
    pub preview_every_n: usize,
    pub prefer_full_precision: bool,
}

impl Default for Params {
    fn default() -> Self {
        Self { prompt: String::new(), negative_prompt: String::new(), guidance_scale: 7.5, strength: 0.7, steps: 30, preview_every_n: 0, prefer_full_precision: true }
    }
}

impl Params {
    /// Why an edit cannot start with these settings, if it cannot.
    pub fn problem(&self) -> Option<&'static str> {
        if self.prompt.trim().is_empty() { Some("Describe the change first") } else { None }
    }

    /// The prompt with `preset` set, or added to the end when `append`.
    pub fn with_preset(&self, preset: &str, append: bool) -> String {
        if append && !self.prompt.trim().is_empty() { format!("{}. {}", self.prompt.trim().trim_end_matches('.'), preset) } else { preset.to_string() }
    }
}

/// Where the model comes from.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelSettings {
    pub model_repo: String,
    pub gguf_repo: String,
    pub gguf_file: String,
    pub gguf_prefer: bool,
    /// The local path of the GGUF file once it is fetched.
    pub gguf_path: Option<String>,
}

impl Default for ModelSettings {
    fn default() -> Self {
        Self { model_repo: "Qwen/Qwen-Image-Edit".to_string(), gguf_repo: "QuantStack/Qwen-Image-Edit-GGUF".to_string(), gguf_file: "Qwen_Image_Edit-Q6_K.gguf".to_string(), gguf_prefer: true, gguf_path: None }
    }
}

/// What went wrong, as far as the advice differs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    OutOfMemory,
    ModelLoad,
    Source,
    Other,
}

/// A failed edit or model load with what to do about it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EditError {
    pub kind: ErrorKind,
    pub message: String,
}

impl EditError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into() }
    }

    /// An error of the kind the text suggests.
    pub fn classified(message: impl Into<String>) -> Self {
        let message = message.into();
        Self { kind: classify(&message), message }
    }

    /// What the user can do about it.
    pub fn advice(&self) -> &'static str {
        match self.kind {
            ErrorKind::OutOfMemory => "The GPU ran out of memory. Try fewer steps, turn off Full precision, close other programs that use the GPU, or load a smaller GGUF model.",
            ErrorKind::ModelLoad => "The model could not be loaded. Check the model repo under Model settings, the network and the free disk space; fetch the GGUF file first if you use one.",
            ErrorKind::Source => "The source image could not be read. Check that the file still exists and is a picture, or choose another image.",
            ErrorKind::Other => "The edit failed. Try again, or change the settings; Logs has the details.",
        }
    }

    /// True when a lower precision may help.
    pub fn may_retry_lower_precision(&self) -> bool {
        self.kind == ErrorKind::OutOfMemory
    }
}

/// The kind of failure a message describes.
pub fn classify(message: &str) -> ErrorKind {
    let m = message.to_ascii_lowercase();
    if m.contains("out of memory") || m.contains("oom") || m.contains("cuda_error_out_of_memory") || m.contains("alloc") || m.contains("named symbol not found") || m.contains("cuda") && m.contains("memory") {
        ErrorKind::OutOfMemory
    } else if m.contains("no such file") || m.contains("cannot open") || m.contains("unsupported image") || m.contains("decod") || m.contains("source image") {
        ErrorKind::Source
    } else if m.contains("load") || m.contains("repo") || m.contains("hf-hub") || m.contains("download") || m.contains("safetensors") || m.contains("gguf") {
        ErrorKind::ModelLoad
    } else {
        ErrorKind::Other
    }
}

/// Where a job is.
#[derive(Clone, Debug, PartialEq)]
pub enum Phase {
    Idle,
    LoadingModel,
    Running { done: u64, total: u64, detail: String },
    /// The cancel was asked for; the pipeline stops at its next step.
    Cancelling,
    Done,
    Failed(EditError),
    Cancelled,
}

impl Phase {
    /// True while a job holds the model.
    pub fn is_busy(&self) -> bool {
        matches!(self, Self::LoadingModel | Self::Running { .. } | Self::Cancelling)
    }

    pub fn fraction(&self) -> Option<f32> {
        match self {
            Self::Running { done, total, .. } if *total > 0 => Some((*done as f32 / *total as f32).clamp(0.0, 1.0)),
            _ => None,
        }
    }

    /// One line for the status row.
    pub fn label(&self) -> String {
        match self {
            Self::Idle => "Ready".to_string(),
            Self::LoadingModel => "Loading the model".to_string(),
            Self::Running { done, total, detail } if *total > 0 => {
                if detail.is_empty() { format!("Editing, step {done} of {total}") } else { format!("{detail} ({done} of {total})") }
            }
            Self::Running { .. } => "Editing".to_string(),
            Self::Cancelling => "Stopping".to_string(),
            Self::Done => "Done".to_string(),
            Self::Failed(e) => format!("Failed: {}", e.message),
            Self::Cancelled => "Cancelled".to_string(),
        }
    }
}

/// How the source and the result are compared.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CompareMode {
    #[default]
    SideBySide,
    Swipe,
    Toggle,
}

impl CompareMode {
    pub const ALL: [CompareMode; 3] = [Self::SideBySide, Self::Swipe, Self::Toggle];

    pub fn label(self) -> &'static str {
        match self {
            Self::SideBySide => "Side by side",
            Self::Swipe => "Swipe",
            Self::Toggle => "Flip",
        }
    }
}

/// A path next to `source` that does not exist: `name-edited.png`, then `name-edited-2.png` and so on.
pub fn unique_output_path(source: &str, extension: &str, exists: impl Fn(&Path) -> bool) -> PathBuf {
    let src = Path::new(source);
    let dir = src.parent().unwrap_or_else(|| Path::new("."));
    let stem = src.file_stem().and_then(|s| s.to_str()).filter(|s| !s.is_empty()).unwrap_or("image");
    let first = dir.join(format!("{stem}-edited.{extension}"));
    if !exists(&first) {
        return first;
    }
    (2..10_000).map(|n| dir.join(format!("{stem}-edited-{n}.{extension}"))).find(|p| !exists(p)).unwrap_or(first)
}

/// True when `path` is the file the edit started from.
pub fn is_same_file(a: &str, b: &str) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(x), Ok(y)) => x == y,
        _ => a == b,
    }
}

/// File name of a path for display.
pub fn file_name(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

/// Pixels of an image at most `max` on its longer side, keeping the ratio.
pub fn fit_dims(width: u32, height: u32, max: u32) -> (u32, u32) {
    let longest = width.max(height);
    if longest <= max || longest == 0 {
        return (width, height);
    }
    let scale = max as f64 / longest as f64;
    (((width as f64 * scale).round() as u32).max(1), ((height as f64 * scale).round() as u32).max(1))
}
