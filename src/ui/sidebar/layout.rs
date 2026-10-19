//! Window layout kept between runs: chrome visibility, sidebar width and section state in `<config>/smart-media/ui_layout.json`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Version written to the file; a newer file is ignored and never overwritten.
pub const LAYOUT_VERSION: u32 = 1;
/// File name inside the app's config directory.
pub const FILE_NAME: &str = "ui_layout.json";
pub const DEFAULT_WIDTH: f32 = 240.0;
pub const MIN_WIDTH: f32 = 170.0;
pub const MAX_WIDTH: f32 = 420.0;
/// Most section entries the file keeps.
const MAX_SECTIONS: usize = 64;

fn version() -> u32 {
    LAYOUT_VERSION
}

fn yes() -> bool {
    true
}

fn default_width() -> f32 {
    DEFAULT_WIDTH
}

/// The saved window layout.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UiLayout {
    #[serde(default = "version")]
    pub version: u32,
    #[serde(default = "yes")]
    pub sidebar_open: bool,
    #[serde(default = "default_width")]
    pub sidebar_width: f32,
    /// Open state per sidebar section key; absent keys use the section default.
    #[serde(default)]
    pub section_open: BTreeMap<String, bool>,
    #[serde(default = "yes")]
    pub bottom_bar_open: bool,
    #[serde(default)]
    pub right_rail_open: bool,
}

impl Default for UiLayout {
    fn default() -> Self {
        Self {
            version: LAYOUT_VERSION,
            sidebar_open: true,
            sidebar_width: DEFAULT_WIDTH,
            section_open: BTreeMap::new(),
            bottom_bar_open: true,
            right_rail_open: false,
        }
    }
}

/// Why a layout text was not used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayoutError {
    Invalid(String),
    TooNew(u32),
}

impl std::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(reason) => write!(f, "unreadable layout: {reason}"),
            Self::TooNew(v) => write!(f, "layout version {v} is newer than {LAYOUT_VERSION}"),
        }
    }
}

impl UiLayout {
    /// Reads a layout; missing fields take defaults and out-of-range values are clamped.
    pub fn parse(text: &str) -> Result<Self, LayoutError> {
        let mut layout: Self = serde_json::from_str(text).map_err(|e| LayoutError::Invalid(e.to_string()))?;
        if layout.version > LAYOUT_VERSION {
            return Err(LayoutError::TooNew(layout.version));
        }
        layout.sanitize();
        Ok(layout)
    }

    /// Pretty JSON of the layout.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    fn sanitize(&mut self) {
        self.version = LAYOUT_VERSION;
        self.sidebar_width = clamp_width(self.sidebar_width);
        while self.section_open.len() > MAX_SECTIONS {
            let Some(key) = self.section_open.keys().next_back().cloned() else { break };
            self.section_open.remove(&key);
        }
    }

    /// Open state of the section `key`, `default` when never set.
    pub fn is_section_open(&self, key: &str, default: bool) -> bool {
        self.section_open.get(key).copied().unwrap_or(default)
    }

    pub fn set_section_open(&mut self, key: &str, open: bool) {
        self.section_open.insert(key.to_string(), open);
    }
}

/// `width` limited to the sidebar range; a non-finite width becomes the default.
pub fn clamp_width(width: f32) -> f32 {
    if width.is_finite() { width.clamp(MIN_WIDTH, MAX_WIDTH) } else { DEFAULT_WIDTH }
}

/// `<config dir>/smart-media/ui_layout.json`.
pub fn default_path() -> Option<PathBuf> {
    directories::BaseDirs::new().map(|dirs| dirs.config_dir().join(crate::ui::titlebar::APP_ID).join(FILE_NAME))
}

/// What loading the file found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoadNote {
    /// No path: the layout lives in memory only.
    Memory,
    Missing,
    Loaded,
    Corrupt(String),
    TooNew(u32),
}

impl LoadNote {
    pub fn text(&self) -> String {
        match self {
            Self::Memory => "memory only".to_string(),
            Self::Missing => "no saved file, defaults".to_string(),
            Self::Loaded => "loaded".to_string(),
            Self::Corrupt(why) => format!("unreadable, defaults ({why})"),
            Self::TooNew(v) => format!("file version {v} is newer, defaults and no saving"),
        }
    }
}

/// Reads the layout at `path`; never fails.
pub fn load_from(path: &Path) -> (UiLayout, LoadNote) {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return (UiLayout::default(), LoadNote::Missing),
        Err(e) => return (UiLayout::default(), LoadNote::Corrupt(e.to_string())),
    };
    match UiLayout::parse(&text) {
        Ok(layout) => (layout, LoadNote::Loaded),
        Err(LayoutError::TooNew(v)) => (UiLayout::default(), LoadNote::TooNew(v)),
        Err(LayoutError::Invalid(why)) => (UiLayout::default(), LoadNote::Corrupt(why)),
    }
}

/// Writes the layout to a temporary file, then renames it over `path`.
pub fn save_to(path: &Path, layout: &UiLayout) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, layout.to_json())?;
    std::fs::rename(&tmp, path)
}

/// The live layout with its file and change tracking.
#[derive(Clone, Debug)]
pub struct LayoutStore {
    pub layout: UiLayout,
    saved: UiLayout,
    path: Option<PathBuf>,
    pub note: LoadNote,
    pub last_error: Option<String>,
    pub saves: u32,
    writable: bool,
}

impl Default for LayoutStore {
    fn default() -> Self {
        Self::memory()
    }
}

impl LayoutStore {
    /// A store that never touches the disk.
    pub fn memory() -> Self {
        Self { layout: UiLayout::default(), saved: UiLayout::default(), path: None, note: LoadNote::Memory, last_error: None, saves: 0, writable: false }
    }

    /// The store of the config directory file.
    pub fn load() -> Self {
        match default_path() {
            Some(path) => Self::at(path),
            None => Self::memory(),
        }
    }

    /// The store of the file at `path`.
    pub fn at(path: PathBuf) -> Self {
        let (layout, note) = load_from(&path);
        let writable = !matches!(note, LoadNote::TooNew(_));
        Self { saved: layout.clone(), layout, path: Some(path), note, last_error: None, saves: 0, writable }
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    pub fn is_dirty(&self) -> bool {
        self.layout != self.saved
    }

    /// Writes the layout when it differs from what was last written; true when a write happened.
    pub fn save_if_changed(&mut self) -> bool {
        if !self.is_dirty() {
            return false;
        }
        let Some(path) = self.path.clone().filter(|_| self.writable) else {
            self.saved = self.layout.clone();
            return false;
        };
        match save_to(&path, &self.layout) {
            Ok(()) => {
                self.saved = self.layout.clone();
                self.last_error = None;
                self.saves += 1;
                true
            }
            Err(e) => {
                if self.last_error.is_none() {
                    log::warn!("[layout] cannot save {}: {e}", path.display());
                }
                self.last_error = Some(e.to_string());
                self.saved = self.layout.clone();
                false
            }
        }
    }

    /// Back to the defaults; the next save writes them.
    pub fn reset(&mut self) {
        self.layout = UiLayout::default();
    }
}
