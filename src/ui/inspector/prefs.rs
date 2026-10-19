//! Inspector preferences kept in `<config>/smart-media/inspector_prefs.json`; a missing or damaged file gives the defaults.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use super::view::PreviewMode;

pub const MIN_WIDTH: f32 = 280.0;
pub const MAX_WIDTH: f32 = 760.0;
pub const DEFAULT_WIDTH: f32 = 360.0;
/// Window width below which the rail becomes an overlay drawer.
pub const OVERLAY_BELOW: f32 = 900.0;
/// Delay between the last change and the write.
const SAVE_DELAY: Duration = Duration::from_millis(600);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RailTab {
    #[default]
    Inspect,
    Ask,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DetailTab {
    #[default]
    Info,
    Metadata,
    Ai,
    Edit,
}

impl DetailTab {
    pub const ALL: [DetailTab; 4] = [DetailTab::Info, DetailTab::Metadata, DetailTab::Ai, DetailTab::Edit];

    pub fn label(self) -> &'static str {
        match self {
            DetailTab::Info => "Info",
            DetailTab::Metadata => "Metadata",
            DetailTab::Ai => "AI",
            DetailTab::Edit => "Edit",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct InspectorPrefs {
    pub open: bool,
    pub width: f32,
    pub rail_tab: RailTab,
    pub detail_tab: DetailTab,
    pub preview_mode: PreviewMode,
}

impl Default for InspectorPrefs {
    fn default() -> Self {
        Self { open: true, width: DEFAULT_WIDTH, rail_tab: RailTab::default(), detail_tab: DetailTab::default(), preview_mode: PreviewMode::default() }
    }
}

impl InspectorPrefs {
    /// The prefs with the width inside its limits.
    pub fn sanitized(mut self) -> Self {
        self.width = clamp_width(self.width);
        self
    }
}

/// `width` inside the rail's limits; a non-finite width gives the default.
pub fn clamp_width(width: f32) -> f32 {
    if width.is_finite() { width.clamp(MIN_WIDTH, MAX_WIDTH) } else { DEFAULT_WIDTH }
}

/// Where the preferences live, when the platform has a config directory.
pub fn default_path() -> Option<PathBuf> {
    directories::BaseDirs::new().map(|d| d.config_dir().join("smart-media").join("inspector_prefs.json"))
}

/// Result of reading the file.
#[derive(Clone, Debug, PartialEq)]
pub struct Loaded {
    pub prefs: InspectorPrefs,
    /// Why the defaults were used although a file exists.
    pub error: Option<String>,
}

/// Reads the preferences; a missing file is not an error, a damaged one is reported and replaced by the defaults.
pub fn load(path: &Path) -> Loaded {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Loaded { prefs: InspectorPrefs::default(), error: None },
        Err(e) => return Loaded { prefs: InspectorPrefs::default(), error: Some(format!("cannot read {}: {e}", path.display())) },
    };
    match serde_json::from_str::<InspectorPrefs>(&text) {
        Ok(prefs) => Loaded { prefs: prefs.sanitized(), error: None },
        Err(e) => Loaded { prefs: InspectorPrefs::default(), error: Some(format!("{} is damaged: {e}", path.display())) },
    }
}

/// Writes the preferences through a temporary file so a crash never leaves half a file.
pub fn save(path: &Path, prefs: &InspectorPrefs) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let json = serde_json::to_string_pretty(prefs).map_err(std::io::Error::other)?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, json)?;
    std::fs::rename(&tmp, path)
}

/// Preferences plus the debounced write.
#[derive(Debug)]
pub struct PrefsStore {
    pub prefs: InspectorPrefs,
    path: Option<PathBuf>,
    dirty_since: Option<Instant>,
    pub load_error: Option<String>,
    pub last_save_error: Option<String>,
    pub saves: usize,
}

impl Default for PrefsStore {
    fn default() -> Self {
        Self { prefs: InspectorPrefs::default(), path: None, dirty_since: None, load_error: None, last_save_error: None, saves: 0 }
    }
}

impl PrefsStore {
    /// Loads `path`; with `None` the store keeps the defaults and never writes.
    pub fn open(path: Option<PathBuf>) -> Self {
        let loaded = path.as_deref().map(load);
        let (prefs, load_error) = match loaded {
            Some(l) => (l.prefs, l.error),
            None => (InspectorPrefs::default(), None),
        };
        if let Some(e) = &load_error {
            log::warn!("[inspector] {e}; using the defaults");
        }
        Self { prefs, path, dirty_since: None, load_error, last_save_error: None, saves: 0 }
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Applies `edit`; the file is written after the changes settle.
    pub fn update(&mut self, now: Instant, edit: impl FnOnce(&mut InspectorPrefs)) {
        let before = self.prefs.clone();
        edit(&mut self.prefs);
        self.prefs = std::mem::take(&mut self.prefs).sanitized();
        if self.prefs != before {
            self.dirty_since = Some(now);
        }
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty_since.is_some()
    }

    /// Writes the file when the changes have settled; returns whether it wrote.
    pub fn save_if_due(&mut self, now: Instant) -> bool {
        match self.dirty_since {
            Some(since) if now.saturating_duration_since(since) >= SAVE_DELAY => self.save_now(),
            _ => false,
        }
    }

    pub fn save_now(&mut self) -> bool {
        self.dirty_since = None;
        let Some(path) = self.path.as_deref() else { return false };
        match save(path, &self.prefs) {
            Ok(()) => {
                self.saves += 1;
                self.last_save_error = None;
                true
            }
            Err(e) => {
                log::warn!("[inspector] cannot write {}: {e}", path.display());
                self.last_save_error = Some(e.to_string());
                false
            }
        }
    }

    /// Time until the pending write is due.
    pub fn due_in(&self, now: Instant) -> Option<Duration> {
        self.dirty_since.map(|since| SAVE_DELAY.saturating_sub(now.saturating_duration_since(since)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("smart-media").join("inspector_prefs.json");
        (dir, path)
    }

    #[test]
    fn the_defaults_open_the_rail_at_its_default_width() {
        let p = InspectorPrefs::default();
        assert!(p.open);
        assert_eq!(p.width, DEFAULT_WIDTH);
        assert_eq!((p.rail_tab, p.detail_tab, p.preview_mode), (RailTab::Inspect, DetailTab::Info, PreviewMode::Fit));
    }

    #[test]
    fn a_missing_file_gives_the_defaults_without_an_error() {
        let (_dir, path) = file();
        assert_eq!(load(&path), Loaded { prefs: InspectorPrefs::default(), error: None });
    }

    #[test]
    fn preferences_survive_a_write_and_read() {
        let (_dir, path) = file();
        let prefs = InspectorPrefs { open: false, width: 512.0, rail_tab: RailTab::Ask, detail_tab: DetailTab::Metadata, preview_mode: PreviewMode::Actual };
        save(&path, &prefs).unwrap();
        assert_eq!(load(&path), Loaded { prefs, error: None });
        assert!(!path.with_extension("json.tmp").exists(), "the temporary file is renamed away");
    }

    #[test]
    fn damaged_files_give_the_defaults_and_an_error_instead_of_a_panic() {
        let (_dir, path) = file();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        for junk in ["", "{", "[1,2]", "{\"open\": \"yes\"}", "{\"rail_tab\": \"sideways\"}", "\u{0}\u{1}\u{2}"] {
            std::fs::write(&path, junk).unwrap();
            let loaded = load(&path);
            assert_eq!(loaded.prefs, InspectorPrefs::default(), "{junk:?}");
            assert!(loaded.error.as_deref().is_some_and(|e| e.contains("damaged")), "{junk:?}: {:?}", loaded.error);
        }
        std::fs::write(&path, [0xff, 0xfe, 0x00]).unwrap();
        assert!(load(&path).error.is_some(), "invalid UTF-8 is reported too");
    }

    #[test]
    fn a_directory_in_place_of_the_file_is_reported() {
        let (_dir, path) = file();
        std::fs::create_dir_all(&path).unwrap();
        assert!(load(&path).error.is_some());
    }

    #[test]
    fn unknown_and_missing_fields_are_tolerated() {
        let (_dir, path) = file();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, r#"{"width": 400, "future_field": [1]}"#).unwrap();
        let loaded = load(&path);
        assert_eq!(loaded.error, None);
        assert_eq!((loaded.prefs.width, loaded.prefs.open), (400.0, true));
    }

    #[test]
    fn widths_are_clamped_and_non_finite_widths_reset() {
        assert_eq!(clamp_width(10.0), MIN_WIDTH);
        assert_eq!(clamp_width(5000.0), MAX_WIDTH);
        assert_eq!(clamp_width(f32::NAN), DEFAULT_WIDTH);
        assert_eq!(clamp_width(f32::INFINITY), DEFAULT_WIDTH);
        let (_dir, path) = file();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, r#"{"width": 99999}"#).unwrap();
        assert_eq!(load(&path).prefs.width, MAX_WIDTH);
    }

    #[test]
    fn the_store_writes_after_the_changes_settle() {
        let (_dir, path) = file();
        let mut store = PrefsStore::open(Some(path.clone()));
        let t0 = Instant::now();
        store.update(t0, |p| p.width = 420.0);
        assert!(store.is_dirty());
        assert!(!store.save_if_due(t0 + Duration::from_millis(100)), "still settling");
        assert!(!path.exists());
        store.update(t0 + Duration::from_millis(300), |p| p.width = 430.0);
        assert!(!store.save_if_due(t0 + Duration::from_millis(700)), "a newer change restarts the wait");
        assert!(store.save_if_due(t0 + Duration::from_millis(1000)));
        assert_eq!(load(&path).prefs.width, 430.0);
        assert!(!store.is_dirty());
        store.update(t0 + Duration::from_secs(2), |p| p.width = 430.0);
        assert!(!store.is_dirty(), "an unchanged value writes nothing");
    }

    #[test]
    fn a_store_without_a_path_never_writes_and_a_failed_write_is_remembered() {
        let mut none = PrefsStore::open(None);
        none.update(Instant::now(), |p| p.open = false);
        assert!(!none.save_now());
        assert_eq!(none.saves, 0);

        let dir = tempfile::tempdir().unwrap();
        let blocker = dir.path().join("file");
        std::fs::write(&blocker, "x").unwrap();
        let mut blocked = PrefsStore::open(Some(blocker.join("sub").join("p.json")));
        blocked.update(Instant::now(), |p| p.open = false);
        assert!(!blocked.save_now());
        assert!(blocked.last_save_error.is_some());
    }

    #[test]
    fn a_store_opened_on_a_damaged_file_keeps_the_error_for_the_devtools() {
        let (_dir, path) = file();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "nope").unwrap();
        let store = PrefsStore::open(Some(path));
        assert!(store.load_error.is_some());
        assert_eq!(store.prefs, InspectorPrefs::default());
    }
}
