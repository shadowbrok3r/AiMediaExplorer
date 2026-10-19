//! View mode and tile settings, and their app-wide persistence in `<config>/smart-media/view_prefs.json`.

#[cfg(test)]
use std::cell::RefCell;
use std::path::{Path, PathBuf};
#[cfg(not(test))]
use std::sync::{Mutex, OnceLock};

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const TILE_MIN: f32 = 64.0;
pub const TILE_MAX: f32 = 320.0;
pub const TILE_DEFAULT: f32 = 160.0;
/// Size change of one keyboard or wheel step.
pub const TILE_STEP: f32 = 16.0;
pub const PREFS_FILE: &str = "view_prefs.json";
const APP_DIR: &str = "smart-media";

/// How a browser tab shows its rows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ViewMode {
    #[default]
    Grid,
    Details,
}

impl ViewMode {
    pub const ALL: [ViewMode; 2] = [ViewMode::Grid, ViewMode::Details];

    pub fn label(self) -> &'static str {
        match self {
            ViewMode::Grid => "Grid",
            ViewMode::Details => "Details",
        }
    }

    pub fn other(self) -> ViewMode {
        match self {
            ViewMode::Grid => ViewMode::Details,
            ViewMode::Details => ViewMode::Grid,
        }
    }
}

/// How a thumbnail fills its tile.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TileFit {
    /// Cropped to a square.
    #[default]
    Crop,
    /// Whole image inside the square.
    Fit,
}

impl TileFit {
    pub fn label(self) -> &'static str {
        match self {
            TileFit::Crop => "Crop to square",
            TileFit::Fit => "Fit whole image",
        }
    }
}

/// The three tile size presets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SizePreset {
    Small,
    Medium,
    Large,
}

impl SizePreset {
    pub const ALL: [SizePreset; 3] = [SizePreset::Small, SizePreset::Medium, SizePreset::Large];

    pub fn px(self) -> f32 {
        match self {
            SizePreset::Small => 96.0,
            SizePreset::Medium => 160.0,
            SizePreset::Large => 256.0,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            SizePreset::Small => "S",
            SizePreset::Medium => "M",
            SizePreset::Large => "L",
        }
    }

    /// The preset whose size is `tile`, when it is exactly one.
    pub fn of(tile: f32) -> Option<SizePreset> {
        SizePreset::ALL.into_iter().find(|p| (p.px() - tile).abs() < 0.5)
    }
}

/// Clamps a tile size into the supported range; anything not finite becomes the default.
pub fn clamp_tile(tile: f32) -> f32 {
    if tile.is_finite() { tile.clamp(TILE_MIN, TILE_MAX) } else { TILE_DEFAULT }
}

/// Settings of the browser view that outlive a session.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ViewPrefs {
    pub mode: ViewMode,
    pub tile: f32,
    pub fit: TileFit,
    pub captions: bool,
    pub badges: bool,
}

impl Default for ViewPrefs {
    fn default() -> Self {
        Self { mode: ViewMode::Grid, tile: TILE_DEFAULT, fit: TileFit::Crop, captions: true, badges: true }
    }
}

impl ViewPrefs {
    /// The prefs with every field in its valid range.
    pub fn sanitized(mut self) -> Self {
        self.tile = clamp_tile(self.tile);
        self
    }

    /// Reads the prefs from JSON text; each field that is missing or invalid keeps its default.
    pub fn from_json(text: &str) -> Self {
        let mut prefs = Self::default();
        let Ok(Value::Object(map)) = serde_json::from_str::<Value>(text) else { return prefs };
        let field = |name: &str| map.get(name).cloned();
        if let Some(v) = field("mode").and_then(|v| serde_json::from_value(v).ok()) {
            prefs.mode = v;
        }
        if let Some(v) = field("tile").and_then(|v| v.as_f64()) {
            prefs.tile = v as f32;
        }
        if let Some(v) = field("fit").and_then(|v| serde_json::from_value(v).ok()) {
            prefs.fit = v;
        }
        if let Some(v) = field("captions").and_then(|v| v.as_bool()) {
            prefs.captions = v;
        }
        if let Some(v) = field("badges").and_then(|v| v.as_bool()) {
            prefs.badges = v;
        }
        prefs.sanitized()
    }
}

/// Loads prefs from `path`; a missing, unreadable or corrupt file gives the defaults.
pub fn load_from(path: &Path) -> ViewPrefs {
    match std::fs::read_to_string(path) {
        Ok(text) => ViewPrefs::from_json(&text),
        Err(e) => {
            if e.kind() != std::io::ErrorKind::NotFound {
                log::warn!("[view] cannot read {}: {e}", path.display());
            }
            ViewPrefs::default()
        }
    }
}

/// Writes `prefs` to `path` through a temporary file so a crash never leaves half a file.
pub fn save_to(path: &Path, prefs: &ViewPrefs) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    static WRITES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let json = serde_json::to_string_pretty(&prefs.sanitized()).map_err(std::io::Error::other)?;
    let tmp = path.with_extension(format!("json.{}.tmp", WRITES.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
    std::fs::write(&tmp, json)?;
    std::fs::rename(&tmp, path)
}

/// `<config dir>/smart-media/view_prefs.json`; none when the OS reports no config directory.
pub fn default_path() -> Option<PathBuf> {
    directories::BaseDirs::new().map(|d| d.config_dir().join(APP_DIR).join(PREFS_FILE))
}

/// The prefs, where they are saved and whether a change is waiting to be written.
#[derive(Debug)]
pub struct PrefsStore {
    prefs: ViewPrefs,
    path: Option<PathBuf>,
    dirty: bool,
}

impl PrefsStore {
    /// A store over `path`, loading what is saved there.
    pub fn open(path: Option<PathBuf>) -> Self {
        let prefs = path.as_deref().map(load_from).unwrap_or_default();
        Self { prefs, path, dirty: false }
    }

    pub fn prefs(&self) -> ViewPrefs {
        self.prefs
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Applies `edit`; the change is written by the next [`flush`](Self::flush).
    pub fn update(&mut self, edit: impl FnOnce(&mut ViewPrefs)) {
        let before = self.prefs;
        edit(&mut self.prefs);
        self.prefs = self.prefs.sanitized();
        self.dirty |= self.prefs != before;
    }

    /// The path and prefs of a pending change, which then counts as handed over.
    pub fn take_pending(&mut self) -> Option<(PathBuf, ViewPrefs)> {
        if !self.dirty {
            return None;
        }
        self.dirty = false;
        self.path.clone().map(|path| (path, self.prefs))
    }

    /// Marks the prefs as not yet written.
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Writes pending changes; a failed write is logged and retried by the next flush.
    pub fn flush(&mut self) -> bool {
        if !self.dirty {
            return false;
        }
        let Some(path) = self.path.as_deref() else {
            self.dirty = false;
            return false;
        };
        match save_to(path, &self.prefs) {
            Ok(()) => {
                self.dirty = false;
                true
            }
            Err(e) => {
                log::warn!("[view] cannot save {}: {e}", path.display());
                false
            }
        }
    }
}

#[cfg(not(test))]
fn global() -> &'static Mutex<PrefsStore> {
    static STORE: OnceLock<Mutex<PrefsStore>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(PrefsStore::open(default_path())))
}

#[cfg(test)]
thread_local! {
    static TEST_STORE: RefCell<PrefsStore> = RefCell::new(PrefsStore::open(None));
}

// Tests get a store per thread that never touches the disk.
fn with_store<R>(f: impl FnOnce(&mut PrefsStore) -> R) -> R {
    #[cfg(test)]
    {
        TEST_STORE.with(|s| f(&mut s.borrow_mut()))
    }
    #[cfg(not(test))]
    {
        f(&mut global().lock().unwrap_or_else(|e| e.into_inner()))
    }
}

/// The app-wide prefs new tabs start from.
pub fn app_prefs() -> ViewPrefs {
    with_store(|s| s.prefs())
}

/// Changes the app-wide prefs.
pub fn update_app_prefs(edit: impl FnOnce(&mut ViewPrefs)) {
    with_store(|s| s.update(edit));
}

/// Writes pending app-wide prefs to disk.
pub fn flush_app_prefs() -> bool {
    with_store(|s| s.flush())
}

/// Hands pending app-wide prefs to the writer thread; true when something was handed over.
pub fn flush_app_prefs_in_background() -> bool {
    match with_store(|s| s.take_pending()) {
        Some((path, prefs)) => {
            write_in_background(path, prefs);
            true
        }
        None => false,
    }
}

/// Writes `prefs` to `path` on the writer thread; a newer write queued meanwhile replaces an older one.
pub fn write_in_background(path: PathBuf, prefs: ViewPrefs) {
    static WRITER: std::sync::OnceLock<crossbeam::channel::Sender<(PathBuf, ViewPrefs)>> = std::sync::OnceLock::new();
    let tx = WRITER.get_or_init(|| {
        let (tx, rx) = crossbeam::channel::unbounded::<(PathBuf, ViewPrefs)>();
        let spawned = std::thread::Builder::new().name("view-prefs-writer".into()).spawn(move || {
            while let Ok(mut job) = rx.recv() {
                while let Ok(newer) = rx.try_recv() {
                    job = newer;
                }
                if let Err(e) = save_to(&job.0, &job.1) {
                    log::warn!("[view] cannot save {}: {e}", job.0.display());
                    #[cfg(not(test))]
                    global().lock().unwrap_or_else(|e| e.into_inner()).mark_dirty();
                }
            }
        });
        if let Err(e) = spawned {
            log::warn!("[view] no writer thread: {e}");
        }
        tx
    });
    if tx.send((path.clone(), prefs)).is_err() {
        let _ = save_to(&path, &prefs);
    }
}

/// Whether app-wide prefs wait to be written.
pub fn app_prefs_dirty() -> bool {
    with_store(|s| s.is_dirty())
}

/// Where the app-wide prefs are saved.
pub fn app_prefs_path() -> Option<PathBuf> {
    with_store(|s| s.path().map(Path::to_path_buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_is_the_grid_at_a_medium_size() {
        let p = ViewPrefs::default();
        assert_eq!((p.mode, p.tile, p.fit, p.captions, p.badges), (ViewMode::Grid, 160.0, TileFit::Crop, true, true));
        assert_eq!(SizePreset::of(p.tile), Some(SizePreset::Medium));
    }

    #[test]
    fn prefs_round_trip_through_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sub").join(PREFS_FILE);
        let prefs = ViewPrefs { mode: ViewMode::Details, tile: 212.0, fit: TileFit::Fit, captions: false, badges: false };
        save_to(&path, &prefs).unwrap();
        assert_eq!(load_from(&path), prefs);
        let left_over = std::fs::read_dir(path.parent().unwrap()).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().ends_with(".tmp")).count();
        assert_eq!(left_over, 0, "the temporary file is renamed away");
    }

    #[test]
    fn a_missing_file_gives_the_defaults() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(load_from(&dir.path().join("nope.json")), ViewPrefs::default());
    }

    #[test]
    fn corrupt_files_never_panic_and_give_the_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(PREFS_FILE);
        for junk in ["", "{", "not json", "[1,2,3]", "null", "\u{0}\u{1}\u{2}", "{\"mode\": 5}", "{\"tile\": \"wide\"}"] {
            std::fs::write(&path, junk).unwrap();
            assert_eq!(load_from(&path), ViewPrefs::default(), "{junk:?}");
        }
        std::fs::write(&path, [0xFFu8, 0xFE, 0x00, 0x80]).unwrap();
        assert_eq!(load_from(&path), ViewPrefs::default(), "invalid UTF-8");
    }

    #[test]
    fn each_valid_field_survives_next_to_an_invalid_one() {
        let p = ViewPrefs::from_json(r#"{"mode":"details","tile":"big","fit":"fit","captions":"yes","extra":1}"#);
        assert_eq!((p.mode, p.tile, p.fit, p.captions), (ViewMode::Details, TILE_DEFAULT, TileFit::Fit, true));
        let q = ViewPrefs::from_json(r#"{"mode":"carousel","tile":96}"#);
        assert_eq!((q.mode, q.tile), (ViewMode::Grid, 96.0));
    }

    #[test]
    fn out_of_range_sizes_are_clamped() {
        assert_eq!(ViewPrefs::from_json(r#"{"tile": 5}"#).tile, TILE_MIN);
        assert_eq!(ViewPrefs::from_json(r#"{"tile": 9000}"#).tile, TILE_MAX);
        assert_eq!(clamp_tile(f32::NAN), TILE_DEFAULT);
        assert_eq!(clamp_tile(f32::INFINITY), TILE_DEFAULT);
        assert_eq!(clamp_tile(-3.0), TILE_MIN);
    }

    #[test]
    fn the_store_writes_only_after_a_change_and_only_on_flush() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(PREFS_FILE);
        let mut store = PrefsStore::open(Some(path.clone()));
        assert!(!store.flush(), "nothing changed");
        assert!(!path.exists());
        store.update(|p| p.mode = ViewMode::Grid);
        assert!(!store.is_dirty(), "setting the current value is not a change");
        store.update(|p| p.mode = ViewMode::Details);
        assert!(store.is_dirty() && !path.exists());
        assert!(store.flush() && !store.is_dirty());
        assert_eq!(PrefsStore::open(Some(path)).prefs().mode, ViewMode::Details);
    }

    #[test]
    fn writes_at_the_same_time_never_fail_and_leave_one_of_the_values() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(PREFS_FILE);
        let writers: Vec<_> = (0..8)
            .map(|t| {
                let path = path.clone();
                std::thread::spawn(move || {
                    for i in 0..50 {
                        save_to(&path, &ViewPrefs { tile: 100.0 + (t * 50 + i) as f32 % 200.0, ..ViewPrefs::default() }).unwrap();
                    }
                })
            })
            .collect();
        writers.into_iter().for_each(|w| w.join().unwrap());
        let got = load_from(&path);
        assert!((100.0..300.0).contains(&got.tile), "{}", got.tile);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1, "only the prefs file is left");
    }

    #[test]
    fn a_pending_change_is_handed_over_once() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(PREFS_FILE);
        let mut store = PrefsStore::open(Some(path.clone()));
        assert_eq!(store.take_pending(), None, "nothing changed");
        store.update(|p| p.tile = 200.0);
        let (to, prefs) = store.take_pending().expect("a change is pending");
        assert_eq!((to, prefs.tile), (path, 200.0));
        assert!(!store.is_dirty());
        assert_eq!(store.take_pending(), None, "it is handed over once");
        store.mark_dirty();
        assert!(store.is_dirty());
    }

    #[test]
    fn a_store_without_a_path_hands_nothing_over() {
        let mut store = PrefsStore::open(None);
        store.update(|p| p.tile = 200.0);
        assert_eq!(store.take_pending(), None);
        assert!(!store.is_dirty());
    }

    #[test]
    fn the_writer_thread_writes_the_newest_prefs_without_the_caller_waiting() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("deep").join(PREFS_FILE);
        for tile in [100.0, 150.0, 220.0] {
            write_in_background(path.clone(), ViewPrefs { tile, ..ViewPrefs::default() });
        }
        let started = std::time::Instant::now();
        let wanted = loop {
            let got = load_from(&path);
            if got.tile == 220.0 || started.elapsed() > std::time::Duration::from_secs(5) {
                break got;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        };
        assert_eq!(wanted.tile, 220.0, "the newest write wins");
    }

    #[test]
    fn a_failed_write_keeps_the_change_pending() {
        let dir = tempfile::tempdir().unwrap();
        let blocker = dir.path().join("file");
        std::fs::write(&blocker, "x").unwrap();
        let mut store = PrefsStore::open(Some(blocker.join("inside").join(PREFS_FILE)));
        store.update(|p| p.tile = 100.0);
        assert!(!store.flush());
        assert!(store.is_dirty(), "the next flush retries");
    }

    #[test]
    fn updates_are_clamped() {
        let mut store = PrefsStore::open(None);
        store.update(|p| p.tile = 10_000.0);
        assert_eq!(store.prefs().tile, TILE_MAX);
        assert!(!store.flush(), "a store without a path never writes");
        assert!(!store.is_dirty());
    }

    #[test]
    fn app_prefs_are_per_thread_in_tests() {
        update_app_prefs(|p| p.tile = 200.0);
        assert_eq!(app_prefs().tile, 200.0);
        assert_eq!(std::thread::spawn(app_prefs).join().unwrap().tile, TILE_DEFAULT);
        assert!(app_prefs_path().is_none());
    }

    #[test]
    fn presets_map_to_their_sizes_and_back() {
        for p in SizePreset::ALL {
            assert_eq!(SizePreset::of(p.px()), Some(p));
        }
        assert_eq!(SizePreset::of(100.0), None);
        assert!(SizePreset::Small.px() < SizePreset::Medium.px() && SizePreset::Medium.px() < SizePreset::Large.px());
    }

    #[test]
    fn modes_serialize_in_snake_case() {
        assert_eq!(serde_json::to_string(&ViewMode::Details).unwrap(), "\"details\"");
        assert_eq!(serde_json::from_str::<TileFit>("\"fit\"").unwrap(), TileFit::Fit);
        assert_eq!(ViewMode::Grid.other(), ViewMode::Details);
    }
}
