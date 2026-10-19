//! Interface preferences kept between runs in `<config>/smart-media/ui_prefs.json`: startup behaviour, confirmations, watched folders and the setup checklist.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Version written to the file; a newer file is ignored and never overwritten.
pub const VERSION: u32 = 1;
/// File name inside the app's config directory.
pub const FILE_NAME: &str = "ui_prefs.json";
/// Most watched folders and restored tabs the file keeps.
const MAX_LIST: usize = 64;

fn version() -> u32 {
    VERSION
}

/// What the first-run checklist knows about one step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepState {
    #[default]
    Pending,
    Done,
    Skipped,
}

impl StepState {
    pub fn is_finished(self) -> bool {
        self != Self::Pending
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Pending => "to do",
            Self::Done => "done",
            Self::Skipped => "skipped",
        }
    }
}

/// How the AI features are meant to run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AiMode {
    #[default]
    Off,
    Local,
    Zeroclaw,
}

impl AiMode {
    pub const ALL: [AiMode; 3] = [Self::Off, Self::Local, Self::Zeroclaw];

    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "Off",
            Self::Local => "Local models",
            Self::Zeroclaw => "ZeroClaw",
        }
    }
}

/// What the application does when it starts.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Startup {
    /// Restore the workspace (`workspace.json`: tabs, layout and window) at launch.
    pub restore_tabs: bool,
    /// Restore only the scope the focused tab showed, in the main tab.
    pub restore_last_scope: bool,
    /// Scope of the focused browser as older versions recorded it; read only while no workspace file exists.
    pub last_scope: Option<String>,
    /// Scopes of the open browsers as older versions recorded them; read only while no workspace file exists.
    pub tabs: Vec<String>,
}

/// Questions the application asks before it acts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Confirmations {
    pub delete_from_library: bool,
    pub move_files: bool,
    pub import_backup: bool,
    pub restore_snapshot: bool,
}

impl Default for Confirmations {
    fn default() -> Self {
        Self { delete_from_library: true, move_files: true, import_backup: true, restore_snapshot: true }
    }
}

/// One confirmation the General section can switch off.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Confirm {
    DeleteFromLibrary,
    MoveFiles,
    ImportBackup,
    RestoreSnapshot,
}

impl Confirm {
    pub const ALL: [Confirm; 4] = [Self::DeleteFromLibrary, Self::MoveFiles, Self::ImportBackup, Self::RestoreSnapshot];

    pub fn label(self) -> &'static str {
        match self {
            Self::DeleteFromLibrary => "Before removing files from the library",
            Self::MoveFiles => "Before moving or copying files",
            Self::ImportBackup => "Before importing a backup into the library",
            Self::RestoreSnapshot => "Before restoring a snapshot",
        }
    }
}

impl Confirmations {
    pub fn get(&self, which: Confirm) -> bool {
        match which {
            Confirm::DeleteFromLibrary => self.delete_from_library,
            Confirm::MoveFiles => self.move_files,
            Confirm::ImportBackup => self.import_backup,
            Confirm::RestoreSnapshot => self.restore_snapshot,
        }
    }

    pub fn set(&mut self, which: Confirm, on: bool) {
        match which {
            Confirm::DeleteFromLibrary => self.delete_from_library = on,
            Confirm::MoveFiles => self.move_files = on,
            Confirm::ImportBackup => self.import_backup = on,
            Confirm::RestoreSnapshot => self.restore_snapshot = on,
        }
    }
}

/// Folders the library is built from.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct LibraryPrefs {
    pub watched: Vec<String>,
}

/// The first-run checklist.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Checklist {
    /// The user closed the checklist card for good.
    pub dismissed: bool,
    pub location: StepState,
    pub folders: StepState,
    pub ai: StepState,
    pub scan: StepState,
    pub ai_mode: AiMode,
}

/// The saved interface preferences.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UiPrefs {
    #[serde(default = "version")]
    pub version: u32,
    #[serde(default)]
    pub startup: Startup,
    #[serde(default)]
    pub confirmations: Confirmations,
    #[serde(default)]
    pub library: LibraryPrefs,
    #[serde(default)]
    pub checklist: Checklist,
}

impl Default for UiPrefs {
    fn default() -> Self {
        Self { version: VERSION, startup: Startup::default(), confirmations: Confirmations::default(), library: LibraryPrefs::default(), checklist: Checklist::default() }
    }
}

/// Why a prefs text was not used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PrefsError {
    Invalid(String),
    TooNew(u32),
}

impl std::fmt::Display for PrefsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(reason) => write!(f, "unreadable preferences: {reason}"),
            Self::TooNew(v) => write!(f, "preferences version {v} is newer than {VERSION}"),
        }
    }
}

impl UiPrefs {
    /// Reads preferences; missing fields take defaults and over-long lists are cut.
    pub fn parse(text: &str) -> Result<Self, PrefsError> {
        let mut prefs: Self = serde_json::from_str(text).map_err(|e| PrefsError::Invalid(e.to_string()))?;
        if prefs.version > VERSION {
            return Err(PrefsError::TooNew(prefs.version));
        }
        prefs.sanitize();
        Ok(prefs)
    }

    /// Pretty JSON of the preferences.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    fn sanitize(&mut self) {
        self.version = VERSION;
        self.library.watched.retain(|p| !p.trim().is_empty());
        self.library.watched.truncate(MAX_LIST);
        self.startup.tabs.truncate(MAX_LIST);
        self.startup.last_scope = self.startup.last_scope.take().filter(|s| !s.trim().is_empty());
    }

    /// Adds `path` to the watched folders; false when it was there already or is empty.
    pub fn watch(&mut self, path: &str) -> bool {
        let path = path.trim();
        if path.is_empty() || self.library.watched.iter().any(|p| p == path) || self.library.watched.len() >= MAX_LIST {
            return false;
        }
        self.library.watched.push(path.to_string());
        true
    }

    /// Removes `path` from the watched folders; false when it was not there.
    pub fn unwatch(&mut self, path: &str) -> bool {
        let before = self.library.watched.len();
        self.library.watched.retain(|p| p != path);
        self.library.watched.len() != before
    }
}

/// `<config dir>/smart-media/ui_prefs.json`.
pub fn default_path() -> Option<PathBuf> {
    directories::BaseDirs::new().map(|dirs| dirs.config_dir().join(crate::ui::titlebar::APP_ID).join(FILE_NAME))
}

/// What loading the file found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoadNote {
    /// No path: the preferences live in memory only.
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

/// Reads the preferences at `path`; never fails.
pub fn load_from(path: &Path) -> (UiPrefs, LoadNote) {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return (UiPrefs::default(), LoadNote::Missing),
        Err(e) => return (UiPrefs::default(), LoadNote::Corrupt(e.to_string())),
    };
    match UiPrefs::parse(&text) {
        Ok(prefs) => (prefs, LoadNote::Loaded),
        Err(PrefsError::TooNew(v)) => (UiPrefs::default(), LoadNote::TooNew(v)),
        Err(PrefsError::Invalid(why)) => (UiPrefs::default(), LoadNote::Corrupt(why)),
    }
}

/// Writes the preferences to a temporary file, then renames it over `path`.
pub fn save_to(path: &Path, prefs: &UiPrefs) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, prefs.to_json())?;
    std::fs::rename(&tmp, path)
}

/// The live preferences with their file and change tracking.
#[derive(Clone, Debug)]
pub struct UiPrefsStore {
    pub prefs: UiPrefs,
    saved: UiPrefs,
    path: Option<PathBuf>,
    pub note: LoadNote,
    pub last_error: Option<String>,
    pub saves: u32,
    writable: bool,
}

impl Default for UiPrefsStore {
    fn default() -> Self {
        Self::memory()
    }
}

impl UiPrefsStore {
    /// A store that never touches the disk.
    pub fn memory() -> Self {
        Self { prefs: UiPrefs::default(), saved: UiPrefs::default(), path: None, note: LoadNote::Memory, last_error: None, saves: 0, writable: false }
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
        let (prefs, note) = load_from(&path);
        let writable = !matches!(note, LoadNote::TooNew(_));
        Self { saved: prefs.clone(), prefs, path: Some(path), note, last_error: None, saves: 0, writable }
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    pub fn is_dirty(&self) -> bool {
        self.prefs != self.saved
    }

    /// Applies `edit` and writes the file when anything changed; true when a write happened.
    pub fn update(&mut self, edit: impl FnOnce(&mut UiPrefs)) -> bool {
        edit(&mut self.prefs);
        self.prefs.sanitize();
        self.save_if_changed()
    }

    /// Writes the preferences when they differ from what was last written.
    pub fn save_if_changed(&mut self) -> bool {
        if !self.is_dirty() {
            return false;
        }
        let Some(path) = self.path.clone().filter(|_| self.writable) else {
            self.saved = self.prefs.clone();
            return false;
        };
        match save_to(&path, &self.prefs) {
            Ok(()) => {
                self.saved = self.prefs.clone();
                self.last_error = None;
                self.saves += 1;
                true
            }
            Err(e) => {
                if self.last_error.is_none() {
                    log::warn!("[ui_prefs] cannot save {}: {e}", path.display());
                }
                self.last_error = Some(e.to_string());
                self.saved = self.prefs.clone();
                false
            }
        }
    }

    /// Back to the defaults; the file is written.
    pub fn reset(&mut self) -> bool {
        self.update(|p| *p = UiPrefs::default())
    }

    /// Raw text of the file on disk, or why there is none.
    pub fn file_text(&self) -> Result<String, String> {
        let path = self.path.as_deref().ok_or_else(|| "memory only, no file".to_string())?;
        std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))
    }
}

#[cfg(not(test))]
fn global() -> &'static std::sync::Mutex<UiPrefsStore> {
    static STORE: std::sync::OnceLock<std::sync::Mutex<UiPrefsStore>> = std::sync::OnceLock::new();
    STORE.get_or_init(|| std::sync::Mutex::new(UiPrefsStore::load()))
}

#[cfg(test)]
thread_local! {
    static TEST_STORE: std::cell::RefCell<UiPrefsStore> = std::cell::RefCell::new(UiPrefsStore::memory());
}

/// Runs `f` on the process-wide store; tests get one memory-only store per thread.
pub fn with<R>(f: impl FnOnce(&mut UiPrefsStore) -> R) -> R {
    #[cfg(test)]
    {
        TEST_STORE.with(|s| f(&mut s.borrow_mut()))
    }
    #[cfg(not(test))]
    {
        f(&mut global().lock().unwrap_or_else(|e| e.into_inner()))
    }
}

/// Copy of the live preferences.
pub fn get() -> UiPrefs {
    with(|s| s.prefs.clone())
}

/// Applies `edit` to the live preferences and saves them; true when a write happened.
pub fn update(edit: impl FnOnce(&mut UiPrefs)) -> bool {
    with(|s| s.update(edit))
}

/// Whether the application asks before `which`.
pub fn confirms(which: Confirm) -> bool {
    with(|s| s.prefs.confirmations.get(which))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("smart-media").join(FILE_NAME);
        (dir, path)
    }

    #[test]
    fn the_defaults_confirm_everything_and_restore_nothing() {
        let p = UiPrefs::default();
        assert_eq!(p.version, VERSION);
        assert!(!p.startup.restore_tabs && !p.startup.restore_last_scope);
        assert!(Confirm::ALL.iter().all(|c| p.confirmations.get(*c)));
        assert_eq!(p.checklist, Checklist::default());
        assert!(p.library.watched.is_empty());
    }

    #[test]
    fn preferences_survive_a_write_and_read() {
        let (_dir, path) = file();
        let mut prefs = UiPrefs::default();
        prefs.startup = Startup { restore_tabs: true, restore_last_scope: true, last_scope: Some("fs:/a".into()), tabs: vec!["fs:/a".into(), "tag:sun".into()] };
        prefs.confirmations.set(Confirm::MoveFiles, false);
        prefs.watch("/pics");
        prefs.checklist = Checklist { dismissed: false, location: StepState::Done, folders: StepState::Skipped, ai: StepState::Pending, scan: StepState::Pending, ai_mode: AiMode::Local };
        save_to(&path, &prefs).unwrap();
        let (loaded, note) = load_from(&path);
        assert_eq!((loaded, note), (prefs, LoadNote::Loaded));
        assert!(!path.with_extension("json.tmp").exists(), "the temporary file is renamed away");
    }

    #[test]
    fn a_missing_file_gives_the_defaults() {
        let (_dir, path) = file();
        assert_eq!(load_from(&path), (UiPrefs::default(), LoadNote::Missing));
    }

    #[test]
    fn corrupt_files_give_the_defaults_without_a_panic() {
        let (_dir, path) = file();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        for junk in ["", "{", "null", "7", "{\"startup\": 5}", "{\"version\": \"x\"}", "\u{0}\u{1}binary"] {
            std::fs::write(&path, junk).unwrap();
            let (prefs, note) = load_from(&path);
            assert_eq!(prefs, UiPrefs::default(), "{junk:?}");
            assert!(matches!(note, LoadNote::Corrupt(_)), "{junk:?}: {note:?}");
        }
    }

    #[test]
    fn a_newer_file_gives_the_defaults_and_is_never_overwritten() {
        let (_dir, path) = file();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let newer = format!("{{\"version\": {}, \"startup\": {{\"restore_tabs\": true}}, \"future\": 1}}", VERSION + 1);
        std::fs::write(&path, &newer).unwrap();
        let mut store = UiPrefsStore::at(path.clone());
        assert_eq!(store.note, LoadNote::TooNew(VERSION + 1));
        assert_eq!(store.prefs, UiPrefs::default());
        assert!(!store.update(|p| p.startup.restore_tabs = true), "nothing is written");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), newer);
    }

    #[test]
    fn missing_fields_take_defaults_and_unknown_ones_are_ignored() {
        let prefs = UiPrefs::parse("{\"startup\": {\"restore_tabs\": true}, \"extra\": [1,2]}").unwrap();
        assert!(prefs.startup.restore_tabs && !prefs.startup.restore_last_scope);
        assert_eq!(prefs.confirmations, Confirmations::default());
        assert_eq!(prefs.version, VERSION);
    }

    #[test]
    fn lists_are_cut_and_blank_entries_dropped() {
        let many: Vec<String> = (0..200).map(|i| format!("/p{i}")).collect();
        let json = serde_json::json!({"library": {"watched": many, "x": 1}, "startup": {"tabs": many, "last_scope": "  "}}).to_string();
        let prefs = UiPrefs::parse(&json).unwrap();
        assert_eq!(prefs.library.watched.len(), MAX_LIST);
        assert_eq!(prefs.startup.tabs.len(), MAX_LIST);
        assert_eq!(prefs.startup.last_scope, None);
        let blanks = UiPrefs::parse("{\"library\": {\"watched\": [\"\", \"  \", \"/a\"]}}").unwrap();
        assert_eq!(blanks.library.watched, vec!["/a".to_string()]);
    }

    #[test]
    fn watching_a_folder_twice_keeps_one_entry() {
        let mut p = UiPrefs::default();
        assert!(p.watch(" /a "));
        assert!(!p.watch("/a"));
        assert!(!p.watch("  "));
        assert_eq!(p.library.watched, vec!["/a".to_string()]);
        assert!(p.unwatch("/a"));
        assert!(!p.unwatch("/a"));
    }

    #[test]
    fn the_store_writes_only_when_something_changed() {
        let (_dir, path) = file();
        let mut store = UiPrefsStore::at(path.clone());
        assert!(!store.update(|_| {}), "no change, no write");
        assert!(!path.exists());
        assert!(store.update(|p| p.startup.restore_tabs = true));
        assert_eq!(store.saves, 1);
        assert_eq!(UiPrefsStore::at(path.clone()).prefs.startup.restore_tabs, true);
        assert!(!store.update(|p| p.startup.restore_tabs = true));
        assert_eq!(store.saves, 1);
        assert!(store.file_text().unwrap().contains("\"restore_tabs\": true"));
    }

    #[test]
    fn reset_writes_the_defaults() {
        let (_dir, path) = file();
        let mut store = UiPrefsStore::at(path.clone());
        store.update(|p| {
            p.watch("/x");
        });
        assert!(store.reset());
        assert_eq!(UiPrefsStore::at(path).prefs, UiPrefs::default());
    }

    #[test]
    fn a_memory_store_never_touches_the_disk() {
        let mut store = UiPrefsStore::memory();
        assert!(!store.update(|p| p.startup.restore_tabs = true));
        assert!(store.prefs.startup.restore_tabs);
        assert!(store.file_text().is_err());
    }

    #[test]
    fn the_global_store_in_tests_is_memory_only() {
        assert!(update(|p| p.startup.restore_last_scope = true) == false);
        assert!(get().startup.restore_last_scope);
        update(|p| *p = UiPrefs::default());
        assert!(confirms(Confirm::ImportBackup));
    }
}
