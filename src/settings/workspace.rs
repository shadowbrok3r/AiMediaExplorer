//! Workspace kept between runs in `<config>/smart-media/workspace.json`: window geometry, the dock layout and the state of every browser tab.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

use super::ui_prefs::Startup;
use crate::command::{QueryMode, Scope};
use crate::ui::file_table::rows::{FilterSpec, SortSpec};
use crate::ui::file_table::table::view_mode::ViewPrefs;
use crate::ui::titlebar::{MIN_HEIGHT, MIN_WIDTH};

/// Version written to the file; a newer file is ignored and never overwritten.
pub const VERSION: u32 = 1;
/// File name inside the app's config directory.
pub const FILE_NAME: &str = "workspace.json";
/// Quiet time after the last change before the file is written.
pub const SAVE_DELAY: Duration = Duration::from_secs(1);
/// Most browser tabs the file keeps.
pub const MAX_BROWSERS: usize = 64;
/// Most selected paths kept per tab.
pub const MAX_SELECTION: usize = 50;
/// Longest query text kept per tab.
pub const MAX_QUERY_CHARS: usize = 512;
/// Deepest split nesting the file keeps; deeper subtrees become one leaf.
pub const MAX_DEPTH: usize = 16;
/// Smallest share a split gives to either side.
pub const MIN_FRACTION: f32 = 0.05;
/// Most floating dock windows the file keeps.
pub const MAX_FLOATING: usize = 8;
/// Smallest floating window.
const MIN_FLOATING: [f32; 2] = [120.0, 80.0];
/// Largest window side kept when the screen is unknown.
pub const MAX_WINDOW_EXTENT: f32 = 16_384.0;
/// Pixels of the window's top edge that must stay reachable.
pub const VISIBLE_MARGIN: f32 = 80.0;
/// Farthest a window position can be from the origin.
const POSITION_LIMIT: f32 = 32_000.0;

fn version() -> u32 {
    VERSION
}

fn layout_file() -> String {
    crate::ui::sidebar::layout::FILE_NAME.to_string()
}

fn half() -> f32 {
    0.5
}

fn lenient<'de, D, T>(d: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: DeserializeOwned + Default,
{
    Ok(serde_json::from_value(Value::deserialize(d)?).unwrap_or_default())
}

fn lenient_option<'de, D, T>(d: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: DeserializeOwned,
{
    Ok(serde_json::from_value(Value::deserialize(d)?).ok())
}

fn lenient_list<'de, D, T>(d: D) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: DeserializeOwned,
{
    match Value::deserialize(d)? {
        Value::Array(items) => Ok(items.into_iter().filter_map(|v| serde_json::from_value(v).ok()).collect()),
        _ => Ok(Vec::new()),
    }
}

/* ------------------------------------------------ window ------------------------------------------------ */

/// Size and place of the window.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct WindowState {
    /// Inner size in points.
    pub size: [f32; 2],
    /// Outer position in points; unknown where the system does not tell it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<[f32; 2]>,
    #[serde(default)]
    pub maximized: bool,
}

impl WindowState {
    /// The window within the limits: the minimum size, no larger than `screen` when it is known, and with a strip of its top edge still reachable. `None` for a size that is not a number.
    pub fn clamped(self, screen: Option<[f32; 2]>) -> Option<WindowState> {
        if !self.size.iter().all(|v| v.is_finite()) {
            return None;
        }
        let max_w = screen.map_or(MAX_WINDOW_EXTENT, |s| s[0].max(MIN_WIDTH));
        let max_h = screen.map_or(MAX_WINDOW_EXTENT, |s| s[1].max(MIN_HEIGHT));
        let size = [self.size[0].clamp(MIN_WIDTH, max_w), self.size[1].clamp(MIN_HEIGHT, max_h)];
        let position = self.position.filter(|p| p.iter().all(|v| v.is_finite())).map(|[x, y]| match screen {
            Some([sw, sh]) => [x.clamp(VISIBLE_MARGIN - size[0] - sw, 2.0 * sw - VISIBLE_MARGIN), y.clamp(0.0, 2.0 * sh - VISIBLE_MARGIN)],
            None => [x.clamp(-POSITION_LIMIT, POSITION_LIMIT), y.clamp(-POSITION_LIMIT, POSITION_LIMIT)],
        });
        Some(WindowState { size, position, maximized: self.maximized })
    }
}

/* ------------------------------------------------- dock ------------------------------------------------- */

/// A tab of the dock as it is saved; browsers are named by the id they have in this file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SavedTab {
    Browser { id: u64 },
    Refine,
    ImageEdit,
    Settings,
    Devtools,
    Logs,
}

/// How a split lays out its two sides.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SplitDir {
    /// Side by side, the first on the left.
    Horizontal,
    /// Stacked, the first on top.
    Vertical,
}

/// One node of the dock's main surface.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "node", rename_all = "snake_case")]
pub enum SavedNode {
    Leaf {
        #[serde(deserialize_with = "lenient_list")]
        tabs: Vec<SavedTab>,
        /// Index of the tab in front.
        #[serde(default)]
        active: usize,
    },
    Split {
        dir: SplitDir,
        /// Share of the first side.
        #[serde(default = "half")]
        fraction: f32,
        first: Box<SavedNode>,
        second: Box<SavedNode>,
    },
}

impl SavedNode {
    /// One leaf with `tabs`.
    pub fn leaf(tabs: Vec<SavedTab>) -> Self {
        Self::Leaf { tabs, active: 0 }
    }

    /// Every tab in layout order, the first side before the second.
    pub fn tabs(&self) -> Vec<SavedTab> {
        match self {
            Self::Leaf { tabs, .. } => tabs.clone(),
            Self::Split { first, second, .. } => first.tabs().into_iter().chain(second.tabs()).collect(),
        }
    }

    /// The tab in front of the first leaf that has one.
    pub fn first_active(&self) -> Option<SavedTab> {
        match self {
            Self::Leaf { tabs, active } => tabs.get(*active).copied(),
            Self::Split { first, second, .. } => first.first_active().or_else(|| second.first_active()),
        }
    }

    /// The tab of the first leaf, which a split's new side is seeded with.
    pub fn seed(&self) -> Vec<SavedTab> {
        match self {
            Self::Leaf { tabs, .. } => tabs.clone(),
            Self::Split { first, .. } => first.seed(),
        }
    }

    /// The node with only the tabs `keep` accepts: empty leaves are dropped, a split with one side left becomes that side, fractions are brought into range and deeper than [`MAX_DEPTH`] becomes one leaf.
    fn normalized(self, keep: &mut dyn FnMut(&SavedTab) -> bool, depth: usize) -> Option<SavedNode> {
        match self {
            Self::Leaf { tabs, active } => {
                let front = tabs.get(active).copied();
                let tabs: Vec<SavedTab> = tabs.into_iter().filter(|t| keep(t)).collect();
                if tabs.is_empty() {
                    return None;
                }
                let active = front.and_then(|f| tabs.iter().position(|t| *t == f)).unwrap_or(0);
                Some(Self::Leaf { tabs, active })
            }
            Self::Split { dir, fraction, first, second } => {
                if depth >= MAX_DEPTH {
                    let all = Self::Split { dir, fraction, first, second }.tabs();
                    return Self::leaf(all).normalized(keep, depth);
                }
                let fraction = if fraction.is_finite() { fraction.clamp(MIN_FRACTION, 1.0 - MIN_FRACTION) } else { 0.5 };
                match (first.normalized(keep, depth + 1), second.normalized(keep, depth + 1)) {
                    (Some(a), Some(b)) => Some(Self::Split { dir, fraction, first: Box::new(a), second: Box::new(b) }),
                    (Some(only), None) | (None, Some(only)) => Some(only),
                    (None, None) => None,
                }
            }
        }
    }
}

/// A floating window of the dock: its tabs and where it was.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FloatingWindow {
    pub dock: SavedNode,
    /// Position inside the application window, in points.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<[f32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<[f32; 2]>,
}

impl FloatingWindow {
    /// The window with a place and size that are numbers within range.
    fn sanitized(mut self) -> Self {
        self.position = self.position.filter(|p| p.iter().all(|v| v.is_finite())).map(|[x, y]| [x.clamp(-POSITION_LIMIT, POSITION_LIMIT), y.clamp(-POSITION_LIMIT, POSITION_LIMIT)]);
        self.size = self.size.filter(|s| s.iter().all(|v| v.is_finite())).map(|[w, h]| [w.clamp(MIN_FLOATING[0], MAX_WINDOW_EXTENT), h.clamp(MIN_FLOATING[1], MAX_WINDOW_EXTENT)]);
        self
    }
}

/* ------------------------------------------------ browsers ----------------------------------------------- */

/// The query bar of a tab.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryState {
    #[serde(default)]
    pub mode: QueryMode,
    #[serde(default)]
    pub text: String,
}

/// What a browser tab shows and how it shows it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BrowserState {
    /// Name of the tab in this file, referenced by the dock.
    pub id: u64,
    /// Scope string of the tab.
    pub scope: String,
    #[serde(default, deserialize_with = "lenient")]
    pub view: ViewPrefs,
    #[serde(default, deserialize_with = "lenient", skip_serializing_if = "SortSpec::is_empty")]
    pub sort: SortSpec,
    #[serde(default, deserialize_with = "lenient")]
    pub query: QueryState,
    /// The tab's filter when it differs from the defaults of the settings.
    #[serde(default, deserialize_with = "lenient_option", skip_serializing_if = "Option::is_none")]
    pub filter: Option<FilterSpec>,
    /// Selected paths; only a small selection is kept.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub selection: Vec<String>,
}

impl BrowserState {
    /// A tab showing `scope` with everything else at its defaults.
    pub fn new(id: u64, scope: &Scope) -> Self {
        Self { id, scope: scope.format(), view: ViewPrefs::default(), sort: SortSpec::default(), query: QueryState::default(), filter: None, selection: Vec::new() }
    }

    /// The scope of the tab, when its string reads.
    pub fn parsed_scope(&self) -> Option<Scope> {
        Scope::parse(&self.scope).ok()
    }

    /// Brings every field into range; false when the tab cannot be used at all.
    fn sanitize(&mut self) -> bool {
        let Some(scope) = self.parsed_scope() else { return false };
        self.scope = scope.format();
        self.view = self.view.sanitized();
        self.sort = SortSpec::from_keys(self.sort.keys().to_vec());
        self.query.text = self.query.text.chars().take(MAX_QUERY_CHARS).collect();
        self.filter = self.filter.take().map(FilterSpec::normalized);
        let mut seen = HashSet::new();
        self.selection.retain(|p| !p.is_empty() && seen.insert(p.clone()));
        self.selection.truncate(MAX_SELECTION);
        true
    }
}

/* ----------------------------------------------- workspace ---------------------------------------------- */

/// Why a workspace text was not used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkspaceError {
    Invalid(String),
    TooNew(u32),
}

impl std::fmt::Display for WorkspaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(reason) => write!(f, "unreadable workspace: {reason}"),
            Self::TooNew(v) => write!(f, "workspace version {v} is newer than {VERSION}"),
        }
    }
}

/// The saved workspace.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Workspace {
    #[serde(default = "version")]
    pub version: u32,
    /// The sidebar and right rail state is kept in this file next to it, not here.
    #[serde(default = "layout_file")]
    pub layout_file: String,
    #[serde(default, deserialize_with = "lenient_option", skip_serializing_if = "Option::is_none")]
    pub window: Option<WindowState>,
    #[serde(default, deserialize_with = "lenient_option", skip_serializing_if = "Option::is_none")]
    pub dock: Option<SavedNode>,
    /// Windows the dock's tabs were dragged out into.
    #[serde(default, deserialize_with = "lenient_list", skip_serializing_if = "Vec::is_empty")]
    pub floating: Vec<FloatingWindow>,
    /// The tab that had the focus.
    #[serde(default, deserialize_with = "lenient_option", skip_serializing_if = "Option::is_none")]
    pub focused: Option<SavedTab>,
    #[serde(default, deserialize_with = "lenient_list")]
    pub browsers: Vec<BrowserState>,
}

impl Default for Workspace {
    fn default() -> Self {
        Self { version: VERSION, layout_file: layout_file(), window: None, dock: None, floating: Vec::new(), focused: None, browsers: Vec::new() }
    }
}

impl Workspace {
    /// Reads a workspace; whatever is unusable in it is dropped and the rest is brought into range.
    pub fn parse(text: &str) -> Result<Self, WorkspaceError> {
        let value: Value = serde_json::from_str(text).map_err(|e| WorkspaceError::Invalid(e.to_string()))?;
        if !value.is_object() {
            return Err(WorkspaceError::Invalid("not a JSON object".to_string()));
        }
        match value.get("version") {
            None | Some(Value::Null) => {}
            Some(v) => match v.as_u64().and_then(|v| u32::try_from(v).ok()) {
                Some(v) if v > VERSION => return Err(WorkspaceError::TooNew(v)),
                Some(_) => {}
                None => return Err(WorkspaceError::Invalid("the version is not a number".to_string())),
            },
        }
        let mut workspace: Self = serde_json::from_value(value).map_err(|e| WorkspaceError::Invalid(e.to_string()))?;
        workspace.sanitize();
        Ok(workspace)
    }

    /// Pretty JSON of the workspace.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Drops what cannot be restored and keeps the dock and the browsers consistent with each other.
    pub fn sanitize(&mut self) {
        self.version = VERSION;
        self.layout_file = layout_file();
        self.window = self.window.take().and_then(|w| w.clamped(None));
        let mut ids = HashSet::new();
        self.browsers.retain_mut(|b| b.sanitize() && ids.insert(b.id));
        self.browsers.truncate(MAX_BROWSERS);
        let known: HashSet<u64> = self.browsers.iter().map(|b| b.id).collect();
        let mut seen = HashSet::new();
        let mut keep = |tab: &SavedTab| match tab {
            SavedTab::Browser { id } => known.contains(id) && seen.insert(*tab),
            _ => seen.insert(*tab),
        };
        self.dock = self.dock.take().and_then(|d| d.normalized(&mut keep, 0));
        let windows = std::mem::take(&mut self.floating);
        self.floating = windows.into_iter().take(MAX_FLOATING).filter_map(|w| Some(FloatingWindow { dock: w.dock.normalized(&mut keep, 0)?, ..w }.sanitized())).collect();
        let shown = self.all_tabs();
        let used: HashSet<u64> = shown.iter().filter_map(|t| if let SavedTab::Browser { id } = t { Some(*id) } else { None }).collect();
        self.browsers.retain(|b| used.contains(&b.id));
        self.focused = self.focused.filter(|f| shown.contains(f));
    }

    /// Every tab: the dock's in layout order, then those of the floating windows.
    pub fn all_tabs(&self) -> Vec<SavedTab> {
        self.dock.iter().flat_map(SavedNode::tabs).chain(self.floating.iter().flat_map(|w| w.dock.tabs())).collect()
    }

    /// True when there is a tab to bring back.
    pub fn has_tabs(&self) -> bool {
        self.dock.is_some() || !self.floating.is_empty()
    }

    /// The saved state of browser `id`.
    pub fn browser(&self, id: u64) -> Option<&BrowserState> {
        self.browsers.iter().find(|b| b.id == id)
    }

    /// The tab that had the focus when it is a browser, else the browser in front of the first leaf, else the first browser.
    pub fn focused_browser(&self) -> Option<&BrowserState> {
        let by_id = |tab: SavedTab| if let SavedTab::Browser { id } = tab { self.browser(id) } else { None };
        self.focused.and_then(by_id).or_else(|| self.dock.as_ref().and_then(SavedNode::first_active).and_then(by_id)).or_else(|| self.browsers.first())
    }

    /// A workspace of the tabs the preferences of older versions kept: the focused scope first, one leaf with every tab.
    pub fn from_legacy(startup: &Startup, all_tabs: bool) -> Option<Self> {
        let scopes: Vec<&String> = if all_tabs { startup.tabs.iter().collect() } else { startup.last_scope.iter().collect() };
        let browsers: Vec<BrowserState> = scopes.into_iter().filter_map(|text| Scope::parse(text).ok()).enumerate().map(|(i, scope)| BrowserState::new(i as u64, &scope)).collect();
        if browsers.is_empty() {
            return None;
        }
        let tabs: Vec<SavedTab> = browsers.iter().map(|b| SavedTab::Browser { id: b.id }).chain(std::iter::once(SavedTab::Logs)).collect();
        let mut workspace = Self { dock: Some(SavedNode::leaf(tabs)), focused: Some(SavedTab::Browser { id: 0 }), browsers, ..Self::default() };
        workspace.sanitize();
        Some(workspace)
    }
}

/* -------------------------------------------------- file -------------------------------------------------- */

/// `<config dir>/smart-media/workspace.json`.
pub fn default_path() -> Option<PathBuf> {
    directories::BaseDirs::new().map(|dirs| dirs.config_dir().join(crate::ui::titlebar::APP_ID).join(FILE_NAME))
}

/// What loading the file found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoadNote {
    /// No path: the workspace lives in memory only.
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
            Self::Missing => "no saved file".to_string(),
            Self::Loaded => "loaded".to_string(),
            Self::Corrupt(why) => format!("unreadable, ignored ({why})"),
            Self::TooNew(v) => format!("file version {v} is newer, ignored and never overwritten"),
        }
    }
}

/// Reads the workspace at `path`; a missing, unreadable or newer file gives `None` and says why.
pub fn load_from(path: &Path) -> (Option<Workspace>, LoadNote) {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return (None, LoadNote::Missing),
        Err(e) => return (None, LoadNote::Corrupt(e.to_string())),
    };
    match Workspace::parse(&text) {
        Ok(workspace) => (Some(workspace), LoadNote::Loaded),
        Err(WorkspaceError::TooNew(v)) => (None, LoadNote::TooNew(v)),
        Err(WorkspaceError::Invalid(why)) => (None, LoadNote::Corrupt(why)),
    }
}

/// Writes the workspace to a temporary file, then renames it over `path`.
pub fn save_to(path: &Path, workspace: &Workspace) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, workspace.to_json())?;
    std::fs::rename(&tmp, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })
}

/// The file with its write schedule: a change is written once it has been quiet for [`SAVE_DELAY`], and on demand.
#[derive(Debug)]
pub struct Saver {
    path: Option<PathBuf>,
    writable: bool,
    pub note: LoadNote,
    /// What is on disk as far as this run knows.
    written: Option<Workspace>,
    pending: Option<(Workspace, Instant)>,
    pub saves: u32,
    pub last_error: Option<String>,
    pub last_write: Option<SystemTime>,
    delay: Duration,
}

impl Saver {
    /// A saver that never touches the disk.
    pub fn memory() -> Self {
        Self { path: None, writable: false, note: LoadNote::Memory, written: None, pending: None, saves: 0, last_error: None, last_write: None, delay: SAVE_DELAY }
    }

    /// The saver of `path` after `workspace` and `note` were read from it; an unreadable file is kept next to it with the extension `.bad`.
    pub fn from_load(path: Option<PathBuf>, workspace: Option<Workspace>, note: LoadNote) -> Self {
        if let (Some(path), LoadNote::Corrupt(_)) = (&path, &note) {
            let _ = std::fs::copy(path, path.with_extension("json.bad"));
        }
        let last_write = path.as_deref().and_then(|p| std::fs::metadata(p).ok()).and_then(|m| m.modified().ok());
        Self { writable: path.is_some() && !matches!(note, LoadNote::TooNew(_)), path, note, written: workspace, pending: None, saves: 0, last_error: None, last_write, delay: SAVE_DELAY }
    }

    /// The saver of the file at `path`, with what it held.
    pub fn at(path: PathBuf) -> (Self, Option<Workspace>) {
        let (workspace, note) = load_from(&path);
        (Self::from_load(Some(path), workspace.clone(), note), workspace)
    }

    /// Replaces the quiet time, for tests.
    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// False when writing is not allowed: no path, or the file is from a newer version.
    pub fn is_writable(&self) -> bool {
        self.writable
    }

    /// What this run believes is on disk.
    pub fn written(&self) -> Option<&Workspace> {
        self.written.as_ref()
    }

    /// True while a change waits for its quiet time.
    pub fn is_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Takes note of the current state; a state equal to the one on disk cancels the wait, a new one starts it over.
    pub fn observe(&mut self, now: Instant, state: &Workspace) {
        if self.written.as_ref() == Some(state) {
            self.pending = None;
        } else if self.pending.as_ref().is_none_or(|(p, _)| p != state) {
            self.pending = Some((state.clone(), now));
        }
    }

    /// Writes the change that has been quiet long enough; true when a write happened.
    pub fn poll(&mut self, now: Instant) -> bool {
        match self.pending.take_if(|(_, since)| now.saturating_duration_since(*since) >= self.delay) {
            Some((state, _)) => self.write(&state),
            None => false,
        }
    }

    /// How long until the pending change is due.
    pub fn wake_in(&self, now: Instant) -> Option<Duration> {
        self.pending.as_ref().map(|(_, since)| self.delay.saturating_sub(now.saturating_duration_since(*since)))
    }

    /// Writes `state` at once, even when it equals what is on disk.
    pub fn save_now(&mut self, state: &Workspace) -> Result<(), String> {
        if !self.writable {
            return Err(match (&self.path, &self.note) {
                (None, _) => "memory only, no file".to_string(),
                (_, note) => note.text(),
            });
        }
        self.pending = None;
        if self.write(state) { Ok(()) } else { Err(self.last_error.clone().unwrap_or_else(|| "not written".to_string())) }
    }

    /// Writes `state` when it differs from what is on disk; for the exit of the application.
    pub fn flush(&mut self, state: &Workspace) -> bool {
        self.pending = None;
        self.written.as_ref() != Some(state) && self.write(state)
    }

    /// Deletes the file; a file of a newer version is left alone.
    pub fn forget(&mut self) -> Result<(), String> {
        let Some(path) = self.path.clone() else { return Err("memory only, no file".to_string()) };
        if !self.writable {
            return Err(self.note.text());
        }
        self.pending = None;
        match std::fs::remove_file(&path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.to_string()),
        }
        self.written = None;
        self.last_write = None;
        self.note = LoadNote::Missing;
        Ok(())
    }

    fn write(&mut self, state: &Workspace) -> bool {
        let Some(path) = self.path.clone().filter(|_| self.writable) else {
            self.written = Some(state.clone());
            return false;
        };
        self.written = Some(state.clone());
        match save_to(&path, state) {
            Ok(()) => {
                self.last_error = None;
                self.last_write = Some(SystemTime::now());
                self.note = LoadNote::Loaded;
                self.saves += 1;
                true
            }
            Err(e) => {
                if self.last_error.is_none() {
                    log::warn!("[workspace] cannot save {}: {e}", path.display());
                }
                self.last_error = Some(e.to_string());
                false
            }
        }
    }

    /// Raw text of the file on disk, or why there is none.
    pub fn file_text(&self) -> Result<String, String> {
        let path = self.path.as_deref().ok_or_else(|| "memory only, no file".to_string())?;
        std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::file_table::table::view_mode::{TileFit, ViewMode};

    fn browser(id: u64, scope: &str) -> BrowserState {
        BrowserState::new(id, &Scope::parse(scope).unwrap())
    }

    fn sample() -> Workspace {
        let mut a = browser(0, "fs:/pics");
        a.view = ViewPrefs { mode: ViewMode::Details, tile: 200.0, fit: TileFit::Fit, captions: false, badges: true };
        a.sort = SortSpec::single(2, true);
        a.query = QueryState { mode: QueryMode::Text, text: "beach".into() };
        a.filter = Some(FilterSpec::from_json(r#"{"tags":["sun"],"size":{"min":null}}"#).unwrap());
        a.selection = vec!["/pics/a.jpg".into(), "/pics/b.jpg".into()];
        let b = browser(1, "tag:sunset");
        let c = browser(2, "lib:all");
        let dock = SavedNode::Split {
            dir: SplitDir::Horizontal,
            fraction: 0.4,
            first: Box::new(SavedNode::Leaf { tabs: vec![SavedTab::Browser { id: 0 }, SavedTab::Browser { id: 1 }, SavedTab::Settings], active: 1 }),
            second: Box::new(SavedNode::Split {
                dir: SplitDir::Vertical,
                fraction: 0.7,
                first: Box::new(SavedNode::leaf(vec![SavedTab::Browser { id: 2 }])),
                second: Box::new(SavedNode::leaf(vec![SavedTab::Logs, SavedTab::Refine])),
            }),
        };
        Workspace { window: Some(WindowState { size: [1200.0, 800.0], position: Some([40.0, 30.0]), maximized: false }), dock: Some(dock), focused: Some(SavedTab::Browser { id: 1 }), browsers: vec![a, b, c], ..Workspace::default() }
    }

    fn dir() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    /* ----- model ----- */

    #[test]
    fn a_workspace_survives_the_json_round_trip() {
        let ws = sample();
        let back = Workspace::parse(&ws.to_json()).unwrap();
        assert_eq!(back, ws);
        assert!(ws.to_json().contains("ui_layout.json"), "the layout file is referenced, not copied");
    }

    #[test]
    fn missing_fields_take_defaults_and_unknown_ones_are_ignored() {
        let ws = Workspace::parse(r#"{"future": [1, 2], "browsers": [{"id": 0, "scope": "lib:all", "extra": true}], "dock": {"node": "leaf", "tabs": [{"kind": "browser", "id": 0}]}}"#).unwrap();
        assert_eq!(ws.version, VERSION);
        assert_eq!(ws.browsers.len(), 1);
        assert_eq!(ws.browsers[0].view, ViewPrefs::default());
        assert_eq!(ws.dock, Some(SavedNode::leaf(vec![SavedTab::Browser { id: 0 }])));
        assert!(Workspace::parse("{}").unwrap().dock.is_none());
    }

    #[test]
    fn garbage_is_an_error_and_never_a_panic() {
        let deep_dock = format!("{{\"dock\": {}}}", "{\"node\":\"split\",\"dir\":\"horizontal\",\"first\":".repeat(300));
        let texts = ["", "null", "[]", "42", "\"x\"", "{", "{\"browsers\": ", "\u{0}\u{1}\u{2}", "{\"version\": \"one\"}", "{\"version\": -3}", &"[".repeat(5000), &deep_dock];
        for text in texts {
            assert!(matches!(Workspace::parse(text), Err(WorkspaceError::Invalid(_))), "{}", &text[..text.len().min(40)]);
        }
    }

    #[test]
    fn a_newer_version_is_refused_with_its_number() {
        assert_eq!(Workspace::parse(&format!("{{\"version\": {}}}", VERSION + 1)), Err(WorkspaceError::TooNew(VERSION + 1)));
        assert!(Workspace::parse(&format!("{{\"version\": {VERSION}}}")).is_ok());
    }

    #[test]
    fn a_broken_part_is_dropped_and_the_rest_is_kept() {
        let text = r#"{
            "window": {"size": "big"},
            "browsers": [
                {"id": 0, "scope": "lib:all", "filter": {"nonsense": 1}, "view": {"mode": 7}, "sort": "up"},
                {"id": 1, "scope": "not a scope"},
                {"scope": "lib:all"},
                "junk",
                {"id": 3, "scope": "tag:sun", "query": {"mode": "text", "text": "x"}}
            ],
            "dock": {"node": "leaf", "tabs": [{"kind": "browser", "id": 0}, {"kind": "browser", "id": 1}, {"kind": "browser", "id": 3}, {"kind": "mystery"}]}
        }"#;
        let ws = Workspace::parse(text).unwrap();
        assert_eq!(ws.window, None);
        assert_eq!(ws.browsers.iter().map(|b| b.id).collect::<Vec<_>>(), vec![0, 3]);
        assert_eq!(ws.browsers[0].filter, None);
        assert_eq!(ws.browsers[0].view, ViewPrefs::default());
        assert!(ws.browsers[0].sort.is_empty());
        assert_eq!(ws.browsers[1].query, QueryState { mode: QueryMode::Text, text: "x".into() });
        assert_eq!(ws.dock.unwrap().tabs(), vec![SavedTab::Browser { id: 0 }, SavedTab::Browser { id: 3 }], "tabs without a browser are gone");
    }

    #[test]
    fn the_dock_is_made_consistent_with_the_browsers() {
        let mut ws = Workspace {
            browsers: vec![browser(0, "lib:all"), browser(1, "tag:a"), browser(1, "tag:dup"), browser(5, "tag:unused")],
            dock: Some(SavedNode::Split {
                dir: SplitDir::Vertical,
                fraction: f32::NAN,
                first: Box::new(SavedNode::Leaf { tabs: vec![SavedTab::Browser { id: 0 }, SavedTab::Logs, SavedTab::Browser { id: 0 }], active: 9 }),
                second: Box::new(SavedNode::Split {
                    dir: SplitDir::Horizontal,
                    fraction: 2.0,
                    first: Box::new(SavedNode::leaf(vec![SavedTab::Browser { id: 9 }])),
                    second: Box::new(SavedNode::Leaf { tabs: vec![SavedTab::Logs, SavedTab::Browser { id: 1 }], active: 1 }),
                }),
            }),
            focused: Some(SavedTab::Browser { id: 5 }),
            ..Workspace::default()
        };
        ws.sanitize();
        assert_eq!(ws.browsers.iter().map(|b| (b.id, b.scope.as_str())).collect::<Vec<_>>(), vec![(0, "lib:all"), (1, "tag:a")], "a duplicate id keeps the first, an unused tab goes");
        let dock = ws.dock.clone().unwrap();
        assert_eq!(dock.tabs(), vec![SavedTab::Browser { id: 0 }, SavedTab::Logs, SavedTab::Browser { id: 1 }], "each tab appears once");
        match dock {
            SavedNode::Split { fraction, first, second, .. } => {
                assert_eq!(fraction, 0.5, "a fraction that is not a number becomes a half");
                assert!(matches!(*first, SavedNode::Leaf { active: 0, .. }), "an index out of range falls back to the first tab");
                assert!(matches!(*second, SavedNode::Leaf { .. }), "the split with an empty side became the other side");
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(ws.focused, None, "the focus needs a tab that is shown");
    }

    #[test]
    fn the_active_tab_stays_the_same_tab_when_others_are_dropped() {
        let mut ws = Workspace {
            browsers: vec![browser(1, "lib:all")],
            dock: Some(SavedNode::Leaf { tabs: vec![SavedTab::Browser { id: 7 }, SavedTab::Logs, SavedTab::Browser { id: 1 }], active: 2 }),
            ..Workspace::default()
        };
        ws.sanitize();
        assert_eq!(ws.dock.unwrap(), SavedNode::Leaf { tabs: vec![SavedTab::Logs, SavedTab::Browser { id: 1 }], active: 1 });
    }

    #[test]
    fn a_split_deeper_than_the_limit_becomes_one_leaf() {
        let mut node = SavedNode::leaf(vec![SavedTab::Browser { id: 0 }]);
        for i in 1..40u64 {
            node = SavedNode::Split { dir: SplitDir::Horizontal, fraction: 0.5, first: Box::new(SavedNode::leaf(vec![SavedTab::Browser { id: i }])), second: Box::new(node) };
        }
        let mut ws = Workspace { browsers: (0..40).map(|i| browser(i, "lib:all")).collect(), dock: Some(node), ..Workspace::default() };
        ws.sanitize();
        let dock = ws.dock.unwrap();
        fn depth(n: &SavedNode) -> usize {
            match n {
                SavedNode::Leaf { .. } => 0,
                SavedNode::Split { first, second, .. } => 1 + depth(first).max(depth(second)),
            }
        }
        assert!(depth(&dock) <= MAX_DEPTH);
        assert_eq!(dock.tabs().len(), 40, "no tab is lost by the flattening");
    }

    #[test]
    fn floating_windows_are_kept_with_their_place_and_size() {
        let mut ws = sample();
        ws.browsers.push(browser(3, "tag:float"));
        ws.floating = vec![FloatingWindow { dock: leaf_of(&[SavedTab::Browser { id: 3 }, SavedTab::Devtools]), position: Some([30.0, 40.0]), size: Some([500.0, 300.0]) }];
        ws.sanitize();
        assert_eq!(ws.floating.len(), 1);
        let back = Workspace::parse(&ws.to_json()).unwrap();
        assert_eq!(back, ws);
        assert_eq!(back.all_tabs().len(), ws.dock.as_ref().unwrap().tabs().len() + 2);
        assert!(back.has_tabs());
    }

    fn leaf_of(tabs: &[SavedTab]) -> SavedNode {
        SavedNode::leaf(tabs.to_vec())
    }

    #[test]
    fn a_floating_window_is_dropped_when_its_tabs_are_gone_or_already_in_the_dock() {
        let mut ws = sample();
        ws.floating = vec![
            FloatingWindow { dock: leaf_of(&[SavedTab::Browser { id: 0 }, SavedTab::Logs]), position: None, size: None },
            FloatingWindow { dock: leaf_of(&[SavedTab::Browser { id: 99 }]), position: None, size: None },
            FloatingWindow { dock: leaf_of(&[SavedTab::ImageEdit]), position: Some([f32::NAN, 1e9]), size: Some([1.0, f32::INFINITY]) },
        ];
        ws.sanitize();
        assert_eq!(ws.floating.len(), 1, "only the window with a tab that was not shown yet is left");
        assert_eq!(ws.floating[0].dock, leaf_of(&[SavedTab::ImageEdit]));
        assert_eq!(ws.floating[0].position, None);
        assert_eq!(ws.floating[0].size, None);
        let placed = FloatingWindow { dock: leaf_of(&[SavedTab::Settings]), position: Some([-5.0, 1e9]), size: Some([1.0, 99_999.0]) }.sanitized();
        assert_eq!(placed.position, Some([-5.0, POSITION_LIMIT]));
        assert_eq!(placed.size, Some([MIN_FLOATING[0], MAX_WINDOW_EXTENT]));
    }

    #[test]
    fn a_workspace_of_floating_windows_alone_has_tabs_and_keeps_its_browsers() {
        let mut ws = Workspace { browsers: vec![browser(2, "lib:all")], floating: vec![FloatingWindow { dock: leaf_of(&[SavedTab::Browser { id: 2 }]), position: None, size: None }], ..Workspace::default() };
        ws.sanitize();
        assert!(ws.has_tabs() && ws.dock.is_none());
        assert_eq!(ws.browsers.len(), 1);
        assert!(!Workspace::default().has_tabs());
    }

    #[test]
    fn no_more_floating_windows_than_the_limit_are_kept() {
        let tabs = [SavedTab::Refine, SavedTab::ImageEdit, SavedTab::Settings, SavedTab::Devtools, SavedTab::Logs];
        let mut ws = Workspace::default();
        ws.browsers = (0..10).map(|i| browser(i, "lib:all")).collect();
        ws.floating = (0..10).map(|i| FloatingWindow { dock: leaf_of(&[SavedTab::Browser { id: i }]), position: None, size: None }).collect();
        ws.floating.push(FloatingWindow { dock: leaf_of(&tabs), position: None, size: None });
        ws.sanitize();
        assert_eq!(ws.floating.len(), MAX_FLOATING);
        assert_eq!(ws.browsers.len(), MAX_FLOATING, "a browser no window shows is dropped");
    }

    #[test]
    fn a_workspace_keeps_at_most_the_limit_of_browsers_and_their_tabs() {
        let browsers: Vec<BrowserState> = (0..MAX_BROWSERS as u64 + 20).map(|i| BrowserState::new(i, &Scope::Library)).collect();
        let tabs: Vec<SavedTab> = browsers.iter().map(|b| SavedTab::Browser { id: b.id }).collect();
        let mut ws = Workspace { browsers, dock: Some(SavedNode::leaf(tabs)), ..Workspace::default() };
        ws.sanitize();
        assert_eq!(ws.browsers.len(), MAX_BROWSERS);
        assert_eq!(ws.dock.unwrap().tabs().len(), MAX_BROWSERS, "no tab is left without its browser");
    }

    #[test]
    fn a_tab_keeps_a_small_selection_and_a_short_query() {
        let mut b = browser(0, "lib:all");
        b.selection = (0..200).map(|i| format!("/p/{i}.jpg")).chain(["".to_string(), "/p/1.jpg".to_string()]).collect();
        b.query.text = "x".repeat(MAX_QUERY_CHARS * 2);
        b.view.tile = 99_999.0;
        let mut ws = Workspace { browsers: vec![b], dock: Some(SavedNode::leaf(vec![SavedTab::Browser { id: 0 }])), ..Workspace::default() };
        ws.sanitize();
        let b = &ws.browsers[0];
        assert_eq!(b.selection.len(), MAX_SELECTION);
        assert!(b.selection.iter().all(|p| !p.is_empty()) && b.selection.iter().filter(|p| *p == "/p/1.jpg").count() == 1);
        assert_eq!(b.query.text.chars().count(), MAX_QUERY_CHARS);
        assert_eq!(b.view.tile, crate::ui::file_table::table::view_mode::TILE_MAX);
    }

    #[test]
    fn the_focused_browser_falls_back_to_the_front_tab_then_the_first_browser() {
        let mut ws = sample();
        assert_eq!(ws.focused_browser().unwrap().id, 1);
        ws.focused = Some(SavedTab::Settings);
        assert_eq!(ws.focused_browser().unwrap().id, 1, "the front tab of the first leaf");
        ws.focused = None;
        assert_eq!(ws.focused_browser().unwrap().id, 1);
        ws.dock = Some(SavedNode::leaf(vec![SavedTab::Logs]));
        assert_eq!(ws.focused_browser().unwrap().id, 0, "the first browser");
        assert!(Workspace::default().focused_browser().is_none());
    }

    #[test]
    fn the_tabs_of_an_older_version_make_a_workspace() {
        let startup = Startup { restore_tabs: true, restore_last_scope: true, last_scope: Some("tag:b".into()), tabs: vec!["tag:b".into(), "garbage".into(), "lib:all".into()] };
        let all = Workspace::from_legacy(&startup, true).unwrap();
        assert_eq!(all.browsers.iter().map(|b| b.scope.as_str()).collect::<Vec<_>>(), vec!["tag:b", "lib:all"]);
        assert_eq!(all.dock.as_ref().unwrap().tabs(), vec![SavedTab::Browser { id: 0 }, SavedTab::Browser { id: 1 }, SavedTab::Logs]);
        assert_eq!(all.focused_browser().unwrap().scope, "tag:b");
        let one = Workspace::from_legacy(&startup, false).unwrap();
        assert_eq!(one.browsers.len(), 1);
        assert!(Workspace::from_legacy(&Startup::default(), true).is_none());
    }

    /* ----- window ----- */

    fn window(w: f32, h: f32, pos: Option<[f32; 2]>) -> WindowState {
        WindowState { size: [w, h], position: pos, maximized: false }
    }

    #[test]
    fn a_window_is_never_smaller_than_the_minimum() {
        let w = window(100.0, 50.0, None).clamped(None).unwrap();
        assert_eq!(w.size, [MIN_WIDTH, MIN_HEIGHT]);
        assert_eq!(window(MIN_WIDTH, MIN_HEIGHT, None).clamped(None).unwrap().size, [700.0, 400.0]);
    }

    #[test]
    fn a_window_is_never_larger_than_the_screen() {
        let w = window(5000.0, 3000.0, None).clamped(Some([1920.0, 1080.0])).unwrap();
        assert_eq!(w.size, [1920.0, 1080.0]);
        assert_eq!(window(50_000.0, 30_000.0, None).clamped(None).unwrap().size, [MAX_WINDOW_EXTENT, MAX_WINDOW_EXTENT]);
        let tiny_screen = window(900.0, 700.0, None).clamped(Some([500.0, 300.0])).unwrap();
        assert_eq!(tiny_screen.size, [MIN_WIDTH, MIN_HEIGHT], "the minimum wins over a smaller screen");
    }

    #[test]
    fn a_window_keeps_a_reachable_strip_on_screen() {
        let screen = Some([1920.0, 1080.0]);
        let far = window(1000.0, 700.0, Some([90_000.0, -50.0])).clamped(screen).unwrap();
        assert_eq!(far.position, Some([2.0 * 1920.0 - VISIBLE_MARGIN, 0.0]));
        let left = window(1000.0, 700.0, Some([-9_000.0, 5_000.0])).clamped(screen).unwrap();
        assert_eq!(left.position, Some([VISIBLE_MARGIN - 1000.0 - 1920.0, 2.0 * 1080.0 - VISIBLE_MARGIN]));
        let second_screen = window(1000.0, 700.0, Some([2000.0, 100.0])).clamped(screen).unwrap();
        assert_eq!(second_screen.position, Some([2000.0, 100.0]), "a window on a neighbouring screen stays there");
        assert_eq!(window(1000.0, 700.0, Some([10.0, 20.0])).clamped(screen).unwrap().position, Some([10.0, 20.0]));
    }

    #[test]
    fn a_position_that_is_not_a_number_is_dropped_and_a_size_that_is_not_gives_no_window() {
        let w = window(1000.0, 700.0, Some([f32::NAN, 5.0])).clamped(Some([1920.0, 1080.0])).unwrap();
        assert_eq!(w.position, None);
        assert!(window(f32::INFINITY, 700.0, None).clamped(None).is_none());
        let kept = WindowState { size: [900.0, 600.0], position: None, maximized: true }.clamped(None).unwrap();
        assert!(kept.maximized);
    }

    #[test]
    fn a_saved_window_is_clamped_when_the_workspace_is_read() {
        let ws = Workspace::parse(r#"{"window": {"size": [10, 10], "position": [1e9, 5]}}"#).unwrap();
        let w = ws.window.unwrap();
        assert_eq!(w.size, [MIN_WIDTH, MIN_HEIGHT]);
        assert_eq!(w.position, Some([POSITION_LIMIT, 5.0]));
    }

    /* ----- file ----- */

    #[test]
    fn a_missing_file_gives_nothing_and_a_note() {
        let d = dir();
        let (ws, note) = load_from(&d.path().join("workspace.json"));
        assert!(ws.is_none());
        assert_eq!(note, LoadNote::Missing);
    }

    #[test]
    fn saving_writes_through_a_temporary_file_and_loading_reads_it_back() {
        let d = dir();
        let path = d.path().join("nested").join(FILE_NAME);
        save_to(&path, &sample()).unwrap();
        assert!(!path.with_extension("json.tmp").exists(), "the temporary file is renamed away");
        let (ws, note) = load_from(&path);
        assert_eq!((ws, note), (Some(sample()), LoadNote::Loaded));
    }

    #[test]
    fn a_corrupt_file_is_ignored_kept_aside_and_replaced_by_the_next_save() {
        let d = dir();
        let path = d.path().join(FILE_NAME);
        std::fs::write(&path, "{ this is not json").unwrap();
        let (mut saver, ws) = Saver::at(path.clone());
        assert!(ws.is_none() && matches!(saver.note, LoadNote::Corrupt(_)));
        assert!(saver.is_writable());
        assert_eq!(std::fs::read_to_string(path.with_extension("json.bad")).unwrap(), "{ this is not json", "the unreadable text is kept");
        assert!(saver.flush(&sample()));
        assert_eq!(load_from(&path).0, Some(sample()));
    }

    #[test]
    fn a_newer_file_is_never_overwritten() {
        let d = dir();
        let path = d.path().join(FILE_NAME);
        let newer = format!("{{\"version\": {}, \"future\": true}}", VERSION + 1);
        std::fs::write(&path, &newer).unwrap();
        let (mut saver, ws) = Saver::at(path.clone());
        assert!(ws.is_none());
        assert_eq!(saver.note, LoadNote::TooNew(VERSION + 1));
        assert!(!saver.is_writable());
        let t0 = Instant::now();
        saver.observe(t0, &sample());
        assert!(!saver.poll(t0 + Duration::from_secs(60)));
        assert!(!saver.flush(&sample()));
        assert!(saver.save_now(&sample()).is_err());
        assert!(saver.forget().is_err(), "forgetting does not delete it either");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), newer);
        assert!(!path.with_extension("json.tmp").exists());
    }

    #[test]
    fn a_memory_saver_never_writes() {
        let mut saver = Saver::memory();
        let t0 = Instant::now();
        saver.observe(t0, &sample());
        assert!(!saver.poll(t0 + Duration::from_secs(5)));
        assert_eq!(saver.saves, 0);
        assert!(saver.save_now(&sample()).is_err() && saver.forget().is_err());
        assert!(saver.file_text().is_err());
    }

    #[test]
    fn a_change_is_written_once_it_has_been_quiet_and_each_new_change_starts_the_wait_over() {
        let d = dir();
        let path = d.path().join(FILE_NAME);
        let (mut saver, _) = Saver::at(path.clone());
        let t0 = Instant::now();
        let mut state = sample();
        saver.observe(t0, &state);
        assert!(saver.is_pending());
        assert!(!saver.poll(t0 + Duration::from_millis(999)), "not yet");
        assert_eq!(saver.wake_in(t0 + Duration::from_millis(400)), Some(Duration::from_millis(600)));
        state.focused = Some(SavedTab::Logs);
        saver.observe(t0 + Duration::from_millis(900), &state);
        assert!(!saver.poll(t0 + Duration::from_millis(1500)), "a newer change restarted the wait");
        saver.observe(t0 + Duration::from_millis(1600), &state);
        assert!(saver.poll(t0 + Duration::from_millis(1900)));
        assert!(!saver.is_pending());
        assert_eq!(saver.saves, 1);
        assert_eq!(load_from(&path).0.unwrap().focused, Some(SavedTab::Logs));
        saver.observe(t0 + Duration::from_secs(10), &state);
        assert!(!saver.is_pending() && !saver.poll(t0 + Duration::from_secs(20)), "an unchanged state is not written again");
        assert_eq!(saver.saves, 1);
    }

    #[test]
    fn going_back_to_what_is_on_disk_cancels_the_write() {
        let d = dir();
        let (mut saver, _) = Saver::at(d.path().join(FILE_NAME));
        let t0 = Instant::now();
        let base = sample();
        assert!(saver.flush(&base));
        let mut changed = base.clone();
        changed.focused = None;
        saver.observe(t0, &changed);
        assert!(saver.is_pending());
        saver.observe(t0 + Duration::from_millis(10), &base);
        assert!(!saver.is_pending());
    }

    #[test]
    fn flushing_writes_only_a_state_that_differs() {
        let d = dir();
        let (mut saver, _) = Saver::at(d.path().join(FILE_NAME));
        assert!(saver.flush(&sample()));
        assert!(!saver.flush(&sample()), "nothing changed");
        assert_eq!(saver.saves, 1);
        assert!(saver.save_now(&sample()).is_ok(), "save now writes even an unchanged state");
        assert_eq!(saver.saves, 2);
        assert!(saver.last_write.is_some());
    }

    #[test]
    fn forgetting_deletes_the_file_and_what_was_known_of_it() {
        let d = dir();
        let path = d.path().join(FILE_NAME);
        let (mut saver, _) = Saver::at(path.clone());
        saver.flush(&sample());
        assert!(path.exists());
        saver.forget().unwrap();
        assert!(!path.exists() && saver.written().is_none() && saver.last_write.is_none());
        assert_eq!(saver.note, LoadNote::Missing);
        saver.forget().unwrap();
        assert!(saver.flush(&sample()), "it can be written again");
    }

    #[test]
    fn a_failed_write_is_reported_and_not_retried_every_frame() {
        let d = dir();
        let blocker = d.path().join("blocked");
        std::fs::write(&blocker, "a file where a folder should be").unwrap();
        let (mut saver, _) = Saver::at(blocker.join(FILE_NAME));
        let t0 = Instant::now();
        saver.observe(t0, &sample());
        assert!(!saver.poll(t0 + Duration::from_secs(2)));
        assert!(saver.last_error.is_some());
        saver.observe(t0 + Duration::from_secs(3), &sample());
        assert!(!saver.is_pending(), "the same state is not retried until it changes");
    }
}
