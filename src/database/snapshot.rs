use std::collections::VecDeque;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant, SystemTime};

use chrono::{DateTime, Local, NaiveDate, NaiveDateTime, TimeZone};
use fs2::FileExt;
use serde::{Deserialize, Serialize};

use crate::ui::status::{DB_STATUS, GlobalStatusIndicator, StatusState};

pub const LOCK_FILE: &str = "LOCK";
pub const STAMP_FILE: &str = "engine-stamp.json";
pub const SETTINGS_FILE: &str = "maintenance.json";
pub const PENDING_RESTORE_FILE: &str = "pending-restore.json";
pub const SNAPSHOTS_DIR: &str = "snapshots";
pub const MANIFEST_FILE: &str = "SNAPSHOT-MANIFEST.json";
pub const SPACE_MARGIN: f64 = 1.1;
pub const NAME_TS_FORMAT: &str = "%Y%m%d-%H%M%S";
const PARTIAL_SUFFIX: &str = ".partial";
const STALE_PARTIAL_AGE: Duration = Duration::from_secs(24 * 3600);
const COPY_BUF: usize = 1 << 20;
const AUTO_EXPORT_DELAY: Duration = Duration::from_secs(3);

#[derive(Clone, Debug)]
pub struct Event {
    pub at: DateTime<Local>,
    pub level: log::Level,
    pub msg: String,
}

static EVENTS: LazyLock<Mutex<VecDeque<Event>>> = LazyLock::new(|| Mutex::new(VecDeque::new()));
static EVENT_HOOK: LazyLock<Mutex<Option<Arc<dyn Fn() + Send + Sync>>>> = LazyLock::new(|| Mutex::new(None));
const EVENT_CAP: usize = 300;

/// Log a maintenance event to the `log` facade and the panel ring buffer.
pub fn log_event(level: log::Level, msg: impl Into<String>) {
    let msg = msg.into();
    log::log!(level, "[DB] {msg}");
    if let Ok(mut q) = EVENTS.lock() {
        if q.len() >= EVENT_CAP { q.pop_front(); }
        q.push_back(Event { at: Local::now(), level, msg });
    }
    if let Ok(g) = EVENT_HOOK.lock() {
        if let Some(h) = g.as_ref() { h(); }
    }
}

/// Snapshot of the retained maintenance events, oldest first.
pub fn recent_events() -> Vec<Event> {
    EVENTS.lock().map(|q| q.iter().cloned().collect()).unwrap_or_default()
}

/// Register a callback invoked after every logged event (used to request UI repaints).
pub fn set_event_hook(hook: Option<Arc<dyn Fn() + Send + Sync>>) {
    if let Ok(mut g) = EVENT_HOOK.lock() { *g = hook; }
}

#[derive(Debug)]
pub enum SnapshotError {
    InsufficientSpace { needed: u64, available: u64, dir: PathBuf },
    InUse(PathBuf),
    Missing(PathBuf),
    Corrupt(String),
    Io { context: String, source: io::Error },
    Other(String),
}

impl SnapshotError {
    fn io(context: impl Into<String>, source: io::Error) -> Self {
        Self::Io { context: context.into(), source }
    }

    /// Message suitable for showing to the user, including the recovery hint.
    pub fn user_message(&self) -> String {
        match self {
            Self::InsufficientSpace { needed, available, dir } => format!(
                "Not enough free disk space for the pre-upgrade database snapshot in {}: need {} but only {} available. \
                 Free up space, or use 'Open without snapshot' in Database Maintenance to proceed without a backup.",
                dir.display(), human(*needed), human(*available)
            ),
            Self::InUse(p) => format!("The database at {} is in use by another process. Close the other instance and retry.", p.display()),
            Self::Missing(p) => format!("Path not found: {}", p.display()),
            Self::Corrupt(m) => format!("Snapshot is damaged: {m}"),
            Self::Io { context, source } => format!(
                "{context}: {source}. The database was not opened. Fix the problem, or use 'Open without snapshot' in Database Maintenance to proceed without a backup."
            ),
            Self::Other(m) => m.clone(),
        }
    }
}

impl std::fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.user_message())
    }
}

impl std::error::Error for SnapshotError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

pub fn human(bytes: u64) -> String {
    humansize::format_size(bytes, humansize::BINARY)
}

const CARGO_LOCK: &str = include_str!("../../Cargo.lock");

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EngineVersions {
    pub surrealdb: String,
    pub surrealkv: Option<String>,
}

fn lock_package_versions(lock: &str, name: &str) -> Vec<String> {
    let header = format!("name = \"{name}\"");
    let mut out = Vec::new();
    let mut lines = lock.lines();
    while let Some(line) = lines.next() {
        if line.trim() == header {
            if let Some(v) = lines.next().and_then(|l| l.trim().strip_prefix("version = \"")) {
                out.push(v.trim_end_matches('"').to_string());
            }
        }
    }
    out
}

fn version_key(v: &str) -> (u64, u64, u64, bool, String) {
    let (core, pre) = match v.split_once('-') {
        Some((c, p)) => (c, Some(p)),
        None => (v, None),
    };
    let mut parts = core.split('.').map(|p| p.parse::<u64>().unwrap_or(0));
    (
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
        pre.is_none(),
        pre.unwrap_or("").to_string(),
    )
}

/// Highest version of `name` recorded in a Cargo.lock text.
pub fn lock_package_version(lock: &str, name: &str) -> Option<String> {
    lock_package_versions(lock, name).into_iter().max_by_key(|v| version_key(v))
}

static ENGINE: LazyLock<EngineVersions> = LazyLock::new(|| EngineVersions {
    surrealdb: lock_package_version(CARGO_LOCK, "surrealdb").unwrap_or_else(|| "unknown".into()),
    surrealkv: lock_package_version(CARGO_LOCK, "surrealkv"),
});

/// Engine versions parsed from the Cargo.lock embedded at build time.
pub fn engine_versions() -> &'static EngineVersions {
    &ENGINE
}

pub fn db_parent(store: &Path) -> PathBuf {
    match store.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    }
}

pub fn store_name(store: &Path) -> String {
    store.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "db".into())
}

pub fn snapshots_dir(store: &Path) -> PathBuf {
    db_parent(store).join(SNAPSHOTS_DIR)
}

pub fn stamp_path(store: &Path) -> PathBuf {
    db_parent(store).join(STAMP_FILE)
}

pub fn settings_path(store: &Path) -> PathBuf {
    db_parent(store).join(SETTINGS_FILE)
}

pub fn pending_restore_path(store: &Path) -> PathBuf {
    db_parent(store).join(PENDING_RESTORE_FILE)
}

/// True when the folder holds anything besides the SurrealKV LOCK file.
pub fn store_has_data(store: &Path) -> bool {
    match fs::read_dir(store) {
        Ok(rd) => rd.flatten().any(|e| e.file_name() != LOCK_FILE),
        Err(_) => false,
    }
}

/// True when another handle currently holds the SurrealKV LOCK file.
pub fn store_in_use(store: &Path) -> bool {
    let Ok(f) = File::open(store.join(LOCK_FILE)) else { return false };
    match f.try_lock_exclusive() {
        Ok(()) => {
            let _ = FileExt::unlock(&f);
            false
        }
        Err(e) => e.kind() == io::ErrorKind::WouldBlock || e.raw_os_error() == fs2::lock_contended_error().raw_os_error(),
    }
}

/// Wait until the engine has shut down and released the store LOCK; false on timeout.
pub async fn wait_released(store: &Path, timeout: Duration) -> bool {
    let started = Instant::now();
    while store_in_use(store) {
        if started.elapsed() > timeout {
            log_event(log::Level::Warn, format!("Store {} is still locked after {:?}", store.display(), timeout));
            return false;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    true
}

/// Total size of regular files below `root`, skipping LOCK files.
pub fn dir_size(root: &Path) -> u64 {
    let mut total = 0u64;
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = fs::read_dir(&dir) else { continue };
        for e in rd.flatten() {
            let Ok(ft) = e.file_type() else { continue };
            if ft.is_dir() {
                stack.push(e.path());
            } else if ft.is_file() && e.file_name() != LOCK_FILE {
                total += e.metadata().map(|m| m.len()).unwrap_or(0);
            }
        }
    }
    total
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct MaintenanceSettings {
    pub snapshot_on_engine_change: bool,
    pub auto_export_on_engine_change: bool,
    pub export_size_threshold_bytes: u64,
    pub retention_keep: usize,
    pub retention_min_age_days: u32,
    pub deep_verify_snapshots: bool,
}

impl Default for MaintenanceSettings {
    fn default() -> Self {
        Self {
            snapshot_on_engine_change: true,
            auto_export_on_engine_change: true,
            export_size_threshold_bytes: 2 * 1024 * 1024 * 1024,
            retention_keep: 5,
            retention_min_age_days: 7,
            deep_verify_snapshots: false,
        }
    }
}

impl MaintenanceSettings {
    pub fn load(store: &Path) -> Self {
        match fs::read_to_string(settings_path(store)) {
            Ok(s) => serde_json::from_str(&s).unwrap_or_else(|e| {
                log::warn!("[DB] maintenance settings unreadable ({e}); using defaults");
                Self::default()
            }),
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self, store: &Path) -> io::Result<()> {
        let json = serde_json::to_string_pretty(self).map_err(io::Error::other)?;
        write_atomic(&settings_path(store), json.as_bytes())
    }

    pub fn min_age(&self) -> Duration {
        Duration::from_secs(self.retention_min_age_days as u64 * 24 * 3600)
    }
}

fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    if let Some(parent) = path.parent() { fs::create_dir_all(parent)?; }
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    let mut f = File::create(&tmp)?;
    f.write_all(bytes)?;
    let _ = f.sync_all();
    drop(f);
    fs::rename(&tmp, path)
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EngineStamp {
    pub surrealdb_version: String,
    #[serde(default)]
    pub surrealkv_version: Option<String>,
    pub app_version: String,
    pub last_opened_rfc3339: String,
    pub store_path: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StampState {
    Missing,
    Unreadable(String),
    Present(EngineStamp),
}

pub fn read_stamp(store: &Path) -> StampState {
    match fs::read_to_string(stamp_path(store)) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => StampState::Missing,
        Err(e) => StampState::Unreadable(e.to_string()),
        Ok(s) => match serde_json::from_str::<EngineStamp>(&s) {
            Ok(st) => StampState::Present(st),
            Err(e) => StampState::Unreadable(e.to_string()),
        },
    }
}

pub fn stamp_for(store: &Path, versions: &EngineVersions, now: DateTime<Local>) -> EngineStamp {
    EngineStamp {
        surrealdb_version: versions.surrealdb.clone(),
        surrealkv_version: versions.surrealkv.clone(),
        app_version: env!("CARGO_PKG_VERSION").to_string(),
        last_opened_rfc3339: now.to_rfc3339(),
        store_path: store.to_string_lossy().into_owned(),
    }
}

pub fn write_stamp(store: &Path, stamp: &EngineStamp) -> io::Result<()> {
    let json = serde_json::to_string_pretty(stamp).map_err(io::Error::other)?;
    write_atomic(&stamp_path(store), json.as_bytes())
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum EngineChange {
    NoStore,
    Unchanged,
    Changed { from: Option<String>, to: String, reason: String },
}

impl EngineChange {
    pub fn is_changed(&self) -> bool {
        matches!(self, Self::Changed { .. })
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::NoStore => "no_store",
            Self::Unchanged => "unchanged",
            Self::Changed { from: None, .. } => "unknown",
            Self::Changed { .. } => "changed",
        }
    }
}

/// Decide whether the store was last opened by a different (or unknown) engine.
pub fn classify(store: &Path, has_data: bool, stamp: &StampState, current: &EngineVersions) -> EngineChange {
    if !has_data {
        return EngineChange::NoStore;
    }
    let to = current.surrealdb.clone();
    match stamp {
        StampState::Missing => EngineChange::Changed { from: None, to, reason: "no engine stamp found (unknown engine)".into() },
        StampState::Unreadable(e) => EngineChange::Changed { from: None, to, reason: format!("engine stamp unreadable: {e}") },
        StampState::Present(st) => {
            let stamp_store = Path::new(&st.store_path).file_name().map(|n| n.to_string_lossy().into_owned());
            if stamp_store.as_deref() != Some(store_name(store).as_str()) {
                EngineChange::Changed { from: None, to, reason: format!("engine stamp belongs to another store ({})", st.store_path) }
            } else if st.surrealdb_version != current.surrealdb {
                EngineChange::Changed { from: Some(st.surrealdb_version.clone()), to, reason: format!("surrealdb {} -> {}", st.surrealdb_version, current.surrealdb) }
            } else if matches!((&st.surrealkv_version, &current.surrealkv), (Some(a), Some(b)) if a != b) {
                let from = st.surrealkv_version.clone().unwrap_or_default();
                EngineChange::Changed { from: Some(st.surrealdb_version.clone()), to, reason: format!("surrealkv {} -> {}", from, current.surrealkv.clone().unwrap_or_default()) }
            } else {
                EngineChange::Unchanged
            }
        }
    }
}

pub type SpaceFn = dyn Fn(&Path) -> io::Result<u64> + Send + Sync;

/// Free bytes on the volume holding `path` (nearest existing ancestor).
pub fn available_space(path: &Path) -> io::Result<u64> {
    let mut p = path.to_path_buf();
    loop {
        if p.exists() {
            return fs2::available_space(&p);
        }
        if !p.pop() || p.as_os_str().is_empty() {
            return fs2::available_space(Path::new("."));
        }
    }
}

pub fn required_space(store_bytes: u64) -> u64 {
    (store_bytes as f64 * SPACE_MARGIN).ceil() as u64
}

pub fn check_free_space(needed: u64, dir: &Path, space: &SpaceFn) -> Result<(), SnapshotError> {
    let available = space(dir).map_err(|e| SnapshotError::io(format!("Cannot determine free space for {}", dir.display()), e))?;
    if available < needed {
        return Err(SnapshotError::InsufficientSpace { needed, available, dir: dir.to_path_buf() });
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParsedName {
    pub store: String,
    pub label: String,
    pub engine: String,
    pub created: Option<DateTime<Local>>,
}

static NAME_RE: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(r"^(?P<store>.+)-(?P<label>pre|post|manual|export)-(?P<engine>[0-9][0-9A-Za-z.+_-]*?)-(?P<date>\d{8})(?:-(?P<time>\d{6}))?(?:-(?P<n>\d+))?$").unwrap()
});

pub fn snapshot_file_name(store_name: &str, label: &str, engine: &str, now: DateTime<Local>) -> String {
    format!("{store_name}-{label}-{engine}-{}", now.format(NAME_TS_FORMAT))
}

/// Parse `<store>-<pre|post|manual|export>-<engine>-<yyyymmdd[-hhmmss]>` snapshot names.
pub fn parse_snapshot_name(name: &str) -> Option<ParsedName> {
    let stem = name.strip_suffix(".surql").unwrap_or(name);
    let c = NAME_RE.captures(stem)?;
    let date = NaiveDate::parse_from_str(&c["date"], "%Y%m%d").ok();
    let created = match (date, c.name("time")) {
        (Some(d), Some(t)) => NaiveDateTime::parse_from_str(&format!("{}-{}", &c["date"], t.as_str()), NAME_TS_FORMAT)
            .ok()
            .and_then(|n| Local.from_local_datetime(&n).earliest()),
        (Some(d), None) => d.and_hms_opt(0, 0, 0).and_then(|n| Local.from_local_datetime(&n).earliest()),
        _ => None,
    };
    Some(ParsedName { store: c["store"].to_string(), label: c["label"].to_string(), engine: c["engine"].to_string(), created })
}

pub fn unique_path(dir: &Path, base: &str, ext: &str) -> PathBuf {
    let first = dir.join(format!("{base}{ext}"));
    if !first.exists() && !dir.join(format!("{base}{ext}{PARTIAL_SUFFIX}")).exists() {
        return first;
    }
    let mut n = 2;
    loop {
        let p = dir.join(format!("{base}-{n}{ext}"));
        if !p.exists() && !dir.join(format!("{base}-{n}{ext}{PARTIAL_SUFFIX}")).exists() {
            return p;
        }
        n += 1;
    }
}

pub type ProgressFn = Arc<dyn Fn(u64, u64, &str) + Send + Sync>;

/// Progress sink that discards updates.
pub fn no_progress() -> ProgressFn {
    Arc::new(|_, _, _| {})
}

/// Progress sink that mirrors updates into the DB status card (throttled).
pub fn status_progress(prefix: &'static str) -> ProgressFn {
    let last = Mutex::new(Instant::now() - Duration::from_secs(1));
    Arc::new(move |cur, total, what| {
        let Ok(mut l) = last.lock() else { return };
        if cur < total && l.elapsed() < Duration::from_millis(100) {
            return;
        }
        *l = Instant::now();
        DB_STATUS.set_progress(cur, total);
        if total == 0 {
            DB_STATUS.set_detail(format!("{prefix}: {what} ({})", human(cur)));
        } else {
            DB_STATUS.set_detail(format!("{prefix}: {what} ({} of {})", human(cur), human(total)));
        }
    })
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileEntry {
    pub path: String,
    pub len: u64,
    pub blake3: String,
}

#[derive(Clone, Debug)]
struct SrcFile {
    rel: PathBuf,
    abs: PathBuf,
    len: u64,
    modified: Option<SystemTime>,
}

#[derive(Debug)]
pub struct CopyOutcome {
    pub files: Vec<FileEntry>,
    pub bytes: u64,
    pub consistent: bool,
    pub attempts: u32,
}

fn rel_string(rel: &Path) -> String {
    rel.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect::<Vec<_>>().join("/")
}

fn scan_tree(root: &Path, skip_top: &[&str], skip_any: &[&str]) -> io::Result<(Vec<PathBuf>, Vec<SrcFile>)> {
    let mut dirs = Vec::new();
    let mut files = Vec::new();
    let mut stack = vec![PathBuf::new()];
    while let Some(rel_dir) = stack.pop() {
        for e in fs::read_dir(root.join(&rel_dir))? {
            let e = e?;
            let name = e.file_name();
            if (rel_dir.as_os_str().is_empty() && skip_top.iter().any(|s| name == *s)) || skip_any.iter().any(|s| name == *s) {
                continue;
            }
            let rel = rel_dir.join(&name);
            let ft = e.file_type()?;
            if ft.is_dir() {
                dirs.push(rel.clone());
                stack.push(rel);
            } else if ft.is_file() {
                let md = e.metadata()?;
                files.push(SrcFile { abs: e.path(), len: md.len(), modified: md.modified().ok(), rel });
            }
        }
    }
    files.sort_by(|a, b| a.rel.cmp(&b.rel));
    Ok((dirs, files))
}

fn same_files(a: &[SrcFile], b: &[SrcFile]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.rel == y.rel && x.len == y.len && x.modified == y.modified)
}

fn copy_file_hashed(src: &Path, dst: &Path, on_bytes: &mut dyn FnMut(u64)) -> io::Result<(u64, String)> {
    let mut r = File::open(src)?;
    let mut w = File::create(dst)?;
    let mut hasher = blake3::Hasher::new();
    let mut buf = vec![0u8; COPY_BUF];
    let mut total = 0u64;
    loop {
        let n = r.read(&mut buf)?;
        if n == 0 { break; }
        w.write_all(&buf[..n])?;
        hasher.update(&buf[..n]);
        total += n as u64;
        on_bytes(n as u64);
    }
    let _ = w.sync_all();
    if let Ok(m) = r.metadata().and_then(|m| m.modified()) {
        let _ = w.set_modified(m);
    }
    Ok((total, hasher.finalize().to_hex().to_string()))
}

fn sync_dir(dir: &Path) {
    if let Ok(f) = File::open(dir) {
        let _ = f.sync_all();
    }
}

/// Copy a directory tree (LOCK files skipped) with blake3 hashes; retries if the source changes mid-copy.
pub fn copy_tree(src: &Path, dst: &Path, skip_top: &[&str], attempts: u32, progress: &mut dyn FnMut(u64, u64)) -> io::Result<CopyOutcome> {
    const SKIP_ANY: &[&str] = &[LOCK_FILE];
    let attempts = attempts.max(1);
    for attempt in 1..=attempts {
        let (dirs, files) = scan_tree(src, skip_top, SKIP_ANY)?;
        let total: u64 = files.iter().map(|f| f.len).sum();
        if dst.exists() {
            fs::remove_dir_all(dst)?;
        }
        fs::create_dir_all(dst)?;
        for d in &dirs {
            fs::create_dir_all(dst.join(d))?;
        }
        let mut copied = 0u64;
        let mut entries = Vec::with_capacity(files.len());
        let mut vanished = false;
        for f in &files {
            let target = dst.join(&f.rel);
            let res = copy_file_hashed(&f.abs, &target, &mut |n| {
                copied += n;
                progress(copied, total);
            });
            match res {
                Ok((len, hash)) => entries.push(FileEntry { path: rel_string(&f.rel), len, blake3: hash }),
                Err(e) if e.kind() == io::ErrorKind::NotFound && !target.exists() => {
                    vanished = true;
                    break;
                }
                Err(e) => return Err(e),
            }
        }
        let stable = !vanished && scan_tree(src, skip_top, SKIP_ANY).map(|(_, after)| same_files(&files, &after)).unwrap_or(false);
        if stable || attempt == attempts {
            for d in dirs.iter().chain(std::iter::once(&PathBuf::new())) {
                sync_dir(&dst.join(d));
            }
            progress(copied, total.max(copied));
            return Ok(CopyOutcome { files: entries, bytes: copied, consistent: stable, attempts: attempt });
        }
        log_event(log::Level::Warn, format!("Source changed while copying (attempt {attempt}/{attempts}); retrying"));
    }
    unreachable!("copy_tree loop always returns on the last attempt")
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SnapshotManifest {
    pub version: u32,
    pub store_name: String,
    pub label: String,
    pub engine_from: Option<String>,
    pub engine_running: String,
    pub created_rfc3339: String,
    pub source_path: String,
    pub hot: bool,
    pub consistent: bool,
    pub total_bytes: u64,
    pub files: Vec<FileEntry>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum SnapshotKind {
    Dir,
    Surql,
}

impl SnapshotKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Dir => "dir",
            Self::Surql => "surql",
        }
    }
}

fn ser_dt<S: serde::Serializer>(v: &Option<DateTime<Local>>, s: S) -> Result<S::Ok, S::Error> {
    match v {
        Some(d) => s.serialize_str(&d.to_rfc3339()),
        None => s.serialize_none(),
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct SnapshotInfo {
    pub name: String,
    pub path: PathBuf,
    pub kind: SnapshotKind,
    pub size_bytes: u64,
    #[serde(serialize_with = "ser_dt")]
    pub created: Option<DateTime<Local>>,
    pub engine_version: Option<String>,
    pub store_name: Option<String>,
    pub label: Option<String>,
    pub has_manifest: bool,
}

fn info_for(path: &Path) -> Option<SnapshotInfo> {
    let name = path.file_name()?.to_string_lossy().into_owned();
    if name.starts_with('.') || name.ends_with(PARTIAL_SUFFIX) {
        return None;
    }
    let md = fs::metadata(path).ok()?;
    let kind = if md.is_dir() {
        SnapshotKind::Dir
    } else if name.ends_with(".surql") {
        SnapshotKind::Surql
    } else {
        return None;
    };
    let size_bytes = if md.is_dir() { dir_size(path) } else { md.len() };
    let parsed = parse_snapshot_name(&name);
    let created = parsed.as_ref().and_then(|p| p.created).or_else(|| md.modified().ok().map(DateTime::<Local>::from));
    Some(SnapshotInfo {
        has_manifest: md.is_dir() && path.join(MANIFEST_FILE).is_file(),
        engine_version: parsed.as_ref().map(|p| p.engine.clone()),
        store_name: parsed.as_ref().map(|p| p.store.clone()),
        label: parsed.as_ref().map(|p| p.label.clone()),
        name,
        path: path.to_path_buf(),
        kind,
        size_bytes,
        created,
    })
}

/// Snapshots and exports found next to `store`, newest first.
pub fn list_snapshots_for(store: &Path) -> Vec<SnapshotInfo> {
    let dir = snapshots_dir(store);
    let mut out: Vec<SnapshotInfo> = match fs::read_dir(&dir) {
        Ok(rd) => rd.flatten().filter_map(|e| info_for(&e.path())).collect(),
        Err(_) => Vec::new(),
    };
    out.sort_by(|a, b| b.created.cmp(&a.created).then_with(|| b.name.cmp(&a.name)));
    out
}

/// Snapshots and exports next to the configured store, newest first.
pub async fn list_snapshots() -> Vec<SnapshotInfo> {
    let store = PathBuf::from(super::get_db_path());
    tokio::task::spawn_blocking(move || list_snapshots_for(&store)).await.unwrap_or_default()
}

/// Copy `store` into `<store-parent>/snapshots/<store>-<label>-<engine>-<ts>/` with a hash manifest.
pub fn create_snapshot_blocking(
    store: &Path,
    label: &str,
    engine_from: Option<String>,
    now: DateTime<Local>,
    space: &SpaceFn,
    progress: &ProgressFn,
) -> Result<SnapshotInfo, SnapshotError> {
    if !store.is_dir() {
        return Err(SnapshotError::Missing(store.to_path_buf()));
    }
    let versions = engine_versions();
    let dir = snapshots_dir(store);
    fs::create_dir_all(&dir).map_err(|e| SnapshotError::io(format!("Cannot create {}", dir.display()), e))?;
    let size = dir_size(store);
    check_free_space(required_space(size), &dir, space)?;

    let hot = store_in_use(store);
    let base = snapshot_file_name(&store_name(store), label, &versions.surrealdb, now);
    let final_path = unique_path(&dir, &base, "");
    let mut partial = final_path.as_os_str().to_owned();
    partial.push(PARTIAL_SUFFIX);
    let partial = PathBuf::from(partial);

    log_event(log::Level::Info, format!("Snapshot: copying {} ({}) to {}", store.display(), human(size), final_path.display()));
    let started = Instant::now();
    let p = progress.clone();
    let outcome = copy_tree(store, &partial, &[LOCK_FILE], if hot { 2 } else { 1 }, &mut |cur, total| p(cur, total, "Copying database files"))
        .map_err(|e| {
            let _ = fs::remove_dir_all(&partial);
            SnapshotError::io(format!("Snapshot copy to {} failed", partial.display()), e)
        })?;

    let manifest = SnapshotManifest {
        version: 1,
        store_name: store_name(store),
        label: label.to_string(),
        engine_from,
        engine_running: versions.surrealdb.clone(),
        created_rfc3339: now.to_rfc3339(),
        source_path: store.to_string_lossy().into_owned(),
        hot,
        consistent: outcome.consistent,
        total_bytes: outcome.bytes,
        files: outcome.files,
    };
    let finish = (|| -> io::Result<()> {
        let json = serde_json::to_vec_pretty(&manifest).map_err(io::Error::other)?;
        write_atomic(&partial.join(MANIFEST_FILE), &json)?;
        sync_dir(&partial);
        fs::rename(&partial, &final_path)?;
        sync_dir(&dir);
        Ok(())
    })();
    if let Err(e) = finish {
        let _ = fs::remove_dir_all(&partial);
        return Err(SnapshotError::io("Finalizing snapshot failed", e));
    }
    if !outcome.consistent {
        log_event(log::Level::Warn, format!("Snapshot {} was taken while the database was changing; it may be inconsistent", final_path.display()));
    }
    log_event(log::Level::Info, format!("Snapshot done: {} files, {} in {:.1}s ({})", manifest.files.len(), human(outcome.bytes), started.elapsed().as_secs_f32(), final_path.display()));
    info_for(&final_path).ok_or_else(|| SnapshotError::Other("snapshot vanished after creation".into()))
}

/// Create a snapshot of the store on a blocking thread.
pub async fn create_snapshot(store: PathBuf, label: &'static str, engine_from: Option<String>, progress: ProgressFn) -> Result<SnapshotInfo, SnapshotError> {
    tokio::task::spawn_blocking(move || {
        create_snapshot_blocking(&store, label, engine_from, Local::now(), &available_space, &progress)
    })
    .await
    .map_err(|e| SnapshotError::Other(format!("snapshot task failed: {e}")))?
}

/// Snapshot of the configured store, labelled "manual".
pub async fn snapshot_now() -> Result<SnapshotInfo, SnapshotError> {
    snapshot_now_with(status_progress("Snapshot")).await
}

/// Same as `snapshot_now` with a caller-supplied progress sink.
pub async fn snapshot_now_with(progress: ProgressFn) -> Result<SnapshotInfo, SnapshotError> {
    let store = PathBuf::from(super::get_db_path());
    let _ga = super::db_activity("Snapshot DB");
    let info = create_snapshot(store.clone(), "manual", None, progress).await?;
    let settings = MaintenanceSettings::load(&store);
    prune_snapshots(&store, &settings, Local::now());
    Ok(info)
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct SnapshotCheck {
    pub ok: bool,
    pub has_manifest: bool,
    pub files_checked: usize,
    pub bytes_checked: u64,
    pub missing: Vec<String>,
    pub mismatched: Vec<String>,
    pub extra: Vec<String>,
    pub consistent_when_taken: Option<bool>,
    pub notes: Vec<String>,
}

/// Recompute file hashes of a directory snapshot and compare with its manifest.
pub fn verify_snapshot_dir(dir: &Path, progress: &ProgressFn) -> Result<SnapshotCheck, SnapshotError> {
    if !dir.is_dir() {
        return Err(SnapshotError::Missing(dir.to_path_buf()));
    }
    let mut check = SnapshotCheck::default();
    let manifest_path = dir.join(MANIFEST_FILE);
    let (_, files) = scan_tree(dir, &[], &[]).map_err(|e| SnapshotError::io(format!("Cannot read {}", dir.display()), e))?;
    let actual: Vec<&SrcFile> = files.iter().filter(|f| f.rel != Path::new(MANIFEST_FILE)).collect();
    if !manifest_path.is_file() {
        check.notes.push("No manifest (legacy snapshot): only a structural check was possible".into());
        check.files_checked = actual.len();
        check.bytes_checked = actual.iter().map(|f| f.len).sum();
        check.ok = !actual.is_empty() && dir.join("manifest").is_dir();
        if !check.ok {
            check.notes.push("Folder does not look like a SurrealKV store (no manifest/ directory)".into());
        }
        return Ok(check);
    }
    check.has_manifest = true;
    let text = fs::read_to_string(&manifest_path).map_err(|e| SnapshotError::io("Cannot read snapshot manifest", e))?;
    let manifest: SnapshotManifest = serde_json::from_str(&text).map_err(|e| SnapshotError::Corrupt(format!("manifest unreadable: {e}")))?;
    check.consistent_when_taken = Some(manifest.consistent);
    if !manifest.consistent {
        check.notes.push("Snapshot was taken while the database was changing".into());
    }
    let total: u64 = manifest.files.iter().map(|f| f.len).sum();
    let mut done = 0u64;
    for entry in &manifest.files {
        let path = dir.join(&entry.path);
        match File::open(&path) {
            Err(_) => check.missing.push(entry.path.clone()),
            Ok(mut f) => {
                let mut hasher = blake3::Hasher::new();
                let mut buf = vec![0u8; COPY_BUF];
                let mut len = 0u64;
                loop {
                    let n = f.read(&mut buf).map_err(|e| SnapshotError::io(format!("Cannot read {}", path.display()), e))?;
                    if n == 0 { break; }
                    hasher.update(&buf[..n]);
                    len += n as u64;
                    done += n as u64;
                    progress(done, total, "Verifying snapshot files");
                }
                if len != entry.len || hasher.finalize().to_hex().as_str() != entry.blake3 {
                    check.mismatched.push(entry.path.clone());
                }
                check.files_checked += 1;
                check.bytes_checked += len;
            }
        }
    }
    let known: std::collections::HashSet<&str> = manifest.files.iter().map(|f| f.path.as_str()).collect();
    for f in &actual {
        let rel = rel_string(&f.rel);
        if !known.contains(rel.as_str()) {
            check.extra.push(rel);
        }
    }
    check.ok = check.missing.is_empty() && check.mismatched.is_empty() && check.extra.is_empty();
    Ok(check)
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct PruneReport {
    pub deleted: Vec<PathBuf>,
    pub kept: usize,
    pub protected: Vec<PathBuf>,
    pub errors: Vec<String>,
}

fn remove_snapshot_path(info_path: &Path, is_dir: bool) -> io::Result<()> {
    if is_dir { fs::remove_dir_all(info_path) } else { fs::remove_file(info_path) }
}

/// Keep the newest N per kind, never deleting snapshots younger than the minimum age or the latest pre/post ones.
pub fn prune_snapshots(store: &Path, settings: &MaintenanceSettings, now: DateTime<Local>) -> PruneReport {
    let mut report = PruneReport::default();
    let me = store_name(store);
    let all: Vec<SnapshotInfo> = list_snapshots_for(store).into_iter().filter(|s| s.store_name.as_deref() == Some(me.as_str())).collect();
    let min_age = chrono::Duration::from_std(settings.min_age()).unwrap_or_else(|_| chrono::Duration::days(7));
    for (kind, protected_label) in [(SnapshotKind::Dir, "pre"), (SnapshotKind::Surql, "post")] {
        let group: Vec<&SnapshotInfo> = all.iter().filter(|s| s.kind == kind).collect();
        let protected = group.iter().find(|s| s.label.as_deref() == Some(protected_label)).map(|s| s.path.clone());
        for (i, s) in group.iter().enumerate() {
            let young = s.created.map(|c| now.signed_duration_since(c) < min_age).unwrap_or(true);
            if i < settings.retention_keep || young {
                report.kept += 1;
                continue;
            }
            if protected.as_ref() == Some(&s.path) {
                report.protected.push(s.path.clone());
                report.kept += 1;
                continue;
            }
            match remove_snapshot_path(&s.path, kind == SnapshotKind::Dir) {
                Ok(()) => {
                    log_event(log::Level::Info, format!("Retention: deleted old snapshot {}", s.path.display()));
                    report.deleted.push(s.path.clone());
                }
                Err(e) => {
                    log_event(log::Level::Warn, format!("Retention: could not delete {}: {e}", s.path.display()));
                    report.errors.push(format!("{}: {e}", s.path.display()));
                }
            }
        }
    }
    if let Ok(rd) = fs::read_dir(snapshots_dir(store)) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            let old = e.metadata().and_then(|m| m.modified()).ok().and_then(|m| m.elapsed().ok()).map(|a| a > STALE_PARTIAL_AGE).unwrap_or(false);
            if name.ends_with(PARTIAL_SUFFIX) && old {
                let is_dir = e.path().is_dir();
                match remove_snapshot_path(&e.path(), is_dir) {
                    Ok(()) => log_event(log::Level::Info, format!("Retention: deleted stale partial {}", e.path().display())),
                    Err(err) => log_event(log::Level::Warn, format!("Retention: could not delete {}: {err}", e.path().display())),
                }
            }
        }
    }
    report
}

/// Delete one snapshot or export, refusing paths outside the snapshots folder.
pub fn delete_snapshot(store: &Path, path: &Path) -> Result<(), SnapshotError> {
    let dir = snapshots_dir(store);
    let canon_dir = fs::canonicalize(&dir).map_err(|e| SnapshotError::io(format!("Cannot resolve {}", dir.display()), e))?;
    let canon = fs::canonicalize(path).map_err(|_| SnapshotError::Missing(path.to_path_buf()))?;
    if canon.parent() != Some(canon_dir.as_path()) {
        return Err(SnapshotError::Other(format!("{} is not inside the snapshots folder", path.display())));
    }
    let is_dir = canon.is_dir();
    remove_snapshot_path(&canon, is_dir).map_err(|e| SnapshotError::io(format!("Cannot delete {}", path.display()), e))?;
    log_event(log::Level::Info, format!("Deleted snapshot {}", path.display()));
    Ok(())
}

#[derive(Clone, Debug, Serialize)]
pub struct RestoreReport {
    pub store: PathBuf,
    pub restored_from: PathBuf,
    pub replaced_to: Option<PathBuf>,
    pub bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PendingRestore {
    pub snapshot: PathBuf,
    pub requested_rfc3339: String,
}

/// Replace `store` with a copy of `snapshot_dir`, moving the old store aside to `<store>.replaced-<ts>`; the store must not be open.
pub fn restore_snapshot_blocking(
    snapshot_dir: &Path,
    store: &Path,
    now: DateTime<Local>,
    space: &SpaceFn,
    progress: &ProgressFn,
) -> Result<RestoreReport, SnapshotError> {
    if !snapshot_dir.is_dir() {
        return Err(SnapshotError::Missing(snapshot_dir.to_path_buf()));
    }
    if store_in_use(store) {
        return Err(SnapshotError::InUse(store.to_path_buf()));
    }
    let check = verify_snapshot_dir(snapshot_dir, progress)?;
    if !check.ok {
        return Err(SnapshotError::Corrupt(format!(
            "verification failed (missing: {}, mismatched: {}, extra: {}, notes: {})",
            check.missing.len(), check.mismatched.len(), check.extra.len(), check.notes.join("; ")
        )));
    }
    let parent = db_parent(store);
    fs::create_dir_all(&parent).map_err(|e| SnapshotError::io(format!("Cannot create {}", parent.display()), e))?;
    check_free_space(required_space(check.bytes_checked), &parent, space)?;

    let ts = now.format(NAME_TS_FORMAT).to_string();
    let name = store_name(store);
    let staging = parent.join(format!("{name}.restoring-{ts}"));
    log_event(log::Level::Info, format!("Restore: copying {} to {}", snapshot_dir.display(), staging.display()));
    let p = progress.clone();
    let outcome = copy_tree(snapshot_dir, &staging, &[LOCK_FILE, MANIFEST_FILE], 1, &mut |cur, total| p(cur, total, "Restoring snapshot files")).map_err(|e| {
        let _ = fs::remove_dir_all(&staging);
        SnapshotError::io("Restore copy failed", e)
    })?;

    let mut replaced_to = None;
    if store.exists() {
        let aside = unique_path(&parent, &format!("{name}.replaced-{ts}"), "");
        fs::rename(store, &aside).map_err(|e| {
            let _ = fs::remove_dir_all(&staging);
            SnapshotError::io(format!("Cannot move {} aside", store.display()), e)
        })?;
        log_event(log::Level::Info, format!("Restore: previous store kept at {}", aside.display()));
        replaced_to = Some(aside);
    }
    if let Err(e) = fs::rename(&staging, store) {
        if let Some(aside) = &replaced_to {
            let _ = fs::rename(aside, store);
        }
        return Err(SnapshotError::io("Cannot move restored store into place", e));
    }
    sync_dir(&parent);

    restamp_after_restore(store, snapshot_dir, now);
    log_event(log::Level::Info, format!("Restore complete: {} restored from {}", store.display(), snapshot_dir.display()));
    Ok(RestoreReport { store: store.to_path_buf(), restored_from: snapshot_dir.to_path_buf(), replaced_to, bytes: outcome.bytes })
}

/// Point the engine stamp at the engine that wrote the restored data, or drop it when that is unknown.
fn restamp_after_restore(store: &Path, snapshot_dir: &Path, now: DateTime<Local>) {
    let name = snapshot_dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let parsed = parse_snapshot_name(&name);
    let manifest: Option<SnapshotManifest> = fs::read_to_string(snapshot_dir.join(MANIFEST_FILE)).ok().and_then(|t| serde_json::from_str(&t).ok());
    let engine = match parsed.as_ref().map(|p| p.label.as_str()) {
        Some("pre") => manifest.and_then(|m| m.engine_from),
        Some(_) => parsed.map(|p| p.engine),
        None => None,
    };
    match engine {
        Some(engine) => {
            let mut stamp = stamp_for(store, engine_versions(), now);
            stamp.surrealdb_version = engine;
            stamp.surrealkv_version = None;
            if let Err(e) = write_stamp(store, &stamp) {
                log_event(log::Level::Warn, format!("Restore: could not write engine stamp: {e}"));
            }
        }
        None => {
            let _ = fs::remove_file(stamp_path(store));
        }
    }
}

/// Restore over the configured store; errors while the global DB is connected (use stage_restore and restart).
pub async fn restore_snapshot(snapshot_dir: &Path) -> Result<RestoreReport, SnapshotError> {
    if super::is_connected() {
        return Err(SnapshotError::Other("The database is open in this process and its handle cannot be closed. Use stage_restore() and restart the app.".into()));
    }
    let store = PathBuf::from(super::get_db_path());
    restore_snapshot_to(snapshot_dir.to_path_buf(), store, status_progress("Restore")).await
}

/// Restore `snapshot_dir` over an explicit store folder on a blocking thread.
pub async fn restore_snapshot_to(snapshot_dir: PathBuf, store: PathBuf, progress: ProgressFn) -> Result<RestoreReport, SnapshotError> {
    tokio::task::spawn_blocking(move || restore_snapshot_blocking(&snapshot_dir, &store, Local::now(), &available_space, &progress))
        .await
        .map_err(|e| SnapshotError::Other(format!("restore task failed: {e}")))?
}

/// Record a restore to apply the next time the database is opened (before it connects).
pub fn stage_restore(store: &Path, snapshot_dir: &Path) -> io::Result<()> {
    let pending = PendingRestore { snapshot: snapshot_dir.to_path_buf(), requested_rfc3339: Local::now().to_rfc3339() };
    let json = serde_json::to_vec_pretty(&pending).map_err(io::Error::other)?;
    write_atomic(&pending_restore_path(store), &json)?;
    log_event(log::Level::Info, format!("Restore of {} staged; it will be applied the next time the database opens", snapshot_dir.display()));
    Ok(())
}

pub fn pending_restore(store: &Path) -> Option<PendingRestore> {
    serde_json::from_str(&fs::read_to_string(pending_restore_path(store)).ok()?).ok()
}

pub fn clear_pending_restore(store: &Path) {
    let _ = fs::remove_file(pending_restore_path(store));
}

static OVERRIDE_ONCE: AtomicBool = AtomicBool::new(false);
static BLOCKED: Mutex<Option<Blocked>> = Mutex::new(None);
static LAST_OPEN: Mutex<Option<OpenSummary>> = Mutex::new(None);

/// Skip the pre-open snapshot (and its free-space guard) for the next open only.
pub fn override_snapshot_once() {
    OVERRIDE_ONCE.store(true, Ordering::SeqCst);
    log_event(log::Level::Warn, "Pre-open snapshot will be skipped for the next open (user override)");
}

pub fn override_pending() -> bool {
    OVERRIDE_ONCE.load(Ordering::SeqCst)
}

#[derive(Clone, Debug)]
pub struct Blocked {
    pub message: String,
    pub overridable: bool,
}

/// Record why the last open attempt was refused before connecting.
pub fn set_blocked(err: &SnapshotError) {
    let overridable = matches!(err, SnapshotError::InsufficientSpace { .. } | SnapshotError::Io { .. });
    let message = err.user_message();
    log_event(log::Level::Error, format!("Database open blocked: {message}"));
    if let Ok(mut b) = BLOCKED.lock() { *b = Some(Blocked { message, overridable }); }
}

pub fn clear_blocked() {
    if let Ok(mut b) = BLOCKED.lock() { *b = None; }
}

/// Refusal recorded by the last open attempt, if any.
pub fn blocked() -> Option<Blocked> {
    BLOCKED.lock().ok().and_then(|b| b.clone())
}

#[derive(Clone, Debug, Serialize)]
pub struct OpenSummary {
    pub change: EngineChange,
    pub snapshot: Option<PathBuf>,
    pub note: Option<String>,
    pub auto_export: Option<PathBuf>,
    pub restored: Option<PathBuf>,
}

/// Outcome of the pre-open phase, handed to `finish_open` and `spawn_auto_export`.
#[derive(Clone, Debug)]
pub struct PreOpen {
    pub change: EngineChange,
    pub snapshot: Option<SnapshotInfo>,
    pub note: Option<String>,
    pub settings: MaintenanceSettings,
    pub restored: Option<RestoreReport>,
}

pub fn last_open_summary() -> Option<OpenSummary> {
    LAST_OPEN.lock().ok().and_then(|g| g.clone())
}

fn record_open(pre: &PreOpen) {
    if let Ok(mut g) = LAST_OPEN.lock() {
        *g = Some(OpenSummary {
            change: pre.change.clone(),
            snapshot: pre.snapshot.as_ref().map(|s| s.path.clone()),
            note: pre.note.clone(),
            auto_export: g.as_ref().and_then(|s| s.auto_export.clone()),
            restored: pre.restored.as_ref().map(|r| r.restored_from.clone()),
        });
    }
}

/// Apply a staged restore, then snapshot the store if the engine changed; call before connecting.
pub async fn prepare_open(store: &Path) -> Result<PreOpen, SnapshotError> {
    let settings = MaintenanceSettings::load(store);
    let mut restored = None;
    if let Some(pending) = pending_restore(store) {
        log_event(log::Level::Info, format!("Applying staged restore from {}", pending.snapshot.display()));
        DB_STATUS.set_state(StatusState::Initializing, "Restoring database snapshot");
        let res = restore_snapshot_to(pending.snapshot.clone(), store.to_path_buf(), status_progress("Restore")).await;
        clear_pending_restore(store);
        match res {
            Ok(r) => restored = Some(r),
            Err(e) => log_event(log::Level::Error, format!("Staged restore failed, opening the existing store: {}", e.user_message())),
        }
    }
    let mut pre = prepare_open_with(store, settings, Arc::new(available_space), Local::now(), status_progress("Snapshot")).await?;
    pre.restored = restored;
    DB_STATUS.set_progress(0, 0);
    record_open(&pre);
    Ok(pre)
}

/// Pre-open flow with injectable free-space probe, clock and progress sink.
pub async fn prepare_open_with(
    store: &Path,
    settings: MaintenanceSettings,
    space: Arc<SpaceFn>,
    now: DateTime<Local>,
    progress: ProgressFn,
) -> Result<PreOpen, SnapshotError> {
    let override_requested = OVERRIDE_ONCE.swap(false, Ordering::SeqCst);
    let has_data = store_has_data(store);
    let stamp = read_stamp(store);
    let change = classify(store, has_data, &stamp, engine_versions());
    let mut pre = PreOpen { change: change.clone(), snapshot: None, note: None, settings: settings.clone(), restored: None };
    let EngineChange::Changed { from, to, reason } = &change else {
        return Ok(pre);
    };
    log_event(log::Level::Info, format!("Engine change detected for {}: {reason}", store.display()));
    if !settings.snapshot_on_engine_change {
        pre.note = Some("pre-open snapshot disabled in settings".into());
        log_event(log::Level::Warn, "Pre-open snapshot disabled in settings; opening without a backup");
        return Ok(pre);
    }
    if override_requested {
        pre.note = Some("pre-open snapshot skipped by user override".into());
        log_event(log::Level::Warn, "Pre-open snapshot skipped by user override; opening without a backup");
        return Ok(pre);
    }
    if store_in_use(store) {
        return Err(SnapshotError::InUse(store.to_path_buf()));
    }
    DB_STATUS.set_state(StatusState::Initializing, format!("Snapshotting database before engine upgrade to {to}"));
    let store_buf = store.to_path_buf();
    let engine_from = from.clone();
    let handle = tokio::task::spawn_blocking(move || create_snapshot_blocking(&store_buf, "pre", engine_from, now, space.as_ref(), &progress));
    let info = handle.await.map_err(|e| SnapshotError::Other(format!("snapshot task failed: {e}")))??;
    pre.snapshot = Some(info);
    let report = prune_snapshots(store, &settings, now);
    if !report.deleted.is_empty() {
        log_event(log::Level::Info, format!("Retention removed {} old snapshot(s)", report.deleted.len()));
    }
    Ok(pre)
}

/// Record the running engine in the stamp; call only after connect and schema define succeeded.
pub fn finish_open(store: &Path) {
    let stamp = stamp_for(store, engine_versions(), Local::now());
    match write_stamp(store, &stamp) {
        Ok(()) => log::info!("[DB] engine stamp updated: surrealdb {}", stamp.surrealdb_version),
        Err(e) => log_event(log::Level::Warn, format!("Could not write engine stamp {}: {e}", stamp_path(store).display())),
    }
}

#[derive(Clone, Debug, Serialize)]
pub enum AutoExport {
    Skipped(String),
    Exported(PathBuf, u64),
    Failed(String),
}

/// Export the open DB to `<snapshots>/<store>-post-<engine>-<ts>.surql` when the engine changed and the store is small enough.
pub async fn auto_export_if_changed(db: &surrealdb::Surreal<surrealdb::engine::local::Db>, store: &Path, pre: &PreOpen) -> AutoExport {
    if !pre.change.is_changed() {
        return AutoExport::Skipped("engine unchanged".into());
    }
    if !pre.settings.auto_export_on_engine_change {
        return AutoExport::Skipped("auto export disabled in settings".into());
    }
    let size_store = store.to_path_buf();
    let size = tokio::task::spawn_blocking(move || dir_size(&size_store)).await.unwrap_or(0);
    if size > pre.settings.export_size_threshold_bytes {
        let msg = format!("store is {} which exceeds the auto-export threshold of {}", human(size), human(pre.settings.export_size_threshold_bytes));
        log_event(log::Level::Info, format!("Auto export skipped: {msg}"));
        return AutoExport::Skipped(msg);
    }
    let dir = snapshots_dir(store);
    if let Err(e) = fs::create_dir_all(&dir) {
        log_event(log::Level::Error, format!("Auto export skipped: cannot create {}: {e}", dir.display()));
        return AutoExport::Failed(format!("cannot create {}: {e}", dir.display()));
    }
    if let Err(e) = check_free_space(size, &dir, &available_space) {
        log_event(log::Level::Warn, format!("Auto export skipped: {}", e.user_message()));
        return AutoExport::Skipped(e.user_message());
    }
    let base = snapshot_file_name(&store_name(store), "post", &engine_versions().surrealdb, Local::now());
    let path = unique_path(&dir, &base, ".surql");
    log_event(log::Level::Info, format!("Auto export started: {}", path.display()));
    let _ga = super::db_activity("Auto export DB");
    match super::surql::export_atomic(db, &path, &status_progress("Export")).await {
        Ok(rep) => {
            log_event(log::Level::Info, format!("Auto export done: {} ({} in {:.1}s)", path.display(), human(rep.bytes), rep.millis as f32 / 1000.0));
            if let Ok(mut g) = LAST_OPEN.lock() {
                if let Some(s) = g.as_mut() { s.auto_export = Some(path.clone()); }
            }
            prune_snapshots(store, &pre.settings, Local::now());
            AutoExport::Exported(path, rep.bytes)
        }
        Err(e) => {
            log_event(log::Level::Error, format!("Auto export failed: {e}"));
            AutoExport::Failed(e.to_string())
        }
    }
}

/// Run the post-upgrade auto export on the global DB in the background.
pub fn spawn_auto_export(store: PathBuf, pre: PreOpen) {
    if !pre.change.is_changed() {
        return;
    }
    let Ok(handle) = tokio::runtime::Handle::try_current() else { return };
    handle.spawn(async move {
        tokio::time::sleep(AUTO_EXPORT_DELAY).await;
        let _ = auto_export_if_changed(&super::DB, &store, &pre).await;
    });
}
