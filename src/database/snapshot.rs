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

use crate::status::{DB_STATUS, GlobalStatusIndicator, StatusState};

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
    pub export_include_thumb_blobs: bool,
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
            export_include_thumb_blobs: false,
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

pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
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
    regex::Regex::new(r"^(?P<store>.+)-(?P<label>pre|post|manual|export|thumbs|collections)-(?P<engine>[0-9][0-9A-Za-z.+_-]*?)-(?P<date>\d{8})(?:-(?P<time>\d{6}))?(?:-(?P<n>\d+))?$").unwrap()
});

pub fn snapshot_file_name(store_name: &str, label: &str, engine: &str, now: DateTime<Local>) -> String {
    format!("{store_name}-{label}-{engine}-{}", now.format(NAME_TS_FORMAT))
}

/// Parse `<store>-<pre|post|manual|export|thumbs|collections>-<engine>-<yyyymmdd[-hhmmss]>` snapshot names.
pub fn parse_snapshot_name(name: &str) -> Option<ParsedName> {
    let stem = name.strip_suffix(".surql").unwrap_or(name);
    let c = NAME_RE.captures(stem)?;
    let date = NaiveDate::parse_from_str(&c["date"], "%Y%m%d").ok();
    let created = match (date, c.name("time")) {
        (Some(_), Some(t)) => NaiveDateTime::parse_from_str(&format!("{}-{}", &c["date"], t.as_str()), NAME_TS_FORMAT)
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

/// Manifest of a directory snapshot, if present and readable.
pub fn read_manifest(dir: &Path) -> Option<SnapshotManifest> {
    fs::read_to_string(dir.join(MANIFEST_FILE)).ok().and_then(|t| serde_json::from_str(&t).ok())
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

mod retention;
pub use retention::*;

mod restore;
pub use restore::*;

mod open;
pub use open::*;
