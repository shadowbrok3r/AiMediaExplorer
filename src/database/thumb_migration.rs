//! Resumable migration of inline `thumbnail_b64` data into the `thumb_blobs` table.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, anyhow};
use chrono::{DateTime, Local, Utc};
use serde::{Deserialize, Serialize};
use surrealdb::Surreal;
use surrealdb::engine::local::Db;
use surrealdb::types::{Datetime, RecordId, SurrealValue};

use super::snapshot::{self, ProgressFn, db_parent, human, log_event};
use super::thumb_blobs::{self, ThumbBlobIn};
use super::{db_activity, db_set_detail, db_set_progress};

pub const STATE_FILE: &str = "thumb-migration.json";
pub const BATCH: usize = 64;
pub const FRESH_SNAPSHOT_AGE: Duration = Duration::from_secs(3600);
pub const SNAPSHOT_LABEL: &str = "thumbs";
const MAX_RECORDED_FAILURES: usize = 200;
const SNAPSHOT_ATTEMPTS: u32 = 3;
const MISSING_SCAN_BATCH: usize = 500;
const SAMPLE_CAP: usize = 4096;

pub fn state_path(store: &Path) -> PathBuf {
    db_parent(store).join(STATE_FILE)
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct FailedRow {
    pub path: String,
    pub reason: String,
}

/// Persisted migration progress; rollback is restoring the directory snapshot named in `snapshot`.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct MigrationState {
    pub started: Option<String>,
    pub last_key: Option<String>,
    pub migrated: u64,
    pub skipped_existing: u64,
    pub failed: u64,
    pub rows_total: Option<u64>,
    pub copy_done: bool,
    pub snapshot: Option<String>,
    pub cleared: u64,
    pub finished: Option<String>,
    pub failures: Vec<FailedRow>,
}

impl MigrationState {
    pub fn load(store: &Path) -> Self {
        match std::fs::read_to_string(state_path(store)) {
            Ok(s) => serde_json::from_str(&s).unwrap_or_else(|e| {
                log::warn!("[thumb-migration] state file unreadable ({e}); starting from scratch");
                Self::default()
            }),
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self, store: &Path) -> std::io::Result<()> {
        let json = serde_json::to_vec_pretty(self).map_err(std::io::Error::other)?;
        snapshot::write_atomic(&state_path(store), &json)
    }

    pub fn is_finished(&self) -> bool {
        self.finished.is_some()
    }
}

/// Removes the state file; used after a snapshot restore and by the devtools reset.
pub fn reset_state(store: &Path) {
    if std::fs::remove_file(state_path(store)).is_ok() {
        log_event(log::Level::Info, "Thumbnail migration state reset");
    }
    refresh_inline_flag();
}

static INLINE: AtomicU8 = AtomicU8::new(0);

/// True while rows of the configured store may still carry inline `thumbnail_b64` data.
pub fn inline_may_exist() -> bool {
    match INLINE.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let finished = MigrationState::load(Path::new(&super::get_db_path())).is_finished();
            INLINE.store(if finished { 2 } else { 1 }, Ordering::Relaxed);
            !finished
        }
    }
}

/// Forget the cached answer of `inline_may_exist` so it is read from the state file again.
pub fn refresh_inline_flag() {
    INLINE.store(0, Ordering::Relaxed);
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Phase {
    #[default]
    Idle,
    Snapshot,
    Counting,
    Copying,
    Paused,
    CopyDone,
    Verifying,
    Clearing,
    Finished,
    Failed,
}

impl Phase {
    pub fn label(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Snapshot => "taking snapshot",
            Self::Counting => "counting thumbnail rows",
            Self::Copying => "copying",
            Self::Paused => "paused",
            Self::CopyDone => "copy complete",
            Self::Verifying => "verifying",
            Self::Clearing => "clearing inline data",
            Self::Finished => "finished",
            Self::Failed => "failed",
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MigrationProgress {
    pub phase: Phase,
    pub message: String,
    pub rows_total: Option<u64>,
    pub processed: u64,
    pub migrated: u64,
    pub skipped_existing: u64,
    pub failed: u64,
    pub cleared: u64,
    pub bytes_in: u64,
    pub bytes_out: u64,
    pub started: Option<Instant>,
    pub ended: Option<Instant>,
    pub error: Option<String>,
}

impl MigrationProgress {
    pub fn rows_per_sec(&self) -> f64 {
        let Some(start) = self.started else { return 0.0 };
        let secs = self.ended.unwrap_or_else(Instant::now).duration_since(start).as_secs_f64();
        if secs > 0.05 { self.processed as f64 / secs } else { 0.0 }
    }

    pub fn fraction(&self) -> Option<f32> {
        let total = self.rows_total.filter(|t| *t > 0)?;
        Some((self.processed as f32 / total as f32).clamp(0.0, 1.0))
    }

    pub fn eta(&self) -> Option<Duration> {
        let total = self.rows_total?;
        let rate = self.rows_per_sec();
        (rate > 0.0 && total > self.processed).then(|| Duration::from_secs_f64((total - self.processed) as f64 / rate))
    }
}

#[derive(Clone, Debug)]
pub struct CopyOptions {
    pub require_snapshot: bool,
    pub restart: bool,
    pub batch: usize,
    pub workers: usize,
    pub max_batches: Option<u64>,
}

impl Default for CopyOptions {
    fn default() -> Self {
        let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2);
        Self { require_snapshot: true, restart: false, batch: BATCH, workers: (cpus / 2).clamp(1, 4), max_batches: None }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct CopyReport {
    pub processed: u64,
    pub migrated: u64,
    pub skipped_existing: u64,
    pub failed: u64,
    pub bytes_in: u64,
    pub bytes_out: u64,
    pub millis: u128,
    pub cancelled: bool,
    pub snapshot: Option<String>,
}

#[derive(Clone, Debug)]
pub struct FinishOptions {
    pub require_snapshot: bool,
    pub clear_undecodable: bool,
    pub sample: usize,
    pub batch: usize,
}

impl Default for FinishOptions {
    fn default() -> Self {
        Self { require_snapshot: true, clear_undecodable: false, sample: 200, batch: BATCH }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct FinishReport {
    pub ok: bool,
    pub legacy_rows: u64,
    pub covered: u64,
    pub repaired: u64,
    pub undecodable: Vec<FailedRow>,
    pub undecodable_count: u64,
    pub sample_checked: u64,
    pub sample_failures: Vec<FailedRow>,
    pub cleared: u64,
    pub remaining: u64,
    pub summary: String,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct MissingReport {
    pub scanned: u64,
    pub missing: u64,
    pub paths: Vec<String>,
    pub truncated: bool,
}

#[derive(Debug, SurrealValue)]
struct LegacyRow {
    id: RecordId,
    path: String,
    thumbnail_b64: Option<String>,
    modified: Option<Datetime>,
    size: Option<i64>,
}

#[derive(Debug, SurrealValue)]
struct CountRow {
    count: i64,
}

struct BatchOutcome {
    migrated: u64,
    skipped: u64,
    bad: Vec<FailedRow>,
    bytes_in: u64,
    bytes_out: u64,
}

struct Shared {
    progress: Mutex<MigrationProgress>,
    pause: AtomicBool,
    cancel: AtomicBool,
    running: AtomicBool,
}

struct RunGuard<'a>(&'a Shared);

impl Drop for RunGuard<'_> {
    fn drop(&mut self) {
        self.0.running.store(false, Ordering::SeqCst);
        self.0.pause.store(false, Ordering::SeqCst);
    }
}

pub struct Migrator {
    handle: Option<Surreal<Db>>,
    store: PathBuf,
    shared: Shared,
}

static GLOBAL: LazyLock<Arc<Migrator>> = LazyLock::new(|| Migrator::build(None, PathBuf::from(super::get_db_path())));

impl Migrator {
    /// Migrator on a private handle (selected namespace and database required).
    pub fn new(db: Surreal<Db>, store: PathBuf) -> Arc<Self> {
        Self::build(Some(db), store)
    }

    fn build(handle: Option<Surreal<Db>>, store: PathBuf) -> Arc<Self> {
        Arc::new(Self {
            handle,
            store,
            shared: Shared {
                progress: Mutex::new(MigrationProgress::default()),
                pause: AtomicBool::new(false),
                cancel: AtomicBool::new(false),
                running: AtomicBool::new(false),
            },
        })
    }

    /// Migrator bound to the global database and configured store.
    pub fn global() -> Arc<Self> {
        Arc::clone(&GLOBAL)
    }

    pub fn store(&self) -> &Path {
        &self.store
    }

    fn db(&self) -> &Surreal<Db> {
        self.handle.as_ref().unwrap_or(&super::DB)
    }

    pub fn progress(&self) -> MigrationProgress {
        self.shared.progress.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub fn state(&self) -> MigrationState {
        MigrationState::load(&self.store)
    }

    pub fn is_running(&self) -> bool {
        self.shared.running.load(Ordering::SeqCst)
    }

    pub fn is_paused(&self) -> bool {
        self.shared.pause.load(Ordering::SeqCst)
    }

    pub fn pause(&self) {
        if self.is_running() {
            self.shared.pause.store(true, Ordering::SeqCst);
        }
    }

    pub fn resume(&self) {
        self.shared.pause.store(false, Ordering::SeqCst);
    }

    pub fn cancel(&self) {
        if self.is_running() {
            self.shared.cancel.store(true, Ordering::SeqCst);
        }
    }

    fn update(&self, f: impl FnOnce(&mut MigrationProgress)) {
        f(&mut self.shared.progress.lock().unwrap_or_else(|e| e.into_inner()));
    }

    fn set_phase(&self, phase: Phase, message: impl Into<String>) {
        let message = message.into();
        self.update(|p| {
            p.phase = phase;
            p.message = message;
        });
    }

    /// Marks a step as running; the step's `RunGuard` releases it.
    fn claim(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.shared.running.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_ok(),
            "a thumbnail migration step is already running"
        );
        self.shared.cancel.store(false, Ordering::SeqCst);
        self.shared.pause.store(false, Ordering::SeqCst);
        self.update(|p| *p = MigrationProgress { started: Some(Instant::now()), ..Default::default() });
        Ok(())
    }

    fn fail(&self, e: &anyhow::Error) {
        let msg = format!("{e:#}");
        log_event(log::Level::Error, format!("Thumbnail migration failed: {msg}"));
        self.update(|p| {
            p.phase = Phase::Failed;
            p.error = Some(msg.clone());
            p.message = msg;
        });
    }

    /// Waits while paused; true when the run was cancelled.
    async fn checkpoint(&self, resume_phase: Phase) -> bool {
        if self.shared.pause.load(Ordering::SeqCst) {
            self.set_phase(Phase::Paused, "paused");
            while self.shared.pause.load(Ordering::SeqCst) && !self.shared.cancel.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(150)).await;
            }
            self.set_phase(resume_phase, "");
        }
        self.shared.cancel.load(Ordering::SeqCst)
    }

    /// Runs the copy phase on the runtime; false when a step is already running.
    pub fn spawn_copy(self: &Arc<Self>, opts: CopyOptions) -> bool {
        if self.claim().is_err() {
            return false;
        }
        let me = Arc::clone(self);
        tokio::spawn(async move {
            if let Err(e) = me.copy_claimed(opts).await {
                log::warn!("[thumb-migration] copy ended: {e:#}");
            }
        });
        true
    }

    /// Runs the finish step on the runtime; false when a step is already running.
    pub fn spawn_finish(self: &Arc<Self>, opts: FinishOptions) -> bool {
        if self.claim().is_err() {
            return false;
        }
        let me = Arc::clone(self);
        tokio::spawn(async move {
            if let Err(e) = me.finish_claimed(opts).await {
                log::warn!("[thumb-migration] finish ended: {e:#}");
            }
        });
        true
    }

    /// Number of rows that still carry an inline thumbnail (reads every row).
    pub async fn count_legacy(&self) -> anyhow::Result<u64> {
        let rows: Vec<CountRow> = self.db().query("SELECT count() AS count FROM thumbnails WHERE thumbnail_b64 != NONE GROUP ALL").await?.take(0)?;
        Ok(rows.first().map(|r| r.count.max(0) as u64).unwrap_or(0))
    }

    /// Number of rows in the thumbnails table (key count only).
    pub async fn count_rows(&self) -> anyhow::Result<u64> {
        let rows: Vec<CountRow> = self.db().query("SELECT count() AS count FROM thumbnails GROUP ALL").await?.take(0)?;
        Ok(rows.first().map(|r| r.count.max(0) as u64).unwrap_or(0))
    }

    async fn legacy_batch(&self, after: Option<&str>, n: usize, with_data: bool) -> anyhow::Result<Vec<LegacyRow>> {
        let cols = if with_data { "id, path, thumbnail_b64, modified, size" } else { "id, path, modified, size" };
        let filter = if after.is_some() { "path > $after AND thumbnail_b64 != NONE" } else { "thumbnail_b64 != NONE" };
        let mut q = self.db().query(format!("SELECT {cols} FROM thumbnails WHERE {filter} ORDER BY path LIMIT $n")).bind(("n", n as i64));
        if let Some(k) = after {
            q = q.bind(("after", k.to_string()));
        }
        Ok(q.await?.take(0)?)
    }

    fn recent_snapshot_blocking(store: &Path, now: DateTime<Local>) -> Option<snapshot::SnapshotInfo> {
        let me = snapshot::store_name(store);
        snapshot::list_snapshots_for(store).into_iter().find(|s| {
            s.kind == snapshot::SnapshotKind::Dir
                && s.has_manifest
                && s.store_name.as_deref().is_none_or(|n| n == me)
                && s.created.is_some_and(|c| now.signed_duration_since(c).to_std().is_ok_and(|age| age <= FRESH_SNAPSHOT_AGE))
                && snapshot::read_manifest(&s.path).is_some_and(|m| m.consistent)
        })
    }

    /// Path of a consistent directory snapshot taken within the last hour, creating one when none exists.
    pub async fn ensure_snapshot(&self, progress: ProgressFn) -> anyhow::Result<String> {
        let store = self.store.clone();
        let recent = tokio::task::spawn_blocking(move || Self::recent_snapshot_blocking(&store, Local::now())).await?;
        if let Some(info) = recent {
            log_event(log::Level::Info, format!("Thumbnail migration: using snapshot {} taken within the last hour", info.path.display()));
            return Ok(absolute(&info.path));
        }
        let mut last_err = String::new();
        for attempt in 1..=SNAPSHOT_ATTEMPTS {
            log_event(log::Level::Info, format!("Thumbnail migration: taking a snapshot of {} (attempt {attempt}/{SNAPSHOT_ATTEMPTS})", self.store.display()));
            let info = snapshot::create_snapshot(self.store.clone(), SNAPSHOT_LABEL, None, progress.clone()).await.map_err(|e| anyhow!(e.user_message()))?;
            if snapshot::read_manifest(&info.path).is_some_and(|m| m.consistent) {
                return Ok(absolute(&info.path));
            }
            last_err = format!("snapshot {} was taken while the database was changing", info.path.display());
            let _ = snapshot::delete_snapshot(&self.store, &info.path);
        }
        anyhow::bail!("no consistent snapshot could be taken ({last_err}); stop scans and other database activity, then retry")
    }

    /// Converts one batch of legacy rows into blobs, skipping rows that already have one.
    async fn copy_batch(&self, rows: Vec<LegacyRow>, pool: &Arc<rayon::ThreadPool>) -> anyhow::Result<BatchOutcome> {
        let keys: Vec<String> = rows.iter().map(|r| thumb_blobs::key_for_path(&r.path)).collect();
        let existing = thumb_blobs::exists_many_with(self.db(), &keys).await?;
        let (todo, skipped): (Vec<(LegacyRow, String)>, Vec<(LegacyRow, String)>) = rows.into_iter().zip(keys).partition(|(_, k)| !existing.contains(k));
        let bytes_in: u64 = todo.iter().map(|(r, _)| r.thumbnail_b64.as_ref().map_or(0, |s| s.len() as u64)).sum();
        let pool = Arc::clone(pool);
        let converted = tokio::task::spawn_blocking(move || {
            use rayon::prelude::*;
            pool.install(|| todo.into_par_iter().map(|(row, _)| convert_row(row)).collect::<Vec<_>>())
        })
        .await?;
        let mut ok = Vec::with_capacity(converted.len());
        let mut bad = Vec::new();
        for c in converted {
            match c {
                Ok(b) => ok.push(b),
                Err(f) => bad.push(f),
            }
        }
        let bytes_out: u64 = ok.iter().map(|b| b.bytes.len() as u64).sum();
        let attempted = ok.len() as u64;
        let migrated = thumb_blobs::put_if_absent_with(self.db(), ok).await? as u64;
        Ok(BatchOutcome { migrated, skipped: skipped.len() as u64 + attempted.saturating_sub(migrated), bad, bytes_in, bytes_out })
    }

    /// Copy phase: converts every inline thumbnail into a blob, never touching the originals.
    pub async fn run_copy(&self, opts: CopyOptions) -> anyhow::Result<CopyReport> {
        self.claim()?;
        self.copy_claimed(opts).await
    }

    async fn copy_claimed(&self, opts: CopyOptions) -> anyhow::Result<CopyReport> {
        let _run = RunGuard(&self.shared);
        let result = self.copy_inner(opts).await;
        if let Err(e) = &result {
            self.fail(e);
        }
        self.update(|p| p.ended = Some(Instant::now()));
        result
    }

    async fn copy_inner(&self, opts: CopyOptions) -> anyhow::Result<CopyReport> {
        let started = Instant::now();
        let mut state = if opts.restart { MigrationState { snapshot: self.state().snapshot, ..Default::default() } } else { self.state() };
        let mut report = CopyReport { snapshot: state.snapshot.clone(), ..Default::default() };
        let status_progress = snapshot::status_progress("Snapshot");
        if opts.require_snapshot {
            self.set_phase(Phase::Snapshot, "taking a snapshot of the store");
            db_set_detail("Thumbnail migration: snapshot");
            let snap = self.ensure_snapshot(status_progress).await?;
            state.snapshot = Some(snap.clone());
            report.snapshot = Some(snap);
        }
        if state.started.is_none() {
            state.started = Some(Local::now().to_rfc3339());
        }
        state.finished = None;
        state.copy_done = false;
        state.save(&self.store).context("cannot write migration state")?;
        refresh_inline_flag();

        let rows_total = match state.rows_total {
            Some(n) => Some(n),
            None => {
                self.set_phase(Phase::Counting, "counting thumbnail rows");
                let n = self.count_rows().await.ok();
                state.rows_total = n;
                state.save(&self.store).context("cannot write migration state")?;
                n
            }
        };
        let already = state.migrated + state.skipped_existing + state.failed;
        self.update(|p| {
            p.rows_total = rows_total;
            p.processed = 0;
            p.migrated = state.migrated;
            p.skipped_existing = state.skipped_existing;
            p.failed = state.failed;
        });
        log_event(log::Level::Info, format!("Thumbnail migration: copying inline thumbnails of {} rows (resuming after {} already handled)", rows_total.unwrap_or(0), already));

        let pool = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(opts.workers.max(1)).thread_name(|i| format!("thumb-migrate-{i}")).build()?);
        self.set_phase(Phase::Copying, "");
        let batch = opts.batch.max(1);
        let mut batches = 0u64;
        let mut prefetched: Option<Vec<LegacyRow>> = None;
        loop {
            if self.checkpoint(Phase::Copying).await || opts.max_batches.is_some_and(|m| batches >= m) {
                report.cancelled = true;
                break;
            }
            batches += 1;
            let _ga = db_activity("Migrating thumbnails");
            let rows = match prefetched.take() {
                Some(r) => r,
                None => self.legacy_batch(state.last_key.as_deref(), batch, true).await?,
            };
            let Some(last) = rows.last() else { break };
            let last_key = last.path.clone();
            let (done, upcoming) = tokio::join!(self.copy_batch(rows, &pool), self.legacy_batch(Some(&last_key), batch, true));
            let out = done?;

            let handled = out.migrated + out.skipped + out.bad.len() as u64;
            state.migrated += out.migrated;
            state.skipped_existing += out.skipped;
            state.failed += out.bad.len() as u64;
            for f in &out.bad {
                if state.failures.len() < MAX_RECORDED_FAILURES {
                    state.failures.push(f.clone());
                }
            }
            state.last_key = Some(last_key);
            state.save(&self.store).context("cannot write migration state")?;

            report.processed += handled;
            report.migrated += out.migrated;
            report.skipped_existing += out.skipped;
            report.failed += out.bad.len() as u64;
            report.bytes_in += out.bytes_in;
            report.bytes_out += out.bytes_out;
            self.update(|p| {
                p.processed += handled;
                p.migrated = state.migrated;
                p.skipped_existing = state.skipped_existing;
                p.failed = state.failed;
                p.bytes_in += out.bytes_in;
                p.bytes_out += out.bytes_out;
            });
            if let Some(total) = rows_total.filter(|t| *t > 0) {
                db_set_progress((state.migrated + state.skipped_existing + state.failed).min(total), total);
            }
            db_set_detail(format!("Thumbnail migration: {} migrated, {} failed", state.migrated, state.failed));
            prefetched = Some(upcoming?);
        }

        if !report.cancelled {
            state.copy_done = true;
            state.save(&self.store).context("cannot write migration state")?;
        }
        db_set_progress(0, 0);
        report.millis = started.elapsed().as_millis();
        let (label, phase, message) = if report.cancelled { ("stopped", Phase::Idle, "stopped; progress is kept") } else { ("complete", Phase::CopyDone, "") };
        self.set_phase(phase, message);
        log_event(
            log::Level::Info,
            format!(
                "Thumbnail migration copy {label}: {} migrated, {} already present, {} failed, {} -> {} in {:.1}s",
                report.migrated,
                report.skipped_existing,
                report.failed,
                human(report.bytes_in),
                human(report.bytes_out),
                report.millis as f32 / 1000.0
            ),
        );
        Ok(report)
    }

    /// Finish step: verify coverage, sample-decode blobs, then clear `thumbnail_b64` in batches.
    pub async fn run_finish(&self, opts: FinishOptions) -> anyhow::Result<FinishReport> {
        self.claim()?;
        self.finish_claimed(opts).await
    }

    async fn finish_claimed(&self, opts: FinishOptions) -> anyhow::Result<FinishReport> {
        let _run = RunGuard(&self.shared);
        let result = self.finish_inner(opts).await;
        match &result {
            Ok(r) if !r.ok => self.update(|p| {
                p.phase = Phase::CopyDone;
                p.message = r.summary.clone();
            }),
            Err(e) => self.fail(e),
            _ => {}
        }
        self.update(|p| p.ended = Some(Instant::now()));
        result
    }

    async fn finish_inner(&self, opts: FinishOptions) -> anyhow::Result<FinishReport> {
        let mut state = self.state();
        anyhow::ensure!(state.copy_done, "the copy phase has not completed; run it first");
        if opts.require_snapshot {
            let have_snapshot = state.snapshot.as_ref().is_some_and(|p| Path::new(p).is_dir());
            if !have_snapshot {
                self.set_phase(Phase::Snapshot, "taking a snapshot of the store");
                state.snapshot = Some(self.ensure_snapshot(snapshot::status_progress("Snapshot")).await?);
                state.save(&self.store)?;
            }
        }
        let mut report = FinishReport::default();

        self.set_phase(Phase::Verifying, "checking that every legacy row has a blob");
        let mut after: Option<String> = None;
        let mut sample: Vec<String> = Vec::new();
        let mut seen_covered = 0u64;
        let mut rng = rand_seed();
        loop {
            if self.checkpoint(Phase::Verifying).await {
                self.set_phase(Phase::CopyDone, "finish cancelled before any data was cleared");
                report.summary = "cancelled before any data was cleared".into();
                return Ok(report);
            }
            let _ga = db_activity("Verifying thumbnail migration");
            let rows = self.legacy_batch(after.as_deref(), opts.batch.max(1), false).await?;
            let Some(last) = rows.last() else { break };
            after = Some(last.path.clone());
            let keys: Vec<String> = rows.iter().map(|r| thumb_blobs::key_for_path(&r.path)).collect();
            let existing = thumb_blobs::exists_many_with(self.db(), &keys).await?;
            report.legacy_rows += rows.len() as u64;
            for (row, key) in rows.into_iter().zip(keys) {
                if existing.contains(&key) {
                    report.covered += 1;
                    seen_covered += 1;
                    reservoir_push(&mut sample, key, seen_covered, opts.sample.min(SAMPLE_CAP), &mut rng);
                    continue;
                }
                match self.repair_row(&row).await {
                    Ok(()) => {
                        report.repaired += 1;
                        report.covered += 1;
                        seen_covered += 1;
                        reservoir_push(&mut sample, key, seen_covered, opts.sample.min(SAMPLE_CAP), &mut rng);
                    }
                    Err(reason) => {
                        report.undecodable_count += 1;
                        if report.undecodable.len() < MAX_RECORDED_FAILURES {
                            report.undecodable.push(FailedRow { path: row.path, reason });
                        }
                    }
                }
            }
            self.update(|p| {
                p.processed = report.legacy_rows;
                p.message = format!("{} rows checked", report.legacy_rows);
            });
        }

        if !sample.is_empty() {
            let blobs = thumb_blobs::get_many_with(self.db(), &sample).await?;
            for key in &sample {
                report.sample_checked += 1;
                match blobs.get(key) {
                    None => report.sample_failures.push(FailedRow { path: key.clone(), reason: "blob vanished during verification".into() }),
                    Some(b) => {
                        if let Err(e) = thumb_blobs::verify_blob(b) {
                            report.sample_failures.push(FailedRow { path: b.path.clone(), reason: format!("{e:#}") });
                        }
                    }
                }
            }
        }

        let undecodable_blocks = report.undecodable_count > 0 && !opts.clear_undecodable;
        if !report.sample_failures.is_empty() || undecodable_blocks {
            report.summary = if !report.sample_failures.is_empty() {
                format!("{} of {} sampled blobs failed to decode; nothing was cleared", report.sample_failures.len(), report.sample_checked)
            } else {
                format!("{} legacy rows have no blob and cannot be decoded; nothing was cleared (enable clearing undecodable data to proceed, or rebuild them from the source files)", report.undecodable_count)
            };
            log_event(log::Level::Warn, format!("Thumbnail migration finish refused: {}", report.summary));
            return Ok(report);
        }

        self.set_phase(Phase::Clearing, "clearing inline thumbnail data");
        let undecodable_keys: HashSet<String> = report.undecodable.iter().map(|f| thumb_blobs::key_for_path(&f.path)).collect();
        let mut after: Option<String> = None;
        let mut cleared = 0u64;
        loop {
            if self.checkpoint(Phase::Clearing).await {
                break;
            }
            let _ga = db_activity("Clearing inline thumbnails");
            let rows = self.legacy_batch(after.as_deref(), opts.batch.max(1), false).await?;
            let Some(last) = rows.last() else { break };
            after = Some(last.path.clone());
            let keys: Vec<String> = rows.iter().map(|r| thumb_blobs::key_for_path(&r.path)).collect();
            let existing = thumb_blobs::exists_many_with(self.db(), &keys).await?;
            let ids: Vec<RecordId> = rows
                .iter()
                .zip(&keys)
                .filter(|(_, k)| existing.contains(*k) || (opts.clear_undecodable && undecodable_keys.contains(*k)))
                .map(|(r, _)| r.id.clone())
                .collect();
            if !ids.is_empty() {
                thumb_blobs::retry_conflicts(|| async {
                    self.db().query("UPDATE $ids SET thumbnail_b64 = NONE RETURN NONE").bind(("ids", ids.clone())).await?.check()?;
                    Ok(())
                })
                .await?;
                cleared += ids.len() as u64;
                state.cleared += ids.len() as u64;
                state.save(&self.store)?;
            }
            self.update(|p| {
                p.cleared = cleared;
                p.message = format!("{cleared} rows cleared");
            });
            db_set_detail(format!("Thumbnail migration: {cleared} inline thumbnails cleared"));
        }
        report.cleared = cleared;
        report.remaining = self.count_legacy().await?;
        report.ok = report.remaining == 0;
        if report.ok {
            state.finished = Some(Local::now().to_rfc3339());
            state.save(&self.store)?;
            INLINE.store(2, Ordering::Relaxed);
            self.set_phase(Phase::Finished, format!("{cleared} inline thumbnails cleared"));
            report.summary = format!("migration finished: {cleared} inline thumbnails cleared, {} repaired during verification", report.repaired);
        } else {
            self.set_phase(Phase::CopyDone, format!("{} rows still carry inline data", report.remaining));
            report.summary = format!("{} rows still carry inline data after clearing; run finish again", report.remaining);
        }
        log_event(if report.ok { log::Level::Info } else { log::Level::Warn }, format!("Thumbnail migration finish: {}", report.summary));
        Ok(report)
    }

    async fn repair_row(&self, row: &LegacyRow) -> Result<(), String> {
        let inline: Option<String> = self
            .db()
            .query("SELECT VALUE thumbnail_b64 FROM $id")
            .bind(("id", row.id.clone()))
            .await
            .and_then(|mut r| r.take(0))
            .map_err(|e| format!("read failed: {e}"))?;
        let full = LegacyRow { id: row.id.clone(), path: row.path.clone(), thumbnail_b64: inline, modified: row.modified.clone(), size: row.size };
        let blob = convert_row(full).map_err(|f| f.reason)?;
        thumb_blobs::put_if_absent_with(self.db(), vec![blob]).await.map_err(|e| format!("write failed: {e:#}"))?;
        Ok(())
    }

    /// Paths of thumbnail rows that have no blob, for the scan/thumb pipeline to rebuild.
    pub async fn missing_paths(&self, limit: usize) -> anyhow::Result<MissingReport> {
        let mut report = MissingReport::default();
        let mut after: Option<String> = None;
        loop {
            let filter = if after.is_some() { "WHERE path > $after" } else { "" };
            let mut q = self.db().query(format!("SELECT id, path FROM thumbnails {filter} ORDER BY path LIMIT $n")).bind(("n", MISSING_SCAN_BATCH as i64));
            if let Some(k) = &after {
                q = q.bind(("after", k.clone()));
            }
            #[derive(Debug, SurrealValue)]
            struct IdPath {
                id: RecordId,
                path: String,
            }
            let rows: Vec<IdPath> = q.await?.take(0)?;
            let Some(last) = rows.last() else { break };
            after = Some(last.path.clone());
            let keys: Vec<String> = rows.iter().map(|r| thumb_blobs::key_for_path(&r.path)).collect();
            let existing = thumb_blobs::exists_many_with(self.db(), &keys).await?;
            report.scanned += rows.len() as u64;
            for (row, key) in rows.into_iter().zip(keys) {
                if existing.contains(&key) {
                    continue;
                }
                report.missing += 1;
                if report.paths.len() < limit {
                    report.paths.push(row.path);
                } else {
                    report.truncated = true;
                }
            }
        }
        Ok(report)
    }
}

fn absolute(path: &Path) -> String {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf()).to_string_lossy().into_owned()
}

/// Decodes one legacy row into a blob entry.
fn convert_row(row: LegacyRow) -> Result<ThumbBlobIn, FailedRow> {
    let fail = |reason: String| FailedRow { path: row.path.clone(), reason };
    let inline = row.thumbnail_b64.as_deref().filter(|s| !s.trim().is_empty()).ok_or_else(|| fail("inline thumbnail is empty".into()))?;
    let blob = ThumbBlobIn::from_thumb_string(row.path.clone(), inline).map_err(|e| fail(format!("{e:#}")))?;
    let modified: Option<DateTime<Utc>> = row.modified.map(Into::into);
    Ok(blob.with_source(modified, row.size.map(|s| s.max(0) as u64)))
}

fn rand_seed() -> u64 {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0x9E37_79B9_7F4A_7C15);
    t ^ 0xA076_1D64_78BD_642F
}

fn next_rand(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Reservoir sampling of `cap` items from a stream where `seen` counts items including this one.
fn reservoir_push(sample: &mut Vec<String>, item: String, seen: u64, cap: usize, rng: &mut u64) {
    if cap == 0 {
        return;
    }
    if sample.len() < cap {
        sample.push(item);
        return;
    }
    let j = next_rand(rng) % seen;
    if (j as usize) < cap {
        sample[j as usize] = item;
    }
}

/// Random offset below `bound` for sampling devtools.
pub fn random_below(bound: u64) -> u64 {
    if bound == 0 {
        return 0;
    }
    let mut seed = rand_seed();
    next_rand(&mut seed) % bound
}
