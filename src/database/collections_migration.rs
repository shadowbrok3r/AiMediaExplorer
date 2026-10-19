//! One-way, snapshot-first, resumable migration of logical groups into collections; the legacy table and links are never changed.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, anyhow};
use chrono::{DateTime, Local};
use egui_theme_kit::store::MAX_NAME_CHARS;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use surrealdb::engine::local::Db;
use surrealdb::types::{Datetime, RecordId, SurrealValue};
use surrealdb::{Connection, Surreal};

use super::collections::{self, Collection, NewCollection, UNFILED_NAME};
use super::snapshot::{self, ProgressFn, log_event};
use super::{db_activity, db_set_detail, db_set_progress};

pub const SNAPSHOT_LABEL: &str = "collections";
pub const FRESH_SNAPSHOT_AGE: Duration = Duration::from_secs(3600);
const STATE_RECORD: &str = "collection_migration:state";
const SNAPSHOT_ATTEMPTS: u32 = 3;
const CHUNK: usize = 1000;
const MAX_LISTED: usize = 200;
const LEGACY_GROUPS: &str = "logical_groups";

/// A legacy group whose name was changed to fit the collection name rules.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct RenamedGroup {
    pub legacy_key: String,
    pub original: String,
    pub assigned: String,
}

/// Where the migration stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    NotStarted,
    Copying,
    Copied,
    Finished,
    RolledBack,
}

impl Phase {
    pub fn label(self) -> &'static str {
        match self {
            Self::NotStarted => "not started",
            Self::Copying => "copy in progress or interrupted",
            Self::Copied => "copied, not finished",
            Self::Finished => "finished",
            Self::RolledBack => "rolled back",
        }
    }
}

/// Persisted migration progress, stored in the database so it travels with exports and snapshots.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct MigrationState {
    pub started: Option<String>,
    pub snapshot: Option<String>,
    pub runs: u32,
    pub groups_done: Vec<String>,
    pub groups_mapped: u64,
    pub collections_created: u64,
    pub memberships_added: u64,
    pub memberships_existing: u64,
    pub unfiled_rows: u64,
    pub dangling_links: u64,
    pub renamed: Vec<RenamedGroup>,
    pub warnings: Vec<String>,
    pub copy_done: bool,
    pub copied_at: Option<String>,
    pub finished: Option<String>,
    pub rolled_back: Option<String>,
    pub needs_resync: bool,
    pub verify: Option<VerifyReport>,
}

impl MigrationState {
    pub fn phase(&self) -> Phase {
        if self.finished.is_some() {
            Phase::Finished
        } else if self.started.is_none() {
            if self.rolled_back.is_some() { Phase::RolledBack } else { Phase::NotStarted }
        } else if self.copy_done {
            Phase::Copied
        } else {
            Phase::Copying
        }
    }
}

/// Reads the state record; absent means the migration never ran.
pub async fn load_state<C: Connection>(db: &Surreal<C>) -> anyhow::Result<MigrationState> {
    let rows: Vec<Value> = db.query(format!("SELECT state FROM {STATE_RECORD}")).await?.take(0)?;
    match rows.first().and_then(|r| r.get("state")).filter(|v| v.is_object()) {
        Some(v) => Ok(serde_json::from_value(v.clone()).unwrap_or_else(|e| {
            log::warn!("[collections] migration state unreadable ({e}); starting from scratch");
            MigrationState::default()
        })),
        None => Ok(MigrationState::default()),
    }
}

pub async fn save_state<C: Connection>(db: &Surreal<C>, state: &MigrationState) -> anyhow::Result<()> {
    db.query(format!("UPSERT {STATE_RECORD} SET state = $state, updated = time::now() RETURN NONE"))
        .bind(json!({ "state": serde_json::to_value(state)? }))
        .await?
        .check()
        .context("cannot write the migration state")?;
    Ok(())
}

/// Marks a finished or copied migration stale after legacy-shaped rows were imported.
pub async fn note_legacy_import<C: Connection>(db: &Surreal<C>) -> anyhow::Result<()> {
    let mut state = load_state(db).await?;
    if state.started.is_some() && !state.needs_resync {
        state.needs_resync = true;
        save_state(db, &state).await?;
    }
    Ok(())
}

#[derive(Debug, SurrealValue)]
struct LegacyGroupRow {
    id: RecordId,
    name: String,
    created: Option<Datetime>,
    updated: Option<Datetime>,
}

#[derive(Debug, SurrealValue)]
struct GroupCountRow {
    logical_group: Option<RecordId>,
    n: i64,
}

/// A legacy group as read from `logical_groups`.
#[derive(Clone, Debug)]
pub struct LegacyGroup {
    pub id: RecordId,
    pub key: String,
    pub name: String,
    pub created: Option<DateTime<chrono::Utc>>,
    pub updated: Option<DateTime<chrono::Utc>>,
}

impl LegacyGroup {
    /// The group that means "filed nowhere".
    pub fn is_unfiled(&self) -> bool {
        self.name == UNFILED_NAME
    }
}

/// Every legacy group.
pub async fn legacy_groups<C: Connection>(db: &Surreal<C>) -> anyhow::Result<Vec<LegacyGroup>> {
    let rows: Vec<LegacyGroupRow> = db.query(format!("SELECT id, name, created, updated FROM {LEGACY_GROUPS}")).await?.take(0)?;
    Ok(rows
        .into_iter()
        .map(|r| LegacyGroup { key: collections::key_of(&r.id), id: r.id, name: r.name, created: r.created.map(Into::into), updated: r.updated.map(Into::into) })
        .collect())
}

/// Rows linked to each legacy group key, plus rows whose link is missing.
pub async fn legacy_link_counts<C: Connection>(db: &Surreal<C>) -> anyhow::Result<(HashMap<String, u64>, u64)> {
    let rows: Vec<GroupCountRow> = db.query("SELECT logical_group, count() AS n FROM thumbnails GROUP BY logical_group").await?.take(0)?;
    let mut out = HashMap::new();
    let mut unlinked = 0;
    for r in rows {
        match r.logical_group {
            Some(g) => *out.entry(collections::key_of(&g)).or_insert(0) += r.n.max(0) as u64,
            None => unlinked += r.n.max(0) as u64,
        }
    }
    Ok((out, unlinked))
}

/// A legacy group to assign a collection name to.
#[derive(Clone, Debug)]
pub struct GroupIn {
    pub key: String,
    pub name: String,
}

/// The collection a legacy group maps to.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Assignment {
    pub legacy_key: String,
    pub legacy_name: String,
    pub name: String,
    /// A collection already exists for this group.
    pub reuse: bool,
    /// The collection name differs from the group name.
    pub renamed: bool,
}

/// A usable collection name from any text: control characters dropped, whitespace collapsed, length capped.
pub fn sanitize_name(name: &str) -> String {
    let cleaned: String = name.chars().filter(|c| !c.is_control() || c.is_whitespace()).collect();
    let collapsed = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
    let capped: String = collapsed.chars().take(MAX_NAME_CHARS).collect();
    let capped = capped.trim_end().to_string();
    if capped.is_empty() { "Untitled".to_string() } else { capped }
}

fn with_suffix(base: &str, n: usize) -> String {
    let suffix = format!(" ({n})");
    let room = MAX_NAME_CHARS.saturating_sub(suffix.chars().count());
    let head: String = base.chars().take(room).collect();
    format!("{}{suffix}", head.trim_end())
}

/// Assigns a distinct, valid collection name to every legacy group except the unfiled one, reusing `existing_by_legacy` and avoiding the `taken` keys.
pub fn assign_names(groups: &[GroupIn], existing_by_legacy: &HashMap<String, String>, taken: &HashSet<String>) -> Vec<Assignment> {
    let mut ordered: Vec<&GroupIn> = groups.iter().filter(|g| g.name != UNFILED_NAME).collect();
    ordered.sort_by(|a, b| (a.name.as_str(), a.key.as_str()).cmp(&(b.name.as_str(), b.key.as_str())));
    let mut used: HashSet<String> = taken.clone();
    let mut out = Vec::with_capacity(ordered.len());
    for g in ordered.iter().filter(|g| existing_by_legacy.contains_key(&g.key)) {
        let name = existing_by_legacy[&g.key].clone();
        out.push(Assignment { legacy_key: g.key.clone(), legacy_name: g.name.clone(), renamed: name != g.name, name, reuse: true });
    }
    for g in ordered.iter().filter(|g| !existing_by_legacy.contains_key(&g.key)) {
        let base = sanitize_name(&g.name);
        let mut candidate = base.clone();
        let mut n = 1;
        let key = loop {
            match collections::normalize(&candidate) {
                Ok((_, key)) if !used.contains(&key) => break key,
                _ => {
                    n += 1;
                    candidate = with_suffix(&base, n);
                }
            }
        };
        used.insert(key);
        out.push(Assignment { legacy_key: g.key.clone(), legacy_name: g.name.clone(), renamed: candidate != g.name, name: candidate, reuse: false });
    }
    out
}

/// One group in a [`Plan`].
#[derive(Clone, Debug, Serialize)]
pub struct PlannedGroup {
    #[serde(flatten)]
    pub assignment: Assignment,
    pub memberships: u64,
}

/// What a copy run would do, computed without writing.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Plan {
    pub groups: Vec<PlannedGroup>,
    pub memberships: u64,
    pub unfiled_rows: u64,
    pub dangling_links: u64,
    pub to_create: usize,
    pub renamed: usize,
    pub warnings: Vec<String>,
}

/// Counts and state for display.
#[derive(Clone, Debug, Serialize)]
pub struct Status {
    pub phase: Phase,
    pub phase_label: &'static str,
    pub state: MigrationState,
    pub legacy_groups: u64,
    pub collections: u64,
    pub memberships: u64,
    pub orphan_memberships: u64,
}

/// Per-group result of [`Migrator::verify`].
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct GroupCheck {
    pub legacy_key: String,
    pub collection: Option<String>,
    pub legacy_count: u64,
    pub present: u64,
}

/// Result of comparing the legacy links with the collections.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct VerifyReport {
    pub ok: bool,
    pub groups_checked: u64,
    pub legacy_memberships: u64,
    pub present: u64,
    pub missing: u64,
    pub groups_without_collection: Vec<String>,
    pub orphan_memberships: u64,
    pub memberships_total: u64,
    pub mismatched: Vec<GroupCheck>,
    pub summary: String,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Step {
    #[default]
    Idle,
    Snapshot,
    Planning,
    Mapping,
    Copying,
    Verifying,
    Rolling,
    Done,
    Failed,
}

impl Step {
    pub fn label(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Snapshot => "taking snapshot",
            Self::Planning => "counting legacy links",
            Self::Mapping => "mapping groups",
            Self::Copying => "copying memberships",
            Self::Verifying => "verifying",
            Self::Rolling => "removing collections",
            Self::Done => "done",
            Self::Failed => "failed",
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Progress {
    pub step: Step,
    pub message: String,
    pub groups_total: u64,
    pub groups_done: u64,
    pub memberships_total: u64,
    pub processed: u64,
    pub added: u64,
    pub started: Option<Instant>,
    pub ended: Option<Instant>,
    pub error: Option<String>,
}

impl Progress {
    pub fn fraction(&self) -> Option<f32> {
        (self.memberships_total > 0).then(|| (self.processed as f32 / self.memberships_total as f32).clamp(0.0, 1.0))
    }
}

#[derive(Clone, Debug)]
pub struct CopyOptions {
    pub require_snapshot: bool,
    pub restart: bool,
    pub max_groups: Option<usize>,
}

impl Default for CopyOptions {
    fn default() -> Self {
        Self { require_snapshot: true, restart: false, max_groups: None }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct CopyReport {
    pub groups_mapped: u64,
    pub collections_created: u64,
    pub groups_copied: u64,
    pub memberships_added: u64,
    pub memberships_existing: u64,
    pub unfiled_rows: u64,
    pub dangling_links: u64,
    pub renamed: Vec<RenamedGroup>,
    pub warnings: Vec<String>,
    pub millis: u128,
    pub cancelled: bool,
    pub snapshot: Option<String>,
}

#[derive(Clone, Debug)]
pub struct FinishOptions {
    pub require_snapshot: bool,
    pub repair: bool,
}

impl Default for FinishOptions {
    fn default() -> Self {
        Self { require_snapshot: true, repair: true }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct FinishReport {
    pub ok: bool,
    pub repaired: bool,
    pub verify: VerifyReport,
    pub legacy_cleanup_pending: bool,
    pub summary: String,
}

#[derive(Clone, Debug, Default)]
pub struct RollbackOptions {
    pub require_snapshot: bool,
    pub force: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct RollbackReport {
    pub ok: bool,
    pub collections_removed: u64,
    pub memberships_removed: u64,
    pub created_after_migration: u64,
    pub memberships_after_migration: u64,
    pub snapshot: Option<String>,
    pub summary: String,
}

struct Shared {
    progress: Mutex<Progress>,
    cancel: AtomicBool,
    running: AtomicBool,
}

struct RunGuard<'a>(&'a Shared);

impl Drop for RunGuard<'_> {
    fn drop(&mut self) {
        self.0.running.store(false, Ordering::SeqCst);
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
            shared: Shared { progress: Mutex::new(Progress::default()), cancel: AtomicBool::new(false), running: AtomicBool::new(false) },
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

    pub fn progress(&self) -> Progress {
        self.shared.progress.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub fn is_running(&self) -> bool {
        self.shared.running.load(Ordering::SeqCst)
    }

    pub fn cancel(&self) {
        if self.is_running() {
            self.shared.cancel.store(true, Ordering::SeqCst);
        }
    }

    fn update(&self, f: impl FnOnce(&mut Progress)) {
        f(&mut self.shared.progress.lock().unwrap_or_else(|e| e.into_inner()));
    }

    fn set_step(&self, step: Step, message: impl Into<String>) {
        let message = message.into();
        self.update(|p| {
            p.step = step;
            p.message = message;
        });
    }

    fn claim(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.shared.running.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_ok(),
            "a collection migration step is already running"
        );
        self.shared.cancel.store(false, Ordering::SeqCst);
        self.update(|p| *p = Progress { started: Some(Instant::now()), ..Default::default() });
        Ok(())
    }

    fn fail(&self, e: &anyhow::Error) {
        let msg = format!("{e:#}");
        log_event(log::Level::Error, format!("Collection migration failed: {msg}"));
        self.update(|p| {
            p.step = Step::Failed;
            p.error = Some(msg.clone());
            p.message = msg;
        });
    }

    fn cancelled(&self) -> bool {
        self.shared.cancel.load(Ordering::SeqCst)
    }

    pub async fn state(&self) -> anyhow::Result<MigrationState> {
        collections::ensure_schema(self.db()).await?;
        load_state(self.db()).await
    }

    /// Phase, state and live counts.
    pub async fn status(&self) -> anyhow::Result<Status> {
        let db = self.db();
        collections::ensure_schema(db).await?;
        let state = load_state(db).await?;
        let legacy_groups = legacy_groups(db).await?.iter().filter(|g| !g.is_unfiled()).count() as u64;
        let counts = collections::list_with_counts(db).await?;
        let orphan_memberships = collections::orphan_count(db).await?;
        Ok(Status {
            phase: state.phase(),
            phase_label: state.phase().label(),
            state,
            legacy_groups,
            collections: counts.len() as u64,
            memberships: counts.iter().map(|c| c.count).sum::<u64>() + orphan_memberships,
            orphan_memberships,
        })
    }

    /// What a copy run would do, without writing anything.
    pub async fn plan(&self) -> anyhow::Result<Plan> {
        let db = self.db();
        collections::ensure_schema(db).await?;
        let legacy = legacy_groups(db).await?;
        let (links, unlinked) = legacy_link_counts(db).await?;
        let assignments = self.assignments(&legacy).await?;
        let mut plan = Plan { dangling_links: unlinked, ..Default::default() };
        for a in assignments {
            let memberships = links.get(&a.legacy_key).copied().unwrap_or(0);
            plan.memberships += memberships;
            plan.to_create += usize::from(!a.reuse);
            plan.renamed += usize::from(a.renamed && !a.reuse);
            plan.groups.push(PlannedGroup { assignment: a, memberships });
        }
        let (unfiled, dangling) = link_split(&legacy, &links);
        plan.unfiled_rows = unfiled;
        plan.dangling_links += dangling;
        plan.warnings = unfiled_warnings(&legacy, &links);
        Ok(plan)
    }

    /// Collection names for every legacy group except the unfiled one, reusing collections migrated before.
    async fn assignments(&self, legacy: &[LegacyGroup]) -> anyhow::Result<Vec<Assignment>> {
        let existing = collections::list(self.db()).await?;
        let taken: HashSet<String> = existing.iter().map(Collection::key).collect();
        let present: HashSet<&str> = legacy.iter().map(|g| g.key.as_str()).collect();
        let by_legacy: HashMap<String, String> = existing
            .iter()
            .filter_map(|c| c.legacy_key.clone().filter(|k| present.contains(k.as_str())).map(|k| (k, c.name.clone())))
            .collect();
        let ins: Vec<GroupIn> = legacy.iter().map(|g| GroupIn { key: g.key.clone(), name: g.name.clone() }).collect();
        Ok(assign_names(&ins, &by_legacy, &taken))
    }

    /// The collection for each assignment, creating those that do not exist yet.
    async fn ensure_collections(&self, assignments: Vec<Assignment>, legacy: &[LegacyGroup], state: &mut MigrationState) -> anyhow::Result<Vec<(Assignment, Collection)>> {
        let db = self.db();
        let by_key: HashMap<&str, &LegacyGroup> = legacy.iter().map(|g| (g.key.as_str(), g)).collect();
        let mut existing: HashMap<String, Collection> = collections::list(db).await?.into_iter().filter_map(|c| c.legacy_key.clone().map(|k| (k, c))).collect();
        let mut out = Vec::with_capacity(assignments.len());
        for a in assignments {
            let coll = match existing.remove(&a.legacy_key) {
                Some(c) => c,
                None => {
                    let g = by_key.get(a.legacy_key.as_str()).context("legacy group vanished during the migration")?;
                    let c = collections::create_with(
                        db,
                        NewCollection { name: a.name.clone(), created: g.created, updated: g.updated, color: None, note: None, legacy_key: Some(a.legacy_key.clone()) },
                    )
                    .await?;
                    state.collections_created += 1;
                    if a.renamed {
                        state.renamed.push(RenamedGroup { legacy_key: a.legacy_key.clone(), original: a.legacy_name.clone(), assigned: c.name.clone() });
                    }
                    c
                }
            };
            out.push((a, coll));
        }
        Ok(out)
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
            log_event(log::Level::Info, format!("Collection migration: using snapshot {} taken within the last hour", info.path.display()));
            return Ok(absolute(&info.path));
        }
        let mut last_err = String::new();
        for attempt in 1..=SNAPSHOT_ATTEMPTS {
            log_event(log::Level::Info, format!("Collection migration: taking a snapshot of {} (attempt {attempt}/{SNAPSHOT_ATTEMPTS})", self.store.display()));
            let info = snapshot::create_snapshot(self.store.clone(), SNAPSHOT_LABEL, None, progress.clone()).await.map_err(|e| anyhow!(e.user_message()))?;
            if snapshot::read_manifest(&info.path).is_some_and(|m| m.consistent) {
                return Ok(absolute(&info.path));
            }
            last_err = format!("snapshot {} was taken while the database was changing", info.path.display());
            let _ = snapshot::delete_snapshot(&self.store, &info.path);
        }
        anyhow::bail!("no consistent snapshot could be taken ({last_err}); stop scans and other database activity, then retry")
    }

    /// Runs the copy phase on the runtime; false when a step is already running.
    pub fn spawn_copy(self: &Arc<Self>, opts: CopyOptions) -> bool {
        if self.claim().is_err() {
            return false;
        }
        let me = Arc::clone(self);
        tokio::spawn(async move {
            if let Err(e) = me.copy_claimed(opts).await {
                log::warn!("[collections] copy ended: {e:#}");
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
                log::warn!("[collections] finish ended: {e:#}");
            }
        });
        true
    }

    /// Runs the rollback on the runtime; false when a step is already running.
    pub fn spawn_rollback(self: &Arc<Self>, opts: RollbackOptions) -> bool {
        if self.claim().is_err() {
            return false;
        }
        let me = Arc::clone(self);
        tokio::spawn(async move {
            if let Err(e) = me.rollback_claimed(opts).await {
                log::warn!("[collections] rollback ended: {e:#}");
            }
        });
        true
    }

    pub async fn run_copy(&self, opts: CopyOptions) -> anyhow::Result<CopyReport> {
        self.claim()?;
        self.copy_claimed(opts).await
    }

    pub async fn run_finish(&self, opts: FinishOptions) -> anyhow::Result<FinishReport> {
        self.claim()?;
        self.finish_claimed(opts).await
    }

    pub async fn run_rollback(&self, opts: RollbackOptions) -> anyhow::Result<RollbackReport> {
        self.claim()?;
        self.rollback_claimed(opts).await
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
        let db = self.db();
        collections::ensure_schema(db).await?;
        let mut state = load_state(db).await?;
        let mut report = CopyReport { snapshot: state.snapshot.clone(), ..Default::default() };

        self.set_step(Step::Planning, "counting legacy links");
        let legacy = legacy_groups(db).await?;
        let (links, unlinked) = legacy_link_counts(db).await?;
        let assignments = self.assignments(&legacy).await?;
        let legacy_total: u64 = assignments.iter().map(|a| links.get(&a.legacy_key).copied().unwrap_or(0)).sum();
        if opts.require_snapshot && (!assignments.is_empty() || state.started.is_some()) {
            self.set_step(Step::Snapshot, "taking a snapshot of the store");
            db_set_detail("Collection migration: snapshot");
            let snap = self.ensure_snapshot(snapshot::status_progress("Snapshot")).await?;
            state.snapshot = Some(snap.clone());
            report.snapshot = Some(snap);
        }

        let fresh_pass = opts.restart || state.copy_done || state.needs_resync || state.started.is_none();
        if fresh_pass {
            state.groups_done.clear();
            state.memberships_added = 0;
            state.memberships_existing = 0;
            state.collections_created = 0;
            state.renamed.clear();
            state.warnings.clear();
        }
        state.started.get_or_insert_with(|| Local::now().to_rfc3339());
        state.runs += u32::from(fresh_pass);
        state.finished = None;
        state.rolled_back = None;
        state.copy_done = false;
        state.needs_resync = false;
        state.verify = None;
        save_state(db, &state).await?;

        self.update(|p| {
            p.groups_total = assignments.len() as u64;
            p.memberships_total = legacy_total;
        });
        log_event(log::Level::Info, format!("Collection migration: {} groups, {legacy_total} memberships (resuming after {} finished groups)", assignments.len(), state.groups_done.len()));

        let by_key: HashMap<&str, &LegacyGroup> = legacy.iter().map(|g| (g.key.as_str(), g)).collect();
        self.set_step(Step::Mapping, "creating collections");
        let mapped = self.ensure_collections(assignments, &legacy, &mut state).await?;
        state.groups_mapped = mapped.len() as u64;
        state.warnings = unfiled_warnings(&legacy, &links);
        let (unfiled, dangling) = link_split(&legacy, &links);
        state.unfiled_rows = unfiled;
        state.dangling_links = dangling + unlinked;
        save_state(db, &state).await?;

        self.set_step(Step::Copying, "");
        let mut copied_groups = 0usize;
        for (a, coll) in &mapped {
            if state.groups_done.contains(&a.legacy_key) {
                self.update(|p| {
                    p.groups_done += 1;
                    p.processed += links.get(&a.legacy_key).copied().unwrap_or(0);
                });
                continue;
            }
            if self.cancelled() || opts.max_groups.is_some_and(|m| copied_groups >= m) {
                report.cancelled = true;
                break;
            }
            let g = by_key.get(a.legacy_key.as_str()).context("legacy group vanished during the migration")?;
            let _ga = db_activity("Migrating collections");
            match self.copy_group(&g.id, coll).await? {
                Some((added, existing)) => {
                    state.memberships_added += added;
                    state.memberships_existing += existing;
                    state.groups_done.push(a.legacy_key.clone());
                    save_state(db, &state).await?;
                    copied_groups += 1;
                    self.update(|p| p.groups_done += 1);
                }
                None => {
                    report.cancelled = true;
                    break;
                }
            }
        }

        if !report.cancelled {
            state.copy_done = true;
            state.copied_at = Some(Local::now().to_rfc3339());
            save_state(db, &state).await?;
        }
        db_set_progress(0, 0);
        report.groups_mapped = state.groups_mapped;
        report.collections_created = state.collections_created;
        report.groups_copied = state.groups_done.len() as u64;
        report.memberships_added = state.memberships_added;
        report.memberships_existing = state.memberships_existing;
        report.unfiled_rows = state.unfiled_rows;
        report.dangling_links = state.dangling_links;
        report.renamed = state.renamed.clone();
        report.warnings = state.warnings.clone();
        report.millis = started.elapsed().as_millis();
        self.set_step(if report.cancelled { Step::Idle } else { Step::Done }, if report.cancelled { "stopped; progress is kept" } else { "copy complete" });
        log_event(
            log::Level::Info,
            format!(
                "Collection migration copy {}: {} collections created, {} memberships added, {} already present in {:.1}s",
                if report.cancelled { "stopped" } else { "complete" },
                report.collections_created,
                report.memberships_added,
                report.memberships_existing,
                report.millis as f32 / 1000.0
            ),
        );
        Ok(report)
    }

    /// Copies the files of one legacy group; None when cancelled part-way.
    async fn copy_group(&self, group: &RecordId, coll: &Collection) -> anyhow::Result<Option<(u64, u64)>> {
        let db = self.db();
        let ids: Vec<RecordId> = db.query("SELECT VALUE id FROM thumbnails WHERE logical_group = $g").bind(("g", group.clone())).await?.take(0)?;
        let key = coll.key();
        let (mut added, mut existing) = (0u64, 0u64);
        for chunk in ids.chunks(CHUNK) {
            if self.cancelled() {
                return Ok(None);
            }
            let paths: Vec<String> = db.query("SELECT VALUE path FROM $ids").bind(("ids", chunk.to_vec())).await?.take(0)?;
            let unique: Vec<&str> = paths.iter().map(String::as_str).filter(|p| !p.is_empty()).collect::<HashSet<_>>().into_iter().collect();
            let n = collections::insert_paths(db, &coll.id, &key, &unique).await? as u64;
            added += n;
            existing += chunk.len() as u64 - n.min(chunk.len() as u64);
            self.update(|p| {
                p.processed += chunk.len() as u64;
                p.added += n;
            });
            let (done, total) = self.update_progress_numbers();
            db_set_progress(done, total);
        }
        Ok(Some((added, existing)))
    }

    fn update_progress_numbers(&self) -> (u64, u64) {
        let p = self.progress();
        (p.processed.min(p.memberships_total), p.memberships_total)
    }

    /// Compares legacy links with the collections without changing anything.
    pub async fn verify(&self) -> anyhow::Result<VerifyReport> {
        let db = self.db();
        collections::ensure_schema(db).await?;
        let legacy = legacy_groups(db).await?;
        let by_legacy: HashMap<String, Collection> = collections::list(db).await?.into_iter().filter_map(|c| c.legacy_key.clone().map(|k| (k, c))).collect();
        let mut report = VerifyReport::default();
        for g in legacy.iter().filter(|g| !g.is_unfiled()) {
            let ids: Vec<RecordId> = db.query("SELECT VALUE id FROM thumbnails WHERE logical_group = $g").bind(("g", g.id.clone())).await?.take(0)?;
            report.groups_checked += 1;
            let mut check = GroupCheck { legacy_key: g.key.clone(), collection: by_legacy.get(&g.key).map(|c| c.name.clone()), legacy_count: ids.len() as u64, present: 0 };
            match by_legacy.get(&g.key) {
                None => report.groups_without_collection.push(g.name.clone()),
                Some(c) => {
                    let key = c.key();
                    for chunk in ids.chunks(CHUNK) {
                        let paths: Vec<String> = db.query("SELECT VALUE path FROM $ids").bind(("ids", chunk.to_vec())).await?.take(0)?;
                        let distinct: HashSet<String> = paths.into_iter().collect();
                        let items: Vec<RecordId> = distinct.iter().map(|p| collections::item_id(&key, p)).collect();
                        let found: Vec<RecordId> = db.query("SELECT VALUE id FROM $items").bind(("items", items)).await?.take(0)?;
                        check.present += found.len() as u64;
                    }
                }
            }
            report.legacy_memberships += check.legacy_count;
            report.present += check.present;
            if check.present < check.legacy_count && report.mismatched.len() < MAX_LISTED {
                report.mismatched.push(check);
            }
        }
        report.missing = report.legacy_memberships.saturating_sub(report.present);
        report.orphan_memberships = collections::orphan_count(db).await?;
        report.memberships_total = collections::count_items(db).await?;
        report.ok = report.missing == 0 && report.groups_without_collection.is_empty() && report.orphan_memberships == 0;
        report.summary = if report.ok {
            format!("{} groups verified: all {} legacy memberships are in collections ({} memberships in total)", report.groups_checked, report.legacy_memberships, report.memberships_total)
        } else {
            format!(
                "{} legacy memberships are missing from collections, {} groups have no collection, {} memberships point at a missing collection",
                report.missing,
                report.groups_without_collection.len(),
                report.orphan_memberships
            )
        };
        Ok(report)
    }

    async fn finish_claimed(&self, opts: FinishOptions) -> anyhow::Result<FinishReport> {
        let _run = RunGuard(&self.shared);
        let result = self.finish_inner(opts).await;
        if let Err(e) = &result {
            self.fail(e);
        }
        self.update(|p| p.ended = Some(Instant::now()));
        result
    }

    async fn finish_inner(&self, opts: FinishOptions) -> anyhow::Result<FinishReport> {
        let db = self.db();
        collections::ensure_schema(db).await?;
        let mut state = load_state(db).await?;
        anyhow::ensure!(state.started.is_some() && state.copy_done, "the copy phase has not completed; run it first");
        if opts.require_snapshot && !state.snapshot.as_ref().is_some_and(|p| Path::new(p).is_dir()) {
            self.set_step(Step::Snapshot, "taking a snapshot of the store");
            state.snapshot = Some(self.ensure_snapshot(snapshot::status_progress("Snapshot")).await?);
            save_state(db, &state).await?;
        }
        self.set_step(Step::Verifying, "comparing legacy links with collections");
        let mut report = FinishReport { legacy_cleanup_pending: true, ..Default::default() };
        let mut verify = self.verify().await?;
        if !verify.ok && opts.repair {
            self.set_step(Step::Copying, "catching up missing memberships");
            let copied = self.copy_for_repair().await?;
            report.repaired = copied;
            self.set_step(Step::Verifying, "comparing legacy links with collections");
            verify = self.verify().await?;
        }
        report.ok = verify.ok;
        report.summary = if verify.ok { format!("migration finished: {}", verify.summary) } else { format!("not finished: {}", verify.summary) };
        state = load_state(db).await?;
        state.verify = Some(verify.clone());
        if verify.ok {
            state.finished = Some(Local::now().to_rfc3339());
            state.needs_resync = false;
        }
        save_state(db, &state).await?;
        report.verify = verify;
        self.set_step(if report.ok { Step::Done } else { Step::Idle }, report.summary.clone());
        log_event(if report.ok { log::Level::Info } else { log::Level::Warn }, format!("Collection migration finish: {}", report.summary));
        Ok(report)
    }

    /// Re-runs the idempotent copy for every group, creating collections that are missing.
    async fn copy_for_repair(&self) -> anyhow::Result<bool> {
        let db = self.db();
        let legacy = legacy_groups(db).await?;
        let assignments = self.assignments(&legacy).await?;
        let mut state = load_state(db).await?;
        let created_before = state.collections_created;
        let mapped = self.ensure_collections(assignments, &legacy, &mut state).await?;
        let by_key: HashMap<&str, &LegacyGroup> = legacy.iter().map(|g| (g.key.as_str(), g)).collect();
        let mut any = state.collections_created > created_before;
        for (a, coll) in &mapped {
            let g = by_key.get(a.legacy_key.as_str()).context("legacy group vanished during the migration")?;
            if let Some((added, _)) = self.copy_group(&g.id, coll).await? {
                any |= added > 0;
            }
        }
        save_state(db, &state).await?;
        Ok(any)
    }

    async fn rollback_claimed(&self, opts: RollbackOptions) -> anyhow::Result<RollbackReport> {
        let _run = RunGuard(&self.shared);
        let result = self.rollback_inner(opts).await;
        if let Err(e) = &result {
            self.fail(e);
        }
        self.update(|p| p.ended = Some(Instant::now()));
        result
    }

    async fn rollback_inner(&self, opts: RollbackOptions) -> anyhow::Result<RollbackReport> {
        let db = self.db();
        collections::ensure_schema(db).await?;
        let mut state = load_state(db).await?;
        anyhow::ensure!(state.started.is_some(), "no collection migration has been started, so there is nothing to roll back");
        let all = collections::list_with_counts(db).await?;
        let (links, _) = legacy_link_counts(db).await?;
        let created_after = all.iter().filter(|c| c.collection.legacy_key.is_none()).count() as u64;
        let extra: u64 = all
            .iter()
            .map(|c| match &c.collection.legacy_key {
                Some(k) => c.count.saturating_sub(links.get(k).copied().unwrap_or(0)),
                None => c.count,
            })
            .sum();
        let mut report = RollbackReport { created_after_migration: created_after, memberships_after_migration: extra, ..Default::default() };
        if !opts.force && (created_after > 0 || extra > 0) {
            report.summary = format!("refused: {created_after} collections and {extra} memberships were created after the migration and would be lost; pass force to discard them");
            return Ok(report);
        }
        if opts.require_snapshot {
            self.set_step(Step::Snapshot, "taking a snapshot of the store");
            let snap = self.ensure_snapshot(snapshot::status_progress("Snapshot")).await?;
            state.snapshot = Some(snap.clone());
            report.snapshot = Some(snap);
        }
        self.set_step(Step::Rolling, "removing collections");
        for c in &all {
            if self.cancelled() {
                report.summary = "stopped; run rollback again to continue".into();
                return Ok(report);
            }
            report.memberships_removed += collections::delete(db, &c.collection.id).await?;
            report.collections_removed += 1;
            self.update(|p| p.processed += c.count);
        }
        report.memberships_removed += collections::prune_orphans(db).await?;
        let runs = state.runs;
        state = MigrationState { rolled_back: Some(Local::now().to_rfc3339()), runs, snapshot: state.snapshot.clone(), ..Default::default() };
        save_state(db, &state).await?;
        report.ok = true;
        report.summary = format!("rolled back: {} collections and {} memberships removed; legacy groups were not touched", report.collections_removed, report.memberships_removed);
        self.set_step(Step::Done, report.summary.clone());
        log_event(log::Level::Info, format!("Collection migration rollback: {}", report.summary));
        Ok(report)
    }
}

/// Rows linked to the unfiled group and rows linked to a group that does not exist.
fn link_split(groups: &[LegacyGroup], links: &HashMap<String, u64>) -> (u64, u64) {
    let known: HashSet<&str> = groups.iter().map(|g| g.key.as_str()).collect();
    let unfiled = groups.iter().filter(|g| g.is_unfiled()).map(|g| links.get(&g.key).copied().unwrap_or(0)).sum();
    let dangling = links.iter().filter(|(k, _)| !known.contains(k.as_str())).map(|(_, n)| n).sum();
    (unfiled, dangling)
}

/// Warnings about legacy groups whose meaning needs attention.
fn unfiled_warnings(groups: &[LegacyGroup], links: &HashMap<String, u64>) -> Vec<String> {
    let mut out = Vec::new();
    for g in groups {
        if g.key == UNFILED_NAME && !g.is_unfiled() {
            out.push(format!(
                "the group record \"{UNFILED_NAME}\" was renamed to \"{}\" and is migrated as a collection with {} files; delete that collection if it only held files that were never filed",
                g.name,
                links.get(&g.key).copied().unwrap_or(0)
            ));
        }
    }
    out
}

fn absolute(path: &Path) -> String {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf()).to_string_lossy().into_owned()
}

/// True when legacy groups other than the unfiled one exist and the migration has not finished.
pub async fn legacy_pending<C: Connection>(db: &Surreal<C>) -> anyhow::Result<bool> {
    let state = load_state(db).await?;
    if state.finished.is_some() && !state.needs_resync {
        return Ok(false);
    }
    Ok(legacy_groups(db).await?.iter().any(|g| !g.is_unfiled()))
}

#[cfg(test)]
mod tests;
