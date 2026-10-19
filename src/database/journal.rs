//! Metadata edit journal: one entry per path per change, grouped in batches, with undo, redo and a retention cap.
//!
//! A writer that edits metadata records what it changed and never lets the journal fail the edit:
//!
//! ```ignore
//! let batch = journal::begin_batch("Add tags");
//! journal::record(batch, &path, Field::Tags, Value::tags(&before), Value::tags(&after));
//! journal::commit(batch).await;
//! ```
//!
//! `begin_batch` and `record` only collect entries in memory; `commit` writes all of them in one transaction, so a batch is stored whole or not at all.
//! Entries whose before and after are equal are dropped. Collection membership is a `Field::Collection` entry with the collection key as `target` and `Value::Member` values.
//!
//! `undo_last` reverts the newest batch that is still applied, `redo_last` re-applies the batch undone most recently. Each entry is applied only when the row still holds the value the entry expects; an entry whose row changed since is skipped and reported as a conflict, never overwritten. A new commit clears the redo stack. Rows created by an edit stay in the library when the edit is undone.
//!
//! Every function has a `_with` form over any SurrealDB connection (kv-mem in tests); the plain form uses the application database.

mod wire;

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

use anyhow::Context;
use surrealdb::types::{RecordId, SurrealValue};
use surrealdb::{Connection, Surreal};

use super::collections;
use super::metadata_edit::MetaPatch;

pub use wire::{MetaSnapshot, add_to_collection_with, copy_collection_with, delete_tag_with, prune_tags_with, record_patch, remove_from_collection_with, rename_category_with, rename_tag_with, snapshots_with};

pub const TABLE: &str = "metadata_journal";
/// Entries kept; the oldest whole batches are dropped beyond this.
pub const RETENTION: usize = 5000;
const CHUNK: usize = 400;
const IDLE_WAIT: Duration = Duration::from_secs(2);
const PENDING_MAX_AGE_MS: i64 = 10 * 60 * 1000;

const DDL: &str = r#"
    DEFINE TABLE IF NOT EXISTS metadata_journal TYPE NORMAL SCHEMAFULL PERMISSIONS FULL;
    DEFINE FIELD IF NOT EXISTS batch ON metadata_journal TYPE int PERMISSIONS FULL;
    DEFINE FIELD IF NOT EXISTS seq ON metadata_journal TYPE int PERMISSIONS FULL;
    DEFINE FIELD IF NOT EXISTS at_ms ON metadata_journal TYPE int PERMISSIONS FULL;
    DEFINE FIELD IF NOT EXISTS source ON metadata_journal TYPE string PERMISSIONS FULL;
    DEFINE FIELD IF NOT EXISTS path ON metadata_journal TYPE string PERMISSIONS FULL;
    DEFINE FIELD IF NOT EXISTS prop ON metadata_journal TYPE string PERMISSIONS FULL;
    DEFINE FIELD IF NOT EXISTS target ON metadata_journal TYPE option<string> PERMISSIONS FULL;
    DEFINE FIELD IF NOT EXISTS before ON metadata_journal TYPE string PERMISSIONS FULL;
    DEFINE FIELD IF NOT EXISTS after ON metadata_journal TYPE string PERMISSIONS FULL;
    DEFINE FIELD IF NOT EXISTS state ON metadata_journal TYPE string PERMISSIONS FULL;
    DEFINE FIELD IF NOT EXISTS undone_seq ON metadata_journal TYPE option<int> PERMISSIONS FULL;
    DEFINE INDEX IF NOT EXISTS metadata_journal_batch_idx ON metadata_journal FIELDS batch;
    DEFINE INDEX IF NOT EXISTS metadata_journal_seq_idx ON metadata_journal FIELDS seq UNIQUE;
"#;

/// Defines the journal table on a handle already pointed at NS/DB; safe to run on every call.
pub async fn ensure_schema<C: Connection>(db: &Surreal<C>) -> anyhow::Result<()> {
    db.query(DDL).await?.check().context("defining the metadata journal failed")?;
    Ok(())
}

/// Identity of one batch of changes; later batches have larger numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BatchId(pub i64);

impl std::fmt::Display for BatchId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "b{}", self.0)
    }
}

/// What an entry changes about a path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Field {
    Caption,
    Description,
    Category,
    Tags,
    /// Membership of the path in the collection named by the entry's target.
    Collection,
}

impl Field {
    pub const ALL: [Field; 5] = [Self::Caption, Self::Description, Self::Category, Self::Tags, Self::Collection];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Caption => "caption",
            Self::Description => "description",
            Self::Category => "category",
            Self::Tags => "tags",
            Self::Collection => "collection",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|f| f.as_str() == s)
    }

    /// True for the fields stored as optional text on the library row.
    pub fn is_text(self) -> bool {
        matches!(self, Self::Caption | Self::Description | Self::Category)
    }
}

/// A value of a field: optional text, a tag list or a membership flag.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Value {
    Text(Option<String>),
    Tags(Vec<String>),
    Member(bool),
}

impl Value {
    /// Text where blank means none.
    pub fn text(value: Option<&str>) -> Self {
        Self::Text(value.filter(|v| !v.trim().is_empty()).map(str::to_string))
    }

    pub fn tags(tags: &[String]) -> Self {
        Self::Tags(tags.to_vec())
    }

    fn to_json(&self) -> String {
        match self {
            Self::Text(None) => "null".to_string(),
            Self::Text(Some(s)) => serde_json::Value::String(s.clone()).to_string(),
            Self::Tags(t) => serde_json::to_string(t).unwrap_or_else(|_| "[]".to_string()),
            Self::Member(b) => b.to_string(),
        }
    }

    fn from_json(field: Field, json: &str) -> Self {
        let value: serde_json::Value = serde_json::from_str(json).unwrap_or(serde_json::Value::Null);
        match field {
            Field::Tags => Self::Tags(value.as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect()).unwrap_or_default()),
            Field::Collection => Self::Member(value.as_bool().unwrap_or(false)),
            _ => Self::text(value.as_str()),
        }
    }

    /// Short text for lists and conflict messages.
    pub fn brief(&self) -> String {
        match self {
            Self::Text(None) => "(none)".to_string(),
            Self::Text(Some(s)) => clip(s, 40),
            Self::Tags(t) if t.is_empty() => "(no tags)".to_string(),
            Self::Tags(t) => clip(&t.join(", "), 40),
            Self::Member(true) => "in the collection".to_string(),
            Self::Member(false) => "not in the collection".to_string(),
        }
    }
}

fn clip(text: &str, max: usize) -> String {
    let one_line: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if one_line.chars().count() <= max { one_line } else { format!("{}...", one_line.chars().take(max).collect::<String>()) }
}

/// Where an entry is in the undo and redo stacks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntryState {
    Applied,
    Undone,
    /// Could not be undone or redone because the row changed since.
    Skipped,
}

impl EntryState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Applied => "applied",
            Self::Undone => "undone",
            Self::Skipped => "skipped",
        }
    }

    fn parse(s: &str) -> Self {
        match s {
            "undone" => Self::Undone,
            "skipped" => Self::Skipped,
            _ => Self::Applied,
        }
    }
}

/// One journaled change of one path.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub seq: i64,
    pub batch: BatchId,
    pub at_ms: i64,
    pub source: String,
    pub path: String,
    pub field: Field,
    pub target: Option<String>,
    pub before: Value,
    pub after: Value,
    pub state: EntryState,
}

impl Entry {
    fn key(&self) -> Key {
        (self.path.clone(), self.field, self.target.clone())
    }
}

type Key = (String, Field, Option<String>);

#[derive(Clone, Debug, SurrealValue)]
struct Row {
    id: RecordId,
    batch: i64,
    seq: i64,
    at_ms: i64,
    source: String,
    path: String,
    prop: String,
    target: Option<String>,
    before: String,
    after: String,
    state: String,
    undone_seq: Option<i64>,
}

impl Row {
    fn from_entry(e: &Entry, undone_seq: Option<i64>) -> Self {
        Self {
            id: RecordId::new(TABLE, e.seq.to_string()),
            batch: e.batch.0,
            seq: e.seq,
            at_ms: e.at_ms,
            source: e.source.clone(),
            path: e.path.clone(),
            prop: e.field.as_str().to_string(),
            target: e.target.clone(),
            before: e.before.to_json(),
            after: e.after.to_json(),
            state: e.state.as_str().to_string(),
            undone_seq,
        }
    }

    fn into_entry(self) -> Option<Entry> {
        let field = Field::parse(&self.prop)?;
        Some(Entry {
            seq: self.seq,
            batch: BatchId(self.batch),
            at_ms: self.at_ms,
            source: self.source,
            path: self.path,
            field,
            target: self.target,
            before: Value::from_json(field, &self.before),
            after: Value::from_json(field, &self.after),
            state: EntryState::parse(&self.state),
        })
    }
}

#[derive(Debug, SurrealValue)]
struct CountRow {
    n: i64,
}

#[derive(Debug, SurrealValue)]
struct BatchSeq {
    batch: i64,
    seq: i64,
}

static LAST_STAMP: AtomicI64 = AtomicI64::new(0);
static REVISION: AtomicU64 = AtomicU64::new(0);
static IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);
static PENDING: LazyLock<Mutex<HashMap<i64, Pending>>> = LazyLock::new(|| Mutex::new(HashMap::new()));
static GATE: LazyLock<tokio::sync::Mutex<()>> = LazyLock::new(|| tokio::sync::Mutex::new(()));
static LAST_REPORT: LazyLock<Mutex<Option<UndoReport>>> = LazyLock::new(|| Mutex::new(None));

fn unix_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

/// A number larger than any handed out before in this process, close to the clock in microseconds.
fn stamp() -> i64 {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_micros() as i64).unwrap_or(0);
    let mut last = LAST_STAMP.load(Ordering::Relaxed);
    loop {
        let next = now.max(last + 1);
        match LAST_STAMP.compare_exchange_weak(last, next, Ordering::SeqCst, Ordering::Relaxed) {
            Ok(_) => return next,
            Err(seen) => last = seen,
        }
    }
}

/// Counter that grows with every commit, undo, redo and clear, for views that refresh on change.
pub fn revision() -> u64 {
    REVISION.load(Ordering::Relaxed)
}

fn bump() {
    REVISION.fetch_add(1, Ordering::Relaxed);
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

struct NewEntry {
    path: String,
    field: Field,
    target: Option<String>,
    before: Value,
    after: Value,
}

struct Pending {
    source: String,
    started_ms: i64,
    entries: Vec<NewEntry>,
    index: HashMap<Key, usize>,
}

/// Starts collecting the changes of one edit under the label `source`.
pub fn begin_batch(source: &str) -> BatchId {
    let id = BatchId(stamp());
    let now = unix_ms();
    let mut pending = lock(&PENDING);
    let before = pending.len();
    pending.retain(|_, p| now - p.started_ms < PENDING_MAX_AGE_MS);
    IN_FLIGHT.fetch_sub(before - pending.len(), Ordering::SeqCst);
    pending.insert(id.0, Pending { source: source.trim().to_string(), started_ms: now, entries: Vec::new(), index: HashMap::new() });
    IN_FLIGHT.fetch_add(1, Ordering::SeqCst);
    id
}

/// Adds one change to the batch; a repeat of the same path and field keeps the first before and the last after, and a change that ends where it began is dropped.
pub fn record(batch: BatchId, path: &str, field: Field, before: Value, after: Value) {
    record_target(batch, path, field, None, before, after);
}

/// Adds a collection membership change to the batch; `collection_key` is the key of the collection.
pub fn record_member(batch: BatchId, path: &str, collection_key: &str, before: bool, after: bool) {
    record_target(batch, path, Field::Collection, Some(collection_key.to_string()), Value::Member(before), Value::Member(after));
}

fn record_target(batch: BatchId, path: &str, field: Field, target: Option<String>, before: Value, after: Value) {
    if path.is_empty() {
        return;
    }
    let mut pending = lock(&PENDING);
    let Some(p) = pending.get_mut(&batch.0) else { return };
    let key = (path.to_string(), field, target.clone());
    match p.index.get(&key).copied() {
        Some(i) => p.entries[i].after = after,
        None => {
            p.index.insert(key, p.entries.len());
            p.entries.push(NewEntry { path: path.to_string(), field, target, before, after });
        }
    }
}

/// Drops the batch without writing it.
pub fn abandon(batch: BatchId) {
    if lock(&PENDING).remove(&batch.0).is_some() {
        IN_FLIGHT.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Number of changes collected in a batch that is not committed yet.
pub fn pending_len(batch: BatchId) -> usize {
    lock(&PENDING).get(&batch.0).map_or(0, |p| p.entries.iter().filter(|e| e.before != e.after).count())
}

/// Waits until no batch is being collected, so an undo does not race the edit it follows.
pub async fn wait_idle() {
    let started = std::time::Instant::now();
    while IN_FLIGHT.load(Ordering::SeqCst) > 0 && started.elapsed() < IDLE_WAIT {
        tokio::time::sleep(Duration::from_millis(15)).await;
    }
}

/// Writes the batch to the application database; returns the number of entries stored.
pub async fn commit(batch: BatchId) -> anyhow::Result<usize> {
    commit_with(&super::DB, batch).await
}

/// Writes the batch in one transaction, clears the redo stack and applies the retention cap.
pub async fn commit_with<C: Connection>(db: &Surreal<C>, batch: BatchId) -> anyhow::Result<usize> {
    let Some(pending) = lock(&PENDING).remove(&batch.0) else { return Ok(0) };
    IN_FLIGHT.fetch_sub(1, Ordering::SeqCst);
    let entries: Vec<NewEntry> = pending.entries.into_iter().filter(|e| e.before != e.after).collect();
    if entries.is_empty() {
        return Ok(0);
    }
    let _gate = GATE.lock().await;
    ensure_schema(db).await?;
    let at_ms = unix_ms();
    let rows: Vec<Row> = entries
        .into_iter()
        .map(|e| {
            let entry = Entry { seq: stamp(), batch, at_ms, source: pending.source.clone(), path: e.path, field: e.field, target: e.target, before: e.before, after: e.after, state: EntryState::Applied };
            Row::from_entry(&entry, None)
        })
        .collect();
    let count = rows.len();
    db.query("BEGIN; DELETE metadata_journal WHERE state = 'undone'; INSERT INTO metadata_journal $rows RETURN NONE; COMMIT;")
        .bind(("rows", rows))
        .await?
        .check()
        .context("writing the journal entries failed")?;
    bump();
    if let Err(e) = trim_locked(db, RETENTION).await {
        log::warn!("[journal] retention failed: {e:#}");
    }
    Ok(count)
}

/// Commits a batch and logs a failure instead of returning it; for writers that must not fail on a journal error.
pub async fn commit_quietly_with<C: Connection>(db: &Surreal<C>, batch: BatchId) -> usize {
    match commit_with(db, batch).await {
        Ok(n) => n,
        Err(e) => {
            log::warn!("[journal] could not record the change: {e:#}");
            0
        }
    }
}

/// Collects the changes of one edit and commits them, whatever connection the edit used.
pub struct Recorder {
    batch: BatchId,
    done: bool,
}

impl Recorder {
    pub fn new(source: &str) -> Self {
        Self { batch: begin_batch(source), done: false }
    }

    pub fn batch(&self) -> BatchId {
        self.batch
    }

    pub fn text(&self, path: &str, field: Field, before: Option<&str>, after: Option<&str>) {
        record(self.batch, path, field, Value::text(before), Value::text(after));
    }

    pub fn tags(&self, path: &str, before: &[String], after: &[String]) {
        record(self.batch, path, Field::Tags, Value::tags(before), Value::tags(after));
    }

    pub fn member(&self, path: &str, collection_key: &str, before: bool, after: bool) {
        record_member(self.batch, path, collection_key, before, after);
    }

    /// Stores the collected changes; a journal failure is logged and returns zero.
    pub async fn commit_with<C: Connection>(mut self, db: &Surreal<C>) -> usize {
        self.done = true;
        commit_quietly_with(db, self.batch).await
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        if !self.done {
            abandon(self.batch);
        }
    }
}

/// Direction of an undo run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Undo,
    Redo,
}

impl Direction {
    pub fn verb(self) -> &'static str {
        match self {
            Self::Undo => "Undid",
            Self::Redo => "Redid",
        }
    }

    fn nothing(self) -> &'static str {
        match self {
            Self::Undo => "Nothing to undo",
            Self::Redo => "Nothing to redo",
        }
    }
}

/// A value that was written back.
#[derive(Clone, Debug, PartialEq)]
pub struct Change {
    pub path: String,
    pub field: Field,
    pub target: Option<String>,
    pub value: Value,
}

/// An entry that was left alone because its row changed since or its target is gone.
#[derive(Clone, Debug, PartialEq)]
pub struct Conflict {
    pub path: String,
    pub field: Field,
    pub target: Option<String>,
    pub reason: String,
}

/// What an undo or redo did.
#[derive(Clone, Debug, PartialEq)]
pub struct UndoReport {
    pub direction: Direction,
    /// The batch that was reverted or re-applied; none when there was nothing to do.
    pub batch: Option<BatchId>,
    pub source: String,
    pub applied: Vec<Change>,
    pub conflicts: Vec<Conflict>,
}

impl UndoReport {
    fn empty(direction: Direction) -> Self {
        Self { direction, batch: None, source: String::new(), applied: Vec::new(), conflicts: Vec::new() }
    }

    /// True when there was no batch to revert.
    pub fn is_empty(&self) -> bool {
        self.batch.is_none()
    }

    /// Distinct files that got a value back.
    pub fn files(&self) -> usize {
        self.applied.iter().map(|c| c.path.as_str()).collect::<HashSet<_>>().len()
    }

    /// The sentence for the toast: counts of changes, files and skipped entries.
    pub fn summary(&self) -> String {
        if self.is_empty() {
            return self.direction.nothing().to_string();
        }
        let n = self.applied.len();
        let changes = if n == 1 { "1 change".to_string() } else { format!("{n} changes") };
        let mut text = format!("{} {changes}", self.direction.verb());
        if self.files() > 1 {
            text.push_str(&format!(" in {} files", self.files()));
        }
        if !self.source.is_empty() {
            text.push_str(&format!(" ({})", self.source));
        }
        if !self.conflicts.is_empty() {
            text.push_str(&format!("; {} skipped because the file changed since", self.conflicts.len()));
        }
        text
    }

    /// Row patches for the values that were written, one per path, for the rows shown in the browsers.
    pub fn patches(&self) -> Vec<(String, MetaPatch)> {
        let mut order: Vec<String> = Vec::new();
        let mut by_path: HashMap<String, MetaPatch> = HashMap::new();
        for change in &self.applied {
            let patch = by_path.entry(change.path.clone()).or_insert_with(|| {
                order.push(change.path.clone());
                MetaPatch::default()
            });
            match (&change.field, &change.value) {
                (Field::Caption, Value::Text(v)) => patch.caption = Some(v.clone()),
                (Field::Description, Value::Text(v)) => patch.description = Some(v.clone()),
                (Field::Category, Value::Text(v)) => patch.category = Some(v.clone()),
                (Field::Tags, Value::Tags(v)) => patch.tags = Some(v.clone()),
                _ => {}
            }
        }
        order.into_iter().filter_map(|p| by_path.remove(&p).filter(|m| !m.is_empty()).map(|m| (p, m))).collect()
    }

    /// Paths whose collection membership changed.
    pub fn membership_paths(&self) -> Vec<String> {
        let mut seen = HashSet::new();
        self.applied.iter().filter(|c| c.field == Field::Collection && seen.insert(c.path.clone())).map(|c| c.path.clone()).collect()
    }
}

/// The report of the most recent undo or redo in this process, for the Devtools window.
pub fn last_report() -> Option<UndoReport> {
    lock(&LAST_REPORT).clone()
}

/// One write the plan wants to make.
#[derive(Clone, Debug, PartialEq)]
pub struct Write {
    pub seq: i64,
    pub path: String,
    pub field: Field,
    pub target: Option<String>,
    pub value: Value,
}

/// An entry the plan leaves alone.
#[derive(Clone, Debug, PartialEq)]
pub struct Skip {
    pub seq: i64,
    pub conflict: Conflict,
}

/// What can be applied to the current values and what conflicts with them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Plan {
    pub writes: Vec<Write>,
    pub skips: Vec<Skip>,
}

/// Decides, entry by entry, whether `entries` can be undone or redone given the values the rows hold now.
pub fn plan(entries: &[Entry], direction: Direction, current: &HashMap<(String, Field, Option<String>), Value>) -> Plan {
    let mut ordered: Vec<&Entry> = entries.iter().collect();
    ordered.sort_by_key(|e| e.seq);
    if direction == Direction::Undo {
        ordered.reverse();
    }
    let mut now = current.clone();
    let mut plan = Plan::default();
    for e in ordered {
        let (expected, write) = match direction {
            Direction::Undo => (&e.after, &e.before),
            Direction::Redo => (&e.before, &e.after),
        };
        let skip = |reason: String| Skip { seq: e.seq, conflict: Conflict { path: e.path.clone(), field: e.field, target: e.target.clone(), reason } };
        match now.get(&e.key()) {
            None => plan.skips.push(skip("the file is not in the library any more".to_string())),
            Some(cur) if cur == expected => {
                now.insert(e.key(), write.clone());
                plan.writes.push(Write { seq: e.seq, path: e.path.clone(), field: e.field, target: e.target.clone(), value: write.clone() });
            }
            Some(cur) => plan.skips.push(skip(format!("it is now {} instead of {}", cur.brief(), expected.brief()))),
        }
    }
    plan
}

#[derive(Debug, SurrealValue)]
struct CurrentRow {
    path: String,
    caption: Option<String>,
    description: Option<String>,
    category: Option<String>,
    tags: Option<Vec<String>>,
}

/// The values the rows and memberships named by `entries` hold now.
async fn current_values<C: Connection>(db: &Surreal<C>, entries: &[Entry]) -> anyhow::Result<HashMap<Key, Value>> {
    let mut out: HashMap<Key, Value> = HashMap::new();
    let row_paths: Vec<String> = {
        let mut seen = HashSet::new();
        entries.iter().filter(|e| e.field != Field::Collection && seen.insert(e.path.clone())).map(|e| e.path.clone()).collect()
    };
    for chunk in row_paths.chunks(CHUNK) {
        let rows: Vec<CurrentRow> = db.query("SELECT path, caption, description, category, tags FROM thumbnails WHERE path IN $paths").bind(("paths", chunk.to_vec())).await?.check()?.take(0)?;
        for r in rows {
            out.insert((r.path.clone(), Field::Caption, None), Value::text(r.caption.as_deref()));
            out.insert((r.path.clone(), Field::Description, None), Value::text(r.description.as_deref()));
            out.insert((r.path.clone(), Field::Category, None), Value::text(r.category.as_deref()));
            out.insert((r.path, Field::Tags, None), Value::Tags(r.tags.unwrap_or_default()));
        }
    }
    let mut by_target: HashMap<String, Vec<String>> = HashMap::new();
    for e in entries.iter().filter(|e| e.field == Field::Collection) {
        if let Some(t) = &e.target {
            by_target.entry(t.clone()).or_default().push(e.path.clone());
        }
    }
    for (target, paths) in by_target {
        let id = collections::collection_id(&target);
        for chunk in paths.chunks(CHUNK) {
            let members: HashSet<String> = db
                .query("SELECT VALUE path FROM collection_items WHERE collection = $c AND path IN $paths")
                .bind(("c", id.clone()))
                .bind(("paths", chunk.to_vec()))
                .await?
                .check()?
                .take::<Vec<String>>(0)?
                .into_iter()
                .collect();
            for p in chunk {
                out.insert((p.clone(), Field::Collection, Some(target.clone())), Value::Member(members.contains(p)));
            }
        }
    }
    Ok(out)
}

#[derive(Clone, Debug, SurrealValue)]
struct TextUpdate {
    path: String,
    value: Option<String>,
}

#[derive(Clone, Debug, SurrealValue)]
struct TagUpdate {
    path: String,
    value: Vec<String>,
}

/// Writes the planned values; returns the seqs that could not be written and why.
async fn apply_writes<C: Connection>(db: &Surreal<C>, writes: &[Write]) -> Vec<(i64, String)> {
    let mut failed: Vec<(i64, String)> = Vec::new();
    for field in [Field::Caption, Field::Description, Field::Category] {
        let batch: Vec<&Write> = writes.iter().filter(|w| w.field == field).collect();
        for chunk in batch.chunks(CHUNK) {
            let updates: Vec<TextUpdate> = chunk.iter().map(|w| TextUpdate { path: w.path.clone(), value: match &w.value { Value::Text(v) => v.clone(), _ => None } }).collect();
            let sql = format!("FOR $u IN $updates {{ UPDATE thumbnails SET {0} = $u.value WHERE path = $u.path; }};", field.as_str());
            let res = super::thumb_blobs::retry_conflicts(|| async { db.query(&sql).bind(("updates", updates.clone())).await?.check()?; Ok(()) }).await;
            if let Err(e) = res {
                failed.extend(chunk.iter().map(|w| (w.seq, format!("{e:#}"))));
            }
        }
    }
    let tag_writes: Vec<&Write> = writes.iter().filter(|w| w.field == Field::Tags).collect();
    for chunk in tag_writes.chunks(CHUNK) {
        let updates: Vec<TagUpdate> = chunk.iter().map(|w| TagUpdate { path: w.path.clone(), value: match &w.value { Value::Tags(t) => t.clone(), _ => Vec::new() } }).collect();
        let res = super::thumb_blobs::retry_conflicts(|| async {
            db.query("FOR $u IN $updates { UPDATE thumbnails SET tags = $u.value WHERE path = $u.path; };").bind(("updates", updates.clone())).await?.check()?;
            Ok(())
        })
        .await;
        if let Err(e) = res {
            failed.extend(chunk.iter().map(|w| (w.seq, format!("{e:#}"))));
        }
    }
    let mut targets: Vec<String> = writes.iter().filter(|w| w.field == Field::Collection).filter_map(|w| w.target.clone()).collect();
    targets.sort();
    targets.dedup();
    for target in targets {
        let id = collections::collection_id(&target);
        for want in [true, false] {
            let group: Vec<&Write> = writes.iter().filter(|w| w.field == Field::Collection && w.target.as_deref() == Some(target.as_str()) && w.value == Value::Member(want)).collect();
            if group.is_empty() {
                continue;
            }
            let paths: Vec<String> = group.iter().map(|w| w.path.clone()).collect();
            let res = if want { collections::add_paths(db, &id, &paths).await.map(|_| ()) } else { collections::remove_paths(db, &id, &paths).await.map(|_| ()) };
            if let Err(e) = res {
                failed.extend(group.iter().map(|w| (w.seq, format!("{e:#}"))));
            }
        }
    }
    failed
}

async fn set_states<C: Connection>(db: &Surreal<C>, seqs: &[i64], state: EntryState, undone_seq: Option<i64>) -> anyhow::Result<()> {
    for chunk in seqs.chunks(CHUNK) {
        let ids: Vec<RecordId> = chunk.iter().map(|s| RecordId::new(TABLE, s.to_string())).collect();
        db.query("UPDATE $ids SET state = $state, undone_seq = $undone RETURN NONE").bind(("ids", ids)).bind(("state", state.as_str().to_string())).bind(("undone", undone_seq)).await?.check().context("updating the journal entries failed")?;
    }
    Ok(())
}

async fn read_entries<C: Connection>(db: &Surreal<C>, sql: &str, binds: Vec<(&'static str, i64)>) -> anyhow::Result<Vec<Entry>> {
    let mut query = db.query(sql);
    for (name, value) in binds {
        query = query.bind((name, value));
    }
    let rows: Vec<Row> = query.await?.check()?.take(0)?;
    Ok(rows.into_iter().filter_map(Row::into_entry).collect())
}

/// Reverts the newest applied batch of the application database.
pub async fn undo_last() -> anyhow::Result<UndoReport> {
    undo_last_with(&super::DB).await
}

/// Re-applies the batch undone most recently in the application database.
pub async fn redo_last() -> anyhow::Result<UndoReport> {
    redo_last_with(&super::DB).await
}

/// Reverts the newest batch that still has applied entries.
pub async fn undo_last_with<C: Connection>(db: &Surreal<C>) -> anyhow::Result<UndoReport> {
    run_direction(db, Direction::Undo).await
}

/// Re-applies the batch that was undone most recently.
pub async fn redo_last_with<C: Connection>(db: &Surreal<C>) -> anyhow::Result<UndoReport> {
    run_direction(db, Direction::Redo).await
}

async fn run_direction<C: Connection>(db: &Surreal<C>, direction: Direction) -> anyhow::Result<UndoReport> {
    wait_idle().await;
    let _gate = GATE.lock().await;
    ensure_schema(db).await?;
    let head_sql = match direction {
        Direction::Undo => "SELECT * FROM metadata_journal WHERE state = 'applied' ORDER BY seq DESC LIMIT 1",
        Direction::Redo => "SELECT * FROM metadata_journal WHERE state = 'undone' ORDER BY undone_seq DESC LIMIT 1",
    };
    let Some(head) = read_entries(db, head_sql, Vec::new()).await?.into_iter().next() else {
        let report = UndoReport::empty(direction);
        *lock(&LAST_REPORT) = Some(report.clone());
        return Ok(report);
    };
    let wanted = if direction == Direction::Undo { "applied" } else { "undone" };
    let entries = read_entries(db, &format!("SELECT * FROM metadata_journal WHERE batch = $batch AND state = '{wanted}' ORDER BY seq"), vec![("batch", head.batch.0)]).await?;
    let current = current_values(db, &entries).await?;
    let plan = plan(&entries, direction, &current);
    let failed = apply_writes(db, &plan.writes).await;
    let failed_seqs: HashSet<i64> = failed.iter().map(|(s, _)| *s).collect();
    let done_seqs: Vec<i64> = plan.writes.iter().map(|w| w.seq).filter(|s| !failed_seqs.contains(s)).collect();
    let mut skipped_seqs: Vec<i64> = plan.skips.iter().map(|s| s.seq).collect();
    skipped_seqs.extend(failed_seqs.iter().copied());
    match direction {
        Direction::Undo => set_states(db, &done_seqs, EntryState::Undone, Some(stamp())).await?,
        Direction::Redo => set_states(db, &done_seqs, EntryState::Applied, None).await?,
    }
    set_states(db, &skipped_seqs, EntryState::Skipped, None).await?;
    let mut conflicts: Vec<Conflict> = plan.skips.into_iter().map(|s| s.conflict).collect();
    let by_seq: HashMap<i64, &Entry> = entries.iter().map(|e| (e.seq, e)).collect();
    for (seq, why) in &failed {
        if let Some(e) = by_seq.get(seq) {
            conflicts.push(Conflict { path: e.path.clone(), field: e.field, target: e.target.clone(), reason: why.clone() });
        }
    }
    let applied: Vec<Change> = plan.writes.into_iter().filter(|w| !failed_seqs.contains(&w.seq)).map(|w| Change { path: w.path, field: w.field, target: w.target, value: w.value }).collect();
    let report = UndoReport { direction, batch: Some(head.batch), source: head.source, applied, conflicts };
    bump();
    *lock(&LAST_REPORT) = Some(report.clone());
    Ok(report)
}

/// The newest entries of the application database, newest first.
pub async fn recent(n: usize) -> anyhow::Result<Vec<Entry>> {
    recent_with(&super::DB, n).await
}

/// The newest `n` entries, newest first.
pub async fn recent_with<C: Connection>(db: &Surreal<C>, n: usize) -> anyhow::Result<Vec<Entry>> {
    ensure_schema(db).await?;
    read_entries(db, "SELECT * FROM metadata_journal ORDER BY seq DESC LIMIT $n", vec![("n", n as i64)]).await
}

/// One batch as listed in Devtools.
#[derive(Clone, Debug, PartialEq)]
pub struct BatchSummary {
    pub batch: BatchId,
    pub source: String,
    pub at_ms: i64,
    pub entries: usize,
    pub files: usize,
    pub applied: usize,
    pub undone: usize,
    pub skipped: usize,
}

/// The newest `n` batches, newest first.
pub async fn batches_with<C: Connection>(db: &Surreal<C>, n: usize) -> anyhow::Result<Vec<BatchSummary>> {
    ensure_schema(db).await?;
    let entries = read_entries(db, "SELECT * FROM metadata_journal ORDER BY seq", Vec::new()).await?;
    Ok(summarize(&entries, n))
}

/// Groups entries into batches, newest first.
pub fn summarize(entries: &[Entry], n: usize) -> Vec<BatchSummary> {
    let mut order: Vec<BatchId> = Vec::new();
    let mut map: HashMap<BatchId, (BatchSummary, HashSet<&str>)> = HashMap::new();
    for e in entries {
        let slot = map.entry(e.batch).or_insert_with(|| {
            order.push(e.batch);
            (BatchSummary { batch: e.batch, source: e.source.clone(), at_ms: e.at_ms, entries: 0, files: 0, applied: 0, undone: 0, skipped: 0 }, HashSet::new())
        });
        slot.0.entries += 1;
        slot.1.insert(e.path.as_str());
        match e.state {
            EntryState::Applied => slot.0.applied += 1,
            EntryState::Undone => slot.0.undone += 1,
            EntryState::Skipped => slot.0.skipped += 1,
        }
    }
    let mut out: Vec<BatchSummary> = order
        .into_iter()
        .filter_map(|b| map.remove(&b))
        .map(|(mut s, files)| {
            s.files = files.len();
            s
        })
        .collect();
    out.sort_by_key(|s| std::cmp::Reverse(s.batch));
    out.truncate(n);
    out
}

/// Totals over the journal.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub entries: usize,
    pub batches: usize,
    pub applied: usize,
    pub undone: usize,
    pub skipped: usize,
}

/// Counts of entries by state and the number of batches.
pub async fn stats_with<C: Connection>(db: &Surreal<C>) -> anyhow::Result<Stats> {
    ensure_schema(db).await?;
    let entries = read_entries(db, "SELECT * FROM metadata_journal", Vec::new()).await?;
    let mut stats = Stats { entries: entries.len(), ..Default::default() };
    stats.batches = entries.iter().map(|e| e.batch).collect::<HashSet<_>>().len();
    for e in &entries {
        match e.state {
            EntryState::Applied => stats.applied += 1,
            EntryState::Undone => stats.undone += 1,
            EntryState::Skipped => stats.skipped += 1,
        }
    }
    Ok(stats)
}

/// Removes every entry of the application database.
pub async fn clear() -> anyhow::Result<usize> {
    clear_with(&super::DB).await
}

/// Removes every entry; returns how many there were.
pub async fn clear_with<C: Connection>(db: &Surreal<C>) -> anyhow::Result<usize> {
    let _gate = GATE.lock().await;
    ensure_schema(db).await?;
    let rows: Vec<CountRow> = db.query("SELECT count() AS n FROM metadata_journal GROUP ALL").await?.check()?.take(0)?;
    let n = rows.first().map_or(0, |r| r.n.max(0) as usize);
    db.query("DELETE metadata_journal RETURN NONE").await?.check()?;
    bump();
    Ok(n)
}

/// Drops the oldest whole batches while more than `cap` entries are stored; the newest batch always stays. Returns the entries dropped.
pub async fn trim_with<C: Connection>(db: &Surreal<C>, cap: usize) -> anyhow::Result<usize> {
    let _gate = GATE.lock().await;
    ensure_schema(db).await?;
    trim_locked(db, cap).await
}

async fn trim_locked<C: Connection>(db: &Surreal<C>, cap: usize) -> anyhow::Result<usize> {
    let rows: Vec<CountRow> = db.query("SELECT count() AS n FROM metadata_journal GROUP ALL").await?.check()?.take(0)?;
    let total = rows.first().map_or(0, |r| r.n.max(0) as usize);
    if total <= cap {
        return Ok(0);
    }
    let seqs: Vec<BatchSeq> = db.query("SELECT batch, seq FROM metadata_journal").await?.check()?.take(0)?;
    let mut per_batch: HashMap<i64, (usize, i64)> = HashMap::new();
    for s in seqs {
        let slot = per_batch.entry(s.batch).or_insert((0, i64::MAX));
        slot.0 += 1;
        slot.1 = slot.1.min(s.seq);
    }
    let mut oldest_first: Vec<(i64, usize, i64)> = per_batch.into_iter().map(|(b, (n, first))| (b, n, first)).collect();
    oldest_first.sort_by_key(|(_, _, first)| *first);
    let (mut excess, mut dropped, mut left) = (total - cap, 0usize, oldest_first.len());
    let mut doomed: Vec<i64> = Vec::new();
    for (batch, n, _) in oldest_first {
        if excess == 0 || left <= 1 {
            break;
        }
        doomed.push(batch);
        excess = excess.saturating_sub(n);
        dropped += n;
        left -= 1;
    }
    if !doomed.is_empty() {
        db.query("DELETE metadata_journal WHERE batch IN $batches RETURN NONE").bind(("batches", doomed)).await?.check()?;
        bump();
    }
    Ok(dropped)
}

#[cfg(test)]
mod tests;
