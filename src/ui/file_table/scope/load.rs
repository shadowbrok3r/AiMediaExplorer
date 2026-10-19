//! Scope loaders: stream the rows of a non-filesystem scope from the database in batches.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use surrealdb::types::Datetime;
use surrealdb::{Connection, Surreal};

use super::duplicates::{self, Hasher, SizeIndex};
use super::failed;
use crate::command::{Scope, SmartView};
use crate::database::Thumbnail;
use crate::database::clip_embeddings::{knn_with, similarity_of};
use crate::database::collections;
use crate::database::scope_queries::{self as queries, PAGE, RECENT_LIMIT};

/// Rows per batch sent to the view.
pub const BATCH: usize = 500;
/// CLIP checks of earlier pages that run while the next page is read.
const CHECKS_IN_FLIGHT: usize = 4;
/// Results one page of a semantic or similarity search holds.
pub const SEARCH_PAGE: usize = 100;
/// Effort of the nearest-neighbour search.
const SEARCH_EF: usize = 128;

/// Turns text into the CLIP vector it is searched by.
#[async_trait::async_trait]
pub trait TextEmbedder: Send + Sync {
    async fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>>;
}

/// The CLIP engine of the application.
pub struct EngineEmbedder;

#[async_trait::async_trait]
impl TextEmbedder for EngineEmbedder {
    async fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        crate::ai::GLOBAL_AI_ENGINE.ensure_clip_engine().await?;
        let mut guard = crate::ai::GLOBAL_AI_ENGINE.clip_engine.lock().await;
        let engine = guard.as_mut().ok_or_else(|| anyhow::anyhow!("the CLIP engine is not loaded"))?;
        engine.embed_text(text)
    }
}

/// What to load.
#[derive(Clone, Debug)]
pub struct Request {
    pub scope: Scope,
    /// Results already shown, for the next page of a search.
    pub start: usize,
    /// Query vector of an earlier page of the same search.
    pub vector: Option<Arc<Vec<f32>>>,
}

impl Request {
    pub fn first(scope: Scope) -> Self {
        Self { scope, start: 0, vector: None }
    }
}

/// Time a load spent waiting for each kind of query.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stages {
    /// Reading pages of rows.
    pub page: Duration,
    /// Checking which rows have a CLIP embedding.
    pub embed: Duration,
}

/// Rows for the view with what the loader knows about them.
#[derive(Clone, Debug, Default)]
pub struct RowBatch {
    /// Time the CLIP presence check of this batch took.
    pub embed_time: Duration,
    pub rows: Vec<Thumbnail>,
    /// Similarity of rows to the query, for search scopes.
    pub scores: Vec<(String, f32)>,
    /// Paths of rows that are not in the library.
    pub unindexed: Vec<String>,
    /// Paths of rows that have a CLIP embedding.
    pub embedded: Vec<String>,
}

/// How a load ended.
#[derive(Clone, Debug, Default)]
pub struct Outcome {
    pub rows: usize,
    pub batches: usize,
    /// More results can be loaded after these.
    pub more: bool,
    /// Where the next page of a search starts.
    pub next_start: usize,
    pub vector: Option<Arc<Vec<f32>>>,
    /// Rows the scope holds, when known up front.
    pub total: Option<u64>,
    pub first_batch: Option<Duration>,
    pub elapsed: Duration,
    pub stages: Stages,
    pub cancelled: bool,
    /// Better name for the tab than the scope gives, such as the title of a cached scan.
    pub title: Option<String>,
}

/// What a loader uses beside the database.
#[derive(Clone)]
pub struct Env {
    pub cancel: Arc<AtomicBool>,
    pub embedder: Arc<dyn TextEmbedder>,
    pub hasher: Hasher,
    pub progress: Arc<dyn Fn(u64, u64) + Send + Sync>,
}

impl Env {
    pub fn live(cancel: Arc<AtomicBool>, progress: Arc<dyn Fn(u64, u64) + Send + Sync>) -> Self {
        Self { cancel, embedder: Arc::new(EngineEmbedder), hasher: Arc::new(duplicates::file_hash), progress }
    }

    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

/// Collects the counters of a load while batches go out.
struct Run<'a> {
    emit: &'a mut (dyn FnMut(RowBatch) -> bool + Send),
    out: Outcome,
    started: Instant,
    open: bool,
}

impl<'a> Run<'a> {
    fn new(emit: &'a mut (dyn FnMut(RowBatch) -> bool + Send)) -> Self {
        Self { emit, out: Outcome::default(), started: Instant::now(), open: true }
    }

    /// Sends a batch; false once the receiver is gone.
    fn send(&mut self, batch: RowBatch) -> bool {
        if batch.rows.is_empty() {
            return self.open;
        }
        self.out.rows += batch.rows.len();
        self.out.batches += 1;
        self.out.stages.embed += batch.embed_time;
        self.out.first_batch.get_or_insert_with(|| self.started.elapsed());
        self.open = (self.emit)(batch);
        self.open
    }

    fn finish(mut self, env: &Env) -> Outcome {
        self.out.elapsed = self.started.elapsed();
        self.out.cancelled = env.cancelled();
        self.out
    }
}

/// A row for a file that has none in the library, read from the file system.
pub fn row_from_path(path: &str) -> Thumbnail {
    let p = std::path::Path::new(path);
    let filename = p.file_name().and_then(|n| n.to_str()).unwrap_or(path).to_string();
    let mut row = Thumbnail::new(&filename);
    let meta = if path.contains("://") { None } else { std::fs::metadata(p).ok() };
    row.path = path.to_string();
    row.parent_dir = p.parent().map(|d| d.to_string_lossy().to_string()).unwrap_or_default();
    row.file_type = match &meta {
        Some(m) if m.is_dir() => "<DIR>".to_string(),
        _ => p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).unwrap_or_default(),
    };
    if let Some(m) = &meta {
        row.size = m.len();
        row.modified = m.modified().ok().map(|t| Datetime::from(chrono::DateTime::<chrono::Utc>::from(t)));
    }
    row.filename = filename;
    row
}

/// Rows for `paths` in their order: library rows where they exist, file system rows for the rest.
async fn rows_with_fallback<C: Connection>(db: &Surreal<C>, paths: &[String]) -> anyhow::Result<(Vec<Thumbnail>, Vec<String>)> {
    let mut found: std::collections::HashMap<String, Thumbnail> = queries::rows_of_paths(db, paths).await?.into_iter().map(|r| (r.path.clone(), r)).collect();
    let mut rows = Vec::with_capacity(paths.len());
    let mut missing: Vec<String> = Vec::new();
    for path in paths {
        match found.remove(path) {
            Some(row) => rows.push(row),
            None => missing.push(path.clone()),
        }
    }
    let unindexed = missing.clone();
    let synthesized = tokio::task::spawn_blocking(move || missing.iter().map(|p| row_from_path(p)).collect::<Vec<_>>()).await?;
    rows.extend(synthesized);
    Ok((rows, unindexed))
}

async fn batch_of<C: Connection>(db: &Surreal<C>, rows: Vec<Thumbnail>) -> anyhow::Result<RowBatch> {
    batch_with(db, rows, Presence::Lookup).await
}

/// How the CLIP presence of the rows of a batch is found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Presence {
    /// Not at all.
    Skip,
    /// One index lookup per row.
    Lookup,
    /// One scan of the index over the range the rows span.
    Range,
}

async fn batch_with<C: Connection>(db: &Surreal<C>, rows: Vec<Thumbnail>, presence: Presence) -> anyhow::Result<RowBatch> {
    if presence == Presence::Skip {
        return Ok(RowBatch { rows, ..Default::default() });
    }
    let started = Instant::now();
    let paths: Vec<String> = rows.iter().map(|r| r.path.clone()).collect();
    let found = match presence {
        Presence::Range => queries::embedded_in_range(db, &paths).await?,
        _ => queries::embedded_paths(db, &paths).await?,
    };
    Ok(RowBatch { rows, embedded: found.into_iter().collect(), embed_time: started.elapsed(), ..Default::default() })
}

fn progress(env: &Env, done: usize, total: Option<u64>) {
    (env.progress)(done as u64, total.unwrap_or(0));
}

/// Waits until the database is open with its namespace selected, or `timeout` passes.
pub async fn wait_for_connection(cancel: &AtomicBool, timeout: Duration) -> bool {
    let until = Instant::now() + timeout;
    while !crate::database::is_ready() {
        if cancel.load(Ordering::Relaxed) || Instant::now() >= until {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    true
}

/// Loads the scope of `req`, calling `emit` with each batch; `emit` returns false to stop.
pub async fn run<C: Connection + 'static>(db: &Surreal<C>, req: &Request, env: &Env, emit: &mut (dyn FnMut(RowBatch) -> bool + Send)) -> anyhow::Result<Outcome> {
    let mut run = Run::new(emit);
    match &req.scope {
        Scope::Folder(_) | Scope::Archive(_) => anyhow::bail!("folders and archives are listed by the file scanner"),
        Scope::Library => library(db, env, &mut run).await?,
        Scope::Recent => {
            let rows = queries::recent_rows(db, RECENT_LIMIT).await?;
            run.out.total = Some(rows.len() as u64);
            for chunk in rows.chunks(BATCH) {
                if env.cancelled() || !run.send(batch_of(db, chunk.to_vec()).await?) {
                    break;
                }
            }
        }
        Scope::Collection(name) => collection(db, name, env, &mut run).await?,
        Scope::Tag(tag) => keyset(db, env, &mut run, |after| async move { queries::tag_page(db, tag, after.as_deref(), BATCH).await }).await?,
        Scope::Category(category) => keyset(db, env, &mut run, |after| async move { queries::category_page(db, category, after.as_deref(), BATCH).await }).await?,
        Scope::Scan(id) => scan(db, *id, env, &mut run).await?,
        Scope::Smart(view) => smart(db, *view, env, &mut run).await?,
        Scope::Semantic(text) => {
            let vector = match &req.vector {
                Some(v) => v.clone(),
                None => Arc::new(env.embedder.embed(text).await?),
            };
            search(db, &vector, None, req.start, &mut run).await?;
        }
        Scope::Similar(path) => {
            let vector = match &req.vector {
                Some(v) => v.clone(),
                None => Arc::new(queries::embedding_of(db, path).await?.ok_or_else(|| anyhow::anyhow!("{path} has no CLIP embedding yet; generate one first"))?),
            };
            search(db, &vector, Some(path), req.start, &mut run).await?;
        }
    }
    Ok(run.finish(env))
}

async fn library<C: Connection + 'static>(db: &Surreal<C>, env: &Env, run: &mut Run<'_>) -> anyhow::Result<()> {
    let total = queries::count_library(db).await?;
    run.out.total = Some(total);
    stream_pages(db, env, run, Presence::Range, |after| async move { queries::library_page(db, after.as_deref(), BATCH).await }).await
}

/// Reads keyset pages until the last one, sending each as a batch.
async fn keyset<C, F, Fut>(db: &Surreal<C>, env: &Env, run: &mut Run<'_>, page: F) -> anyhow::Result<()>
where
    C: Connection + 'static,
    F: FnMut(Option<String>) -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<queries::KeysetPage>>,
{
    stream_pages(db, env, run, Presence::Lookup, page).await
}

/// A page read and how long it took.
async fn timed<Fut>(read: Fut) -> (anyhow::Result<queries::KeysetPage>, Duration)
where
    Fut: std::future::Future<Output = anyhow::Result<queries::KeysetPage>>,
{
    let started = Instant::now();
    let got = read.await;
    (got, started.elapsed())
}

/// Sends keyset pages as batches in page order, reading ahead while up to [`CHECKS_IN_FLIGHT`] earlier pages are checked.
async fn stream_pages<C, F, Fut>(db: &Surreal<C>, env: &Env, run: &mut Run<'_>, presence: Presence, mut page: F) -> anyhow::Result<()>
where
    C: Connection + 'static,
    F: FnMut(Option<String>) -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<queries::KeysetPage>>,
{
    let mut checks: VecDeque<tokio::task::JoinHandle<anyhow::Result<RowBatch>>> = VecDeque::new();
    let mut after: Option<String> = None;
    let mut reading = true;
    let result = loop {
        while reading && checks.len() < CHECKS_IN_FLIGHT && !env.cancelled() {
            let (got, took) = timed(page(after.take())).await;
            run.out.stages.page += took;
            let queries::KeysetPage { rows, next } = match got {
                Ok(got) => got,
                Err(e) => break_with(&mut checks, e)?,
            };
            reading = next.is_some();
            after = next;
            let db = db.clone();
            checks.push_back(tokio::spawn(async move { batch_with(&db, rows, presence).await }));
        }
        let Some(oldest) = checks.pop_front() else { break Ok(()) };
        let waiting = Instant::now();
        let mut batch = match oldest.await {
            Ok(Ok(batch)) => batch,
            Ok(Err(e)) => break Err(e),
            Err(e) => break Err(anyhow::anyhow!("a page check stopped: {e}")),
        };
        if presence != Presence::Skip {
            // Time the load waited for this check.
            batch.embed_time = waiting.elapsed();
        }
        if !run.send(batch) || env.cancelled() {
            break Ok(());
        }
        progress(env, run.out.rows, run.out.total);
        tokio::task::yield_now().await;
    };
    checks.drain(..).for_each(|check| check.abort());
    result
}

// Stops the checks that run and hands `error` on.
fn break_with(checks: &mut VecDeque<tokio::task::JoinHandle<anyhow::Result<RowBatch>>>, error: anyhow::Error) -> anyhow::Result<queries::KeysetPage> {
    checks.drain(..).for_each(|check| check.abort());
    Err(error)
}

async fn collection<C: Connection>(db: &Surreal<C>, name: &str, env: &Env, run: &mut Run<'_>) -> anyhow::Result<()> {
    let found = collections::get_by_name(db, name).await?.ok_or_else(|| anyhow::anyhow!("no collection named \"{name}\""))?;
    let total = collections::count_members(db, &found.id).await?;
    run.out.total = Some(total);
    let mut offset = 0;
    while !env.cancelled() {
        let paths = collections::members(db, &found.id, offset, BATCH).await?;
        if paths.is_empty() {
            break;
        }
        offset += paths.len();
        let (rows, unindexed) = rows_with_fallback(db, &paths).await?;
        let mut batch = batch_of(db, rows).await?;
        batch.unindexed = unindexed;
        if !run.send(batch) {
            break;
        }
        progress(env, run.out.rows, run.out.total);
        if paths.len() < BATCH {
            break;
        }
        tokio::task::yield_now().await;
    }
    Ok(())
}

async fn scan<C: Connection>(db: &Surreal<C>, id: u64, env: &Env, run: &mut Run<'_>) -> anyhow::Result<()> {
    let record = queries::cached_scan(db, id).await?.ok_or_else(|| anyhow::anyhow!("no cached scan #{id}"))?;
    run.out.title = queries::cached_scan_title(db, &record).await?;
    let mut offset = 0;
    while !env.cancelled() {
        let paths = queries::cached_scan_paths(db, &record, offset, BATCH).await?;
        if paths.is_empty() {
            break;
        }
        offset += paths.len();
        let (rows, unindexed) = rows_with_fallback(db, &paths).await?;
        let mut batch = batch_of(db, rows).await?;
        batch.unindexed = unindexed;
        if !run.send(batch) {
            break;
        }
        progress(env, run.out.rows, None);
        if paths.len() < BATCH {
            break;
        }
        tokio::task::yield_now().await;
    }
    Ok(())
}

async fn smart<C: Connection + 'static>(db: &Surreal<C>, view: SmartView, env: &Env, run: &mut Run<'_>) -> anyhow::Result<()> {
    match view {
        SmartView::FailedThumbs => {
            let failed = failed::snapshot();
            run.out.total = Some(failed.len() as u64);
            let rows = crate::command::sink::failed_thumb_rows(&failed);
            let unindexed = rows.iter().map(|r| r.path.clone()).collect();
            run.send(RowBatch { rows, unindexed, ..Default::default() });
            Ok(())
        }
        SmartView::Duplicates => duplicates_view(db, env, run).await,
        SmartView::NoClip | SmartView::Undescribed | SmartView::Untagged => stream_pages(db, env, run, if view == SmartView::NoClip { Presence::Skip } else { Presence::Lookup }, |after| async move { queries::smart_page(db, view, after.as_deref(), BATCH).await }).await,
    }
}

async fn duplicates_view<C: Connection>(db: &Surreal<C>, env: &Env, run: &mut Run<'_>) -> anyhow::Result<()> {
    let total = queries::count_library(db).await?;
    let mut index = SizeIndex::default();
    let mut after: Option<String> = None;
    let mut read = 0usize;
    loop {
        if env.cancelled() {
            return Ok(());
        }
        let (rows, next) = queries::size_rows_page(db, after.as_deref(), PAGE).await?;
        read += rows.len();
        index.add(rows);
        (env.progress)(read as u64, total);
        match next {
            Some(n) => after = Some(n),
            None => break,
        }
        tokio::task::yield_now().await;
    }
    let candidates = index.candidates();
    let mut pending: Vec<Thumbnail> = Vec::new();
    let mut hashes: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    for group in candidates {
        if env.cancelled() {
            return Ok(());
        }
        for found in duplicates::confirm(group, env.hasher.clone(), &env.cancel).await {
            let paths = found.paths.clone();
            let (rows, _) = rows_with_fallback(db, &paths).await?;
            for row in rows {
                hashes.insert(row.path.clone(), found.hash.clone());
                pending.push(row);
            }
        }
        if pending.len() >= BATCH {
            flush_duplicates(db, run, &mut pending, &hashes).await?;
        }
    }
    flush_duplicates(db, run, &mut pending, &hashes).await
}

async fn flush_duplicates<C: Connection>(db: &Surreal<C>, run: &mut Run<'_>, pending: &mut Vec<Thumbnail>, hashes: &std::collections::HashMap<String, String>) -> anyhow::Result<()> {
    if pending.is_empty() {
        return Ok(());
    }
    let mut rows = std::mem::take(pending);
    for row in &mut rows {
        row.hash = hashes.get(&row.path).cloned().or(row.hash.take());
    }
    run.send(batch_of(db, rows).await?);
    Ok(())
}

/// One page of nearest neighbours of `vector`, skipping `origin` itself.
async fn search<C: Connection>(db: &Surreal<C>, vector: &Arc<Vec<f32>>, origin: Option<&String>, start: usize, run: &mut Run<'_>) -> anyhow::Result<()> {
    let hits = knn_with(db, vector, SEARCH_PAGE, SEARCH_EF, start).await?;
    let got = hits.len();
    let mut rows = Vec::with_capacity(got);
    let mut scores = Vec::with_capacity(got);
    let mut missing: Vec<String> = Vec::new();
    for hit in hits {
        if origin == Some(&hit.path) {
            continue;
        }
        scores.push((hit.path.clone(), similarity_of(hit.dist)));
        match hit.thumb_ref {
            Some(row) => rows.push(row),
            None => missing.push(hit.path),
        }
    }
    if !missing.is_empty() {
        rows.extend(queries::rows_of_paths(db, &missing).await?);
    }
    let embedded = rows.iter().map(|r| r.path.clone()).collect();
    run.out.more = got >= SEARCH_PAGE;
    run.out.next_start = start + got;
    run.out.vector = Some(vector.clone());
    run.send(RowBatch { rows, scores, embedded, ..Default::default() });
    Ok(())
}

#[cfg(test)]
mod tests;
