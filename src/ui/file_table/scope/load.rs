//! Scope loaders: stream the rows of a non-filesystem scope from the database in batches.

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

/// Rows for the view with what the loader knows about them.
#[derive(Clone, Debug, Default)]
pub struct RowBatch {
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
    let paths: Vec<String> = rows.iter().map(|r| r.path.clone()).collect();
    let embedded = queries::embedded_paths(db, &paths).await?.into_iter().collect();
    Ok(RowBatch { rows, embedded, ..Default::default() })
}

fn progress(env: &Env, done: usize, total: Option<u64>) {
    (env.progress)(done as u64, total.unwrap_or(0));
}

/// Waits until the database handle is connected or `timeout` passes.
pub async fn wait_for_connection(cancel: &AtomicBool, timeout: Duration) -> bool {
    let until = Instant::now() + timeout;
    while !crate::database::is_connected() {
        if cancel.load(Ordering::Relaxed) || Instant::now() >= until {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    true
}

/// Loads the scope of `req`, calling `emit` with each batch; `emit` returns false to stop.
pub async fn run<C: Connection>(db: &Surreal<C>, req: &Request, env: &Env, emit: &mut (dyn FnMut(RowBatch) -> bool + Send)) -> anyhow::Result<Outcome> {
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

async fn library<C: Connection>(db: &Surreal<C>, env: &Env, run: &mut Run<'_>) -> anyhow::Result<()> {
    let total = queries::count_library(db).await?;
    run.out.total = Some(total);
    keyset(db, env, run, |after| async move { queries::library_page(db, after.as_deref(), BATCH).await }).await
}

/// Reads keyset pages until the last one, sending each as a batch.
async fn keyset<C, F, Fut>(db: &Surreal<C>, env: &Env, run: &mut Run<'_>, mut page: F) -> anyhow::Result<()>
where
    C: Connection,
    F: FnMut(Option<String>) -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<queries::KeysetPage>>,
{
    let mut after: Option<String> = None;
    loop {
        if env.cancelled() {
            return Ok(());
        }
        let got = page(after.clone()).await?;
        let last = got.is_last();
        after = got.next.clone();
        if !run.send(batch_of(db, got.rows).await?) {
            return Ok(());
        }
        progress(env, run.out.rows, run.out.total);
        if last {
            return Ok(());
        }
        tokio::task::yield_now().await;
    }
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

async fn smart<C: Connection>(db: &Surreal<C>, view: SmartView, env: &Env, run: &mut Run<'_>) -> anyhow::Result<()> {
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
        SmartView::NoClip | SmartView::Undescribed | SmartView::Untagged => {
            let mut after: Option<String> = None;
            loop {
                if env.cancelled() {
                    return Ok(());
                }
                let page = queries::smart_page(db, view, after.as_deref(), BATCH).await?;
                let last = page.is_last();
                after = page.next.clone();
                let mut batch = batch_of(db, page.rows).await?;
                if view == SmartView::NoClip {
                    batch.embedded.clear();
                }
                if !run.send(batch) {
                    return Ok(());
                }
                progress(env, run.out.rows, None);
                if last {
                    return Ok(());
                }
                tokio::task::yield_now().await;
            }
        }
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
