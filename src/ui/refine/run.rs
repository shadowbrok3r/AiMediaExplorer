//! The background run that generates proposals, reported in the task list with progress and a cancel handle.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crossbeam::channel::Sender;

use crate::ai::refine::{self, HeuristicReranker, JinaReranker, ProposalGenerator, RICH_ROW_LIMIT, RerankerType, Reranker, RowScope, RunCtl};
use crate::status::GlobalStatusIndicator as _;
use crate::tasks::{self, TaskKind, TaskSpec};

use super::accept::Outcome;
use super::model::{Proposal, ProposalKey};

/// Proposals sent to the panel at a time.
const BATCH: usize = 12;
/// Files a cloud run asks about at most, to bound the cost.
const CLOUD_LIMIT: usize = 64;

/// What a background job tells the panel.
#[derive(Debug)]
pub enum Msg {
    Progress { done: u64, total: u64 },
    Generated(Vec<refine::Proposal>),
    Finished { cancelled: bool, scanned: u64 },
    Failed(String),
    /// The pending proposals of the store.
    Stored(Result<Vec<Proposal>, String>),
    Applied(Outcome),
    Rejected(Outcome),
}

/// The provider and model of a cloud run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CloudChoice {
    pub provider: String,
    pub model: String,
}

/// What one run does.
#[derive(Clone, Debug)]
pub struct RunSpec {
    pub reranker: RerankerType,
    pub cloud: Option<CloudChoice>,
    /// The files to refine; without it the library's described files are used.
    pub scope: Option<Vec<String>>,
}

impl RunSpec {
    /// Task title of the run.
    pub fn title(&self) -> String {
        match (&self.cloud, &self.scope) {
            (Some(c), _) => format!("Generating refinements ({})", c.provider),
            (None, Some(paths)) => format!("Generating refinements for {} files", paths.len()),
            (None, None) => "Generating refinements".to_string(),
        }
    }
}

/// Starts the run on the runtime; false when there is none.
pub fn spawn(spec: RunSpec, tx: Sender<Msg>, cancel: Arc<AtomicBool>) -> bool {
    let Ok(rt) = tokio::runtime::Handle::try_current() else { return false };
    rt.spawn(run(spec, tx, cancel));
    true
}

async fn run(spec: RunSpec, tx: Sender<Msg>, cancel: Arc<AtomicBool>) {
    let handle = tasks::start(TaskSpec::new(TaskKind::Refine, spec.title()).cancel_flag(cancel.clone()).user_initiated());
    let reporter = handle.reporter();
    let stop = {
        let cancel = cancel.clone();
        move || cancel.load(Ordering::Relaxed) || reporter.cancel_requested()
    };
    let progress_tx = tx.clone();
    let progress = move |done: u64, total: u64| {
        reporter.progress(done, total);
        let _ = progress_tx.send(Msg::Progress { done, total });
    };
    let ctl = RunCtl { should_stop: &stop, progress: &progress };
    let result = match &spec.cloud {
        Some(cloud) => run_cloud(&spec, cloud, &tx, &ctl).await,
        None => run_local(&spec, &tx, &ctl).await,
    };
    match result {
        Ok(scanned) if stop() => {
            handle.cancelled();
            let _ = tx.send(Msg::Finished { cancelled: true, scanned });
        }
        Ok(scanned) => {
            handle.complete();
            let _ = tx.send(Msg::Finished { cancelled: false, scanned });
        }
        Err(e) => {
            handle.fail(e.clone());
            let _ = tx.send(Msg::Failed(e));
        }
    }
}

async fn rows_for(generator: &ProposalGenerator, spec: &RunSpec, limit: usize) -> Vec<crate::Thumbnail> {
    match spec.scope.as_deref() {
        Some(paths) => generator.rows(RowScope::Paths(paths)).await,
        None => generator.rows(RowScope::Rich(limit)).await,
    }
}

/// Sends the buffered proposals in batches.
struct Batcher {
    tx: Sender<Msg>,
    buf: Vec<refine::Proposal>,
}

impl Batcher {
    fn push(&mut self, p: refine::Proposal) {
        self.buf.push(p);
        if self.buf.len() >= BATCH {
            self.flush();
        }
    }

    fn flush(&mut self) {
        if !self.buf.is_empty() {
            let _ = self.tx.send(Msg::Generated(std::mem::take(&mut self.buf)));
        }
    }
}

async fn run_local(spec: &RunSpec, tx: &Sender<Msg>, ctl: &RunCtl<'_>) -> Result<u64, String> {
    match spec.reranker {
        RerankerType::Heuristic => crate::ai::unload_heavy_models_except("CLIP").await,
        RerankerType::JinaM0 => {
            crate::ai::unload_heavy_models_except("RERANK").await;
            crate::ai::reranker::ensure_reranker_from_settings().await.map_err(|e| format!("the Jina reranker could not be loaded: {e:#}"))?;
        }
    }
    let generator = ProposalGenerator::new(&crate::ai::GLOBAL_AI_ENGINE);
    let rows = rows_for(&generator, spec, RICH_ROW_LIMIT).await;
    let scanned = rows.len() as u64;
    let mut heuristic = HeuristicReranker;
    let mut jina = JinaReranker;
    let reranker: &mut dyn Reranker = match spec.reranker {
        RerankerType::Heuristic => &mut heuristic,
        RerankerType::JinaM0 => &mut jina,
    };
    let mut batcher = Batcher { tx: tx.clone(), buf: Vec::new() };
    generator.stream_rows(rows, reranker, ctl, |p| batcher.push(p)).await;
    batcher.flush();
    Ok(scanned)
}

async fn run_cloud(spec: &RunSpec, cloud: &CloudChoice, tx: &Sender<Msg>, ctl: &RunCtl<'_>) -> Result<u64, String> {
    let generator = ProposalGenerator::new(&crate::ai::GLOBAL_AI_ENGINE);
    let rows: Vec<crate::Thumbnail> = rows_for(&generator, spec, CLOUD_LIMIT).await.into_iter().take(CLOUD_LIMIT).collect();
    let cfg = refine::provider_config(&cloud.provider, &cloud.model);
    crate::status::RERANK_STATUS.set_model(&format!("{}:{}", cfg.provider, cfg.model));
    let total = rows.len() as u64;
    let mut batcher = Batcher { tx: tx.clone(), buf: Vec::new() };
    let (mut produced, mut first_error) = (0usize, None::<String>);
    for (i, row) in rows.iter().enumerate() {
        if (ctl.should_stop)() {
            break;
        }
        (ctl.progress)(i as u64, total);
        match refine::refine_with_cloud(&cfg, row).await {
            Ok(Some(p)) => {
                produced += 1;
                batcher.push(p);
            }
            Ok(None) => {}
            Err(e) => {
                log::warn!("[Refine/Cloud] {}: {e}", row.path);
                first_error.get_or_insert(e);
            }
        }
    }
    batcher.flush();
    match first_error {
        Some(e) if produced == 0 => Err(e),
        _ => Ok(total),
    }
}

/// The proposals a batch of generated results becomes, with keys numbered from `next`.
pub fn into_proposals(batch: Vec<refine::Proposal>, next: &mut u64, now_ms: u64) -> Vec<Proposal> {
    batch
        .iter()
        .filter_map(|g| {
            let key = ProposalKey::Local(*next);
            let made = Proposal::from_generated(key, g, now_ms);
            if made.is_some() {
                *next += 1;
            }
            made
        })
        .collect()
}
