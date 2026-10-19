//! The AI Refinements tab: proposals made in this session and proposals tools filed in the MCP store, with per-field choices and accepts that go through the metadata journal.

mod accept;
pub mod model;
mod run;
mod store;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod zc_tests;
mod view;

use std::collections::BTreeSet;
use std::collections::HashSet;
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crossbeam::channel::{Receiver, Sender, unbounded};
use egui_toast::ToastKind;
use once_cell::sync::Lazy;

use crate::ai::mcp::ctx::DbSource;
use crate::ai::refine::RerankerType;
use crate::zc::jobs::ZcJobs;
use crate::ui::file_table::thumbs::{ScopeId, ThumbService};

pub use accept::Outcome;
pub use model::{BulkPlan, FieldKind, ListFilter, Proposal, ProposalKey, ProposalSource, SortKey};
pub use run::{CloudChoice, Msg, RunSpec, ZcChoice};
pub use store::StoreSource;

/// Paths the command dispatcher hands to the refinements panel until it is shown
pub static REFINE_SELECTION_REQUESTS: Lazy<Mutex<Vec<Vec<String>>>> = Lazy::new(|| Mutex::new(Vec::new()));

/// A ZeroClaw run the `zc` command asked for: the files and the route hint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZcRunRequest {
    pub paths: Vec<String>,
    pub hint: String,
}

/// ZeroClaw runs the `zc` command queued until the tab is shown.
pub static REFINE_ZC_REQUESTS: Lazy<Mutex<Vec<ZcRunRequest>>> = Lazy::new(|| Mutex::new(Vec::new()));

/// Queues a ZeroClaw run over `paths` for the Refine tab; the tab itself is opened by the `refine` command.
pub fn request_zeroclaw_run(paths: Vec<String>, hint: Option<String>) {
    REFINE_ZC_REQUESTS.lock().unwrap_or_else(|e| e.into_inner()).push(ZcRunRequest { paths, hint: hint.unwrap_or_default() });
    crate::command::request_repaint();
}

pub fn request_refine_for_paths(paths: Vec<String>) {
    if paths.is_empty() { return; }
    crate::command::send(crate::command::AppCommand::Refine { paths });
}

/// Proposals drawn before "Show more".
const PAGE: usize = 40;
/// How long a listing of the store stays fresh while the tab is shown.
const STORE_STALE: std::time::Duration = std::time::Duration::from_secs(8);

/// Where the generator run is.
#[derive(Clone, Debug, PartialEq)]
pub enum RunState {
    Idle,
    Running { done: u64, total: u64 },
    Done { scanned: u64, produced: usize, cancelled: bool },
    Failed(String),
}

/// A bulk action waiting for the user's answer.
#[derive(Clone, Debug, PartialEq)]
pub enum Confirm {
    Accept(BulkPlan),
    Reject(Vec<ProposalKey>),
}

/// The state of the listing of the MCP store.
#[derive(Clone, Debug, Default)]
pub struct StoreStatus {
    pub loading: bool,
    pub at: Option<Instant>,
    pub error: Option<String>,
    pub pending: usize,
}

/// What this session has decided.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tally {
    pub applied_fields: usize,
    pub applied_files: usize,
    pub rejected: usize,
    pub failed: usize,
}

pub struct RefinementsPanel {
    pub reranker_choice: RerankerType,
    pub use_cloud_model: bool,
    pub cloud_model_name: String,
    pub cloud_provider: String,
    /// Ask the ZeroClaw gateway instead of the local or cloud generators.
    pub use_zeroclaw: bool,
    /// Route hint of a ZeroClaw run; blank takes the profile's.
    pub zc_hint: String,
    zc_jobs: Option<Arc<ZcJobs>>,
    pub filter: ListFilter,
    pub(crate) proposals: Vec<Proposal>,
    scope: BTreeSet<String>,
    run: RunState,
    run_produced: usize,
    cancel: Option<Arc<AtomicBool>>,
    pub(crate) confirm: Option<Confirm>,
    applying: HashSet<ProposalKey>,
    shown: usize,
    db: DbSource,
    store: StoreSource,
    pub(crate) store_status: StoreStatus,
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
    toast_tx: Sender<(ToastKind, String)>,
    next_local: u64,
    thumbs: Arc<ThumbService>,
    thumb_scope: ScopeId,
    pub(crate) tally: Tally,
    pub(crate) notice: Option<(bool, String)>,
    global_requests: bool,
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

impl RefinementsPanel {
    pub fn new(toast_tx: Sender<(ToastKind, String)>) -> Self {
        let mut panel = Self::with_sources(toast_tx, DbSource::Global, StoreSource::Hub, crate::ui::file_table::thumbs::service());
        panel.global_requests = true;
        panel
    }

    /// A panel over given sources, for tests and tools that bring their own database or store.
    pub fn with_sources(toast_tx: Sender<(ToastKind, String)>, db: DbSource, store: StoreSource, thumbs: Arc<ThumbService>) -> Self {
        let (tx, rx) = unbounded();
        Self {
            reranker_choice: Default::default(),
            use_cloud_model: false,
            cloud_model_name: String::new(),
            cloud_provider: "openrouter".into(),
            use_zeroclaw: false,
            zc_hint: String::new(),
            zc_jobs: None,
            filter: ListFilter::default(),
            proposals: Vec::new(),
            scope: BTreeSet::new(),
            run: RunState::Idle,
            run_produced: 0,
            cancel: None,
            confirm: None,
            applying: HashSet::new(),
            shown: PAGE,
            db,
            store,
            store_status: StoreStatus::default(),
            tx,
            rx,
            toast_tx,
            next_local: 1,
            thumb_scope: crate::ui::file_table::thumbs::alloc_scope(),
            thumbs,
            tally: Tally::default(),
            notice: None,
            global_requests: false,
        }
    }

    /// The job service ZeroClaw runs go through.
    pub fn zc_jobs(&self) -> Arc<ZcJobs> {
        self.zc_jobs.clone().unwrap_or_else(|| crate::zc::jobs::jobs().clone())
    }

    /// Uses `jobs` for ZeroClaw runs, for tests and tools with their own gateway.
    pub fn set_zc_jobs(&mut self, jobs: Arc<ZcJobs>) {
        self.zc_jobs = Some(jobs);
    }

    /// Why a ZeroClaw run cannot start now, if it cannot.
    pub fn zc_unavailable(&self) -> Option<String> {
        self.zc_jobs().availability().reason().map(str::to_string)
    }

    /// Sender that background jobs report to; tests use it to feed messages.
    #[cfg(test)]
    pub fn sender(&self) -> Sender<Msg> {
        self.tx.clone()
    }

    pub fn proposals(&self) -> &[Proposal] {
        &self.proposals
    }

    pub fn run_state(&self) -> &RunState {
        &self.run
    }

    pub fn is_running(&self) -> bool {
        matches!(self.run, RunState::Running { .. })
    }

    pub fn scope_len(&self) -> usize {
        self.scope.len()
    }

    pub fn clear_scope(&mut self) {
        self.scope.clear();
    }

    fn toast(&self, kind: ToastKind, text: impl Into<String>) {
        let _ = self.toast_tx.try_send((kind, text.into()));
    }

    fn note(&mut self, ok: bool, text: impl Into<String>) {
        self.notice = Some((ok, text.into()));
    }

    fn spawn<F: Future<Output = ()> + Send + 'static>(&self, fut: F) -> bool {
        match tokio::runtime::Handle::try_current() {
            Ok(rt) => {
                rt.spawn(fut);
                true
            }
            Err(_) => false,
        }
    }

    /// Takes the selection the dispatcher queued for the tab; only the panel of the application reads the shared mailbox.
    pub fn take_requests(&mut self) {
        if !self.global_requests {
            return;
        }
        let batches: Vec<Vec<String>> = std::mem::take(&mut *REFINE_SELECTION_REQUESTS.lock().unwrap_or_else(|e| e.into_inner()));
        self.absorb_requests(batches);
        let runs: Vec<ZcRunRequest> = std::mem::take(&mut *REFINE_ZC_REQUESTS.lock().unwrap_or_else(|e| e.into_inner()));
        self.absorb_zc_runs(runs);
    }

    /// Starts a ZeroClaw run over the files of each request.
    pub fn absorb_zc_runs(&mut self, runs: Vec<ZcRunRequest>) {
        for run in runs {
            self.absorb_requests(vec![run.paths]);
            self.use_zeroclaw = true;
            self.use_cloud_model = false;
            self.zc_hint = run.hint;
            self.start_run();
        }
    }

    /// Adds the files of the requests to the scope of the next run.
    pub fn absorb_requests(&mut self, batches: Vec<Vec<String>>) {
        for batch in batches {
            self.scope.extend(batch);
        }
    }

    /// Starts generating proposals for the scope, or for the library's described files without one; the proposals of this session are replaced, the stored ones stay.
    pub fn start_run(&mut self) {
        if self.is_running() {
            return;
        }
        if self.use_zeroclaw && !self.zc_jobs().is_settling() {
            if let Some(reason) = self.zc_unavailable() {
                self.run = RunState::Failed(reason);
                return;
            }
        }
        let spec = RunSpec {
            reranker: self.reranker_choice,
            cloud: (self.use_cloud_model && !self.use_zeroclaw).then(|| CloudChoice { provider: self.cloud_provider.clone(), model: self.cloud_model_name.clone() }),
            zeroclaw: self.use_zeroclaw.then(|| ZcChoice { jobs: self.zc_jobs(), db: self.db.clone(), hint: self.zc_hint.trim().to_string() }),
            scope: (!self.scope.is_empty()).then(|| self.scope.iter().cloned().collect()),
        };
        let cancel = Arc::new(AtomicBool::new(false));
        if !run::spawn(spec, self.tx.clone(), cancel.clone()) {
            self.run = RunState::Failed("no async runtime is available".to_string());
            return;
        }
        self.proposals.retain(|p| p.source.is_mcp());
        self.cancel = Some(cancel);
        self.run_produced = 0;
        self.run = RunState::Running { done: 0, total: 0 };
        self.notice = None;
    }

    /// Asks the running generation to stop.
    pub fn cancel_run(&mut self) {
        if let Some(flag) = &self.cancel {
            flag.store(true, Ordering::Relaxed);
        }
    }

    /// Drops the proposals of this session; the stored ones stay.
    pub fn clear_local(&mut self) {
        self.proposals.retain(|p| p.source.is_mcp());
        self.confirm = None;
    }

    /// Reads the pending proposals of the store in the background.
    pub fn refresh_store(&mut self) {
        if self.store_status.loading {
            return;
        }
        let Some(store) = self.store.get() else { return };
        self.store_status.loading = true;
        let (db, tx) = (self.db.clone(), self.tx.clone());
        if !self.spawn(async move {
            let listed = store::list_pending(&store, &db).await;
            let _ = tx.send(Msg::Stored(listed));
            crate::command::request_repaint();
        }) {
            self.store_status.loading = false;
        }
    }

    /// Lists the store again when the last listing is old.
    pub fn refresh_store_if_stale(&mut self) {
        let stale = self.store_status.at.is_none_or(|t| t.elapsed() > STORE_STALE);
        if stale && !self.store_status.loading && !matches!(self.store, StoreSource::Off) && self.db.get().is_ok() {
            self.refresh_store();
        }
    }

    /// Takes in what the background jobs reported.
    pub fn pump(&mut self) {
        let msgs: Vec<Msg> = self.rx.try_iter().collect();
        for msg in msgs {
            self.ingest(msg);
        }
    }

    /// Applies one report of a background job.
    pub fn ingest(&mut self, msg: Msg) {
        match msg {
            Msg::Progress { done, total } => {
                if self.is_running() {
                    self.run = RunState::Running { done, total };
                }
            }
            Msg::Generated(batch) => {
                let made = run::into_proposals(batch, &mut self.next_local, now_ms());
                self.run_produced += made.len();
                for p in made {
                    match self.proposals.iter().position(|old| !old.source.is_mcp() && old.path == p.path) {
                        Some(i) => self.proposals[i] = p,
                        None => self.proposals.push(p),
                    }
                }
            }
            Msg::Finished { cancelled, scanned } => {
                self.run = RunState::Done { scanned, produced: self.run_produced, cancelled };
                self.cancel = None;
                if self.run_produced == 0 && !cancelled {
                    self.toast(ToastKind::Info, format!("No proposals: {scanned} files checked and nothing new to propose"));
                } else if !cancelled {
                    self.toast(ToastKind::Info, format!("Proposals ready ({})", self.run_produced));
                }
            }
            Msg::Notice(text) => self.note(false, text),
            Msg::Failed(e) => {
                self.toast(ToastKind::Error, format!("Refine failed: {e}"));
                self.run = RunState::Failed(e);
                self.cancel = None;
            }
            Msg::Stored(listed) => {
                self.store_status.loading = false;
                self.store_status.at = Some(Instant::now());
                match listed {
                    Ok(list) => {
                        self.store_status.error = None;
                        self.store_status.pending = list.len();
                        self.replace_stored(list);
                    }
                    Err(e) => self.store_status.error = Some(e),
                }
            }
            Msg::Applied(out) => self.finish_decision(out, true),
            Msg::Rejected(out) => self.finish_decision(out, false),
        }
    }

    /// Replaces the stored proposals with `list`, keeping the choices made on the ones that are still there.
    fn replace_stored(&mut self, list: Vec<Proposal>) {
        let kept: std::collections::HashMap<ProposalKey, Proposal> = self.proposals.iter().filter(|p| p.source.is_mcp()).map(|p| (p.key.clone(), p.clone())).collect();
        self.proposals.retain(|p| !p.source.is_mcp());
        for mut p in list {
            if let Some(old) = kept.get(&p.key) {
                for c in &mut p.changes {
                    if let Some(o) = old.change(c.kind) {
                        c.accepted = o.accepted;
                    }
                }
                p.error = old.error.clone();
            }
            self.proposals.push(p);
        }
    }

    fn finish_decision(&mut self, out: Outcome, applied: bool) {
        for key in out.done.iter().chain(out.failed.iter().map(|(k, _)| k)) {
            self.applying.remove(key);
        }
        let touched_store = out.done.iter().chain(out.failed.iter().map(|(k, _)| k)).any(|k| matches!(k, ProposalKey::Mcp(_)));
        self.proposals.retain(|p| !out.done.contains(&p.key));
        for (key, why) in &out.failed {
            if let Some(p) = self.proposals.iter_mut().find(|p| p.key == *key) {
                p.error = Some(why.clone());
            }
        }
        self.tally.failed += out.failed.len();
        let mut text = if applied {
            self.tally.applied_fields += out.fields;
            self.tally.applied_files += out.files();
            format!("Applied {} field{} in {} file{}. Ctrl+Z undoes it.", out.fields, if out.fields == 1 { "" } else { "s" }, out.files(), if out.files() == 1 { "" } else { "s" })
        } else {
            self.tally.rejected += out.done.len();
            format!("Rejected {} proposal{}", out.done.len(), if out.done.len() == 1 { "" } else { "s" })
        };
        if !out.failed.is_empty() {
            text.push_str(&format!(" {} could not be {}: {}", out.failed.len(), if applied { "applied" } else { "rejected" }, out.failed[0].1));
        }
        let ok = out.failed.is_empty();
        self.toast(if ok { ToastKind::Success } else { ToastKind::Warning }, text.clone());
        self.note(ok, text);
        crate::ui::journal_ui::push_edits(out.edits);
        if touched_store {
            self.store_status.at = None;
        }
    }

    /// Indices of the proposals the filter shows, in its sort order.
    pub fn visible(&self) -> Vec<usize> {
        model::visible(&self.proposals, &self.filter)
    }

    /// What accepting the shown proposals would write.
    pub fn plan_view(&self) -> BulkPlan {
        model::plan_accept(&self.proposals, &self.visible())
    }

    /// Asks to apply the checked fields of the shown proposals; the answer comes from `confirm_pending`.
    pub fn ask_accept_view(&mut self) {
        let plan = self.plan_view();
        if plan.is_empty() {
            self.note(false, "Nothing is checked in the shown proposals");
            return;
        }
        self.confirm = Some(Confirm::Accept(plan));
    }

    /// Asks to reject the shown proposals.
    pub fn ask_reject_view(&mut self) {
        let keys: Vec<ProposalKey> = self.visible().into_iter().map(|i| self.proposals[i].key.clone()).collect();
        if keys.is_empty() {
            self.note(false, "No proposals are shown");
            return;
        }
        self.confirm = Some(Confirm::Reject(keys));
    }

    /// Runs the bulk action that was confirmed.
    pub fn confirm_pending(&mut self) {
        match self.confirm.take() {
            Some(Confirm::Accept(plan)) => self.execute_accept(plan),
            Some(Confirm::Reject(keys)) => self.execute_reject(keys),
            None => {}
        }
    }

    /// Applies the checked fields of one proposal at once.
    pub fn accept_one(&mut self, key: &ProposalKey) {
        let Some(i) = self.proposals.iter().position(|p| p.key == *key) else { return };
        let plan = model::plan_accept(&self.proposals, &[i]);
        if plan.is_empty() {
            self.note(false, "Check at least one field first");
            return;
        }
        self.execute_accept(plan);
    }

    pub fn reject_one(&mut self, key: &ProposalKey) {
        self.execute_reject(vec![key.clone()]);
    }

    fn execute_accept(&mut self, plan: BulkPlan) {
        let items: Vec<ProposalKey> = plan.items.iter().map(|i| i.key.clone()).collect();
        let (db, store, tx) = (self.db.clone(), self.store.get(), self.tx.clone());
        let started = self.spawn(async move {
            let outcome = accept::apply(&db, store.as_deref(), plan).await;
            let _ = tx.send(Msg::Applied(outcome));
            crate::command::request_repaint();
        });
        if started {
            self.applying.extend(items);
        } else {
            self.note(false, "No async runtime is available to write the changes");
        }
    }

    fn execute_reject(&mut self, keys: Vec<ProposalKey>) {
        let (stored, local): (Vec<ProposalKey>, Vec<ProposalKey>) = keys.into_iter().partition(|k| matches!(k, ProposalKey::Mcp(_)));
        if !local.is_empty() {
            self.finish_decision(Outcome { done: local, ..Default::default() }, false);
        }
        if stored.is_empty() {
            return;
        }
        let (store, tx) = (self.store.get(), self.tx.clone());
        let started = self.spawn(async move {
            let outcome = accept::reject(store.as_deref(), &stored).await;
            let _ = tx.send(Msg::Rejected(outcome));
            crate::command::request_repaint();
        });
        if !started {
            self.note(false, "No async runtime is available");
        }
    }

    pub fn is_applying(&self, key: &ProposalKey) -> bool {
        self.applying.contains(key)
    }

    /// Proposals being written now.
    pub fn applying_len(&self) -> usize {
        self.applying.len()
    }

    /// Adds proposals with made-up values for these files of the library; for trying the tab without a model.
    pub fn inject_samples(&mut self, paths: Vec<String>) {
        let (db, tx) = (self.db.clone(), self.tx.clone());
        let started = self.spawn(async move {
            let rows = match db.get() {
                Ok(handle) => crate::database::collections::rows_for_paths(handle, &paths).await.unwrap_or_default(),
                Err(_) => Vec::new(),
            };
            let _ = tx.send(Msg::Generated(sample_proposals(&rows)));
            crate::command::request_repaint();
        });
        if !started {
            self.note(false, "No async runtime is available");
        }
    }
}

/// Proposals with made-up values for `rows`: a new category, one more tag, a caption and a falling confidence.
pub fn sample_proposals(rows: &[crate::Thumbnail]) -> Vec<crate::ai::refine::Proposal> {
    const GENERATORS: [&str; 3] = ["Heuristic", "Jina M0", "cloud:sample"];
    rows.iter()
        .enumerate()
        .map(|(i, row)| {
            let mut p = crate::ai::refine::Proposal::for_row(row, GENERATORS[i % GENERATORS.len()]);
            p.new_category = Some(format!("sample-{}", i + 1));
            p.new_tags = row.tags.iter().cloned().chain([format!("sample-tag-{}", i + 1)]).collect();
            p.new_caption = Some(format!("Sample caption for {}", row.filename));
            p.confidence = Some((0.9 - 0.2 * i as f32).max(0.1));
            p
        })
        .collect()
}
