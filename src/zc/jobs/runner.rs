//! The job service: a single-flight queue over `/api/complete` with retries on transient errors, cancel, task tracking and a bounded log.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use tokio::sync::mpsc::{UnboundedSender, unbounded_channel};
use zeroclaw_client::{CompleteRequest, CompleteResponse, Error, RestClient};

use super::record::{JobRecord, StepRecord, Totals, clip_text};
use super::{Availability, ErrorKind, JobError, JobEvent, JobHandle, JobId, JobKind, JobOutcome, JobSpec, Step, StepReply, SubmitError, extract};
use crate::tasks::{self, TaskHandle, TaskSpec};
use crate::zc::events::now_ms;
use crate::zc::health::{self, Health};
use crate::zc::runtime::ZcRuntime;

/// Longest error text kept from the gateway.
const ERROR_CLIP: usize = 300;
/// Age below which the status bar leaves a short job out.
const SHORT_JOB: Duration = Duration::from_millis(400);

/// Limits of the job service.
#[derive(Clone, Debug)]
pub struct JobsConfig {
    /// Jobs that may wait behind the running one.
    pub queue_limit: usize,
    /// Requests per step, the first included, when the error is transient.
    pub attempts: u32,
    pub backoff_initial: Duration,
    pub backoff_max: Duration,
    /// Finished jobs kept for the Devtools window.
    pub history: usize,
    /// Bytes of a request or reply kept in the log.
    pub text_cap: usize,
}

impl Default for JobsConfig {
    fn default() -> Self {
        Self { queue_limit: 8, attempts: 3, backoff_initial: Duration::from_millis(800), backoff_max: Duration::from_secs(10), history: 50, text_cap: 128 * 1024 }
    }
}

impl JobsConfig {
    /// Millisecond delays, for tests.
    pub fn quick() -> Self {
        Self { backoff_initial: Duration::from_millis(5), backoff_max: Duration::from_millis(20), ..Self::default() }
    }

    /// The wait before request number `attempt + 1`.
    pub fn delay(&self, attempt: u32) -> Duration {
        let factor = 1u32 << attempt.saturating_sub(1).min(10);
        self.backoff_initial.saturating_mul(factor).min(self.backoff_max)
    }
}

/// A job waiting for its turn.
pub struct QueueRow {
    pub id: JobId,
    pub kind: JobKind,
    pub title: String,
    pub steps: usize,
    pub waited: Duration,
}

/// The job running now.
pub struct RunningRow {
    pub id: JobId,
    pub kind: JobKind,
    pub title: String,
    /// Requests finished.
    pub done: usize,
    pub total: usize,
    pub elapsed: Duration,
    pub record: Arc<JobRecord>,
}

/// Everything the Devtools window shows, copied out of the service.
pub struct Snapshot {
    pub availability: Availability,
    pub hint: String,
    pub queue_limit: usize,
    pub queue: Vec<QueueRow>,
    pub running: Option<RunningRow>,
    pub history: Vec<Arc<JobRecord>>,
    pub totals: Totals,
}

struct QueuedJob {
    id: JobId,
    spec: JobSpec,
    hint: String,
    tx: UnboundedSender<JobEvent>,
    cancel: Arc<AtomicBool>,
    task: TaskHandle,
    record: JobRecord,
    queued: Instant,
}

struct Running {
    id: JobId,
    kind: JobKind,
    title: String,
    cancel: Arc<AtomicBool>,
    record: Arc<JobRecord>,
    done: usize,
    total: usize,
    started: Instant,
}

#[derive(Default)]
struct State {
    queue: VecDeque<QueuedJob>,
    running: Option<Running>,
    worker: bool,
    history: VecDeque<Arc<JobRecord>>,
    totals: Totals,
}

/// Why a request ended without a reply.
enum Stop {
    Cancelled,
    Failed(Error, u32),
}

pub struct ZcJobs {
    rt: Arc<ZcRuntime>,
    cfg: JobsConfig,
    next_id: AtomicU64,
    state: Mutex<State>,
}

impl std::fmt::Debug for ZcJobs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZcJobs").finish_non_exhaustive()
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

async fn wait_cancelled(flag: &AtomicBool) {
    while !flag.load(Ordering::Relaxed) {
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

fn request_for(step: &Step, hint: &str) -> CompleteRequest {
    let mut request = CompleteRequest::new(step.prompt.clone()).with_hint(hint).with_temperature(step.temperature);
    if let Some(system) = &step.system {
        request = request.with_system(system.clone());
    }
    request
}

impl ZcJobs {
    pub fn new(rt: Arc<ZcRuntime>, cfg: JobsConfig) -> Arc<Self> {
        Arc::new(Self { rt, cfg, next_id: AtomicU64::new(0), state: Mutex::new(State::default()) })
    }

    pub fn config(&self) -> &JobsConfig {
        &self.cfg
    }

    pub fn runtime(&self) -> &Arc<ZcRuntime> {
        &self.rt
    }

    /// The route hint jobs use unless they name one.
    pub fn default_hint(&self) -> String {
        self.rt.profile().hint().to_string()
    }

    /// Jobs can run only while the gateway is Online.
    pub fn availability(&self) -> Availability {
        let health = self.rt.health();
        let reason = match &health {
            Health::Online { .. } => return Availability::Ready,
            Health::Unconfigured(why) => format!("ZeroClaw is not set up: {why}"),
            Health::Checking => "Checking the gateway".to_string(),
            Health::Offline(why) => format!("The gateway is offline: {why}"),
            Health::Unauthorized => health.detail(),
            Health::UnknownAgent { .. } => health.detail(),
        };
        Availability::Unavailable { reason, repair: health.is_refused() }
    }

    /// A configured gateway whose first probe has not finished, so [`availability`](Self::availability) may still turn Ready.
    pub fn is_settling(&self) -> bool {
        match self.rt.health() {
            Health::Checking => true,
            Health::Unconfigured(_) => self.rt.profile().is_configured() && self.rt.has_token(),
            _ => false,
        }
    }

    /// The availability once a settling gateway has answered, or after `limit`.
    pub async fn wait_ready(&self, limit: Duration) -> Availability {
        let started = Instant::now();
        loop {
            let now = self.availability();
            if now.is_ready() || !self.is_settling() || started.elapsed() >= limit {
                return now;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// Nothing is queued or running.
    pub fn is_idle(&self) -> bool {
        let st = lock(&self.state);
        st.queue.is_empty() && st.running.is_none()
    }

    pub fn snapshot(&self) -> Snapshot {
        let st = lock(&self.state);
        let now = Instant::now();
        Snapshot {
            availability: self.availability(),
            hint: self.default_hint(),
            queue_limit: self.cfg.queue_limit,
            queue: st.queue.iter().map(|j| QueueRow { id: j.id, kind: j.spec.kind, title: j.spec.title.clone(), steps: j.spec.steps.len(), waited: now.saturating_duration_since(j.queued) }).collect(),
            running: st.running.as_ref().map(|r| RunningRow { id: r.id, kind: r.kind, title: r.title.clone(), done: r.done, total: r.total, elapsed: now.saturating_duration_since(r.started), record: r.record.clone() }),
            history: st.history.iter().cloned().collect(),
            totals: st.totals.clone(),
        }
    }

    pub fn clear_history(&self) {
        lock(&self.state).history.clear();
    }

    /// Queues a job; it runs when the ones before it are done.
    pub fn submit(self: &Arc<Self>, spec: JobSpec) -> Result<JobHandle, SubmitError> {
        if let Availability::Unavailable { reason, repair } = self.availability() {
            return Err(SubmitError::Unavailable { reason, repair });
        }
        if spec.steps.is_empty() {
            return Err(SubmitError::NoSteps);
        }
        let hint = spec.hint.clone().unwrap_or_else(|| self.default_hint());
        for (i, step) in spec.steps.iter().enumerate() {
            request_for(step, &hint).validate().map_err(|e| SubmitError::Invalid(format!("request {} of {}: {e}", i + 1, spec.steps.len())))?;
        }
        let (tx, rx) = unbounded_channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let id = self.next_id.fetch_add(1, Ordering::SeqCst) + 1;
        {
            let mut st = lock(&self.state);
            if st.queue.len() >= self.cfg.queue_limit {
                return Err(SubmitError::QueueFull { limit: self.cfg.queue_limit });
            }
            let ahead = st.queue.len() + usize::from(st.running.is_some());
            let mut task = TaskSpec::new(spec.kind.task_kind(), spec.title.clone()).total(spec.steps.len() as u64).cancel_flag(cancel.clone());
            if spec.kind != JobKind::Refine {
                task = task.visible_after(SHORT_JOB);
            }
            if ahead > 0 {
                task = task.detail(format!("Waiting for {ahead} ZeroClaw job{}", if ahead == 1 { "" } else { "s" }));
                let _ = tx.send(JobEvent::Queued { position: ahead + 1 });
            }
            let record = JobRecord { id, kind: spec.kind, title: spec.title.clone(), hint: hint.clone(), queued_ms: now_ms(), started_ms: None, finished_ms: None, steps_total: spec.steps.len(), steps: Vec::new(), outcome: None };
            st.queue.push_back(QueuedJob { id, spec, hint, tx, cancel: cancel.clone(), task: tasks::start(task), record, queued: Instant::now() });
        }
        self.kick();
        Ok(JobHandle::new(id, self.clone(), cancel, rx))
    }

    /// Stops a job: a queued one leaves the queue at once, the running one stops at its next check. False when no such job is active.
    pub fn cancel(&self, id: JobId) -> bool {
        let removed = {
            let mut st = lock(&self.state);
            match st.queue.iter().position(|j| j.id == id) {
                Some(i) => st.queue.remove(i),
                None => {
                    return match &st.running {
                        Some(r) if r.id == id => {
                            r.cancel.store(true, Ordering::Relaxed);
                            true
                        }
                        _ => false,
                    };
                }
            }
        };
        match removed {
            Some(job) => {
                job.cancel.store(true, Ordering::Relaxed);
                self.finish_unstarted(job, JobOutcome::Cancelled);
                true
            }
            None => false,
        }
    }

    /// Cancels the running job and every queued one; returns how many were active.
    pub fn cancel_all(&self) -> usize {
        let ids: Vec<JobId> = {
            let st = lock(&self.state);
            st.running.iter().map(|r| r.id).chain(st.queue.iter().map(|j| j.id)).collect()
        };
        ids.into_iter().filter(|id| self.cancel(*id)).count()
    }

    fn kick(self: &Arc<Self>) {
        {
            let mut st = lock(&self.state);
            if st.worker || st.queue.is_empty() {
                return;
            }
            st.worker = true;
        }
        let me = self.clone();
        let _ = self.rt.job(move |_| async move { me.work().await });
    }

    async fn work(self: Arc<Self>) {
        loop {
            let next = {
                let mut st = lock(&self.state);
                match st.queue.pop_front() {
                    Some(job) => {
                        if !job.cancel.load(Ordering::Relaxed) {
                            let mut record = job.record.clone();
                            record.started_ms = Some(now_ms());
                            st.running = Some(Running { id: job.id, kind: job.spec.kind, title: job.spec.title.clone(), cancel: job.cancel.clone(), record: Arc::new(record), done: 0, total: job.spec.steps.len(), started: Instant::now() });
                        }
                        job
                    }
                    None => {
                        st.worker = false;
                        return;
                    }
                }
            };
            self.run_job(next).await;
        }
    }

    fn finish_unstarted(&self, job: QueuedJob, outcome: JobOutcome) {
        let QueuedJob { tx, task, mut record, .. } = job;
        record.finished_ms = Some(now_ms());
        record.outcome = Some(outcome.clone());
        match &outcome {
            JobOutcome::Cancelled => task.cancelled(),
            JobOutcome::Failed(e) => task.fail(e.message.clone()),
            JobOutcome::Completed => task.complete(),
        }
        self.archive(record);
        let _ = tx.send(JobEvent::Finished(outcome));
        crate::command::request_repaint();
    }

    fn archive(&self, record: JobRecord) {
        let mut st = lock(&self.state);
        st.totals.add_job(&record);
        st.history.push_front(Arc::new(record));
        st.history.truncate(self.cfg.history);
    }

    async fn run_job(&self, job: QueuedJob) {
        if job.cancel.load(Ordering::Relaxed) {
            self.finish_unstarted(job, JobOutcome::Cancelled);
            return;
        }
        let QueuedJob { id, spec, hint, tx, cancel, task, mut record, .. } = job;
        let total = spec.steps.len();
        record.started_ms = Some(now_ms());
        task.detail("Running");
        let _ = tx.send(JobEvent::Started { steps: total });
        log::info!("[ZcJobs] #{id} {} started: {total} request(s), hint {hint}", spec.kind.label());
        let outcome = self.drive(id, &spec, &hint, &tx, &cancel, &task, &mut record).await;
        record.finished_ms = Some(now_ms());
        record.outcome = Some(outcome.clone());
        match &outcome {
            JobOutcome::Completed => {
                task.progress(total as u64, total as u64);
                task.complete();
            }
            JobOutcome::Cancelled => task.cancelled(),
            JobOutcome::Failed(e) => task.fail(e.message.clone()),
        }
        log::info!("[ZcJobs] #{id} {}", outcome.label());
        self.archive(record);
        lock(&self.state).running = None;
        let _ = tx.send(JobEvent::Finished(outcome));
        crate::command::request_repaint();
    }

    fn publish_running(&self, id: JobId, record: &JobRecord, done: usize) {
        let mut st = lock(&self.state);
        if let Some(r) = st.running.as_mut().filter(|r| r.id == id) {
            r.record = Arc::new(record.clone());
            r.done = done;
        }
        drop(st);
        crate::command::request_repaint();
    }

    #[allow(clippy::too_many_arguments)]
    async fn drive(&self, id: JobId, spec: &JobSpec, hint: &str, tx: &UnboundedSender<JobEvent>, cancel: &AtomicBool, task: &TaskHandle, record: &mut JobRecord) -> JobOutcome {
        let shared = self.rt.shared().clone();
        let rest: RestClient = match shared.rest() {
            Ok(rest) => rest,
            Err(why) => return JobOutcome::Failed(JobError { kind: ErrorKind::Other, message: shared.redact(&why), attempts: 0 }),
        };
        let total = spec.steps.len();
        for (index, step) in spec.steps.iter().enumerate() {
            if cancel.load(Ordering::Relaxed) {
                return JobOutcome::Cancelled;
            }
            task.progress(index as u64, total as u64);
            task.detail(format!("Request {} of {total}", index + 1));
            let _ = tx.send(JobEvent::Progress { done: index, total });
            self.publish_running(id, record, index);
            let request = request_for(step, hint);
            let started = Instant::now();
            let mut attempts = 0;
            let result = self.call(&rest, &request, cancel, tx, index, &mut attempts).await;
            let duration = started.elapsed();
            let mut entry = StepRecord {
                index,
                label: step.label.clone(),
                system: step.system.as_deref().map(|s| clip_text(&shared.redact(s), self.cfg.text_cap)),
                prompt: clip_text(&shared.redact(&step.prompt), self.cfg.text_cap),
                response: None,
                model: String::new(),
                error: None,
                attempts,
                prompt_bytes: request.input_len(),
                response_bytes: 0,
                duration_ms: duration.as_millis() as u64,
                json: None,
            };
            match result {
                Ok(response) => {
                    let text = shared.redact(&response.response);
                    let (json, json_error) = match step.expect {
                        Some(shape) => match extract(&text, shape) {
                            Ok(found) => (Some(found), None),
                            Err(e) => (None, Some(e)),
                        },
                        None => (None, None),
                    };
                    entry.response_bytes = text.len();
                    entry.response = Some(clip_text(&text, self.cfg.text_cap));
                    entry.model = response.model.clone();
                    entry.json = json.as_ref().map(|j| j.quality.label().to_string()).or_else(|| json_error.as_ref().map(ToString::to_string));
                    log::info!("[ZcJobs] #{id} request {}/{total}: {} bytes in, {} bytes out, {} ms, {attempts} attempt(s)", index + 1, entry.prompt_bytes, entry.response_bytes, entry.duration_ms);
                    record.steps.push(entry);
                    self.publish_running(id, record, index + 1);
                    let reply = StepReply { index, label: step.label.clone(), text, model: response.model, json, json_error, prompt_bytes: request.input_len(), duration, attempts };
                    let _ = tx.send(JobEvent::Reply(reply));
                }
                Err(Stop::Cancelled) => {
                    entry.error = Some("cancelled".to_string());
                    record.steps.push(entry);
                    return JobOutcome::Cancelled;
                }
                Err(Stop::Failed(error, attempts)) => {
                    let failure = self.describe(&error, attempts);
                    log::warn!("[ZcJobs] #{id} request {}/{total} failed after {attempts} attempt(s): {}", index + 1, failure.message);
                    entry.error = Some(failure.message.clone());
                    record.steps.push(entry);
                    return JobOutcome::Failed(failure);
                }
            }
        }
        let _ = tx.send(JobEvent::Progress { done: total, total });
        JobOutcome::Completed
    }

    /// One request with retries on transient errors; the token is never retried.
    async fn call(&self, rest: &RestClient, request: &CompleteRequest, cancel: &AtomicBool, tx: &UnboundedSender<JobEvent>, step: usize, attempts: &mut u32) -> Result<CompleteResponse, Stop> {
        loop {
            *attempts += 1;
            let result = tokio::select! {
                r = rest.complete(request) => r,
                _ = wait_cancelled(cancel) => return Err(Stop::Cancelled),
            };
            let error = match result {
                Ok(response) => return Ok(response),
                Err(e) => e,
            };
            let limit = if matches!(error, Error::Timeout(_)) { self.cfg.attempts.min(2) } else { self.cfg.attempts };
            if error.is_auth() || !error.is_retryable() || *attempts >= limit {
                return Err(Stop::Failed(error, *attempts));
            }
            let wait = self.cfg.delay(*attempts);
            let _ = tx.send(JobEvent::Retrying { step, attempt: *attempts + 1, wait, error: self.rt.shared().redact(&error.to_string()) });
            tokio::select! {
                _ = tokio::time::sleep(wait) => {}
                _ = wait_cancelled(cancel) => return Err(Stop::Cancelled),
            }
        }
    }

    /// The failure of a request as the job reports it; a refused token and an unreachable gateway are also told to health.
    fn describe(&self, error: &Error, attempts: u32) -> JobError {
        let shared = self.rt.shared();
        let kind = if error.is_auth() {
            shared.refused();
            ErrorKind::Auth
        } else {
            match error {
                Error::Connect(_) => {
                    shared.apply_health(health::classify_error(error));
                    ErrorKind::Unreachable
                }
                Error::Timeout(_) => ErrorKind::Unreachable,
                Error::TooLarge { .. } => ErrorKind::TooLarge,
                Error::Http { status, .. } | Error::Rejected { status, .. } if matches!(*status, 408 | 425 | 429) => ErrorKind::Unreachable,
                Error::Http { status, .. } | Error::Rejected { status, .. } if *status < 500 => ErrorKind::Rejected,
                _ => ErrorKind::Other,
            }
        };
        let message = match kind {
            ErrorKind::Auth => "The gateway refused the token. Pair again in Settings > Assistant.".to_string(),
            _ => clip_text(&shared.redact(&error.to_string()), ERROR_CLIP),
        };
        JobError { kind, message, attempts }
    }
}
