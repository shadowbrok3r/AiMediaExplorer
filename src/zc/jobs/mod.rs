//! One-shot AI jobs on the gateway's `/api/complete`: a single-flight queue of text-only requests with batching, retries, cancel, task tracking and a log for the Devtools window.

pub mod ask;
pub mod batch;
pub mod extract;
pub mod record;
pub mod runner;
pub mod suggest;

#[cfg(test)]
mod tests;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use serde::Serialize;
use tokio::sync::mpsc::{UnboundedReceiver, error::TryRecvError};

use crate::tasks::TaskKind;

pub use extract::{ExtractError, Extracted, Quality, Shape, extract};
pub use record::{JobRecord, StepRecord, Totals};
pub use runner::{JobsConfig, QueueRow, RunningRow, Snapshot, ZcJobs};

pub type JobId = u64;

static JOBS: OnceLock<Arc<ZcJobs>> = OnceLock::new();

/// The job service of the application, over the application's ZeroClaw runtime.
pub fn jobs() -> &'static Arc<ZcJobs> {
    JOBS.get_or_init(|| ZcJobs::new(super::runtime().clone(), JobsConfig::default()))
}

/// Runs a one-request job and waits for its reply; `cancel` stops the wait and the job.
pub async fn run_one(jobs: &Arc<ZcJobs>, spec: JobSpec, cancel: &AtomicBool) -> Result<StepReply, String> {
    if let Availability::Unavailable { reason, .. } = jobs.wait_ready(Duration::from_secs(8)).await {
        return Err(reason);
    }
    let mut handle = jobs.submit(spec).map_err(|e| e.to_string())?;
    let mut reply = None;
    let outcome = loop {
        if cancel.load(Ordering::Relaxed) {
            handle.cancel();
        }
        match tokio::time::timeout(Duration::from_millis(60), handle.next()).await {
            Err(_) => continue,
            Ok(None) => break JobOutcome::Cancelled,
            Ok(Some(JobEvent::Reply(r))) => reply = Some(r),
            Ok(Some(JobEvent::Finished(outcome))) => break outcome,
            Ok(Some(_)) => {}
        }
    };
    match outcome {
        JobOutcome::Completed => reply.ok_or_else(|| "the gateway sent no reply".to_string()),
        JobOutcome::Cancelled => Err("Cancelled".to_string()),
        JobOutcome::Failed(e) => Err(e.message),
    }
}

/// What a job is for; it picks the task kind and the Devtools label.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JobKind {
    Refine,
    Ask,
    Suggest,
    Manual,
}

impl JobKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Refine => "Refine",
            Self::Ask => "Ask",
            Self::Suggest => "Suggest",
            Self::Manual => "Manual",
        }
    }

    pub fn task_kind(self) -> TaskKind {
        match self {
            Self::Refine => TaskKind::Refine,
            _ => TaskKind::Assistant,
        }
    }
}

/// One request of a job.
#[derive(Clone, Debug)]
pub struct Step {
    pub label: String,
    pub system: Option<String>,
    pub prompt: String,
    /// The JSON the reply should hold; the reply is read with the tolerant extractor.
    pub expect: Option<Shape>,
    pub temperature: f64,
}

impl Step {
    pub fn new(label: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self { label: label.into(), system: None, prompt: prompt.into(), expect: None, temperature: 0.2 }
    }

    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    pub fn expect(mut self, shape: Shape) -> Self {
        self.expect = Some(shape);
        self
    }

    pub fn temperature(mut self, t: f64) -> Self {
        self.temperature = t;
        self
    }

    /// Bytes the gateway's input cap counts for this step.
    pub fn input_len(&self) -> usize {
        self.prompt.len() + self.system.as_deref().map_or(0, str::len)
    }
}

/// Requests that run one after the other as one task.
#[derive(Clone, Debug)]
pub struct JobSpec {
    pub kind: JobKind,
    pub title: String,
    /// The route hint; the profile's when none is given.
    pub hint: Option<String>,
    pub steps: Vec<Step>,
}

impl JobSpec {
    pub fn new(kind: JobKind, title: impl Into<String>) -> Self {
        Self { kind, title: title.into(), hint: None, steps: Vec::new() }
    }

    pub fn hint(mut self, hint: impl Into<String>) -> Self {
        let hint = hint.into();
        self.hint = Some(hint).filter(|h| !h.trim().is_empty());
        self
    }

    pub fn step(mut self, step: Step) -> Self {
        self.steps.push(step);
        self
    }

    pub fn steps(mut self, steps: impl IntoIterator<Item = Step>) -> Self {
        self.steps.extend(steps);
        self
    }
}

/// Why a job could not start or queue.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SubmitError {
    Unavailable { reason: String, repair: bool },
    QueueFull { limit: usize },
    NoSteps,
    /// A step the gateway would refuse, such as one over the input cap.
    Invalid(String),
}

impl std::fmt::Display for SubmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable { reason, .. } => f.write_str(reason),
            Self::QueueFull { limit } => write!(f, "{limit} ZeroClaw jobs are already waiting"),
            Self::NoSteps => f.write_str("the job has nothing to ask"),
            Self::Invalid(why) => f.write_str(why),
        }
    }
}

/// Whether jobs can run now, with the reason when they cannot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Availability {
    Ready,
    Unavailable {
        reason: String,
        /// The token was refused: the Re-pair entry of Needs attention applies.
        repair: bool,
    },
}

impl Availability {
    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready)
    }

    /// The reason jobs cannot run, if they cannot.
    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Ready => None,
            Self::Unavailable { reason, .. } => Some(reason),
        }
    }
}

/// The kind of failure that ended a job.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// The gateway refused the token; never retried.
    Auth,
    /// The gateway could not be reached or did not answer.
    Unreachable,
    /// The gateway answered with an error that repeating will not fix.
    Rejected,
    TooLarge,
    Other,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JobError {
    pub kind: ErrorKind,
    pub message: String,
    /// Requests sent for the step that failed.
    pub attempts: u32,
}

impl JobError {
    pub fn needs_repair(&self) -> bool {
        self.kind == ErrorKind::Auth
    }
}

impl std::fmt::Display for JobError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JobOutcome {
    Completed,
    Cancelled,
    Failed(JobError),
}

impl JobOutcome {
    pub fn label(&self) -> String {
        match self {
            Self::Completed => "completed".to_string(),
            Self::Cancelled => "cancelled".to_string(),
            Self::Failed(e) => format!("failed: {}", record::clip_text(&e.message, 80)),
        }
    }
}

/// A successful request of a job.
#[derive(Clone, Debug)]
pub struct StepReply {
    pub index: usize,
    pub label: String,
    pub text: String,
    pub model: String,
    /// The JSON read from the reply, when the step expected some and it could be read.
    pub json: Option<Extracted>,
    /// Why the JSON could not be read.
    pub json_error: Option<ExtractError>,
    pub prompt_bytes: usize,
    pub duration: Duration,
    pub attempts: u32,
}

/// What a job reports while it runs; `Finished` comes last, exactly once.
#[derive(Clone, Debug)]
pub enum JobEvent {
    /// Waiting behind other jobs; `position` counts from 1.
    Queued { position: usize },
    Started { steps: usize },
    Progress { done: usize, total: usize },
    Retrying { step: usize, attempt: u32, wait: Duration, error: String },
    Reply(StepReply),
    Finished(JobOutcome),
}

/// The result of waiting for a whole job.
#[derive(Clone, Debug)]
pub struct JobResult {
    pub replies: Vec<StepReply>,
    pub outcome: JobOutcome,
}

/// The caller's side of a submitted job.
pub struct JobHandle {
    id: JobId,
    jobs: Arc<ZcJobs>,
    cancel: Arc<AtomicBool>,
    rx: UnboundedReceiver<JobEvent>,
    finished: bool,
}

impl std::fmt::Debug for JobHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JobHandle").field("id", &self.id).field("finished", &self.finished).finish()
    }
}

impl JobHandle {
    pub(super) fn new(id: JobId, jobs: Arc<ZcJobs>, cancel: Arc<AtomicBool>, rx: UnboundedReceiver<JobEvent>) -> Self {
        Self { id, jobs, cancel, rx, finished: false }
    }

    pub fn id(&self) -> JobId {
        self.id
    }

    /// Stops the job: a queued one leaves the queue, a running one stops at once and keeps what it produced.
    pub fn cancel(&self) {
        self.jobs.cancel(self.id);
    }

    pub fn is_cancel_requested(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// The events that arrived since the last call.
    pub fn poll(&mut self) -> Vec<JobEvent> {
        let mut out = Vec::new();
        loop {
            match self.rx.try_recv() {
                Ok(event) => {
                    self.finished |= matches!(event, JobEvent::Finished(_));
                    out.push(event);
                }
                Err(TryRecvError::Empty) => return out,
                Err(TryRecvError::Disconnected) => {
                    self.finished = true;
                    return out;
                }
            }
        }
    }

    /// The next event; none once the job has finished.
    pub async fn next(&mut self) -> Option<JobEvent> {
        if self.finished {
            return None;
        }
        let event = self.rx.recv().await;
        match &event {
            Some(JobEvent::Finished(_)) | None => self.finished = true,
            _ => {}
        }
        event
    }

    /// Waits for the job to end.
    pub async fn wait(mut self) -> JobResult {
        let mut replies = Vec::new();
        while let Some(event) = self.next().await {
            match event {
                JobEvent::Reply(r) => replies.push(r),
                JobEvent::Finished(outcome) => return JobResult { replies, outcome },
                _ => {}
            }
        }
        JobResult { replies, outcome: JobOutcome::Cancelled }
    }
}
