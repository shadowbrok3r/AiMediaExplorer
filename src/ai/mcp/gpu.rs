//! One-at-a-time queue for jobs that load or run local models.

use std::collections::VecDeque;
use std::future::Future;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde::Serialize;
use tokio_util::sync::CancellationToken;

use super::clock::Clock;

/// Finished jobs kept for the status view.
const HISTORY: usize = 20;

#[derive(Clone, Debug)]
pub struct GpuLimits {
    /// Jobs allowed to wait; one more is refused.
    pub max_queued: usize,
    /// Longest a job may wait for its turn.
    pub queue_timeout: Duration,
    /// Longest a job may run.
    pub job_timeout: Duration,
}

impl Default for GpuLimits {
    fn default() -> Self {
        Self { max_queued: 8, queue_timeout: Duration::from_secs(120), job_timeout: Duration::from_secs(300) }
    }
}

/// What the caller says about a job.
#[derive(Clone, Debug)]
pub struct JobSpec {
    pub kind: &'static str,
    pub label: String,
    pub client: String,
    /// Longest wait for the turn and longest run of this job; the queue limits apply when `None`.
    pub limits: Option<(Duration, Duration)>,
}

impl JobSpec {
    pub fn new(kind: &'static str, label: impl Into<String>, client: impl Into<String>) -> Self {
        Self { kind, label: label.into(), client: client.into(), limits: None }
    }

    /// Replaces the queue's wait and run limits for this job.
    pub fn with_limits(mut self, wait: Duration, run: Duration) -> Self {
        self.limits = Some((wait, run));
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Queued,
    Running,
    Done,
    Failed,
    TimedOut,
    Cancelled,
}

#[derive(Clone, Debug, Serialize)]
pub struct JobInfo {
    pub id: u64,
    pub kind: &'static str,
    pub label: String,
    pub client: String,
    pub state: JobState,
    pub queued_ms: u64,
    pub ran_ms: u64,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuError {
    /// The queue holds as many jobs as it may.
    QueueFull { depth: usize, max: usize },
    /// The job waited longer than the queue timeout.
    QueueTimeout(Duration),
    /// The job ran longer than the job timeout.
    JobTimeout(Duration),
    Cancelled,
    /// The job itself failed.
    Job(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::QueueFull { depth, max } => write!(f, "the GPU queue is full ({depth} of {max} places taken); try again later"),
            Self::QueueTimeout(d) => write!(f, "the job waited {}s for the GPU and was dropped", d.as_secs()),
            Self::JobTimeout(d) => write!(f, "the job ran longer than {}s and was stopped", d.as_secs()),
            Self::Cancelled => write!(f, "the job was cancelled"),
            Self::Job(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for GpuError {}

/// What a finished job reports about its place in the queue.
#[derive(Clone, Debug, Serialize)]
pub struct JobStats {
    pub job_id: u64,
    pub queue_depth_at_submit: usize,
    pub waited_ms: u64,
    pub ran_ms: u64,
}

/// A job's view of itself while it runs.
#[derive(Clone)]
pub struct JobCtx {
    pub id: u64,
    pub cancel: CancellationToken,
}

#[derive(Clone, Debug, Serialize)]
pub struct GpuStatus {
    pub running: Option<JobInfo>,
    pub queued: Vec<JobInfo>,
    pub max_queued: usize,
    pub queue_timeout_secs: u64,
    pub job_timeout_secs: u64,
    pub completed: u64,
    pub failed: u64,
    pub cancelled: u64,
    pub recent: Vec<JobInfo>,
}

struct Entry {
    info: JobInfo,
    submitted: Instant,
    started: Option<Instant>,
    cancel: CancellationToken,
}

#[derive(Default)]
struct State {
    queued: VecDeque<Entry>,
    running: Option<Entry>,
    recent: VecDeque<JobInfo>,
    completed: u64,
    failed: u64,
    cancelled: u64,
}

pub struct GpuQueue {
    limits: GpuLimits,
    clock: Arc<dyn Clock>,
    gate: tokio::sync::Mutex<()>,
    state: Mutex<State>,
    next_id: AtomicU64,
}

fn ms(d: Duration) -> u64 {
    d.as_millis() as u64
}

impl GpuQueue {
    pub fn new(limits: GpuLimits, clock: Arc<dyn Clock>) -> Self {
        Self { limits, clock, gate: tokio::sync::Mutex::new(()), state: Mutex::new(State::default()), next_id: AtomicU64::new(1) }
    }

    pub fn limits(&self) -> &GpuLimits {
        &self.limits
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Jobs waiting plus the one running.
    pub fn depth(&self) -> usize {
        let s = self.lock();
        s.queued.len() + usize::from(s.running.is_some())
    }

    /// Stops a queued or running job; false when no such job is live.
    pub fn cancel(&self, id: u64) -> bool {
        let s = self.lock();
        let token = s.queued.iter().chain(s.running.iter()).find(|e| e.info.id == id).map(|e| e.cancel.clone());
        drop(s);
        token.map(|t| t.cancel()).is_some()
    }

    pub fn status(&self) -> GpuStatus {
        let now = self.clock.now();
        let s = self.lock();
        let view = |e: &Entry| {
            let mut info = e.info.clone();
            info.queued_ms = ms(e.started.unwrap_or(now).saturating_duration_since(e.submitted));
            info.ran_ms = e.started.map(|st| ms(now.saturating_duration_since(st))).unwrap_or(0);
            info
        };
        GpuStatus {
            running: s.running.as_ref().map(view),
            queued: s.queued.iter().map(view).collect(),
            max_queued: self.limits.max_queued,
            queue_timeout_secs: self.limits.queue_timeout.as_secs(),
            job_timeout_secs: self.limits.job_timeout.as_secs(),
            completed: s.completed,
            failed: s.failed,
            cancelled: s.cancelled,
            recent: s.recent.iter().rev().cloned().collect(),
        }
    }

    /// Moves the job `id` from the queue or the running slot into the history.
    fn finish(&self, id: u64, state: JobState, error: Option<String>) -> (u64, u64) {
        let now = self.clock.now();
        let mut s = self.lock();
        let entry = match s.running.take() {
            Some(e) if e.info.id == id => Some(e),
            other => {
                s.running = other;
                s.queued.iter().position(|e| e.info.id == id).and_then(|i| s.queued.remove(i))
            }
        };
        let Some(mut e) = entry else { return (0, 0) };
        let queued = ms(e.started.unwrap_or(now).saturating_duration_since(e.submitted));
        let ran = e.started.map(|st| ms(now.saturating_duration_since(st))).unwrap_or(0);
        e.info.state = state;
        e.info.error = error;
        e.info.queued_ms = queued;
        e.info.ran_ms = ran;
        match state {
            JobState::Done => s.completed += 1,
            JobState::Cancelled => s.cancelled += 1,
            _ => s.failed += 1,
        }
        s.recent.push_back(e.info);
        while s.recent.len() > HISTORY {
            s.recent.pop_front();
        }
        (queued, ran)
    }

    /// Runs `work` when every earlier job is done; `outer` cancels it from the caller's side.
    pub async fn run<T, F, Fut>(&self, spec: JobSpec, outer: &CancellationToken, work: F) -> Result<(T, JobStats), GpuError>
    where
        F: FnOnce(JobCtx) -> Fut,
        Fut: Future<Output = Result<T, String>>,
    {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (queue_timeout, job_timeout) = spec.limits.unwrap_or((self.limits.queue_timeout, self.limits.job_timeout));
        let cancel = outer.child_token();
        let submitted = self.clock.now();
        let depth = {
            let mut s = self.lock();
            let depth = s.queued.len() + usize::from(s.running.is_some());
            if s.queued.len() >= self.limits.max_queued {
                return Err(GpuError::QueueFull { depth, max: self.limits.max_queued });
            }
            s.queued.push_back(Entry {
                info: JobInfo { id, kind: spec.kind, label: spec.label.clone(), client: spec.client.clone(), state: JobState::Queued, queued_ms: 0, ran_ms: 0, error: None },
                submitted,
                started: None,
                cancel: cancel.clone(),
            });
            depth
        };
        let mut abandon = AbandonGuard { queue: self, id, armed: true };

        let permit = tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                abandon.armed = false;
                self.finish(id, JobState::Cancelled, None);
                return Err(GpuError::Cancelled);
            }
            guard = self.gate.lock() => guard,
            _ = tokio::time::sleep(queue_timeout) => {
                abandon.armed = false;
                let e = GpuError::QueueTimeout(queue_timeout);
                self.finish(id, JobState::TimedOut, Some(e.to_string()));
                return Err(e);
            }
        };

        {
            let mut s = self.lock();
            if let Some(i) = s.queued.iter().position(|e| e.info.id == id) {
                if let Some(mut entry) = s.queued.remove(i) {
                    entry.started = Some(self.clock.now());
                    entry.info.state = JobState::Running;
                    s.running = Some(entry);
                }
            }
        }

        let ctx = JobCtx { id, cancel: cancel.clone() };
        let outcome = tokio::select! {
            biased;
            _ = cancel.cancelled() => Err(GpuError::Cancelled),
            r = work(ctx) => r.map_err(GpuError::Job),
            _ = tokio::time::sleep(job_timeout) => Err(GpuError::JobTimeout(job_timeout)),
        };
        let (state, error) = match &outcome {
            Ok(_) => (JobState::Done, None),
            Err(GpuError::Cancelled) => (JobState::Cancelled, None),
            Err(e @ GpuError::JobTimeout(_)) => (JobState::TimedOut, Some(e.to_string())),
            Err(e) => (JobState::Failed, Some(e.to_string())),
        };
        abandon.armed = false;
        let (waited, ran) = self.finish(id, state, error);
        drop(permit);
        outcome.map(|value| (value, JobStats { job_id: id, queue_depth_at_submit: depth, waited_ms: waited, ran_ms: ran }))
    }
}

/// Records a job as cancelled when its caller drops the future before the job ended.
struct AbandonGuard<'a> {
    queue: &'a GpuQueue,
    id: u64,
    armed: bool,
}

impl Drop for AbandonGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.queue.finish(self.id, JobState::Cancelled, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::mcp::clock::SystemClock;
    use std::sync::atomic::AtomicUsize;

    fn queue(max_queued: usize, queue_secs: u64, job_secs: u64) -> Arc<GpuQueue> {
        Arc::new(GpuQueue::new(
            GpuLimits { max_queued, queue_timeout: Duration::from_secs(queue_secs), job_timeout: Duration::from_secs(job_secs) },
            Arc::new(SystemClock),
        ))
    }

    fn spec(label: &str) -> JobSpec {
        JobSpec::new("test", label, "tester")
    }

    #[tokio::test(start_paused = true)]
    async fn jobs_never_overlap_and_run_in_submission_order() {
        let q = queue(8, 600, 600);
        let active = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));
        let order = Arc::new(Mutex::new(Vec::new()));
        let mut handles = Vec::new();
        for i in 0..5u32 {
            let (q, active, max_seen, order) = (q.clone(), active.clone(), max_seen.clone(), order.clone());
            handles.push(tokio::spawn(async move {
                let token = CancellationToken::new();
                let r = q
                    .run(spec(&format!("job{i}")), &token, |_ctx| async move {
                        let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                        max_seen.fetch_max(now, Ordering::SeqCst);
                        order.lock().unwrap().push(i);
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        active.fetch_sub(1, Ordering::SeqCst);
                        Ok(i)
                    })
                    .await;
                r.map(|(v, stats)| (v, stats.queue_depth_at_submit))
            }));
            tokio::task::yield_now().await;
        }
        let mut depths = Vec::new();
        for h in handles {
            let (_, depth) = h.await.unwrap().unwrap();
            depths.push(depth);
        }
        assert_eq!(max_seen.load(Ordering::SeqCst), 1);
        assert_eq!(*order.lock().unwrap(), vec![0, 1, 2, 3, 4]);
        assert_eq!(depths, vec![0, 1, 2, 3, 4]);
        let status = q.status();
        assert_eq!((status.completed, status.failed, status.cancelled), (5, 0, 0));
        assert!(status.running.is_none() && status.queued.is_empty());
        assert_eq!(status.recent.len(), 5);
        assert_eq!(status.recent[0].label, "job4");
    }

    #[tokio::test(start_paused = true)]
    async fn a_full_queue_refuses_more_jobs() {
        let q = queue(1, 600, 600);
        let release = Arc::new(tokio::sync::Semaphore::new(0));
        let spawn_blocking_job = |label: &'static str| {
            let (q, release) = (q.clone(), release.clone());
            tokio::spawn(async move {
                q.run(spec(label), &CancellationToken::new(), |_| async move {
                    release.acquire().await.unwrap().forget();
                    Ok(())
                })
                .await
                .map(|_| ())
            })
        };
        let first = spawn_blocking_job("first");
        tokio::task::yield_now().await;
        assert_eq!(q.status().running.as_ref().map(|j| j.label.as_str()), Some("first"));
        let second = spawn_blocking_job("second");
        tokio::task::yield_now().await;
        assert_eq!(q.status().queued.len(), 1);
        let third = q.run(spec("third"), &CancellationToken::new(), |_| async { Ok(()) }).await.unwrap_err();
        assert_eq!(third, GpuError::QueueFull { depth: 2, max: 1 });
        assert!(third.to_string().contains("queue is full"));
        release.add_permits(2);
        tokio::task::yield_now().await;
        first.await.unwrap().unwrap();
        second.await.unwrap().unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn a_waiting_job_that_is_cancelled_leaves_the_queue_and_the_others_continue() {
        let q = queue(8, 600, 600);
        let release = Arc::new(tokio::sync::Semaphore::new(0));
        let (q1, r1) = (q.clone(), release.clone());
        let first = tokio::spawn(async move {
            q1.run(spec("first"), &CancellationToken::new(), |_| async move {
                r1.acquire().await.unwrap().forget();
                Ok(1)
            })
            .await
        });
        tokio::task::yield_now().await;
        let token = CancellationToken::new();
        let (q2, t2) = (q.clone(), token.clone());
        let second = tokio::spawn(async move { q2.run(spec("second"), &t2, |_| async { Ok(2) }).await });
        tokio::task::yield_now().await;
        let queued_id = q.status().queued[0].id;
        assert!(q.cancel(queued_id));
        assert_eq!(second.await.unwrap().unwrap_err(), GpuError::Cancelled);
        assert!(q.status().queued.is_empty());
        assert!(!q.cancel(queued_id), "a finished job cannot be cancelled again");
        release.add_permits(1);
        assert_eq!(first.await.unwrap().unwrap().0, 1);
        let status = q.status();
        assert_eq!((status.completed, status.cancelled), (1, 1));
    }

    #[tokio::test(start_paused = true)]
    async fn the_caller_token_cancels_a_running_job() {
        let q = queue(8, 600, 600);
        let token = CancellationToken::new();
        let (q2, t2) = (q.clone(), token.clone());
        let job = tokio::spawn(async move {
            q2.run(spec("long"), &t2, |ctx| async move {
                ctx.cancel.cancelled().await;
                Ok(())
            })
            .await
        });
        tokio::task::yield_now().await;
        assert!(q.status().running.is_some());
        token.cancel();
        assert_eq!(job.await.unwrap().unwrap_err(), GpuError::Cancelled);
        assert!(q.status().running.is_none());
        assert_eq!(q.depth(), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn a_job_over_its_time_is_stopped_and_the_next_one_runs() {
        let q = queue(8, 600, 5);
        let token = CancellationToken::new();
        let slow = q.run(spec("slow"), &token, |_| async {
            tokio::time::sleep(Duration::from_secs(60)).await;
            Ok(())
        });
        assert_eq!(slow.await.unwrap_err(), GpuError::JobTimeout(Duration::from_secs(5)));
        let next = q.run(spec("next"), &CancellationToken::new(), |_| async { Ok("fine") }).await.unwrap();
        assert_eq!(next.0, "fine");
        let status = q.status();
        assert_eq!((status.completed, status.failed), (1, 1));
        assert!(status.recent.iter().any(|j| j.state == JobState::TimedOut && j.error.is_some()));
    }

    #[tokio::test(start_paused = true)]
    async fn a_job_that_waits_too_long_for_its_turn_is_dropped() {
        let q = queue(8, 10, 600);
        let (q1, release) = (q.clone(), Arc::new(tokio::sync::Semaphore::new(0)));
        let r1 = release.clone();
        let blocker = tokio::spawn(async move {
            q1.run(spec("blocker"), &CancellationToken::new(), |_| async move {
                r1.acquire().await.unwrap().forget();
                Ok(())
            })
            .await
        });
        tokio::task::yield_now().await;
        let waiting = q.run(spec("waiting"), &CancellationToken::new(), |_| async { Ok(()) }).await.unwrap_err();
        assert_eq!(waiting, GpuError::QueueTimeout(Duration::from_secs(10)));
        assert!(q.status().queued.is_empty());
        release.add_permits(1);
        blocker.await.unwrap().unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn dropping_a_job_frees_the_gpu_and_counts_it_as_cancelled() {
        let q = queue(8, 600, 600);
        let token = CancellationToken::new();
        let long = q.run(spec("dropped"), &token, |_| async {
            tokio::time::sleep(Duration::from_secs(100)).await;
            Ok(())
        });
        tokio::select! {
            _ = long => panic!("the job should still run"),
            _ = tokio::time::sleep(Duration::from_secs(1)) => {}
        }
        let status = q.status();
        assert!(status.running.is_none() && status.queued.is_empty());
        assert_eq!((status.cancelled, q.depth()), (1, 0));
        assert!(q.run(spec("next"), &CancellationToken::new(), |_| async { Ok(()) }).await.is_ok());
    }

    #[tokio::test(start_paused = true)]
    async fn dropping_a_waiting_job_removes_it_from_the_queue() {
        let q = queue(8, 600, 600);
        let release = Arc::new(tokio::sync::Semaphore::new(0));
        let (q1, r1) = (q.clone(), release.clone());
        let blocker = tokio::spawn(async move {
            q1.run(spec("blocker"), &CancellationToken::new(), |_| async move {
                r1.acquire().await.unwrap().forget();
                Ok(())
            })
            .await
        });
        tokio::task::yield_now().await;
        let token = CancellationToken::new();
        let waiting = q.run(spec("waiting"), &token, |_| async { Ok(()) });
        tokio::select! {
            _ = waiting => panic!("the GPU is busy"),
            _ = tokio::time::sleep(Duration::from_secs(1)) => {}
        }
        assert!(q.status().queued.is_empty());
        assert_eq!(q.status().cancelled, 1);
        release.add_permits(1);
        blocker.await.unwrap().unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn a_failing_job_reports_its_error_and_frees_the_gpu() {
        let q = queue(8, 600, 600);
        let err = q.run::<(), _, _>(spec("bad"), &CancellationToken::new(), |_| async { Err("model missing".to_string()) }).await.unwrap_err();
        assert_eq!(err, GpuError::Job("model missing".into()));
        assert!(q.run(spec("good"), &CancellationToken::new(), |_| async { Ok(()) }).await.is_ok());
        let status = q.status();
        assert_eq!(status.failed, 1);
        assert_eq!(status.recent.iter().find(|j| j.label == "bad").and_then(|j| j.error.clone()).as_deref(), Some("model missing"));
    }
}
