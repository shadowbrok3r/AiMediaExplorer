//! Task registry: one list of everything the app is working on, fed by workers and by sources that watch existing engines.

pub mod chips;
pub mod model;
pub mod registry;
pub mod sources;

use std::sync::Mutex;

use once_cell::sync::Lazy;

pub use chips::{ActivityChip, activity_chips};
pub use model::*;
pub use registry::{Counts, Notice, Outcome, TaskRegistry};
pub use sources::{AppProbe, ExplorerProbe, Poller, SourceCtx, TaskSource};

/// How often the frame loop wakes to poll the sources while nothing else repaints.
pub const IDLE_POLL: std::time::Duration = std::time::Duration::from_secs(1);

static REGISTRY: Lazy<Mutex<TaskRegistry>> = Lazy::new(|| Mutex::new(TaskRegistry::new()));

/// Runs `f` on the application's task registry.
pub fn with<R>(f: impl FnOnce(&mut TaskRegistry) -> R) -> R {
    f(&mut REGISTRY.lock().unwrap_or_else(|e| e.into_inner()))
}

fn wake() {
    crate::command::request_repaint();
}

/// Polls the sources of `poller` into the application's registry.
pub fn poll(poller: &mut Poller, app: &AppProbe) {
    poller.poll(&REGISTRY, app, std::time::Instant::now());
}

/// Starts a task on the application's registry and returns the worker's handle.
pub fn start(spec: TaskSpec) -> TaskHandle {
    let id = with(|r| r.begin(spec));
    wake();
    TaskHandle { reporter: Reporter { id }, finished: false }
}

/// Progress reporting for one task, copyable into worker closures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reporter {
    id: TaskId,
}

impl Reporter {
    pub fn id(&self) -> TaskId {
        self.id
    }

    pub fn progress(&self, done: u64, total: u64) {
        with(|r| r.set_progress(self.id, done, total));
        wake();
    }

    pub fn detail(&self, text: impl Into<String>) {
        with(|r| r.set_detail(self.id, text));
        wake();
    }

    /// True once the user asked the task to stop.
    pub fn cancel_requested(&self) -> bool {
        with(|r| r.is_cancel_requested(self.id))
    }
}

/// A worker's ownership of its task; an unfinished handle ends the task as cancelled when dropped.
#[derive(Debug)]
pub struct TaskHandle {
    reporter: Reporter,
    finished: bool,
}

impl TaskHandle {
    pub fn id(&self) -> TaskId {
        self.reporter.id
    }

    pub fn reporter(&self) -> Reporter {
        self.reporter
    }

    pub fn progress(&self, done: u64, total: u64) {
        self.reporter.progress(done, total);
    }

    pub fn detail(&self, text: impl Into<String>) {
        self.reporter.detail(text);
    }

    pub fn cancel_requested(&self) -> bool {
        self.reporter.cancel_requested()
    }

    pub fn complete(mut self) {
        self.end(Outcome::Completed);
    }

    pub fn fail(mut self, error: impl Into<String>) {
        self.end(Outcome::Failed(error.into()));
    }

    pub fn cancelled(mut self) {
        self.end(Outcome::Cancelled);
    }

    fn end(&mut self, outcome: Outcome) {
        self.finished = true;
        with(|r| r.finish(self.reporter.id, outcome));
        wake();
    }
}

impl Drop for TaskHandle {
    fn drop(&mut self) {
        if !self.finished {
            self.end(Outcome::Cancelled);
        }
    }
}

/// Runs `work` as a task: it completes on `Ok` and fails with the error text on `Err`.
pub async fn track<T, E, Fut>(spec: TaskSpec, work: impl FnOnce(Reporter) -> Fut) -> Result<T, E>
where
    Fut: std::future::Future<Output = Result<T, E>>,
    E: std::fmt::Display,
{
    let handle = start(spec);
    match work(handle.reporter()).await {
        Ok(value) => {
            handle.complete();
            Ok(value)
        }
        Err(e) => {
            handle.fail(e.to_string());
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_handle_reports_progress_and_completes_its_task() {
        let handle = start(TaskSpec::new(TaskKind::Debug, "handle-test-complete").total(4));
        let id = handle.id();
        handle.progress(3, 4);
        assert_eq!(with(|r| r.get(id).map(|t| (t.done, t.total))), Some((3, 4)));
        handle.complete();
        assert_eq!(with(|r| r.get(id).map(|t| t.state)), Some(TaskState::Completed));
    }

    #[test]
    fn a_dropped_handle_ends_the_task_as_cancelled() {
        let id = {
            let handle = start(TaskSpec::new(TaskKind::Debug, "handle-test-drop"));
            handle.id()
        };
        assert_eq!(with(|r| r.get(id).map(|t| t.state)), Some(TaskState::Cancelled));
    }

    #[test]
    fn a_failed_handle_keeps_the_error() {
        let handle = start(TaskSpec::new(TaskKind::Debug, "handle-test-fail"));
        let id = handle.id();
        handle.fail("nope");
        assert_eq!(with(|r| r.get(id).and_then(|t| t.error.clone())).as_deref(), Some("nope"));
    }

    #[test]
    fn a_handle_sees_a_cancel_request() {
        let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let handle = start(TaskSpec::new(TaskKind::Debug, "handle-test-cancel").cancel_flag(flag.clone()));
        assert!(!handle.cancel_requested());
        with(|r| r.request_cancel(handle.id()));
        assert!(handle.cancel_requested());
        assert!(flag.load(std::sync::atomic::Ordering::Relaxed));
    }

    #[tokio::test]
    async fn track_completes_on_ok_and_fails_on_err() {
        let ok: Result<u8, String> = track(TaskSpec::new(TaskKind::Debug, "track-ok"), |reporter| async move {
            reporter.progress(1, 1);
            Ok(7)
        })
        .await;
        assert_eq!(ok, Ok(7));
        let err: Result<u8, String> = track(TaskSpec::new(TaskKind::Debug, "track-err").user_initiated(), |_| std::future::ready(Err("bad".to_string()))).await;
        assert_eq!(err, Err("bad".to_string()));
        let (done, failed) = with(|r| {
            let state = |title: &str| r.tasks().iter().find(|t| t.title == title).map(|t| t.state);
            (state("track-ok"), state("track-err"))
        });
        assert_eq!((done, failed), (Some(TaskState::Completed), Some(TaskState::Failed)));
    }
}
