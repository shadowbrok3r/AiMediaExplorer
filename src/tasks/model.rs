//! Data model of the task registry: what a task is, how it is described and what it can report.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::command::{AppCommand, Scope};

pub type TaskId = u64;

/// What a task does; chips of the status bar are grouped by it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskKind {
    Scan,
    Thumbnails,
    Describe,
    Clip,
    Refine,
    ImageEdit,
    Database,
    LegacyImport,
    ThumbMigration,
    Backup,
    ModelLoad,
    Assistant,
    /// Made by the Tasks devtools window or a script.
    Debug,
}

impl TaskKind {
    pub const ALL: [TaskKind; 13] = [
        Self::Scan,
        Self::Thumbnails,
        Self::Describe,
        Self::Clip,
        Self::Refine,
        Self::ImageEdit,
        Self::Database,
        Self::LegacyImport,
        Self::ThumbMigration,
        Self::Backup,
        Self::ModelLoad,
        Self::Assistant,
        Self::Debug,
    ];

    /// Short name on chips.
    pub fn label(self) -> &'static str {
        match self {
            Self::Scan => "Scan",
            Self::Thumbnails => "Thumbnails",
            Self::Describe => "Describe",
            Self::Clip => "CLIP",
            Self::Refine => "Refine",
            Self::ImageEdit => "Edit",
            Self::Database => "Database",
            Self::LegacyImport => "Import",
            Self::ThumbMigration => "Thumb migration",
            Self::Backup => "Copy",
            Self::ModelLoad => "Models",
            Self::Assistant => "Assistant",
            Self::Debug => "Debug",
        }
    }

    /// True when the numbers of this kind count items, false when they are a share of a whole.
    pub fn counts_items(self) -> bool {
        matches!(self, Self::Thumbnails | Self::Describe | Self::Clip | Self::Backup | Self::Debug)
    }
}

/// Where a task is in its life.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskState {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl TaskState {
    pub fn is_active(self) -> bool {
        matches!(self, Self::Queued | Self::Running)
    }
}

/// A change a script or worker reports for a task.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskEvent {
    /// Starts a new task; the id of the command is ignored.
    Begin {
        kind: TaskKind,
        title: String,
        #[serde(default)]
        total: u64,
    },
    Progress { done: u64, total: u64 },
    Detail { text: String },
    Complete,
    Fail { error: String },
    /// Asks the task to stop through its cancel handle.
    Cancel,
}

/// Stops a running task; cheap and callable from any thread.
#[derive(Clone)]
pub struct CancelHandle(Arc<dyn Fn() + Send + Sync>);

impl CancelHandle {
    pub fn new(f: impl Fn() + Send + Sync + 'static) -> Self {
        Self(Arc::new(f))
    }

    /// Sets `flag` when called.
    pub fn from_flag(flag: Arc<AtomicBool>) -> Self {
        Self::new(move || flag.store(true, Ordering::Relaxed))
    }

    pub fn call(&self) {
        (self.0)();
    }
}

impl fmt::Debug for CancelHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("CancelHandle")
    }
}

/// One unit of background or user-started work.
#[derive(Clone, Debug)]
pub struct Task {
    pub id: TaskId,
    /// Identity of the task inside its source.
    pub key: Option<String>,
    pub source: Option<&'static str>,
    pub kind: TaskKind,
    pub title: String,
    pub state: TaskState,
    pub done: u64,
    pub total: u64,
    pub detail: String,
    pub started: Instant,
    pub ended: Option<Instant>,
    pub cancel: Option<CancelHandle>,
    pub cancel_requested: bool,
    pub error: Option<String>,
    /// Command that starts the work again.
    pub retry: Option<AppCommand>,
    /// Where the failed items can be listed.
    pub failure_scope: Option<Scope>,
    /// Started by a click or command; only these toast when they finish.
    pub user_initiated: bool,
    /// Age below which the status bar does not show the task.
    pub visible_after: Duration,
    /// Driven by commands only; cancelling ends it at once.
    pub scripted: bool,
    /// Shortest run that earns a completion toast.
    pub toast_after: Duration,
}

impl Task {
    /// Share done in 0..=1 when the total is known.
    pub fn fraction(&self) -> Option<f32> {
        (self.total > 0).then(|| (self.done as f32 / self.total as f32).clamp(0.0, 1.0))
    }

    /// Time the task ran, or has been running at `now`.
    pub fn elapsed(&self, now: Instant) -> Duration {
        self.ended.unwrap_or(now).saturating_duration_since(self.started)
    }

    /// Active and old enough to show.
    pub fn is_visible(&self, now: Instant) -> bool {
        self.state.is_active() && self.elapsed(now) >= self.visible_after
    }
}

/// Describes a task that is about to start.
#[derive(Clone, Debug)]
pub struct TaskSpec {
    pub kind: TaskKind,
    pub title: String,
    pub total: u64,
    pub detail: String,
    pub cancel: Option<CancelHandle>,
    pub retry: Option<AppCommand>,
    pub failure_scope: Option<Scope>,
    pub user_initiated: bool,
    pub visible_after: Duration,
    pub scripted: bool,
    pub toast_after: Duration,
}

impl TaskSpec {
    pub fn new(kind: TaskKind, title: impl Into<String>) -> Self {
        Self {
            kind,
            title: title.into(),
            total: 0,
            detail: String::new(),
            cancel: None,
            retry: None,
            failure_scope: None,
            user_initiated: false,
            visible_after: Duration::ZERO,
            scripted: false,
            toast_after: Duration::ZERO,
        }
    }

    pub fn total(mut self, total: u64) -> Self {
        self.total = total;
        self
    }

    pub fn detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = detail.into();
        self
    }

    pub fn cancel(mut self, cancel: CancelHandle) -> Self {
        self.cancel = Some(cancel);
        self
    }

    pub fn cancel_flag(self, flag: Arc<AtomicBool>) -> Self {
        self.cancel(CancelHandle::from_flag(flag))
    }

    pub fn retry(mut self, command: AppCommand) -> Self {
        self.retry = Some(command);
        self
    }

    pub fn failure_scope(mut self, scope: Scope) -> Self {
        self.failure_scope = Some(scope);
        self
    }

    pub fn user_initiated(mut self) -> Self {
        self.user_initiated = true;
        self
    }

    pub fn visible_after(mut self, delay: Duration) -> Self {
        self.visible_after = delay;
        self
    }

    /// Completion of a user-initiated task toasts only when it ran at least this long.
    pub fn toast_after(mut self, runtime: Duration) -> Self {
        self.toast_after = runtime;
        self
    }

    /// Marks the task as driven by commands only and gives it a no-op cancel handle.
    pub fn scripted(mut self) -> Self {
        self.scripted = true;
        self.cancel = Some(CancelHandle::new(|| {}));
        self
    }
}

/// What a source sees of one unit of work at poll time.
#[derive(Clone, Debug)]
pub struct Observation {
    pub key: String,
    pub spec: TaskSpec,
    pub done: u64,
    /// Waiting for a worker rather than running.
    pub queued: bool,
    /// The work stopped with this error.
    pub failed: Option<String>,
}

impl Observation {
    pub fn new(key: impl Into<String>, spec: TaskSpec) -> Self {
        Self { key: key.into(), spec, done: 0, queued: false, failed: None }
    }

    pub fn done(mut self, done: u64) -> Self {
        self.done = done;
        self
    }

    pub fn queued(mut self) -> Self {
        self.queued = true;
        self
    }

    pub fn failed(mut self, error: impl Into<String>) -> Self {
        self.failed = Some(error.into());
        self
    }
}

/// Something that needs the user, listed under "Needs attention".
#[derive(Clone, Debug, PartialEq)]
pub struct AttentionItem {
    pub key: String,
    pub kind: AttentionKind,
    pub title: String,
    pub detail: String,
    pub count: Option<usize>,
    /// Command behind the item's button.
    pub action: Option<AppCommand>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AttentionKind {
    ArchivePassword,
    FailedThumbs,
    ModelError,
    /// The library database could not be opened.
    Database,
    /// The ZeroClaw gateway refused the token.
    Gateway,
}

/// `3s`, `2m 05s` or `1h 02m`.
pub fn format_elapsed(d: Duration) -> String {
    let secs = d.as_secs();
    match secs {
        0..=59 => format!("{secs}s"),
        60..=3599 => format!("{}m {:02}s", secs / 60, secs % 60),
        _ => format!("{}h {:02}m", secs / 3600, (secs % 3600) / 60),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task(done: u64, total: u64) -> Task {
        Task {
            id: 1,
            key: None,
            source: None,
            kind: TaskKind::Debug,
            title: "t".into(),
            state: TaskState::Running,
            done,
            total,
            detail: String::new(),
            started: Instant::now(),
            ended: None,
            cancel: None,
            cancel_requested: false,
            error: None,
            retry: None,
            failure_scope: None,
            user_initiated: false,
            visible_after: Duration::ZERO,
            scripted: false,
            toast_after: Duration::ZERO,
        }
    }

    #[test]
    fn fraction_needs_a_total_and_clamps() {
        assert_eq!(task(1, 0).fraction(), None);
        assert_eq!(task(1, 4).fraction(), Some(0.25));
        assert_eq!(task(9, 4).fraction(), Some(1.0));
    }

    #[test]
    fn a_task_is_visible_only_when_active_and_old_enough() {
        let mut t = task(0, 0);
        t.visible_after = Duration::from_millis(500);
        assert!(!t.is_visible(t.started + Duration::from_millis(100)));
        assert!(t.is_visible(t.started + Duration::from_millis(500)));
        t.state = TaskState::Completed;
        assert!(!t.is_visible(t.started + Duration::from_secs(5)));
    }

    #[test]
    fn elapsed_stops_at_the_end_time() {
        let mut t = task(0, 0);
        t.ended = Some(t.started + Duration::from_secs(7));
        assert_eq!(t.elapsed(t.started + Duration::from_secs(100)), Duration::from_secs(7));
    }

    #[test]
    fn elapsed_text_scales_with_the_duration() {
        assert_eq!(format_elapsed(Duration::from_secs(5)), "5s");
        assert_eq!(format_elapsed(Duration::from_secs(125)), "2m 05s");
        assert_eq!(format_elapsed(Duration::from_secs(3725)), "1h 02m");
    }

    #[test]
    fn a_cancel_flag_is_set_by_the_handle() {
        let flag = Arc::new(AtomicBool::new(false));
        CancelHandle::from_flag(flag.clone()).call();
        assert!(flag.load(Ordering::Relaxed));
    }

    #[test]
    fn task_events_use_the_documented_json() {
        let begin: TaskEvent = serde_json::from_str(r#"{"begin":{"kind":"debug","title":"x","total":5}}"#).unwrap();
        assert_eq!(begin, TaskEvent::Begin { kind: TaskKind::Debug, title: "x".into(), total: 5 });
        assert_eq!(serde_json::from_str::<TaskEvent>("\"complete\"").unwrap(), TaskEvent::Complete);
        assert_eq!(serde_json::to_string(&TaskEvent::Cancel).unwrap(), "\"cancel\"");
    }
}
