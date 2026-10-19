//! The task registry: lifecycle, source reconciliation, toast policy and the needs-attention list.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::Instant;

use super::model::*;
use crate::command::ToastLevel;

/// Finished tasks kept for the history list.
pub const FINISHED_KEPT: usize = 40;

/// A toast the policy decided to show.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Notice {
    pub level: ToastLevel,
    pub text: String,
}

/// How a task ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    Completed,
    Failed(String),
    Cancelled,
}

/// Counts for the status bar and the Perf window.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub running: usize,
    pub queued: usize,
    pub finished: usize,
    pub failed: usize,
    pub attention: usize,
}

#[derive(Default)]
pub struct TaskRegistry {
    tasks: Vec<Task>,
    next_id: TaskId,
    notices: Vec<Notice>,
    attention: BTreeMap<&'static str, Vec<AttentionItem>>,
    last_failure: HashMap<String, String>,
    revision: u64,
}

impl TaskRegistry {
    pub fn new() -> Self {
        Self { next_id: 1, ..Default::default() }
    }

    /// Changes whenever anything a view of the registry shows changes.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    fn bump(&mut self) {
        self.revision += 1;
    }

    fn task_mut(&mut self, id: TaskId) -> Option<&mut Task> {
        self.tasks.iter_mut().find(|t| t.id == id)
    }

    pub fn get(&self, id: TaskId) -> Option<&Task> {
        self.tasks.iter().find(|t| t.id == id)
    }

    /// Every task, oldest first.
    pub fn tasks(&self) -> &[Task] {
        &self.tasks
    }

    pub fn active(&self) -> impl Iterator<Item = &Task> {
        self.tasks.iter().filter(|t| t.state.is_active())
    }

    pub fn finished(&self) -> impl DoubleEndedIterator<Item = &Task> {
        self.tasks.iter().filter(|t| !t.state.is_active())
    }

    pub fn attention(&self) -> Vec<AttentionItem> {
        self.attention.values().flatten().cloned().collect()
    }

    pub fn counts(&self) -> Counts {
        let mut c = Counts { attention: self.attention.values().map(Vec::len).sum(), ..Default::default() };
        for t in &self.tasks {
            match t.state {
                TaskState::Running => c.running += 1,
                TaskState::Queued => c.queued += 1,
                TaskState::Failed => {
                    c.finished += 1;
                    c.failed += 1;
                }
                TaskState::Completed | TaskState::Cancelled => c.finished += 1,
            }
        }
        c
    }

    /// Starts a task now.
    pub fn begin(&mut self, spec: TaskSpec) -> TaskId {
        self.begin_at(spec, Instant::now())
    }

    pub fn begin_at(&mut self, spec: TaskSpec, now: Instant) -> TaskId {
        self.insert(spec, None, None, 0, false, now)
    }

    fn insert(&mut self, spec: TaskSpec, source: Option<&'static str>, key: Option<String>, done: u64, queued: bool, now: Instant) -> TaskId {
        let id = self.next_id.max(1);
        self.next_id = id + 1;
        self.tasks.push(Task {
            id,
            key,
            source,
            kind: spec.kind,
            title: spec.title,
            state: if queued { TaskState::Queued } else { TaskState::Running },
            done,
            total: spec.total,
            detail: spec.detail,
            started: now,
            ended: None,
            cancel: spec.cancel,
            cancel_requested: false,
            error: None,
            retry: spec.retry,
            failure_scope: spec.failure_scope,
            user_initiated: spec.user_initiated,
            visible_after: spec.visible_after,
            scripted: spec.scripted,
            toast_after: spec.toast_after,
        });
        self.bump();
        id
    }

    pub fn set_progress(&mut self, id: TaskId, done: u64, total: u64) -> bool {
        let Some(t) = self.task_mut(id).filter(|t| t.state.is_active()) else { return false };
        if (t.done, t.total) != (done, total) {
            t.done = done;
            t.total = total;
            self.bump();
        }
        true
    }

    pub fn set_detail(&mut self, id: TaskId, text: impl Into<String>) -> bool {
        let text = text.into();
        let Some(t) = self.task_mut(id).filter(|t| t.state.is_active()) else { return false };
        if t.detail != text {
            t.detail = text;
            self.bump();
        }
        true
    }

    /// Ends an active task; a task whose cancel was requested ends as cancelled unless it failed.
    pub fn finish(&mut self, id: TaskId, outcome: Outcome) -> bool {
        self.finish_at(id, outcome, Instant::now())
    }

    pub fn finish_at(&mut self, id: TaskId, outcome: Outcome, now: Instant) -> bool {
        let Some(t) = self.task_mut(id).filter(|t| t.state.is_active()) else { return false };
        let outcome = match outcome {
            Outcome::Completed if t.cancel_requested => Outcome::Cancelled,
            other => other,
        };
        let runtime = now.saturating_duration_since(t.started);
        if outcome == Outcome::Completed && runtime < t.visible_after {
            self.tasks.retain(|t| t.id != id);
            self.bump();
            return true;
        }
        t.ended = Some(now);
        t.cancel = None;
        let notice = match outcome {
            Outcome::Completed => {
                t.state = TaskState::Completed;
                if t.total > 0 {
                    t.done = t.total;
                }
                (t.user_initiated && runtime >= t.toast_after).then(|| Notice { level: ToastLevel::Success, text: format!("{} finished", t.title) })
            }
            Outcome::Failed(error) => {
                t.state = TaskState::Failed;
                let text = format!("{} failed: {error}", t.title);
                t.error = Some(error);
                t.user_initiated.then(|| Notice { level: ToastLevel::Error, text })
            }
            Outcome::Cancelled => {
                t.state = TaskState::Cancelled;
                None
            }
        };
        self.notices.extend(notice);
        self.prune();
        self.bump();
        true
    }

    /// Asks an active task to stop; false when it has no cancel handle.
    pub fn request_cancel(&mut self, id: TaskId) -> bool {
        let Some(t) = self.task_mut(id).filter(|t| t.state.is_active()) else { return false };
        let Some(handle) = t.cancel.clone() else { return false };
        if !t.cancel_requested {
            t.cancel_requested = true;
            handle.call();
            self.bump();
        }
        true
    }

    pub fn is_cancel_requested(&self, id: TaskId) -> bool {
        self.get(id).is_some_and(|t| t.cancel_requested)
    }

    /// Applies a command-borne event; `Begin` returns the new task's id in the text.
    pub fn apply_event(&mut self, id: TaskId, event: TaskEvent, now: Instant) -> Result<String, String> {
        let missing = || format!("no active task {id}");
        match event {
            TaskEvent::Begin { kind, title, total } => {
                let spec = TaskSpec::new(kind, title).total(total).user_initiated().scripted();
                Ok(format!("started task {}", self.begin_at(spec, now)))
            }
            TaskEvent::Progress { done, total } => self.set_progress(id, done, total).then(|| format!("task {id} at {done}/{total}")).ok_or_else(missing),
            TaskEvent::Detail { text } => self.set_detail(id, text).then(|| format!("task {id} detail set")).ok_or_else(missing),
            TaskEvent::Complete => self.finish_at(id, Outcome::Completed, now).then(|| format!("task {id} completed")).ok_or_else(missing),
            TaskEvent::Fail { error } => self.finish_at(id, Outcome::Failed(error), now).then(|| format!("task {id} failed")).ok_or_else(missing),
            TaskEvent::Cancel => {
                if !self.get(id).is_some_and(|t| t.state.is_active()) {
                    Err(missing())
                } else if self.request_cancel(id) {
                    if self.get(id).is_some_and(|t| t.scripted) {
                        self.finish_at(id, Outcome::Cancelled, now);
                    }
                    Ok(format!("task {id} cancel requested"))
                } else {
                    Err(format!("task {id} cannot be cancelled"))
                }
            }
        }
    }

    /// Brings the tasks of `source` in line with what it observes now: seen work is created or updated, unseen work finishes.
    pub fn reconcile(&mut self, source: &'static str, observations: Vec<Observation>) {
        self.reconcile_at(source, observations, Instant::now());
    }

    pub fn reconcile_at(&mut self, source: &'static str, observations: Vec<Observation>, now: Instant) {
        let mut seen: HashSet<String> = HashSet::new();
        for obs in observations {
            seen.insert(obs.key.clone());
            match obs.failed.clone() {
                Some(error) => self.record_failure(source, obs, error, now),
                None => self.observe_active(source, obs, now),
            }
        }
        let gone: Vec<TaskId> = self
            .tasks
            .iter()
            .filter(|t| t.state.is_active() && t.source == Some(source) && t.key.as_ref().is_some_and(|k| !seen.contains(k)))
            .map(|t| t.id)
            .collect();
        for id in gone {
            self.finish_at(id, Outcome::Completed, now);
        }
    }

    fn active_by_key(&mut self, source: &'static str, key: &str) -> Option<&mut Task> {
        self.tasks.iter_mut().find(|t| t.state.is_active() && t.source == Some(source) && t.key.as_deref() == Some(key))
    }

    fn observe_active(&mut self, source: &'static str, obs: Observation, now: Instant) {
        self.last_failure.remove(&obs.key);
        let Observation { key, spec, done, queued, .. } = obs;
        let state = if queued { TaskState::Queued } else { TaskState::Running };
        if let Some(t) = self.active_by_key(source, &key) {
            let changed = (t.done, t.total, &t.title, &t.detail, t.state) != (done, spec.total, &spec.title, &spec.detail, state);
            t.done = done;
            t.total = spec.total;
            t.title = spec.title;
            t.detail = spec.detail;
            t.state = state;
            t.cancel = spec.cancel;
            t.retry = spec.retry;
            t.failure_scope = spec.failure_scope;
            if changed {
                self.bump();
            }
            return;
        }
        self.insert(spec, Some(source), Some(key), done, queued, now);
    }

    fn record_failure(&mut self, source: &'static str, obs: Observation, error: String, now: Instant) {
        if let Some(id) = self.active_by_key(source, &obs.key).map(|t| t.id) {
            self.last_failure.insert(obs.key, error.clone());
            self.finish_at(id, Outcome::Failed(error), now);
            return;
        }
        if self.last_failure.get(&obs.key) == Some(&error) {
            return;
        }
        self.last_failure.insert(obs.key.clone(), error.clone());
        let id = self.insert(obs.spec, Some(source), Some(obs.key), obs.done, false, now);
        self.finish_at(id, Outcome::Failed(error), now);
    }

    /// Replaces the needs-attention items `source` reports.
    pub fn set_attention(&mut self, source: &'static str, items: Vec<AttentionItem>) {
        let unchanged = self.attention.get(source).map_or(items.is_empty(), |old| *old == items);
        if unchanged {
            return;
        }
        if items.is_empty() {
            self.attention.remove(source);
        } else {
            self.attention.insert(source, items);
        }
        self.bump();
    }

    /// Toasts the policy produced since the last call.
    pub fn take_notices(&mut self) -> Vec<Notice> {
        std::mem::take(&mut self.notices)
    }

    /// Removes one finished task from the list.
    pub fn dismiss(&mut self, id: TaskId) -> bool {
        let before = self.tasks.len();
        self.tasks.retain(|t| t.id != id || t.state.is_active());
        let removed = self.tasks.len() != before;
        if removed {
            self.bump();
        }
        removed
    }

    /// Removes every finished task.
    pub fn clear_finished(&mut self) -> usize {
        let before = self.tasks.len();
        self.tasks.retain(|t| t.state.is_active());
        let removed = before - self.tasks.len();
        if removed > 0 {
            self.bump();
        }
        removed
    }

    fn prune(&mut self) {
        let finished = self.tasks.iter().filter(|t| !t.state.is_active()).count();
        if finished <= FINISHED_KEPT {
            return;
        }
        let mut drop_count = finished - FINISHED_KEPT;
        self.tasks.retain(|t| {
            if drop_count > 0 && !t.state.is_active() {
                drop_count -= 1;
                false
            } else {
                true
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    fn spec(title: &str) -> TaskSpec {
        TaskSpec::new(TaskKind::Debug, title)
    }

    #[test]
    fn a_task_runs_makes_progress_and_completes() {
        let mut r = TaskRegistry::new();
        let t0 = Instant::now();
        let id = r.begin_at(spec("work").total(10), t0);
        assert_eq!(r.get(id).unwrap().state, TaskState::Running);
        assert!(r.set_progress(id, 4, 10));
        assert_eq!(r.get(id).unwrap().fraction(), Some(0.4));
        assert!(r.set_detail(id, "halfway"));
        assert!(r.finish_at(id, Outcome::Completed, t0 + Duration::from_secs(3)));
        let t = r.get(id).unwrap();
        assert_eq!((t.state, t.done, t.elapsed(t0 + Duration::from_secs(99))), (TaskState::Completed, 10, Duration::from_secs(3)));
        assert!(!r.set_progress(id, 1, 10), "a finished task takes no more progress");
        assert!(!r.finish_at(id, Outcome::Completed, t0), "a finished task finishes once");
    }

    #[test]
    fn ids_are_unique_and_increasing() {
        let mut r = TaskRegistry::new();
        let a = r.begin(spec("a"));
        let b = r.begin(spec("b"));
        assert!(a >= 1 && b > a);
    }

    #[test]
    fn a_failure_keeps_its_error_and_user_started_tasks_toast_once() {
        let mut r = TaskRegistry::new();
        let id = r.begin(spec("Export").user_initiated());
        r.finish(id, Outcome::Failed("disk full".into()));
        assert_eq!(r.get(id).unwrap().error.as_deref(), Some("disk full"));
        assert_eq!(r.take_notices(), vec![Notice { level: ToastLevel::Error, text: "Export failed: disk full".into() }]);
        assert!(r.take_notices().is_empty());
    }

    #[test]
    fn user_started_completion_toasts_and_background_work_never_does() {
        let mut r = TaskRegistry::new();
        let user = r.begin(spec("Import").user_initiated());
        let background = r.begin(spec("Thumbnails"));
        let failing_background = r.begin(spec("Index"));
        r.finish(user, Outcome::Completed);
        r.finish(background, Outcome::Completed);
        r.finish(failing_background, Outcome::Failed("x".into()));
        assert_eq!(r.take_notices(), vec![Notice { level: ToastLevel::Success, text: "Import finished".into() }]);
    }

    #[test]
    fn a_short_user_task_completes_without_a_toast_and_a_long_one_toasts() {
        let mut r = TaskRegistry::new();
        let t0 = Instant::now();
        let quick = r.begin_at(spec("Quick").user_initiated().toast_after(Duration::from_secs(3)), t0);
        let long = r.begin_at(spec("Long").user_initiated().toast_after(Duration::from_secs(3)), t0);
        r.finish_at(quick, Outcome::Completed, t0 + Duration::from_secs(1));
        r.finish_at(long, Outcome::Completed, t0 + Duration::from_secs(5));
        assert_eq!(r.take_notices(), vec![Notice { level: ToastLevel::Success, text: "Long finished".into() }]);
    }

    #[test]
    fn a_failure_toasts_whatever_its_runtime() {
        let mut r = TaskRegistry::new();
        let t0 = Instant::now();
        let id = r.begin_at(spec("Quick").user_initiated().toast_after(Duration::from_secs(3)), t0);
        r.finish_at(id, Outcome::Failed("x".into()), t0);
        assert_eq!(r.take_notices().len(), 1);
    }

    #[test]
    fn a_task_that_ends_before_it_was_ever_shown_leaves_no_trace() {
        let mut r = TaskRegistry::new();
        let t0 = Instant::now();
        let blink = r.begin_at(spec("Blink").visible_after(Duration::from_millis(400)), t0);
        r.finish_at(blink, Outcome::Completed, t0 + Duration::from_millis(100));
        assert!(r.get(blink).is_none());
        let shown = r.begin_at(spec("Shown").visible_after(Duration::from_millis(400)), t0);
        r.finish_at(shown, Outcome::Completed, t0 + Duration::from_millis(500));
        assert_eq!(r.get(shown).unwrap().state, TaskState::Completed);
        let failed = r.begin_at(spec("Failed fast").visible_after(Duration::from_millis(400)), t0);
        r.finish_at(failed, Outcome::Failed("e".into()), t0 + Duration::from_millis(10));
        assert_eq!(r.get(failed).unwrap().state, TaskState::Failed, "a failure is always kept");
    }

    #[test]
    fn progress_updates_never_toast() {
        let mut r = TaskRegistry::new();
        let id = r.begin(spec("Scan").total(100).user_initiated());
        for i in 0..100 {
            r.set_progress(id, i, 100);
        }
        assert!(r.take_notices().is_empty());
    }

    #[test]
    fn cancelling_calls_the_handle_once_and_the_task_ends_cancelled_without_a_toast() {
        let flag = Arc::new(AtomicBool::new(false));
        let mut r = TaskRegistry::new();
        let id = r.begin(spec("Scan").user_initiated().cancel_flag(flag.clone()));
        assert!(r.request_cancel(id));
        assert!(flag.load(Ordering::Relaxed));
        assert!(r.is_cancel_requested(id));
        r.finish(id, Outcome::Completed);
        assert_eq!(r.get(id).unwrap().state, TaskState::Cancelled);
        assert!(r.take_notices().is_empty());
    }

    #[test]
    fn a_task_without_a_cancel_handle_cannot_be_cancelled() {
        let mut r = TaskRegistry::new();
        let id = r.begin(spec("No stop"));
        assert!(!r.request_cancel(id));
        assert!(!r.is_cancel_requested(id));
    }

    #[test]
    fn events_drive_the_lifecycle_like_commands_do() {
        let mut r = TaskRegistry::new();
        let now = Instant::now();
        let started = r.apply_event(0, TaskEvent::Begin { kind: TaskKind::Scan, title: "Fake".into(), total: 8 }, now).unwrap();
        let id: TaskId = started.rsplit(' ').next().unwrap().parse().unwrap();
        assert!(r.apply_event(id, TaskEvent::Progress { done: 2, total: 8 }, now).is_ok());
        assert!(r.apply_event(id, TaskEvent::Detail { text: "x".into() }, now).is_ok());
        let plain = r.begin_at(spec("plain"), now);
        assert!(r.apply_event(plain, TaskEvent::Cancel, now).is_err(), "no handle");
        assert!(r.apply_event(id, TaskEvent::Fail { error: "bad".into() }, now).is_ok());
        assert_eq!(r.get(id).unwrap().state, TaskState::Failed);
        assert!(r.apply_event(id, TaskEvent::Complete, now).is_err(), "already finished");
        assert!(r.apply_event(999, TaskEvent::Complete, now).is_err());
    }

    #[test]
    fn a_fake_task_cancels_at_once_because_nothing_works_on_it() {
        let mut r = TaskRegistry::new();
        let now = Instant::now();
        let started = r.apply_event(0, TaskEvent::Begin { kind: TaskKind::Debug, title: "Fake".into(), total: 8 }, now).unwrap();
        let id: TaskId = started.rsplit(' ').next().unwrap().parse().unwrap();
        assert!(r.apply_event(id, TaskEvent::Cancel, now).is_ok());
        assert_eq!(r.get(id).unwrap().state, TaskState::Cancelled);
        assert!(r.take_notices().is_empty());
    }

    fn obs(key: &str, done: u64, total: u64) -> Observation {
        Observation::new(key, spec(key).total(total)).done(done)
    }

    #[test]
    fn a_source_poll_creates_updates_and_finishes_one_task_per_key() {
        let mut r = TaskRegistry::new();
        let t0 = Instant::now();
        r.reconcile_at("src", vec![obs("a", 1, 10)], t0);
        r.reconcile_at("src", vec![obs("a", 5, 10), obs("b", 0, 0)], t0);
        assert_eq!(r.active().count(), 2);
        let a = r.tasks().iter().find(|t| t.key.as_deref() == Some("a")).unwrap();
        assert_eq!((a.done, a.started), (5, t0), "the same task keeps its start time");
        r.reconcile_at("src", vec![obs("b", 0, 0)], t0 + Duration::from_secs(2));
        let a = r.tasks().iter().find(|t| t.key.as_deref() == Some("a")).unwrap();
        assert_eq!(a.state, TaskState::Completed);
        assert_eq!(r.active().count(), 1);
    }

    #[test]
    fn sources_do_not_finish_each_others_tasks() {
        let mut r = TaskRegistry::new();
        r.reconcile("one", vec![obs("a", 0, 0)]);
        r.reconcile("two", vec![]);
        assert_eq!(r.active().count(), 1);
        let manual = r.begin(spec("manual"));
        r.reconcile("one", vec![]);
        assert!(r.get(manual).unwrap().state.is_active(), "tasks without a source stay");
    }

    #[test]
    fn an_unchanged_poll_does_not_change_the_revision() {
        let mut r = TaskRegistry::new();
        r.reconcile("src", vec![obs("a", 1, 10)]);
        let rev = r.revision();
        for _ in 0..5 {
            r.reconcile("src", vec![obs("a", 1, 10)]);
        }
        assert_eq!(r.revision(), rev);
        r.reconcile("src", vec![obs("a", 2, 10)]);
        assert!(r.revision() > rev);
    }

    #[test]
    fn a_queued_observation_becomes_running_when_it_starts() {
        let mut r = TaskRegistry::new();
        r.reconcile("src", vec![Observation::new("a", spec("a")).queued()]);
        assert_eq!(r.tasks()[0].state, TaskState::Queued);
        r.reconcile("src", vec![obs("a", 0, 0)]);
        assert_eq!(r.tasks()[0].state, TaskState::Running);
    }

    #[test]
    fn a_reported_failure_is_recorded_once_until_the_work_runs_again() {
        let mut r = TaskRegistry::new();
        let failing = || Observation::new("m", spec("Model")).failed("out of memory");
        r.reconcile("src", vec![failing()]);
        r.reconcile("src", vec![failing()]);
        r.reconcile("src", vec![failing()]);
        assert_eq!(r.finished().count(), 1);
        assert_eq!(r.finished().next().unwrap().error.as_deref(), Some("out of memory"));
        r.reconcile("src", vec![obs("m", 0, 0)]);
        r.reconcile("src", vec![failing()]);
        assert_eq!(r.finished().filter(|t| t.state == TaskState::Failed).count(), 2, "a new failure after a retry is a new record");
    }

    #[test]
    fn a_failure_of_running_work_ends_that_task() {
        let mut r = TaskRegistry::new();
        r.reconcile("src", vec![obs("m", 1, 2)]);
        r.reconcile("src", vec![Observation::new("m", spec("m")).failed("broke")]);
        assert_eq!(r.active().count(), 0);
        assert_eq!(r.finished().next().unwrap().error.as_deref(), Some("broke"));
    }

    #[test]
    fn the_finished_history_is_capped_and_never_drops_active_tasks() {
        let mut r = TaskRegistry::new();
        let running = r.begin(spec("long"));
        for i in 0..(FINISHED_KEPT + 10) {
            let id = r.begin(spec(&i.to_string()));
            r.finish(id, Outcome::Completed);
        }
        assert_eq!(r.finished().count(), FINISHED_KEPT);
        assert!(r.get(running).is_some());
        assert_eq!(r.finished().next().unwrap().title, "10", "the oldest finished tasks go first");
    }

    #[test]
    fn dismiss_and_clear_remove_only_finished_tasks() {
        let mut r = TaskRegistry::new();
        let live = r.begin(spec("live"));
        let done = r.begin(spec("done"));
        r.finish(done, Outcome::Completed);
        assert!(!r.dismiss(live));
        assert!(r.dismiss(done));
        let again = r.begin(spec("again"));
        r.finish(again, Outcome::Failed("e".into()));
        assert_eq!(r.clear_finished(), 1);
        assert_eq!(r.tasks().len(), 1);
    }

    #[test]
    fn counts_summarise_the_states() {
        let mut r = TaskRegistry::new();
        r.begin(spec("running"));
        r.reconcile("s", vec![Observation::new("q", spec("q")).queued()]);
        let ok = r.begin(spec("ok"));
        let bad = r.begin(spec("bad"));
        r.finish(ok, Outcome::Completed);
        r.finish(bad, Outcome::Failed("x".into()));
        assert_eq!(r.counts(), Counts { running: 1, queued: 1, finished: 2, failed: 1, attention: 0 });
    }

    fn item(key: &str) -> AttentionItem {
        AttentionItem { key: key.into(), kind: AttentionKind::FailedThumbs, title: "t".into(), detail: String::new(), count: Some(3), action: None }
    }

    #[test]
    fn attention_items_are_replaced_per_source_and_cleared_when_empty() {
        let mut r = TaskRegistry::new();
        r.set_attention("a", vec![item("x")]);
        r.set_attention("b", vec![item("y"), item("z")]);
        assert_eq!(r.counts().attention, 3);
        let rev = r.revision();
        r.set_attention("a", vec![item("x")]);
        assert_eq!(r.revision(), rev, "an identical report is not a change");
        r.set_attention("a", vec![]);
        assert_eq!(r.attention().len(), 2);
        r.set_attention("a", vec![]);
        assert_eq!(r.revision(), rev + 1);
    }
}
