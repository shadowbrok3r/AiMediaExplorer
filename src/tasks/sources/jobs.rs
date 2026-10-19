//! Long database jobs (legacy import, thumbnail migration) and the needs-attention items the explorers report.

use std::time::{Duration, Instant};

use super::{Report, SourceCtx, TaskSource, Throttle};
use crate::command::{AppCommand, Scope, SmartView};
use crate::tasks::model::{AttentionItem, AttentionKind, CancelHandle, Observation, TaskKind, TaskSpec};

pub const LEGACY_SOURCE: &str = "legacy-import";
pub const MIGRATION_SOURCE: &str = "thumb-migration";
pub const ATTENTION_SOURCE: &str = "explorer-attention";

/// How long a legacy import counts as running after its last running phase ended.
const PHASE_GAP: Duration = Duration::from_millis(1500);
/// How often the job engines are read.
const READ_EVERY: Duration = Duration::from_millis(250);

/// What the legacy import job shows right now.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LegacyView {
    /// Phase running now.
    pub running: Option<LegacyPhase>,
    /// Phase that failed, with the reason.
    pub failed: Option<(String, String)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LegacyPhase {
    pub label: String,
    pub done: u64,
    pub total: u64,
    pub what: String,
}

type LegacyReader = Box<dyn FnMut() -> LegacyView + Send>;

pub struct LegacyImportSource {
    read: LegacyReader,
    cancel: CancelHandle,
    throttle: Throttle,
    view: LegacyView,
    last_running: Option<(Instant, LegacyPhase)>,
}

impl LegacyImportSource {
    pub fn new(read: LegacyReader, cancel: CancelHandle) -> Self {
        Self { read, cancel, throttle: Throttle::new(READ_EVERY), view: LegacyView::default(), last_running: None }
    }

    /// Reads the job of the Legacy import window.
    pub fn live() -> Self {
        Self::new(
            Box::new(|| {
                let Some(job) = crate::ui::devtools::legacy_import::current_job() else { return LegacyView::default() };
                let running = job.running_phase().map(|p| {
                    let v = job.view(p);
                    LegacyPhase { label: p.label().to_string(), done: v.done, total: v.total, what: v.what }
                });
                let failed = job.failed_phase().map(|p| (p.label().to_string(), job.view(p).what));
                LegacyView { running, failed }
            }),
            CancelHandle::new(|| {
                if let Some(job) = crate::ui::devtools::legacy_import::current_job() {
                    job.cancel();
                }
            }),
        )
    }
}

impl TaskSource for LegacyImportSource {
    fn name(&self) -> &'static str {
        LEGACY_SOURCE
    }

    fn poll(&mut self, cx: &SourceCtx<'_>, out: &mut Report) {
        if self.throttle.ready(cx.now) {
            self.view = (self.read)();
            if let Some(phase) = self.view.running.clone() {
                self.last_running = Some((cx.now, phase));
            }
        }
        let view = self.view.clone();
        let held = self.last_running.as_ref().filter(|(at, _)| cx.now.saturating_duration_since(*at) < PHASE_GAP);
        let Some((_, phase)) = held else {
            self.last_running = None;
            if let Some((label, why)) = view.failed {
                let spec = TaskSpec::new(TaskKind::LegacyImport, format!("Legacy import: {label}")).user_initiated();
                out.observations.push(Observation::new("legacy-import", spec).failed(why));
            }
            return;
        };
        let spec = TaskSpec::new(TaskKind::LegacyImport, format!("Legacy import: {}", phase.label))
            .total(phase.total)
            .detail(phase.what.clone())
            .cancel(self.cancel.clone())
            .user_initiated();
        out.observations.push(Observation::new("legacy-import", spec).done(phase.done));
    }
}

/// What the thumbnail migrator shows right now.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MigrationView {
    pub running: bool,
    pub paused: bool,
    pub phase: String,
    pub message: String,
    pub processed: u64,
    pub total: Option<u64>,
    pub error: Option<String>,
}

type MigrationReader = Box<dyn FnMut() -> MigrationView + Send>;

pub struct MigrationSource {
    read: MigrationReader,
    cancel: CancelHandle,
    throttle: Throttle,
    view: MigrationView,
}

impl MigrationSource {
    pub fn new(read: MigrationReader, cancel: CancelHandle) -> Self {
        Self { read, cancel, throttle: Throttle::new(READ_EVERY), view: MigrationView::default() }
    }

    /// Reads the global thumbnail migrator.
    pub fn live() -> Self {
        use crate::database::thumb_migration::{Migrator, Phase};
        Self::new(
            Box::new(|| {
                let m = Migrator::global();
                let p = m.progress();
                MigrationView {
                    running: m.is_running(),
                    paused: m.is_paused(),
                    phase: p.phase.label().to_string(),
                    message: p.message,
                    processed: p.processed,
                    total: p.rows_total,
                    error: (p.phase == Phase::Failed).then(|| p.error.unwrap_or_else(|| "migration failed".to_string())),
                }
            }),
            CancelHandle::new(|| crate::database::thumb_migration::Migrator::global().cancel()),
        )
    }
}

impl TaskSource for MigrationSource {
    fn name(&self) -> &'static str {
        MIGRATION_SOURCE
    }

    fn poll(&mut self, cx: &SourceCtx<'_>, out: &mut Report) {
        if self.throttle.ready(cx.now) {
            self.view = (self.read)();
        }
        let v = self.view.clone();
        if !v.running {
            if let Some(error) = v.error {
                out.observations.push(Observation::new("thumb-migration", TaskSpec::new(TaskKind::ThumbMigration, "Thumbnail migration").user_initiated()).failed(error));
            }
            return;
        }
        let detail = if v.paused { "Paused".to_string() } else if v.message.is_empty() { v.phase.clone() } else { v.message.clone() };
        let spec = TaskSpec::new(TaskKind::ThumbMigration, "Thumbnail migration")
            .total(v.total.unwrap_or(0))
            .detail(detail)
            .cancel(self.cancel.clone())
            .user_initiated();
        out.observations.push(Observation::new("thumb-migration", spec).done(v.processed));
    }
}

/// Command behind the Show failures button of failed thumbnails.
pub fn failed_thumbs_command() -> AppCommand {
    AppCommand::open(Scope::Smart(SmartView::FailedThumbs))
}

/// Items for the needs-attention list from the explorers: failed thumbnails and archives waiting for a password.
pub struct ExplorerAttentionSource;

impl TaskSource for ExplorerAttentionSource {
    fn name(&self) -> &'static str {
        ATTENTION_SOURCE
    }

    fn poll(&mut self, cx: &SourceCtx<'_>, out: &mut Report) {
        let failed: usize = cx.app.explorers.iter().map(|e| e.failed_thumbs).sum();
        if failed > 0 {
            out.attention.push(AttentionItem {
                key: "failed-thumbs".into(),
                kind: AttentionKind::FailedThumbs,
                title: "Failed thumbnails".into(),
                detail: format!("{failed} files could not be decoded"),
                count: Some(failed),
                action: Some(failed_thumbs_command()),
            });
        }
        let prompts: usize = cx.app.explorers.iter().map(|e| e.archive_prompts).sum();
        if prompts > 0 {
            out.attention.push(AttentionItem {
                key: "archive-passwords".into(),
                kind: AttentionKind::ArchivePassword,
                title: "Encrypted archives".into(),
                detail: format!("{prompts} archives are waiting for a password"),
                count: Some(prompts),
                action: None,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::sources::{AppProbe, ExplorerProbe};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    fn poll(s: &mut dyn TaskSource, app: &AppProbe, at: Instant) -> Report {
        let mut out = Report::default();
        s.poll(&SourceCtx { now: at, app }, &mut out);
        out
    }

    fn phase(label: &str, done: u64, total: u64) -> LegacyPhase {
        LegacyPhase { label: label.into(), done, total, what: format!("{label} work") }
    }

    fn legacy(view: Arc<Mutex<LegacyView>>, flag: Arc<AtomicBool>) -> LegacyImportSource {
        LegacyImportSource::new(Box::new(move || view.lock().unwrap().clone()), CancelHandle::from_flag(flag))
    }

    #[test]
    fn a_running_phase_is_a_cancellable_user_task() {
        let view = Arc::new(Mutex::new(LegacyView { running: Some(phase("Copy", 25, 100)), failed: None }));
        let flag = Arc::new(AtomicBool::new(false));
        let mut s = legacy(view, flag.clone());
        let out = poll(&mut s, &AppProbe::default(), Instant::now());
        let o = &out.observations[0];
        assert_eq!((o.spec.title.as_str(), o.done, o.spec.total, o.spec.detail.as_str()), ("Legacy import: Copy", 25, 100, "Copy work"));
        assert!(o.spec.user_initiated && o.failed.is_none());
        o.spec.cancel.as_ref().unwrap().call();
        assert!(flag.load(Ordering::Relaxed));
    }

    #[test]
    fn the_gap_between_phases_does_not_end_the_task() {
        let view = Arc::new(Mutex::new(LegacyView { running: Some(phase("Copy", 1, 1)), failed: None }));
        let mut s = legacy(view.clone(), Arc::default());
        let t0 = Instant::now();
        poll(&mut s, &AppProbe::default(), t0);
        *view.lock().unwrap() = LegacyView::default();
        assert_eq!(poll(&mut s, &AppProbe::default(), t0 + Duration::from_millis(500)).observations.len(), 1);
        assert!(poll(&mut s, &AppProbe::default(), t0 + Duration::from_secs(3)).observations.is_empty());
    }

    #[test]
    fn a_failed_phase_is_reported_once_the_run_has_stopped() {
        let view = Arc::new(Mutex::new(LegacyView { running: None, failed: Some(("Import".into(), "bad record".into())) }));
        let mut s = legacy(view, Arc::default());
        let out = poll(&mut s, &AppProbe::default(), Instant::now());
        assert_eq!(out.observations[0].failed.as_deref(), Some("bad record"));
        assert_eq!(out.observations[0].spec.title, "Legacy import: Import");
    }

    #[test]
    fn no_job_means_no_task() {
        let mut s = legacy(Arc::default(), Arc::default());
        assert!(poll(&mut s, &AppProbe::default(), Instant::now()).observations.is_empty());
    }

    fn migration(view: Arc<Mutex<MigrationView>>, flag: Arc<AtomicBool>) -> MigrationSource {
        MigrationSource::new(Box::new(move || view.lock().unwrap().clone()), CancelHandle::from_flag(flag))
    }

    #[test]
    fn a_running_migration_shows_rows_processed_and_can_be_cancelled() {
        let view = Arc::new(Mutex::new(MigrationView { running: true, phase: "copying".into(), message: "Copying rows".into(), processed: 30, total: Some(120), ..Default::default() }));
        let flag = Arc::new(AtomicBool::new(false));
        let mut s = migration(view.clone(), flag.clone());
        let out = poll(&mut s, &AppProbe::default(), Instant::now());
        let o = &out.observations[0];
        assert_eq!((o.done, o.spec.total, o.spec.detail.as_str(), o.spec.kind), (30, 120, "Copying rows", TaskKind::ThumbMigration));
        o.spec.cancel.as_ref().unwrap().call();
        assert!(flag.load(Ordering::Relaxed));
        view.lock().unwrap().paused = true;
        assert_eq!(poll(&mut s, &AppProbe::default(), Instant::now() + Duration::from_secs(1)).observations[0].spec.detail, "Paused");
    }

    #[test]
    fn a_stopped_migration_is_silent_unless_it_failed() {
        let view = Arc::new(Mutex::new(MigrationView::default()));
        let mut s = migration(view.clone(), Arc::default());
        assert!(poll(&mut s, &AppProbe::default(), Instant::now()).observations.is_empty());
        view.lock().unwrap().error = Some("disk full".into());
        let out = poll(&mut s, &AppProbe::default(), Instant::now() + Duration::from_secs(1));
        assert_eq!(out.observations[0].failed.as_deref(), Some("disk full"));
    }

    fn explorer(failed: usize, prompts: usize) -> ExplorerProbe {
        ExplorerProbe { failed_thumbs: failed, archive_prompts: prompts, ..Default::default() }
    }

    #[test]
    fn failed_thumbnails_and_password_prompts_are_summed_over_the_explorers() {
        let app = AppProbe { explorers: vec![explorer(2, 1), explorer(3, 0)], ..Default::default() };
        let out = poll(&mut ExplorerAttentionSource, &app, Instant::now());
        assert_eq!(out.attention.len(), 2);
        assert_eq!((out.attention[0].kind, out.attention[0].count), (AttentionKind::FailedThumbs, Some(5)));
        assert_eq!(out.attention[0].action, Some(failed_thumbs_command()));
        assert_eq!((out.attention[1].kind, out.attention[1].count), (AttentionKind::ArchivePassword, Some(1)));
    }

    #[test]
    fn nothing_to_report_means_an_empty_list() {
        let app = AppProbe { explorers: vec![explorer(0, 0)], ..Default::default() };
        assert!(poll(&mut ExplorerAttentionSource, &app, Instant::now()).attention.is_empty());
    }

    #[test]
    fn the_failures_command_is_the_smart_scope() {
        assert_eq!(failed_thumbs_command().log_json(), r#"{"cmd":"open_scope","scope":"smart:failed-thumbs","target":"current"}"#);
    }
}
