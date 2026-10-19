//! Task sources: adapters that turn the counters of existing engines into task observations.

pub mod describe;
pub mod engines;
pub mod jobs;
pub mod scan;
pub mod thumbs;

use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::model::{AttentionItem, CancelHandle, Observation, TaskKind};
use super::registry::TaskRegistry;

/// Application state the sources may read; built once per frame without touching rows.
#[derive(Clone, Debug, Default)]
pub struct AppProbe {
    pub explorers: Vec<ExplorerProbe>,
    /// Stops the running image edit.
    pub image_edit_cancel: Option<CancelHandle>,
}

/// The few facts one explorer tab contributes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExplorerProbe {
    pub tab: String,
    /// Id of the scan feeding the tab, when the explorer exposes it.
    pub scan_id: Option<u64>,
    pub scan_progress: f32,
    pub scan_done: bool,
    pub failed_thumbs: usize,
    /// Encrypted archives waiting for a password.
    pub archive_prompts: usize,
}

impl ExplorerProbe {
    pub fn is_scanning(&self) -> bool {
        self.scan_progress > 0.0 && !self.scan_done
    }
}

/// What a source can see when polled.
pub struct SourceCtx<'a> {
    pub now: Instant,
    pub app: &'a AppProbe,
}

/// What one poll of a source found.
#[derive(Default)]
pub struct Report {
    pub observations: Vec<Observation>,
    pub attention: Vec<AttentionItem>,
}

/// A watcher of one engine; `poll` runs every frame and must not walk rows.
pub trait TaskSource: Send {
    fn name(&self) -> &'static str;
    fn poll(&mut self, cx: &SourceCtx<'_>, out: &mut Report);
}

/// Limits how often a source reads its engine.
#[derive(Debug)]
pub struct Throttle {
    every: Duration,
    last: Option<Instant>,
}

impl Throttle {
    pub fn new(every: Duration) -> Self {
        Self { every, last: None }
    }

    /// True when `every` has passed since the last time this returned true.
    pub fn ready(&mut self, now: Instant) -> bool {
        match self.last {
            Some(last) if now.saturating_duration_since(last) < self.every => false,
            _ => {
                self.last = Some(now);
                true
            }
        }
    }
}

/// Progress of work that arrives in bursts and has no fixed total: done counts since the burst began, total adds what is left.
#[derive(Debug, Default)]
pub struct Burst {
    baseline: Option<u64>,
    quiet_since: Option<Instant>,
}

/// How long a burst must stay empty before it counts as over.
pub const BURST_QUIET: Duration = Duration::from_millis(600);

impl Burst {
    /// `(done, total)` while the burst lasts and `None` once it has been quiet for [`BURST_QUIET`].
    pub fn observe(&mut self, now: Instant, finished_total: u64, remaining: u64) -> Option<(u64, u64)> {
        if remaining > 0 {
            self.quiet_since = None;
            let base = *self.baseline.get_or_insert(finished_total);
            let done = finished_total.saturating_sub(base);
            return Some((done, done + remaining));
        }
        let Some(base) = self.baseline else { return None };
        let since = *self.quiet_since.get_or_insert(now);
        if now.saturating_duration_since(since) < BURST_QUIET {
            let done = finished_total.saturating_sub(base);
            return Some((done, done));
        }
        self.baseline = None;
        self.quiet_since = None;
        None
    }
}

/// Owns the sources and feeds the registry from them.
#[derive(Default)]
pub struct Poller {
    sources: Vec<Box<dyn TaskSource>>,
}

impl Poller {
    pub fn new() -> Self {
        Self::default()
    }

    /// The sources of the running application.
    pub fn standard() -> Self {
        let mut p = Self::new();
        p.add(Box::new(scan::ScanSource::live()));
        p.add(Box::new(thumbs::ThumbSource::live()));
        p.add(Box::new(describe::DescribeSource::live()));
        p.add(Box::new(engines::EngineSource::live()));
        p.add(Box::new(jobs::LegacyImportSource::live()));
        p.add(Box::new(jobs::MigrationSource::live()));
        p.add(Box::new(jobs::ExplorerAttentionSource));
        p
    }

    pub fn add(&mut self, source: Box<dyn TaskSource>) {
        self.sources.push(source);
    }

    pub fn len(&self) -> usize {
        self.sources.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    /// Polls every source and reconciles `registry` with what they found.
    pub fn poll(&mut self, registry: &Mutex<TaskRegistry>, app: &AppProbe, now: Instant) {
        let cx = SourceCtx { now, app };
        let reports: Vec<(&'static str, Report)> = self
            .sources
            .iter_mut()
            .map(|source| {
                let mut report = Report::default();
                source.poll(&cx, &mut report);
                (source.name(), report)
            })
            .collect();
        let mut registry = registry.lock().unwrap_or_else(|e| e.into_inner());
        let busy_elsewhere = busy_with_dedicated_task(&reports, &registry);
        for (name, mut report) in reports {
            if busy_elsewhere {
                report.observations.retain(|o| !(name == engines::SOURCE && o.key == engines::DB_KEY));
            }
            registry.reconcile_at(name, report.observations, now);
            registry.set_attention(name, report.attention);
        }
    }
}

/// Tasks of these kinds already explain the DB engine's activity.
const DB_EXPLAINING_KINDS: [TaskKind; 3] = [TaskKind::Database, TaskKind::LegacyImport, TaskKind::ThumbMigration];

fn busy_with_dedicated_task(reports: &[(&'static str, Report)], registry: &TaskRegistry) -> bool {
    let explaining = |kind: TaskKind| DB_EXPLAINING_KINDS.contains(&kind);
    let from_reports = reports.iter().filter(|(name, _)| *name != engines::SOURCE).flat_map(|(_, r)| &r.observations).any(|o| o.failed.is_none() && explaining(o.spec.kind));
    from_reports || registry.active().any(|t| t.source != Some(engines::SOURCE) && explaining(t.kind))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::model::{TaskSpec, TaskState};

    #[test]
    fn a_throttle_lets_one_read_through_per_interval() {
        let mut t = Throttle::new(Duration::from_millis(100));
        let t0 = Instant::now();
        assert!(t.ready(t0));
        assert!(!t.ready(t0 + Duration::from_millis(99)));
        assert!(t.ready(t0 + Duration::from_millis(100)));
        assert!(!t.ready(t0 + Duration::from_millis(150)));
    }

    #[test]
    fn a_burst_reports_done_since_it_began_and_ends_after_a_quiet_spell() {
        let mut b = Burst::default();
        let t0 = Instant::now();
        assert_eq!(b.observe(t0, 100, 0), None, "no work, no burst");
        assert_eq!(b.observe(t0, 100, 50), Some((0, 50)));
        assert_eq!(b.observe(t0, 120, 30), Some((20, 50)));
        assert_eq!(b.observe(t0 + Duration::from_millis(10), 150, 0), Some((50, 50)), "briefly empty still counts");
        assert_eq!(b.observe(t0 + Duration::from_millis(700), 150, 0), None, "quiet long enough ends it");
        assert_eq!(b.observe(t0 + Duration::from_millis(800), 150, 10), Some((0, 10)), "a new burst starts from zero");
    }

    #[test]
    fn work_resuming_during_the_quiet_spell_continues_the_burst() {
        let mut b = Burst::default();
        let t0 = Instant::now();
        b.observe(t0, 0, 10);
        b.observe(t0 + Duration::from_millis(100), 10, 0);
        assert_eq!(b.observe(t0 + Duration::from_millis(300), 10, 5), Some((10, 15)));
    }

    struct Fixed(&'static str, Vec<Observation>, Vec<AttentionItem>);

    impl TaskSource for Fixed {
        fn name(&self) -> &'static str {
            self.0
        }

        fn poll(&mut self, _: &SourceCtx<'_>, out: &mut Report) {
            out.observations = self.1.clone();
            out.attention = self.2.clone();
        }
    }

    fn obs(key: &str, kind: TaskKind) -> Observation {
        Observation::new(key, TaskSpec::new(kind, key))
    }

    #[test]
    fn the_poller_reconciles_each_source_into_the_registry() {
        let registry = Mutex::new(TaskRegistry::new());
        let mut poller = Poller::new();
        poller.add(Box::new(Fixed("one", vec![obs("a", TaskKind::Scan)], vec![])));
        poller.add(Box::new(Fixed("two", vec![obs("b", TaskKind::Clip), obs("c", TaskKind::Refine)], vec![])));
        poller.poll(&registry, &AppProbe::default(), Instant::now());
        assert_eq!(registry.lock().unwrap().active().count(), 3);
        let mut poller = Poller::new();
        poller.add(Box::new(Fixed("one", vec![], vec![])));
        poller.poll(&registry, &AppProbe::default(), Instant::now());
        let r = registry.lock().unwrap();
        assert_eq!(r.active().count(), 2, "only the silent source's task finished");
        assert_eq!(r.finished().next().unwrap().state, TaskState::Completed);
    }

    #[test]
    fn the_db_engine_task_yields_to_a_dedicated_database_task() {
        let registry = Mutex::new(TaskRegistry::new());
        let engine_db = || Fixed(engines::SOURCE, vec![obs(engines::DB_KEY, TaskKind::Database)], vec![]);
        let mut poller = Poller::new();
        poller.add(Box::new(engine_db()));
        poller.poll(&registry, &AppProbe::default(), Instant::now());
        assert_eq!(registry.lock().unwrap().active().count(), 1);

        let mut poller = Poller::new();
        poller.add(Box::new(engine_db()));
        poller.add(Box::new(Fixed("migration", vec![obs("m", TaskKind::ThumbMigration)], vec![])));
        poller.poll(&registry, &AppProbe::default(), Instant::now());
        let r = registry.lock().unwrap();
        let kinds: Vec<_> = r.active().map(|t| t.kind).collect();
        assert_eq!(kinds, vec![TaskKind::ThumbMigration], "the generic database task ended");
    }

    #[test]
    fn the_standard_poller_watches_every_engine_and_polls_without_panicking() {
        let mut poller = Poller::standard();
        assert_eq!(poller.len(), 7);
        let registry = Mutex::new(TaskRegistry::new());
        let now = Instant::now();
        for frame in 0..3 {
            poller.poll(&registry, &AppProbe::default(), now + Duration::from_millis(frame * 300));
        }
    }

    #[test]
    fn attention_reports_follow_their_source() {
        use crate::tasks::model::AttentionKind;
        let registry = Mutex::new(TaskRegistry::new());
        let item = AttentionItem { key: "k".into(), kind: AttentionKind::FailedThumbs, title: "t".into(), detail: String::new(), count: None, action: None };
        let mut poller = Poller::new();
        poller.add(Box::new(Fixed("src", vec![], vec![item])));
        poller.poll(&registry, &AppProbe::default(), Instant::now());
        assert_eq!(registry.lock().unwrap().attention().len(), 1);
        let mut poller = Poller::new();
        poller.add(Box::new(Fixed("src", vec![], vec![])));
        poller.poll(&registry, &AppProbe::default(), Instant::now());
        assert!(registry.lock().unwrap().attention().is_empty());
    }
}
