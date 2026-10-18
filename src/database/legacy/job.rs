//! Shared progress state for a legacy import run, polled by the CLI reporter and the devtools window.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::database::snapshot::ProgressFn;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum PhaseId {
    Engine,
    Copy,
    Export,
    Convert,
    Snapshot,
    Import,
    Verify,
    Thumbs,
}

pub const PHASES: [PhaseId; 8] = [PhaseId::Engine, PhaseId::Copy, PhaseId::Export, PhaseId::Convert, PhaseId::Snapshot, PhaseId::Import, PhaseId::Verify, PhaseId::Thumbs];

impl PhaseId {
    pub fn label(self) -> &'static str {
        match self {
            PhaseId::Engine => "Engine",
            PhaseId::Copy => "Copy",
            PhaseId::Export => "Export",
            PhaseId::Convert => "Convert",
            PhaseId::Snapshot => "Snapshot",
            PhaseId::Import => "Import",
            PhaseId::Verify => "Verify",
            PhaseId::Thumbs => "Thumbnails",
        }
    }

    fn index(self) -> usize {
        PHASES.iter().position(|p| *p == self).unwrap_or(0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum PhaseState {
    Pending,
    Running,
    Done,
    Skipped,
    Failed,
}

impl PhaseState {
    fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Running,
            2 => Self::Done,
            3 => Self::Skipped,
            4 => Self::Failed,
            _ => Self::Pending,
        }
    }
}

#[derive(Default)]
struct Slot {
    state: AtomicU8,
    done: AtomicU64,
    total: AtomicU64,
    what: Mutex<String>,
    started: Mutex<Option<Instant>>,
    took_ms: AtomicU64,
}

#[derive(Clone, Debug, Serialize)]
pub struct PhaseView {
    pub phase: PhaseId,
    pub state: PhaseState,
    pub done: u64,
    pub total: u64,
    pub what: String,
    pub elapsed_ms: u64,
    pub eta_ms: Option<u64>,
}

impl PhaseView {
    pub fn fraction(&self) -> Option<f32> {
        (self.total > 0).then(|| (self.done as f64 / self.total as f64).clamp(0.0, 1.0) as f32)
    }

    pub fn elapsed(&self) -> Duration {
        Duration::from_millis(self.elapsed_ms)
    }

    pub fn eta(&self) -> Option<Duration> {
        self.eta_ms.map(Duration::from_millis)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum EngineSource {
    Explicit,
    Environment,
    Cache,
    Downloaded,
}

impl EngineSource {
    pub fn label(&self) -> &'static str {
        match self {
            EngineSource::Explicit => "given on the command line",
            EngineSource::Environment => "from AME_SURREAL_BIN",
            EngineSource::Cache => "cached",
            EngineSource::Downloaded => "downloaded",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub enum EngineStatus {
    #[default]
    Unknown,
    Checking,
    Found { path: String, version: String, source: EngineSource, checksum_ok: bool },
    Downloading { done: u64, total: u64 },
    Verifying,
    Failed(String),
}

#[derive(Debug)]
pub struct Cancelled;

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("cancelled")
    }
}

impl std::error::Error for Cancelled {}

pub fn is_cancelled(e: &anyhow::Error) -> bool {
    e.downcast_ref::<Cancelled>().is_some()
}

const LOG_CAP: usize = 300;

pub struct Job {
    slots: [Slot; 8],
    cancel: AtomicBool,
    log: Mutex<VecDeque<String>>,
    engine: Mutex<EngineStatus>,
    repaint: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
    started: Instant,
}

impl Job {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            slots: Default::default(),
            cancel: AtomicBool::new(false),
            log: Mutex::new(VecDeque::new()),
            engine: Mutex::new(EngineStatus::Unknown),
            repaint: Mutex::new(None),
            started: Instant::now(),
        })
    }

    pub fn set_repaint(&self, f: Option<Arc<dyn Fn() + Send + Sync>>) {
        if let Ok(mut g) = self.repaint.lock() {
            *g = f;
        }
    }

    fn poke(&self) {
        if let Some(f) = self.repaint.lock().ok().and_then(|g| g.clone()) {
            f();
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
        self.note("Cancel requested");
    }

    pub fn is_cancel_requested(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }

    pub fn check_cancel(&self) -> anyhow::Result<()> {
        if self.is_cancel_requested() { Err(anyhow::Error::new(Cancelled)) } else { Ok(()) }
    }

    pub fn note(&self, msg: impl Into<String>) {
        let msg = msg.into();
        log::info!("[legacy] {msg}");
        if let Ok(mut l) = self.log.lock() {
            if l.len() >= LOG_CAP {
                l.pop_front();
            }
            l.push_back(format!("{}  {msg}", chrono::Local::now().format("%H:%M:%S")));
        }
        self.poke();
    }

    pub fn log_lines(&self) -> Vec<String> {
        self.log.lock().map(|l| l.iter().cloned().collect()).unwrap_or_default()
    }

    pub fn set_engine(&self, s: EngineStatus) {
        if let Ok(mut e) = self.engine.lock() {
            *e = s;
        }
        self.poke();
    }

    pub fn engine_status(&self) -> EngineStatus {
        self.engine.lock().map(|e| e.clone()).unwrap_or_default()
    }

    fn slot(&self, p: PhaseId) -> &Slot {
        &self.slots[p.index()]
    }

    pub fn begin(&self, p: PhaseId, total: u64, what: &str) {
        let s = self.slot(p);
        s.done.store(0, Ordering::Relaxed);
        s.total.store(total, Ordering::Relaxed);
        s.took_ms.store(0, Ordering::Relaxed);
        if let Ok(mut w) = s.what.lock() {
            *w = what.to_string();
        }
        if let Ok(mut t) = s.started.lock() {
            *t = Some(Instant::now());
        }
        s.state.store(1, Ordering::SeqCst);
        self.poke();
    }

    pub fn update(&self, p: PhaseId, done: u64, total: u64, what: &str) {
        let s = self.slot(p);
        if s.state.load(Ordering::Relaxed) == 0 {
            self.begin(p, total, what);
        }
        s.done.store(done, Ordering::Relaxed);
        s.total.store(total, Ordering::Relaxed);
        if !what.is_empty() {
            if let Ok(mut w) = s.what.lock() {
                if *w != what {
                    *w = what.to_string();
                }
            }
        }
        self.poke();
    }

    fn end(&self, p: PhaseId, state: u8, what: Option<&str>) {
        let s = self.slot(p);
        let took = s.started.lock().ok().and_then(|t| *t).map(|t| t.elapsed().as_millis() as u64).unwrap_or(0);
        s.took_ms.store(took, Ordering::Relaxed);
        if let Some(w) = what {
            if let Ok(mut g) = s.what.lock() {
                *g = w.to_string();
            }
        }
        s.state.store(state, Ordering::SeqCst);
        self.poke();
    }

    pub fn finish(&self, p: PhaseId) {
        let s = self.slot(p);
        let total = s.total.load(Ordering::Relaxed);
        if total > 0 {
            s.done.store(total, Ordering::Relaxed);
        }
        self.end(p, 2, None);
    }

    pub fn skip(&self, p: PhaseId, why: &str) {
        self.end(p, 3, Some(why));
    }

    pub fn fail(&self, p: PhaseId, why: &str) {
        self.end(p, 4, Some(why));
    }

    pub fn failed_phase(&self) -> Option<PhaseId> {
        PHASES.iter().copied().find(|p| self.view(*p).state == PhaseState::Failed)
    }

    pub fn running_phase(&self) -> Option<PhaseId> {
        PHASES.iter().copied().find(|p| self.view(*p).state == PhaseState::Running)
    }

    pub fn view(&self, p: PhaseId) -> PhaseView {
        let s = self.slot(p);
        let state = PhaseState::from_u8(s.state.load(Ordering::SeqCst));
        let done = s.done.load(Ordering::Relaxed);
        let total = s.total.load(Ordering::Relaxed);
        let elapsed_ms = match state {
            PhaseState::Running => s.started.lock().ok().and_then(|t| *t).map(|t| t.elapsed().as_millis() as u64).unwrap_or(0),
            _ => s.took_ms.load(Ordering::Relaxed),
        };
        let eta_ms = (state == PhaseState::Running && done > 0 && total > done && elapsed_ms > 500).then(|| (elapsed_ms as f64 * (total - done) as f64 / done as f64) as u64);
        PhaseView { phase: p, state, done, total, what: s.what.lock().map(|w| w.clone()).unwrap_or_default(), elapsed_ms, eta_ms }
    }

    pub fn views(&self) -> Vec<PhaseView> {
        PHASES.iter().map(|p| self.view(*p)).collect()
    }

    /// Progress sink for the snapshot and surql helpers, writing into `phase`.
    pub fn progress_fn(self: &Arc<Self>, phase: PhaseId) -> ProgressFn {
        let job = Arc::clone(self);
        Arc::new(move |cur, total, what| job.update(phase, cur, total, what))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phases_track_state_and_eta() {
        let job = Job::new();
        assert_eq!(job.view(PhaseId::Copy).state, PhaseState::Pending);
        job.begin(PhaseId::Copy, 100, "copy");
        job.update(PhaseId::Copy, 25, 100, "copy");
        let v = job.view(PhaseId::Copy);
        assert_eq!((v.state, v.done, v.total), (PhaseState::Running, 25, 100));
        assert_eq!(v.fraction(), Some(0.25));
        job.finish(PhaseId::Copy);
        let v = job.view(PhaseId::Copy);
        assert_eq!((v.state, v.done), (PhaseState::Done, 100));
        job.skip(PhaseId::Thumbs, "not requested");
        assert_eq!(job.view(PhaseId::Thumbs).state, PhaseState::Skipped);
        job.fail(PhaseId::Import, "boom");
        assert_eq!(job.failed_phase(), Some(PhaseId::Import));
    }

    #[test]
    fn cancel_surfaces_as_a_typed_error() {
        let job = Job::new();
        assert!(job.check_cancel().is_ok());
        job.cancel();
        let err = job.check_cancel().unwrap_err();
        assert!(is_cancelled(&err));
        assert!(!is_cancelled(&anyhow::anyhow!("other")));
    }
}
