//! The description queue of the AI engine: one task while files wait to be described.

use super::{Burst, Report, SourceCtx, TaskSource};
use crate::tasks::model::{CancelHandle, Observation, TaskKind, TaskSpec};

pub const SOURCE: &str = "describe";
const KEY: &str = "describe";

/// The queue counters of the engine.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DescribeCounters {
    /// Files queued, including the ones being described.
    pub queue_len: u64,
    /// Files being described right now.
    pub active: u64,
    /// Files described since the process started.
    pub completed: u64,
}

type Reader = Box<dyn FnMut() -> DescribeCounters + Send>;

pub struct DescribeSource {
    read: Reader,
    cancel: CancelHandle,
    burst: Burst,
}

impl DescribeSource {
    pub fn new(read: Reader, cancel: CancelHandle) -> Self {
        Self { read, cancel, burst: Burst::default() }
    }

    /// Reads the counters of the global AI engine and stops the bulk run through `cancel_bulk_descriptions`.
    pub fn live() -> Self {
        use std::sync::atomic::Ordering::Relaxed;
        Self::new(
            Box::new(|| {
                let e = &crate::ai::GLOBAL_AI_ENGINE;
                DescribeCounters {
                    queue_len: e.index_queue_len.load(Relaxed) as u64,
                    active: e.index_active.load(Relaxed) as u64,
                    completed: e.index_completed.load(Relaxed) as u64,
                }
            }),
            CancelHandle::new(|| crate::ai::GLOBAL_AI_ENGINE.cancel_bulk_descriptions()),
        )
    }
}

impl TaskSource for DescribeSource {
    fn name(&self) -> &'static str {
        SOURCE
    }

    fn poll(&mut self, cx: &SourceCtx<'_>, out: &mut Report) {
        let c = (self.read)();
        let remaining = c.queue_len.max(c.active);
        if let Some((done, total)) = self.burst.observe(cx.now, c.completed, remaining) {
            let detail = format!("{} active, {} queued", c.active, c.queue_len.saturating_sub(c.active));
            let spec = TaskSpec::new(TaskKind::Describe, "Describe files").total(total).detail(detail).cancel(self.cancel.clone());
            out.observations.push(Observation::new(KEY, spec).done(done));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::sources::AppProbe;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    fn source(c: Arc<Mutex<DescribeCounters>>, flag: Arc<AtomicBool>) -> DescribeSource {
        DescribeSource::new(Box::new(move || *c.lock().unwrap()), CancelHandle::from_flag(flag))
    }

    fn poll(s: &mut DescribeSource, at: Instant) -> Vec<Observation> {
        let app = AppProbe::default();
        let mut out = Report::default();
        s.poll(&SourceCtx { now: at, app: &app }, &mut out);
        out.observations
    }

    #[test]
    fn an_empty_queue_makes_no_task() {
        let mut s = source(Arc::default(), Arc::default());
        assert!(poll(&mut s, Instant::now()).is_empty());
    }

    #[test]
    fn a_queue_becomes_a_task_counting_completed_over_the_burst_total() {
        let c = Arc::new(Mutex::new(DescribeCounters { queue_len: 340, active: 1, completed: 500 }));
        let mut s = source(c.clone(), Arc::default());
        let t0 = Instant::now();
        let first = poll(&mut s, t0);
        assert_eq!((first[0].done, first[0].spec.total), (0, 340));
        *c.lock().unwrap() = DescribeCounters { queue_len: 328, active: 1, completed: 512 };
        let later = poll(&mut s, t0);
        assert_eq!((later[0].done, later[0].spec.total), (12, 340));
        assert_eq!(later[0].spec.detail, "1 active, 327 queued");
        assert_eq!(later[0].spec.kind, TaskKind::Describe);
    }

    #[test]
    fn the_task_offers_cancel_through_the_engine_flag() {
        let flag = Arc::new(AtomicBool::new(false));
        let c = Arc::new(Mutex::new(DescribeCounters { queue_len: 3, active: 1, completed: 0 }));
        let mut s = source(c, flag.clone());
        let obs = poll(&mut s, Instant::now());
        obs[0].spec.cancel.as_ref().unwrap().call();
        assert!(flag.load(Ordering::Relaxed));
    }

    #[test]
    fn the_task_ends_when_the_queue_has_been_empty_for_a_moment() {
        let c = Arc::new(Mutex::new(DescribeCounters { queue_len: 2, active: 1, completed: 0 }));
        let mut s = source(c.clone(), Arc::default());
        let t0 = Instant::now();
        poll(&mut s, t0);
        *c.lock().unwrap() = DescribeCounters { queue_len: 0, active: 0, completed: 2 };
        assert_eq!(poll(&mut s, t0 + Duration::from_millis(100)).len(), 1);
        assert!(poll(&mut s, t0 + Duration::from_secs(2)).is_empty());
    }
}
