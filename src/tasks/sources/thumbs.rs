//! The thumbnail pipeline: one task while the ThumbService has work in flight.

use std::time::Duration;

use super::{Burst, Report, SourceCtx, TaskSource, Throttle};
use crate::tasks::model::{Observation, TaskKind, TaskSpec};

pub const SOURCE: &str = "thumbnails";
const KEY: &str = "thumbnails";
const READ_EVERY: Duration = Duration::from_millis(250);

/// The pipeline counters the task is built from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ThumbCounters {
    /// Requests that are queued, being fetched, decoded or generated.
    pub in_flight: u64,
    /// Requests that ended, successfully or not, since the process started.
    pub finished: u64,
    /// Encoded thumbnails waiting to be written to the store.
    pub pending_writes: u64,
}

type Reader = Box<dyn FnMut() -> Option<ThumbCounters> + Send>;

pub struct ThumbSource {
    read: Reader,
    throttle: Throttle,
    burst: Burst,
    last: Option<(u64, u64, u64)>,
}

impl ThumbSource {
    pub fn new(read: Reader) -> Self {
        Self { read, throttle: Throttle::new(READ_EVERY), burst: Burst::default(), last: None }
    }

    /// Reads the global ThumbService.
    pub fn live() -> Self {
        Self::new(Box::new(|| {
            let s = crate::ui::file_table::thumbs::service().snapshot();
            let d = s.depths;
            Some(ThumbCounters {
                in_flight: (d.queued + d.fetching + d.decode_ready + d.decoding + d.gen_wait + d.generating + d.legacy_wait) as u64,
                finished: s.core.decode_ok + s.core.decode_failed + s.core.generate_failed + s.counters.produced,
                pending_writes: s.pending_writes as u64,
            })
        }))
    }
}

impl TaskSource for ThumbSource {
    fn name(&self) -> &'static str {
        SOURCE
    }

    fn poll(&mut self, cx: &SourceCtx<'_>, out: &mut Report) {
        if self.throttle.ready(cx.now) {
            let counters = (self.read)().unwrap_or_default();
            let remaining = counters.in_flight + counters.pending_writes;
            self.last = self.burst.observe(cx.now, counters.finished, remaining).map(|(done, total)| (done, total, counters.in_flight));
        }
        if let Some((done, total, in_flight)) = self.last {
            let detail = if in_flight > 0 { format!("{in_flight} in the pipeline") } else { "Saving thumbnails".to_string() };
            let spec = TaskSpec::new(TaskKind::Thumbnails, "Thumbnails").total(total).detail(detail);
            out.observations.push(Observation::new(KEY, spec).done(done));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::sources::AppProbe;
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    fn source(counters: Arc<Mutex<Option<ThumbCounters>>>, reads: Arc<Mutex<u32>>) -> ThumbSource {
        ThumbSource::new(Box::new(move || {
            *reads.lock().unwrap() += 1;
            *counters.lock().unwrap()
        }))
    }

    fn poll(s: &mut ThumbSource, at: Instant) -> Vec<Observation> {
        let app = AppProbe::default();
        let mut out = Report::default();
        s.poll(&SourceCtx { now: at, app: &app }, &mut out);
        out.observations
    }

    fn set(c: &Arc<Mutex<Option<ThumbCounters>>>, in_flight: u64, finished: u64, pending_writes: u64) {
        *c.lock().unwrap() = Some(ThumbCounters { in_flight, finished, pending_writes });
    }

    #[test]
    fn an_idle_pipeline_makes_no_task() {
        let c = Arc::new(Mutex::new(Some(ThumbCounters::default())));
        let mut s = source(c, Arc::default());
        assert!(poll(&mut s, Instant::now()).is_empty());
    }

    #[test]
    fn work_in_flight_is_one_task_with_progress_since_it_began() {
        let c = Arc::new(Mutex::new(None));
        let mut s = source(c.clone(), Arc::default());
        let t0 = Instant::now();
        set(&c, 40, 1000, 0);
        let first = poll(&mut s, t0);
        assert_eq!((first[0].done, first[0].spec.total, first[0].spec.kind), (0, 40, TaskKind::Thumbnails));
        set(&c, 10, 1030, 0);
        let second = poll(&mut s, t0 + Duration::from_millis(300));
        assert_eq!((second[0].done, second[0].spec.total), (30, 40));
        assert_eq!(second[0].spec.detail, "10 in the pipeline");
        assert!(!second[0].spec.user_initiated, "background work never toasts");
    }

    #[test]
    fn pending_writes_keep_the_task_alive_after_the_pipeline_drains() {
        let c = Arc::new(Mutex::new(None));
        let mut s = source(c.clone(), Arc::default());
        let t0 = Instant::now();
        set(&c, 5, 0, 0);
        poll(&mut s, t0);
        set(&c, 0, 5, 3);
        let obs = poll(&mut s, t0 + Duration::from_millis(300));
        assert_eq!(obs[0].spec.detail, "Saving thumbnails");
    }

    #[test]
    fn the_task_ends_after_the_pipeline_stays_empty() {
        let c = Arc::new(Mutex::new(None));
        let mut s = source(c.clone(), Arc::default());
        let t0 = Instant::now();
        set(&c, 5, 0, 0);
        poll(&mut s, t0);
        set(&c, 0, 5, 0);
        assert_eq!(poll(&mut s, t0 + Duration::from_millis(300)).len(), 1, "a short gap keeps it");
        assert!(poll(&mut s, t0 + Duration::from_millis(1200)).is_empty());
    }

    #[test]
    fn the_engine_is_read_at_most_four_times_a_second() {
        let c = Arc::new(Mutex::new(Some(ThumbCounters::default())));
        let reads = Arc::new(Mutex::new(0));
        let mut s = source(c, reads.clone());
        let t0 = Instant::now();
        for frame in 0..100 {
            poll(&mut s, t0 + Duration::from_millis(frame * 10));
        }
        assert!(*reads.lock().unwrap() <= 5, "read {} times in one second", reads.lock().unwrap());
    }

    #[test]
    fn frames_between_reads_repeat_the_last_observation() {
        let c = Arc::new(Mutex::new(None));
        let mut s = source(c.clone(), Arc::default());
        let t0 = Instant::now();
        set(&c, 8, 0, 0);
        poll(&mut s, t0);
        assert_eq!(poll(&mut s, t0 + Duration::from_millis(16)).len(), 1);
    }
}
