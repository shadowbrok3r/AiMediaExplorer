//! Bounded concurrency for video decoding and counters for the debug window.

use super::{Cancel, VideoError};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock};
use std::time::Duration;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Counting semaphore with cancellable waits.
pub struct Gate {
    state: Mutex<usize>,
    cv: Condvar,
    limit: AtomicUsize,
    waiting: AtomicUsize,
}

pub struct Permit<'a> {
    gate: &'a Gate,
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        let mut busy = lock(&self.gate.state);
        *busy = busy.saturating_sub(1);
        drop(busy);
        self.gate.cv.notify_one();
    }
}

impl Gate {
    pub fn new(limit: usize) -> Self {
        Self { state: Mutex::new(0), cv: Condvar::new(), limit: AtomicUsize::new(limit.max(1)), waiting: AtomicUsize::new(0) }
    }

    pub fn limit(&self) -> usize {
        self.limit.load(Ordering::Relaxed)
    }

    pub fn set_limit(&self, limit: usize) {
        self.limit.store(limit.max(1), Ordering::Relaxed);
        self.cv.notify_all();
    }

    pub fn in_flight(&self) -> usize {
        *lock(&self.state)
    }

    pub fn waiting(&self) -> usize {
        self.waiting.load(Ordering::Relaxed)
    }

    /// Waits for a free slot; fails once `cancel` is raised.
    pub fn acquire(&self, cancel: &Cancel) -> Result<Permit<'_>, VideoError> {
        self.waiting.fetch_add(1, Ordering::Relaxed);
        let result = self.wait_for_slot(cancel);
        self.waiting.fetch_sub(1, Ordering::Relaxed);
        result
    }

    fn wait_for_slot(&self, cancel: &Cancel) -> Result<Permit<'_>, VideoError> {
        let mut busy = lock(&self.state);
        loop {
            if cancel.is_cancelled() {
                return Err(VideoError::Cancelled);
            }
            if *busy < self.limit() {
                *busy += 1;
                return Ok(Permit { gate: self });
            }
            busy = self.cv.wait_timeout(busy, Duration::from_millis(50)).unwrap_or_else(|e| e.into_inner()).0;
        }
    }
}

/// Concurrent video decodes: `AME_VIDEO_JOBS` or a quarter of the cores, between 2 and 4.
fn default_limit() -> usize {
    std::env::var("AME_VIDEO_JOBS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or_else(|| (std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4) / 4).clamp(2, 4))
        .max(1)
}

static GATE: OnceLock<Gate> = OnceLock::new();

pub fn global() -> &'static Gate {
    GATE.get_or_init(|| Gate::new(default_limit()))
}

/// Running totals of poster work.
#[derive(Default)]
pub struct Counters {
    pub ok: AtomicU64,
    pub placeholders: AtomicU64,
    pub failed: AtomicU64,
    pub timeouts: AtomicU64,
    pub blank_retries: AtomicU64,
    pub total_micros: AtomicU64,
    pub last_micros: AtomicU64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CountersSnapshot {
    pub ok: u64,
    pub placeholders: u64,
    pub failed: u64,
    pub timeouts: u64,
    pub blank_retries: u64,
    pub total_micros: u64,
    pub last_micros: u64,
}

impl Counters {
    pub const fn new() -> Self {
        Self {
            ok: AtomicU64::new(0),
            placeholders: AtomicU64::new(0),
            failed: AtomicU64::new(0),
            timeouts: AtomicU64::new(0),
            blank_retries: AtomicU64::new(0),
            total_micros: AtomicU64::new(0),
            last_micros: AtomicU64::new(0),
        }
    }

    pub fn snapshot(&self) -> CountersSnapshot {
        let get = |a: &AtomicU64| a.load(Ordering::Relaxed);
        CountersSnapshot {
            ok: get(&self.ok),
            placeholders: get(&self.placeholders),
            failed: get(&self.failed),
            timeouts: get(&self.timeouts),
            blank_retries: get(&self.blank_retries),
            total_micros: get(&self.total_micros),
            last_micros: get(&self.last_micros),
        }
    }

    pub fn record_time(&self, elapsed: Duration) {
        let micros = elapsed.as_micros().min(u128::from(u64::MAX)) as u64;
        self.total_micros.fetch_add(micros, Ordering::Relaxed);
        self.last_micros.store(micros, Ordering::Relaxed);
    }
}

static COUNTERS: Counters = Counters::new();

pub fn counters() -> &'static Counters {
    &COUNTERS
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Instant;

    #[test]
    fn never_more_than_the_limit_run_at_once() {
        let gate = Arc::new(Gate::new(2));
        let (peak, current) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let (gate, peak, current) = (gate.clone(), peak.clone(), current.clone());
                std::thread::spawn(move || {
                    let _permit = gate.acquire(&Cancel::new()).unwrap();
                    let now = current.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(now, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(15));
                    current.fetch_sub(1, Ordering::SeqCst);
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(peak.load(Ordering::SeqCst), 2);
        assert_eq!(gate.in_flight(), 0);
        assert_eq!(gate.waiting(), 0);
    }

    #[test]
    fn a_waiter_gives_up_when_cancelled() {
        let gate = Arc::new(Gate::new(1));
        let _held = gate.acquire(&Cancel::new()).unwrap();
        let cancel = Cancel::new();
        let waiter = {
            let (gate, cancel) = (gate.clone(), cancel.clone());
            std::thread::spawn(move || gate.acquire(&cancel).err())
        };
        std::thread::sleep(Duration::from_millis(80));
        assert_eq!(gate.waiting(), 1);
        let started = Instant::now();
        cancel.cancel();
        assert_eq!(waiter.join().unwrap(), Some(VideoError::Cancelled));
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn dropping_a_permit_frees_the_slot_and_the_limit_never_drops_below_one() {
        let gate = Gate::new(1);
        let permit = gate.acquire(&Cancel::new()).unwrap();
        assert_eq!(gate.in_flight(), 1);
        drop(permit);
        assert_eq!(gate.in_flight(), 0);
        gate.set_limit(0);
        assert_eq!(gate.limit(), 1);
        gate.set_limit(3);
        assert_eq!(gate.limit(), 3);
    }

    #[test]
    fn counters_accumulate_time() {
        let c = Counters::new();
        c.record_time(Duration::from_micros(1500));
        c.record_time(Duration::from_micros(500));
        let s = c.snapshot();
        assert_eq!((s.total_micros, s.last_micros), (2000, 500));
    }
}
