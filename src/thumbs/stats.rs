//! Rolling timing windows for the thumbnail pipeline stages and the frame-time series.

use std::time::Duration;

/// Fixed-size window of the most recent samples in milliseconds.
pub struct Ring {
    buf: Vec<f32>,
    cap: usize,
    next: usize,
    total: u64,
    max_ever: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StageStat {
    pub samples: u64,
    pub last: f32,
    pub avg: f32,
    pub p95: f32,
    pub max_window: f32,
    pub max_ever: f32,
}

impl Ring {
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        Self { buf: Vec::with_capacity(cap), cap, next: 0, total: 0, max_ever: 0.0 }
    }

    pub fn push(&mut self, ms: f32) {
        if self.buf.len() < self.cap {
            self.buf.push(ms);
        } else {
            self.buf[self.next] = ms;
        }
        self.next = (self.next + 1) % self.cap;
        self.total += 1;
        self.max_ever = self.max_ever.max(ms);
    }

    pub fn push_duration(&mut self, d: Duration) {
        self.push(d.as_secs_f32() * 1000.0);
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Samples from oldest to newest.
    pub fn series(&self) -> Vec<f32> {
        if self.buf.len() < self.cap {
            return self.buf.clone();
        }
        let mut out = Vec::with_capacity(self.buf.len());
        out.extend_from_slice(&self.buf[self.next..]);
        out.extend_from_slice(&self.buf[..self.next]);
        out
    }

    pub fn stat(&self) -> StageStat {
        if self.buf.is_empty() {
            return StageStat::default();
        }
        let mut sorted = self.buf.clone();
        sorted.sort_by(f32::total_cmp);
        let last = if self.buf.len() < self.cap { self.buf[self.buf.len() - 1] } else { self.buf[(self.next + self.cap - 1) % self.cap] };
        let p95_index = ((sorted.len() as f32 * 0.95).ceil() as usize).saturating_sub(1).min(sorted.len() - 1);
        StageStat {
            samples: self.total,
            last,
            avg: sorted.iter().sum::<f32>() / sorted.len() as f32,
            p95: sorted[p95_index],
            max_window: sorted[sorted.len() - 1],
            max_ever: self.max_ever,
        }
    }

    pub fn percentile(&self, p: f32) -> f32 {
        if self.buf.is_empty() {
            return 0.0;
        }
        let mut sorted = self.buf.clone();
        sorted.sort_by(f32::total_cmp);
        let i = ((sorted.len() - 1) as f32 * p.clamp(0.0, 1.0)).round() as usize;
        sorted[i]
    }
}

pub struct Timings {
    pub db_batch: Ring,
    pub decode: Ring,
    pub upload: Ring,
    pub store_put: Ring,
    pub frame: Ring,
    pub ui: Ring,
}

impl Default for Timings {
    fn default() -> Self {
        Self { db_batch: Ring::new(64), decode: Ring::new(128), upload: Ring::new(128), store_put: Ring::new(64), frame: Ring::new(240), ui: Ring::new(240) }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TimingsSnapshot {
    pub db_batch: StageStat,
    pub decode: StageStat,
    pub upload: StageStat,
    pub store_put: StageStat,
    pub frame: StageStat,
    pub ui: StageStat,
    pub ui_series: Vec<f32>,
    pub frame_series: Vec<f32>,
    pub frame_p50: f32,
}

impl Timings {
    pub fn snapshot(&self) -> TimingsSnapshot {
        TimingsSnapshot {
            db_batch: self.db_batch.stat(),
            decode: self.decode.stat(),
            upload: self.upload.stat(),
            store_put: self.store_put.stat(),
            frame: self.frame.stat(),
            ui: self.ui.stat(),
            ui_series: self.ui.series(),
            frame_series: self.frame.series(),
            frame_p50: self.frame.percentile(0.5),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_keeps_the_newest_samples_in_order() {
        let mut r = Ring::new(4);
        for i in 1..=6 {
            r.push(i as f32);
        }
        assert_eq!(r.series(), vec![3.0, 4.0, 5.0, 6.0]);
        let s = r.stat();
        assert_eq!(s.samples, 6);
        assert_eq!(s.last, 6.0);
        assert_eq!(s.max_window, 6.0);
        assert!((s.avg - 4.5).abs() < 1e-6);
    }

    #[test]
    fn partial_ring_reports_what_it_has() {
        let mut r = Ring::new(10);
        assert_eq!(r.stat(), StageStat::default());
        r.push(2.0);
        r.push(8.0);
        assert_eq!(r.series(), vec![2.0, 8.0]);
        assert_eq!(r.stat().last, 8.0);
        assert_eq!(r.stat().p95, 8.0);
        assert_eq!(r.percentile(0.5), 8.0);
    }

    #[test]
    fn max_ever_survives_window_rollover() {
        let mut r = Ring::new(2);
        r.push(100.0);
        r.push(1.0);
        r.push(2.0);
        let s = r.stat();
        assert_eq!(s.max_window, 2.0);
        assert_eq!(s.max_ever, 100.0);
    }

    #[test]
    fn p95_picks_the_high_tail() {
        let mut r = Ring::new(100);
        for i in 1..=100 {
            r.push(i as f32);
        }
        assert_eq!(r.stat().p95, 95.0);
        assert_eq!(r.percentile(1.0), 100.0);
    }
}
