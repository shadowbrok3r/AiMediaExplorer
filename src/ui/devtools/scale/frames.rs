//! Per-frame timings of the whole application and the summary statistics of a series.

use std::collections::VecDeque;
use std::sync::{LazyLock, Mutex};
use std::time::Instant;

use serde::Serialize;

/// Frames kept in the global log.
pub const LOG_CAPACITY: usize = 16_384;

/// Timings of one drawn frame in milliseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Frame {
    /// Time since the previous frame started.
    pub dt_ms: f32,
    /// Time the application spent building its interface.
    pub ui_ms: f32,
    /// CPU time of the whole frame as eframe measures it; the frame before for the one being drawn.
    pub cpu_ms: f32,
    /// When the frame ended, in seconds on the clock of [`clock_s`].
    pub at_s: f32,
    /// Allocations the interface thread made during the frame; zero unless the build counts them.
    pub allocs: u32,
}

static EPOCH: LazyLock<Instant> = LazyLock::new(Instant::now);

/// Seconds since the first call; the clock frames are stamped with.
pub fn clock_s() -> f32 {
    EPOCH.elapsed().as_secs_f32()
}

/// A frame that took long, placed in its run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct SlowFrame {
    /// Seconds after the first frame of the run.
    pub at_s: f32,
    pub dt_ms: f32,
    pub ui_ms: f32,
    pub cpu_ms: f32,
}

/// The `n` frames with the longest time since the previous frame, slowest first.
pub fn slowest(frames: &[Frame], n: usize) -> Vec<SlowFrame> {
    let Some(first) = frames.first() else { return Vec::new() };
    let mut order: Vec<&Frame> = frames.iter().collect();
    order.sort_by(|a, b| b.dt_ms.total_cmp(&a.dt_ms));
    order.into_iter().take(n).map(|f| SlowFrame { at_s: (f.at_s - first.at_s).max(0.0), dt_ms: f.dt_ms, ui_ms: f.ui_ms, cpu_ms: f.cpu_ms }).collect()
}

/// Smallest, mean, median, 95th percentile and largest value of a series.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct Spread {
    pub min: f32,
    pub avg: f32,
    pub p50: f32,
    pub p95: f32,
    pub max: f32,
}

impl Spread {
    /// The spread of `values`; `None` when there are none.
    pub fn of(values: &[f32]) -> Option<Spread> {
        if values.is_empty() {
            return None;
        }
        let mut sorted: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
        if sorted.is_empty() {
            return None;
        }
        sorted.sort_by(f32::total_cmp);
        let rank = |p: f32| sorted[(((sorted.len() as f32 * p).ceil() as usize).clamp(1, sorted.len())) - 1];
        Some(Spread { min: sorted[0], avg: sorted.iter().sum::<f32>() / sorted.len() as f32, p50: rank(0.5), p95: rank(0.95), max: sorted[sorted.len() - 1] })
    }
}

/// The three series of a run of frames.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct FrameSpread {
    pub dt: Spread,
    pub ui: Spread,
    pub cpu: Spread,
    /// Allocations of the interface thread per frame.
    pub allocs: Spread,
}

impl FrameSpread {
    pub fn of(frames: &[Frame]) -> Option<FrameSpread> {
        let dt: Vec<f32> = frames.iter().map(|f| f.dt_ms).collect();
        let ui: Vec<f32> = frames.iter().map(|f| f.ui_ms).collect();
        let cpu: Vec<f32> = frames.iter().map(|f| f.cpu_ms).collect();
        let allocs: Vec<f32> = frames.iter().map(|f| f.allocs as f32).collect();
        Some(FrameSpread { dt: Spread::of(&dt)?, ui: Spread::of(&ui)?, cpu: Spread::of(&cpu).unwrap_or_default(), allocs: Spread::of(&allocs).unwrap_or_default() })
    }
}

/// The newest frames, each with a running number.
#[derive(Default)]
pub struct FrameLog {
    next: u64,
    frames: VecDeque<Frame>,
}

impl FrameLog {
    pub fn push(&mut self, frame: Frame) {
        if self.frames.len() == LOG_CAPACITY {
            self.frames.pop_front();
        }
        self.frames.push_back(frame);
        self.next += 1;
    }

    /// Number the next pushed frame gets; frames before it were seen already.
    pub fn mark(&self) -> u64 {
        self.next
    }

    /// Frames pushed since `mark`, oldest first, as far as the log still holds them.
    pub fn since(&self, mark: u64) -> Vec<Frame> {
        let first = self.next - self.frames.len() as u64;
        let from = mark.max(first) - first;
        self.frames.iter().skip(from as usize).copied().collect()
    }

    /// The newest `n` frames.
    pub fn latest(&self, n: usize) -> Vec<Frame> {
        self.frames.iter().skip(self.frames.len().saturating_sub(n)).copied().collect()
    }
}

static LOG: Mutex<FrameLog> = Mutex::new(FrameLog { next: 0, frames: VecDeque::new() });

fn log() -> std::sync::MutexGuard<'static, FrameLog> {
    LOG.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn push(frame: Frame) {
    log().push(frame);
}

pub fn mark() -> u64 {
    log().mark()
}

pub fn since(mark: u64) -> Vec<Frame> {
    log().since(mark)
}

pub fn latest(n: usize) -> Vec<Frame> {
    log().latest(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(ms: f32) -> Frame {
        Frame { dt_ms: ms, ui_ms: ms / 2.0, cpu_ms: ms / 4.0, at_s: ms, allocs: ms as u32 * 10 }
    }

    #[test]
    fn the_slowest_frames_come_first_and_are_placed_after_the_first_frame() {
        let frames = [frame(10.0), frame(500.0), frame(20.0), frame(300.0)];
        let slow = slowest(&frames, 2);
        assert_eq!(slow.iter().map(|s| s.dt_ms).collect::<Vec<_>>(), vec![500.0, 300.0]);
        assert_eq!(slow[0].at_s, 490.0);
        assert!(slowest(&[], 3).is_empty());
        assert_eq!(slowest(&frames, 10).len(), 4);
    }

    #[test]
    fn a_spread_orders_the_values_and_takes_nearest_rank_percentiles() {
        let values: Vec<f32> = (1..=100).rev().map(|v| v as f32).collect();
        let s = Spread::of(&values).unwrap();
        assert_eq!((s.min, s.p50, s.p95, s.max), (1.0, 50.0, 95.0, 100.0));
        assert!((s.avg - 50.5).abs() < 1e-3);
        assert_eq!(Spread::of(&[7.0]).unwrap(), Spread { min: 7.0, avg: 7.0, p50: 7.0, p95: 7.0, max: 7.0 });
        assert_eq!(Spread::of(&[]), None);
        assert_eq!(Spread::of(&[f32::NAN]), None);
    }

    #[test]
    fn the_frame_spread_covers_each_series() {
        let s = FrameSpread::of(&[frame(8.0), frame(16.0), frame(40.0)]).unwrap();
        assert_eq!(s.dt.max, 40.0);
        assert_eq!(s.ui.max, 20.0);
        assert_eq!(s.cpu.min, 2.0);
        assert_eq!(s.allocs.max, 400.0);
        assert_eq!(FrameSpread::of(&[]), None);
    }

    #[test]
    fn the_log_hands_out_only_frames_after_a_mark() {
        let mut log = FrameLog::default();
        log.push(frame(1.0));
        let mark = log.mark();
        assert!(log.since(mark).is_empty());
        log.push(frame(2.0));
        log.push(frame(3.0));
        assert_eq!(log.since(mark), vec![frame(2.0), frame(3.0)]);
        assert_eq!(log.since(0).len(), 3);
        assert_eq!(log.latest(2), vec![frame(2.0), frame(3.0)]);
        assert_eq!(log.latest(10).len(), 3);
    }

    #[test]
    fn the_log_drops_its_oldest_frames_at_capacity() {
        let mut log = FrameLog::default();
        for i in 0..LOG_CAPACITY + 10 {
            log.push(frame(i as f32));
        }
        assert_eq!(log.latest(usize::MAX).len(), LOG_CAPACITY);
        let all = log.since(0);
        assert_eq!(all.len(), LOG_CAPACITY);
        assert_eq!(all[0].dt_ms, 10.0);
        assert_eq!(log.since(log.mark() - 3).len(), 3);
    }
}
