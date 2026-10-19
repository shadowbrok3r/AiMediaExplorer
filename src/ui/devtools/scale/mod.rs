//! Devtools window for performance at scale: live counters and scripted runs.

pub mod alloc;
pub mod bench;
pub mod frames;
pub mod marks;
pub mod proc;
pub mod resolve;
mod window;

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use eframe::egui::{Context, Id};

use crate::ui::file_table::rows::{RefreshCounts, ViewStats};

pub use window::show;

/// Numbers of the tab that drew last.
#[derive(Clone, Debug, Default)]
pub struct Facts {
    pub label: String,
    pub scope: String,
    pub library: bool,
    pub details: bool,
    pub store_len: usize,
    pub view_len: usize,
    pub selected: usize,
    pub stats: ViewStats,
    pub store_revision: u64,
    pub view_revision: u64,
    pub refreshes: RefreshCounts,
    pub loading: bool,
    /// Bytes the rows, their keys and the key index hold.
    pub row_bytes: usize,
}

fn open_id() -> Id {
    Id::new("ame_devtools_scale_open")
}

fn facts_id() -> Id {
    Id::new("ame_devtools_scale_facts")
}

pub fn is_open(ctx: &Context) -> bool {
    ctx.data(|d| d.get_temp::<bool>(open_id())).unwrap_or(false)
}

pub fn set_open(ctx: &Context, open: bool) {
    ctx.data_mut(|d| d.insert_temp(open_id(), open));
}

pub fn toggle(ctx: &Context) {
    set_open(ctx, !is_open(ctx));
}

/// Makes `facts` the numbers the window shows.
pub fn publish(ctx: &Context, facts: Facts) {
    ctx.data_mut(|d| d.insert_temp(facts_id(), facts));
}

pub fn latest(ctx: &Context) -> Option<Facts> {
    ctx.data(|d| d.get_temp::<Facts>(facts_id()))
}

static TIMINGS: Mutex<Vec<(String, f32)>> = Mutex::new(Vec::new());
static CAUSES: Mutex<BTreeMap<String, u64>> = Mutex::new(BTreeMap::new());
static PROGRESS: Mutex<Option<bench::Progress>> = Mutex::new(None);
static LAST: Mutex<Option<Arc<bench::Report>>> = Mutex::new(None);
static CANCEL: AtomicBool = AtomicBool::new(false);

fn lock<T>(m: &'static Mutex<T>) -> std::sync::MutexGuard<'static, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Records how long `name` took, for the report of the next run.
pub fn note_timing(name: &str, took: Duration) {
    lock(&TIMINGS).push((name.to_string(), took.as_secs_f32() * 1000.0));
}

/// Every timing noted so far in milliseconds.
pub fn timings() -> Vec<(String, f32)> {
    lock(&TIMINGS).clone()
}

/// Where the running run is, `None` when none runs.
pub fn progress() -> Option<bench::Progress> {
    lock(&PROGRESS).clone()
}

pub fn set_progress(progress: Option<bench::Progress>) {
    *lock(&PROGRESS) = progress;
}

/// The report of the run that ended last.
pub fn last_report() -> Option<Arc<bench::Report>> {
    lock(&LAST).clone()
}

pub fn set_report(report: bench::Report) -> Arc<bench::Report> {
    let report = Arc::new(report);
    *lock(&LAST) = Some(report.clone());
    report
}

/// Serializes the tests that use the global cancel flag.
#[cfg(test)]
pub(crate) static CANCEL_TESTS: Mutex<()> = Mutex::new(());

/// Asks the running run to stop at its next frame.
pub fn request_cancel() {
    CANCEL.store(true, Ordering::Relaxed);
}

/// True once after [`request_cancel`].
pub fn take_cancel() -> bool {
    CANCEL.swap(false, Ordering::Relaxed)
}

/// Frames drawn per place that asked for a repaint.
pub fn repaint_causes() -> BTreeMap<String, u64> {
    lock(&CAUSES).clone()
}

// Allocations the interface thread made since the previous frame ended.
fn ui_allocations() -> u32 {
    static LAST: AtomicU64 = AtomicU64::new(0);
    static MARKED: AtomicBool = AtomicBool::new(false);
    if !MARKED.swap(true, Ordering::Relaxed) {
        alloc::mark_ui_thread();
    }
    let now = alloc::ui_thread().allocs;
    now.saturating_sub(LAST.swap(now, Ordering::Relaxed)).min(u32::MAX as u64) as u32
}

fn note_causes(ctx: &Context) {
    let causes = ctx.repaint_causes();
    if causes.is_empty() {
        return;
    }
    let mut map = lock(&CAUSES);
    for cause in causes {
        *map.entry(cause.to_string()).or_default() += 1;
    }
}

/// Records the frame that just ended; called once per frame by the application.
pub fn frame_end(ctx: &Context, ui_time: Duration, cpu_secs: Option<f32>) {
    if is_open(ctx) || progress().is_some() {
        note_causes(ctx);
    }
    frames::push(frames::Frame {
        dt_ms: ctx.input(|i| i.unstable_dt) * 1000.0,
        ui_ms: ui_time.as_secs_f32() * 1000.0,
        cpu_ms: cpu_secs.unwrap_or(0.0) * 1000.0,
        at_s: frames::clock_s(),
        allocs: ui_allocations(),
    });
    marks::observe_runtime();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_window_opens_and_toggles() {
        let ctx = crate::ui::test_support::test_ctx();
        assert!(!is_open(&ctx));
        set_open(&ctx, true);
        assert!(is_open(&ctx));
        toggle(&ctx);
        assert!(!is_open(&ctx));
    }

    #[test]
    fn published_facts_are_the_latest_ones() {
        let ctx = crate::ui::test_support::test_ctx();
        assert!(latest(&ctx).is_none());
        publish(&ctx, Facts { label: "a".into(), store_len: 3, ..Default::default() });
        publish(&ctx, Facts { label: "b".into(), store_len: 5, ..Default::default() });
        let facts = latest(&ctx).unwrap();
        assert_eq!((facts.label.as_str(), facts.store_len), ("b", 5));
    }

    #[test]
    fn a_cancel_request_is_taken_once() {
        let _guard = CANCEL_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        take_cancel();
        assert!(!take_cancel());
        request_cancel();
        assert!(take_cancel());
        assert!(!take_cancel());
    }

    #[test]
    fn a_frame_end_adds_one_frame_with_milliseconds() {
        let ctx = crate::ui::test_support::test_ctx();
        let before = frames::mark();
        frame_end(&ctx, Duration::from_millis(7), Some(0.004));
        let added = frames::since(before);
        assert!(!added.is_empty());
        let last = added.last().unwrap();
        assert!((last.ui_ms - 7.0).abs() < 1e-3 && (last.cpu_ms - 4.0).abs() < 1e-3, "{last:?}");
    }
}
