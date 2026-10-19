use std::cell::{Cell, RefCell};

use super::*;

#[derive(Default)]
struct Fake {
    calls: RefCell<Vec<String>>,
    frames: RefCell<Vec<Frame>>,
    loading_polls: Cell<usize>,
    load_polls_per_open: usize,
    scrolls: RefCell<Vec<f32>>,
    resolutions: RefCell<VecDeque<Resolved>>,
    open_error: Option<String>,
    queries: Cell<u64>,
    cpu: Cell<f64>,
    causes: RefCell<BTreeMap<String, u64>>,
}

impl Fake {
    fn note(&self, call: impl Into<String>) {
        self.calls.borrow_mut().push(call.into());
    }

    fn draw(&self, dt_ms: f32) {
        let at_s = self.frames.borrow().len() as f32 * 0.016;
        self.frames.borrow_mut().push(Frame { dt_ms, ui_ms: dt_ms / 2.0, cpu_ms: dt_ms / 4.0, at_s, allocs: 100 });
    }
}

impl Target for Fake {
    fn view_rows(&self) -> usize {
        1_000
    }
    fn selected(&self) -> usize {
        7
    }
    fn is_loading(&self) -> bool {
        let left = self.loading_polls.get();
        if left > 0 {
            self.loading_polls.set(left - 1);
        }
        left > 0
    }
    fn open(&mut self, scope: &str) -> Result<(), String> {
        self.note(format!("open {scope}"));
        if let Some(e) = &self.open_error {
            return Err(e.clone());
        }
        self.loading_polls.set(self.load_polls_per_open);
        Ok(())
    }
    fn last_load(&self) -> Option<LoadSummary> {
        Some(LoadSummary { scope: "lib:all".into(), rows: 1_000, batches: 2, first_batch_ms: Some(12.0), elapsed_ms: 80.0, page_ms: 50.0, embed_ms: 20.0, total: Some(1_000), error: None, cancelled: false })
    }
    fn scroll_to(&mut self, fraction: f32) {
        self.scrolls.borrow_mut().push(fraction);
    }
    fn set_tile(&mut self, px: f32) {
        self.note(format!("tile {px}"));
    }
    fn set_mode(&mut self, mode: ViewMode) {
        self.note(format!("mode {}", mode.label()));
    }
    fn sort(&mut self, by: Option<(SortBy, bool)>) {
        self.note(format!("sort {by:?}"));
    }
    fn filter(&mut self, patch: &serde_json::Value) -> Result<(), String> {
        self.note(format!("filter {patch}"));
        Ok(())
    }
    fn clear_filters(&mut self) {
        self.note("clear_filters");
    }
    fn select_all(&mut self) {
        self.note("select_all");
    }
    fn clear_selection(&mut self) {
        self.note("clear_selection");
    }
    fn task_center(&mut self, open: bool) {
        self.note(format!("task_center {open}"));
    }
    fn resolve(&mut self, scope: &str) -> Resolved {
        self.note(format!("resolve {scope}"));
        self.resolutions.borrow_mut().pop_front().unwrap_or(Resolved::Pending)
    }
    fn meters(&self) -> Meters {
        let mut queries = Counts::default();
        queries.by_site[Site::ScopePage as usize] = self.queries.get();
        queries.by_site[Site::ThumbRead as usize] = self.queries.get() / 2;
        Meters { rss_split: Some((150 * 1024 * 1024, 50 * 1024 * 1024)), rss: Some(200 * 1024 * 1024), peak_rss: Some(300 * 1024 * 1024), cpu_secs: Some(self.cpu.get()), queries, causes: self.causes.borrow().clone(), thumbs: None }
    }
    fn frame_mark(&self) -> u64 {
        self.frames.borrow().len() as u64
    }
    fn frames_since(&self, mark: u64) -> Vec<Frame> {
        self.frames.borrow().iter().skip(mark as usize).copied().collect()
    }
}

/// Ticks a run once per simulated frame of `dt_ms` until it ends or `limit_ms` passes.
fn run(bench: &mut Bench, target: &mut Fake, dt_ms: u64, limit_ms: u64) -> (Option<Report>, Instant) {
    let start = Instant::now();
    let mut now = start;
    while now.duration_since(start) < Duration::from_millis(limit_ms) {
        target.draw(dt_ms as f32);
        if let Tick::Done(report) = bench.tick(target, now) {
            return (Some(*report), now);
        }
        now += Duration::from_millis(dt_ms);
    }
    (None, now)
}

fn bench(steps: Vec<Step>) -> Bench {
    Bench::new(Suite::Ui, steps, Instant::now())
}

#[test]
fn every_suite_has_a_plan_and_all_chains_them() {
    assert_eq!(plan(Suite::Scroll, &[], None).len(), 9);
    assert_eq!(plan(Suite::Scopes, &[], None).len(), DEFAULT_SCOPES.len());
    assert!(plan(Suite::Ui, &[], None).iter().any(|s| s.op == Op::SelectAll));
    let all = plan(Suite::All, &[], None);
    assert_eq!(all.len(), DEFAULT_SCOPES.len() + 4 + plan(Suite::Scroll, &[], None).len() + plan(Suite::Ui, &[], None).len() + 1);
    assert_eq!(all.last().unwrap().op, Op::Record { secs: IDLE_SECS });
    assert_eq!(all[DEFAULT_SCOPES.len()].op, Op::TaskCenter(true));
    assert_eq!(all[DEFAULT_SCOPES.len() + 1].op, Op::Open("lib:all".into()), "a load runs with the Task Center showing");
    assert_eq!(all[DEFAULT_SCOPES.len() + 3].op, Op::Open("lib:all".into()), "the library is opened again before the view steps");
}

#[test]
fn named_scopes_and_the_duration_replace_the_defaults() {
    let steps = plan(Suite::Scopes, &["tag:a".into(), "cat:b".into()], None);
    assert_eq!(steps.iter().map(|s| s.op.clone()).collect::<Vec<_>>(), vec![Op::Open("tag:a".into()), Op::Open("cat:b".into())]);
    assert!(matches!(plan(Suite::Scroll, &[], Some(3.0))[0].op, Op::Scroll { secs } if secs == 3.0));
    assert!(matches!(plan(Suite::Scroll, &[], Some(-1.0))[0].op, Op::Scroll { secs } if secs == SCROLL_SECS));
    assert!(matches!(plan(Suite::Record, &[], Some(5.0))[0].op, Op::Record { secs } if secs == 5.0));
}

#[test]
fn the_scroll_and_ui_suites_open_named_scopes_first_and_only_those() {
    for suite in [Suite::Scroll, Suite::Ui] {
        let plain = plan(suite, &[], None);
        assert!(!plain.iter().any(|s| matches!(s.op, Op::Open(_))), "{suite:?} opens nothing by default");
        let named = plan(suite, &["lib:all".into()], None);
        assert_eq!(named[0].op, Op::Open("lib:all".into()));
        assert_eq!(named.len(), plain.len() + 1);
    }
}

#[test]
fn a_recording_opens_its_named_scopes_first() {
    let steps = plan(Suite::Record, &["lib:all".into()], Some(4.0));
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0].op, Op::Open("lib:all".into()));
    assert!(matches!(steps[1].op, Op::Record { secs } if secs == 4.0));
}

#[test]
fn a_quick_step_applies_once_and_is_measured_over_its_settle_frames() {
    let mut target = Fake::default();
    let mut b = bench(vec![step("tiles", Op::Tile(96.0))]);
    let (report, _) = run(&mut b, &mut target, 16, 10_000);
    let report = report.expect("the run ends");
    assert_eq!(target.calls.borrow().as_slice(), ["tile 96"]);
    let s = report.step("tiles").unwrap();
    assert!(s.frames >= SETTLE_FRAMES, "{}", s.frames);
    assert!((s.spread.unwrap().dt.avg - 16.0).abs() < 1e-3);
    assert_eq!(s.view_rows, 1_000);
    assert_eq!(s.selected, 7);
}

#[test]
fn every_op_reaches_the_target() {
    let mut target = Fake::default();
    let steps = vec![
        step("mode", Op::Mode(ViewMode::Details)),
        step("sort", Op::Sort(Some((SortBy::Size, true)))),
        step("sort off", Op::Sort(None)),
        step("filter", Op::Filter(serde_json::json!({"types": {"videos": false}}))),
        step("clear", Op::ClearFilters),
        step("all", Op::SelectAll),
        step("none", Op::ClearSelection),
        step("center", Op::TaskCenter(true)),
    ];
    let mut b = bench(steps);
    let (report, _) = run(&mut b, &mut target, 10, 600_000);
    assert_eq!(report.unwrap().steps.len(), 8);
    assert_eq!(
        target.calls.borrow().as_slice(),
        ["mode Details", "sort Some((Size, true))", "sort None", "filter {\"types\":{\"videos\":false}}", "clear_filters", "select_all", "clear_selection", "task_center true"]
    );
}

#[test]
fn a_scroll_walks_from_the_first_row_to_the_last_over_its_duration() {
    let mut target = Fake::default();
    let mut b = bench(vec![step("scroll", Op::Scroll { secs: 2.0 })]);
    let (report, end) = run(&mut b, &mut target, 20, 20_000);
    let report = report.unwrap();
    let scrolls = target.scrolls.borrow();
    assert_eq!(scrolls[0], 0.0);
    assert!(scrolls.windows(2).all(|w| w[1] >= w[0]), "the scroll never goes back");
    assert_eq!(*scrolls.last().unwrap(), 1.0);
    let s = report.step("scroll").unwrap();
    assert!((s.seconds - 2.0).abs() < 0.1, "{}", s.seconds);
    assert!(s.frames >= 90 && s.frames <= 110, "{}", s.frames);
    assert!(end.elapsed() < Duration::from_secs(5), "simulated time, not wall time");
}

#[test]
fn opening_a_scope_waits_for_the_load_and_carries_its_summary() {
    let mut target = Fake { load_polls_per_open: 25, ..Default::default() };
    let mut b = bench(vec![step("load", Op::Open("lib:all".into()))]);
    let (report, _) = run(&mut b, &mut target, 16, 60_000);
    let s = report.unwrap().steps.remove(0);
    assert_eq!(target.calls.borrow().as_slice(), ["open lib:all"]);
    assert_eq!(s.scope.as_deref(), Some("lib:all"));
    assert_eq!(s.load.as_ref().unwrap().rows, 1_000);
    assert!(s.frames >= 25 + 3, "frames while loading and a few after: {}", s.frames);
}

#[test]
fn a_load_that_never_ends_times_out_with_a_note() {
    let mut target = Fake { load_polls_per_open: usize::MAX, ..Default::default() };
    let mut b = bench(vec![step("load", Op::Open("lib:all".into()))]);
    let (report, _) = run(&mut b, &mut target, 1_000, 1_000_000);
    let s = report.unwrap().steps.remove(0);
    assert!(s.note.unwrap().contains("timed out"));
}

#[test]
fn a_top_name_is_looked_up_until_it_is_known() {
    let mut target = Fake::default();
    target.resolutions.borrow_mut().extend([Resolved::Pending, Resolved::Pending, Resolved::Ready("tag:sunset".into())]);
    let mut b = bench(vec![step("load", Op::Open("tag:@top".into()))]);
    let (report, _) = run(&mut b, &mut target, 16, 60_000);
    let s = report.unwrap().steps.remove(0);
    assert_eq!(s.scope.as_deref(), Some("tag:sunset"));
    assert!(target.calls.borrow().contains(&"open tag:sunset".to_string()));
    assert_eq!(target.calls.borrow().iter().filter(|c| c.starts_with("resolve")).count(), 3);
}

#[test]
fn a_top_name_that_does_not_exist_skips_the_step() {
    let mut target = Fake::default();
    target.resolutions.borrow_mut().push_back(Resolved::Missing);
    let mut b = bench(vec![step("load", Op::Open("col:@top".into())), step("tiles", Op::Tile(64.0))]);
    let (report, _) = run(&mut b, &mut target, 16, 60_000);
    let report = report.unwrap();
    assert_eq!(report.steps.len(), 2, "the run goes on");
    assert_eq!(report.steps[0].frames, 0);
    assert!(report.steps[0].note.as_ref().unwrap().contains("nothing to open"));
    assert!(!target.calls.borrow().iter().any(|c| c.starts_with("open ")));
}

#[test]
fn a_top_name_that_never_resolves_gives_up() {
    let mut target = Fake::default();
    let mut b = bench(vec![step("load", Op::Open("cat:@top".into()))]);
    let (report, _) = run(&mut b, &mut target, 1_000, 1_000_000);
    assert!(report.unwrap().steps[0].note.as_ref().unwrap().contains("nothing to open"));
}

#[test]
fn a_scope_that_cannot_be_opened_is_reported_and_skipped() {
    let mut target = Fake { open_error: Some("not a folder".into()), ..Default::default() };
    let mut b = bench(vec![step("load", Op::Open("fs:/nope".into()))]);
    let (report, _) = run(&mut b, &mut target, 16, 60_000);
    let s = report.unwrap().steps.remove(0);
    assert!(s.note.unwrap().contains("not a folder"));
    assert_eq!(s.frames, 0);
}

#[test]
fn queries_and_cpu_are_the_difference_over_the_step() {
    struct Busy(Fake);
    impl Target for Busy {
        fn view_rows(&self) -> usize { self.0.view_rows() }
        fn selected(&self) -> usize { 0 }
        fn is_loading(&self) -> bool { false }
        fn open(&mut self, _: &str) -> Result<(), String> { Ok(()) }
        fn last_load(&self) -> Option<LoadSummary> { None }
        fn scroll_to(&mut self, _: f32) {
            self.0.queries.set(self.0.queries.get() + 3);
            self.0.cpu.set(self.0.cpu.get() + 0.01);
        }
        fn set_tile(&mut self, _: f32) {}
        fn set_mode(&mut self, _: ViewMode) {}
        fn sort(&mut self, _: Option<(SortBy, bool)>) {}
        fn filter(&mut self, _: &serde_json::Value) -> Result<(), String> { Ok(()) }
        fn clear_filters(&mut self) {}
        fn select_all(&mut self) {}
        fn clear_selection(&mut self) {}
        fn task_center(&mut self, _: bool) {}
        fn resolve(&mut self, _: &str) -> Resolved { Resolved::Missing }
        fn meters(&self) -> Meters { self.0.meters() }
        fn frame_mark(&self) -> u64 { self.0.frame_mark() }
        fn frames_since(&self, mark: u64) -> Vec<Frame> { self.0.frames_since(mark) }
    }
    let mut target = Busy(Fake::default());
    let mut b = bench(vec![step("scroll", Op::Scroll { secs: 1.0 })]);
    let start = Instant::now();
    let mut now = start;
    let report = loop {
        target.0.draw(20.0);
        if let Tick::Done(r) = b.tick(&mut target, now) {
            break *r;
        }
        now += Duration::from_millis(20);
    };
    let s = report.step("scroll").unwrap();
    assert!(s.queries >= 100, "{}", s.queries);
    let pages = s.queries_by_site.get("scope page").copied().unwrap();
    let reads = s.queries_by_site.get("thumb read").copied().unwrap();
    assert_eq!(s.queries, pages + reads, "the total is the sum of the sites");
    assert!(reads * 2 <= pages + 1 && reads * 2 + 2 >= pages, "the fake makes one read per two pages: {reads} and {pages}");
    let cpu = s.cpu_percent.unwrap();
    assert!(cpu > 40.0 && cpu < 60.0, "0.01 s of CPU every 20 ms is about 50 percent, got {cpu}");
    assert_eq!(report.rss_end_mb, Some(200.0));
    assert_eq!((s.rss_anon_mb, s.rss_file_mb), (Some(150.0), Some(50.0)));
    assert_eq!(report.peak_rss_mb, Some(300.0));
}

#[test]
fn a_recording_asks_for_repaints_no_more_than_once_every_few_seconds() {
    let mut target = Fake::default();
    let mut b = bench(vec![step("idle", Op::Record { secs: 30.0 })]);
    let now = Instant::now();
    target.draw(1000.0);
    match b.tick(&mut target, now) {
        Tick::Running(Repaint::After(d)) => assert_eq!(d, RECORD_WAKE),
        _ => panic!("a recording waits"),
    }
    match b.tick(&mut target, now + Duration::from_millis(29_900)) {
        Tick::Running(Repaint::After(d)) => assert!(d >= Duration::from_millis(90) && d <= Duration::from_millis(110), "{d:?}"),
        _ => panic!("a recording near its end wakes for it"),
    }
    assert!(matches!(b.tick(&mut target, now + Duration::from_secs(31)), Tick::Running(Repaint::After(_)) | Tick::Done(_)));
}

#[test]
fn the_report_is_valid_json_and_readable_text() {
    let mut target = Fake::default();
    let mut b = bench(vec![step("tiles", Op::Tile(96.0))]);
    let (report, _) = run(&mut b, &mut target, 16, 60_000);
    let report = report.unwrap();
    let json: serde_json::Value = serde_json::from_str(&report.to_json()).unwrap();
    assert_eq!(json["suite"], "ui");
    assert_eq!(json["steps"][0]["label"], "tiles");
    assert!(json["steps"][0]["spread"]["dt"]["p95"].is_number());
    assert!(json["steps"][0]["slowest"][0]["dt_ms"].is_number());
    assert!(json["marks"].is_array());
    let text = report.to_text();
    assert!(text.contains("tiles") && text.contains("dt avg"), "{text}");
}

#[test]
fn repaint_causes_are_the_most_frequent_new_ones_since_the_step_began() {
    let before = Meters { causes: BTreeMap::from([("a.rs:1".to_string(), 5), ("b.rs:2".to_string(), 1)]), ..Default::default() };
    let after = Meters { causes: BTreeMap::from([("a.rs:1".to_string(), 8), ("b.rs:2".to_string(), 1), ("c.rs:3".to_string(), 20), ("d.rs:4".to_string(), 3)]), ..Default::default() };
    assert_eq!(after.causes_since(&before, 10), vec![("c.rs:3".to_string(), 20), ("a.rs:1".to_string(), 3), ("d.rs:4".to_string(), 3)]);
    assert_eq!(after.causes_since(&before, 1).len(), 1);
    assert!(after.causes_since(&after, 5).is_empty());
}

#[test]
fn progress_counts_finished_and_running_steps() {
    let mut target = Fake::default();
    let now = Instant::now();
    let mut b = Bench::new(Suite::Ui, vec![step("a", Op::Tile(96.0)), step("b", Op::Tile(160.0))], now);
    assert_eq!(b.progress(now), Progress { step: 0, steps: 2, label: String::new(), elapsed: 0.0, quiet: false });
    target.draw(16.0);
    let _ = b.tick(&mut target, now);
    let p = b.progress(now);
    assert_eq!((p.step, p.steps, p.label.as_str()), (1, 2, "a"));
}
