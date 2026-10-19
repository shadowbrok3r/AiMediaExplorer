use std::time::{Duration, Instant};

use super::*;
use crate::ui::devtools::scale::bench::{Op, Step, plan};
use crate::ui::file_table::rows::SortKey;
use crate::ui::file_table::test_rig::{ROWS, Rig};

fn step(label: &str, op: Op) -> Step {
    Step { label: label.to_string(), op }
}

/// Draws the rig and ticks the run once per simulated frame until it ends.
fn run_to_the_end(rig: &mut Rig, suite: bench::Suite, steps: Vec<Step>) -> bench::Report {
    let top = TopNames::known(Some("sunset"), Some("landscape"), Some("Holidays"));
    let mut b = Bench::new(suite, steps, Instant::now());
    let mut now = Instant::now();
    for _ in 0..20_000 {
        rig.frame(Vec::new());
        scale::frame_end(&rig.ctx, Duration::from_millis(2), Some(0.001));
        now += Duration::from_millis(16);
        if let Tick::Done(report) = b.tick(&mut Driver { ex: &mut rig.ex, ctx: &rig.ctx, top: &top }, now) {
            return *report;
        }
    }
    panic!("the run did not end");
}

#[test]
fn sorting_and_tile_size_reach_the_explorer() {
    let mut rig = Rig::new(900.0, 700.0);
    let report = run_to_the_end(&mut rig, bench::Suite::Ui, vec![step("tiles", Op::Tile(256.0)), step("sort", Op::Sort(Some((SortBy::Size, true))))]);
    assert_eq!(report.steps.len(), 2);
    assert_eq!(rig.ex.browse.tile, 256.0);
    assert_eq!(rig.ex.view.sort().keys(), &[SortKey { column: Col::Size.id(), descending: true }]);
}

#[test]
fn switching_the_sort_off_and_the_view_mode_reaches_the_explorer() {
    let mut rig = Rig::new(900.0, 700.0);
    rig.ex.view.set_sort(SortSpec::single(Col::Name.id(), false));
    run_to_the_end(&mut rig, bench::Suite::Ui, vec![step("off", Op::Sort(None)), step("details", Op::Mode(ViewMode::Details))]);
    assert!(rig.ex.view.sort().is_empty());
    assert_eq!(rig.ex.browse.mode, ViewMode::Details);
}

#[test]
fn select_all_selects_every_visible_row_and_clearing_empties_it() {
    let mut rig = Rig::new(900.0, 700.0);
    let report = run_to_the_end(&mut rig, bench::Suite::Ui, vec![step("all", Op::SelectAll), step("none", Op::ClearSelection)]);
    assert_eq!(report.steps[0].selected, ROWS + 1, "the folder row and every image");
    assert_eq!(report.steps[1].selected, 0);
    assert_eq!(rig.ex.selection.len(), 0);
}

#[test]
fn a_filter_step_changes_what_the_view_lists_and_clearing_brings_it_back() {
    let mut rig = Rig::new(900.0, 700.0);
    rig.ex.clear_filters();
    rig.ex.sync_view();
    let before = rig.ex.view.len();
    let report = run_to_the_end(&mut rig, bench::Suite::Ui, vec![step("no images", Op::Filter(serde_json::json!({"types": {"images": false}})))]);
    assert!(report.steps[0].view_rows < before, "{} of {before}", report.steps[0].view_rows);
    run_to_the_end(&mut rig, bench::Suite::Ui, vec![step("clear", Op::ClearFilters)]);
    rig.ex.sync_view();
    assert_eq!(rig.ex.view.len(), before);
}

#[test]
fn a_bad_filter_is_reported_on_its_step_and_the_run_goes_on() {
    let mut rig = Rig::new(900.0, 700.0);
    let report = run_to_the_end(&mut rig, bench::Suite::Ui, vec![step("bad", Op::Filter(serde_json::json!({"nonsense": 1}))), step("tiles", Op::Tile(96.0))]);
    assert!(report.steps[0].note.is_some());
    assert_eq!(report.steps[1].label, "tiles");
}

#[test]
fn scrolling_runs_in_both_view_modes_without_rows_to_spare() {
    let mut rig = Rig::new(900.0, 700.0);
    let report = run_to_the_end(&mut rig, bench::Suite::Scroll, plan(bench::Suite::Scroll, &[], Some(0.2)));
    assert_eq!(report.steps.len(), 9);
    assert_eq!(rig.ex.browse.mode, ViewMode::Grid, "the scroll suite ends where it began");
}

#[test]
fn the_task_center_step_opens_and_closes_the_popover() {
    let mut rig = Rig::new(900.0, 700.0);
    run_to_the_end(&mut rig, bench::Suite::Ui, vec![step("open", Op::TaskCenter(true))]);
    assert!(crate::ui::taskbar::is_center_open(&rig.ctx));
    run_to_the_end(&mut rig, bench::Suite::Ui, vec![step("close", Op::TaskCenter(false))]);
    assert!(!crate::ui::taskbar::is_center_open(&rig.ctx));
}

#[test]
fn a_run_waits_for_the_database_and_a_second_run_is_refused() {
    let _guard = scale::CANCEL_TESTS.lock().unwrap_or_else(|e| e.into_inner());
    let mut rig = Rig::new(900.0, 700.0);
    assert!(!rig.ex.bench_running());
    let started = rig.ex.start_bench(bench::Suite::Ui, None, &[], None);
    assert!(started.is_ok(), "{started:?}");
    assert!(rig.ex.bench_running());
    assert!(rig.ex.start_bench(bench::Suite::Scroll, None, &[], None).unwrap_err().contains("already"));
    if !crate::database::is_ready() {
        rig.settle(3);
        assert!(rig.ex.bench_running(), "nothing runs before the database is ready");
        assert_eq!(scale::progress().map(|p| p.step), Some(0));
    }
    scale::request_cancel();
    rig.settle(2);
    assert!(!rig.ex.bench_running(), "a cancel ends the run");
    assert!(scale::progress().is_none());
}

#[test]
fn a_scope_list_with_a_malformed_scope_is_refused_before_anything_runs() {
    let _guard = scale::CANCEL_TESTS.lock().unwrap_or_else(|e| e.into_inner());
    let mut rig = Rig::new(900.0, 700.0);
    let err = rig.ex.start_bench(bench::Suite::Scopes, None, &["nonsense:1".to_string()], None).unwrap_err();
    assert!(err.contains("nonsense:1"), "{err}");
    assert!(!rig.ex.bench_running());
    assert!(rig.ex.start_bench(bench::Suite::Scopes, None, &["tag:@top".to_string(), "lib:all".to_string()], None).is_ok());
    scale::request_cancel();
    rig.settle(2);
}

#[test]
fn the_tab_publishes_its_numbers_while_the_window_is_open() {
    let mut rig = Rig::new(900.0, 700.0);
    rig.settle(2);
    assert!(scale::latest(&rig.ctx).is_none());
    scale::set_open(&rig.ctx, true);
    rig.settle(2);
    let facts = scale::latest(&rig.ctx).expect("published");
    assert_eq!(facts.store_len, ROWS + 1);
    assert_eq!(facts.view_len, rig.ex.view.len());
    assert!(!facts.library);
}
