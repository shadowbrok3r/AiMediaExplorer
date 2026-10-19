use std::sync::Mutex;

use eframe::egui::{Pos2, RawInput, Rect, vec2};

use super::*;
use crate::database::collections::Collection;
use crate::database::collections_migration::{Assignment, MigrationState, PlannedGroup};
use crate::ui::test_support::{test_ctx, texts};

/// Serializes the tests that use the process-wide panel state.
static PANEL_LOCK: Mutex<()> = Mutex::new(());

fn drawn(ctx: &Context) -> Vec<String> {
    let mut last = Vec::new();
    for _ in 0..3 {
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1100.0, 1400.0))), ..Default::default() };
        let out = ctx.run_ui(input, |ui| show(ui.ctx()));
        last = texts(&out.shapes).into_iter().map(|(t, _)| t).collect();
        out.drop_without_applying_deltas();
    }
    last
}

fn drawn_body(ctx: &Context) -> Vec<String> {
    let mut last = Vec::new();
    for _ in 0..3 {
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1100.0, 2400.0))), ..Default::default() };
        let out = ctx.run_ui(input, |ui| body(ui));
        last = texts(&out.shapes).into_iter().map(|(t, _)| t).collect();
        out.drop_without_applying_deltas();
    }
    last
}

fn count(name: &str, n: u64) -> CollectionCount {
    let id = collections::collection_id(&name.to_lowercase());
    CollectionCount {
        collection: Collection { id, name: name.to_string(), created: None, updated: None, color: None, note: None, legacy_key: Some(format!("legacy-{name}")) },
        count: n,
    }
}

fn reset() {
    with(|p| *p = Panel::default());
}

fn assignment(name: &str) -> Assignment {
    Assignment { legacy_key: name.into(), legacy_name: name.into(), name: name.into(), reuse: false, renamed: false }
}

#[test]
fn pasted_paths_are_trimmed_unquoted_and_deduplicated_in_order() {
    let text = "  /a/one.jpg \r\n\r\n\"/b/two words.jpg\"\n'/c/three.jpg'\n/a/one.jpg\n   \n\"\n/d/\"quoted\"/x.jpg";
    assert_eq!(parse_paths(text), vec!["/a/one.jpg", "/b/two words.jpg", "/c/three.jpg", "\"", "/d/\"quoted\"/x.jpg"]);
    assert!(parse_paths("").is_empty());
    assert!(parse_paths(" \n\t\n").is_empty());
    assert_eq!(parse_paths("C:\\Users\\me\\a.jpg"), vec!["C:\\Users\\me\\a.jpg"]);
}

#[tokio::test]
async fn pasted_paths_are_added_and_removed_with_a_summary() {
    let db = crate::database::collections::tests::db().await;
    let c = collections::create(&db, "Manual").await.unwrap();
    let msg = apply_paths(&db, &c.id, "/m/1.jpg\n/m/2.jpg\n/m/2.jpg\n", true).await.unwrap();
    assert_eq!(msg, "2 added, 0 already in the collection");
    let again = apply_paths(&db, &c.id, "/m/1.jpg\n/m/3.jpg", true).await.unwrap();
    assert_eq!(again, "1 added, 1 already in the collection");
    let gone = apply_paths(&db, &c.id, "/m/1.jpg\n/m/9.jpg", false).await.unwrap();
    assert_eq!(gone, "1 removed, 1 were not in the collection");
    assert_eq!(collections::count_members(&db, &c.id).await.unwrap(), 2);
    assert!(apply_paths(&db, &c.id, "  \n", true).await.is_err());
    assert!(apply_paths(&db, &collections::collection_id("ghost"), "/m/1.jpg", true).await.is_err());
}

#[test]
fn the_open_flag_toggles_per_context() {
    let ctx = test_ctx();
    assert!(!is_open(&ctx));
    toggle(&ctx);
    assert!(is_open(&ctx));
    toggle(&ctx);
    assert!(!is_open(&ctx));
}

#[tokio::test]
async fn a_closed_window_draws_nothing_and_an_open_one_draws_its_title_and_sections() {
    let _guard = PANEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    reset();
    let ctx = test_ctx();
    assert!(drawn(&ctx).is_empty());
    set_open(&ctx, true);
    let window = drawn(&ctx);
    assert!(window.iter().any(|t| t == "Devtools: Collections"), "{window:?}");
    let all = drawn_body(&ctx);
    for want in [
        "Collections (0)",
        "Add or remove paths",
        "Migration from logical groups",
        "Orphan memberships",
        "Timing probe",
        "The database is not connected in this process; actions are disabled.",
        "No collections yet.",
        "Dry run",
        "Verify",
        "Cancel",
        "Create",
        "Phase",
        "not started",
    ] {
        assert!(all.iter().any(|t| t == want), "missing {want:?} in {all:?}");
    }
}

#[tokio::test]
async fn the_list_status_plan_and_messages_are_rendered_from_the_panel_state() {
    let _guard = PANEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    reset();
    let ctx = test_ctx();
    set_open(&ctx, true);
    let state = MigrationState { started: Some("t".into()), copy_done: true, memberships_added: 7, unfiled_rows: 4, needs_resync: true, warnings: vec!["a warning".into()], ..Default::default() };
    with(|p| {
        p.list = vec![count("Holiday", 12), count("Trip", 0)];
        p.status = Some(Status { phase: Phase::Copied, phase_label: Phase::Copied.label(), state, legacy_groups: 2, collections: 2, memberships: 12, orphan_memberships: 0 });
        p.note = Some((true, "all good".into()));
        p.target = Some("holiday".into());
        p.orphans = Some((3, 5));
        p.plan = Some(Plan { groups: vec![PlannedGroup { assignment: assignment("Holiday"), memberships: 12 }], memberships: 12, to_create: 1, ..Default::default() });
        p.probe = Some(vec![ProbeStep { step: "add_paths".into(), millis: 42, detail: "10 added".into() }]);
    });
    let all = drawn_body(&ctx);
    for want in [
        "Collections (2)",
        "Holiday",
        "12",
        "Trip",
        "all good",
        "copied, not finished",
        "legacy rows were imported after the last copy; run the copy again",
        "a warning",
        "Migrate again",
        "Finish",
        "Rollback...",
    ] {
        assert!(all.iter().any(|t| t == want), "missing {want:?} in {all:?}");
    }
    assert!(all.iter().any(|t| t.starts_with("Dry run: 1 groups, 12 memberships, 1 to create")), "{all:?}");
    assert!(all.iter().any(|t| t.starts_with("create  \"Holiday\" -> \"Holiday\"")), "{all:?}");
    reset();
}

#[tokio::test]
async fn confirmation_frames_show_their_warnings() {
    let _guard = PANEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    reset();
    let ctx = test_ctx();
    set_open(&ctx, true);
    with(|p| {
        p.list = vec![count("Holiday", 3)];
        p.confirm_delete = Some(collections::collection_id("holiday"));
        p.confirm_rollback = true;
        p.status = Some(Status { phase: Phase::Copied, phase_label: "x", state: MigrationState { started: Some("t".into()), ..Default::default() }, legacy_groups: 0, collections: 1, memberships: 3, orphan_memberships: 0 });
    });
    let all = drawn_body(&ctx);
    assert!(all.iter().any(|t| t.starts_with("Delete \"Holiday\"? Its memberships are removed")), "{all:?}");
    assert!(all.iter().any(|t| t.starts_with("Rollback deletes every collection and membership.")), "{all:?}");
    assert!(all.iter().any(|t| t == "Roll back"));
    reset();
}
