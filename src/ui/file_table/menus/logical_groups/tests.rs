use std::sync::Mutex;

use eframe::egui::{Context, Pos2, RawInput, Rect, vec2};

use super::*;
use crate::database::collections::tests::{db, insert_rows, seed_row};
use crate::database::ui_themes::tests::RT;
use crate::ui::file_table::FileExplorer;
use crate::ui::test_support::{press_release, test_ctx, texts};

/// Serializes the tests that use the menu's process-wide state.
static MENU_LOCK: Mutex<()> = Mutex::new(());

fn thumb(path: &str, file_type: &str) -> Thumbnail {
    let mut t = Thumbnail::new(path.rsplit('/').next().unwrap_or(path));
    t.path = path.to_string();
    t.file_type = file_type.to_string();
    t
}

fn group(name: &str) -> LogicalGroup {
    LogicalGroup { id: collections::collection_id(&name.to_lowercase()), name: name.to_string() }
}

fn reset_menu() {
    menu(|m| *m = MenuState { refreshed: Some(Instant::now()), ..Default::default() });
}

/// An explorer with three files and a folder in view, two files selected and two collections listed.
fn explorer() -> FileExplorer {
    let mut ex = FileExplorer::new(true);
    for (path, ty) in [("/v/a.jpg", "jpg"), ("/v/b.jpg", "jpg"), ("/v/c.png", "png"), ("/v/sub", "<DIR>")] {
        ex.table.push(thumb(path, ty));
    }
    ex.sync_view();
    ex.select_paths(["/v/a.jpg", "/v/c.png", "/v/sub"]);
    ex.logical_groups = vec![LogicalGroup::default(), group("Holiday"), group("Trip")];
    ex
}

fn body_frame(ctx: &Context, ex: &mut FileExplorer, events: Vec<eframe::egui::Event>) -> Vec<(String, Pos2)> {
    let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(900.0, 1500.0))), events, ..Default::default() };
    let out = ctx.run_ui(input, |ui| ex.collections_menu_body(ui));
    let all = texts(&out.shapes);
    out.drop_without_applying_deltas();
    all
}

fn settled(ctx: &Context, ex: &mut FileExplorer) -> Vec<(String, Pos2)> {
    let mut last = Vec::new();
    for _ in 0..3 {
        last = body_frame(ctx, ex, Vec::new());
    }
    last
}

fn has(all: &[(String, Pos2)], text: &str) -> bool {
    all.iter().any(|(t, _)| t == text)
}

#[test]
fn the_button_label_names_the_active_collection_and_cuts_long_names() {
    assert_eq!(menu_label(None), "Collections");
    assert_eq!(menu_label(Some("Holiday")), "Collections (Holiday)");
    assert_eq!(menu_label(Some("A very long collection name")), "Collections (A very long col...)");
    assert_eq!(menu_label(Some(&"x".repeat(18))), format!("Collections ({})", "x".repeat(18)));
    assert_eq!(menu_label(Some("日本語日本語日本語日本語日本語日本語日本語日本語")).chars().filter(|c| *c == '.').count(), 3);
}

#[test]
fn the_unfiled_marker_is_not_an_active_collection() {
    assert_eq!(collection_of_active(None), None);
    assert_eq!(collection_of_active(Some("Default")), None);
    assert_eq!(collection_of_active(Some("Holiday")), Some("Holiday"));
    assert_eq!(collection_of_active(Some("default")), Some("default"), "only the exact marker means unfiled");
}

#[tokio::test]
async fn adding_creates_missing_collections_reuses_existing_ones_ignoring_case_and_reports() {
    let db = db().await;
    let paths: Vec<String> = ["/v/a.jpg", "/v/b.jpg"].iter().map(|s| s.to_string()).collect();
    let first = add_to_named(&db, "Holiday", &paths).await.unwrap();
    assert_eq!(first, "Added 2 files to \"Holiday\" (new collection); 0 were already there");
    let again = add_to_named(&db, "holiday", &paths[..1]).await.unwrap();
    assert_eq!(again, "Added 0 files to \"Holiday\"; 1 were already there");
    assert_eq!(collections::list(&db).await.unwrap().len(), 1);
    assert!(add_to_named(&db, "Default", &paths).await.is_err(), "the unfiled marker is not a collection");
    assert!(add_to_named(&db, "  ", &paths).await.is_err());
}

#[tokio::test]
async fn a_file_added_to_two_collections_stays_in_both_and_removal_touches_one() {
    let db = db().await;
    let paths = vec!["/v/a.jpg".to_string(), "/v/b.jpg".to_string(), "/v/c.jpg".to_string()];
    add_to_named(&db, "One", &paths).await.unwrap();
    add_to_named(&db, "Two", &paths[..2]).await.unwrap();
    let one = collections::get_by_name(&db, "One").await.unwrap().unwrap();
    let msg = remove_from(&db, &one.id, "One", &["/v/a.jpg".to_string(), "/v/zzz.jpg".to_string()]).await.unwrap();
    assert_eq!(msg, "Removed 1 files from \"One\"; 1 were not in it");
    assert_eq!(collections::count_members(&db, &one.id).await.unwrap(), 2);
    let two = collections::get_by_name(&db, "Two").await.unwrap().unwrap();
    assert_eq!(collections::count_members(&db, &two.id).await.unwrap(), 2, "the other collection keeps the file");
    assert_eq!(collections::collections_of(&db, "/v/a.jpg").await.unwrap().len(), 1);
}

#[tokio::test]
async fn copying_between_collections_creates_the_destination_and_refuses_bad_pairs() {
    let db = db().await;
    let paths: Vec<String> = (0..5).map(|i| format!("/v/{i}.jpg")).collect();
    add_to_named(&db, "Source", &paths).await.unwrap();
    let msg = copy_between(&db, "source", "Destination").await.unwrap();
    assert_eq!(msg, "Copied 5 files from \"Source\" to \"Destination\"");
    assert_eq!(copy_between(&db, "Source", "destination").await.unwrap(), "Copied 0 files from \"Source\" to \"Destination\"");
    assert!(copy_between(&db, "Source", "SOURCE").await.is_err());
    assert!(copy_between(&db, "Missing", "Destination").await.is_err());
    assert_eq!(collections::list(&db).await.unwrap().len(), 2);
}

#[tokio::test]
async fn files_in_no_collection_can_be_added_in_bulk() {
    let db = db().await;
    insert_rows(&db, (0..6).map(|i| seed_row(i, &format!("/v/{i}.jpg"))).collect()).await;
    add_to_named(&db, "Existing", &["/v/0.jpg".to_string(), "/v/1.jpg".to_string()]).await.unwrap();
    let msg = add_unfiled(&db, "Rest").await.unwrap();
    assert_eq!(msg, "Added 4 files to \"Rest\" (new collection); 0 were already there");
    assert_eq!(add_unfiled(&db, "Rest").await.unwrap(), "Every indexed file is already in a collection");
}

#[tokio::test]
async fn the_menu_shows_the_selection_count_the_list_with_counts_and_the_active_collection() {
    let _guard = MENU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _active = crate::command::context::serial();
    crate::command::context::set_active_collection(None);
    reset_menu();
    let _rt = RT.enter();
    let ctx = test_ctx();
    let mut ex = explorer();
    menu(|m| {
        m.counts.insert("holiday".into(), 12);
        m.counts.insert("trip".into(), 1);
    });
    let all = settled(&ctx, &mut ex);
    assert!(has(&all, "Selected files: 2"), "the folder in the selection is not counted: {all:?}");
    assert!(has(&all, "Add selection (2) to:"), "{all:?}");
    assert!(has(&all, "Add current view (3) to:"), "{all:?}");
    assert!(has(&all, "Holiday") && has(&all, "12 files") && has(&all, "Trip") && has(&all, "1 files"), "{all:?}");
    assert!(has(&all, "No active collection: scanned files are not added to one."));
    assert!(!has(&all, "Default"), "the unfiled entry is not listed");
    assert!(has(&all, "Create New Collection"));

    crate::command::context::set_active_collection(Some("Holiday".into()));
    let active = settled(&ctx, &mut ex);
    assert!(has(&active, "Active collection: Holiday"), "{active:?}");
    crate::command::context::set_active_collection(Some("Deleted Elsewhere".into()));
    let stale = settled(&ctx, &mut ex);
    assert!(has(&stale, "Active collection \"Deleted Elsewhere\" no longer exists."), "{stale:?}");
    assert!(has(&stale, "Clear"));
    crate::command::context::set_active_collection(Some("Default".into()));
    let unfiled = settled(&ctx, &mut ex);
    assert!(has(&unfiled, "No active collection: scanned files are not added to one."), "Default reads as none");
    reset_menu();
}

#[tokio::test]
async fn nothing_selected_means_nothing_to_add_and_the_count_says_so() {
    let _guard = MENU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _active = crate::command::context::serial();
    crate::command::context::set_active_collection(None);
    reset_menu();
    let _rt = RT.enter();
    let ctx = test_ctx();
    let mut ex = explorer();
    ex.clear_selection();
    assert_eq!(ex.selected_file_count(), 0);
    assert_eq!(ex.view_file_count(), 3);
    let all = settled(&ctx, &mut ex);
    assert!(has(&all, "Selected files: 0") && has(&all, "Add selection (0) to:"), "{all:?}");
    ex.select_paths(["/v/sub", "/elsewhere/x.jpg"]);
    assert_eq!(ex.selected_file_count(), 0, "folders and paths that are not in view do not count");
    assert!(ex.selected_file_rows().is_empty());
    assert_eq!(ex.view_file_count(), 3);
    reset_menu();
}

#[tokio::test]
async fn the_migration_hint_the_message_and_the_armed_delete_are_drawn_from_the_menu_state() {
    let _guard = MENU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _active = crate::command::context::serial();
    crate::command::context::set_active_collection(None);
    reset_menu();
    let _rt = RT.enter();
    let ctx = test_ctx();
    let mut ex = explorer();
    menu(|m| {
        m.pending_migration = true;
        m.message = Some((true, "Added 2 files".into()));
        m.confirm_delete = Some("Trip".into());
    });
    let all = settled(&ctx, &mut ex);
    assert!(has(&all, "Logical groups from the old database are not collections yet."), "{all:?}");
    assert!(has(&all, "Migrate now") && has(&all, "Added 2 files") && has(&all, "Confirm delete") && has(&all, "Keep"), "{all:?}");
    reset_menu();
}

#[tokio::test]
async fn clearing_the_active_collection_returns_to_the_unfiled_marker() {
    let _guard = MENU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _active = crate::command::context::serial();
    crate::command::context::set_active_collection(None);
    reset_menu();
    let _rt = RT.enter();
    let ctx = test_ctx();
    let mut ex = explorer();
    crate::command::context::set_active_collection(Some("Trip".into()));
    let all = settled(&ctx, &mut ex);
    let clear = all.iter().find(|(t, _)| t == "Clear").map(|(_, p)| *p).expect("a Clear button is drawn for an active collection");
    body_frame(&ctx, &mut ex, press_release(clear + vec2(3.0, 3.0)));
    settled(&ctx, &mut ex);
    assert_eq!(crate::command::context::active_collection(), None);
    assert_eq!(ex.active_collection(), None);
    let after = settled(&ctx, &mut ex);
    assert!(has(&after, "No active collection: scanned files are not added to one."));
    reset_menu();
}

fn quick_frame(ctx: &Context, ex: &mut FileExplorer) -> Vec<(String, Pos2)> {
    let mut last = Vec::new();
    for _ in 0..3 {
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(400.0, 800.0))), ..Default::default() };
        let out = ctx.run_ui(input, |ui| ex.collections_quick_list(ui));
        last = texts(&out.shapes);
        out.drop_without_applying_deltas();
    }
    last
}

#[tokio::test]
async fn the_quick_access_list_shows_counts_the_active_collection_and_the_migration_hint() {
    let _guard = MENU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _active = crate::command::context::serial();
    crate::command::context::set_active_collection(None);
    reset_menu();
    let _rt = RT.enter();
    let ctx = test_ctx();
    let mut ex = explorer();
    menu(|m| {
        m.counts.insert("holiday".into(), 12);
        m.counts.insert("trip".into(), 1);
    });
    let none = quick_frame(&ctx, &mut ex);
    assert!(has(&none, "Active: (none)") && has(&none, "Holiday (12)") && has(&none, "Trip (1)"), "{none:?}");
    assert!(!has(&none, "Default"));
    crate::command::context::set_active_collection(Some("Trip".into()));
    assert!(has(&quick_frame(&ctx, &mut ex), "Active: Trip"));
    menu(|m| m.pending_migration = true);
    assert!(has(&quick_frame(&ctx, &mut ex), "Old logical groups are not migrated; use the Collections menu."));
    ex.logical_groups = vec![LogicalGroup::default()];
    assert!(has(&quick_frame(&ctx, &mut ex), "No collections yet."));
    reset_menu();
}


fn files(paths: &[&str]) -> Vec<Thumbnail> {
    paths.iter().map(|p| thumb(p, "jpg")).collect()
}

#[test]
fn a_group_action_on_nothing_selected_is_refused_and_says_to_select_files() {
    let err = plan_group_action(&[], Some("Trip")).unwrap_err();
    assert!(err.starts_with("Select files first"), "{err}");
    let folders = [thumb("/v/sub", "<DIR>")];
    let refs: Vec<&Thumbnail> = folders.iter().collect();
    assert!(plan_group_action(&refs, Some("Trip")).is_err(), "a selected folder is not a file");
}

#[test]
fn a_group_action_needs_an_active_collection() {
    let rows = files(&["/v/a.jpg"]);
    let refs: Vec<&Thumbnail> = rows.iter().collect();
    for active in [None, Some("Default")] {
        let err = plan_group_action(&refs, active).unwrap_err();
        assert!(err.contains("active collection"), "{err}");
    }
}

#[test]
fn a_group_action_plans_the_selected_files_only() {
    let rows = vec![thumb("/v/a.jpg", "jpg"), thumb("/v/sub", "<DIR>"), thumb("/v/b.png", "png")];
    let refs: Vec<&Thumbnail> = rows.iter().collect();
    let (name, planned) = plan_group_action(&refs, Some("Trip")).unwrap();
    assert_eq!(name, "Trip");
    assert_eq!(planned.iter().map(|r| r.path.as_str()).collect::<Vec<_>>(), vec!["/v/a.jpg", "/v/b.png"]);
}

#[tokio::test]
async fn adding_the_selection_saves_the_files_and_files_them_in_the_collection_with_counts() {
    let db = db().await;
    collections::create(&db, "Trip").await.unwrap();
    let msg = apply_group_action(&db, GroupAction::Add, "Trip", files(&["/v/a.jpg", "/v/b.jpg"])).await.unwrap();
    assert_eq!(msg, "Added 2 files to \"Trip\"; 0 were already there");
    let c = collections::get_by_name(&db, "Trip").await.unwrap().unwrap();
    assert_eq!(collections::count_members(&db, &c.id).await.unwrap(), 2);
    let indexed: Vec<String> = db.query("SELECT VALUE path FROM thumbnails ORDER BY path").await.unwrap().take(0).unwrap();
    assert_eq!(indexed, vec!["/v/a.jpg".to_string(), "/v/b.jpg".to_string()]);
    let again = apply_group_action(&db, GroupAction::Add, "Trip", files(&["/v/b.jpg", "/v/c.jpg"])).await.unwrap();
    assert_eq!(again, "Added 1 files to \"Trip\"; 1 were already there");
}

#[tokio::test]
async fn removing_the_selection_takes_only_those_files_out_and_reports_the_counts() {
    let db = db().await;
    apply_group_action(&db, GroupAction::Add, "Trip", files(&["/v/a.jpg", "/v/b.jpg", "/v/c.jpg"])).await.unwrap();
    let msg = apply_group_action(&db, GroupAction::Remove, "trip", files(&["/v/a.jpg", "/v/zzz.jpg"])).await.unwrap();
    assert_eq!(msg, "Removed 1 files from \"Trip\"; 1 were not in it");
    let c = collections::get_by_name(&db, "Trip").await.unwrap().unwrap();
    assert_eq!(collections::count_members(&db, &c.id).await.unwrap(), 2);
    let err = apply_group_action(&db, GroupAction::Remove, "Gone", files(&["/v/a.jpg"])).await.unwrap_err();
    assert!(err.to_string().contains("Gone"), "{err}");
}
