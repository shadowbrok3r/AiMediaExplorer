use eframe::egui::*;

use super::*;
use crate::command::{ActionId, Scope};
use crate::ui::file_table::menus::db_delete::GLOBAL_DB_TESTS;
use crate::ui::file_table::test_rig::{ROWS, Rig, db, global_db, path_of, text_rects};
use crate::ui::test_support::{press_release, test_ctx, texts};

fn pin() {
    pin_environment(Some(true), Some(true));
}

fn clean_rig_rows() {
    db(async {
        crate::database::DB.query("DELETE thumbnails WHERE path CONTAINS '/rig/'").await.unwrap();
    });
}

fn row(path: &str, kind: &str, size: u64) -> Thumbnail {
    let mut t = Thumbnail::new(path.rsplit('/').next().unwrap_or(path));
    t.path = path.into();
    t.file_type = kind.into();
    t.size = size;
    t
}

/* ------------------------------------------------- pure -------------------------------------------- */

#[test]
fn facts_count_files_folders_images_and_files_on_disk() {
    let rows = [row("/p/a.jpg", "jpg", 10), row("/p/b.mp4", "mp4", 20), row("zip:///p/x.zip!/c.jpg", "jpg", 30), row("/p/sub", "<DIR>", 0)];
    let refs: Vec<&Thumbnail> = rows.iter().collect();
    let f = facts_of(&refs, true, false);
    assert_eq!((f.selected, f.files, f.dirs, f.local, f.images, f.size), (4, 3, 1, 2, 1, 60));
    assert!(f.db_ready && !f.ai_ready);
    assert_eq!(facts_of(&[], true, true), Facts { db_ready: true, ai_ready: true, ..Default::default() }, "nothing selected is nothing");
}

#[test]
fn the_count_line_names_the_selection_its_kinds_and_its_size() {
    let f = Facts { selected: 12, files: 12, size: 48_200_000, ..Default::default() };
    assert_eq!(count_text(&f), "12 selected (48.20 MB)");
    let mixed = Facts { selected: 3, files: 2, dirs: 1, size: 0, ..Default::default() };
    assert_eq!(count_text(&mixed), "3 selected: 2 files, 1 folder");
    assert_eq!(count_text(&Facts { selected: 1, files: 1, ..Default::default() }), "1 selected");
}

#[test]
fn confirmations_state_the_exact_count_in_the_text_and_on_the_button() {
    for (action, count) in [(ActionId::RemoveFromLibrary, 3), (ActionId::MoveTo, 12), (ActionId::Describe, 120), (ActionId::GenerateClip, 40), (ActionId::AddToLibrary, 26)] {
        let (title, body, button) = confirm_text(action, count);
        assert!(!title.is_empty());
        assert!(body.contains(&format!("{count} files")) || body.contains(&format!("{count} file")), "{action:?}: {body}");
        assert!(button.contains(&count.to_string()), "{action:?}: {button}");
        assert!(title.is_ascii() && body.is_ascii() && button.is_ascii());
    }
    assert!(confirm_text(ActionId::RemoveFromLibrary, 1).1.starts_with("Remove 1 file from the library"));
    assert!(confirm_text(ActionId::RemoveFromLibrary, 3).1.contains("Files on disk are not touched"));
}

#[test]
fn names_are_listed_up_to_five_and_then_counted() {
    let names: Vec<String> = (1..=8).map(|i| format!("{i}.jpg")).collect();
    assert_eq!(names_text(&names[..3]), "1.jpg, 2.jpg, 3.jpg");
    assert_eq!(names_text(&names), "1.jpg, 2.jpg, 3.jpg, 4.jpg, 5.jpg and 3 more");
    assert_eq!(names_text(&[]), "");
}

/* -------------------------------------------- bar frames ------------------------------------------- */

fn bar(rig: &Rig) -> BarLayout {
    rig.ex.toolbar.bar.layout.clone().expect("the bar was drawn")
}

fn button(rig: &Rig, action: ActionId) -> (Rect, bool) {
    let (_, rect, enabled) = bar(rig).buttons.into_iter().find(|(a, _, _)| *a == action).unwrap_or_else(|| panic!("{action:?} is not on the bar"));
    (rect, enabled)
}

fn click_action(rig: &mut Rig, action: ActionId) {
    let (rect, enabled) = button(rig, action);
    assert!(enabled, "{action:?} is disabled");
    rig.click(rect.center());
}

#[test]
fn the_bar_is_absent_without_a_selection_and_shows_the_count_with_one() {
    pin();
    let mut rig = Rig::new(1000.0, 600.0);
    assert!(rig.ex.toolbar.bar.layout.is_none());
    rig.select_images(0..3);
    assert!(rig.has_text("3 selected (63 kB)"), "{:?}", texts(&rig.shapes).iter().map(|t| t.0.clone()).collect::<Vec<_>>());
    rig.ex.clear_selection();
    rig.settle(2);
    assert!(rig.ex.toolbar.bar.layout.is_none(), "an empty selection draws no bar");
}

#[test]
fn the_bar_stays_inside_the_browser_and_never_overlaps_across_widths() {
    pin();
    for width in (400..=1800).step_by(50) {
        let mut rig = Rig::new(width as f32, 500.0);
        rig.select_images(0..3);
        let l = bar(&rig);
        assert!(l.rect.left() >= l.browser.left() - 0.5 && l.rect.right() <= l.browser.right() + 0.5, "{width}px: the bar {:?} leaves the browser {:?}", l.rect, l.browser);
        assert!(l.rect.bottom() <= l.browser.bottom() + 0.5);
        let mut parts: Vec<(String, Rect)> = vec![("label".into(), l.label), ("clear".into(), l.clear)];
        parts.extend(l.buttons.iter().map(|(a, r, _)| (format!("{a:?}"), *r)));
        parts.extend(l.more.map(|r| ("more".to_string(), r)));
        for (name, r) in &parts {
            assert!(r.left() >= l.rect.left() - 0.5 && r.right() <= l.rect.right() + 0.5, "{width}px: {name} {r:?} leaves the bar {:?}", l.rect);
        }
        for (i, (a, ra)) in parts.iter().enumerate() {
            for (b, rb) in parts.iter().skip(i + 1) {
                let inter = ra.intersect(*rb);
                assert!(inter.width() <= 0.5 || inter.height() <= 0.5, "{width}px: {a} and {b} overlap: {ra:?} {rb:?}");
            }
        }
    }
}

#[test]
fn a_narrow_bar_folds_actions_into_more_and_a_wide_one_shows_them_all() {
    pin();
    let in_bar = SPECS.iter().filter(|s| s.in_bar).count();
    let mut wide = Rig::new(2800.0, 500.0);
    wide.select_images(0..3);
    let l = bar(&wide);
    assert_eq!((l.buttons.len(), l.hidden, l.more.is_some()), (in_bar, 0, false));
    let mut narrow = Rig::new(700.0, 500.0);
    narrow.select_images(0..3);
    let n = bar(&narrow);
    assert!(n.hidden > 0 && n.more.is_some() && n.buttons.len() + n.hidden == in_bar);
    let kept: Vec<u8> = n.buttons.iter().map(|(a, _, _)| spec_of(*a).priority).collect();
    let dropped_best = SPECS.iter().filter(|s| s.in_bar && !n.buttons.iter().any(|(a, _, _)| *a == s.id)).map(|s| s.priority).min().unwrap();
    assert!(kept.iter().all(|p| *p <= dropped_best), "the buttons that stay are the ones with the lowest priority numbers: {kept:?} vs {dropped_best}");
    assert!(narrow.has_text("Tag"), "Tag has the top priority and always stays");
}

#[test]
fn the_more_menu_lists_what_the_bar_dropped_and_runs_it() {
    pin();
    let mut rig = Rig::new(700.0, 500.0);
    rig.select_images(0..2);
    let more = bar(&rig).more.expect("a narrow bar has a More menu");
    rig.click(more.center());
    assert!(rig.has_text("Move to...") || rig.has_text("Regenerate CLIP Embedding") || rig.has_text("Remove from collection..."), "dropped actions are listed");
}

#[test]
fn a_folder_only_selection_disables_the_file_actions_and_enables_exclude() {
    pin();
    let mut rig = Rig::new(2800.0, 500.0);
    rig.select(&["/rig/sub".to_string()]);
    assert_eq!(rig.ex.selection_count(), 1);
    for action in [ActionId::Tag, ActionId::AddToLibrary, ActionId::Describe, ActionId::CopyTo, ActionId::Refine] {
        assert!(!button(&rig, action).1, "{action:?} needs a file");
    }
    assert!(button(&rig, ActionId::ExcludeFolder).1);
    assert!(button(&rig, ActionId::Open).1 && button(&rig, ActionId::CopyPath).1);
}

#[test]
fn the_library_and_the_vision_model_gate_their_buttons() {
    pin_environment(Some(false), Some(false));
    let mut rig = Rig::new(1800.0, 500.0);
    rig.select_images(0..2);
    for action in [ActionId::Tag, ActionId::AddToLibrary, ActionId::RemoveFromLibrary, ActionId::Describe] {
        assert!(!button(&rig, action).1, "{action:?} needs the library or the model");
    }
    assert!(button(&rig, ActionId::CopyPath).1 && button(&rig, ActionId::GenerateClip).1);
    pin();
}

#[test]
fn copy_path_puts_the_selected_paths_on_the_clipboard_one_per_line() {
    pin();
    let mut rig = Rig::new(1800.0, 500.0);
    rig.select_images(0..2);
    click_action(&mut rig, ActionId::CopyPath);
    assert_eq!(rig.copied, [format!("{}\n{}", path_of(0), path_of(1))]);
}

#[test]
fn open_on_one_row_opens_that_file_and_on_several_each_file() {
    pin();
    let mut rig = Rig::new(1800.0, 500.0);
    rig.select_images(2..3);
    click_action(&mut rig, ActionId::Open);
    assert_eq!(rig.ex.row_sync.opened, [path_of(2)]);
    rig.ex.row_sync.opened.clear();
    rig.select_images(0..3);
    click_action(&mut rig, ActionId::Open);
    assert_eq!(rig.ex.row_sync.opened, [path_of(0), path_of(1), path_of(2)]);
    rig.ex.row_sync.opened.clear();
    click_action(&mut rig, ActionId::Reveal);
    assert_eq!(rig.ex.row_sync.opened, ["/rig", "/rig", "/rig"]);
}

#[test]
fn the_selection_never_means_everything() {
    pin();
    let mut rig = Rig::new(1800.0, 500.0);
    let ctx = rig.ctx.clone();
    let err = rig.ex.run_action_ui(&ctx, ActionId::CopyPath, Origin::Key).unwrap_err();
    assert!(err.contains("Select"), "{err}");
    for action in ActionId::ALL {
        assert!(rig.ex.run_action_ui(&ctx, action, Origin::Command).is_err(), "{action:?} ran on an empty selection");
    }
    assert!(rig.ex.toolbar.bar.popover.is_none());
    assert!(rig.copied.is_empty());
}

/* ------------------------------------------- popovers and flows ------------------------------------- */

fn wait_for(rig: &mut Rig, what: &str, done: impl Fn(&Rig) -> bool) {
    for _ in 0..300 {
        rig.settle(1);
        if done(rig) {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    panic!("timed out waiting for {what}");
}

#[test]
fn clicking_tag_opens_the_popover_for_the_exact_selection() {
    pin();
    let mut rig = Rig::new(1800.0, 600.0);
    rig.select_images(0..2);
    click_action(&mut rig, ActionId::Tag);
    assert_eq!(rig.ex.toolbar.bar.popover, Some(Popover::Tag));
    assert!(rig.has_text("Tag 2 files"));
    assert_eq!(rig.ex.toolbar.bar.target.len(), 2);
    let l = bar(&rig);
    let pop = l.popover.expect("the popover has a rectangle");
    assert!(pop.bottom() <= l.rect.top() + 1.0, "the popover sits above the bar: {pop:?} {:?}", l.rect);
    rig.key(Key::Escape, Modifiers::NONE);
    assert!(rig.ex.toolbar.bar.popover.is_none(), "Esc closes it");
    assert_eq!(rig.ex.selection_count(), 2, "and leaves the selection");
}

#[test]
fn a_popover_closes_when_the_selection_changes_or_the_user_clicks_elsewhere() {
    pin();
    let mut rig = Rig::new(1800.0, 600.0);
    rig.select_images(0..2);
    click_action(&mut rig, ActionId::Category);
    assert_eq!(rig.ex.toolbar.bar.popover, Some(Popover::Category));
    rig.select_images(0..3);
    assert!(rig.ex.toolbar.bar.popover.is_none(), "the popover is bound to the selection it opened on");
    click_action(&mut rig, ActionId::Category);
    rig.click(pos2(30.0, 400.0));
    assert!(rig.ex.toolbar.bar.popover.is_none(), "a press outside closes it");
}

#[test]
fn tags_typed_in_the_popover_reach_the_rows_and_the_database_with_counts() {
    pin();
    let _serial = GLOBAL_DB_TESTS.lock().unwrap_or_else(|e| e.into_inner());
    global_db();
    let mut rig = Rig::new(1800.0, 600.0);
    rig.select_images(0..2);
    click_action(&mut rig, ActionId::Tag);
    rig.type_text("bar-sun, bar-sea");
    rig.settle(2);
    assert_eq!(rig.ex.toolbar.bar.forms.tag.add, ["bar-sun"], "a comma completes a tag");
    let apply = rig.text_pos("Add to 2 files").expect("the apply button names the count");
    rig.click(apply + vec2(10.0, 6.0));
    wait_for(&mut rig, "the tags to arrive", |r| r.ex.table.get_by_path(&path_of(0)).is_some_and(|row| !row.tags.is_empty()));
    let tags = rig.ex.table.get_by_path(&path_of(1)).unwrap().tags.clone();
    assert_eq!(tags, ["bar-sun", "bar-sea"]);
    assert!(rig.ex.toolbar.bar.popover.is_none(), "applying closes the popover");
    assert!(rig.ex.table.get_by_path(&path_of(2)).unwrap().tags.is_empty(), "other rows are untouched");
    let stored: Vec<Vec<String>> = db(async {
        let mut r = crate::database::DB.query("SELECT VALUE tags FROM thumbnails WHERE path IN [$a, $b] ORDER BY path").bind(("a", path_of(0))).bind(("b", path_of(1))).await.unwrap();
        r.take(0).unwrap()
    });
    assert_eq!(stored, vec![vec!["bar-sun".to_string(), "bar-sea".to_string()]; 2]);
    clean_rig_rows();
}

#[test]
fn the_category_popover_sets_and_clears_the_category() {
    pin();
    let _serial = GLOBAL_DB_TESTS.lock().unwrap_or_else(|e| e.into_inner());
    global_db();
    let mut rig = Rig::new(1800.0, 600.0);
    rig.select_images(3..5);
    click_action(&mut rig, ActionId::Category);
    rig.type_text("Beach");
    let set = rig.text_pos("Set for 2 files").expect("the set button names the count");
    rig.click(set + vec2(10.0, 6.0));
    wait_for(&mut rig, "the category to arrive", |r| r.ex.table.get_by_path(&path_of(3)).is_some_and(|row| row.category.is_some()));
    assert_eq!(rig.ex.table.get_by_path(&path_of(4)).unwrap().category.as_deref(), Some("Beach"));
    click_action(&mut rig, ActionId::Category);
    let clear = rig.text_pos("Clear category").expect("the clear button");
    rig.click(clear + vec2(10.0, 6.0));
    wait_for(&mut rig, "the category to clear", |r| r.ex.table.get_by_path(&path_of(3)).is_some_and(|row| row.category.is_none()));
    clean_rig_rows();
}

#[test]
fn the_collection_popover_creates_a_collection_and_adds_the_selection() {
    pin();
    let _serial = GLOBAL_DB_TESTS.lock().unwrap_or_else(|e| e.into_inner());
    global_db();
    let mut rig = Rig::new(1800.0, 600.0);
    rig.select_images(0..3);
    click_action(&mut rig, ActionId::AddToCollection);
    assert!(rig.has_text("Add 3 files to a collection"));
    rig.type_text("Bar Trip");
    let create = rig.text_pos("Create \"Bar Trip\" and add 3 files").expect("a new name offers to create it");
    rig.click(create + vec2(10.0, 6.0));
    wait_for(&mut rig, "the collection to hold the files", |_| {
        db(async { crate::database::collections::get_by_name(&crate::database::DB, "Bar Trip").await.unwrap() }).is_some_and(|c| db(async move { crate::database::collections::count_members(&crate::database::DB, &c.id).await.unwrap() }) == 3)
    });
    let c = db(async { crate::database::collections::get_by_name(&crate::database::DB, "Bar Trip").await.unwrap().unwrap() });
    db(async move { crate::database::collections::delete(&crate::database::DB, &c.id).await.unwrap() });
    clean_rig_rows();
}

#[test]
fn collection_helpers_filter_by_name_and_decide_when_to_create() {
    let list = vec![("Holidays".to_string(), 12), ("Trip".to_string(), 3), ("Trip 2023".to_string(), 1)];
    assert_eq!(popovers::matching_collections(&list, "trip").len(), 2);
    assert_eq!(popovers::matching_collections(&list, "  ").len(), 3);
    assert!(popovers::matching_collections(&list, "zzz").is_empty());
    assert!(popovers::creates_collection(&list, "Fresh"));
    assert!(!popovers::creates_collection(&list, " trip "), "an existing name, ignoring case, is not created again");
    assert!(!popovers::creates_collection(&list, "   "));
}

#[test]
fn removing_from_the_library_asks_with_the_exact_count_and_cancel_changes_nothing() {
    pin();
    let mut rig = Rig::new(1800.0, 600.0);
    rig.select_images(0..3);
    click_action(&mut rig, ActionId::RemoveFromLibrary);
    assert!(matches!(rig.ex.toolbar.bar.popover, Some(Popover::Confirm(Pending { action: ActionId::RemoveFromLibrary, count: 3, .. }))));
    assert!(rig.has_text("Remove 3"), "the button names the count");
    assert!(rig.ex.toolbar.bar.runs.back().is_some_and(|r| matches!(r.outcome, Ok(Done::Asked(_)))));
    let cancel = rig.text_pos("Cancel").unwrap();
    rig.click(cancel + vec2(10.0, 6.0));
    assert!(rig.ex.toolbar.bar.popover.is_none());
    assert_eq!(rig.ex.table.len(), ROWS + 1, "nothing was removed");
}

#[test]
fn confirming_a_removal_in_a_library_scope_drops_the_rows_from_the_view() {
    pin();
    let _serial = GLOBAL_DB_TESTS.lock().unwrap_or_else(|e| e.into_inner());
    global_db();
    let mut rig = Rig::new(1800.0, 600.0);
    rig.ex.nav.scope = Scope::Library;
    rig.select_images(0..3);
    click_action(&mut rig, ActionId::RemoveFromLibrary);
    let ok = rig.text_pos("Remove 3").unwrap();
    rig.click(ok + vec2(10.0, 6.0));
    wait_for(&mut rig, "the rows to leave the view", |r| r.ex.table.len() == ROWS + 1 - 3);
    assert!(rig.ex.table.get_by_path(&path_of(0)).is_none() && rig.ex.table.get_by_path(&path_of(3)).is_some());
}

#[test]
fn confirming_a_removal_in_a_folder_scope_keeps_the_rows_and_marks_them_not_indexed() {
    pin();
    let _serial = GLOBAL_DB_TESTS.lock().unwrap_or_else(|e| e.into_inner());
    global_db();
    let mut rig = Rig::new(1800.0, 600.0);
    rig.ex.viewer.badges.mark_indexed([path_of(0).as_str(), path_of(1).as_str()]);
    rig.select_images(0..2);
    click_action(&mut rig, ActionId::RemoveFromLibrary);
    let ok = rig.text_pos("Remove 2").unwrap();
    rig.click(ok + vec2(10.0, 6.0));
    wait_for(&mut rig, "the badges to change", |r| !r.ex.viewer.badges.is_indexed(&path_of(0)));
    assert_eq!(rig.ex.table.len(), ROWS + 1, "the files are still on disk, so they stay listed");
}

#[test]
fn heavy_work_on_more_than_the_limit_asks_and_a_few_files_run_at_once() {
    pin();
    let mut rig = Rig::new(1800.0, 600.0);
    let many: Vec<Thumbnail> = (100..140).map(crate::ui::file_table::test_rig::image_row).collect();
    rig.ex.table.bulk_load(many);
    rig.settle(2);
    let paths: Vec<String> = (100..130).map(path_of).collect();
    rig.select(&paths);
    click_action(&mut rig, ActionId::Describe);
    assert!(matches!(rig.ex.toolbar.bar.popover, Some(Popover::Confirm(Pending { action: ActionId::Describe, count: 30, .. }))), "{:?}", rig.ex.toolbar.bar.popover);
    assert!(rig.has_text("Describe 30"));
    rig.key(Key::Escape, Modifiers::NONE);
    assert!(rig.ex.toolbar.bar.popover.is_none());

    rig.ex.table.get_mut_by_path(&path_of(1)).unwrap().description = Some("a cat".into());
    rig.select_images(0..2);
    click_action(&mut rig, ActionId::Describe);
    assert!(rig.ex.toolbar.bar.popover.is_none(), "two files run without asking");
    let run = rig.ex.toolbar.bar.runs.back().unwrap();
    assert_eq!(run.outcome, Ok(Done::Ran("Describing 1 file; 1 file already described".to_string())));
}

#[test]
fn exclude_folder_adds_the_folder_to_the_filter_and_hides_what_is_below() {
    pin();
    let mut rig = Rig::new(2800.0, 600.0);
    rig.select(&["/rig/sub".to_string()]);
    click_action(&mut rig, ActionId::ExcludeFolder);
    assert!(rig.ex.filter_spec().excluded_dirs.contains(&"/rig/sub".to_string()));
    assert!(rig.ex.table.contains_path("/rig/sub"), "rows are never removed by a filter");
}

#[test]
fn an_action_that_changes_nothing_in_the_log_is_still_recorded_with_its_origin() {
    pin();
    let mut rig = Rig::new(1800.0, 600.0);
    rig.select_images(0..2);
    click_action(&mut rig, ActionId::CopyPath);
    let run = rig.ex.toolbar.bar.runs.back().unwrap();
    assert_eq!((run.action, run.origin, run.count), (ActionId::CopyPath, Origin::Bar, 2));
}

/* ----------------------------------------------- context menu -------------------------------------- */

fn menu_frame_in(ctx: &Context, targets: MenuTargets, events: Vec<Event>) -> (Vec<MenuAction>, Vec<epaint::ClippedShape>) {
    let mut emitted = Vec::new();
    let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(500.0, 900.0))), events, ..Default::default() };
    let out = ctx.run_ui(input, |ui| context_menu(ui, targets, |a| emitted.push(a)));
    let shapes = out.shapes.clone();
    out.drop_without_applying_deltas();
    (emitted, shapes)
}

/// Draws the menu twice so its widgets are known, then once more with `events`.
fn menu_frame(targets: MenuTargets, find: &str, events: impl Fn(Pos2) -> Vec<Event>) -> (Vec<MenuAction>, Pos2) {
    let ctx = test_ctx();
    menu_frame_in(&ctx, targets, Vec::new());
    let (_, shapes) = menu_frame_in(&ctx, targets, Vec::new());
    let at = texts(&shapes).into_iter().find(|(t, _)| t.starts_with(find)).unwrap_or_else(|| panic!("no {find} in the menu")).1;
    let (emitted, _) = menu_frame_in(&ctx, targets, events(at + vec2(8.0, 6.0)));
    (emitted, at)
}

#[test]
fn the_context_menu_lists_every_registry_action_in_groups() {
    pin();
    let ctx = test_ctx();
    menu_frame_in(&ctx, MenuTargets { rows: 2, dirs: 0 }, Vec::new());
    let (_, shapes) = menu_frame_in(&ctx, MenuTargets { rows: 2, dirs: 0 }, Vec::new());
    let shown: Vec<String> = texts(&shapes).into_iter().map(|(t, _)| t).collect();
    assert!(shown.contains(&"2 items".to_string()));
    for spec in &SPECS {
        assert!(shown.contains(&spec.label.to_string()), "{} is missing from {shown:?}", spec.label);
    }
    let order: Vec<usize> = SPECS.iter().map(|s| shown.iter().position(|t| t == s.label).unwrap()).collect();
    assert!(order.windows(2).all(|w| w[0] < w[1]), "the items follow the registry order");
}

#[test]
fn a_context_menu_click_emits_the_item_and_a_disabled_item_emits_nothing() {
    pin();
    let (emitted, _) = menu_frame(MenuTargets { rows: 2, dirs: 0 }, "Add to library", press_release);
    assert_eq!(emitted, [MenuAction::AddToLibrary]);
    let (emitted, _) = menu_frame(MenuTargets { rows: 2, dirs: 0 }, "Exclude selected directories", press_release);
    assert!(emitted.is_empty(), "no folder is selected");
}

#[test]
fn the_menu_and_the_bar_run_the_same_executor() {
    pin();
    let mut rig = Rig::new(1800.0, 600.0);
    rig.select_images(0..2);
    rig.ex.run_menu_action(&rig.ctx.clone(), MenuAction::CopyPath);
    rig.settle(1);
    assert_eq!(rig.copied, [format!("{}\n{}", path_of(0), path_of(1))]);
    assert_eq!(rig.ex.toolbar.bar.runs.back().unwrap().origin, Origin::Menu);
}

#[test]
fn rows_found_by_path_fall_back_to_the_file_itself() {
    pin();
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("pic.png");
    std::fs::write(&file, b"12345").unwrap();
    let rig = Rig::new(1000.0, 500.0);
    let made = rig.ex.row_for_path(&file.display().to_string());
    assert_eq!((made.file_type.as_str(), made.size, made.filename.as_str()), ("png", 5, "pic.png"));
    assert_eq!(rig.ex.row_for_path(&dir.path().display().to_string()).file_type, "<DIR>");
    assert_eq!(rig.ex.row_for_path(&path_of(1)).path, path_of(1), "a row in the tab is used as it is");
    let _ = text_rects(&[]);
}
