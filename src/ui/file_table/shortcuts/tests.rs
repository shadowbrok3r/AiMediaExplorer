use eframe::egui::*;

use crate::command::{ActionId, AppCommand, QueryMode, Selector, dispatch};
use crate::ui::file_table::action_bar::{Popover, pin_environment};
use crate::ui::file_table::query::text_id;
use crate::ui::file_table::test_rig::{AppRig, Rig, ROWS, path_of};

fn pin() {
    pin_environment(Some(true), Some(true));
}

/// Moves the pointer over the tab, as a user working in it does.
fn hover(rig: &mut Rig) {
    rig.frame(vec![Event::PointerMoved(pos2(400.0, 300.0))]);
    rig.settle(2);
}

#[test]
fn t_c_d_and_delete_act_on_the_selection_while_the_pointer_is_over_the_tab() {
    pin();
    let mut rig = Rig::new(1400.0, 600.0);
    rig.select_images(0..2);
    hover(&mut rig);
    rig.key(Key::T, Modifiers::NONE);
    assert_eq!(rig.ex.toolbar.bar.popover, Some(Popover::Tag));
    rig.key(Key::Escape, Modifiers::NONE);
    assert!(rig.ex.toolbar.bar.popover.is_none());
    rig.key(Key::C, Modifiers::NONE);
    assert_eq!(rig.ex.toolbar.bar.popover, Some(Popover::AddToCollection));
    rig.key(Key::Escape, Modifiers::NONE);
    rig.key(Key::Delete, Modifiers::NONE);
    assert!(matches!(rig.ex.toolbar.bar.popover, Some(Popover::Confirm(ref p)) if p.action == ActionId::RemoveFromLibrary && p.count == 2), "Delete asks first with the count");
    rig.key(Key::Escape, Modifiers::NONE);
    rig.key(Key::D, Modifiers::NONE);
    let run = rig.ex.toolbar.bar.runs.back().unwrap();
    assert_eq!((run.action, run.origin), (ActionId::Describe, crate::ui::file_table::action_bar::Origin::Key));
    assert_eq!(rig.ex.selection_count(), 2, "the keys leave the selection alone");
}

#[test]
fn the_keys_do_nothing_without_a_selection_or_when_the_browser_does_not_have_the_keyboard() {
    pin();
    let mut rig = Rig::new(1400.0, 600.0);
    hover(&mut rig);
    for key in [Key::T, Key::C, Key::D, Key::Delete] {
        rig.key(key, Modifiers::NONE);
    }
    assert!(rig.ex.toolbar.bar.popover.is_none() && rig.ex.toolbar.bar.runs.is_empty(), "an empty selection never means everything");
    let mut away = Rig::new(1400.0, 600.0);
    away.select_images(0..2);
    away.frame(vec![Event::PointerMoved(pos2(-50.0, -50.0))]);
    away.key(Key::T, Modifiers::NONE);
    assert!(away.ex.toolbar.bar.popover.is_none(), "the pointer is not over the tab and nothing was clicked in it");
}

#[test]
fn typing_in_a_field_never_fires_a_key() {
    pin();
    let mut rig = Rig::new(1400.0, 600.0);
    rig.select_images(0..2);
    hover(&mut rig);
    rig.ex.toolbar.query.focus = true;
    rig.settle(3);
    rig.type_text("t");
    rig.type_text("d");
    assert_eq!(rig.ex.toolbar.query.text, "td");
    assert!(rig.ex.toolbar.bar.popover.is_none() && rig.ex.toolbar.bar.runs.is_empty());
    let id = text_id(&rig.ex);
    assert!(rig.ctx.memory(|m| m.has_focus(id)));
}

#[test]
fn modifier_chords_are_not_the_plain_keys() {
    pin();
    let mut rig = Rig::new(1400.0, 600.0);
    rig.select_images(0..2);
    hover(&mut rig);
    rig.key(Key::T, Modifiers::COMMAND);
    rig.key(Key::D, Modifiers::COMMAND);
    rig.key(Key::C, Modifiers::COMMAND);
    assert!(rig.ex.toolbar.bar.popover.is_none() && rig.ex.toolbar.bar.runs.is_empty());
}

#[test]
fn escape_clears_the_selection_first_and_then_the_query() {
    pin();
    let mut rig = Rig::new(1400.0, 600.0);
    rig.ex.apply_query_command(QueryMode::Name, "img_0", false).unwrap();
    rig.select_images(1..3);
    hover(&mut rig);
    assert_eq!(rig.ex.selection_count(), 2);
    rig.key(Key::Escape, Modifiers::NONE);
    assert_eq!((rig.ex.selection_count(), rig.ex.toolbar.query.text.as_str()), (0, "img_0"), "the first Esc clears the selection");
    rig.key(Key::Escape, Modifiers::NONE);
    assert_eq!(rig.ex.toolbar.query.text, "");
    assert_eq!(rig.ex.filter_spec().name, "", "and the second Esc the query with its filter");
    assert_eq!(rig.ex.view.len(), ROWS + 1);
}

#[test]
fn the_keys_wait_while_a_dialog_a_popover_or_the_address_field_is_open() {
    pin();
    let mut rig = Rig::new(1400.0, 600.0);
    rig.select_images(0..2);
    hover(&mut rig);
    rig.ex.toolbar.scan.open = true;
    rig.settle(2);
    rig.key(Key::T, Modifiers::NONE);
    assert!(rig.ex.toolbar.bar.popover.is_none(), "the scan dialog has the keyboard");
    rig.ex.toolbar.scan.open = false;
    rig.ex.edit_address();
    rig.settle(2);
    rig.key(Key::T, Modifiers::NONE);
    assert!(rig.ex.toolbar.bar.popover.is_none(), "the address field has the keyboard");
}

/* ------------------------------------------ keys from anywhere ---------------------------------------- */

#[test]
fn ctrl_f_focuses_the_query_ctrl_l_the_address_and_f5_and_ctrl_r_refresh() {
    let _serial = crate::command::bus::testing::serial();
    let mut rig = AppRig::new(1400.0, 700.0);
    rig.key(Key::F, Modifiers::COMMAND);
    let id = text_id(rig.ex());
    assert!(rig.ctx.memory(|m| m.has_focus(id)), "Ctrl+F puts the cursor in the query field");
    rig.key(Key::Escape, Modifiers::NONE);
    assert!(!rig.ctx.memory(|m| m.has_focus(id)));
    rig.key(Key::L, Modifiers::COMMAND);
    assert!(rig.ex().toolbar.path.editing, "Ctrl+L edits the address");
    rig.key(Key::Escape, Modifiers::NONE);
    assert!(!rig.ex().toolbar.path.editing);
    let before = crate::command::with_log(|log| log.entries().filter(|e| e.json.contains("\"op\":\"refresh\"")).count());
    rig.key(Key::F5, Modifiers::NONE);
    rig.key(Key::R, Modifiers::COMMAND);
    let after = crate::command::with_log(|log| log.entries().filter(|e| e.json.contains("\"op\":\"refresh\"")).count());
    assert_eq!(after - before, 2, "F5 and Ctrl+R each read the scope again");
}

#[test]
fn the_global_keys_wait_for_the_palette() {
    let _serial = crate::command::bus::testing::serial();
    let mut rig = AppRig::new(1400.0, 700.0);
    rig.app.shell.palette.open_with(None);
    rig.settle(2);
    rig.key(Key::L, Modifiers::COMMAND);
    assert!(!rig.ex().toolbar.path.editing, "Ctrl+L belongs to the palette while it is open");
}

/* --------------------------------------------- command round trips ----------------------------------- */

#[test]
fn the_run_command_reaches_the_executor_and_opens_the_popovers_of_the_focused_tab() {
    pin();
    let _serial = crate::command::bus::testing::serial();
    let mut rig = AppRig::new(1400.0, 700.0);
    let paths: Vec<String> = (0..3).map(path_of).collect();
    dispatch(&mut rig.app, AppCommand::Run { action: ActionId::Tag, on: Selector::Paths(paths.clone()) }).unwrap();
    assert_eq!(rig.ex().toolbar.bar.popover, Some(Popover::Tag));
    assert_eq!(rig.ex().toolbar.bar.target.len(), 3);
    rig.ex().close_popover();
    let text = dispatch(&mut rig.app, AppCommand::Run { action: ActionId::RemoveFromLibrary, on: Selector::Paths(paths) }).unwrap();
    assert!(text.contains("asked to confirm") && text.contains("3 files"), "{text}");
    assert!(matches!(rig.ex().toolbar.bar.popover, Some(Popover::Confirm(ref p)) if p.count == 3), "a command never removes without the user's answer");
}

#[test]
fn run_on_the_selection_uses_exactly_the_selected_rows_and_an_empty_selection_is_an_error() {
    pin();
    let _serial = crate::command::bus::testing::serial();
    let mut rig = AppRig::new(1400.0, 700.0);
    assert!(dispatch(&mut rig.app, AppCommand::Run { action: ActionId::Tag, on: Selector::Selection }).is_err());
    rig.ex().sync_view();
    rig.ex().select_paths([path_of(4).as_str(), path_of(5).as_str()]);
    dispatch(&mut rig.app, AppCommand::Run { action: ActionId::Category, on: Selector::Selection }).unwrap();
    let target: Vec<String> = rig.ex().toolbar.bar.target.iter().map(|r| r.path.clone()).collect();
    assert_eq!(target, [path_of(4), path_of(5)]);
}

#[test]
fn run_refuses_what_cannot_run_and_says_why() {
    pin();
    let _serial = crate::command::bus::testing::serial();
    let mut rig = AppRig::new(1400.0, 700.0);
    let err = dispatch(&mut rig.app, AppCommand::Run { action: ActionId::ExcludeFolder, on: Selector::Paths(vec![path_of(1)]) }).unwrap_err();
    assert_eq!(err.kind, crate::command::ErrorKind::Invalid);
    assert!(err.message.contains("Select a folder first"), "{}", err.message);
    let err = dispatch(&mut rig.app, AppCommand::Run { action: ActionId::Describe, on: Selector::Paths(vec!["/p/clip.mp4".into()]) }).unwrap_err();
    assert!(err.message.contains("image"), "{}", err.message);
}

#[test]
fn copy_path_and_exclude_folder_run_straight_from_a_command() {
    pin();
    let _serial = crate::command::bus::testing::serial();
    let mut rig = AppRig::new(1400.0, 700.0);
    let text = dispatch(&mut rig.app, AppCommand::Run { action: ActionId::CopyPath, on: Selector::Paths(vec![path_of(0), path_of(1)]) }).unwrap();
    assert_eq!(text, "copied 2 paths");
    let text = dispatch(&mut rig.app, AppCommand::Run { action: ActionId::ExcludeFolder, on: Selector::Paths(vec!["/rig/sub".into()]) }).unwrap();
    assert_eq!(text, "excluded 1 folder from scans");
    assert!(rig.ex().filter_spec().excluded_dirs.contains(&"/rig/sub".to_string()));
}

#[test]
fn the_query_command_drives_the_focused_tabs_bar() {
    let _serial = crate::command::bus::testing::serial();
    let mut rig = AppRig::new(1400.0, 700.0);
    dispatch(&mut rig.app, AppCommand::Query { mode: QueryMode::Name, text: "img_05".into(), run: false }).unwrap();
    rig.settle(2);
    assert_eq!(rig.ex().view.len(), 1);
    assert_eq!(rig.ex().toolbar.query.text, "img_05");
    dispatch(&mut rig.app, AppCommand::Query { mode: QueryMode::Semantic, text: "a red car".into(), run: true }).unwrap();
    assert_eq!(rig.ex().scope(), &crate::command::Scope::Semantic("a red car".into()));
    let err = dispatch(&mut rig.app, AppCommand::Query { mode: QueryMode::Semantic, text: "  ".into(), run: true }).unwrap_err();
    assert_eq!(err.kind, crate::command::ErrorKind::Invalid);
}

#[test]
fn the_dialog_command_opens_the_scan_and_process_dialogs_of_the_focused_tab() {
    let _serial = crate::command::bus::testing::serial();
    let dir = tempfile::tempdir().unwrap();
    let mut rig = AppRig::new(1400.0, 700.0);
    let folder = dir.path().display().to_string();
    dispatch(&mut rig.app, AppCommand::Dialog { dialog: crate::command::DialogKind::Scan, folder: Some(folder.clone()) }).unwrap();
    assert!(rig.ex().toolbar.scan.open);
    assert_eq!(rig.ex().toolbar.scan.folder, folder);
    rig.settle(2);
    assert!(texts_contain(&rig, "Scan folder"));
    dispatch(&mut rig.app, AppCommand::Dialog { dialog: crate::command::DialogKind::Process, folder: Some(folder.clone()) }).unwrap();
    assert!(rig.ex().toolbar.process.open);
    assert_eq!(rig.ex().toolbar.process.scope, crate::ui::file_table::dialogs::process::ProcessScope::Folder);
}

fn texts_contain(rig: &AppRig, text: &str) -> bool {
    crate::ui::test_support::texts(&rig.shapes).iter().any(|(t, _)| t == text)
}

#[test]
fn the_tools_menu_scan_entry_and_the_palette_open_the_scan_dialog() {
    let _serial = crate::command::bus::testing::serial();
    let mut rig = AppRig::new(1400.0, 700.0);
    let ctx = rig.ctx.clone();
    rig.app.run_shell_action(&ctx, crate::ui::shell::ShellAction::ScanFolder);
    rig.settle(3);
    assert!(rig.ex().toolbar.scan.open, "Scan folder opens the dialog instead of a bare folder picker");
}
