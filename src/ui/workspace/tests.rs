//! Workspace tests over the whole application drawn headless: capture and restore, lazy and offline tabs, the write schedule, the commands and the state dump.

use std::path::Path;
use std::time::Duration;

use std::sync::MutexGuard;

use eframe::egui::vec2;

use super::capture::{WindowInfo, capture};
use super::*;
use crate::command::bus::testing::{serial, take_matching};
use crate::UiSettings;
use crate::command::{AppCommand, Scope};
use crate::settings::workspace::{BrowserState, FloatingWindow, QueryState, SavedNode, SavedTab, SplitDir};
use crate::ui::file_table::rows::{FilterSpec, SortSpec};
use crate::ui::file_table::scope::PendingKind;
use crate::ui::file_table::table::view_mode::{TileFit, ViewMode, ViewPrefs};
use crate::ui::file_table::test_rig::AppRig;
use crate::ui::tabs::{BrowserId, TabKind};
use crate::ui::test_support::{press_release, text_pos, texts};

fn folder(dir: &Path) -> Scope {
    Scope::Folder(dir.display().to_string())
}

fn leaf(tabs: &[SavedTab], active: usize) -> SavedNode {
    SavedNode::Leaf { tabs: tabs.to_vec(), active }
}

fn b(id: u64) -> SavedTab {
    SavedTab::Browser { id }
}

fn startup(tabs: bool, scope: bool) -> Startup {
    Startup { restore_tabs: tabs, restore_last_scope: scope, last_scope: None, tabs: Vec::new() }
}

fn args(list: &[&str]) -> Vec<String> {
    list.iter().map(|a| a.to_string()).collect()
}

/// A workspace of three browsers in a split dock: two folders and the library.
fn sample(a: &Path, b_dir: &Path) -> Workspace {
    let mut main = BrowserState::new(0, &folder(a));
    main.view = ViewPrefs { mode: ViewMode::Details, tile: 200.0, fit: TileFit::Fit, captions: false, badges: true };
    main.sort = SortSpec::single(2, true);
    main.query = QueryState { mode: crate::command::QueryMode::Text, text: "beach".into() };
    main.filter = Some(FilterSpec::from_json(r#"{"tags":["sun"],"query":"beach"}"#).unwrap());
    main.selection = vec![format!("{}/a.jpg", a.display()), format!("{}/b.jpg", a.display())];
    let second = BrowserState::new(1, &folder(b_dir));
    let library = BrowserState::new(2, &Scope::Library);
    let dock = SavedNode::Split {
        dir: SplitDir::Horizontal,
        fraction: 0.4,
        first: Box::new(leaf(&[b(0), b(1), SavedTab::Settings], 1)),
        second: Box::new(SavedNode::Split { dir: SplitDir::Vertical, fraction: 0.7, first: Box::new(leaf(&[b(2)], 0)), second: Box::new(leaf(&[SavedTab::Logs, SavedTab::Refine], 0)) }),
    };
    let mut ws = Workspace { dock: Some(dock), focused: Some(b(1)), browsers: vec![main, second, library], ..Workspace::default() };
    ws.sanitize();
    ws
}

/// The application drawn headless, holding the lock of the tests that use the global command bus.
struct Rig {
    rig: AppRig,
    _guard: MutexGuard<'static, ()>,
}

impl std::ops::Deref for Rig {
    type Target = AppRig;

    fn deref(&self) -> &AppRig {
        &self.rig
    }
}

impl std::ops::DerefMut for Rig {
    fn deref_mut(&mut self) -> &mut AppRig {
        &mut self.rig
    }
}

fn new_rig() -> Rig {
    let guard = serial();
    let _ = take_matching(|_| true);
    Rig { rig: AppRig::new(1000.0, 700.0), _guard: guard }
}

fn texts_of(rig: &Rig) -> String {
    texts(&rig.shapes).into_iter().map(|(t, _)| t).collect::<Vec<_>>().join("\n")
}

/* ------------------------------------------------ launch ------------------------------------------------ */

#[test]
fn the_startup_settings_pick_how_much_is_restored_and_the_command_line_can_veto() {
    assert_eq!(mode_for(&startup(false, false), &[]), Mode::Off);
    assert_eq!(mode_for(&startup(false, true), &[]), Mode::Scope);
    assert_eq!(mode_for(&startup(true, false), &[]), Mode::Full);
    assert_eq!(mode_for(&startup(true, true), &[]), Mode::Full, "the whole workspace includes the focused scope");
    assert_eq!(mode_for(&startup(true, true), &args(&["--scope", "lib:all"])), Mode::Off);
    assert_eq!(mode_for(&startup(true, true), &args(&["--run={\"cmd\":\"x\"}"])), Mode::Off);
    assert_eq!(mode_for(&startup(true, true), &args(&["--devtools", "tasks", "--wizard", "folders", "--dump-state", "/tmp/s.json"])), Mode::Full);
    assert!(explicit_request(&args(&["--scope=fs:/a"])) && !explicit_request(&args(&["--db-info"])));
}

#[test]
fn a_launch_reads_the_file_and_restores_only_when_asked() {
    let d = tempfile::tempdir().unwrap();
    let (a, c) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let path = d.path().join("workspace.json");
    std::fs::write(&path, sample(a.path(), c.path()).to_json()).unwrap();
    let off = Launch::read(&[], &startup(false, false), Some(path.clone()));
    assert!(off.file.is_some() && off.workspace.is_some());
    assert!(!off.restoring() && off.window().is_none());
    let full = Launch::read(&[], &startup(true, false), Some(path.clone()));
    assert!(full.restoring());
    assert_eq!(full.note, LoadNote::Loaded);
    let scope_only = Launch::read(&[], &startup(false, true), Some(path.clone()));
    assert!(scope_only.restoring() && scope_only.window().is_none(), "only a full restore moves the window");
    let explicit = Launch::read(&args(&["--scope", "lib:all"]), &startup(true, true), Some(path));
    assert!(!explicit.restoring(), "what the command line opens wins");
}

#[test]
fn a_launch_that_finds_no_file_uses_the_tabs_older_versions_kept() {
    let d = tempfile::tempdir().unwrap();
    let legacy = Startup { restore_tabs: true, restore_last_scope: false, last_scope: Some("tag:a".into()), tabs: vec!["tag:a".into(), "lib:all".into()] };
    let launch = Launch::read(&[], &legacy, Some(d.path().join("workspace.json")));
    assert_eq!(launch.note, LoadNote::Missing);
    assert!(launch.file.is_none() && launch.restoring());
    assert_eq!(launch.workspace.as_ref().unwrap().browsers.len(), 2);
    assert!(launch.saver().written().is_none(), "the migrated tabs are not on disk yet");
    let none = Launch::read(&[], &startup(true, false), Some(d.path().join("workspace.json")));
    assert!(!none.restoring());
}

#[test]
fn a_broken_or_newer_file_restores_nothing_and_the_launch_goes_on() {
    let d = tempfile::tempdir().unwrap();
    let path = d.path().join("workspace.json");
    std::fs::write(&path, "{ not json").unwrap();
    let corrupt = Launch::read(&[], &startup(true, false), Some(path.clone()));
    assert!(matches!(corrupt.note, LoadNote::Corrupt(_)) && !corrupt.restoring() && corrupt.workspace.is_none());
    std::fs::write(&path, format!("{{\"version\": {}}}", crate::settings::workspace::VERSION + 1)).unwrap();
    let newer = Launch::read(&[], &startup(true, false), Some(path.clone()));
    assert!(matches!(newer.note, LoadNote::TooNew(_)) && !newer.restoring());
    assert!(!newer.saver().is_writable(), "a newer file is never overwritten");
    let legacy = Startup { restore_tabs: true, restore_last_scope: false, last_scope: None, tabs: vec!["lib:all".into()] };
    assert!(Launch::read(&[], &legacy, Some(path)).workspace.is_none(), "an unusable file is not replaced by old tabs");
}

#[test]
fn a_launch_without_a_dock_does_not_hold_back_the_first_scan() {
    let d = tempfile::tempdir().unwrap();
    let path = d.path().join("workspace.json");
    std::fs::write(&path, Workspace::default().to_json()).unwrap();
    assert!(!Launch::read(&[], &startup(true, false), Some(path.clone())).restoring());
    assert!(!Launch::read(&[], &startup(false, true), Some(path)).restoring());
}

#[test]
fn the_saved_window_goes_into_the_viewport() {
    let state = WindowState { size: [1280.0, 800.0], position: Some([50.0, 60.0]), maximized: false };
    let v = with_saved_window(ViewportBuilder::default(), Some(state));
    assert_eq!(v.inner_size, Some(vec2(1280.0, 800.0)));
    assert_eq!(v.position, Some(pos2(50.0, 60.0)));
    assert_eq!(v.maximized, Some(false));
    let max = with_saved_window(ViewportBuilder::default(), Some(WindowState { maximized: true, position: None, ..state }));
    assert_eq!((max.maximized, max.position), (Some(true), None));
    let tiny = with_saved_window(ViewportBuilder::default(), Some(WindowState { size: [10.0, 10.0], position: None, maximized: false }));
    assert_eq!(tiny.inner_size, Some(vec2(crate::ui::titlebar::MIN_WIDTH, crate::ui::titlebar::MIN_HEIGHT)), "never below the minimum");
    let untouched = with_saved_window(ViewportBuilder::default().with_inner_size([1.0, 2.0]), None);
    assert_eq!(untouched.inner_size, Some(vec2(1.0, 2.0)));
}

#[test]
fn a_restored_window_that_does_not_fit_the_monitor_is_corrected() {
    use eframe::egui::{Rect, pos2};
    let info = |w: f32, h: f32, x: f32, y: f32, monitor: Option<[f32; 2]>| WindowInfo { inner: Some(Rect::from_min_size(pos2(x, y), vec2(w, h))), outer: Some(Rect::from_min_size(pos2(x, y), vec2(w, h))), monitor, ..Default::default() };
    assert_eq!(window_correction(&info(1000.0, 700.0, 10.0, 10.0, Some([1920.0, 1080.0]))), None, "a window that fits is left alone");
    let big = window_correction(&info(4000.0, 3000.0, 0.0, 0.0, Some([1920.0, 1080.0]))).unwrap();
    assert_eq!(big.size, [1920.0, 1080.0]);
    let lost = window_correction(&info(1000.0, 700.0, 90_000.0, 10.0, Some([1920.0, 1080.0]))).unwrap();
    assert_eq!(lost.position, Some([2.0 * 1920.0 - crate::settings::workspace::VISIBLE_MARGIN, 10.0]));
    assert_eq!(window_correction(&info(1000.0, 700.0, 90_000.0, 10.0, None)), None, "nothing to compare with before the monitor is known");
    assert_eq!(window_correction(&WindowInfo { maximized: true, ..info(4000.0, 3000.0, 0.0, 0.0, Some([1920.0, 1080.0])) }), None);
    assert_eq!(window_correction(&WindowInfo { inner: None, ..info(1.0, 1.0, 0.0, 0.0, Some([1920.0, 1080.0])) }), None);
}

/* ---------------------------------------- capture and restore ----------------------------------------- */

#[test]
fn a_fresh_application_captures_its_two_default_tabs() {
    let rig = new_rig();
    let ws = capture(&rig.app, None);
    assert_eq!(ws.dock.as_ref().unwrap().tabs(), vec![b(0), SavedTab::Logs]);
    assert_eq!(ws.browsers.len(), 1);
    assert_eq!(ws.browsers[0].id, 0);
    assert!(ws.browsers[0].selection.is_empty());
}

#[test]
fn what_is_restored_is_captured_again_unchanged() {
    let (a, c) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let ws = sample(a.path(), c.path());
    let mut rig = new_rig();
    let applied = restore::apply_full(&mut rig.app, &ws).unwrap();
    assert_eq!((applied.browsers, applied.tabs), (3, 6));
    assert_eq!(applied.focused.as_deref(), Some(format!("fs:{}", c.path().display()).as_str()));
    assert_eq!(capture(&rig.app, None), ws, "layout, order, front tabs, splits, focus and every tab's state survive");
    assert_eq!(rig.app.context.filtered_tabs.len(), 2);
    assert_eq!(rig.app.context.open_tabs.len(), 6);
}

#[test]
fn a_second_restore_of_the_capture_gives_the_same_workspace() {
    let (a, c) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let mut rig = new_rig();
    restore::apply_full(&mut rig.app, &sample(a.path(), c.path())).unwrap();
    let captured = capture(&rig.app, None);
    let reread = Workspace::parse(&captured.to_json()).unwrap();
    rig.app.context.filtered_tabs.clear();
    restore::apply_full(&mut rig.app, &reread).unwrap();
    assert_eq!(capture(&rig.app, None), captured);
    assert_eq!(rig.app.context.filtered_tabs.len(), 2, "the tabs of the first restore were replaced, not added to");
}

#[test]
fn the_main_tab_keeps_its_place_and_the_others_get_new_ids() {
    let (a, c) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let mut rig = new_rig();
    restore::apply_full(&mut rig.app, &sample(a.path(), c.path())).unwrap();
    assert_eq!(rig.app.context.file_explorer.scope(), &folder(a.path()), "the label 0 is the main explorer");
    let ids: Vec<BrowserId> = rig.app.context.filtered_tabs.keys().copied().collect();
    assert_eq!(ids.len(), 2);
    assert!(ids.iter().all(|id| *id != BrowserId::MAIN));
    let focused = rig.app.context.active_tab.unwrap();
    assert_eq!(rig.app.context.explorer_mut(focused.browser_id().unwrap()).unwrap().scope(), &folder(c.path()));
}

#[test]
fn restoring_replaces_the_tabs_that_were_open() {
    let (a, c) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let mut rig = new_rig();
    let old = BrowserId::next();
    let mut ex = crate::ui::file_table::FileExplorer::new(true);
    ex.restore_state(&BrowserState::new(9, &Scope::Library));
    rig.app.context.filtered_tabs.insert(old, ex);
    rig.app.tree.main_surface_mut().split_right(egui_dock::NodeIndex::root(), 0.5, vec![TabKind::browser(old)]);
    rig.app.context.open_tabs.insert(TabKind::browser(old));
    rig.app.tree.add_window(vec![TabKind::Settings]);
    restore::apply_full(&mut rig.app, &sample(a.path(), c.path())).unwrap();
    assert!(!rig.app.context.filtered_tabs.contains_key(&old));
    assert!(rig.app.tree.find_tab(&TabKind::browser(old)).is_none());
    assert_eq!(rig.app.tree.iter_all_tabs().count(), 6, "floating windows are gone too");
}

#[test]
fn floating_windows_come_back_with_their_tabs_place_and_size() {
    let (a, c, f) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let mut ws = sample(a.path(), c.path());
    ws.browsers.push(BrowserState::new(3, &folder(f.path())));
    ws.floating = vec![FloatingWindow { dock: leaf(&[b(3), SavedTab::Devtools], 1), position: Some([60.0, 70.0]), size: Some([480.0, 320.0]) }];
    ws.sanitize();
    let mut rig = new_rig();
    let applied = restore::apply_full(&mut rig.app, &ws).unwrap();
    assert_eq!((applied.browsers, applied.tabs), (4, 8));
    assert_eq!(rig.app.tree.surfaces_count(), 2, "one window next to the main surface");
    assert_eq!(rig.app.context.open_tabs.len(), 8, "the tabs of the window are open too");
    rig.settle(3);
    let captured = capture(&rig.app, None);
    assert_eq!(captured.floating.len(), 1);
    assert_eq!(captured.floating[0].dock, ws.floating[0].dock);
    assert_eq!(captured.browsers.len(), 4);
    let again = Workspace::parse(&captured.to_json()).unwrap();
    assert_eq!(again, captured);
    assert_eq!(captured.floating[0].position.is_some(), captured.floating[0].size.is_some(), "a window has both its place and its size or neither");
}

#[test]
fn a_workspace_of_floating_windows_alone_restores_with_an_empty_main_surface() {
    let a = tempfile::tempdir().unwrap();
    let mut ws = Workspace { browsers: vec![BrowserState::new(0, &folder(a.path()))], floating: vec![FloatingWindow { dock: leaf(&[b(0)], 0), position: None, size: None }], ..Workspace::default() };
    ws.sanitize();
    let mut rig = new_rig();
    let applied = restore::apply_full(&mut rig.app, &ws).unwrap();
    assert_eq!((applied.browsers, applied.tabs), (1, 1));
    assert!(rig.app.tree.find_tab(&TabKind::MAIN).is_some());
    rig.settle(3);
    let captured = capture(&rig.app, None);
    assert!(captured.dock.is_none() && captured.floating.len() == 1);
}

#[test]
fn undocking_a_tab_keeps_it_in_the_saved_workspace() {
    let mut rig = new_rig();
    rig.app.tree.add_window(vec![TabKind::Settings]);
    let ws = capture(&rig.app, None);
    assert_eq!(ws.floating.len(), 1);
    assert_eq!(ws.floating[0].dock.tabs(), vec![SavedTab::Settings]);
    assert!(ws.dock.as_ref().unwrap().tabs().contains(&SavedTab::Logs));
}

#[test]
fn a_workspace_without_tabs_is_refused_and_changes_nothing() {
    let mut rig = new_rig();
    let before = capture(&rig.app, None);
    assert!(restore::apply_full(&mut rig.app, &Workspace::default()).is_err());
    assert_eq!(capture(&rig.app, None), before);
    assert!(restore::apply_scope(&mut rig.app, &Workspace::default()).is_err());
}

#[test]
fn a_workspace_that_leaves_out_the_main_tab_still_leaves_the_main_explorer_usable() {
    let a = tempfile::tempdir().unwrap();
    let mut ws = Workspace { dock: Some(leaf(&[b(5), SavedTab::Logs], 0)), browsers: vec![BrowserState::new(5, &folder(a.path()))], ..Workspace::default() };
    ws.sanitize();
    let mut rig = new_rig();
    restore::apply_full(&mut rig.app, &ws).unwrap();
    assert!(!rig.app.context.open_tabs.contains(&TabKind::MAIN));
    assert!(matches!(rig.app.context.file_explorer.nav.pending.as_ref().map(|p| &p.kind), Some(PendingKind::Lazy)), "it reads its folder when it is shown again");
    assert_eq!(rig.app.context.filtered_tabs.len(), 1);
}

#[test]
fn a_scope_only_restore_shows_the_focused_scope_in_the_main_tab_and_leaves_the_dock() {
    let (a, c) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let ws = sample(a.path(), c.path());
    let mut rig = new_rig();
    let dock_before = capture(&rig.app, None).dock;
    let applied = restore::apply_scope(&mut rig.app, &ws).unwrap();
    assert_eq!(applied.focused, Some(format!("fs:{}", c.path().display())));
    assert_eq!(rig.app.context.file_explorer.scope(), &folder(c.path()));
    assert_eq!(capture(&rig.app, None).dock, dock_before);
    assert!(rig.app.context.filtered_tabs.is_empty());
}

#[test]
fn the_state_of_a_tab_comes_back_through_the_explorer_api() {
    let a = tempfile::tempdir().unwrap();
    let mut state = BrowserState::new(4, &folder(a.path()));
    state.view = ViewPrefs { mode: ViewMode::Details, tile: 88.0, fit: TileFit::Fit, captions: false, badges: false };
    state.sort = SortSpec::from_keys(vec![crate::ui::file_table::rows::SortKey { column: 3, descending: true }, crate::ui::file_table::rows::SortKey { column: 1, descending: false }]);
    state.query = QueryState { mode: crate::command::QueryMode::Name, text: "img".into() };
    state.filter = Some(FilterSpec::from_json(r#"{"skip_icons": true, "tags": ["x", "y"], "tags_all": true, "name": "img"}"#).unwrap().normalized());
    state.selection = vec!["/a/1.jpg".into(), "/a/2.jpg".into()];
    let mut rig = new_rig();
    let mut ex = crate::ui::file_table::FileExplorer::new(true);
    ex.restore_state(&state);
    assert!(ex.nav.pending.is_some() && ex.table.is_empty(), "nothing is read yet");
    assert_eq!(ex.workspace_state(4), Some(state.clone()));
    assert_eq!(ex.scope(), &folder(a.path()));
    assert_eq!(ex.nav.history.len(), 1);
    assert!(!ex.tab_title().is_empty());
    rig.app.context.filtered_tabs.insert(BrowserId::next(), ex);
}

#[test]
fn a_tab_that_shows_rows_held_in_memory_is_not_saved() {
    let mut rig = new_rig();
    rig.app.context.file_explorer.show_rows(Some("Selected".into()), vec![crate::ui::file_table::test_rig::image_row(1)]);
    let ws = capture(&rig.app, None);
    assert!(ws.browsers.is_empty(), "no scope can read those rows again");
    assert!(ws.dock.as_ref().is_none_or(|d| d.tabs() == vec![SavedTab::Logs]));
}

#[test]
fn a_large_selection_is_left_out() {
    let mut rig = new_rig();
    let paths: Vec<String> = (0..crate::ui::file_table::test_rig::ROWS).map(crate::ui::file_table::test_rig::path_of).collect();
    rig.app.context.file_explorer.sync_view();
    rig.app.context.file_explorer.select_paths(paths.iter().map(String::as_str));
    assert_eq!(capture(&rig.app, None).browsers[0].selection.len(), paths.len(), "a small selection is kept");
    let many = crate::settings::workspace::MAX_SELECTION + 5;
    let rows: Vec<_> = (0..many).map(|i| {
        let mut t = crate::database::Thumbnail::new(&format!("m_{i}.jpg"));
        t.path = format!("/many/m_{i}.jpg");
        t.file_type = "jpg".into();
        t.size = 50_000;
        t
    }).collect();
    let ex = &mut rig.app.context.file_explorer;
    ex.table.bulk_load(rows);
    ex.sync_view();
    let all: Vec<String> = (0..many).map(|i| format!("/many/m_{i}.jpg")).collect();
    ex.select_paths(all.iter().map(String::as_str));
    assert!(ex.selection_count() > crate::settings::workspace::MAX_SELECTION);
    assert!(capture(&rig.app, None).browsers[0].selection.is_empty());
}

/* ------------------------------------------- lazy and offline ------------------------------------------ */

#[test]
fn only_the_tabs_that_are_shown_read_their_scope() {
    let (a, c) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let mut rig = new_rig();
    restore::apply_full(&mut rig.app, &sample(a.path(), c.path())).unwrap();
    rig.settle(4);
    let shown = rig.app.context.active_tab.unwrap().browser_id().unwrap();
    assert!(rig.app.context.explorer_mut(shown).unwrap().nav.pending.is_none(), "the tab in front read its folder");
    assert!(matches!(rig.app.context.file_explorer.nav.pending.as_ref().map(|p| &p.kind), Some(PendingKind::Lazy)), "the tab behind it has not been drawn and reads nothing");
    assert!(rig.app.context.file_explorer.table.is_empty());
}

#[test]
fn a_background_tab_reads_its_scope_when_it_is_brought_to_the_front() {
    let (a, c) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    std::fs::write(a.path().join("x.txt"), "x").unwrap();
    let mut rig = new_rig();
    restore::apply_full(&mut rig.app, &sample(a.path(), c.path())).unwrap();
    rig.settle(3);
    let path = rig.app.tree.find_tab(&TabKind::MAIN).unwrap();
    rig.app.tree.set_active_tab(path).unwrap();
    rig.app.tree.set_focused_node_and_surface(path.node_path());
    rig.settle(3);
    let main = &rig.app.context.file_explorer;
    assert!(main.nav.pending.is_none());
    assert!(main.nav.fs_started.is_some() || !main.nav.log.is_empty(), "the folder listing started");
}

#[test]
fn a_folder_that_is_gone_opens_as_a_placeholder_with_a_retry_that_works_once_it_is_back() {
    let base = tempfile::tempdir().unwrap();
    let gone = base.path().join("gone");
    let mut ws = Workspace { dock: Some(leaf(&[b(0), b(1)], 1)), browsers: vec![BrowserState::new(0, &folder(base.path())), BrowserState::new(1, &folder(&gone))], focused: Some(b(1)), ..Workspace::default() };
    ws.sanitize();
    let mut rig = new_rig();
    restore::apply_full(&mut rig.app, &ws).unwrap();
    rig.settle(4);
    let id = rig.app.context.active_tab.unwrap().browser_id().unwrap();
    match rig.app.context.explorer_mut(id).unwrap().nav.pending.as_ref().map(|p| p.kind.clone()) {
        Some(PendingKind::Offline(reason)) => assert!(reason.contains("gone"), "{reason}"),
        other => panic!("{other:?}"),
    }
    let text = texts_of(&rig);
    assert!(text.contains("Not available right now") && text.contains("Retry"), "{text}");
    std::fs::create_dir(&gone).unwrap();
    let at = text_pos(&rig.shapes, "Retry");
    rig.frame(press_release(at + vec2(8.0, 8.0)));
    rig.settle(3);
    assert!(rig.app.context.explorer_mut(id).unwrap().nav.pending.is_none(), "the folder is back and loads");
    assert!(!texts_of(&rig).contains("Not available right now"));
}

#[test]
fn a_retry_while_the_folder_is_still_gone_keeps_the_placeholder_and_says_why() {
    let base = tempfile::tempdir().unwrap();
    let _rt = crate::database::ui_themes::tests::RT.enter();
    let mut ex = crate::ui::file_table::FileExplorer::new(true);
    ex.restore_state(&BrowserState::new(0, &folder(&base.path().join("nope"))));
    ex.retry_pending();
    assert!(matches!(ex.nav.pending.as_ref().map(|p| &p.kind), Some(PendingKind::Offline(r)) if r.contains("nope")));
}

#[test]
fn a_saved_search_is_not_run_until_the_user_asks() {
    let mut ws = Workspace { dock: Some(leaf(&[b(0), b(1)], 1)), browsers: vec![BrowserState::new(0, &Scope::Library), BrowserState::new(1, &Scope::Semantic("a red car".into()))], focused: Some(b(1)), ..Workspace::default() };
    ws.sanitize();
    let mut rig = new_rig();
    restore::apply_full(&mut rig.app, &ws).unwrap();
    rig.settle(4);
    let id = rig.app.context.active_tab.unwrap().browser_id().unwrap();
    assert!(matches!(rig.app.context.explorer_mut(id).unwrap().nav.pending.as_ref().map(|p| &p.kind), Some(PendingKind::Search)));
    let text = texts_of(&rig);
    assert!(text.contains("Search not run") && text.contains("Run search"), "{text}");
    assert!(rig.app.context.explorer_mut(id).unwrap().nav.active.is_none(), "no engine was asked");
}

#[test]
fn any_navigation_replaces_the_placeholder() {
    let base = tempfile::tempdir().unwrap();
    let _rt = crate::database::ui_themes::tests::RT.enter();
    let mut ex = crate::ui::file_table::FileExplorer::new(true);
    ex.restore_state(&BrowserState::new(0, &folder(&base.path().join("gone"))));
    assert!(ex.nav.pending.is_some());
    ex.open_scope(folder(base.path())).unwrap();
    assert!(ex.nav.pending.is_none());
}

#[test]
fn the_selection_of_a_restored_tab_is_made_once_its_rows_are_in() {
    let _rt = crate::database::ui_themes::tests::RT.enter();
    let mut rig = crate::ui::file_table::test_rig::Rig::new(900.0, 600.0);
    let (first, rest) = (crate::ui::file_table::test_rig::path_of(2), vec![crate::ui::file_table::test_rig::path_of(4), crate::ui::file_table::test_rig::path_of(5)]);
    rig.ex.nav.restore = Some(crate::ui::file_table::scope::Restore { scroll_row: None, anchor: Some(first) });
    rig.ex.nav.restore_select = rest;
    rig.settle(3);
    assert_eq!(rig.ex.selection_count(), 3);
    assert!(rig.ex.nav.restore_select.is_empty() && rig.ex.nav.restore.is_none());
}

#[test]
fn a_selection_that_is_not_in_the_rows_is_dropped_without_a_trace() {
    let _rt = crate::database::ui_themes::tests::RT.enter();
    let mut rig = crate::ui::file_table::test_rig::Rig::new(900.0, 600.0);
    rig.ex.nav.restore = Some(crate::ui::file_table::scope::Restore { scroll_row: None, anchor: Some("/rig/missing.jpg".into()) });
    rig.ex.nav.restore_select = vec![crate::ui::file_table::test_rig::path_of(4)];
    rig.settle(3);
    assert_eq!(rig.ex.selection_count(), 0);
    assert!(rig.ex.nav.restore_select.is_empty());
}

#[test]
fn starting_a_restored_scope_hands_the_selection_to_the_restore_step() {
    let dir = tempfile::tempdir().unwrap();
    let _rt = crate::database::ui_themes::tests::RT.enter();
    let mut state = BrowserState::new(0, &folder(dir.path()));
    state.selection = vec!["/p/a.jpg".into(), "/p/b.jpg".into(), "/p/c.jpg".into()];
    let mut ex = crate::ui::file_table::FileExplorer::new(true);
    ex.restore_state(&state);
    ex.retry_pending();
    assert!(ex.nav.pending.is_none(), "the folder exists, so it was started");
    assert_eq!(ex.nav.restore_select, vec!["/p/b.jpg".to_string(), "/p/c.jpg".to_string()]);
    assert_eq!(ex.nav.restore.as_ref().and_then(|r| r.anchor.clone()), Some("/p/a.jpg".to_string()));
}

/* ----------------------------------------- the write schedule ------------------------------------------ */

fn file_backed(rig: &mut Rig, dir: &Path, delay: Duration) -> std::path::PathBuf {
    let path = dir.join("workspace.json");
    let (saver, _) = Saver::at(path.clone());
    rig.app.context.workspace.saver = saver.with_delay(delay);
    path
}

#[test]
fn a_change_is_written_by_the_tick_and_an_unchanged_state_is_not_written_again() {
    let d = tempfile::tempdir().unwrap();
    let mut rig = new_rig();
    let path = file_backed(&mut rig, d.path(), Duration::ZERO);
    let ctx = rig.ctx.clone();
    rig.app.workspace_tick(&ctx);
    assert_eq!(rig.app.context.workspace.saver.saves, 1);
    assert_eq!(load_from(&path).0, Some(capture(&rig.app, None)));
    rig.app.workspace_tick(&ctx);
    rig.app.workspace_tick(&ctx);
    assert_eq!(rig.app.context.workspace.saver.saves, 1, "nothing changed");
    rig.app.context.file_explorer.toolbar.query.text = "beach".into();
    rig.app.workspace_tick(&ctx);
    assert_eq!(rig.app.context.workspace.saver.saves, 2);
    assert_eq!(load_from(&path).0.unwrap().browsers[0].query.text, "beach");
}

#[test]
fn a_change_waits_for_its_quiet_time_and_the_exit_writes_it_at_once() {
    let d = tempfile::tempdir().unwrap();
    let mut rig = new_rig();
    let path = file_backed(&mut rig, d.path(), Duration::from_secs(3600));
    let ctx = rig.ctx.clone();
    rig.app.workspace_tick(&ctx);
    assert!(rig.app.context.workspace.saver.is_pending());
    assert!(!path.exists(), "the quiet time has not passed");
    rig.app.workspace_exit(&ctx);
    assert!(path.exists() && !rig.app.context.workspace.saver.is_pending());
    assert_eq!(rig.app.context.workspace.saver.saves, 1);
}

#[test]
fn a_restored_workspace_is_not_rewritten_by_the_first_frames() {
    let (a, c, d) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let path = d.path().join("workspace.json");
    std::fs::write(&path, sample(a.path(), c.path()).to_json()).unwrap();
    let launch = Launch::read(&[], &startup(true, false), Some(path.clone()));
    let mut rig = new_rig();
    let applied = restore_launch(&mut rig.app, &launch);
    rig.app.context.workspace = Session::live(&launch, applied.as_ref());
    assert!(rig.app.context.workspace.restored.as_deref().is_some_and(|t| t.contains("3 browsers")));
    let ctx = rig.ctx.clone();
    rig.app.workspace_tick(&ctx);
    assert!(!rig.app.context.workspace.saver.is_pending(), "the state is what the file holds, ids and all");
    assert_eq!(rig.app.context.workspace.saver.saves, 0);
}

#[test]
fn the_launch_restore_does_nothing_when_the_setting_is_off() {
    let (a, c, d) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let path = d.path().join("workspace.json");
    std::fs::write(&path, sample(a.path(), c.path()).to_json()).unwrap();
    let launch = Launch::read(&[], &startup(false, false), Some(path));
    let mut rig = new_rig();
    let before = capture(&rig.app, None);
    assert!(restore_launch(&mut rig.app, &launch).is_none());
    assert_eq!(capture(&rig.app, None), before);
    assert_eq!(Session::live(&launch, None).restored, None);
}

#[test]
fn a_scope_only_launch_changes_only_the_main_tab() {
    let (a, c, d) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let path = d.path().join("workspace.json");
    std::fs::write(&path, sample(a.path(), c.path()).to_json()).unwrap();
    let launch = Launch::read(&[], &startup(false, true), Some(path));
    let mut rig = new_rig();
    let applied = restore_launch(&mut rig.app, &launch).unwrap();
    assert_eq!(applied.tabs, 0);
    assert_eq!(rig.app.context.file_explorer.scope(), &folder(c.path()));
    assert!(rig.app.context.filtered_tabs.is_empty());
}

/* ------------------------------------------------ commands ------------------------------------------- */

#[test]
fn the_workspace_commands_save_reload_and_forget_the_file() {
    let (a, c, d) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let mut rig = new_rig();
    let path = file_backed(&mut rig, d.path(), Duration::from_secs(3600));
    assert!(rig.app.run_workspace_op(Op::Restore).is_err(), "there is nothing to read yet");
    restore::apply_full(&mut rig.app, &sample(a.path(), c.path())).unwrap();
    let saved = rig.app.run_workspace_op(Op::Save).unwrap();
    assert!(saved.contains("workspace.json"), "{saved}");
    let on_disk = load_from(&path).0.unwrap();
    assert_eq!(on_disk.browsers.len(), 3);
    rig.app.context.filtered_tabs.clear();
    rig.app.tree.main_surface_mut().retain_tabs(|t| t.browser_id().is_none());
    let done = rig.app.run_workspace_op(Op::Restore).unwrap();
    assert!(done.contains("3 browsers"), "{done}");
    assert_eq!(rig.app.context.filtered_tabs.len(), 2);
    assert_eq!(capture(&rig.app, None), on_disk);
    rig.app.run_workspace_op(Op::Forget).unwrap();
    assert!(!path.exists());
    assert!(rig.app.run_workspace_op(Op::Restore).is_err());
}

#[test]
fn the_workspace_command_goes_through_the_command_bus() {
    let d = tempfile::tempdir().unwrap();
    let mut rig = new_rig();
    let path = file_backed(&mut rig, d.path(), Duration::from_secs(3600));
    crate::command::send(AppCommand::Workspace { op: Op::Save });
    rig.settle(1);
    assert!(path.exists());
    let json = r#"{"cmd":"workspace","op":"forget"}"#;
    assert_eq!(crate::command::send_text(json), Ok(1));
    rig.settle(1);
    assert!(!path.exists());
}

#[test]
fn saving_a_memory_only_workspace_is_a_failed_command() {
    let mut rig = new_rig();
    let err = rig.app.run_workspace_op(Op::Save).unwrap_err();
    assert_eq!(err.kind, crate::command::ErrorKind::Failed);
    assert!(rig.app.run_workspace_op(Op::Forget).is_err());
}

#[test]
fn saving_over_a_newer_file_is_refused_and_leaves_it() {
    let d = tempfile::tempdir().unwrap();
    let path = d.path().join("workspace.json");
    let newer = format!("{{\"version\": {}}}", crate::settings::workspace::VERSION + 1);
    std::fs::write(&path, &newer).unwrap();
    let mut rig = new_rig();
    let (saver, _) = Saver::at(path.clone());
    rig.app.context.workspace.saver = saver;
    assert!(rig.app.run_workspace_op(Op::Save).is_err());
    assert!(rig.app.run_workspace_op(Op::Forget).is_err());
    let ctx = rig.ctx.clone();
    rig.app.workspace_exit(&ctx);
    assert_eq!(std::fs::read_to_string(&path).unwrap(), newer);
}

/* ------------------------------------------------ state dump ----------------------------------------- */

fn read_json(path: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn a_dump_lists_the_tabs_with_their_scopes_counts_and_view() {
    let d = tempfile::tempdir().unwrap();
    let (a, c) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let mut rig = new_rig();
    restore::apply_full(&mut rig.app, &sample(a.path(), c.path())).unwrap();
    rig.settle(3);
    let path = d.path().join("out").join("state.json");
    let done = rig.app.request_dump(path.display().to_string(), false).unwrap();
    assert!(done.contains("state.json"), "{done}");
    let dump = read_json(&path);
    assert_eq!(dump["version"], 1);
    assert_eq!(dump["build"]["name"], "smart-media");
    let tabs = dump["tabs"].as_array().unwrap();
    assert_eq!(tabs.len(), 3);
    let main = tabs.iter().find(|t| t["scope"] == format!("fs:{}", a.path().display())).unwrap();
    assert_eq!(main["view"], "details");
    assert_eq!(main["query"]["mode"], "text");
    assert_eq!(main["query"]["text"], "beach");
    assert_eq!(main["selected"], 0);
    assert_eq!(main["pending"], "lazy", "the tab behind has not read its folder");
    assert!(main["filter"].as_str().unwrap().contains("\"sun\""));
    assert_eq!(main["filters_active"].as_u64().unwrap() >= 1, true);
    assert_eq!(main["sort"][0]["descending"], true);
    let shown = tabs.iter().find(|t| t["active"] == true).unwrap();
    assert_eq!(shown["scope"], format!("fs:{}", c.path().display()));
    assert_eq!(dump["active_tab"], shown["scope"]);
    let others: Vec<&str> = dump["other_tabs"].as_array().unwrap().iter().filter_map(|v| v.as_str()).collect();
    assert!(others.contains(&"Settings") && others.contains(&"Logs") && others.contains(&"AI Refinements"), "{others:?}");
    assert!(dump["engines"].is_array() && dump["tasks"].is_array());
    assert!(dump["settings"]["loaded"].is_boolean());
}

#[test]
fn a_dump_counts_the_rows_and_the_selection_of_the_tab() {
    let d = tempfile::tempdir().unwrap();
    let mut rig = new_rig();
    rig.settle(2);
    let paths: Vec<String> = (0..3).map(crate::ui::file_table::test_rig::path_of).collect();
    rig.app.context.file_explorer.sync_view();
    rig.app.context.file_explorer.select_paths(paths.iter().map(String::as_str));
    let path = d.path().join("state.json");
    rig.app.request_dump(path.display().to_string(), false).unwrap();
    let tab = &read_json(&path)["tabs"][0];
    assert_eq!(tab["rows"], 13);
    assert!((12..=13).contains(&tab["filtered"].as_u64().unwrap()), "{tab}");
    assert_eq!(tab["selected"], 3);
    assert_eq!(tab["loading"], false);
}

#[test]
fn a_dump_with_keys_in_the_settings_never_contains_them() {
    let (a, c) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let mut rig = new_rig();
    restore::apply_full(&mut rig.app, &sample(a.path(), c.path())).unwrap();
    rig.app.context.file_explorer.toolbar.query.text = "my key sk-ABCDEFGHIJKLMNOPQRSTUV".into();
    let mut settings = UiSettings::default();
    settings.openai_api_key = Some("sk-FAKEOPENAIKEY1234567890abcdef".into());
    settings.gemini_api_key = Some("FAKEGEMINISECRET99999".into());
    settings.openai_organization = Some("org-SECRETORG777".into());
    settings.openai_base_url = Some("https://bob:swordfish@llm.example/v1?key=URLKEY1234".into());
    settings.ai_chat_provider = Some("openai".into());
    let report = super::dump::build(&rig.app, 3, &settings, true, &ui_prefs::UiPrefs::default());
    let text = serde_json::to_string_pretty(&report).unwrap();
    for secret in ["sk-FAKEOPENAIKEY1234567890abcdef", "FAKEGEMINISECRET99999", "org-SECRETORG777", "swordfish", "URLKEY1234", "sk-ABCDEFGHIJKLMNOPQRSTUV"] {
        assert!(!text.contains(secret), "{secret} leaked:\n{text}");
    }
    assert!(text.contains("\"providers_with_key\"") && text.contains("openai") && text.contains("gemini"));
    assert!(text.contains("https://llm.example/v1"));
    for field in ["api_key", "apikey", "password", "token", "secret", "organization"] {
        assert!(!text.to_lowercase().contains(&format!("\"{field}")), "a field named like {field} is in the dump");
    }
}

#[test]
fn a_dump_asked_for_at_start_up_waits_for_the_first_frames_and_then_writes_once() {
    let d = tempfile::tempdir().unwrap();
    let mut rig = new_rig();
    let path = d.path().join("state.json");
    let queued = rig.app.request_dump(path.display().to_string(), true).unwrap();
    assert!(queued.contains("idle"), "{queued}");
    let ctx = rig.ctx.clone();
    for _ in 0..4 {
        rig.app.workspace_tick(&ctx);
    }
    assert!(!path.exists(), "the first frames have not run");
    for _ in 0..12 {
        rig.app.workspace_tick(&ctx);
    }
    assert!(path.exists() && rig.app.context.workspace.dump.is_none());
    let first = std::fs::read_to_string(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    for _ in 0..12 {
        rig.app.workspace_tick(&ctx);
    }
    assert!(!path.exists(), "it is written once");
    assert!(first.contains("\"version\""));
}

#[test]
fn a_dump_waits_while_a_tab_is_still_reading() {
    let d = tempfile::tempdir().unwrap();
    let mut rig = new_rig();
    let mut state = BrowserState::new(0, &Scope::Library);
    state.selection = Vec::new();
    rig.app.context.file_explorer.restore_state(&state);
    if let Some(p) = rig.app.context.file_explorer.nav.pending.as_mut() {
        p.kind = PendingKind::Waiting;
    }
    assert!(super::dump::is_loading(&rig.app));
    let path = d.path().join("state.json");
    rig.app.request_dump(path.display().to_string(), true).unwrap();
    let ctx = rig.ctx.clone();
    for _ in 0..40 {
        rig.app.workspace_tick(&ctx);
    }
    assert!(!path.exists(), "the library has not connected, so the application is not idle yet");
}

#[test]
fn a_tab_behind_another_does_not_hold_the_dump_back() {
    let (a, c) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let mut rig = new_rig();
    restore::apply_full(&mut rig.app, &sample(a.path(), c.path())).unwrap();
    if let Some(p) = rig.app.context.file_explorer.nav.pending.as_mut() {
        p.kind = PendingKind::Waiting;
    }
    assert!(!super::dump::is_loading(&rig.app), "the main tab is behind the front tab of its leaf and is not drawn");
    let path = rig.app.tree.find_tab(&TabKind::MAIN).unwrap();
    rig.app.tree.set_active_tab(path).unwrap();
    assert!(super::dump::is_loading(&rig.app), "in front it is waiting for the library");
}

#[test]
fn a_dump_to_a_path_that_cannot_be_written_is_a_failed_command() {
    let d = tempfile::tempdir().unwrap();
    let blocker = d.path().join("file");
    std::fs::write(&blocker, "x").unwrap();
    let mut rig = new_rig();
    let err = rig.app.request_dump(blocker.join("state.json").display().to_string(), false).unwrap_err();
    assert_eq!(err.kind, crate::command::ErrorKind::Failed);
    assert!(matches!(rig.app.context.workspace.last_dump, Some(Err(_))));
}

#[test]
fn the_dump_command_goes_through_the_command_bus() {
    let d = tempfile::tempdir().unwrap();
    let mut rig = new_rig();
    let path = d.path().join("bus.json");
    crate::command::send(AppCommand::DumpState { path: path.display().to_string(), when_idle: false });
    rig.settle(1);
    assert!(path.exists());
}
