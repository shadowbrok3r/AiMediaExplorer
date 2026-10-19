//! The real application as a command sink: tabs, panels, mailboxes, tasks and the failure toasts.

use std::sync::Arc;

use super::bus::testing::{serial, take_matching};
use super::*;
use crate::app::SmartMediaApp;
use crate::database::Thumbnail;
use crate::database::ui_themes::tests::{Fixture, RT};
use crate::tasks::{TaskEvent, TaskKind, TaskState};
use crate::theme::Appearance;
use crate::theme::chrome::Chrome;
use crate::ui::file_table::FileExplorer;
use crate::ui::tabs::{BrowserId, TabKind};
use crate::ui::test_support::test_ctx;

struct Rig {
    app: SmartMediaApp,
    dir: tempfile::TempDir,
    _fixture: Fixture,
    _runtime: tokio::runtime::EnterGuard<'static>,
}

impl Rig {
    fn new() -> Self {
        let fixture = Fixture::new();
        let runtime = RT.enter();
        let ctx = test_ctx();
        let appearance = Appearance::attach(&ctx, None, Arc::new(fixture.store()));
        let mut app = SmartMediaApp::assemble(ctx, appearance, Chrome { frameless: false, available: true }, FileExplorer::new(true));
        app.tree.set_focused_node_and_surface(egui_dock::NodePath { surface: egui_dock::SurfaceIndex::main(), node: egui_dock::NodeIndex::root() });
        Self { app, dir: tempfile::tempdir().expect("empty folder"), _fixture: fixture, _runtime: runtime }
    }

    fn title_of(&self, kind: TabKind) -> String {
        match kind.browser_id() {
            Some(id) if id == BrowserId::MAIN => self.app.context.file_explorer.tab_title(),
            Some(id) => self.app.context.filtered_tabs.get(&id).map(|ex| ex.tab_title()).unwrap_or_default(),
            None => kind.fixed_title().to_string(),
        }
    }

    fn titles(&self) -> Vec<String> {
        self.app.tree.iter_all_tabs().map(|(_, t)| self.title_of(*t)).collect()
    }

    fn focused(&mut self) -> String {
        let kind = self.app.tree.find_active_focused().map(|(_, t)| *t);
        kind.map(|k| self.title_of(k)).unwrap_or_default()
    }

    fn focused_kind(&mut self) -> Option<TabKind> {
        self.app.tree.find_active_focused().map(|(_, t)| *t)
    }

    fn toasts(&self) -> Vec<(egui_toast::ToastKind, String)> {
        self.app.context.toast_rx.try_iter().collect()
    }
}

fn row(path: &str) -> Thumbnail {
    let mut t = Thumbnail::default();
    t.path = path.into();
    t.filename = path.rsplit('/').next().unwrap_or(path).into();
    t.file_type = "jpg".into();
    t
}

fn open_folder(rig: &mut Rig, title: &str, target: Target) -> CommandResult {
    let dir = rig.dir.path().display().to_string();
    rig.app.open_scope(&Scope::Folder(dir), target, Some(title), false)
}

#[test]
fn a_folder_scope_opens_a_titled_tab_that_takes_focus() {
    let mut rig = Rig::new();
    assert!(open_folder(&mut rig, "Sink folder", Target::NewTab).is_ok());
    assert!(rig.titles().contains(&"Sink folder".to_string()));
    assert!(rig.app.context.browser_titled("Sink folder").is_some());
    assert_eq!(rig.focused(), "Sink folder");
}

#[test]
fn a_background_scope_leaves_focus_alone_and_each_tab_keeps_its_own_explorer() {
    let mut rig = Rig::new();
    open_folder(&mut rig, "Sink a", Target::NewTab).unwrap();
    open_folder(&mut rig, "Sink b", Target::Background).unwrap();
    assert_eq!(rig.focused(), "Sink a");
    open_folder(&mut rig, "Sink a", Target::NewTab).unwrap();
    assert_eq!(rig.titles().iter().filter(|t| t.as_str() == "Sink a").count(), 2, "titles come from the scope, so two tabs may share one");
    assert_eq!(rig.app.context.filtered_tabs.len(), 3);
}

#[test]
fn row_tabs_hold_their_rows_and_similarity_data() {
    let mut rig = Rig::new();
    let tab = RowsTab { title: "Sink rows".into(), rows: vec![row("/a/1.jpg"), row("/a/2.jpg")], showing_similarity: true, origin_path: Some("/a/1.jpg".into()), ..Default::default() };
    let text = rig.app.open_rows(tab).unwrap();
    assert!(text.contains("2 rows"), "{text}");
    let (_, ex) = rig.app.context.browser_titled("Sink rows").expect("the rows tab");
    assert_eq!(ex.table.len(), 2);
    assert!(ex.viewer.showing_similarity);
}

#[test]
fn the_image_edit_tool_opens_once_and_loads_the_file() {
    let mut rig = Rig::new();
    rig.app.open_tool(Tool::ImageEdit, Some("/img/one.png"), Target::NewTab).unwrap();
    rig.app.open_tool(Tool::ImageEdit, Some("/img/two.png"), Target::NewTab).unwrap();
    assert_eq!(rig.titles().iter().filter(|t| t.as_str() == "Image Edit").count(), 1, "no duplicate tool tab");
    assert_eq!(rig.focused(), "Image Edit");
}

#[test]
fn every_query_scope_opens_in_the_focused_tab_and_back_returns_to_where_it_was() {
    let mut rig = Rig::new();
    let start = rig.app.context.file_explorer.scope().clone();
    let scopes = [
        Scope::Library,
        Scope::Recent,
        Scope::Collection("c".into()),
        Scope::Tag("t".into()),
        Scope::Category("c".into()),
        Scope::Semantic("q".into()),
        Scope::Similar("/p".into()),
        Scope::Scan(1),
        Scope::Smart(SmartView::NoClip),
        Scope::Smart(SmartView::Undescribed),
        Scope::Smart(SmartView::Untagged),
        Scope::Smart(SmartView::FailedThumbs),
        Scope::Smart(SmartView::Duplicates),
    ];
    for scope in &scopes {
        let text = rig.app.open_scope(scope, Target::Current, None, false).unwrap_or_else(|e| panic!("{scope}: {e}"));
        assert!(text.contains("focused tab"), "{text}");
        assert_eq!(rig.app.context.file_explorer.scope(), scope);
        assert_eq!(rig.app.context.filtered_tabs.len(), 0, "{scope} did not spawn a tab");
    }
    assert_eq!(rig.titles().len(), 2, "the main browser and the logs tab");
    for _ in 0..scopes.len() {
        rig.app.navigate(NavOp::Back).unwrap();
    }
    assert_eq!(rig.app.context.file_explorer.scope(), &start);
    assert_eq!(rig.app.navigate(NavOp::Back).unwrap_err().kind, ErrorKind::Invalid, "nothing is further back");
    rig.app.navigate(NavOp::Forward).unwrap();
    assert_eq!(rig.app.context.file_explorer.scope(), &scopes[0]);
    rig.app.context.file_explorer.cancel_load();
}

#[test]
fn every_query_scope_can_open_in_a_new_tab_that_starts_with_that_scope() {
    let mut rig = Rig::new();
    for scope in [Scope::Library, Scope::Tag("t".into()), Scope::Smart(SmartView::Duplicates), Scope::Semantic("q".into())] {
        rig.app.open_scope(&scope, Target::NewTab, None, false).unwrap();
        let kind = rig.focused_kind().unwrap();
        let id = kind.browser_id().unwrap();
        let ex = &rig.app.context.filtered_tabs[&id];
        assert_eq!(ex.scope(), &scope);
        assert!(!ex.can_go_back(), "a new tab starts its history at its scope");
    }
    for ex in rig.app.context.filtered_tabs.values_mut() {
        ex.cancel_load();
    }
}

#[test]
fn an_explicit_title_names_the_tab_until_it_navigates() {
    let mut rig = Rig::new();
    open_folder(&mut rig, "Named", Target::NewTab).unwrap();
    let id = rig.focused_kind().unwrap().browser_id().unwrap();
    assert_eq!(rig.app.context.filtered_tabs[&id].tab_title(), "Named");
    let ex = rig.app.context.explorer_mut(id).unwrap();
    ex.open_scope(Scope::Library).unwrap();
    assert_eq!(ex.tab_title(), "Entire Database");
    ex.cancel_load();
}

#[test]
fn a_folder_that_does_not_exist_is_not_found_and_changes_nothing() {
    let mut rig = Rig::new();
    let before = rig.app.context.file_explorer.scope().clone();
    let err = rig.app.open_scope(&Scope::Folder("/definitely/not/here".into()), Target::Current, None, false).unwrap_err();
    assert_eq!(err.kind, ErrorKind::NotFound);
    assert_eq!(rig.app.context.file_explorer.scope(), &before);
    assert!(rig.app.open_scope(&Scope::Folder("/definitely/not/here".into()), Target::NewTab, None, false).is_err());
    assert!(rig.app.context.filtered_tabs.is_empty(), "no tab is left behind");
}

#[test]
fn opening_in_the_focused_tab_brings_the_browser_forward_when_a_tool_tab_is_in_front() {
    let mut rig = Rig::new();
    rig.app.open_tool(Tool::Logs, None, Target::NewTab).unwrap();
    assert_eq!(rig.focused_kind(), Some(TabKind::Logs));
    rig.app.open_scope(&Scope::Library, Target::Current, None, false).unwrap();
    assert_eq!(rig.focused_kind(), Some(TabKind::MAIN), "the scope is shown where the user can see it");
    rig.app.context.file_explorer.cancel_load();
}

#[test]
fn navigation_commands_report_what_they_did_or_why_not() {
    let mut rig = Rig::new();
    let dir = rig.dir.path().display().to_string();
    let sub = rig.dir.path().join("sub");
    std::fs::create_dir_all(&sub).unwrap();
    rig.app.open_scope(&Scope::Folder(dir.clone()), Target::Current, None, false).unwrap();
    rig.app.open_scope(&Scope::Folder(sub.display().to_string()), Target::Current, None, false).unwrap();
    assert!(rig.app.navigate(NavOp::Up).unwrap().contains(&dir));
    assert!(rig.app.navigate(NavOp::Back).unwrap().contains(&sub.display().to_string()));
    assert!(rig.app.navigate(NavOp::Forward).is_ok());
    assert_eq!(rig.app.navigate(NavOp::Forward).unwrap_err().kind, ErrorKind::Invalid);
    assert_eq!(rig.app.navigate(NavOp::LoadMore).unwrap_err().kind, ErrorKind::Invalid, "a folder has no further pages");
    assert!(rig.app.navigate(NavOp::Refresh).is_ok());
    assert!(rig.app.navigate(NavOp::Cancel).is_ok());
}

#[test]
fn the_database_command_is_a_no_op_when_the_database_is_already_connected() {
    if !crate::database::is_connected() {
        return;
    }
    let mut rig = Rig::new();
    assert!(rig.app.connect_database().unwrap().contains("already connected"));
}

#[test]
fn failed_thumbnails_are_listed_without_needing_any_tab_to_hold_them() {
    let _serial = crate::ui::file_table::scope::failed::serial();
    crate::ui::file_table::scope::failed::clear();
    crate::ui::file_table::scope::failed::record("/a/bad.jpg", "corrupt");
    assert_eq!(crate::ui::file_table::scope::failed::count(), 1);
    crate::ui::file_table::scope::failed::clear();
}

#[test]
fn failed_thumb_rows_carry_the_marker_category_and_the_error() {
    let rows = super::sink::failed_thumb_rows(&[("/x/y/Photo.JPG".to_string(), "bad header".to_string())]);
    assert_eq!(rows[0].filename, "Photo.JPG");
    assert_eq!(rows[0].file_type, "jpg");
    assert_eq!(rows[0].category.as_deref(), Some("<FAILED_THUMB>"));
    assert_eq!(rows[0].parent_dir, "/x/y");
}

#[test]
fn settings_open_the_window_the_section_names() {
    let mut rig = Rig::new();
    rig.app.show_settings(None).unwrap();
    assert!(rig.app.context.open_settings_modal);
    rig.app.show_settings(Some(Section::Advanced)).unwrap();
    assert!(rig.app.context.open_ui_settings);
}

#[test]
fn attach_and_refine_fill_the_mailboxes_and_show_their_panels() {
    let mut rig = Rig::new();
    crate::ui::assistant::ATTACH_REQUESTS.lock().unwrap().clear();
    rig.app.attach_to_assistant(vec!["/a.jpg".into()]).unwrap();
    assert_eq!(crate::ui::assistant::ATTACH_REQUESTS.lock().unwrap().last(), Some(&vec!["/a.jpg".to_string()]));
    assert!(rig.app.context.assistant_window_open);
    crate::ui::refine::REFINE_SELECTION_REQUESTS.lock().unwrap().clear();
    rig.app.refine(vec!["/b.jpg".into()]).unwrap();
    assert_eq!(crate::ui::refine::REFINE_SELECTION_REQUESTS.lock().unwrap().last(), Some(&vec!["/b.jpg".to_string()]));
    assert_eq!(rig.focused(), "AI Refinements");
}

#[test]
fn toasts_reach_the_toast_channel() {
    let mut rig = Rig::new();
    rig.app.toast(ToastLevel::Success, "hello".into()).unwrap();
    assert_eq!(rig.toasts(), vec![(egui_toast::ToastKind::Success, "hello".to_string())]);
}

#[test]
fn task_updates_drive_the_registry() {
    let mut rig = Rig::new();
    let text = rig.app.task_update(0, TaskEvent::Begin { kind: TaskKind::Debug, title: "sink-task".into(), total: 4 }).unwrap();
    let id: u64 = text.rsplit(' ').next().unwrap().parse().unwrap();
    rig.app.task_update(id, TaskEvent::Progress { done: 1, total: 4 }).unwrap();
    assert_eq!(crate::tasks::with(|r| r.get(id).map(|t| (t.done, t.state))), Some((1, TaskState::Running)));
    rig.app.task_update(id, TaskEvent::Complete).unwrap();
    assert_eq!(crate::tasks::with(|r| r.get(id).map(|t| t.state)), Some(TaskState::Completed));
    let err = rig.app.task_update(id, TaskEvent::Complete).unwrap_err();
    assert_eq!(err.kind, ErrorKind::NotFound);
}

#[test]
fn selectors_resolve_against_the_focused_tab_without_falling_back_to_everything() {
    let mut rig = Rig::new();
    {
        let ex = &mut rig.app.context.file_explorer;
        ex.table.push(row("/p/a.jpg"));
        ex.table.push(row("/p/b.jpg"));
        ex.table.push(row("/p/c.jpg"));
    }
    assert!(rig.app.resolve(&Selector::Selection).is_empty(), "nothing selected resolves to nothing");
    assert!(rig.app.resolve(&Selector::Focus).is_empty(), "nothing is focused before a row is chosen");
    {
        let ex = &mut rig.app.context.file_explorer;
        ex.sync_view();
        ex.select_paths(["/p/b.jpg", "/p/a.jpg"]);
    }
    assert_eq!(rig.app.resolve(&Selector::Selection), vec!["/p/a.jpg", "/p/b.jpg"]);
    assert_eq!(rig.app.resolve(&Selector::View).len(), 3);
    assert_eq!(rig.app.resolve(&Selector::Focus).len(), 1, "the focus is one of the selected rows");
    rig.app.focus_path("/p/c.jpg").unwrap();
    assert_eq!(rig.app.resolve(&Selector::Focus), vec!["/p/c.jpg"]);
    assert_eq!(rig.app.resolve(&Selector::Selection), vec!["/p/c.jpg"]);
    assert_eq!(rig.app.focus_path("/missing").unwrap_err().kind, ErrorKind::NotFound);
    rig.app.clear_selection().unwrap();
    assert!(rig.app.resolve(&Selector::Selection).is_empty());
}

#[test]
fn an_action_on_an_empty_selection_never_reaches_the_app() {
    let mut rig = Rig::new();
    let err = dispatch(&mut rig.app, AppCommand::Run { action: ActionId::GenerateClip, on: Selector::Selection }).unwrap_err();
    assert_eq!(err.kind, ErrorKind::Invalid);
    let too_many: Vec<String> = (0..30).map(|i| format!("/x/{i}.jpg")).collect();
    let err = rig.app.run_action(ActionId::Open, too_many).unwrap_err();
    assert_eq!(err.kind, ErrorKind::Invalid);
    assert_eq!(rig.app.run_action(ActionId::Describe, vec!["/a".into()]).unwrap_err().kind, ErrorKind::Invalid, "a path that is no image cannot be described");
}

#[test]
fn devtools_and_the_task_center_open_by_command() {
    let mut rig = Rig::new();
    rig.app.open_devtools(DevtoolsWindow::Tasks).unwrap();
    rig.app.open_devtools(DevtoolsWindow::CommandLog).unwrap();
    assert!(crate::ui::devtools::tasks::is_open(&rig.app.egui_ctx));
    assert!(crate::ui::devtools::command_log::is_open(&rig.app.egui_ctx));
    rig.app.show_task_center().unwrap();
    assert!(crate::ui::taskbar::is_center_open(&rig.app.egui_ctx));
}

#[test]
fn the_global_pump_runs_queued_commands_logs_them_and_toasts_failures() {
    let _serial = serial();
    let mut rig = Rig::new();
    take_matching(|_| true);
    with_log(|log| log.clear());
    send(AppCommand::ClearSelection);
    send(AppCommand::open(Scope::Folder("/definitely/not/here".into())));
    send(AppCommand::Toast { kind: ToastLevel::Info, text: "pumped".into() });
    assert_eq!(pump_global(&mut rig.app), 3);
    let (entries, failures) = with_log(|log| (log.entries().map(|e| e.json.clone()).collect::<Vec<_>>(), log.failures()));
    assert_eq!(entries.len(), 3);
    assert!(entries[1].contains("fs:/definitely/not/here"));
    assert_eq!(failures, 1);
    let toasts = rig.toasts();
    assert!(toasts.iter().any(|(k, t)| *k == egui_toast::ToastKind::Info && t == "pumped"));
    assert!(toasts.iter().any(|(k, t)| *k == egui_toast::ToastKind::Warning && t.starts_with("open_scope: ") && t.contains("not a folder that exists")), "{toasts:?}");
    with_log(|log| log.clear());
}

#[test]
fn a_command_sent_from_another_thread_runs_on_the_next_pump() {
    let _serial = serial();
    let mut rig = Rig::new();
    take_matching(|_| true);
    with_log(|log| log.clear());
    std::thread::spawn(|| send(AppCommand::Toast { kind: ToastLevel::Info, text: "from-thread".into() })).join().unwrap();
    pump_global(&mut rig.app);
    assert!(rig.toasts().iter().any(|(_, t)| t == "from-thread"));
    with_log(|log| log.clear());
}

fn sized_row(path: &str, size: u64) -> Thumbnail {
    let mut t = row(path);
    t.size = size;
    t
}

#[test]
fn filter_commands_change_the_focused_tab_and_never_remove_its_rows() {
    let mut rig = Rig::new();
    {
        let ex = &mut rig.app.context.file_explorer;
        ex.table.push(sized_row("/p/small.jpg", 500));
        ex.table.push(sized_row("/p/big.jpg", 5_000));
    }
    let done = dispatch(&mut rig.app, AppCommand::SetFilter { filter: serde_json::json!({"size": {"min": 1000}}), replace: false }).unwrap();
    assert_eq!(done, "1 active filters");
    assert_eq!(rig.app.resolve(&Selector::View), vec!["/p/big.jpg"], "a view selector reaches only the listed rows");
    assert_eq!(rig.app.context.file_explorer.table.len(), 2);

    let err = dispatch(&mut rig.app, AppCommand::SetFilter { filter: serde_json::json!({"size": {"min": "big"}}), replace: false }).unwrap_err();
    assert_eq!(err.kind, ErrorKind::Invalid);
    assert_eq!(rig.app.resolve(&Selector::View), vec!["/p/big.jpg"], "a rejected filter changes nothing");

    let done = dispatch(&mut rig.app, AppCommand::ClearFilters).unwrap();
    assert_eq!(done, "cleared 1 filters");
    assert_eq!(rig.app.resolve(&Selector::View), vec!["/p/small.jpg", "/p/big.jpg"]);
}

#[test]
fn a_saved_view_is_applied_by_name_and_an_unknown_name_is_not_found() {
    let mut rig = Rig::new();
    let err = dispatch(&mut rig.app, AppCommand::ApplySavedView { name: "Big".into() }).unwrap_err();
    assert_eq!(err.kind, ErrorKind::NotFound);
    let mut spec = crate::ui::file_table::rows::FilterSpec::none();
    spec.size.min = Some(1000);
    rig.app.context.file_explorer.set_saved_views(vec![crate::ui::file_table::filters::group_from_spec("Big", &spec)]);
    {
        let ex = &mut rig.app.context.file_explorer;
        ex.table.push(sized_row("/p/small.jpg", 500));
        ex.table.push(sized_row("/p/big.jpg", 5_000));
    }
    assert_eq!(dispatch(&mut rig.app, AppCommand::ApplySavedView { name: "big".into() }).unwrap(), "applied view 'Big'");
    assert_eq!(rig.app.resolve(&Selector::View), vec!["/p/big.jpg"]);
}

#[test]
fn the_filters_window_opens_by_command() {
    let mut rig = Rig::new();
    rig.app.open_devtools(DevtoolsWindow::Filters).unwrap();
    assert!(crate::ui::devtools::filters::is_open(&rig.app.egui_ctx));
}

#[test]
fn the_view_command_sets_the_focused_tabs_mode_size_and_fit() {
    use crate::ui::file_table::table::view_mode::{TileFit, ViewMode};
    let mut rig = Rig::new();
    open_folder(&mut rig, "Sink view", Target::NewTab).unwrap();
    let id = rig.app.context.browser_titled("Sink view").map(|(id, _)| id).expect("the view tab");
    rig.app.context.active_tab = Some(crate::ui::tabs::TabKind::Browser { id });
    let text = rig.app.set_view_mode(ViewMode::Details, Some(200.0), Some(TileFit::Fit), Some(false), None).unwrap();
    assert!(text.contains("Details") && text.contains("200"), "{text}");
    let ex = &rig.app.context.filtered_tabs[&id];
    assert_eq!((ex.browse.mode, ex.browse.tile, ex.browse.fit, ex.browse.captions), (ViewMode::Details, 200.0, TileFit::Fit, false));
    rig.app.set_view_mode(ViewMode::Grid, Some(9_999.0), None, None, Some(30.0)).unwrap();
    let ex = &rig.app.context.filtered_tabs[&id];
    assert_eq!((ex.browse.mode, ex.browse.tile), (ViewMode::Grid, 320.0), "sizes are clamped");
    assert_eq!(ex.browse.grid.stress().map(|s| s.rows_per_s), Some(30.0));
    let err = rig.app.set_view_mode(ViewMode::Grid, Some(f32::NAN), None, None, None).unwrap_err();
    assert_eq!(err.kind, ErrorKind::Invalid);
}
