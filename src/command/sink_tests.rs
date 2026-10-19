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

    fn titles(&self) -> Vec<String> {
        self.app.tree.iter_all_tabs().map(|(_, t)| t.clone()).collect()
    }

    fn focused(&mut self) -> String {
        self.app.tree.find_active_focused().map(|(_, t)| t.clone()).unwrap_or_default()
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
    assert!(rig.app.context.filtered_tabs.contains_key("Sink folder"));
    assert_eq!(rig.focused(), "Sink folder");
}

#[test]
fn a_background_scope_leaves_focus_alone_and_titles_stay_unique() {
    let mut rig = Rig::new();
    open_folder(&mut rig, "Sink a", Target::NewTab).unwrap();
    open_folder(&mut rig, "Sink b", Target::Background).unwrap();
    assert_eq!(rig.focused(), "Sink a");
    open_folder(&mut rig, "Sink a", Target::NewTab).unwrap();
    assert!(rig.titles().contains(&"Sink a (2)".to_string()));
}

#[test]
fn row_tabs_hold_their_rows_and_similarity_data() {
    let mut rig = Rig::new();
    let tab = RowsTab { title: "Sink rows".into(), rows: vec![row("/a/1.jpg"), row("/a/2.jpg")], showing_similarity: true, origin_path: Some("/a/1.jpg".into()), ..Default::default() };
    let text = rig.app.open_rows(tab).unwrap();
    assert!(text.contains("2 rows"), "{text}");
    let ex = &rig.app.context.filtered_tabs["Sink rows"];
    assert_eq!(ex.table.len(), 2);
    assert!(ex.viewer.showing_similarity);
    assert_eq!(ex.current_path, "/a/1.jpg");
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
fn the_unsupported_scopes_say_so() {
    let mut rig = Rig::new();
    for scope in [Scope::Recent, Scope::Collection("c".into()), Scope::Semantic("q".into()), Scope::Similar("/p".into()), Scope::Scan(1), Scope::Smart(SmartView::NoClip)] {
        let err = rig.app.open_scope(&scope, Target::NewTab, None, false).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Unsupported, "{scope}");
    }
}

#[test]
fn failed_thumbnails_open_as_a_tab_and_report_when_there_are_none() {
    let mut rig = Rig::new();
    let none = rig.app.open_scope(&Scope::Smart(SmartView::FailedThumbs), Target::NewTab, None, false).unwrap_err();
    assert_eq!(none.kind, ErrorKind::NotFound);
    rig.app.context.file_explorer.failed_thumbs.push(("/a/bad.jpg".into(), "corrupt".into()));
    rig.app.open_scope(&Scope::Smart(SmartView::FailedThumbs), Target::NewTab, None, false).unwrap();
    let ex = &rig.app.context.filtered_tabs["Failed thumbnails (1)"];
    assert_eq!(ex.table.len(), 1);
    assert_eq!(ex.table.iter().next().unwrap().description.as_deref(), Some("Thumbnail failed: corrupt"));
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
    rig.app.context.file_explorer.viewer.selected.extend(["/p/b.jpg".to_string(), "/p/a.jpg".to_string()]);
    assert_eq!(rig.app.resolve(&Selector::Selection), vec!["/p/a.jpg", "/p/b.jpg"]);
    assert_eq!(rig.app.resolve(&Selector::View).len(), 3);
    assert!(rig.app.resolve(&Selector::Focus).is_empty());
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
    assert_eq!(rig.app.run_action(ActionId::Describe, vec!["/a".into()]).unwrap_err().kind, ErrorKind::Unsupported);
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
    send(AppCommand::open(Scope::Scan(77)));
    send(AppCommand::Toast { kind: ToastLevel::Info, text: "pumped".into() });
    assert_eq!(pump_global(&mut rig.app), 3);
    let (entries, failures) = with_log(|log| (log.entries().map(|e| e.json.clone()).collect::<Vec<_>>(), log.failures()));
    assert_eq!(entries.len(), 3);
    assert!(entries[1].contains("scan:77"));
    assert_eq!(failures, 1);
    let toasts = rig.toasts();
    assert!(toasts.iter().any(|(k, t)| *k == egui_toast::ToastKind::Info && t == "pumped"));
    assert!(toasts.iter().any(|(k, t)| *k == egui_toast::ToastKind::Warning && t.starts_with("open_scope: no loader for scan scopes")), "{toasts:?}");
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
