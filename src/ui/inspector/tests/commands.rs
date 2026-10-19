use std::sync::Arc;

use egui::{Pos2, RawInput, Rect, vec2};

use super::*;
use crate::app::SmartMediaApp;
use crate::command::{AppCommand, CommandSink, DevtoolsWindow, ErrorKind, dispatch};
use crate::database::ui_themes::tests::{Fixture, RT};
use crate::theme::Appearance;
use crate::theme::chrome::Chrome;
use crate::ui::devtools::inspector as dev;

struct AppRig {
    app: SmartMediaApp,
    ctx: Context,
    _fixture: Fixture,
    _runtime: tokio::runtime::EnterGuard<'static>,
}

impl AppRig {
    fn new() -> Self {
        let fixture = Fixture::new();
        let runtime = RT.enter();
        let ctx = test_ctx();
        egui_extras::install_image_loaders(&ctx);
        let appearance = Appearance::attach(&ctx, None, Arc::new(fixture.store()));
        let mut app = SmartMediaApp::assemble(ctx.clone(), appearance, Chrome { frameless: false, available: true }, FileExplorer::new(true));
        let rows: Vec<Thumbnail> = ["/pics/a.jpg", "/pics/b.jpg", "/pics/c.jpg"].iter().map(|p| file_row(Path::new("/pics"), p.rsplit('/').next().unwrap(), "jpg")).collect();
        app.context.file_explorer.table.bulk_load(rows);
        app.context.file_explorer.sync_view();
        Self { app, ctx, _fixture: fixture, _runtime: runtime }
    }

    fn frame(&mut self) -> Vec<String> {
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1200.0, 700.0))), ..Default::default() };
        let app = &mut self.app;
        let full = self.ctx.run_ui(input, |ui| {
            app.inspector_rail(ui);
            crate::ui::devtools::show(ui.ctx());
        });
        let drawn = texts(&full.shapes).into_iter().map(|(t, _)| t).collect();
        full.drop_without_applying_deltas();
        drawn
    }
}

#[test]
fn the_toggle_command_sets_or_flips_the_rail() {
    let mut rig = AppRig::new();
    assert!(rig.app.context.inspector.is_open(), "the rail starts open");
    assert_eq!(dispatch(&mut rig.app, AppCommand::InspectorToggle { open: Some(false) }).unwrap(), "inspector hidden");
    assert!(!rig.app.context.inspector.is_open());
    assert_eq!(dispatch(&mut rig.app, AppCommand::InspectorToggle { open: Some(false) }).unwrap(), "inspector hidden", "setting the same state twice changes nothing");
    assert_eq!(dispatch(&mut rig.app, AppCommand::InspectorToggle { open: None }).unwrap(), "inspector shown");
    assert!(rig.app.context.inspector.is_open());
    assert_eq!(dispatch(&mut rig.app, AppCommand::InspectorToggle { open: None }).unwrap(), "inspector hidden");
    assert!(rig.frame().iter().all(|t| t != "Inspect"), "a hidden rail draws nothing");
}

#[test]
fn focusing_a_path_selects_it_shows_the_rail_and_the_rail_draws_it() {
    let mut rig = AppRig::new();
    rig.app.inspector_toggle(Some(false)).unwrap();
    assert!(!rig.frame().iter().any(|t| t == "b.jpg"));
    assert_eq!(rig.app.focus_path("/pics/b.jpg").unwrap(), "focused /pics/b.jpg");
    assert!(rig.app.context.inspector.is_open(), "focusing a file shows the rail");
    assert_eq!(rig.app.context.file_explorer.selection.len(), 1);
    rig.frame();
    let drawn = rig.frame();
    assert!(drawn.iter().any(|t| t == "b.jpg") && drawn.iter().any(|t| t == "Info"), "{drawn:?}");
    assert_eq!(rig.app.context.inspector.last_target, "file /pics/b.jpg");

    let err = rig.app.focus_path("/pics/missing.jpg").unwrap_err();
    assert_eq!(err.kind, ErrorKind::NotFound);
    assert_eq!(rig.app.context.inspector.last_target, "file /pics/b.jpg", "a failed focus keeps the selection");

    rig.app.inspector_toggle(Some(false)).unwrap();
    assert!(rig.app.focus_path("/pics/missing.jpg").is_err());
    assert!(!rig.app.context.inspector.is_open(), "a failed focus does not open the rail");
}

#[test]
fn clearing_the_selection_by_command_gives_the_empty_state() {
    let mut rig = AppRig::new();
    rig.app.focus_path("/pics/a.jpg").unwrap();
    rig.frame();
    rig.app.clear_selection().unwrap();
    let drawn = rig.frame();
    assert!(drawn.iter().any(|t| t == "Select a file"), "{drawn:?}");
}

#[test]
fn the_devtools_window_opens_by_command_and_reports_the_focus() {
    let mut rig = AppRig::new();
    rig.app.open_devtools(DevtoolsWindow::Inspector).unwrap();
    assert!(dev::is_open(&rig.ctx));
    rig.app.focus_path("/pics/c.jpg").unwrap();
    let drawn = rig.frame();
    let snapshot = dev::latest(&rig.ctx).expect("the rail published its state");
    assert_eq!(snapshot.focus.as_deref(), Some("/pics/c.jpg"));
    assert_eq!(snapshot.selection, 1);
    assert!(snapshot.open && !snapshot.overlay);
    assert!(drawn.iter().any(|t| t == "Devtools: Inspector") || drawn.iter().any(|t| t == "Force save"), "{drawn:?}");
}

#[test]
fn the_devtools_buttons_drive_the_rail_without_the_mouse() {
    let mut rig = AppRig::new();
    rig.app.focus_path("/pics/a.jpg").unwrap();
    rig.frame();
    dev::push_command(&rig.ctx, dev::Command::Next);
    rig.frame();
    rig.frame();
    assert_eq!(rig.app.context.inspector.last_target, "file /pics/b.jpg");
    dev::push_command(&rig.ctx, dev::Command::Previous);
    rig.frame();
    rig.frame();
    assert_eq!(rig.app.context.inspector.last_target, "file /pics/a.jpg");
    dev::push_command(&rig.ctx, dev::Command::ForceOverlay(Some(true)));
    rig.frame();
    assert!(rig.app.context.inspector.overlay_active, "the overlay can be forced on a wide window");
    dev::push_command(&rig.ctx, dev::Command::ClearSelection);
    rig.frame();
    rig.frame();
    assert_eq!(rig.app.context.inspector.last_target, "empty");
    dev::push_command(&rig.ctx, dev::Command::ToggleOpen);
    rig.frame();
    assert!(!rig.app.context.inspector.is_open());
}
