//! Shell tests: panel layout at several window sizes, shortcuts, persistence and the shared actions.

use std::sync::Arc;

use eframe::egui::epaint::ClippedShape;
use eframe::egui::{CentralPanel, Context, Event, Key, Modifiers, Pos2, RawInput, Rect, Shape, vec2};

use super::*;
use crate::command::CommandSink;
use crate::command::bus::testing::{serial, take_matching};
use crate::database::ui_themes::tests::{Fixture, RT};
use crate::status::StatusMeta;
use crate::theme::Appearance;
use crate::theme::chrome::Chrome;
use crate::ui::sidebar::data::PresetData;
use crate::ui::sidebar::layout::{DEFAULT_WIDTH, FILE_NAME};
use crate::ui::sidebar::links::{CollectionEntry, CountEntry};
use crate::ui::sidebar::{RAIL_WIDTH, SectionId};
use crate::ui::test_support::{press_release, test_ctx, text_pos, texts};

fn engines() -> Vec<StatusMeta> {
    ["DB", "VISION", "CLIP", "RERANK", "QWEN-EDIT", "ASSIST"].into_iter().map(|name| StatusMeta { name, ..Default::default() }).collect()
}

struct Out {
    shapes: Vec<ClippedShape>,
    strip: crate::ui::titlebar::StripLayout,
    central: Rect,
    sidebar: SidebarLayout,
}

impl Out {
    fn has(&self, text: &str) -> bool {
        texts(&self.shapes).iter().any(|(t, _)| t == text)
    }

    fn pos(&self, text: &str) -> Pos2 {
        text_pos(&self.shapes, text)
    }
}

/// Text shapes with their rectangle and clip rectangle, leaving out text clipped away entirely.
fn text_rects(shapes: &[ClippedShape]) -> Vec<(String, Rect)> {
    fn walk(shape: &Shape, clip: Rect, out: &mut Vec<(String, Rect)>) {
        match shape {
            Shape::Text(t) if clip.width() > 0.0 && clip.height() > 0.0 => out.push((t.galley.text().to_string(), Rect::from_min_size(t.pos, t.galley.size()))),
            Shape::Vec(v) => v.iter().for_each(|s| walk(s, clip, out)),
            _ => {}
        }
    }
    let mut out = Vec::new();
    shapes.iter().for_each(|s| walk(&s.shape, s.clip_rect, &mut out));
    out
}

struct Rig {
    app: SmartMediaApp,
    ctx: Context,
    size: (f32, f32),
    _fixture: Fixture,
    _runtime: tokio::runtime::EnterGuard<'static>,
}

impl Rig {
    fn new(width: f32, height: f32) -> Self {
        let fixture = Fixture::new();
        let runtime = RT.enter();
        let ctx = test_ctx();
        let appearance = Appearance::attach(&ctx, None, Arc::new(fixture.store()));
        let mut app = SmartMediaApp::assemble(ctx.clone(), appearance, Chrome { frameless: true, available: true }, FileExplorer::new(true));
        app.shell.sidebar.data.preset(fill);
        app.context.db_ready = true;
        Self { app, ctx, size: (width, height), _fixture: fixture, _runtime: runtime }
    }

    fn frame(&mut self, events: Vec<Event>) -> Out {
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(self.size.0, self.size.1))), events, ..Default::default() };
        let app = &mut self.app;
        let (mut strip, mut central) = (None, Rect::NOTHING);
        let engines = engines();
        let out = self.ctx.run_ui(input, |ui| {
            let ctx = ui.ctx().clone();
            strip = Some(app.title_strip_with(ui, &engines));
            app.status_bar(ui);
            app.side_panels(ui);
            central = CentralPanel::default().show(ui, |ui| {
                egui_dock::DockArea::new(&mut app.tree).show_inside(ui, &mut app.context);
            }).response.rect;
            crate::ui::devtools::show(&ctx);
            app.shell_overlays(&ctx);
        });
        let shapes = out.shapes.clone();
        out.drop_without_applying_deltas();
        let sidebar = last_sidebar(&self.ctx).expect("the sidebar recorded its layout");
        Out { shapes, strip: strip.expect("the strip ran"), central, sidebar }
    }

    fn settle(&mut self) -> Out {
        (0..3).map(|_| self.frame(Vec::new())).last().unwrap()
    }

    fn click(&mut self, at: Pos2) -> Out {
        self.frame(press_release(at));
        self.settle()
    }
}

fn fill(p: &mut PresetData) {
    p.total = Some(10);
    p.places.quick = vec![crate::utilities::types::QuickAccess { label: "Camera roll".into(), path: "/home/u/Camera".into(), icon: String::new() }];
    p.collections = vec![CollectionEntry { id: surrealdb::types::RecordId::new("collections", "holidays"), name: "Holidays".into(), count: 12 }];
    p.tags = vec![CountEntry { name: "sunset".into(), count: 9 }];
    p.categories = vec![CountEntry { name: "landscape".into(), count: 3 }];
}

fn key(key: Key, modifiers: Modifiers) -> Event {
    Event::Key { key, physical_key: Some(key), pressed: true, repeat: false, modifiers }
}

/* ---------------------------------------------- pure -------------------------------------------- */

#[test]
fn a_browser_stands_for_the_scope_it_shows() {
    let ex = FileExplorer::new(false);
    assert_eq!(focused_scope(&ex), Some(ex.scope().clone()));
}

#[test]
fn picked_folders_open_here_or_scan_in_a_new_tab() {
    let dir = std::path::Path::new("/home/u/Pictures");
    assert_eq!(serde_json::to_value(open_folder_command(dir, false)).unwrap(), serde_json::json!({"cmd": "open_scope", "scope": "fs:/home/u/Pictures", "target": "current"}));
    assert_eq!(serde_json::to_value(open_folder_command(dir, true)).unwrap(), serde_json::json!({"cmd": "open_scope", "scope": "fs:/home/u/Pictures", "target": "new_tab", "recursive": true}));
}

#[test]
fn shell_action_ids_and_labels_are_unique() {
    let ids: std::collections::BTreeSet<_> = ShellAction::ALL.iter().map(|a| a.id()).collect();
    let labels: std::collections::BTreeSet<_> = ShellAction::ALL.iter().map(|a| a.label()).collect();
    assert_eq!((ids.len(), labels.len()), (ShellAction::ALL.len(), ShellAction::ALL.len()));
}

/* ------------------------------------------- panel layout --------------------------------------- */

#[test]
fn the_panels_do_not_overlap_at_700_1000_and_1600_wide() {
    for (width, height, mode) in [(700.0_f32, 420.0_f32, Mode::Rail), (1000.0, 700.0, Mode::Full), (1600.0, 900.0, Mode::Full)] {
        let mut rig = Rig::new(width, height);
        let out = rig.settle();
        let tag = format!("{width}x{height}");
        assert_eq!(out.sidebar.mode, mode, "{tag}");
        let side = out.sidebar.rect;
        let expected = if mode == Mode::Rail { RAIL_WIDTH } else { DEFAULT_WIDTH };
        assert!((side.width() - expected).abs() < 1.0, "{tag}: sidebar is {} wide", side.width());
        assert!(side.left() <= 0.5, "{tag}");
        assert!(side.right() <= out.central.left() + 0.5, "{tag}: sidebar {side:?} overlaps the dock {:?}", out.central);
        assert!(out.strip.rect.bottom() <= side.top() + 0.5, "{tag}: the strip overlaps the sidebar");
        let bar = crate::theme::metrics(&rig.ctx).bar_h;
        assert!(side.bottom() <= height - bar + 1.0, "{tag}: the sidebar reaches into the bottom bar");
        assert!(out.central.width() > 300.0, "{tag}: the dock keeps room, {:?}", out.central);

        let l = out.strip;
        assert!(l.left.right() <= l.right.left() + 0.5, "{tag}: menus overlap the chips");
        assert!(l.right.right() <= width - 3.0 * crate::theme::title_bar::CAPTION_W + 0.5, "{tag}: chips reach the caption buttons");
        assert!(l.search.width() == 0.0 || (l.search.left() >= l.left.left() && l.search.right() <= l.left.right() + 0.5), "{tag}: the palette entry leaves the left slot");

        for (text, rect) in text_rects(&out.shapes) {
            assert!(rect.left() >= -1.0 && rect.right() <= width + 1.0, "{tag}: {text:?} leaves the window at {rect:?}");
            let in_column = rect.center().x < side.right() && rect.top() >= side.top() && rect.bottom() <= side.bottom();
            if in_column {
                assert!(rect.right() <= side.right() + 1.0, "{tag}: {text:?} overflows the sidebar at {rect:?}");
            }
        }
    }
}

#[test]
fn the_sidebar_keeps_its_rows_clear_of_each_other() {
    for (width, height) in [(1000.0_f32, 700.0_f32), (1600.0, 900.0)] {
        let mut rig = Rig::new(width, height);
        for section in SectionId::ALL {
            rig.app.shell.layout.layout.set_section_open(section.key(), true);
        }
        let out = rig.settle();
        let side = out.sidebar.rect;
        let rects: Vec<_> = text_rects(&out.shapes).into_iter().filter(|(t, r)| t != "\u{26a0}\u{fe0f}" && r.center().x < side.right() && r.top() > out.strip.rect.bottom()).collect();
        assert!(rects.len() > 10);
        for (i, (a, ra)) in rects.iter().enumerate() {
            for (b, rb) in &rects[i + 1..] {
                let overlap = ra.intersect(*rb);
                assert!(overlap.width() <= 1.0 || overlap.height() <= 1.0, "{a:?} {ra:?} overlaps {b:?} {rb:?}");
            }
        }
    }
}

#[test]
fn the_window_minimum_matches_the_narrowest_layout_tested() {
    assert_eq!((crate::ui::titlebar::MIN_WIDTH, crate::ui::titlebar::MIN_HEIGHT), (700.0, 400.0));
}

#[test]
fn growing_the_window_through_900_switches_from_the_rail_to_the_full_sidebar() {
    let mut rig = Rig::new(899.0, 700.0);
    assert_eq!(rig.settle().sidebar.mode, Mode::Rail);
    rig.size.0 = 900.0;
    let out = rig.settle();
    assert_eq!(out.sidebar.mode, Mode::Full);
    assert!(out.has("Camera roll"));
}

#[test]
fn a_closed_sidebar_takes_no_room() {
    let mut rig = Rig::new(1000.0, 700.0);
    let open = rig.settle();
    rig.app.shell.layout.layout.sidebar_open = false;
    let closed = rig.settle();
    assert_eq!(closed.sidebar.mode, Mode::Hidden);
    assert!(closed.central.width() > open.central.width() + 200.0);
    assert!(!closed.has("Camera roll"));
}

#[test]
fn the_bottom_bar_can_be_hidden() {
    let mut rig = Rig::new(1000.0, 700.0);
    rig.settle();
    let owner = crate::app::MAIN_EXPLORER_TAB;
    assert!(crate::ui::taskbar::last_layout(&rig.ctx, owner).is_some());
    rig.app.shell.layout.layout.bottom_bar_open = false;
    let out = rig.settle();
    let bar = crate::theme::metrics(&rig.ctx).bar_h;
    assert!(out.sidebar.rect.bottom() > 700.0 - bar + 1.0, "the sidebar takes the bar's room: {:?}", out.sidebar.rect);
}

/* --------------------------------------- resizing and persistence ----------------------------------- */

#[test]
fn dragging_the_edge_resizes_the_sidebar_within_its_limits() {
    let mut rig = Rig::new(1200.0, 700.0);
    let out = rig.settle();
    let edge = Pos2::new(out.sidebar.rect.right() - 1.0, 300.0);
    let button = |pressed| Event::PointerButton { pos: edge, button: eframe::egui::PointerButton::Primary, pressed, modifiers: Modifiers::NONE };
    rig.frame(vec![Event::PointerMoved(edge), button(true)]);
    rig.frame(vec![Event::PointerMoved(edge + vec2(80.0, 0.0))]);
    rig.frame(vec![Event::PointerMoved(edge + vec2(80.0, 0.0))]);
    let wide = rig.settle();
    assert!(rig.app.shell.layout.layout.sidebar_width > DEFAULT_WIDTH + 40.0, "{}", rig.app.shell.layout.layout.sidebar_width);
    assert!(wide.sidebar.rect.width() > out.sidebar.rect.width() + 40.0);
    rig.frame(vec![Event::PointerMoved(edge + vec2(900.0, 0.0))]);
    rig.frame(vec![Event::PointerMoved(edge + vec2(1500.0, 0.0))]);
    assert!(rig.app.shell.layout.layout.sidebar_width <= crate::ui::sidebar::layout::MAX_WIDTH);
    rig.frame(vec![Event::PointerButton { pos: edge, button: eframe::egui::PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE }]);
}

#[test]
fn the_layout_is_written_after_a_toggle_and_read_back_by_the_next_run() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("smart-media").join(FILE_NAME);
    let mut rig = Rig::new(1000.0, 700.0);
    rig.app.shell.layout = LayoutStore::at(path.clone());
    rig.settle();
    assert!(!path.exists(), "an untouched layout writes nothing");
    rig.frame(vec![key(Key::B, Modifiers::COMMAND)]);
    rig.settle();
    assert!(!rig.app.shell.layout.layout.sidebar_open);
    let saved: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(saved["sidebar_open"], false);
    assert_eq!(saved["version"], 1);
    let next = LayoutStore::at(path);
    assert!(!next.layout.sidebar_open);
}

#[test]
fn ctrl_b_toggles_the_sidebar_and_ctrl_k_the_palette() {
    let mut rig = Rig::new(1000.0, 700.0);
    rig.settle();
    rig.frame(vec![key(Key::B, Modifiers::COMMAND)]);
    assert!(!rig.app.shell.layout.layout.sidebar_open);
    rig.frame(vec![key(Key::B, Modifiers::COMMAND)]);
    assert!(rig.app.shell.layout.layout.sidebar_open);
    rig.frame(vec![key(Key::K, Modifiers::COMMAND)]);
    assert!(rig.app.shell.palette.open);
    let out = rig.settle();
    assert!(out.has("Library: All media"), "the palette lists commands");
    rig.frame(vec![key(Key::K, Modifiers::COMMAND)]);
    assert!(!rig.app.shell.palette.open);
}

#[test]
fn a_corrupt_layout_file_does_not_stop_the_shell() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(FILE_NAME);
    std::fs::write(&path, "}}}{").unwrap();
    let mut rig = Rig::new(1000.0, 700.0);
    rig.app.shell.layout = LayoutStore::at(path);
    let out = rig.settle();
    assert_eq!(out.sidebar.mode, Mode::Full);
    assert!((out.sidebar.rect.width() - DEFAULT_WIDTH).abs() < 1.0);
}

/* ------------------------------------------- the palette entry ------------------------------------- */

#[test]
fn the_strip_has_a_palette_entry_that_opens_the_palette() {
    let mut rig = Rig::new(1200.0, 700.0);
    let out = rig.settle();
    assert!(out.strip.search.width() > 100.0 && !out.strip.search_compact, "{:?}", out.strip.search);
    assert!(out.has("Search or run a command"));
    rig.click(out.strip.search.center());
    assert!(rig.app.shell.palette.open);
}

#[test]
fn the_palette_entry_shrinks_to_an_icon_in_a_narrow_window() {
    let mut rig = Rig::new(700.0, 420.0);
    let out = rig.settle();
    assert!(out.strip.search_compact, "{:?}", out.strip);
    assert!(!out.has("Search or run a command"));
    assert!(out.strip.search.width() > 10.0);
    rig.click(out.strip.search.center());
    assert!(rig.app.shell.palette.open);
}

#[test]
fn every_menu_sits_in_the_strip_and_the_search_follows_them() {
    let mut rig = Rig::new(1200.0, 700.0);
    let out = rig.settle();
    let xs: Vec<f32> = ["File", "View", "Tools", "Help"].iter().map(|m| out.pos(m).x).collect();
    assert!(xs.windows(2).all(|w| w[0] < w[1]), "{xs:?}");
    assert!(out.strip.search.left() > xs[3], "the palette entry comes after Help");
}

/* ------------------------------------------- actions and commands ------------------------------------ */

#[test]
fn shell_actions_flip_the_chrome_flags() {
    let mut rig = Rig::new(1000.0, 700.0);
    let ctx = rig.ctx.clone();
    for (action, read) in [
        (ShellAction::ToggleSidebar, (|l: &UiLayout| l.sidebar_open) as fn(&UiLayout) -> bool),
        (ShellAction::ToggleBottomBar, |l| l.bottom_bar_open),
        (ShellAction::ToggleRightRail, |l| l.right_rail_open),
    ] {
        assert!(read(&rig.app.shell.layout.layout));
        rig.app.run_shell_action(&ctx, action);
        assert!(!read(&rig.app.shell.layout.layout), "{action:?}");
        rig.app.run_shell_action(&ctx, action);
        assert!(read(&rig.app.shell.layout.layout), "{action:?}");
    }
}

#[test]
fn show_collections_opens_the_sidebar_and_its_collections_section() {
    let mut rig = Rig::new(1000.0, 700.0);
    let ctx = rig.ctx.clone();
    rig.app.shell.layout.layout.sidebar_open = false;
    rig.app.shell.layout.layout.set_section_open("collections", false);
    rig.app.run_shell_action(&ctx, ShellAction::ShowCollections);
    assert!(rig.app.shell.layout.layout.sidebar_open);
    assert!(rig.app.shell.layout.layout.is_section_open("collections", false));
    assert!(rig.settle().has("Holidays"));
}

#[test]
fn connect_actions_open_the_database_setup() {
    let mut rig = Rig::new(1000.0, 700.0);
    let ctx = rig.ctx.clone();
    assert!(!rig.app.context.show_db_select_modal);
    rig.app.run_shell_action(&ctx, ShellAction::ConnectLibrary);
    assert!(rig.app.context.show_db_select_modal);
    rig.app.context.show_db_select_modal = false;
    rig.app.context.db_ready = false;
    let out = rig.settle();
    rig.click(out.pos("Connect") + vec2(6.0, 6.0));
    assert!(rig.app.context.show_db_select_modal, "the sidebar button asks for the setup");
}

#[test]
fn about_and_new_collection_open_their_windows() {
    let mut rig = Rig::new(1000.0, 700.0);
    let ctx = rig.ctx.clone();
    rig.app.run_shell_action(&ctx, ShellAction::ShowAbout);
    rig.app.run_shell_action(&ctx, ShellAction::NewCollection);
    let out = rig.settle();
    assert!(out.has("About Smart Media") && out.has("Smart Media Explorer"), "{:?}", texts(&out.shapes));
    assert!(out.has("New collection") && out.has("Create") && out.has("Cancel"));
    let cancel = out.pos("Cancel") + vec2(6.0, 6.0);
    rig.click(cancel);
    assert!(rig.app.shell.new_collection.is_none());
}

#[test]
fn the_open_palette_command_opens_the_palette_with_the_query() {
    let mut rig = Rig::new(1000.0, 700.0);
    let result = rig.app.open_palette(Some("no clip".into()));
    assert_eq!(result, Ok("palette opened".to_string()));
    assert!(rig.app.shell.palette.open);
    assert_eq!(rig.app.shell.palette.query, "no clip");
    let out = rig.settle();
    assert!(out.has("Library: No CLIP"), "{:?}", texts(&out.shapes));
}

#[test]
fn palette_entries_cover_the_cached_sidebar_data() {
    let mut rig = Rig::new(1000.0, 700.0);
    let entries = rig.app.palette_entries();
    let ids: Vec<_> = entries.iter().map(|e| e.id.as_str()).collect();
    for id in ["link.place:/home/u/Camera", "link.col:Holidays", "link.tag:sunset", "link.cat:landscape", "open_scope.library", "shell.toggle_sidebar"] {
        assert!(ids.contains(&id), "{id} missing from {ids:?}");
    }
}

#[test]
fn a_chosen_palette_entry_reaches_the_command_bus() {
    let _serial = serial();
    take_matching(|_| true);
    let mut rig = Rig::new(1000.0, 700.0);
    let ctx = rig.ctx.clone();
    rig.app.run_effect(&ctx, crate::ui::palette::Effect::Command(AppCommand::ShowTaskCenter));
    rig.app.run_effect(&ctx, crate::ui::palette::Effect::Shell(ShellAction::ToggleBottomBar));
    assert_eq!(take_matching(|c| matches!(c, AppCommand::ShowTaskCenter)).len(), 1);
    assert!(!rig.app.shell.layout.layout.bottom_bar_open);
}

#[test]
fn clicking_a_sidebar_link_sends_the_open_command() {
    let _serial = serial();
    take_matching(|_| true);
    let mut rig = Rig::new(1000.0, 700.0);
    let out = rig.settle();
    rig.click(out.pos("Camera roll") + vec2(6.0, 6.0));
    let sent = take_matching(|c| matches!(c, AppCommand::OpenScope { .. }));
    assert_eq!(sent.len(), 1);
    assert_eq!(serde_json::to_value(&sent[0]).unwrap(), serde_json::json!({"cmd": "open_scope", "scope": "fs:/home/u/Camera", "target": "current"}));
}

#[test]
fn the_sidebar_devtools_window_reports_and_edits_the_layout() {
    use crate::ui::devtools::sidebar as dev;
    let mut rig = Rig::new(1000.0, 700.0);
    dev::set_open(&rig.ctx, true);
    rig.settle();
    let snapshot = dev::latest(&rig.ctx).expect("the shell published the window data");
    assert_eq!(snapshot.mode, Mode::Full);
    assert_eq!(snapshot.sections.len(), SectionId::ALL.len());
    assert_eq!(snapshot.sources.len(), crate::ui::sidebar::Source::ALL.len());
    assert!(snapshot.layout_json.contains("\"sidebar_open\": true"));
    dev::push_request(&rig.ctx, dev::Request::SetSection(SectionId::Tags, true));
    dev::push_request(&rig.ctx, dev::Request::ForceDisconnected(true));
    let out = rig.settle();
    assert!(rig.app.shell.layout.layout.is_section_open("tags", false));
    assert!(out.has("Library not connected"));
    dev::push_request(&rig.ctx, dev::Request::ResetLayout);
    rig.settle();
    assert_eq!(rig.app.shell.layout.layout, UiLayout::default());
}

#[test]
fn the_actions_devtools_window_lists_the_registry_and_runs_enabled_entries() {
    use crate::ui::devtools::actions as dev;
    let mut rig = Rig::new(1000.0, 700.0);
    dev::set_open(&rig.ctx, true);
    rig.ctx.data_mut(|d| d.insert_temp(eframe::egui::Id::new("ame_devtools_actions_filter"), "toggle sidebar".to_string()));
    rig.settle();
    let snapshot = dev::latest(&rig.ctx).expect("the shell published the registry");
    assert!(snapshot.entries.iter().any(|e| e.id == "shell.toggle_sidebar"));
    let out = rig.settle();
    assert!(out.has("Toggle sidebar") && out.has("Ctrl+B"), "{:?}", texts(&out.shapes).len());
    dev::push_request(&rig.ctx, "shell.toggle_bottom_bar".into());
    dev::push_request(&rig.ctx, "run.open".into());
    rig.settle();
    assert!(!rig.app.shell.layout.layout.bottom_bar_open, "an enabled entry ran");
}
