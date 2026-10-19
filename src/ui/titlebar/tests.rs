use super::*;
use std::sync::{Arc, Mutex};

use eframe::egui::epaint::ClippedShape;
use eframe::egui::{Event, Pos2, RawInput, ViewportCommand, ViewportId, vec2};

use crate::database::ui_themes::tests::{Fixture, RT};
use crate::theme::Appearance;
use crate::theme::chrome::ChromePrefs;
use crate::theme::title_bar::CAPTION_W;
use crate::ui::file_table::FileExplorer;
use crate::status::StatusMeta;
use crate::ui::test_support::{press_release, test_ctx, text_pos, texts};

/// Serializes the tests that save the title bar choice under the process-wide app id.
static CHROME_PREFS: Mutex<()> = Mutex::new(());
const MEMORY_ID: &str = "memory/ame-title-strip-tests";

/// The six core engines, all idle.
fn engines() -> Vec<StatusMeta> {
    ["DB", "VISION", "CLIP", "RERANK", "QWEN-EDIT", "ASSIST"].into_iter().map(|name| StatusMeta { name, ..Default::default() }).collect()
}

/// Every engine the app registers, the gateway included, all idle.
fn every_engine() -> Vec<StatusMeta> {
    crate::status::INDICATOR_ORDER.into_iter().map(|name| StatusMeta { name, ..Default::default() }).collect()
}

struct Frame {
    layout: StripLayout,
    shapes: Vec<ClippedShape>,
    commands: Vec<ViewportCommand>,
}

impl Frame {
    fn texts(&self) -> Vec<(String, Pos2)> {
        texts(&self.shapes)
    }

    fn text_pos(&self, text: &str) -> Pos2 {
        text_pos(&self.shapes, text)
    }
}

/// The real app over an in-memory theme store, drawing only the title strip.
struct Rig {
    app: SmartMediaApp,
    ctx: Context,
    width: f32,
    height: f32,
    all_engines: bool,
    _fixture: Fixture,
    _runtime: tokio::runtime::EnterGuard<'static>,
}

impl Rig {
    fn new(frameless: bool, width: f32) -> Self {
        let fixture = Fixture::new();
        let runtime = RT.enter();
        let ctx = test_ctx();
        let appearance = Appearance::attach(&ctx, None, Arc::new(fixture.store()));
        let chrome = Chrome { frameless, available: true };
        let app = SmartMediaApp::assemble(ctx.clone(), appearance, chrome, FileExplorer::new(true));
        Self { app, ctx, width, height: 600.0, all_engines: false, _fixture: fixture, _runtime: runtime }
    }

    fn frame(&mut self, events: Vec<Event>) -> Frame {
        let input = RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(self.width, self.height))),
            events,
            ..Default::default()
        };
        let app = &mut self.app;
        let mut layout = None;
        let engines = if self.all_engines { every_engine() } else { engines() };
        let out = self.ctx.run_ui(input, |ui| layout = Some(app.title_strip_with(ui, &engines)));
        let commands = out.viewport_output.get(&ViewportId::ROOT).map(|v| v.commands.clone()).unwrap_or_default();
        let shapes = out.shapes.clone();
        out.drop_without_applying_deltas();
        Frame { layout: layout.expect("the strip ran"), shapes, commands }
    }

    fn settle(&mut self) -> Frame {
        (0..3).map(|_| self.frame(Vec::new())).last().unwrap()
    }

    fn click(&mut self, pos: Pos2) -> Frame {
        self.frame(press_release(pos));
        self.settle()
    }
}

#[test]
fn the_logo_keeps_its_duotone_layer() {
    match logo_mark() {
        KitIcon::Duotone { front, back } => {
            assert_eq!(front.uri, svg::AME_LOGO.uri());
            assert_eq!(back.uri, svg::AME_LOGO.duo_uri());
        }
        other => panic!("expected a duotone icon, got {other:?}"),
    }
    assert_eq!(matches!(kit_icon(svg::FOLDER), KitIcon::Duotone { .. }), svg::FOLDER.has_duo());
}

#[test]
fn the_viewport_is_frameless_only_for_a_frameless_chrome() {
    let custom = viewport(&Chrome { frameless: true, available: true });
    assert_eq!(custom.decorations, Some(false));
    assert_eq!(custom.inner_size, Some(vec2(1000.0, 750.0)));
    assert_eq!(custom.min_inner_size, Some(vec2(MIN_WIDTH, MIN_HEIGHT)));
    assert_eq!(custom.drag_and_drop, Some(true));
    let system = viewport(&Chrome { frameless: false, available: true });
    assert_eq!(system.decorations, Some(true));
    assert_eq!(system.min_inner_size, Some(vec2(MIN_WIDTH, MIN_HEIGHT)));
    assert_eq!(viewport(&Chrome::UNAVAILABLE).decorations, Some(true));
}

#[test]
fn the_system_titlebar_flag_wins_for_one_launch_and_other_flags_still_parse() {
    let args = |list: &[&str]| list.iter().map(|a| a.to_string()).collect::<Vec<_>>();
    let saved = ChromePrefs { custom_title_bar: true };
    let forced = chrome::resolve_with(saved, args(&["--system-titlebar"]), |_| None);
    assert_eq!((forced.frameless, forced.available), (false, true));
    assert!(chrome::resolve_with(saved, args(&["--db-info"]), |_| None).frameless);
    let env = |key: &str| (key == chrome::SYSTEM_TITLEBAR_ENV).then(|| "1".to_string());
    assert!(!chrome::resolve_with(saved, args(&[]), env).frameless);
    assert!(!chrome::resolve_with(ChromePrefs { custom_title_bar: false }, args(&[]), |_| None).frameless);
    assert_eq!(crate::database::cli::run(&args(&["--system-titlebar"])), None);
}

#[test]
fn the_strip_never_overlaps_from_640_to_1600() {
    for frameless in [true, false] {
        let mut last_rank = 0;
        for width in [640.0_f32, 700.0, 800.0, 1000.0, 1200.0, 1400.0, 1600.0] {
            let mut rig = Rig::new(frameless, width);
            let l = rig.settle().layout;
            let tag = format!("frameless {frameless} at {width}px: {l:?}");
            let right_edge = if frameless { width - 3.0 * CAPTION_W } else { width };
            assert!(l.left.right() <= l.right.left() + 0.5, "menus overlap the chips: {tag}");
            assert!(l.right.right() <= right_edge + 0.5, "chips reach the caption buttons: {tag}");
            assert!(l.left.left() >= -0.5 && l.gap.width() >= 0.0, "{tag}");
            assert!(l.left.width() >= 2.0 * 40.0, "menus are squeezed: {tag}");
            let rank = match l.chips {
                ChipMode::Summary => 0,
                ChipMode::Icon => 1,
                ChipMode::Full => 2,
            };
            assert!(rank >= last_rank, "a wider window showed fewer chips: {tag}");
            last_rank = rank;
            assert_eq!(l.title.is_empty(), !frameless, "{tag}");
        }
        assert_eq!(last_rank, 2, "every chip is labelled at 1600px");
    }
}

#[test]
fn the_strip_with_every_engine_chip_never_overlaps_and_folds_to_icons_before_the_summary() {
    let mut last_rank = 0;
    for width in [640.0_f32, 700.0, 800.0, 1000.0, 1200.0, 1400.0, 1700.0, 2000.0] {
        let mut rig = Rig::new(true, width);
        rig.all_engines = true;
        let l = rig.settle().layout;
        let tag = format!("{width}px: {l:?}");
        assert!(l.left.right() <= l.right.left() + 0.5, "menus overlap the chips: {tag}");
        assert!(l.right.right() <= width - 3.0 * CAPTION_W + 0.5, "chips reach the caption buttons: {tag}");
        let rank = match l.chips {
            ChipMode::Summary => 0,
            ChipMode::Icon => 1,
            ChipMode::Full => 2,
        };
        assert!(rank >= last_rank, "a wider window showed fewer chips: {tag}");
        last_rank = rank;
        if l.chips == ChipMode::Icon {
            assert!(l.right.width() >= 7.0 * 14.0, "seven icons need seven icon widths: {tag}");
        }
    }
    assert_eq!(last_rank, 2, "every chip is labelled at 2000px");
}

#[test]
fn the_title_takes_the_version_only_when_it_fits() {
    let mut rig = Rig::new(true, 1600.0);
    let drawn = |frame: &Frame| frame.texts().into_iter().map(|(t, _)| t).find(|t| t == TITLE_LONG || t == TITLE_SHORT);
    let mut seen = std::collections::BTreeSet::<String>::new();
    for width in (MIN_WIDTH as u32..=1600).step_by(2) {
        rig.width = width as f32;
        let frame = rig.settle();
        let title = drawn(&frame).unwrap_or_else(|| panic!("{width}px draws no title: {:?}", frame.layout));
        if width >= 1500 {
            assert_eq!(title, TITLE_LONG, "{width}px: {:?}", frame.layout);
        }
        seen.insert(title);
    }
    assert!(seen.contains(TITLE_SHORT) && seen.contains(TITLE_LONG), "{seen:?}");
}

/// Opens `menu` and returns the frame with its entries; each entry must draw under the strip.
fn open_menu(rig: &mut Rig, menu: &str, entries: &[&str]) -> Frame {
    let l = rig.settle().layout;
    let at = rig.settle().text_pos(menu) + vec2(6.0, 6.0);
    let frame = rig.click(at);
    let texts = frame.texts();
    for entry in entries {
        let (_, at) = texts.iter().find(|(t, _)| t == entry).unwrap_or_else(|| panic!("{menu} menu lacks {entry}: {texts:?}"));
        assert!(at.y > l.rect.bottom(), "{entry} is not under the strip");
    }
    frame
}

/// Closes any open menu by clicking empty space.
fn close_menus(rig: &mut Rig) {
    rig.click(pos2(500.0, 500.0));
}

#[test]
fn file_view_tools_and_help_open_under_the_strip_with_every_entry() {
    let mut rig = Rig::new(true, 1200.0);
    open_menu(&mut rig, "File", &["Open folder...", "Scan folder...", "Database", "Preferences", "Quit"]);
    let view = open_menu(&mut rig, "View", &["Sidebar", "Bottom bar", "Right rail", "Appearance…", "Custom title bar", "Tabs", "AI Assistant Window", "Devtools"]);
    assert!(view.texts().iter().all(|(t, _)| t != "Quit"), "File closed when View opened");
    open_menu(&mut rig, "Tools", &["Scan folder...", "AI processing", "Refine (AI, DB-only)", "Image edit", "Collections", "Scan settings"]);
    open_menu(&mut rig, "Help", &["Logs", "Command palette", "About"]);
}

#[test]
fn preferences_and_appearance_ask_for_the_settings_tab_through_the_command() {
    use crate::command::bus::testing::{serial, take_matching};
    use crate::command::{AppCommand, Section};
    let _serial = serial();
    take_matching(|_| true);
    let mut rig = Rig::new(true, 1200.0);
    let file = open_menu(&mut rig, "File", &["Preferences"]);
    rig.click(file.text_pos("Preferences") + vec2(6.0, 6.0));
    assert_eq!(take_matching(|c| matches!(c, AppCommand::ShowSettings { section: None })).len(), 1);
    close_menus(&mut rig);
    let view = open_menu(&mut rig, "View", &["Appearance…"]);
    rig.click(view.text_pos("Appearance…") + vec2(6.0, 6.0));
    assert_eq!(take_matching(|c| matches!(c, AppCommand::ShowSettings { section: Some(Section::Appearance) })).len(), 1);
}

#[test]
fn submenus_hold_the_rest_of_the_old_menu_actions() {
    let mut rig = Rig::new(true, 1200.0);
    rig.height = 1400.0;
    let cases: [(&str, &str, &[&str]); 6] = [
        ("File", "Database", &["View Entire Database", "Database backup and restore", "Export backup...", "Import backup...", "Choose database folder..."]),
        ("View", "Tabs", &["File Explorer", "Logs", "AI Refinements"]),
        ("View", "Devtools", &["Icon gallery", "Theme panel", "Settings store", "Wizard", "Tasks", "Command log", "Perf", "Row store", "Collections", "Video", "Sidebar", "Actions"]),
        ("Tools", "AI processing", &["Enable AI search", "Enable auto indexing", "Enable auto CLIP", "Generate CLIP for missing files", "Generate CLIP for the previewed file", "Count files missing CLIP"]),
        ("Tools", "Collections", &["Show collections", "New collection..."]),
        ("Tools", "Scan settings", &["FoundBatch size"]),
    ];
    for (menu, item, entries) in cases {
        close_menus(&mut rig);
        let open = open_menu(&mut rig, menu, &[item]);
        let at = open.text_pos(item) + vec2(6.0, 6.0);
        let sub = rig.click(at);
        let texts = sub.texts();
        for entry in entries {
            assert!(texts.iter().any(|(t, _)| t == entry), "{menu} > {item} lacks {entry}: {texts:?}");
        }
    }
}

#[test]
fn the_devtools_submenu_scrolls_inside_a_short_window() {
    let mut rig = Rig::new(true, 1200.0);
    rig.height = 400.0;
    let open = open_menu(&mut rig, "View", &["Devtools"]);
    let sub = rig.click(open.text_pos("Devtools") + vec2(6.0, 6.0));
    let texts = sub.texts();
    assert!(texts.iter().any(|(t, _)| t == "Icon gallery"));
    assert!(texts.iter().all(|(t, _)| t != "ZeroClaw usage"), "the last entry is below the fold");
    let over = texts.iter().find(|(t, _)| t == "Theme panel").map(|(_, at)| *at).unwrap() + vec2(6.0, 6.0);
    rig.frame(vec![Event::PointerMoved(over)]);
    let scrolled = (0..4).map(|_| rig.frame(vec![Event::MouseWheel { unit: egui::MouseWheelUnit::Point, delta: vec2(0.0, -2000.0), phase: egui::TouchPhase::Move, source: egui::MouseWheelSource::Wheel, modifiers: egui::Modifiers::NONE }])).last().unwrap();
    assert!(scrolled.texts().iter().any(|(t, _)| t == "ZeroClaw usage"), "scrolling reaches the last entry");
}

#[test]
fn the_view_menu_shows_which_chrome_is_on() {
    let mut rig = Rig::new(true, 1200.0);
    let open = open_menu(&mut rig, "View", &["Sidebar"]);
    let before = rig.app.shell.layout.layout.sidebar_open;
    let toggled = rig.click(open.text_pos("Sidebar") + vec2(6.0, 6.0));
    assert_ne!(rig.app.shell.layout.layout.sidebar_open, before, "the entry flips the sidebar");
    assert!(toggled.layout.rect.height() > 0.0);
}

#[test]
fn the_palette_entry_opens_the_palette_and_shows_its_shortcut_when_wide() {
    let mut rig = Rig::new(true, 1600.0);
    let l = rig.settle();
    assert!(l.texts().iter().any(|(t, _)| t == "Search or run a command"));
    assert!(l.texts().iter().any(|(t, _)| t == "Ctrl+K"));
    assert!(l.layout.search.width() > 200.0 && !l.layout.search_compact);
    assert!(!rig.app.shell.palette.open);
    rig.click(l.layout.search.center());
    assert!(rig.app.shell.palette.open);
}

#[test]
fn the_palette_entry_gives_way_before_the_menus_do() {
    let mut last_width = 0.0_f32;
    for width in [640.0_f32, 700.0, 800.0, 1000.0, 1200.0, 1600.0] {
        let mut rig = Rig::new(true, width);
        let l = rig.settle().layout;
        assert!(l.search.width() == 0.0 || l.search.left() >= l.left.left(), "{width}: {l:?}");
        assert!(l.search.width() + 0.5 >= last_width, "a wider window shrank the palette entry at {width}px: {l:?}");
        last_width = l.search.width();
        assert_eq!(l.search_compact, l.search.width() < 150.0, "{width}: {l:?}");
        let menus_right = l.search.left();
        assert!(menus_right >= l.left.left() + 4.0 * 40.0 || l.search.width() == 0.0, "{width}: the four menus keep their room");
    }
}

#[test]
fn the_view_menu_switches_the_title_bar_live_and_remembers_it() {
    let _lock = CHROME_PREFS.lock().unwrap_or_else(|e| e.into_inner());
    chrome::resolve(MEMORY_ID, Vec::new(), |_| None);
    let mut rig = Rig::new(true, 1000.0);
    rig.settle();

    let open = open_menu(&mut rig, "View", &["Custom title bar"]);
    let row = open.text_pos("Custom title bar");
    let switched = rig.click(row + vec2(12.0, 6.0));
    assert!(!rig.app.chrome.frameless, "the click turned the custom bar off");
    assert_eq!(switched.layout.title, "", "the strip drops its title with the system bar");
    assert!(!ChromePrefs::load(MEMORY_ID).custom_title_bar, "the choice is saved");

    let again = open_menu(&mut rig, "View", &["Custom title bar"]);
    let row = again.text_pos("Custom title bar");
    let back = rig.click(row + vec2(12.0, 6.0));
    assert!(rig.app.chrome.frameless);
    assert!(back.layout.title == TITLE_LONG || back.layout.title == TITLE_SHORT, "{}", back.layout.title);
    assert!(ChromePrefs::load(MEMORY_ID).custom_title_bar);
}

#[test]
fn switching_sends_the_decorations_command() {
    let _lock = CHROME_PREFS.lock().unwrap_or_else(|e| e.into_inner());
    chrome::resolve(MEMORY_ID, Vec::new(), |_| None);
    let mut rig = Rig::new(true, 1000.0);
    let open = open_menu(&mut rig, "View", &["Custom title bar"]);
    let row = open.text_pos("Custom title bar");
    let click = rig.frame(press_release(row + vec2(12.0, 6.0)));
    assert!(click.commands.iter().any(|c| matches!(c, ViewportCommand::Decorations(true))), "{:?}", click.commands);
}
