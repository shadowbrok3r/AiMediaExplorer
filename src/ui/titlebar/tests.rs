use super::*;
use std::sync::{Arc, Mutex};

use eframe::egui::epaint::ClippedShape;
use eframe::egui::{Event, Pos2, RawInput, ViewportCommand, ViewportId, vec2};

use crate::database::ui_themes::tests::{Fixture, RT};
use crate::theme::Appearance;
use crate::theme::chrome::ChromePrefs;
use crate::theme::title_bar::CAPTION_W;
use crate::ui::file_table::FileExplorer;
use crate::ui::status::StatusMeta;
use crate::ui::test_support::{press_release, test_ctx, text_pos, texts};

/// Serializes the tests that save the title bar choice under the process-wide app id.
static CHROME_PREFS: Mutex<()> = Mutex::new(());
const MEMORY_ID: &str = "memory/ame-title-strip-tests";

/// The six engines the app registers, all idle.
fn engines() -> Vec<StatusMeta> {
    ["DB", "VISION", "CLIP", "RERANK", "QWEN-EDIT", "ASSIST"].into_iter().map(|name| StatusMeta { name, ..Default::default() }).collect()
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
        Self { app, ctx, width, _fixture: fixture, _runtime: runtime }
    }

    fn frame(&mut self, events: Vec<Event>) -> Frame {
        let input = RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(self.width, 600.0))),
            events,
            ..Default::default()
        };
        let app = &mut self.app;
        let mut layout = None;
        let engines = engines();
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
fn the_title_takes_the_version_only_when_it_fits() {
    let mut rig = Rig::new(true, 1600.0);
    let drawn = |frame: &Frame| frame.texts().into_iter().map(|(t, _)| t).find(|t| t == TITLE_LONG || t == TITLE_SHORT);
    let mut seen = std::collections::BTreeSet::<String>::new();
    for width in (MIN_WIDTH as u32..=1600).step_by(2) {
        rig.width = width as f32;
        let frame = rig.settle();
        let title = drawn(&frame).unwrap_or_else(|| panic!("{width}px draws no title: {:?}", frame.layout));
        if width >= 1000 {
            assert_eq!(title, TITLE_LONG, "{width}px: {:?}", frame.layout);
        }
        seen.insert(title);
    }
    assert!(seen.contains(TITLE_SHORT) && seen.contains(TITLE_LONG), "{seen:?}");
}

#[test]
fn file_and_view_open_under_the_strip_with_every_entry() {
    let mut rig = Rig::new(true, 1000.0);
    let l = rig.settle().layout;
    let strip_bottom = l.rect.bottom();

    let file = rig.click(pos2(l.left.left() + 12.0, l.rect.center().y));
    let texts = file.texts();
    for entry in ["Database", "AI", "Preferences", "Appearance…", "Exit"] {
        let (_, at) = texts.iter().find(|(t, _)| t == entry).unwrap_or_else(|| panic!("File menu lacks {entry}: {texts:?}"));
        assert!(at.y > strip_bottom, "{entry} is not under the strip");
    }

    let view = rig.click(pos2(l.left.right() - 12.0, l.rect.center().y));
    let texts = view.texts();
    for entry in ["Appearance…", "Egui Settings", "File Explorer", "Logs", "AI Refinements", "AI Assistant Window", "Icon gallery", "Devtools", "Custom title bar"] {
        let (_, at) = texts.iter().find(|(t, _)| t == entry).unwrap_or_else(|| panic!("View menu lacks {entry}: {texts:?}"));
        assert!(at.y > strip_bottom, "{entry} is not under the strip");
    }
    assert!(texts.iter().all(|(t, _)| t != "Exit"), "File closed when View opened");
}

#[test]
fn the_view_menu_switches_the_title_bar_live_and_remembers_it() {
    let _lock = CHROME_PREFS.lock().unwrap_or_else(|e| e.into_inner());
    chrome::resolve(MEMORY_ID, Vec::new(), |_| None);
    let mut rig = Rig::new(true, 1000.0);
    let l = rig.settle().layout;
    let view_at = pos2(l.left.right() - 12.0, l.rect.center().y);

    let open = rig.click(view_at);
    let row = open.text_pos("Custom title bar");
    let switched = rig.click(row + vec2(12.0, 6.0));
    assert!(!rig.app.chrome.frameless, "the click turned the custom bar off");
    assert_eq!(switched.layout.title, "", "the strip drops its title with the system bar");
    assert!(!ChromePrefs::load(MEMORY_ID).custom_title_bar, "the choice is saved");

    let view_again = pos2(switched.layout.left.right() - 12.0, l.rect.center().y);
    let again = rig.click(view_again);
    let row = again.text_pos("Custom title bar");
    let back = rig.click(row + vec2(12.0, 6.0));
    assert!(rig.app.chrome.frameless);
    assert_eq!(back.layout.title, TITLE_LONG);
    assert!(ChromePrefs::load(MEMORY_ID).custom_title_bar);
}

#[test]
fn switching_sends_the_decorations_command() {
    let _lock = CHROME_PREFS.lock().unwrap_or_else(|e| e.into_inner());
    chrome::resolve(MEMORY_ID, Vec::new(), |_| None);
    let mut rig = Rig::new(true, 1000.0);
    let l = rig.settle().layout;
    let open = rig.click(pos2(l.left.right() - 12.0, l.rect.center().y));
    let row = open.text_pos("Custom title bar");
    let click = rig.frame(press_release(row + vec2(12.0, 6.0)));
    assert!(click.commands.iter().any(|c| matches!(c, ViewportCommand::Decorations(true))), "{:?}", click.commands);
}
