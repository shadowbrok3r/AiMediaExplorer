//! Where the rail sits in the window: a resizable right panel, an overlay drawer on narrow windows, the keys and the toggle shared with the browsers.

use std::time::Instant;

use egui::{Area, Context, CursorIcon, Frame, Id, Key, Margin, Modifiers, Order, Panel, Rect, Sense, Ui, vec2};

use super::lightbox;
use super::panel::{self, RailEnv};
use super::prefs::{MIN_WIDTH, OVERLAY_BELOW, clamp_width};
use super::state::{Inspector, RowEdit, apply_edit};
use super::target::{Target, is_image_path, navigate};
use super::video;
use crate::command::ToastLevel;
use crate::ui::devtools::inspector as dev;
use crate::ui::file_table::FileExplorer;

/// Width of the strip at the rail's left edge that resizes it.
const HANDLE: f32 = 7.0;

/// What the host needs from the application.
pub struct HostArgs<'a> {
    /// The focused browser.
    pub ex: &'a mut FileExplorer,
    /// Title of the focused browser tab.
    pub tab: &'a str,
    pub assistant_open: &'a mut bool,
}

/// What the application must do after the rail drew.
#[derive(Debug, Default)]
pub struct HostOutput {
    pub edits: Vec<RowEdit>,
    pub notices: Vec<(ToastLevel, String)>,
    /// Where the rail was drawn.
    pub rect: Option<Rect>,
    pub overlay: bool,
}

/// Follows the browser's own preview flag: Space and the eye button flip it, the host turns that into the global state.
pub fn sync_flag(insp: &mut Inspector, ex: &mut FileExplorer, tab: &str) {
    let open = insp.is_open();
    let same_tab = insp.flag_sync.as_ref().is_some_and(|(t, _)| t == tab);
    let last = insp.flag_sync.as_ref().map(|(_, v)| *v);
    if !same_tab {
        ex.open_preview_pane = open;
        insp.flag_sync = Some((tab.to_string(), open));
        return;
    }
    if Some(ex.open_preview_pane) != last {
        let wanted = ex.open_preview_pane;
        insp.set_open(wanted);
        insp.flag_sync = Some((tab.to_string(), wanted));
    } else if ex.open_preview_pane != open {
        ex.open_preview_pane = open;
        insp.flag_sync = Some((tab.to_string(), open));
    }
}

fn run_dev_commands(ctx: &Context, insp: &mut Inspector, ex: &mut FileExplorer, target: &Target, now: Instant) {
    for command in dev::take_commands(ctx) {
        match command {
            dev::Command::ForceSave => insp.flush_now(now),
            dev::Command::Simulate(state) => insp.meta.simulate(state),
            dev::Command::EditCaption => {
                let text = format!("{} edited", insp.meta.draft.caption);
                insp.meta.set_text(super::meta::Field::Caption, text, now);
            }
            dev::Command::Next => {
                navigate(ex, true);
            }
            dev::Command::Previous => {
                navigate(ex, false);
            }
            dev::Command::ClearSelection => ex.clear_selection(),
            dev::Command::ToggleViewer => {
                if insp.lightbox.open {
                    insp.lightbox.open = false;
                } else {
                    insp.lightbox.show();
                }
            }
            dev::Command::ForceOverlay(mode) => insp.force_overlay = mode,
            dev::Command::FakeStream(on) => {
                if let Some(path) = target.focus_path() {
                    if on {
                        ex.streaming_interim.insert(path.to_string(), "A simulated description that streams in while the model writes it.".to_string());
                    } else {
                        ex.streaming_interim.remove(path);
                    }
                }
            }
            dev::Command::ResetPrefs => insp.prefs.update(now, |p| *p = Default::default()),
            dev::Command::ToggleOpen => {
                insp.toggle();
            }
        }
    }
}

/// Arrow keys step through the files, Enter opens the viewer and Esc leaves the rail, unless a text field had the focus.
fn rail_keys(ctx: &Context, insp: &mut Inspector, ex: &mut FileExplorer, target: &Target, was_typing: bool) {
    if !insp.rail_focused || insp.lightbox.open || was_typing || ctx.egui_wants_keyboard_input() {
        return;
    }
    let pressed = |key: Key| ctx.input_mut(|i| i.consume_key(Modifiers::NONE, key));
    if pressed(Key::ArrowRight) || pressed(Key::ArrowDown) {
        navigate(ex, true);
    }
    if pressed(Key::ArrowLeft) || pressed(Key::ArrowUp) {
        navigate(ex, false);
    }
    if pressed(Key::Enter) && target.focus_path().is_some_and(is_image_path) {
        insp.lightbox.show();
    }
    if pressed(Key::Escape) {
        insp.rail_focused = false;
    }
}

fn track_focus(ctx: &Context, insp: &mut Inspector, ex: &mut FileExplorer, rail: Rect) {
    let pressed_at = ctx.input(|i| if i.pointer.any_pressed() { i.pointer.interact_pos() } else { None });
    if let Some(pos) = pressed_at {
        let inside = rail.contains(pos);
        if inside && !insp.rail_focused {
            ex.details.set_keyboard_active(false);
        }
        insp.rail_focused = inside;
    }
    if ex.details.keyboard_active() {
        insp.rail_focused = false;
    }
}

fn draw_contents(ui: &mut Ui, insp: &mut Inspector, args: &mut HostArgs, target: &Target) -> bool {
    let mut env = RailEnv { assistant_open: &mut *args.assistant_open, close: false };
    panel::show(ui, insp, args.ex, &mut env, target);
    env.close
}

fn docked(ui: &mut Ui, insp: &mut Inspector, args: &mut HostArgs, target: &Target) -> (Rect, bool) {
    let width = clamp_width(insp.prefs.prefs.width);
    let mut close = false;
    let shown = Panel::right("ame_inspector_rail").resizable(false).exact_size(width).frame(Frame::side_top_panel(ui.style()).inner_margin(Margin::symmetric(8, 6))).show(ui, |ui| {
        close = draw_contents(ui, insp, args, target);
    });
    let rect = shown.response.rect;
    let grip = Rect::from_min_size(rect.left_top() - vec2(HANDLE / 2.0, 0.0), vec2(HANDLE, rect.height()));
    let response = ui.interact(grip, Id::new("ame_inspector_grip"), Sense::drag());
    if response.hovered() || response.dragged() {
        ui.ctx().set_cursor_icon(CursorIcon::ResizeHorizontal);
    }
    if response.dragged() {
        let wanted = clamp_width(width - response.drag_delta().x);
        if (wanted - width).abs() > 0.5 {
            insp.edit_prefs(|p| p.width = wanted);
        }
    }
    (rect, close)
}

fn drawer(ui: &mut Ui, insp: &mut Inspector, args: &mut HostArgs, target: &Target) -> (Rect, bool) {
    let free = ui.available_rect_before_wrap();
    let width = clamp_width(insp.prefs.prefs.width).min(free.width() * 0.94).max(MIN_WIDTH.min(free.width()));
    let rect = Rect::from_min_size(egui::pos2(free.right() - width, free.top()), vec2(width, free.height()));
    let mut close = false;
    let style = ui.style().clone();
    Area::new(Id::new("ame_inspector_drawer")).order(Order::Foreground).fixed_pos(rect.min).constrain(false).show(ui.ctx(), |ui| {
        Frame::popup(&style).inner_margin(Margin::symmetric(8, 6)).show(ui, |ui| {
            ui.set_width(width - 18.0);
            ui.set_height((rect.height() - 14.0).max(60.0));
            close = draw_contents(ui, insp, args, target);
        });
    });
    (rect, close)
}

/// Draws the rail for the focused browser and returns what the application has to apply.
pub fn show(ui: &mut Ui, insp: &mut Inspector, mut args: HostArgs) -> HostOutput {
    let ctx = ui.ctx().clone();
    let now = Instant::now();
    let mut out = HostOutput::default();
    let was_typing = ctx.egui_wants_keyboard_input();
    sync_flag(insp, args.ex, args.tab);
    if ctx.input_mut(|i| i.consume_key(Modifiers::COMMAND, Key::I)) {
        insp.toggle();
    }
    out.edits = insp.pump(&ctx, now);
    for edit in &out.edits {
        apply_edit(args.ex, edit);
    }
    let target = insp.begin_frame(&ctx, args.ex, now, insp.is_open());
    run_dev_commands(&ctx, insp, args.ex, &target, now);
    if !insp.is_open() {
        video::stop(&ctx, args.ex.viewer.scope);
        insp.lightbox.open = false;
        insp.rail_focused = false;
        insp.preview = Default::default();
    } else {
        let width = ui.available_width();
        let overlay = insp.force_overlay.unwrap_or(width < OVERLAY_BELOW);
        insp.overlay_active = overlay;
        let (rect, close) = if overlay { drawer(ui, insp, &mut args, &target) } else { docked(ui, insp, &mut args, &target) };
        out.rect = Some(rect);
        out.overlay = overlay;
        if close {
            insp.set_open(false);
        }
        track_focus(&ctx, insp, args.ex, rect);
        rail_keys(&ctx, insp, args.ex, &target, was_typing);
        lightbox::show(&ctx, insp, args.ex, &target, ctx.content_rect());
    }
    if dev::is_open(&ctx) {
        dev::publish(&ctx, dev::Snapshot::capture(insp, args.ex, &target, out.rect.map(|r| r.width())));
    }
    out.notices = insp.take_notices();
    out
}
