//! Window top strip: logo mark, File, View, Tools and Help menus, the palette entry, title, engine status chips and the caption buttons.

use eframe::egui::text::LayoutJob;
use eframe::egui::{Align, Color32, Context, Id, Layout, Rect, Response, Sense, StrokeKind, TextStyle, TextWrapMode, Ui, UiBuilder, WidgetText, pos2, vec2};
use egui_toast::ToastKind;

use crate::app::SmartMediaApp;
use crate::icons::{self, svg};
use crate::theme::chrome::{self, Chrome};
use crate::theme::controls::{Icon as KitIcon, MenuButton, StaticImage};
use crate::theme::title_bar::{Side, TitleBar};
use crate::status::StatusMeta;
use crate::ui::status::{self, ChipMode};

/// App id naming the saved window chrome preference, `<config dir>/smart-media/window_settings.json`.
pub const APP_ID: &str = "smart-media";
/// Smallest window width at which the strip, explorer toolbar and status bar do not overlap.
pub const MIN_WIDTH: f32 = 700.0;
/// Smallest window height.
pub const MIN_HEIGHT: f32 = 400.0;
/// Strip title when the free space is narrow.
const TITLE_SHORT: &str = "Smart Media";
/// Strip title when the free space allows the version.
const TITLE_LONG: &str = concat!("Smart Media ", env!("CARGO_PKG_VERSION"));
/// Space the title strip keeps clear on both sides of its title and between its slots.
const SLOT_GAP: f32 = 16.0;
/// Menus of the strip, left to right.
const MENU_NAMES: [&str; 4] = ["File", "View", "Tools", "Help"];
/// Widest the palette entry grows.
const SEARCH_MAX: f32 = 300.0;
/// Narrowest width at which the palette entry shows its hint text.
const SEARCH_WIDE_MIN: f32 = 150.0;
/// Width from which the palette entry shows its shortcut.
const SEARCH_SHORTCUT_MIN: f32 = 240.0;
/// Text of the palette entry.
const SEARCH_HINT: &str = "Search or run a command";

/// What the strip laid out in the last frame.
#[derive(Clone, Copy, Debug)]
pub struct StripLayout {
    pub rect: Rect,
    /// Space used by the File and View buttons.
    pub left: Rect,
    /// Space used by the status chips.
    pub right: Rect,
    /// Free space between the two slots.
    pub gap: Rect,
    pub chips: ChipMode,
    /// Title passed to the strip; empty with the system title bar.
    pub title: &'static str,
    /// The palette entry; zero-sized when the strip had no room for it.
    pub search: Rect,
    /// The palette entry shows only its icon.
    pub search_compact: bool,
    /// The palette entry was clicked this frame.
    pub search_clicked: bool,
}

fn layout_id() -> Id {
    Id::unique("ame.title_strip.layout")
}

/// The layout recorded by the last [`show`] on `ctx`.
pub fn last_layout(ctx: &Context) -> Option<StripLayout> {
    ctx.data(|d| d.get_temp::<StripLayout>(layout_id()))
}

/// The chrome for this run: the saved choice, `--system-titlebar` and `EGUI_THEME_KIT_SYSTEM_TITLEBAR`.
pub fn resolve_chrome() -> Chrome {
    chrome::resolve_from_process(APP_ID)
}

/// The window builder with the system decorations dropped for a frameless `chrome`.
pub fn viewport(chrome: &Chrome) -> eframe::egui::ViewportBuilder {
    chrome::viewport(
        eframe::egui::ViewportBuilder::default()
            .with_inner_size([1000.0, 750.0])
            .with_min_inner_size([MIN_WIDTH, MIN_HEIGHT])
            .with_drag_and_drop(true),
        chrome,
    )
}

/// The app logo as the kit's two-layer icon.
pub fn logo_mark() -> KitIcon {
    kit_icon(svg::AME_LOGO)
}

/// `icon` in the kit's icon type, keeping its duotone back layer.
pub fn kit_icon(icon: icons::Icon) -> KitIcon {
    let front = StaticImage::new(icon.uri(), icon.svg());
    match icon.duo_svg() {
        Some(back) => KitIcon::Duotone { front, back: StaticImage::new(icon.duo_uri(), back) },
        None => KitIcon::Image(front),
    }
}

/// Width of `text` in the body font, padded for the strip's medium-weight face.
fn text_width(ui: &Ui, text: &str) -> f32 {
    let galley = WidgetText::from(text).into_galley(ui, Some(TextWrapMode::Extend), f32::INFINITY, TextStyle::Body);
    galley.size().x * 1.08 + 2.0
}

/// Width the menu buttons take, with the gaps between them and before the palette entry.
fn menus_width(ui: &Ui) -> f32 {
    let menus: f32 = MENU_NAMES.iter().map(|name| MenuButton::new(name).size(ui).x).sum();
    menus + ui.spacing().item_spacing.x * MENU_NAMES.len() as f32
}

/// The richest chip layout that still leaves room for the menus, a palette entry sized for that layout and the title.
fn pick_chips(widths: &status::ChipWidths, slot: f32, menus_w: f32, reserve: f32) -> ChipMode {
    let room = |search: f32| (slot - menus_w - search - SLOT_GAP).max(0.0);
    if widths.full + reserve <= room(SEARCH_MAX) {
        ChipMode::Full
    } else if widths.icon + reserve <= room(SEARCH_WIDE_MIN) {
        ChipMode::Icon
    } else {
        ChipMode::Summary
    }
}

/// What the palette entry did.
struct SearchDraw {
    rect: Rect,
    compact: bool,
    clicked: bool,
}

/// Draws the palette entry in the room left after the menus, keeping `reserve` free for the title.
fn search_entry(ui: &mut Ui, reserve: f32) -> SearchDraw {
    let (t, m) = (crate::theme::tokens(ui.ctx()), crate::theme::metrics(ui.ctx()));
    let room = ui.available_width() - reserve;
    let compact = room < SEARCH_WIDE_MIN;
    let width = if compact { m.control_h } else { room.min(SEARCH_MAX) };
    if ui.available_width() < width {
        return SearchDraw { rect: Rect::NOTHING, compact, clicked: false };
    }
    let (rect, response) = ui.allocate_exact_size(vec2(width, m.control_h), Sense::click());
    response.widget_info(|| eframe::egui::WidgetInfo::labeled(eframe::egui::Role::Button, true, SEARCH_HINT));
    if ui.is_rect_visible(rect) {
        paint_search(ui, &response, &t, &m, compact);
    }
    let clicked = response.clone().on_hover_text("Search or run a command (Ctrl+K)").clicked();
    SearchDraw { rect, compact, clicked }
}

fn paint_search(ui: &Ui, response: &Response, t: &crate::theme::Tokens, m: &crate::theme::Metrics, compact: bool) {
    let rect = response.rect;
    let fill = if response.hovered() { t.hover } else { t.field };
    ui.painter().rect(rect, t.radius_sm, fill, eframe::egui::Stroke::new(1.0, t.field_border), StrokeKind::Inside);
    let icon_rect = Rect::from_center_size(pos2(if compact { rect.center().x } else { rect.left() + m.pad.x + m.icon_sm / 2.0 }, rect.center().y), vec2(m.icon_sm, m.icon_sm));
    crate::theme::controls::icons::SEARCH.paint(ui, icon_rect, m.icon_sm, t.text_dim);
    if compact {
        return;
    }
    let small = TextStyle::Small.resolve(ui.style());
    let mut right = rect.right() - m.pad.x;
    if rect.width() >= SEARCH_SHORTCUT_MIN {
        let galley = ui.painter().layout_no_wrap("Ctrl+K".to_string(), small, t.text_faint);
        right -= galley.size().x;
        ui.painter().galley(pos2(right, rect.center().y - galley.size().y / 2.0), galley, t.text_faint);
        right -= m.gap.x;
    }
    let left = icon_rect.right() + m.gap.x;
    let mut job = LayoutJob::simple_singleline(SEARCH_HINT.to_string(), TextStyle::Body.resolve(ui.style()), t.text_dim);
    job.wrap.max_width = (right - left).max(0.0);
    job.wrap.max_rows = 1;
    job.wrap.break_anywhere = true;
    job.wrap.overflow_character = Some('…');
    let galley = ui.painter().layout_job(job);
    ui.painter().galley(pos2(left, rect.center().y - galley.size().y / 2.0), galley, Color32::PLACEHOLDER);
}

/// Draws the status chips of `metas` right to left inside the slot, in the richest mode that leaves `menus_w` and `reserve` free.
fn chips_slot(ui: &mut Ui, menus_w: f32, reserve: f32, metas: &[StatusMeta]) -> ChipMode {
    let slot = ui.max_rect();
    let widths = status::chip_widths(ui, metas);
    let mode = pick_chips(&widths, slot.width(), menus_w, reserve);
    let width = match mode {
        ChipMode::Full => widths.full,
        ChipMode::Icon => widths.icon,
        ChipMode::Summary => widths.summary,
    };
    let bounds = Rect::from_min_max(pos2(slot.right() - width.min(slot.width()), slot.top()), slot.right_bottom());
    let mut child = ui.new_child(UiBuilder::new().max_rect(bounds).layout(Layout::right_to_left(Align::Center)));
    child.set_clip_rect(bounds.intersect(ui.clip_rect()));
    status::status_chips(&mut child, mode, metas);
    ui.allocate_rect(child.min_rect().intersect(bounds), Sense::hover());
    mode
}

/// Draws the strip as the first top panel: `menus` and the palette entry fill the left slot, the chips of `engines` the right.
pub fn show(ui: &mut Ui, frameless: bool, engines: &[StatusMeta], mut menus: impl FnMut(&mut Ui)) -> StripLayout {
    let ctx = ui.ctx().clone();
    let (short_w, long_w) = (text_width(ui, TITLE_SHORT), text_width(ui, TITLE_LONG));
    let free = last_layout(&ctx).map_or(0.0, |l| l.gap.width());
    let title = match (frameless, free >= long_w + 2.0 * SLOT_GAP) {
        (false, _) => "",
        (true, false) => TITLE_SHORT,
        (true, true) => TITLE_LONG,
    };
    let reserve = if frameless { short_w + 2.0 * SLOT_GAP } else { 0.0 };
    let menus_w = menus_width(ui);
    let mut bar = TitleBar::new("ame.title_strip", title).frameless(frameless);
    if frameless {
        bar = bar.mark(logo_mark());
    }
    let mut chips = ChipMode::Full;
    let mut search = SearchDraw { rect: Rect::NOTHING, compact: true, clicked: false };
    let out = bar.show(ui, |ui, side| {
        match side {
            Side::Right => chips = chips_slot(ui, menus_w, reserve, engines),
            Side::Left => {
                menus(ui);
                search = search_entry(ui, reserve);
            }
        }
        ui.min_rect()
    });
    let layout = StripLayout { rect: out.rect, left: out.left, right: out.right, gap: out.gap, chips, title, search: search.rect, search_compact: search.compact, search_clicked: search.clicked };
    ctx.data_mut(|d| d.insert_temp(layout_id(), layout));
    layout
}

impl SmartMediaApp {
    /// The window strip with the real menus and the registered engines.
    pub fn title_strip(&mut self, ui: &mut Ui) -> StripLayout {
        self.title_strip_with(ui, &crate::status::engine_snapshots())
    }

    /// The window strip showing `engines`; reports a failed save of the title bar choice.
    pub fn title_strip_with(&mut self, ui: &mut Ui, engines: &[StatusMeta]) -> StripLayout {
        let layout = show(ui, self.chrome.frameless, engines, |ui| {
            self.file_menu(ui);
            self.view_menu(ui);
            self.tools_menu(ui);
            self.help_menu(ui);
        });
        if layout.search_clicked {
            self.shell.palette.open_with(None);
        }
        if let Some(error) = chrome::take_save_error(ui.ctx()) {
            super::dialogs::toast(ToastKind::Error, format!("Could not save the title bar setting: {error}"));
        }
        layout
    }

    /// Resize zones and outline of the frameless window; call last in the frame.
    pub fn window_edges(&self, ui: &mut Ui) {
        if self.chrome.frameless {
            crate::theme::title_bar::window_edges(ui);
        }
    }
}

#[cfg(test)]
mod tests;
