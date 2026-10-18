//! Window top strip: logo mark, File and View menus, title, engine status chips and the caption buttons.

use eframe::egui::{Align, Context, Id, Layout, Rect, Sense, TextStyle, TextWrapMode, Ui, UiBuilder, WidgetText, pos2};
use egui_toast::ToastKind;

use crate::app::SmartMediaApp;
use crate::icons::{self, svg};
use crate::theme::chrome::{self, Chrome};
use crate::theme::controls::{Icon as KitIcon, MenuButton, StaticImage};
use crate::theme::title_bar::{Side, TitleBar};
use crate::ui::status::{self, ChipMode, StatusMeta};

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
}

fn layout_id() -> Id {
    Id::new("ame.title_strip.layout")
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

/// Width the File and View buttons take, with the gap between them.
fn menus_width(ui: &Ui) -> f32 {
    MenuButton::new("File").size(ui).x + MenuButton::new("View").size(ui).x + ui.spacing().item_spacing.x
}

/// Draws the status chips of `metas` right to left inside the slot, in the richest mode that leaves `menus_w` and `reserve` free.
fn chips_slot(ui: &mut Ui, menus_w: f32, reserve: f32, metas: &[StatusMeta]) -> ChipMode {
    let slot = ui.max_rect();
    let widths = status::chip_widths(ui, metas);
    let avail = (slot.width() - menus_w - SLOT_GAP).max(0.0);
    let mode = status::chip_mode(&widths, avail, reserve);
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

/// Draws the strip as the first top panel: `menus` fills the left slot, the chips of `engines` the right.
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
    let out = bar.show(ui, |ui, side| {
        match side {
            Side::Right => chips = chips_slot(ui, menus_w, reserve, engines),
            Side::Left => menus(ui),
        }
        ui.min_rect()
    });
    let layout = StripLayout { rect: out.rect, left: out.left, right: out.right, gap: out.gap, chips, title };
    ctx.data_mut(|d| d.insert_temp(layout_id(), layout));
    layout
}

impl SmartMediaApp {
    /// The window strip with the real File and View menus and the registered engines.
    pub fn title_strip(&mut self, ui: &mut Ui) -> StripLayout {
        self.title_strip_with(ui, &status::engine_snapshots())
    }

    /// The window strip showing `engines`; reports a failed save of the title bar choice.
    pub fn title_strip_with(&mut self, ui: &mut Ui, engines: &[StatusMeta]) -> StripLayout {
        let layout = show(ui, self.chrome.frameless, engines, |ui| {
            self.file_menu(ui);
            self.view_menu(ui);
        });
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
