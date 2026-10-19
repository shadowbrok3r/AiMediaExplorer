//! The viewer overlay: the picture over the whole window with arrows, zoom and Esc.

use egui::{Area, Color32, Context, Id, Key, Modifiers, Order, Rect, RichText, Sense, Ui, UiBuilder, Vec2, vec2};

use super::preview::{canvas, resolve};
use super::state::Inspector;
use super::target::{Target, is_dir, is_image_path, navigate, position};
use super::view::{PreviewMode, ViewState, ZOOM_STEP, image_size};
use crate::icons::{self, svg};
use crate::ui::file_table::FileExplorer;
use crate::ui::file_table::thumbs::ThumbView;

const BAR_HEIGHT: f32 = 42.0;

/// What a key press asks the viewer to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewerKey {
    Close,
    Previous,
    Next,
    ZoomIn,
    ZoomOut,
    Reset,
}

/// The viewer keys pressed this frame, consumed so nothing else reacts to them.
pub fn take_keys(ctx: &Context) -> Vec<ViewerKey> {
    const KEYS: [(Key, ViewerKey); 8] = [
        (Key::Escape, ViewerKey::Close),
        (Key::ArrowLeft, ViewerKey::Previous),
        (Key::ArrowRight, ViewerKey::Next),
        (Key::Plus, ViewerKey::ZoomIn),
        (Key::Equals, ViewerKey::ZoomIn),
        (Key::Minus, ViewerKey::ZoomOut),
        (Key::Num0, ViewerKey::Reset),
        (Key::Num1, ViewerKey::Reset),
    ];
    ctx.input_mut(|i| KEYS.iter().filter(|(key, _)| i.consume_key(Modifiers::NONE, *key)).map(|(_, action)| *action).collect())
}

fn bar(ui: &mut Ui, insp: &mut Inspector, ex: &mut FileExplorer, name: &str, area: Vec2, image: Vec2) {
    ui.horizontal_centered(|ui| {
        ui.add_space(8.0);
        if icons::icon_button(ui, svg::X, "Close (Esc)").clicked() {
            insp.lightbox.open = false;
        }
        ui.label(RichText::new(name).color(Color32::WHITE).strong());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add_space(8.0);
            if icons::icon_button(ui, svg::ARROW_RIGHT, "Next (Right arrow)").clicked() {
                navigate(ex, true);
            }
            let (n, total) = position(ex).unwrap_or((0, 0));
            ui.label(RichText::new(format!("{n} of {total}")).color(Color32::WHITE));
            if icons::icon_button(ui, svg::ARROW_LEFT, "Previous (Left arrow)").clicked() {
                navigate(ex, false);
            }
            ui.separator();
            if icons::icon_button(ui, svg::MAGNIFYING_GLASS_PLUS, "Zoom in (+)").clicked() {
                insp.lightbox.view.zoom_at(image, area, Vec2::ZERO, ZOOM_STEP);
            }
            ui.label(RichText::new(format!("{:.0}%", insp.lightbox.view.percent(image, area))).monospace().color(Color32::WHITE));
            if icons::icon_button(ui, svg::MAGNIFYING_GLASS_MINUS, "Zoom out (-)").clicked() {
                insp.lightbox.view.zoom_at(image, area, Vec2::ZERO, 1.0 / ZOOM_STEP);
            }
            for mode in [PreviewMode::Actual, PreviewMode::Fit] {
                if ui.selectable_label(insp.lightbox.view.mode == mode, mode.label()).clicked() {
                    insp.lightbox.view.set_mode(mode);
                }
            }
        });
    });
}

/// Draws the viewer over `content` when it is open; keys are handled here.
pub fn show(ctx: &Context, insp: &mut Inspector, ex: &mut FileExplorer, target: &Target, content: Rect) {
    if !insp.lightbox.open {
        return;
    }
    let row = match target {
        Target::Single(row) if !is_dir(row) && is_image_path(&row.path) => Some(row.clone()),
        Target::Multi { focus: Some(row), .. } if !is_dir(row) && is_image_path(&row.path) => Some(row.clone()),
        _ => None,
    };
    let Some(row) = row else {
        insp.lightbox.open = false;
        return;
    };
    for key in take_keys(ctx) {
        match key {
            ViewerKey::Close => insp.lightbox.open = false,
            ViewerKey::Previous => {
                navigate(ex, false);
            }
            ViewerKey::Next => {
                navigate(ex, true);
            }
            ViewerKey::ZoomIn | ViewerKey::ZoomOut => {
                let image = insp.preview.image.map_or(Vec2::ZERO, |i| vec2(i[0], i[1]));
                let factor = if key == ViewerKey::ZoomIn { ZOOM_STEP } else { 1.0 / ZOOM_STEP };
                let area = vec2(content.width(), content.height() - BAR_HEIGHT);
                insp.lightbox.view.zoom_at(image, area, Vec2::ZERO, factor);
            }
            ViewerKey::Reset => insp.lightbox.view = ViewState::default(),
        }
    }
    if !insp.lightbox.open {
        return;
    }
    ex.details.set_keyboard_active(false);
    let facts = insp.facts.get(&row.path).cloned();
    let (view, _) = resolve(&ex.viewer.thumbs, ex.viewer.scope, &row);
    let name = row.filename.clone();
    Area::new(Id::new("ame_inspector_lightbox")).order(Order::Foreground).fixed_pos(content.min).constrain(false).show(ctx, |ui| {
        ui.set_clip_rect(content);
        ui.allocate_rect(content, Sense::hover());
        ui.painter().rect_filled(content, 0.0, Color32::from_black_alpha(238));
        let canvas_rect = Rect::from_min_max(content.min + vec2(0.0, BAR_HEIGHT), content.max).shrink(12.0);
        let (texture, image) = match view {
            ThumbView::Ready { id, size } => (Some((id, size)), image_size(size, facts.as_ref().and_then(|f| f.dims))),
            _ => (None, Vec2::ZERO),
        };
        let out = canvas(ui, canvas_rect, &mut insp.lightbox.view, texture, image);
        if out.double_clicked {
            let next = if insp.lightbox.view.mode == PreviewMode::Actual { PreviewMode::Fit } else { PreviewMode::Actual };
            insp.lightbox.view.set_mode(next);
        }
        if texture.is_none() {
            ui.painter().text(canvas_rect.center(), egui::Align2::CENTER_CENTER, "Loading...", egui::TextStyle::Body.resolve(ui.style()), Color32::WHITE);
        }
        let bar_rect = Rect::from_min_size(content.min, vec2(content.width(), BAR_HEIGHT));
        ui.scope_builder(UiBuilder::new().max_rect(bar_rect), |ui| bar(ui, insp, ex, &name, canvas_rect.size(), image));
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn press(ctx: &Context, key: Key) -> Vec<ViewerKey> {
        let input = egui::RawInput { events: vec![egui::Event::Key { key, physical_key: Some(key), pressed: true, repeat: false, modifiers: Modifiers::NONE }], ..Default::default() };
        let mut keys = Vec::new();
        ctx.run_ui(input, |ui| {
            keys = take_keys(ui.ctx());
        })
        .drop_without_applying_deltas();
        keys
    }

    #[test]
    fn the_viewer_keys_map_to_actions() {
        let ctx = Context::default();
        assert_eq!(press(&ctx, Key::Escape), vec![ViewerKey::Close]);
        assert_eq!(press(&ctx, Key::ArrowLeft), vec![ViewerKey::Previous]);
        assert_eq!(press(&ctx, Key::ArrowRight), vec![ViewerKey::Next]);
        assert_eq!(press(&ctx, Key::Plus), vec![ViewerKey::ZoomIn]);
        assert_eq!(press(&ctx, Key::Minus), vec![ViewerKey::ZoomOut]);
        assert_eq!(press(&ctx, Key::Num0), vec![ViewerKey::Reset]);
        assert!(press(&ctx, Key::A).is_empty());
    }
}
