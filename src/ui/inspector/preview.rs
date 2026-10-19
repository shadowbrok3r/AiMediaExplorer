//! The preview area: the picture with fit modes, zoom and pan, or the video player, and the strip of controls below it.

use egui::{Color32, CornerRadius, Rect, Response, Sense, Stroke, StrokeKind, TextureId, Ui, Vec2, pos2, vec2};

use super::facts::FileFacts;
use super::state::{Inspector, PreviewReport};
use super::target::{is_dir, is_video_path, position};
use super::view::{PreviewMode, ViewState, ZOOM_STEP, image_size};
use crate::Thumbnail;
use crate::icons::{self, svg};
use crate::ui::file_table::FileExplorer;
use crate::ui::file_table::thumbs::{ThumbService, ThumbView, Tier, can_preview};

/// Height of the preview area relative to its width.
const ASPECT: f32 = 0.72;
const MIN_HEIGHT: f32 = 150.0;
const MAX_HEIGHT: f32 = 380.0;

/// Where the picture on screen comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// Folders and archives have no picture.
    None,
    Thumb,
    Preview,
    Pending,
    Missing,
}

impl Source {
    pub fn label(self) -> &'static str {
        match self {
            Source::None => "none",
            Source::Thumb => "thumbnail",
            Source::Preview => "preview",
            Source::Pending => "loading",
            Source::Missing => "missing",
        }
    }
}

/// The best picture available for `row`: the larger preview once decoded, else the stored thumbnail.
pub fn resolve(thumbs: &ThumbService, scope: u32, row: &Thumbnail) -> (ThumbView, Source) {
    let path = row.path.as_str();
    if path.is_empty() || is_dir(row) || row.file_type == "<ARCHIVE>" {
        return (ThumbView::Missing, Source::None);
    }
    let thumb = thumbs.request(scope, path, Tier::Thumb, 0.0);
    let preview = if can_preview(path) { thumbs.request(scope, path, Tier::Preview, 0.0) } else { ThumbView::Missing };
    if preview.is_ready() {
        (preview, Source::Preview)
    } else if thumb.is_ready() {
        (thumb, Source::Thumb)
    } else if thumb == ThumbView::Pending || preview == ThumbView::Pending {
        (ThumbView::Pending, Source::Pending)
    } else {
        (ThumbView::Missing, Source::Missing)
    }
}

/// Height of the preview area for a rail `width` points wide.
pub fn area_height(width: f32) -> f32 {
    (width * ASPECT).clamp(MIN_HEIGHT, MAX_HEIGHT)
}

/// What the user did in the preview.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PreviewEvent {
    #[default]
    None,
    OpenViewer,
    Navigate { forward: bool },
}

/// What the zoomable canvas reported.
#[derive(Clone, Copy, Debug, Default)]
pub struct CanvasOutput {
    pub double_clicked: bool,
    pub hovered: bool,
}

/// Draws `texture` in `rect` placed by `view` and applies wheel zoom and drag panning to it.
pub fn canvas(ui: &mut Ui, rect: Rect, view: &mut ViewState, texture: Option<(TextureId, Vec2)>, image: Vec2) -> CanvasOutput {
    let response: Response = ui.interact(rect, ui.id().with(("inspector_canvas", rect.min.x as i32, rect.min.y as i32)), Sense::click_and_drag());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, CornerRadius::same(4), ui.visuals().extreme_bg_color);
    let mut out = CanvasOutput { hovered: response.hovered(), double_clicked: response.double_clicked() };
    if let Some((id, _)) = texture {
        if response.hovered() {
            let scroll = ui.ctx().input_mut(|i| std::mem::take(&mut i.smooth_scroll_delta));
            if scroll.y != 0.0 {
                let cursor = response.hover_pos().map_or(Vec2::ZERO, |p| p - rect.center());
                view.zoom_at(image, rect.size(), cursor, ViewState::wheel_factor(scroll.y));
            }
        }
        if response.dragged() {
            view.pan += response.drag_delta();
            view.clamp(image, rect.size());
        }
        if response.hovered() || response.dragged() {
            ui.ctx().set_cursor_icon(if response.dragged() { egui::CursorIcon::Grabbing } else { egui::CursorIcon::Grab });
        }
        view.clamp(image, rect.size());
        let dest = view.dest(image, rect);
        painter.image(id, dest, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
    }
    painter.rect_stroke(rect, CornerRadius::same(4), Stroke::new(1.0, ui.visuals().widgets.noninteractive.bg_stroke.color), StrokeKind::Inside);
    out.hovered = response.hovered();
    out
}

fn placeholder(ui: &mut Ui, rect: Rect, text: &str) {
    ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, text, egui::TextStyle::Body.resolve(ui.style()), ui.visuals().weak_text_color());
}

/// The picture of `row` in a fixed-height area; returns the double click of the canvas.
pub fn picture(ui: &mut Ui, insp: &mut Inspector, ex: &FileExplorer, row: &Thumbnail, facts: Option<&FileFacts>) -> bool {
    let width = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(vec2(width, area_height(width)), Sense::hover());
    let (view, source) = resolve(&ex.viewer.thumbs, ex.viewer.scope, row);
    let mut report = PreviewReport { source: source.label(), texture: None, image: None };
    let mut double = false;
    match view {
        ThumbView::Ready { id, size } => {
            let known = facts.and_then(|f| f.dims);
            let image = image_size(size, known);
            report.texture = Some([size.x as usize, size.y as usize]);
            report.image = Some([image.x, image.y]);
            let out = canvas(ui, rect, &mut insp.view, Some((id, size)), image);
            double = out.double_clicked;
        }
        ThumbView::Pending => {
            canvas(ui, rect, &mut insp.view, None, Vec2::ZERO);
            placeholder(ui, rect, "Loading...");
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(120));
        }
        ThumbView::Missing => {
            canvas(ui, rect, &mut insp.view, None, Vec2::ZERO);
            placeholder(ui, rect, if source == Source::None { "No picture" } else { "No preview available" });
        }
    }
    insp.preview = report;
    double
}

/// The strip under the preview: previous and next, the fit modes, zoom and the viewer button.
pub fn controls(ui: &mut Ui, insp: &mut Inspector, ex: &FileExplorer, row: &Thumbnail, zoomable: bool) -> PreviewEvent {
    let mut event = PreviewEvent::None;
    let area = vec2(ui.available_width(), area_height(ui.available_width()));
    let image = vec2(insp.preview.image.map_or(0.0, |i| i[0]), insp.preview.image.map_or(0.0, |i| i[1]));
    ui.horizontal_wrapped(|ui| {
        let (n, total) = position(ex).unwrap_or((0, 0));
        if icons::icon_button(ui, svg::ARROW_LEFT, "Previous file (Left arrow)").clicked() {
            event = PreviewEvent::Navigate { forward: false };
        }
        ui.label(egui::RichText::new(format!("{n} of {total}")).weak());
        if icons::icon_button(ui, svg::ARROW_RIGHT, "Next file (Right arrow)").clicked() {
            event = PreviewEvent::Navigate { forward: true };
        }
        if zoomable {
            ui.separator();
            for mode in PreviewMode::ALL {
                if ui.selectable_label(insp.view.mode == mode, mode.label()).clicked() {
                    insp.view.set_mode(mode);
                    insp.edit_prefs(|p| p.preview_mode = mode);
                }
            }
            ui.separator();
            if icons::icon_button(ui, svg::MAGNIFYING_GLASS_MINUS, "Zoom out").clicked() {
                insp.view.zoom_at(image, area, Vec2::ZERO, 1.0 / ZOOM_STEP);
            }
            ui.label(egui::RichText::new(format!("{:.0}%", insp.view.percent(image, area))).monospace());
            if icons::icon_button(ui, svg::MAGNIFYING_GLASS_PLUS, "Zoom in").clicked() {
                insp.view.zoom_at(image, area, Vec2::ZERO, ZOOM_STEP);
            }
            if icons::icon_button(ui, svg::ARROWS_OUT_SIMPLE, "Open the viewer (Enter or double click)").clicked() && !is_video_path(&row.path) {
                event = PreviewEvent::OpenViewer;
            }
        }
    });
    event
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_preview_height_follows_the_width_within_limits() {
        assert_eq!(area_height(100.0), MIN_HEIGHT);
        assert_eq!(area_height(360.0), 360.0 * ASPECT);
        assert_eq!(area_height(4000.0), MAX_HEIGHT);
    }

    #[test]
    fn sources_have_distinct_labels() {
        let labels: std::collections::HashSet<_> = [Source::None, Source::Thumb, Source::Preview, Source::Pending, Source::Missing].into_iter().map(Source::label).collect();
        assert_eq!(labels.len(), 5);
    }
}
