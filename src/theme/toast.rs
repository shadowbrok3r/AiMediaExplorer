//! Theme-native look for `egui-toast` notifications, registered through its custom-contents hook.

use eframe::egui::{
    Button, Color32, CornerRadius, CursorIcon, Frame, Margin, Response, Sense, Shape, Stroke,
    StrokeKind, Ui, Vec2, epaint::RectShape,
};
use egui_toast::{Toast, ToastKind, Toasts};

use super::{metrics::Metrics, tokens::Tokens};

/// Toast kinds drawn by [`contents`].
const KINDS: [ToastKind; 4] = [
    ToastKind::Info,
    ToastKind::Warning,
    ToastKind::Error,
    ToastKind::Success,
];

/// Registers the themed renderer for every toast kind.
pub fn style(toasts: Toasts) -> Toasts {
    KINDS.into_iter().fold(toasts, |toasts, kind| {
        toasts.custom_contents(kind, contents)
    })
}

/// The accent color of a toast of `kind`.
pub fn tone(kind: ToastKind, t: &Tokens) -> Color32 {
    match kind {
        ToastKind::Warning => t.warn,
        ToastKind::Error => t.error,
        ToastKind::Success => t.success,
        ToastKind::Info | ToastKind::Custom(_) => t.info,
    }
}

/// Draws one toast: a window-surface card with a status dot, the message and a dismiss button.
pub fn contents(ui: &mut Ui, toast: &mut Toast) -> Response {
    let (t, m) = (Tokens::get(ui.ctx()), Metrics::get(ui.ctx()));
    let tone = tone(toast.kind, &t);
    let frame = Frame::window(ui.style())
        .inner_margin(Margin::symmetric(m.window_margin, m.gap.x as i8))
        .stroke(Stroke::NONE);
    let response = frame
        .show(ui, |ui| {
            ui.set_max_width(m.tooltip_w * 1.6);
            ui.horizontal(|ui| {
                if toast.options.show_icon {
                    let (slot, _) = ui.allocate_exact_size(Vec2::splat(m.icon_md), Sense::hover());
                    ui.painter()
                        .circle_filled(slot.center(), m.icon_md * 0.3, tone);
                }
                ui.label(toast.text.clone());
                let close = ui
                    .add(Button::new("x").frame(false))
                    .on_hover_text("Dismiss");
                if close.clicked() {
                    toast.close();
                }
            })
        })
        .response;

    if toast.options.show_progress {
        progress_bar(ui, &response, toast, tone);
    }
    ui.painter().add(Shape::Rect(RectShape::stroke(
        response.rect,
        frame.corner_radius,
        Stroke::new(1.0, t.card_border),
        StrokeKind::Inside,
    )));
    let response = response.on_hover_cursor(CursorIcon::PointingHand);
    if response.interact(Sense::click()).clicked() {
        toast.close();
    }
    response
}

/// The two-point remaining-time strip along the bottom edge.
fn progress_bar(ui: &mut Ui, response: &Response, toast: &Toast, tone: Color32) {
    let rounding = CornerRadius {
        nw: 0,
        ne: 0,
        ..ui.visuals().window_corner_radius
    };
    let mut clip_rect = response.rect;
    clip_rect.set_top(clip_rect.bottom() - 2.0);
    clip_rect.set_right(clip_rect.left() + clip_rect.width() * toast.options.progress() as f32);
    ui.painter()
        .with_clip_rect(clip_rect)
        .rect_filled(response.rect, rounding, tone);
}

#[cfg(test)]
mod tests {
    use super::*;
    use eframe::egui::{Align2, Area, Context, Order, Pos2, RawInput, Rect, vec2};
    use egui_toast::ToastOptions;

    fn run_frame(ctx: &Context, toasts: &mut Toasts) {
        let input = RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(800.0, 600.0))),
            ..Default::default()
        };
        ctx.run_ui(input, |ui| {
            Area::new(eframe::egui::Id::unique("toast_test"))
                .order(Order::Foreground)
                .anchor(Align2::LEFT_TOP, vec2(0.0, 0.0))
                .show(ui.ctx(), |ui| toasts.show(ui));
        })
        .drop_without_applying_deltas();
    }

    #[test]
    fn every_kind_has_a_distinct_tone_from_the_tokens() {
        let t = Tokens::derive(
            &eframe::egui::Style::default(),
            Color32::from_rgb(10, 200, 10),
            Color32::from_rgb(200, 10, 200),
        );
        assert_eq!(tone(ToastKind::Error, &t), t.error);
        assert_eq!(tone(ToastKind::Warning, &t), t.warn);
        assert_eq!(tone(ToastKind::Success, &t), t.success);
        assert_eq!(tone(ToastKind::Info, &t), t.info);
        assert_eq!(tone(ToastKind::Custom(3), &t), t.info);
    }

    #[test]
    fn themed_toasts_render_and_expire() {
        let ctx = Context::default();
        let mut toasts = style(Toasts::new());
        for kind in KINDS {
            toasts.add(Toast {
                kind,
                text: format!("{kind:?} message").into(),
                options: ToastOptions::default().duration_in_seconds(30.0),
                ..Default::default()
            });
        }
        run_frame(&ctx, &mut toasts);
        run_frame(&ctx, &mut toasts);
        let shown: Vec<Toast> = ctx.data(|d| {
            d.get_temp(eframe::egui::Id::unique("__toasts"))
                .unwrap_or_default()
        });
        assert_eq!(shown.len(), KINDS.len());
    }
}
