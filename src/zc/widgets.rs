//! Small drawing helpers shared by the connection card, the assistant header and the Devtools window.

use egui::{Color32, Rect, Response, RichText, Sense, Ui, vec2};

use super::health::{Health, HealthKind};

/// Dot colour of a health state.
pub fn health_color(ui: &Ui, kind: HealthKind) -> Color32 {
    match kind {
        HealthKind::Online => Color32::from_rgb(88, 196, 120),
        HealthKind::Checking | HealthKind::Unconfigured => ui.visuals().weak_text_color(),
        HealthKind::Offline => ui.visuals().warn_fg_color,
        HealthKind::Unauthorized | HealthKind::UnknownAgent => ui.visuals().error_fg_color,
    }
}

/// A filled circle sized to the text height.
pub fn dot(ui: &mut Ui, color: Color32) -> Response {
    let edge = ui.text_style_height(&egui::TextStyle::Body) * 0.7;
    let (rect, response) = ui.allocate_exact_size(vec2(edge, edge), Sense::hover());
    let r = Rect::from_center_size(rect.center(), vec2(edge * 0.8, edge * 0.8));
    ui.painter().circle_filled(r.center(), r.width() / 2.0, color);
    response
}

/// Dot, state name and detail; the detail wraps under narrow widths.
pub fn health_row(ui: &mut Ui, health: &Health) {
    ui.horizontal(|ui| {
        dot(ui, health_color(ui, health.kind()));
        ui.label(RichText::new(health.label()).strong());
    });
    let detail = health.detail();
    if !detail.is_empty() {
        ui.label(RichText::new(detail).weak());
    }
}

/// `3s ago`, `4m ago` or `2h ago` for a unix-millisecond stamp.
pub fn ago(now_ms: u64, at_ms: u64) -> String {
    let secs = now_ms.saturating_sub(at_ms) / 1000;
    match secs {
        0..=59 => format!("{secs}s ago"),
        60..=3599 => format!("{}m ago", secs / 60),
        _ => format!("{}h ago", secs / 3600),
    }
}

/// `12:34:56` for a unix-millisecond stamp, in local time.
pub fn clock(at_ms: u64) -> String {
    chrono::DateTime::<chrono::Utc>::from_timestamp_millis(at_ms as i64)
        .map(|t| t.with_timezone(&chrono::Local).format("%H:%M:%S").to_string())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ages_read_in_the_largest_whole_unit() {
        assert_eq!(ago(10_000, 7_000), "3s ago");
        assert_eq!(ago(300_000, 0), "5m ago");
        assert_eq!(ago(7_300_000, 0), "2h ago");
        assert_eq!(ago(0, 5_000), "0s ago");
    }

    #[test]
    fn a_stamp_formats_as_a_clock() {
        assert_eq!(clock(0).len(), 8);
    }
}
