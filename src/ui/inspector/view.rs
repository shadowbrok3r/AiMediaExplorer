//! Picture placement in the preview and the viewer: scale for each fit mode, zoom about the pointer and panning limits.

use egui::{Rect, Vec2, pos2, vec2};
use serde::{Deserialize, Serialize};

pub const ZOOM_MIN: f32 = 0.1;
pub const ZOOM_MAX: f32 = 32.0;
/// Zoom change per scroll point.
const WHEEL_RATE: f32 = 0.0015;
/// Zoom change of the plus and minus buttons.
pub const ZOOM_STEP: f32 = 1.25;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreviewMode {
    /// The whole picture inside the area, never enlarged past one picture pixel per point.
    #[default]
    Fit,
    /// The area filled; the picture is cropped on one axis.
    Fill,
    /// One picture pixel per screen point.
    Actual,
}

impl PreviewMode {
    pub const ALL: [PreviewMode; 3] = [PreviewMode::Fit, PreviewMode::Fill, PreviewMode::Actual];

    pub fn label(self) -> &'static str {
        match self {
            PreviewMode::Fit => "Fit",
            PreviewMode::Fill => "Fill",
            PreviewMode::Actual => "100%",
        }
    }
}

/// Fit mode, zoom on top of it and pan of the picture centre from the area centre.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewState {
    pub mode: PreviewMode,
    pub zoom: f32,
    pub pan: Vec2,
}

impl Default for ViewState {
    fn default() -> Self {
        Self { mode: PreviewMode::Fit, zoom: 1.0, pan: Vec2::ZERO }
    }
}

/// Screen points per picture pixel of `mode` before zoom; zero for an empty picture or area.
pub fn base_scale(mode: PreviewMode, image: Vec2, area: Vec2) -> f32 {
    if image.x <= 0.0 || image.y <= 0.0 || area.x <= 0.0 || area.y <= 0.0 {
        return 0.0;
    }
    let (sx, sy) = (area.x / image.x, area.y / image.y);
    match mode {
        PreviewMode::Fit => sx.min(sy).min(1.0),
        PreviewMode::Fill => sx.max(sy),
        PreviewMode::Actual => 1.0,
    }
}

impl ViewState {
    pub fn with_mode(mode: PreviewMode) -> Self {
        Self { mode, ..Self::default() }
    }

    /// Screen points per picture pixel.
    pub fn scale(&self, image: Vec2, area: Vec2) -> f32 {
        base_scale(self.mode, image, area) * self.zoom
    }

    /// Size of the picture on screen.
    pub fn shown(&self, image: Vec2, area: Vec2) -> Vec2 {
        image * self.scale(image, area)
    }

    /// Back to the plain mode: no zoom, no pan.
    pub fn reset(&mut self) {
        self.zoom = 1.0;
        self.pan = Vec2::ZERO;
    }

    pub fn set_mode(&mut self, mode: PreviewMode) {
        self.mode = mode;
        self.reset();
    }

    /// Keeps the pan inside the limits for a picture of `image` pixels in `area`.
    pub fn clamp(&mut self, image: Vec2, area: Vec2) {
        self.zoom = self.zoom.clamp(ZOOM_MIN, ZOOM_MAX);
        self.pan = clamp_pan(self.pan, self.shown(image, area), area);
    }

    /// Multiplies the zoom by `factor` keeping the picture point under `cursor` (relative to the area centre) in place.
    pub fn zoom_at(&mut self, image: Vec2, area: Vec2, cursor: Vec2, factor: f32) {
        let old = self.scale(image, area);
        if old <= 0.0 || !factor.is_finite() || factor <= 0.0 {
            return;
        }
        let point = (cursor - self.pan) / old;
        self.zoom = (self.zoom * factor).clamp(ZOOM_MIN, ZOOM_MAX);
        let new = self.scale(image, area);
        self.pan = cursor - point * new;
        self.clamp(image, area);
    }

    /// Zoom change for a scroll of `points`.
    pub fn wheel_factor(points: f32) -> f32 {
        (points * WHEEL_RATE).exp()
    }

    /// Where the picture is drawn inside `area_rect`.
    pub fn dest(&self, image: Vec2, area_rect: Rect) -> Rect {
        Rect::from_center_size(area_rect.center() + self.pan, self.shown(image, area_rect.size()))
    }

    /// Zoom shown to the user, relative to one picture pixel per point.
    pub fn percent(&self, image: Vec2, area: Vec2) -> f32 {
        self.scale(image, area) * 100.0
    }
}

/// Pan limited so a picture larger than the area never leaves a gap and a smaller one stays centred.
pub fn clamp_pan(pan: Vec2, shown: Vec2, area: Vec2) -> Vec2 {
    let axis = |p: f32, s: f32, a: f32| {
        let slack = (s - a) / 2.0;
        if slack <= 0.0 { 0.0 } else { p.clamp(-slack, slack) }
    };
    vec2(axis(pan.x, shown.x, area.x), axis(pan.y, shown.y, area.y))
}

/// Rect of `size` points centred in `outer`, for placing the picture without zoom.
pub fn centered(outer: Rect, size: Vec2) -> Rect {
    Rect::from_min_size(pos2(outer.center().x - size.x / 2.0, outer.center().y - size.y / 2.0), size)
}

/// Picture size of the largest texture being shown, from the best texture available.
pub fn image_size(texture: Vec2, known: Option<(u32, u32)>) -> Vec2 {
    match known {
        Some((w, h)) if w > 0 && h > 0 => {
            let aspect = texture.x / texture.y.max(1.0);
            let real = w as f32 / h as f32;
            if (aspect - real).abs() < 0.05 { vec2(w as f32, h as f32) } else { texture }
        }
        _ => texture,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMG: Vec2 = vec2(1600.0, 800.0);
    const AREA: Vec2 = vec2(400.0, 300.0);

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-3
    }

    #[test]
    fn each_mode_has_its_base_scale() {
        assert!(close(base_scale(PreviewMode::Fit, IMG, AREA), 0.25));
        assert!(close(base_scale(PreviewMode::Fill, IMG, AREA), 0.375));
        assert!(close(base_scale(PreviewMode::Actual, IMG, AREA), 1.0));
        assert_eq!(base_scale(PreviewMode::Fit, Vec2::ZERO, AREA), 0.0);
        assert_eq!(base_scale(PreviewMode::Fit, IMG, vec2(0.0, 10.0)), 0.0);
    }

    #[test]
    fn fit_never_enlarges_a_small_picture_but_fill_does() {
        let small = vec2(40.0, 30.0);
        assert!(close(base_scale(PreviewMode::Fit, small, AREA), 1.0));
        assert!(close(base_scale(PreviewMode::Fill, small, AREA), 10.0));
        assert!(close(ViewState::default().percent(small, AREA), 100.0));
    }

    #[test]
    fn fit_shows_the_whole_picture_and_fill_covers_the_area() {
        let fit = ViewState::with_mode(PreviewMode::Fit).shown(IMG, AREA);
        assert!(fit.x <= AREA.x + 1e-3 && fit.y <= AREA.y + 1e-3);
        assert!(close(fit.x, 400.0) && close(fit.y, 200.0));
        let fill = ViewState::with_mode(PreviewMode::Fill).shown(IMG, AREA);
        assert!(fill.x >= AREA.x && fill.y >= AREA.y - 1e-3);
        assert!(close(fill.y, 300.0));
    }

    #[test]
    fn zooming_about_the_pointer_keeps_the_point_under_it() {
        let mut v = ViewState::default();
        let cursor = vec2(60.0, -40.0);
        let point_before = (cursor - v.pan) / v.scale(IMG, AREA);
        v.zoom_at(IMG, AREA, cursor, 2.0);
        assert!(close(v.zoom, 2.0));
        let point_after = (cursor - v.pan) / v.scale(IMG, AREA);
        assert!(close(point_before.x, point_after.x) && close(point_before.y, point_after.y), "{point_before:?} vs {point_after:?}");
    }

    #[test]
    fn zoom_stays_inside_its_limits() {
        let mut v = ViewState::default();
        v.zoom_at(IMG, AREA, Vec2::ZERO, 1e9);
        assert_eq!(v.zoom, ZOOM_MAX);
        v.zoom_at(IMG, AREA, Vec2::ZERO, 1e-9);
        assert_eq!(v.zoom, ZOOM_MIN);
        let before = v;
        v.zoom_at(IMG, AREA, Vec2::ZERO, f32::NAN);
        v.zoom_at(IMG, AREA, Vec2::ZERO, -1.0);
        assert_eq!(v, before);
    }

    #[test]
    fn a_picture_smaller_than_the_area_cannot_be_panned_and_a_larger_one_stays_covering() {
        assert_eq!(clamp_pan(vec2(50.0, 50.0), vec2(100.0, 100.0), AREA), Vec2::ZERO);
        let p = clamp_pan(vec2(500.0, -500.0), vec2(800.0, 600.0), AREA);
        assert_eq!(p, vec2(200.0, -150.0));
        let p = clamp_pan(vec2(10.0, 20.0), vec2(800.0, 300.0), AREA);
        assert_eq!(p, vec2(10.0, 0.0), "the axis that fits exactly stays centred");
    }

    #[test]
    fn dragging_is_limited_by_clamp_after_zooming_in() {
        let mut v = ViewState::default();
        v.zoom_at(IMG, AREA, Vec2::ZERO, 4.0);
        v.pan += vec2(10_000.0, 10_000.0);
        v.clamp(IMG, AREA);
        let shown = v.shown(IMG, AREA);
        assert!(close(v.pan.x, (shown.x - AREA.x) / 2.0) && close(v.pan.y, (shown.y - AREA.y) / 2.0));
        let rect = v.dest(IMG, Rect::from_min_size(pos2(0.0, 0.0), AREA));
        assert!(rect.min.x <= 1e-3 && rect.min.y <= 1e-3, "the picture still covers the area: {rect:?}");
    }

    #[test]
    fn changing_the_mode_resets_zoom_and_pan() {
        let mut v = ViewState { mode: PreviewMode::Fit, zoom: 3.0, pan: vec2(5.0, 5.0) };
        v.set_mode(PreviewMode::Actual);
        assert_eq!((v.mode, v.zoom, v.pan), (PreviewMode::Actual, 1.0, Vec2::ZERO));
        assert!(close(v.percent(IMG, AREA), 100.0));
        v.zoom = 2.0;
        assert!(close(v.percent(IMG, AREA), 200.0));
    }

    #[test]
    fn scrolling_up_zooms_in_and_down_zooms_out_symmetrically() {
        assert!(ViewState::wheel_factor(40.0) > 1.0);
        assert!(ViewState::wheel_factor(-40.0) < 1.0);
        assert!(close(ViewState::wheel_factor(40.0) * ViewState::wheel_factor(-40.0), 1.0));
        assert_eq!(ViewState::wheel_factor(0.0), 1.0);
    }

    #[test]
    fn the_known_size_replaces_the_texture_size_only_for_the_same_shape() {
        let tex = vec2(256.0, 128.0);
        assert_eq!(image_size(tex, Some((4000, 2000))), vec2(4000.0, 2000.0));
        assert_eq!(image_size(tex, Some((2000, 4000))), tex, "a rotated or different picture keeps the texture size");
        assert_eq!(image_size(tex, None), tex);
        assert_eq!(image_size(tex, Some((0, 0))), tex);
    }

    #[test]
    fn the_destination_rect_is_centred_on_the_area_plus_pan() {
        let area = Rect::from_min_size(pos2(10.0, 20.0), AREA);
        let v = ViewState::default();
        let dest = v.dest(IMG, area);
        assert!(close(dest.center().x, area.center().x) && close(dest.center().y, area.center().y));
        assert_eq!(centered(area, vec2(100.0, 50.0)).center(), area.center());
    }
}
