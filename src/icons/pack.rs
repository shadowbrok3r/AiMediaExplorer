//! White SVG icons (Phosphor Regular + custom `AME_*`) drawn through egui_extras' SVG loader.

use std::collections::HashMap;
use std::sync::OnceLock;

use egui::{Color32, Image, ImageSource, Pos2, Rect, Response, Sense, Ui, Vec2, load::Bytes};

/// One icon of the pack. Constants are generated from `assets/icons/regular/*.svg`.
#[derive(Clone, Copy, Debug)]
pub struct Icon {
    name: &'static str,
    uri: &'static str,
    duo_uri: &'static str,
    svg: &'static [u8],
    duo: Option<&'static [u8]>,
    glyph: &'static str,
}

impl Icon {
    /// File stem, e.g. `"hard-drive"` or `"ame-logo"`.
    pub const fn name(self) -> &'static str {
        self.name
    }

    /// Phosphor font glyph for inline text.
    pub const fn glyph(self) -> &'static str {
        self.glyph
    }

    /// Raw SVG bytes of the front layer.
    pub const fn svg(self) -> &'static [u8] {
        self.svg
    }

    /// Raw SVG bytes of the duotone back layer.
    pub const fn duo_svg(self) -> Option<&'static [u8]> {
        self.duo
    }

    pub const fn has_duo(self) -> bool {
        self.duo.is_some()
    }

    /// Name of the generated const, e.g. `"HARD_DRIVE"`.
    pub fn const_name(self) -> String {
        self.name.replace('-', "_").to_ascii_uppercase()
    }

    /// True for icons drawn for this app (`ame-*`), false for Phosphor.
    pub fn is_custom(self) -> bool {
        BORROWED.contains(&self.name)
    }

    /// Looks up an icon by file stem.
    pub fn find(name: &str) -> Option<Self> {
        ALL.binary_search_by(|icon| icon.name.cmp(name))
            .ok()
            .map(|i| ALL[i])
    }

    /// The Phosphor icon whose font glyph is `glyph`.
    pub fn from_glyph(glyph: &str) -> Option<Self> {
        static BY_GLYPH: OnceLock<HashMap<&'static str, Icon>> = OnceLock::new();
        BY_GLYPH
            .get_or_init(|| {
                ALL.iter()
                    .filter(|icon| !BORROWED.contains(&icon.name))
                    .map(|icon| (icon.glyph, *icon))
                    .collect()
            })
            .get(glyph)
            .copied()
    }

    pub fn source(self) -> ImageSource<'static> {
        ImageSource::Bytes {
            uri: self.uri.into(),
            bytes: Bytes::Static(self.svg),
        }
    }

    pub fn duo_source(self) -> Option<ImageSource<'static>> {
        self.duo.map(|bytes| ImageSource::Bytes {
            uri: self.duo_uri.into(),
            bytes: Bytes::Static(bytes),
        })
    }

    /// Untinted (white) image fixed at `size` points square; chain `.tint(..)` to colour it.
    pub fn image(self, size: f32) -> Image<'static> {
        Image::new(self.source()).fit_to_exact_size(Vec2::splat(size))
    }

    /// Allocates a `size` square and paints the icon in the current text colour.
    pub fn show(self, ui: &mut Ui, size: f32) -> Response {
        let tint = ui.visuals().text_color();
        self.show_tinted(ui, size, tint)
    }

    /// Allocates a `size` square and paints the icon tinted `tint`.
    pub fn show_tinted(self, ui: &mut Ui, size: f32, tint: Color32) -> Response {
        let (rect, response) = ui.allocate_exact_size(Vec2::splat(size), Sense::hover());
        if ui.is_rect_visible(rect) {
            self.paint(ui, rect, size, tint);
        }
        response
    }

    /// Paints the icon `size` points square, centred in `rect` and snapped to whole pixels.
    pub fn paint(self, ui: &Ui, rect: Rect, size: f32, tint: Color32) {
        let target = snapped_square(rect.center(), size, ui.pixels_per_point());
        Image::new(self.source()).tint(tint).paint_at(ui, target);
    }

    /// Paints the back layer tinted `bg` under the front layer tinted `fg`.
    pub fn paint_duo(self, ui: &Ui, rect: Rect, size: f32, fg: Color32, bg: Color32) {
        let target = snapped_square(rect.center(), size, ui.pixels_per_point());
        if let Some(back) = self.duo_source() {
            Image::new(back).tint(bg).paint_at(ui, target);
        }
        Image::new(self.source()).tint(fg).paint_at(ui, target);
    }
}

impl PartialEq for Icon {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for Icon {}

impl std::hash::Hash for Icon {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl From<Icon> for ImageSource<'static> {
    fn from(icon: Icon) -> Self {
        icon.source()
    }
}

/// Square of a whole number of physical pixels whose origin sits on the pixel grid.
pub(crate) fn snapped_square(center: Pos2, size: f32, pixels_per_point: f32) -> Rect {
    let px = (size * pixels_per_point).round().max(1.0);
    let min =
        (center.to_vec2() * pixels_per_point - Vec2::splat(px * 0.5)).round() / pixels_per_point;
    Rect::from_min_size(min.to_pos2(), Vec2::splat(px / pixels_per_point))
}

include!(concat!(env!("OUT_DIR"), "/icon_table.rs"));
