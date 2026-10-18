//! Semantic color accessors backed by the active egui theme.

use eframe::egui::{Color32, Context, Id, Ui};

use super::tokens::Tokens;

const SUCCESS_KEY: &str = "smart_media.theme.success_color";
const ACCENT2_KEY: &str = "smart_media.theme.accent_secondary";

/// Success green used when a theme sets none.
pub const DEFAULT_SUCCESS: Color32 = Color32::from_rgb(72, 199, 142);
/// Secondary accent used when a theme sets none.
pub const DEFAULT_ACCENT_SECONDARY: Color32 = Color32::from_rgb(191, 33, 101);

fn success_id() -> Id {
    Id::unique(SUCCESS_KEY)
}

fn accent2_id() -> Id {
    Id::unique(ACCENT2_KEY)
}

pub fn set_success_color(ctx: &Context, c: Color32) {
    ctx.data_mut(|d| d.insert_temp(success_id(), c));
}

pub fn set_accent_secondary(ctx: &Context, c: Color32) {
    ctx.data_mut(|d| d.insert_temp(accent2_id(), c));
}

pub fn error(ui: &Ui) -> Color32 {
    ui.visuals().error_fg_color
}

pub fn warn(ui: &Ui) -> Color32 {
    ui.visuals().warn_fg_color
}

pub fn info(ui: &Ui) -> Color32 {
    ui.visuals().hyperlink_color
}

pub fn success(ui: &Ui) -> Color32 {
    success_ctx(ui.ctx())
}

pub fn success_ctx(ctx: &Context) -> Color32 {
    ctx.data(|d| d.get_temp::<Color32>(success_id()))
        .unwrap_or(DEFAULT_SUCCESS)
}

/// Selection wash, for fills only; use [`primary`] for text and icon ink.
pub fn accent(ui: &Ui) -> Color32 {
    ui.visuals().selection.bg_fill
}

pub fn accent_secondary(ui: &Ui) -> Color32 {
    accent_secondary_ctx(ui.ctx())
}

pub fn accent_secondary_ctx(ctx: &Context) -> Color32 {
    ctx.data(|d| d.get_temp::<Color32>(accent2_id()))
        .unwrap_or(DEFAULT_ACCENT_SECONDARY)
}

/// Primary accent role from the published [`Tokens`].
pub fn primary(ui: &Ui) -> Color32 {
    primary_ctx(ui.ctx())
}

pub fn primary_ctx(ctx: &Context) -> Color32 {
    Tokens::get(ctx).primary
}

/// Secondary accent role from the published [`Tokens`].
pub fn secondary(ui: &Ui) -> Color32 {
    secondary_ctx(ui.ctx())
}

pub fn secondary_ctx(ctx: &Context) -> Color32 {
    Tokens::get(ctx).secondary
}

/// Tertiary accent role from the published [`Tokens`].
pub fn tertiary(ui: &Ui) -> Color32 {
    tertiary_ctx(ui.ctx())
}

pub fn tertiary_ctx(ctx: &Context) -> Color32 {
    Tokens::get(ctx).tertiary
}

pub fn strong_text(ui: &Ui) -> Color32 {
    ui.visuals().strong_text_color()
}

/// Body text in the current widget state; brightens on hover and press on themes that leave `override_text_color` unset.
pub fn text(ui: &Ui) -> Color32 {
    ui.visuals().text_color()
}

pub fn weak_text(ui: &Ui) -> Color32 {
    ui.visuals().weak_text_color()
}

/// Dimmer than [`weak_text`], for placeholders and inert separators.
pub fn faint_text(ui: &Ui) -> Color32 {
    weak_text(ui).gamma_multiply(0.65)
}

/// Number of distinct tag accents [`tag_color`] cycles through.
pub const TAG_LEN: usize = 5;

/// Theme-following accent for a categorical tag; wraps at [`TAG_LEN`].
pub fn tag_color(ui: &Ui, index: usize) -> Color32 {
    match index % TAG_LEN {
        0 => info(ui),
        1 => success(ui),
        2 => warn(ui),
        3 => accent_secondary(ui),
        _ => strong_text(ui),
    }
}

pub fn border(ui: &Ui) -> Color32 {
    ui.visuals().window_stroke.color
}

pub fn bg_surface(ui: &Ui) -> Color32 {
    ui.visuals().panel_fill
}

pub fn bg_faint(ui: &Ui) -> Color32 {
    ui.visuals().faint_bg_color
}

pub fn bg_extreme(ui: &Ui) -> Color32 {
    ui.visuals().extreme_bg_color
}

/// Number of distinct series the validated categorical palette supports.
pub const SERIES_LEN: usize = 6;

/// Categorical series colors; the order is CVD-validated and must not be reordered or extended.
pub const SERIES_PALETTE: [Color32; SERIES_LEN] = [
    Color32::from_rgb(0x41, 0x84, 0xE4),
    Color32::from_rgb(0xE5, 0x53, 0x4B),
    Color32::from_rgb(0x1F, 0x9D, 0xA6),
    Color32::from_rgb(0xCC, 0x6B, 0x2C),
    Color32::from_rgb(0x82, 0x56, 0xD0),
    Color32::from_rgb(0x46, 0x95, 0x4A),
];

/// Color for series `index`, or `None` past `SERIES_LEN`.
pub fn series_color(index: usize) -> Option<Color32> {
    SERIES_PALETTE.get(index).copied()
}

/// Neutral color for the folded "Other" bucket that absorbs series past `SERIES_LEN`.
pub fn other_series(ui: &Ui) -> Color32 {
    weak_text(ui)
}

/// Four-level status scale for meters and tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Good,
    Warn,
    Serious,
    Critical,
}

impl Status {
    /// Short label token for this status.
    pub fn label(self) -> &'static str {
        match self {
            Status::Good => "OK",
            Status::Warn => "Warn",
            Status::Serious => "Serious",
            Status::Critical => "Critical",
        }
    }
}

/// Status color paired with its label token.
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct StatusStyle {
    pub color: Color32,
    pub label: &'static str,
}

/// Color and label token for `status`, drawn from the theme's success/warn/error slots.
pub fn status_style(ui: &Ui, status: Status) -> StatusStyle {
    let color = match status {
        Status::Good => success(ui),
        Status::Warn => warn(ui),
        Status::Serious => lerp_rgb(warn(ui), error(ui), 0.5),
        Status::Critical => error(ui),
    };
    StatusStyle {
        color,
        label: status.label(),
    }
}

/// Status for a 0..100 usage percentage.
pub fn usage_status(pct: f32) -> Status {
    if pct >= 90.0 {
        Status::Critical
    } else if pct >= 70.0 {
        Status::Warn
    } else {
        Status::Good
    }
}

/// Color of a 0..100 usage percentage on the success, warn and error slots.
pub fn usage_level(ui: &Ui, pct: f32) -> Color32 {
    status_style(ui, usage_status(pct)).color
}

fn lerp_rgb(a: Color32, b: Color32, t: f32) -> Color32 {
    let mix = |x: u8, y: u8| {
        (x as f32 + (y as f32 - x as f32) * t)
            .round()
            .clamp(0.0, 255.0) as u8
    };
    Color32::from_rgb(mix(a.r(), b.r()), mix(a.g(), b.g()), mix(a.b(), b.b()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_series_palette_keeps_its_validated_order() {
        let expected = [
            (0x41, 0x84, 0xE4),
            (0xE5, 0x53, 0x4B),
            (0x1F, 0x9D, 0xA6),
            (0xCC, 0x6B, 0x2C),
            (0x82, 0x56, 0xD0),
            (0x46, 0x95, 0x4A),
        ];
        assert_eq!(SERIES_PALETTE.len(), SERIES_LEN);
        for (slot, (r, g, b)) in expected.iter().enumerate() {
            let c = series_color(slot).expect("slot inside the palette");
            assert_eq!((c.r(), c.g(), c.b()), (*r, *g, *b), "slot {slot} moved");
        }
    }

    #[test]
    fn a_series_past_the_palette_gets_no_colour_of_its_own() {
        assert!(series_color(SERIES_LEN).is_none());
        assert!(series_color(SERIES_LEN + 7).is_none());
    }

    #[test]
    fn usage_grades_on_its_documented_thresholds() {
        assert_eq!(usage_status(69.9), Status::Good);
        assert_eq!(usage_status(70.0), Status::Warn);
        assert_eq!(usage_status(89.9), Status::Warn);
        assert_eq!(usage_status(90.0), Status::Critical);
    }

    #[test]
    fn unset_semantic_colors_fall_back_to_the_defaults() {
        let ctx = Context::default();
        assert_eq!(success_ctx(&ctx), DEFAULT_SUCCESS);
        assert_eq!(accent_secondary_ctx(&ctx), DEFAULT_ACCENT_SECONDARY);
        set_success_color(&ctx, Color32::RED);
        set_accent_secondary(&ctx, Color32::BLUE);
        assert_eq!(success_ctx(&ctx), Color32::RED);
        assert_eq!(accent_secondary_ctx(&ctx), Color32::BLUE);
    }
}
