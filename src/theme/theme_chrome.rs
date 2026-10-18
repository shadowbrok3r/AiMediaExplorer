//! Baseline widget chrome presets: the shipped Mastertech file, MTech Noir and vanilla egui dark.

use std::sync::LazyLock;

use eframe::egui::{Color32, Shadow, Stroke, Style, Visuals};

use super::{
    glass::GlassParams,
    mtech_glass::glassify,
    saved::{SavedTheme, style_from_json, theme_from_json},
};

/// The shipped default theme file.
const MASTERTECH_JSON: &str = include_str!("../../assets/themes/mastertech.json");
const MTECH_NOIR_JSON: &str = include_str!("../../assets/themes/mtech_noir_style.json");

static MASTERTECH: LazyLock<SavedTheme> = LazyLock::new(|| {
    theme_from_json(MASTERTECH_JSON.as_bytes())
        .expect("mastertech.json must deserialize to a SavedTheme")
});

static MTECH_NOIR_CHROME: LazyLock<Style> = LazyLock::new(|| {
    style_from_json(MTECH_NOIR_JSON.as_bytes())
        .expect("mtech_noir_style.json must deserialize to egui::Style")
});

static DEFAULT_EGUI_CHROME: LazyLock<Style> = LazyLock::new(|| Style {
    visuals: Visuals::dark(),
    ..Style::default()
});

/// The embedded default theme: style, glass material, density and text scale.
pub fn mastertech_theme() -> SavedTheme {
    MASTERTECH.clone()
}

/// The embedded default theme's style, before the metrics overlay.
pub fn mastertech_chrome() -> Style {
    MASTERTECH.style.clone()
}

/// The embedded default theme's glass material.
pub fn mastertech_glass_params() -> GlassParams {
    MASTERTECH.glass.unwrap_or(GlassParams::OFF)
}

/// Vanilla egui dark mode defaults.
pub fn default_egui_chrome() -> Style {
    DEFAULT_EGUI_CHROME.clone()
}

/// OLED-black chrome with violet strokes, crimson press, cyan warn and rose error.
pub fn mtech_noir_chrome() -> Style {
    MTECH_NOIR_CHROME.clone()
}

/// Opacity of the noir glass floating surfaces over the frosted backdrop.
const NOIR_GLASS_WINDOW_ALPHA: f32 = 0.55;
const NOIR_GLASS_EDGE: Color32 = Color32::from_rgb(136, 128, 219);
const NOIR_GLASS_GLOW: Color32 = Color32::from_rgb(52, 40, 120);

/// [`mtech_noir_chrome`] as tinted glass: translucent widget panes and floating surfaces sheer enough for the frost.
pub fn mtech_noir_glass_chrome() -> Style {
    let mut style = glassify(&mtech_noir_chrome());
    let v = &mut style.visuals;
    v.window_fill = Color32::from_rgb(6, 5, 12).gamma_multiply(NOIR_GLASS_WINDOW_ALPHA);
    v.window_stroke = Stroke::new(1.0, NOIR_GLASS_EDGE.gamma_multiply(0.55));
    v.window_shadow = Shadow {
        offset: [0, 0],
        blur: 20,
        spread: 2,
        color: NOIR_GLASS_GLOW.gamma_multiply(0.38),
    };
    v.popup_shadow = Shadow {
        offset: [0, 0],
        blur: 14,
        spread: 1,
        color: NOIR_GLASS_GLOW.gamma_multiply(0.30),
    };
    style
}

/// The glass material [`mtech_noir_glass_chrome`] is drawn against.
pub fn mtech_noir_glass_params() -> GlassParams {
    GlassParams {
        enabled: true,
        blur_radius: 28.0,
        tint: Color32::from_rgba_unmultiplied(14, 12, 26, 72),
        corner_radius: 6.0,
        presence: 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::super::{fonts::TextScale, metrics::Density};
    use super::*;

    #[test]
    fn the_embedded_theme_is_the_mastertech_file() {
        let theme = mastertech_theme();
        assert_eq!(theme.density, Density::Compact);
        assert_eq!(theme.text_scale, TextScale::Small);
        let glass = theme.glass.expect("the file carries a glass material");
        assert!(glass.enabled);
        assert_eq!(glass.blur_radius, 32.0);
        assert_eq!(glass.tint, Color32::from_rgba_premultiplied(5, 5, 7, 81));
        assert_eq!(theme.style.visuals.panel_fill, Color32::from_rgb(8, 8, 10));
        assert_eq!(
            theme.style.visuals.window_fill,
            Color32::from_rgb(17, 17, 21)
        );
    }

    #[test]
    fn mtech_noir_chrome_decodes_to_its_own_palette() {
        let noir = mtech_noir_chrome();
        assert_ne!(noir, Style::default());
        assert_eq!(noir.visuals.panel_fill, Color32::BLACK);
        assert_eq!(noir.visuals.warn_fg_color, Color32::from_rgb(76, 219, 255));
        assert_eq!(noir.visuals.error_fg_color, Color32::from_rgb(255, 73, 137));
    }

    #[test]
    fn mtech_noir_glass_chrome_keeps_the_palette_but_opens_up_windows() {
        let noir = mtech_noir_chrome();
        let glass = mtech_noir_glass_chrome();
        assert_eq!(glass.visuals.panel_fill, noir.visuals.panel_fill);
        assert_eq!(glass.visuals.warn_fg_color, noir.visuals.warn_fg_color);
        assert_eq!(glass.text_styles, noir.text_styles);
        assert_eq!(glass.spacing, noir.spacing);
        assert!(glass.visuals.window_fill.a() < 180);
        assert!(glass.visuals.widgets.hovered.weak_bg_fill.a() < 255);
    }

    #[test]
    fn mtech_noir_glass_params_are_visible() {
        assert!(mtech_noir_glass_params().is_visible());
    }
}
