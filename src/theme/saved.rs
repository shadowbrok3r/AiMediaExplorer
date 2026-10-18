//! A saved theme: the egui `Style` plus the parts of a theme with no `Style` slot, and its JSON forms.

use anyhow::Context as _;
use eframe::egui::{Context, FontFamily, Style};

use super::{
    fonts::{self, TextScale},
    glass::{self, GlassParams},
    metrics::{self, Density},
};

const STYLE_KEY: &str = "style";
const GLASS_KEY: &str = "glass";
const DENSITY_KEY: &str = "density";
const TEXT_SCALE_KEY: &str = "text_scale";

/// A saved theme: the egui `Style`, its glass material, density and text scale.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SavedTheme {
    pub style: Style,
    /// `None` on an imported bare `egui::Style` file; callers recover it from the matching preset.
    #[serde(default)]
    pub glass: Option<GlassParams>,
    #[serde(default)]
    pub density: Density,
    #[serde(default)]
    pub text_scale: TextScale,
}

impl SavedTheme {
    /// The theme as currently applied to `ctx`: the live style plus the glass, density and text scale on the context.
    pub fn current(ctx: &Context) -> Self {
        let mut style = (*ctx.global_style()).clone();
        strip_weight_families(&mut style);
        Self {
            style,
            glass: Some(glass::params(ctx)),
            density: metrics::density(ctx),
            text_scale: fonts::text_scale(ctx),
        }
    }
}

/// Resets text styles in the overlay's weight families to `Proportional`.
pub(crate) fn strip_weight_families(style: &mut Style) {
    for font in style.text_styles.values_mut() {
        if let FontFamily::Name(name) = &font.family
            && (&**name == fonts::MEDIUM || &**name == fonts::SEMIBOLD)
        {
            font.family = FontFamily::Proportional;
        }
    }
}

/// Serializes a theme to the JSON shape used for storage and file export.
pub fn theme_to_json(theme: &SavedTheme) -> anyhow::Result<Vec<u8>> {
    let mut theme = theme.clone();
    strip_weight_families(&mut theme.style);
    serde_json::to_vec(&theme).context("serialize theme to JSON")
}

/// Reads a theme from JSON in either shape: the wrapper written by [`theme_to_json`] or a bare `egui::Style`, told apart by the `style` key.
pub fn theme_from_json(json: &[u8]) -> anyhow::Result<SavedTheme> {
    let value: serde_json::Value =
        serde_json::from_slice(json).context("theme is not valid JSON")?;
    match value {
        serde_json::Value::Object(mut map) if map.contains_key(STYLE_KEY) => {
            let glass = take_field(&mut map, GLASS_KEY);
            let density = take_field(&mut map, DENSITY_KEY).unwrap_or_default();
            let text_scale = take_field(&mut map, TEXT_SCALE_KEY).unwrap_or_default();
            let style = map.remove(STYLE_KEY).unwrap_or(serde_json::Value::Null);
            Ok(SavedTheme {
                style: style_from_value(style),
                glass,
                density,
                text_scale,
            })
        }
        bare => Ok(SavedTheme {
            style: style_from_value(bare),
            glass: None,
            density: Default::default(),
            text_scale: Default::default(),
        }),
    }
}

/// Removes `key` from `map` and decodes it, or `None` when absent or malformed.
fn take_field<T: serde::de::DeserializeOwned>(
    map: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Option<T> {
    map.remove(key).and_then(|v| serde_json::from_value(v).ok())
}

/// Deserializes an egui `Style` from JSON, migrating across egui versions.
pub fn style_from_json(json: &[u8]) -> anyhow::Result<Style> {
    let value: serde_json::Value =
        serde_json::from_slice(json).context("theme is not valid JSON; cannot migrate")?;
    Ok(style_from_value(value))
}

/// Decodes a `Style`, overlaying the stored values on `Style::default()` when the strict decode fails so older themes still load.
fn style_from_value(value: serde_json::Value) -> Style {
    match serde_json::from_value::<Style>(value.clone()) {
        Ok(style) => style,
        Err(e) => {
            log::debug!(
                "stored theme predates the current egui Style schema ({e}); migrating against egui Style defaults"
            );
            let default = serde_json::to_value(Style::default()).unwrap_or_default();
            match serde_json::from_value::<Style>(merge_over_default(default, value)) {
                Ok(style) => {
                    log::debug!("theme migrated to the current egui Style schema");
                    style
                }
                Err(e2) => {
                    log::error!("theme migration failed ({e2}); using default Style");
                    Style::default()
                }
            }
        }
    }
}

/// Overlays `stored` onto `default` by the default's structure; a shape mismatch keeps the default.
fn merge_over_default(default: serde_json::Value, stored: serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    match (default, stored) {
        (Value::Object(mut d), Value::Object(s)) => {
            for (k, dv) in d.iter_mut() {
                if let Some(sv) = s.get(k) {
                    *dv = merge_over_default(dv.take(), sv.clone());
                }
            }
            Value::Object(d)
        }
        (Value::Object(d), _) => Value::Object(d),
        (_, stored) => stored,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eframe::egui::{Color32, TextStyle};

    fn sample() -> SavedTheme {
        let mut style = Style::default();
        style.visuals.hyperlink_color = Color32::from_rgb(1, 2, 3);
        SavedTheme {
            style,
            glass: Some(GlassParams {
                enabled: true,
                blur_radius: 12.0,
                tint: Color32::from_rgba_unmultiplied(4, 5, 6, 70),
                corner_radius: 5.0,
                presence: 0.8,
            }),
            density: Density::Comfortable,
            text_scale: TextScale::Large,
        }
    }

    #[test]
    fn the_wrapper_round_trips_every_part() {
        let theme = sample();
        let decoded = theme_from_json(&theme_to_json(&theme).unwrap()).unwrap();
        assert_eq!(decoded.glass, theme.glass);
        assert_eq!(decoded.density, Density::Comfortable);
        assert_eq!(decoded.text_scale, TextScale::Large);
        assert_eq!(
            decoded.style.visuals.hyperlink_color,
            theme.style.visuals.hyperlink_color
        );
    }

    #[test]
    fn a_bare_style_loads_without_glass_density_or_scale() {
        let mut style = Style::default();
        style.visuals.hyperlink_color = Color32::from_rgb(9, 8, 7);
        let json = serde_json::to_vec(&style).unwrap();
        let theme = theme_from_json(&json).unwrap();
        assert!(theme.glass.is_none());
        assert_eq!(theme.density, Density::Regular);
        assert_eq!(theme.text_scale, TextScale::Normal);
        assert_eq!(
            theme.style.visuals.hyperlink_color,
            Color32::from_rgb(9, 8, 7)
        );
    }

    #[test]
    fn a_wrapper_without_density_or_scale_loads_the_defaults() {
        let json = serde_json::json!({ "style": Style::default() });
        let theme = theme_from_json(json.to_string().as_bytes()).unwrap();
        assert_eq!(theme.density, Density::Regular);
        assert_eq!(theme.text_scale, TextScale::Normal);
    }

    #[test]
    fn malformed_json_is_an_error_not_a_default() {
        assert!(theme_from_json(b"not json").is_err());
        assert!(style_from_json(b"{").is_err());
    }

    #[test]
    fn a_style_missing_new_fields_migrates_against_the_defaults() {
        let mut value = serde_json::to_value(Style::default()).unwrap();
        let visuals = value["visuals"].as_object_mut().unwrap();
        visuals.remove("text_options");
        visuals.remove("disabled_alpha");
        value["visuals"]["hyperlink_color"] = serde_json::json!([5, 6, 7, 255]);
        let style = style_from_value(value);
        assert_eq!(style.visuals.hyperlink_color, Color32::from_rgb(5, 6, 7));
        assert_eq!(
            style.visuals.disabled_alpha,
            Style::default().visuals.disabled_alpha
        );
    }

    #[test]
    fn saved_themes_never_name_the_weight_families() {
        let mut style = Style::default();
        style
            .text_styles
            .get_mut(&TextStyle::Heading)
            .unwrap()
            .family = FontFamily::Name(fonts::SEMIBOLD.into());
        let theme = SavedTheme { style, ..sample() };
        let decoded = theme_from_json(&theme_to_json(&theme).unwrap()).unwrap();
        for font in decoded.style.text_styles.values() {
            assert!(!matches!(font.family, FontFamily::Name(_)), "{font:?}");
        }
    }
}
