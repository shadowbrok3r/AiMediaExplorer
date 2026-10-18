//! Theme presets and the install path: applying a preset or a saved theme to a [`Context`].

use eframe::egui::{Color32, Context, Id, Style, Visuals};

use super::{
    colors::{self, DEFAULT_ACCENT_SECONDARY, DEFAULT_SUCCESS},
    fonts::{self, TextScale},
    glass::{self, GlassParams},
    metrics::{self, Density, Metrics, apply_metrics},
    mtech_glass::{self, glass_params_for_style, glass_style},
    mtech_studio,
    saved::SavedTheme,
    theme_chrome::{
        default_egui_chrome, mastertech_chrome, mastertech_glass_params, mastertech_theme,
        mtech_noir_chrome, mtech_noir_glass_chrome, mtech_noir_glass_params,
    },
    tokens::Tokens,
};

/// One group of the preset picker.
pub struct PresetGroup {
    pub title: &'static str,
    pub presets: &'static [PresetStyles],
}

/// The preset the picker lists above its groups.
pub const TOP_PRESET: PresetStyles = PresetStyles::Mastertech;

/// Preset picker groups, in menu order, between [`TOP_PRESET`] and Custom.
pub static PRESET_GROUPS: [PresetGroup; 2] = [
    PresetGroup {
        title: "MTech",
        presets: &[
            PresetStyles::MtechStudio,
            PresetStyles::MtechNoir,
            PresetStyles::MtechNoirGlass,
            PresetStyles::MtechGlass,
        ],
    },
    PresetGroup {
        title: "Classics",
        presets: &[PresetStyles::DefaultEgui],
    },
];

/// Every preset a saved style can be recognised as.
pub const MATCHABLE_PRESETS: [PresetStyles; 6] = [
    PresetStyles::Mastertech,
    PresetStyles::MtechStudio,
    PresetStyles::DefaultEgui,
    PresetStyles::MtechNoir,
    PresetStyles::MtechNoirGlass,
    PresetStyles::MtechGlass,
];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PresetStyles {
    #[default]
    Mastertech,
    MtechStudio,
    DefaultEgui,
    MtechNoir,
    MtechNoirGlass,
    MtechGlass,
    Custom,
}

impl PresetStyles {
    pub fn as_str(&self) -> &str {
        match self {
            PresetStyles::Mastertech => "Mastertech",
            PresetStyles::MtechStudio => mtech_studio::NAME,
            PresetStyles::DefaultEgui => "Default egui",
            PresetStyles::MtechNoir => "MTech Noir",
            PresetStyles::MtechNoirGlass => "MTech Noir Glass",
            PresetStyles::MtechGlass => "MTech Glass",
            PresetStyles::Custom => "Custom",
        }
    }

    /// The preset a label names; unknown labels are Custom.
    pub fn from_label(label: &str) -> Self {
        MATCHABLE_PRESETS
            .into_iter()
            .find(|p| p.as_str() == label)
            .unwrap_or(Self::Custom)
    }
}

/// The picker group listing `preset`; `None` for the top-level rows.
pub fn group_of(preset: PresetStyles) -> Option<&'static PresetGroup> {
    PRESET_GROUPS.iter().find(|g| g.presets.contains(&preset))
}

/// Applies the embedded default theme: its style, glass material, density and text scale.
pub fn bootstrap_startup_theme(ctx: &Context) {
    apply_saved_theme(ctx, mastertech_theme());
}

/// Installs `style` under the current density's metrics overlay and publishes its tokens.
pub fn install_style(ctx: &Context, style: Style) {
    metrics::install_overlaid_style(ctx, style);
    if let Some(tokens) = exact_tokens(&ctx.global_style()) {
        Tokens::publish(ctx, tokens);
    }
}

/// The tokens [`apply_preset`] publishes for `preset` at the current density; installs nothing.
pub fn preset_tokens(ctx: &Context, preset: PresetStyles) -> Tokens {
    let density = metrics::density(ctx);
    let id = preset_tokens_id(preset, density);
    if let Some(tokens) = ctx.data(|d| d.get_temp::<Tokens>(id)) {
        return tokens;
    }
    let tokens = hand_tuned_tokens(preset).unwrap_or_else(|| {
        let (success, secondary) = semantic_colors_for_preset(preset);
        let m = Metrics::for_density(density);
        metrics::overlay(style_for_preset(preset), &m, success, secondary).1
    });
    ctx.data_mut(|d| d.insert_temp(id, tokens));
    tokens
}

/// Temp-data key of the [`preset_tokens`] cache entry for `preset` at `density`.
fn preset_tokens_id(preset: PresetStyles, density: Density) -> Id {
    Id::unique(("smart_media.theme.preset_tokens", preset, density))
}

/// Hand-tuned tokens of a preset, for presets that define them.
fn hand_tuned_tokens(preset: PresetStyles) -> Option<Tokens> {
    match preset {
        PresetStyles::MtechStudio => Some(mtech_studio::TOKENS),
        _ => None,
    }
}

/// The hand-tuned tokens of the preset `style` matches, if it defines any.
fn exact_tokens(style: &Style) -> Option<Tokens> {
    let style = normalized(style);
    MATCHABLE_PRESETS
        .into_iter()
        .filter_map(|p| hand_tuned_tokens(p).map(|t| (p, t)))
        .find(|(p, _)| styles_visually_equal(&style_for_preset(*p), &style))
        .map(|(_, t)| t)
}

/// Switches the density and re-installs the current style under it.
pub fn set_density(ctx: &Context, density: Density) {
    if metrics::density(ctx) == density {
        return;
    }
    metrics::set_density(ctx, density);
    install_style(ctx, (*ctx.global_style()).clone());
}

/// Switches the UI text size, rebuilding the font atlas when it changes.
pub fn set_text_scale(ctx: &Context, scale: TextScale) {
    fonts::apply_scale(ctx, scale);
}

/// Applies a saved theme: its style, its preset's semantic colors when recognized (Custom otherwise), and its glass material.
pub fn apply_saved_theme(ctx: &Context, theme: SavedTheme) {
    let preset = exact_preset(&normalized(&theme.style)).unwrap_or(PresetStyles::Custom);
    metrics::set_density(ctx, theme.density);
    apply_preset_semantics(ctx, preset);
    install_style(ctx, theme.style);
    set_text_scale(ctx, theme.text_scale);
    if let Some(glass) = theme.glass {
        glass::set_params(ctx, glass);
    }
}

/// Clears egui memory, then re-installs the current theme, density and text size into it.
pub fn reset_memory(ctx: &Context) {
    let saved = SavedTheme::current(ctx);
    ctx.memory_mut(|mem| *mem = Default::default());
    apply_saved_theme(ctx, saved);
}

/// Stores a preset's name, its semantic colors, read by [`install_style`], and its glass material.
fn apply_preset_semantics(ctx: &Context, preset: PresetStyles) {
    set_installed_preset(ctx, preset);
    let (success, accent2) = semantic_colors_for_preset(preset);
    colors::set_success_color(ctx, success);
    colors::set_accent_secondary(ctx, accent2);
    glass::set_params(ctx, glass_params_for_preset(preset));
}

fn installed_preset_id() -> Id {
    Id::unique("smart_media.theme.installed_preset")
}

/// The preset the installed style was applied as; Custom when none was recorded.
pub fn installed_preset(ctx: &Context) -> PresetStyles {
    ctx.data(|d| d.get_temp::<PresetStyles>(installed_preset_id()))
        .unwrap_or(PresetStyles::Custom)
}

/// Records `preset` as the one the installed style was applied as.
pub fn set_installed_preset(ctx: &Context, preset: PresetStyles) {
    ctx.data_mut(|d| d.insert_temp(installed_preset_id(), preset));
}

/// Compares styles ignoring `number_formatter`, whose PartialEq is Arc pointer identity.
pub fn styles_visually_equal(a: &Style, b: &Style) -> bool {
    let mut a = a.clone();
    a.number_formatter = b.number_formatter.clone();
    a == *b
}

/// `style` under the Regular metrics overlay, the form presets are compared in.
pub fn normalized(style: &Style) -> Style {
    let mut style = style.clone();
    apply_metrics(&mut style, &Metrics::for_density(Density::Regular));
    style
}

/// The preset a normalized style equals.
pub fn exact_preset(normal: &Style) -> Option<PresetStyles> {
    MATCHABLE_PRESETS
        .into_iter()
        .find(|p| styles_visually_equal(&style_for_preset(*p), normal))
}

/// Whether `style` is a blank egui default, dark or light.
pub fn is_blank_default_style(style: &Style) -> bool {
    let style = normalized(style);
    let dark_default = Style {
        visuals: Visuals::dark(),
        ..Style::default()
    };
    [Style::default(), dark_default]
        .iter()
        .any(|blank| styles_visually_equal(&style, &normalized(blank)))
}

/// A preset's style under the Regular metrics overlay.
pub fn style_for_preset(preset: PresetStyles) -> Style {
    let mut style = preset_chrome(preset);
    apply_metrics(&mut style, &Metrics::for_density(Density::Regular));
    style
}

/// A preset's style as its theme defines it, before the metrics overlay.
fn preset_chrome(preset: PresetStyles) -> Style {
    match preset {
        PresetStyles::Mastertech | PresetStyles::Custom => mastertech_chrome(),
        PresetStyles::MtechStudio => mtech_studio::studio_style(),
        PresetStyles::DefaultEgui => default_egui_chrome(),
        PresetStyles::MtechNoir => mtech_noir_chrome(),
        PresetStyles::MtechNoirGlass => mtech_noir_glass_chrome(),
        PresetStyles::MtechGlass => glass_style(),
    }
}

/// Success and secondary-accent colors of a preset, which have no `egui::Visuals` slot.
pub fn semantic_colors_for_preset(preset: PresetStyles) -> (Color32, Color32) {
    match preset {
        PresetStyles::Mastertech | PresetStyles::Custom => {
            (DEFAULT_SUCCESS, DEFAULT_ACCENT_SECONDARY)
        }
        PresetStyles::DefaultEgui => (
            Color32::from_rgb(42, 172, 170),
            Color32::from_rgb(56, 114, 238),
        ),
        PresetStyles::MtechStudio => mtech_studio::studio_semantic_colors(),
        PresetStyles::MtechNoir | PresetStyles::MtechNoirGlass => (
            Color32::from_rgb(90, 220, 160),
            Color32::from_rgb(233, 130, 255),
        ),
        PresetStyles::MtechGlass => mtech_glass::semantic_colors(),
    }
}

/// The backdrop-blur material a preset draws its glass against; opaque presets return [`GlassParams::OFF`].
pub fn glass_params_for_preset(preset: PresetStyles) -> GlassParams {
    match preset {
        PresetStyles::Mastertech => mastertech_glass_params(),
        PresetStyles::MtechStudio => mtech_studio::studio_glass_params(),
        PresetStyles::MtechNoirGlass => mtech_noir_glass_params(),
        PresetStyles::MtechGlass => glass_params_for_style(&glass_style()),
        PresetStyles::MtechNoir | PresetStyles::DefaultEgui | PresetStyles::Custom => {
            GlassParams::OFF
        }
    }
}

/// Installs a preset's style and semantic colors, keeping the current density and text scale.
pub fn apply_preset(ctx: &Context, preset: PresetStyles) {
    apply_preset_semantics(ctx, preset);
    install_style(ctx, style_for_preset(preset));
}

#[cfg(test)]
mod tests {
    use super::super::{
        saved::{theme_from_json, theme_to_json},
        tokens::composite,
    };
    use super::*;

    fn saved(preset: PresetStyles) -> SavedTheme {
        SavedTheme {
            style: style_for_preset(preset),
            glass: Some(glass_params_for_preset(preset)),
            density: Density::default(),
            text_scale: TextScale::default(),
        }
    }

    fn all_presets() -> impl Iterator<Item = PresetStyles> {
        MATCHABLE_PRESETS.into_iter().chain([PresetStyles::Custom])
    }

    fn round_trip(theme: &SavedTheme) -> SavedTheme {
        theme_from_json(&theme_to_json(theme).unwrap()).unwrap()
    }

    #[test]
    fn every_preset_round_trips_as_itself_under_the_overlay() {
        for preset in MATCHABLE_PRESETS {
            let decoded = round_trip(&saved(preset));
            assert_eq!(
                exact_preset(&normalized(&decoded.style)),
                Some(preset),
                "{} did not round trip",
                preset.as_str(),
            );
        }
    }

    #[test]
    fn every_preset_shares_the_overlay_geometry() {
        let regular = Metrics::for_density(Density::Regular);
        for preset in all_presets() {
            let style = style_for_preset(preset);
            let name = preset.as_str();
            assert_eq!(
                style.spacing.interact_size.y, regular.control_h,
                "{name}: control height"
            );
            assert_eq!(style.spacing.button_padding, regular.pad, "{name}: padding");
            assert!(style.override_font_id.is_none(), "{name}: font override");

            let w = &style.visuals.widgets;
            let radius = w.inactive.corner_radius;
            for state in [
                &w.noninteractive,
                &w.inactive,
                &w.hovered,
                &w.active,
                &w.open,
            ] {
                for width in [state.bg_stroke.width, state.fg_stroke.width] {
                    assert_eq!(width, width.round(), "{name}: fractional stroke {width}");
                }
                assert_eq!(state.expansion, 0.0, "{name}: widget grows");
                assert_eq!(
                    state.corner_radius, radius,
                    "{name}: states disagree on radius"
                );
            }
        }
    }

    #[test]
    fn presets_stay_distinct() {
        for (i, a) in MATCHABLE_PRESETS.iter().enumerate() {
            for b in &MATCHABLE_PRESETS[i + 1..] {
                assert!(
                    !styles_visually_equal(&style_for_preset(*a), &style_for_preset(*b)),
                    "{} and {} produce the same style",
                    a.as_str(),
                    b.as_str(),
                );
            }
        }
    }

    #[test]
    fn bootstrap_applies_the_embedded_mastertech_file() {
        let ctx = Context::default();
        bootstrap_startup_theme(&ctx);
        assert_eq!(installed_preset(&ctx), PresetStyles::Mastertech);
        assert_eq!(metrics::density(&ctx), Density::Compact);
        assert_eq!(fonts::text_scale(&ctx), TextScale::Small);
        assert_eq!(Metrics::get(&ctx), Metrics::for_density(Density::Compact));
        assert_eq!(glass::params(&ctx), mastertech_glass_params());
        assert!(glass::params(&ctx).is_visible());
        let style = ctx.global_style();
        let compact = Metrics::for_density(Density::Compact);
        assert_eq!(style.spacing.interact_size.y, compact.control_h);
        assert_eq!(style.spacing.item_spacing, compact.gap);
        assert_eq!(style.spacing.button_padding, compact.pad);
        for (text_style, size) in [
            (eframe::egui::TextStyle::Small, 10.5),
            (eframe::egui::TextStyle::Body, 12.5),
            (eframe::egui::TextStyle::Button, 12.5),
            (eframe::egui::TextStyle::Monospace, 12.0),
            (eframe::egui::TextStyle::Heading, 15.0),
        ] {
            assert_eq!(style.text_styles[&text_style].size, size, "{text_style:?}");
        }
    }

    #[test]
    fn the_mastertech_preset_reproduces_the_file_style_under_its_own_density() {
        let file = mastertech_theme();
        let ctx = Context::default();
        apply_saved_theme(&ctx, file.clone());
        let mut live = (*ctx.global_style()).clone();
        live.number_formatter = file.style.number_formatter.clone();
        let mut expected = file.style.clone();
        apply_metrics(&mut expected, &Metrics::for_density(Density::Compact));
        assert!(
            live == expected,
            "the installed style drifted from the file"
        );
    }

    #[test]
    fn the_file_semantic_colors_are_the_app_wide_defaults() {
        let ctx = Context::default();
        bootstrap_startup_theme(&ctx);
        assert_eq!(colors::success_ctx(&ctx), DEFAULT_SUCCESS);
        assert_eq!(colors::accent_secondary_ctx(&ctx), DEFAULT_ACCENT_SECONDARY);
    }

    #[test]
    fn density_and_text_scale_round_trip_through_a_saved_theme() {
        let ctx = Context::default();
        apply_preset(&ctx, PresetStyles::MtechGlass);
        set_density(&ctx, Density::Comfortable);
        set_text_scale(&ctx, TextScale::XLarge);
        let decoded = round_trip(&SavedTheme::current(&ctx));
        assert_eq!(decoded.density, Density::Comfortable);
        assert_eq!(decoded.text_scale, TextScale::XLarge);

        let fresh = Context::default();
        apply_saved_theme(&fresh, decoded);
        assert_eq!(metrics::density(&fresh), Density::Comfortable);
        assert_eq!(fonts::text_scale(&fresh), TextScale::XLarge);
        assert_eq!(installed_preset(&fresh), PresetStyles::MtechGlass);
        assert_eq!(
            fresh.global_style().spacing.interact_size.y,
            Metrics::for_density(Density::Comfortable).control_h
        );
    }

    #[test]
    fn switching_density_reinstalls_the_current_style() {
        let ctx = Context::default();
        apply_preset(&ctx, PresetStyles::MtechGlass);
        let fill = ctx.global_style().visuals.widgets.inactive.weak_bg_fill;

        set_density(&ctx, Density::Comfortable);
        let style = ctx.global_style();
        assert_eq!(
            style.spacing.interact_size.y,
            Metrics::for_density(Density::Comfortable).control_h
        );
        assert_eq!(style.visuals.widgets.inactive.weak_bg_fill, fill);
    }

    #[test]
    fn a_blank_default_style_is_recognised() {
        let legacy = Style {
            visuals: Visuals::dark(),
            ..Style::default()
        };
        assert!(is_blank_default_style(&legacy));
        assert!(is_blank_default_style(&normalized(&legacy)));
        assert!(!is_blank_default_style(&style_for_preset(
            PresetStyles::Mastertech
        )));
    }

    #[test]
    fn a_hand_tuned_material_survives_a_save() {
        let tuned = GlassParams {
            enabled: true,
            blur_radius: 41.0,
            tint: Color32::from_rgba_unmultiplied(9, 30, 44, 96),
            corner_radius: 13.0,
            presence: 0.72,
        };
        let ctx = Context::default();
        apply_preset(&ctx, PresetStyles::MtechGlass);
        glass::set_params(&ctx, tuned);
        let saved = round_trip(&SavedTheme::current(&ctx));

        let fresh = Context::default();
        bootstrap_startup_theme(&fresh);
        apply_saved_theme(&fresh, saved);
        assert_eq!(glass::params(&fresh), tuned);
    }

    #[test]
    fn a_bare_style_record_takes_its_presets_material() {
        let style = style_for_preset(PresetStyles::MtechNoirGlass);
        let theme = theme_from_json(&serde_json::to_vec(&style).unwrap()).unwrap();
        assert!(theme.glass.is_none());
        let ctx = Context::default();
        apply_saved_theme(&ctx, theme);
        assert_eq!(installed_preset(&ctx), PresetStyles::MtechNoirGlass);
        assert_eq!(glass::params(&ctx), mtech_noir_glass_params());
    }

    #[test]
    fn applying_a_preset_carries_its_glass_material() {
        let ctx = Context::default();
        apply_preset(&ctx, PresetStyles::MtechNoirGlass);
        let params = glass::params(&ctx);
        assert_eq!(params, mtech_noir_glass_params());
        assert!(params.is_visible());

        apply_preset(&ctx, PresetStyles::MtechNoir);
        assert_eq!(glass::params(&ctx), GlassParams::OFF);
    }

    #[test]
    fn mtech_studio_publishes_its_exact_tokens_on_every_install_path() {
        let ctx = Context::default();
        apply_preset(&ctx, PresetStyles::MtechStudio);
        assert_eq!(Tokens::get(&ctx), mtech_studio::TOKENS);
        assert_eq!(glass::params(&ctx), mtech_studio::studio_glass_params());

        set_density(&ctx, Density::Compact);
        assert_eq!(Tokens::get(&ctx), mtech_studio::TOKENS);

        let decoded = round_trip(&SavedTheme::current(&ctx));
        assert_eq!(decoded.density, Density::Compact);
        let fresh = Context::default();
        bootstrap_startup_theme(&fresh);
        assert_ne!(Tokens::get(&fresh), mtech_studio::TOKENS);
        apply_saved_theme(&fresh, decoded);
        assert_eq!(Tokens::get(&fresh), mtech_studio::TOKENS);
    }

    #[test]
    fn leaving_mtech_studio_publishes_derived_tokens() {
        let ctx = Context::default();
        apply_preset(&ctx, PresetStyles::MtechStudio);
        apply_preset(&ctx, PresetStyles::Mastertech);
        let (success, accent2) = semantic_colors_for_preset(PresetStyles::Mastertech);
        assert_eq!(
            Tokens::get(&ctx),
            Tokens::derive(&ctx.global_style(), success, accent2)
        );
    }

    #[test]
    fn every_preset_publishes_opaque_tokens_with_its_semantic_colors() {
        let ctx = Context::default();
        for preset in all_presets() {
            apply_preset(&ctx, preset);
            let tokens = Tokens::get(&ctx);
            let name = preset.as_str();
            for (slot, c) in tokens.colors() {
                assert_eq!(c.a(), 255, "{name}: {slot} is translucent");
            }
            let (success, accent2) = semantic_colors_for_preset(preset);
            assert_eq!(
                tokens.success,
                composite(success, tokens.base),
                "{name}: success"
            );
            assert_eq!(
                tokens.secondary,
                composite(accent2, tokens.base),
                "{name}: secondary"
            );
        }
    }

    #[test]
    fn every_preset_name_round_trips() {
        for preset in MATCHABLE_PRESETS {
            assert_eq!(
                PresetStyles::from_label(preset.as_str()),
                preset,
                "{}",
                preset.as_str()
            );
        }
        assert_eq!(
            PresetStyles::from_label("no such preset"),
            PresetStyles::Custom
        );
    }

    #[test]
    fn saved_themes_never_name_the_weight_families() {
        let ctx = Context::default();
        for preset in all_presets() {
            apply_preset(&ctx, preset);
            let current = SavedTheme::current(&ctx);
            for style in [&current.style, &round_trip(&current).style] {
                for (text_style, font) in &style.text_styles {
                    assert!(
                        !matches!(font.family, eframe::egui::FontFamily::Name(_)),
                        "{}: {text_style:?} saved as {:?}",
                        preset.as_str(),
                        font.family,
                    );
                }
            }
        }
    }

    #[test]
    fn resetting_memory_keeps_the_theme_density_and_text_size() {
        let ctx = Context::default();
        apply_preset(&ctx, PresetStyles::MtechStudio);
        set_density(&ctx, Density::Compact);
        set_text_scale(&ctx, TextScale::Large);

        reset_memory(&ctx);

        assert_eq!(installed_preset(&ctx), PresetStyles::MtechStudio);
        assert_eq!(Metrics::get(&ctx), Metrics::for_density(Density::Compact));
        assert_eq!(fonts::text_scale(&ctx), TextScale::Large);
        assert_eq!(Tokens::get(&ctx), mtech_studio::TOKENS);
        assert_eq!(glass::params(&ctx), mtech_studio::studio_glass_params());
    }

    #[test]
    fn an_edited_style_is_custom_with_the_default_semantics() {
        let ctx = Context::default();
        let mut theme = mastertech_theme();
        theme.style.visuals.hyperlink_color = Color32::from_rgb(1, 2, 3);
        apply_saved_theme(&ctx, theme);
        assert_eq!(installed_preset(&ctx), PresetStyles::Custom);
        assert_eq!(colors::success_ctx(&ctx), DEFAULT_SUCCESS);
    }
}
