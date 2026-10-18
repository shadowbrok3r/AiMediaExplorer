//! The app theme: tokens, metrics, fonts, widget styling, dock chrome, presets and frosted glass, installed by [`bootstrap`].

pub mod cli;
pub mod colors;
pub mod debug_panel;
pub mod dock_style;
pub mod fonts;
pub mod glass;
pub mod kit;
pub mod metrics;
pub mod persist;
pub mod presets;
pub mod saved;
pub mod table_style;
pub mod toast;
pub mod tokens;
pub mod widget_theme;

mod mtech_glass;
mod mtech_studio;
#[cfg(test)]
mod tests;
mod theme_chrome;

use eframe::egui::Context;

pub use colors::{
    Status, StatusStyle, accent, accent_secondary, accent_secondary_ctx, bg_extreme, bg_faint,
    bg_surface, border, error, faint_text, info, other_series, primary, primary_ctx, secondary,
    secondary_ctx, series_color, set_accent_secondary, set_success_color, status_style,
    strong_text, success, success_ctx, tag_color, tertiary, tertiary_ctx, text, usage_level,
    usage_status, warn, weak_text,
};
pub use kit::{ButtonKind, danger_button, ghost_button, primary_button};
pub use metrics::{Density, Metrics};
pub use presets::{PresetStyles, apply_preset, apply_saved_theme, bootstrap_startup_theme};
pub use saved::SavedTheme;
pub use tokens::Tokens;

/// Installs glass, fonts and the startup theme (`--theme` file, saved theme or embedded default) from the creation context.
pub fn bootstrap(cc: &eframe::CreationContext<'_>) {
    glass::install(cc);
    bootstrap_context(&cc.egui_ctx, cc.storage);
}

/// [`bootstrap`] without the GL backend: fonts and the startup theme on `ctx`.
pub fn bootstrap_context(ctx: &Context, storage: Option<&dyn eframe::Storage>) {
    ctx.set_theme(eframe::egui::Theme::Dark);
    fonts::install_for_context(ctx);
    let options = cli::options();
    let (theme, source) = persist::resolve(storage, options.theme_path.as_deref());
    log::info!("theme: applying {}", source.label());
    persist::set_source(ctx, source);
    apply_saved_theme(ctx, theme);
    if options.theme_panel {
        debug_panel::set_open(ctx, true);
    }
}

/// Replaces the context options with `options`, keeping the installed styles and theme preference.
pub fn apply_preferences(ctx: &Context, options: eframe::egui::Options) {
    ctx.options_mut(|o| {
        let (dark, light) = (o.dark_style.clone(), o.light_style.clone());
        let (preference, fallback) = (o.theme_preference, o.fallback_theme);
        *o = options;
        o.dark_style = dark;
        o.light_style = light;
        o.theme_preference = preference;
        o.fallback_theme = fallback;
    });
}

/// The tokens published on `ctx`.
pub fn tokens(ctx: &Context) -> Tokens {
    Tokens::get(ctx)
}

/// The metrics published on `ctx`.
pub fn metrics(ctx: &Context) -> Metrics {
    Metrics::get(ctx)
}

/// Switches the UI text size, rebuilding the font atlas when it changes.
pub fn set_text_scale(ctx: &Context, scale: fonts::TextScale) {
    presets::set_text_scale(ctx, scale);
}

/// Switches the density and re-installs the current style under it.
pub fn set_density(ctx: &Context, density: Density) {
    presets::set_density(ctx, density);
}

/// The dock chrome built from the published tokens and metrics.
pub fn dock_style(ctx: &Context) -> egui_dock::Style {
    dock_style::style(ctx)
}

/// Writes the live theme to eframe storage; call from `eframe::App::save`.
pub fn save(ctx: &Context, storage: &mut dyn eframe::Storage) {
    persist::save(ctx, storage);
}
