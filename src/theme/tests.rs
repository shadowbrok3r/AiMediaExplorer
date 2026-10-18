//! App theme tests: the default look, the font stack and the legacy-styling ratchet.

use std::fs;
use std::path::{Path, PathBuf};

use eframe::egui::{Context, FontFamily, FontId};
use regex::Regex;
use serde_json::{Value, json};

use super::{
    cli,
    fonts::{self, TextScale},
    metrics::Metrics,
    presets::{PresetStyles, bootstrap_startup_theme, installed_preset},
    tokens::Tokens,
};

fn default_context() -> Context {
    let ctx = Context::default();
    ctx.set_fonts(fonts::definitions(TextScale::Normal));
    bootstrap_startup_theme(&ctx);
    ctx.run_ui(Default::default(), |_| {}).drop_without_applying_deltas();
    ctx
}

#[test]
fn the_default_theme_keeps_the_mastertech_look() {
    let ctx = default_context();
    assert_eq!(installed_preset(&ctx), PresetStyles::Graphite);
    let tokens = cli::tokens_json(&Tokens::get(&ctx));
    let expected: Value = json!({
        "base": "#08080a", "card_border": "#7f8190", "error": "#ec6284", "field": "#0f0f12",
        "field_border": "#747683", "hover": "#221724", "icon": "#b7b7bd", "info": "#e598c2",
        "on_primary": "#0c0c0e", "pressed": "#291d30", "primary": "#d184de", "primary_border": "#8d5b97",
        "primary_soft": "#2c1e30", "radius_lg": 5, "radius_md": 4, "radius_sm": 3, "raised": "#0d0d12",
        "secondary": "#bf2165", "secondary_soft": "#290d1a", "separator": "#35353c", "success": "#48c78e",
        "sunken": "#0f0f12", "tertiary": "#e598c2", "tertiary_soft": "#30222b", "text": "#e2e2ea",
        "text_dim": "#8b8b90", "text_faint": "#5d5d61", "warn": "#7ec8e6", "window": "#111115",
    });
    assert_eq!(tokens, expected);
    let metrics = cli::metrics_json(&Metrics::get(&ctx));
    let expected: Value = json!({
        "bar_h": 32.0, "button_h": 26.0, "combo_w": 110.0, "control_h": 22.0, "footer_h": 28.0,
        "gap": [6.0, 4.0], "header_h": 26.0, "icon_lg": 18.0, "icon_md": 14.0, "icon_sm": 12.0,
        "menu_margin": 4, "pad": [8.0, 3.0], "radius_lg": 12, "radius_md": 8, "radius_sm": 6,
        "row_h": 22.0, "slider_w": 130.0, "tooltip_w": 280.0, "window_margin": 10,
    });
    assert_eq!(metrics, expected);
    let resolved = cli::resolved_json(&ctx);
    assert_eq!(resolved["density"], "Compact");
    assert_eq!(resolved["text_scale"], "Small (90%)");
    assert_eq!(resolved["glass"]["blur_radius"], 32.0);
    assert_eq!(resolved["text_styles"]["Heading"]["family"], "semibold");
}

#[test]
fn the_font_stack_orders_the_app_faces_under_inter_and_the_icon_font() {
    let defs = fonts::definitions(TextScale::Normal);
    let keys = |family: FontFamily| defs.families[&family].clone();
    assert_eq!(
        keys(FontFamily::Proportional),
        ["Inter", "Ubuntu-Light", "phosphor", "egui-icons", "NotoEmoji-Regular", "emoji-icon-font"]
    );
    assert_eq!(
        keys(FontFamily::Monospace),
        ["UbuntuSansMono", "phosphor", "UbuntuMonoNerdFont", "Hack", "Ubuntu-Light", "egui-icons", "NotoEmoji-Regular", "emoji-icon-font"]
    );
    assert_eq!(keys(FontFamily::Name("Regular".into())), ["Regular", "phosphor"]);
    assert_eq!(keys(FontFamily::Name("Bold".into())), ["Bold", "phosphor"]);
    assert_eq!(keys(FontFamily::Name("medium".into()))[0], "Inter-Medium");
    assert_eq!(keys(FontFamily::Name("semibold".into()))[0], "Inter-SemiBold");
    for family in defs.families.values() {
        assert!(family.iter().all(|key| defs.font_data.contains_key(key)), "{family:?}");
    }
}

#[test]
fn icon_glyphs_resolve_to_phosphor_in_the_text_and_monospace_families() {
    use eframe::epaint::text::{Fonts, TextOptions};

    let mut defs = fonts::definitions(TextScale::Normal);
    defs.families.insert(FontFamily::Name("icon-only".into()), vec!["phosphor".to_owned()]);
    let mut atlas = Fonts::new(TextOptions::default(), defs);
    let mut view = atlas.with_pixels_per_point(1.0);
    let glyph = egui_phosphor::regular::GEAR.chars().next().expect("glyph");
    let icon = view.glyph_width(&FontId::new(14.0, FontFamily::Name("icon-only".into())), glyph);
    assert!(icon > 0.0, "the icon face lacks the glyph");
    for family in [FontFamily::Proportional, FontFamily::Monospace, FontFamily::Name("semibold".into())] {
        let width = view.glyph_width(&FontId::new(14.0, family.clone()), glyph);
        assert_eq!(width, icon, "{glyph:?} in {family:?} is not the icon face's");
    }
}

#[test]
fn the_theme_dump_lists_the_app_faces() {
    let dump = cli::dump_with(&cli::Options::default(), &super::font_config()).expect("dump");
    let dump: Value = serde_json::from_str(&dump).expect("json");
    assert_eq!(dump["font_families"]["Monospace"][0], "UbuntuSansMono");
    assert_eq!(dump["font_families"]["Proportional"][0], "Inter");
    assert_eq!(dump["source"], "embedded default");
}

/// Most `Color32` constructors and named colours in the UI files this theme restyled.
const COLOR_LITERALS: usize = 2;
/// Most `add_space` calls whose argument starts with a number literal in the UI files this theme restyled.
const ADD_SPACE_LITERALS: usize = 42;

/// Files and folders under `src/` whose chrome follows the theme.
const THEMED: &[&str] = &["receive.rs", "ui/tabs.rs", "ui/status.rs", "ui/file_table"];

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

fn themed_sources() -> Vec<(String, String)> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    for entry in THEMED {
        let path = root.join(entry);
        if path.is_dir() {
            collect(&path, &mut files);
        } else {
            files.push(path);
        }
    }
    assert!(
        files.len() > 10,
        "only {} themed source files found",
        files.len()
    );
    files
        .into_iter()
        .filter_map(|p| {
            let rel = p
                .strip_prefix(&root)
                .ok()?
                .to_string_lossy()
                .replace('\\', "/");
            Some((rel, fs::read_to_string(&p).ok()?))
        })
        .collect()
}

fn ratchet(name: &str, ceiling: usize, pattern: &Regex, skip: impl Fn(&str) -> bool) {
    let mut per_file: Vec<(usize, String)> = themed_sources()
        .into_iter()
        .map(|(path, text)| {
            let n = pattern
                .find_iter(&text)
                .filter(|m| !skip(m.as_str()))
                .count();
            (n, path)
        })
        .filter(|(n, _)| *n > 0)
        .collect();
    let total: usize = per_file.iter().map(|(n, _)| n).sum();
    per_file.sort_unstable_by(|a, b| b.cmp(a));
    let top: Vec<String> = per_file
        .iter()
        .take(6)
        .map(|(n, p)| format!("{n} {p}"))
        .collect();
    assert!(
        total <= ceiling,
        "{name}: {total} found, ceiling {ceiling}; use theme tokens and semantic colors instead. Top files: {top:#?}"
    );
}

#[test]
fn color_literals_in_themed_ui_files_only_go_down() {
    let pattern = Regex::new(r"Color32::(?:from_\w+\(|[A-Z][A-Z0-9_]*\b)").unwrap();
    ratchet("Color32 literals", COLOR_LITERALS, &pattern, |m| {
        matches!(m, "Color32::TRANSPARENT" | "Color32::PLACEHOLDER")
    });
}

#[test]
fn add_space_literals_in_themed_ui_files_only_go_down() {
    let pattern = Regex::new(r"add_space\(\s*[-+]?\d[\d_.]*(?:_?f32)?\s*\)").unwrap();
    ratchet("add_space(<literal>)", ADD_SPACE_LITERALS, &pattern, |_| {
        false
    });
}

#[test]
fn the_ratchet_patterns_see_each_pattern_and_skip_lookalikes() {
    let color = Regex::new(r"Color32::(?:from_\w+\(|[A-Z][A-Z0-9_]*\b)").unwrap();
    let text = "let a = Color32::from_rgb(1, 2, 3); let b = egui::Color32::RED; let c = Color32::TRANSPARENT;";
    let counted = color
        .find_iter(text)
        .filter(|m| m.as_str() != "Color32::TRANSPARENT")
        .count();
    assert_eq!(counted, 2);
    let space = Regex::new(r"add_space\(\s*[-+]?\d[\d_.]*(?:_?f32)?\s*\)").unwrap();
    assert_eq!(
        space
            .find_iter("ui.add_space(4.0); ui.add_space(m.gap.y); ui.add_space(2.0 * m.gap.y);")
            .count(),
        1
    );
}
