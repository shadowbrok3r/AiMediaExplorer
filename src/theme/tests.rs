//! Cross-module theme tests: stock control geometry per preset and density, and the legacy-styling ratchet.

use std::fs;
use std::path::{Path, PathBuf};

use eframe::egui::{
    Button, CentralPanel, ComboBox, Context, DragValue, Pos2, RawInput, Rect, TextEdit, Ui, vec2,
};
use regex::Regex;

use super::{
    dock_style,
    fonts::{self, TextScale},
    metrics::{Density, Metrics},
    presets::{
        MATCHABLE_PRESETS, PresetStyles, apply_preset, bootstrap_startup_theme, set_density,
    },
    tokens::Tokens,
};

fn context() -> Context {
    let ctx = Context::default();
    ctx.set_fonts(fonts::definitions(TextScale::Normal));
    bootstrap_startup_theme(&ctx);
    ctx
}

/// Runs `passes` frames of `f` in a central panel; returns the last result.
fn run<R>(ctx: &Context, passes: usize, mut f: impl FnMut(&mut Ui) -> R) -> R {
    let mut out = None;
    for _ in 0..passes {
        let input = RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1200.0, 400.0))),
            ..Default::default()
        };
        ctx.run_ui(input, |ui| {
            CentralPanel::default().show(ui, |ui| out = Some(f(ui)));
        })
        .drop_without_applying_deltas();
    }
    out.expect("the ui closure ran")
}

/// Rects of a stock Button, ComboBox, DragValue and TextEdit in one row.
fn stock_row(ui: &mut Ui) -> Vec<Rect> {
    let mut text = "Text".to_owned();
    let mut size = 20.0_f32;
    ui.horizontal(|ui| {
        let combo = ComboBox::from_id_salt("stock_row")
            .width(100.0)
            .selected_text("One")
            .show_ui(ui, |ui| {
                let _ = ui.selectable_label(true, "One");
            });
        vec![
            ui.button("Apply").rect,
            combo.response.rect,
            ui.add(DragValue::new(&mut size)).rect,
            ui.add(TextEdit::singleline(&mut text).desired_width(120.0))
                .rect,
        ]
    })
    .inner
}

#[test]
fn stock_controls_share_the_control_height_and_centre_line_for_every_preset_and_density() {
    let ctx = context();
    for preset in MATCHABLE_PRESETS {
        apply_preset(&ctx, preset);
        for density in Density::ALL {
            set_density(&ctx, density);
            let rects = run(&ctx, 2, stock_row);
            let height = Metrics::for_density(density).control_h;
            let first = rects[0];
            for (i, r) in rects.iter().enumerate() {
                assert!(
                    (r.height() - height).abs() < 0.01,
                    "{} at {density:?}: control {i} is {r:?}, want height {height}",
                    preset.as_str()
                );
                assert!(
                    (r.center().y - first.center().y).abs() < 0.01,
                    "{} at {density:?}: control {i} is off the centre line",
                    preset.as_str()
                );
            }
        }
    }
}

#[test]
fn bootstrap_pins_the_dark_slot_and_applies_the_embedded_file() {
    let ctx = Context::default();
    ctx.set_theme(eframe::egui::Theme::Light);
    super::bootstrap_context(&ctx, None);
    assert_eq!(ctx.theme(), eframe::egui::Theme::Dark);
    assert_eq!(
        super::presets::installed_preset(&ctx),
        PresetStyles::Mastertech
    );
    assert_eq!(
        super::persist::source(&ctx),
        super::persist::ThemeSource::Embedded
    );
}

#[test]
fn preferences_replace_the_options_but_keep_the_installed_style() {
    let ctx = context();
    let style = ctx.global_style();
    let mut options = eframe::egui::Options::default();
    options.zoom_factor = 1.5;
    super::apply_preferences(&ctx, options);
    assert_eq!(ctx.options(|o| o.zoom_factor), 1.5);
    assert!(std::sync::Arc::ptr_eq(&style, &ctx.global_style()));
}

#[test]
fn a_prominent_primary_button_is_taller_than_a_stock_one() {
    use eframe::egui::class::HasClasses as _;
    let ctx = context();
    let (stock, prominent) = run(&ctx, 2, |ui| {
        ui.horizontal(|ui| {
            (
                ui.button("Apply").rect,
                ui.add(Button::new("Save").with_class(super::widget_theme::PROMINENT))
                    .rect,
            )
        })
        .inner
    });
    assert!(prominent.height() > stock.height());
}

#[test]
fn dock_chrome_builds_from_the_tokens_for_every_preset_and_density() {
    let ctx = context();
    for preset in MATCHABLE_PRESETS {
        apply_preset(&ctx, preset);
        for density in Density::ALL {
            set_density(&ctx, density);
            let dock = dock_style::style(&ctx);
            let t = Tokens::get(&ctx);
            let m = Metrics::for_density(density);
            assert_eq!(dock.tab.tab_body.bg_fill, t.raised, "{}", preset.as_str());
            let margin = dock.tab_bar.inner_margin;
            let pill = dock.tab_bar.height - f32::from(margin.top + margin.bottom);
            assert_eq!(pill, m.control_h, "{} at {density:?}", preset.as_str());
        }
    }
}

#[test]
fn leaving_and_returning_to_the_default_preset_restores_the_published_look() {
    let ctx = context();
    let before = Tokens::get(&ctx);
    apply_preset(&ctx, PresetStyles::MtechNoirGlass);
    assert_ne!(Tokens::get(&ctx), before);
    apply_preset(&ctx, PresetStyles::Mastertech);
    assert_eq!(Tokens::get(&ctx), before);
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
