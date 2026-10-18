//! The "Theme" debug window: [`show`] draws it when open, [`toggle`] and [`is_open`] drive it, and Ctrl+Alt+T toggles it.

use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Duration;

use eframe::egui::class::HasClasses as _;
use eframe::egui::{
    self, Align, CollapsingHeader, Color32, ComboBox, Context, DragValue, Grid, Id, Key,
    KeyboardShortcut, Layout, Modifiers, ProgressBar, RichText, ScrollArea, Sense, Slider, Spinner,
    Stroke, StrokeKind, TextEdit, TextStyle, Ui, Vec2, Window, vec2,
};
use egui_toast::ToastKind;

use super::{
    colors,
    fonts::{self, RichTextExt as _, TextScale},
    glass::{self, GlassParams},
    kit::{self, ButtonKind},
    metrics::{self, Density, Metrics},
    persist,
    presets::{
        self, PRESET_GROUPS, PresetStyles, TOP_PRESET, apply_preset, apply_saved_theme,
        installed_preset,
    },
    saved::{SavedTheme, theme_from_json, theme_to_json},
    theme_chrome::mastertech_theme,
    tokens::{Tokens, contrast},
    widget_theme,
};

const SHORTCUT: KeyboardShortcut =
    KeyboardShortcut::new(Modifiers::CTRL.plus(Modifiers::ALT), Key::T);
/// Smoothing of the frame-time readout per sample.
const EMA: f32 = 0.05;
/// Tallest the JSON box grows before it scrolls.
const JSON_BOX_HEIGHT: f32 = 110.0;
/// Width of one swatch chip.
const CHIP: Vec2 = vec2(22.0, 14.0);

static DIALOG_RESULTS: Mutex<Vec<DialogResult>> = Mutex::new(Vec::new());
static TOAST_REQUESTS: Mutex<Vec<(ToastKind, String)>> = Mutex::new(Vec::new());

/// Outcome of a file dialog run off the UI thread.
enum DialogResult {
    Imported(Result<SavedTheme, String>, PathBuf),
    Exported(Result<PathBuf, String>),
}

fn open_id() -> Id {
    Id::unique("smart_media.theme.panel.open")
}

fn state_id() -> Id {
    Id::unique("smart_media.theme.panel.state")
}

fn stats_id() -> Id {
    Id::unique("smart_media.theme.panel.frame_stats")
}

/// Whether the Theme window is open.
pub fn is_open(ctx: &Context) -> bool {
    ctx.data(|d| d.get_temp::<bool>(open_id())).unwrap_or(false)
}

/// Opens or closes the Theme window.
pub fn set_open(ctx: &Context, open: bool) {
    ctx.data_mut(|d| d.insert_temp(open_id(), open));
    ctx.request_repaint();
}

/// Flips the Theme window.
pub fn toggle(ctx: &Context) {
    set_open(ctx, !is_open(ctx));
}

/// Toast requests raised by the panel's tester buttons, drained by the app each frame.
pub fn take_toast_requests() -> Vec<(ToastKind, String)> {
    TOAST_REQUESTS
        .lock()
        .map(|mut q| std::mem::take(&mut *q))
        .unwrap_or_default()
}

#[derive(Clone, Copy, Default)]
struct FrameStats {
    ui_ms: f32,
    frame_ms: f32,
}

/// Moves `current` toward `sample`, starting from the first sample.
fn smooth(current: f32, sample: f32) -> f32 {
    if current == 0.0 {
        sample
    } else {
        current + (sample - current) * EMA
    }
}

/// Records how long building the UI took this frame, for the frame-time readout.
pub fn record_ui_time(ctx: &Context, elapsed: Duration) {
    let sample = elapsed.as_secs_f32() * 1000.0;
    ctx.data_mut(|d| {
        let stats = d.get_temp_mut_or_default::<FrameStats>(stats_id());
        stats.ui_ms = smooth(stats.ui_ms, sample);
    });
}

/// Records the previous frame's CPU seconds from eframe, UI and rendering included.
pub fn record_frame_cpu(ctx: &Context, seconds: Option<f32>) {
    let Some(seconds) = seconds else {
        return;
    };
    ctx.data_mut(|d| {
        let stats = d.get_temp_mut_or_default::<FrameStats>(stats_id());
        stats.frame_ms = smooth(stats.frame_ms, seconds * 1000.0);
    });
}

#[derive(Clone)]
struct Gallery {
    text: String,
    multiline: String,
    check: bool,
    check_off: bool,
    radio: u8,
    slider: f32,
    drag: f32,
    combo: usize,
    progress: f32,
    selected: usize,
    collapse_open: bool,
}

impl Default for Gallery {
    fn default() -> Self {
        Self {
            text: "Sample text".to_owned(),
            multiline: "Multi-line\ntext edit".to_owned(),
            check: true,
            check_off: false,
            radio: 0,
            slider: 40.0,
            drag: 12.5,
            combo: 1,
            progress: 0.62,
            selected: 1,
            collapse_open: true,
        }
    }
}

#[derive(Clone, Default)]
struct PanelState {
    json: String,
    json_message: Option<(bool, String)>,
    user_name: String,
    user_message: Option<(bool, String)>,
    gallery: Gallery,
}

/// Draws the Theme window when it is open and handles its shortcut; call once per frame.
pub fn show(ctx: &Context) {
    if ctx.input_mut(|i| i.consume_shortcut(&SHORTCUT)) {
        toggle(ctx);
    }
    if !is_open(ctx) {
        return;
    }
    let mut open = true;
    let mut state = ctx
        .data_mut(|d| d.remove_temp::<PanelState>(state_id()))
        .unwrap_or_default();
    drain_dialogs(ctx, &mut state);
    Window::new("Theme")
        .id(Id::unique("smart_media_theme_panel"))
        .open(&mut open)
        .default_size(vec2(560.0, 700.0))
        .show(ctx, |ui| contents(ui, &mut state));
    ctx.data_mut(|d| d.insert_temp(state_id(), state));
    if !open {
        set_open(ctx, false);
    }
}

fn drain_dialogs(ctx: &Context, state: &mut PanelState) {
    let results = DIALOG_RESULTS
        .lock()
        .map(|mut q| std::mem::take(&mut *q))
        .unwrap_or_default();
    for result in results {
        match result {
            DialogResult::Imported(Ok(theme), path) => {
                apply_saved_theme(ctx, theme);
                state.json_message = Some((true, format!("Imported {}", path.display())));
            }
            DialogResult::Imported(Err(e), path) => {
                state.json_message = Some((false, format!("{}: {e}", path.display())));
            }
            DialogResult::Exported(Ok(path)) => {
                state.json_message = Some((true, format!("Exported to {}", path.display())));
            }
            DialogResult::Exported(Err(e)) => state.json_message = Some((false, e)),
        }
    }
}

fn contents(ui: &mut Ui, state: &mut PanelState) {
    let ctx = ui.ctx().clone();
    status_line(ui, &ctx);
    ui.separator();
    ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            section(ui, "Preset, density and text size", true, |ui| {
                presets_section(ui, &ctx);
            });
            section(ui, "Glass", true, |ui| glass_section(ui, &ctx));
            section(ui, "Token swatches", false, |ui| token_section(ui, &ctx));
            section(ui, "Semantic colors", false, |ui| semantic_section(ui));
            section(ui, "Metrics", false, |ui| metrics_section(ui, &ctx));
            section(ui, "Widget gallery", false, |ui| {
                gallery(ui, &mut state.gallery)
            });
            section(ui, "JSON", false, |ui| json_section(ui, &ctx, state));
            section(ui, "User themes", false, |ui| user_section(ui, &ctx, state));
            section(ui, "Toast tester", false, |ui| toast_section(ui));
        });
}

fn section(ui: &mut Ui, title: &str, open: bool, add: impl FnOnce(&mut Ui)) {
    CollapsingHeader::new(RichText::new(title).semibold())
        .default_open(open)
        .show(ui, add);
}

fn status_line(ui: &mut Ui, ctx: &Context) {
    let preset = installed_preset(ctx);
    let stats = ctx
        .data(|d| d.get_temp::<FrameStats>(stats_id()))
        .unwrap_or_default();
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(preset.as_str()).semibold());
        ui.label(
            RichText::new(format!(
                "{} / {} / source: {}",
                metrics::density(ctx).label(),
                fonts::text_scale(ctx).label(),
                persist::source(ctx).label()
            ))
            .weak(),
        );
        ui.label(
            RichText::new(format!(
                "UI {:.1} ms, frame CPU {:.1} ms",
                stats.ui_ms, stats.frame_ms
            ))
            .weak()
            .small(),
        );
    });
}

fn presets_section(ui: &mut Ui, ctx: &Context) {
    let current = installed_preset(ctx);
    ui.horizontal(|ui| {
        ui.label("Preset");
        let mut picked = None;
        ComboBox::from_id_salt("theme_panel_preset")
            .selected_text(current.as_str())
            .show_ui(ui, |ui| {
                if ui
                    .selectable_label(current == TOP_PRESET, TOP_PRESET.as_str())
                    .clicked()
                {
                    picked = Some(TOP_PRESET);
                }
                for group in &PRESET_GROUPS {
                    ui.separator();
                    ui.label(RichText::new(group.title).weak().small());
                    for &preset in group.presets {
                        if ui
                            .selectable_label(current == preset, preset.as_str())
                            .clicked()
                        {
                            picked = Some(preset);
                        }
                    }
                }
                ui.separator();
                ui.add_enabled(
                    false,
                    egui::Button::new("Custom").selected(current == PresetStyles::Custom),
                );
            });
        if let Some(preset) = picked.filter(|p| *p != current) {
            apply_preset(ctx, preset);
        }
        if ui
            .add(kit::button(ButtonKind::Ghost, "Reset to default"))
            .on_hover_text(
                "Apply the embedded Mastertech theme with its density, text size and glass.",
            )
            .clicked()
        {
            apply_saved_theme(ctx, mastertech_theme());
        }
    });

    ui.horizontal(|ui| {
        let mut density = metrics::density(ctx);
        ui.label("Density");
        ComboBox::from_id_salt("theme_panel_density")
            .selected_text(density.label())
            .show_ui(ui, |ui| {
                for d in Density::ALL {
                    ui.selectable_value(&mut density, d, d.label());
                }
            });
        presets::set_density(ctx, density);

        let mut scale = fonts::text_scale(ctx);
        ui.label("Text size");
        ComboBox::from_id_salt("theme_panel_scale")
            .selected_text(scale.label())
            .show_ui(ui, |ui| {
                for s in TextScale::ALL {
                    ui.selectable_value(&mut scale, s, s.label());
                }
            });
        presets::set_text_scale(ctx, scale);
    });

    let mut stock = !widget_theme::enabled(ctx);
    if ui
        .checkbox(
            &mut stock,
            "Stock egui widget look (theme widget styling off)",
        )
        .changed()
    {
        widget_theme::set_enabled(ctx, !stock);
    }
}

fn glass_section(ui: &mut Ui, ctx: &Context) {
    let mut params = glass::params(ctx);
    let before = params;
    let status = glass::status();

    ui.horizontal_wrapped(|ui| {
        ui.add_enabled_ui(status.backend_installed && !status.disabled, |ui| {
            ui.checkbox(&mut params.enabled, "Backdrop blur");
        });
        ui.label(RichText::new(format!("frost: {}", status.summary())).weak());
    });
    ui.add_enabled_ui(params.enabled, |ui| {
        Grid::new("theme_panel_glass_grid")
            .num_columns(2)
            .spacing(vec2(12.0, 4.0))
            .show(ui, |ui| {
                ui.label("Blur radius");
                ui.add(
                    DragValue::new(&mut params.blur_radius)
                        .range(0.0..=64.0)
                        .speed(0.5)
                        .suffix(" px"),
                );
                ui.end_row();
                ui.label("Tint");
                ui.color_edit_button_srgba(&mut params.tint);
                ui.end_row();
                ui.label("Corner radius");
                ui.add(
                    DragValue::new(&mut params.corner_radius)
                        .range(0.0..=32.0)
                        .speed(0.2)
                        .suffix(" px"),
                );
                ui.end_row();
                ui.label("Presence");
                ui.add(Slider::new(&mut params.presence, 0.0..=1.0).step_by(0.01));
                ui.end_row();
            });
    });
    if params != before {
        glass::set_params(ctx, params);
    }

    ui.horizontal_wrapped(|ui| {
        if ui
            .add(kit::button(ButtonKind::Ghost, "Material off"))
            .clicked()
        {
            glass::set_params(ctx, GlassParams::OFF);
        }
        if ui
            .add(kit::button(ButtonKind::Ghost, "Disable frost"))
            .on_hover_text("Trips the process-wide kill switch, like --no-frost.")
            .clicked()
        {
            glass::disable("disabled from the Theme panel");
        }
        if ui
            .add(kit::button(ButtonKind::Ghost, "Re-enable frost"))
            .on_hover_text(
                "Clears the kill switch and the failure run; the environment switch stays.",
            )
            .clicked()
        {
            glass::reenable();
        }
    });

    Grid::new("theme_panel_frost_grid")
        .num_columns(2)
        .spacing(vec2(12.0, 2.0))
        .show(ui, |ui| {
            let row = |ui: &mut Ui, name: &str, value: String| {
                ui.label(RichText::new(name).weak());
                ui.label(value);
                ui.end_row();
            };
            row(ui, "GL backend", yes_no(status.backend_installed));
            row(ui, "Kill switch", yes_no(status.disabled));
            row(ui, "Environment switch", yes_no(status.env_disabled));
            row(
                ui,
                "Last frame",
                status
                    .last
                    .map_or("none yet".to_owned(), |r| r.label().to_owned()),
            );
            row(
                ui,
                "Composited frames",
                status.composited_frames.to_string(),
            );
            row(ui, "Failed run", status.consecutive_failures.to_string());
            row(
                ui,
                "Reason",
                status.reason.clone().unwrap_or_else(|| "none".to_owned()),
            );
        });

    glass::preview(ui, glass::params(ctx));
}

fn yes_no(v: bool) -> String {
    if v { "yes" } else { "no" }.to_owned()
}

fn chip(ui: &mut Ui, color: Color32) {
    let (rect, _) = ui.allocate_exact_size(CHIP, Sense::hover());
    ui.painter().rect_filled(rect, 3.0, color);
    ui.painter().rect_stroke(
        rect,
        3.0,
        Stroke::new(1.0, colors::border(ui)),
        StrokeKind::Inside,
    );
}

fn hex(c: Color32) -> String {
    format!("#{:02x}{:02x}{:02x}", c.r(), c.g(), c.b())
}

fn token_section(ui: &mut Ui, ctx: &Context) {
    let t = Tokens::get(ctx);
    Grid::new("theme_panel_tokens")
        .num_columns(6)
        .spacing(vec2(8.0, 3.0))
        .show(ui, |ui| {
            for (i, (name, color)) in t.colors().into_iter().enumerate() {
                chip(ui, color);
                ui.label(name);
                ui.label(RichText::new(hex(color)).weak().monospace());
                if i % 2 == 1 {
                    ui.end_row();
                }
            }
        });
    ui.add_space(4.0);
    ui.label(format!(
        "radii: sm {}  md {}  lg {}   contrast text/base {:.1}, text_dim/base {:.1}",
        t.radius_sm,
        t.radius_md,
        t.radius_lg,
        contrast(t.text, t.base),
        contrast(t.text_dim, t.base)
    ));
}

fn semantic_section(ui: &mut Ui) {
    let rows: [(&str, Color32); 17] = [
        ("error", colors::error(ui)),
        ("warn", colors::warn(ui)),
        ("info", colors::info(ui)),
        ("success", colors::success(ui)),
        ("accent", colors::accent(ui)),
        ("accent_secondary", colors::accent_secondary(ui)),
        ("primary", colors::primary(ui)),
        ("secondary", colors::secondary(ui)),
        ("tertiary", colors::tertiary(ui)),
        ("strong_text", colors::strong_text(ui)),
        ("text", colors::text(ui)),
        ("weak_text", colors::weak_text(ui)),
        ("faint_text", colors::faint_text(ui)),
        ("border", colors::border(ui)),
        ("bg_surface", colors::bg_surface(ui)),
        ("bg_faint", colors::bg_faint(ui)),
        ("bg_extreme", colors::bg_extreme(ui)),
    ];
    Grid::new("theme_panel_semantic")
        .num_columns(4)
        .spacing(vec2(8.0, 3.0))
        .show(ui, |ui| {
            for (i, (name, color)) in rows.into_iter().enumerate() {
                chip(ui, color);
                ui.label(name);
                if i % 2 == 1 {
                    ui.end_row();
                }
            }
        });
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.label("tag colors");
        for i in 0..colors::TAG_LEN {
            chip(ui, colors::tag_color(ui, i));
        }
        ui.label("series");
        for i in 0..colors::SERIES_LEN {
            if let Some(c) = colors::series_color(i) {
                chip(ui, c);
            }
        }
    });
    ui.horizontal(|ui| {
        ui.label("usage");
        for pct in [10.0, 75.0, 95.0] {
            ui.colored_label(colors::usage_level(ui, pct), format!("{pct:.0}%"));
        }
    });
}

fn metrics_section(ui: &mut Ui, ctx: &Context) {
    let m = Metrics::get(ctx);
    let rows = [
        ("control_h", m.control_h),
        ("button_h", m.button_h),
        ("pad.x", m.pad.x),
        ("pad.y", m.pad.y),
        ("gap.x", m.gap.x),
        ("gap.y", m.gap.y),
        ("window_margin", m.window_margin as f32),
        ("menu_margin", m.menu_margin as f32),
        ("combo_w", m.combo_w),
        ("slider_w", m.slider_w),
        ("tooltip_w", m.tooltip_w),
        ("icon_sm", m.icon_sm),
        ("icon_md", m.icon_md),
        ("icon_lg", m.icon_lg),
        ("row_h", m.row_h),
        ("header_h", m.header_h),
        ("bar_h", m.bar_h),
        ("footer_h", m.footer_h),
    ];
    Grid::new("theme_panel_metrics")
        .num_columns(6)
        .spacing(vec2(8.0, 2.0))
        .show(ui, |ui| {
            for (i, (name, value)) in rows.into_iter().enumerate() {
                ui.label(RichText::new(name).weak());
                ui.label(RichText::new(format!("{value}")).monospace());
                if i % 3 == 2 {
                    ui.end_row();
                }
            }
        });
    let style = ctx.global_style();
    ui.add_space(4.0);
    ui.label(format!(
        "type scale: small {}  body {}  button {}  monospace {}  heading {}",
        style.text_styles[&TextStyle::Small].size,
        style.text_styles[&TextStyle::Body].size,
        style.text_styles[&TextStyle::Button].size,
        style.text_styles[&TextStyle::Monospace].size,
        style.text_styles[&TextStyle::Heading].size,
    ));
}

fn gallery(ui: &mut Ui, g: &mut Gallery) {
    ui.label(RichText::new("Buttons").weak().small());
    ui.horizontal_wrapped(|ui| {
        for kind in ButtonKind::ALL {
            ui.add(kit::button(kind, kind.label()));
        }
        ui.add(kit::button(ButtonKind::Primary, "Prominent").with_class(widget_theme::PROMINENT));
        ui.add(egui::Button::new("Small").small());
        ui.add_enabled(false, egui::Button::new("Disabled"));
        ui.add(egui::Button::new("Selected").selected(true));
        ui.add(egui::Button::new("Frameless").frame(false));
    });

    ui.label(RichText::new("Toggles").weak().small());
    ui.horizontal_wrapped(|ui| {
        ui.checkbox(&mut g.check, "Checked");
        ui.checkbox(&mut g.check_off, "Unchecked");
        ui.radio_value(&mut g.radio, 0, "Radio A");
        ui.radio_value(&mut g.radio, 1, "Radio B");
    });

    ui.label(RichText::new("Fields").weak().small());
    ui.horizontal_wrapped(|ui| {
        ui.add(TextEdit::singleline(&mut g.text).desired_width(140.0));
        ui.add(
            TextEdit::singleline(&mut String::new())
                .hint_text("Hint text")
                .desired_width(120.0),
        );
        ComboBox::from_id_salt("theme_panel_gallery_combo")
            .selected_text(["Alpha", "Beta", "Gamma"][g.combo])
            .show_ui(ui, |ui| {
                for (i, name) in ["Alpha", "Beta", "Gamma"].into_iter().enumerate() {
                    ui.selectable_value(&mut g.combo, i, name);
                }
            });
        ui.add(DragValue::new(&mut g.drag).speed(0.1).suffix(" px"));
    });
    ui.add(
        TextEdit::multiline(&mut g.multiline)
            .desired_rows(2)
            .desired_width(f32::INFINITY),
    );
    ui.add(Slider::new(&mut g.slider, 0.0..=100.0).text("Slider"));
    ui.add(ProgressBar::new(g.progress).show_percentage());
    ui.horizontal(|ui| {
        ui.add(Spinner::new());
        ui.label("Spinner");
        ui.hyperlink_to("Hyperlink", "https://example.com");
        ui.code("code");
    });

    ui.label(RichText::new("Text").weak().small());
    ui.horizontal_wrapped(|ui| {
        ui.heading("Heading");
        ui.label("Body");
        ui.label(RichText::new("Medium").medium());
        ui.label(RichText::new("Semibold").semibold());
        ui.label(RichText::new("Small").small());
        ui.label(RichText::new("Monospace 0123").monospace());
        ui.label(RichText::new("Weak").weak());
        ui.label(RichText::new("Strong").strong());
        ui.colored_label(colors::error(ui), "Error");
        ui.colored_label(colors::warn(ui), "Warn");
        ui.colored_label(colors::success(ui), "Success");
        ui.label("Emoji \u{1F5FA} \u{1F4DC} \u{1F5D1} \u{26A1} \u{2714} \u{2716}");
    });

    ui.label(RichText::new("Selection").weak().small());
    ui.horizontal_wrapped(|ui| {
        for (i, name) in ["Row one", "Row two", "Row three"].into_iter().enumerate() {
            if ui.selectable_label(g.selected == i, name).clicked() {
                g.selected = i;
            }
        }
    });

    CollapsingHeader::new("Collapsing header")
        .default_open(g.collapse_open)
        .show(ui, |ui| {
            ui.label("Inside the header.");
            ui.separator();
            ui.label("After a separator.");
        });

    ui.label(RichText::new("Table").weak().small());
    table(ui, g);

    ui.label(RichText::new("Overlays").weak().small());
    ui.horizontal_wrapped(|ui| {
        ui.label("Hover for a tooltip")
            .on_hover_text("Plain tooltip text.");
        ui.label("Rich tooltip").on_hover_ui(|ui| {
            ui.heading("Tooltip");
            ui.label("With several widgets.");
            ui.add(ProgressBar::new(0.4));
        });
        let popup_btn = ui.button("Popup");
        egui::Popup::from_toggle_button_response(&popup_btn).show(|ui| {
            ui.label("Popup contents");
            let _ = ui.button("Popup action");
        });
        ui.menu_button("Menu", |ui| {
            let _ = ui.button("Menu item");
            ui.checkbox(&mut g.check, "Menu check");
            ui.separator();
            ui.menu_button("Submenu", |ui| {
                let _ = ui.button("Nested item");
            });
        });
    });
}

fn table(ui: &mut Ui, g: &mut Gallery) {
    use egui_extras::{Column, TableBuilder};
    TableBuilder::new(ui)
        .striped(true)
        .id_salt("theme_panel_gallery_table")
        .column(Column::exact(120.0))
        .column(Column::exact(80.0))
        .column(Column::remainder())
        .sense(Sense::click())
        .max_scroll_height(130.0)
        .header(20.0, |mut header| {
            for title in ["Name", "Size", "Modified"] {
                header.col(|ui| {
                    ui.strong(title);
                });
            }
        })
        .body(|mut body| {
            for (i, (name, size)) in [
                ("sunset.png", "2.4 MB"),
                ("clip.mp4", "48 MB"),
                ("notes.txt", "1 KB"),
                ("scan.tiff", "9.1 MB"),
            ]
            .into_iter()
            .enumerate()
            {
                body.row(22.0, |mut row| {
                    row.set_selected(g.selected == i);
                    row.col(|ui| {
                        ui.label(name);
                    });
                    row.col(|ui| {
                        ui.label(size);
                    });
                    row.col(|ui| {
                        ui.label(RichText::new("2026-10-18").weak());
                    });
                    if row.response().clicked() {
                        g.selected = i;
                    }
                });
            }
        });
}

fn json_section(ui: &mut Ui, ctx: &Context, state: &mut PanelState) {
    ui.horizontal_wrapped(|ui| {
        if ui.button("Export to file").clicked() {
            export_file(ctx);
        }
        if ui.button("Import from file").clicked() {
            import_file(ctx);
        }
        if ui.button("Load current into box").clicked() {
            match theme_to_json(&SavedTheme::current(ctx)) {
                Ok(bytes) => {
                    state.json = String::from_utf8_lossy(&bytes).into_owned();
                    state.json_message = None;
                }
                Err(e) => state.json_message = Some((false, e.to_string())),
            }
        }
        if ui.button("Copy to clipboard").clicked() {
            if let Ok(bytes) = theme_to_json(&SavedTheme::current(ctx)) {
                ctx.copy_text(String::from_utf8_lossy(&bytes).into_owned());
            }
        }
    });
    ScrollArea::vertical()
        .id_salt("theme_panel_json_scroll")
        .max_height(JSON_BOX_HEIGHT)
        .show(ui, |ui| {
            ui.add(
                TextEdit::multiline(&mut state.json)
                    .code_editor()
                    .hint_text("Paste theme JSON (wrapper or bare egui Style)")
                    .desired_rows(5)
                    .desired_width(f32::INFINITY),
            );
        });
    ui.horizontal(|ui| {
        if ui
            .add_enabled(
                !state.json.trim().is_empty(),
                kit::button(ButtonKind::Primary, "Apply JSON"),
            )
            .clicked()
        {
            state.json_message = Some(match theme_from_json(state.json.as_bytes()) {
                Ok(theme) => {
                    apply_saved_theme(ctx, theme);
                    (true, "Applied".to_owned())
                }
                Err(e) => (false, format!("{e:#}")),
            });
        }
        if let Some((ok, message)) = &state.json_message {
            let color = if *ok {
                colors::success(ui)
            } else {
                colors::error(ui)
            };
            ui.colored_label(color, message);
        }
    });
}

fn export_file(ctx: &Context) {
    let theme = SavedTheme::current(ctx);
    let ctx = ctx.clone();
    std::thread::spawn(move || {
        let Some(path) = rfd::FileDialog::new()
            .set_title("Export theme")
            .set_file_name("smart_media_theme.json")
            .add_filter("Json", &["json"])
            .save_file()
        else {
            return;
        };
        let result = theme_to_json(&theme)
            .map_err(|e| e.to_string())
            .and_then(|json| std::fs::write(&path, json).map_err(|e| e.to_string()))
            .map(|_| path);
        push_dialog(&ctx, DialogResult::Exported(result));
    });
}

fn import_file(ctx: &Context) {
    let ctx = ctx.clone();
    std::thread::spawn(move || {
        let Some(path) = rfd::FileDialog::new()
            .set_title("Import theme")
            .add_filter("Json", &["json"])
            .pick_file()
        else {
            return;
        };
        let result = persist::load_file(&path).map_err(|e| format!("{e:#}"));
        push_dialog(&ctx, DialogResult::Imported(result, path));
    });
}

fn push_dialog(ctx: &Context, result: DialogResult) {
    if let Ok(mut q) = DIALOG_RESULTS.lock() {
        q.push(result);
    }
    ctx.request_repaint();
}

/// Directory holding user-saved theme files.
pub fn user_theme_dir() -> Option<PathBuf> {
    directories::ProjectDirs::from("", "", "smart-media")
        .map(|dirs| dirs.config_dir().join("themes"))
}

fn user_theme_files() -> Vec<PathBuf> {
    let Some(dir) = user_theme_dir() else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    files
}

/// A file-name-safe form of `name`; empty when nothing usable remains.
pub fn sanitized_name(name: &str) -> String {
    name.trim()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .to_owned()
}

/// Writes the live theme as `<name>.json` in [`user_theme_dir`]; returns the path.
pub fn save_user_theme(ctx: &Context, name: &str) -> anyhow::Result<PathBuf> {
    let dir =
        user_theme_dir().ok_or_else(|| anyhow::anyhow!("no config directory on this host"))?;
    write_user_theme(&dir, ctx, name)
}

/// Writes the live theme as `<name>.json` in `dir`, creating it; returns the path.
fn write_user_theme(dir: &std::path::Path, ctx: &Context, name: &str) -> anyhow::Result<PathBuf> {
    let name = sanitized_name(name);
    anyhow::ensure!(
        !name.is_empty(),
        "enter a name made of letters, digits, - or _"
    );
    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!("{name}.json"));
    std::fs::write(&path, theme_to_json(&SavedTheme::current(ctx))?)?;
    Ok(path)
}

fn user_section(ui: &mut Ui, ctx: &Context, state: &mut PanelState) {
    ui.horizontal(|ui| {
        ui.add(
            TextEdit::singleline(&mut state.user_name)
                .hint_text("Theme name")
                .desired_width(160.0),
        );
        if ui
            .add(kit::button(ButtonKind::Primary, "Save as user theme"))
            .clicked()
        {
            state.user_message = Some(match save_user_theme(ctx, &state.user_name) {
                Ok(path) => (true, format!("Saved {}", path.display())),
                Err(e) => (false, format!("{e:#}")),
            });
        }
    });
    if let Some((ok, message)) = &state.user_message {
        let color = if *ok {
            colors::success(ui)
        } else {
            colors::error(ui)
        };
        ui.colored_label(color, message);
    }
    if let Some(dir) = user_theme_dir() {
        ui.label(RichText::new(dir.display().to_string()).weak().small());
    }
    for path in user_theme_files() {
        let label = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        ui.horizontal(|ui| {
            ui.label(&label);
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui.add(kit::button(ButtonKind::Danger, "Delete")).clicked() {
                    let _ = std::fs::remove_file(&path);
                }
                if ui.button("Load").clicked() {
                    state.user_message = Some(match persist::load_file(&path) {
                        Ok(theme) => {
                            apply_saved_theme(ctx, theme);
                            (true, format!("Loaded {label}"))
                        }
                        Err(e) => (false, format!("{label}: {e:#}")),
                    });
                }
            });
        });
    }
}

fn toast_section(ui: &mut Ui) {
    ui.horizontal_wrapped(|ui| {
        for (kind, label) in [
            (ToastKind::Info, "Info"),
            (ToastKind::Warning, "Warning"),
            (ToastKind::Error, "Error"),
            (ToastKind::Success, "Success"),
        ] {
            if ui.button(label).clicked()
                && let Ok(mut q) = TOAST_REQUESTS.lock()
            {
                q.push((kind, format!("{label} toast from the Theme panel")));
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use eframe::egui::{CentralPanel, Pos2, RawInput, Rect};

    use super::super::{fonts::TextScale, presets::bootstrap_startup_theme};

    fn frame(ctx: &Context) {
        let input = RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1400.0, 900.0))),
            ..Default::default()
        };
        ctx.run_ui(input, |ui| {
            CentralPanel::default().show(ui, |_| {});
            show(ui.ctx());
        })
        .drop_without_applying_deltas();
    }

    fn themed_context() -> Context {
        let ctx = Context::default();
        ctx.set_fonts(fonts::definitions(TextScale::Normal));
        bootstrap_startup_theme(&ctx);
        ctx
    }

    #[test]
    fn the_panel_toggles_and_draws_every_section() {
        let ctx = themed_context();
        assert!(!is_open(&ctx));
        frame(&ctx);
        toggle(&ctx);
        assert!(is_open(&ctx));
        for _ in 0..3 {
            frame(&ctx);
        }
        toggle(&ctx);
        frame(&ctx);
        assert!(!is_open(&ctx));
    }

    #[test]
    fn the_panel_draws_under_every_preset_and_density() {
        let ctx = themed_context();
        set_open(&ctx, true);
        for preset in presets::MATCHABLE_PRESETS {
            apply_preset(&ctx, preset);
            for density in Density::ALL {
                presets::set_density(&ctx, density);
                frame(&ctx);
                frame(&ctx);
            }
        }
    }

    #[test]
    fn names_are_made_file_safe() {
        assert_eq!(sanitized_name("  My Theme! "), "My_Theme");
        assert_eq!(sanitized_name("a/b\\c"), "a_b_c");
        assert_eq!(sanitized_name("???"), "");
    }

    #[test]
    fn a_user_theme_is_written_and_loads_back_as_the_same_theme() {
        let ctx = themed_context();
        apply_preset(&ctx, PresetStyles::MtechNoirGlass);
        let dir =
            std::env::temp_dir().join(format!("smart_media_user_themes_{}", std::process::id()));
        let path = write_user_theme(&dir, &ctx, " My Noir! ").unwrap();
        assert_eq!(path.file_name().unwrap(), "My_Noir.json");
        assert!(write_user_theme(&dir, &ctx, "???").is_err());

        let fresh = themed_context();
        apply_saved_theme(&fresh, persist::load_file(&path).unwrap());
        assert_eq!(installed_preset(&fresh), PresetStyles::MtechNoirGlass);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn frame_time_smooths_toward_new_samples() {
        let ctx = Context::default();
        record_ui_time(&ctx, Duration::from_millis(10));
        let first = ctx
            .data(|d| d.get_temp::<FrameStats>(stats_id()))
            .unwrap()
            .ui_ms;
        assert!((first - 10.0).abs() < 0.01);
        record_ui_time(&ctx, Duration::from_millis(20));
        let second = ctx
            .data(|d| d.get_temp::<FrameStats>(stats_id()))
            .unwrap()
            .ui_ms;
        assert!(second > first && second < 20.0);
    }

    #[test]
    fn frame_cpu_time_is_recorded_and_a_missing_sample_is_ignored() {
        let ctx = Context::default();
        record_frame_cpu(&ctx, None);
        assert!(ctx.data(|d| d.get_temp::<FrameStats>(stats_id())).is_none());
        record_frame_cpu(&ctx, Some(0.004));
        let stats = ctx.data(|d| d.get_temp::<FrameStats>(stats_id())).unwrap();
        assert!((stats.frame_ms - 4.0).abs() < 0.01);
    }

    #[test]
    fn toast_requests_are_drained_once() {
        TOAST_REQUESTS
            .lock()
            .unwrap()
            .push((ToastKind::Info, "x".to_owned()));
        assert!(!take_toast_requests().is_empty());
        assert!(take_toast_requests().is_empty());
    }
}
