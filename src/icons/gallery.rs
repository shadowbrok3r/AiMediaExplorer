//! Debug gallery: searchable icon grid with size, tint, duotone, backdrop, widget samples and stress mode.

use egui::{
    Align2, Color32, Context, CornerRadius, FontId, Id, Rect, RichText, ScrollArea, Sense, Slider,
    Stroke, StrokeKind, Ui, Window, pos2, vec2,
};

use super::pack::{self as svg, ALL, Icon};
use super::widgets;

const STATE_ID: &str = "ame_icon_gallery_state";
const OPEN_ID: &str = "ame_icon_gallery_open";
const STRESS_SIZES: [f32; 7] = [12.0, 16.0, 20.0, 24.0, 32.0, 48.0, 64.0];
const DARK_BG: Color32 = Color32::from_rgb(27, 27, 31);
const LIGHT_BG: Color32 = Color32::from_rgb(242, 242, 244);

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    All,
    Custom,
    Phosphor,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Backdrop {
    Theme,
    Dark,
    Light,
    Checker,
}

#[derive(Clone)]
struct State {
    query: String,
    kind: Kind,
    size: f32,
    tint: Option<Color32>,
    duo: bool,
    names: bool,
    backdrop: Backdrop,
    stress: bool,
    stress_animate: bool,
    frame_ms: f32,
    selected: Option<&'static str>,
    copied: Option<(String, f64)>,
    toggle_sample: bool,
}

impl Default for State {
    fn default() -> Self {
        Self {
            query: String::new(),
            kind: Kind::All,
            size: 32.0,
            tint: None,
            duo: false,
            names: true,
            backdrop: Backdrop::Theme,
            stress: false,
            stress_animate: false,
            frame_ms: 0.0,
            selected: None,
            copied: None,
            toggle_sample: false,
        }
    }
}

/// True while the gallery window is open.
pub fn is_open(ctx: &Context) -> bool {
    ctx.data(|d| d.get_temp::<bool>(Id::unique(OPEN_ID))).unwrap_or(false)
}

/// Opens or closes the gallery window.
pub fn set_open(ctx: &Context, open: bool) {
    ctx.data_mut(|d| d.insert_temp(Id::unique(OPEN_ID), open));
}

/// Flips the gallery window.
pub fn toggle(ctx: &Context) {
    set_open(ctx, !is_open(ctx));
}

/// Draws the gallery window when open; call once per frame.
pub fn show_window(ctx: &Context) {
    let mut open = is_open(ctx);
    if !open {
        return;
    }
    Window::new("Icon gallery")
        .open(&mut open)
        .default_size(vec2(780.0, 620.0))
        .resizable(true)
        .show(ctx, show);
    set_open(ctx, open);
}

/// Gallery contents, usable inside any `Ui`.
pub fn show(ui: &mut Ui) {
    let id = Id::unique(STATE_ID);
    let mut state: State = ui.data(|d| d.get_temp(id)).unwrap_or_default();
    controls(ui, &mut state);
    ui.separator();
    if state.stress {
        stress(ui, &mut state);
    } else {
        grid(ui, &mut state);
    }
    ui.data_mut(|d| d.insert_temp(id, state));
}

fn visible(state: &State) -> Vec<Icon> {
    let query = state.query.trim().to_lowercase();
    ALL.iter()
        .copied()
        .filter(|icon| match state.kind {
            Kind::All => true,
            Kind::Custom => icon.is_custom(),
            Kind::Phosphor => !icon.is_custom(),
        })
        .filter(|icon| {
            query.is_empty()
                || icon.name().contains(&query)
                || icon.const_name().to_lowercase().contains(&query)
        })
        .collect()
}

fn controls(ui: &mut Ui, state: &mut State) {
    ui.horizontal_wrapped(|ui| {
        ui.add(
            egui::TextEdit::singleline(&mut state.query)
                .hint_text("Search icons")
                .desired_width(160.0),
        );
        ui.selectable_value(&mut state.kind, Kind::All, "All");
        ui.selectable_value(&mut state.kind, Kind::Custom, "Custom only");
        ui.selectable_value(&mut state.kind, Kind::Phosphor, "Phosphor only");
        ui.separator();
        ui.label("Size");
        ui.add(Slider::new(&mut state.size, 12.0..=96.0).suffix(" px").step_by(1.0));
    });
    ui.horizontal_wrapped(|ui| {
        let mut custom = state.tint.is_some();
        if ui.checkbox(&mut custom, "Custom tint").changed() {
            state.tint = custom.then(|| ui.visuals().text_color());
        }
        if let Some(color) = state.tint.as_mut() {
            ui.color_edit_button_srgba(color);
        }
        ui.checkbox(&mut state.duo, "Duotone");
        ui.checkbox(&mut state.names, "Names");
        ui.separator();
        ui.label("Backdrop");
        ui.selectable_value(&mut state.backdrop, Backdrop::Theme, "Theme");
        ui.selectable_value(&mut state.backdrop, Backdrop::Dark, "Dark");
        ui.selectable_value(&mut state.backdrop, Backdrop::Light, "Light");
        ui.selectable_value(&mut state.backdrop, Backdrop::Checker, "Checker");
        ui.separator();
        ui.checkbox(&mut state.stress, "Stress");
        if state.stress {
            ui.checkbox(&mut state.stress_animate, "Animate");
        }
    });
    ui.horizontal_wrapped(|ui| {
        let total = ALL.len();
        let custom = ALL.iter().filter(|i| i.is_custom()).count();
        ui.weak(format!(
            "{total} icons ({custom} custom, {} Phosphor), {} with duotone",
            total - custom,
            ALL.iter().filter(|i| i.has_duo()).count()
        ));
        let now = ui.input(|i| i.time);
        if let Some((text, at)) = state.copied.clone() {
            if now - at < 2.5 {
                ui.label(RichText::new(format!("Copied {text}")).color(ui.visuals().selection.stroke.color));
                ui.ctx().request_repaint_after(std::time::Duration::from_millis(250));
            } else {
                state.copied = None;
            }
        }
    });
}

fn tint_for(state: &State, ui: &Ui) -> Color32 {
    state.tint.unwrap_or_else(|| match state.backdrop {
        Backdrop::Theme => ui.visuals().text_color(),
        Backdrop::Dark | Backdrop::Checker => Color32::from_gray(235),
        Backdrop::Light => Color32::from_gray(28),
    })
}

fn paint_backdrop(ui: &Ui, rect: Rect, backdrop: Backdrop) {
    let painter = ui.painter();
    let radius = CornerRadius::same(4);
    match backdrop {
        Backdrop::Theme => {
            painter.rect_filled(rect, radius, ui.visuals().extreme_bg_color);
        }
        Backdrop::Dark => {
            painter.rect_filled(rect, radius, DARK_BG);
        }
        Backdrop::Light => {
            painter.rect_filled(rect, radius, LIGHT_BG);
        }
        Backdrop::Checker => {
            painter.rect_filled(rect, radius, Color32::from_gray(54));
            let step = 8.0;
            let cols = (rect.width() / step).ceil() as i32;
            let rows = (rect.height() / step).ceil() as i32;
            for y in 0..rows {
                for x in 0..cols {
                    if (x + y) % 2 == 0 {
                        let cell = Rect::from_min_size(
                            rect.min + vec2(x as f32 * step, y as f32 * step),
                            vec2(step, step),
                        )
                        .intersect(rect);
                        painter.rect_filled(cell, 0.0, Color32::from_gray(74));
                    }
                }
            }
        }
    }
}

fn paint_icon(ui: &Ui, icon: Icon, rect: Rect, size: f32, tint: Color32, duo: bool) {
    if duo && icon.has_duo() {
        icon.paint_duo(ui, rect, size, tint, tint.gamma_multiply(0.4));
    } else {
        icon.paint(ui, rect, size, tint);
    }
}

fn elide(ui: &Ui, text: &str, font: &FontId, max_width: f32) -> String {
    let width = |s: &str| {
        ui.painter()
            .layout_no_wrap(s.to_owned(), font.clone(), Color32::WHITE)
            .size()
            .x
    };
    if width(text) <= max_width {
        return text.to_owned();
    }
    let mut chars: Vec<char> = text.chars().collect();
    while !chars.is_empty() {
        chars.pop();
        let candidate: String = chars.iter().collect::<String>() + "\u{2026}";
        if width(&candidate) <= max_width {
            return candidate;
        }
    }
    String::new()
}

fn glyph_label(icon: Icon) -> String {
    let code = icon.glyph().chars().next().map_or(0, |c| c as u32);
    let source = svg::Icon::from_glyph(icon.glyph())
        .map(|i| i.const_name())
        .unwrap_or_else(|| "?".to_owned());
    format!("U+{code:04X} ({source})")
}

fn hover(ui: &mut Ui, icon: Icon) {
    ui.label(RichText::new(icon.const_name()).strong());
    ui.label(format!("name: {}", icon.name()));
    ui.label(format!("glyph: {}", glyph_label(icon)));
    ui.label(format!("has_duo: {}", icon.has_duo()));
    ui.label(format!("kind: {}", if icon.is_custom() { "custom" } else { "Phosphor" }));
    ui.label(format!("svg: {} bytes", icon.svg().len()));
}

fn copy(ui: &Ui, state: &mut State, text: String) {
    ui.ctx().copy_text(text.clone());
    state.copied = Some((text, ui.input(|i| i.time)));
}

fn grid(ui: &mut Ui, state: &mut State) {
    let icons = visible(state);
    let tint = tint_for(state, ui);
    let spacing = 6.0;
    let pad = 8.0;
    let name_h = if state.names { 14.0 } else { 0.0 };
    let cell = vec2((state.size + pad * 2.0).max(78.0), state.size + pad * 2.0 + name_h);
    let detail_h = if state.selected.is_some() { 156.0 } else { 0.0 };

    egui::CollapsingHeader::new("Widget samples")
        .id_salt("ame_icon_gallery_widgets")
        .show(ui, |ui| widget_samples(ui, state));

    ui.spacing_mut().item_spacing = vec2(spacing, spacing);
    let cols = (((ui.available_width() + spacing) / (cell.x + spacing)).floor() as usize).max(1);
    let rows = icons.len().div_ceil(cols);
    let height = (ui.available_height() - detail_h - 8.0).max(120.0);

    if icons.is_empty() {
        ui.weak("No icons match.");
    }
    ScrollArea::vertical()
        .id_salt("ame_icon_gallery_grid")
        .max_height(height)
        .auto_shrink([false, true])
        .show_rows(ui, cell.y, rows, |ui, range| {
            for row in range {
                ui.horizontal(|ui| {
                    for icon in icons.iter().skip(row * cols).take(cols) {
                        draw_cell(ui, state, *icon, cell, pad, tint);
                    }
                });
            }
        });

    if let Some(name) = state.selected {
        if let Some(icon) = Icon::find(name) {
            ui.separator();
            detail(ui, state, icon);
        }
    }
}

fn draw_cell(ui: &mut Ui, state: &mut State, icon: Icon, cell: egui::Vec2, pad: f32, tint: Color32) {
    let (rect, response) = ui.allocate_exact_size(cell, Sense::click());
    if ui.is_rect_visible(rect) {
        paint_backdrop(ui, rect, state.backdrop);
        let icon_rect = Rect::from_center_size(
            pos2(rect.center().x, rect.top() + pad + state.size * 0.5),
            vec2(state.size, state.size),
        );
        paint_icon(ui, icon, icon_rect, state.size, tint, state.duo);
        if state.names {
            let font = FontId::proportional(9.0);
            let text = elide(ui, icon.name(), &font, cell.x - 4.0);
            let color = ui.visuals().weak_text_color();
            ui.painter().text(
                pos2(rect.center().x, rect.bottom() - 2.0),
                Align2::CENTER_BOTTOM,
                text,
                font,
                color,
            );
        }
        let selected = state.selected == Some(icon.name());
        if selected || response.hovered() {
            let color = if selected {
                ui.visuals().selection.stroke.color
            } else {
                ui.visuals().widgets.hovered.bg_stroke.color
            };
            ui.painter().rect_stroke(
                rect,
                CornerRadius::same(4),
                Stroke::new(1.0, color),
                StrokeKind::Inside,
            );
        }
    }
    if response.clicked() {
        state.selected = Some(icon.name());
        copy(ui, state, format!("icons::{}", icon.const_name()));
    }
    response.on_hover_ui(|ui| hover(ui, icon));
}

fn detail(ui: &mut Ui, state: &mut State, icon: Icon) {
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(icon.const_name()).strong());
        ui.weak(icon.name());
        if ui.button("Copy const").clicked() {
            copy(ui, state, format!("icons::{}", icon.const_name()));
        }
        if ui.button("Copy name").clicked() {
            copy(ui, state, icon.name().to_owned());
        }
        if ui.button("Copy glyph path").clicked() {
            copy(ui, state, format!("egui_phosphor::regular::{}", glyph_const(icon)));
        }
        if ui.button("Close").clicked() {
            state.selected = None;
        }
    });
    ui.horizontal(|ui| {
        for (bg, ink) in [(DARK_BG, Color32::from_gray(235)), (LIGHT_BG, Color32::from_gray(28))] {
            let (rect, _) = ui.allocate_exact_size(vec2(300.0, 104.0), Sense::hover());
            ui.painter().rect_filled(rect, CornerRadius::same(4), bg);
            let mut x = rect.left() + 12.0;
            for size in [96.0, 48.0, 32.0, 24.0, 16.0] {
                let center = pos2(x + size * 0.5, rect.center().y);
                paint_icon(ui, icon, Rect::from_center_size(center, vec2(size, size)), size, ink, state.duo);
                x += size + 12.0;
            }
        }
    });
}

fn glyph_const(icon: Icon) -> String {
    Icon::from_glyph(icon.glyph()).map_or_else(|| "?".to_owned(), |i| i.const_name())
}

fn widget_samples(ui: &mut Ui, state: &mut State) {
    ui.horizontal_wrapped(|ui| {
        widgets::button(ui, svg::SPARKLE, "Button");
        widgets::small_button(ui, svg::GEAR, "Small");
        if widgets::selectable_button(ui, svg::STAR, state.toggle_sample, "Selectable").clicked() {
            state.toggle_sample = !state.toggle_sample;
        }
        widgets::icon_button(ui, svg::AME_SCAN, "Icon button with tooltip");
        if widgets::toggle_button(ui, svg::AME_LOGICAL_GROUP, state.toggle_sample, "Toggle button").clicked() {
            state.toggle_sample = !state.toggle_sample;
        }
        widgets::label(ui, svg::AME_DATABASE, "Icon label");
        ui.add_enabled_ui(false, |ui| {
            widgets::button(ui, svg::TRASH, "Disabled");
        });
        widgets::menu_button(ui, svg::AME_ASSISTANT, "Menu button", |ui| {
            widgets::button(ui, svg::AME_AI_DESCRIBE, "Describe");
            widgets::button(ui, svg::AME_CLIP_EMBED, "Embed");
            widgets::menu_button(ui, svg::AME_MCP, "Sub-menu", |ui| {
                widgets::button(ui, svg::PLUGS_CONNECTED, "Connect");
            });
        });
    });
}

fn stress(ui: &mut Ui, state: &mut State) {
    let dt = ui.input(|i| i.unstable_dt) * 1000.0;
    state.frame_ms = if state.frame_ms == 0.0 { dt } else { state.frame_ms * 0.9 + dt * 0.1 };
    ui.horizontal_wrapped(|ui| {
        ui.label(format!(
            "{} icons x {} sizes = {} images per frame, frame {:.1} ms",
            ALL.len(),
            STRESS_SIZES.len(),
            ALL.len() * STRESS_SIZES.len(),
            state.frame_ms
        ));
    });
    if state.stress_animate {
        ui.ctx().request_repaint();
    }
    let tint = tint_for(state, ui);
    ScrollArea::both().id_salt("ame_icon_gallery_stress").auto_shrink([false, false]).show(ui, |ui| {
        for size in STRESS_SIZES {
            ui.weak(format!("{size} px"));
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing = vec2(2.0, 2.0);
                for icon in ALL {
                    let (rect, _) = ui.allocate_exact_size(vec2(size, size), Sense::hover());
                    if ui.is_rect_visible(rect) {
                        paint_icon(ui, *icon, rect, size, tint, state.duo);
                    }
                }
            });
        }
    });
}
