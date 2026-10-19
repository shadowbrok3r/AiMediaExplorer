use crate::icons::svg;
use eframe::egui::*;

pub use crate::status::*;

/// Egui color of a [`StatusState`].
pub trait StatusStateColor {
    fn color(self, style: &Style) -> Color32;
}

impl StatusStateColor for StatusState {
    fn color(self, style: &Style) -> Color32 {
        match self {
            StatusState::Idle => style.visuals.warn_fg_color,
            StatusState::Initializing => style.visuals.warn_fg_color,
            StatusState::Running => style.visuals.warn_fg_color,
            StatusState::Waiting => style.visuals.hyperlink_color,
            StatusState::Error => style.visuals.error_fg_color,
        }
    }
}

/// How much of each engine indicator fits in the space offered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChipMode {
    /// Gear, name and progress per engine.
    Full,
    /// Gear only per engine; the hover card names it.
    Icon,
    /// One gear whose menu lists every engine.
    Summary,
}

/// Width each [`ChipMode`] takes at the current style.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChipWidths {
    pub full: f32,
    pub icon: f32,
    pub summary: f32,
}

/// Gear size inside an indicator button.
const CHIP_ICON: f32 = 14.0;

/// Widths of the three indicator layouts for `metas`, measured in hidden passes.
pub fn chip_widths(ui: &mut Ui, metas: &[StatusMeta]) -> ChipWidths {
    ChipWidths {
        full: measure(ui, ChipMode::Full, metas),
        icon: measure(ui, ChipMode::Icon, metas),
        summary: measure(ui, ChipMode::Summary, metas),
    }
}

fn measure(ui: &mut Ui, mode: ChipMode, metas: &[StatusMeta]) -> f32 {
    let rect = Rect::from_min_size(pos2(-100_000.0, -100_000.0), vec2(10_000.0, ui.max_rect().height()));
    let builder = UiBuilder::new().id_salt(("ame.chip_measure", mode as u8)).max_rect(rect).layout(Layout::left_to_right(Align::Center));
    let mut offscreen = ui.new_child(builder);
    offscreen.set_clip_rect(Rect::NOTHING);
    status_chips(&mut offscreen, mode, metas);
    offscreen.min_rect().width()
}

/// The richest layout that fits `avail`; `full_reserve` is kept free for other content when choosing [`ChipMode::Full`].
pub fn chip_mode(widths: &ChipWidths, avail: f32, full_reserve: f32) -> ChipMode {
    if widths.full + full_reserve <= avail {
        ChipMode::Full
    } else if widths.icon <= avail {
        ChipMode::Icon
    } else {
        ChipMode::Summary
    }
}

/// Draws the indicators of `metas` in `mode`, right to left when the layout is.
pub fn status_chips(ui: &mut Ui, mode: ChipMode, metas: &[StatusMeta]) {
    match mode {
        ChipMode::Full => metas.iter().for_each(|meta| indicator_small(ui, meta)),
        ChipMode::Icon => metas.iter().for_each(|meta| indicator_icon(ui, meta)),
        ChipMode::Summary => indicator_summary(ui, metas),
    }
}

/// Button text of an indicator: name and progress, padded when the spinner overlays the start.
fn chip_label(meta: &StatusMeta) -> String {
    let (pct, has_prog) = progress_of(meta);
    let label = if has_prog { format!("{} {:.0}%", meta.name, pct * 100.0) } else { meta.name.to_string() };
    if is_active(meta) { format!("   {label}") } else { label }
}

/// Paints the gear of an indicator button into `rect`, spinning while the engine works.
fn paint_gear(ui: &mut Ui, rect: Rect, meta: &StatusMeta) {
    if is_active(meta) {
        let angle = ui.input(|i| i.time) as f32 * std::f32::consts::TAU * 0.6;
        ui.ctx().request_repaint();
        let tint = crate::theme::primary(ui);
        svg::GEAR.image(CHIP_ICON).rotate(angle, Vec2::splat(0.5)).tint(tint).paint_at(ui, rect);
    } else {
        let tint = ui.visuals().weak_text_color();
        svg::GEAR.image(CHIP_ICON).tint(tint).paint_at(ui, rect);
    }
}

fn indicator_small(ui: &mut Ui, meta: &StatusMeta) {
    ui.add_space(5.);
    let custom_button_id = IdSalt::new("custom_button");
    let atom_layout = Button::new((
        Atom::custom(custom_button_id, Vec2::splat(CHIP_ICON)),
        RichText::new(chip_label(meta)).color(meta.state.color(ui.style())),
    ))
    .atom_ui(ui);

    if let Some(rect) = atom_layout.rect(custom_button_id) {
        paint_gear(ui, rect, meta);
    }

    atom_layout.response.on_hover_ui(|ui| hover_card(ui, meta));
    ui.add_space(5.);
}

/// Gear-only indicator; its hover card carries the name.
fn indicator_icon(ui: &mut Ui, meta: &StatusMeta) {
    let custom_button_id = IdSalt::new("custom_button");
    let atom_layout = Button::new(Atom::custom(custom_button_id, Vec2::splat(CHIP_ICON))).atom_ui(ui);
    if let Some(rect) = atom_layout.rect(custom_button_id) {
        match meta.state {
            StatusState::Error | StatusState::Waiting => {
                let tint = meta.state.color(ui.style());
                svg::GEAR.image(CHIP_ICON).tint(tint).paint_at(ui, rect);
            }
            _ => paint_gear(ui, rect, meta),
        }
    }
    atom_layout.response.on_hover_ui(|ui| hover_card(ui, meta));
}

/// One gear tinted by the most urgent engine state, opening a list of every indicator.
fn indicator_summary(ui: &mut Ui, metas: &[StatusMeta]) {
    let urgency = |s: StatusState| match s {
        StatusState::Error => 3,
        StatusState::Running | StatusState::Initializing => 2,
        StatusState::Waiting => 1,
        StatusState::Idle => 0,
    };
    let tint = match metas.iter().map(|m| m.state).max_by_key(|s| urgency(*s)) {
        Some(state) if urgency(state) > 0 => state.color(ui.style()),
        _ => ui.visuals().weak_text_color(),
    };
    let button = Button::new(svg::GEAR.image(CHIP_ICON).tint(tint));
    let (response, _) = containers::menu::MenuButton::from_button(button).ui(ui, |ui| {
        for meta in metas {
            indicator_small(ui, meta);
        }
    });
    response.on_hover_text("Engine status");
}

fn hover_card(ui: &mut Ui, meta: &StatusMeta) {
    let (pct, has_prog) = progress_of(meta);
    ui.style_mut().interaction.selectable_labels = true;
    ui.set_max_width(500.);
    ui.vertical_centered(|ui| {
        ui.heading(meta.name);
        ui.separator();
    });

    if let Some(model) = meta.model {
        ui.horizontal(|ui| {
            ui.colored_label(ui.style().visuals.warn_fg_color, RichText::new("Model").underline());
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.label(RichText::new(model).underline());
            });
        });
    }

    ui.horizontal(|ui| {
        ui.colored_label(ui.style().visuals.warn_fg_color, RichText::new("State").underline());
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.label(format!("{:?}", meta.state));
        });
    });

    if let Some(dev) = meta.device {
        ui.horizontal(|ui| {
            ui.colored_label(ui.style().visuals.warn_fg_color, RichText::new("Device").underline());
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let txt = match dev { DeviceKind::CPU => "CPU", DeviceKind::GPU => "GPU" };
                ui.label(txt);
            });
        });
    }

    if !meta.detail.is_empty() {
        ui.horizontal(|ui| {
            ui.colored_label(ui.style().visuals.warn_fg_color, RichText::new("Detail").underline());
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.label(&meta.detail);
            });
        });
    }

    for (k, v) in &meta.extra {
        ui.horizontal(|ui| {
            ui.colored_label(ui.style().visuals.warn_fg_color, RichText::new(k).underline());
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.label(v);
            });
        });
    }

    if let Some(err) = &meta.error {
        ui.horizontal(|ui| {
            ui.colored_label(ui.style().visuals.error_fg_color, RichText::new("Error").underline());
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.colored_label(ui.style().visuals.error_fg_color, err);
            });
        });
    }

    if has_prog {
        ui.vertical_centered(|ui| {
            ProgressBar::new(pct).show_percentage().ui(ui);
        });
    }
}

#[derive(Default)]
pub struct GearSpinner {
    pub size: Option<f32>,
    pub color: Option<Color32>,
}

impl GearSpinner {
    /// Create a new spinner that uses the style's `interact_size` unless changed.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the spinner's size. The size sets both the height and width, as the spinner is always
    /// square. If the size isn't set explicitly, the active style's `interact_size` is used.
    #[inline]
    pub fn size(mut self, size: f32) -> Self {
        self.size = Some(size);
        self
    }

    /// Sets the spinner's color.
    #[inline]
    pub fn color(mut self, color: impl Into<Color32>) -> Self {
        self.color = Some(color.into());
        self
    }

    /// Paint the spinner in the given rectangle.
    pub fn paint_at(&self, ui: &Ui, button_rect: Rect) {
        if !ui.is_rect_visible(button_rect) {
            return;
        }
        ui.ctx().request_repaint(); // ensure animation
        let h = button_rect.height();
        // Square area for gear near left edge (after button's default padding ~4px)
        let side = h.min(18.0); // cap size a bit so it doesn't dominate
        let gear_rect = Rect::from_min_size(
            pos2(
                button_rect.left() + 4.0,
                button_rect.center().y - side * 0.5,
            ),
            vec2(side, side),
        );
        let angle = ui.input(|i| i.time as f32) * std::f32::consts::TAU * 0.6; // rotation speed factor
        // Build rotated text shape using egui font system.
        let shape = ui.ctx().fonts_mut(|f| {
            let mut s = Shape::text(
                f,
                gear_rect.center(),
                Align2::CENTER_CENTER,
                "⚙",
                FontId::new(side * 0.9, FontFamily::Proportional),
                if let Some(color) = self.color {
                    color
                } else {
                    crate::theme::primary(ui)
                },
            );
            if let Shape::Text(ts) = &mut s {
                let rotated = ts
                    .clone()
                    .with_angle_and_anchor(angle, Align2::CENTER_CENTER);
                *ts = rotated;
            }
            s
        });
        ui.painter().add(shape);
    }
}

impl Widget for GearSpinner {
    fn ui(self, ui: &mut Ui) -> Response {
        let size = self
            .size
            .unwrap_or_else(|| ui.style().spacing.interact_size.y);
        let (rect, response) = ui.allocate_exact_size(vec2(size, size), Sense::hover());
        response.widget_info(|| WidgetInfo::new(Role::ProgressIndicator));
        self.paint_at(ui, rect);

        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn widths() -> ChipWidths {
        ChipWidths { full: 480.0, icon: 150.0, summary: 26.0 }
    }

    #[test]
    fn the_richest_mode_that_fits_wins() {
        assert_eq!(chip_mode(&widths(), 1000.0, 0.0), ChipMode::Full);
        assert_eq!(chip_mode(&widths(), 480.0, 0.0), ChipMode::Full);
        assert_eq!(chip_mode(&widths(), 479.0, 0.0), ChipMode::Icon);
        assert_eq!(chip_mode(&widths(), 150.0, 0.0), ChipMode::Icon);
        assert_eq!(chip_mode(&widths(), 149.0, 0.0), ChipMode::Summary);
        assert_eq!(chip_mode(&widths(), 0.0, 0.0), ChipMode::Summary);
    }

    #[test]
    fn the_reserve_only_holds_back_the_labelled_mode() {
        assert_eq!(chip_mode(&widths(), 500.0, 100.0), ChipMode::Icon);
        assert_eq!(chip_mode(&widths(), 580.0, 100.0), ChipMode::Full);
        assert_eq!(chip_mode(&widths(), 150.0, 100.0), ChipMode::Icon);
    }

    fn engines() -> Vec<StatusMeta> {
        let meta = |name, state, detail: &str| StatusMeta { name, state, detail: detail.to_string(), ..Default::default() };
        let mut busy = meta("VISION", StatusState::Running, "busy");
        busy.progress_current = 45;
        busy.progress_total = 100;
        vec![
            meta("DB", StatusState::Idle, ""),
            busy,
            meta("CLIP", StatusState::Error, "failed"),
            meta("RERANK", StatusState::Waiting, ""),
            meta("QWEN-EDIT", StatusState::Idle, ""),
            meta("ASSIST", StatusState::Idle, ""),
        ]
    }

    #[test]
    fn measured_widths_match_the_drawn_indicators() {
        let ctx = Context::default();
        ctx.set_fonts(crate::theme::fonts::definitions(crate::theme::fonts::TextScale::Normal));
        crate::theme::bootstrap_startup_theme(&ctx);
        let metas = engines();
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1600.0, 600.0))), ..Default::default() };
        let mut measured = Vec::new();
        ctx.run_ui(input, |ui| {
            let widths = chip_widths(ui, &metas);
            for (mode, predicted) in [(ChipMode::Full, widths.full), (ChipMode::Icon, widths.icon), (ChipMode::Summary, widths.summary)] {
                let scope = UiBuilder::new().layout(Layout::right_to_left(Align::Center));
                let used = ui.scope_builder(scope, |ui| status_chips(ui, mode, &metas)).response.rect.width();
                measured.push((mode, predicted, used));
            }
        })
        .drop_without_applying_deltas();
        for (mode, predicted, used) in measured {
            assert!((predicted - used).abs() <= 0.5, "{mode:?} measures {predicted} but draws {used}");
        }
    }

    #[test]
    fn a_busy_engine_widens_the_labelled_mode_only_by_its_progress_text() {
        let ctx = Context::default();
        ctx.set_fonts(crate::theme::fonts::definitions(crate::theme::fonts::TextScale::Normal));
        crate::theme::bootstrap_startup_theme(&ctx);
        let idle: Vec<StatusMeta> = engines().into_iter().map(|m| StatusMeta { state: StatusState::Idle, progress_total: 0, ..m }).collect();
        let (mut busy, mut quiet) = (None, None);
        ctx.run_ui(RawInput::default(), |ui| {
            busy = Some(chip_widths(ui, &engines()));
            quiet = Some(chip_widths(ui, &idle));
        })
        .drop_without_applying_deltas();
        let (busy_w, idle_w) = (busy.unwrap(), quiet.unwrap());
        assert!(busy_w.full > idle_w.full, "{busy_w:?} vs {idle_w:?}");
        assert_eq!(busy_w.icon, idle_w.icon);
        assert_eq!(busy_w.summary, idle_w.summary);
    }
}
