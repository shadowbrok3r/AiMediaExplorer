use eframe::egui::*;
use once_cell::sync::Lazy;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, AtomicUsize};

/// High-level lifecycle state for a backend component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusState {
    Idle,
    Initializing,
    Running,
    Waiting,
    Error,
}

impl StatusState {
    pub fn color(self, style: &Style) -> Color32 {
        match self {
            StatusState::Idle => style.visuals.warn_fg_color,
            StatusState::Initializing => style.visuals.warn_fg_color,
            StatusState::Running => style.visuals.warn_fg_color,
            StatusState::Waiting => Color32::from_rgb(120, 160, 230),
            StatusState::Error => style.visuals.error_fg_color,
        }
    }
}

/// Shared metadata describing a component status.
#[derive(Debug, Clone)]
pub struct StatusMeta {
    pub name: &'static str,
    pub model: Option<&'static str>,
    pub detail: String,
    pub state: StatusState,
    // Device where the model is running.
    pub device: Option<DeviceKind>,
    pub started_at_ms: u64,
    pub progress_current: u64,
    pub progress_total: u64,
    pub extra: Vec<(String, String)>,
    pub error: Option<String>,
}

impl Default for StatusMeta {
    fn default() -> Self {
        Self {
            name: "",
            model: None,
            detail: String::new(),
            state: StatusState::Idle,
            device: None,
            started_at_ms: 0,
            progress_current: 0,
            progress_total: 0,
            extra: vec![],
            error: None,
        }
    }
}

/// Trait for anything that can render itself as a compact indicator + optional hover card.
pub trait GlobalStatusIndicator {
    /// Name / short key.
    fn key(&self) -> &'static str;
    /// Retrieve current meta snapshot.
    fn snapshot(&self) -> StatusMeta;
    /// Update convenience helpers.
    fn set_state(&self, state: StatusState, detail: impl Into<String>);
    fn set_progress(&self, current: u64, total: u64);
    /// Set or update the model string shown in the hover card.
    /// Note: stored as a leaked &'static str to match StatusMeta's lifetime.
    fn set_model(&self, model: &str);
    /// Set device where this model is running.
    fn set_device(&self, device: DeviceKind);
    /// Set an error message and mark state as Error (displayed in hover card).
    fn set_error(&self, err: impl Into<String>);
    /// Clear any existing error message (does not change state).
    fn clear_error(&self);
    /// Update just the detail string without changing the state.
    fn set_detail(&self, detail: impl Into<String>);
}

// Internal representation stored globally.
#[derive(Debug)]
struct GlobalStatusInner {
    meta: StatusMeta,
}

static STATUSES: Lazy<RwLock<std::collections::HashMap<&'static str, GlobalStatusInner>>> =
    Lazy::new(|| RwLock::new(Default::default()));

fn now_ms() -> u64 {
    (std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()) as u64
}

/// Handle for a registered global status.
#[derive(Clone)]
pub struct RegisteredStatus {
    key: &'static str,
}

impl RegisteredStatus {
    pub fn register(name: &'static str, model: Option<&'static str>) -> Self {
        let mut w = STATUSES.write().unwrap();
        w.entry(name).or_insert_with(|| GlobalStatusInner {
            meta: StatusMeta {
                name,
                model,
                device: None,
                ..Default::default()
            },
        });
        Self { key: name }
    }
}

impl GlobalStatusIndicator for RegisteredStatus {
    fn key(&self) -> &'static str {
        self.key
    }
    fn snapshot(&self) -> StatusMeta {
        STATUSES
            .read()
            .unwrap()
            .get(self.key)
            .map(|i| i.meta.clone())
            .unwrap_or_default()
    }
    fn set_state(&self, state: StatusState, detail: impl Into<String>) {
        if let Some(inner) = STATUSES.write().unwrap().get_mut(self.key) {
            inner.meta.state = state;
            inner.meta.detail = detail.into();
            if matches!(state, StatusState::Running | StatusState::Initializing) {
                inner.meta.started_at_ms = now_ms();
            }
        }
    }
    fn set_progress(&self, current: u64, total: u64) {
        if let Some(inner) = STATUSES.write().unwrap().get_mut(self.key) {
            inner.meta.progress_current = current;
            inner.meta.progress_total = total;
        }
    }
    fn set_model(&self, model: &str) {
        if let Some(inner) = STATUSES.write().unwrap().get_mut(self.key) {
            // Leak to obtain 'static lifetime acceptable for process lifetime.
            let leaked: &'static str = Box::leak(model.to_string().into_boxed_str());
            inner.meta.model = Some(leaked);
        }
    }
    fn set_device(&self, device: DeviceKind) {
        if let Some(inner) = STATUSES.write().unwrap().get_mut(self.key) {
            inner.meta.device = Some(device);
        }
    }
    fn set_error(&self, err: impl Into<String>) {
        if let Some(inner) = STATUSES.write().unwrap().get_mut(self.key) {
            inner.meta.error = Some(err.into());
            inner.meta.state = StatusState::Error;
        }
    }
    fn clear_error(&self) {
        if let Some(inner) = STATUSES.write().unwrap().get_mut(self.key) {
            inner.meta.error = None;
        }
    }
    fn set_detail(&self, detail: impl Into<String>) {
        if let Some(inner) = STATUSES.write().unwrap().get_mut(self.key) {
            inner.meta.detail = detail.into();
        }
    }
}

/// Access snapshot for arbitrary key (used by UI aggregation).
pub fn snapshot(key: &'static str) -> Option<StatusMeta> {
    STATUSES.read().ok()?.get(key).map(|i| i.meta.clone())
}
pub fn all_snapshots() -> Vec<StatusMeta> {
    STATUSES
        .read()
        .map(|m| m.values().map(|i| i.meta.clone()).collect())
        .unwrap_or_default()
}

/// Convenience registration for core components.

/// Coarse device classification for display.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind { CPU, GPU }

pub static DB_STATUS: Lazy<RegisteredStatus> = Lazy::new(|| RegisteredStatus::register("DB", None));
pub static VISION_STATUS: Lazy<RegisteredStatus> =
    Lazy::new(|| RegisteredStatus::register("VISION", None));
pub static CLIP_STATUS: Lazy<RegisteredStatus> =
    Lazy::new(|| RegisteredStatus::register("CLIP", None));
pub static RERANK_STATUS: Lazy<RegisteredStatus> =
    Lazy::new(|| RegisteredStatus::register("RERANK", None));
pub static QWEN_EDIT_STATUS: Lazy<RegisteredStatus> =
    Lazy::new(|| RegisteredStatus::register("QWEN-EDIT", None));
pub static ASSIST_STATUS: Lazy<RegisteredStatus> =
    Lazy::new(|| RegisteredStatus::register("ASSIST", None));

/// Token-progress tracker for generation (vision description) from anywhere.
pub static VISION_TOKENS: Lazy<AtomicUsize> = Lazy::new(|| AtomicUsize::new(0));
pub static VISION_MAX_TOKENS: Lazy<AtomicUsize> =
    Lazy::new(|| AtomicUsize::new(crate::app::MAX_NEW_TOKENS));
pub static LAST_ERROR_TS_MS: Lazy<AtomicU64> = Lazy::new(|| AtomicU64::new(0));

/// Render a compact horizontal status bar section suitable for embedding in a toolbar.
pub fn status_bar_inline(ui: &mut Ui) {
    for meta in all_snapshots() {
        indicator_small(ui, &meta);
    }
}

fn indicator_small(ui: &mut Ui, meta: &StatusMeta) {
    ui.add_space(5.);
    let (pct, has_prog) = if meta.progress_total > 0 {
        (
            (meta.progress_current as f32 / meta.progress_total as f32).clamp(0.0, 1.0),
            true,
        )
    } else {
        (0.0, false)
    };

    let label = if has_prog {
        format!("{} {:.0}%", meta.name, pct * 100.0)
    } else {
        meta.name.to_string()
    };
    // Reserve space for a spinner glyph inside the button when active.
    let show_spinner = matches!(meta.state, StatusState::Initializing | StatusState::Running);
    let mut display_label = label.clone();
    if show_spinner {
        // prepend a bit of spacing so our overlaid gear doesn't cover text
        display_label = format!("   {}", display_label);
    }

    let custom_button_id = Id::new(format!("custom_button {display_label:?}"));
    let atom_layout = Button::new((
        Atom::custom(custom_button_id, Vec2::splat(14.0)),
        RichText::new(display_label).color(meta.state.color(ui.style())),
    ))
    .atom_ui(ui);

    if let Some(rect) = atom_layout.rect(custom_button_id) {
        if show_spinner {
            ui.put(
                rect,
                GearSpinner::new()
                    .color(ui.style().visuals.error_fg_color)
                    .size(18.),
            );
        } else {
            ui.put(rect, Label::new(RichText::new("⚙").weak()).selectable(false));
        }
    }

    atom_layout.response.on_hover_ui(|ui| {
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
        let _ = ui.button("X");
    });
    ui.add_space(5.);
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
                    ui.style().visuals.error_fg_color
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
        response.widget_info(|| WidgetInfo::new(WidgetType::ProgressIndicator));
        self.paint_at(ui, rect);

        response
    }
}
