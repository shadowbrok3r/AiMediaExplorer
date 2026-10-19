//! Devtools window for the sidebar: section state, list caches, the saved layout and a disconnected preview.

use std::time::{Duration, Instant};

use eframe::egui::{Context, Grid, Id, RichText, ScrollArea, TextEdit, Ui, Window};

use crate::ui::shell::Shell;
use crate::ui::sidebar::{Mode, SectionId, Source};

/// A button pressed in the window, applied by the shell next frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Request {
    SetSection(SectionId, bool),
    SetSidebarOpen(bool),
    Refresh(Source),
    RefreshAll,
    ForceDisconnected(bool),
    ResetLayout,
    SaveNow,
}

/// One cached list.
#[derive(Clone, Debug)]
pub struct SourceRow {
    pub source: Source,
    pub items: usize,
    pub age: Option<Duration>,
    pub loading: bool,
    pub error: Option<String>,
    pub loads: u32,
    pub ttl: Duration,
}

/// What the window shows.
#[derive(Clone, Debug)]
pub struct Snapshot {
    pub mode: Mode,
    pub width: f32,
    pub sidebar_open: bool,
    pub sections: Vec<(SectionId, bool)>,
    pub sources: Vec<SourceRow>,
    pub force_disconnected: bool,
    pub layout_json: String,
    pub layout_path: Option<String>,
    pub layout_note: String,
    pub layout_dirty: bool,
    pub layout_saves: u32,
    pub layout_error: Option<String>,
    pub bottom_bar_open: bool,
    pub right_rail_open: bool,
}

fn open_id() -> Id {
    Id::unique("ame_devtools_sidebar_open")
}

fn snapshot_id() -> Id {
    Id::unique("ame_devtools_sidebar_snapshot")
}

fn requests_id() -> Id {
    Id::unique("ame_devtools_sidebar_requests")
}

pub fn is_open(ctx: &Context) -> bool {
    ctx.data(|d| d.get_temp::<bool>(open_id())).unwrap_or(false)
}

pub fn set_open(ctx: &Context, open: bool) {
    ctx.data_mut(|d| d.insert_temp(open_id(), open));
}

/// Records what the window shows; does nothing while it is closed.
pub fn publish(ctx: &Context, shell: &Shell) {
    if !is_open(ctx) {
        return;
    }
    let now = Instant::now();
    let layout = &shell.layout.layout;
    let data = &shell.sidebar.data;
    let items = |source: Source| match source {
        Source::Places => data.places().quick.len() + data.places().drives.len(),
        Source::Library => usize::from(data.total().is_some()),
        Source::Collections => data.collections().len(),
        Source::Tags => data.tags().len(),
        Source::Categories => data.categories().len(),
        Source::Scans => data.scans().len(),
    };
    let snapshot = Snapshot {
        mode: crate::ui::sidebar::mode_for(ctx.content_rect().width(), layout.sidebar_open),
        width: layout.sidebar_width,
        sidebar_open: layout.sidebar_open,
        sections: SectionId::ALL.iter().map(|s| (*s, layout.is_section_open(s.key(), s.default_open()))).collect(),
        sources: Source::ALL
            .iter()
            .map(|s| {
                let meta = data.meta(*s);
                SourceRow { source: *s, items: items(*s), age: meta.age(now), loading: meta.loading, error: meta.error.clone(), loads: meta.loads, ttl: s.ttl() }
            })
            .collect(),
        force_disconnected: shell.sidebar.force_disconnected,
        layout_json: layout.to_json(),
        layout_path: shell.layout.path().map(|p| p.display().to_string()),
        layout_note: shell.layout.note.text(),
        layout_dirty: shell.layout.is_dirty(),
        layout_saves: shell.layout.saves,
        layout_error: shell.layout.last_error.clone(),
        bottom_bar_open: layout.bottom_bar_open,
        right_rail_open: layout.right_rail_open,
    };
    ctx.data_mut(|d| d.insert_temp(snapshot_id(), snapshot));
}

pub fn latest(ctx: &Context) -> Option<Snapshot> {
    ctx.data(|d| d.get_temp::<Snapshot>(snapshot_id()))
}

/// Queues `request` for the shell.
pub fn push_request(ctx: &Context, request: Request) {
    ctx.data_mut(|d| d.get_temp_mut_or_default::<Vec<Request>>(requests_id()).push(request));
}

pub fn take_requests(ctx: &Context) -> Vec<Request> {
    ctx.data_mut(|d| d.remove_temp::<Vec<Request>>(requests_id())).unwrap_or_default()
}

/// Applies a request from the window to the shell.
pub fn apply(request: Request, shell: &mut Shell, ctx: &Context) {
    match request {
        Request::SetSection(section, open) => shell.layout.layout.set_section_open(section.key(), open),
        Request::SetSidebarOpen(open) => shell.layout.layout.sidebar_open = open,
        Request::Refresh(source) => shell.sidebar.data.invalidate(source),
        Request::RefreshAll => {
            shell.sidebar.data.invalidate_all();
            crate::ui::sidebar::request_refresh();
        }
        Request::ForceDisconnected(on) => shell.sidebar.force_disconnected = on,
        Request::ResetLayout => shell.layout.reset(),
        Request::SaveNow => {
            shell.layout.save_if_changed();
        }
    }
    ctx.request_repaint();
}

fn mode_text(mode: Mode) -> &'static str {
    match mode {
        Mode::Hidden => "hidden",
        Mode::Rail => "icon rail",
        Mode::Full => "full",
    }
}

fn age_text(age: Option<Duration>) -> String {
    match age {
        Some(d) if d.as_secs() >= 1 => format!("{} s ago", d.as_secs()),
        Some(_) => "just now".to_string(),
        None => "never".to_string(),
    }
}

fn body(ctx: &Context, ui: &mut Ui, s: &Snapshot) {
    Grid::new("ame_devtools_sidebar_grid").num_columns(2).striped(true).show(ui, |ui| {
        ui.label("Mode");
        ui.label(format!("{} ({:.0} px wide)", mode_text(s.mode), s.width));
        ui.end_row();
        ui.label("Sidebar open");
        let mut open = s.sidebar_open;
        if ui.checkbox(&mut open, "").changed() {
            push_request(ctx, Request::SetSidebarOpen(open));
        }
        ui.end_row();
        ui.label("Bottom bar / right rail");
        ui.label(format!("{} / {}", if s.bottom_bar_open { "shown" } else { "hidden" }, if s.right_rail_open { "shown" } else { "hidden" }));
        ui.end_row();
        for (section, open) in &s.sections {
            ui.label(format!("Section {}", section.title()));
            let mut value = *open;
            if ui.checkbox(&mut value, "open").changed() {
                push_request(ctx, Request::SetSection(*section, value));
            }
            ui.end_row();
        }
        ui.label("Library preview");
        let mut forced = s.force_disconnected;
        if ui.checkbox(&mut forced, "show as disconnected").changed() {
            push_request(ctx, Request::ForceDisconnected(forced));
        }
        ui.end_row();
    });
    ui.separator();
    ui.horizontal(|ui| {
        ui.strong("List caches");
        if ui.button("Refresh all").clicked() {
            push_request(ctx, Request::RefreshAll);
        }
    });
    Grid::new("ame_devtools_sidebar_sources").num_columns(6).striped(true).show(ui, |ui| {
        for head in ["List", "Items", "Loaded", "Loads", "State", ""] {
            ui.label(RichText::new(head).strong());
        }
        ui.end_row();
        for row in &s.sources {
            ui.label(row.source.name());
            ui.label(row.items.to_string());
            ui.label(format!("{} (every {} s)", age_text(row.age), row.ttl.as_secs()));
            ui.label(row.loads.to_string());
            match (&row.error, row.loading) {
                (Some(e), _) => ui.label(RichText::new(e).color(ui.visuals().error_fg_color)),
                (None, true) => ui.label("loading"),
                (None, false) => ui.label("idle"),
            };
            if ui.small_button("Refresh").clicked() {
                push_request(ctx, Request::Refresh(row.source));
            }
            ui.end_row();
        }
    });
    ui.separator();
    ui.horizontal(|ui| {
        ui.strong("Saved layout");
        if ui.button("Reset").on_hover_text("Back to the default layout").clicked() {
            push_request(ctx, Request::ResetLayout);
        }
        if ui.add_enabled(s.layout_dirty, eframe::egui::Button::new("Save now")).clicked() {
            push_request(ctx, Request::SaveNow);
        }
    });
    ui.label(format!("File: {}", s.layout_path.as_deref().unwrap_or("none (memory only)")));
    ui.label(format!("Load: {}; saved {} times{}", s.layout_note, s.layout_saves, if s.layout_dirty { "; unsaved changes" } else { "" }));
    if let Some(error) = &s.layout_error {
        ui.colored_label(ui.visuals().error_fg_color, format!("Last save failed: {error}"));
    }
    let mut json = s.layout_json.as_str();
    ScrollArea::vertical().max_height(180.0).show(ui, |ui| {
        ui.add(TextEdit::multiline(&mut json).code_editor().desired_width(f32::INFINITY).desired_rows(8));
    });
}

/// Draws the window when open.
pub fn show(ctx: &Context) {
    let mut open = is_open(ctx);
    if !open {
        return;
    }
    ctx.request_repaint_after(Duration::from_millis(500));
    Window::new("Devtools: Sidebar").open(&mut open).default_width(480.0).default_height(560.0).show(ctx, |ui| match latest(ctx) {
        Some(snapshot) => ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| body(ctx, ui, &snapshot)).inner,
        None => {
            ui.weak("Waiting for the first frame.");
        }
    });
    set_open(ctx, open);
}
