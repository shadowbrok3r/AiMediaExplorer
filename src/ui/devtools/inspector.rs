//! Devtools window for the inspector: focus, preview source and texture cost, the edit state machine with buttons that drive each state, and the preferences file.

use eframe::egui::{Button, Context, Grid, Id, RichText, ScrollArea, Ui, Window};
use humansize::{BINARY, format_size};

use crate::icons::{self, svg};
use crate::ui::file_table::FileExplorer;
use crate::ui::inspector::meta::SaveState;
use crate::ui::inspector::prefs::{DetailTab, RailTab};
use crate::ui::inspector::state::Inspector;
use crate::ui::inspector::target::{Target, current_ai_state, stream_view};

#[derive(Clone, Debug)]
pub struct Snapshot {
    pub open: bool,
    pub overlay: bool,
    pub forced_overlay: Option<bool>,
    pub width: f32,
    pub drawn_width: Option<f32>,
    pub rail_tab: RailTab,
    pub detail_tab: DetailTab,
    pub target: String,
    pub focus: Option<String>,
    pub selection: usize,
    pub source: &'static str,
    pub texture: Option<[usize; 2]>,
    pub texture_bytes: usize,
    pub image: Option<[f32; 2]>,
    pub zoom_percent: Option<f32>,
    pub edit_state: SaveState,
    pub simulated: bool,
    pub dirty: String,
    pub last_error: Option<String>,
    pub bound: Option<String>,
    pub saves_started: usize,
    pub saves_ok: usize,
    pub saves_failed: usize,
    pub streaming: Option<usize>,
    pub streaming_files: usize,
    pub follow_ai: bool,
    pub ai: &'static str,
    pub rail_focused: bool,
    pub viewer_open: bool,
    pub prefs_path: Option<String>,
    pub prefs_error: Option<String>,
    pub prefs_saves: usize,
    pub known_tags: usize,
    pub collections: usize,
    pub last_batch: Option<String>,
}

impl Snapshot {
    pub fn capture(insp: &Inspector, ex: &FileExplorer, target: &Target, drawn_width: Option<f32>) -> Self {
        let focus = target.focus_path().map(str::to_string);
        let streaming = stream_view(focus.as_deref(), &ex.streaming_interim).map(str::len);
        Self {
            open: insp.is_open(),
            overlay: insp.overlay_active,
            forced_overlay: insp.force_overlay,
            width: insp.prefs.prefs.width,
            drawn_width,
            rail_tab: insp.prefs.prefs.rail_tab,
            detail_tab: insp.prefs.prefs.detail_tab,
            target: target.label(),
            focus,
            selection: ex.selection.len(),
            source: insp.preview.source,
            texture: insp.preview.texture,
            texture_bytes: insp.preview.texture_bytes(),
            image: insp.preview.image,
            zoom_percent: insp.preview.image.map(|i| insp.view.percent(eframe::egui::vec2(i[0], i[1]), eframe::egui::vec2(insp.prefs.prefs.width, insp.prefs.prefs.width * 0.72))),
            edit_state: insp.meta.state(),
            simulated: insp.meta.is_simulated(),
            dirty: insp.meta.dirty_fields().labels(),
            last_error: insp.meta.last_error().map(str::to_string),
            bound: insp.meta.path().map(str::to_string),
            saves_started: insp.counters.saves_started,
            saves_ok: insp.counters.saves_ok,
            saves_failed: insp.counters.saves_failed,
            streaming,
            streaming_files: ex.streaming_interim.len(),
            follow_ai: ex.follow_active_vision,
            ai: match current_ai_state() {
                crate::ui::inspector::target::AiState::Ready => "ready",
                crate::ui::inspector::target::AiState::NotConfigured => "not configured",
            },
            rail_focused: insp.rail_focused,
            viewer_open: insp.lightbox.open,
            prefs_path: insp.prefs.path().map(|p| p.display().to_string()),
            prefs_error: insp.prefs.load_error.clone().or_else(|| insp.prefs.last_save_error.clone()),
            prefs_saves: insp.prefs.saves,
            known_tags: insp.known_tags.len(),
            collections: insp.collections.len(),
            last_batch: insp.last_batch.clone(),
        }
    }
}

/// Something the window asks the host to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Command {
    ForceSave,
    /// Shows a save state in the indicator; `None` goes back to the real one.
    Simulate(Option<SaveState>),
    /// Appends text to the caption of the bound file, making it dirty.
    EditCaption,
    Next,
    Previous,
    ClearSelection,
    ToggleViewer,
    ForceOverlay(Option<bool>),
    FakeStream(bool),
    ResetPrefs,
    ToggleOpen,
}

fn open_id() -> Id {
    Id::unique("ame_devtools_inspector_open")
}

fn snapshot_id() -> Id {
    Id::unique("ame_devtools_inspector_snapshot")
}

fn commands_id() -> Id {
    Id::unique("ame_devtools_inspector_commands")
}

pub fn is_open(ctx: &Context) -> bool {
    ctx.data(|d| d.get_temp::<bool>(open_id())).unwrap_or(false)
}

pub fn set_open(ctx: &Context, open: bool) {
    ctx.data_mut(|d| d.insert_temp(open_id(), open));
}

pub fn toggle(ctx: &Context) {
    set_open(ctx, !is_open(ctx));
}

pub fn publish(ctx: &Context, snapshot: Snapshot) {
    ctx.data_mut(|d| d.insert_temp(snapshot_id(), snapshot));
}

pub fn latest(ctx: &Context) -> Option<Snapshot> {
    ctx.data(|d| d.get_temp::<Snapshot>(snapshot_id()))
}

pub fn push_command(ctx: &Context, command: Command) {
    ctx.data_mut(|d| d.get_temp_mut_or_default::<Vec<Command>>(commands_id()).push(command));
    ctx.request_repaint();
}

pub fn take_commands(ctx: &Context) -> Vec<Command> {
    ctx.data_mut(|d| d.remove_temp::<Vec<Command>>(commands_id())).unwrap_or_default()
}

/// Draws the window when open.
pub fn show(ctx: &Context) {
    let mut open = is_open(ctx);
    if !open {
        return;
    }
    Window::new("Devtools: Inspector").open(&mut open).default_width(440.0).default_height(620.0).show(ctx, |ui| {
        ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| match latest(ctx) {
            Some(s) => body(ui, ctx, &s),
            None => {
                ui.weak("The rail has not drawn since the window opened.");
                if ui.button("Open the inspector").clicked() {
                    push_command(ctx, Command::ToggleOpen);
                }
            }
        });
    });
    set_open(ctx, open);
}

fn yes(v: bool) -> &'static str {
    if v { "yes" } else { "no" }
}

fn body(ui: &mut Ui, ctx: &Context, s: &Snapshot) {
    ui.ctx().request_repaint_after(std::time::Duration::from_millis(250));
    icons::heading(ui, svg::SIDEBAR_SIMPLE, "Rail");
    Grid::new("dev_inspector_rail").num_columns(2).spacing([16.0, 3.0]).show(ui, |ui| {
        ui.label("Open / overlay");
        ui.monospace(format!("{} / {}{}", yes(s.open), yes(s.overlay), s.forced_overlay.map_or(String::new(), |f| format!(" (forced {})", yes(f)))));
        ui.end_row();
        ui.label("Width");
        ui.monospace(format!("{:.0} saved, {} drawn", s.width, s.drawn_width.map_or("-".to_string(), |w| format!("{w:.0}"))));
        ui.end_row();
        ui.label("Tabs");
        ui.monospace(format!("{:?} / {:?}", s.rail_tab, s.detail_tab));
        ui.end_row();
        ui.label("Rail has the keys / viewer open");
        ui.monospace(format!("{} / {}", yes(s.rail_focused), yes(s.viewer_open)));
        ui.end_row();
    });
    ui.horizontal_wrapped(|ui| {
        if ui.button("Toggle rail").clicked() {
            push_command(ctx, Command::ToggleOpen);
        }
        if ui.button("Force overlay").clicked() {
            push_command(ctx, Command::ForceOverlay(Some(true)));
        }
        if ui.button("Force docked").clicked() {
            push_command(ctx, Command::ForceOverlay(Some(false)));
        }
        if ui.button("Automatic").clicked() {
            push_command(ctx, Command::ForceOverlay(None));
        }
        if ui.button("Toggle viewer").clicked() {
            push_command(ctx, Command::ToggleViewer);
        }
    });
    ui.separator();
    icons::heading(ui, svg::CURSOR_CLICK, "Focus");
    Grid::new("dev_inspector_focus").num_columns(2).spacing([16.0, 3.0]).show(ui, |ui| {
        ui.label("Target");
        ui.monospace(&s.target);
        ui.end_row();
        ui.label("Focus path");
        ui.monospace(s.focus.as_deref().unwrap_or("-"));
        ui.end_row();
        ui.label("Selected rows");
        ui.monospace(s.selection.to_string());
        ui.end_row();
        ui.label("Editor bound to");
        ui.monospace(s.bound.as_deref().unwrap_or("-"));
        ui.end_row();
    });
    ui.horizontal(|ui| {
        if ui.button("Previous file").clicked() {
            push_command(ctx, Command::Previous);
        }
        if ui.button("Next file").clicked() {
            push_command(ctx, Command::Next);
        }
        if ui.button("Clear selection").clicked() {
            push_command(ctx, Command::ClearSelection);
        }
    });
    ui.separator();
    icons::heading(ui, svg::IMAGE, "Preview");
    Grid::new("dev_inspector_preview").num_columns(2).spacing([16.0, 3.0]).show(ui, |ui| {
        ui.label("Source");
        ui.monospace(s.source);
        ui.end_row();
        ui.label("Texture");
        ui.monospace(match s.texture {
            Some([w, h]) => format!("{w} x {h}, {}", format_size(s.texture_bytes as u64, BINARY)),
            None => "none".to_string(),
        });
        ui.end_row();
        ui.label("Picture size / zoom");
        ui.monospace(match (s.image, s.zoom_percent) {
            (Some([w, h]), Some(z)) => format!("{w:.0} x {h:.0}, {z:.0}%"),
            _ => "-".to_string(),
        });
        ui.end_row();
    });
    ui.separator();
    icons::heading(ui, svg::FLOPPY_DISK, "Metadata edit");
    let color = match s.edit_state {
        SaveState::Error => crate::theme::error(ui),
        SaveState::Dirty => crate::theme::warn(ui),
        SaveState::Saved => crate::theme::success(ui),
        _ => ui.visuals().text_color(),
    };
    Grid::new("dev_inspector_edit").num_columns(2).spacing([16.0, 3.0]).show(ui, |ui| {
        ui.label("State");
        ui.label(RichText::new(format!("{:?}{}", s.edit_state, if s.simulated { " (simulated)" } else { "" })).color(color).monospace());
        ui.end_row();
        ui.label("Unsaved fields");
        ui.monospace(if s.dirty.is_empty() { "-" } else { s.dirty.as_str() });
        ui.end_row();
        ui.label("Last save error");
        ui.monospace(s.last_error.as_deref().unwrap_or("-"));
        ui.end_row();
        ui.label("Saves started / ok / failed");
        ui.monospace(format!("{} / {} / {}", s.saves_started, s.saves_ok, s.saves_failed));
        ui.end_row();
    });
    ui.horizontal_wrapped(|ui| {
        if ui.add(Button::new("Force save")).clicked() {
            push_command(ctx, Command::ForceSave);
        }
        if ui.button("Edit caption").on_hover_text("Appends text to the caption of the bound file").clicked() {
            push_command(ctx, Command::EditCaption);
        }
    });
    ui.horizontal_wrapped(|ui| {
        ui.label("Simulate:");
        for (label, state) in [("Dirty", SaveState::Dirty), ("Saving", SaveState::Saving), ("Saved", SaveState::Saved), ("Error", SaveState::Error)] {
            if ui.button(label).clicked() {
                push_command(ctx, Command::Simulate(Some(state)));
            }
        }
        if ui.button("Real state").clicked() {
            push_command(ctx, Command::Simulate(None));
        }
    });
    ui.separator();
    icons::heading(ui, svg::AME_AI_DESCRIBE, "AI");
    Grid::new("dev_inspector_ai").num_columns(2).spacing([16.0, 3.0]).show(ui, |ui| {
        ui.label("Vision model");
        ui.monospace(s.ai);
        ui.end_row();
        ui.label("Streams / shown for focus");
        ui.monospace(format!("{} / {}", s.streaming_files, s.streaming.map_or("none".to_string(), |n| format!("{n} bytes"))));
        ui.end_row();
        ui.label("Follow other files");
        ui.monospace(yes(s.follow_ai));
        ui.end_row();
        ui.label("Known tags / collections");
        ui.monospace(format!("{} / {}", s.known_tags, s.collections));
        ui.end_row();
        ui.label("Last batch");
        ui.monospace(s.last_batch.as_deref().unwrap_or("-"));
        ui.end_row();
    });
    ui.horizontal(|ui| {
        if ui.button("Fake a stream").clicked() {
            push_command(ctx, Command::FakeStream(true));
        }
        if ui.button("Clear the stream").clicked() {
            push_command(ctx, Command::FakeStream(false));
        }
    });
    ui.separator();
    icons::heading(ui, svg::FILE_TEXT, "Preferences file");
    Grid::new("dev_inspector_prefs").num_columns(2).spacing([16.0, 3.0]).show(ui, |ui| {
        ui.label("Path");
        ui.monospace(s.prefs_path.as_deref().unwrap_or("none (not persisted)"));
        ui.end_row();
        ui.label("Writes / error");
        ui.monospace(format!("{} / {}", s.prefs_saves, s.prefs_error.as_deref().unwrap_or("-")));
        ui.end_row();
    });
    if ui.button("Reset preferences").clicked() {
        push_command(ctx, Command::ResetPrefs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commands_queue_in_order_and_are_taken_once() {
        let ctx = crate::ui::test_support::test_ctx();
        push_command(&ctx, Command::ForceSave);
        push_command(&ctx, Command::Simulate(Some(SaveState::Error)));
        assert_eq!(take_commands(&ctx), vec![Command::ForceSave, Command::Simulate(Some(SaveState::Error))]);
        assert!(take_commands(&ctx).is_empty());
    }

    #[test]
    fn the_window_opens_and_closes() {
        let ctx = crate::ui::test_support::test_ctx();
        assert!(!is_open(&ctx));
        toggle(&ctx);
        assert!(is_open(&ctx));
        set_open(&ctx, false);
        assert!(!is_open(&ctx));
    }
}
