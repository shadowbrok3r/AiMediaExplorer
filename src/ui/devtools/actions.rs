//! Devtools window listing the palette's action registry: name, shortcut, enabled state and reason.

use eframe::egui::{Context, Grid, Id, RichText, ScrollArea, TextEdit, Ui, Window};

use crate::ui::palette::{Effect, Entry, matcher};

/// Entries published by the shell while the window is open.
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub entries: Vec<Entry>,
}

fn open_id() -> Id {
    Id::unique("ame_devtools_actions_open")
}

fn snapshot_id() -> Id {
    Id::unique("ame_devtools_actions_snapshot")
}

fn requests_id() -> Id {
    Id::unique("ame_devtools_actions_requests")
}

fn filter_id() -> Id {
    Id::unique("ame_devtools_actions_filter")
}

pub fn is_open(ctx: &Context) -> bool {
    ctx.data(|d| d.get_temp::<bool>(open_id())).unwrap_or(false)
}

pub fn set_open(ctx: &Context, open: bool) {
    ctx.data_mut(|d| d.insert_temp(open_id(), open));
}

/// Records the entries the window shows.
pub fn publish(ctx: &Context, entries: Vec<Entry>) {
    ctx.data_mut(|d| d.insert_temp(snapshot_id(), Snapshot { entries }));
}

pub fn latest(ctx: &Context) -> Option<Snapshot> {
    ctx.data(|d| d.get_temp::<Snapshot>(snapshot_id()))
}

/// Asks the shell to run the entry called `id`.
pub fn push_request(ctx: &Context, id: String) {
    ctx.data_mut(|d| d.get_temp_mut_or_default::<Vec<String>>(requests_id()).push(id));
}

pub fn take_requests(ctx: &Context) -> Vec<String> {
    ctx.data_mut(|d| d.remove_temp::<Vec<String>>(requests_id())).unwrap_or_default()
}

/// What the effect of an entry is called in the window.
pub fn effect_name(effect: &Effect) -> String {
    match effect {
        Effect::Command(cmd) => cmd.name().to_string(),
        Effect::Shell(action) => format!("shell:{}", action.id()),
    }
}

fn body(ctx: &Context, ui: &mut Ui, snapshot: &Snapshot) {
    let mut filter = ctx.data(|d| d.get_temp::<String>(filter_id())).unwrap_or_default();
    ui.horizontal(|ui| {
        ui.label("Filter");
        ui.add(TextEdit::singleline(&mut filter).desired_width(220.0));
        let disabled = snapshot.entries.iter().filter(|e| e.enabled.is_err()).count();
        ui.label(RichText::new(format!("{} actions, {} disabled", snapshot.entries.len(), disabled)).weak());
    });
    ctx.data_mut(|d| d.insert_temp(filter_id(), filter.clone()));
    let shown = matcher::rank(&filter, &snapshot.entries, |e| (e.title.as_str(), e.id.as_str()));
    ScrollArea::both().auto_shrink([false, false]).show(ui, |ui| {
        Grid::new("ame_devtools_actions_grid").num_columns(6).striped(true).show(ui, |ui| {
            for head in ["Name", "Id", "Runs", "Shortcut", "Enabled", ""] {
                ui.label(RichText::new(head).strong());
            }
            ui.end_row();
            for ranked in shown {
                let entry = &snapshot.entries[ranked.index];
                ui.label(&entry.title);
                ui.label(RichText::new(&entry.id).monospace());
                ui.label(effect_name(&entry.effect));
                ui.label(entry.shortcut.unwrap_or(""));
                match &entry.enabled {
                    Ok(()) => ui.label("yes"),
                    Err(reason) => ui.label(RichText::new(format!("no: {reason}")).color(ui.visuals().warn_fg_color)),
                };
                if ui.add_enabled(entry.enabled.is_ok(), eframe::egui::Button::new("Run").small()).clicked() {
                    push_request(ctx, entry.id.clone());
                }
                ui.end_row();
            }
        });
    });
}

/// Draws the window when open.
pub fn show(ctx: &Context) {
    let mut open = is_open(ctx);
    if !open {
        return;
    }
    Window::new("Devtools: Actions").open(&mut open).default_width(960.0).default_height(520.0).show(ctx, |ui| match latest(ctx) {
        Some(snapshot) => body(ctx, ui, &snapshot),
        None => {
            ui.weak("Waiting for the first frame.");
        }
    });
    set_open(ctx, open);
}
