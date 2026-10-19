//! Devtools window for the metadata journal: entries, batches, conflicts of the last undo, Undo and Redo buttons, a manual edit form and Clear.

use std::sync::{LazyLock, Mutex};

use eframe::egui::{CollapsingHeader, ComboBox, Context, Grid, Id, RichText, ScrollArea, TextEdit, Ui, Window};

use crate::database::DB;
use crate::database::journal::{self, BatchSummary, Direction, Entry, EntryState, Field, Stats, UndoReport};
use crate::database::metadata_edit::{self, MetaPatch, RowSeed};
use crate::icons::{self, svg};

const LISTED_ENTRIES: usize = 200;
const LISTED_BATCHES: usize = 40;

fn open_id() -> Id {
    Id::new("ame_devtools_journal_open")
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

/// What the window shows of the journal.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Listing {
    pub entries: Vec<Entry>,
    pub batches: Vec<BatchSummary>,
    pub stats: Stats,
}

struct Panel {
    ctx: Option<Context>,
    listing: Listing,
    loaded_at: Option<u64>,
    loading: bool,
    busy: bool,
    note: Option<(bool, String)>,
    error: Option<String>,
    confirm_clear: bool,
    form_path: String,
    form_field: Field,
    form_value: String,
}

impl Default for Panel {
    fn default() -> Self {
        Self {
            ctx: None,
            listing: Listing::default(),
            loaded_at: None,
            loading: false,
            busy: false,
            note: None,
            error: None,
            confirm_clear: false,
            form_path: String::new(),
            form_field: Field::Caption,
            form_value: String::new(),
        }
    }
}

static PANEL: LazyLock<Mutex<Panel>> = LazyLock::new(|| Mutex::new(Panel::default()));

fn with<R>(f: impl FnOnce(&mut Panel) -> R) -> R {
    f(&mut PANEL.lock().unwrap_or_else(|e| e.into_inner()))
}

fn repaint() {
    if let Some(ctx) = with(|p| p.ctx.clone()) {
        ctx.request_repaint();
    }
}

fn note(ok: bool, text: impl Into<String>) {
    let text = text.into();
    with(|p| p.note = Some((ok, text)));
    repaint();
}

/// Reads the journal into `Listing` form.
pub async fn load_listing<C: surrealdb::Connection>(db: &surrealdb::Surreal<C>) -> anyhow::Result<Listing> {
    Ok(Listing {
        entries: journal::recent_with(db, LISTED_ENTRIES).await?,
        batches: journal::batches_with(db, LISTED_BATCHES).await?,
        stats: journal::stats_with(db).await?,
    })
}

fn refresh() {
    if with(|p| std::mem::replace(&mut p.loading, true)) {
        return;
    }
    let revision = journal::revision();
    tokio::spawn(async move {
        let listing = load_listing(&DB).await;
        with(|p| {
            match listing {
                Ok(l) => {
                    p.listing = l;
                    p.error = None;
                }
                Err(e) => p.error = Some(format!("{e:#}")),
            }
            p.loaded_at = Some(revision);
            p.loading = false;
        });
        repaint();
    });
}

fn maybe_refresh(ctx: &Context) {
    let due = with(|p| {
        p.ctx = Some(ctx.clone());
        p.loaded_at != Some(journal::revision()) && !p.loading
    });
    if due && crate::database::is_ready() {
        refresh();
    }
}

fn run_direction(direction: Direction) {
    if with(|p| std::mem::replace(&mut p.busy, true)) {
        return;
    }
    tokio::spawn(async move {
        let outcome = match direction {
            Direction::Undo => journal::undo_last().await,
            Direction::Redo => journal::redo_last().await,
        };
        match outcome {
            Ok(report) => {
                note(report.conflicts.is_empty(), report.summary());
                crate::ui::journal_ui::push_result(report);
            }
            Err(e) => note(false, format!("{} failed: {e:#}", if direction == Direction::Undo { "Undo" } else { "Redo" })),
        }
        with(|p| p.busy = false);
        repaint();
    });
}

fn clear_all() {
    tokio::spawn(async {
        match journal::clear().await {
            Ok(n) => note(true, format!("Removed {n} entries")),
            Err(e) => note(false, format!("Clear failed: {e:#}")),
        }
    });
}

/// The patch the form writes: tags are comma separated, an empty value clears a text field.
pub fn form_patch(field: Field, value: &str) -> Option<MetaPatch> {
    let text = Some(Some(value.trim().to_string()).filter(|v| !v.is_empty()));
    match field {
        Field::Caption => Some(MetaPatch { caption: text, ..Default::default() }),
        Field::Description => Some(MetaPatch { description: text, ..Default::default() }),
        Field::Category => Some(MetaPatch { category: text, ..Default::default() }),
        Field::Tags => Some(MetaPatch { tags: Some(value.split(',').map(str::to_string).collect()), ..Default::default() }),
        Field::Collection => None,
    }
}

fn apply_form(path: String, field: Field, value: String) {
    let Some(patch) = form_patch(field, &value) else {
        return note(false, "Collection changes are made from the browser");
    };
    tokio::spawn(async move {
        match metadata_edit::save_patch_as(&DB, &RowSeed::from_path(&path), &patch, "Devtools edit").await {
            Ok(_) => note(true, format!("Wrote {} of {path}", field.as_str())),
            Err(e) => note(false, format!("Write failed: {e:#}")),
        }
    });
}

fn state_label(state: EntryState) -> &'static str {
    state.as_str()
}

fn time_label(at_ms: i64) -> String {
    chrono::DateTime::from_timestamp_millis(at_ms).map(|t| t.with_timezone(&chrono::Local).format("%H:%M:%S").to_string()).unwrap_or_default()
}

fn file_name(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

/// Draws the window when open.
pub fn show(ctx: &Context) {
    let mut open = is_open(ctx);
    if !open {
        return;
    }
    Window::new("Devtools: Journal").open(&mut open).default_width(860.0).default_height(560.0).show(ctx, |ui| {
        ScrollArea::vertical().auto_shrink([false, false]).show(ui, body);
    });
    set_open(ctx, open);
}

fn body(ui: &mut Ui) {
    maybe_refresh(ui.ctx());
    let ready = crate::database::is_ready();
    let (ok_color, err_color, warn_color) = (crate::theme::success(ui), crate::theme::error(ui), crate::theme::warn(ui));
    if !ready {
        ui.label(RichText::new("The database is not connected in this process; the journal is empty and the buttons are disabled.").color(warn_color));
    }
    if let Some((ok, text)) = with(|p| p.note.clone()) {
        ui.label(RichText::new(text).color(if ok { ok_color } else { err_color }));
    }
    if let Some(e) = with(|p| p.error.clone()) {
        ui.label(RichText::new(format!("Reading the journal failed: {e}")).color(err_color));
    }
    let (listing, busy) = with(|p| (p.listing.clone(), p.busy));
    ui.horizontal_wrapped(|ui| {
        if ui.add_enabled(ready && !busy, icons::button_widget(ui, svg::ARROW_COUNTER_CLOCKWISE, "Undo")).clicked() {
            run_direction(Direction::Undo);
        }
        if ui.add_enabled(ready && !busy, icons::button_widget(ui, svg::ARROW_CLOCKWISE, "Redo")).clicked() {
            run_direction(Direction::Redo);
        }
        if icons::button(ui, svg::ARROWS_CLOCKWISE, "Refresh").clicked() {
            with(|p| p.loaded_at = None);
        }
        let confirm = with(|p| p.confirm_clear);
        if confirm {
            if ui.add_enabled(ready, icons::button_widget(ui, svg::TRASH, "Confirm clear")).clicked() {
                with(|p| p.confirm_clear = false);
                clear_all();
            }
            if icons::button(ui, svg::X, "Keep").clicked() {
                with(|p| p.confirm_clear = false);
            }
        } else if ui.add_enabled(ready, icons::button_widget(ui, svg::TRASH, "Clear")).clicked() {
            with(|p| p.confirm_clear = true);
        }
        let s = listing.stats;
        ui.weak(format!("{} entries in {} batches: {} applied, {} undone, {} skipped; keeps {}", s.entries, s.batches, s.applied, s.undone, s.skipped, journal::RETENTION));
    });
    ui.separator();
    last_report_section(ui);
    batches_section(ui, &listing);
    entries_section(ui, &listing);
    form_section(ui, ready);
}

fn last_report_section(ui: &mut Ui) {
    let Some(report) = journal::last_report() else { return };
    ui.collapsing("Last undo or redo", |ui| {
        ui.label(report.summary());
        conflict_rows(ui, &report);
    });
}

fn conflict_rows(ui: &mut Ui, report: &UndoReport) {
    if report.conflicts.is_empty() {
        ui.weak("No conflicts.");
        return;
    }
    Grid::new("ame_journal_conflicts").num_columns(3).striped(true).show(ui, |ui| {
        for c in &report.conflicts {
            ui.label(file_name(&c.path));
            ui.label(c.field.as_str());
            ui.label(&c.reason);
            ui.end_row();
        }
    });
}

fn batches_section(ui: &mut Ui, listing: &Listing) {
    CollapsingHeader::new(format!("Batches ({})", listing.batches.len())).default_open(true).show(ui, |ui| {
        Grid::new("ame_journal_batches").num_columns(7).striped(true).spacing([10.0, 3.0]).show(ui, |ui| {
            for head in ["Time", "Batch", "Source", "Entries", "Files", "State", ""] {
                ui.label(RichText::new(head).strong());
            }
            ui.end_row();
            for b in &listing.batches {
                ui.label(time_label(b.at_ms));
                ui.label(RichText::new(b.batch.to_string()).monospace());
                ui.label(&b.source);
                ui.label(b.entries.to_string());
                ui.label(b.files.to_string());
                ui.label(format!("{} applied, {} undone, {} skipped", b.applied, b.undone, b.skipped));
                ui.end_row();
            }
        });
    });
}

fn entries_section(ui: &mut Ui, listing: &Listing) {
    CollapsingHeader::new(format!("Entries (newest {})", listing.entries.len())).default_open(true).show(ui, |ui| {
        if listing.entries.is_empty() {
            ui.weak("The journal is empty.");
        }
        Grid::new("ame_journal_entries").num_columns(7).striped(true).spacing([10.0, 3.0]).show(ui, |ui| {
            for head in ["Time", "Batch", "File", "Field", "Before", "After", "State"] {
                ui.label(RichText::new(head).strong());
            }
            ui.end_row();
            for e in &listing.entries {
                ui.label(time_label(e.at_ms));
                ui.label(RichText::new(e.batch.to_string()).monospace());
                ui.label(file_name(&e.path)).on_hover_text(&e.path);
                let field = match &e.target {
                    Some(t) => format!("{} {t}", e.field.as_str()),
                    None => e.field.as_str().to_string(),
                };
                ui.label(field);
                ui.label(e.before.brief());
                ui.label(e.after.brief());
                ui.label(state_label(e.state));
                ui.end_row();
            }
        });
    });
}

fn form_section(ui: &mut Ui, ready: bool) {
    ui.collapsing("Edit a file by hand", |ui| {
        let (mut path, mut field, mut value) = with(|p| (p.form_path.clone(), p.form_field, p.form_value.clone()));
        ui.horizontal(|ui| {
            ui.label("Path");
            ui.add(TextEdit::singleline(&mut path).desired_width(360.0));
        });
        ui.horizontal(|ui| {
            ComboBox::from_id_salt("ame_journal_form_field").selected_text(field.as_str()).show_ui(ui, |ui| {
                for f in [Field::Caption, Field::Description, Field::Category, Field::Tags] {
                    ui.selectable_value(&mut field, f, f.as_str());
                }
            });
            ui.add(TextEdit::singleline(&mut value).hint_text("New value (tags: comma separated; empty clears)").desired_width(300.0));
            if ui.add_enabled(ready && !path.trim().is_empty(), icons::button_widget(ui, svg::CHECK, "Apply")).clicked() {
                apply_form(path.trim().to_string(), field, value.clone());
            }
        });
        with(|p| {
            p.form_path = path;
            p.form_field = field;
            p.form_value = value;
        });
        ui.weak("The edit goes through the same journaled writer as the Inspector; press Undo to revert it.");
    });
}

#[cfg(test)]
mod tests;
