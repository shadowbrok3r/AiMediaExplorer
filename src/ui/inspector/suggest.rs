//! The Suggest section of the AI tab: tags, a category and a caption suggested by the ZeroClaw gateway for the focused file, each accepted with one click through the metadata journal.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};

use crossbeam::channel::{Receiver, Sender, unbounded};
use egui::{Button, RichText, Ui};

use super::state::RowEdit;
use crate::Thumbnail;
use crate::ai::mcp::ctx::DbSource;
use crate::ai::refine::zeroclaw::StoredNeighbours;
use crate::database::journal;
use crate::database::metadata_edit::{self, MetaPatch, RowSeed, TagChange};
use crate::icons::{self, svg};
use crate::zc::jobs::ZcJobs;
use crate::zc::jobs::suggest::{self, Suggestions};

/// Journal label of an accepted suggestion.
pub const LABEL: &str = "Accepted suggestion";

/// The values one click writes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Pick {
    pub tags: Vec<String>,
    pub category: Option<String>,
    pub caption: Option<String>,
}

impl Pick {
    fn is_empty(&self) -> bool {
        self.tags.is_empty() && self.category.is_none() && self.caption.is_none()
    }

    /// Words for the confirmation line.
    fn describe(&self) -> String {
        let mut parts = Vec::new();
        if !self.tags.is_empty() {
            parts.push(format!("tag{} {}", if self.tags.len() == 1 { "" } else { "s" }, self.tags.join(", ")));
        }
        if let Some(c) = &self.category {
            parts.push(format!("category {c}"));
        }
        if self.caption.is_some() {
            parts.push("the caption".to_string());
        }
        parts.join(" and ")
    }
}

/// Writes `pick` to the file as one journaled change; the edit tells the browsers what is stored now.
pub async fn accept(db: &DbSource, row: &Thumbnail, pick: &Pick) -> Result<RowEdit, String> {
    if pick.is_empty() {
        return Err("nothing to write".to_string());
    }
    let handle = db.get().map_err(|e| e.message())?;
    let stored = journal::snapshots_with(handle, std::slice::from_ref(&row.path)).await.ok().and_then(|mut m| m.remove(&row.path));
    let mut patch = MetaPatch::default();
    if !pick.tags.is_empty() {
        let existing = stored.as_ref().and_then(|s| s.tags.clone()).unwrap_or_else(|| row.tags.clone());
        patch.tags = Some(metadata_edit::apply_tag_change(&existing, &TagChange { add: pick.tags.clone(), remove: Vec::new() }));
    }
    if let Some(category) = &pick.category {
        patch.category = Some(Some(category.clone()));
    }
    if let Some(caption) = &pick.caption {
        patch.caption = Some(Some(caption.clone()));
    }
    metadata_edit::save_patch_as(handle, &RowSeed::from_row(row), &patch, LABEL).await.map_err(|e| format!("{e:#}"))?;
    Ok(RowEdit::Patch { path: row.path.clone(), patch })
}

enum Msg {
    Suggested(Result<Suggestions, String>),
    Accepted { pick: Pick, result: Result<(), String> },
}

enum Phase {
    Idle,
    Asking(Arc<AtomicBool>),
    Ready(Suggestions),
    Failed(String),
}

struct Entry {
    phase: Phase,
    /// What the last click wrote, or why it failed.
    note: Option<(bool, String)>,
    writing: bool,
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
}

impl Default for Entry {
    fn default() -> Self {
        let (tx, rx) = unbounded();
        Self { phase: Phase::Idle, note: None, writing: false, tx, rx }
    }
}

impl Entry {
    fn poll(&mut self) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                Msg::Suggested(Ok(s)) => self.phase = Phase::Ready(s),
                Msg::Suggested(Err(e)) if e == "Cancelled" => self.phase = Phase::Idle,
                Msg::Suggested(Err(e)) => self.phase = Phase::Failed(e),
                Msg::Accepted { pick, result: Ok(()) } => {
                    self.writing = false;
                    self.note = Some((true, format!("Added {}. Ctrl+Z undoes it.", pick.describe())));
                    if let Phase::Ready(s) = &mut self.phase {
                        s.tags.retain(|t| !pick.tags.contains(t));
                        if pick.category.is_some() {
                            s.category = None;
                        }
                        if pick.caption.is_some() {
                            s.caption = None;
                        }
                        if s.is_empty() {
                            self.phase = Phase::Idle;
                        }
                    }
                }
                Msg::Accepted { result: Err(e), .. } => {
                    self.writing = false;
                    self.note = Some((false, format!("Not saved: {e}")));
                }
            }
        }
    }
}

/// Suggestions per file, shared by the frames.
#[derive(Default)]
pub struct Store {
    entries: Mutex<HashMap<String, Entry>>,
}

impl Store {
    fn lock(&self) -> MutexGuard<'_, HashMap<String, Entry>> {
        self.entries.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The suggestions waiting for `path`.
    pub fn ready(&self, path: &str) -> Option<Suggestions> {
        match self.lock().get_mut(path) {
            Some(Entry { phase: Phase::Ready(s), .. }) => Some(s.clone()),
            _ => None,
        }
    }

    pub fn is_asking(&self, path: &str) -> bool {
        matches!(self.lock().get(path), Some(Entry { phase: Phase::Asking(_), .. }))
    }

    /// The error of the last question about `path`.
    pub fn failure(&self, path: &str) -> Option<String> {
        match self.lock().get(path) {
            Some(Entry { phase: Phase::Failed(e), .. }) => Some(e.clone()),
            _ => None,
        }
    }

    /// The line the last accept left.
    pub fn note(&self, path: &str) -> Option<(bool, String)> {
        self.lock().get(path).and_then(|e| e.note.clone())
    }

    /// Takes in what the background work reported for `path`.
    pub fn pump(&self, path: &str) {
        if let Some(e) = self.lock().get_mut(path) {
            e.poll();
        }
    }
}

static GLOBAL: LazyLock<Arc<Store>> = LazyLock::new(|| Arc::new(Store::default()));

/// The environment tests draw the section with; `None` leaves the application's.
#[cfg(test)]
static TEST_ENV: Mutex<Option<Env>> = Mutex::new(None);

#[cfg(test)]
pub(crate) fn set_test_env(env: Option<Env>) {
    *TEST_ENV.lock().unwrap_or_else(|e| e.into_inner()) = env;
}

#[cfg(test)]
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Held by tests that set the environment or depend on it being the application's.
#[cfg(test)]
pub(crate) fn env_lock() -> MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// What the section works with: the job service, the library and the suggestions kept per file.
#[derive(Clone)]
pub struct Env {
    pub jobs: Arc<ZcJobs>,
    pub db: DbSource,
    pub store: Arc<Store>,
}

impl Env {
    /// The application's gateway, library and store.
    pub fn global() -> Self {
        #[cfg(test)]
        if let Some(env) = TEST_ENV.lock().unwrap_or_else(|e| e.into_inner()).clone() {
            return env;
        }
        Self { jobs: crate::zc::jobs::jobs().clone(), db: DbSource::Global, store: GLOBAL.clone() }
    }

    /// Why Suggest cannot run now; none while the gateway is Online or being checked.
    pub fn unavailable(&self) -> Option<String> {
        match self.jobs.availability() {
            a if a.is_ready() || self.jobs.is_settling() => None,
            a => a.reason().map(str::to_string),
        }
    }

    /// Asks for suggestions for `row`; false when one is already on its way or Suggest is unavailable.
    pub fn ask(&self, row: &Thumbnail) -> bool {
        self.ask_with_hint(row, None)
    }

    /// [`ask`](Self::ask) through the route `hint`.
    pub fn ask_with_hint(&self, row: &Thumbnail, hint: Option<String>) -> bool {
        if self.unavailable().is_some() || self.store.is_asking(&row.path) {
            return false;
        }
        let Ok(runtime) = tokio::runtime::Handle::try_current() else { return false };
        let cancel = Arc::new(AtomicBool::new(false));
        let tx = {
            let mut entries = self.store.lock();
            let entry = entries.entry(row.path.clone()).or_default();
            entry.phase = Phase::Asking(cancel.clone());
            entry.note = None;
            entry.tx.clone()
        };
        let (jobs, db, row) = (self.jobs.clone(), self.db.clone(), row.clone());
        runtime.spawn(async move {
            let neighbours = StoredNeighbours { db: db.clone() };
            let result = suggest::suggest(&jobs, &db, &neighbours, &row, hint, &cancel).await;
            let _ = tx.send(Msg::Suggested(result));
            crate::command::request_repaint();
        });
        true
    }

    /// Stops the question about `path`.
    pub fn cancel(&self, path: &str) {
        if let Some(Entry { phase: Phase::Asking(cancel), .. }) = self.store.lock().get(path) {
            cancel.store(true, Ordering::Relaxed);
        }
    }

    pub fn dismiss(&self, path: &str) {
        if let Some(e) = self.store.lock().get_mut(path) {
            if !matches!(e.phase, Phase::Asking(_)) {
                e.phase = Phase::Idle;
            }
        }
    }

    /// Writes `pick` to the file in the background; the browsers learn of it through the journal queue.
    pub fn accept(&self, row: &Thumbnail, pick: Pick) -> bool {
        let Ok(runtime) = tokio::runtime::Handle::try_current() else { return false };
        let tx = {
            let mut entries = self.store.lock();
            let entry = entries.entry(row.path.clone()).or_default();
            if entry.writing {
                return false;
            }
            entry.writing = true;
            entry.tx.clone()
        };
        let (db, row) = (self.db.clone(), row.clone());
        runtime.spawn(async move {
            let result = match accept(&db, &row, &pick).await {
                Ok(edit) => {
                    crate::ui::journal_ui::push_edits(vec![edit]);
                    Ok(())
                }
                Err(e) => Err(e),
            };
            let _ = tx.send(Msg::Accepted { pick, result });
            crate::command::request_repaint();
        });
        true
    }
}

/// What the user pressed.
enum Click {
    Suggest,
    Cancel,
    Dismiss,
    Pick(Pick),
}

fn chips(ui: &mut Ui, label: &str, values: &[String], writing: bool, click: &mut Option<Click>, make: impl Fn(&str) -> Pick) {
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(label).weak());
        for value in values {
            let button = Button::new(format!("+ {value}")).small();
            if ui.add_enabled(!writing, button).on_hover_text(format!("Add {value}; Ctrl+Z undoes it")).clicked() {
                *click = Some(Click::Pick(make(value)));
            }
        }
    });
}

/// Draws the section for `row` with the application's gateway and library.
pub fn show(ui: &mut Ui, row: &Thumbnail) {
    show_with(ui, &Env::global(), row);
}

/// Draws the section for `row` with `env`.
pub fn show_with(ui: &mut Ui, env: &Env, row: &Thumbnail) {
    env.store.pump(&row.path);
    ui.label(RichText::new("Suggestions").strong());
    ui.label(RichText::new("Tags, a category and a caption from this file's metadata and the files most like it. Only text is sent.").weak());
    let unavailable = env.unavailable();
    let asking = env.store.is_asking(&row.path);
    let suggestions = env.store.ready(&row.path);
    let failure = env.store.failure(&row.path);
    let writing = env.store.lock().get(&row.path).is_some_and(|e| e.writing);
    let mut click = None;
    ui.horizontal_wrapped(|ui| {
        if asking {
            ui.add(egui::Spinner::new().size(14.0));
            ui.label("Asking ZeroClaw...");
            if icons::button(ui, svg::STOP, "Cancel").clicked() {
                click = Some(Click::Cancel);
            }
        } else {
            let ready = unavailable.is_none();
            let response = ui.add_enabled(ready, icons::button_widget(ui, svg::SPARKLE, "Suggest"));
            let response = match &unavailable {
                Some(why) => response.on_disabled_hover_text(why.as_str()),
                None => response.on_hover_text("Ask the ZeroClaw gateway for tags and a caption"),
            };
            if response.clicked() {
                click = Some(Click::Suggest);
            }
        }
    });
    if let Some(why) = &unavailable {
        ui.label(RichText::new(format!("ZeroClaw is unavailable: {why}")).small().color(crate::theme::warn(ui))).on_hover_text(why.as_str());
    }
    if let Some(error) = &failure {
        ui.label(RichText::new(crate::ui::refine::model::clipped(error, 240)).color(crate::theme::error(ui))).on_hover_text(error.as_str());
        ui.label(RichText::new("Nothing was changed.").small().weak());
    }
    if let Some(s) = &suggestions {
        if s.is_empty() {
            ui.label(RichText::new("The model had nothing to add.").weak());
        }
        if !s.tags.is_empty() {
            chips(ui, "Tags", &s.tags, writing, &mut click, |t| Pick { tags: vec![t.to_string()], ..Default::default() });
        }
        if let Some(category) = &s.category {
            chips(ui, "Category", std::slice::from_ref(category), writing, &mut click, |c| Pick { category: Some(c.to_string()), ..Default::default() });
        }
        if let Some(caption) = &s.caption {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new("Caption").weak());
                ui.label(RichText::new(crate::ui::refine::model::clipped(caption, 200))).on_hover_text(caption.as_str());
            });
            if ui.add_enabled(!writing, icons::button_widget(ui, svg::CHECK, "Use caption")).on_hover_text("Replace the caption; Ctrl+Z undoes it").clicked() {
                click = Some(Click::Pick(Pick { caption: Some(caption.clone()), ..Default::default() }));
            }
        }
        if !s.is_empty() {
            ui.horizontal(|ui| {
                if ui.add_enabled(!writing, icons::button_widget(ui, svg::CHECKS, "Add all")).on_hover_text("Write every suggestion as one change").clicked() {
                    click = Some(Click::Pick(Pick { tags: s.tags.clone(), category: s.category.clone(), caption: s.caption.clone() }));
                }
                if icons::button(ui, svg::X, "Dismiss").clicked() {
                    click = Some(Click::Dismiss);
                }
                if !s.model.is_empty() {
                    ui.label(RichText::new(format!("Answered by {}", s.model)).small().weak());
                }
            });
        }
    }
    if let Some((ok, text)) = env.store.note(&row.path) {
        ui.label(RichText::new(text).color(if ok { crate::theme::success(ui) } else { crate::theme::warn(ui) }));
    }
    if asking || writing {
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(150));
    }
    match click {
        Some(Click::Suggest) => {
            env.ask(row);
        }
        Some(Click::Cancel) => env.cancel(&row.path),
        Some(Click::Dismiss) => env.dismiss(&row.path),
        Some(Click::Pick(pick)) => {
            env.accept(row, pick);
        }
        None => {}
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn a_pick_reads_as_a_sentence() {
        assert_eq!(Pick { tags: vec!["sea".into()], ..Default::default() }.describe(), "tag sea");
        assert_eq!(Pick { tags: vec!["a".into(), "b".into()], category: Some("c".into()), caption: Some("x".into()) }.describe(), "tags a, b and category c and the caption");
        assert!(Pick::default().is_empty());
    }
}
