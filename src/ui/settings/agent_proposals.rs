//! Proposals of the agent that wait for a decision: tag and category renames with the number of items they touch, and finished image edits to save.

use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam::channel::{Receiver, Sender, unbounded};
use eframe::egui::{Button, Frame, RichText, Ui};

use crate::ai::mcp::agent::AgentCtx;
use crate::ai::mcp::ctx::DbSource;
use crate::ai::mcp::edits::{EditJob, EditState, Review};
use crate::ai::mcp::proposals::Status;
use crate::ai::mcp::taxonomy::{Preview, TaxProposal};
use crate::command::{AppCommand, ToastLevel};
use crate::icons::{self, svg};
use crate::zc::events::now_ms;
use crate::zc::widgets::ago;

/// How often the list of proposals is read again while the section is shown.
const REFRESH: Duration = Duration::from_secs(8);
/// Proposals read per refresh.
const LIMIT: usize = 50;

/// A pending rename and what it would touch now.
#[derive(Clone, Debug)]
pub struct TaxCard {
    pub proposal: TaxProposal,
    pub preview: Option<Preview>,
}

enum Msg {
    Loaded(Result<(Vec<TaxCard>, u64), String>),
    Done(Result<String, String>),
}

pub struct Panel {
    agent: Option<Arc<AgentCtx>>,
    db: Option<DbSource>,
    pub cards: Vec<TaxCard>,
    pub pending_total: u64,
    loaded: bool,
    loading: bool,
    last: Option<Instant>,
    /// The rename the person is about to apply.
    pub confirm: Option<String>,
    pub working: bool,
    pub note: Option<(bool, String)>,
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
}

impl Default for Panel {
    fn default() -> Self {
        let (tx, rx) = unbounded();
        Self { agent: None, db: None, cards: Vec::new(), pending_total: 0, loaded: false, loading: false, last: None, confirm: None, working: false, note: None, tx, rx }
    }
}

/// Runs `work` on the current runtime; false when there is none.
fn spawn(work: impl std::future::Future<Output = ()> + Send + 'static) -> bool {
    match tokio::runtime::Handle::try_current() {
        Ok(rt) => {
            rt.spawn(work);
            true
        }
        Err(_) => false,
    }
}

fn toast(kind: ToastLevel, text: String) {
    crate::command::send(AppCommand::Toast { kind, text });
}

impl Panel {
    /// A panel over the stores in `agent`, adding library rows through `db`.
    pub fn with_agent(agent: Arc<AgentCtx>, db: DbSource) -> Self {
        Self { agent: Some(agent), db: Some(db), ..Self::default() }
    }

    fn agent(&mut self) -> Arc<AgentCtx> {
        self.agent.get_or_insert_with(|| crate::ai::mcp::hub().ctx.agent.clone()).clone()
    }

    fn db(&mut self) -> DbSource {
        self.db.get_or_insert(DbSource::Global).clone()
    }

    /// Image edit results that wait for the person.
    pub fn results(&mut self) -> Vec<EditJob> {
        self.agent().edits.list().into_iter().filter(|j| j.state == EditState::Done && j.review == Review::Pending).collect()
    }

    /// Reads the pending renames and counts what each would touch.
    pub fn reload(&mut self) {
        if self.loading {
            return;
        }
        self.loading = true;
        self.last = Some(Instant::now());
        let (agent, tx) = (self.agent(), self.tx.clone());
        let started = spawn(async move {
            let read = async {
                let (rows, total) = agent.taxonomy.list(Some(Status::Pending), 0, LIMIT).await.map_err(|e| e.message())?;
                let mut cards = Vec::with_capacity(rows.len());
                for proposal in rows {
                    let preview = agent.taxonomy.preview(&proposal.rename()).await.ok();
                    cards.push(TaxCard { proposal, preview });
                }
                Ok((cards, total))
            };
            let _ = tx.send(Msg::Loaded(read.await));
            crate::command::request_repaint();
        });
        if !started {
            self.loading = false;
            self.loaded = true;
        }
    }

    /// Applies the rename `id` through the journaled tag or category functions.
    pub fn accept(&mut self, id: String) {
        if self.working {
            return;
        }
        self.working = true;
        self.confirm = None;
        let (agent, tx) = (self.agent(), self.tx.clone());
        self.working = spawn(async move {
            let outcome = match agent.taxonomy.accept(&id).await {
                Ok(p) => {
                    crate::ui::sidebar::request_refresh();
                    let n = p.applied.unwrap_or(0);
                    Ok(format!("{}: {n} {} written. Ctrl+Z undoes {}.", p.rename().title(), if n == 1 { "change" } else { "changes" }, if n == 1 { "it" } else { "them" }))
                }
                Err(e) => Err(e.message()),
            };
            let _ = tx.send(Msg::Done(outcome));
            crate::command::request_repaint();
        });
    }

    pub fn reject(&mut self, id: String) {
        if self.working {
            return;
        }
        self.working = true;
        self.confirm = None;
        let (agent, tx) = (self.agent(), self.tx.clone());
        self.working = spawn(async move {
            let outcome = agent.taxonomy.reject(&id).await.map(|p| format!("Rejected: {}", p.rename().title())).map_err(|e| e.message());
            let _ = tx.send(Msg::Done(outcome));
            crate::command::request_repaint();
        });
    }

    /// Writes the result of an edit next to its original, and adds it to the library when `library` is set.
    pub fn save_result(&mut self, id: String, library: bool) {
        if self.working {
            return;
        }
        self.working = true;
        let (agent, db, tx) = (self.agent(), self.db(), self.tx.clone());
        self.working = spawn(async move {
            let saved = if library { agent.edits.add_to_library(&id, &db).await } else { agent.edits.save_new(&id) };
            let outcome = match saved {
                Ok((_, path)) => {
                    if library {
                        crate::ui::sidebar::request_refresh();
                    }
                    Ok(format!("Saved {}{}", path.display(), if library { " and added it to the library" } else { "" }))
                }
                Err(e) => Err(e.message()),
            };
            let _ = tx.send(Msg::Done(outcome));
            crate::command::request_repaint();
        });
    }

    pub fn reject_result(&mut self, id: &str) {
        match self.agent().edits.reject(id) {
            Ok(job) => self.note = Some((true, format!("Rejected the edit of {}.", job.file_name()))),
            Err(e) => self.note = Some((false, e.message())),
        }
    }

    fn poll(&mut self) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                Msg::Loaded(Ok((cards, total))) => {
                    self.loading = false;
                    self.loaded = true;
                    self.cards = cards;
                    self.pending_total = total;
                    if self.confirm.as_ref().is_some_and(|id| !self.cards.iter().any(|c| &c.proposal.id == id)) {
                        self.confirm = None;
                    }
                }
                Msg::Loaded(Err(e)) => {
                    self.loading = false;
                    self.loaded = true;
                    self.note = Some((false, format!("The proposals could not be read: {e}")));
                }
                Msg::Done(result) => {
                    self.working = false;
                    match &result {
                        Ok(text) => toast(ToastLevel::Success, text.clone()),
                        Err(text) => toast(ToastLevel::Error, text.clone()),
                    }
                    self.note = Some(match result {
                        Ok(text) => (true, text),
                        Err(text) => (false, text),
                    });
                    self.last = None;
                }
            }
        }
        if !self.loading && (!self.loaded || self.last.is_none_or(|at| at.elapsed() > REFRESH)) {
            self.reload();
        }
    }

    pub fn ui(&mut self, ui: &mut Ui, db_ready: bool) {
        self.poll();
        ui.ctx().request_repaint_after(REFRESH);
        let results = self.results();
        ui.horizontal_wrapped(|ui| {
            icons::heading(ui, svg::AME_ASSISTANT, "Agent proposals");
            let plural = |n: u64, one: &str, many: &str| format!("{n} {}", if n == 1 { one } else { many });
            ui.weak(format!("{}, {}", plural(self.pending_total, "rename", "renames"), plural(results.len() as u64, "edited picture", "edited pictures")));
            if ui.add_enabled(!self.loading, icons::button_widget(ui, svg::ARROW_CLOCKWISE, "Refresh")).clicked() {
                self.last = None;
            }
        });
        ui.weak("Renames and edited pictures the agent filed. Nothing changes until you accept it here.");
        if !db_ready {
            ui.colored_label(ui.visuals().warn_fg_color, "The library is not open. Accepting changes the library.");
        }
        let enabled = db_ready && !self.working;
        let mut accept = None;
        let mut reject = None;
        let mut ask = None;
        let mut cancel = false;
        for card in &self.cards {
            let p = &card.proposal;
            Frame::group(ui.style()).show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.label(RichText::new(p.rename().title()).strong());
                let counts = card.preview.as_ref().map(Preview::summary).unwrap_or_else(|| "Could not count the items.".to_string());
                ui.label(counts);
                if let Some(reason) = &p.reason {
                    ui.weak(format!("Reason: {reason}"));
                }
                ui.weak(format!("From {}, {}", p.source, ago(now_ms(), p.created_ms)));
                let touches = card.preview.as_ref().is_some_and(Preview::changes_anything);
                ui.horizontal_wrapped(|ui| {
                    if self.confirm.as_deref() == Some(p.id.as_str()) {
                        let n = card.preview.as_ref().map_or(0, |v| v.items);
                        if ui.add_enabled(enabled, crate::theme::primary_button(format!("Apply to {n} {}", if n == 1 { "item" } else { "items" }))).clicked() {
                            accept = Some(p.id.clone());
                        }
                        if icons::button(ui, svg::X, "Cancel").clicked() {
                            cancel = true;
                        }
                    } else {
                        if ui.add_enabled(enabled && touches, icons::button_widget(ui, svg::CHECK, "Accept")).on_hover_text("Shows what changes first; the change is journaled and Ctrl+Z undoes it").clicked() {
                            ask = Some(p.id.clone());
                        }
                        if ui.add_enabled(!self.working, icons::button_widget(ui, svg::X, "Reject")).clicked() {
                            reject = Some(p.id.clone());
                        }
                    }
                });
            });
        }
        let mut save = None;
        let mut library = None;
        let mut drop_result = None;
        for job in &results {
            Frame::group(ui.style()).show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.label(RichText::new(format!("Edited copy of {}", job.file_name())).strong());
                if let Some(r) = &job.result {
                    ui.label(format!("{}x{} pixels, {} KB", r.width, r.height, r.bytes.div_ceil(1024)));
                }
                ui.weak(format!("Prompt: {}", job.prompt));
                ui.weak(format!("From {}. The original is untouched.", job.client));
                ui.horizontal_wrapped(|ui| {
                    if ui.add_enabled(!self.working, icons::button_widget(ui, svg::FLOPPY_DISK, "Save as new file")).on_hover_text("Writes name-edited.png next to the original").clicked() {
                        save = Some(job.id.clone());
                    }
                    if ui.add_enabled(enabled, icons::button_widget(ui, svg::AME_DATABASE, "Add to library")).on_hover_text("Saves it next to the original and adds it to the library").clicked() {
                        library = Some(job.id.clone());
                    }
                    if ui.add_enabled(!self.working, Button::new("Discard")).clicked() {
                        drop_result = Some(job.id.clone());
                    }
                });
            });
        }
        if self.cards.is_empty() && results.is_empty() && self.loaded {
            ui.weak("Nothing waits for review.");
        }
        if self.working {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Working...");
            });
        }
        match &self.note {
            Some((true, text)) => {
                ui.colored_label(crate::theme::success_ctx(ui.ctx()), text);
            }
            Some((false, text)) => {
                ui.colored_label(ui.visuals().error_fg_color, text);
            }
            None => {}
        }
        if let Some(id) = ask {
            self.confirm = Some(id);
        }
        if cancel {
            self.confirm = None;
        }
        if let Some(id) = accept {
            self.accept(id);
        }
        if let Some(id) = reject {
            self.reject(id);
        }
        if let Some(id) = save {
            self.save_result(id, false);
        }
        if let Some(id) = library {
            self.save_result(id, true);
        }
        if let Some(id) = drop_result {
            self.reject_result(&id);
        }
    }
}
