//! Metadata tools section: rename, merge, delete and prune tags and categories after a preview of what changes.

use eframe::egui::{DragValue, RichText, TextEdit, Ui};

use super::events::{Event, Tx};
use super::metadata_ops::{MAX_PRUNE_LIMIT, Op, Preview, preview, run};
use super::sections::name;
use super::widgets::{Focus, titled};
use crate::icons::{self, svg};

/// State of the metadata tools.
#[derive(Default)]
pub struct Metadata {
    pub category_old: String,
    pub category_new: String,
    pub tag_old: String,
    pub tag_new: String,
    pub tag_delete: String,
    pub prune_limit: u32,
    /// An edit whose count is being read.
    pub loading: Option<Op>,
    /// An edit and what it would touch, waiting for the answer.
    pub pending: Option<(Op, Preview)>,
    pub running: bool,
    pub result: Option<Result<String, String>>,
}

impl Metadata {
    pub fn new() -> Self {
        Self { prune_limit: 8, ..Default::default() }
    }

    pub fn on_event(&mut self, event: &Event) -> bool {
        match event {
            Event::Previewed(op, result) if self.loading.as_ref() == Some(op) => {
                self.loading = None;
                match result {
                    Ok(p) => self.pending = Some((op.clone(), *p)),
                    Err(e) => self.result = Some(Err(e.clone())),
                }
                true
            }
            Event::Edited(result) => {
                self.running = false;
                self.result = Some(result.clone());
                true
            }
            _ => false,
        }
    }

    /// Starts counting what `op` would touch.
    pub fn ask(&mut self, op: Op, tx: &Tx) {
        self.pending = None;
        self.result = None;
        if let Err(why) = op.validate() {
            self.result = Some(Err(why));
            return;
        }
        let op = op.trimmed();
        self.loading = Some(op.clone());
        let tx = tx.clone();
        tokio::spawn(async move {
            let result = preview(&crate::database::DB, &op).await.map_err(|e| format!("{e:#}"));
            let _ = tx.send(Event::Previewed(op, result));
            crate::command::request_repaint();
        });
    }

    /// Runs the edit that was previewed.
    pub fn confirm(&mut self, tx: &Tx) {
        let Some((op, _)) = self.pending.take() else { return };
        self.running = true;
        let tx = tx.clone();
        tokio::spawn(async move {
            let title = op.title();
            let result = run(op).await.map(|n| format!("{title}: {n} items changed")).map_err(|e| format!("{title} failed: {e:#}"));
            if result.is_ok() {
                crate::ui::sidebar::request_refresh();
            }
            let _ = tx.send(Event::Edited(result));
            crate::command::request_repaint();
        });
    }

    pub fn cancel(&mut self) {
        self.pending = None;
        self.loading = None;
    }

    fn busy(&self) -> bool {
        self.loading.is_some() || self.running
    }

    pub fn ui(&mut self, ui: &mut Ui, focus: &Focus, db_ready: bool, tx: &Tx) {
        if !db_ready {
            ui.colored_label(ui.visuals().warn_fg_color, "The library is not open. These tools change the library.");
        }
        ui.weak("Each tool shows how many items it changes before anything is written, and asks for confirmation.");
        let enabled = db_ready && !self.busy();
        let mut ask = None;

        titled(ui, focus, name::RENAME_CATEGORY);
        ui.horizontal_wrapped(|ui| {
            ui.add(TextEdit::singleline(&mut self.category_old).hint_text("Old category").desired_width(200.0).id_salt("meta-cat-old"));
            ui.add(TextEdit::singleline(&mut self.category_new).hint_text("New category").desired_width(200.0).id_salt("meta-cat-new"));
            if ui.add_enabled(enabled, eframe::egui::Button::new("Preview")).clicked() {
                ask = Some(Op::RenameCategory { old: self.category_old.clone(), new: self.category_new.clone() });
            }
        });

        titled(ui, focus, name::RENAME_TAG);
        ui.horizontal_wrapped(|ui| {
            ui.add(TextEdit::singleline(&mut self.tag_old).hint_text("Old tag").desired_width(200.0).id_salt("meta-tag-old"));
            ui.add(TextEdit::singleline(&mut self.tag_new).hint_text("New tag").desired_width(200.0).id_salt("meta-tag-new"));
            if ui.add_enabled(enabled, eframe::egui::Button::new("Preview")).clicked() {
                ask = Some(Op::RenameTag { old: self.tag_old.clone(), new: self.tag_new.clone() });
            }
        });

        titled(ui, focus, name::DELETE_TAG);
        ui.horizontal_wrapped(|ui| {
            ui.add(TextEdit::singleline(&mut self.tag_delete).hint_text("Tag to delete").desired_width(200.0).id_salt("meta-tag-delete"));
            if ui.add_enabled(enabled, eframe::egui::Button::new("Preview")).clicked() {
                ask = Some(Op::DeleteTag { tag: self.tag_delete.clone() });
            }
        });

        titled(ui, focus, name::PRUNE_TAGS);
        ui.horizontal_wrapped(|ui| {
            ui.label("Keep at most");
            ui.add(DragValue::new(&mut self.prune_limit).range(0..=MAX_PRUNE_LIMIT));
            ui.label("tags per item");
            if ui.add_enabled(enabled, eframe::egui::Button::new("Preview")).clicked() {
                ask = Some(Op::PruneTags { limit: self.prune_limit });
            }
        });

        if let Some(op) = ask {
            self.ask(op, tx);
        }
        ui.add_space(8.0);
        self.answer_ui(ui, tx);
    }

    fn answer_ui(&mut self, ui: &mut Ui, tx: &Tx) {
        if self.loading.is_some() {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Counting what would change...");
            });
        }
        if self.running {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Writing the change...");
            });
        }
        let mut confirm = false;
        let mut cancel = false;
        if let Some((op, p)) = &self.pending {
            eframe::egui::Frame::group(ui.style()).stroke(eframe::egui::Stroke::new(1.0, ui.visuals().warn_fg_color)).show(ui, |ui| {
                ui.label(RichText::new(op.title()).strong());
                ui.label(p.summary(op));
                ui.horizontal(|ui| {
                    if ui.add_enabled(p.changes_anything(), crate::theme::primary_button("Apply to the library")).clicked() {
                        confirm = true;
                    }
                    if icons::button(ui, svg::X, if p.changes_anything() { "Cancel" } else { "Close" }).clicked() {
                        cancel = true;
                    }
                });
            });
        }
        if confirm {
            self.confirm(tx);
        }
        if cancel {
            self.cancel();
        }
        match &self.result {
            Some(Ok(text)) => {
                ui.colored_label(crate::theme::success_ctx(ui.ctx()), text);
            }
            Some(Err(text)) => {
                ui.colored_label(ui.visuals().error_fg_color, text);
            }
            None => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::settings::events::channel;

    fn rename() -> Op {
        Op::RenameTag { old: "a".into(), new: "b".into() }
    }

    #[test]
    fn a_preview_answer_waits_for_confirmation_and_only_the_asked_edit_counts() {
        let mut m = Metadata::new();
        m.loading = Some(rename());
        assert!(!m.on_event(&Event::Previewed(Op::DeleteTag { tag: "x".into() }, Ok(Preview::default()))), "an answer to another edit is ignored");
        assert!(m.loading.is_some());
        assert!(m.on_event(&Event::Previewed(rename(), Ok(Preview { affected: 4, merged: 1, removed: 0 }))));
        assert!(m.loading.is_none());
        assert_eq!(m.pending.as_ref().map(|(_, p)| p.affected), Some(4));
        m.cancel();
        assert!(m.pending.is_none());
    }

    #[test]
    fn a_failed_preview_is_reported_and_nothing_waits_for_confirmation() {
        let mut m = Metadata::new();
        m.loading = Some(rename());
        m.on_event(&Event::Previewed(rename(), Err("database error".into())));
        assert!(m.pending.is_none());
        assert_eq!(m.result, Some(Err("database error".to_string())));
    }

    #[test]
    fn an_invalid_edit_is_refused_without_a_preview() {
        let (tx, rx) = channel();
        let mut m = Metadata::new();
        m.ask(Op::RenameTag { old: "a".into(), new: "a".into() }, &tx);
        assert!(m.loading.is_none() && m.pending.is_none());
        assert!(matches!(m.result, Some(Err(_))));
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn confirming_without_a_preview_does_nothing() {
        let (tx, rx) = channel();
        let mut m = Metadata::new();
        m.confirm(&tx);
        assert!(!m.running);
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn the_finished_edit_is_shown_and_frees_the_buttons() {
        let mut m = Metadata::new();
        m.running = true;
        assert!(m.busy());
        m.on_event(&Event::Edited(Ok("Rename tag 'a' to 'b': 3 items changed".into())));
        assert!(!m.busy());
        assert_eq!(m.result, Some(Ok("Rename tag 'a' to 'b': 3 items changed".to_string())));
    }
}
