//! Undo and redo in the window: the keys, the background run, the toast and the refresh of the rows that were reverted.

use std::sync::Mutex;

use eframe::egui::{Context, Key, Modifiers};
use once_cell::sync::Lazy;

use crate::command::{AppCommand, CommandError, CommandResult, ToastLevel};
use crate::database::journal::{self, Direction, UndoReport};
use crate::ui::inspector::{RowEdit, state::apply_edit};

static RESULTS: Lazy<Mutex<Vec<UndoReport>>> = Lazy::new(|| Mutex::new(Vec::new()));
static EDITS: Lazy<Mutex<Vec<RowEdit>>> = Lazy::new(|| Mutex::new(Vec::new()));

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The command for Ctrl+Z or Ctrl+Shift+Z when no text field has focus.
pub fn read_keys(ctx: &Context) -> Option<AppCommand> {
    if ctx.egui_wants_keyboard_input() {
        return None;
    }
    ctx.input_mut(|i| {
        if i.consume_key(Modifiers::COMMAND | Modifiers::SHIFT, Key::Z) {
            Some(AppCommand::Redo)
        } else if i.consume_key(Modifiers::COMMAND, Key::Z) {
            Some(AppCommand::Undo)
        } else {
            None
        }
    })
}

/// Level of the toast that reports `report`.
pub fn toast_level(report: &UndoReport) -> ToastLevel {
    if report.is_empty() {
        ToastLevel::Info
    } else if report.conflicts.is_empty() {
        ToastLevel::Success
    } else {
        ToastLevel::Warning
    }
}

/// Starts an undo or redo in the background; the toast and the row refresh follow when it is done.
pub fn start(direction: Direction) -> CommandResult {
    if !crate::database::is_ready() {
        return Err(CommandError::failed("the library database is not connected"));
    }
    tokio::spawn(async move {
        let outcome = match direction {
            Direction::Undo => journal::undo_last().await,
            Direction::Redo => journal::redo_last().await,
        };
        match outcome {
            Ok(report) => {
                crate::command::send(AppCommand::Toast { kind: toast_level(&report), text: report.summary() });
                lock(&RESULTS).push(report);
                crate::command::request_repaint();
            }
            Err(e) => crate::command::send(AppCommand::Toast { kind: ToastLevel::Error, text: format!("{} failed: {e:#}", if direction == Direction::Undo { "Undo" } else { "Redo" }) }),
        }
    });
    Ok(format!("{} the last metadata change", if direction == Direction::Undo { "undoing" } else { "redoing" }))
}

/// Queues a finished report for the next frame.
pub fn push_result(report: UndoReport) {
    lock(&RESULTS).push(report);
}

/// Reports finished since the last call.
pub fn take_results() -> Vec<UndoReport> {
    std::mem::take(&mut *lock(&RESULTS))
}

/// The row edits that show `report` in the browsers.
pub fn row_edits(report: &UndoReport) -> Vec<RowEdit> {
    report.patches().into_iter().map(|(path, patch)| RowEdit::Patch { path, patch }).collect()
}

/// Queues edits that were written to the library for the rows shown in the browsers.
pub fn push_edits(edits: Vec<RowEdit>) {
    if edits.is_empty() {
        return;
    }
    lock(&EDITS).extend(edits);
    crate::command::request_repaint();
}

/// Edits queued since the last call.
pub fn take_edits() -> Vec<RowEdit> {
    std::mem::take(&mut *lock(&EDITS))
}

impl crate::app::SmartMediaApp {
    /// Once per frame: shows the values an undo, a redo or a tool wrote in every browser, refreshes what the inspector knows, and runs the tool tabs' background reports and devtools requests.
    pub fn apply_journal_results(&mut self) {
        let ctx = &mut self.context;
        ctx.refinements.pump();
        ctx.image_edit.pump(&self.egui_ctx);
        crate::ui::devtools::refine::frame(&mut ctx.refinements, &mut ctx.image_edit, &self.egui_ctx);
        let mut edits = take_edits();
        let mut membership = Vec::new();
        for report in take_results() {
            edits.extend(row_edits(&report));
            membership.extend(report.membership_paths());
        }
        let ctx = &mut self.context;
        for edit in &edits {
            apply_edit(&mut ctx.file_explorer, edit);
            ctx.filtered_tabs.values_mut().for_each(|ex| apply_edit(ex, edit));
        }
        for path in membership {
            ctx.inspector.library.invalidate(&path);
        }
    }
}

#[cfg(test)]
mod tests;
