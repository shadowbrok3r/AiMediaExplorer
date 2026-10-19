//! Shortcuts section: every action and command the palette registers, with its key binding.

use eframe::egui::{Grid, ScrollArea, TextEdit, Ui};

use super::sections::name;
use super::widgets::{Focus, titled};
use crate::ui::palette::{self, ActionContext, Effect};

/// Bindings that no palette entry carries.
pub const EXTRA: [(&str, &str); 5] = [
    ("Toggle the right rail", "Ctrl+I"),
    ("Back in the focused tab", "Alt+Left or mouse back"),
    ("Forward in the focused tab", "Alt+Right or mouse forward"),
    ("Up one level", "Alt+Up"),
    ("Close the palette", "Esc"),
];

/// One line of the table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    pub title: String,
    pub binding: String,
    /// The command or window action behind it.
    pub runs: String,
}

fn runs_of(effect: &Effect) -> String {
    match effect {
        Effect::Command(cmd) => cmd.log_json(),
        Effect::Shell(action) => format!("shell action {}", action.id()),
    }
}

/// Every registered action with its binding, then the keys the registry does not list.
pub fn rows() -> Vec<Row> {
    let mut rows: Vec<Row> = palette::registry(&ActionContext::default())
        .into_iter()
        .map(|e| Row { title: e.title, binding: e.shortcut.unwrap_or("").to_string(), runs: runs_of(&e.effect) })
        .collect();
    rows.extend(EXTRA.iter().map(|(title, binding)| Row { title: (*title).to_string(), binding: (*binding).to_string(), runs: "keyboard".to_string() }));
    rows
}

/// The rows whose title, binding or command contain every word of `filter`.
pub fn filtered<'a>(rows: &'a [Row], filter: &str) -> Vec<&'a Row> {
    let words: Vec<String> = filter.split_whitespace().map(str::to_lowercase).collect();
    rows.iter().filter(|r| {
        let hay = format!("{} {} {}", r.title, r.binding, r.runs).to_lowercase();
        words.iter().all(|w| hay.contains(w))
    }).collect()
}

/// State of the shortcuts table.
#[derive(Default)]
pub struct Shortcuts {
    rows: Vec<Row>,
    pub filter: String,
}

impl Shortcuts {
    pub fn ui(&mut self, ui: &mut Ui, focus: &Focus) {
        if self.rows.is_empty() {
            self.rows = rows();
        }
        titled(ui, focus, name::BINDINGS);
        ui.add(TextEdit::singleline(&mut self.filter).hint_text("Filter").desired_width(220.0));
        ui.weak("Bindings cannot be changed yet. Every action is also a JSON command (docs/commands.md) and runs from the palette (Ctrl+K).");
        let shown = filtered(&self.rows, &self.filter);
        ScrollArea::vertical().id_salt("settings-shortcuts").auto_shrink([false, false]).show(ui, |ui| {
            Grid::new("settings-shortcuts-grid").striped(true).num_columns(3).spacing([16.0, 4.0]).show(ui, |ui| {
                for header in ["Action", "Binding", "Runs"] {
                    ui.strong(header);
                }
                ui.end_row();
                for row in shown {
                    ui.label(row.title.as_str());
                    ui.monospace(row.binding.as_str());
                    ui.add(eframe::egui::Label::new(eframe::egui::RichText::new(row.runs.as_str()).weak().monospace()).truncate()).on_hover_text(row.runs.as_str());
                    ui.end_row();
                }
            });
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_lists_every_registered_action_and_the_extra_keys() {
        let all = rows();
        let registry = palette::registry(&ActionContext::default());
        assert_eq!(all.len(), registry.len() + EXTRA.len());
        assert!(all.iter().any(|r| r.title == "Command palette" && r.binding == "Ctrl+K"));
        assert!(all.iter().any(|r| r.title == "Toggle sidebar" && r.binding == "Ctrl+B"));
        assert!(all.iter().any(|r| r.runs.contains("\"cmd\":\"show_settings\"")));
        assert!(all.iter().any(|r| r.runs.starts_with("shell action ")));
        assert!(all.iter().all(|r| !r.title.is_empty() && !r.runs.is_empty()));
    }

    #[test]
    fn filtering_needs_every_word_and_ignores_case() {
        let all = rows();
        assert!(filtered(&all, "").len() == all.len());
        let hits = filtered(&all, "CTRL k");
        assert!(hits.iter().any(|r| r.title == "Command palette"));
        assert!(hits.len() < all.len());
        assert!(filtered(&all, "zzzz-nothing").is_empty());
    }
}
