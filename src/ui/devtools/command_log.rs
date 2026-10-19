//! Devtools window with the command log and a runner for commands given as JSON or a scope string.

use eframe::egui::{Color32, Context, FontId, Id, RichText, ScrollArea, TextEdit, TextStyle, Ui, Window};

use crate::command::{self, LogEntry};
use crate::icons::{self, svg};

/// Longest JSON shown on one log line before it is cut.
const LINE_CHARS: usize = 150;

fn open_id() -> Id {
    Id::unique("ame_devtools_command_log_open")
}

fn state_id() -> Id {
    Id::unique("ame_devtools_command_log_state")
}

pub fn is_open(ctx: &Context) -> bool {
    ctx.data(|d| d.get_temp::<bool>(open_id())).unwrap_or(false)
}

pub fn set_open(ctx: &Context, open: bool) {
    ctx.data_mut(|d| d.insert_temp(open_id(), open));
}

#[derive(Clone, Default)]
struct State {
    input: String,
    message: Option<(bool, String)>,
    failures_only: bool,
}

/// Cuts `text` to `max` characters, marking the cut.
pub fn clip(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        text.to_string()
    } else {
        text.chars().take(max.saturating_sub(3)).collect::<String>() + "..."
    }
}

/// The runner's feedback for `text`: the commands were queued, or why they were not.
pub fn run_text(text: &str) -> (bool, String) {
    match command::send_text(text) {
        Ok(1) => (true, "Queued 1 command".to_string()),
        Ok(n) => (true, format!("Queued {n} commands")),
        Err(e) => (false, e),
    }
}

fn entry_row(ui: &mut Ui, entry: &LogEntry) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(entry.at.format("%H:%M:%S").to_string()).weak());
        let (mark, color) = if entry.outcome.is_ok() { ("ok", ui.visuals().text_color()) } else { ("error", crate::theme::error(ui)) };
        ui.label(RichText::new(mark).color(color));
        let json = ui.label(RichText::new(clip(&entry.json, LINE_CHARS)).font(FontId::monospace(12.0)));
        json.on_hover_text(&entry.json);
    });
    let result = entry.outcome.text();
    if !result.is_empty() {
        let color = if entry.outcome.is_ok() { ui.visuals().weak_text_color() } else { crate::theme::error(ui) };
        ui.horizontal(|ui| {
            ui.add_space(24.0);
            ui.label(RichText::new(result).color(color));
            if icons::icon_button(ui, svg::ARROW_CLOCKWISE, "Run this command again").clicked() {
                let _ = command::send_text(&entry.json);
            }
            if icons::icon_button(ui, svg::COPY, "Copy the JSON").clicked() {
                ui.ctx().copy_text(entry.json.clone());
            }
        });
    }
}

fn body(ctx: &Context, ui: &mut Ui) {
    let mut state = ctx.data(|d| d.get_temp::<State>(state_id())).unwrap_or_default();
    ui.label("Run a command as JSON, a list of commands, or a scope such as fs:/path, lib:all, tag:x.");
    let edit = ui.add(
        TextEdit::multiline(&mut state.input)
            .font(TextStyle::Monospace)
            .desired_rows(3)
            .desired_width(f32::INFINITY)
            .hint_text(r#"{"cmd":"open_scope","scope":"lib:all","target":"new_tab"}"#),
    );
    ui.horizontal(|ui| {
        let run = icons::button(ui, svg::PLAY, "Run").clicked() || (edit.has_focus() && ui.input(|i| i.modifiers.command && i.key_pressed(egui::Key::Enter)));
        if run {
            state.message = Some(run_text(&state.input));
        }
        if ui.button("Clear log").clicked() {
            command::with_log(|log| log.clear());
        }
        ui.checkbox(&mut state.failures_only, "Failures only");
        if let Some((ok, text)) = &state.message {
            let color = if *ok { Color32::GRAY } else { crate::theme::error(ui) };
            ui.label(RichText::new(text).color(color));
        }
    });
    ui.separator();
    let (entries, total, failures): (Vec<LogEntry>, usize, usize) =
        command::with_log(|log| (log.entries().rev().filter(|e| !state.failures_only || !e.outcome.is_ok()).cloned().collect(), log.len(), log.failures()));
    ui.label(RichText::new(format!("{total} commands logged, {failures} failed (newest first)")).weak());
    ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        for entry in &entries {
            entry_row(ui, entry);
            ui.add_space(2.0);
        }
    });
    ctx.data_mut(|d| d.insert_temp(state_id(), state));
}

/// Draws the window when open.
pub fn show(ctx: &Context) {
    let mut open = is_open(ctx);
    if !open {
        return;
    }
    Window::new("Devtools: Command log").open(&mut open).default_width(640.0).default_height(460.0).show(ctx, |ui| body(ctx, ui));
    set_open(ctx, open);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::bus::testing::{serial, take_matching};
    use crate::command::{AppCommand, CommandError, Outcome};

    #[test]
    fn long_json_is_cut_with_a_marker() {
        assert_eq!(clip("short", 10), "short");
        assert_eq!(clip("0123456789abcdef", 10), "0123456...");
    }

    #[test]
    fn the_runner_queues_json_lists_and_scopes_and_reports_bad_input() {
        let _serial = serial();
        take_matching(|c| matches!(c, AppCommand::ClearSelection | AppCommand::ShowTaskCenter));
        assert_eq!(run_text(r#"{"cmd":"clear_selection"}"#), (true, "Queued 1 command".to_string()));
        assert_eq!(run_text(r#"[{"cmd":"clear_selection"},{"cmd":"show_task_center"}]"#), (true, "Queued 2 commands".to_string()));
        assert_eq!(take_matching(|c| matches!(c, AppCommand::ClearSelection | AppCommand::ShowTaskCenter)).len(), 3);
        let (ok, message) = run_text("{not json");
        assert!(!ok && message.starts_with("bad command"));
        let (ok, _) = run_text("nonsense");
        assert!(!ok);
    }

    #[test]
    fn the_window_shows_logged_commands_and_their_results() {
        let _serial = serial();
        let ctx = crate::ui::test_support::test_ctx();
        command::with_log(|log| {
            log.clear();
            log.push(r#"{"cmd":"clear_selection"}"#.into(), Outcome::Ok("cleared 2 selected".into()));
            log.push(r#"{"cmd":"focus_path","path":"/x"}"#.into(), Outcome::Err(CommandError::not_found("no such row")));
        });
        set_open(&ctx, true);
        let mut texts = Vec::new();
        for _ in 0..3 {
            let out = ctx.run_ui(Default::default(), |ui| show(ui.ctx()));
            texts = crate::ui::test_support::texts(&out.shapes).into_iter().map(|(t, _)| t).collect();
            out.drop_without_applying_deltas();
        }
        assert!(texts.iter().any(|t| t.contains("clear_selection")));
        assert!(texts.iter().any(|t| t == "cleared 2 selected"));
        assert!(texts.iter().any(|t| t == "not found: no such row"));
        command::with_log(|log| log.clear());
    }
}
