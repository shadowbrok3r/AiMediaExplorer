//! A small tool tester for the assistant: pick a tool of the MCP server, edit arguments generated from its schema, run it in this process.

use std::sync::{Arc, Mutex};

use egui::{ComboBox, Context, RichText, ScrollArea, TextEdit, Window};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::ai::mcp::hub;
use crate::ui::devtools::mcp::skeleton;

/// Longest result kept for display.
const RESULT_CHARS: usize = 8000;

#[derive(Default)]
pub struct ToolsDebug {
    pub open: bool,
    tool: usize,
    args: String,
    args_for: Option<usize>,
    running: Option<CancellationToken>,
    result: Arc<Mutex<Option<(bool, String)>>>,
}

/// Runs `name` with the JSON text `args` and returns the result text, or why it failed.
pub async fn run_tool(name: &str, args: &str, cancel: CancellationToken) -> Result<String, String> {
    let args: Value = serde_json::from_str(args).map_err(|e| format!("The arguments are not valid JSON: {e}"))?;
    let started = std::time::Instant::now();
    let out = hub().call_local(name, args, cancel).await.map_err(|e| e.to_string())?;
    let mut text = format!("{name} ok in {} ms\n{}", started.elapsed().as_millis(), out.text);
    if !out.data.is_null() {
        text.push('\n');
        text.push_str(&serde_json::to_string_pretty(&out.data).unwrap_or_default());
    }
    if text.chars().count() > RESULT_CHARS {
        text = text.chars().take(RESULT_CHARS).collect::<String>() + "\n...";
    }
    Ok(text)
}

impl ToolsDebug {
    pub fn show(&mut self, ctx: &Context) {
        if !self.open {
            return;
        }
        let catalog = hub().registry.catalog();
        if self.tool >= catalog.len() {
            self.tool = 0;
        }
        if self.args_for != Some(self.tool) {
            self.args = catalog.get(self.tool).map(|r| serde_json::to_string_pretty(&skeleton(&r.input_schema)).unwrap_or_default()).unwrap_or_default();
            self.args_for = Some(self.tool);
        }
        let mut open = self.open;
        Window::new("Tools tester").open(&mut open).default_width(420.0).show(ctx, |ui| {
            ui.label(RichText::new("Runs a tool of the application's MCP server in this process, as the agent would.").weak());
            let current = catalog.get(self.tool).map(|r| r.name.clone()).unwrap_or_default();
            ComboBox::from_id_salt("ame_assistant_tool_pick").selected_text(current).show_ui(ui, |ui| {
                for (i, row) in catalog.iter().enumerate() {
                    ui.selectable_value(&mut self.tool, i, format!("{} ({})", row.name, row.class.label()));
                }
            });
            if let Some(row) = catalog.get(self.tool) {
                ui.label(RichText::new(&row.description).weak().small());
            }
            ui.add(TextEdit::multiline(&mut self.args).code_editor().desired_rows(5).desired_width(ui.available_width()));
            ui.horizontal(|ui| {
                if self.running.is_none() {
                    if ui.button("Run").clicked() {
                        let name = catalog.get(self.tool).map(|r| r.name.clone()).unwrap_or_default();
                        let (args, cancel, slot, ctx) = (self.args.clone(), CancellationToken::new(), self.result.clone(), ui.ctx().clone());
                        self.running = Some(cancel.clone());
                        tokio::spawn(async move {
                            let out = run_tool(&name, &args, cancel).await;
                            *slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(match out {
                                Ok(t) => (true, t),
                                Err(e) => (false, e),
                            });
                            ctx.request_repaint();
                        });
                    }
                } else if ui.button("Cancel").clicked() {
                    if let Some(c) = &self.running {
                        c.cancel();
                    }
                }
            });
            if self.running.is_some() && self.result.lock().unwrap_or_else(|e| e.into_inner()).is_some() {
                self.running = None;
            }
            if let Some((ok, text)) = self.result.lock().unwrap_or_else(|e| e.into_inner()).clone() {
                let color = if ok { ui.visuals().text_color() } else { ui.visuals().error_fg_color };
                ScrollArea::vertical().max_height(240.0).show(ui, |ui| {
                    ui.label(RichText::new(text).monospace().small().color(color));
                });
            }
        });
        self.open = open;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_tool_runs_and_reports_its_text() {
        let text = run_tool("util_ping", r#"{"message":"hello"}"#, CancellationToken::new()).await.unwrap();
        assert!(text.starts_with("util_ping ok in"), "{text}");
        assert!(text.contains("hello"), "{text}");
    }

    #[tokio::test]
    async fn bad_json_and_unknown_tools_are_errors() {
        assert!(run_tool("util_ping", "{nope", CancellationToken::new()).await.unwrap_err().contains("not valid JSON"));
        assert!(run_tool("no_such_tool", "{}", CancellationToken::new()).await.is_err());
    }

    #[test]
    fn the_window_lists_the_catalog() {
        let ctx = crate::ui::test_support::test_ctx();
        let mut dbg = ToolsDebug { open: true, ..Default::default() };
        let mut drawn = Vec::new();
        for _ in 0..3 {
            let out = ctx.run_ui(egui::RawInput::default(), |ui| dbg.show(&ui.ctx().clone()));
            drawn = crate::ui::test_support::texts(&out.shapes);
            out.drop_without_applying_deltas();
        }
        assert!(drawn.iter().any(|(t, _)| t.contains("Tools tester")), "{drawn:?}");
        assert!(dbg.args.starts_with('{'));
    }
}
