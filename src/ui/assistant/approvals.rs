//! The approval card: a tool the agent wants to run, the arguments, a countdown and Approve, Always and Deny.

use egui::{Frame, Label, Margin, RichText, Ui};
use serde_json::Value;
use zeroclaw_client::ApprovalDecision;

use super::entry::{Decision, tool_name};
use crate::icons::{self, svg};

/// An icon followed by text that wraps to the width that is left.
pub(super) fn wrap_label(ui: &mut Ui, icon: icons::Icon, text: RichText) {
    ui.horizontal(|ui| {
        let edge = icons::IconSize::Md.px(ui);
        ui.add(icon.image(edge).tint(ui.visuals().weak_text_color()));
        ui.add(Label::new(text).wrap());
    });
}

/// `4:32` for the time left, `0:00` once it is over.
pub fn countdown(now_ms: u64, deadline_ms: u64) -> String {
    let secs = deadline_ms.saturating_sub(now_ms).div_ceil(1000);
    format!("{}:{:02}", secs / 60, secs % 60)
}

/// The arguments as indented JSON, cut at `max` characters.
pub fn pretty_args(args: &Value, max: usize) -> String {
    let text = match args {
        Value::Null => String::new(),
        other => serde_json::to_string_pretty(other).unwrap_or_default(),
    };
    if text.chars().count() > max { text.chars().take(max).collect::<String>() + "\n..." } else { text }
}

pub fn decision_to_wire(d: Decision) -> Option<ApprovalDecision> {
    match d {
        Decision::Approved => Some(ApprovalDecision::Approve),
        Decision::Always => Some(ApprovalDecision::Always),
        Decision::Denied => Some(ApprovalDecision::Deny),
        Decision::Pending | Decision::Expired => None,
    }
}

/// Draws the card; returns the answer when a button was pressed.
pub fn card(ui: &mut Ui, key: &str, tool: &str, summary: &str, args: &Value, deadline_ms: u64, decision: Decision, now_ms: u64) -> Option<Decision> {
    let mut answer = None;
    let stroke = if decision == Decision::Pending { ui.visuals().warn_fg_color } else { ui.visuals().widgets.noninteractive.bg_stroke.color };
    Frame::new().stroke(egui::Stroke::new(1.0, stroke)).corner_radius(6).inner_margin(Margin::same(8)).show(ui, |ui| {
        ui.set_width(ui.available_width());
        wrap_label(ui, svg::SHIELD_CHECK, RichText::new(format!("Allow {}?", tool_name(tool))).strong());
        if !summary.trim().is_empty() {
            ui.label(summary);
        }
        let pretty = pretty_args(args, 1500);
        if !pretty.is_empty() && pretty != "{}" {
            ui.collapsing(RichText::new("Arguments").weak(), |ui| {
                ui.label(RichText::new(pretty).monospace().small());
            });
        }
        match decision {
            Decision::Pending => {
                ui.label(RichText::new(format!("Denied automatically in {}", countdown(now_ms, deadline_ms))).weak());
                ui.horizontal_wrapped(|ui| {
                    ui.push_id(key, |ui| {
                        if icons::button(ui, svg::CHECK, "Approve").clicked() {
                            answer = Some(Decision::Approved);
                        }
                        if icons::button(ui, svg::CHECKS, "Always").on_hover_text("Approve and stop asking for this tool").clicked() {
                            answer = Some(Decision::Always);
                        }
                        if icons::button(ui, svg::X_CIRCLE, "Deny").clicked() {
                            answer = Some(Decision::Denied);
                        }
                    });
                });
            }
            other => {
                ui.label(RichText::new(format!("Answer: {}", other.label())).weak());
            }
        }
    });
    answer
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_countdown_reads_minutes_and_seconds() {
        assert_eq!(countdown(0, 300_000), "5:00");
        assert_eq!(countdown(1_000, 300_000), "4:59");
        assert_eq!(countdown(0, 59_500), "1:00", "a part second counts as a whole one");
        assert_eq!(countdown(400_000, 300_000), "0:00");
    }

    #[test]
    fn arguments_are_pretty_and_bounded() {
        assert_eq!(pretty_args(&Value::Null, 100), "");
        assert!(pretty_args(&json!({"a":1}), 100).contains("\"a\": 1"));
        let big = json!({"text": "x".repeat(500)});
        let cut = pretty_args(&big, 50);
        assert!(cut.ends_with("...") && cut.chars().count() < 60);
    }

    #[test]
    fn answers_map_to_wire_decisions() {
        assert_eq!(decision_to_wire(Decision::Approved), Some(ApprovalDecision::Approve));
        assert_eq!(decision_to_wire(Decision::Always), Some(ApprovalDecision::Always));
        assert_eq!(decision_to_wire(Decision::Denied), Some(ApprovalDecision::Deny));
        assert_eq!(decision_to_wire(Decision::Pending), None);
        assert_eq!(decision_to_wire(Decision::Expired), None);
    }
}
