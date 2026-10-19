//! The tools-server and privacy sections of the connection card.

use egui::{CollapsingHeader, DragValue, RichText, TextEdit, Ui};

use super::card::Card;
use super::profile::{LOOPBACK, ZcProfile};
use super::{tools, widgets};
use crate::icons::{self, svg};

/// Edits of the tools section that are not part of the profile.
#[derive(Default)]
pub struct ToolsDraft {
    /// The tailnet checkbox; `None` follows the profile.
    pub expose: Option<bool>,
    pub confirm_rotate: bool,
    pub note: Option<(bool, String)>,
}

fn note_from(draft: &mut ToolsDraft, result: crate::command::CommandResult) {
    draft.note = Some(match result {
        Ok(text) => (true, text),
        Err(e) => (false, e.message),
    });
}

pub(super) fn tools_section(ui: &mut Ui, card: &mut Card, draft: &mut ZcProfile) {
    CollapsingHeader::new(RichText::new("Tools server").strong()).default_open(true).show(ui, |ui| {
        ui.label(RichText::new("Lets the agent on the gateway read your library. Writes always wait for your review.").weak());
        let status = tools::status();
        ui.horizontal(|ui| {
            let (kind, color) = match status.state {
                "running" => ("Running", egui::Color32::from_rgb(88, 196, 120)),
                "failed" => ("Failed", ui.visuals().error_fg_color),
                "starting" => ("Starting", ui.visuals().warn_fg_color),
                _ => ("Stopped", ui.visuals().weak_text_color()),
            };
            widgets::dot(ui, color);
            ui.label(RichText::new(kind).strong());
            ui.label(RichText::new(format!("{} clients", status.clients)).weak());
        });
        ui.label(RichText::new(&status.detail).weak());
        ui.horizontal_wrapped(|ui| {
            ui.label("URL");
            ui.monospace(&status.url);
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("Token");
            ui.monospace(tools::masked_token().unwrap_or_else(|| "none".to_string()));
            if card.tools.confirm_rotate {
                if ui.button("Confirm new token").clicked() {
                    note_from(&mut card.tools, tools::rotate_token());
                    card.tools.confirm_rotate = false;
                }
                if ui.button("Keep").clicked() {
                    card.tools.confirm_rotate = false;
                }
            } else if ui.button("Rotate").on_hover_text("Clients need the new token").clicked() {
                card.tools.confirm_rotate = true;
            }
        });
        ui.checkbox(&mut draft.tools_enabled, "Serve tools to the gateway (start with the application)");
        ui.horizontal_wrapped(|ui| {
            ui.label("Port");
            ui.add(DragValue::new(&mut draft.tools_port).range(1..=65535));
        });
        let mut expose = card.tools.expose.unwrap_or_else(|| draft.tools_on_tailnet());
        if ui.checkbox(&mut expose, "Expose on the tailnet").on_hover_text("Binds one interface address; loopback otherwise").changed() {
            card.tools.expose = Some(expose);
            draft.tools_bind = if expose { String::new() } else { LOOPBACK.to_string() };
        }
        if expose {
            ui.add(TextEdit::singleline(&mut draft.tools_bind).hint_text("interface address, e.g. from tailscale ip -4").desired_width(ui.available_width()));
        }
        ui.horizontal_wrapped(|ui| {
            let running = status.running;
            if ui.add_enabled(!running, icons::button_widget(ui, svg::PLAY, "Start")).clicked() {
                if let Err(e) = tools::apply_profile(draft) {
                    card.tools.note = Some((false, e));
                } else {
                    note_from(&mut card.tools, tools::start());
                }
            }
            if ui.add_enabled(running, icons::button_widget(ui, svg::STOP, "Stop")).clicked() {
                note_from(&mut card.tools, tools::stop());
            }
            if icons::button(ui, svg::COPY, "Copy gateway config").on_hover_text("Configuration for the gateway with a placeholder for the token").clicked() {
                ui.ctx().copy_text(tools::gateway_snippet());
                card.tools.note = Some((true, "Copied. Paste it into the gateway configuration and fill in the token.".to_string()));
            }
        });
        if let Some((ok, text)) = &card.tools.note {
            let color = if *ok { ui.visuals().text_color() } else { ui.visuals().error_fg_color };
            ui.label(RichText::new(text).color(color));
        }
    });
}

pub(super) fn privacy_section(ui: &mut Ui, card: &mut Card, draft: &mut ZcProfile) {
    CollapsingHeader::new(RichText::new("Privacy").strong()).default_open(true).show(ui, |ui| {
        ui.label(RichText::new("Selections, folders and filters go to the agent as text. Pictures are sent only when you attach one and this is on.").weak());
        let local = draft.is_local_origin();
        let mut allow = draft.allow_images;
        if ui.checkbox(&mut allow, "Allow images in chat messages").changed() {
            draft.allow_images = allow && (local || card.images_confirmed);
        }
        if !local {
            ui.label(RichText::new(format!("{} is outside your network: attached pictures would leave it.", draft.host_label())).color(ui.visuals().warn_fg_color));
            if ui.checkbox(&mut card.images_confirmed, "I understand, allow pictures anyway").changed() && !card.images_confirmed {
                draft.allow_images = false;
            }
        }
    });
}
