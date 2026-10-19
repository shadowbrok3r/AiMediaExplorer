//! The message box: context chips, attachments, the text field with its slash-command hints, and the Send and Stop buttons.

use egui::{Id, Key, Layout, RichText, TextEdit, Ui, Widget};

use super::context::{ChipKind, ContextInput};
use super::local::SLASH_COMMANDS;
use super::transcript::Thumbs;
use crate::icons::{self, svg};
use crate::ui::dialogs::{self, DialogResult};

#[derive(Default)]
pub struct Composer {
    pub text: String,
    /// Chips that are on.
    pub chips: Vec<ChipKind>,
    pub attachments: Vec<String>,
}

/// What the person asked for this frame.
#[derive(Debug, PartialEq, Eq)]
pub enum Act {
    Send,
    Stop,
    AttachSelection,
}

/// What the box needs to know to draw itself.
pub struct Env<'a> {
    /// A turn is running: Send becomes Stop.
    pub running: bool,
    /// Why sending is not possible now, when it is not.
    pub blocked: Option<&'a str>,
    pub input: &'a ContextInput,
}

impl Composer {
    pub fn toggle_chip(&mut self, kind: ChipKind) {
        match self.chips.iter().position(|k| *k == kind) {
            Some(i) => {
                self.chips.remove(i);
            }
            None => self.chips.push(kind),
        }
    }

    pub fn add_attachment(&mut self, path: String) {
        if !self.attachments.contains(&path) {
            self.attachments.push(path);
        }
    }

    pub fn can_send(&self) -> bool {
        !self.text.trim().is_empty() || !self.attachments.is_empty()
    }

    /// Takes the draft for sending: the text and the attachments; the chips stay on.
    pub fn take(&mut self) -> (String, Vec<String>) {
        (std::mem::take(&mut self.text), std::mem::take(&mut self.attachments))
    }
}

fn focus_id() -> Id {
    Id::unique("ame_assistant_composer_focus")
}

/// Asks the composer to take the keyboard focus on its next draw.
pub fn request_focus(ctx: &egui::Context) {
    ctx.data_mut(|d| d.insert_temp(focus_id(), true));
    ctx.request_repaint();
}

/// Removes a plain Enter key press from this frame's input so the text field does not add a line; true when there was one.
fn take_enter(ui: &Ui) -> bool {
    ui.input_mut(|i| {
        let is_enter = |e: &egui::Event| matches!(e, egui::Event::Key { key: Key::Enter, pressed: true, modifiers, .. } if !modifiers.shift && !modifiers.alt);
        let hit = i.events.iter().any(is_enter);
        if hit {
            i.events.retain(|e| !is_enter(e));
        }
        hit
    })
}

/// Commands whose name starts with what was typed after the slash.
pub fn slash_suggestions(text: &str) -> Vec<(&'static str, &'static str)> {
    let t = text.trim_start();
    if !t.starts_with('/') || t.contains(char::is_whitespace) {
        return Vec::new();
    }
    let typed = t.trim_start_matches('/').to_lowercase();
    SLASH_COMMANDS.iter().filter(|(c, _)| c.trim_start_matches('/').starts_with(&typed)).copied().collect()
}

fn chip_row(ui: &mut Ui, c: &mut Composer, input: &ContextInput) {
    ui.horizontal_wrapped(|ui| {
        for kind in ChipKind::ALL {
            let on = c.chips.contains(&kind);
            let n = input.count(kind);
            let label = if n > 0 { format!("{} ({n})", kind.label()) } else { kind.label().to_string() };
            if ui.selectable_label(on, label).on_hover_text(kind.hint()).clicked() {
                c.toggle_chip(kind);
            }
        }
    });
}

fn attachment_row(ui: &mut Ui, c: &mut Composer, thumbs: &mut Thumbs) {
    if c.attachments.is_empty() {
        return;
    }
    let mut remove = None;
    ui.horizontal_wrapped(|ui| {
        for (i, p) in c.attachments.iter().enumerate() {
            match thumbs.get(p) {
                Some(tex) => {
                    ui.image((tex.id(), egui::vec2(20.0, 20.0)));
                }
                None => thumbs.request(p),
            }
            let name = std::path::Path::new(p).file_name().and_then(|s| s.to_str()).unwrap_or(p);
            ui.label(RichText::new(name).monospace().small()).on_hover_text(p);
            if icons::icon_button(ui, svg::X, "Remove attachment").clicked() {
                remove = Some(i);
            }
        }
    });
    if let Some(i) = remove {
        c.attachments.remove(i);
    }
}

/// Draws the box. The text field keeps the focus after a send so typing can go on.
pub fn show(ui: &mut Ui, c: &mut Composer, env: &Env, thumbs: &mut Thumbs) -> Option<Act> {
    let mut act = None;
    if let Some(DialogResult::AssistantAttach(path)) = dialogs::take(|r| matches!(r, DialogResult::AssistantAttach(_)).then(|| r.clone())) {
        c.add_attachment(path.display().to_string());
    }
    chip_row(ui, c, env.input);
    attachment_row(ui, c, thumbs);
    if let Some(why) = env.blocked {
        ui.label(RichText::new(why).color(ui.visuals().warn_fg_color));
    }
    let hint = if env.blocked.is_some() { "Quick actions still work: type /help" } else { "Message. Enter sends, Shift+Enter adds a line, / lists quick actions" };
    let id = Id::unique("ame_assistant_composer");
    let enter = ui.memory(|m| m.has_focus(id)) && take_enter(ui);
    let edit = TextEdit::multiline(&mut c.text).id(id).desired_rows(3).desired_width(ui.available_width()).hint_text(hint).ui(ui);
    if ui.ctx().data_mut(|d| d.remove_temp::<bool>(focus_id())).unwrap_or(false) {
        edit.request_focus();
    }
    for (cmd, desc) in slash_suggestions(&c.text) {
        let name = cmd.split_whitespace().next().unwrap_or(cmd);
        if ui.small_button(format!("{cmd}")).on_hover_text(desc).clicked() {
            c.text = format!("{name} ");
            edit.request_focus();
        }
    }
    ui.horizontal_wrapped(|ui| {
        icons::icon_menu_button(ui, svg::PLUS, "Attach", |ui| {
            if icons::button(ui, svg::CURSOR_CLICK, "Attach current selection").clicked() {
                act = Some(Act::AttachSelection);
                ui.close_kind(egui::UiKind::Menu);
            }
            if icons::button(ui, svg::PAPERCLIP, "Attach file...").clicked() {
                let ctx = ui.ctx().clone();
                dialogs::pick_file(&ctx, "Attach image or file", {
                    let ctx = ctx.clone();
                    move |file| dialogs::deliver(&ctx, DialogResult::AssistantAttach(file))
                });
                ui.close_kind(egui::UiKind::Menu);
            }
        });
        ui.with_layout(Layout::right_to_left(egui::Align::Center), |ui| {
            if env.running {
                if icons::button(ui, svg::STOP, "Stop").clicked() {
                    act = Some(Act::Stop);
                }
            } else {
                let can = c.can_send() && (env.blocked.is_none() || c.text.trim_start().starts_with('/'));
                if ui.add_enabled(can, icons::button_widget(ui, svg::PAPER_PLANE_RIGHT, "Send")).clicked() {
                    act = Some(Act::Send);
                }
            }
        });
    });
    if enter && !env.running && c.can_send() {
        act = Some(Act::Send);
    }
    act
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chips_toggle_and_attachments_do_not_repeat() {
        let mut c = Composer::default();
        c.toggle_chip(ChipKind::Selection);
        c.toggle_chip(ChipKind::Folder);
        c.toggle_chip(ChipKind::Selection);
        assert_eq!(c.chips, [ChipKind::Folder]);
        c.add_attachment("/a.jpg".into());
        c.add_attachment("/a.jpg".into());
        assert_eq!(c.attachments.len(), 1);
    }

    #[test]
    fn a_draft_needs_text_or_an_attachment() {
        let mut c = Composer::default();
        assert!(!c.can_send());
        c.text = "  ".into();
        assert!(!c.can_send());
        c.add_attachment("/a.jpg".into());
        assert!(c.can_send());
        c.text = "describe".into();
        let (text, attachments) = c.take();
        assert_eq!((text.as_str(), attachments.len()), ("describe", 1));
        assert!(c.text.is_empty() && c.attachments.is_empty());
    }

    #[test]
    fn taking_the_draft_keeps_the_chips_on() {
        let mut c = Composer::default();
        c.toggle_chip(ChipKind::Filters);
        c.text = "x".into();
        c.take();
        assert_eq!(c.chips, [ChipKind::Filters]);
    }

    #[test]
    fn slash_hints_follow_what_is_typed() {
        assert!(slash_suggestions("hello").is_empty());
        assert_eq!(slash_suggestions("/").len(), SLASH_COMMANDS.len());
        let s = slash_suggestions("/se");
        assert_eq!(s.len(), 1);
        assert!(s[0].0.starts_with("/search"));
        assert!(slash_suggestions("/ping text").is_empty(), "no hints once an argument is being typed");
        assert!(slash_suggestions("/zzz").is_empty());
    }
}
