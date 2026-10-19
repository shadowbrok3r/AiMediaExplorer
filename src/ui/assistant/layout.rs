//! Drawing the assistant: the rail layout and the pop-out window layout around the same body.

use egui::{CentralPanel, Context, Frame, Margin, Panel, RichText, ScrollArea, Ui};

use super::AssistantPanel;
use super::backend::Resolved;
use super::composer::{self, Act as ComposerAct};
use super::context;
use super::header::{self, Act as HeaderAct};
use super::providers;
use super::transcript::{self, Action as TranscriptAction, usage_ring};
use crate::command::{AppCommand, Section};
use crate::icons::{self, svg};
use crate::ui::file_table::FileExplorer;
use crate::zc::card::{self, Mode};
use crate::zc::events::now_ms;

/// What the layout wants the host to do after drawing.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct PanelOut {
    pub pop_out: bool,
    pub dock: bool,
}

fn open_settings() {
    crate::command::send(AppCommand::ShowSettings { section: Some(Section::Assistant) });
}

impl AssistantPanel {
    /// The rail layout: header, transcript, composer.
    pub fn ui_rail(&mut self, ui: &mut Ui, ex: &mut FileExplorer, popped_out: bool) -> PanelOut {
        self.body(ui, ex, popped_out, false)
    }

    /// The window layout: the session list beside the same body.
    pub fn ui(&mut self, ui: &mut Ui, ex: &mut FileExplorer) -> PanelOut {
        Panel::left("ame_assistant_sessions_panel").default_size(190.0).resizable(true).show(ui, |ui| self.session_list(ui));
        self.body(ui, ex, true, true)
    }

    fn session_list(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.heading("Chats");
            if icons::icon_button(ui, svg::AME_NEW_CHAT, "New chat").clicked() {
                self.new_chat();
            }
        });
        let mut switch = None;
        let mut ask = None;
        let mut confirm = None;
        ScrollArea::vertical().id_salt("ame_assistant_session_scroll").auto_shrink([false, false]).show(ui, |ui| {
            for (i, s) in self.sessions.list.iter().enumerate() {
                ui.horizontal(|ui| {
                    let legacy = if s.is_legacy() { " (old)" } else { "" };
                    if ui.selectable_label(self.sessions.active == Some(i), format!("{}{legacy}", s.title)).clicked() {
                        switch = Some(i);
                    }
                    if self.sessions.confirm_delete == Some(i) {
                        if ui.small_button("Delete").clicked() {
                            confirm = Some(i);
                        }
                    } else if icons::icon_button(ui, svg::TRASH, "Delete this chat (the gateway keeps its own copy)").clicked() {
                        ask = Some(i);
                    }
                });
            }
        });
        if let Some(i) = switch {
            self.switch_to(i);
        }
        if let Some(i) = ask {
            self.sessions.confirm_delete = Some(i);
        }
        if let Some(i) = confirm {
            self.delete_session(i);
        }
    }

    fn blocked_reason(&self) -> Option<&'static str> {
        match &self.resolved {
            Resolved::Ready(_) => None,
            Resolved::NeedsSetup => Some("Not set up. Quick actions still work: type /help."),
            Resolved::Unavailable { .. } => Some("Chat is unavailable right now. Quick actions still work."),
        }
    }

    fn apply_header(&mut self, ctx: &Context, acts: Vec<HeaderAct>, out: &mut PanelOut) {
        for act in acts {
            match act {
                HeaderAct::NewChat => self.new_chat(),
                HeaderAct::Switch(i) => self.switch_to(i),
                HeaderAct::AskDelete(i) => self.sessions.confirm_delete = Some(i),
                HeaderAct::ConfirmDelete(i) => self.delete_session(i),
                HeaderAct::CancelDelete => self.sessions.confirm_delete = None,
                HeaderAct::SetBackend(pref) => {
                    let mut profile = self.rt.profile();
                    profile.backend = pref.as_str().to_string();
                    crate::zc::save_profile_with(&self.rt, profile);
                    self.connect_tried = None;
                }
                HeaderAct::OpenSettings | HeaderAct::Pair => open_settings(),
                HeaderAct::PopOut => out.pop_out = true,
                HeaderAct::Dock => out.dock = true,
                HeaderAct::Retry => self.retry_connection(),
                HeaderAct::StartTools => {
                    let mut profile = self.rt.profile();
                    profile.tools_enabled = true;
                    crate::zc::save_profile_with(&self.rt, profile.clone());
                    crate::zc::tools::ensure_running(&profile);
                }
                HeaderAct::BrowseModels => self.models.show_and_load(&self.settings),
                HeaderAct::SetProvider(p) => {
                    crate::settings::update("assistant:provider", |s| s.ai_chat_provider = Some(p));
                }
                HeaderAct::SetModel(m) => {
                    self.model_override = Some(m.clone());
                    crate::settings::update("assistant:model", |s| s.openai_default_model = Some(m));
                }
                HeaderAct::ToolsDebug => self.tools_dbg.open = true,
            }
        }
        ctx.request_repaint();
    }

    fn apply_transcript(&mut self, actions: Vec<TranscriptAction>) {
        for action in actions {
            match action {
                TranscriptAction::Decide { request_id, decision } => self.decide(&request_id, decision),
                TranscriptAction::Notice(a) => match a {
                    super::entry::NoticeAction::RePair | super::entry::NoticeAction::OpenSettings => open_settings(),
                    super::entry::NoticeAction::Retry => self.retry_connection(),
                    super::entry::NoticeAction::ReloadHistory => {
                        self.rt.load_history(50, false);
                    }
                },
                TranscriptAction::LoadEarlier => {
                    self.rt.load_history(50, true);
                }
                TranscriptAction::Thumb(path) => self.view.thumbs.request(&path),
            }
        }
    }

    fn usage_row(&self, ui: &mut Ui) {
        let Some(usage) = &self.transcript.usage else { return };
        let fraction = usage.context_fraction();
        let tip = match (usage.last_input_tokens.or(usage.input_tokens), usage.context_window) {
            (Some(used), Some(window)) => format!("{used} of {window} tokens of context in use"),
            (Some(used), None) => format!("{used} tokens of context in use"),
            _ => "Context use is not reported".to_string(),
        };
        ui.horizontal_wrapped(|ui| {
            usage_ring(ui, fraction, &tip);
            if let Some(f) = fraction {
                ui.label(RichText::new(format!("{:.0}%", f * 100.0)).weak().small());
            }
            if let Some(model) = usage.model.as_deref() {
                ui.label(RichText::new(model).weak().small());
            }
            if let Some(cost) = usage.cost_usd.filter(|c| *c > 0.0) {
                ui.label(RichText::new(format!("${cost:.4}")).weak().small());
            }
        });
    }

    fn body(&mut self, ui: &mut Ui, ex: &mut FileExplorer, popped_out: bool, list_beside: bool) -> PanelOut {
        let mut out = PanelOut::default();
        let ctx = ui.ctx().clone();
        self.ai_update_tx = Some(ex.viewer.ai_update_tx.clone());
        self.last_drawn = Some(std::time::Instant::now());
        if self.sessions.loaded && self.sessions.active.is_none() && self.sessions.list.is_empty() && !self.sessions.creating {
            self.new_chat();
        }
        self.input = context::gather(ex);
        if std::mem::take(&mut self.pending_send) {
            self.send_current(ex);
        }
        let tools_offline = self.profile.tools_enabled && !crate::zc::tools::status().running;
        let acts = {
            let env = header::Env { resolved: &self.resolved, profile: &self.profile, health: &self.health, link: self.rt.link().state, sessions: &self.sessions, direct: &self.direct, settings: &self.settings, tools_offline, popped_out, list_beside };
            Panel::top("ame_assistant_header").frame(Frame::NONE.inner_margin(Margin { bottom: 4, ..Default::default() })).show(ui, |ui| header::show(ui, &env)).inner
        };
        self.apply_header(&ctx, acts, &mut out);

        let legacy = self.sessions.active_info().is_some_and(|s| s.is_legacy());
        let blocked = self.blocked_reason();
        let mut composer_act = None;
        let mut want_new_chat = false;
        Panel::bottom("ame_assistant_composer_panel").frame(Frame::NONE.inner_margin(Margin { top: 6, ..Default::default() })).show(ui, |ui| {
            self.usage_row(ui);
            if legacy {
                ui.horizontal_wrapped(|ui| {
                    ui.label(RichText::new("This chat is from an older version and is read-only.").color(ui.visuals().warn_fg_color));
                    if icons::small_button(ui, svg::AME_NEW_CHAT, "Continue in a new chat").clicked() {
                        want_new_chat = true;
                    }
                });
            }
            let env = composer::Env { running: self.transcript.running, blocked, input: &self.input };
            composer_act = composer::show(ui, &mut self.composer, &env, &mut self.view.thumbs);
        });
        if want_new_chat {
            self.continue_in_new_chat();
        }
        match composer_act {
            Some(ComposerAct::Send) => self.send_current(ex),
            Some(ComposerAct::Stop) => self.stop_current(),
            Some(ComposerAct::AttachSelection) => {
                ex.sync_view();
                let paths: Vec<String> = ex.selected_rows().iter().filter(|r| r.file_type != "<DIR>").map(|r| r.path.clone()).collect();
                for p in paths {
                    self.composer.add_attachment(p);
                }
            }
            None => {}
        }

        let auto_down = self.profile.backend_pref() == crate::zc::profile::BackendPref::Auto && matches!(self.resolved, Resolved::Unavailable { .. });
        let setup = (matches!(self.resolved, Resolved::NeedsSetup) || auto_down) && !self.setup_skipped && self.transcript.entries.is_empty();
        let mut actions = Vec::new();
        let mut skipped = false;
        CentralPanel::default().frame(Frame::NONE).show(ui, |ui| {
            if setup {
                ScrollArea::vertical().id_salt("ame_assistant_setup").auto_shrink([false, false]).show(ui, |ui| {
                    ui.label(RichText::new("Connect the assistant").heading());
                    ui.label(RichText::new("The assistant runs on a ZeroClaw gateway. Pair with one, or import it from ZC Codex.").weak());
                    ui.add_space(6.0);
                    skipped = card::show(ui, &self.rt, Mode::Wizard);
                });
            } else {
                actions = transcript::show(ui, &self.transcript, &mut self.view, now_ms());
            }
        });
        if skipped && !self.health.is_online() {
            self.setup_skipped = true;
        }
        self.apply_transcript(actions);

        if let Some(model) = providers::model_browser(&ctx, &mut self.models, &self.settings) {
            self.model_override = Some(model);
            self.new_chat();
        }
        self.tools_dbg.show(&ctx);
        out
    }
}
