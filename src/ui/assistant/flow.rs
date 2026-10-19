//! What the assistant does: sessions, sending, stopping, approvals, titles, and applying the events of the backends to the transcript.

use std::time::Instant;

use egui::Context;
use surrealdb::types::RecordId;
use zeroclaw_client::CompleteRequest;

use super::AssistantPanel;
use super::STOP_GRACE;
use super::backend::{AssistantBackend, BackendKind, Resolved, TurnRequest};
use super::context::{self, ChipKind};
use super::entry::{Decision, Level, NoticeAction, entry_from_row};
use super::images;
use super::local;
use super::sessions::{Loaded, NEW_TITLE, SessionInfo, clean_title, title_from_text};
use crate::database::assistant_store::NewRow;
use crate::status::{GlobalStatusIndicator, StatusState};
use crate::ui::file_table::FileExplorer;
use crate::zc::events::{ChatEv, Tagged, now_ms};
use crate::zc::health::HealthKind;

/// Messages of the stored transcript asked for at once.
const HISTORY_PAGE: usize = 50;
/// Characters of earlier conversation sent to a provider that keeps none.
const HISTORY_CHARS: usize = 24_000;

impl AssistantPanel {

    pub(super) fn take_mailboxes(&mut self) {
        if !self.mailboxes {
            return;
        }
        for path in super::api::take_attachments() {
            self.composer.add_attachment(path);
        }
        for text in super::api::take_submits() {
            if !self.send_text(text.clone(), Vec::new(), Vec::new()) && self.composer.text.trim().is_empty() {
                self.composer.text = text;
            }
        }
    }

    /// Adds a session to the list and makes it the open one.
    pub(crate) fn adopt_session(&mut self, info: SessionInfo) {
        self.sessions.list.insert(0, info);
        self.sessions.active = Some(0);
        self.begin_session();
    }

    /// Clears what belongs to the session that was open and starts listening for the new one.
    fn begin_session(&mut self) {
        self.deny_pending();
        self.transcript.clear();
        self.view.expanded.clear();
        self.queued = None;
        self.running_on = None;
        self.stopping_since = None;
        self.local_epoch += 1;
        self.epoch = 0;
        self.connect_tried = None;
        if self.rt.live_session().is_some() {
            self.rt.disconnect();
        }
    }

    pub(super) fn sessions_step(&mut self) {
        if !self.sessions.listed && crate::database::is_ready() {
            self.sessions.request_list();
        }
        for loaded in self.sessions.poll() {
            match loaded {
                Loaded::List(list) => {
                    let open = self.sessions.active_info().cloned();
                    self.sessions.list = list;
                    match open {
                        Some(info) => {
                            let at = match self.sessions.find(&info.key) {
                                Some(i) => i,
                                None => {
                                    let rows = self.transcript.entries.iter().filter_map(super::entry::to_row).collect();
                                    self.sessions.save_new(&info, &self.profile.host_label(), rows);
                                    self.sessions.list.insert(0, info);
                                    0
                                }
                            };
                            self.sessions.active = Some(at);
                        }
                        None if self.sessions.list.is_empty() => {}
                        None => self.switch_to(0),
                    }
                }
                Loaded::Created(info) => {
                    self.adopt_session(info);
                    if std::mem::take(&mut self.send_when_created) {
                        self.pending_send = true;
                    }
                }
                Loaded::Messages { key, rows } => {
                    if self.sessions.active_info().is_some_and(|s| s.key == key) {
                        let scope = format!("{key}:c");
                        let mut loaded: Vec<_> = rows.iter().enumerate().map(|(i, r)| entry_from_row(&scope, i, r)).collect();
                        loaded.append(&mut self.transcript.entries);
                        self.transcript.entries = loaded;
                    }
                }
                Loaded::Failed(why) => {
                    log::warn!("[assistant] {why}");
                    self.transcript.notice(format!("sessions:{}", now_ms()), Level::Warn, why, None, now_ms());
                    self.sessions.listed = false;
                }
            }
        }
    }

    /// Starts a new chat for the backend that would answer it.
    pub fn new_chat(&mut self) {
        let backend = self.resolved.kind().map(|k| k.as_str()).unwrap_or("local");
        let agent = (self.resolved.kind() == Some(BackendKind::ZeroClaw)).then(|| self.profile.agent.trim().to_string());
        if crate::database::is_ready() {
            self.sessions.create(backend, agent, Some(self.profile.host_label()));
        } else {
            let key = surrealdb_types::Uuid::new_v4().to_string();
            self.adopt_session(SessionInfo { id: RecordId::new(crate::ASSISTANT_SESSIONS, key.clone()), key, title: NEW_TITLE.to_string(), backend: Some(backend.to_string()), agent });
        }
    }

    pub fn switch_to(&mut self, index: usize) {
        if index >= self.sessions.list.len() || self.sessions.active == Some(index) {
            return;
        }
        self.sessions.active = Some(index);
        self.begin_session();
        self.sessions.load_messages(index);
    }

    pub fn delete_session(&mut self, index: usize) {
        let was_active = self.sessions.active == Some(index);
        self.sessions.delete(index);
        if was_active {
            self.begin_session();
            match self.sessions.list.is_empty() {
                true => self.new_chat(),
                false => self.switch_to(0),
            }
        }
    }

    /// Opens the gateway socket of the open session when ZeroClaw is the backend and the gateway is reachable.
    pub(super) fn ensure_connected(&mut self) {
        if !matches!(self.resolved, Resolved::Ready(BackendKind::ZeroClaw)) {
            self.connect_tried = None;
            return;
        }
        if !self.is_on_screen() {
            return;
        }
        let Some(info) = self.sessions.active_info() else { return };
        if info.is_legacy() || self.rt.live_session().as_deref() == Some(info.key.as_str()) {
            return;
        }
        let tried = (info.key.clone(), self.health.kind());
        if self.connect_tried.as_ref() == Some(&tried) {
            return;
        }
        let (key, title) = (info.key.clone(), (!info.has_default_title()).then(|| info.title.clone()));
        self.connect_tried = Some(tried);
        self.epoch = self.rt.connect(&key, title.as_deref());
    }

    /// True while the panel is being drawn.
    pub(super) fn is_on_screen(&self) -> bool {
        self.last_drawn.is_some_and(|at| at.elapsed() < std::time::Duration::from_millis(1500))
    }

    /// Tries the connection again at the person's request.
    pub fn retry_connection(&mut self) {
        self.connect_tried = None;
        self.rt.check();
    }

    pub(super) fn drain_events(&mut self, now: u64) {
        for t in self.rt.drain(256) {
            self.on_event(t, true, now);
        }
        let local: Vec<Tagged> = self.local_rx.try_iter().take(256).collect();
        for t in local {
            self.on_event(t, false, now);
        }
    }

    fn on_event(&mut self, t: Tagged, gateway: bool, now: u64) {
        let expected = if gateway { self.epoch } else { self.local_epoch };
        if self.sessions.active_info().map(|s| s.key.as_str()) != Some(t.session.as_str()) || t.epoch != expected {
            self.stale_dropped += 1;
            return;
        }
        let scope = format!("{}:{}{}", t.session, if gateway { 'z' } else { 'l' }, t.epoch);
        match &t.ev {
            ChatEv::Connected { message_count, reconnect, .. } => {
                self.dispatch_queued();
                if *message_count > 0 && (!*reconnect || !self.transcript.running) {
                    self.rt.load_history(HISTORY_PAGE, false);
                }
            }
            ChatEv::Link { state: crate::zc::events::LinkState::Failed, detail } => {
                self.give_back_queued();
                if !self.rt.health().is_refused() {
                    self.transcript.notice(format!("{scope}:link"), Level::Error, format!("Could not connect to the gateway: {detail}"), Some(NoticeAction::Retry), now);
                }
            }
            _ => {}
        }
        let applied = self.transcript.apply(&scope, &t.ev, now);
        self.persist(applied.persist);
        if let Some(draft) = applied.restore_draft {
            if self.composer.text.trim().is_empty() {
                self.composer.text = draft;
            }
        }
        if applied.reload_history {
            self.rt.load_history(HISTORY_PAGE, false);
        }
        if applied.finished.is_some() {
            self.after_turn();
        }
        if !self.transcript.running {
            if self.running_on == Some(BackendKind::ZeroClaw) {
                crate::status::ASSIST_STATUS.set_state(StatusState::Idle, "Idle");
            }
            self.running_on = None;
            self.stopping_since = None;
        }
    }

    /// Applies an event as if the backend had sent it; used when a stop is not confirmed in time.
    fn apply_local(&mut self, ev: ChatEv, now: u64) {
        let key = self.sessions.active_info().map(|s| s.key.clone()).unwrap_or_default();
        let scope = format!("{key}:z{}", self.epoch);
        let applied = self.transcript.apply(&scope, &ev, now);
        self.persist(applied.persist);
        self.running_on = None;
        self.stopping_since = None;
    }

    pub(super) fn timers(&mut self, ctx: &Context, now: u64) {
        if self.transcript.expire_approvals(now) {
            ctx.request_repaint();
        }
        if self.stopping_since.is_some_and(|at| at.elapsed() > STOP_GRACE) && self.transcript.running {
            let turn = self.transcript.entries.iter().rev().find_map(|e| e.turn).unwrap_or(0);
            self.apply_local(ChatEv::Aborted { turn }, now);
        }
        if let Some((key, job)) = &mut self.title_job {
            if let Some(result) = job.take() {
                let key = key.clone();
                self.title_job = None;
                if let (Ok(reply), Some(index)) = (result, self.sessions.find(&key)) {
                    if let Some(title) = clean_title(&reply.response) {
                        self.sessions.rename(index, &title);
                        let session = key.clone();
                        let _ = self.rt.job(move |sh| async move {
                            let rest = sh.rest()?;
                            rest.rename_session(&session, &title).await.map_err(|e| sh.redact(&e.to_string()))
                        });
                    }
                }
            } else {
                ctx.request_repaint_after(std::time::Duration::from_millis(200));
            }
        }
        if self.transcript.running || self.transcript.pending_approvals() > 0 {
            ctx.request_repaint_after(std::time::Duration::from_millis(250));
        }
    }

    pub(super) fn persist(&self, rows: Vec<NewRow>) {
        if rows.is_empty() || !crate::database::is_ready() {
            return;
        }
        let Some(info) = self.sessions.active_info() else { return };
        let id = info.id.clone();
        tokio::spawn(async move {
            if let Err(e) = crate::database::assistant_store::append_rows(&id, rows).await {
                log::warn!("[assistant] the chat could not be saved: {e:#}");
            }
        });
    }

    /// Sends what is in the composer, reading the focused browser for the context chips.
    pub fn send_current(&mut self, ex: &mut FileExplorer) {
        self.ai_update_tx = Some(ex.viewer.ai_update_tx.clone());
        self.input = context::gather(ex);
        if !self.composer.can_send() {
            return;
        }
        let chips = self.composer.chips.clone();
        let (draft, attachments) = self.composer.take();
        if !self.send_text(draft.trim().to_string(), attachments.clone(), chips) {
            self.composer.text = draft;
            self.composer.attachments = attachments;
        }
    }

    /// Sends `text`; true when it was taken (sent, queued or answered as a quick action).
    pub(crate) fn send_text(&mut self, text: String, attachments: Vec<String>, chips: Vec<ChipKind>) -> bool {
        let now = now_ms();
        let scope = format!("assistant:{now}");
        if self.sessions.active_info().is_none() {
            if !self.sessions.creating {
                self.new_chat();
            }
            if self.sessions.active_info().is_none() {
                self.send_when_created = true;
                self.transcript.notice(format!("{scope}:wait"), Level::Info, "Starting a chat. Your message goes out as soon as it exists.", None, now);
                return false;
            }
        }
        if self.transcript.running {
            self.transcript.notice(format!("{scope}:busy"), Level::Info, "A reply is still running. Stop it first, or wait.", None, now);
            return false;
        }
        let slash = local::is_slash(&text);
        let kind = match (&self.resolved, slash) {
            (_, true) => BackendKind::Local,
            (Resolved::Ready(k), false) => *k,
            (Resolved::NeedsSetup, false) => {
                self.transcript.notice(format!("{scope}:setup"), Level::Warn, "The assistant is not set up. Connect a gateway above, or use quick actions (/help).", None, now);
                return false;
            }
            (Resolved::Unavailable { reason, .. }, false) => {
                let reason = reason.clone();
                self.transcript.notice(format!("{scope}:unavailable"), Level::Warn, format!("{reason}. The message was not sent."), None, now);
                return false;
            }
        };
        let info = self.sessions.active_info().cloned().unwrap();
        if info.is_legacy() && !slash {
            self.transcript.notice(format!("{scope}:legacy"), Level::Warn, "This chat is from an older version and is read-only. Start a new chat to continue.", None, now);
            return false;
        }
        let labels = if slash { Vec::new() } else { context::labels(&chips, &self.input) };
        let snapshot = if slash { String::new() } else { context::snapshot(&chips, &self.input) };
        let mut prompt = format!("{snapshot}{text}");
        let mut pictures = Vec::new();
        if kind == BackendKind::ZeroClaw && !attachments.is_empty() {
            let prepared = images::prepare(&attachments, self.rt.profile().allow_images);
            prompt.push_str(&images::skipped_line(&prepared.skipped));
            if let Some(note) = prepared.notice {
                self.transcript.notice(format!("{scope}:images"), Level::Info, note, None, now);
            }
            pictures = prepared.images;
        }
        self.local_epoch += 1;
        self.local_turn += 1;
        let history = if kind == BackendKind::Direct { self.transcript.conversation(HISTORY_CHARS) } else { Vec::new() };
        let ai_update_tx = self.ai_update_tx.clone().unwrap_or_else(|| crossbeam::channel::unbounded().0);
        let epoch = if kind == BackendKind::ZeroClaw { self.epoch } else { self.local_epoch };
        let request = TurnRequest {
            session: info.key.clone(),
            epoch,
            turn: self.local_turn,
            text: text.clone(),
            prompt,
            attachments: attachments.clone(),
            images: pictures,
            history,
            settings: self.settings.clone(),
            model: self.model_override.clone(),
            temperature: self.temp,
            ai_update_tx,
        };
        let key = format!("{}:u{now}", info.key);
        let row = self.transcript.push_user(key, &text, attachments, labels, now);
        self.persist(vec![row]);
        self.running_on = Some(kind);
        if kind == BackendKind::ZeroClaw && !slash {
            crate::status::ASSIST_STATUS.set_state(StatusState::Running, "Gateway reply");
        }
        match (kind, slash) {
            (_, true) => {
                let selected = self.input.selection.first().map(|f| f.path.clone()).unwrap_or_default();
                self.local_backend.slash(&request, selected);
            }
            (BackendKind::ZeroClaw, _) => {
                self.queued = Some(request);
                self.connect_tried = None;
                self.ensure_connected();
                self.dispatch_queued();
            }
            (BackendKind::Direct, _) => self.start(BackendKind::Direct, request),
            (BackendKind::Local, _) => self.start(BackendKind::Local, request),
        }
        true
    }

    fn start(&mut self, kind: BackendKind, request: TurnRequest) {
        let result = match kind {
            BackendKind::Direct => self.direct_backend.send(request),
            _ => self.local_backend.send(request),
        };
        if let Err(why) = result {
            let now = now_ms();
            let text = self.transcript.last_user_text().unwrap_or_default().to_string();
            self.apply_error(why, text, now);
        }
    }

    /// Removes the question and gives the draft back after a send that could not start.
    fn apply_error(&mut self, why: String, draft: String, now: u64) {
        let key = self.sessions.active_info().map(|s| s.key.clone()).unwrap_or_default();
        let ev = ChatEv::SendFailed { text: draft, error: why };
        let applied = self.transcript.apply(&format!("{key}:l{}", self.local_epoch), &ev, now);
        if let Some(d) = applied.restore_draft {
            if self.composer.text.trim().is_empty() {
                self.composer.text = d;
            }
        }
        self.running_on = None;
    }

    /// Sends the queued turn once the chat socket for its session is open.
    pub(super) fn dispatch_queued(&mut self) {
        let Some(mut request) = self.queued.take() else { return };
        let open = self.rt.live_session().as_deref() == Some(request.session.as_str()) && self.rt.link().state == crate::zc::events::LinkState::Open;
        if !open {
            self.queued = Some(request);
            return;
        }
        request.epoch = self.epoch;
        if let Err(why) = self.zc.send(request) {
            let text = self.transcript.last_user_text().unwrap_or_default().to_string();
            self.apply_error(why, text, now_ms());
        }
    }

    /// A queued turn that could not be sent goes back to the composer.
    fn give_back_queued(&mut self) {
        if let Some(request) = self.queued.take() {
            self.apply_error("the gateway could not be reached".to_string(), request.text, now_ms());
        }
    }

    pub fn stop_current(&mut self) {
        self.queued = None;
        match self.running_on {
            Some(BackendKind::ZeroClaw) => {
                self.zc.stop();
                self.stopping_since = Some(Instant::now());
            }
            Some(BackendKind::Direct) => self.direct_backend.stop(),
            Some(BackendKind::Local) => self.local_backend.stop(),
            None => {}
        }
    }

    /// Answers an approval card.
    pub fn decide(&mut self, request_id: &str, decision: Decision) {
        if let Some(wire) = super::approvals::decision_to_wire(decision) {
            if !self.zc.approve(request_id, wire) {
                self.transcript.notice(format!("approval:{request_id}"), Level::Warn, "The answer could not be delivered: the chat socket is closed.", None, now_ms());
                return;
            }
        }
        self.transcript.decide(request_id, decision);
    }

    /// Denies every approval that is waiting.
    pub fn deny_pending(&mut self) {
        let ids: Vec<String> = self
            .transcript
            .entries
            .iter()
            .filter_map(|e| match &e.kind {
                super::entry::Kind::Approval { request_id, decision: Decision::Pending, .. } => Some(request_id.clone()),
                _ => None,
            })
            .collect();
        for id in ids {
            self.decide(&id, Decision::Denied);
        }
    }

    /// Names the session after its first exchange: the gateway writes the title for ZeroClaw chats, the first words do for the rest.
    fn after_turn(&mut self) {
        let Some(index) = self.sessions.active else { return };
        let Some(info) = self.sessions.list.get(index).cloned() else { return };
        if info.is_legacy() || !info.has_default_title() || self.title_job.is_some() {
            return;
        }
        let Some((question, answer)) = self.transcript.first_exchange() else { return };
        if info.backend.as_deref() == Some("zeroclaw") && self.health.kind() == HealthKind::Online {
            let prompt = format!("Question: {}\n\nAnswer: {}", clip(&question, 600), clip(&answer, 900));
            let request = CompleteRequest::new(prompt).with_hint(self.profile.hint()).with_system("Write a title of at most six words for this chat. Reply with the title only, no quotes and no full stop.").with_temperature(0.2);
            self.title_job = Some((info.key.clone(), self.rt.complete(request)));
        } else {
            self.sessions.rename(index, &title_from_text(&question));
        }
    }

    /// Continues an old read-only chat as a new one, with its last question as the draft.
    pub fn continue_in_new_chat(&mut self) {
        let draft = self.transcript.last_user_text().map(str::to_string);
        self.new_chat();
        if let Some(d) = draft {
            if self.composer.text.trim().is_empty() {
                self.composer.text = d;
            }
        }
    }
}

fn clip(text: &str, max: usize) -> String {
    if text.chars().count() > max { text.chars().take(max).collect::<String>() + "..." } else { text.to_string() }
}
