//! The gateway session browser: the chats the gateway holds for the profile's agent, with open, rename and delete.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use egui::{Context, Id, RichText, ScrollArea, TextEdit, Ui, Window};
use once_cell::sync::Lazy;
use surrealdb::types::RecordId;
use zeroclaw_client::SessionRow;

use super::AssistantPanel;
use super::backend::BackendKind;
use super::entry::{Level, NoticeAction};
use super::sessions::SessionInfo;
use crate::command::{AppCommand, Section};
use crate::icons::{self, svg};
use crate::zc::events::{Pending, now_ms};
use crate::zc::runtime::ZcRuntime;
use crate::zc::sessions::{self, Fault, PAGE_SIZE, SessionsError};
use crate::zc::widgets::ago;

/// A chat the person asked to open.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Open {
    pub id: String,
    pub name: Option<String>,
}

#[derive(Default)]
struct State {
    rows: Vec<SessionRow>,
    loaded: bool,
    list: Option<Pending<Result<Vec<SessionRow>, SessionsError>>>,
    error: Option<SessionsError>,
    page: usize,
    /// The chat being renamed and the text typed so far.
    rename: Option<(String, String)>,
    /// The chat the trash button asked about.
    confirm_delete: Option<String>,
    renaming: Option<(String, Pending<Result<(), SessionsError>>)>,
    deleting: Option<(String, Pending<Result<bool, SessionsError>>)>,
    note: Option<(bool, String)>,
}

fn state(ctx: &Context) -> Arc<Mutex<State>> {
    let id = Id::new("ame_gateway_sessions_state");
    ctx.data_mut(|d| d.get_temp_mut_or_insert_with::<Arc<Mutex<State>>>(id, Default::default).clone())
}

fn open_id() -> Id {
    Id::new("ame_gateway_sessions_open")
}

static OPEN_REQUESTS: Lazy<Mutex<Vec<Open>>> = Lazy::new(|| Mutex::new(Vec::new()));
static BROWSER_REQUEST: AtomicBool = AtomicBool::new(false);

/// Asks the assistant to open the gateway chat `open` and shows the assistant.
pub fn request_open(open: Open) {
    OPEN_REQUESTS.lock().unwrap_or_else(|e| e.into_inner()).push(open);
    super::api::request_open();
}

/// Chats queued with [`request_open`].
pub fn take_open_requests() -> Vec<Open> {
    std::mem::take(&mut *OPEN_REQUESTS.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Asks for the session browser to be shown, and the assistant with it.
pub fn request_browser() {
    BROWSER_REQUEST.store(true, Ordering::SeqCst);
    super::api::request_open();
}

pub fn is_open(ctx: &Context) -> bool {
    ctx.data(|d| d.get_temp::<bool>(open_id())).unwrap_or(false)
}

/// Shows or hides the browser; showing it reads the list again.
pub fn set_open(ctx: &Context, open: bool) {
    ctx.data_mut(|d| d.insert_temp(open_id(), open));
    if open {
        state(ctx).lock().unwrap_or_else(|e| e.into_inner()).loaded = false;
    }
    ctx.request_repaint();
}

impl State {
    fn busy(&self) -> bool {
        self.list.is_some() || self.renaming.is_some() || self.deleting.is_some()
    }

    fn reload(&mut self, rt: &ZcRuntime) {
        if self.list.is_none() {
            self.list = Some(sessions::list(rt));
            self.error = None;
        }
    }

    fn poll(&mut self, rt: &ZcRuntime) {
        if let Some(result) = self.list.as_mut().and_then(Pending::take) {
            self.list = None;
            self.loaded = true;
            match result {
                Ok(rows) => {
                    self.rows = rows;
                    self.error = None;
                }
                Err(e) => self.error = Some(e),
            }
        }
        if let Some((id, job)) = &mut self.renaming {
            if let Some(result) = job.take() {
                let id = id.clone();
                self.renaming = None;
                match result {
                    Ok(()) => {
                        self.rename = None;
                        self.note = Some((true, "Renamed on the gateway.".to_string()));
                        self.loaded = false;
                    }
                    Err(e) => self.note = Some((false, format!("Could not rename {id}: {}", e.message))),
                }
            }
        }
        if let Some((id, job)) = &mut self.deleting {
            if let Some(result) = job.take() {
                let id = id.clone();
                self.deleting = None;
                self.confirm_delete = None;
                match result {
                    Ok(_) => {
                        self.note = Some((true, "Deleted from the gateway.".to_string()));
                        self.loaded = false;
                    }
                    Err(e) => self.note = Some((false, format!("Could not delete {id}: {}", e.message))),
                }
            }
        }
        if !self.loaded && self.list.is_none() && rt.profile().is_configured() && rt.has_token() {
            self.reload(rt);
        }
    }
}

fn title_of(row: &SessionRow) -> String {
    match row.name.as_deref().map(str::trim).filter(|n| !n.is_empty()) {
        Some(n) => n.to_string(),
        None => {
            let short: String = row.id.chars().take(18).collect();
            format!("Chat {short}")
        }
    }
}

/// `5m ago`, `3h ago`, or the date once it is more than two days back.
fn when(now_ms: u64, at_ms: u64) -> String {
    if now_ms.saturating_sub(at_ms) < 48 * 3600 * 1000 {
        return ago(now_ms, at_ms);
    }
    chrono::DateTime::<chrono::Utc>::from_timestamp_millis(at_ms as i64).map(|t| t.with_timezone(&chrono::Local).format("%Y-%m-%d").to_string()).unwrap_or_default()
}

fn activity(row: &SessionRow) -> String {
    let count = if row.message_count == 1 { "1 message".to_string() } else { format!("{} messages", row.message_count) };
    match row.last_activity_epoch {
        Some(secs) if secs > 0 => format!("{count}, {}", when(now_ms(), secs as u64 * 1000)),
        _ => count,
    }
}

fn open_settings() {
    crate::command::send(AppCommand::ShowSettings { section: Some(Section::Assistant) });
}

/// Draws the browser's content; returns the chat the person chose to open.
fn body(ui: &mut Ui, rt: &Arc<ZcRuntime>, st: &mut State) -> Option<Open> {
    st.poll(rt);
    if st.busy() {
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
    }
    let profile = rt.profile();
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(format!("Chats of agent {}", if profile.agent.trim().is_empty() { "(none)" } else { profile.agent.trim() })).strong());
        if ui.add_enabled(!st.busy(), icons::button_widget(ui, svg::ARROW_CLOCKWISE, "Refresh")).clicked() {
            st.loaded = false;
        }
    });
    if !profile.is_configured() || !rt.has_token() {
        ui.label(RichText::new("The gateway is not set up yet. Add its address, an agent and a token in the connection settings.").color(ui.visuals().warn_fg_color));
        if icons::button(ui, svg::GEAR, "Connection settings").clicked() {
            open_settings();
        }
        return None;
    }
    if let Some(error) = st.error.clone() {
        match error.fault {
            Fault::Unauthorized => {
                ui.label(RichText::new("The gateway refused the token. Pair again to see its chats.").color(ui.visuals().error_fg_color));
                if icons::button(ui, svg::LINK, "Pair again").clicked() {
                    open_settings();
                }
            }
            Fault::Offline => {
                ui.label(RichText::new(format!("The gateway cannot be reached: {}", error.message)).color(ui.visuals().warn_fg_color));
                if ui.add_enabled(!st.busy(), icons::button_widget(ui, svg::ARROW_CLOCKWISE, "Try again")).clicked() {
                    st.loaded = false;
                }
            }
            Fault::Failed => {
                ui.label(RichText::new(format!("The chats could not be listed: {}", error.message)).color(ui.visuals().error_fg_color));
                if ui.add_enabled(!st.busy(), icons::button_widget(ui, svg::ARROW_CLOCKWISE, "Try again")).clicked() {
                    st.loaded = false;
                }
            }
        }
        if st.rows.is_empty() {
            return None;
        }
        ui.separator();
        ui.label(RichText::new("Last list that was read:").weak());
    }
    if st.list.is_some() && st.rows.is_empty() {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label("Reading the chats...");
        });
        return None;
    }
    if st.rows.is_empty() && st.loaded && st.error.is_none() {
        ui.label(RichText::new(format!("The gateway has no chats with messages for agent {}.", profile.agent.trim())).weak());
        return None;
    }
    let total = st.rows.len();
    let (page_rows, page, pages) = sessions::page(&st.rows, st.page, PAGE_SIZE);
    let page_rows: Vec<SessionRow> = page_rows.to_vec();
    st.page = page;
    let mut open = None;
    let mut want_rename = None;
    let mut ask_delete = None;
    let mut confirm_delete = None;
    let mut save_rename = None;
    ScrollArea::vertical().id_salt("ame_gateway_sessions_scroll").auto_shrink([false, true]).max_height(ui.available_height().clamp(120.0, 520.0)).show(ui, |ui| {
        for row in &page_rows {
            ui.group(|ui| {
                ui.set_width(ui.available_width());
                let renaming_here = st.rename.as_ref().is_some_and(|(id, _)| *id == row.id);
                if renaming_here {
                    if let Some((_, text)) = st.rename.as_mut() {
                        ui.add(TextEdit::singleline(text).desired_width(ui.available_width()).hint_text("New name").id_salt(("gw-rename", &row.id)));
                    }
                    ui.horizontal_wrapped(|ui| {
                        let typed = st.rename.as_ref().is_some_and(|(_, t)| !t.trim().is_empty());
                        if ui.add_enabled(typed && st.renaming.is_none(), icons::button_widget(ui, svg::CHECK, "Save name")).clicked() {
                            save_rename = st.rename.clone();
                        }
                        if icons::button(ui, svg::X, "Cancel").clicked() {
                            st.rename = None;
                        }
                    });
                } else {
                    ui.label(RichText::new(title_of(row)).strong());
                    ui.label(RichText::new(activity(row)).weak().small());
                    ui.horizontal_wrapped(|ui| {
                        if icons::button(ui, svg::FOLDER_OPEN, "Open").on_hover_text("Continue this chat here; the gateway keeps the history").clicked() {
                            open = Some(Open { id: row.id.clone(), name: row.name.clone() });
                        }
                        if icons::button(ui, svg::PENCIL_SIMPLE, "Rename").clicked() {
                            want_rename = Some((row.id.clone(), row.name.clone().unwrap_or_default()));
                        }
                        if st.confirm_delete.as_deref() == Some(row.id.as_str()) {
                            if ui.add_enabled(st.deleting.is_none(), icons::button_widget(ui, svg::TRASH, "Delete for good")).clicked() {
                                confirm_delete = Some(row.clone());
                            }
                            if icons::button(ui, svg::X, "Keep").clicked() {
                                st.confirm_delete = None;
                            }
                        } else if icons::button(ui, svg::TRASH, "Delete").on_hover_text("Remove this chat from the gateway").clicked() {
                            ask_delete = Some(row.id.clone());
                        }
                    });
                }
            });
        }
    });
    if pages > 1 {
        ui.horizontal_wrapped(|ui| {
            if ui.add_enabled(page > 0, egui::Button::new("Previous")).clicked() {
                st.page = page - 1;
            }
            ui.label(RichText::new(format!("Page {} of {pages}, {total} chats", page + 1)).weak());
            if ui.add_enabled(page + 1 < pages, egui::Button::new("Next")).clicked() {
                st.page = page + 1;
            }
        });
    }
    if let Some(r) = want_rename {
        st.rename = Some(r);
        st.confirm_delete = None;
    }
    if let Some(id) = ask_delete {
        st.confirm_delete = Some(id);
        st.rename = None;
    }
    if let Some((id, name)) = save_rename {
        st.note = None;
        let keys = st.rows.iter().find(|r| r.id == id).map(|r| r.keys.clone()).unwrap_or_else(|| vec![id.clone()]);
        st.renaming = Some((id.clone(), sessions::rename(rt, keys, name)));
    }
    if let Some(row) = confirm_delete {
        st.note = None;
        st.deleting = Some((row.id.clone(), sessions::delete(rt, row.keys.clone())));
    }
    if let Some((ok, text)) = &st.note {
        let color = if *ok { ui.visuals().text_color() } else { ui.visuals().error_fg_color };
        ui.label(RichText::new(text).color(color));
    }
    open
}

/// Draws the browser window while it is open; returns the chat to open.
pub fn show(ctx: &Context, rt: &Arc<ZcRuntime>) -> Option<Open> {
    if BROWSER_REQUEST.swap(false, Ordering::SeqCst) {
        set_open(ctx, true);
    }
    if !is_open(ctx) {
        return None;
    }
    let shared = state(ctx);
    let mut visible = true;
    let mut chosen = None;
    let max = (ctx.content_rect().width() - 24.0).max(220.0);
    Window::new("Gateway sessions").id(Id::new("ame_gateway_sessions_window")).open(&mut visible).default_width(360.0_f32.min(max)).max_width(max).resizable(true).show(ctx, |ui| {
        let mut st = shared.lock().unwrap_or_else(|e| e.into_inner());
        chosen = body(ui, rt, &mut st);
    });
    if chosen.is_some() || !visible {
        set_open_flag(ctx, false);
    }
    chosen
}

fn set_open_flag(ctx: &Context, open: bool) {
    ctx.data_mut(|d| d.insert_temp(open_id(), open));
}

impl AssistantPanel {
    /// Opens a chat the gateway already holds: it joins the list under the gateway's own id and its history is read lazily from the gateway.
    pub fn open_gateway_session(&mut self, open: Open) {
        let key = open.id.trim().to_string();
        if key.is_empty() {
            return;
        }
        let known = self.sessions.find(&key);
        match known {
            Some(i) if self.sessions.active == Some(i) => self.connect_tried = None,
            Some(i) => self.switch_to(i),
            None => {
                let title = open.name.as_deref().map(str::trim).filter(|n| !n.is_empty()).map(str::to_string).unwrap_or_else(|| format!("Chat {}", key.chars().take(18).collect::<String>()));
                let info = SessionInfo { id: RecordId::new(crate::ASSISTANT_SESSIONS, key.clone()), key: key.clone(), title, backend: Some("zeroclaw".to_string()), agent: Some(self.profile.agent.trim().to_string()) };
                self.sessions.save_new(&info, &self.profile.host_label(), Vec::new());
                self.adopt_session(info);
            }
        }
        if self.resolved.kind() != Some(BackendKind::ZeroClaw) {
            self.transcript.notice(format!("gateway-open:{key}"), Level::Warn, "This chat lives on the gateway. Switch the backend to ZeroClaw to read it.", Some(NoticeAction::OpenSettings), now_ms());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::test_support::{press_release, test_ctx};
    use crate::zc::runtime_tests::runtime_for;
    use egui::{CentralPanel, Event, Pos2, RawInput, Rect, vec2};
    use serde_json::json;
    use std::time::Duration;
    use zeroclaw_client::mock::{MockGateway, MockOptions};

    struct Rig {
        ctx: Context,
        rt: Arc<ZcRuntime>,
        shapes: Vec<egui::epaint::ClippedShape>,
        width: f32,
        opened: Option<Open>,
    }

    impl Rig {
        fn new(rt: Arc<ZcRuntime>, width: f32) -> Self {
            Self { ctx: test_ctx(), rt, shapes: Vec::new(), width, opened: None }
        }

        fn frame_with(&mut self, events: Vec<Event>) {
            let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(self.width, 900.0))), events, ..Default::default() };
            let (rt, shared) = (self.rt.clone(), state(&self.ctx));
            let mut opened = None;
            let full = self.ctx.run_ui(input, |ui| {
                CentralPanel::default().show(ui, |ui| {
                    let mut st = shared.lock().unwrap();
                    opened = body(ui, &rt, &mut st).or(opened.take());
                });
            });
            if opened.is_some() {
                self.opened = opened;
            }
            self.shapes = full.shapes.clone();
            full.drop_without_applying_deltas();
        }

        fn frame(&mut self) {
            self.frame_with(Vec::new());
        }

        fn texts(&self) -> Vec<(String, egui::Rect)> {
            fn walk(shape: &egui::Shape, out: &mut Vec<(String, egui::Rect)>) {
                match shape {
                    egui::Shape::Text(t) => out.push((t.galley.text().to_string(), t.galley.rect.translate(t.pos.to_vec2()))),
                    egui::Shape::Vec(v) => v.iter().for_each(|s| walk(s, out)),
                    _ => {}
                }
            }
            let mut out = Vec::new();
            self.shapes.iter().for_each(|s| walk(&s.shape, &mut out));
            out
        }

        fn names(&self) -> Vec<String> {
            self.texts().into_iter().map(|t| t.0).collect()
        }

        fn has(&self, text: &str) -> bool {
            self.texts().iter().any(|(t, _)| t == text)
        }

        fn has_containing(&self, part: &str) -> bool {
            self.texts().iter().any(|(t, _)| t.contains(part))
        }

        /// Clicks the first drawn text equal to `text`.
        fn click(&mut self, text: &str) {
            let rect = self.texts().into_iter().find(|(t, _)| t == text).map(|(_, r)| r).unwrap_or_else(|| panic!("no {text:?}: {:?}", self.names()));
            self.frame_with(press_release(rect.center()));
            self.frame();
        }

        async fn until(&mut self, what: &str, mut cond: impl FnMut(&mut Rig) -> bool) {
            let end = std::time::Instant::now() + Duration::from_secs(8);
            while std::time::Instant::now() < end {
                self.frame();
                if cond(self) {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(15)).await;
            }
            panic!("timed out waiting for {what}: {:?}", self.names());
        }
    }

    fn unique(prefix: &str) -> String {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        format!("{prefix}-{}", N.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }

    async fn gateway(extra: usize) -> (MockGateway, Arc<ZcRuntime>) {
        let mock = MockGateway::start_with(MockOptions { agents: vec!["media".into(), "voice".into()], ..Default::default() }).await;
        mock.add_session_row(json!({"session_id": "s-new", "session_key": "gw_s-new", "agent_alias": "media", "name": "Tag the sunsets", "message_count": 7, "last_activity": "2026-09-11T10:00:00+00:00"}));
        mock.add_session("s-old", "media", 1, "2026-09-09T12:00:00+00:00");
        mock.add_session("s-empty", "media", 0, "2026-09-11T11:59:00+00:00");
        mock.add_session("s-other", "voice", 9, "2026-09-11T11:00:00+00:00");
        for i in 0..extra {
            mock.add_session(&format!("bulk-{i:02}"), "media", 2, &format!("2026-08-{:02}T10:00:00+00:00", i + 1));
        }
        let rt = runtime_for(&mock, "media", &unique("gw-browser"));
        (mock, rt)
    }

    #[test]
    fn recent_activity_reads_as_an_age_and_old_activity_as_a_date() {
        let now = 1_760_000_000_000;
        assert_eq!(when(now, now - 5_000), "5s ago");
        assert_eq!(when(now, now - 3 * 3600 * 1000), "3h ago");
        let old = when(now, now - 5 * 24 * 3600 * 1000);
        assert_eq!(old.len(), 10, "{old}");
        assert!(old.starts_with("2025-"), "{old}");
        let row = |epoch| SessionRow { id: "a".into(), name: None, keys: vec!["a".into()], message_count: 1, last_activity: None, last_activity_epoch: epoch };
        assert_eq!(activity(&row(None)), "1 message");
        assert_eq!(title_of(&row(None)), "Chat a");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn the_list_shows_title_activity_and_count_without_empty_or_foreign_chats() {
        let (_mock, rt) = gateway(0).await;
        let mut r = Rig::new(rt, 700.0);
        r.until("the chats", |r| r.has("Tag the sunsets")).await;
        assert!(r.has_containing("7 messages"), "{:?}", r.names());
        assert!(r.has_containing("1 message,") || r.has("1 message"), "{:?}", r.names());
        assert!(r.has_containing("Chat s-old"), "a chat without a name gets a short title: {:?}", r.names());
        assert!(!r.has_containing("s-empty") && !r.has_containing("s-other"), "{:?}", r.names());
        for button in ["Open", "Rename", "Delete", "Refresh"] {
            assert!(r.has(button), "{button}: {:?}", r.names());
        }
        assert!(r.has("Chats of agent media"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn open_hands_back_the_chat_without_touching_the_gateway() {
        let (mock, rt) = gateway(0).await;
        let mut r = Rig::new(rt, 700.0);
        r.until("the chats", |r| r.has("Tag the sunsets")).await;
        let writes_before = mock.requests().iter().filter(|q| q.method != "GET").count();
        r.click("Open");
        assert_eq!(r.opened, Some(Open { id: "s-new".into(), name: Some("Tag the sunsets".into()) }));
        assert_eq!(mock.requests().iter().filter(|q| q.method != "GET").count(), writes_before);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn renaming_writes_the_new_name_and_the_list_follows() {
        let (mock, rt) = gateway(0).await;
        let mut r = Rig::new(rt, 700.0);
        r.until("the chats", |r| r.has("Tag the sunsets")).await;
        r.click("Rename");
        assert!(r.has("Save name") && r.has("Cancel"));
        let shared = state(&r.ctx);
        shared.lock().unwrap().rename.as_mut().unwrap().1 = "Sunset batch".to_string();
        r.frame();
        r.click("Save name");
        r.until("the rename", |r| r.has_containing("Renamed on the gateway")).await;
        assert!(mock.session_rows().iter().any(|s| s["name"] == "Sunset batch"));
        assert!(mock.requests().iter().any(|q| q.method == "PUT" && q.path == "/api/sessions/gw_s-new" && q.body["name"] == "Sunset batch"), "{:?}", mock.requests().iter().map(|q| format!("{} {}", q.method, q.path)).collect::<Vec<_>>());
        r.until("the list to show the new name", |r| r.has("Sunset batch") && !r.has("Save name")).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn deleting_asks_first_and_then_removes_only_that_chat() {
        let (mock, rt) = gateway(0).await;
        let mut r = Rig::new(rt, 700.0);
        r.until("the chats", |r| r.has("Tag the sunsets")).await;
        r.click("Delete");
        assert!(r.has("Delete for good") && r.has("Keep"));
        assert!(mock.requests().iter().all(|q| q.method != "DELETE"), "nothing is deleted before the confirmation");
        r.click("Keep");
        assert!(!r.has("Delete for good"));
        assert!(mock.requests().iter().all(|q| q.method != "DELETE"));
        r.click("Delete");
        r.click("Delete for good");
        r.until("the chat to go", |r| !r.has("Tag the sunsets")).await;
        let left: Vec<String> = mock.session_rows().iter().map(|s| s["session_id"].as_str().unwrap().to_string()).collect();
        assert!(!left.contains(&"s-new".to_string()) && left.contains(&"s-old".to_string()), "{left:?}");
        assert!(r.has_containing("Chat s-old"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn long_lists_are_paged() {
        let (_mock, rt) = gateway(PAGE_SIZE + 3).await;
        let mut r = Rig::new(rt, 700.0);
        r.until("the first page", |r| r.has_containing("Page 1 of 2")).await;
        let first_page = r.names().iter().filter(|t| t.as_str() == "Open").count();
        assert!((1..=PAGE_SIZE).contains(&first_page), "rows beyond the scroll area are not drawn: {first_page}");
        assert_eq!(state(&r.ctx).lock().unwrap().rows.len(), PAGE_SIZE + 3 + 2, "the page counts every chat with messages");
        assert!(r.has("Next") && r.has("Previous"));
        r.click("Next");
        assert!(r.has_containing("Page 2 of 2"), "{:?}", r.names());
        assert_eq!(r.names().iter().filter(|t| t.as_str() == "Open").count(), 5, "the rest of the rows");
        r.click("Previous");
        assert!(r.has_containing("Page 1 of 2"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_refused_token_says_so_and_offers_to_pair_again() {
        let (_mock, rt) = gateway(0).await;
        rt.save_token("not-the-token").unwrap();
        let mut r = Rig::new(rt, 700.0);
        r.until("the refusal", |r| r.has_containing("refused the token")).await;
        assert!(r.has("Pair again"));
        assert!(!r.has("Open"), "no chats are shown without a valid token");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn an_unreachable_gateway_shows_why_and_can_be_tried_again() {
        let (mock, rt) = gateway(0).await;
        drop(mock);
        let mut r = Rig::new(rt, 700.0);
        r.until("the offline message", |r| r.has_containing("cannot be reached")).await;
        assert!(r.has("Try again"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn an_empty_gateway_and_an_unconfigured_profile_say_what_is_missing() {
        let mock = MockGateway::start().await;
        let rt = runtime_for(&mock, "default", &unique("gw-browser-empty"));
        let mut r = Rig::new(rt, 700.0);
        r.until("the empty text", |r| r.has_containing("no chats with messages")).await;
        let blank = ZcRuntime::new(tokio::runtime::Handle::current(), None);
        let mut b = Rig::new(blank, 700.0);
        b.frame();
        assert!(b.has_containing("not set up yet") && b.has("Connection settings"), "{:?}", b.names());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn the_browser_fits_every_width() {
        let (_mock, rt) = gateway(PAGE_SIZE + 3).await;
        for width in [280.0, 360.0, 700.0, 1000.0, 1600.0] {
            let mut r = Rig::new(rt.clone(), width);
            r.until("the chats", |r| r.has("Open")).await;
            r.frame();
            for (text, rect) in r.texts() {
                assert!(rect.left() >= -0.5 && rect.right() <= width + 0.5, "{text:?} spills out at {width}: {rect:?}");
            }
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn the_window_opens_reads_and_closes_when_a_chat_is_chosen() {
        let (_mock, rt) = gateway(0).await;
        let ctx = test_ctx();
        assert!(!is_open(&ctx));
        assert!(show(&ctx, &rt).is_none(), "a closed browser draws nothing");
        set_open(&ctx, true);
        let end = std::time::Instant::now() + Duration::from_secs(8);
        let mut texts = Vec::new();
        while std::time::Instant::now() < end && !texts.iter().any(|(t, _): &(String, Pos2)| t == "Tag the sunsets") {
            let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(700.0, 900.0))), ..Default::default() };
            let rt = rt.clone();
            let out = ctx.run_ui(input, |ui| {
                show(&ui.ctx().clone(), &rt);
            });
            texts = crate::ui::test_support::texts(&out.shapes);
            out.drop_without_applying_deltas();
            tokio::time::sleep(Duration::from_millis(15)).await;
        }
        assert!(texts.iter().any(|(t, _)| t == "Gateway sessions"), "{texts:?}");
        assert!(texts.iter().any(|(t, _)| t == "Tag the sunsets"), "{texts:?}");
        assert!(is_open(&ctx));
    }
}
