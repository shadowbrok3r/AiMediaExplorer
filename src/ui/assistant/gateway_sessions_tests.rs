//! Headless tests of opening a gateway chat from the browser into the assistant panel, against the loopback mock gateway.

use std::sync::Arc;
use std::time::Duration;

use egui::epaint::ClippedShape;
use egui::{CentralPanel, Context, Pos2, RawInput, Rect, Shape, vec2};
use serde_json::json;
use zeroclaw_client::mock::MockGateway;

use super::entry::Kind;
use super::gateway_sessions::Open;
use super::*;
use crate::ui::file_table::FileExplorer;
use crate::ui::test_support::{press_release, test_ctx};
use crate::zc::runtime_tests::runtime_for;

fn text_rects(shapes: &[ClippedShape]) -> Vec<(String, Rect)> {
    fn walk(shape: &Shape, out: &mut Vec<(String, Rect)>) {
        match shape {
            Shape::Text(t) => out.push((t.galley.text().to_string(), t.galley.rect.translate(t.pos.to_vec2()))),
            Shape::Vec(v) => v.iter().for_each(|s| walk(s, out)),
            _ => {}
        }
    }
    let mut out = Vec::new();
    shapes.iter().for_each(|s| walk(&s.shape, &mut out));
    out
}

struct Rig {
    ctx: Context,
    panel: AssistantPanel,
    ex: FileExplorer,
    rt: Arc<ZcRuntime>,
    shapes: Vec<ClippedShape>,
    width: f32,
    clock: f64,
}

impl Rig {
    fn new(width: f32, rt: Arc<ZcRuntime>) -> Self {
        let mut panel = AssistantPanel::with_runtime(rt.clone());
        panel.mailboxes = false;
        Self { ctx: test_ctx(), panel, ex: FileExplorer::new(true), rt, shapes: Vec::new(), width, clock: 0.0 }
    }

    fn frame_with(&mut self, events: Vec<egui::Event>) {
        self.clock += 0.016;
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(self.width, 900.0))), events, time: Some(self.clock), ..Default::default() };
        let Self { ctx, panel, ex, .. } = self;
        let full = ctx.run_ui(input, |ui| {
            panel.receive(&ui.ctx().clone());
            CentralPanel::default().show(ui, |ui| {
                panel.ui_rail(ui, ex, false);
            });
        });
        self.shapes = full.shapes.clone();
        full.drop_without_applying_deltas();
    }

    fn frame(&mut self) {
        self.frame_with(Vec::new());
    }

    fn texts(&self) -> Vec<String> {
        text_rects(&self.shapes).into_iter().map(|(t, _)| t).collect()
    }

    fn has_text(&self, text: &str) -> bool {
        text_rects(&self.shapes).iter().any(|(t, _)| t == text)
    }

    fn has_text_containing(&self, part: &str) -> bool {
        text_rects(&self.shapes).iter().any(|(t, _)| t.contains(part))
    }

    fn click_text(&mut self, text: &str) {
        let rect = match text_rects(&self.shapes).into_iter().find(|(t, _)| t == text) {
            Some((_, r)) => r,
            None => panic!("no {text:?} drawn: {:?}", self.texts()),
        };
        self.frame_with(press_release(rect.center()));
        self.frame();
    }

    async fn until(&mut self, what: &str, mut done: impl FnMut(&mut Rig) -> bool) {
        let end = std::time::Instant::now() + Duration::from_secs(8);
        while std::time::Instant::now() < end {
            self.frame();
            if done(self) {
                self.frame();
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("timed out waiting for {what}; link {:?}, texts {:?}", self.rt.link(), self.texts());
    }

    fn user_texts(&self) -> Vec<String> {
        self.panel.transcript.entries.iter().filter_map(|e| if let Kind::User { text, .. } = &e.kind { Some(text.clone()) } else { None }).collect()
    }
}

fn stored(role: &str, n: usize) -> serde_json::Value {
    json!({"role": role, "content": format!("{role} message {n}"), "created_at": format!("2026-09-11T10:{:02}:00+00:00", n % 60)})
}

fn unique(prefix: &str) -> String {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    format!("{prefix}-{}", N.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
}

async fn gateway() -> (MockGateway, Arc<ZcRuntime>) {
    let mock = MockGateway::start().await;
    mock.add_session_row(json!({"session_id": "g1", "session_key": "gw_g1", "agent_alias": "default", "name": "Tag the sunsets", "message_count": 60, "last_activity": "2026-09-11T10:59:00+00:00"}));
    mock.set_messages("gw_g1", (0..60).map(|i| stored(if i % 2 == 0 { "user" } else { "assistant" }, i)).collect());
    let rt = runtime_for(&mock, "default", &unique("gw-open"));
    (mock, rt)
}

#[tokio::test(flavor = "multi_thread")]
async fn opening_a_gateway_chat_resumes_it_and_reads_only_the_newest_page_of_history() {
    let (mock, rt) = gateway().await;
    let mut r = Rig::new(420.0, rt);
    r.until("the gateway to be online", |r| r.panel.resolved.is_ready()).await;
    r.panel.open_gateway_session(Open { id: "g1".into(), name: Some("Tag the sunsets".into()) });
    assert_eq!(r.panel.sessions.active_info().map(|s| (s.key.clone(), s.title.clone(), s.backend.clone())), Some(("g1".to_string(), "Tag the sunsets".to_string(), Some("zeroclaw".to_string()))));
    r.until("the history", |r| r.user_texts().iter().any(|t| t == "user message 58")).await;
    assert!(!r.user_texts().iter().any(|t| t == "user message 0"), "the oldest messages are not read up front");
    assert!(r.panel.transcript.has_more, "older messages are left on the gateway for Load earlier messages");
    let conn = mock.connections().into_iter().find(|c| c.requested_session.as_deref() == Some("g1")).expect("the socket asked to resume g1");
    assert_eq!(conn.session_id.as_deref(), Some("g1"), "the same session answered");
    assert_eq!(mock.session_rows().len(), 1, "no new session was made on the gateway");
    assert!(mock.requests().iter().any(|q| q.method == "GET" && q.path.starts_with("/api/sessions/g1/messages")));
    assert!(r.panel.transcript.has_more);
    r.rt.load_history(50, true);
    r.until("older messages", |r| r.user_texts().iter().any(|t| t == "user message 0")).await;
    assert!(!r.panel.transcript.has_more, "all 60 messages are in");
}

#[tokio::test(flavor = "multi_thread")]
async fn opening_a_chat_that_is_already_in_the_list_switches_to_it_instead_of_adding_a_second_one() {
    let (_mock, rt) = gateway().await;
    let mut r = Rig::new(420.0, rt);
    r.until("the gateway to be online", |r| r.panel.resolved.is_ready()).await;
    r.panel.open_gateway_session(Open { id: "g1".into(), name: None });
    assert_eq!(r.panel.sessions.active_info().unwrap().title, "Chat g1", "a chat without a name gets a short title");
    r.panel.new_chat();
    r.until("a second chat", |r| r.panel.sessions.list.len() >= 2 && r.panel.sessions.active_info().is_some_and(|s| s.key != "g1")).await;
    let before = r.panel.sessions.list.len();
    r.panel.open_gateway_session(Open { id: "g1".into(), name: None });
    assert_eq!(r.panel.sessions.list.len(), before, "no duplicate");
    assert_eq!(r.panel.sessions.active_info().unwrap().key, "g1");
    r.panel.open_gateway_session(Open { id: "  ".into(), name: None });
    assert_eq!(r.panel.sessions.active_info().unwrap().key, "g1", "a blank id is ignored");
}

#[tokio::test(flavor = "multi_thread")]
async fn without_the_zeroclaw_backend_the_chat_opens_with_a_note_that_says_why_it_is_empty() {
    let (_mock, rt) = gateway().await;
    let mut profile = rt.profile();
    profile.backend = "local".into();
    rt.set_profile(profile);
    let mut r = Rig::new(420.0, rt);
    r.frame();
    r.panel.open_gateway_session(Open { id: "g1".into(), name: Some("x".into()) });
    r.frame();
    assert!(r.has_text_containing("Switch the backend to ZeroClaw"), "{:?}", r.texts());
}

#[tokio::test(flavor = "multi_thread")]
async fn the_header_menu_opens_the_browser_and_choosing_a_chat_opens_it_in_the_panel() {
    let (_mock, rt) = gateway().await;
    let mut r = Rig::new(420.0, rt);
    r.until("the gateway to be online", |r| r.panel.resolved.is_ready()).await;
    assert!(!gateway_sessions::is_open(&r.ctx));
    r.click_text("ZeroClaw - default");
    r.click_text("Gateway sessions");
    assert!(gateway_sessions::is_open(&r.ctx), "the header entry opens the browser");
    r.until("the list", |r| r.has_text("Tag the sunsets") && r.has_text("Open")).await;
    r.click_text("Open");
    assert_eq!(r.panel.sessions.active_info().map(|s| s.key.clone()), Some("g1".to_string()));
    assert!(!gateway_sessions::is_open(&r.ctx), "the browser closes once a chat is chosen");
    r.until("the history", |r| r.user_texts().iter().any(|t| t == "user message 58")).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn notices_from_agent_jobs_appear_in_the_open_chat() {
    let _g = notices::serial();
    notices::take();
    let (_mock, rt) = gateway().await;
    let mut r = Rig::new(420.0, rt);
    r.until("the gateway to be online", |r| r.panel.resolved.is_ready()).await;
    r.panel.mailboxes = true;
    notices::push(entry::Level::Info, "Image edit of a.png is ready. Review it in Settings > Metadata tools.");
    notices::push(entry::Level::Error, "Image edit of b.png failed: out of memory");
    r.frame();
    assert!(r.panel.sessions.active.is_none() && !r.has_text_containing("a.png is ready"), "without an open chat the notices wait");
    r.panel.new_chat();
    r.until("the chat", |r| r.panel.sessions.active.is_some()).await;
    r.frame();
    assert!(r.has_text_containing("a.png is ready"), "{:?}", r.texts());
    assert!(r.has_text_containing("b.png failed"));
    assert!(notices::take().is_empty(), "the panel took them");
}
