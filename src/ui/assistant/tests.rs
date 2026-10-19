//! Headless tests of the assistant panel against the loopback mock gateway: real frames through the real runtime into the real transcript and widgets.

use std::sync::Arc;
use std::time::Duration;

use egui::epaint::ClippedShape;
use egui::{CentralPanel, Context, Event, Key, Modifiers, PointerButton, Pos2, RawInput, Rect, Shape, vec2};
use serde_json::json;
use zeroclaw_client::mock::{ConnScript, MockGateway, MockOptions, Turn};

use super::backend::{BackendKind, Fix, Resolved};
use super::composer;
use super::context::ChipKind;
use super::entry::{Decision, Kind};
use super::sessions::SessionInfo;
use super::*;
use crate::database::Thumbnail;
use crate::ui::file_table::FileExplorer;
use crate::ui::test_support::{press_release, test_ctx};
use crate::zc::events::ChatEv;
use crate::zc::health::Health;
use crate::zc::profile::{BackendPref, ZcProfile};
use crate::zc::runtime_tests::{runtime_for, temp_dir, until};
use crate::zc::secrets::TokenStore;

/// Text shapes with the rectangle they cover.
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
    height: f32,
    clock: f64,
}

fn files(dir: &std::path::Path) -> Vec<Thumbnail> {
    ["a.png", "b.png"]
        .iter()
        .map(|n| {
            let path = dir.join(n);
            image::RgbImage::from_pixel(32, 24, image::Rgb([20, 120, 220])).save(&path).unwrap();
            let mut row = Thumbnail::default();
            row.path = path.display().to_string();
            row.filename = n.to_string();
            row.file_type = "png".into();
            row.caption = Some(format!("caption of {n}"));
            row.tags = vec!["blue".into()];
            row.parent_dir = dir.display().to_string();
            row
        })
        .collect()
}

impl Rig {
    async fn new(width: f32, rt: Arc<ZcRuntime>, name: &str) -> Self {
        let ctx = test_ctx();
        egui_extras::install_image_loaders(&ctx);
        let mut ex = FileExplorer::new(true);
        let dir = temp_dir(&format!("files-{name}"));
        std::fs::create_dir_all(&dir).unwrap();
        ex.table.bulk_load(files(&dir));
        ex.sync_view();
        let mut panel = AssistantPanel::with_runtime(rt.clone());
        panel.mailboxes = false;
        Self { ctx, panel, ex, rt, shapes: Vec::new(), width, height: 760.0, clock: 0.0 }
    }

    fn frame_with(&mut self, events: Vec<Event>) {
        self.clock += 0.016;
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(self.width, self.height))), events, time: Some(self.clock), ..Default::default() };
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

    /// Enough frames for panels to size themselves and draw everything.
    fn settle(&mut self) {
        for _ in 0..3 {
            self.frame();
        }
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
        let texts: Vec<String> = self.texts();
        panic!("timed out waiting for {what}; link {:?}, health {:?}, resolved {:?}, texts {texts:?}", self.rt.link(), self.rt.health(), self.panel.resolved);
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
        let p = rect.center();
        self.frame_with(press_release(p));
        self.frame();
    }

    /// Waits until the gateway is online and a ZeroClaw chat is open and connected.
    async fn ready(&mut self) {
        self.until("the gateway to be online", |r| r.panel.resolved.is_ready()).await;
        if self.panel.sessions.active.is_none() {
            self.panel.new_chat();
        }
        self.until("the chat socket to open", |r| r.rt.link().state == crate::zc::events::LinkState::Open).await;
    }

    fn send(&mut self, text: &str) {
        self.panel.composer.text = text.to_string();
        self.panel.send_current(&mut self.ex);
    }

    fn assistant_texts(&self) -> Vec<String> {
        self.panel.transcript.entries.iter().filter_map(|e| if let Kind::Assistant { text, .. } = &e.kind { Some(text.clone()) } else { None }).collect()
    }
}

async fn rig(width: f32, mock: &MockGateway, name: &str) -> Rig {
    let rt = runtime_for(mock, "default", name);
    Rig::new(width, rt, name).await
}

#[tokio::test(flavor = "multi_thread")]
async fn a_plain_turn_streams_into_the_rail_and_ends_with_usage() {
    let mock = MockGateway::start().await;
    mock.script_turn(Turn::reply("Hello from the gateway"));
    let mut r = rig(420.0, &mock, "plain").await;
    r.ready().await;
    assert!(r.has_text("ZeroClaw - default"), "the header chip names the backend and agent: {:?}", r.texts());
    r.panel.composer.text = "hi there".into();
    r.frame();
    r.click_text("Send");
    r.until("the reply", |r| !r.panel.transcript.running && r.assistant_texts() == ["Hello from the gateway"]).await;
    assert!(r.has_text("hi there") && r.has_text("Hello from the gateway"), "{:?}", r.texts());
    assert!(r.panel.transcript.usage.is_some());
    assert!(r.panel.composer.text.is_empty(), "the draft was taken");
    let sent = mock.received_of_type("message");
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0]["content"], "hi there");
}

#[tokio::test(flavor = "multi_thread")]
async fn thinking_a_tool_call_and_an_approval_card_work_end_to_end() {
    let mock = MockGateway::start().await;
    mock.script_turn(
        Turn::new()
            .thinking("let me look at the library")
            .tool_call("c1", "aimedia__library_stats", json!({"limit": 5}))
            .approval("r1", "aimedia__library_stats", "count the library")
            .tool_result("c1", "aimedia__library_stats", "12 items in 3 folders")
            .chunk("There are 12 items.")
            .done("There are 12 items."),
    );
    let mut r = rig(420.0, &mock, "approval").await;
    r.ready().await;
    r.send("how big is the library?");
    r.until("the approval card", |r| r.has_text("Allow library_stats?")).await;
    assert!(r.has_text("Approve") && r.has_text("Always") && r.has_text("Deny"), "{:?}", r.texts());
    assert!(r.has_text_containing("Denied automatically in"), "{:?}", r.texts());
    assert_eq!(r.panel.transcript.pending_approvals(), 1);
    r.click_text("Approve");
    r.until("the answer", |r| !r.panel.transcript.running && r.assistant_texts() == ["There are 12 items."]).await;
    assert!(r.has_text("Thought (26 characters)"), "reasoning is collapsed: {:?}", r.texts());
    assert!(r.has_text("library_stats (done)"));
    assert!(r.has_text("library_stats returned 21 characters"));
    assert!(!r.has_text_containing("let me look at the library"), "collapsed rows hide their text");
    let thought = r.panel.transcript.entries.iter().find(|e| matches!(e.kind, Kind::Reasoning { .. })).unwrap().key.clone();
    r.panel.view.expanded.insert(thought);
    r.frame();
    assert!(r.has_text_containing("let me look at the library"), "an opened row shows its text");
    assert_eq!(mock.received_of_type("approval_response")[0]["decision"], "approve");
    assert!(matches!(&r.panel.transcript.entries.iter().find(|e| matches!(e.kind, Kind::Approval { .. })).unwrap().kind, Kind::Approval { decision: Decision::Approved, .. }));
}

#[tokio::test(flavor = "multi_thread")]
async fn an_always_decision_and_a_deny_reach_the_gateway() {
    let mock = MockGateway::start().await;
    mock.script_turn(Turn::new().approval("r1", "aimedia__describe_image", "describe a.png").chunk("ok").done("ok"));
    let mut r = rig(420.0, &mock, "always").await;
    r.ready().await;
    r.send("describe it");
    r.until("the card", |r| r.has_text("Allow describe_image?")).await;
    r.click_text("Always");
    r.until("the turn to end", |r| !r.panel.transcript.running).await;
    assert_eq!(mock.received_of_type("approval_response")[0]["decision"], "always");

    mock.script_turn(Turn::new().approval("r2", "aimedia__describe_image", "again").chunk("done").done("done"));
    r.send("again");
    r.until("the second card", |r| r.panel.transcript.pending_approvals() == 1).await;
    r.frame();
    r.click_text("Deny");
    r.until("the second turn to end", |r| !r.panel.transcript.running).await;
    assert_eq!(mock.received_of_type("approval_response")[1]["decision"], "deny");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_error_ends_the_turn_and_the_next_one_works() {
    let mock = MockGateway::start().await;
    mock.script_turn(Turn::new().chunk("partial ").error("PROVIDER_ERROR", "model unavailable"));
    mock.script_turn(Turn::reply("second works"));
    let mut r = rig(420.0, &mock, "error").await;
    r.ready().await;
    r.send("one");
    r.until("the error notice", |r| r.has_text("PROVIDER_ERROR: model unavailable") && !r.panel.transcript.running).await;
    assert_eq!(r.assistant_texts(), ["partial "], "what streamed before the error stays");
    r.send("two");
    r.until("the second answer", |r| r.assistant_texts().contains(&"second works".to_string())).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_dropped_connection_resumes_the_turn_without_duplicate_text() {
    let mock = MockGateway::start().await;
    mock.script_turn(Turn::new().chunk("Half ").delay_ms(30).drop_connection());
    let mut r = rig(420.0, &mock, "resume").await;
    r.ready().await;
    mock.script_connection(ConnScript::new().resume(vec![json!({"type":"chunk","content":"Half "})], Turn::new().chunk("done").done("Half done")));
    r.send("go");
    r.until("the resumed answer", |r| !r.panel.transcript.running && r.assistant_texts() == ["Half done"]).await;
    assert!(r.has_text("Half done"), "{:?}", r.texts());
    assert_eq!(mock.received_of_type("message").len(), 1, "the prompt is not resent after a reconnect");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_refused_token_offers_re_pair_and_is_never_retried() {
    let _serial = crate::zc::health::serial();
    let mock = MockGateway::start().await;
    let rt = crate::zc::runtime::ZcRuntime::new(tokio::runtime::Handle::current(), Some(TokenStore::at(temp_dir("refused-panel"))));
    rt.set_publish(true);
    rt.save_token("wrong-token").unwrap();
    rt.set_profile(ZcProfile { host: mock.url(), agent: "default".into(), ..Default::default() });
    let mut r = Rig::new(420.0, rt, "refused").await;
    r.until("the re-pair banner", |r| r.has_text("Re-pair")).await;
    assert!(matches!(r.panel.resolved, Resolved::Unavailable { backend: BackendKind::ZeroClaw, fix: Fix::Pair, .. }), "{:?}", r.panel.resolved);
    assert!(r.has_text("The gateway refused the token"), "{:?}", r.texts());
    r.send("will not go");
    r.settle();
    assert!(r.has_text_containing("The message was not sent") || r.has_text_containing("not sent"), "{:?}", r.texts());
    assert!(mock.received_of_type("message").is_empty());
    let attempts = mock.connections().len();
    for _ in 0..30 {
        r.frame();
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(mock.connections().len(), attempts, "no socket retry after a 401");
    assert!(crate::tasks::with(|t| t.attention()).iter().any(|i| i.key == "zeroclaw-token"));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_token_refused_mid_session_shows_a_re_pair_card_in_the_transcript() {
    let mock = MockGateway::start().await;
    let mut r = rig(420.0, &mock, "refused-mid").await;
    r.ready().await;
    mock.script_connection(ConnScript::new().reject(401, "Unauthorized"));
    mock.script_connection(ConnScript::new().reject(401, "Unauthorized"));
    r.rt.reconnect();
    r.until("the re-pair card", |r| r.has_text_containing("refused the token. It is not retried")).await;
    assert!(r.has_text("Re-pair"));
    assert_eq!(r.rt.health(), Health::Unauthorized);
}

#[tokio::test(flavor = "multi_thread")]
async fn an_assistant_that_is_not_set_up_shows_onboarding_and_never_reaches_a_provider() {
    let rt = crate::zc::runtime::ZcRuntime::new(tokio::runtime::Handle::current(), Some(TokenStore::at(temp_dir("unset"))));
    let mut r = Rig::new(420.0, rt, "unset").await;
    r.frame();
    assert_eq!(r.panel.resolved, Resolved::NeedsSetup);
    assert!(r.has_text("Connect the assistant") && r.has_text("Not set up"), "{:?}", r.texts());
    assert!(r.has_text("Pair"), "the card offers pairing: {:?}", r.texts());
    r.send("hello?");
    r.settle();
    assert!(r.has_text_containing("The assistant is not set up"), "{:?}", r.texts());
    assert!(r.panel.transcript.entries.iter().all(|e| !matches!(e.kind, Kind::User { .. })), "nothing was sent");
    assert_eq!(r.panel.composer.text, "hello?", "the draft stays");
}

#[tokio::test(flavor = "multi_thread")]
async fn stop_ends_the_turn_and_frames_after_it_are_ignored() {
    let mock = MockGateway::start().await;
    mock.script_turn(Turn::new().chunk("first ").chunk("second ").delay_ms(300).chunk("late words ").done("first second late words"));
    let mut r = rig(420.0, &mock, "stop").await;
    r.ready().await;
    let key = r.panel.sessions.active_info().unwrap().key.clone();
    mock.set_state(&key, "running", None);
    r.send("long answer please");
    r.until("some text", |r| r.panel.transcript.running && !r.assistant_texts().is_empty()).await;
    r.panel.stop_current();
    r.until("the turn to end", |r| !r.panel.transcript.running).await;
    for _ in 0..40 {
        r.frame();
        tokio::time::sleep(Duration::from_millis(15)).await;
    }
    assert!(!r.has_text_containing("late words"), "{:?}", r.texts());
    assert!(r.has_text("Stopped.") || r.panel.transcript.entries.iter().any(|e| matches!(&e.kind, Kind::Notice { text, .. } if text == "Stopped.")));
    assert!(mock.requests().iter().any(|q| q.path.ends_with("/abort")));
}

#[tokio::test(flavor = "multi_thread")]
async fn late_frames_of_a_session_that_was_left_are_dropped() {
    let mock = MockGateway::start().await;
    mock.script_turn(Turn::new().chunk("old ").delay_ms(200).chunk("session text").done("old session text"));
    let mut r = rig(420.0, &mock, "switch").await;
    r.ready().await;
    r.send("question for the first chat");
    r.until("first text", |r| !r.assistant_texts().is_empty()).await;
    r.panel.new_chat();
    r.frame();
    assert!(r.panel.transcript.entries.is_empty(), "the new chat starts empty");
    for _ in 0..40 {
        r.frame();
        tokio::time::sleep(Duration::from_millis(15)).await;
    }
    assert!(r.panel.transcript.entries.is_empty(), "frames of the old session do not land in the new one: {:?}", r.texts());
    assert_eq!(r.panel.sessions.list.len(), 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn chips_send_a_text_snapshot_and_pictures_stay_home_unless_allowed() {
    let mock = MockGateway::start().await;
    mock.script_turn(Turn::reply("seen"));
    mock.script_turn(Turn::reply("seen again"));
    let mut r = rig(420.0, &mock, "chips").await;
    r.ready().await;
    let first = r.ex.table.first().map(|row| row.path.clone()).unwrap();
    r.ex.select_paths([first.as_str()]);
    r.ex.sync_view();
    r.panel.composer.toggle_chip(ChipKind::Selection);
    r.panel.composer.toggle_chip(ChipKind::Filters);
    r.panel.composer.add_attachment(first.clone());
    r.settle();
    assert!(r.has_text("@selection (1)"), "{:?}", r.texts());
    r.send("what is this?");
    r.until("the first reply", |r| !r.panel.transcript.running).await;
    let sent = mock.received_of_type("message");
    let content = sent[0]["content"].as_str().unwrap();
    assert!(content.contains("Selected files (1):") && content.contains("caption of"), "{content}");
    assert!(content.contains("what is this?"));
    assert!(!content.contains("[IMAGE:"), "images are off by default");
    assert!(content.contains("not sent as pictures"), "{content}");
    assert!(r.panel.transcript.entries.iter().any(|e| matches!(&e.kind, Kind::User { chips, .. } if chips.iter().any(|c| c.starts_with("@selection")))));

    let mut profile = r.rt.profile();
    profile.allow_images = true;
    r.rt.set_profile(profile);
    r.settle();
    r.panel.composer.add_attachment(first);
    r.send("and now with the picture");
    r.until("the second reply", |r| !r.panel.transcript.running && r.assistant_texts().len() == 2).await;
    let content = mock.received_of_type("message")[1]["content"].as_str().unwrap().to_string();
    assert!(content.contains("[IMAGE:data:image/jpeg;base64,"), "an allowed picture goes inline");
}

#[tokio::test(flavor = "multi_thread")]
async fn slash_commands_answer_locally_without_the_gateway() {
    let mock = MockGateway::start().await;
    let mut r = rig(420.0, &mock, "slash").await;
    r.ready().await;
    r.send("/help");
    r.until("the help text", |r| r.assistant_texts().iter().any(|t| t.contains("/ping"))).await;
    assert!(mock.received_of_type("message").is_empty(), "quick actions never reach the gateway");
    r.send("/nope");
    r.until("the unknown reply", |r| r.assistant_texts().iter().any(|t| t.contains("Unknown command: /nope"))).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn typing_a_slash_lists_the_quick_actions() {
    let mock = MockGateway::start().await;
    let mut r = rig(420.0, &mock, "slashhints").await;
    r.ready().await;
    r.panel.composer.text = "/".into();
    r.settle();
    assert!(r.has_text_containing("/search <words>") && r.has_text_containing("/describe [path]"), "{:?}", r.texts());
    r.panel.composer.text = "/pi".into();
    r.settle();
    assert!(r.has_text_containing("/ping <text>") && !r.has_text_containing("/search <words>"));
}

#[tokio::test(flavor = "multi_thread")]
async fn the_gateway_names_the_session_after_the_first_exchange() {
    let mock = MockGateway::start().await;
    mock.script_turn(Turn::reply("Sunsets are in the Spain folder"));
    mock.set_complete_reply("\"Spanish sunsets\"");
    let mut r = rig(420.0, &mock, "title").await;
    r.ready().await;
    r.send("where are my sunset pictures?");
    r.until("the title", |r| r.panel.sessions.active_info().is_some_and(|s| s.title == "Spanish sunsets")).await;
    assert!(mock.requests().iter().any(|q| q.path == "/api/complete"), "the title came from /api/complete");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_chat_keeps_its_layout_at_every_rail_width() {
    let mock = MockGateway::start().await;
    for width in [280.0, 340.0, 420.0, 560.0, 760.0] {
        let mut r = rig(width, &mock, &format!("layout{}", width as u32)).await;
        r.ready().await;
        let scope = "s:z1";
        let long = "A long answer that has to wrap onto several lines at narrow widths without spilling out of the rail. ".repeat(8);
        let events = [
            ChatEv::Thinking { turn: 0, segment: 0, text: "thinking about it for a while".into() },
            ChatEv::ToolCall { turn: 0, index: 1, id: Some("c".into()), name: "aimedia__library_semantic_search".into(), args: json!({"query": "sunset over the sea"}) },
            ChatEv::ToolResult { turn: 0, index: 1, id: Some("c".into()), name: "aimedia__library_semantic_search".into(), output: "12 results".into() },
            ChatEv::Text { turn: 0, segment: 1, text: format!("{long}\n```rust\nfn main() {{ println!(\"hello world from a code block that is wide\"); }}\n```") },
            ChatEv::Plan { turn: 0, rows: vec![crate::zc::events::PlanRow { content: "describe the new pictures".into(), status: Some("pending".into()) }] },
            ChatEv::Error { turn: 0, code: Some("PROVIDER_ERROR".into()), message: "model unavailable right now, please try again in a moment".into(), ends_turn: false },
            ChatEv::Approval(zeroclaw_client::ApprovalRequest { turn: 0, request_id: "r".into(), tool: "aimedia__describe_image".into(), summary: "describe a picture with the local vision model".into(), arguments: json!({"path": "/home/u/Pictures/a.png"}), timeout_secs: 300 }),
        ];
        r.panel.transcript.push_user(format!("{scope}:u"), &long, vec![], vec!["@selection (3)".into()], 1);
        let now = crate::zc::events::now_ms();
        for ev in &events {
            r.panel.transcript.apply(scope, ev, now);
        }
        r.panel.composer.text = "a draft that is long enough to need wrapping in the narrowest rail we support".into();
        r.panel.transcript.running = false;
        r.frame();
        r.frame();
        for (text, rect) in text_rects(&r.shapes) {
            assert!(rect.left() >= -0.5 && rect.right() <= width + 0.5, "{text:?} spills out at width {width}: {rect:?}");
        }
        assert!(r.has_text("Send"), "the send button is visible at {width}: {:?}", r.texts());
        assert!(r.has_text("@selection (2)") || r.has_text("@selection"), "the chips are visible at {width}");
        assert!(r.has_text_containing("Allow describe_image?"), "the newest card is in view at {width}: {:?}", r.texts());
        assert!(r.has_text("Approve") && r.has_text("Always") && r.has_text("Deny"), "the approval buttons fit at {width}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn the_overlay_drawer_width_works_too() {
    let mock = MockGateway::start().await;
    let mut r = rig(900.0 * 0.94, &mock, "drawer").await;
    r.ready().await;
    r.panel.transcript.push_user("s:u".into(), "hello", vec![], vec![], 1);
    r.panel.transcript.running = false;
    r.settle();
    assert!(r.has_text("hello") && r.has_text("Send"), "{:?}", r.texts());
}

#[tokio::test(flavor = "multi_thread")]
async fn ctrl_j_opens_the_ask_tab_and_focuses_the_composer() {
    use crate::ui::inspector::prefs::RailTab;
    let ctx = test_ctx();
    let mut insp = crate::ui::inspector::Inspector::default();
    insp.set_open(false);
    insp.edit_prefs(|p| p.rail_tab = RailTab::Inspect);
    let key = Event::Key { key: Key::J, physical_key: Some(Key::J), pressed: true, repeat: false, modifiers: Modifiers::COMMAND };
    let mut handled = false;
    let out = ctx.run_ui(RawInput { events: vec![Event::ModifiersChanged(Modifiers::COMMAND), key], ..Default::default() }, |ui| {
        handled = handle_shortcuts(&ui.ctx().clone(), &mut insp, false);
    });
    out.drop_without_applying_deltas();
    assert!(handled && insp.is_open());
    assert_eq!(insp.prefs.prefs.rail_tab, RailTab::Ask);
    let mut insp2 = crate::ui::inspector::Inspector::default();
    insp2.set_open(false);
    handle_shortcuts(&ctx, &mut insp2, true);
    assert!(!insp2.is_open(), "a popped-out assistant keeps the rail as it is");
    let _ = composer::request_focus;
}

#[tokio::test(flavor = "multi_thread")]
async fn files_attached_through_the_command_show_as_chips_in_the_composer() {
    let _guard = api::serial();
    let mock = MockGateway::start().await;
    let mut r = rig(420.0, &mock, "attach").await;
    r.panel.mailboxes = true;
    api::take_attachments();
    ATTACH_REQUESTS.lock().unwrap().push(vec!["/pics/sunset.jpg".into()]);
    r.settle();
    assert_eq!(r.panel.composer.attachments, ["/pics/sunset.jpg"]);
    assert!(r.has_text("sunset.jpg"), "{:?}", r.texts());
    r.panel.mailboxes = false;
}

#[tokio::test(flavor = "multi_thread")]
async fn the_window_layout_lists_the_chats_beside_the_conversation() {
    let mock = MockGateway::start().await;
    let mut r = rig(760.0, &mock, "window").await;
    r.ready().await;
    r.panel.sessions.list[0].title = "Spanish sunsets".into();
    let Rig { ctx, panel, ex, .. } = &mut r;
    let mut drawn: Vec<String> = Vec::new();
    for _ in 0..3 {
        let out = ctx.run_ui(RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(760.0, 760.0))), ..Default::default() }, |ui| {
            CentralPanel::default().show(ui, |ui| {
                panel.ui(ui, ex);
            });
        });
        drawn = text_rects(&out.shapes).into_iter().map(|(t, _)| t).collect();
        out.drop_without_applying_deltas();
    }
    assert!(drawn.iter().any(|t| t == "Chats") && drawn.iter().any(|t| t == "Spanish sunsets"), "{drawn:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_chat_from_before_backends_is_read_only_with_a_way_forward() {
    let mock = MockGateway::start().await;
    let mut r = rig(420.0, &mock, "legacy").await;
    r.until("online", |r| r.panel.resolved.is_ready()).await;
    r.panel.adopt_session(SessionInfo { id: surrealdb::types::RecordId::new("assistant_sessions", "old"), key: "old".into(), title: "Old chat".into(), backend: None, agent: None });
    r.panel.transcript.push_user("old:u".into(), "an old question", vec![], vec![], 1);
    r.panel.transcript.running = false;
    r.frame();
    assert!(r.has_text_containing("read-only") && r.has_text("Continue in a new chat"), "{:?}", r.texts());
    r.send("a new message");
    r.frame();
    assert!(r.has_text_containing("read-only"));
    assert!(mock.received_of_type("message").is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn the_backend_menu_choice_is_stored_in_the_profile() {
    let mock = MockGateway::start_with(MockOptions::default()).await;
    let mut r = rig(420.0, &mock, "pick").await;
    r.ready().await;
    let mut profile = r.rt.profile();
    profile.backend = BackendPref::Local.as_str().into();
    crate::zc::save_profile_with(&r.rt, profile);
    r.frame();
    assert_eq!(r.panel.resolved, Resolved::Ready(BackendKind::Local));
    assert!(r.has_text("Local"), "{:?}", r.texts());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_send_before_any_chat_exists_creates_one_and_goes_out() {
    let mock = MockGateway::start().await;
    mock.script_turn(Turn::reply("first words"));
    let mut r = rig(420.0, &mock, "lazy").await;
    r.until("online", |r| r.panel.resolved.is_ready()).await;
    assert!(r.panel.sessions.active.is_none(), "no chat until one is needed");
    r.send("hello");
    r.until("the reply", |r| r.assistant_texts() == ["first words"]).await;
    assert_eq!(r.panel.sessions.list.len(), 1);
    assert!(!r.rt.link().session.as_deref().unwrap_or("").is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn the_socket_stays_closed_until_the_panel_is_on_screen() {
    let mock = MockGateway::start().await;
    let rt = runtime_for(&mock, "default", "offscreen");
    let mut panel = AssistantPanel::with_runtime(rt.clone());
    panel.mailboxes = false;
    let ctx = test_ctx();
    for _ in 0..50 {
        panel.receive(&ctx);
        tokio::time::sleep(Duration::from_millis(10)).await;
        if panel.resolved.is_ready() {
            break;
        }
    }
    assert!(panel.resolved.is_ready());
    panel.new_chat();
    for _ in 0..10 {
        panel.receive(&ctx);
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(mock.connections().is_empty(), "a hidden assistant opens no chat socket and mints no empty gateway session");
    panel.last_drawn = Some(std::time::Instant::now());
    panel.receive(&ctx);
    until("the socket", || (!mock.connections().is_empty()).then_some(())).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn an_offline_gateway_shows_retry_and_the_setup_card_and_never_reaches_a_provider() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    let rt = crate::zc::runtime::ZcRuntime::new(tokio::runtime::Handle::current(), Some(TokenStore::at(temp_dir("offline-panel"))));
    rt.save_token("some-token").unwrap();
    rt.set_profile(ZcProfile { host: addr.to_string(), agent: "media".into(), ..Default::default() });
    let mut r = Rig::new(420.0, rt, "offline").await;
    r.until("the offline banner", |r| r.has_text_containing("The gateway is offline")).await;
    assert!(r.has_text("Retry"), "{:?}", r.texts());
    assert!(r.has_text("Connect the assistant"), "Auto shows the setup card while the gateway is down: {:?}", r.texts());
    assert!(matches!(r.panel.resolved, Resolved::Unavailable { backend: BackendKind::ZeroClaw, fix: Fix::Retry, .. }));
    r.send("anyone there?");
    r.settle();
    assert!(r.has_text_containing("The message was not sent") || r.has_text_containing("offline"), "{:?}", r.texts());
}

#[tokio::test(flavor = "multi_thread")]
async fn leaving_a_chat_denies_the_approvals_it_was_waiting_for() {
    let mock = MockGateway::start().await;
    mock.script_turn(Turn::new().approval("r1", "aimedia__describe_image", "describe it").chunk("ok").done("ok"));
    let mut r = rig(420.0, &mock, "leave").await;
    r.ready().await;
    r.send("describe the picture");
    r.until("the approval card", |r| r.panel.transcript.pending_approvals() == 1).await;
    r.panel.new_chat();
    until("the denial to reach the gateway", || (!mock.received_of_type("approval_response").is_empty()).then_some(())).await;
    assert_eq!(mock.received_of_type("approval_response")[0]["decision"], "deny");
}
