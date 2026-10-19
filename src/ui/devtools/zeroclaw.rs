//! Devtools window for the ZeroClaw connection: state machine, frame log with a JSON viewer and recording, raw frame sender, session inspector, `/api/complete` playground and usage totals.

use std::sync::{LazyLock, Mutex};
use std::time::Instant;

use egui::{CollapsingHeader, Color32, ComboBox, Context, Id, RichText, ScrollArea, Slider, TextEdit, Ui, Window};
use egui_json_tree::{DefaultExpand, JsonTree};
use zeroclaw_client::{CompleteRequest, CompleteResponse, Direction, RawFrame};

use crate::zc::debug::{self, FRAME_CAP};
use crate::zc::events::{LinkState, Pending, now_ms};
use crate::zc::runtime::ZcRuntime;
use crate::zc::widgets::{ago, clock, health_row};

const ROW_HEIGHT: f32 = 18.0;

fn open_id() -> Id {
    Id::unique("ame_devtools_zeroclaw_open")
}

pub fn is_open(ctx: &Context) -> bool {
    ctx.data(|d| d.get_temp::<bool>(open_id())).unwrap_or(false)
}

pub fn set_open(ctx: &Context, open: bool) {
    ctx.data_mut(|d| d.insert_temp(open_id(), open));
}

pub fn toggle(ctx: &Context) {
    set_open(ctx, !is_open(ctx));
}

/// What the session inspector read from the gateway.
#[derive(Clone, Debug, Default)]
struct SessionReport {
    state: String,
    running: Vec<String>,
    messages: usize,
}

struct State {
    filter: String,
    dir: Option<Direction>,
    selected: Option<usize>,
    record_path: String,
    record_note: Option<(bool, String)>,
    raw: String,
    session: String,
    report: Option<Pending<Result<SessionReport, String>>>,
    report_shown: Option<Result<SessionReport, String>>,
    hint: String,
    system: String,
    prompt: String,
    temperature: f32,
    job: Option<(Instant, Pending<Result<CompleteResponse, String>>)>,
    result: Option<(bool, String, u128)>,
}

impl Default for State {
    fn default() -> Self {
        Self {
            filter: String::new(),
            dir: None,
            selected: None,
            record_path: String::new(),
            record_note: None,
            raw: r#"{"type":"message","content":"hello"}"#.to_string(),
            session: String::new(),
            report: None,
            report_shown: None,
            hint: String::new(),
            system: String::new(),
            prompt: "Reply with the single word: pong".to_string(),
            temperature: 0.2,
            job: None,
            result: None,
        }
    }
}

static STATE: LazyLock<Mutex<State>> = LazyLock::new(|| Mutex::new(State::default()));

fn default_record_path() -> String {
    let dir = crate::ai::mcp::config::default_dir().unwrap_or_else(std::env::temp_dir);
    dir.join(format!("zc-frames-{}.jsonl", now_ms() / 1000)).display().to_string()
}

fn link_color(ui: &Ui, state: LinkState) -> Color32 {
    match state {
        LinkState::Open => Color32::from_rgb(88, 196, 120),
        LinkState::Connecting | LinkState::Reconnecting => ui.visuals().warn_fg_color,
        LinkState::Failed => ui.visuals().error_fg_color,
        LinkState::Idle | LinkState::Closed => ui.visuals().weak_text_color(),
    }
}

fn connection_section(ui: &mut Ui, rt: &ZcRuntime, st: &mut State) {
    let profile = rt.profile();
    health_row(ui, &rt.health());
    ui.label(RichText::new(format!("{} - {} - agent {} - backend {}", profile.display_name(), profile.host_label(), profile.agent.trim(), profile.backend)).weak());
    ui.label(RichText::new(match rt.token_path() {
        Some(p) => format!("Token: {} ({})", if rt.has_token() { "stored" } else { "none" }, p.display()),
        None => "Token: no storage".to_string(),
    })
    .weak());
    let link = rt.link();
    ui.horizontal_wrapped(|ui| {
        ui.label("Socket");
        for state in [LinkState::Idle, LinkState::Connecting, LinkState::Open, LinkState::Reconnecting, LinkState::Closed, LinkState::Failed] {
            let current = link.state == state;
            let text = RichText::new(state.label()).color(if current { link_color(ui, state) } else { ui.visuals().weak_text_color() });
            ui.label(if current { text.strong().underline() } else { text });
        }
    });
    ui.label(RichText::new(format!("{} {}", link.session.as_deref().unwrap_or("no session"), link.detail)).weak());
    ui.horizontal_wrapped(|ui| {
        if ui.button("Check").clicked() {
            rt.check();
        }
        if ui.button("Connect new session").clicked() {
            let id = format!("devtools-{}", now_ms() / 1000);
            rt.connect(&id, Some("Devtools"));
        }
        if ui.add_enabled(link.state == LinkState::Open || link.state == LinkState::Reconnecting, egui::Button::new("Force reconnect")).clicked() {
            rt.reconnect();
        }
        if ui.add_enabled(link.session.is_some(), egui::Button::new("Drop socket")).on_hover_text("Closes the socket and ends the session in this window").clicked() {
            rt.disconnect();
        }
        if ui.button("Ask gateway to ping tools").on_hover_text("Sends a message asking the agent to call aimedia__util_ping").clicked() {
            crate::ui::assistant::api::submit(crate::zc::command::PING_PROMPT.to_string());
            crate::ui::assistant::api::request_open();
        }
    });
    if st.session.is_empty() {
        st.session = link.session.clone().unwrap_or_default();
    }
    CollapsingHeader::new("State history").default_open(false).show(ui, |ui| {
        let now = now_ms();
        for step in link.steps.iter().rev().take(30) {
            ui.label(RichText::new(format!("{} {:>12} {}", clock(step.at_ms), step.state.label(), step.detail)).monospace().small().weak()).on_hover_text(ago(now, step.at_ms));
        }
    });
}

fn frame_line(i: usize, f: &RawFrame) -> String {
    let arrow = if f.dir == Direction::In { "<-" } else { "->" };
    let summary = match f.frame.get("content").and_then(|c| c.as_str()).or_else(|| f.frame.get("name").and_then(|c| c.as_str())).or_else(|| f.frame.get("message").and_then(|c| c.as_str())) {
        Some(text) => text.chars().take(60).collect::<String>().replace('\n', " "),
        None => String::new(),
    };
    format!("{i:>3} +{:>7.2}s {arrow} {} {summary}", f.t_ms as f64 / 1000.0, debug::frame_type(&f.frame))
}

fn frames_section(ui: &mut Ui, rt: &ZcRuntime, st: &mut State) {
    let (total_in, total_out, kept, recording) = rt.with_frames(|f| (f.total_in, f.total_out, f.len(), f.recording().map(|r| (r.path().display().to_string(), r.lines))));
    ui.horizontal_wrapped(|ui| {
        ui.add(TextEdit::singleline(&mut st.filter).hint_text("filter: type or text").desired_width(160.0));
        ComboBox::from_id_salt("zc_dir_filter").selected_text(match st.dir {
            None => "both",
            Some(Direction::In) => "in",
            Some(Direction::Out) => "out",
        }).show_ui(ui, |ui| {
            ui.selectable_value(&mut st.dir, None, "both");
            ui.selectable_value(&mut st.dir, Some(Direction::In), "in");
            ui.selectable_value(&mut st.dir, Some(Direction::Out), "out");
        });
        if ui.button("Clear").clicked() {
            rt.with_frames(|f| f.clear());
            st.selected = None;
        }
        ui.label(RichText::new(format!("{kept} kept of {FRAME_CAP}, {total_in} in, {total_out} out")).weak());
    });
    ui.horizontal_wrapped(|ui| match &recording {
        Some((path, lines)) => {
            ui.label(RichText::new(format!("Recording {lines} lines to {path}")).color(ui.visuals().warn_fg_color));
            if ui.button("Stop recording").clicked() {
                let stopped = rt.with_frames(|f| f.stop_recording());
                st.record_note = stopped.map(|(p, n)| (true, format!("Saved {n} lines to {}", p.display())));
            }
        }
        None => {
            if st.record_path.is_empty() {
                st.record_path = default_record_path();
            }
            ui.add(TextEdit::singleline(&mut st.record_path).desired_width(ui.available_width().min(360.0)));
            if ui.button("Record to JSONL").on_hover_text("Writes every frame, with the token removed, in the format zcprobe and the mock gateway replay").clicked() {
                let path = std::path::PathBuf::from(st.record_path.trim());
                st.record_note = Some(match rt.with_frames(|f| f.start_recording(&path)) {
                    Ok(()) => (true, "Recording".to_string()),
                    Err(e) => (false, format!("Cannot record: {e}")),
                });
            }
        }
    });
    if let Some((ok, note)) = &st.record_note {
        ui.label(RichText::new(note).color(if *ok { ui.visuals().text_color() } else { ui.visuals().error_fg_color }));
    }
    let rows: Vec<(usize, RawFrame)> = rt.with_frames(|f| f.frames().cloned().enumerate().filter(|(_, fr)| debug::matches(fr, st.dir, &st.filter)).collect());
    ScrollArea::vertical().id_salt("zc_frames_scroll").max_height(180.0).stick_to_bottom(true).auto_shrink([false, true]).show_rows(ui, ROW_HEIGHT, rows.len(), |ui, range| {
        for (idx, frame) in &rows[range] {
            let text = RichText::new(frame_line(*idx, frame)).monospace().small();
            if ui.selectable_label(st.selected == Some(*idx), text).clicked() {
                st.selected = Some(*idx);
            }
        }
    });
    if let Some(idx) = st.selected {
        if let Some((_, frame)) = rows.iter().find(|(i, _)| *i == idx) {
            ui.separator();
            JsonTree::new(format!("zc-frame-{idx}-{}", frame.t_ms), &frame.frame).default_expand(DefaultExpand::All).show(ui);
        }
    }
}

fn raw_section(ui: &mut Ui, rt: &ZcRuntime, st: &mut State) {
    ui.label(RichText::new("The gateway accepts only message and approval_response frames from a client.").weak());
    ui.add(TextEdit::multiline(&mut st.raw).code_editor().desired_rows(2).desired_width(ui.available_width()));
    if ui.button("Send frame").clicked() && !rt.raw_send(&st.raw) {
        rt.note("raw frame not sent: no chat socket is open");
    }
}

fn session_section(ui: &mut Ui, rt: &ZcRuntime, st: &mut State) {
    ui.horizontal_wrapped(|ui| {
        ui.add(TextEdit::singleline(&mut st.session).hint_text("session id").desired_width(260.0));
        if ui.button("Read state").clicked() {
            let id = st.session.trim().to_string();
            st.report = Some(rt.job(move |sh| async move {
                let rest = sh.rest()?;
                let state = rest.session_state(&id).await.map_err(|e| sh.redact(&e.to_string()))?;
                let running = rest.running().await.map_err(|e| sh.redact(&e.to_string()))?;
                let messages = rest.messages(&id, false).await.map(|m| m.len()).unwrap_or(0);
                Ok(SessionReport { state: state.state, running, messages })
            }));
        }
    });
    if let Some(job) = st.report.as_mut() {
        if let Some(result) = job.take() {
            st.report_shown = Some(result);
            st.report = None;
        } else {
            ui.label(RichText::new("Asking the gateway...").weak());
        }
    }
    match &st.report_shown {
        Some(Ok(r)) => {
            ui.label(format!("State {} - {} stored messages - {} sessions running", r.state, r.messages, r.running.len()));
            if !r.running.is_empty() {
                ui.label(RichText::new(r.running.join(", ")).monospace().small().weak());
            }
        }
        Some(Err(e)) => {
            ui.label(RichText::new(e).color(ui.visuals().error_fg_color));
        }
        None => {}
    }
}

fn complete_section(ui: &mut Ui, rt: &ZcRuntime, st: &mut State) {
    if st.hint.is_empty() {
        st.hint = rt.profile().hint().to_string();
    }
    ui.horizontal_wrapped(|ui| {
        ui.label("Hint");
        ui.add(TextEdit::singleline(&mut st.hint).desired_width(90.0));
        ui.label("Temperature");
        ui.add(Slider::new(&mut st.temperature, 0.0..=2.0));
    });
    ui.add(TextEdit::singleline(&mut st.system).hint_text("system (optional)").desired_width(ui.available_width()));
    ui.add(TextEdit::multiline(&mut st.prompt).hint_text("prompt").desired_rows(3).desired_width(ui.available_width()));
    if ui.add_enabled(st.job.is_none(), egui::Button::new("Run")).clicked() {
        let mut request = CompleteRequest::new(st.prompt.clone()).with_hint(st.hint.trim()).with_temperature(st.temperature as f64);
        if !st.system.trim().is_empty() {
            request = request.with_system(st.system.clone());
        }
        st.job = Some((Instant::now(), rt.complete(request)));
    }
    if let Some((started, job)) = st.job.as_mut() {
        let started = *started;
        if let Some(result) = job.take() {
            let ms = started.elapsed().as_millis();
            st.result = Some(match result {
                Ok(r) => (true, format!("[{}] {}", r.model, r.response), ms),
                Err(e) => (false, e, ms),
            });
            st.job = None;
        } else {
            ui.label(RichText::new("Waiting for the model...").weak());
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
        }
    }
    if let Some((ok, text, ms)) = &st.result {
        ui.label(RichText::new(format!("{ms} ms")).weak());
        ui.label(RichText::new(text).color(if *ok { ui.visuals().text_color() } else { ui.visuals().error_fg_color }));
    }
}

fn usage_section(ui: &mut Ui, rt: &ZcRuntime) {
    let u = rt.usage();
    ui.label(format!("{} turns, {} tokens in, {} out, ${:.4}", u.turns, u.input_tokens, u.output_tokens, u.cost_usd));
    egui::Grid::new("zc_usage_grid").striped(true).show(ui, |ui| {
        for (model, turns, input, output, cost) in &u.by_model {
            ui.label(model);
            ui.label(format!("{turns} turns"));
            ui.label(format!("{input} in"));
            ui.label(format!("{output} out"));
            ui.label(format!("${cost:.4}"));
            ui.end_row();
        }
    });
    if let Some(last) = &u.last {
        ui.label(RichText::new(format!("Last: {:?} of {:?} tokens of context, model {:?}", last.last_input_tokens.or(last.input_tokens), last.context_window, last.model)).weak().small());
    }
    ui.label(RichText::new(format!("{} events waiting, {} dropped", rt.backlog(), rt.dropped())).weak().small());
}

fn notes_section(ui: &mut Ui, rt: &ZcRuntime) {
    let now = now_ms();
    for (at, text) in rt.notes().iter().take(40) {
        ui.label(RichText::new(format!("{} {text}", clock(*at))).monospace().small()).on_hover_text(ago(now, *at));
    }
}

/// Draws the window when it is open.
pub fn show(ctx: &Context) {
    if !is_open(ctx) {
        return;
    }
    let rt = crate::zc::runtime();
    rt.attach_ctx(ctx);
    let mut open = true;
    Window::new("ZeroClaw").open(&mut open).default_size([580.0, 680.0]).resizable(true).show(ctx, |ui| {
        let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
        ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            CollapsingHeader::new(RichText::new("Connection").strong()).default_open(true).show(ui, |ui| connection_section(ui, rt, &mut st));
            CollapsingHeader::new(RichText::new("Frames").strong()).default_open(true).show(ui, |ui| frames_section(ui, rt, &mut st));
            CollapsingHeader::new(RichText::new("Send a frame").strong()).default_open(false).show(ui, |ui| raw_section(ui, rt, &mut st));
            CollapsingHeader::new(RichText::new("Session inspector").strong()).default_open(false).show(ui, |ui| session_section(ui, rt, &mut st));
            CollapsingHeader::new(RichText::new("/api/complete playground").strong()).default_open(false).show(ui, |ui| complete_section(ui, rt, &mut st));
            CollapsingHeader::new(RichText::new("Usage").strong()).default_open(false).show(ui, |ui| usage_section(ui, rt));
            CollapsingHeader::new(RichText::new("Notes").strong()).default_open(false).show(ui, |ui| notes_section(ui, rt));
            CollapsingHeader::new(RichText::new("Connection card (as in Settings)").strong()).default_open(false).show(ui, |ui| crate::zc::connection_ui(ui));
        });
    });
    if !open {
        set_open(ctx, false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_line_has_the_index_time_direction_type_and_a_summary() {
        let f = RawFrame { t_ms: 1234, dir: Direction::In, frame: serde_json::json!({"type":"chunk","content":"Hello\nworld"}) };
        let line = frame_line(7, &f);
        assert!(line.contains("  7") && line.contains("+   1.23s") && line.contains("<-") && line.contains("chunk") && line.contains("Hello world"), "{line}");
        let out = RawFrame { t_ms: 0, dir: Direction::Out, frame: serde_json::json!({"type":"message"}) };
        assert!(frame_line(0, &out).contains("->"));
    }

    #[test]
    fn the_window_opens_toggles_and_draws() {
        let ctx = crate::ui::test_support::test_ctx();
        assert!(!is_open(&ctx));
        toggle(&ctx);
        assert!(is_open(&ctx));
        let mut drawn = Vec::new();
        for _ in 0..3 {
            let out = ctx.run_ui(egui::RawInput::default(), |ui| show(&ui.ctx().clone()));
            drawn = crate::ui::test_support::texts(&out.shapes);
            out.drop_without_applying_deltas();
        }
        for expected in ["Connection", "Frames", "Usage"] {
            assert!(drawn.iter().any(|(t, _)| t == expected), "{expected} in {drawn:?}");
        }
        set_open(&ctx, false);
        assert!(!is_open(&ctx));
    }
}
