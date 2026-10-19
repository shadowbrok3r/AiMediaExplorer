//! Devtools window for the ZeroClaw one-shot jobs: the queue, the running job, the history with sizes, durations and usage, a viewer of the raw request and reply with the file paths optionally masked, a manual submit box and Cancel.

use std::sync::{Arc, LazyLock, Mutex};

use egui::{CollapsingHeader, ComboBox, Context, Id, ProgressBar, RichText, ScrollArea, TextEdit, Ui, Window};
use egui_json_tree::{DefaultExpand, JsonTree};

use crate::zc::events::now_ms;
use crate::zc::jobs::record::{JobRecord, StepRecord, mask_paths};
use crate::zc::jobs::{JobId, JobKind, JobSpec, Snapshot, Step, ZcJobs};
use crate::zc::widgets::ago;

fn open_id() -> Id {
    Id::new("ame_devtools_zc_jobs_open")
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Part {
    System,
    Prompt,
    Reply,
    Parsed,
}

struct State {
    selected: Option<(JobId, usize)>,
    part: Part,
    mask_paths: bool,
    hint: String,
    system: String,
    prompt: String,
    temperature: f32,
    note: Option<(bool, String)>,
}

impl Default for State {
    fn default() -> Self {
        Self { selected: None, part: Part::Prompt, mask_paths: true, hint: String::new(), system: String::new(), prompt: "Reply with the single word: pong".to_string(), temperature: 0.2, note: None }
    }
}

static STATE: LazyLock<Mutex<State>> = LazyLock::new(|| Mutex::new(State::default()));

fn size(bytes: usize) -> String {
    if bytes >= 10 * 1024 { format!("{:.1} KiB", bytes as f64 / 1024.0) } else { format!("{bytes} B") }
}

fn millis(ms: u64) -> String {
    if ms >= 10_000 { format!("{:.1} s", ms as f64 / 1000.0) } else { format!("{ms} ms") }
}

fn status_section(ui: &mut Ui, jobs: &Arc<ZcJobs>, snap: &Snapshot) {
    match snap.availability.reason() {
        None => {
            ui.label(RichText::new("Gateway online: jobs can run").color(crate::theme::success(ui)));
        }
        Some(why) => {
            ui.label(RichText::new(format!("Jobs cannot run: {why}")).color(crate::theme::warn(ui)));
        }
    }
    let t = &snap.totals;
    ui.label(RichText::new(format!("Hint {} - queue {} of {} - {} jobs ({} done, {} failed, {} cancelled)", snap.hint, snap.queue.len(), snap.queue_limit, t.jobs, t.completed, t.failed, t.cancelled)).weak());
    ui.label(RichText::new(format!("{} requests ({} retries) - {} sent (~{} tokens), {} received (~{} tokens) - {} on the gateway", t.calls, t.retries, size(t.prompt_bytes as usize), t.est_tokens_in(), size(t.response_bytes as usize), t.est_tokens_out(), millis(t.duration_ms))).weak());
    ui.label(RichText::new("The gateway reports no token counts for /api/complete; tokens are estimated at four bytes each.").weak().small());
    ui.horizontal(|ui| {
        if ui.add_enabled(snap.running.is_some() || !snap.queue.is_empty(), egui::Button::new("Cancel all")).clicked() {
            jobs.cancel_all();
        }
        if ui.add_enabled(!snap.history.is_empty(), egui::Button::new("Clear history")).clicked() {
            jobs.clear_history();
            STATE.lock().unwrap_or_else(|e| e.into_inner()).selected = None;
        }
    });
}

fn running_section(ui: &mut Ui, jobs: &Arc<ZcJobs>, snap: &Snapshot) {
    match &snap.running {
        None => {
            ui.label(RichText::new("Nothing is running").weak());
        }
        Some(r) => {
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("#{} {} - {}", r.id, r.kind.label(), r.title)).strong());
                if ui.button("Cancel").clicked() {
                    jobs.cancel(r.id);
                }
            });
            let fraction = if r.total == 0 { 0.0 } else { r.done as f32 / r.total as f32 };
            ui.add(ProgressBar::new(fraction).desired_width(260.0).text(format!("{} of {} requests, {:.1} s", r.done, r.total, r.elapsed.as_secs_f32())));
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(200));
        }
    }
    for q in &snap.queue {
        ui.horizontal(|ui| {
            ui.label(format!("Queued #{} {} - {} ({} requests, waiting {:.0} s)", q.id, q.kind.label(), q.title, q.steps, q.waited.as_secs_f32()));
            if ui.small_button("Cancel").clicked() {
                jobs.cancel(q.id);
            }
        });
    }
}

fn submit_section(ui: &mut Ui, jobs: &Arc<ZcJobs>, st: &mut State) {
    if st.hint.is_empty() {
        st.hint = jobs.default_hint();
    }
    ui.horizontal_wrapped(|ui| {
        ui.label("Hint");
        ui.add(TextEdit::singleline(&mut st.hint).desired_width(90.0));
        ui.label("Temperature");
        ui.add(egui::Slider::new(&mut st.temperature, 0.0..=2.0));
    });
    ui.add(TextEdit::singleline(&mut st.system).hint_text("system (optional)").desired_width(ui.available_width()));
    ui.add(TextEdit::multiline(&mut st.prompt).hint_text("prompt").desired_rows(3).desired_width(ui.available_width()));
    if ui.button("Submit job").on_hover_text("Queues one request as a Manual job; it runs after the jobs before it").clicked() {
        let title = format!("Manual: {}", crate::ui::refine::model::clipped(st.prompt.trim(), 40));
        let mut step = Step::new("manual", st.prompt.clone()).temperature(f64::from(st.temperature));
        if !st.system.trim().is_empty() {
            step = step.system(st.system.clone());
        }
        let spec = JobSpec::new(JobKind::Manual, title).hint(st.hint.trim()).step(step);
        st.note = Some(match jobs.submit(spec) {
            Ok(handle) => (true, format!("Queued job #{}", handle.id())),
            Err(e) => (false, e.to_string()),
        });
    }
    if let Some((ok, text)) = &st.note {
        ui.label(RichText::new(text).color(if *ok { ui.visuals().text_color() } else { ui.visuals().error_fg_color }));
    }
}

fn history_section(ui: &mut Ui, snap: &Snapshot, st: &mut State) {
    let now = now_ms();
    if snap.history.is_empty() {
        ui.label(RichText::new("No finished jobs yet").weak());
        return;
    }
    ScrollArea::vertical().id_salt("zc_jobs_history").max_height(190.0).auto_shrink([false, true]).show(ui, |ui| {
        for record in &snap.history {
            let on = st.selected.is_some_and(|(id, _)| id == record.id);
            let title = crate::ui::refine::model::clipped(&record.title, 60);
            if ui.selectable_label(on, format!("#{} {} - {}", record.id, record.kind.label(), title)).on_hover_text(ago(now, record.finished_ms.unwrap_or(record.queued_ms))).clicked() {
                st.selected = Some((record.id, 0));
            }
            let sent = record.prompt_bytes();
            let received = record.response_bytes();
            let summary = format!(
                "{}/{} requests, {} attempts - {} sent (~{} tokens), {} received (~{} tokens) - {} - {}",
                record.steps.len(),
                record.steps_total,
                record.attempts(),
                size(sent),
                crate::zc::jobs::record::est_tokens(sent as u64),
                size(received),
                crate::zc::jobs::record::est_tokens(received as u64),
                record.duration_ms().map_or("not finished".to_string(), millis),
                record.outcome_label()
            );
            ui.label(RichText::new(summary).small().weak());
        }
    });
}

fn step_text(step: &StepRecord, part: Part, mask: bool) -> String {
    let text = match part {
        Part::System => step.system.clone().unwrap_or_default(),
        Part::Prompt => step.prompt.clone(),
        Part::Reply => step.response.clone().or_else(|| step.error.clone()).unwrap_or_default(),
        Part::Parsed => String::new(),
    };
    if mask { mask_paths(&text) } else { text }
}

fn viewer_section(ui: &mut Ui, snap: &Snapshot, st: &mut State) {
    ui.checkbox(&mut st.mask_paths, "Mask file paths").on_hover_text("Shows <path> for anything that looks like a file path. The token is never in the log.");
    let record: Option<Arc<JobRecord>> = st.selected.and_then(|(id, _)| snap.history.iter().find(|r| r.id == id).cloned().or_else(|| snap.running.as_ref().filter(|r| r.id == id).map(|r| r.record.clone())));
    let Some(record) = record else {
        ui.label(RichText::new("Pick a job in the history to read its request and reply").weak());
        return;
    };
    if record.steps.is_empty() {
        ui.label(RichText::new("This job has no finished request").weak());
        return;
    }
    let mut index = st.selected.map_or(0, |(_, i)| i).min(record.steps.len() - 1);
    ui.horizontal_wrapped(|ui| {
        ComboBox::from_id_salt("zc_jobs_step").selected_text(format!("Request {} of {}: {}", index + 1, record.steps_total, record.steps[index].label)).show_ui(ui, |ui| {
            for (i, s) in record.steps.iter().enumerate() {
                ui.selectable_value(&mut index, i, format!("{}: {}", i + 1, s.label));
            }
        });
        for (part, label) in [(Part::System, "System"), (Part::Prompt, "Prompt"), (Part::Reply, "Reply"), (Part::Parsed, "Parsed JSON")] {
            ui.selectable_value(&mut st.part, part, label);
        }
    });
    if let Some((id, _)) = st.selected {
        st.selected = Some((id, index));
    }
    let step = &record.steps[index];
    ui.label(RichText::new(format!("{} in (~{} tok), {} out (~{} tok), {}, {} attempt(s), model {}, JSON {}", size(step.prompt_bytes), step.est_tokens_in(), size(step.response_bytes), step.est_tokens_out(), millis(step.duration_ms), step.attempts, if step.model.is_empty() { "-" } else { step.model.as_str() }, step.json.as_deref().unwrap_or("not expected"))).weak().small());
    if let Some(error) = &step.error {
        ui.label(RichText::new(error).color(ui.visuals().error_fg_color));
    }
    if st.part == Part::Parsed {
        let reply = step.response.clone().unwrap_or_default();
        match crate::zc::jobs::extract(&reply, crate::zc::jobs::Shape::Any) {
            Ok(found) => {
                let value = if st.mask_paths { serde_json::from_str(&mask_paths(&found.value.to_string())).unwrap_or(found.value) } else { found.value };
                ScrollArea::vertical().id_salt("zc_jobs_json").max_height(240.0).show(ui, |ui| {
                    JsonTree::new(format!("zc-job-{}-{index}", record.id), &value).default_expand(DefaultExpand::All).show(ui);
                });
            }
            Err(e) => {
                ui.label(RichText::new(e.to_string()).weak());
            }
        }
        return;
    }
    let mut text = step_text(step, st.part, st.mask_paths);
    ScrollArea::vertical().id_salt("zc_jobs_text").max_height(260.0).show(ui, |ui| {
        ui.add(TextEdit::multiline(&mut text).code_editor().desired_rows(6).desired_width(ui.available_width()));
    });
}

/// Draws the window with the jobs of `jobs` when it is open.
pub fn show_with(ctx: &Context, jobs: &Arc<ZcJobs>) {
    if !is_open(ctx) {
        return;
    }
    jobs.runtime().attach_ctx(ctx);
    let mut open = true;
    Window::new("ZeroClaw jobs").open(&mut open).default_size([640.0, 700.0]).resizable(true).show(ctx, |ui| {
        let snap = jobs.snapshot();
        let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
        ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            CollapsingHeader::new(RichText::new("Status").strong()).default_open(true).show(ui, |ui| status_section(ui, jobs, &snap));
            CollapsingHeader::new(RichText::new("Queue").strong()).default_open(true).show(ui, |ui| running_section(ui, jobs, &snap));
            CollapsingHeader::new(RichText::new("Submit a job").strong()).default_open(false).show(ui, |ui| submit_section(ui, jobs, &mut st));
            CollapsingHeader::new(RichText::new("History").strong()).default_open(true).show(ui, |ui| history_section(ui, &snap, &mut st));
            CollapsingHeader::new(RichText::new("Request and reply").strong()).default_open(true).show(ui, |ui| viewer_section(ui, &snap, &mut st));
        });
    });
    if !open {
        set_open(ctx, false);
    }
}

/// Draws the window of the application's jobs when it is open.
pub fn show(ctx: &Context) {
    if is_open(ctx) {
        show_with(ctx, crate::zc::jobs::jobs());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_and_times_read_as_short_text() {
        assert_eq!((size(512).as_str(), size(20 * 1024).as_str()), ("512 B", "20.0 KiB"));
        assert_eq!((millis(850).as_str(), millis(12_500).as_str()), ("850 ms", "12.5 s"));
    }

    fn draw(ctx: &Context, jobs: &Arc<ZcJobs>, width: f32) -> Vec<(String, egui::Pos2)> {
        let mut drawn = Vec::new();
        for _ in 0..3 {
            let input = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(width, 1200.0))), ..Default::default() };
            let out = ctx.run_ui(input, |ui| show_with(&ui.ctx().clone(), jobs));
            drawn = crate::ui::test_support::texts(&out.shapes);
            out.drop_without_applying_deltas();
        }
        drawn
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn the_window_lists_jobs_and_shows_the_raw_request_with_paths_masked_on_request() {
        use crate::zc::jobs::JobsConfig;
        use crate::zc::runtime_tests::{runtime_for, until};
        let mock = zeroclaw_client::mock::MockGateway::start().await;
        let rt = runtime_for(&mock, "default", "dev-jobs");
        until("online", || rt.health().is_online().then_some(())).await;
        let jobs = ZcJobs::new(rt, JobsConfig::quick());
        mock.script_complete_reply(r#"[{"id":1,"file":"/home/u/Pictures/a.jpg"}]"#);
        let spec = JobSpec::new(JobKind::Manual, "dev-jobs job").step(Step::new("only", "Describe /home/u/Pictures/trip/a.jpg please").expect(crate::zc::jobs::Shape::Rows));
        let handle = jobs.submit(spec).unwrap();
        let id = handle.id();
        handle.wait().await;
        let ctx = crate::ui::test_support::test_ctx();
        set_open(&ctx, true);
        *STATE.lock().unwrap() = State { selected: Some((id, 0)), ..State::default() };
        for width in [700.0, 1000.0, 1600.0] {
            let drawn = draw(&ctx, &jobs, width);
            let has = |needle: &str| drawn.iter().any(|(t, _)| t.contains(needle));
            for expected in ["Gateway online: jobs can run", "Nothing is running", "dev-jobs job", "completed", "Request 1 of 1: only", "Mask file paths", "1 jobs (1 done, 0 failed, 0 cancelled)"] {
                assert!(has(expected), "{width}: missing {expected:?} in {drawn:?}");
            }
            assert!(has("Describe <path> please") && !has("/home/u/Pictures"), "{width}: the path is masked: {drawn:?}");
            assert!(drawn.iter().all(|(_, pos)| pos.x < width), "{width}: nothing starts beyond the right edge");
        }
        STATE.lock().unwrap().mask_paths = false;
        assert!(draw(&ctx, &jobs, 1000.0).iter().any(|(t, _)| t.contains("Describe /home/u/Pictures/trip/a.jpg please")), "unmasked on request");
        STATE.lock().unwrap().part = Part::Reply;
        assert!(draw(&ctx, &jobs, 1000.0).iter().any(|(t, _)| t.contains("\"file\":\"/home/u/Pictures/a.jpg\"")));
        *STATE.lock().unwrap() = State::default();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn the_submit_box_queues_a_manual_job_and_cancel_buttons_stop_jobs() {
        use crate::zc::jobs::JobsConfig;
        use crate::zc::runtime_tests::{runtime_for, until};
        let mock = zeroclaw_client::mock::MockGateway::start().await;
        let rt = runtime_for(&mock, "default", "dev-jobs-submit");
        until("online", || rt.health().is_online().then_some(())).await;
        let jobs = ZcJobs::new(rt, JobsConfig::quick());
        mock.script_complete(200, serde_json::json!({"response": "late", "model": "m", "hint": "quick"}), std::time::Duration::from_secs(30));
        let mut st = State { prompt: "slow manual prompt".into(), ..State::default() };
        let ctx = crate::ui::test_support::test_ctx();
        ctx.run_ui(egui::RawInput::default(), |ui| submit_section(ui, &jobs, &mut st)).drop_without_applying_deltas();
        assert_eq!(st.hint, "quick", "the hint field starts with the profile's");
        let queued = jobs.submit(JobSpec::new(JobKind::Manual, "Manual: slow").step(Step::new("manual", "slow manual prompt").temperature(0.5))).unwrap();
        until("running", || jobs.snapshot().running.map(|_| ())).await;
        let snap = jobs.snapshot();
        assert_eq!(snap.running.as_ref().map(|r| r.id), Some(queued.id()));
        assert_eq!(jobs.cancel_all(), 1);
        until("idle", || jobs.is_idle().then_some(())).await;
        assert_eq!(jobs.snapshot().totals.cancelled, 1);
    }

    #[test]
    fn the_window_opens_toggles_and_draws_its_sections() {
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
        for expected in ["Status", "Queue", "Submit a job", "History", "Request and reply"] {
            assert!(drawn.iter().any(|(t, _)| t == expected), "{expected} in {drawn:?}");
        }
        set_open(&ctx, false);
        assert!(!is_open(&ctx));
    }
}
