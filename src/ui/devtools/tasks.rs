//! Devtools window listing the task registry, with buttons that start, advance, fail, complete and cancel fake tasks.

use std::time::{Duration, Instant};

use eframe::egui::{Button, Context, DragValue, Grid, Id, ProgressBar, RichText, ScrollArea, TextEdit, Ui, Window};

use crate::command::{AppCommand, send};
use crate::icons::{self, svg};
use crate::tasks::{self, Task, TaskEvent, TaskId, TaskKind, format_elapsed};

/// How often simulated tasks report progress.
const SIM_TICK: Duration = Duration::from_millis(100);

fn open_id() -> Id {
    Id::new("ame_devtools_tasks_open")
}

fn state_id() -> Id {
    Id::new("ame_devtools_tasks_state")
}

pub fn is_open(ctx: &Context) -> bool {
    ctx.data(|d| d.get_temp::<bool>(open_id())).unwrap_or(false)
}

pub fn set_open(ctx: &Context, open: bool) {
    ctx.data_mut(|d| d.insert_temp(open_id(), open));
}

/// A fake task that advances on its own.
#[derive(Clone, Debug, PartialEq)]
pub struct Sim {
    pub id: TaskId,
    pub started: Instant,
    pub duration: Duration,
    pub total: u64,
    pub fail_at: Option<f32>,
    pub last_tick: Option<Instant>,
}

/// Where a simulated task is at `now`: the commands to send for it, or `None` when it is over.
pub fn sim_step(sim: &Sim, now: Instant) -> (Vec<TaskEvent>, bool) {
    let share = (now.saturating_duration_since(sim.started).as_secs_f32() / sim.duration.as_secs_f32().max(0.001)).clamp(0.0, 1.0);
    let mut events = vec![TaskEvent::Progress { done: (share * sim.total as f32) as u64, total: sim.total }];
    let mut finished = false;
    match sim.fail_at {
        Some(at) if share >= at => {
            events.push(TaskEvent::Fail { error: "Simulated failure".to_string() });
            finished = true;
        }
        _ if share >= 1.0 => {
            events.push(TaskEvent::Complete);
            finished = true;
        }
        _ => {}
    }
    (events, finished)
}

#[derive(Clone)]
struct State {
    title: String,
    kind: TaskKind,
    total: u64,
    seconds: f32,
    sims: Vec<Sim>,
}

impl Default for State {
    fn default() -> Self {
        Self { title: "Fake task".to_string(), kind: TaskKind::Debug, total: 100, seconds: 10.0, sims: Vec::new() }
    }
}

fn begin(kind: TaskKind, title: &str, total: u64) {
    send(AppCommand::TaskUpdate { id: 0, event: TaskEvent::Begin { kind, title: title.to_string(), total } });
}

fn event(id: TaskId, event: TaskEvent) {
    send(AppCommand::TaskUpdate { id, event });
}

/// Starts the demo set: three tasks of different kinds, one of which fails half way.
pub fn start_demo_set(ctx: &Context) {
    let mut state = ctx.data(|d| d.get_temp::<State>(state_id())).unwrap_or_default();
    let now = Instant::now();
    for (kind, title, seconds, total, fail_at) in [
        (TaskKind::Scan, "Demo scan", 40.0, 100, None),
        (TaskKind::Describe, "Demo describe", 60.0, 340, None),
        (TaskKind::Clip, "Demo CLIP", 30.0, 50, Some(0.6)),
    ] {
        let id = tasks::with(|r| r.begin_at(tasks::TaskSpec::new(kind, title).total(total).user_initiated().scripted(), now));
        state.sims.push(Sim { id, started: now, duration: Duration::from_secs_f32(seconds), total, fail_at, last_tick: None });
    }
    ctx.data_mut(|d| d.insert_temp(state_id(), state));
    ctx.request_repaint();
}

fn advance_sims(ctx: &Context, state: &mut State) {
    let now = Instant::now();
    state.sims.retain_mut(|sim| {
        let gone = !tasks::with(|r| r.get(sim.id).is_some_and(|t| t.state.is_active()));
        if gone {
            return false;
        }
        if sim.last_tick.is_some_and(|t| now.saturating_duration_since(t) < SIM_TICK) {
            return true;
        }
        sim.last_tick = Some(now);
        let (events, finished) = sim_step(sim, now);
        for e in events {
            event(sim.id, e);
        }
        !finished
    });
    if !state.sims.is_empty() {
        ctx.request_repaint_after(SIM_TICK);
    }
}

fn row(ui: &mut Ui, task: &Task, now: Instant) {
    ui.label(task.id.to_string());
    ui.label(task.kind.label());
    ui.label(format!("{:?}", task.state));
    ui.label(&task.title);
    match task.fraction() {
        Some(f) => ui.add(ProgressBar::new(f).desired_width(80.0).text(format!("{}/{}", task.done, task.total))),
        None => ui.label("-"),
    };
    ui.label(format_elapsed(task.elapsed(now)));
    ui.label(task.source.unwrap_or("manual"));
    ui.horizontal(|ui| {
        let active = task.state.is_active();
        if ui.add_enabled(active, Button::new("+10%")).clicked() {
            let total = task.total.max(100);
            event(task.id, TaskEvent::Progress { done: (task.done + total / 10).min(total), total });
        }
        if ui.add_enabled(active, Button::new("Complete")).clicked() {
            event(task.id, TaskEvent::Complete);
        }
        if ui.add_enabled(active, Button::new("Fail")).clicked() {
            event(task.id, TaskEvent::Fail { error: "Failed from devtools".to_string() });
        }
        if ui.add_enabled(active && task.cancel.is_some(), Button::new("Cancel")).clicked() {
            event(task.id, TaskEvent::Cancel);
        }
        if ui.add_enabled(!active, Button::new("Remove")).clicked() {
            tasks::with(|r| r.dismiss(task.id));
        }
    });
    ui.end_row();
}

fn body(ctx: &Context, ui: &mut Ui) {
    let mut state = ctx.data(|d| d.get_temp::<State>(state_id())).unwrap_or_default();
    advance_sims(ctx, &mut state);

    ui.horizontal(|ui| {
        ui.add(TextEdit::singleline(&mut state.title).desired_width(140.0));
        egui::ComboBox::from_id_salt("ame_devtools_tasks_kind").selected_text(state.kind.label()).show_ui(ui, |ui| {
            for kind in TaskKind::ALL {
                ui.selectable_value(&mut state.kind, kind, kind.label());
            }
        });
        ui.add(DragValue::new(&mut state.total).range(0..=100_000).prefix("total "));
        ui.add(DragValue::new(&mut state.seconds).range(1.0..=600.0).suffix(" s"));
    });
    let mut demo = false;
    ui.horizontal(|ui| {
        if icons::button(ui, svg::PLUS_CIRCLE, "Add fake task").on_hover_text("Starts a task that waits for the buttons below").clicked() {
            begin(state.kind, &state.title, state.total);
        }
        if icons::button(ui, svg::PLAY, "Add running task").on_hover_text("Starts a task that advances on its own").clicked() {
            let now = Instant::now();
            let (kind, title, total) = (state.kind, state.title.clone(), state.total);
            let id = tasks::with(|r| r.begin_at(tasks::TaskSpec::new(kind, title).total(total).user_initiated().scripted(), now));
            state.sims.push(Sim { id, started: now, duration: Duration::from_secs_f32(state.seconds), total: total.max(1), fail_at: None, last_tick: None });
            ctx.request_repaint();
        }
        demo = ui.button("Demo set").on_hover_text("Scan, describe and a CLIP run that fails half way").clicked();
        if ui.button("Clear finished").clicked() {
            tasks::with(|r| r.clear_finished());
        }
        if ui.button("Open Task Center").clicked() {
            send(AppCommand::ShowTaskCenter);
        }
    });
    ctx.data_mut(|d| d.insert_temp(state_id(), state));
    if demo {
        start_demo_set(ctx);
    }

    let (counts, revision, attention, list) = tasks::with(|r| (r.counts(), r.revision(), r.attention(), r.tasks().to_vec()));
    ui.separator();
    ui.label(format!(
        "{} running, {} queued, {} finished ({} failed), {} need attention. Revision {revision}.",
        counts.running, counts.queued, counts.finished, counts.failed, counts.attention
    ));
    for item in &attention {
        ui.label(RichText::new(format!("Attention: {} - {}", item.title, item.detail)).weak());
    }
    let now = Instant::now();
    ScrollArea::both().auto_shrink([false, false]).show(ui, |ui| {
        Grid::new("ame_devtools_tasks_grid").striped(true).num_columns(8).show(ui, |ui| {
            for head in ["Id", "Kind", "State", "Title", "Progress", "Elapsed", "Source", "Actions"] {
                ui.label(RichText::new(head).strong());
            }
            ui.end_row();
            for task in list.iter().rev() {
                row(ui, task, now);
            }
        });
    });
}

/// Draws the window when open.
pub fn show(ctx: &Context) {
    let mut open = is_open(ctx);
    if !open {
        return;
    }
    Window::new("Devtools: Tasks").open(&mut open).default_width(760.0).default_height(420.0).show(ctx, |ui| body(ctx, ui));
    set_open(ctx, open);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sim(duration_s: f32, fail_at: Option<f32>) -> Sim {
        Sim { id: 1, started: Instant::now(), duration: Duration::from_secs_f32(duration_s), total: 200, fail_at, last_tick: None }
    }

    #[test]
    fn a_simulated_task_reports_its_share_and_completes_at_the_end() {
        let s = sim(10.0, None);
        let (events, finished) = sim_step(&s, s.started + Duration::from_secs(5));
        assert_eq!(events, vec![TaskEvent::Progress { done: 100, total: 200 }]);
        assert!(!finished);
        let (events, finished) = sim_step(&s, s.started + Duration::from_secs(11));
        assert_eq!(events, vec![TaskEvent::Progress { done: 200, total: 200 }, TaskEvent::Complete]);
        assert!(finished);
    }

    #[test]
    fn a_simulated_failure_happens_at_its_share() {
        let s = sim(10.0, Some(0.5));
        assert!(!sim_step(&s, s.started + Duration::from_secs(4)).1);
        let (events, finished) = sim_step(&s, s.started + Duration::from_secs(6));
        assert!(finished);
        assert_eq!(events.last(), Some(&TaskEvent::Fail { error: "Simulated failure".into() }));
    }

    #[test]
    fn the_window_draws_rows_for_every_registry_task() {
        let ctx = crate::ui::test_support::test_ctx();
        let id = tasks::start(tasks::TaskSpec::new(TaskKind::Debug, "devtools-window-row").total(10));
        set_open(&ctx, true);
        let mut drawn = false;
        for _ in 0..3 {
            let out = ctx.run_ui(Default::default(), |ui| show(ui.ctx()));
            drawn = crate::ui::test_support::texts(&out.shapes).into_iter().any(|(t, _)| t == "devtools-window-row");
            out.drop_without_applying_deltas();
        }
        id.complete();
        assert!(drawn);
    }
}
