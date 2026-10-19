//! A scripted run against a browser tab: ordered steps, each measured over the frames, queries and CPU time it spans.

use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use super::frames::{self, Frame, FrameSpread, SlowFrame};
use super::proc;
use crate::database::scope_queries::counter::{self, Counts, Site};
use crate::ui::file_table::table::view_mode::ViewMode;

/// Longest wait a recording asks for between two frames.
const RECORD_WAKE: Duration = Duration::from_secs(3);
/// Repaint causes a step report names.
const CAUSES_SHOWN: usize = 6;
/// Long frames a step report names.
const SLOWEST: usize = 3;
/// Frames a quick step is measured over.
pub const SETTLE_FRAMES: usize = 30;
/// Shortest time a quick step is measured over.
const SETTLE_MIN: Duration = Duration::from_millis(400);
/// Longest time a quick step waits for its frames.
const SETTLE_MAX: Duration = Duration::from_secs(6);
/// Longest time a scope load is waited for.
const LOAD_TIMEOUT: Duration = Duration::from_secs(300);
/// Longest time a scope name is waited for.
const RESOLVE_TIMEOUT: Duration = Duration::from_secs(30);
/// Seconds the scroll step takes from the first row to the last.
pub const SCROLL_SECS: f32 = 10.0;
/// Seconds the idle step measures.
pub const IDLE_SECS: f32 = 30.0;

/// What a run measures.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Suite {
    /// Scroll the shown rows from the first to the last.
    #[default]
    Scroll,
    /// Resize, sort, switch view, filter and select in the shown rows.
    Ui,
    /// Open each scope and wait for its rows.
    Scopes,
    /// Measure the application for a while without touching it.
    Record,
    /// Every suite in order.
    All,
}

impl Suite {
    pub fn label(self) -> &'static str {
        match self {
            Self::Scroll => "scroll",
            Self::Ui => "ui",
            Self::Scopes => "scopes",
            Self::Record => "record",
            Self::All => "all",
        }
    }
}

/// A column the run sorts by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SortBy {
    Name,
    Size,
    Modified,
}

/// One thing a step does.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    /// Opens a scope string and waits for its rows; `tag:@top`, `cat:@top` and `col:@top` name the biggest tag, category and collection.
    Open(String),
    /// Scrolls from the first row to the last over `secs`.
    Scroll { secs: f32 },
    /// Does nothing for `secs`.
    Record { secs: f32 },
    Tile(f32),
    Mode(ViewMode),
    Sort(Option<(SortBy, bool)>),
    Filter(serde_json::Value),
    ClearFilters,
    SelectAll,
    ClearSelection,
    TaskCenter(bool),
}

impl Op {
    fn describe(&self) -> String {
        match self {
            Self::Open(scope) => format!("open {scope}"),
            Self::Scroll { secs } => format!("scroll over {secs} s"),
            Self::Record { secs } => format!("record {secs} s"),
            Self::Tile(px) => format!("tiles {px} px"),
            Self::Mode(mode) => format!("{} view", mode.label()),
            Self::Sort(Some((by, down))) => format!("sort {by:?} {}", if *down { "descending" } else { "ascending" }),
            Self::Sort(None) => "sort off".to_string(),
            Self::Filter(patch) => format!("filter {patch}"),
            Self::ClearFilters => "clear filters".to_string(),
            Self::SelectAll => "select all".to_string(),
            Self::ClearSelection => "clear selection".to_string(),
            Self::TaskCenter(open) => format!("task center {}", if *open { "open" } else { "closed" }),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    pub label: String,
    pub op: Op,
}

fn step(label: impl Into<String>, op: Op) -> Step {
    Step { label: label.into(), op }
}

/// Scopes the scope suite opens when none are named.
pub const DEFAULT_SCOPES: [&str; 8] = ["lib:all", "recent:all", "smart:undescribed", "smart:untagged", "tag:@top", "cat:@top", "col:@top", "smart:duplicates"];

fn scope_steps(scopes: &[String]) -> Vec<Step> {
    scopes.iter().map(|s| step(format!("load {s}"), Op::Open(s.clone()))).collect()
}

fn scroll_steps(secs: f32) -> Vec<Step> {
    vec![
        step("scroll grid", Op::Scroll { secs }),
        step("mode details", Op::Mode(ViewMode::Details)),
        step("scroll details", Op::Scroll { secs }),
        step("mode grid", Op::Mode(ViewMode::Grid)),
        step("scroll grid again", Op::Scroll { secs }),
        step("mode details again", Op::Mode(ViewMode::Details)),
        step("scroll details again", Op::Scroll { secs }),
        step("mode grid again", Op::Mode(ViewMode::Grid)),
        step("scroll grid third", Op::Scroll { secs }),
    ]
}

fn ui_steps() -> Vec<Step> {
    let mut steps = vec![step("tiles 96", Op::Tile(96.0)), step("tiles 256", Op::Tile(256.0)), step("tiles 160", Op::Tile(160.0))];
    for (name, by, down) in [("name", SortBy::Name, false), ("size", SortBy::Size, true), ("modified", SortBy::Modified, false)] {
        steps.push(step(format!("sort {name}"), Op::Sort(Some((by, down)))));
    }
    steps.push(step("sort off", Op::Sort(None)));
    steps.push(step("mode details", Op::Mode(ViewMode::Details)));
    steps.push(step("mode grid", Op::Mode(ViewMode::Grid)));
    steps.push(step("chip videos off", Op::Filter(serde_json::json!({"types": {"videos": false}}))));
    steps.push(step("chip images off", Op::Filter(serde_json::json!({"types": {"images": false}}))));
    steps.push(step("chip filters cleared", Op::ClearFilters));
    steps.push(step("select all", Op::SelectAll));
    steps.push(step("task center open", Op::TaskCenter(true)));
    steps.push(step("task center closed", Op::TaskCenter(false)));
    steps.push(step("selection cleared", Op::ClearSelection));
    steps
}

/// The steps of `suite`; `scopes` replaces the default scope list and `secs` the scroll duration.
pub fn plan(suite: Suite, scopes: &[String], secs: Option<f32>) -> Vec<Step> {
    let scroll = secs.filter(|s| s.is_finite() && *s > 0.0).unwrap_or(SCROLL_SECS);
    let named: Vec<String> = if scopes.is_empty() { DEFAULT_SCOPES.iter().map(|s| s.to_string()).collect() } else { scopes.to_vec() };
    // Scopes named for a suite that measures a loaded tab are opened first.
    let opened_first = |steps: Vec<Step>| {
        let mut all = scope_steps(scopes);
        all.extend(steps);
        all
    };
    match suite {
        Suite::Scroll => opened_first(scroll_steps(scroll)),
        Suite::Ui => opened_first(ui_steps()),
        Suite::Scopes => scope_steps(&named),
        Suite::Record => opened_first(vec![step("record", Op::Record { secs: secs.filter(|s| s.is_finite() && *s > 0.0).unwrap_or(IDLE_SECS) })]),
        Suite::All => {
            let mut steps = scope_steps(&named);
            steps.push(step("task center open", Op::TaskCenter(true)));
            steps.push(step("reload lib:all with task center", Op::Open("lib:all".to_string())));
            steps.push(step("task center closed", Op::TaskCenter(false)));
            steps.push(step("reload lib:all", Op::Open("lib:all".to_string())));
            steps.extend(scroll_steps(scroll));
            steps.extend(ui_steps());
            steps.push(step("idle", Op::Record { secs: IDLE_SECS }));
            steps
        }
    }
}

/// How a scope load went, as the tab logged it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LoadSummary {
    pub scope: String,
    pub rows: usize,
    pub batches: usize,
    pub first_batch_ms: Option<f32>,
    pub elapsed_ms: f32,
    /// Time spent reading pages of rows.
    pub page_ms: f32,
    /// Time spent checking CLIP presence.
    pub embed_ms: f32,
    pub total: Option<u64>,
    pub error: Option<String>,
    pub cancelled: bool,
}

/// The texture and encoded caches of the thumbnail service at one moment.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct ThumbMeter {
    pub texture_mb: f32,
    pub budget_mb: f32,
    pub textures: usize,
    pub encoded_mb: f32,
    pub evictions: u64,
    pub hits: u64,
    pub decode_started: u64,
    pub queued: usize,
}

impl ThumbMeter {
    pub fn read() -> Self {
        const MB: f32 = 1024.0 * 1024.0;
        let s = crate::ui::file_table::thumbs::service().snapshot();
        Self {
            texture_mb: s.used as f32 / MB,
            budget_mb: s.budget as f32 / MB,
            textures: s.entries,
            encoded_mb: s.recent_bytes as f32 / MB,
            evictions: s.cache.evictions,
            hits: s.cache.hits,
            decode_started: s.core.decode_started,
            queued: s.depths.queued,
        }
    }
}

/// Process meters at one moment.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Meters {
    pub thumbs: Option<ThumbMeter>,
    /// Heap and stack memory, and memory of mapped files, in bytes.
    pub rss_split: Option<(u64, u64)>,
    pub rss: Option<u64>,
    pub peak_rss: Option<u64>,
    pub cpu_secs: Option<f64>,
    pub queries: Counts,
    /// Frames drawn so far per place that asked for a repaint.
    pub causes: BTreeMap<String, u64>,
}

impl Meters {
    pub fn read() -> Self {
        Self { rss_split: proc::rss_split(), rss: proc::rss_bytes(), peak_rss: proc::peak_rss_bytes(), cpu_secs: proc::cpu_seconds(), queries: counter::snapshot(), causes: super::repaint_causes(), thumbs: Some(ThumbMeter::read()) }
    }

    /// Repaints per cause since `earlier`, the most frequent first.
    pub fn causes_since(&self, earlier: &Meters, n: usize) -> Vec<(String, u64)> {
        let mut delta: Vec<(String, u64)> = self.causes.iter().map(|(k, v)| (k.clone(), v.saturating_sub(earlier.causes.get(k).copied().unwrap_or(0)))).filter(|(_, v)| *v > 0).collect();
        delta.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        delta.truncate(n);
        delta
    }
}

/// What a scope token turned into.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Resolved {
    Ready(String),
    Pending,
    Missing,
}

/// The tab a run acts on.
pub trait Target {
    fn view_rows(&self) -> usize;
    fn selected(&self) -> usize;
    fn is_loading(&self) -> bool;
    fn open(&mut self, scope: &str) -> Result<(), String>;
    /// The newest finished load of the tab.
    fn last_load(&self) -> Option<LoadSummary>;
    fn scroll_to(&mut self, fraction: f32);
    fn set_tile(&mut self, px: f32);
    fn set_mode(&mut self, mode: ViewMode);
    fn sort(&mut self, by: Option<(SortBy, bool)>);
    fn filter(&mut self, patch: &serde_json::Value) -> Result<(), String>;
    fn clear_filters(&mut self);
    fn select_all(&mut self);
    fn clear_selection(&mut self);
    fn task_center(&mut self, open: bool);
    /// A scope containing `@top`, with the name filled in.
    fn resolve(&mut self, scope: &str) -> Resolved;
    fn meters(&self) -> Meters {
        Meters::read()
    }
    /// Number the next drawn frame gets.
    fn frame_mark(&self) -> u64 {
        frames::mark()
    }
    /// Frames drawn since `mark`.
    fn frames_since(&self, mark: u64) -> Vec<Frame> {
        frames::since(mark)
    }
}

/// What the host should do for the next frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Repaint {
    Now,
    After(Duration),
    Never,
}

/// What a step measured.
#[derive(Clone, Debug, Serialize)]
pub struct StepReport {
    pub label: String,
    pub op: String,
    pub seconds: f32,
    pub frames: usize,
    pub fps: f32,
    pub spread: Option<FrameSpread>,
    /// The longest frames of the step.
    pub slowest: Vec<SlowFrame>,
    pub queries: u64,
    pub queries_by_site: BTreeMap<&'static str, u64>,
    /// Who asked for the frames of the step: file, line and reason with the frame count.
    pub repaint_causes: Vec<(String, u64)>,
    pub cpu_percent: Option<f32>,
    pub rss_mb: Option<f32>,
    /// Heap and stack part of the resident memory.
    pub rss_anon_mb: Option<f32>,
    /// Part of the resident memory that maps files, such as the database store.
    pub rss_file_mb: Option<f32>,
    pub peak_rss_mb: Option<f32>,
    pub view_rows: usize,
    pub selected: usize,
    /// The thumbnail caches when the step ended.
    pub thumbs: Option<ThumbMeter>,
    /// Textures dropped from the cache during the step.
    pub evictions: u64,
    /// Thumbnails the service decoded during the step.
    pub decoded: u64,
    /// The scope an open step showed, with `@top` names filled in.
    pub scope: Option<String>,
    pub load: Option<LoadSummary>,
    pub note: Option<String>,
}

/// What a whole run measured.
#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub suite: Suite,
    pub seconds: f32,
    pub rss_start_mb: Option<f32>,
    pub rss_end_mb: Option<f32>,
    pub peak_rss_mb: Option<f32>,
    /// Start-up milestones in seconds since the process started.
    pub marks: Vec<(String, f32)>,
    /// Lookups the run waited on, in milliseconds.
    pub timings: Vec<(String, f32)>,
    pub steps: Vec<StepReport>,
}

impl Report {
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// One line per step for the log.
    pub fn to_text(&self) -> String {
        let mut out = format!("scale {} run: {:.1} s\n", self.suite.label(), self.seconds);
        for s in &self.steps {
            let dt = s.spread.map_or("no frames".to_string(), |f| format!("dt avg {:.1} p95 {:.1} max {:.1} ms", f.dt.avg, f.dt.p95, f.dt.max));
            out.push_str(&format!("  {:<24} {:>5} frames  {dt}  {} queries\n", s.label, s.frames, s.queries));
        }
        out
    }

    pub fn step(&self, label: &str) -> Option<&StepReport> {
        self.steps.iter().find(|s| s.label == label)
    }
}

fn delta_of(after: Option<ThumbMeter>, before: Option<ThumbMeter>, pick: impl Fn(&ThumbMeter) -> u64) -> u64 {
    match (after, before) {
        (Some(a), Some(b)) => pick(&a).saturating_sub(pick(&b)),
        _ => 0,
    }
}

fn mb(bytes: Option<u64>) -> Option<f32> {
    bytes.map(|b| b as f32 / (1024.0 * 1024.0))
}

struct Active {
    step: Step,
    began: Instant,
    mark: u64,
    before: Meters,
    opened: bool,
    saw_loading: bool,
    quiet_since: Option<usize>,
    scope: Option<String>,
    note: Option<String>,
}

/// Where a run is, for display.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Progress {
    pub step: usize,
    pub steps: usize,
    pub label: String,
    pub elapsed: f32,
    /// The running step measures idle behaviour.
    pub quiet: bool,
}

/// A run in progress.
pub struct Bench {
    suite: Suite,
    queue: VecDeque<Step>,
    total: usize,
    active: Option<Active>,
    done: Vec<StepReport>,
    started: Instant,
    start_meters: Meters,
}

/// How a tick left the run.
pub enum Tick {
    Running(Repaint),
    Done(Box<Report>),
}

impl Bench {
    pub fn new(suite: Suite, steps: Vec<Step>, now: Instant) -> Self {
        Self { suite, total: steps.len(), queue: steps.into(), active: None, done: Vec::new(), started: now, start_meters: Meters::read() }
    }

    pub fn suite(&self) -> Suite {
        self.suite
    }

    pub fn progress(&self, now: Instant) -> Progress {
        Progress {
            step: self.done.len() + usize::from(self.active.is_some()),
            steps: self.total,
            label: self.active.as_ref().map(|a| a.step.label.clone()).unwrap_or_default(),
            elapsed: now.saturating_duration_since(self.started).as_secs_f32(),
            quiet: self.active.as_ref().is_some_and(|a| matches!(a.step.op, Op::Record { .. })),
        }
    }

    /// Advances the run by one frame: starts the next step, drives the current one and finishes it when it has its frames.
    pub fn tick(&mut self, target: &mut dyn Target, now: Instant) -> Tick {
        loop {
            if self.active.is_none() {
                let Some(next) = self.queue.pop_front() else {
                    return Tick::Done(Box::new(self.finish(target, now)));
                };
                self.begin(next, target, now);
            }
            match self.drive(target, now) {
                Some(report) => self.done.push(report),
                None => break,
            }
        }
        let repaint = match self.active.as_ref().map(|a| &a.step.op) {
            Some(Op::Record { secs }) => {
                let left = Duration::from_secs_f32(*secs).saturating_sub(now.saturating_duration_since(self.active.as_ref().map_or(now, |a| a.began)));
                Repaint::After(left.clamp(Duration::from_millis(50), RECORD_WAKE))
            }
            Some(Op::Open(_)) => Repaint::After(Duration::from_millis(40)),
            _ => Repaint::Now,
        };
        Tick::Running(repaint)
    }

    fn begin(&mut self, step: Step, target: &mut dyn Target, now: Instant) {
        self.active = Some(Active { step, began: now, mark: target.frame_mark(), before: target.meters(), opened: false, saw_loading: false, quiet_since: None, scope: None, note: None });
    }

    // Applies and watches the active step; the report when it is complete.
    fn drive(&mut self, target: &mut dyn Target, now: Instant) -> Option<StepReport> {
        let active = self.active.as_mut()?;
        let elapsed = now.saturating_duration_since(active.began);
        let seen = target.frames_since(active.mark).len();
        let mut finished = false;
        match active.step.op.clone() {
            Op::Open(scope) => {
                if !active.opened {
                    match resolve(target, &scope, elapsed) {
                        Resolved::Pending => return None,
                        Resolved::Missing => {
                            active.note = Some(format!("{scope} has nothing to open in this library"));
                            return Some(self.complete(target, now, true));
                        }
                        Resolved::Ready(name) => {
                            if let Err(e) = target.open(&name) {
                                active.note = Some(format!("could not open {name}: {e}"));
                                return Some(self.complete(target, now, true));
                            }
                            active.opened = true;
                            active.scope = Some(name);
                            active.mark = target.frame_mark();
                            return None;
                        }
                    }
                }
                let loading = target.is_loading();
                active.saw_loading |= loading;
                let started = active.saw_loading || seen >= 3 || elapsed > Duration::from_secs(1);
                if started && !loading {
                    let quiet = *active.quiet_since.get_or_insert(seen);
                    finished = seen >= quiet + 3;
                } else {
                    active.quiet_since = None;
                }
                if elapsed >= LOAD_TIMEOUT {
                    active.note = Some(format!("timed out after {} s", LOAD_TIMEOUT.as_secs()));
                    finished = true;
                }
            }
            Op::Scroll { secs } => {
                let secs = secs.max(0.1);
                target.scroll_to((elapsed.as_secs_f32() / secs).clamp(0.0, 1.0));
                finished = elapsed.as_secs_f32() >= secs;
            }
            Op::Record { secs } => finished = elapsed.as_secs_f32() >= secs,
            other => {
                if !active.opened {
                    active.opened = true;
                    active.mark = target.frame_mark();
                    if let Err(e) = apply(target, &other) {
                        active.note = Some(e);
                        return Some(self.complete(target, now, true));
                    }
                    return None;
                }
                finished = (seen >= SETTLE_FRAMES && elapsed >= SETTLE_MIN) || elapsed >= SETTLE_MAX;
            }
        }
        finished.then(|| self.complete(target, now, false))
    }

    // Closes the active step into a report.
    fn complete(&mut self, target: &mut dyn Target, now: Instant, skipped: bool) -> StepReport {
        let active = self.active.take().expect("a step is active");
        let after = target.meters();
        let seconds = now.saturating_duration_since(active.began).as_secs_f32();
        let frames = if skipped { Vec::new() } else { target.frames_since(active.mark) };
        let delta = after.queries.since(&active.before.queries);
        let by_site = Site::ALL.iter().filter(|s| delta.of(**s) > 0).map(|s| (s.label(), delta.of(*s))).collect();
        let cpu = match (active.before.cpu_secs, after.cpu_secs) {
            (Some(a), Some(b)) if seconds > 0.0 => Some(((b - a) / seconds as f64 * 100.0) as f32),
            _ => None,
        };
        let load = matches!(active.step.op, Op::Open(_)).then(|| target.last_load()).flatten();
        StepReport {
            label: active.step.label,
            op: active.step.op.describe(),
            seconds,
            frames: frames.len(),
            fps: if seconds > 0.0 { frames.len() as f32 / seconds } else { 0.0 },
            spread: FrameSpread::of(&frames),
            slowest: frames::slowest(&frames, SLOWEST),
            queries: delta.total(),
            queries_by_site: by_site,
            repaint_causes: after.causes_since(&active.before, CAUSES_SHOWN),
            cpu_percent: cpu,
            rss_mb: mb(after.rss),
            rss_anon_mb: mb(after.rss_split.map(|s| s.0)),
            rss_file_mb: mb(after.rss_split.map(|s| s.1)),
            peak_rss_mb: mb(after.peak_rss),
            view_rows: target.view_rows(),
            selected: target.selected(),
            thumbs: after.thumbs,
            evictions: delta_of(after.thumbs, active.before.thumbs, |t| t.evictions),
            decoded: delta_of(after.thumbs, active.before.thumbs, |t| t.decode_started),
            scope: active.scope,
            load,
            note: active.note,
        }
    }

    fn finish(&mut self, target: &mut dyn Target, now: Instant) -> Report {
        let end = target.meters();
        Report {
            suite: self.suite,
            seconds: now.saturating_duration_since(self.started).as_secs_f32(),
            rss_start_mb: mb(self.start_meters.rss),
            rss_end_mb: mb(end.rss),
            peak_rss_mb: mb(end.peak_rss),
            marks: super::marks::all(),
            timings: super::timings(),
            steps: std::mem::take(&mut self.done),
        }
    }
}

// Names the scope of an open step, giving the lookup of `@top` names time to finish.
fn resolve(target: &mut dyn Target, scope: &str, waited: Duration) -> Resolved {
    if !scope.contains("@top") {
        return Resolved::Ready(scope.to_string());
    }
    match target.resolve(scope) {
        Resolved::Pending if waited >= RESOLVE_TIMEOUT => Resolved::Missing,
        other => other,
    }
}

fn apply(target: &mut dyn Target, op: &Op) -> Result<(), String> {
    match op {
        Op::Tile(px) => target.set_tile(*px),
        Op::Mode(mode) => target.set_mode(*mode),
        Op::Sort(by) => target.sort(*by),
        Op::Filter(patch) => return target.filter(patch),
        Op::ClearFilters => target.clear_filters(),
        Op::SelectAll => target.select_all(),
        Op::ClearSelection => target.clear_selection(),
        Op::TaskCenter(open) => target.task_center(*open),
        Op::Open(_) | Op::Scroll { .. } | Op::Record { .. } => {}
    }
    Ok(())
}

#[cfg(test)]
#[path = "bench_tests.rs"]
mod tests;
