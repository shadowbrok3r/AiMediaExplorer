use std::sync::Arc;
use std::time::{Duration, Instant};

use eframe::egui::epaint::ClippedShape;
use eframe::egui::{Context, Pos2, RawInput, Rect, Ui, vec2};

use super::center::TaskView;
use super::*;
use crate::database::Thumbnail;
use crate::tasks::{AttentionItem, AttentionKind, Outcome, TaskKind, TaskRegistry, TaskSpec};
use crate::ui::test_support::{press_release, test_ctx, text_pos, texts};

fn row(path: &str, kind: &str, size: u64) -> Thumbnail {
    let mut t = Thumbnail::default();
    t.path = path.into();
    t.filename = path.rsplit('/').next().unwrap_or(path).into();
    t.file_type = kind.into();
    t.size = size;
    t
}

fn explorer() -> FileExplorer {
    let mut ex = FileExplorer::new(true);
    for r in [row("/p/a.jpg", "jpg", 100), row("/p/b.png", "png", 200), row("/p/c.mp4", "mp4", 4000), row("/p/sub", "<DIR>", 0)] {
        ex.table.push(r);
    }
    ex
}

fn view_of(r: &TaskRegistry) -> TaskView {
    TaskView { revision: r.revision(), tasks: Arc::new(r.tasks().to_vec()), attention: Arc::new(r.attention()) }
}

/// A registry with running scan, thumbnails, describe, CLIP and refine tasks, a finished failure and one attention item.
fn busy_registry(now: Instant) -> TaskRegistry {
    let mut r = TaskRegistry::new();
    let mut start = |kind, title: &str, done, total| {
        let id = r.begin_at(TaskSpec::new(kind, title).total(total).user_initiated().scripted(), now);
        r.set_progress(id, done, total);
        id
    };
    start(TaskKind::Scan, "Scan: Photos", 420, 1000);
    start(TaskKind::Thumbnails, "Thumbnails", 120, 800);
    start(TaskKind::Describe, "Describe files", 12, 340);
    start(TaskKind::Clip, "CLIP embeddings", 5, 20);
    start(TaskKind::Refine, "Refine proposals", 0, 0);
    let failed = r.begin_at(TaskSpec::new(TaskKind::Database, "Export database to a.surql"), now);
    r.finish_at(failed, Outcome::Failed("disk full".into()), now);
    r.set_attention(
        "test",
        vec![AttentionItem { key: "k".into(), kind: AttentionKind::FailedThumbs, title: "Failed thumbnails".into(), detail: "5 files could not be decoded".into(), count: Some(5), action: None }],
    );
    r
}

fn run_frame(ctx: &Context, width: f32, events: Vec<eframe::egui::Event>, mut f: impl FnMut(&mut Ui)) -> Vec<ClippedShape> {
    let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(width, 600.0))), events, ..Default::default() };
    let out = ctx.run_ui(input, |ui| f(ui));
    let shapes = out.shapes.clone();
    out.drop_without_applying_deltas();
    shapes
}

fn draw(ctx: &Context, width: f32, ex: &mut FileExplorer, owner: &str, view: &TaskView, now: Instant) -> Vec<ClippedShape> {
    run_frame(ctx, width, Vec::new(), |ui| show_with(ui, ex, owner, view, SystemMetrics::default(), now))
}

fn settle(ctx: &Context, width: f32, ex: &mut FileExplorer, owner: &str, view: &TaskView, now: Instant) -> Vec<ClippedShape> {
    (0..3).map(|_| draw(ctx, width, ex, owner, view, now)).last().unwrap()
}

#[tokio::test]
async fn the_bar_never_overlaps_with_several_tasks_at_700_1000_and_1600_px() {
    let ctx = test_ctx();
    let mut ex = explorer();
    let now = Instant::now();
    let view = view_of(&busy_registry(now));
    let mut last_overflowed = usize::MAX;
    for width in [700.0_f32, 1000.0, 1600.0] {
        settle(&ctx, width, &mut ex, "overlap", &view, now);
        let l = last_layout(&ctx, "overlap").expect("the bar records its layout");
        assert!(l.left.right() <= l.right.left() + 0.5, "{width}px: left {:?} overlaps right {:?}", l.left, l.right);
        assert!(l.left.left() >= -0.5 && l.right.right() <= width + 0.5, "{width}px: the bar leaves the window: {l:?}");
        assert!(l.shown >= 2, "{width}px: at least the count and the task button stay visible");
        assert!(l.overflowed <= last_overflowed, "{width}px: a wider window collapsed more items");
        last_overflowed = l.overflowed;
    }
    assert_eq!(last_overflowed, 0, "everything fits at 1600px");
}

#[tokio::test]
async fn the_bar_survives_extreme_widths_without_leaving_the_window() {
    let ctx = test_ctx();
    let mut ex = explorer();
    let now = Instant::now();
    let view = view_of(&busy_registry(now));
    for width in [420.0_f32, 520.0, 640.0] {
        settle(&ctx, width, &mut ex, "narrow", &view, now);
        let l = last_layout(&ctx, "narrow").unwrap();
        assert!(l.left.right() <= l.right.left() + 0.5, "{width}px: {l:?}");
    }
    settle(&ctx, 420.0, &mut ex, "narrow", &view, now);
    assert!(last_layout(&ctx, "narrow").unwrap().overflowed > 0, "items collapse into the overflow menu below the supported width");
}

#[tokio::test]
async fn the_bar_shows_counts_activity_chips_and_the_task_button() {
    let ctx = test_ctx();
    let mut ex = explorer();
    let now = Instant::now();
    let view = view_of(&busy_registry(now));
    let shapes = settle(&ctx, 1600.0, &mut ex, "texts", &view, now);
    let all: Vec<String> = texts(&shapes).into_iter().map(|(t, _)| t).collect();
    let has = |needle: &str| all.iter().any(|t| t == needle);
    assert!(all.iter().any(|t| t.ends_with("items")), "item count: {all:?}");
    assert!(has("Tasks (5)"), "{all:?}");
    assert!(has("Scan 42%"), "{all:?}");
    assert!(has("Thumbnails 120/800"), "{all:?}");
    assert!(has("Describe 12/340"), "{all:?}");
    assert!(has("CLIP 5/20"), "{all:?}");
    assert!(has("Refine"), "{all:?}");
    assert!(has("1 need attention"), "{all:?}");
    assert!(has("Logs"), "{all:?}");
}

#[tokio::test]
async fn an_idle_bar_has_no_chips_and_a_plain_task_button() {
    let ctx = test_ctx();
    let mut ex = explorer();
    let now = Instant::now();
    let view = view_of(&TaskRegistry::new());
    let shapes = settle(&ctx, 1000.0, &mut ex, "idle", &view, now);
    let all: Vec<String> = texts(&shapes).into_iter().map(|(t, _)| t).collect();
    assert!(all.iter().any(|t| t == "Tasks"));
    assert!(!all.iter().any(|t| t.contains("need attention") || t.starts_with("Scan") || t.starts_with("CLIP")));
}

#[tokio::test]
async fn a_young_database_task_is_not_shown() {
    let ctx = test_ctx();
    let mut ex = explorer();
    let now = Instant::now();
    let mut r = TaskRegistry::new();
    r.begin_at(TaskSpec::new(TaskKind::Database, "Database").visible_after(Duration::from_millis(800)), now);
    let view = view_of(&r);
    let early = texts(&settle(&ctx, 1000.0, &mut ex, "young", &view, now)).into_iter().map(|(t, _)| t).collect::<Vec<_>>();
    assert!(!early.iter().any(|t| t.starts_with("Database")), "{early:?}");
    let later = texts(&settle(&ctx, 1000.0, &mut ex, "young", &view, now + Duration::from_secs(1))).into_iter().map(|(t, _)| t).collect::<Vec<_>>();
    assert!(later.iter().any(|t| t.starts_with("Database")), "{later:?}");
}

#[tokio::test]
async fn stats_are_recomputed_only_when_the_view_changes() {
    let ctx = test_ctx();
    let mut ex = explorer();
    let now = Instant::now();
    let view = view_of(&TaskRegistry::new());
    for _ in 0..8 {
        draw(&ctx, 1000.0, &mut ex, "cache", &view, now);
    }
    assert_eq!(stats_computes(&ctx, "cache"), 1, "idle frames reuse the cached stats");
    ex.table.push(row("/p/new.jpg", "jpg", 5));
    for _ in 0..3 {
        draw(&ctx, 1000.0, &mut ex, "cache", &view, now);
    }
    assert_eq!(stats_computes(&ctx, "cache"), 2, "a new row is one recompute");
    ex.sync_view();
    ex.select_paths(["/p/new.jpg"]);
    for _ in 0..3 {
        draw(&ctx, 1000.0, &mut ex, "cache", &view, now);
    }
    assert_eq!(stats_computes(&ctx, "cache"), 3, "a selection change is one recompute");
}

#[tokio::test]
async fn the_selection_appears_in_the_bar_with_its_size() {
    let ctx = test_ctx();
    let mut ex = explorer();
    ex.sync_view();
    ex.select_paths(["/p/c.mp4"]);
    let view = view_of(&TaskRegistry::new());
    let shapes = settle(&ctx, 1200.0, &mut ex, "selection", &view, Instant::now());
    let all: Vec<String> = texts(&shapes).into_iter().map(|(t, _)| t).collect();
    assert!(all.iter().any(|t| t == "1 selected (4 kB)"), "{all:?}");
}

#[tokio::test]
async fn the_bar_hides_meters_the_platform_cannot_provide() {
    let none = SystemMetrics::default();
    assert!(!none.any() && none.items().is_empty());
    let all = SystemMetrics { cpu01: Some(0.4), ram: Some((8000.0, 16000.0)), vram: Some((3000.0, 8000.0)) };
    assert_eq!(all.items().len(), 3);
    assert!(all.items().iter().all(Item::is_meter));
}

#[tokio::test]
async fn meters_that_exist_are_drawn_without_breaking_the_layout() {
    let ctx = test_ctx();
    let mut ex = explorer();
    let now = Instant::now();
    let view = view_of(&busy_registry(now));
    let meters = SystemMetrics { cpu01: Some(0.4), ram: Some((8000.0, 16000.0)), vram: Some((3000.0, 8000.0)) };
    for width in [700.0_f32, 1000.0, 1600.0] {
        for _ in 0..3 {
            run_frame(&ctx, width, Vec::new(), |ui| show_with(ui, &mut ex, "meters", &view, meters, now));
        }
        let l = last_layout(&ctx, "meters").unwrap();
        assert!(l.left.right() <= l.right.left() + 0.5, "{width}px: {l:?}");
    }
}

#[tokio::test]
async fn clicking_the_task_button_opens_the_center_and_a_second_click_closes_it() {
    let ctx = test_ctx();
    let mut ex = explorer();
    let now = Instant::now();
    let view = view_of(&busy_registry(now));
    let width = 1400.0;
    let shapes = settle(&ctx, width, &mut ex, "center", &view, now);
    assert!(!is_center_open(&ctx));
    let at = text_pos(&shapes, "Tasks (5)") + vec2(4.0, 4.0);

    run_frame(&ctx, width, press_release(at), |ui| show_with(ui, &mut ex, "center", &view, SystemMetrics::default(), now));
    let shapes = settle(&ctx, width, &mut ex, "center", &view, now);
    assert!(is_center_open(&ctx), "the click opens the task center");
    let all: Vec<String> = texts(&shapes).into_iter().map(|(t, _)| t).collect();
    for needle in ["Needs attention (1)", "Running (5)", "Scan: Photos", "Failed thumbnails", "Describe files"] {
        assert!(all.iter().any(|t| t == needle), "missing {needle:?} in {all:?}");
    }

    run_frame(&ctx, width, press_release(at), |ui| show_with(ui, &mut ex, "center", &view, SystemMetrics::default(), now));
    settle(&ctx, width, &mut ex, "center", &view, now);
    assert!(!is_center_open(&ctx), "a second click closes it");
}

#[tokio::test]
async fn a_click_outside_closes_the_center() {
    let ctx = test_ctx();
    let mut ex = explorer();
    let now = Instant::now();
    let view = view_of(&busy_registry(now));
    set_center_open(&ctx, true);
    settle(&ctx, 1200.0, &mut ex, "outside", &view, now);
    assert!(is_center_open(&ctx));
    run_frame(&ctx, 1200.0, press_release(Pos2::new(900.0, 40.0)), |ui| show_with(ui, &mut ex, "outside", &view, SystemMetrics::default(), now));
    settle(&ctx, 1200.0, &mut ex, "outside", &view, now);
    assert!(!is_center_open(&ctx));
}

#[tokio::test]
async fn the_center_lists_finished_tasks_with_their_errors_and_actions() {
    let ctx = test_ctx();
    let mut ex = explorer();
    let now = Instant::now();
    let mut r = TaskRegistry::new();
    let failed = r.begin_at(
        TaskSpec::new(TaskKind::Database, "Export database to a.surql")
            .retry(crate::command::AppCommand::ShowTaskCenter)
            .failure_scope(crate::command::Scope::Smart(crate::command::SmartView::FailedThumbs)),
        now,
    );
    r.finish_at(failed, Outcome::Failed("disk full".into()), now);
    let done = r.begin_at(TaskSpec::new(TaskKind::Clip, "CLIP embeddings"), now);
    r.finish_at(done, Outcome::Completed, now);
    let view = view_of(&r);
    set_center_open(&ctx, true);
    let shapes = settle(&ctx, 1200.0, &mut ex, "finished", &view, now);
    let all: Vec<String> = texts(&shapes).into_iter().map(|(t, _)| t).collect();
    for needle in ["Recent (2)", "Export database to a.surql", "disk full", "Retry", "Show failures", "CLIP embeddings"] {
        assert!(all.iter().any(|t| t == needle), "missing {needle:?} in {all:?}");
    }
    assert!(!all.iter().any(|t| t.starts_with("Running")), "{all:?}");
}

#[tokio::test]
async fn the_center_lists_nothing_running_for_an_empty_registry() {
    let ctx = test_ctx();
    let mut ex = explorer();
    let view = view_of(&TaskRegistry::new());
    set_center_open(&ctx, true);
    let shapes = settle(&ctx, 1200.0, &mut ex, "empty", &view, Instant::now());
    assert!(texts(&shapes).iter().any(|(t, _)| t == "Nothing is running."));
}

#[tokio::test]
async fn the_task_view_is_rebuilt_only_when_the_registry_revision_moves() {
    let ctx = test_ctx();
    let first = task_view(&ctx);
    let again = task_view(&ctx);
    if first.revision == again.revision {
        assert!(Arc::ptr_eq(&first.tasks, &again.tasks), "an unchanged registry reuses the snapshot");
    }
    let handle = crate::tasks::start(TaskSpec::new(TaskKind::Debug, "taskbar-view-test"));
    let changed = task_view(&ctx);
    assert!(changed.tasks.iter().any(|t| t.title == "taskbar-view-test"));
    handle.complete();
}

#[tokio::test]
async fn center_actions_reach_the_registry_and_the_bus() {
    use crate::command::bus::testing::{serial, take_matching};
    use crate::command::AppCommand;
    let _serial = serial();
    let ctx = test_ctx();
    let flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let handle = crate::tasks::start(TaskSpec::new(TaskKind::Debug, "center-action-test").cancel_flag(flag.clone()));
    apply_center_action(&ctx, center::CenterAction::Cancel(handle.id()));
    assert!(flag.load(std::sync::atomic::Ordering::Relaxed));
    apply_center_action(&ctx, center::CenterAction::Command(AppCommand::ShowTaskCenter));
    assert_eq!(take_matching(|c| matches!(c, AppCommand::ShowTaskCenter)).len(), 1);
    let id = handle.id();
    handle.cancelled();
    apply_center_action(&ctx, center::CenterAction::Dismiss(id));
    assert!(crate::tasks::with(|r| r.get(id).is_none()));
}
