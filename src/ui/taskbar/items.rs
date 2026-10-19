//! The items of the status bar: what each shows in its full and compact form and what a click does.

use eframe::egui::*;

use super::layout::{Entry, Mode};
use super::view_stats::ViewStats;
use crate::icons::{self, Icon, svg};
use crate::tasks::{ActivityChip, TaskKind};

pub const BAR_FULL_W: f32 = 64.0;
pub const BAR_COMPACT_W: f32 = 28.0;
pub const BAR_H: f32 = 5.0;

#[derive(Clone, Debug)]
pub enum Item {
    Count { items: usize, hidden: usize, detail: String },
    Selection { count: usize, bytes: u64 },
    Filters { count: usize, labels: Vec<String> },
    /// Opens the task list; shows how many tasks run.
    Tasks { active: usize },
    Activity(ActivityChip),
    Attention { count: usize },
    Meter { icon: Icon, label: &'static str, pct: f32, detail: String },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BarAction {
    ClearFilters,
    ToggleCenter,
}

/// What drawing an item produced.
#[derive(Default)]
pub struct Drawn {
    pub action: Option<BarAction>,
    /// The widget the task list popover opens from.
    pub anchor: Option<Response>,
}

fn kind_icon(kind: TaskKind) -> Icon {
    match kind {
        TaskKind::Scan => svg::AME_SCAN,
        TaskKind::Thumbnails => svg::AME_THUMBNAIL_GRID,
        TaskKind::Describe => svg::AME_AI_DESCRIBE,
        TaskKind::Clip => svg::AME_CLIP_EMBED,
        TaskKind::Refine => svg::AME_REFINE,
        TaskKind::ImageEdit => svg::AME_IMAGE_EDIT,
        TaskKind::Database => svg::AME_DATABASE,
        TaskKind::LegacyImport => svg::FILE_ARROW_DOWN,
        TaskKind::ThumbMigration => svg::ARROWS_CLOCKWISE,
        TaskKind::Backup => svg::COPY,
        TaskKind::ModelLoad => svg::AME_MODEL,
        TaskKind::Assistant => svg::AME_ASSISTANT,
        TaskKind::Debug => svg::WRENCH,
    }
}

/// Icon of a task kind.
pub fn icon_of(kind: TaskKind) -> Icon {
    kind_icon(kind)
}

fn plural(n: usize, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}

fn bytes(b: u64) -> String {
    humansize::format_size(b, humansize::DECIMAL)
}

impl Item {
    /// Layout priority: higher items collapse last.
    pub fn priority(&self) -> u8 {
        match self {
            Self::Tasks { .. } => 255,
            Self::Attention { .. } => 200,
            Self::Count { .. } => 150,
            Self::Selection { .. } => 140,
            Self::Filters { .. } => 130,
            Self::Activity(chip) => 120u8.saturating_sub(chip.kind as u8),
            Self::Meter { label, .. } => match *label {
                "CPU" => 40,
                "RAM" => 35,
                _ => 30,
            },
        }
    }

    pub fn is_meter(&self) -> bool {
        matches!(self, Self::Meter { .. })
    }

    pub fn is_tasks_button(&self) -> bool {
        matches!(self, Self::Tasks { .. })
    }

    /// Items measured in both forms.
    pub fn has_compact(&self) -> bool {
        true
    }

    pub fn draw(&self, ui: &mut Ui, mode: Mode) -> Drawn {
        let compact = mode == Mode::Compact;
        let mut out = Drawn::default();
        match self {
            Self::Count { items, hidden, detail } => {
                let text = match (compact, *hidden) {
                    (true, _) => items.to_string(),
                    (false, 0) => plural(*items, "item", "items"),
                    (false, h) => format!("{items} of {} items", items + h),
                };
                icons::label(ui, svg::SQUARES_FOUR, text).on_hover_text(detail.as_str());
            }
            Self::Selection { count, bytes: size } => {
                let text = if compact { count.to_string() } else { format!("{count} selected ({})", bytes(*size)) };
                icons::label(ui, svg::CURSOR_CLICK, text).on_hover_text(format!("{} selected, {}", plural(*count, "file", "files"), bytes(*size)));
            }
            Self::Filters { count, labels } => {
                let text = if compact { count.to_string() } else { plural(*count, "filter", "filters") };
                icons::label(ui, svg::FUNNEL, text).on_hover_text(labels.join("\n"));
                let clear = if compact { icons::icon_button(ui, svg::X, "Clear filters") } else { ui.link("Clear").on_hover_text("Clear every filter") };
                if clear.clicked() {
                    out.action = Some(BarAction::ClearFilters);
                }
            }
            Self::Tasks { active } => {
                let resp = if compact {
                    let text = if *active > 0 { active.to_string() } else { String::new() };
                    icons::button(ui, svg::LIST_CHECKS, text).on_hover_text("Tasks")
                } else {
                    let text = if *active > 0 { format!("Tasks ({active})") } else { "Tasks".to_string() };
                    icons::button(ui, svg::LIST_CHECKS, text).on_hover_text("Running and recent tasks")
                };
                if resp.clicked() {
                    out.action = Some(BarAction::ToggleCenter);
                }
                out.anchor = Some(resp);
            }
            Self::Activity(chip) => {
                let tint = crate::theme::primary(ui);
                let text = if compact { chip.short.as_str() } else { chip.label.as_str() };
                let resp = icons::button(ui, kind_icon(chip.kind), text).on_hover_text(if chip.queued { "Waiting to start" } else { "Open the task list" });
                if let Some(fraction) = chip.fraction {
                    let width = if compact { BAR_COMPACT_W } else { BAR_FULL_W };
                    ui.add(ProgressBar::new(fraction).desired_width(width).desired_height(BAR_H).fill(tint));
                }
                if resp.clicked() {
                    out.action = Some(BarAction::ToggleCenter);
                }
            }
            Self::Attention { count } => {
                let text = if compact { count.to_string() } else { format!("{count} need attention") };
                let tint = crate::theme::warn(ui);
                let resp = icons::button(ui, svg::WARNING, RichText::new(text).color(tint)).on_hover_text("Things that need you");
                if resp.clicked() {
                    out.action = Some(BarAction::ToggleCenter);
                }
            }
            Self::Meter { icon, label, pct, detail } => {
                let color = crate::theme::usage_level(ui, pct * 100.0);
                if compact {
                    icons::label(ui, *icon, format!("{:.0}%", pct * 100.0));
                    ui.add(ProgressBar::new(*pct).desired_width(BAR_COMPACT_W).desired_height(BAR_H).fill(color));
                } else {
                    icons::label(ui, *icon, format!("{label}: {detail}"));
                    ui.add(ProgressBar::new(*pct).desired_width(BAR_FULL_W).desired_height(BAR_H).fill(color));
                }
            }
        }
        out
    }
}

/// Width of an item drawn in `mode`, measured by laying it out invisibly.
pub fn measure(ui: &mut Ui, item: &Item, mode: Mode, spacing: f32) -> f32 {
    let origin = ui.available_rect_before_wrap().min;
    let builder = UiBuilder::new()
        .max_rect(Rect::from_min_size(origin, vec2(4000.0, ui.available_height().max(1.0))))
        .layout(Layout::left_to_right(Align::Center))
        .invisible();
    let mut child = ui.new_child(builder);
    child.spacing_mut().item_spacing.x = spacing;
    item.draw(&mut child, mode);
    child.min_rect().width()
}

/// Layout entry of an item at the current style.
pub fn entry_of(ui: &mut Ui, item: &Item, spacing: f32) -> Entry {
    Entry {
        priority: item.priority(),
        full: measure(ui, item, Mode::Full, spacing),
        compact: item.has_compact().then(|| measure(ui, item, Mode::Compact, spacing)),
    }
}

/// The items for a view and the running work, in display order.
pub fn build(stats: &ViewStats, chips: Vec<ActivityChip>, active: usize, attention: usize, meters: Vec<Item>) -> Vec<Item> {
    let mut items = Vec::new();
    let detail = format!(
        "{}, {}, {}, {}",
        plural(stats.images, "image", "images"),
        plural(stats.videos, "video", "videos"),
        plural(stats.dirs, "folder", "folders"),
        bytes(stats.total_bytes)
    );
    items.push(Item::Count { items: stats.items, hidden: stats.hidden, detail });
    if stats.selected > 0 {
        items.push(Item::Selection { count: stats.selected, bytes: stats.selected_bytes });
    }
    if !stats.filters.is_empty() {
        items.push(Item::Filters { count: stats.filters.len(), labels: stats.filters.iter().map(|f| f.label.clone()).collect() });
    }
    items.push(Item::Tasks { active });
    items.extend(chips.into_iter().map(Item::Activity));
    if attention > 0 {
        items.push(Item::Attention { count: attention });
    }
    items.extend(meters);
    items
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::taskbar::view_stats::{FilterChip, FilterKind};

    fn chip(kind: TaskKind) -> ActivityChip {
        ActivityChip { kind, count: 1, done: 0, total: 0, fraction: None, queued: false, label: kind.label().into(), short: String::new() }
    }

    fn stats() -> ViewStats {
        ViewStats { items: 10, ..Default::default() }
    }

    fn kinds(items: &[Item]) -> Vec<&'static str> {
        items
            .iter()
            .map(|i| match i {
                Item::Count { .. } => "count",
                Item::Selection { .. } => "selection",
                Item::Filters { .. } => "filters",
                Item::Tasks { .. } => "tasks",
                Item::Activity(_) => "activity",
                Item::Attention { .. } => "attention",
                Item::Meter { .. } => "meter",
            })
            .collect()
    }

    #[test]
    fn an_idle_plain_view_shows_the_count_and_the_task_button() {
        assert_eq!(kinds(&build(&stats(), vec![], 0, 0, vec![])), vec!["count", "tasks"]);
    }

    #[test]
    fn selection_filters_chips_and_attention_appear_only_when_there_is_something() {
        let mut s = stats();
        s.selected = 2;
        s.filters = vec![FilterChip { kind: FilterKind::Size, label: "Size >= 10 KiB".into() }];
        let items = build(&s, vec![chip(TaskKind::Scan), chip(TaskKind::Clip)], 2, 1, vec![]);
        assert_eq!(kinds(&items), vec!["count", "selection", "filters", "tasks", "activity", "activity", "attention"]);
    }

    #[test]
    fn the_task_button_collapses_last_and_meters_first() {
        let items = build(&stats(), vec![chip(TaskKind::Scan)], 1, 1, vec![Item::Meter { icon: svg::CPU, label: "CPU", pct: 0.1, detail: String::new() }]);
        let by_priority = |name: &str| items.iter().find(|i| kinds(std::slice::from_ref(*i))[0] == name).unwrap().priority();
        assert_eq!(items.iter().map(Item::priority).max(), Some(by_priority("tasks")));
        assert_eq!(items.iter().map(Item::priority).min(), Some(by_priority("meter")));
        assert!(by_priority("attention") > by_priority("count"));
        assert!(by_priority("count") > by_priority("activity"));
    }

    #[test]
    fn earlier_task_kinds_outrank_later_ones() {
        let scan = Item::Activity(chip(TaskKind::Scan)).priority();
        let clip = Item::Activity(chip(TaskKind::Clip)).priority();
        assert!(scan > clip);
    }

    #[test]
    fn the_count_detail_lists_the_kinds_and_size() {
        let mut s = stats();
        s.images = 3;
        s.videos = 1;
        s.dirs = 2;
        s.total_bytes = 1_500_000;
        match &build(&s, vec![], 0, 0, vec![])[0] {
            Item::Count { detail, .. } => assert_eq!(detail, "3 images, 1 video, 2 folders, 1.50 MB"),
            other => panic!("unexpected {other:?}"),
        }
    }
}
