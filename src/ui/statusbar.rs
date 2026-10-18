//! Bottom status bar: stats cached per table revision, responsive layout and an overflow menu.

use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

use eframe::egui::*;
use humansize::DECIMAL;

use crate::UiSettings;
use crate::database::Thumbnail;
use crate::icons::{self, Icon, IconSize, svg};
use crate::ui::file_table::FileExplorer;

/// Narrowest window width the bar guarantees not to overlap at.
pub const MIN_SUPPORTED_WIDTH: f32 = 640.0;
const STATS_THROTTLE: Duration = Duration::from_millis(250);
const SCAN_DONE_SHOWN: Duration = Duration::from_secs(4);
const METER_REFRESH: Duration = Duration::from_millis(500);
const BAR_FULL_W: f32 = 84.0;
const BAR_COMPACT_W: f32 = 36.0;
const BAR_H: f32 = 5.0;
const NAME_MAX_CHARS: usize = 18;

/* ---------------------------------------------- stats --------------------------------------------- */

/// Counts and sizes shown by the bar, computed in one pass over the rows.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BarStats {
    pub visible: usize,
    pub images: usize,
    pub videos: usize,
    pub dirs: usize,
    pub total_size: u64,
    pub sel_images: usize,
    pub sel_videos: usize,
    pub sel_dirs: usize,
    pub sel_size: u64,
    pub thumbs_total: usize,
    pub thumbs_done: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Image,
    Video,
    Other,
}

fn kind_of(path: &str) -> Kind {
    let name = path.rsplit(['/', '\\']).next().unwrap_or(path);
    let Some((stem, ext)) = name.rsplit_once('.') else { return Kind::Other };
    if stem.is_empty() || ext.is_empty() || ext.len() > 8 {
        return Kind::Other;
    }
    let mut buf = [0u8; 8];
    for (slot, b) in buf.iter_mut().zip(ext.bytes()) {
        *slot = b.to_ascii_lowercase();
    }
    match std::str::from_utf8(&buf[..ext.len()]) {
        Ok(lower) if crate::is_image(lower) => Kind::Image,
        Ok(lower) if crate::is_video(lower) => Kind::Video,
        _ => Kind::Other,
    }
}

/// Single pass over `rows`; `passes` is the view filter and `has_thumb` tells whether a thumbnail is cached.
pub fn compute_stats<'a>(
    rows: impl Iterator<Item = &'a Thumbnail>,
    passes: impl Fn(&Thumbnail) -> bool,
    selected: &HashSet<String>,
    has_thumb: impl Fn(&str) -> bool,
) -> BarStats {
    let mut s = BarStats::default();
    for r in rows {
        let is_dir = r.file_type == "<DIR>";
        let kind = if is_dir { Kind::Other } else { kind_of(&r.path) };
        if passes(r) {
            s.visible += 1;
            if is_dir {
                s.dirs += 1;
            } else {
                s.total_size = s.total_size.saturating_add(r.size);
                match kind {
                    Kind::Image => s.images += 1,
                    Kind::Video => s.videos += 1,
                    Kind::Other => {}
                }
                if kind != Kind::Other {
                    s.thumbs_total += 1;
                    if has_thumb(&r.path) {
                        s.thumbs_done += 1;
                    }
                }
            }
        }
        if !selected.is_empty() && selected.contains(&r.path) {
            if is_dir {
                s.sel_dirs += 1;
            } else {
                s.sel_size = s.sel_size.saturating_add(r.size);
                match kind {
                    Kind::Image => s.sel_images += 1,
                    Kind::Video => s.sel_videos += 1,
                    Kind::Other => {}
                }
            }
        }
    }
    s
}

#[derive(Clone)]
struct CachedStats {
    stamp: u64,
    at: Instant,
    stats: BarStats,
}

fn path_hash(p: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    p.hash(&mut h);
    h.finish()
}

/// Cheap fingerprint of everything the stats depend on.
fn revision_stamp(ex: &FileExplorer) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    ex.table.len().hash(&mut h);
    ex.table.first().map(|r| r.path.as_str()).hash(&mut h);
    ex.table.last().map(|r| r.path.as_str()).hash(&mut h);
    ex.current_path.hash(&mut h);
    ex.viewer.filter.hash(&mut h);
    (ex.viewer.types_show_images, ex.viewer.types_show_videos, ex.viewer.types_show_dirs).hash(&mut h);
    (ex.viewer.mode as u8, ex.viewer.showing_similarity).hash(&mut h);
    ex.viewer.thumb_cache.len().hash(&mut h);
    (ex.is_recursive_scan(), ex.recursive_total_filtered(), ex.recursive_total_unfiltered()).hash(&mut h);
    ex.viewer.selected.len().hash(&mut h);
    let selection = ex.viewer.selected.iter().fold(0u64, |acc, p| acc ^ path_hash(p));
    selection.hash(&mut h);
    h.finish()
}

fn is_scanning(ex: &FileExplorer) -> bool {
    ex.file_scan_progress > 0.0 && !ex.scan_done
}

/// Stats for the explorer, recomputed only when its revision stamp changes (throttled while scanning).
fn cached_stats(ctx: &Context, ex: &FileExplorer, owner: &str) -> BarStats {
    let id = Id::new(("ame_bar_stats", owner));
    let stamp = revision_stamp(ex);
    let cached = ctx.data(|d| d.get_temp::<CachedStats>(id));
    if let Some(c) = &cached {
        if c.stamp == stamp {
            return c.stats.clone();
        }
        let age = c.at.elapsed();
        if is_scanning(ex) && age < STATS_THROTTLE {
            ctx.request_repaint_after(STATS_THROTTLE - age);
            return c.stats.clone();
        }
    }
    #[cfg(test)]
    tests::note_compute(owner);
    let stats = compute_stats(
        ex.table.iter(),
        |r| ex.viewer.row_passes_filter(r),
        &ex.viewer.selected,
        |p| ex.viewer.thumb_cache.contains_key(p),
    );
    ctx.data_mut(|d| d.insert_temp(id, CachedStats { stamp, at: Instant::now(), stats: stats.clone() }));
    stats
}

/* ----------------------------------------------- chips --------------------------------------------- */

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChipKind {
    Size,
    Dates,
    SkipIcons,
    Search,
    HiddenTypes,
}

/// An active filter shown as a removable chip.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChipSpec {
    pub kind: ChipKind,
    pub label: String,
}

fn fmt_bytes(b: u64) -> String {
    humansize::format_size(b, humansize::BINARY)
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    s.chars().take(max.saturating_sub(3)).collect::<String>() + "..."
}

/// Filters currently narrowing the view, including the default minimum size.
pub fn active_chips(s: &UiSettings, search: &str, show_images: bool, show_videos: bool, show_dirs: bool) -> Vec<ChipSpec> {
    let mut chips = Vec::new();
    let size = match (s.db_min_size_bytes, s.db_max_size_bytes) {
        (Some(a), Some(b)) => Some(format!("Size {} - {}", fmt_bytes(a), fmt_bytes(b))),
        (Some(a), None) => Some(format!("Size >= {}", fmt_bytes(a))),
        (None, Some(b)) => Some(format!("Size <= {}", fmt_bytes(b))),
        (None, None) => None,
    };
    if let Some(label) = size {
        chips.push(ChipSpec { kind: ChipKind::Size, label });
    }
    let dates = match (s.filter_modified_after.as_deref(), s.filter_modified_before.as_deref()) {
        (Some(a), Some(b)) => Some(format!("{a} to {b}")),
        (Some(a), None) => Some(format!("Since {a}")),
        (None, Some(b)) => Some(format!("Until {b}")),
        (None, None) => None,
    };
    if let Some(label) = dates {
        chips.push(ChipSpec { kind: ChipKind::Dates, label });
    }
    if s.filter_skip_icons {
        chips.push(ChipSpec { kind: ChipKind::SkipIcons, label: "Skip icons".into() });
    }
    if !search.trim().is_empty() {
        chips.push(ChipSpec { kind: ChipKind::Search, label: format!("Search: {}", truncate_chars(search.trim(), NAME_MAX_CHARS)) });
    }
    let hidden: Vec<&str> = [(!show_images, "Images"), (!show_videos, "Videos"), (!show_dirs, "Folders")]
        .into_iter()
        .filter_map(|(hide, name)| hide.then_some(name))
        .collect();
    if !hidden.is_empty() {
        chips.push(ChipSpec { kind: ChipKind::HiddenTypes, label: format!("Hiding: {}", hidden.join(", ")) });
    }
    chips
}

/* ----------------------------------------------- layout -------------------------------------------- */

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Full,
    Compact,
}

/// One bar item's measured widths; entries are in display order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Entry {
    pub priority: u8,
    pub full: f32,
    pub compact: Option<f32>,
}

/// Which items are shown at which width, and whether the rest moved to the overflow menu.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub modes: Vec<Option<Mode>>,
    pub overflow: bool,
}

fn plan_width(entries: &[Entry], modes: &[Option<Mode>], gap: f32, overflow_w: f32) -> f32 {
    let mut total = 0.0;
    let mut shown = 0;
    for (e, m) in entries.iter().zip(modes) {
        let Some(mode) = m else { continue };
        total += match mode {
            Mode::Full => e.full,
            Mode::Compact => e.compact.unwrap_or(e.full),
        };
        shown += 1;
    }
    if shown > 1 {
        total += gap * (shown - 1) as f32;
    }
    if modes.iter().any(Option::is_none) {
        total += overflow_w + if shown > 0 { gap } else { 0.0 };
    }
    total
}

/// Shrinks items to compact form lowest priority first, then collapses the lowest priority items into the overflow menu.
pub fn plan(entries: &[Entry], gap: f32, overflow_w: f32, avail: f32) -> Plan {
    let mut modes: Vec<Option<Mode>> = vec![Some(Mode::Full); entries.len()];
    while plan_width(entries, &modes, gap, overflow_w) > avail {
        let next = entries
            .iter()
            .enumerate()
            .filter(|(i, e)| modes[*i] == Some(Mode::Full) && e.compact.is_some_and(|c| c < e.full))
            .min_by_key(|(_, e)| e.priority)
            .map(|(i, _)| i);
        match next {
            Some(i) => modes[i] = Some(Mode::Compact),
            None => break,
        }
    }
    while plan_width(entries, &modes, gap, overflow_w) > avail {
        let next = entries
            .iter()
            .enumerate()
            .filter(|(i, _)| modes[*i].is_some())
            .min_by_key(|(_, e)| e.priority)
            .map(|(i, _)| i);
        match next {
            Some(i) => modes[i] = None,
            None => break,
        }
    }
    let overflow = modes.iter().any(Option::is_none);
    Plan { modes, overflow }
}

/// Rectangles the last drawn bar occupied, for tests and devtools.
#[derive(Clone, Copy, Debug)]
pub struct BarLayout {
    pub left: Rect,
    pub right: Rect,
    pub panel: Rect,
    pub overflowed: usize,
}

pub fn last_layout(ctx: &Context, owner: &str) -> Option<BarLayout> {
    ctx.data(|d| d.get_temp::<BarLayout>(Id::new(("ame_bar_layout", owner))))
}

/* ----------------------------------------------- items --------------------------------------------- */

/// Live system meter readings; `None` when the platform provides no data.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemMetrics {
    pub cpu01: Option<f32>,
    pub ram: Option<(f32, f32)>,
    pub vram: Option<(f32, f32)>,
}

impl SystemMetrics {
    /// Readings from the platform helpers; unsupported meters stay `None`.
    pub fn sample() -> Self {
        use crate::utilities::windows::{gpu_mem_mb, smoothed_cpu01, system_mem_mb};
        Self {
            cpu01: cfg!(windows).then(smoothed_cpu01),
            ram: system_mem_mb().filter(|(_, total)| *total > 0.0),
            vram: gpu_mem_mb().filter(|(_, total)| *total > 0.0),
        }
    }
}

#[derive(Clone, Debug)]
enum ItemKind {
    Scan { pct: f32 },
    ScanDone,
    Chip(ChipSpec),
    Filtered { shown: usize, total: usize },
    Stat { icon: Icon, label: &'static str, value: String, selected: bool },
    Thumbs { done: usize, total: usize },
    Meter { icon: Icon, label: &'static str, pct: f32, detail: String },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BarAction {
    ClearChip(ChipKind),
}

impl ItemKind {
    fn priority(&self) -> u8 {
        match self {
            Self::Filtered { .. } => 100,
            Self::Chip(_) => 95,
            Self::Scan { .. } => 90,
            Self::Stat { label, .. } => match *label {
                "Images" => 80,
                "Videos" => 75,
                "Dirs" => 70,
                _ => 65,
            },
            Self::Thumbs { .. } => 60,
            Self::Meter { label, .. } => match *label {
                "CPU" => 40,
                "RAM" => 35,
                _ => 30,
            },
            Self::ScanDone => 20,
        }
    }

    fn has_compact(&self) -> bool {
        !matches!(self, Self::ScanDone)
    }

    fn is_meter(&self) -> bool {
        matches!(self, Self::Meter { .. })
    }

    fn draw(&self, ui: &mut Ui, mode: Mode) -> Option<BarAction> {
        let compact = mode == Mode::Compact;
        match self {
            Self::Scan { pct } => {
                if compact {
                    ui.add(ProgressBar::new(*pct).desired_width(BAR_COMPACT_W).desired_height(BAR_H).fill(crate::theme::primary(ui)).animate(true));
                    ui.label(format!("{:.0}%", pct * 100.0));
                } else {
                    icons::label(ui, svg::AME_SCAN, "Scan");
                    ui.add(ProgressBar::new(*pct).desired_width(BAR_FULL_W + 20.0).show_percentage().fill(crate::theme::primary(ui)).animate(true));
                }
                None
            }
            Self::ScanDone => {
                icons::label(ui, svg::CHECK_CIRCLE, RichText::new("Scan complete").color(crate::theme::success(ui)));
                None
            }
            Self::Chip(chip) => {
                if compact {
                    icons::label(ui, svg::FUNNEL, "");
                } else {
                    icons::label(ui, svg::FUNNEL, chip.label.as_str());
                }
                let clear = icons::icon_button(ui, svg::X, &format!("Clear filter: {}", chip.label));
                clear.clicked().then_some(BarAction::ClearChip(chip.kind))
            }
            Self::Filtered { shown, total } => {
                if compact {
                    icons::label(ui, svg::FUNNEL, format!("{shown}/{total}"));
                } else {
                    let out = total.saturating_sub(*shown);
                    icons::label(ui, svg::FUNNEL, format!("Filtered {shown} of {total} (out: {out})"));
                }
                None
            }
            Self::Stat { icon, label, value, selected } => {
                if compact {
                    icons::label(ui, *icon, value.as_str());
                } else {
                    let prefix = if *selected { "Sel. " } else { "" };
                    icons::label(ui, *icon, format!("{prefix}{label}: {value}"));
                }
                None
            }
            Self::Thumbs { done, total } => {
                let pct = if *total == 0 { 0.0 } else { *done as f32 / *total as f32 };
                if compact {
                    icons::label(ui, svg::AME_THUMBNAIL_GRID, "");
                    ui.add(ProgressBar::new(pct).desired_width(BAR_COMPACT_W).desired_height(BAR_H).fill(crate::theme::primary(ui)));
                } else {
                    icons::label(ui, svg::AME_THUMBNAIL_GRID, format!("Thumbs: {done}/{total}"));
                    ui.add(ProgressBar::new(pct).desired_width(BAR_FULL_W).desired_height(BAR_H).fill(crate::theme::primary(ui)));
                }
                None
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
                None
            }
        }
    }
}

/// Width of an item drawn in `mode`, measured by laying it out invisibly.
fn measure(ui: &mut Ui, item: &ItemKind, mode: Mode, spacing: f32) -> f32 {
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

fn text_button_width(ui: &Ui, text: &str) -> f32 {
    let font = TextStyle::Button.resolve(ui.style());
    let galley = ui.painter().layout_no_wrap(text.to_string(), font, Color32::WHITE);
    galley.size().x + ui.spacing().button_padding.x * 2.0 + 2.0
}

fn separator(ui: &mut Ui, width: f32) {
    let height = ui.available_height().min(18.0);
    let (rect, _) = ui.allocate_exact_size(vec2(width, height), Sense::hover());
    let x = rect.center().x;
    let stroke = ui.visuals().widgets.noninteractive.bg_stroke;
    ui.painter().line_segment([pos2(x, rect.top() + 2.0), pos2(x, rect.bottom() - 2.0)], stroke);
}

fn build_items(
    ctx: &Context,
    ex: &FileExplorer,
    owner: &str,
    stats: &BarStats,
    chips: Vec<ChipSpec>,
    metrics: &SystemMetrics,
) -> Vec<ItemKind> {
    let mut items = Vec::new();
    let done_id = Id::new(("ame_scan_done_at", owner));
    if is_scanning(ex) {
        ctx.data_mut(|d| d.remove::<Instant>(done_id));
        items.push(ItemKind::Scan { pct: ex.file_scan_progress.clamp(0.0, 1.0) });
    } else if ex.scan_done && ex.file_scan_progress > 0.0 {
        let at = ctx.data_mut(|d| *d.get_temp_mut_or_insert_with(done_id, Instant::now));
        if at.elapsed() < SCAN_DONE_SHOWN {
            items.push(ItemKind::ScanDone);
            ctx.request_repaint_after(SCAN_DONE_SHOWN - at.elapsed());
        }
    }

    let (global_filtered, global_total) = if ex.is_recursive_scan() {
        (ex.recursive_total_filtered(), ex.recursive_total_unfiltered())
    } else {
        (stats.visible, ex.table.len())
    };
    items.push(ItemKind::Filtered { shown: global_filtered, total: global_total });
    items.extend(chips.into_iter().map(ItemKind::Chip));

    let selecting = ex.selection_count() > 0;
    let (images, videos, dirs, size) = if selecting {
        (stats.sel_images, stats.sel_videos, stats.sel_dirs, stats.sel_size)
    } else {
        (stats.images, stats.videos, stats.dirs, stats.total_size)
    };
    let stat = |icon: Icon, label: &'static str, value: String| ItemKind::Stat { icon, label, value, selected: selecting };
    items.push(stat(svg::IMAGE, "Images", images.to_string()));
    items.push(stat(svg::VIDEO_CAMERA, "Videos", videos.to_string()));
    items.push(stat(svg::FOLDER, "Dirs", dirs.to_string()));
    items.push(stat(svg::HARD_DRIVE, "Total Size", humansize::format_size(size, DECIMAL)));
    if stats.thumbs_total > 0 {
        items.push(ItemKind::Thumbs { done: stats.thumbs_done, total: stats.thumbs_total });
    }
    if let Some(cpu) = metrics.cpu01 {
        items.push(ItemKind::Meter { icon: svg::CPU, label: "CPU", pct: cpu.clamp(0.0, 1.0), detail: format!("{:.0}%", cpu * 100.0) });
    }
    if let Some((used, total)) = metrics.ram {
        items.push(ItemKind::Meter { icon: svg::MEMORY, label: "RAM", pct: (used / total).clamp(0.0, 1.0), detail: format!("{used:.0}/{total:.0} MiB") });
    }
    if let Some((used, total)) = metrics.vram {
        items.push(ItemKind::Meter { icon: svg::AME_GPU, label: "VRAM", pct: (used / total).clamp(0.0, 1.0), detail: format!("{used:.0}/{total:.0} MiB") });
    }
    items
}

fn apply_action(ex: &mut FileExplorer, action: BarAction) {
    let BarAction::ClearChip(kind) = action;
    match kind {
        ChipKind::Size => {
            crate::settings::update("statusbar:clear-size", |s| {
                s.db_min_size_bytes = None;
                s.db_max_size_bytes = None;
            });
        }
        ChipKind::Dates => {
            crate::settings::update("statusbar:clear-dates", |s| {
                s.filter_modified_after = None;
                s.filter_modified_before = None;
            });
        }
        ChipKind::SkipIcons => {
            crate::settings::update("statusbar:clear-skip-icons", |s| s.filter_skip_icons = false);
        }
        ChipKind::Search => ex.viewer.filter.clear(),
        ChipKind::HiddenTypes => {
            ex.viewer.types_show_images = true;
            ex.viewer.types_show_videos = true;
            ex.viewer.types_show_dirs = true;
        }
    }
}

/// Draws the bar for `ex`, sampling the platform's system meters.
pub fn show(ui: &mut Ui, ex: &mut FileExplorer, owner: &str) {
    show_with_metrics(ui, ex, owner, SystemMetrics::sample());
}

/// Draws the bar with explicit meter readings.
pub fn show_with_metrics(ui: &mut Ui, ex: &mut FileExplorer, owner: &str, metrics: SystemMetrics) {
    let ctx = ui.ctx().clone();
    let m = crate::theme::metrics(&ctx);
    let mut actions: Vec<BarAction> = Vec::new();
    Panel::bottom("FileExplorer Bottom Panel").exact_size(m.bar_h).show(ui, |ui| {
        let stats = cached_stats(&ctx, ex, owner);
        let chips = crate::settings::store().read(|s| {
            active_chips(s, &ex.viewer.filter, ex.viewer.types_show_images, ex.viewer.types_show_videos, ex.viewer.types_show_dirs)
        });
        let items = build_items(&ctx, ex, owner, &stats, chips, &metrics);

        let gap = m.gap.x;
        let inner = gap / 2.0;
        let step = inner * 2.0 + 1.0;
        let rect = ui.available_rect_before_wrap();
        let select_w = text_button_width(ui, &ex.selection_label());
        let groups_w = text_button_width(ui, &ex.logical_group_label());
        let right_w = (select_w + gap + groups_w).min(rect.width());
        let right_rect = Rect::from_min_max(pos2(rect.right() - right_w, rect.top()), rect.right_bottom());
        let left_rect = Rect::from_min_max(rect.left_top(), pos2((right_rect.left() - gap).max(rect.left()), rect.bottom()));

        let overflow_w = ui.spacing().interact_size.y.max(IconSize::Md.px(ui) + 4.0);
        let entries: Vec<Entry> = items
            .iter()
            .map(|it| Entry {
                priority: it.priority(),
                full: measure(ui, it, Mode::Full, inner),
                compact: it.has_compact().then(|| measure(ui, it, Mode::Compact, inner)),
            })
            .collect();
        let plan = plan(&entries, step, overflow_w, left_rect.width());

        let right = ui.scope_builder(UiBuilder::new().max_rect(right_rect).layout(Layout::right_to_left(Align::Center)), |ui| {
            ui.spacing_mut().item_spacing.x = gap;
            ex.logical_group_menu(ui);
            ex.selection_menu(ui, owner);
        });

        let left = ui.scope_builder(UiBuilder::new().max_rect(left_rect).layout(Layout::left_to_right(Align::Center)), |ui| {
            ui.set_clip_rect(left_rect.intersect(ui.clip_rect()));
            ui.spacing_mut().item_spacing.x = inner;
            let mut first = true;
            for (item, mode) in items.iter().zip(&plan.modes) {
                let Some(mode) = mode else { continue };
                if !first {
                    separator(ui, 1.0);
                }
                first = false;
                actions.extend(item.draw(ui, *mode));
            }
            if plan.overflow {
                if !first {
                    separator(ui, 1.0);
                }
                icons::icon_menu_button(ui, svg::DOTS_THREE, "More status items", |ui| {
                    for (item, mode) in items.iter().zip(&plan.modes) {
                        if mode.is_none() {
                            ui.horizontal(|ui| actions.extend(item.draw(ui, Mode::Full)));
                        }
                    }
                });
            }
        });

        let shown_meter = items.iter().zip(&plan.modes).any(|(it, mode)| it.is_meter() && mode.is_some());
        if shown_meter {
            ctx.request_repaint_after(METER_REFRESH);
        }
        let layout = BarLayout {
            left: left.response.rect,
            right: right.response.rect,
            panel: rect,
            overflowed: plan.modes.iter().filter(|m| m.is_none()).count(),
        };
        ctx.data_mut(|d| d.insert_temp(Id::new(("ame_bar_layout", owner)), layout));
    });
    for action in actions {
        apply_action(ex, action);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::Mutex;

    static COMPUTES: once_cell::sync::Lazy<Mutex<std::collections::HashMap<String, usize>>> =
        once_cell::sync::Lazy::new(|| Mutex::new(std::collections::HashMap::new()));

    pub(crate) fn note_compute(owner: &str) {
        *COMPUTES.lock().unwrap().entry(owner.to_string()).or_default() += 1;
    }

    fn computes(owner: &str) -> usize {
        COMPUTES.lock().unwrap().get(owner).copied().unwrap_or(0)
    }

    fn row(path: &str, kind: &str, size: u64) -> Thumbnail {
        let mut t = Thumbnail::default();
        t.path = path.into();
        t.filename = path.rsplit('/').next().unwrap_or(path).into();
        t.file_type = kind.into();
        t.size = size;
        t
    }

    fn sample_rows() -> Vec<Thumbnail> {
        vec![
            row("/p/a.JPG", "jpg", 100),
            row("/p/b.png", "png", 200),
            row("/p/c.mp4", "mp4", 4000),
            row("/p/notes.txt", "txt", 7),
            row("/p/sub", "<DIR>", 0),
        ]
    }

    #[test]
    fn kind_detection_ignores_case_and_dotfiles() {
        assert_eq!(kind_of("/x/A.JPG"), Kind::Image);
        assert_eq!(kind_of("C:\\x\\clip.MP4"), Kind::Video);
        assert_eq!(kind_of("/x/.png"), Kind::Other);
        assert_eq!(kind_of("/x/noext"), Kind::Other);
        assert_eq!(kind_of("/x/file."), Kind::Other);
        assert_eq!(kind_of("/x/a.verylongextension"), Kind::Other);
    }

    #[test]
    fn stats_count_in_one_pass_with_selection_tallies() {
        let rows = sample_rows();
        let selected: HashSet<String> = ["/p/b.png".to_string(), "/p/c.mp4".to_string()].into();
        let cached: HashSet<&str> = ["/p/a.JPG"].into();
        let s = compute_stats(rows.iter(), |_| true, &selected, |p| cached.contains(p));
        assert_eq!((s.visible, s.images, s.videos, s.dirs, s.total_size), (5, 2, 1, 1, 4307));
        assert_eq!((s.sel_images, s.sel_videos, s.sel_dirs, s.sel_size), (1, 1, 0, 4200));
        assert_eq!((s.thumbs_total, s.thumbs_done), (3, 1));
    }

    #[test]
    fn stats_respect_the_view_filter_but_selection_tallies_do_not() {
        let rows = sample_rows();
        let selected: HashSet<String> = ["/p/c.mp4".to_string()].into();
        let s = compute_stats(rows.iter(), |r| r.file_type != "mp4", &selected, |_| false);
        assert_eq!((s.visible, s.videos), (4, 0));
        assert_eq!(s.sel_videos, 1);
    }

    #[test]
    fn the_default_settings_show_the_hidden_minimum_size_filter() {
        let chips = active_chips(&UiSettings::default(), "", true, true, true);
        assert_eq!(chips, vec![ChipSpec { kind: ChipKind::Size, label: "Size >= 10 KiB".into() }]);
    }

    #[test]
    fn chips_cover_every_active_filter_and_none_when_clear() {
        let mut s = UiSettings::default();
        s.db_min_size_bytes = None;
        assert!(active_chips(&s, "", true, true, true).is_empty());
        s.db_max_size_bytes = Some(5 * 1024 * 1024);
        s.filter_modified_after = Some("2024-01-01".into());
        s.filter_skip_icons = true;
        let chips = active_chips(&s, "cat", false, true, true);
        let kinds: Vec<_> = chips.iter().map(|c| c.kind).collect();
        assert_eq!(kinds, vec![ChipKind::Size, ChipKind::Dates, ChipKind::SkipIcons, ChipKind::Search, ChipKind::HiddenTypes]);
        assert_eq!(chips[0].label, "Size <= 5 MiB");
        assert_eq!(chips[1].label, "Since 2024-01-01");
        assert_eq!(chips[4].label, "Hiding: Images");
    }

    fn entries() -> Vec<Entry> {
        vec![
            Entry { priority: 100, full: 150.0, compact: Some(50.0) },
            Entry { priority: 80, full: 70.0, compact: Some(30.0) },
            Entry { priority: 60, full: 130.0, compact: Some(60.0) },
            Entry { priority: 40, full: 160.0, compact: Some(60.0) },
            Entry { priority: 30, full: 160.0, compact: Some(60.0) },
        ]
    }

    #[test]
    fn everything_stays_full_when_it_fits() {
        let p = plan(&entries(), 8.0, 28.0, 2000.0);
        assert!(p.modes.iter().all(|m| *m == Some(Mode::Full)));
        assert!(!p.overflow);
    }

    #[test]
    fn meters_shrink_before_anything_collapses_lowest_priority_first() {
        let es = entries();
        let full: f32 = es.iter().map(|e| e.full).sum::<f32>() + 8.0 * 4.0;
        let p = plan(&es, 8.0, 28.0, full - 10.0);
        assert_eq!(p.modes[4], Some(Mode::Compact), "lowest priority shrinks first");
        assert_eq!(p.modes[0], Some(Mode::Full), "highest priority untouched");
        assert!(!p.overflow);
    }

    #[test]
    fn the_lowest_priority_items_collapse_into_the_overflow_once_all_are_compact() {
        let es = entries();
        let compact_total: f32 = es.iter().map(|e| e.compact.unwrap()).sum::<f32>() + 8.0 * 4.0;
        let p = plan(&es, 8.0, 28.0, compact_total - 1.0);
        assert!(p.overflow);
        assert_eq!(p.modes[4], None);
        assert_eq!(p.modes[0], Some(Mode::Compact));
    }

    #[test]
    fn a_plan_never_exceeds_the_available_width_while_anything_can_still_collapse() {
        let es = entries();
        for avail in (80..=900).step_by(7) {
            let avail = avail as f32;
            let p = plan(&es, 8.0, 28.0, avail);
            let width = plan_width(&es, &p.modes, 8.0, 28.0);
            let shown = p.modes.iter().filter(|m| m.is_some()).count();
            assert!(width <= avail || shown == 0, "avail {avail}: width {width} with {shown} shown");
        }
    }

    #[test]
    fn wider_space_never_shows_fewer_items() {
        let es = entries();
        let mut last_shown = 0;
        for avail in (0..=1000).step_by(5) {
            let p = plan(&es, 8.0, 28.0, avail as f32);
            let shown = p.modes.iter().filter(|m| m.is_some()).count();
            assert!(shown >= last_shown, "avail {avail} showed {shown} after {last_shown}");
            last_shown = shown;
        }
    }

    fn test_ctx() -> Context {
        let ctx = Context::default();
        ctx.set_fonts(crate::theme::fonts::definitions(crate::theme::fonts::TextScale::Normal));
        crate::theme::bootstrap_startup_theme(&ctx);
        ctx
    }

    fn run_frame(ctx: &Context, width: f32, mut f: impl FnMut(&mut Ui)) {
        let input = RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(width, 600.0))),
            ..Default::default()
        };
        ctx.run_ui(input, |ui| f(ui)).drop_without_applying_deltas();
    }

    fn all_meters() -> SystemMetrics {
        SystemMetrics { cpu01: Some(0.4), ram: Some((8000.0, 16000.0)), vram: Some((3000.0, 8000.0)) }
    }

    #[tokio::test]
    async fn the_bar_never_overlaps_from_640_to_1600() {
        let ctx = test_ctx();
        let mut ex = FileExplorer::new(true);
        for r in sample_rows() {
            ex.table.push(r);
        }
        let mut last_overflowed = usize::MAX;
        for width in [640.0_f32, 700.0, 800.0, 900.0, 1000.0, 1200.0, 1400.0, 1600.0] {
            for _ in 0..3 {
                run_frame(&ctx, width, |ui| show_with_metrics(ui, &mut ex, "overlap", all_meters()));
            }
            let l = last_layout(&ctx, "overlap").expect("the bar records its layout");
            assert!(l.left.right() <= l.right.left() + 0.5, "{width}px: left {:?} overlaps right {:?}", l.left, l.right);
            assert!(l.left.left() >= -0.5 && l.right.right() <= width + 0.5, "{width}px: bar leaves the window: {l:?}");
            assert!(l.overflowed <= last_overflowed, "{width}px: wider window collapsed more items");
            last_overflowed = l.overflowed;
        }
        assert_eq!(last_overflowed, 0, "everything fits at 1600px");
        for _ in 0..3 {
            run_frame(&ctx, 420.0, |ui| show_with_metrics(ui, &mut ex, "overlap", all_meters()));
        }
        assert!(last_layout(&ctx, "overlap").unwrap().overflowed > 0, "items collapse into the overflow below the supported width");
    }

    #[tokio::test]
    async fn stats_are_recomputed_only_when_the_table_revision_changes() {
        let ctx = test_ctx();
        let mut ex = FileExplorer::new(true);
        for r in sample_rows() {
            ex.table.push(r);
        }
        for _ in 0..6 {
            run_frame(&ctx, 1000.0, |ui| show_with_metrics(ui, &mut ex, "cache", SystemMetrics::default()));
        }
        assert_eq!(computes("cache"), 1, "idle frames reuse the cached stats");
        ex.table.push(row("/p/new.jpg", "jpg", 5));
        for _ in 0..3 {
            run_frame(&ctx, 1000.0, |ui| show_with_metrics(ui, &mut ex, "cache", SystemMetrics::default()));
        }
        assert_eq!(computes("cache"), 2, "a new row is one recompute");
        ex.viewer.selected.insert("/p/new.jpg".to_string());
        for _ in 0..3 {
            run_frame(&ctx, 1000.0, |ui| show_with_metrics(ui, &mut ex, "cache", SystemMetrics::default()));
        }
        assert_eq!(computes("cache"), 3, "a selection change is one recompute");
    }

    #[tokio::test]
    async fn the_bar_hides_meters_the_platform_cannot_provide() {
        let ctx = test_ctx();
        let mut ex = FileExplorer::new(true);
        for r in sample_rows() {
            ex.table.push(r);
        }
        let none = items_for(&ctx, &ex, SystemMetrics::default());
        assert!(!none.iter().any(ItemKind::is_meter));
        let all = items_for(&ctx, &ex, all_meters());
        assert_eq!(all.iter().filter(|i| i.is_meter()).count(), 3);
    }

    fn items_for(ctx: &Context, ex: &FileExplorer, metrics: SystemMetrics) -> Vec<ItemKind> {
        let stats = cached_stats(ctx, ex, "items");
        build_items(ctx, ex, "items", &stats, Vec::new(), &metrics)
    }

    #[tokio::test]
    async fn clearing_the_size_chip_removes_the_hidden_default_filter() {
        let mut ex = FileExplorer::new(true);
        let store = crate::settings::store();
        store.update("test:seed", |s| s.db_min_size_bytes = Some(10 * 1024));
        apply_action(&mut ex, BarAction::ClearChip(ChipKind::Size));
        assert_eq!(store.read(|s| (s.db_min_size_bytes, s.db_max_size_bytes)), (None, None));
    }
}
