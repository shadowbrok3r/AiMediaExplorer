//! The thumbnail grid: a virtualized view over the row store and view index with the selection, keyboard and menu behaviour of the details view.

use std::collections::VecDeque;
use std::ops::Range;
use std::time::{Duration, Instant};

use egui::scroll_area::{DragScroll, ScrollBarVisibility, ScrollSource};
use egui::style::ScrollAnimation;
use egui::{Context, CornerRadius, Pos2, Rect, ScrollArea, Sense, Stroke, StrokeKind, Ui, vec2};

use super::details::{DetailsEvent, DetailsInputs, row_menu};
use super::grid_facts::{ProbeKind, TileFacts};
use super::grid_layout::{GAP, GridLayout, Motion};
use super::grid_tile::{self, TileColors, TileDraw};
use super::keys::{self, GridKey, KeyAction};
use super::view_mode::{TILE_STEP, TileFit, clamp_tile};
use crate::ui::file_table::rows::{PathKey, RowFacts, RowKind, ViewRef};
use crate::ui::file_table::thumbs::{ThumbView, Tier};

const FRAME_SAMPLES: usize = 240;
/// Pointer travel before a press on empty space becomes a band.
const BAND_START_PX: f32 = 6.0;
/// Distance from the top or bottom edge where a band drag scrolls.
const BAND_EDGE_PX: f32 = 28.0;
/// Width at the right edge where a press grabs the scroll bar instead.
const SCROLL_BAR_PX: f32 = 14.0;
/// Points per second a band drag scrolls at the very edge.
const BAND_SCROLL_SPEED: f32 = 900.0;

/// How tiles look; the explorer fills it from the view prefs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridStyle {
    pub tile: f32,
    pub fit: TileFit,
    pub captions: bool,
    pub badges: bool,
}

/// The tiles a band covers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BandChange {
    /// The first change of a band.
    pub begin: bool,
    /// Ctrl was held at the press: the band adds to the selection.
    pub additive: bool,
    pub rows: Range<usize>,
    pub cols: Range<usize>,
    pub grid_cols: usize,
}

/// Something the user did in the grid.
#[derive(Clone, Debug)]
pub enum GridEvent {
    /// An action shared with the details view.
    Details(DetailsEvent),
    /// Arrow or page key.
    Move { motion: Motion, extend: bool },
    Band(BandChange),
    /// The user asked for another tile size.
    TileSize(f32),
}

/// Where the next frame scrolls to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scroll {
    /// Just far enough to show the tile at this view position.
    Reveal(usize),
    /// The row of this view position at the top.
    Pos(usize),
    Offset(f32),
    Top,
}

/// Automatic scrolling up and down for manual performance checks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stress {
    pub rows_per_s: f32,
    dir: f32,
}

impl Stress {
    pub fn new(rows_per_s: f32) -> Self {
        Self { rows_per_s: rows_per_s.max(0.0), dir: 1.0 }
    }

    /// The offset one `dt` later, turning around at either end.
    pub fn step(&mut self, offset: f32, max: f32, row_h: f32, dt: f32) -> f32 {
        let next = offset + self.dir * self.rows_per_s * row_h * dt;
        if next >= max {
            self.dir = -1.0;
            max.max(0.0)
        } else if next <= 0.0 {
            self.dir = 1.0;
            0.0
        } else {
            next
        }
    }
}

/// Counters of the last drawn frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GridStats {
    pub tiles_drawn: usize,
    pub thumbs_requested: usize,
    pub ahead_requested: usize,
    pub pending: usize,
    pub frames: u64,
}

#[derive(Clone, Copy, Debug)]
struct Press {
    origin: Pos2,
    content: Pos2,
    additive: bool,
    on_selected: bool,
}

#[derive(Clone, Debug, Default)]
struct BandRun {
    last: Option<(Range<usize>, Range<usize>)>,
}

#[derive(Default)]
struct Anchor {
    rev: u64,
    key_rev: u64,
    pos: usize,
    key: Option<PathKey>,
    /// Points the viewport top sits below the top row.
    sub: f32,
}

pub struct GridView {
    layout: Option<GridLayout>,
    offset: f32,
    viewport_h: f32,
    content_h: f32,
    visible: Range<usize>,
    ahead: (Range<usize>, Range<usize>),
    intent: Option<Scroll>,
    kb_active: bool,
    press: Option<Press>,
    band: Option<BandRun>,
    anchor: Anchor,
    stress: Option<Stress>,
    pub facts: TileFacts,
    frame_ms: VecDeque<f32>,
    stats: GridStats,
    cells: Vec<(usize, Rect)>,
    checkboxes: Vec<Rect>,
    order: Vec<(f32, usize)>,
    views: Vec<ThumbView>,
}

impl Default for GridView {
    fn default() -> Self {
        Self {
            layout: None,
            offset: 0.0,
            viewport_h: 0.0,
            content_h: 0.0,
            visible: 0..0,
            ahead: (0..0, 0..0),
            intent: None,
            kb_active: false,
            press: None,
            band: None,
            anchor: Anchor::default(),
            stress: None,
            facts: TileFacts::default(),
            frame_ms: VecDeque::new(),
            stats: GridStats::default(),
            cells: Vec::new(),
            checkboxes: Vec::new(),
            order: Vec::new(),
            views: Vec::new(),
        }
    }
}

fn wants_thumb(kind: RowKind, path: &str) -> bool {
    matches!(kind, RowKind::Image | RowKind::Video) && !path.is_empty()
}

impl GridView {
    pub fn new() -> Self {
        Self::default()
    }

    /// Columns of the last drawn layout.
    pub fn cols(&self) -> usize {
        self.layout.map_or(1, |l| l.cols)
    }

    pub fn layout(&self) -> Option<GridLayout> {
        self.layout
    }

    /// Tile rows that fit the viewport, as of the last frame.
    pub fn page_rows(&self) -> usize {
        self.layout.map_or(8, |l| l.rows_per_screen(self.viewport_h))
    }

    pub fn offset(&self) -> f32 {
        self.offset
    }

    pub fn viewport_height(&self) -> f32 {
        self.viewport_h
    }

    pub fn content_height(&self) -> f32 {
        self.content_h
    }

    /// View positions of the tiles drawn in the last frame.
    pub fn visible(&self) -> Range<usize> {
        self.visible.clone()
    }

    /// Positions requested ahead of the viewport, above and below.
    pub fn ahead(&self) -> (Range<usize>, Range<usize>) {
        self.ahead.clone()
    }

    pub fn stats(&self) -> GridStats {
        self.stats
    }

    /// Position of the first tile in the top visible row.
    pub fn top_pos(&self) -> usize {
        self.anchor.pos
    }

    /// View position and screen rectangle of each tile drawn in the last frame.
    pub fn cell_rects(&self) -> &[(usize, Rect)] {
        &self.cells
    }

    pub fn keyboard_active(&self) -> bool {
        self.kb_active
    }

    pub fn set_keyboard_active(&mut self, active: bool) {
        self.kb_active = active;
    }

    /// CPU time of recent frames in milliseconds, oldest first.
    pub fn frame_times(&self) -> &VecDeque<f32> {
        &self.frame_ms
    }

    pub fn clear_frame_times(&mut self) {
        self.frame_ms.clear();
    }

    /// Scrolls on the next frame.
    pub fn scroll(&mut self, to: Scroll) {
        self.intent = Some(to);
    }

    pub fn stress(&self) -> Option<Stress> {
        self.stress
    }

    /// Starts or, with `None`, stops scrolling up and down by itself.
    pub fn set_stress(&mut self, rows_per_s: Option<f32>) {
        self.stress = rows_per_s.filter(|r| *r > 0.0).map(Stress::new);
    }

    /// Whether a band selection is running.
    pub fn band_active(&self) -> bool {
        self.band.is_some()
    }

    fn record(&mut self, started: Instant) {
        if self.frame_ms.len() >= FRAME_SAMPLES {
            self.frame_ms.pop_front();
        }
        self.frame_ms.push_back(started.elapsed().as_secs_f32() * 1000.0);
        self.stats.frames += 1;
    }

    /// Draws the grid into `ui` and returns what the user did.
    pub fn show(&mut self, ui: &mut Ui, input: &DetailsInputs<'_>, style: &GridStyle) -> Vec<GridEvent> {
        let started = Instant::now();
        let ctx = ui.ctx().clone();
        let mut events = Vec::new();
        let total = input.view.len();
        let area = ui.available_rect_before_wrap();
        let hovered = ui.rect_contains_pointer(area);
        if ui.input(|i| i.pointer.any_pressed()) {
            self.kb_active = hovered;
        }
        self.read_input(ui, hovered, style, &mut events);
        self.facts.pump(&ctx);
        let colors = TileColors::new(ui);
        let caption_h = if style.captions { colors.caption_h } else { 0.0 };
        let scroll_style = &ui.spacing().scroll;
        let est_w = (ui.available_width() - scroll_style.allocated_width() - scroll_style.content_margin.sum().x).max(1.0);
        let est = GridLayout::new(est_w, style.tile, GAP, caption_h);
        let jump = self.next_offset(&ctx, input, &est, total);
        let mut scroll = ScrollArea::vertical()
            .id_salt("ame_grid_scroll")
            .auto_shrink([false, false])
            .scroll_source(ScrollSource { scroll_bar: true, drag: DragScroll::Never, mouse_wheel: true })
            .scroll_bar_visibility(ScrollBarVisibility::AlwaysVisible);
        if let Some(offset) = jump {
            scroll = scroll.vertical_scroll_offset(offset.clamp(0.0, est.max_offset(total, self.viewport_h)));
        }
        scroll.show_viewport(ui, |ui, viewport| self.draw(ui, viewport, input, style, &colors, &mut events));
        self.record(started);
        events
    }

    fn read_input(&mut self, ui: &mut Ui, hovered: bool, style: &GridStyle, events: &mut Vec<GridEvent>) {
        let unfocused = ui.memory(|m| m.focused().is_none());
        if !ui.ctx().egui_wants_keyboard_input() && !egui::Popup::is_any_open(ui.ctx()) && (self.kb_active || (hovered && unfocused)) {
            let page = self.page_rows();
            for key in ui.input_mut(|i| keys::read_grid(i, page)) {
                events.push(match key {
                    GridKey::Move { motion, extend } => GridEvent::Move { motion, extend },
                    GridKey::Zoom(dir) => GridEvent::TileSize(clamp_tile(style.tile + dir as f32 * TILE_STEP)),
                    GridKey::Common(action) => GridEvent::Details(DetailsEvent::Key(action)),
                });
            }
        }
        if hovered {
            let zoom = ui.input(|i| i.zoom_delta());
            if (zoom - 1.0).abs() > 1e-3 {
                events.push(GridEvent::TileSize(clamp_tile(style.tile * zoom)));
            }
        }
    }

    // The offset to scroll to this frame, from a pending request, a changed layout, a changed view or the stress scroller.
    fn next_offset(&mut self, ctx: &Context, input: &DetailsInputs<'_>, est: &GridLayout, total: usize) -> Option<f32> {
        let rev = input.view.revision();
        let mut jump = None;
        if let Some(old) = self.layout {
            let changed = old.cols != est.cols || (old.tile - est.tile).abs() > 0.5 || (old.caption_h - est.caption_h).abs() > 0.5;
            if changed && total > 0 {
                jump = Some(est.offset_of_row(est.row_of(self.anchor.pos)));
            }
        }
        if jump.is_none() && self.anchor.rev != rev && total > 0 && self.offset > 0.5 {
            let moved = self.anchor.key.as_ref().and_then(|key| ViewRef::new(input.store, input.view).pos_of(key));
            if let Some(pos) = moved.filter(|p| *p != self.anchor.pos) {
                jump = Some(est.offset_of_row(est.row_of(pos)) + self.anchor.sub);
            }
        }
        self.anchor.rev = rev;
        match self.intent.take() {
            Some(Scroll::Top) => jump = Some(0.0),
            Some(Scroll::Offset(o)) => jump = Some(o),
            Some(Scroll::Pos(pos)) => jump = Some(est.offset_of_row(est.row_of(pos))),
            Some(Scroll::Reveal(pos)) => jump = est.reveal_offset(pos, self.offset, self.viewport_h, total).or(jump),
            None => {}
        }
        if let Some(stress) = self.stress.as_mut() {
            let dt = ctx.input(|i| i.stable_dt).min(0.1);
            jump = Some(stress.step(self.offset, est.max_offset(total, self.viewport_h), est.row_h(), dt));
            ctx.request_repaint();
        }
        jump
    }

    fn draw(&mut self, ui: &mut Ui, viewport: Rect, input: &DetailsInputs<'_>, style: &GridStyle, colors: &TileColors, events: &mut Vec<GridEvent>) {
        let total = input.view.len();
        let width = ui.available_width();
        let caption_h = if style.captions { colors.caption_h } else { 0.0 };
        let layout = GridLayout::new(width, style.tile, GAP, caption_h);
        self.layout = Some(layout);
        self.offset = viewport.min.y;
        self.viewport_h = viewport.height();
        self.cells.clear();
        self.checkboxes.clear();
        self.stats.tiles_drawn = 0;
        if total == 0 {
            self.visible = 0..0;
            self.ahead = (0..0, 0..0);
            self.content_h = 0.0;
            let hidden = input.store.len();
            let note = if hidden == 0 { "No items".to_string() } else { format!("{hidden} items hidden by the current filter") };
            ui.vertical_centered(|ui| {
                ui.add_space(crate::theme::metrics(ui.ctx()).gap.y * 8.0);
                ui.weak(note);
            });
            return;
        }
        let content_h = layout.content_height(total);
        self.content_h = content_h;
        let (content, _) = ui.allocate_exact_size(vec2(width, content_h), Sense::hover());
        let origin = content.min.to_vec2();
        let background = ui.interact(Rect::from_min_size(content.min, vec2(width, content_h.max(viewport.height()))), ui.scope_id().with("ame_grid_background"), Sense::click());

        let rows = layout.visible_rows(viewport.min.y, viewport.height(), total);
        let vis = layout.positions_of_rows(rows.clone(), total);
        let screen_rows = layout.rows_per_screen(viewport.height());
        let all_rows = layout.rows(total);
        let before = layout.positions_of_rows(rows.start.saturating_sub(screen_rows)..rows.start, total);
        let after = layout.positions_of_rows(rows.end..(rows.end + screen_rows).min(all_rows), total);
        self.visible = vis.clone();
        self.ahead = (before.clone(), after.clone());
        self.request_thumbs(input, &layout, viewport, vis.clone(), before, after);

        let viewer = input.viewer;
        let mods = ui.input(|i| i.modifiers);
        let time = ui.input(|i| i.time);
        let focus = input.selection.focus();
        let multi = !input.selection.is_empty();
        let banding = self.band.is_some();
        let views = std::mem::take(&mut self.views);
        let mut menu_events: Vec<DetailsEvent> = Vec::new();
        let mut pending = 0usize;
        for pos in vis.clone() {
            let Some(idx) = input.view.store_index(pos) else { continue };
            let (Some(row), Some(key)) = (input.store.get(idx), input.store.key_at(idx)) else { continue };
            let thumb = views.get(pos - vis.start).copied().unwrap_or(ThumbView::Missing);
            let kind = row.kind();
            let cell = layout.cell_rect(pos).translate(origin);
            let image = layout.image_rect(pos).translate(origin);
            let selected = input.selection.contains(key);
            let id = ui.scope_id().with(("ame_grid_tile", pos));
            let resp = ui.interact(cell, id, Sense::click());
            let hover = !banding && ui.rect_contains_pointer(cell);
            if matches!(thumb, ThumbView::Pending) {
                pending += 1;
            }
            self.want_facts(input, row, kind, style);
            let facts = self.facts.get(&row.path);
            let draw = TileDraw {
                row,
                kind,
                thumb,
                facts,
                selected,
                focused: focus == Some(key),
                hovered: hover,
                multi,
                indexed: viewer.mode == super::ExplorerMode::Database || row.db_created.is_some(),
                fit: style.fit,
                captions: style.captions,
                badges: style.badges,
                time,
                cell: (layout.row_of(pos), layout.col_of(pos)),
            };
            let parts = grid_tile::paint(ui, colors, cell, image, &draw);
            self.cells.push((pos, cell));
            let mut on_check = false;
            if let Some(check) = parts.checkbox {
                self.checkboxes.push(check);
                let check_resp = ui.interact(check, id.with("check"), Sense::click());
                if check_resp.clicked() {
                    on_check = true;
                    events.push(GridEvent::Details(DetailsEvent::Click { key: key.clone(), ctrl: true, shift: mods.shift }));
                    self.kb_active = true;
                }
            }
            if !on_check {
                if resp.clicked() {
                    events.push(GridEvent::Details(DetailsEvent::Click { key: key.clone(), ctrl: mods.command, shift: mods.shift }));
                    self.kb_active = true;
                }
                if resp.double_clicked() {
                    events.push(GridEvent::Details(DetailsEvent::Open { key: key.clone() }));
                }
                if resp.secondary_clicked() {
                    events.push(GridEvent::Details(DetailsEvent::ContextOn { key: key.clone() }));
                }
            }
            if hover {
                let tip = resp.clone();
                tip.on_hover_ui(|ui| {
                    ui.label(egui::RichText::new(&row.filename).strong());
                    ui.weak(&row.path);
                });
            }
            resp.context_menu(|ui| row_menu(ui, input, key, row, &mut menu_events));
        }
        self.views = views;
        events.extend(menu_events.into_iter().map(GridEvent::Details));
        self.stats.tiles_drawn = self.cells.len();
        self.stats.pending = pending;
        if pending > 0 {
            ui.ctx().request_repaint_after(Duration::from_millis(50));
        }

        self.handle_band(ui, input, &layout, content.min, events);
        if background.clicked() && !self.checkboxes.iter().any(|c| ui.rect_contains_pointer(*c)) {
            events.push(GridEvent::Details(DetailsEvent::Key(KeyAction::ClearSelection)));
        }
        self.track_anchor(input, &layout, &viewport, rows.start, vis.start);
    }

    // Requests the visible tiles nearest the viewport centre first, then the screen before and after.
    fn request_thumbs(&mut self, input: &DetailsInputs<'_>, layout: &GridLayout, viewport: Rect, vis: Range<usize>, before: Range<usize>, after: Range<usize>) {
        let (svc, scope) = (&input.viewer.thumbs, input.viewer.scope);
        let center = viewport.center().y;
        let distance = |pos: usize| (layout.cell_rect(pos).center().y - center).abs();
        self.order.clear();
        self.order.extend(vis.clone().map(|pos| (distance(pos), pos)));
        self.order.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        self.views.clear();
        self.views.resize(vis.len(), ThumbView::Missing);
        let mut requested = 0;
        for &(prio, pos) in &self.order {
            let Some(row) = input.view.store_index(pos).and_then(|i| input.store.get(i)) else { continue };
            if wants_thumb(row.kind(), &row.path) {
                self.views[pos - vis.start] = svc.request(scope, &row.path, Tier::Thumb, prio);
                requested += 1;
            }
        }
        let mut ahead = 0;
        for pos in before.chain(after) {
            let Some(row) = input.view.store_index(pos).and_then(|i| input.store.get(i)) else { continue };
            if wants_thumb(row.kind(), &row.path) {
                svc.request_ahead(scope, &row.path, Tier::Thumb, distance(pos));
                ahead += 1;
            }
        }
        self.stats.thumbs_requested = requested;
        self.stats.ahead_requested = ahead;
    }

    fn want_facts(&mut self, input: &DetailsInputs<'_>, row: &crate::Thumbnail, kind: RowKind, style: &GridStyle) {
        if row.path.is_empty() || row.path.contains("://") {
            return;
        }
        let library = input.viewer.mode == super::ExplorerMode::Database;
        match kind {
            RowKind::Dir if !library => self.facts.want(&row.path, ProbeKind::DirItems),
            RowKind::Dir => {}
            _ => {
                if library {
                    self.facts.want(&row.path, ProbeKind::Offline);
                }
                if kind == RowKind::Video {
                    self.facts.want(&row.path, ProbeKind::Duration);
                }
                if style.badges && crate::database::is_connected() {
                    self.facts.want(&row.path, ProbeKind::Collections);
                }
            }
        }
    }

    // Starts, updates and ends a rubber-band selection from raw pointer state.
    fn handle_band(&mut self, ui: &mut Ui, input: &DetailsInputs<'_>, layout: &GridLayout, origin: Pos2, events: &mut Vec<GridEvent>) {
        let total = input.view.len();
        let clip = ui.clip_rect();
        let (pressed, down, latest, mods) = ui.input(|i| (i.pointer.primary_pressed(), i.pointer.primary_down(), i.pointer.latest_pos(), i.modifiers));
        if pressed {
            self.press = None;
            self.band = None;
            if let Some(at) = latest.filter(|p| clip.contains(*p) && p.x < clip.right() - SCROLL_BAR_PX && ui.rect_contains_pointer(clip)) {
                if !self.checkboxes.iter().any(|c| c.contains(at)) {
                    let content = Pos2::ZERO + (at - origin);
                    let on_selected = layout
                        .pos_at(content, total)
                        .and_then(|pos| ViewRef::new(input.store, input.view).key_at(pos))
                        .is_some_and(|key| input.selection.contains(key));
                    self.press = Some(Press { origin: at, content, additive: mods.command, on_selected });
                }
            }
        }
        if !down {
            self.press = None;
            self.band = None;
            return;
        }
        let (Some(press), Some(at)) = (self.press, latest) else { return };
        if self.band.is_none() {
            if press.on_selected || at.distance(press.origin) < BAND_START_PX {
                return;
            }
            self.band = Some(BandRun { last: None });
        }
        let begin = self.band.as_ref().is_some_and(|b| b.last.is_none());
        let rect = Rect::from_two_pos(press.content, Pos2::ZERO + (at - origin));
        let cells = layout.band_cells(rect, total);
        let changed = self.band.as_ref().is_some_and(|b| b.last != cells || begin);
        if changed {
            let (rows, cols) = cells.clone().unwrap_or((0..0, 0..0));
            events.push(GridEvent::Band(BandChange { begin, additive: press.additive, rows, cols, grid_cols: layout.cols }));
            if let Some(run) = self.band.as_mut() {
                run.last = cells;
            }
        }
        let screen = Rect::from_two_pos(press.content + origin.to_vec2(), at).intersect(clip);
        let sel = ui.visuals().selection;
        ui.painter().rect(screen, CornerRadius::same(2), sel.bg_fill.gamma_multiply(0.35), Stroke::new(1.0, sel.stroke.color), StrokeKind::Inside);
        let dt = ui.input(|i| i.stable_dt).min(0.1);
        let toward_top = (clip.top() + BAND_EDGE_PX - at.y).max(0.0) / BAND_EDGE_PX;
        let toward_bottom = (at.y - (clip.bottom() - BAND_EDGE_PX)).max(0.0) / BAND_EDGE_PX;
        let speed = (toward_top - toward_bottom).clamp(-1.0, 1.0) * BAND_SCROLL_SPEED * dt;
        if speed != 0.0 {
            ui.scroll_with_delta_animation(vec2(0.0, speed), ScrollAnimation::none());
        }
        ui.ctx().request_repaint();
    }

    // Remembers the tile at the top so a changed view can keep it in place.
    fn track_anchor(&mut self, input: &DetailsInputs<'_>, layout: &GridLayout, viewport: &Rect, top_row: usize, top_pos: usize) {
        let rev = input.view.revision();
        if self.anchor.pos != top_pos || self.anchor.key_rev != rev || self.anchor.key.is_none() {
            self.anchor.pos = top_pos;
            self.anchor.key_rev = rev;
            self.anchor.key = ViewRef::new(input.store, input.view).key_at(top_pos).cloned();
        }
        self.anchor.sub = (viewport.min.y - layout.offset_of_row(top_row)).max(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stress_scroller_turns_around_at_both_ends() {
        let mut s = Stress::new(10.0);
        let (max, row_h) = (1000.0, 100.0);
        let mut offset = 0.0;
        let mut turned_down = false;
        let mut turned_up = false;
        for _ in 0..400 {
            let before = s.dir;
            offset = s.step(offset, max, row_h, 0.05);
            assert!((0.0..=max).contains(&offset), "{offset}");
            turned_down |= before > 0.0 && s.dir < 0.0;
            turned_up |= before < 0.0 && s.dir > 0.0;
        }
        assert!(turned_down && turned_up);
    }

    #[test]
    fn the_stress_speed_is_rows_per_second() {
        let mut s = Stress::new(4.0);
        assert!((s.step(0.0, 10_000.0, 50.0, 0.5) - 100.0).abs() < 1e-3);
        let mut none = Stress::new(-3.0);
        assert_eq!(none.step(10.0, 100.0, 50.0, 1.0), 10.0, "a negative speed is stopped");
    }

    #[test]
    fn a_stress_scroller_in_content_that_fits_stays_at_the_top() {
        let mut s = Stress::new(5.0);
        assert_eq!(s.step(0.0, 0.0, 100.0, 0.1), 0.0);
        assert_eq!(s.step(0.0, -20.0, 100.0, 0.1), 0.0);
    }

    #[test]
    fn only_images_and_videos_ask_for_thumbnails() {
        assert!(wants_thumb(RowKind::Image, "/a.jpg"));
        assert!(wants_thumb(RowKind::Video, "/a.mp4"));
        assert!(!wants_thumb(RowKind::Image, ""));
        for kind in [RowKind::Dir, RowKind::Archive, RowKind::Other] {
            assert!(!wants_thumb(kind, "/a"));
        }
    }
}
