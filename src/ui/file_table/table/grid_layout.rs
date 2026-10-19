//! Tile grid geometry: columns for a width, position and row mapping, visible ranges, hit tests, keyboard moves and image fitting.

use std::ops::Range;

use egui::{Pos2, Rect, Vec2, pos2, vec2};

use crate::ui::file_table::rows::Step;

/// Space between tiles.
pub const GAP: f32 = 6.0;
/// Smallest tile a layout produces.
const MIN_TILE: f32 = 16.0;

/// Columns and sizes of a grid laid out for one width.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridLayout {
    pub cols: usize,
    /// Edge of the square image area.
    pub tile: f32,
    pub gap: f32,
    /// Height of the caption under the image; zero without captions.
    pub caption_h: f32,
}

impl GridLayout {
    /// The layout for `width` where tiles are at least `target` wide; the tiles grow to use the whole width.
    pub fn new(width: f32, target: f32, gap: f32, caption_h: f32) -> Self {
        let width = if width.is_finite() { width.max(1.0) } else { 1.0 };
        let target = if target.is_finite() { target.max(MIN_TILE) } else { MIN_TILE };
        let cols = (((width + gap) / (target + gap)).floor() as usize).max(1);
        let tile = ((width - gap * (cols as f32 - 1.0)) / cols as f32).max(1.0);
        Self { cols, tile, gap, caption_h: caption_h.max(0.0) }
    }

    /// Distance between the left edges of neighbouring tiles.
    pub fn cell_w(&self) -> f32 {
        self.tile + self.gap
    }

    /// Height of a tile with its caption.
    pub fn cell_h(&self) -> f32 {
        self.tile + self.caption_h
    }

    /// Distance between the top edges of neighbouring rows.
    pub fn row_h(&self) -> f32 {
        self.cell_h() + self.gap
    }

    /// Rows needed for `total` tiles.
    pub fn rows(&self, total: usize) -> usize {
        total.div_ceil(self.cols)
    }

    pub fn content_height(&self, total: usize) -> f32 {
        self.rows(total) as f32 * self.row_h()
    }

    /// Largest scroll offset for a viewport `viewport_h` tall.
    pub fn max_offset(&self, total: usize, viewport_h: f32) -> f32 {
        (self.content_height(total) - viewport_h).max(0.0)
    }

    pub fn row_of(&self, pos: usize) -> usize {
        pos / self.cols
    }

    pub fn col_of(&self, pos: usize) -> usize {
        pos % self.cols
    }

    /// Position of the tile at `row` and `col`, when there is one.
    pub fn pos_of(&self, row: usize, col: usize, total: usize) -> Option<usize> {
        let pos = row.checked_mul(self.cols)?.checked_add(col)?;
        (col < self.cols && pos < total).then_some(pos)
    }

    /// Scroll offset that puts `row` at the top.
    pub fn offset_of_row(&self, row: usize) -> f32 {
        row as f32 * self.row_h()
    }

    /// Tile with its caption, relative to the content origin.
    pub fn cell_rect(&self, pos: usize) -> Rect {
        let min = pos2(self.col_of(pos) as f32 * self.cell_w(), self.row_of(pos) as f32 * self.row_h());
        Rect::from_min_size(min, vec2(self.tile, self.cell_h()))
    }

    /// Image area of a tile, relative to the content origin.
    pub fn image_rect(&self, pos: usize) -> Rect {
        let cell = self.cell_rect(pos);
        Rect::from_min_size(cell.min, vec2(self.tile, self.tile))
    }

    /// Rows that touch a viewport starting `offset` down and `viewport_h` tall.
    pub fn visible_rows(&self, offset: f32, viewport_h: f32, total: usize) -> Range<usize> {
        let rows = self.rows(total);
        let row_h = self.row_h();
        let first = ((offset.max(0.0) / row_h).floor() as usize).min(rows);
        let last = (((offset.max(0.0) + viewport_h.max(0.0)) / row_h).ceil() as usize).min(rows);
        first..last.max(first)
    }

    /// Positions in `rows`.
    pub fn positions_of_rows(&self, rows: Range<usize>, total: usize) -> Range<usize> {
        let start = (rows.start * self.cols).min(total);
        let end = (rows.end * self.cols).min(total);
        start..end.max(start)
    }

    /// Rows in one screen of `viewport_h`.
    pub fn rows_per_screen(&self, viewport_h: f32) -> usize {
        ((viewport_h.max(0.0) / self.row_h()).floor() as usize).max(1)
    }

    /// The tile under `point` (relative to the content origin); gaps and the space past the last tile give none.
    pub fn pos_at(&self, point: Pos2, total: usize) -> Option<usize> {
        if point.x < 0.0 || point.y < 0.0 {
            return None;
        }
        let (col, row) = ((point.x / self.cell_w()).floor() as usize, (point.y / self.row_h()).floor() as usize);
        let inside = point.x - col as f32 * self.cell_w() <= self.tile && point.y - row as f32 * self.row_h() <= self.cell_h();
        if inside { self.pos_of(row, col, total) } else { None }
    }

    /// Rows and columns whose tiles a band `rect` (relative to the content origin) touches.
    pub fn band_cells(&self, rect: Rect, total: usize) -> Option<(Range<usize>, Range<usize>)> {
        if total == 0 || rect.max.x < 0.0 || rect.max.y < 0.0 {
            return None;
        }
        let first = |lo: f32, size: f32, step: f32| ((lo - size) / step).ceil().max(0.0) as usize;
        let last = |hi: f32, step: f32| (hi / step).floor().max(0.0) as usize;
        let (c0, c1) = (first(rect.min.x, self.tile, self.cell_w()), last(rect.max.x, self.cell_w()).min(self.cols - 1));
        let (r0, r1) = (first(rect.min.y, self.cell_h(), self.row_h()), last(rect.max.y, self.row_h()).min(self.rows(total).saturating_sub(1)));
        (c0 <= c1 && r0 <= r1).then(|| (r0..r1 + 1, c0..c1 + 1))
    }

    /// Scroll offset that brings the row of `pos` fully into a viewport, or none when it already is.
    pub fn reveal_offset(&self, pos: usize, offset: f32, viewport_h: f32, total: usize) -> Option<f32> {
        let top = self.row_of(pos) as f32 * self.row_h();
        let bottom = top + self.cell_h();
        let max = self.max_offset(total, viewport_h);
        if top < offset {
            Some(top.min(max))
        } else if bottom > offset + viewport_h {
            Some((bottom - viewport_h).clamp(0.0, max))
        } else {
            None
        }
    }
}

/// A keyboard move over the grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Motion {
    Left,
    Right,
    Up,
    Down,
    PageUp,
    PageDown,
}

/// Position `motion` leads to from `cur`; a move off the edge stays and a step down into a shorter last row lands on its last tile.
pub fn motion_target(cur: Option<usize>, total: usize, cols: usize, page_rows: usize, motion: Motion) -> Option<usize> {
    if total == 0 || cols == 0 {
        return None;
    }
    let last = total - 1;
    let Some(cur) = cur.map(|c| c.min(last)) else {
        return Some(match motion {
            Motion::Right | Motion::Down | Motion::PageDown => 0,
            Motion::Left | Motion::Up | Motion::PageUp => last,
        });
    };
    let by_rows = |rows: isize| {
        let last_row = (last / cols) as isize;
        let row = (cur / cols) as isize;
        let target_row = (row + rows).clamp(0, last_row) as usize;
        (target_row * cols + cur % cols).min(last)
    };
    let page = page_rows.saturating_sub(1).max(1) as isize;
    Some(match motion {
        Motion::Left => cur.saturating_sub(1),
        Motion::Right => (cur + 1).min(last),
        Motion::Up => by_rows(-1),
        Motion::Down => by_rows(1),
        Motion::PageUp => by_rows(-page),
        Motion::PageDown => by_rows(page),
    })
}

/// The selection step from `cur` to `target`, none when they are the same.
pub fn step_between(cur: usize, target: usize) -> Option<Step> {
    match target.cmp(&cur) {
        std::cmp::Ordering::Greater => Some(Step::Down(target - cur)),
        std::cmp::Ordering::Less => Some(Step::Up(cur - target)),
        std::cmp::Ordering::Equal => None,
    }
}

/// UV rectangle of a texture of `tex` size that fills a square, cropping the longer side.
pub fn cover_uv(tex: Vec2) -> Rect {
    if tex.x <= 0.0 || tex.y <= 0.0 {
        return Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));
    }
    if tex.x >= tex.y {
        let u = tex.y / tex.x;
        Rect::from_min_max(pos2((1.0 - u) / 2.0, 0.0), pos2((1.0 + u) / 2.0, 1.0))
    } else {
        let v = tex.x / tex.y;
        Rect::from_min_max(pos2(0.0, (1.0 - v) / 2.0), pos2(1.0, (1.0 + v) / 2.0))
    }
}

/// The largest rectangle of `tex`'s aspect that fits `bounds`, centred in it.
pub fn fit_rect(tex: Vec2, bounds: Rect) -> Rect {
    if tex.x <= 0.0 || tex.y <= 0.0 {
        return bounds;
    }
    let scale = (bounds.width() / tex.x).min(bounds.height() / tex.y);
    Rect::from_center_size(bounds.center(), tex * scale)
}

/// `m:ss`, or `h:mm:ss` from an hour on.
pub fn duration_text(secs: f32) -> String {
    let total = if secs.is_finite() { secs.max(0.0).round() as u64 } else { 0 };
    let (h, m, s) = (total / 3600, total / 60 % 60, total % 60);
    if h > 0 { format!("{h}:{m:02}:{s:02}") } else { format!("{m}:{s:02}") }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(width: f32, target: f32) -> GridLayout {
        GridLayout::new(width, target, GAP, 20.0)
    }

    #[test]
    fn columns_follow_the_width_and_tiles_fill_it() {
        let l = layout(1000.0, 160.0);
        assert_eq!(l.cols, 6);
        assert!((l.cols as f32 * l.tile + (l.cols as f32 - 1.0) * l.gap - 1000.0).abs() < 0.01, "tiles and gaps span the width");
        assert!(l.tile >= 160.0, "tiles never shrink below the target");
        assert_eq!(layout(160.0, 160.0).cols, 1);
        assert_eq!(layout(166.0, 160.0).cols, 1);
        assert_eq!(layout(326.0, 160.0).cols, 2, "two targets and one gap fit exactly");
        assert_eq!(layout(325.0, 160.0).cols, 1);
    }

    #[test]
    fn degenerate_widths_still_give_a_usable_layout() {
        for width in [0.0, -5.0, 1.0, f32::NAN, f32::INFINITY] {
            let l = layout(width, 160.0);
            assert!(l.cols >= 1 && l.tile >= 1.0 && l.tile.is_finite(), "{width}: {l:?}");
        }
        assert!(layout(500.0, f32::NAN).cols >= 1);
        assert!(layout(500.0, 0.0).cols >= 1, "a zero target is raised to the minimum tile");
    }

    #[test]
    fn a_bigger_target_never_adds_columns() {
        let mut prev = usize::MAX;
        for target in (64..=320).step_by(8) {
            let cols = layout(1100.0, target as f32).cols;
            assert!(cols <= prev, "{target}: {cols} > {prev}");
            prev = cols;
        }
    }

    #[test]
    fn positions_map_to_rows_and_columns_and_back() {
        let l = layout(1000.0, 160.0);
        let total = 100;
        for pos in 0..total {
            assert_eq!(l.pos_of(l.row_of(pos), l.col_of(pos), total), Some(pos));
        }
        assert_eq!(l.pos_of(0, l.cols, total), None, "column past the edge");
        assert_eq!(l.pos_of(l.rows(total), 0, total), None, "row past the end");
        assert_eq!(l.pos_of(l.rows(total) - 1, l.cols - 1, total), None, "ragged last row has no such tile");
        assert_eq!(l.pos_of(usize::MAX, 1, total), None, "overflow is not a position");
    }

    #[test]
    fn rows_and_content_height_cover_a_ragged_last_row() {
        let l = layout(1000.0, 160.0);
        assert_eq!(l.rows(0), 0);
        assert_eq!(l.rows(1), 1);
        assert_eq!(l.rows(6), 1);
        assert_eq!(l.rows(7), 2);
        assert_eq!(l.content_height(0), 0.0);
        assert!((l.content_height(13) - 3.0 * l.row_h()).abs() < 1e-3);
    }

    #[test]
    fn cell_rects_tile_without_overlap() {
        let l = layout(1000.0, 160.0);
        let a = l.cell_rect(0);
        let b = l.cell_rect(1);
        let below = l.cell_rect(l.cols);
        assert!(!a.intersects(b) && !a.intersects(below));
        assert!((b.min.x - a.max.x - l.gap).abs() < 1e-3);
        assert!((below.min.y - a.max.y - l.gap).abs() < 1e-3);
        assert_eq!(l.image_rect(0).height(), l.tile);
        assert!((l.cell_rect(l.cols - 1).max.x - 1000.0).abs() < 0.01, "the last column ends at the right edge");
    }

    fn exact() -> GridLayout {
        GridLayout { cols: 6, tile: 100.0, gap: 10.0, caption_h: 20.0 }
    }

    #[test]
    fn visible_rows_cover_the_viewport_and_clamp_to_the_content() {
        let l = exact();
        let total = 6 * 50;
        assert_eq!(l.row_h(), 130.0);
        assert_eq!(l.visible_rows(0.0, 390.0, total), 0..3);
        assert_eq!(l.visible_rows(65.0, 390.0, total), 0..4, "a partly scrolled row still counts");
        assert_eq!(l.visible_rows(1300.0, 325.0, total), 10..13);
        assert_eq!(l.visible_rows(6370.0, 650.0, total), 49..50);
        assert_eq!(l.visible_rows(65_000.0, 650.0, total), 50..50, "scrolled past the end shows nothing");
        assert_eq!(l.visible_rows(0.0, 400.0, 0), 0..0);
        assert_eq!(l.visible_rows(-30.0, 100.0, total), 0..1, "an overscrolled top clamps");
    }

    #[test]
    fn visible_positions_stop_at_the_last_tile() {
        let l = layout(1000.0, 160.0);
        assert_eq!(l.positions_of_rows(2..4, 100), 12..24);
        assert_eq!(l.positions_of_rows(2..4, 15), 12..15);
        assert_eq!(l.positions_of_rows(5..9, 15), 15..15);
    }

    #[test]
    fn the_visible_tile_count_is_bounded_by_the_screen_not_the_total() {
        let l = layout(1100.0, 96.0);
        for total in [1_000usize, 100_000, 5_000_000] {
            let rows = l.visible_rows(l.offset_of_row(total / l.cols / 2), 700.0, total);
            let pos = l.positions_of_rows(rows, total);
            assert!(pos.len() <= (700.0 / l.row_h()).ceil() as usize * l.cols + 2 * l.cols, "{total}: {} tiles", pos.len());
        }
    }

    #[test]
    fn hit_tests_find_tiles_and_miss_gaps() {
        let l = layout(1000.0, 160.0);
        let total = 20;
        let center = |pos| l.cell_rect(pos).center();
        for pos in [0, 5, 6, 19] {
            assert_eq!(l.pos_at(center(pos), total), Some(pos));
        }
        let in_gap = pos2(l.cell_rect(0).max.x + l.gap / 2.0, 10.0);
        assert_eq!(l.pos_at(in_gap, total), None);
        let in_row_gap = pos2(10.0, l.cell_rect(0).max.y + l.gap / 2.0);
        assert_eq!(l.pos_at(in_row_gap, total), None);
        assert_eq!(l.pos_at(center(19) + vec2(l.cell_w(), 0.0), total), None, "past the ragged end of the last row");
        assert_eq!(l.pos_at(pos2(-1.0, 5.0), total), None);
        assert_eq!(l.pos_at(pos2(5.0, 1e9), total), None);
    }

    #[test]
    fn a_band_selects_the_tiles_it_touches() {
        let l = layout(1000.0, 160.0);
        let total = 100;
        let c = |pos: usize| l.cell_rect(pos).center();
        let band = Rect::from_two_pos(c(1), c(l.cols + 3));
        let (rows, cols) = l.band_cells(band, total).unwrap();
        assert_eq!((rows, cols), (0..2, 1..4));
        let tiny = Rect::from_center_size(c(8), vec2(2.0, 2.0));
        assert_eq!(l.band_cells(tiny, total), Some((1..2, 2..3)));
        let gap_only = Rect::from_center_size(pos2(l.cell_rect(0).max.x + l.gap / 2.0, 30.0), vec2(1.0, 1.0));
        assert_eq!(l.band_cells(gap_only, total), None, "a band inside a gap touches nothing");
        let huge = Rect::from_min_max(pos2(-50.0, -50.0), pos2(1e6, 1e9));
        assert_eq!(l.band_cells(huge, total), Some((0..l.rows(total), 0..l.cols)));
        assert_eq!(l.band_cells(huge, 0), None);
        assert_eq!(l.band_cells(Rect::from_min_max(pos2(-30.0, -30.0), pos2(-5.0, -5.0)), total), None);
    }

    #[test]
    fn reveal_scrolls_the_least_that_shows_the_row() {
        let l = layout(1000.0, 160.0);
        let (rh, vh, total) = (l.row_h(), 500.0, 6 * 100);
        assert_eq!(l.reveal_offset(0, 0.0, vh, total), None);
        let below = l.reveal_offset(6 * 10, 0.0, vh, total).unwrap();
        assert!((below - (10.0 * rh + l.cell_h() - vh)).abs() < 1e-3, "the row ends at the bottom edge");
        let above = l.reveal_offset(6 * 2, 10.0 * rh, vh, total).unwrap();
        assert!((above - 2.0 * rh).abs() < 1e-3, "the row starts at the top edge");
        assert_eq!(l.reveal_offset(6 * 3, 2.0 * rh, vh, total), None);
        let last = l.reveal_offset(total - 1, 0.0, vh, total).unwrap();
        assert!(last <= l.max_offset(total, vh) + 1e-3, "never past the end");
        assert_eq!(l.reveal_offset(0, 50.0, vh, 3), Some(0.0), "never negative");
    }

    #[test]
    fn arrows_move_by_one_tile_or_one_row() {
        let (total, cols) = (20, 6);
        let go = |cur, m| motion_target(Some(cur), total, cols, 4, m);
        assert_eq!(go(7, Motion::Right), Some(8));
        assert_eq!(go(7, Motion::Left), Some(6));
        assert_eq!(go(7, Motion::Down), Some(13));
        assert_eq!(go(7, Motion::Up), Some(1));
        assert_eq!(go(5, Motion::Right), Some(6), "Right wraps to the next row");
        assert_eq!(go(6, Motion::Left), Some(5), "Left wraps to the previous row");
    }

    #[test]
    fn moves_stay_at_the_edges() {
        let (total, cols) = (20, 6);
        let go = |cur, m| motion_target(Some(cur), total, cols, 4, m);
        assert_eq!(go(0, Motion::Left), Some(0));
        assert_eq!(go(2, Motion::Up), Some(2), "the first row has nothing above");
        assert_eq!(go(19, Motion::Right), Some(19));
        assert_eq!(go(15, Motion::Down), Some(19), "a column the ragged last row lacks lands on its last tile");
        assert_eq!(go(19, Motion::Down), Some(19));
        assert_eq!(go(18, Motion::Down), Some(18));
    }

    #[test]
    fn down_into_a_ragged_row_keeps_the_column_when_it_exists() {
        let go = |cur, m| motion_target(Some(cur), 20, 6, 4, m);
        assert_eq!(go(6, Motion::Down), Some(12));
        assert_eq!(go(12, Motion::Down), Some(18));
        assert_eq!(go(13, Motion::Down), Some(19));
        assert_eq!(go(14, Motion::Down), Some(19), "the shorter last row ends at tile 19");
        assert_eq!(go(19, Motion::Up), Some(13));
    }

    #[test]
    fn page_moves_go_a_screen_minus_a_row_and_clamp() {
        let go = |cur, m| motion_target(Some(cur), 200, 6, 5, m);
        assert_eq!(go(0, Motion::PageDown), Some(24));
        assert_eq!(go(30, Motion::PageUp), Some(6));
        assert_eq!(go(3, Motion::PageUp), Some(3));
        assert_eq!(go(190, Motion::PageDown), Some(199));
        assert_eq!(motion_target(Some(0), 200, 6, 0, Motion::PageDown), Some(6), "a zero page still moves a row");
        assert_eq!(motion_target(Some(0), 200, 6, 1, Motion::PageDown), Some(6));
    }

    #[test]
    fn with_no_focus_the_first_move_enters_the_grid() {
        for m in [Motion::Right, Motion::Down, Motion::PageDown] {
            assert_eq!(motion_target(None, 10, 4, 3, m), Some(0));
        }
        for m in [Motion::Left, Motion::Up, Motion::PageUp] {
            assert_eq!(motion_target(None, 10, 4, 3, m), Some(9));
        }
        assert_eq!(motion_target(None, 0, 4, 3, Motion::Down), None);
        assert_eq!(motion_target(Some(5), 0, 4, 3, Motion::Down), None);
        assert_eq!(motion_target(Some(99), 10, 4, 3, Motion::Left), Some(8), "a stale focus is clamped");
    }

    #[test]
    fn every_position_is_reachable_by_arrows_across_a_ragged_grid() {
        let (total, cols) = (23, 5);
        let mut seen = vec![false; total];
        let mut cur = 0;
        loop {
            seen[cur] = true;
            let next = motion_target(Some(cur), total, cols, 3, Motion::Right).unwrap();
            if next == cur {
                break;
            }
            cur = next;
        }
        assert!(seen.iter().all(|s| *s));
    }

    #[test]
    fn steps_between_positions_point_the_right_way() {
        assert!(matches!(step_between(3, 9), Some(Step::Down(6))));
        assert!(matches!(step_between(9, 3), Some(Step::Up(6))));
        assert!(step_between(4, 4).is_none());
    }

    #[test]
    fn cover_crops_the_long_side_and_fit_keeps_the_whole_image() {
        let wide = cover_uv(vec2(200.0, 100.0));
        assert!((wide.width() - 0.5).abs() < 1e-6 && (wide.height() - 1.0).abs() < 1e-6 && (wide.center().x - 0.5).abs() < 1e-6);
        let tall = cover_uv(vec2(100.0, 400.0));
        assert!((tall.height() - 0.25).abs() < 1e-6 && (tall.width() - 1.0).abs() < 1e-6);
        assert_eq!(cover_uv(vec2(50.0, 50.0)), Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)));
        assert_eq!(cover_uv(vec2(0.0, 10.0)), Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)));
        let bounds = Rect::from_min_size(pos2(10.0, 10.0), vec2(100.0, 100.0));
        let fit = fit_rect(vec2(200.0, 100.0), bounds);
        assert!((fit.width() - 100.0).abs() < 1e-3 && (fit.height() - 50.0).abs() < 1e-3 && fit.center() == bounds.center());
        assert_eq!(fit_rect(vec2(0.0, 0.0), bounds), bounds);
    }

    #[test]
    fn durations_print_minutes_and_hours() {
        assert_eq!(duration_text(0.0), "0:00");
        assert_eq!(duration_text(59.4), "0:59");
        assert_eq!(duration_text(75.0), "1:15");
        assert_eq!(duration_text(3600.0), "1:00:00");
        assert_eq!(duration_text(3725.0), "1:02:05");
        assert_eq!(duration_text(f32::NAN), "0:00");
        assert_eq!(duration_text(-4.0), "0:00");
    }
}
