//! Headless frames through the real grid view: virtualized drawing, selection by mouse and keyboard, scrolling, menus and the texture pipeline.

use super::actions::MenuAction;
use super::details::DetailsEvent;
use super::grid::{GridEvent, Scroll};
use super::grid_layout::Motion;
use super::view_mode::{TileFit, ViewMode, ViewPrefs};
use crate::Thumbnail;
use crate::ui::file_table::FileExplorer;
use crate::ui::file_table::thumbs::service::{Producer, ServiceConfig, ThumbService};
use crate::ui::file_table::thumbs::store::{Blob, BlobFormat, MemStore};
use crate::ui::file_table::thumbs::{ThumbPresence, Tier};
use crate::ui::test_support::{test_ctx, texts};
use egui::epaint::ClippedShape;
use egui::{CentralPanel, Context, Event, Key, Modifiers, OutputCommand, PointerButton, Pos2, RawInput, Rect, vec2};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

struct NoProducer;

impl Producer for NoProducer {
    fn request(&self, _path: &Path) -> bool {
        true
    }
}

fn jpeg() -> Blob {
    let img: image::RgbImage = image::ImageBuffer::from_fn(256, 171, |x, y| image::Rgb([(x * 7 + y * 3) as u8, (x ^ y) as u8, ((x * y) >> 3) as u8]));
    let mut bytes = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Jpeg).unwrap();
    Blob { format: BlobFormat::Jpeg, w: 256, h: 171, bytes }
}

fn path_of(i: usize) -> String {
    format!("/cells/img_{i:06}.jpg")
}

fn image_row(i: usize) -> Thumbnail {
    let mut t = Thumbnail::new(&format!("img_{i:06}.jpg"));
    t.path = path_of(i);
    t.file_type = "jpg".into();
    t.size = ((i * 37) % 11) as u64 + 1;
    t
}

fn dir_row(path: &str) -> Thumbnail {
    let mut t = Thumbnail::new(path.rsplit('/').next().unwrap());
    t.path = path.into();
    t.file_type = "<DIR>".into();
    t
}

struct Rig {
    ctx: Context,
    ex: FileExplorer,
    svc: Arc<ThumbService>,
    store: Arc<MemStore>,
    shapes: Vec<ClippedShape>,
    copied: Vec<String>,
    clock: f64,
    screen: egui::Vec2,
    strip: bool,
}

fn service(store: &Arc<MemStore>, cfg: ServiceConfig) -> Arc<ThumbService> {
    ThumbService::new(store.clone(), tokio::runtime::Handle::current(), cfg, Arc::new(NoProducer))
}

fn rig_with(rows: Vec<Thumbnail>, with_blobs: bool, cfg: ServiceConfig) -> Rig {
    let store = MemStore::new();
    let svc = service(&store, cfg);
    let mut ex = FileExplorer::new(true);
    ex.viewer.thumbs = svc.clone();
    ex.viewer.thumb_cache = ThumbPresence::new(svc.clone());
    ex.browse.mode = ViewMode::Grid;
    if with_blobs {
        let proto = jpeg();
        for t in rows.iter().filter(|t| t.file_type != "<DIR>") {
            store.insert(&t.path, proto.clone());
        }
    }
    ex.table.bulk_load(rows);
    let ctx = test_ctx();
    egui_extras::install_image_loaders(&ctx);
    Rig { ctx, ex, svc, store, shapes: Vec::new(), copied: Vec::new(), clock: 0.0, screen: vec2(1100.0, 600.0), strip: false }
}

fn rig(rows: usize) -> Rig {
    rig_with((0..rows).map(image_row).collect(), true, ServiceConfig::default())
}

fn button(pos: Pos2, pressed: bool, modifiers: Modifiers, button: PointerButton) -> Event {
    Event::PointerButton { pos, button, pressed, modifiers }
}

fn key_event(key: Key, modifiers: Modifiers) -> Event {
    Event::Key { key, physical_key: Some(key), pressed: true, repeat: false, modifiers }
}

impl Rig {
    fn frame_with(&mut self, events: Vec<Event>, modifiers: Modifiers) {
        let mut all = vec![Event::ModifiersChanged(modifiers)];
        all.extend(events);
        self.clock += 0.016;
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, self.screen)), events: all, time: Some(self.clock), ..Default::default() };
        let (ex, svc, strip) = (&mut self.ex, self.svc.clone(), self.strip);
        let out = self.ctx.run_ui(input, |ui| {
            svc.poll(ui.ctx());
            CentralPanel::default().show(ui, |ui| if strip { ex.browse_panel(ui) } else { ex.grid_panel(ui) });
        });
        self.shapes = out.shapes.clone();
        self.copied = out.platform_output.commands.iter().filter_map(|c| if let OutputCommand::CopyText(t) = c { Some(t.clone()) } else { None }).collect();
        out.drop_without_applying_deltas();
    }

    fn frame(&mut self) {
        self.frame_with(Vec::new(), Modifiers::NONE);
    }

    fn frames(&mut self, n: usize) {
        for _ in 0..n {
            self.frame();
        }
    }

    async fn settle(&mut self, frames: usize) {
        for _ in 0..frames {
            self.frame();
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    fn cols(&self) -> usize {
        self.ex.browse.grid.cols()
    }

    fn tile(&self) -> f32 {
        self.ex.browse.grid.layout().expect("a layout").tile
    }

    fn cell(&self, pos: usize) -> Rect {
        self.ex.browse.grid.cell_rects().iter().find(|(p, _)| *p == pos).unwrap_or_else(|| panic!("tile {pos} is not on screen: visible {:?}", self.ex.browse.grid.visible())).1
    }

    /// A point inside the image area of the tile at view position `pos`, away from the checkbox corner.
    fn tile_point(&self, pos: usize) -> Pos2 {
        let cell = self.cell(pos);
        Pos2::new(cell.center().x + 10.0, cell.min.y + self.tile() * 0.6)
    }

    fn click_at(&mut self, p: Pos2, mods: Modifiers, count: usize) {
        self.clock += 1.0;
        let mut events = vec![Event::PointerMoved(p)];
        for _ in 0..count {
            events.push(button(p, true, mods, PointerButton::Primary));
            events.push(button(p, false, mods, PointerButton::Primary));
        }
        self.frame_with(events, mods);
        self.frame();
    }

    fn click(&mut self, pos: usize) {
        let p = self.tile_point(pos);
        self.click_at(p, Modifiers::NONE, 1);
    }

    fn click_with(&mut self, pos: usize, mods: Modifiers) {
        let p = self.tile_point(pos);
        self.click_at(p, mods, 1);
    }

    fn press_key(&mut self, key: Key, mods: Modifiers) {
        self.frame_with(vec![key_event(key, mods)], mods);
        self.frame();
        self.frame();
    }

    /// Presses at `from`, moves in steps to `to` and releases there.
    fn drag(&mut self, from: Pos2, to: Pos2, mods: Modifiers) {
        self.clock += 1.0;
        self.frame_with(vec![Event::PointerMoved(from), button(from, true, mods, PointerButton::Primary)], mods);
        for step in 1..=4 {
            let at = from + (to - from) * (step as f32 / 4.0);
            self.frame_with(vec![Event::PointerMoved(at)], mods);
        }
        self.frame_with(vec![button(to, false, mods, PointerButton::Primary)], mods);
        self.frame();
    }

    fn selected_paths(&self) -> Vec<String> {
        self.ex.selected_rows().iter().map(|r| r.path.clone()).collect()
    }

    fn focus_path(&self) -> Option<String> {
        self.ex.selection.focus().map(|k| k.as_str().to_string())
    }

    fn focus_pos(&self) -> Option<usize> {
        let view = self.ex.view_ref();
        self.ex.selection.focus().and_then(|k| view.pos_of(k))
    }

    fn text_point(&self, text: &str) -> Option<Pos2> {
        texts(&self.shapes).into_iter().find(|(t, _)| t == text).map(|(_, p)| p + vec2(6.0, 6.0))
    }

    fn shown_texts(&self) -> Vec<String> {
        texts(&self.shapes).into_iter().map(|(t, _)| t).collect()
    }
}

fn paths(range: impl Iterator<Item = usize>) -> Vec<String> {
    range.map(path_of).collect()
}

/* ------------------------------------------- virtualization ------------------------------------------- */

#[tokio::test(flavor = "multi_thread")]
async fn only_visible_and_nearby_tiles_ask_for_a_thumbnail() {
    let mut rig = rig(1000);
    rig.settle(40).await;
    let grid = &rig.ex.browse.grid;
    let (visible, (before, after)) = (grid.visible(), grid.ahead());
    assert!(!visible.is_empty() && visible.len() <= 60, "a 600 px window shows a few rows of tiles, got {}", visible.len());
    assert_eq!(visible.start, 0);
    assert!(before.is_empty(), "nothing above the top");
    assert_eq!(after.start, visible.end, "the screen ahead follows the visible tiles");
    let tracked: Vec<usize> = (0..1000).filter(|i| rig.svc.is_tracked(&path_of(*i), Tier::Thumb)).collect();
    assert_eq!(tracked, (0..after.end).collect::<Vec<_>>(), "exactly the visible tiles and one screen below them are wanted");
    assert!(tracked.len() < 200);
    let snap = rig.svc.snapshot();
    assert_eq!(snap.core.wants_created as usize, tracked.len(), "one want per tile");
    assert_eq!(rig.store.get_batches().iter().sum::<usize>(), tracked.len(), "the store was asked for each wanted tile once");
    assert!(snap.core.ahead_requests > 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn drawn_tiles_end_up_with_textures_and_steady_frames_cost_nothing() {
    let mut rig = rig(60);
    rig.settle(40).await;
    let visible = rig.ex.browse.grid.visible();
    assert!(visible.clone().all(|i| rig.svc.texture_for(&path_of(i), Tier::Thumb).is_ready()));
    let before = rig.svc.snapshot();
    rig.frames(20);
    let after = rig.svc.snapshot();
    assert_eq!(after.core.wants_created, before.core.wants_created);
    assert_eq!(after.core.decode_started, before.core.decode_started);
    assert_eq!(after.cache.claims_granted, before.cache.claims_granted);
    assert_eq!(rig.ex.browse.grid.stats().pending, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_hundred_thousand_rows_cost_no_more_than_a_thousand() {
    let mut small = rig_with((0..1_000).map(image_row).collect(), false, ServiceConfig::default());
    let mut big = rig_with((0..100_000).map(image_row).collect(), false, ServiceConfig::default());
    for rig in [&mut small, &mut big] {
        rig.settle(10).await;
        rig.ex.browse.grid.scroll(Scroll::Offset(5_000.0));
        rig.settle(10).await;
    }
    let (a, b) = (small.ex.browse.grid.stats(), big.ex.browse.grid.stats());
    assert_eq!(a.tiles_drawn, b.tiles_drawn, "the tile count depends on the window, not the row count");
    assert!(b.tiles_drawn <= 60);
    assert!(big.svc.snapshot().core.wants_created < 400, "thumbnail requests stay bounded by the screen");
    assert!(big.ex.browse.grid.content_height() > 90.0 * small.ex.browse.grid.content_height());
}

#[tokio::test(flavor = "multi_thread")]
async fn scrolling_far_cancels_the_requests_of_tiles_that_left_the_screen() {
    let mut rig = rig_with((0..5_000).map(image_row).collect(), false, ServiceConfig::default());
    rig.settle(10).await;
    let first_screen: Vec<String> = paths(rig.ex.browse.grid.visible());
    assert!(first_screen.iter().all(|p| rig.svc.is_tracked(p, Tier::Thumb)));
    rig.ex.browse.grid.scroll(Scroll::Offset(60_000.0));
    rig.settle(12).await;
    assert!(first_screen.iter().all(|p| !rig.svc.is_tracked(p, Tier::Thumb)), "tiles that left the screen are no longer wanted");
    assert!(rig.svc.snapshot().core.cancelled_stale > 0);
    let now: Vec<String> = paths(rig.ex.browse.grid.visible());
    assert!(now.iter().all(|p| rig.svc.is_tracked(p, Tier::Thumb)));
}

#[tokio::test(flavor = "multi_thread")]
async fn tiles_nearest_the_centre_are_requested_before_the_ones_ahead() {
    let mut rig = rig(2_000);
    rig.settle(6).await;
    let grid = &rig.ex.browse.grid;
    let layout = grid.layout().unwrap();
    let viewport = grid.viewport_height();
    let center = grid.offset() + viewport / 2.0;
    let dist = |pos: usize| (layout.cell_rect(pos).center().y - center).abs();
    let visible = grid.visible();
    let (_, after) = grid.ahead();
    let farthest_visible = visible.clone().map(dist).fold(0.0f32, f32::max);
    let nearest_ahead = after.clone().map(dist).fold(f32::MAX, f32::min);
    assert!(farthest_visible <= nearest_ahead, "visible {farthest_visible} vs ahead {nearest_ahead}");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_texture_budget_holds_while_scrolling_through_a_library() {
    let mut cfg = ServiceConfig::default();
    cfg.core.budget_bytes = 12 * 1024 * 1024;
    let mut rig = rig_with((0..3_000).map(image_row).collect(), true, cfg);
    rig.settle(20).await;
    for step in 1..=12 {
        rig.ex.browse.grid.scroll(Scroll::Offset(step as f32 * 2_500.0));
        rig.settle(12).await;
        let snap = rig.svc.snapshot();
        assert!(snap.used <= snap.budget, "{} > {}", snap.used, snap.budget);
    }
    let snap = rig.svc.snapshot();
    assert!(snap.cache.evictions > 0, "scrolling past the budget evicts");
    let visible = rig.ex.browse.grid.visible();
    assert!(visible.clone().all(|i| rig.svc.texture_for(&path_of(i), Tier::Thumb).is_ready()), "the tiles on screen keep their textures");
}

#[tokio::test(flavor = "multi_thread")]
async fn pending_tiles_show_a_placeholder_and_ask_for_repaints() {
    let mut rig = rig_with((0..40).map(image_row).collect(), true, ServiceConfig::default());
    rig.frame();
    let stats = rig.ex.browse.grid.stats();
    assert_eq!(stats.pending, stats.tiles_drawn, "nothing is loaded in the first frame");
    assert!(stats.tiles_drawn > 0);
    rig.settle(40).await;
    assert_eq!(rig.ex.browse.grid.stats().pending, 0);
}

/* --------------------------------------------- layout ---------------------------------------------- */

#[tokio::test(flavor = "multi_thread")]
async fn columns_follow_the_window_width_and_the_tile_size() {
    let mut rig = rig(300);
    rig.settle(3).await;
    let wide = rig.cols();
    assert!(wide >= 5, "{wide}");
    rig.screen = vec2(500.0, 600.0);
    rig.frames(3);
    assert!(rig.cols() < wide);
    rig.screen = vec2(1100.0, 600.0);
    rig.ex.set_tile_size(64.0);
    rig.frames(3);
    assert!(rig.cols() > wide, "smaller tiles fit more columns");
    rig.ex.set_tile_size(320.0);
    rig.frames(3);
    assert!(rig.cols() < wide);
    assert!(rig.tile() >= 320.0, "tiles are at least the target size");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_last_row_may_be_short_and_every_tile_is_still_drawn() {
    let mut rig = rig(23);
    rig.screen = vec2(1100.0, 1800.0);
    rig.settle(3).await;
    let cols = rig.cols();
    assert_ne!(23 % cols, 0, "this layout has a ragged last row");
    assert_eq!(rig.ex.browse.grid.stats().tiles_drawn, 23);
    assert_eq!(rig.cell(22).min.y, rig.cell(22 - 22 % cols).min.y);
}

/* ------------------------------------------- mouse selection --------------------------------------- */

#[tokio::test(flavor = "multi_thread")]
async fn click_ctrl_click_and_shift_click_select_in_view_order() {
    let mut rig = rig(60);
    rig.settle(3).await;
    let cols = rig.cols();
    rig.click(1);
    assert_eq!(rig.selected_paths(), paths(1..2));
    rig.click_with(3, Modifiers::COMMAND);
    assert_eq!(rig.selected_paths(), vec![path_of(1), path_of(3)]);
    rig.click_with(1, Modifiers::COMMAND);
    assert_eq!(rig.selected_paths(), paths(3..4), "ctrl-click toggles");
    rig.click(2);
    rig.click_with(cols + 1, Modifiers::SHIFT);
    assert_eq!(rig.selected_paths(), paths(2..cols + 2), "shift-click selects the range across rows in view order");
    assert_eq!(rig.focus_pos(), Some(cols + 1));
}

#[tokio::test(flavor = "multi_thread")]
async fn clicking_empty_space_between_tiles_or_below_them_clears_the_selection() {
    let mut rig = rig(10);
    rig.settle(3).await;
    rig.click(2);
    assert_eq!(rig.selected_paths().len(), 1);
    let below = Pos2::new(40.0, rig.cell(0).max.y + 200.0);
    rig.click_at(below, Modifiers::NONE, 1);
    assert!(rig.selected_paths().is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn the_hover_checkbox_toggles_a_tile_without_a_modifier() {
    let mut rig = rig(30);
    rig.settle(3).await;
    let corner = rig.cell(2).min + vec2(14.0, 14.0);
    rig.frame_with(vec![Event::PointerMoved(corner)], Modifiers::NONE);
    rig.frame();
    rig.click_at(corner, Modifiers::NONE, 1);
    assert_eq!(rig.selected_paths(), paths(2..3));
    let other = rig.cell(4).min + vec2(14.0, 14.0);
    rig.frame_with(vec![Event::PointerMoved(other)], Modifiers::NONE);
    rig.click_at(other, Modifiers::NONE, 1);
    assert_eq!(rig.selected_paths(), vec![path_of(2), path_of(4)], "checkboxes add to the selection");
    let corner = rig.cell(2).min + vec2(14.0, 14.0);
    rig.click_at(corner, Modifiers::NONE, 1);
    assert_eq!(rig.selected_paths(), paths(4..5), "a second click deselects");
}

#[tokio::test(flavor = "multi_thread")]
async fn dragging_over_empty_space_selects_the_tiles_in_the_band() {
    let mut rig = rig(60);
    rig.settle(3).await;
    let cols = rig.cols();
    let from = rig.cell(0).min + vec2(rig.tile() + 3.0, rig.tile() * 0.5);
    let to = rig.cell(cols + 2).center();
    rig.drag(from, to, Modifiers::NONE);
    let selected = rig.selected_paths();
    assert!(!selected.is_empty());
    let expected: Vec<String> = [1, 2, cols + 1, cols + 2].iter().map(|p| path_of(*p)).collect();
    assert_eq!(selected, expected, "the band covers columns 1-2 of the first two rows");
    assert!(!rig.ex.browse.grid.band_active(), "releasing ends the band");
}

#[tokio::test(flavor = "multi_thread")]
async fn dragging_from_a_tile_also_starts_a_band_unless_it_is_selected() {
    let mut rig = rig(60);
    rig.settle(3).await;
    let cols = rig.cols();
    rig.click(0);
    let to = rig.cell(cols + 1).center();
    let from = rig.tile_point(0);
    rig.drag(from, to, Modifiers::NONE);
    assert_eq!(rig.selected_paths(), paths(0..1), "a drag from a selected tile starts no band");
    rig.click(7);
    rig.click_at(Pos2::new(5.0, 590.0), Modifiers::NONE, 1);
    let from = rig.tile_point(2);
    let to = rig.cell(cols + 3).center();
    rig.drag(from, to, Modifiers::NONE);
    assert_eq!(rig.selected_paths(), vec![path_of(2), path_of(3), path_of(cols + 2), path_of(cols + 3)]);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_ctrl_drag_adds_the_band_to_the_selection() {
    let mut rig = rig(60);
    rig.settle(3).await;
    let cols = rig.cols();
    rig.click(0);
    let from = rig.tile_point(cols + 2);
    let to = rig.cell(cols + 3).center();
    rig.drag(from, to, Modifiers::COMMAND);
    let selected = rig.selected_paths();
    assert!(selected.contains(&path_of(0)), "the earlier selection stays");
    assert!(selected.contains(&path_of(cols + 3)));
}

/* ------------------------------------------- keyboard ------------------------------------------------ */

#[tokio::test(flavor = "multi_thread")]
async fn arrow_keys_move_by_tile_and_row_over_a_ragged_last_row() {
    let mut rig = rig(23);
    rig.screen = vec2(1100.0, 1800.0);
    rig.settle(3).await;
    let cols = rig.cols();
    rig.click(0);
    rig.press_key(Key::ArrowRight, Modifiers::NONE);
    assert_eq!(rig.focus_pos(), Some(1));
    rig.press_key(Key::ArrowDown, Modifiers::NONE);
    assert_eq!(rig.focus_pos(), Some(1 + cols));
    rig.press_key(Key::ArrowLeft, Modifiers::NONE);
    rig.press_key(Key::ArrowUp, Modifiers::NONE);
    assert_eq!(rig.focus_pos(), Some(0));
    rig.press_key(Key::ArrowUp, Modifiers::NONE);
    assert_eq!(rig.focus_pos(), Some(0), "the first row has nothing above");
    rig.press_key(Key::End, Modifiers::NONE);
    assert_eq!(rig.focus_pos(), Some(22));
    rig.press_key(Key::ArrowDown, Modifiers::NONE);
    assert_eq!(rig.focus_pos(), Some(22));
    rig.press_key(Key::ArrowRight, Modifiers::NONE);
    assert_eq!(rig.focus_pos(), Some(22));
    assert_eq!(rig.selected_paths(), paths(22..23), "moving selects the new tile");
    rig.press_key(Key::Home, Modifiers::NONE);
    assert_eq!(rig.focus_pos(), Some(0));
}

#[tokio::test(flavor = "multi_thread")]
async fn down_from_a_column_the_last_row_lacks_lands_on_its_last_tile() {
    let mut rig = rig(23);
    rig.screen = vec2(1100.0, 1800.0);
    rig.settle(3).await;
    let cols = rig.cols();
    let last_row_start = 22 - 22 % cols;
    let col = 22 % cols + 1;
    assert!(col < cols, "this layout has a column the last row lacks");
    rig.click(last_row_start - cols + col);
    rig.press_key(Key::ArrowDown, Modifiers::NONE);
    assert_eq!(rig.focus_pos(), Some(22));
}

#[tokio::test(flavor = "multi_thread")]
async fn shift_arrows_extend_the_selection_in_view_order() {
    let mut rig = rig(60);
    rig.settle(3).await;
    let cols = rig.cols();
    rig.click(1);
    rig.press_key(Key::ArrowRight, Modifiers::SHIFT);
    rig.press_key(Key::ArrowRight, Modifiers::SHIFT);
    assert_eq!(rig.selected_paths(), paths(1..4));
    rig.press_key(Key::ArrowDown, Modifiers::SHIFT);
    assert_eq!(rig.selected_paths(), paths(1..4 + cols));
}

#[tokio::test(flavor = "multi_thread")]
async fn the_first_arrow_with_no_focus_enters_the_grid() {
    let mut rig = rig(30);
    rig.settle(3).await;
    rig.frame_with(vec![Event::PointerMoved(Pos2::new(300.0, 300.0))], Modifiers::NONE);
    rig.press_key(Key::ArrowRight, Modifiers::NONE);
    assert_eq!(rig.focus_pos(), Some(0));
    assert_eq!(rig.selected_paths(), paths(0..1));
}

#[tokio::test(flavor = "multi_thread")]
async fn page_keys_move_a_screen_and_scroll_to_follow_the_focus() {
    let mut rig = rig(2_000);
    rig.settle(3).await;
    let cols = rig.cols();
    let page = rig.ex.browse.grid.page_rows();
    rig.click(0);
    rig.press_key(Key::PageDown, Modifiers::NONE);
    let focus = rig.focus_pos().unwrap();
    assert_eq!(focus, (page - 1) * cols);
    rig.press_key(Key::PageDown, Modifiers::NONE);
    rig.press_key(Key::PageDown, Modifiers::NONE);
    let focus = rig.focus_pos().unwrap();
    assert!(rig.ex.browse.grid.visible().contains(&focus), "the focused tile is scrolled into view");
    rig.press_key(Key::End, Modifiers::NONE);
    assert_eq!(rig.focus_pos(), Some(1_999));
    assert!(rig.ex.browse.grid.visible().contains(&1_999));
    rig.press_key(Key::Home, Modifiers::NONE);
    assert!(rig.ex.browse.grid.visible().contains(&0));
}

#[tokio::test(flavor = "multi_thread")]
async fn ctrl_a_selects_everything_and_escape_clears() {
    let mut rig = rig(40);
    rig.settle(3).await;
    rig.click(0);
    rig.press_key(Key::A, Modifiers::COMMAND);
    assert_eq!(rig.selected_paths().len(), 40);
    rig.press_key(Key::Escape, Modifiers::NONE);
    assert!(rig.selected_paths().is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn enter_and_double_click_open_the_row_and_space_toggles_the_preview_flag() {
    let mut rig = rig(20);
    rig.settle(3).await;
    rig.click(3);
    rig.press_key(Key::Enter, Modifiers::NONE);
    assert_eq!(rig.ex.row_sync.opened.clone(), vec![path_of(3)]);
    let p = rig.tile_point(5);
    rig.click_at(p, Modifiers::NONE, 2);
    assert_eq!(rig.ex.row_sync.opened.clone(), vec![path_of(3), path_of(5)]);
    let was = rig.ex.open_preview_pane;
    rig.press_key(Key::Space, Modifiers::NONE);
    assert_eq!(rig.ex.open_preview_pane, !was);
}

#[tokio::test(flavor = "multi_thread")]
async fn plus_and_minus_resize_the_tiles_and_remember_the_size() {
    let mut rig = rig(60);
    rig.settle(3).await;
    rig.click(0);
    rig.press_key(Key::Plus, Modifiers::NONE);
    assert_eq!(rig.ex.browse.tile, 176.0);
    rig.press_key(Key::Minus, Modifiers::NONE);
    rig.press_key(Key::Minus, Modifiers::NONE);
    assert_eq!(rig.ex.browse.tile, 144.0);
    assert_eq!(super::view_mode::app_prefs().tile, 144.0, "new tabs start with the last size");
}

#[tokio::test(flavor = "multi_thread")]
async fn ctrl_scroll_zooms_the_tiles() {
    let mut rig = rig(60);
    rig.settle(3).await;
    let before = rig.ex.browse.tile;
    rig.frame_with(vec![Event::PointerMoved(Pos2::new(300.0, 300.0))], Modifiers::NONE);
    rig.frame_with(vec![Event::MouseWheel { unit: egui::MouseWheelUnit::Line, delta: vec2(0.0, 2.0), phase: egui::TouchPhase::Move, source: egui::MouseWheelSource::Wheel, modifiers: Modifiers::COMMAND }], Modifiers::COMMAND);
    rig.frames(2);
    assert!(rig.ex.browse.tile > before, "{} -> {}", before, rig.ex.browse.tile);
}

/* --------------------------------------------- menu ------------------------------------------------- */

#[tokio::test(flavor = "multi_thread")]
async fn right_click_selects_an_unselected_tile_and_opens_the_same_menu_as_the_table() {
    let mut rig = rig(30);
    rig.settle(3).await;
    rig.click(1);
    let p = rig.tile_point(4);
    rig.clock += 1.0;
    rig.frame_with(vec![Event::PointerMoved(p), button(p, true, Modifiers::NONE, PointerButton::Secondary), button(p, false, Modifiers::NONE, PointerButton::Secondary)], Modifiers::NONE);
    rig.frames(2);
    assert_eq!(rig.selected_paths(), paths(4..5), "a right click on an unselected tile selects it");
    let shown = rig.shown_texts();
    for action in MenuAction::ALL.iter().filter(|a| **a != MenuAction::AttachToChatWindow) {
        assert!(shown.iter().any(|t| t == action.label()), "{} is missing from {shown:?}", action.label());
    }
    assert!(shown.iter().any(|t| t == "1 item"));
}

#[tokio::test(flavor = "multi_thread")]
async fn right_click_inside_the_selection_keeps_it() {
    let mut rig = rig(30);
    rig.settle(3).await;
    rig.click(1);
    rig.click_with(2, Modifiers::COMMAND);
    let p = rig.tile_point(2);
    rig.clock += 1.0;
    rig.frame_with(vec![Event::PointerMoved(p), button(p, true, Modifiers::NONE, PointerButton::Secondary), button(p, false, Modifiers::NONE, PointerButton::Secondary)], Modifiers::NONE);
    rig.frames(2);
    assert_eq!(rig.selected_paths(), paths(1..3));
    assert!(rig.shown_texts().iter().any(|t| t == "2 items"));
}

#[tokio::test(flavor = "multi_thread")]
async fn the_menu_acts_on_the_selection_and_does_nothing_without_one() {
    let mut rig = rig(30);
    rig.settle(3).await;
    let _serial = crate::command::bus::testing::serial();
    let is_tab = |c: &crate::command::AppCommand| matches!(c, crate::command::AppCommand::OpenRows(_) | crate::command::AppCommand::OpenScope { .. });
    crate::command::bus::testing::take_matching(is_tab);
    rig.ex.apply_grid_events(&rig.ctx.clone(), vec![GridEvent::Details(DetailsEvent::Menu(MenuAction::OpenSelectionInNewTab))]);
    assert!(crate::command::bus::testing::take_matching(is_tab).is_empty(), "nothing is selected");
    rig.click(1);
    rig.click_with(2, Modifiers::COMMAND);
    rig.ex.apply_grid_events(&rig.ctx.clone(), vec![GridEvent::Details(DetailsEvent::Menu(MenuAction::OpenSelectionInNewTab))]);
    let sent = crate::command::bus::testing::take_matching(is_tab);
    assert_eq!(sent.len(), 1);
    match &sent[0] {
        crate::command::AppCommand::OpenRows(tab) => assert_eq!(tab.rows.len(), 2),
        other => panic!("{other:?}"),
    }
}

/* --------------------------------------------- views ------------------------------------------------ */

#[tokio::test(flavor = "multi_thread")]
async fn a_new_explorer_opens_in_the_grid_and_remembers_the_last_choice() {
    super::view_mode::update_app_prefs(|p| *p = ViewPrefs::default());
    assert_eq!(FileExplorer::new(true).browse.mode, ViewMode::Grid, "the grid is the default");
    let mut ex = FileExplorer::new(true);
    ex.set_view_mode(ViewMode::Details);
    ex.set_tile_size(200.0);
    ex.set_tile_fit(TileFit::Fit);
    let next = FileExplorer::new(true);
    assert_eq!((next.browse.mode, next.browse.tile, next.browse.fit), (ViewMode::Details, 200.0, TileFit::Fit));
    super::view_mode::update_app_prefs(|p| *p = ViewPrefs::default());
}

#[tokio::test(flavor = "multi_thread")]
async fn switching_views_keeps_the_top_row_in_sight() {
    let mut rig = rig(3_000);
    rig.settle(3).await;
    rig.ex.browse.grid.scroll(Scroll::Offset(40_000.0));
    rig.settle(4).await;
    let top = rig.ex.browse.grid.top_pos();
    assert!(top > 500);
    rig.ex.set_view_mode(ViewMode::Details);
    for _ in 0..4 {
        rig.frame_with(Vec::new(), Modifiers::NONE);
    }
    assert_eq!(rig.ex.browse.mode, ViewMode::Details);
    rig.strip = true;
    rig.frames(4);
    let first = rig.ex.details.row_rects().iter().map(|(p, _)| *p).min().expect("details rows are drawn");
    assert!((first as i64 - top as i64).abs() <= 1, "details starts at {first}, the grid was at {top}");
    rig.ex.set_view_mode(ViewMode::Grid);
    rig.frames(4);
    let back = rig.ex.browse.grid.top_pos();
    assert!(back.abs_diff(first) < rig.cols() + 1, "back in the grid at {back}, details was at {first}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_filter_keeps_the_tile_at_the_top_where_it_was() {
    let mut rig = rig(2_000);
    rig.settle(3).await;
    rig.ex.browse.grid.scroll(Scroll::Offset(30_000.0));
    rig.settle(4).await;
    let anchor_pos = rig.ex.browse.grid.top_pos();
    let anchor_path = path_of(anchor_pos);
    rig.ex.viewer.filter = "img_0".into();
    rig.settle(6).await;
    let view = rig.ex.view_ref();
    let moved = view.row_at(rig.ex.browse.grid.top_pos()).map(|r| r.path.clone());
    assert!(moved.is_some());
    let key = rig.ex.table.key_for(&anchor_path);
    let new_pos = view.pos_of(&key);
    assert!(new_pos.is_some(), "the anchor tile still passes the filter");
    assert!(rig.ex.browse.grid.top_pos().abs_diff(new_pos.unwrap()) <= rig.cols(), "the grid stays on the same tile after the filter");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_tile_size_change_keeps_the_top_row() {
    let mut rig = rig(3_000);
    rig.settle(3).await;
    rig.ex.browse.grid.scroll(Scroll::Offset(20_000.0));
    rig.settle(4).await;
    let top = rig.ex.browse.grid.top_pos();
    rig.ex.set_tile_size(96.0);
    rig.settle(4).await;
    let after = rig.ex.browse.grid.top_pos();
    assert!(after.abs_diff(top) <= rig.cols(), "was at {top}, now at {after}");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_empty_view_says_so_and_a_filtered_one_says_how_many_are_hidden() {
    let mut empty = rig_with(Vec::new(), false, ServiceConfig::default());
    empty.settle(2).await;
    assert!(empty.shown_texts().iter().any(|t| t == "No items"));
    let mut hidden = rig(20);
    hidden.ex.viewer.filter = "no such file".into();
    hidden.settle(3).await;
    assert!(hidden.shown_texts().iter().any(|t| t == "20 items hidden by the current filter"));
}

#[tokio::test(flavor = "multi_thread")]
async fn folders_and_other_files_draw_without_asking_for_thumbnails() {
    let mut rows: Vec<Thumbnail> = vec![dir_row("/cells/sub"), dir_row("/cells/other")];
    let mut text = Thumbnail::new("notes.txt");
    text.path = "/cells/notes.txt".into();
    text.file_type = "txt".into();
    rows.push(text);
    rows.extend((0..5).map(image_row));
    let mut rig = rig_with(rows, true, ServiceConfig::default());
    rig.settle(20).await;
    assert_eq!(rig.ex.browse.grid.stats().tiles_drawn, 8);
    assert!(!rig.svc.is_tracked("/cells/sub", Tier::Thumb));
    assert!(!rig.svc.is_tracked("/cells/notes.txt", Tier::Thumb));
    assert_eq!(rig.ex.browse.grid.stats().thumbs_requested, 5);
    let shown = rig.shown_texts();
    assert!(shown.iter().any(|t| t == "sub") && shown.iter().any(|t| t == "notes.txt"), "captions are drawn: {shown:?}");
    rig.click(0);
    rig.press_key(Key::Enter, Modifiers::NONE);
    assert_eq!(rig.ex.row_sync.opened.clone(), Vec::<String>::new(), "a folder is not opened externally");
}

#[tokio::test(flavor = "multi_thread")]
async fn captions_can_be_turned_off() {
    let mut rig = rig(12);
    rig.settle(3).await;
    assert!(rig.shown_texts().iter().any(|t| t == "img_000001.jpg"));
    rig.ex.set_captions(false);
    rig.frames(3);
    assert!(!rig.shown_texts().iter().any(|t| t == "img_000001.jpg"));
}

#[tokio::test(flavor = "multi_thread")]
async fn grid_moves_and_clicks_agree_with_the_details_selection_rules() {
    let mut rig = rig(40);
    rig.settle(3).await;
    rig.click(2);
    rig.ex.apply_grid_events(&rig.ctx.clone(), vec![GridEvent::Move { motion: Motion::Right, extend: true }, GridEvent::Move { motion: Motion::Right, extend: true }]);
    assert_eq!(rig.selected_paths(), paths(2..5));
    rig.ex.apply_grid_events(&rig.ctx.clone(), vec![GridEvent::Details(DetailsEvent::Key(crate::ui::file_table::table::keys::KeyAction::SelectAll))]);
    assert_eq!(rig.selected_paths().len(), 40);
    assert_eq!(rig.ex.check_row_invariants(), Ok(()));
}

#[tokio::test(flavor = "multi_thread")]
async fn the_stress_scroller_moves_the_view_and_stops() {
    let mut rig = rig(5_000);
    rig.settle(3).await;
    rig.ex.browse.grid.set_stress(Some(200.0));
    let mut offsets = Vec::new();
    for _ in 0..30 {
        rig.settle(1).await;
        offsets.push(rig.ex.browse.grid.offset());
    }
    assert!(offsets.windows(2).any(|w| w[1] > w[0] + 1.0), "the view scrolls down");
    rig.ex.browse.grid.set_stress(None);
    rig.settle(2).await;
    let rest = rig.ex.browse.grid.offset();
    rig.settle(5).await;
    assert_eq!(rig.ex.browse.grid.offset(), rest, "stopped");
}

#[tokio::test(flavor = "multi_thread")]
async fn grid_frames_are_recorded_for_the_devtools() {
    let mut rig = rig(200);
    rig.settle(8).await;
    assert!(rig.ex.browse.grid.frame_times().len() >= 8);
    rig.ex.browse.grid.clear_frame_times();
    assert!(rig.ex.browse.grid.frame_times().is_empty());
}

/* --------------------------------------------- strip ------------------------------------------------ */

#[tokio::test(flavor = "multi_thread")]
async fn the_strip_shows_the_controls_and_the_count_and_collapses_when_narrow() {
    let mut rig = rig(1_234);
    rig.strip = true;
    rig.settle(4).await;
    let wide = rig.shown_texts();
    for label in ["Grid", "Details", "S", "M", "L", "Sort", "1,234 items"] {
        assert!(wide.iter().any(|t| t == label), "{label} is missing from {wide:?}");
    }
    rig.screen = vec2(700.0, 600.0);
    rig.frames(4);
    let narrow = rig.shown_texts();
    assert!(!narrow.iter().any(|t| t == "Details" || t == "Sort"), "controls fold into the menu: {narrow:?}");
    assert!(narrow.iter().any(|t| t == "1,234"), "the count stays: {narrow:?}");
    assert_eq!(rig.ex.browse.mode, ViewMode::Grid);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_strip_hides_the_grid_only_controls_in_details() {
    let mut rig = rig(50);
    rig.strip = true;
    rig.settle(4).await;
    assert!(rig.shown_texts().iter().any(|t| t == "L"), "the size presets show in the grid");
    rig.ex.set_view_mode(ViewMode::Details);
    rig.frames(4);
    let shown = rig.shown_texts();
    assert!(!shown.iter().any(|t| t == "S" || t == "M" || t == "L"), "{shown:?}");
    assert!(shown.iter().any(|t| t == "Sort") && shown.iter().any(|t| t == "50 items"));
}

#[tokio::test(flavor = "multi_thread")]
async fn the_mode_switch_toggles_the_view_and_f9_flips_it() {
    let mut rig = rig(50);
    rig.strip = true;
    rig.settle(4).await;
    let details = rig.text_point("Details").expect("a Details button");
    rig.click_at(details, Modifiers::NONE, 1);
    assert_eq!(rig.ex.browse.mode, ViewMode::Details);
    rig.frame_with(vec![Event::PointerMoved(Pos2::new(400.0, 300.0))], Modifiers::NONE);
    rig.press_key(Key::F9, Modifiers::NONE);
    assert_eq!(rig.ex.browse.mode, ViewMode::Grid);
    rig.press_key(Key::Num2, Modifiers::COMMAND);
    assert_eq!(rig.ex.browse.mode, ViewMode::Details);
    rig.press_key(Key::Num1, Modifiers::COMMAND);
    assert_eq!(rig.ex.browse.mode, ViewMode::Grid);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_size_presets_set_the_tile_size() {
    let mut rig = rig(50);
    rig.strip = true;
    rig.settle(4).await;
    let large = rig.text_point("L").expect("an L preset");
    rig.click_at(large, Modifiers::NONE, 1);
    assert_eq!(rig.ex.browse.tile, 256.0);
    let small = rig.text_point("S").expect("an S preset");
    rig.click_at(small, Modifiers::NONE, 1);
    assert_eq!(rig.ex.browse.tile, 96.0);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_sort_selector_shares_the_details_sort() {
    let mut rig = rig(30);
    rig.strip = true;
    rig.settle(4).await;
    assert!(rig.ex.view.sort().is_empty() || rig.ex.view.sort().keys().len() == 1);
    rig.ex.view.set_sort(crate::ui::file_table::rows::SortSpec::single(super::columns::Col::Size.id(), true));
    rig.settle(3).await;
    assert!(rig.shown_texts().iter().any(|t| t == "Size"), "the combo names the sort column");
    let first = rig.ex.view.rows(&rig.ex.table).next().map(|r| r.size).unwrap();
    assert!(rig.ex.view.rows(&rig.ex.table).all(|r| r.size <= first), "descending by size");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_view_command_sets_mode_size_and_fit() {
    use crate::command::{AppCommand, dispatch};
    let mut sink = crate::command::dispatch::fake::FakeSink::default();
    let json = r#"{"cmd":"set_view_mode","mode":"grid","size":200,"fit":"fit","captions":false,"stress":25}"#;
    let cmd: AppCommand = serde_json::from_str(json).unwrap();
    dispatch(&mut sink, cmd).unwrap();
    assert_eq!(sink.calls, vec!["view Grid Some(200.0) Some(Fit) Some(false) Some(25.0)"]);
}

/* --------------------------------------------- bench ------------------------------------------------ */

fn rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/statm").ok().and_then(|s| s.split_whitespace().nth(1).and_then(|p| p.parse::<f64>().ok())).map(|pages| pages * 4.0 / 1024.0).unwrap_or(0.0)
}

// Runs `n` frames, scrolling by a wheel step each when `scroll` is set; returns frame milliseconds and UI-thread allocations after the first 30.
async fn bench_frames(rig: &mut Rig, scroll: bool, n: usize) -> (Vec<f64>, Vec<f64>) {
    use crate::ui::file_table::thumbs::perf::measure;
    use egui::{MouseWheelSource, MouseWheelUnit, TouchPhase, pos2};
    let (mut times, mut allocs) = (Vec::new(), Vec::new());
    for frame in 0..n {
        let mut events = vec![Event::PointerMoved(pos2(500.0, 300.0))];
        if scroll {
            events.push(Event::MouseWheel { unit: MouseWheelUnit::Point, delta: vec2(0.0, -600.0), phase: TouchPhase::Move, source: MouseWheelSource::Wheel, modifiers: Modifiers::NONE });
        }
        rig.clock += 0.016;
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, rig.screen)), events, time: Some(rig.clock), ..Default::default() };
        let (ex, svc, ctx) = (&mut rig.ex, rig.svc.clone(), &rig.ctx);
        let started = std::time::Instant::now();
        let (_, counts) = measure(|| {
            let mut out = ctx.run_ui(input, |ui| {
                svc.poll(ui.ctx());
                CentralPanel::default().show(ui, |ui| ex.grid_panel(ui));
            });
            out.textures_delta.clear();
            let _ = ctx.tessellate(out.shapes, out.pixels_per_point);
        });
        if frame >= 30 {
            times.push(started.elapsed().as_secs_f64() * 1000.0);
            allocs.push(counts.allocs as f64);
        }
        tokio::time::sleep(Duration::from_millis(16)).await;
    }
    (times, allocs)
}

/// Prints frame-time figures for scrolling the grid quickly; run with `--ignored --nocapture`, rows from `AME_BENCH_ROWS`, frames from `AME_BENCH_FRAMES`, stored thumbnails from `AME_BENCH_BLOBS`, tile size from `AME_BENCH_TILE`.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_grid_scrolling() {
    let env = |name: &str, default: usize| std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default);
    let rows = env("AME_BENCH_ROWS", 20_000);
    let frames = env("AME_BENCH_FRAMES", 400);
    let blobs = env("AME_BENCH_BLOBS", 12_000).min(rows);
    let tile = env("AME_BENCH_TILE", 160) as f32;
    let mut rig = rig_with((0..rows).map(image_row).collect(), false, ServiceConfig::default());
    let proto = jpeg();
    for i in 0..blobs {
        rig.store.insert(&path_of(i), proto.clone());
    }
    rig.ex.set_tile_size(tile);
    let rss0 = rss_mb();
    let pct = |v: &mut Vec<f64>, p: f64| {
        v.sort_by(f64::total_cmp);
        v[((v.len() - 1) as f64 * p) as usize]
    };
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    for (label, scroll, n) in [("scroll", true, frames + 30), ("idle", false, 130)] {
        let (mut times, mut allocs) = bench_frames(&mut rig, scroll, n).await;
        println!(
            "BENCH grid rows={rows} tile={tile} {label} frames={}: ms p50 {:.2} p95 {:.2} max {:.2} | ui-thread allocs/frame mean {:.0} p95 {:.0} | tiles drawn {} | rss {rss0:.0} -> {:.0} MB",
            times.len(),
            pct(&mut times.clone(), 0.5),
            pct(&mut times.clone(), 0.95),
            pct(&mut times, 1.0),
            mean(&allocs),
            pct(&mut allocs, 0.95),
            rig.ex.browse.grid.stats().tiles_drawn,
            rss_mb(),
        );
    }
    let snap = rig.svc.snapshot();
    println!(
        "BENCH grid cache: entries {} used {} of {} bytes, hits {}, claims {}, evictions {}, decode ok {}, uploads {}, cancelled stale {}, top row {}",
        snap.entries,
        snap.used,
        snap.budget,
        snap.cache.hits,
        snap.cache.claims_granted,
        snap.cache.evictions,
        snap.core.decode_ok,
        snap.core.uploads_new + snap.core.uploads_in_place,
        snap.core.cancelled_stale,
        rig.ex.browse.grid.top_pos(),
    );
}
