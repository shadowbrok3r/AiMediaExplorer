//! Headless frames through the real details view: virtualized drawing, selection by mouse and keyboard, sorting, menus and the model's invariants.

use super::TabAction;
use super::actions::MenuAction;
use super::columns::Col;
use super::details::DetailsEvent;
use crate::Thumbnail;
use crate::ui::devtools::row_store;
use crate::ui::file_table::FileExplorer;
use crate::ui::file_table::rows::Refresh;
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
    format!("/cells/img_{i:04}.jpg")
}

fn image_row(i: usize) -> Thumbnail {
    let mut t = Thumbnail::new(&format!("img_{i:04}.jpg"));
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
}

fn rig(rows: usize) -> Rig {
    let store = MemStore::new();
    let svc = ThumbService::new(store.clone(), tokio::runtime::Handle::current(), ServiceConfig::default(), Arc::new(NoProducer));
    let mut ex = FileExplorer::new(true);
    ex.viewer.thumbs = svc.clone();
    ex.viewer.thumb_cache = ThumbPresence::new(svc.clone());
    let proto = jpeg();
    let batch: Vec<Thumbnail> = (0..rows).map(image_row).collect();
    for t in &batch {
        store.insert(&t.path, proto.clone());
    }
    ex.table.bulk_load(batch);
    let ctx = test_ctx();
    egui_extras::install_image_loaders(&ctx);
    Rig { ctx, ex, svc, store, shapes: Vec::new(), copied: Vec::new(), clock: 0.0 }
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
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1100.0, 600.0))), events: all, time: Some(self.clock), ..Default::default() };
        let (ex, svc) = (&mut self.ex, self.svc.clone());
        let out = self.ctx.run_ui(input, |ui| {
            svc.poll(ui.ctx());
            CentralPanel::default().show(ui, |ui| ex.details_panel(ui));
        });
        self.shapes = out.shapes.clone();
        self.copied = out.platform_output.commands.iter().filter_map(|c| if let OutputCommand::CopyText(t) = c { Some(t.clone()) } else { None }).collect();
        out.drop_without_applying_deltas();
    }

    fn frame(&mut self) {
        self.frame_with(Vec::new(), Modifiers::NONE);
    }

    async fn settle(&mut self, frames: usize) {
        for _ in 0..frames {
            self.frame();
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// A point inside the Name cell of the row at view position `pos`.
    fn row_point(&self, pos: usize) -> Pos2 {
        let rect = self.ex.details.row_rects().iter().find(|(p, _)| *p == pos).unwrap_or_else(|| panic!("row {pos} is not on screen: {:?}", self.ex.details.row_rects())).1;
        Pos2::new(rect.left() + 150.0, rect.center().y)
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
        let p = self.row_point(pos);
        self.click_at(p, Modifiers::NONE, 1);
    }

    fn click_with(&mut self, pos: usize, mods: Modifiers) {
        let p = self.row_point(pos);
        self.click_at(p, mods, 1);
    }

    fn press_key(&mut self, key: Key, mods: Modifiers) {
        self.frame_with(vec![key_event(key, mods)], mods);
        self.frame();
        self.frame();
    }

    fn selected_paths(&self) -> Vec<String> {
        self.ex.selected_rows().iter().map(|r| r.path.clone()).collect()
    }

    fn focus_path(&self) -> Option<String> {
        self.ex.selection.focus().map(|k| k.as_str().to_string())
    }

    fn text_point(&self, text: &str) -> Option<Pos2> {
        texts(&self.shapes).into_iter().find(|(t, _)| t == text).map(|(_, p)| p + vec2(6.0, 6.0))
    }

    fn view_paths(&self) -> Vec<String> {
        self.ex.view.rows(&self.ex.table).map(|r| r.path.clone()).collect()
    }
}

fn paths(range: impl Iterator<Item = usize>) -> Vec<String> {
    range.map(path_of).collect()
}

/* ---------------------------------------------- drawing --------------------------------------------- */

#[tokio::test(flavor = "multi_thread")]
async fn only_drawn_rows_ask_for_a_thumbnail_and_each_asks_once() {
    let mut rig = rig(400);
    rig.settle(40).await;
    let tracked: Vec<usize> = (0..400).filter(|i| rig.svc.is_tracked(&path_of(*i), Tier::Thumb)).collect();
    assert!(!tracked.is_empty(), "visible rows must request their thumbnails");
    assert!(tracked.len() <= 16, "a 600 px window shows about seven 75 px rows, got {}", tracked.len());
    assert_eq!(tracked, (0..tracked.len()).collect::<Vec<_>>(), "the visible window is the top of the table, nothing below it");
    let snap = rig.svc.snapshot();
    let wanted = snap.core.wants_created as usize;
    assert!(wanted >= tracked.len() && wanted <= tracked.len() + 2, "one request per drawn cell, none for undrawn rows: {wanted} wants for {} tracked rows", tracked.len());
    assert_eq!(rig.store.get_batches().iter().sum::<usize>(), wanted, "the store was asked for exactly the wanted rows, once");
    assert!(snap.core.decode_ok as usize >= tracked.len());
    assert!(snap.core.requests as usize > tracked.len(), "cells re-ask every frame; only the first ask costs anything");
}

#[tokio::test(flavor = "multi_thread")]
async fn drawn_rows_end_up_with_textures_and_steady_frames_cost_nothing() {
    let mut rig = rig(50);
    rig.settle(40).await;
    let tracked: Vec<usize> = (0..50).filter(|i| rig.svc.is_tracked(&path_of(*i), Tier::Thumb)).collect();
    assert!(tracked.iter().all(|i| rig.svc.texture_for(&path_of(*i), Tier::Thumb).is_ready()));
    let before = rig.svc.snapshot();
    for _ in 0..20 {
        rig.frame();
    }
    let after = rig.svc.snapshot();
    assert_eq!(after.core.wants_created, before.core.wants_created);
    assert_eq!(after.core.fetch_batches, before.core.fetch_batches);
    assert_eq!(after.core.decode_started, before.core.decode_started);
    assert_eq!(rig.store.get_batches().iter().sum::<usize>(), after.core.wants_created as usize);
    let drawn = rig.ex.details.drawn_rows() as u64;
    assert!(drawn >= 6 && drawn <= tracked.len() as u64 + 2);
    assert!(after.cache.hits >= before.cache.hits + 20 * drawn, "every steady frame is served from the cache");
}

#[tokio::test(flavor = "multi_thread")]
async fn rows_without_thumbnail_data_show_a_missing_state_not_a_request_loop() {
    let mut rig = rig(3);
    let mut missing = Thumbnail::new("notes.txt");
    missing.path = "/cells/notes.txt".into();
    missing.file_type = "txt".into();
    rig.ex.table.push(missing);
    rig.settle(40).await;
    let snap = rig.svc.snapshot();
    assert!(snap.core.negative_hits > 0);
    assert_eq!(snap.core.fetch_missing, 1);
    let batches_before = rig.store.get_batches().len();
    for _ in 0..30 {
        rig.frame();
    }
    assert_eq!(rig.store.get_batches().len(), batches_before, "a known-missing thumbnail is not looked up again every frame");
}

#[tokio::test(flavor = "multi_thread")]
async fn laid_out_rows_depend_on_the_window_not_on_the_row_count() {
    let mut small = rig(30);
    let mut large = rig(60_000);
    small.settle(4).await;
    large.settle(4).await;
    let (a, b) = (small.ex.details.drawn_rows(), large.ex.details.drawn_rows());
    assert!(a > 0 && a <= 14, "{a}");
    assert_eq!(a, b, "60000 rows lay out as many rows as 30");
    assert!(large.ex.details.page_rows >= 5);
}

#[tokio::test(flavor = "multi_thread")]
async fn scrolling_far_into_a_large_table_only_requests_the_rows_it_lands_on() {
    let mut rig = rig(20_000);
    rig.settle(10).await;
    let before = rig.svc.snapshot().core.wants_created;
    rig.ex.details.scroll_to(15_000);
    rig.settle(20).await;
    let wanted = rig.svc.snapshot().core.wants_created - before;
    assert!(wanted > 0 && wanted <= 40, "{wanted}");
    assert!(rig.ex.details.row_rects().iter().any(|(pos, _)| (14_990..15_010).contains(pos)));
}

#[tokio::test(flavor = "multi_thread")]
async fn an_empty_table_says_so_and_a_filtered_one_counts_what_it_hides() {
    let mut rig = rig(5);
    rig.ex.table.clear();
    rig.settle(3).await;
    assert!(rig.text_point("No items").is_some());
    let mut rig = self::rig(5);
    rig.ex.viewer.filter = "no-such-name".into();
    rig.settle(3).await;
    assert!(rig.text_point("5 items hidden by the current filter").is_some());
    assert_eq!(rig.ex.table.len(), 5, "filtering does not remove rows");
}

/* ------------------------------------------- mouse selection ----------------------------------------- */

#[tokio::test(flavor = "multi_thread")]
async fn a_click_selects_one_row_and_the_preview_follows_it() {
    let mut rig = rig(30);
    rig.settle(3).await;
    rig.click(2);
    assert_eq!(rig.selected_paths(), paths(2..3));
    assert_eq!(rig.focus_path(), Some(path_of(2)));
    assert_eq!(rig.ex.current_thumb.path, path_of(2));
    rig.click(4);
    assert_eq!(rig.selected_paths(), paths(4..5));
    assert_eq!(rig.ex.current_thumb.path, path_of(4));
    assert!(rig.ex.thumbnail_rx.is_empty(), "selection no longer travels through the thumbnail channel");
}

#[tokio::test(flavor = "multi_thread")]
async fn ctrl_click_toggles_rows() {
    let mut rig = rig(30);
    rig.settle(3).await;
    rig.click(1);
    rig.click_with(3, Modifiers::COMMAND);
    rig.click_with(5, Modifiers::COMMAND);
    assert_eq!(rig.selected_paths(), [path_of(1), path_of(3), path_of(5)]);
    rig.click_with(3, Modifiers::COMMAND);
    assert_eq!(rig.selected_paths(), [path_of(1), path_of(5)]);
    assert_eq!(rig.ex.selection_count(), 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn shift_click_selects_the_range_and_multi_select_keeps_the_preview_on_the_focused_row() {
    let mut rig = rig(30);
    rig.settle(3).await;
    rig.click(1);
    rig.click_with(5, Modifiers::SHIFT);
    assert_eq!(rig.selected_paths(), paths(1..6));
    assert_eq!(rig.focus_path(), Some(path_of(5)));
    assert_eq!(rig.ex.current_thumb.path, path_of(5), "one deterministic preview row, no lag behind a stream of events");
    assert!(rig.ex.thumbnail_rx.is_empty());
    rig.click_with(3, Modifiers::SHIFT);
    assert_eq!(rig.selected_paths(), paths(1..4));
    rig.click_with(6, Modifiers::COMMAND | Modifiers::SHIFT);
    assert_eq!(rig.selected_paths(), paths(1..7), "ctrl+shift adds the range");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_range_follows_the_sorted_order() {
    let mut rig = rig(30);
    rig.settle(3).await;
    rig.ex.view.set_sort(crate::ui::file_table::rows::SortSpec::single(Col::Size.id(), true));
    rig.settle(3).await;
    let order = rig.view_paths();
    rig.click(0);
    rig.click_with(3, Modifiers::SHIFT);
    assert_eq!(rig.selected_paths(), order[0..4].to_vec());
}

#[tokio::test(flavor = "multi_thread")]
async fn folder_rows_are_selectable_and_do_not_navigate_on_a_single_click() {
    let mut rig = rig(0);
    rig.ex.table.bulk_load([dir_row("/cells/sub_a"), image_row(1), dir_row("/cells/sub_b")]);
    rig.settle(3).await;
    let start = rig.ex.current_path.clone();
    rig.click(0);
    assert_eq!(rig.selected_paths(), ["/cells/sub_a".to_string()]);
    assert_eq!(rig.ex.current_path, start, "a single click only selects a folder");
    rig.click_with(2, Modifiers::COMMAND);
    assert_eq!(rig.ex.selection_count(), 2);
    assert!(rig.selected_paths().contains(&"/cells/sub_b".to_string()));
    let dirs: Vec<String> = rig.ex.selected_rows().iter().filter(|r| r.file_type == "<DIR>").map(|r| r.path.clone()).collect();
    assert_eq!(dirs, ["/cells/sub_a".to_string(), "/cells/sub_b".to_string()], "selected folders reach the Exclude directories action");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_double_click_opens_a_folder_and_a_category_row() {
    let folder = tempfile::tempdir().unwrap();
    let sub_a = folder.path().join("sub_a");
    std::fs::create_dir(&sub_a).unwrap();
    let sub_a = sub_a.to_string_lossy().to_string();
    let mut rig = rig(0);
    rig.ex.table.bulk_load([dir_row(&sub_a), dir_row("cat://Landscape")]);
    rig.settle(3).await;
    let p = rig.row_point(0);
    rig.click_at(p, Modifiers::NONE, 2);
    assert_eq!(rig.ex.current_path, sub_a);
    assert_eq!(rig.ex.scope(), &crate::command::Scope::Folder(sub_a.clone()));
    let mut rig = self::rig(0);
    rig.ex.table.bulk_load([dir_row("cat://Landscape")]);
    rig.settle(3).await;
    let p = rig.row_point(0);
    rig.click_at(p, Modifiers::NONE, 2);
    assert!(rig.ex.viewer.requested_tabs.iter().any(|t| matches!(t, TabAction::OpenCategory(c) if c == "Landscape")));
}

/* ------------------------------------------------ keyboard ---------------------------------------- */

#[tokio::test(flavor = "multi_thread")]
async fn arrow_keys_move_the_focus_and_select_the_row() {
    let mut rig = rig(30);
    rig.settle(3).await;
    rig.click(2);
    rig.press_key(Key::ArrowDown, Modifiers::NONE);
    assert_eq!((rig.focus_path(), rig.selected_paths()), (Some(path_of(3)), paths(3..4)));
    assert_eq!(rig.ex.current_thumb.path, path_of(3));
    rig.press_key(Key::ArrowUp, Modifiers::NONE);
    rig.press_key(Key::ArrowUp, Modifiers::NONE);
    assert_eq!(rig.focus_path(), Some(path_of(1)));
    rig.press_key(Key::End, Modifiers::NONE);
    assert_eq!(rig.focus_path(), Some(path_of(29)));
    rig.press_key(Key::ArrowDown, Modifiers::NONE);
    assert_eq!(rig.focus_path(), Some(path_of(29)), "the focus stops at the last row");
    rig.press_key(Key::Home, Modifiers::NONE);
    assert_eq!(rig.focus_path(), Some(path_of(0)));
    assert!(rig.ex.details.row_rects().iter().any(|(pos, _)| *pos == 0), "the focused row is scrolled into view");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_focus_row_scrolls_into_view_when_the_keyboard_moves_it_far() {
    let mut rig = rig(200);
    rig.settle(3).await;
    rig.click(0);
    rig.press_key(Key::End, Modifiers::NONE);
    rig.settle(4).await;
    assert!(rig.ex.details.row_rects().iter().any(|(pos, _)| *pos == 199));
    rig.press_key(Key::Home, Modifiers::NONE);
    rig.settle(4).await;
    assert!(rig.ex.details.row_rects().iter().any(|(pos, _)| *pos == 0));
}

#[tokio::test(flavor = "multi_thread")]
async fn shift_arrows_and_page_keys_extend_the_selection() {
    let mut rig = rig(40);
    rig.settle(3).await;
    rig.click(2);
    rig.press_key(Key::ArrowDown, Modifiers::SHIFT);
    rig.press_key(Key::ArrowDown, Modifiers::SHIFT);
    assert_eq!(rig.selected_paths(), paths(2..5));
    rig.press_key(Key::ArrowUp, Modifiers::SHIFT);
    assert_eq!(rig.selected_paths(), paths(2..4));
    let page = rig.ex.details.page_rows - 1;
    rig.press_key(Key::PageDown, Modifiers::NONE);
    assert_eq!(rig.focus_path(), Some(path_of(3 + page)), "page down moves by a screenful minus one row");
    rig.press_key(Key::PageUp, Modifiers::SHIFT);
    assert_eq!(rig.focus_path(), Some(path_of(3)));
    assert_eq!(rig.selected_paths(), paths(3..4 + page), "the range runs from the row the plain page move landed on");
}

#[tokio::test(flavor = "multi_thread")]
async fn ctrl_a_selects_every_visible_row_and_escape_clears() {
    let mut rig = rig(25);
    rig.settle(3).await;
    rig.click(1);
    rig.press_key(Key::A, Modifiers::COMMAND);
    assert_eq!(rig.ex.selection_count(), 25);
    assert_eq!(rig.selected_paths(), paths(0..25));
    rig.press_key(Key::Escape, Modifiers::NONE);
    assert!(rig.ex.selection.is_empty());
    assert!(rig.selected_paths().is_empty(), "an empty selection selects no rows, never all of them");
    assert!(rig.ex.selected_rows().is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn ctrl_a_with_a_filter_selects_only_what_is_visible() {
    let mut rig = rig(25);
    rig.ex.viewer.filter = "img_000".into();
    rig.settle(3).await;
    rig.click(0);
    rig.press_key(Key::A, Modifiers::COMMAND);
    assert_eq!(rig.ex.selection_count(), 10);
    assert!(rig.selected_paths().iter().all(|p| p.contains("img_000")));
}

#[tokio::test(flavor = "multi_thread")]
async fn enter_opens_the_focused_row_and_space_toggles_the_preview_pane() {
    let folder = tempfile::tempdir().unwrap();
    let sub_a = folder.path().join("sub_a");
    std::fs::create_dir(&sub_a).unwrap();
    let sub_a = sub_a.to_string_lossy().to_string();
    let mut rig = rig(0);
    rig.ex.table.bulk_load([dir_row(&sub_a), dir_row("tag://Fox")]);
    rig.settle(3).await;
    rig.click(1);
    rig.press_key(Key::Enter, Modifiers::NONE);
    assert!(rig.ex.viewer.requested_tabs.iter().any(|t| matches!(t, TabAction::OpenTag(t) if t == "Fox")));
    rig.click(0);
    rig.press_key(Key::Enter, Modifiers::NONE);
    assert_eq!(rig.ex.current_path, sub_a);
    let shown = rig.ex.open_preview_pane;
    rig.press_key(Key::Space, Modifiers::NONE);
    assert_eq!(rig.ex.open_preview_pane, !shown);
}

#[tokio::test(flavor = "multi_thread")]
async fn enter_and_double_click_open_a_file_with_the_system_opener_but_never_a_virtual_path() {
    let mut rig = rig(3);
    let mut inner = Thumbnail::new("inner.png");
    inner.path = "zip:///cells/a.zip!/inner.png".into();
    inner.file_type = "png".into();
    rig.ex.table.push(inner);
    rig.settle(3).await;
    rig.click(1);
    rig.press_key(Key::Enter, Modifiers::NONE);
    assert_eq!(rig.ex.row_sync.opened, [path_of(1)]);
    let p = rig.row_point(2);
    rig.click_at(p, Modifiers::NONE, 2);
    assert_eq!(rig.ex.row_sync.opened, [path_of(1), path_of(2)]);
    let p = rig.row_point(3);
    rig.click_at(p, Modifiers::NONE, 2);
    rig.press_key(Key::Enter, Modifiers::NONE);
    assert_eq!(rig.ex.row_sync.opened.len(), 2, "a path inside an archive cannot be handed to the system opener");
}

#[tokio::test(flavor = "multi_thread")]
async fn on_demand_row_work_waits_for_the_focus_to_settle() {
    let mut rig = rig(10);
    rig.settle(3).await;
    rig.click(1);
    rig.press_key(Key::ArrowDown, Modifiers::NONE);
    rig.press_key(Key::ArrowDown, Modifiers::NONE);
    assert_eq!(rig.ex.row_sync.pending_focus().map(|k| k.as_str().to_string()), Some(path_of(3)), "only the row the focus rests on is pending");
    tokio::time::sleep(Duration::from_millis(220)).await;
    rig.frame();
    assert!(rig.ex.row_sync.pending_focus().is_none(), "the work started once the focus settled");
}

#[tokio::test(flavor = "multi_thread")]
async fn ctrl_shift_c_copies_the_selected_paths() {
    let mut rig = rig(10);
    rig.settle(3).await;
    rig.click(2);
    rig.frame_with(vec![key_event(Key::C, Modifiers::COMMAND | Modifiers::SHIFT)], Modifiers::COMMAND | Modifiers::SHIFT);
    assert_eq!(rig.copied, [path_of(2)]);
    rig.click_with(4, Modifiers::SHIFT);
    rig.frame_with(vec![key_event(Key::C, Modifiers::COMMAND | Modifiers::SHIFT)], Modifiers::COMMAND | Modifiers::SHIFT);
    assert_eq!(rig.copied, [paths(2..5).join("\n")]);
}

#[tokio::test(flavor = "multi_thread")]
async fn keys_are_ignored_while_the_pointer_is_elsewhere_and_nothing_is_active() {
    let mut rig = rig(10);
    rig.settle(3).await;
    rig.frame_with(vec![Event::PointerMoved(Pos2::new(5000.0, 5000.0))], Modifiers::NONE);
    rig.press_key(Key::ArrowDown, Modifiers::NONE);
    assert!(rig.ex.selection.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn the_editing_hotkeys_of_the_old_spreadsheet_table_do_nothing() {
    let mut rig = rig(10);
    rig.settle(3).await;
    rig.click(2);
    let before = (rig.ex.table.len(), rig.view_paths());
    for (key, mods) in [(Key::X, Modifiers::COMMAND), (Key::V, Modifiers::COMMAND), (Key::D, Modifiers::COMMAND), (Key::Delete, Modifiers::NONE), (Key::Backspace, Modifiers::NONE), (Key::Delete, Modifiers::COMMAND), (Key::Z, Modifiers::COMMAND), (Key::Y, Modifiers::COMMAND), (Key::Insert, Modifiers::NONE)] {
        rig.press_key(key, mods);
    }
    assert_eq!((rig.ex.table.len(), rig.view_paths()), before);
    assert_eq!(rig.selected_paths(), paths(2..3));
    rig.ex.table.check_invariants().unwrap();
}

/* ----------------------------------------------- sorting ------------------------------------------- */

#[tokio::test(flavor = "multi_thread")]
async fn clicking_a_header_sorts_by_that_column_and_cycles() {
    let mut rig = rig(30);
    rig.settle(3).await;
    let p = rig.text_point("Size").expect("the Size header is drawn");
    rig.click_at(p, Modifiers::NONE, 1);
    rig.settle(2).await;
    let sizes: Vec<u64> = rig.ex.view.rows(&rig.ex.table).map(|r| r.size).collect();
    assert!(sizes.windows(2).all(|w| w[0] <= w[1]), "ascending: {sizes:?}");
    let p = rig.text_point("Size").unwrap();
    rig.click_at(p, Modifiers::NONE, 1);
    rig.settle(2).await;
    let sizes: Vec<u64> = rig.ex.view.rows(&rig.ex.table).map(|r| r.size).collect();
    assert!(sizes.windows(2).all(|w| w[0] >= w[1]), "descending: {sizes:?}");
    let p = rig.text_point("Size").unwrap();
    rig.click_at(p, Modifiers::NONE, 1);
    rig.settle(2).await;
    assert!(rig.ex.view.sort().is_empty());
    assert_eq!(rig.view_paths(), paths(0..30), "sorting off restores the store order");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_shift_click_on_a_second_header_adds_a_sort_key() {
    let mut rig = rig(60);
    rig.settle(3).await;
    let p = rig.text_point("Size").unwrap();
    rig.click_at(p, Modifiers::NONE, 1);
    rig.settle(2).await;
    let p = rig.text_point("Name").unwrap();
    rig.click_at(p, Modifiers::SHIFT, 1);
    rig.settle(2).await;
    let keys: Vec<u8> = rig.ex.view.sort().keys().iter().map(|k| k.column).collect();
    assert_eq!(keys, [Col::Size.id(), Col::Name.id()]);
    let pairs: Vec<(u64, String)> = rig.ex.view.rows(&rig.ex.table).map(|r| (r.size, r.filename.clone())).collect();
    assert!(pairs.windows(2).all(|w| w[0].0 < w[1].0 || (w[0].0 == w[1].0 && w[0].1 <= w[1].1)));
}

#[tokio::test(flavor = "multi_thread")]
async fn sort_and_hidden_columns_are_saved_and_restored_in_a_new_explorer() {
    let mut rig = rig(20);
    rig.settle(3).await;
    let p = rig.text_point("Size").unwrap();
    rig.click_at(p, Modifiers::NONE, 1);
    rig.settle(2).await;
    let events = vec![DetailsEvent::ToggleColumn { column: Col::Hash, hidden: true }, DetailsEvent::ToggleColumn { column: Col::Path, hidden: true }];
    rig.ex.apply_details_events(&rig.ctx.clone(), events);
    rig.settle(2).await;
    assert!(rig.text_point("Path").is_none(), "a hidden column leaves the header");
    let mut fresh = self::rig(20);
    fresh.ctx = rig.ctx.clone();
    fresh.clock = rig.clock + 1.0;
    fresh.settle(3).await;
    assert_eq!(fresh.ex.view.sort().position_of(Col::Size.id()), Some((0, false)));
    assert!(fresh.ex.details.prefs.is_hidden(Col::Path));
    assert!(fresh.text_point("Path").is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn the_column_chooser_never_offers_to_hide_the_name() {
    let mut rig = rig(5);
    rig.settle(3).await;
    let p = rig.text_point("Type").unwrap();
    rig.frame_with(vec![Event::PointerMoved(p), button(p, true, Modifiers::NONE, PointerButton::Secondary), button(p, false, Modifiers::NONE, PointerButton::Secondary)], Modifiers::NONE);
    rig.settle(3).await;
    assert!(rig.text_point("Columns").is_some(), "the header menu opened");
    assert!(rig.text_point("Name").is_some());
    let before = rig.ex.details.prefs.clone();
    rig.ex.details.prefs.set_hidden(Col::Name, true);
    assert_eq!(rig.ex.details.prefs, before);
}

/* -------------------------------------------- filters and stats --------------------------------------- */

#[tokio::test(flavor = "multi_thread")]
async fn the_type_toggles_refresh_the_view_and_never_remove_rows() {
    let mut rig = rig(0);
    rig.ex.table.bulk_load((0..6).map(image_row));
    let mut clip = Thumbnail::new("clip.mp4");
    clip.path = "/cells/clip.mp4".into();
    clip.file_type = "mp4".into();
    rig.ex.table.push(clip);
    rig.ex.table.push(dir_row("/cells/sub"));
    rig.ex.sync_view();
    assert_eq!(rig.ex.view.len(), 8);
    rig.ex.viewer.types_show_images = false;
    assert_eq!(rig.ex.sync_view(), Refresh::Rebuilt);
    assert_eq!(rig.ex.view.len(), 2, "images hidden, the video and the folder stay");
    rig.ex.viewer.types_show_videos = false;
    rig.ex.sync_view();
    assert_eq!(rig.ex.view.len(), 1);
    rig.ex.viewer.types_show_dirs = false;
    rig.ex.sync_view();
    assert!(rig.ex.view.is_empty());
    assert_eq!(rig.ex.table.len(), 8, "the rows are still there");
    rig.ex.viewer.types_show_images = true;
    rig.ex.viewer.types_show_videos = true;
    rig.ex.viewer.types_show_dirs = true;
    rig.ex.sync_view();
    assert_eq!(rig.ex.view.len(), 8, "turning the toggles back on brings every row back");
}

#[tokio::test(flavor = "multi_thread")]
async fn selecting_by_path_picks_only_visible_rows_in_view_order() {
    let mut rig = rig(12);
    rig.ex.viewer.filter = "img_000".into();
    rig.settle(3).await;
    rig.ex.select_paths([path_of(5).as_str(), path_of(1).as_str(), path_of(11).as_str(), "/not/there.jpg"]);
    assert_eq!(rig.selected_paths(), [path_of(1), path_of(5)], "img_0011 is hidden by the filter and the unknown path is ignored");
    assert_eq!(rig.ex.selection_count(), 2);
    assert_eq!(rig.ex.current_thumb.path, path_of(5));
    rig.ex.clear_selection();
    assert!(rig.selected_paths().is_empty() && rig.ex.selection_count() == 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn hiding_rows_deselects_them() {
    let mut rig = rig(10);
    rig.settle(3).await;
    rig.click(1);
    rig.click_with(3, Modifiers::COMMAND);
    rig.ex.viewer.filter = "img_0003".into();
    rig.settle(3).await;
    assert_eq!(rig.selected_paths(), [path_of(3)], "a hidden row is no longer selected");
    assert_eq!(rig.ex.selection_count(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn idle_frames_do_not_touch_the_view_or_recompute_its_stats() {
    let mut rig = rig(500);
    rig.settle(5).await;
    let counts = rig.ex.view.counts();
    let stats = crate::ui::file_table::view_stats(&rig.ex);
    assert_eq!(stats.rows.total, 500);
    assert_eq!(stats.rows.visible, 500);
    for _ in 0..60 {
        rig.frame();
    }
    let after = rig.ex.view.counts();
    assert_eq!(after.rebuilt, counts.rebuilt);
    assert_eq!(after.appended, counts.appended);
    assert!(after.unchanged >= counts.unchanged + 60);
    assert_eq!(crate::ui::file_table::view_stats(&rig.ex), stats, "the cached numbers and their revision stay put");
}

#[tokio::test(flavor = "multi_thread")]
async fn view_stats_follow_rows_filters_and_selection() {
    let mut rig = rig(40);
    rig.settle(3).await;
    let first = crate::ui::file_table::view_stats(&rig.ex);
    assert_eq!(first.rows.images, 40);
    assert_eq!(first.selection.count, 0);
    rig.click(0);
    rig.click_with(3, Modifiers::SHIFT);
    let selected = crate::ui::file_table::view_stats(&rig.ex);
    assert_eq!(selected.selection.count, 4);
    assert_eq!(selected.selection.images, 4);
    assert_eq!(selected.rows, first.rows, "selecting does not touch the row counts");
    assert!(selected.selection_revision > first.selection_revision);
    rig.ex.table.push(image_row(100));
    rig.settle(2).await;
    let grown = crate::ui::file_table::view_stats(&rig.ex);
    assert_eq!(grown.rows.total, 41);
    assert!(grown.revision > selected.revision);
    rig.ex.viewer.filter = "img_000".into();
    rig.settle(2).await;
    let filtered = crate::ui::file_table::view_stats(&rig.ex);
    assert_eq!(filtered.rows.visible, 10);
    assert_eq!(filtered.rows.hidden_by_filter, 31);
}

/* ------------------------------------------------ menus ------------------------------------------- */

#[tokio::test(flavor = "multi_thread")]
async fn the_context_menu_acts_on_the_selection() {
    let _serial = crate::command::bus::testing::serial();
    let mut rig = rig(10);
    rig.settle(3).await;
    rig.click(1);
    rig.click_with(3, Modifiers::SHIFT);
    let at = rig.row_point(2);
    rig.frame_with(vec![Event::PointerMoved(at), button(at, true, Modifiers::NONE, PointerButton::Secondary), button(at, false, Modifiers::NONE, PointerButton::Secondary)], Modifiers::NONE);
    rig.settle(4).await;
    assert_eq!(rig.ex.selection_count(), 3, "a secondary click inside the selection keeps it");
    for label in ["Generate Description", "Find Similar (CLIP)", "Open selection in new tab", "Refine Selection"] {
        assert!(rig.text_point(label).is_some(), "{label} is in the menu");
    }
    assert!(rig.text_point("3 items").is_some());
    assert!(rig.text_point("Attach to chat window").is_some(), "the chat attachment is offered in every scope");
    let p = rig.text_point("Open selection in new tab").unwrap();
    rig.click_at(p, Modifiers::NONE, 1);
    let mine = crate::command::bus::testing::take_matching(|c| matches!(c, crate::command::AppCommand::OpenRows(tab) if tab.title.starts_with("Selected (3)")));
    assert_eq!(mine.len(), 1);
    if let crate::command::AppCommand::OpenRows(tab) = &mine[0] {
        assert_eq!(tab.rows.iter().map(|r| r.path.clone()).collect::<Vec<_>>(), paths(1..4));
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn escape_closes_an_open_menu_without_clearing_the_selection() {
    let mut rig = rig(10);
    rig.settle(3).await;
    rig.click(2);
    let at = rig.row_point(2);
    rig.frame_with(vec![Event::PointerMoved(at), button(at, true, Modifiers::NONE, PointerButton::Secondary), button(at, false, Modifiers::NONE, PointerButton::Secondary)], Modifiers::NONE);
    rig.settle(3).await;
    assert!(egui::Popup::is_any_open(&rig.ctx) && rig.text_point("Find Similar (CLIP)").is_some());
    rig.press_key(Key::Escape, Modifiers::NONE);
    assert!(!egui::Popup::is_any_open(&rig.ctx), "the menu closed");
    assert_eq!(rig.selected_paths(), paths(2..3), "the selection survived the Escape that closed the menu");
    rig.press_key(Key::Escape, Modifiers::NONE);
    assert!(rig.ex.selection.is_empty(), "with no menu open Escape clears the selection");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_secondary_click_outside_the_selection_selects_that_row_alone() {
    let mut rig = rig(10);
    rig.settle(3).await;
    rig.click(1);
    rig.click_with(2, Modifiers::SHIFT);
    let at = rig.row_point(5);
    rig.frame_with(vec![Event::PointerMoved(at), button(at, true, Modifiers::NONE, PointerButton::Secondary), button(at, false, Modifiers::NONE, PointerButton::Secondary)], Modifiers::NONE);
    rig.frame();
    assert_eq!(rig.selected_paths(), paths(5..6));
}

#[tokio::test(flavor = "multi_thread")]
async fn clicking_a_tag_or_category_asks_for_its_tab_without_changing_the_selection() {
    let mut rig = rig(3);
    {
        let mut row = rig.ex.table.get_mut(0).unwrap();
        row.tags = vec!["fox".into()];
        row.category = Some("animals".into());
    }
    for col in [Col::Path, Col::Type, Col::Modified, Col::DbCreated] {
        rig.ex.details.prefs.set_hidden(col, true);
    }
    rig.settle(3).await;
    let tag = rig.text_point("fox").expect("the tag chip is drawn");
    rig.click_at(tag, Modifiers::NONE, 1);
    assert!(rig.ex.viewer.requested_tabs.iter().any(|t| matches!(t, TabAction::OpenTag(t) if t == "fox")));
    assert!(rig.ex.selection.is_empty(), "a chip click is not a row click");
    let cat = rig.text_point("animals").expect("the category button is drawn");
    rig.click_at(cat, Modifiers::NONE, 1);
    assert_eq!(rig.ex.viewer.requested_tabs.iter().filter(|t| matches!(t, TabAction::OpenCategory(c) if c == "animals")).count(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn menu_events_on_an_empty_selection_do_nothing() {
    let mut rig = rig(5);
    rig.settle(3).await;
    let _serial = crate::command::bus::testing::serial();
    let is_tab = |c: &crate::command::AppCommand| matches!(c, crate::command::AppCommand::OpenRows(_) | crate::command::AppCommand::OpenScope { .. });
    crate::command::bus::testing::take_matching(is_tab);
    rig.ex.apply_details_events(&rig.ctx.clone(), vec![DetailsEvent::Menu(MenuAction::OpenSelectionInNewTab), DetailsEvent::Menu(MenuAction::RefineSelection)]);
    assert!(crate::command::bus::testing::take_matching(is_tab).is_empty());
    assert!(rig.ex.selected_rows().is_empty());
}

/* ------------------------------------------- store invariants ---------------------------------------- */

#[tokio::test(flavor = "multi_thread")]
async fn row_removals_during_use_keep_the_store_the_view_and_the_selection_consistent() {
    let mut rig = rig(60);
    rig.settle(3).await;
    rig.click(2);
    rig.click_with(6, Modifiers::SHIFT);
    rig.ex.table.retain(|r| !r.path.ends_with('3') && !r.path.ends_with('4'));
    rig.settle(3).await;
    rig.ex.check_row_invariants().unwrap();
    assert!(rig.selected_paths().iter().all(|p| !p.ends_with('3') && !p.ends_with('4')), "removed rows are no longer selected");
    rig.ex.table.remove_path(&path_of(5));
    rig.ex.table.push(image_row(500));
    rig.ex.table.get_mut_by_path(&path_of(6)).unwrap().size = 9_999;
    rig.settle(3).await;
    rig.ex.check_row_invariants().unwrap();
    rig.ex.table.clear();
    rig.settle(3).await;
    rig.ex.check_row_invariants().unwrap();
    assert!(rig.ex.selection.is_empty());
}

/* ----------------------------------------------- devtools -------------------------------------------- */

#[tokio::test(flavor = "multi_thread")]
async fn the_row_store_window_reports_the_explorer_and_runs_its_commands() {
    let mut rig = rig(50);
    row_store::set_open(&rig.ctx, true);
    rig.settle(3).await;
    let snapshot = row_store::latest(&rig.ctx).expect("the explorer published a snapshot");
    assert_eq!((snapshot.store_len, snapshot.view_len), (50, 50));
    assert!(snapshot.drawn_rows > 0 && snapshot.drawn_rows < 50);
    assert!(!snapshot.frame_ms.is_empty());
    assert!(snapshot.memory.total() > 0);
    row_store::push_command(&rig.ctx, snapshot.owner, row_store::Command::CheckInvariants);
    rig.settle(2).await;
    assert_eq!(rig.ex.row_sync.last_check, Some(Ok(())));
    row_store::push_command(&rig.ctx, snapshot.owner, row_store::Command::Synthetic(1_500));
    rig.settle(2).await;
    assert_eq!(rig.ex.table.len(), 1_500);
    rig.ex.check_row_invariants().unwrap();
    row_store::push_command(&rig.ctx, snapshot.owner, row_store::Command::ClearRows);
    rig.settle(2).await;
    assert!(rig.ex.table.is_empty());
    row_store::push_command(&rig.ctx, snapshot.owner, row_store::Command::ResetFrameStats);
    rig.frame();
    assert!(rig.ex.details.frame_times().len() <= 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_invariant_check_reports_a_corrupted_selection() {
    let mut rig = rig(10);
    rig.settle(3).await;
    rig.click(1);
    rig.ex.check_row_invariants().unwrap();
    let ghost = rig.ex.table.key_for("/not/in/the/store.jpg");
    rig.ex.selection.toggle(ghost);
    assert!(rig.ex.check_row_invariants().unwrap_err().contains("selection"));
}

fn rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/statm").ok().and_then(|s| s.split_whitespace().nth(1).and_then(|p| p.parse::<f64>().ok())).map(|pages| pages * 4.0 / 1024.0).unwrap_or(0.0)
}

/// Prints frame-time and allocation figures for scrolling a large table; run with `--ignored --nocapture`, row count from `AME_BENCH_ROWS`.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_details_view_scrolling() {
    use crate::ui::file_table::thumbs::perf::measure;
    use egui::{MouseWheelSource, MouseWheelUnit, TouchPhase, pos2};
    let rows: usize = std::env::var("AME_BENCH_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(20_000);
    let scroll = std::env::var("AME_BENCH_SCROLL").map(|v| v != "0").unwrap_or(true);
    let frames: usize = std::env::var("AME_BENCH_FRAMES").ok().and_then(|v| v.parse().ok()).unwrap_or(400);
    let mut rig = rig(rows);
    let mut times = Vec::new();
    let mut allocs = Vec::new();
    let mut bytes = Vec::new();
    let rss0 = rss_mb();
    for frame in 0..(frames + 30) {
        let mut events = vec![Event::PointerMoved(pos2(500.0, 300.0))];
        if scroll && frame >= 30 {
            events.push(Event::MouseWheel { unit: MouseWheelUnit::Point, delta: vec2(0.0, -600.0), phase: TouchPhase::Move, source: MouseWheelSource::Wheel, modifiers: Modifiers::NONE });
        }
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1100.0, 600.0))), events, ..Default::default() };
        let (ex, svc, ctx) = (&mut rig.ex, rig.svc.clone(), &rig.ctx);
        let started = std::time::Instant::now();
        let (_, counts) = measure(|| {
            let mut out = ctx.run_ui(input, |ui| {
                svc.poll(ui.ctx());
                CentralPanel::default().show(ui, |ui| ex.details_panel(ui));
            });
            out.textures_delta.clear();
            let _ = ctx.tessellate(out.shapes, out.pixels_per_point);
        });
        if frame >= 30 {
            times.push(started.elapsed().as_secs_f64() * 1000.0);
            allocs.push(counts.allocs as f64);
            bytes.push(counts.bytes as f64);
        }
        tokio::time::sleep(Duration::from_millis(16)).await;
    }
    let pct = |v: &mut Vec<f64>, p: f64| {
        v.sort_by(f64::total_cmp);
        v[((v.len() - 1) as f64 * p) as usize]
    };
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    println!(
        "BENCH rows={rows} frames={frames} scroll={scroll}: frame ms p50 {:.2} p95 {:.2} max {:.2} | ui-thread allocs/frame mean {:.0} p95 {:.0} | bytes/frame mean {:.0} p95 {:.0} | drawn rows {} | rss {rss0:.0} -> {:.0} MB",
        pct(&mut times.clone(), 0.5),
        pct(&mut times.clone(), 0.95),
        pct(&mut times, 1.0),
        mean(&allocs),
        pct(&mut allocs.clone(), 0.95),
        mean(&bytes),
        pct(&mut bytes.clone(), 0.95),
        rig.ex.details.drawn_rows(),
        rss_mb(),
    );
}

/// Prints how long the view, filter and selection operations take on a large store; run with `--ignored --nocapture`, row count from `AME_BENCH_ROWS`.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_view_refresh_and_selection() {
    use crate::ui::file_table::rows::thumbnail::synthetic;
    use crate::ui::file_table::rows::{PathKey, SortSpec, ViewRef};
    use std::time::Instant;
    let rows: usize = std::env::var("AME_BENCH_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(100_000);
    let mut ex = FileExplorer::new(true);
    let ms = |t: Instant| t.elapsed().as_secs_f64() * 1000.0;
    let t = Instant::now();
    ex.table.replace_all(synthetic(rows));
    println!("BENCH rows={rows}: store load {:.1} ms", ms(t));
    let t = Instant::now();
    ex.sync_view();
    println!("BENCH unsorted first refresh {:.2} ms", ms(t));
    for (name, spec) in [
        ("name", SortSpec::single(Col::Name.id(), false)),
        ("size", SortSpec::single(Col::Size.id(), false)),
        ("modified", SortSpec::single(Col::Modified.id(), true)),
        ("tags", SortSpec::single(Col::Tags.id(), false)),
        ("size then name", {
            let mut s = SortSpec::single(Col::Size.id(), false);
            s.click(Col::Name.id(), true);
            s
        }),
    ] {
        ex.view.set_sort(spec);
        let t = Instant::now();
        ex.sync_view();
        println!("BENCH sort by {name}: rebuild {:.1} ms", ms(t));
    }
    let t = Instant::now();
    for _ in 0..1000 {
        ex.sync_view();
    }
    println!("BENCH idle sync_view: {:.4} ms per frame", ms(t) / 1000.0);
    ex.viewer.filter = "img_05".into();
    let t = Instant::now();
    ex.sync_view();
    println!("BENCH text filter 'img_05' (sorted by size then name): rebuild {:.1} ms, {} visible", ms(t), ex.view.len());
    ex.viewer.filter.clear();
    ex.sync_view();
    let more: Vec<Thumbnail> = synthetic(rows + 2_000).skip(rows).collect();
    let t = Instant::now();
    ex.table.bulk_load(more);
    ex.sync_view();
    println!("BENCH append 2000 rows into the sorted view: {:.2} ms (refresh {:?})", ms(t), ex.view.last_refresh());
    let first = ex.view.rows(&ex.table).next().map(|r| PathKey::new(&r.path)).unwrap();
    let last = ex.view.rows(&ex.table).last().map(|r| PathKey::new(&r.path)).unwrap();
    let t = Instant::now();
    ex.selection.click(first);
    let view = ViewRef::new(&ex.table, &ex.view);
    ex.selection.select_range(&view, &last, false);
    println!("BENCH shift-select the whole view ({} rows): {:.2} ms", ex.selection.len(), ms(t));
    let t = Instant::now();
    ex.sync_view();
    println!("BENCH sync after a 100%-selection (mirror + stats): {:.2} ms", ms(t));
    let t = Instant::now();
    let selected = ex.selected_rows().len();
    println!("BENCH selected_rows() of {selected}: {:.2} ms", ms(t));
    let t = Instant::now();
    ex.viewer.filter = "img_05".into();
    ex.sync_view();
    println!("BENCH filter with a full selection (rebuild + prune to {}): {:.1} ms", ex.selection.len(), ms(t));
    let t = Instant::now();
    ex.table.retain(|r| r.size % 2 == 0);
    ex.sync_view();
    println!("BENCH retain half the rows: {:.1} ms", ms(t));
    let t = Instant::now();
    ex.check_row_invariants().unwrap();
    println!("BENCH invariant check: {:.1} ms", ms(t));
}

