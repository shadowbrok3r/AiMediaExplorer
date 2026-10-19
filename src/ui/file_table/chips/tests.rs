//! The filter strip through headless frames at three window widths, plus its note and its menus.

use egui::epaint::ClippedShape;
use egui::{CentralPanel, Context, Event, Pos2, RawInput, Rect, vec2};

use super::*;
use crate::database::Thumbnail;
use crate::ui::file_table::filters::group_from_spec;
use crate::ui::file_table::rows::{FilterSpec, Tri};
use crate::ui::test_support::{press_release, test_ctx, text_pos, texts};

fn rows() -> Vec<Thumbnail> {
    (0..40)
        .map(|i| {
            let mut t = Thumbnail::new(&format!("img_{i:02}.jpg"));
            t.path = format!("/pics/img_{i:02}.jpg");
            t.file_type = "jpg".into();
            t.size = if i % 4 == 0 { 2_000 } else { 80_000 + i as u64 };
            t
        })
        .collect()
}

fn explorer(spec: FilterSpec) -> FileExplorer {
    let mut ex = FileExplorer::new(true);
    ex.table.bulk_load(rows());
    ex.set_filter_spec(spec);
    ex.sync_view();
    ex
}

fn busy() -> FilterSpec {
    let mut s = FilterSpec::default();
    s.types.videos = false;
    s.modified.after = Some("2020-01-01".into());
    s.skip_icons = true;
    s.ext_deny = vec!["gif".into(), "tmp".into(), "bak".into()];
    s.excluded_terms = vec!["thumbs".into(), "cache".into()];
    s.tags = vec!["holiday".into()];
    s.categories = vec!["Travel".into()];
    s.clip = Tri::Yes;
    s.normalized()
}

struct Screen {
    ctx: Context,
    clock: f64,
    width: f32,
    shapes: Vec<ClippedShape>,
    strip: Rect,
}

impl Screen {
    fn new(width: f32) -> Self {
        Self { ctx: test_ctx(), clock: 0.0, width, shapes: Vec::new(), strip: Rect::NOTHING }
    }

    fn frame(&mut self, ex: &mut FileExplorer, events: Vec<Event>) {
        self.clock += 0.1;
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(self.width, 1400.0))), events, time: Some(self.clock), ..Default::default() };
        let mut strip = Rect::NOTHING;
        let out = self.ctx.run_ui(input, |ui| {
            CentralPanel::default().show(ui, |ui| {
                strip = ui.scope(|ui| show(ui, ex)).response.rect;
            });
        });
        self.strip = strip;
        self.shapes = out.shapes.clone();
        out.drop_without_applying_deltas();
    }

    fn settle(&mut self, ex: &mut FileExplorer) {
        for _ in 0..3 {
            self.frame(ex, Vec::new());
        }
    }

    fn click(&mut self, ex: &mut FileExplorer, text: &str) {
        let at = text_pos(&self.shapes, text) + vec2(6.0, 6.0);
        self.frame(ex, press_release(at));
        self.settle(ex);
    }

    fn drawn(&self, text: &str) -> bool {
        texts(&self.shapes).iter().any(|(t, _)| t == text)
    }

    fn drawn_containing(&self, part: &str) -> bool {
        texts(&self.shapes).iter().any(|(t, _)| t.contains(part))
    }
}

#[test]
fn plurals_and_the_hidden_note() {
    assert_eq!(plural(1, "item"), "1 item");
    assert_eq!(plural(2, "filter"), "2 filters");
    assert_eq!(plural(0, "item"), "0 items");
    assert_eq!(hidden_note(30, 40, 2).as_deref(), Some("10 items hidden by 2 filters"));
    assert_eq!(hidden_note(39, 40, 1).as_deref(), Some("1 item hidden by 1 filter"));
    assert_eq!(hidden_note(0, 40, 3).as_deref(), Some("No items match the 3 active filters"));
    assert_eq!(hidden_note(0, 40, 1).as_deref(), Some("No items match the 1 active filter"));
    assert_eq!(hidden_note(40, 40, 2), None, "filters that hide nothing add no note");
    assert_eq!(hidden_note(0, 0, 2), None, "an empty tab needs no explanation");
    assert_eq!(hidden_note(30, 40, 0), None);
}

#[tokio::test]
async fn the_strip_shows_a_chip_for_the_default_filter_and_the_hidden_count() {
    let mut ex = explorer(FilterSpec::default());
    let mut screen = Screen::new(1000.0);
    screen.settle(&mut ex);
    assert!(screen.drawn("Size >= 10 KiB"), "{:?}", texts(&screen.shapes));
    assert!(screen.drawn("Filter") && screen.drawn("Views"));
    assert!(screen.drawn("10 items hidden by 1 filter"));
    assert!(screen.drawn("Clear"));
    assert_eq!(ex.view.len(), 30);
    assert_eq!(ex.table.len(), 40);
}

#[tokio::test]
async fn the_strip_without_filters_shows_only_its_two_buttons() {
    let mut ex = explorer(FilterSpec::none());
    let mut screen = Screen::new(1000.0);
    screen.settle(&mut ex);
    assert!(screen.drawn("Filter") && screen.drawn("Views"));
    assert!(!screen.drawn("Clear") && !screen.drawn_containing("hidden by"));
}

#[tokio::test]
async fn clicking_a_chip_removes_its_filter_and_lists_the_rows_again() {
    let mut ex = explorer(FilterSpec::default());
    let mut screen = Screen::new(1000.0);
    screen.settle(&mut ex);
    screen.click(&mut ex, "Size >= 10 KiB");
    assert!(ex.filter_spec().is_empty());
    assert!(!screen.drawn("Size >= 10 KiB") && !screen.drawn("Clear"));
    assert_eq!(ex.view.len(), 40);
    assert_eq!(ex.table.len(), 40);
}

#[tokio::test]
async fn the_clear_link_removes_every_filter() {
    let mut ex = explorer(busy());
    let mut screen = Screen::new(1000.0);
    screen.settle(&mut ex);
    assert!(screen.drawn("Clear"));
    screen.click(&mut ex, "Clear");
    assert!(ex.filter_spec().is_empty());
    assert_eq!(ex.view.len(), 40);
}

#[tokio::test]
async fn an_empty_result_says_which_filters_caused_it() {
    let mut spec = FilterSpec::none();
    spec.size.min = Some(1 << 40);
    spec.query = "img".into();
    let mut ex = explorer(spec);
    let mut screen = Screen::new(1000.0);
    screen.settle(&mut ex);
    assert_eq!(ex.view.len(), 0);
    assert!(screen.drawn("No items match the 2 active filters"));
}

#[tokio::test]
async fn the_strip_wraps_inside_the_window_at_every_width() {
    let mut heights = Vec::new();
    for width in [700.0f32, 1000.0, 1600.0] {
        let mut ex = explorer(busy());
        let mut screen = Screen::new(width);
        screen.settle(&mut ex);
        assert!(screen.strip.left() >= -0.5 && screen.strip.right() <= width + 0.5, "{width}: strip {:?}", screen.strip);
        for (text, at) in texts(&screen.shapes) {
            assert!(at.x >= 0.0 && at.x <= width, "{width}: {text:?} at {at:?}");
        }
        for label in ["Hiding: Videos", "Size >= 10 KiB", "Skip icons", "Not .gif", "Not matching: cache", "Has CLIP", "Tag: holiday", "Category: Travel", "Filter", "Views"] {
            assert!(screen.drawn(label), "{width}: no {label:?} in {:?}", texts(&screen.shapes));
        }
        assert!(screen.drawn_containing("hidden by 12 filters") || screen.drawn_containing("No items match"), "{width}: note missing");
        heights.push(screen.strip.height());
    }
    assert!(heights[0] > heights[2], "the narrow window wraps the strip onto more lines: {heights:?}");
    assert!(heights[0] >= heights[1] && heights[1] >= heights[2], "{heights:?}");
}

#[tokio::test]
async fn the_filter_button_opens_the_editor() {
    let mut ex = explorer(FilterSpec::none());
    let mut screen = Screen::new(1000.0);
    screen.settle(&mut ex);
    screen.click(&mut ex, "Filter");
    assert!(screen.drawn("Skip likely icons"), "{:?}", texts(&screen.shapes));
    assert!(screen.drawn("Make default") && screen.drawn("Clear all"));
}

#[tokio::test]
async fn the_views_menu_lists_saved_views_and_applies_one() {
    let mut ex = explorer(FilterSpec::none());
    let mut spec = FilterSpec::none();
    spec.size.min = Some(50_000);
    ex.filter_groups = vec![group_from_spec("Big files", &spec), group_from_spec("Anything", &FilterSpec::none())];
    let mut screen = Screen::new(1000.0);
    screen.settle(&mut ex);
    screen.click(&mut ex, "Views");
    assert!(screen.drawn("Saved views") && screen.drawn("Big files") && screen.drawn("Anything"), "{:?}", texts(&screen.shapes));
    assert!(screen.drawn("Save"));
    screen.click(&mut ex, "Big files");
    assert_eq!(ex.active_filter_group.as_deref(), Some("Big files"));
    assert_eq!(ex.filter_spec().size.min, Some(50_000));
    assert_eq!(ex.view.len(), 30);
    assert_eq!(ex.table.len(), 40);
    screen.settle(&mut ex);
    assert!(screen.drawn("View: Big files"));
}

#[tokio::test]
async fn an_empty_views_menu_says_so() {
    let mut ex = explorer(FilterSpec::none());
    let mut screen = Screen::new(1000.0);
    screen.settle(&mut ex);
    screen.click(&mut ex, "Views");
    assert!(screen.drawn("No saved views yet."), "{:?}", texts(&screen.shapes));
}
