use super::super::prefs::{DetailTab, MAX_WIDTH, MIN_WIDTH, RailTab};
use super::super::view::PreviewMode;
use super::*;

#[tokio::test(flavor = "multi_thread")]
async fn every_target_state_draws_at_every_width() {
    for width in [700.0, 1000.0, 1600.0] {
        let mut rig = Rig::new(width).await;
        rig.settle(3).await;
        assert!(rig.has_text("Select a file"), "{width}: empty state: {:?}", rig.texts());

        rig.select(&["a.png"]);
        rig.settle(8).await;
        for text in ["a.png", "Info", "Metadata", "AI", "Edit", "1 of 6", "Open", "Reveal", "Copy path"] {
            assert!(rig.has_text(text), "{width}: image state lacks {text:?}: {:?}", rig.texts());
        }
        assert!(!rig.has_text("Select a file"));
        assert!(rig.insp.preview.texture.is_some(), "{width}: the picture reached a texture ({})", rig.insp.preview.source);

        rig.select(&["clip.mp4"]);
        rig.settle(6).await;
        assert!(rig.has_text("clip.mp4"), "{width}: video state: {:?}", rig.texts());
        assert!(rig.has_text("Info"));

        rig.select(&["sub"]);
        rig.settle(4).await;
        assert!(rig.has_text("sub") && rig.has_text("Open") && rig.has_text("Reveal"), "{width}: folder state: {:?}", rig.texts());
        assert!(!rig.has_text("Metadata"), "{width}: a folder has no metadata tab");

        rig.select(&["a.png", "b.png", "c.png"]);
        rig.settle(6).await;
        for text in ["3 items selected", "Tags", "Category", "Collections", "AI"] {
            assert!(rig.has_text(text), "{width}: multi state lacks {text:?}: {:?}", rig.texts());
        }

        rig.ex.clear_selection();
        rig.settle(3).await;
        assert!(rig.has_text("Select a file"), "{width}: back to the empty state");

        let rect = rig.out.rect.expect("the rail drew");
        if width < 900.0 {
            assert!(rig.out.overlay, "{width}: narrow windows get the drawer");
            assert!((rect.right() - width).abs() < 1.5, "{width}: the drawer hugs the right edge: {rect:?}");
            assert!(rect.width() <= width * 0.95);
        } else {
            assert!(!rig.out.overlay, "{width}: wide windows dock the rail");
            assert!(rect.width() >= 360.0 && rect.width() <= 400.0, "{width}: docked width {}", rect.width());
            assert!((rect.right() - width).abs() < 1.5);
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_closed_rail_draws_nothing_and_looks_nothing_up() {
    let mut rig = Rig::new(1000.0).await;
    rig.select(&["a.png"]);
    rig.insp.set_open(false);
    rig.settle(6).await;
    assert!(rig.out.rect.is_none());
    assert!(!rig.has_text("Info"));
    assert_eq!(rig.insp.counters.lookups, 0, "a hidden rail does no background work");
    assert_eq!(rig.insp.meta.path(), None);
    rig.insp.set_open(true);
    rig.settle(6).await;
    assert!(rig.has_text("Info"));
    assert!(rig.insp.counters.lookups > 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_rail_follows_the_focus_not_the_last_clicked_row_of_a_multi_selection() {
    let mut rig = Rig::new(1000.0).await;
    rig.select(&["a.png", "b.png", "c.png"]);
    rig.settle(4).await;
    let focus = rig.ex.selection.preview_target().map(|k| k.as_str().to_string()).expect("a focus");
    let name = std::path::Path::new(&focus).file_name().unwrap().to_string_lossy().to_string();
    assert!(rig.has_text(&format!("Focus: {name}")), "{:?}", rig.texts());
    assert_eq!(rig.insp.last_target, "3 selected");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_picture_is_shown_from_the_thumbnail_and_reports_its_texture_cost() {
    let mut rig = Rig::new(1000.0).await;
    rig.select(&["b.png"]);
    rig.settle(10).await;
    let report = rig.insp.preview.clone();
    assert!(matches!(report.source, "thumbnail" | "preview"), "{report:?}");
    let [w, h] = report.texture.expect("a texture");
    assert!(w > 0 && h > 0);
    assert_eq!(report.texture_bytes(), w * h * 4);
    rig.select(&["notes.txt"]);
    rig.settle(6).await;
    assert!(matches!(rig.insp.preview.source, "missing" | "loading"), "{:?}", rig.insp.preview);
    assert!(rig.insp.preview.texture.is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn scrolling_over_the_preview_zooms_and_the_mode_buttons_reset_it() {
    let mut rig = Rig::new(1000.0).await;
    rig.select(&["a.png"]);
    rig.settle(10).await;
    let rect = rig.out.rect.unwrap();
    let over = rect.center_top() + vec2(0.0, 170.0);
    let wheel = Event::MouseWheel { unit: egui::MouseWheelUnit::Point, delta: vec2(0.0, 240.0), phase: egui::TouchPhase::Move, source: egui::MouseWheelSource::Wheel, modifiers: Modifiers::NONE };
    rig.frame_with(vec![Event::PointerMoved(over), wheel], Modifiers::NONE);
    rig.settle(2).await;
    assert!(rig.insp.view.zoom > 1.0, "zoom {}", rig.insp.view.zoom);
    rig.click_text("Fill");
    assert_eq!(rig.insp.view.zoom, 1.0);
    assert_eq!(rig.insp.view.mode, PreviewMode::Fill);
    assert_eq!(rig.insp.prefs.prefs.preview_mode, PreviewMode::Fill, "the mode is remembered");
}

#[tokio::test(flavor = "multi_thread")]
async fn dragging_the_edge_resizes_the_rail_and_keeps_it_in_range() {
    let mut rig = Rig::new(1200.0).await;
    rig.select(&["a.png"]);
    rig.settle(6).await;
    let before = rig.insp.prefs.prefs.width;
    let edge = rig.out.rect.unwrap().left_top() + vec2(0.0, 300.0);
    let down = Event::PointerButton { pos: edge, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE };
    let up = Event::PointerButton { pos: edge - vec2(80.0, 0.0), button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE };
    rig.frame_with(vec![Event::PointerMoved(edge), down], Modifiers::NONE);
    for step in 1..=4 {
        rig.frame_with(vec![Event::PointerMoved(edge - vec2(20.0 * step as f32, 0.0))], Modifiers::NONE);
    }
    rig.frame_with(vec![up], Modifiers::NONE);
    rig.settle(2).await;
    let after = rig.insp.prefs.prefs.width;
    assert!(after > before + 40.0, "dragging left widens the rail: {before} -> {after}");
    for _ in 0..40 {
        let edge = rig.out.rect.unwrap().left_top() + vec2(0.0, 300.0);
        let down = Event::PointerButton { pos: edge, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE };
        rig.frame_with(vec![Event::PointerMoved(edge), down], Modifiers::NONE);
        rig.frame_with(vec![Event::PointerMoved(edge - vec2(300.0, 0.0))], Modifiers::NONE);
        rig.frame_with(vec![Event::PointerButton { pos: edge, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE }], Modifiers::NONE);
        if rig.insp.prefs.prefs.width >= MAX_WIDTH {
            break;
        }
    }
    assert!(rig.insp.prefs.prefs.width <= MAX_WIDTH);
    assert!(rig.insp.prefs.is_dirty());
    assert!(rig.insp.prefs.save_now());
    assert_eq!(super::super::prefs::load(rig.insp.prefs.path().unwrap()).prefs.width, rig.insp.prefs.prefs.width);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_ask_tab_is_a_placeholder_that_opens_the_assistant() {
    let mut rig = Rig::new(1000.0).await;
    rig.select(&["a.png"]);
    rig.settle(4).await;
    rig.click_text("Ask");
    assert_eq!(rig.insp.prefs.prefs.rail_tab, RailTab::Ask);
    rig.settle(2).await;
    assert!(rig.has_text("Open the assistant"), "{:?}", rig.texts());
    assert!(!rig.assistant);
    rig.click_text("Open the assistant");
    assert!(rig.assistant, "the placeholder opens the assistant window");
    rig.click_text("Inspect");
    assert!(rig.has_text("Info"));
}

#[tokio::test(flavor = "multi_thread")]
async fn the_info_tab_lists_the_facts_of_the_file_and_flags_a_missing_one() {
    let mut rig = Rig::new(1200.0).await;
    rig.height = 1200.0;
    rig.select(&["a.png"]);
    let path = rig.path("a.png");
    rig.until("the file facts", |r| r.insp.facts.get(&path).is_some()).await;
    rig.settle(2).await;
    for text in ["Dimensions", "40 x 30 px", "Size", "Library", "CLIP", "Description", "Collections"] {
        assert!(rig.has_text(text), "info lacks {text:?}: {:?}", rig.texts());
    }
    assert!(!rig.has_text("Offline"));
    std::fs::remove_file(rig.path("a.png")).unwrap();
    rig.insp.facts.invalidate(&path);
    rig.until("the offline badge", |r| r.has_text("Offline")).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn no_state_or_tab_pushes_the_rail_past_the_window_edge() {
    for (width, rail) in [(1200.0, 360.0), (1000.0, MIN_WIDTH), (1600.0, MAX_WIDTH), (700.0, 360.0)] {
        for tab in DetailTab::ALL {
            let mut rig = Rig::new(width).await;
            rig.height = 1100.0;
            rig.insp.edit_prefs(|p| p.width = rail);
            rig.open_tab(tab);
            for names in [&["a.png"][..], &["clip.mp4"], &["sub"], &["a.png", "b.png", "c.png"]] {
                rig.select(names);
                rig.settle(6).await;
                let rect = rig.out.rect.expect("the rail drew");
                assert!((rect.right() - width).abs() < 1.5 || rig.out.overlay, "{width}/{rail} {tab:?} {names:?}: rail at {rect:?}");
                assert!(rect.left() >= -0.5 && rect.right() <= width + 1.5, "{width}/{rail} {tab:?} {names:?}: rail at {rect:?}");
            }
        }
    }
}
