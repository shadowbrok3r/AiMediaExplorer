use eframe::egui::*;

use super::*;
use crate::command::QueryMode;
use crate::ui::file_table::test_rig::{Rig, ROWS, path_of, text_rects};

fn sizes_fixture() -> Sizes {
    Sizes { side: 24.0, icon_w: 30.0, gap: 8.0, scan_text: 62.0, process_text: 90.0, mode_labels: 330.0, mode_icons: 150.0 }
}

#[test]
fn a_wider_row_never_shows_less() {
    let s = sizes_fixture();
    let rank = |p: &Plan| (p.nav, matches!(p.mode, ModeStyle::Labels) as u8 * 2 + matches!(p.mode, ModeStyle::Icons) as u8, p.text_w as i64, p.scan as u8, p.process as u8);
    let mut last = (0, 0, 0, 0, 0);
    for width in (300..2400).step_by(10) {
        let p = plan_for(width as f32, &s);
        let r = rank(&p);
        assert!(r.0 >= last.0 && r.1 >= last.1, "{width}px shows less than a narrower row: {p:?}");
        last = r;
    }
}

#[test]
fn every_plan_fits_its_row_except_the_tightest_below_its_minimum() {
    let s = sizes_fixture();
    let tight = plan_for(0.0, &s);
    for width in (300..2400).step_by(7) {
        let p = plan_for(width as f32, &s);
        assert!(p.needs(&s) <= width as f32 || p == tight, "{width}px: {p:?} needs {}", p.needs(&s));
    }
    assert!(tight.needs(&s) <= 520.0, "the tightest plan fits half a screen: {}", tight.needs(&s));
}

#[test]
fn the_richest_plan_has_labels_and_text_buttons_and_the_poorest_only_the_menu() {
    let s = sizes_fixture();
    let wide = plan_for(1600.0, &s);
    assert_eq!((wide.nav, wide.mode, wide.scan, wide.process), (5, ModeStyle::Labels, Btn::Text, Btn::Text));
    let narrow = plan_for(1.0, &s);
    assert_eq!((narrow.mode, narrow.scan, narrow.process), (ModeStyle::Menu, Btn::Menu, Btn::Menu));
    assert!(narrow.nav < 5);
}

fn assert_clean(layout: &ToolbarLayout, width: f32) {
    let parts = layout.parts();
    for (name, rect) in &parts {
        assert!(rect.width() > 1.0 && rect.height() > 1.0, "{name} has no size at {width}px: {rect:?}");
        assert!(rect.left() >= layout.row.left() - 0.5 && rect.right() <= layout.row.right() + 0.5, "{name} leaves the row at {width}px: {rect:?} in {:?}", layout.row);
    }
    for (i, (a, ra)) in parts.iter().enumerate() {
        for (b, rb) in parts.iter().skip(i + 1) {
            let inter = ra.intersect(*rb);
            assert!(inter.width() <= 0.5 || inter.height() <= 0.5, "{a} and {b} overlap at {width}px: {ra:?} {rb:?}");
        }
    }
}

#[test]
fn the_row_never_overlaps_or_clips_at_700_1000_and_1600_px() {
    for width in [700.0, 1000.0, 1600.0] {
        let mut rig = Rig::new(width, 500.0);
        let layout = rig.ex.toolbar.layout.clone().expect("the row was drawn");
        assert_clean(&layout, width);
        assert!(layout.path.width() >= 120.0, "{width}px: the path area is {} wide", layout.path.width());
        assert!(layout.text.width() >= 80.0, "{width}px: the query field is {} wide", layout.text.width());
        let hint = rig.ex.toolbar.query.mode.hint();
        let shown = text_rects(&rig.shapes).into_iter().find(|(t, _)| t == hint);
        if let Some((_, r)) = shown {
            assert!(r.right() <= layout.text.right() + 1.0, "{width}px: the hint runs out of the field");
        }
    }
}

#[test]
fn the_row_stays_clean_across_every_width_from_540_to_2000() {
    for width in (540..=2000).step_by(40) {
        let rig = Rig::new(width as f32, 400.0);
        assert_clean(&rig.ex.toolbar.layout.clone().unwrap(), width as f32);
    }
}

#[test]
fn the_row_folds_into_the_menu_as_the_window_narrows() {
    let wide = Rig::new(1600.0, 500.0).ex.toolbar.layout.clone().unwrap();
    let narrow = Rig::new(640.0, 500.0).ex.toolbar.layout.clone().unwrap();
    assert_eq!(wide.nav.len(), 5);
    assert_eq!(wide.actions.len(), 2, "Scan and Process are buttons when there is room");
    assert!(narrow.nav.len() < 5 && narrow.actions.len() < 2);
    assert!(wide.mode.width() > narrow.mode.width(), "the mode control shrinks to an icon");
}

#[test]
fn the_back_button_follows_the_history_and_goes_back() {
    let dir = tempfile::tempdir().unwrap();
    let other = dir.path().display().to_string();
    let mut rig = Rig::new(1000.0, 500.0);
    let start = rig.ex.scope().clone();
    let back = |rig: &Rig| rig.ex.toolbar.layout.as_ref().unwrap().nav.iter().find(|(n, _)| *n == "back").unwrap().1;
    assert!(!rig.ex.can_go_back());
    rig.ex.open_scope(crate::command::Scope::Folder(other.clone())).unwrap();
    rig.settle(3);
    assert!(rig.ex.can_go_back());
    rig.click(back(&rig).center());
    assert_eq!(rig.ex.scope(), &start);
    assert!(rig.ex.can_go_forward());
}

#[test]
fn the_scan_and_process_buttons_open_their_dialogs() {
    let mut rig = Rig::new(1600.0, 500.0);
    let layout = rig.ex.toolbar.layout.clone().unwrap();
    let rect = |name: &str| layout.actions.iter().find(|(n, _)| *n == name).unwrap().1;
    rig.click(rect("scan").center());
    assert!(rig.ex.toolbar.scan.open);
    assert!(rig.has_text("Scan folder"), "the dialog is drawn");
    rig.ex.toolbar.scan.open = false;
    rig.click(rect("process").center());
    assert!(rig.ex.toolbar.process.open);
}

#[test]
fn the_more_menu_lists_the_address_actions_and_the_folded_ones() {
    let mut rig = Rig::new(440.0, 500.0);
    let layout = rig.ex.toolbar.layout.clone().unwrap();
    rig.click(layout.more.center());
    for text in ["Scan folder...", "Process files...", "Scan this folder...", "Copy address", "Open in the file manager"] {
        assert!(rig.has_text(text), "{text} is missing from the More menu");
    }
    rig.click(rig.text_pos("Copy address").unwrap() + vec2(8.0, 8.0));
    assert_eq!(rig.copied, [rig.ex.current_path.clone()]);
}

fn focus_query(rig: &mut Rig) {
    rig.ex.toolbar.query.focus = true;
    rig.settle(3);
}

#[test]
fn typing_in_name_mode_filters_without_removing_rows() {
    let mut rig = Rig::new(1000.0, 500.0);
    focus_query(&mut rig);
    rig.type_text("img_03");
    assert_eq!(rig.ex.filter_spec().name, "img_03");
    assert_eq!(rig.ex.view.len(), 1);
    assert_eq!(rig.ex.table.len(), ROWS + 1, "rows stay in the tab");
    assert!(rig.ex.filter_spec().chips().iter().any(|c| c.label.contains("img_03")), "the filter shows as a chip");
}

#[test]
fn text_mode_filters_the_search_text_instead_of_the_name() {
    let mut rig = Rig::new(1000.0, 500.0);
    rig.ex.table.get_mut_by_path(&path_of(4)).unwrap().tags = vec!["sunset".into()];
    rig.ex.toolbar.query.mode = QueryMode::Text;
    focus_query(&mut rig);
    rig.type_text("sunset");
    let spec = rig.ex.filter_spec();
    assert_eq!((spec.name.as_str(), spec.query.as_str()), ("", "sunset"));
    assert_eq!(rig.ex.view.len(), 1);
}

#[test]
fn switching_the_mode_moves_the_text_between_the_conditions() {
    let mut rig = Rig::new(1000.0, 500.0);
    focus_query(&mut rig);
    rig.type_text("img_03");
    let seg = rig.ex.toolbar.layout.clone().unwrap().mode;
    rig.click(pos2(seg.left() + seg.width() * 0.37, seg.center().y));
    assert_eq!(rig.ex.toolbar.query.mode, QueryMode::Text);
    let spec = rig.ex.filter_spec();
    assert_eq!((spec.name.as_str(), spec.query.as_str()), ("", "img_03"));
    let seg = rig.ex.toolbar.layout.clone().unwrap().mode;
    rig.click(pos2(seg.left() + seg.width() * 0.62, seg.center().y));
    assert_eq!(rig.ex.toolbar.query.mode, QueryMode::Semantic);
    let spec = rig.ex.filter_spec();
    assert_eq!((spec.name.as_str(), spec.query.as_str()), ("", ""), "a search mode lists everything until Enter");
    assert_eq!(rig.ex.view.len(), ROWS + 1);
    assert_eq!(rig.ex.toolbar.query.text, "img_03", "the text is kept for the search");
}

#[test]
fn a_chip_removed_elsewhere_empties_the_bar() {
    let mut rig = Rig::new(1000.0, 500.0);
    focus_query(&mut rig);
    rig.type_text("img_03");
    rig.ex.clear_filters();
    rig.settle(3);
    assert_eq!(rig.ex.toolbar.query.text, "");
}

#[test]
fn escape_clears_the_query_first_and_then_leaves_the_field() {
    let mut rig = Rig::new(1000.0, 500.0);
    rig.select_images(3..4);
    focus_query(&mut rig);
    rig.type_text("img_03");
    rig.key(Key::Escape, Modifiers::NONE);
    assert_eq!(rig.ex.toolbar.query.text, "");
    assert_eq!(rig.ex.filter_spec().name, "");
    assert_eq!(rig.ex.selection_count(), 1, "Esc in the field leaves the selection alone");
    let id = crate::ui::file_table::query::text_id(&rig.ex);
    assert!(rig.ctx.memory(|m| m.has_focus(id)), "the field keeps the focus after clearing");
    rig.key(Key::Escape, Modifiers::NONE);
    assert!(!rig.ctx.memory(|m| m.has_focus(id)));
}

#[test]
fn enter_in_semantic_mode_opens_the_search_scope_and_in_similar_mode_the_similar_scope() {
    let mut rig = Rig::new(1000.0, 500.0);
    rig.ex.toolbar.query.mode = QueryMode::Semantic;
    focus_query(&mut rig);
    rig.type_text("a red car");
    rig.key(Key::Enter, Modifiers::NONE);
    assert_eq!(rig.ex.scope(), &crate::command::Scope::Semantic("a red car".into()));
    assert_eq!(rig.ex.toolbar.query.history.back().unwrap().plan, "open q:sem:\"a red car\"");
    rig.ex.toolbar.query.mode = QueryMode::Similar;
    rig.ex.toolbar.query.text = path_of(3);
    focus_query(&mut rig);
    rig.key(Key::Enter, Modifiers::NONE);
    assert_eq!(rig.ex.scope(), &crate::command::Scope::Similar(path_of(3)));
}

#[test]
fn similar_with_an_empty_field_uses_the_focused_file() {
    let mut rig = Rig::new(1000.0, 500.0);
    rig.select_images(5..6);
    rig.ex.toolbar.query.mode = QueryMode::Similar;
    rig.ex.toolbar.query.text.clear();
    let done = rig.ex.run_query().unwrap();
    assert!(done.contains(&path_of(5)), "{done}");
    assert_eq!(rig.ex.scope(), &crate::command::Scope::Similar(path_of(5)));
}

#[test]
fn an_empty_semantic_query_is_refused_and_logged() {
    let mut rig = Rig::new(1000.0, 500.0);
    rig.ex.toolbar.query.mode = QueryMode::Semantic;
    let err = rig.ex.run_query().unwrap_err();
    assert_eq!(err.kind, crate::command::ErrorKind::Invalid);
    assert!(rig.ex.toolbar.query.history.back().unwrap().outcome.is_err());
}

#[test]
fn the_query_command_sets_the_bar_and_runs_it() {
    let mut rig = Rig::new(1000.0, 500.0);
    let text = rig.ex.apply_query_command(QueryMode::Name, "img_07", false).unwrap();
    assert!(text.contains("img_07"));
    rig.settle(2);
    assert_eq!(rig.ex.view.len(), 1);
    assert_eq!(rig.ex.toolbar.query.text, "img_07");
    rig.ex.apply_query_command(QueryMode::Semantic, "sunset", true).unwrap();
    assert_eq!(rig.ex.scope(), &crate::command::Scope::Semantic("sunset".into()));
    assert_eq!(rig.ex.filter_spec().name, "", "starting a search leaves no live filter behind");
}

#[test]
fn a_scope_change_puts_the_search_text_in_the_bar() {
    let mut rig = Rig::new(1000.0, 500.0);
    rig.ex.open_scope(crate::command::Scope::Semantic("moon".into())).unwrap();
    rig.settle(3);
    assert_eq!((rig.ex.toolbar.query.mode, rig.ex.toolbar.query.text.as_str()), (QueryMode::Semantic, "moon"));
    rig.ex.open_scope(crate::command::Scope::Library).unwrap();
    rig.settle(3);
    assert_eq!((rig.ex.toolbar.query.mode, rig.ex.toolbar.query.text.as_str()), (QueryMode::Name, ""));
}
