use std::sync::Mutex;

use eframe::egui::{Context, Pos2, RawInput, Rect, vec2};

use super::*;
use crate::database::journal::{BatchId, Value};
use crate::ui::test_support::{test_ctx, texts};

/// Serializes the tests that use the process-wide panel state.
static PANEL_LOCK: Mutex<()> = Mutex::new(());

fn entry(seq: i64, path: &str, before: &str, after: &str, state: EntryState) -> Entry {
    Entry { seq, batch: BatchId(seq), at_ms: 1_700_000_000_000, source: "Edit metadata".into(), path: path.into(), field: Field::Caption, target: None, before: Value::text(Some(before)), after: Value::text(Some(after)), state }
}

fn drawn(ctx: &Context) -> Vec<String> {
    let mut last = Vec::new();
    for _ in 0..3 {
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1000.0, 900.0))), ..Default::default() };
        let out = ctx.run_ui(input, |ui| show(ui.ctx()));
        last = texts(&out.shapes).into_iter().map(|(t, _)| t).collect();
        out.drop_without_applying_deltas();
    }
    last
}

#[test]
fn the_form_builds_the_patch_of_its_field() {
    assert_eq!(form_patch(Field::Caption, "  a caption ").unwrap().caption, Some(Some("a caption".to_string())));
    assert_eq!(form_patch(Field::Category, "   ").unwrap().category, Some(None), "an empty value clears the field");
    assert_eq!(form_patch(Field::Tags, "a, b ,,c").unwrap().tags, Some(vec!["a".to_string(), " b ".to_string(), String::new(), "c".to_string()]));
    assert!(form_patch(Field::Collection, "x").is_none());
}

#[test]
fn the_seed_of_a_path_names_the_file() {
    let seed = RowSeed::from_path("/photos/Trip/IMG_1.JPG");
    assert_eq!((seed.filename.as_str(), seed.file_type.as_str(), seed.parent_dir.as_str()), ("IMG_1.JPG", "jpg", "/photos/Trip"));
}

#[test]
fn the_window_is_closed_until_opened_and_lists_the_journal() {
    let _serial = PANEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = test_ctx();
    with(|p| *p = Panel::default());
    assert!(drawn(&ctx).is_empty(), "a closed window draws nothing");
    set_open(&ctx, true);
    let entries = vec![entry(2, "/p/b.jpg", "old", "new", EntryState::Applied), entry(1, "/p/a.jpg", "x", "y", EntryState::Skipped)];
    with(|p| {
        p.listing = Listing { batches: journal::summarize(&entries, 10), entries, stats: Stats { entries: 2, batches: 2, applied: 1, undone: 0, skipped: 1 } };
        p.loaded_at = Some(journal::revision());
    });
    let text = drawn(&ctx);
    for expected in ["Devtools: Journal", "Undo", "Redo", "Clear", "Refresh", "b.jpg", "a.jpg", "old", "new", "skipped", "caption"] {
        assert!(text.iter().any(|t| t.contains(expected)), "missing {expected:?} in {text:?}");
    }
    assert!(text.iter().any(|t| t.contains("2 entries in 2 batches")), "{text:?}");
    toggle(&ctx);
    assert!(!is_open(&ctx));
}

#[test]
fn the_window_says_when_the_database_is_not_connected() {
    let _serial = PANEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = test_ctx();
    with(|p| *p = Panel::default());
    set_open(&ctx, true);
    let text = drawn(&ctx);
    assert!(text.iter().any(|t| t.contains("not connected")), "{text:?}");
}
