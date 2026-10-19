use eframe::egui::{CentralPanel, Context, Event, Id, Key, Modifiers, Pos2, RawInput, Rect, TextEdit, Ui, vec2};

use super::*;
use crate::database::journal::{BatchId, Change, Conflict, Field, Value};
use crate::ui::test_support::test_ctx;

fn key_events(key: Key, modifiers: Modifiers) -> Vec<Event> {
    vec![Event::ModifiersChanged(modifiers), Event::Key { key, physical_key: Some(key), pressed: true, repeat: false, modifiers }]
}

fn frame(ctx: &Context, events: Vec<Event>, mut body: impl FnMut(&mut Ui)) {
    let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(400.0, 300.0))), events, ..Default::default() };
    let mut out = ctx.run_ui(input, &mut body);
    out.textures_delta.clear();
}

fn pressed(ctx: &Context, key: Key, modifiers: Modifiers) -> Option<AppCommand> {
    let mut got = None;
    frame(ctx, key_events(key, modifiers), |ui| got = read_keys(ui.ctx()));
    got
}

#[test]
fn ctrl_z_undoes_and_ctrl_shift_z_redoes() {
    let ctx = test_ctx();
    assert!(matches!(pressed(&ctx, Key::Z, Modifiers::COMMAND), Some(AppCommand::Undo)));
    assert!(matches!(pressed(&ctx, Key::Z, Modifiers::COMMAND | Modifiers::SHIFT), Some(AppCommand::Redo)));
    assert!(pressed(&ctx, Key::Z, Modifiers::NONE).is_none(), "a bare Z is just a letter");
    assert!(pressed(&ctx, Key::Y, Modifiers::COMMAND).is_none());
    assert!(pressed(&ctx, Key::Z, Modifiers::ALT).is_none());
}

#[test]
fn a_focused_text_field_keeps_its_own_undo() {
    let ctx = test_ctx();
    let mut text = String::from("hello");
    let id = Id::new("journal_keys_text");
    frame(&ctx, Vec::new(), |ui| {
        CentralPanel::default().show(ui, |ui| {
            ui.add(TextEdit::singleline(&mut text).id(id)).request_focus();
        });
    });
    let mut got = Some(AppCommand::ClearSelection);
    frame(&ctx, key_events(Key::Z, Modifiers::COMMAND), |ui| {
        CentralPanel::default().show(ui, |ui| {
            ui.add(TextEdit::singleline(&mut text).id(id));
        });
        got = read_keys(ui.ctx());
    });
    assert!(got.is_none(), "no command while a text field has the keyboard");
}

fn report(applied: usize, conflicts: usize) -> UndoReport {
    UndoReport {
        direction: Direction::Undo,
        batch: Some(BatchId(1)),
        source: "Add tags".into(),
        applied: (0..applied).map(|i| Change { path: format!("/p/{i}.jpg"), field: Field::Tags, target: None, value: Value::Tags(vec!["x".into()]) }).collect(),
        conflicts: (0..conflicts).map(|i| Conflict { path: format!("/q/{i}.jpg"), field: Field::Tags, target: None, reason: "changed".into() }).collect(),
    }
}

#[test]
fn the_toast_level_tells_success_from_skips_and_nothing() {
    assert_eq!(toast_level(&report(3, 0)), ToastLevel::Success);
    assert_eq!(toast_level(&report(3, 1)), ToastLevel::Warning);
    assert_eq!(toast_level(&UndoReport { batch: None, ..report(0, 0) }), ToastLevel::Info);
}

#[test]
fn a_report_becomes_one_row_edit_per_file() {
    let edits = row_edits(&report(3, 0));
    assert_eq!(edits.len(), 3);
    assert!(matches!(&edits[1], RowEdit::Patch { path, patch } if path == "/p/1.jpg" && patch.tags == Some(vec!["x".to_string()])));
    assert!(row_edits(&report(0, 2)).is_empty(), "skipped entries change no row");
}

#[test]
fn queued_edits_and_reports_are_taken_once() {
    let _serial = crate::command::bus::testing::serial();
    take_edits();
    take_results();
    push_edits(Vec::new());
    assert!(take_edits().is_empty());
    push_edits(row_edits(&report(2, 0)));
    push_result(report(1, 0));
    assert_eq!(take_edits().len(), 2);
    assert!(take_edits().is_empty());
    assert_eq!(take_results().len(), 1);
    assert!(take_results().is_empty());
}

#[tokio::test]
async fn undo_without_a_connected_database_fails_with_a_reason() {
    if crate::database::is_ready() {
        return;
    }
    let err = start(Direction::Undo).unwrap_err();
    assert_eq!(err.kind, crate::command::ErrorKind::Failed);
    assert!(err.message.contains("not connected"));
    assert!(start(Direction::Redo).is_err());
}
