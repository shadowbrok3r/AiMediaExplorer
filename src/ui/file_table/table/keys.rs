//! Keyboard commands of the details view.

use egui::{InputState, Key, Modifiers};

use crate::ui::file_table::rows::Step;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyAction {
    /// Arrow, Home, End, PageUp or PageDown; `extend` is set while Shift is held.
    Move { step: Step, extend: bool },
    SelectAll,
    /// Opens the focused row.
    Open,
    /// Shows or hides the preview pane.
    TogglePreview,
    CopyPath,
    ClearSelection,
}

/// Consumes the table's keys from `input`; `page_rows` is how many rows fit on screen.
pub fn read(input: &mut InputState, page_rows: usize) -> Vec<KeyAction> {
    let mut out = Vec::new();
    let page = page_rows.saturating_sub(1).max(1);
    if input.consume_key(Modifiers::COMMAND | Modifiers::SHIFT, Key::C) {
        out.push(KeyAction::CopyPath);
    }
    if input.consume_key(Modifiers::COMMAND, Key::A) {
        out.push(KeyAction::SelectAll);
    }
    for extend in [true, false] {
        let mods = if extend { Modifiers::SHIFT } else { Modifiers::NONE };
        for (key, step) in [(Key::ArrowDown, Step::Down(1)), (Key::ArrowUp, Step::Up(1)), (Key::PageDown, Step::Down(page)), (Key::PageUp, Step::Up(page))] {
            let presses = input.count_and_consume_key(mods, key);
            if presses > 0 {
                let step = match step {
                    Step::Down(n) => Step::Down(n * presses),
                    Step::Up(n) => Step::Up(n * presses),
                    other => other,
                };
                out.push(KeyAction::Move { step, extend });
            }
        }
        for (key, step) in [(Key::Home, Step::Home), (Key::End, Step::End)] {
            if input.consume_key(mods, key) {
                out.push(KeyAction::Move { step, extend });
            }
        }
    }
    if input.consume_key(Modifiers::NONE, Key::Enter) {
        out.push(KeyAction::Open);
    }
    if input.consume_key(Modifiers::NONE, Key::Space) {
        out.push(KeyAction::TogglePreview);
    }
    if input.consume_key(Modifiers::NONE, Key::Escape) {
        out.push(KeyAction::ClearSelection);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{Context, Event, RawInput};

    fn press(key: Key, modifiers: Modifiers) -> Event {
        Event::Key { key, physical_key: Some(key), pressed: true, repeat: false, modifiers }
    }

    fn actions(events: Vec<Event>, page_rows: usize) -> Vec<KeyAction> {
        let ctx = Context::default();
        let mut out = Vec::new();
        ctx.run_ui(RawInput { events, ..Default::default() }, |ui| {
            out = ui.input_mut(|i| read(i, page_rows));
        })
        .drop_without_applying_deltas();
        out
    }

    #[test]
    fn arrows_move_by_one_row_and_shift_extends() {
        assert_eq!(actions(vec![press(Key::ArrowDown, Modifiers::NONE)], 10), [KeyAction::Move { step: Step::Down(1), extend: false }]);
        assert_eq!(actions(vec![press(Key::ArrowUp, Modifiers::SHIFT)], 10), [KeyAction::Move { step: Step::Up(1), extend: true }]);
    }

    #[test]
    fn repeated_presses_in_one_frame_move_further() {
        let events = vec![press(Key::ArrowDown, Modifiers::NONE), press(Key::ArrowDown, Modifiers::NONE), press(Key::ArrowDown, Modifiers::NONE)];
        assert_eq!(actions(events, 10), [KeyAction::Move { step: Step::Down(3), extend: false }]);
    }

    #[test]
    fn page_keys_move_a_page_minus_one_row() {
        assert_eq!(actions(vec![press(Key::PageDown, Modifiers::NONE)], 10), [KeyAction::Move { step: Step::Down(9), extend: false }]);
        assert_eq!(actions(vec![press(Key::PageUp, Modifiers::SHIFT)], 10), [KeyAction::Move { step: Step::Up(9), extend: true }]);
        assert_eq!(actions(vec![press(Key::PageDown, Modifiers::NONE)], 0), [KeyAction::Move { step: Step::Down(1), extend: false }]);
    }

    #[test]
    fn home_and_end_jump_to_the_edges() {
        assert_eq!(actions(vec![press(Key::Home, Modifiers::NONE)], 10), [KeyAction::Move { step: Step::Home, extend: false }]);
        assert_eq!(actions(vec![press(Key::End, Modifiers::SHIFT)], 10), [KeyAction::Move { step: Step::End, extend: true }]);
    }

    #[test]
    fn ctrl_a_selects_all_and_ctrl_shift_c_copies_the_path() {
        assert_eq!(actions(vec![press(Key::A, Modifiers::COMMAND)], 10), [KeyAction::SelectAll]);
        assert_eq!(actions(vec![press(Key::C, Modifiers::COMMAND | Modifiers::SHIFT)], 10), [KeyAction::CopyPath]);
        assert!(actions(vec![press(Key::C, Modifiers::COMMAND)], 10).is_empty(), "plain copy is not a table command");
    }

    #[test]
    fn enter_space_and_escape_have_their_own_commands() {
        assert_eq!(actions(vec![press(Key::Enter, Modifiers::NONE)], 10), [KeyAction::Open]);
        assert_eq!(actions(vec![press(Key::Space, Modifiers::NONE)], 10), [KeyAction::TogglePreview]);
        assert_eq!(actions(vec![press(Key::Escape, Modifiers::NONE)], 10), [KeyAction::ClearSelection]);
    }

    #[test]
    fn editing_shortcuts_of_the_old_spreadsheet_table_are_not_commands() {
        for (key, mods) in [(Key::X, Modifiers::COMMAND), (Key::V, Modifiers::COMMAND), (Key::D, Modifiers::COMMAND), (Key::Delete, Modifiers::NONE), (Key::Backspace, Modifiers::NONE), (Key::Z, Modifiers::COMMAND), (Key::Y, Modifiers::COMMAND), (Key::Delete, Modifiers::COMMAND)] {
            assert!(actions(vec![press(key, mods)], 10).is_empty(), "{key:?} {mods:?}");
        }
    }

    #[test]
    fn keys_are_consumed_so_nothing_else_sees_them() {
        let ctx = Context::default();
        let mut left = true;
        ctx.run_ui(RawInput { events: vec![press(Key::ArrowDown, Modifiers::NONE)], ..Default::default() }, |ui| {
            ui.input_mut(|i| read(i, 10));
            left = ui.input(|i| i.key_pressed(Key::ArrowDown));
        })
        .drop_without_applying_deltas();
        assert!(!left);
    }
}
