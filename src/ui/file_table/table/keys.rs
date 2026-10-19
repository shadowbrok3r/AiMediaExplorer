//! Keyboard commands of the details view.

use egui::{InputState, Key, Modifiers};

use super::grid_layout::Motion;
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

/// A key command of the grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridKey {
    /// Arrow, PageUp or PageDown; `extend` is set while Shift is held.
    Move { motion: Motion, extend: bool },
    /// Plus or minus: the tile size grows or shrinks by one step.
    Zoom(i32),
    /// A command shared with the details view.
    Common(KeyAction),
}

/// Consumes the grid's keys from `input`; `page_rows` is how many tile rows fit on screen.
pub fn read_grid(input: &mut InputState, page_rows: usize) -> Vec<GridKey> {
    let mut out = Vec::new();
    for extend in [true, false] {
        let mods = if extend { Modifiers::SHIFT } else { Modifiers::NONE };
        for (key, motion) in [
            (Key::ArrowLeft, Motion::Left),
            (Key::ArrowRight, Motion::Right),
            (Key::ArrowUp, Motion::Up),
            (Key::ArrowDown, Motion::Down),
            (Key::PageUp, Motion::PageUp),
            (Key::PageDown, Motion::PageDown),
        ] {
            for _ in 0..input.count_and_consume_key(mods, key) {
                out.push(GridKey::Move { motion, extend });
            }
        }
    }
    for mods in [Modifiers::SHIFT, Modifiers::NONE] {
        for (key, step) in [(Key::Plus, 1), (Key::Equals, 1), (Key::Minus, -1)] {
            for _ in 0..input.count_and_consume_key(mods, key) {
                out.push(GridKey::Zoom(step));
            }
        }
    }
    out.extend(read(input, page_rows).into_iter().map(GridKey::Common));
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

    fn grid_keys(events: Vec<Event>) -> Vec<GridKey> {
        let ctx = Context::default();
        let mut out = Vec::new();
        ctx.run_ui(RawInput { events, ..Default::default() }, |ui| {
            out = ui.input_mut(|i| read_grid(i, 10));
        })
        .drop_without_applying_deltas();
        out
    }

    #[test]
    fn grid_arrows_move_by_direction_and_shift_extends() {
        assert_eq!(grid_keys(vec![press(Key::ArrowLeft, Modifiers::NONE)]), [GridKey::Move { motion: Motion::Left, extend: false }]);
        assert_eq!(grid_keys(vec![press(Key::ArrowDown, Modifiers::SHIFT)]), [GridKey::Move { motion: Motion::Down, extend: true }]);
        assert_eq!(grid_keys(vec![press(Key::PageUp, Modifiers::NONE)]), [GridKey::Move { motion: Motion::PageUp, extend: false }]);
    }

    #[test]
    fn grid_repeated_presses_are_separate_moves() {
        let events = vec![press(Key::ArrowRight, Modifiers::NONE), press(Key::ArrowRight, Modifiers::NONE)];
        assert_eq!(grid_keys(events).len(), 2);
    }

    #[test]
    fn grid_keeps_the_shared_commands() {
        let keys = grid_keys(vec![press(Key::A, Modifiers::COMMAND), press(Key::Enter, Modifiers::NONE), press(Key::Space, Modifiers::NONE), press(Key::Escape, Modifiers::NONE), press(Key::Home, Modifiers::NONE), press(Key::End, Modifiers::SHIFT)]);
        assert!(keys.contains(&GridKey::Common(KeyAction::SelectAll)));
        assert!(keys.contains(&GridKey::Common(KeyAction::Open)));
        assert!(keys.contains(&GridKey::Common(KeyAction::TogglePreview)));
        assert!(keys.contains(&GridKey::Common(KeyAction::ClearSelection)));
        assert!(keys.contains(&GridKey::Common(KeyAction::Move { step: Step::Home, extend: false })));
        assert!(keys.contains(&GridKey::Common(KeyAction::Move { step: Step::End, extend: true })));
        assert!(keys.iter().all(|k| !matches!(k, GridKey::Common(KeyAction::Move { step: Step::Up(_) | Step::Down(_), .. }))));
    }

    #[test]
    fn plus_and_minus_resize_the_tiles() {
        assert_eq!(grid_keys(vec![press(Key::Plus, Modifiers::NONE)]), [GridKey::Zoom(1)]);
        assert_eq!(grid_keys(vec![press(Key::Equals, Modifiers::NONE)]), [GridKey::Zoom(1)]);
        assert_eq!(grid_keys(vec![press(Key::Plus, Modifiers::SHIFT)]), [GridKey::Zoom(1)]);
        assert_eq!(grid_keys(vec![press(Key::Minus, Modifiers::NONE)]), [GridKey::Zoom(-1)]);
        assert!(grid_keys(vec![press(Key::Minus, Modifiers::COMMAND)]).is_empty(), "Ctrl+minus zooms the whole window");
    }

    #[test]
    fn grid_keys_are_consumed() {
        let ctx = Context::default();
        let mut left = true;
        ctx.run_ui(RawInput { events: vec![press(Key::ArrowLeft, Modifiers::NONE)], ..Default::default() }, |ui| {
            ui.input_mut(|i| read_grid(i, 10));
            left = ui.input(|i| i.key_pressed(Key::ArrowLeft));
        })
        .drop_without_applying_deltas();
        assert!(!left);
    }
}
