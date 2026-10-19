//! The action registry: every action of the browser with its labels, icon, shortcut and the rules for when it can run on a selection.

use super::model::ActionId;
use crate::icons::{Icon, svg};

/// Operations on more files than this ask for confirmation with the exact count.
pub const BULK_CONFIRM_ABOVE: usize = 25;
/// Most items Open and Reveal act on at once.
pub const OPEN_LIMIT: usize = 20;

/// Where an action sits in menus: a separator is drawn between groups.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Group {
    Open,
    Library,
    Organize,
    Ai,
    Files,
}

impl Group {
    pub const ALL: [Group; 5] = [Self::Open, Self::Library, Self::Organize, Self::Ai, Self::Files];

    pub fn label(self) -> &'static str {
        match self {
            Self::Open => "Open",
            Self::Library => "Library",
            Self::Organize => "Organize",
            Self::Ai => "AI",
            Self::Files => "Files",
        }
    }
}

/// When an action asks before it runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Confirm {
    Never,
    /// Above [`BULK_CONFIRM_ABOVE`] files.
    Bulk,
    Always,
}

/// What an action needs from the selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Needs {
    Rows,
    Files,
    Images,
    LocalFiles,
    Folders,
}

impl Needs {
    pub fn label(self) -> &'static str {
        match self {
            Self::Rows => "a selected row",
            Self::Files => "a selected file",
            Self::Images => "a selected image",
            Self::LocalFiles => "a selected file on disk",
            Self::Folders => "a selected folder",
        }
    }
}

/// One action as menus and the action bar show it.
#[derive(Clone, Copy)]
pub struct ActionSpec {
    pub id: ActionId,
    /// Id of the context menu item; the first eleven are the ids of the earlier table.
    pub menu_id: &'static str,
    /// Text in the context menu.
    pub label: &'static str,
    /// Text on the action bar button.
    pub short: &'static str,
    pub icon: Icon,
    pub shortcut: Option<&'static str>,
    pub group: Group,
    /// Lower stays on the action bar longer when it runs out of room.
    pub priority: u8,
    pub confirm: Confirm,
    pub in_bar: bool,
    pub needs: Needs,
}

const fn spec(id: ActionId, menu_id: &'static str, label: &'static str, short: &'static str, icon: Icon, group: Group, priority: u8, needs: Needs) -> ActionSpec {
    ActionSpec { id, menu_id, label, short, icon, shortcut: None, group, priority, confirm: Confirm::Never, in_bar: true, needs }
}

impl ActionSpec {
    const fn key(mut self, shortcut: &'static str) -> Self {
        self.shortcut = Some(shortcut);
        self
    }

    const fn confirm(mut self, confirm: Confirm) -> Self {
        self.confirm = confirm;
        self
    }

    const fn menu_only(mut self) -> Self {
        self.in_bar = false;
        self
    }
}

/// Every action in menu order.
pub const SPECS: [ActionSpec; 20] = [
    spec(ActionId::Open, "open", "Open", "Open", svg::ARROW_SQUARE_OUT, Group::Open, 6, Needs::Rows).key("Enter"),
    spec(ActionId::Reveal, "reveal", "Show in folder", "Reveal", svg::AME_REVEAL_IN_EXPLORER, Group::Open, 7, Needs::Rows),
    spec(ActionId::CopyPath, "copy_path", "Copy path", "Copy path", svg::CLIPBOARD_TEXT, Group::Open, 5, Needs::Rows).key("Ctrl+Shift+C"),
    spec(ActionId::OpenInNewTab, "open_selection_in_new_tab", "Open selection in new tab", "New tab", svg::AME_NEW_TAB, Group::Open, 9, Needs::Rows).menu_only(),
    spec(ActionId::AddToLibrary, "add_to_library", "Add to library", "Add to library", svg::AME_DATABASE, Group::Library, 2, Needs::Files).confirm(Confirm::Bulk),
    spec(ActionId::RemoveFromLibrary, "remove_from_library", "Remove from library", "Remove", svg::TRASH, Group::Library, 3, Needs::Files).key("Delete").confirm(Confirm::Always),
    spec(ActionId::Tag, "tag_selection", "Tag...", "Tag", svg::TAG, Group::Organize, 0, Needs::Files).key("T"),
    spec(ActionId::Category, "category_selection", "Category...", "Category", svg::AME_CATEGORY, Group::Organize, 4, Needs::Files),
    spec(ActionId::AddToCollection, "add_selection_to_group", "Add to collection...", "Add to collection", svg::PLUS_CIRCLE, Group::Organize, 1, Needs::Files).key("C"),
    spec(ActionId::RemoveFromCollection, "remove_selection_from_group", "Remove from collection...", "Remove from collection", svg::MINUS_CIRCLE, Group::Organize, 8, Needs::Files),
    spec(ActionId::Describe, "generate_description", "Generate Description", "Describe", svg::AME_AI_DESCRIBE, Group::Ai, 1, Needs::Images).key("D").confirm(Confirm::Bulk),
    spec(ActionId::RegenerateDescription, "regenerate_description", "Regenerate Description (overwrite)", "Regenerate", svg::AME_REGENERATE, Group::Ai, 8, Needs::Images).confirm(Confirm::Bulk),
    spec(ActionId::GenerateClip, "generate_clip_embeddings", "Generate CLIP Embedding", "CLIP", svg::AME_CLIP_EMBED, Group::Ai, 2, Needs::Images).confirm(Confirm::Bulk),
    spec(ActionId::RegenerateClip, "regenerate_clip_embedding", "Regenerate CLIP Embedding", "Regenerate CLIP", svg::AME_REGENERATE, Group::Ai, 9, Needs::Images).confirm(Confirm::Bulk),
    spec(ActionId::FindSimilar, "find_similar", "Find Similar (CLIP)", "Find similar", svg::AME_FIND_SIMILAR, Group::Ai, 3, Needs::Files),
    spec(ActionId::Refine, "refine_selection", "Refine Selection", "Refine...", svg::AME_REFINE, Group::Ai, 5, Needs::Files),
    spec(ActionId::AttachToAssistant, "attach_to_chat_window", "Attach to chat window", "Ask AI", svg::AME_ASSISTANT, Group::Ai, 4, Needs::Files),
    spec(ActionId::CopyTo, "copy_to", "Copy to...", "Copy to", svg::COPY, Group::Files, 6, Needs::LocalFiles),
    spec(ActionId::MoveTo, "move_to", "Move to...", "Move to", svg::FILE_ARROW_UP, Group::Files, 7, Needs::LocalFiles).confirm(Confirm::Always),
    spec(ActionId::ExcludeFolder, "exclude_dirs_from_scan", "Exclude selected directories from recursive scans", "Exclude folder", svg::AME_EXCLUDE_FOLDER, Group::Files, 10, Needs::Folders),
];

/// The spec of `id`.
pub fn spec_of(id: ActionId) -> &'static ActionSpec {
    SPECS.iter().find(|s| s.id == id).expect("every action has a spec")
}

/// The action whose context menu item is called `menu_id`.
pub fn from_menu_id(menu_id: &str) -> Option<ActionId> {
    SPECS.iter().find(|s| s.menu_id == menu_id).map(|s| s.id)
}

/// The numbers over a selection that decide which actions can run.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Facts {
    pub selected: usize,
    pub files: usize,
    pub dirs: usize,
    /// Selected files on disk, not inside an archive.
    pub local: usize,
    pub images: usize,
    pub size: u64,
    pub db_ready: bool,
    pub ai_ready: bool,
}

impl Facts {
    /// Facts from the counts a context menu knows: every selected file counts as an image and as a file on disk.
    pub fn from_counts(rows: usize, dirs: usize, db_ready: bool, ai_ready: bool) -> Self {
        let files = rows.saturating_sub(dirs);
        Self { selected: rows, files, dirs, local: files, images: files, size: 0, db_ready, ai_ready }
    }
}

fn none_reason(f: &Facts, what: &str) -> String {
    if f.selected == 0 { format!("Select {what} first") } else { format!("No {what} in the selection") }
}

/// Why `id` cannot run on `facts`, or `Ok` when it can. Nothing selected never means everything.
pub fn availability(id: ActionId, f: &Facts) -> Result<(), String> {
    let db = || if f.db_ready { Ok(()) } else { Err("The library is not connected".to_string()) };
    let files = || if f.files == 0 { Err(if f.selected == 0 { "Select files first".to_string() } else { "Only folders are selected".to_string() }) } else { Ok(()) };
    let images = || if f.images == 0 { Err(none_reason(f, "an image")) } else { Ok(()) };
    let ai = || if f.ai_ready { Ok(()) } else { Err("Choose a vision model folder in the settings".to_string()) };
    match id {
        ActionId::Open | ActionId::Reveal => match f.selected {
            0 => Err("Select an item first".to_string()),
            n if n > OPEN_LIMIT => Err(format!("Acts on at most {OPEN_LIMIT} items at once")),
            _ => Ok(()),
        },
        ActionId::CopyPath | ActionId::OpenInNewTab => if f.selected == 0 { Err("Select an item first".to_string()) } else { Ok(()) },
        ActionId::AddToLibrary | ActionId::RemoveFromLibrary | ActionId::Tag | ActionId::Category | ActionId::AddToCollection | ActionId::RemoveFromCollection | ActionId::FindSimilar => {
            files()?;
            db()
        }
        ActionId::Describe | ActionId::RegenerateDescription => {
            images()?;
            ai()
        }
        ActionId::GenerateClip | ActionId::RegenerateClip => images(),
        ActionId::Refine | ActionId::AttachToAssistant => files(),
        ActionId::CopyTo | ActionId::MoveTo => if f.local == 0 { Err(none_reason(f, "a file on disk")) } else { Ok(()) },
        ActionId::ExcludeFolder => if f.dirs == 0 { Err("Select a folder first".to_string()) } else { Ok(()) },
    }
}

/// The count an action reports and confirms with: folders only count for the actions that act on folders.
pub fn target_count(id: ActionId, f: &Facts) -> usize {
    match id {
        ActionId::Open | ActionId::Reveal | ActionId::CopyPath | ActionId::OpenInNewTab => f.selected,
        ActionId::ExcludeFolder => f.dirs,
        ActionId::Describe | ActionId::RegenerateDescription | ActionId::GenerateClip | ActionId::RegenerateClip => f.images,
        ActionId::CopyTo | ActionId::MoveTo => f.local,
        _ => f.files,
    }
}

/// Whether running `id` on `count` files first asks the user.
pub fn needs_confirmation(id: ActionId, count: usize) -> bool {
    match spec_of(id).confirm {
        Confirm::Never => false,
        Confirm::Always => count > 0,
        Confirm::Bulk => count > BULK_CONFIRM_ABOVE,
    }
}

/// Which of `items` (priority, width) fit in `avail`: the highest priority numbers drop first and the later one of two equals; a More button of `more_w` is reserved once anything drops.
pub fn plan_bar(items: &[(u8, f32)], gap: f32, more_w: f32, avail: f32) -> Vec<bool> {
    let mut shown = vec![true; items.len()];
    let width = |shown: &[bool], extra: f32| {
        let widths: Vec<f32> = items.iter().zip(shown).filter(|(_, on)| **on).map(|(i, _)| i.1).collect();
        widths.iter().sum::<f32>() + gap * widths.len().saturating_sub(1) as f32 + extra
    };
    if width(&shown, 0.0) <= avail {
        return shown;
    }
    let mut order: Vec<usize> = (0..items.len()).collect();
    order.sort_by(|a, b| items[*b].0.cmp(&items[*a].0).then(b.cmp(a)));
    for i in order {
        shown[i] = false;
        if width(&shown, more_w + gap) <= avail {
            break;
        }
    }
    shown
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(selected: usize, dirs: usize) -> Facts {
        let mut f = Facts::from_counts(selected, dirs, true, true);
        f.size = 1_000;
        f
    }

    #[test]
    fn every_action_has_exactly_one_spec_with_unique_menu_ids_and_clean_labels() {
        for id in ActionId::ALL {
            assert_eq!(SPECS.iter().filter(|s| s.id == id).count(), 1, "{id:?}");
        }
        assert_eq!(SPECS.len(), ActionId::ALL.len());
        let ids: std::collections::BTreeSet<_> = SPECS.iter().map(|s| s.menu_id).collect();
        assert_eq!(ids.len(), SPECS.len());
        for s in &SPECS {
            assert!(s.label.is_ascii() && s.short.is_ascii() && !s.label.is_empty() && !s.short.is_empty(), "{}", s.menu_id);
            assert_eq!(from_menu_id(s.menu_id), Some(s.id));
        }
        assert_eq!(from_menu_id("nope"), None);
    }

    #[test]
    fn the_menu_ids_of_the_earlier_table_are_kept() {
        for old in [
            "generate_description",
            "regenerate_description",
            "generate_clip_embeddings",
            "regenerate_clip_embedding",
            "add_selection_to_group",
            "remove_selection_from_group",
            "open_selection_in_new_tab",
            "exclude_dirs_from_scan",
            "find_similar",
            "refine_selection",
            "attach_to_chat_window",
        ] {
            assert!(from_menu_id(old).is_some(), "{old}");
        }
    }

    #[test]
    fn an_empty_selection_enables_nothing() {
        let none = Facts { db_ready: true, ai_ready: true, ..Default::default() };
        for id in ActionId::ALL {
            assert!(availability(id, &none).is_err(), "{id:?} must not run on nothing");
        }
    }

    #[test]
    fn file_actions_need_files_and_folder_actions_need_folders() {
        let only_dirs = facts(2, 2);
        for id in [ActionId::Tag, ActionId::AddToLibrary, ActionId::Describe, ActionId::CopyTo, ActionId::Refine] {
            assert!(availability(id, &only_dirs).is_err(), "{id:?}");
        }
        assert_eq!(availability(ActionId::Tag, &only_dirs).unwrap_err(), "Only folders are selected");
        assert!(availability(ActionId::ExcludeFolder, &only_dirs).is_ok());
        assert!(availability(ActionId::OpenInNewTab, &only_dirs).is_ok());
        let files = facts(3, 0);
        assert!(availability(ActionId::ExcludeFolder, &files).is_err());
        for id in ActionId::ALL.into_iter().filter(|i| *i != ActionId::ExcludeFolder) {
            assert!(availability(id, &files).is_ok(), "{id:?}");
        }
    }

    #[test]
    fn the_library_and_the_vision_model_gate_their_actions_with_a_reason() {
        let mut f = facts(3, 0);
        f.db_ready = false;
        for id in [ActionId::AddToLibrary, ActionId::RemoveFromLibrary, ActionId::Tag, ActionId::Category, ActionId::AddToCollection, ActionId::FindSimilar] {
            assert_eq!(availability(id, &f).unwrap_err(), "The library is not connected", "{id:?}");
        }
        assert!(availability(ActionId::Open, &f).is_ok() && availability(ActionId::GenerateClip, &f).is_ok());
        f.db_ready = true;
        f.ai_ready = false;
        assert!(availability(ActionId::Describe, &f).unwrap_err().contains("vision model"));
        assert!(availability(ActionId::RegenerateDescription, &f).is_err());
        assert!(availability(ActionId::Tag, &f).is_ok());
    }

    #[test]
    fn open_and_reveal_stop_at_the_limit() {
        assert!(availability(ActionId::Open, &facts(OPEN_LIMIT, 0)).is_ok());
        assert!(availability(ActionId::Reveal, &facts(OPEN_LIMIT + 1, 0)).unwrap_err().contains("at most"));
        assert!(availability(ActionId::CopyPath, &facts(500, 0)).is_ok());
    }

    #[test]
    fn archive_entries_cannot_be_copied_or_moved() {
        let mut f = facts(4, 0);
        f.local = 0;
        assert!(availability(ActionId::CopyTo, &f).is_err());
        assert!(availability(ActionId::MoveTo, &f).is_err());
        f.local = 1;
        assert!(availability(ActionId::MoveTo, &f).is_ok());
    }

    #[test]
    fn counts_follow_what_the_action_acts_on() {
        let mut f = facts(10, 3);
        f.images = 4;
        f.local = 6;
        assert_eq!(target_count(ActionId::Open, &f), 10);
        assert_eq!(target_count(ActionId::Tag, &f), 7);
        assert_eq!(target_count(ActionId::Describe, &f), 4);
        assert_eq!(target_count(ActionId::CopyTo, &f), 6);
        assert_eq!(target_count(ActionId::ExcludeFolder, &f), 3);
    }

    #[test]
    fn confirmation_is_always_for_removal_and_moves_and_bulk_above_the_limit_for_heavy_work() {
        assert!(needs_confirmation(ActionId::RemoveFromLibrary, 1));
        assert!(needs_confirmation(ActionId::MoveTo, 1));
        assert!(!needs_confirmation(ActionId::RemoveFromLibrary, 0));
        assert!(!needs_confirmation(ActionId::Describe, BULK_CONFIRM_ABOVE));
        assert!(needs_confirmation(ActionId::Describe, BULK_CONFIRM_ABOVE + 1));
        assert!(needs_confirmation(ActionId::AddToLibrary, 1_000));
        assert!(!needs_confirmation(ActionId::Tag, 1_000), "the batch popover already shows the exact count");
        assert!(!needs_confirmation(ActionId::Open, 10));
    }

    #[test]
    fn the_bar_drops_low_priority_buttons_first_and_reserves_room_for_more() {
        let items = [(0u8, 50.0), (1, 50.0), (2, 50.0), (3, 50.0)];
        assert_eq!(plan_bar(&items, 4.0, 30.0, 400.0), vec![true; 4]);
        assert_eq!(plan_bar(&items, 4.0, 30.0, 200.0), vec![true, true, true, false]);
        assert_eq!(plan_bar(&items, 4.0, 30.0, 140.0), vec![true, true, false, false]);
        assert_eq!(plan_bar(&items, 4.0, 30.0, 10.0), vec![false; 4]);
    }

    #[test]
    fn equal_priorities_drop_from_the_end_and_the_plan_never_overflows() {
        let items = [(1u8, 40.0), (1, 40.0), (1, 40.0)];
        assert_eq!(plan_bar(&items, 0.0, 20.0, 100.0), vec![true, true, false]);
        for avail in (0..300).step_by(7) {
            let shown = plan_bar(&items, 4.0, 24.0, avail as f32);
            let n = shown.iter().filter(|s| **s).count();
            let used = n as f32 * 40.0 + 4.0 * n.saturating_sub(1) as f32 + if n < 3 { 28.0 } else { 0.0 };
            assert!(used <= avail as f32 + 0.01 || n == 0, "{avail}: {shown:?}");
        }
    }

    #[test]
    fn shortcuts_are_unique() {
        let keys: Vec<_> = SPECS.iter().filter_map(|s| s.shortcut).collect();
        let unique: std::collections::BTreeSet<_> = keys.iter().collect();
        assert_eq!(keys.len(), unique.len());
    }
}
