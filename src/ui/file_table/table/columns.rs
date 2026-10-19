//! Details view columns: identity, titles, widths, and which ones the user hides.

use serde::{Deserialize, Serialize};

use super::ExplorerMode;
use crate::ui::file_table::rows::SortSpec;

/// A details column; the discriminant is the stable id used by sort specs and saved preferences.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Col {
    Thumb = 0,
    Name = 1,
    Path = 2,
    Size = 3,
    Type = 4,
    Clip = 5,
    Modified = 6,
    DbCreated = 7,
    Category = 8,
    Tags = 9,
    Hash = 10,
    Similarity = 11,
}

/// Width bounds of a column in points.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Width {
    pub initial: f32,
    pub min: f32,
    pub max: f32,
}

impl Col {
    pub const ALL: [Col; 12] = [Col::Thumb, Col::Name, Col::Path, Col::Size, Col::Type, Col::Clip, Col::Modified, Col::DbCreated, Col::Category, Col::Tags, Col::Hash, Col::Similarity];

    pub fn id(self) -> u8 {
        self as u8
    }

    pub fn from_id(id: u8) -> Option<Col> {
        Col::ALL.into_iter().find(|c| c.id() == id)
    }

    /// Text of the column header; the thumbnail column has none.
    pub fn header(self) -> &'static str {
        match self {
            Col::Thumb => "",
            Col::Name => "Name",
            Col::Path => "Path",
            Col::Size => "Size",
            Col::Type => "Type",
            Col::Clip => "CLIP",
            Col::Modified => "Modified",
            Col::DbCreated => "DB Created",
            Col::Category => "Category",
            Col::Tags => "Tags",
            Col::Hash => "Hash",
            Col::Similarity => "Similarity",
        }
    }

    /// Name shown in the column chooser.
    pub fn menu_name(self) -> &'static str {
        match self {
            Col::Thumb => "Thumbnail",
            other => other.header(),
        }
    }

    pub fn sortable(self) -> bool {
        self != Col::Thumb
    }

    pub fn hideable(self) -> bool {
        self != Col::Name
    }

    pub fn width(self) -> Width {
        let (initial, min, max) = match self {
            Col::Thumb => (84.0, 60.0, 160.0),
            Col::Name => (200.0, 120.0, 600.0),
            Col::Path => (280.0, 140.0, 1200.0),
            Col::Size => (74.0, 60.0, 160.0),
            Col::Type => (64.0, 50.0, 140.0),
            Col::Clip => (52.0, 46.0, 100.0),
            Col::Modified => (92.0, 80.0, 200.0),
            Col::DbCreated => (92.0, 80.0, 200.0),
            Col::Category => (120.0, 90.0, 300.0),
            Col::Tags => (170.0, 120.0, 600.0),
            Col::Hash => (160.0, 100.0, 400.0),
            Col::Similarity => (84.0, 70.0, 160.0),
        };
        Width { initial, min, max }
    }

    /// Columns the current data can show, in display order.
    pub fn available(mode: ExplorerMode, showing_similarity: bool) -> Vec<Col> {
        Col::ALL
            .into_iter()
            .filter(|c| match c {
                Col::Hash => mode == ExplorerMode::Database,
                Col::Similarity => showing_similarity,
                _ => true,
            })
            .collect()
    }
}

/// Columns the user hid and the sort order, as saved between sessions.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DetailsPrefs {
    pub hidden: Vec<u8>,
    pub sort: SortSpec,
}

impl DetailsPrefs {
    pub fn is_hidden(&self, col: Col) -> bool {
        col.hideable() && self.hidden.contains(&col.id())
    }

    /// Hides or shows `col`; returns whether anything changed.
    pub fn set_hidden(&mut self, col: Col, hidden: bool) -> bool {
        if !col.hideable() {
            return false;
        }
        let present = self.hidden.contains(&col.id());
        match (hidden, present) {
            (true, false) => self.hidden.push(col.id()),
            (false, true) => self.hidden.retain(|c| *c != col.id()),
            _ => return false,
        }
        true
    }

    /// Available columns the user has not hidden.
    pub fn visible(&self, mode: ExplorerMode, showing_similarity: bool) -> Vec<Col> {
        Col::available(mode, showing_similarity).into_iter().filter(|c| !self.is_hidden(*c)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_stable_and_round_trip() {
        let ids: Vec<u8> = Col::ALL.iter().map(|c| c.id()).collect();
        assert_eq!(ids, (0..12).collect::<Vec<u8>>());
        for c in Col::ALL {
            assert_eq!(Col::from_id(c.id()), Some(c));
        }
        assert_eq!(Col::from_id(12), None);
    }

    #[test]
    fn the_file_system_columns_match_the_old_table_in_order() {
        let names: Vec<&str> = Col::available(ExplorerMode::FileSystem, false).iter().map(|c| c.header()).collect();
        assert_eq!(names, ["", "Name", "Path", "Size", "Type", "CLIP", "Modified", "DB Created", "Category", "Tags"]);
    }

    #[test]
    fn the_database_view_adds_hash_and_similarity_results_add_a_score() {
        let db: Vec<Col> = Col::available(ExplorerMode::Database, false);
        assert_eq!(db.last(), Some(&Col::Hash));
        let sim: Vec<Col> = Col::available(ExplorerMode::Database, true);
        assert_eq!(sim.last(), Some(&Col::Similarity));
        assert_eq!(sim.len(), 12);
        assert_eq!(Col::available(ExplorerMode::FileSystem, true).last(), Some(&Col::Similarity));
        assert!(!Col::available(ExplorerMode::FileSystem, true).contains(&Col::Hash));
    }

    #[test]
    fn only_the_thumbnail_is_unsortable_and_only_the_name_is_unhideable() {
        for c in Col::ALL {
            assert_eq!(c.sortable(), c != Col::Thumb);
            assert_eq!(c.hideable(), c != Col::Name);
        }
    }

    #[test]
    fn width_bounds_are_ordered() {
        for c in Col::ALL {
            let w = c.width();
            assert!(w.min <= w.initial && w.initial <= w.max, "{c:?}");
        }
    }

    #[test]
    fn hiding_columns_removes_them_from_the_visible_set_but_never_the_name() {
        let mut prefs = DetailsPrefs::default();
        assert!(prefs.set_hidden(Col::Path, true));
        assert!(!prefs.set_hidden(Col::Path, true));
        assert!(!prefs.set_hidden(Col::Name, true));
        let visible = prefs.visible(ExplorerMode::FileSystem, false);
        assert!(!visible.contains(&Col::Path) && visible.contains(&Col::Name));
        assert!(prefs.set_hidden(Col::Path, false));
        assert!(prefs.visible(ExplorerMode::FileSystem, false).contains(&Col::Path));
    }

    #[test]
    fn a_hidden_database_only_column_stays_hidden_when_it_is_unavailable() {
        let mut prefs = DetailsPrefs::default();
        prefs.set_hidden(Col::Hash, true);
        assert!(!prefs.visible(ExplorerMode::Database, false).contains(&Col::Hash));
        prefs.set_hidden(Col::Hash, false);
        assert!(prefs.visible(ExplorerMode::Database, false).contains(&Col::Hash));
        assert!(!prefs.visible(ExplorerMode::FileSystem, false).contains(&Col::Hash));
    }

    #[test]
    fn preferences_round_trip_through_json() {
        let mut prefs = DetailsPrefs::default();
        prefs.set_hidden(Col::Tags, true);
        prefs.sort.click(Col::Size.id(), false);
        let json = serde_json::to_string(&prefs).unwrap();
        assert_eq!(serde_json::from_str::<DetailsPrefs>(&json).unwrap(), prefs);
    }
}
