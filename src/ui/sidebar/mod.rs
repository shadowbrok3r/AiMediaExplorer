//! Global left sidebar: places, library views, collections, tags, categories and scans of the focused browser.

pub mod data;
pub mod layout;
pub mod links;
mod view;

#[cfg(test)]
mod tests;

pub use data::{Source, SidebarData, request_refresh};
pub use layout::{LayoutStore, UiLayout};
pub use view::{show_full, show_rail};

use std::time::Instant;

use surrealdb::types::RecordId;

use crate::command::{AppCommand, Scope};
use crate::icons::{Icon, svg};

/// Window width below which the sidebar shrinks to an icon rail.
pub const NARROW_WIDTH: f32 = 900.0;
/// Width of the icon rail including its margins.
pub const RAIL_WIDTH: f32 = 44.0;

/// How much of the sidebar shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Hidden,
    Rail,
    Full,
}

/// The mode for a window `width` wide.
pub fn mode_for(width: f32, open: bool) -> Mode {
    match (open, width < NARROW_WIDTH) {
        (false, _) => Mode::Hidden,
        (true, true) => Mode::Rail,
        (true, false) => Mode::Full,
    }
}

/// A collapsible block of the sidebar.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SectionId {
    Places,
    Library,
    Collections,
    Tags,
    Categories,
    Scans,
}

impl SectionId {
    pub const ALL: [SectionId; 6] = [Self::Places, Self::Library, Self::Collections, Self::Tags, Self::Categories, Self::Scans];

    /// Key in the saved layout.
    pub fn key(self) -> &'static str {
        match self {
            Self::Places => "places",
            Self::Library => "library",
            Self::Collections => "collections",
            Self::Tags => "tags",
            Self::Categories => "categories",
            Self::Scans => "scans",
        }
    }

    pub fn title(self) -> &'static str {
        match self {
            Self::Places => "Places",
            Self::Library => "Library",
            Self::Collections => "Collections",
            Self::Tags => "Tags",
            Self::Categories => "Categories",
            Self::Scans => "Scans",
        }
    }

    pub fn icon(self) -> Icon {
        match self {
            Self::Places => svg::AME_QUICK_ACCESS,
            Self::Library => svg::AME_DATABASE,
            Self::Collections => svg::AME_LOGICAL_GROUP,
            Self::Tags => svg::TAG,
            Self::Categories => svg::AME_CATEGORY,
            Self::Scans => svg::AME_SCAN,
        }
    }

    pub fn default_open(self) -> bool {
        matches!(self, Self::Places | Self::Library | Self::Collections)
    }

    /// True for sections that read the database.
    pub fn needs_library(self) -> bool {
        self != Self::Places
    }
}

/// What the host tells the sidebar each frame.
pub struct SidebarInput<'a> {
    pub db_ready: bool,
    /// Scope the focused browser shows, when known.
    pub active: Option<&'a Scope>,
    pub failed_thumbs: usize,
    pub recent_paths: &'a [String],
    pub now: Instant,
}

/// What a sidebar interaction asks the host to do.
#[derive(Clone, Debug)]
pub enum SidebarAction {
    Command(AppCommand),
    /// Open the database setup.
    Connect,
}

/// A collection being renamed inline.
#[derive(Clone, Debug)]
pub struct RenameState {
    pub id: RecordId,
    pub input: String,
}

/// Interaction state of the sidebar.
#[derive(Default)]
pub struct SidebarState {
    pub data: SidebarData,
    pub tag_filter: String,
    pub category_filter: String,
    pub show_all_tags: bool,
    pub show_all_categories: bool,
    /// Text of the "new collection" field while it is open.
    pub new_collection: Option<String>,
    pub rename: Option<RenameState>,
    /// Collection waiting for a delete confirmation.
    pub confirm_delete: Option<String>,
    /// Debug switch that shows the not-connected layout.
    pub force_disconnected: bool,
}

impl SidebarState {
    /// State that loads from the system and the database.
    pub fn live() -> Self {
        Self { data: SidebarData::live(), ..Default::default() }
    }
}
