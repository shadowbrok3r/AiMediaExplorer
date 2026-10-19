//! Viewer state of an explorer tab and the details view built on it.

use std::collections::HashMap;
use std::sync::Arc;

use crossbeam::channel::Sender;
use serde::Serialize;

use crate::ui::file_table::AIUpdate;
use crate::ui::file_table::filters::FilterState;
use crate::ui::file_table::thumbs::{ScopeId, ThumbPresence, ThumbService, alloc_scope};
use crate::{Thumbnail, UiSettings};

pub mod actions;
pub mod browse;
pub mod cells;
pub mod columns;
pub mod details;
pub mod grid;
pub mod grid_facts;
pub mod grid_layout;
pub mod grid_tile;
pub mod keys;
pub mod scale;
pub mod view_mode;
pub mod view_strip;

#[cfg(test)]
mod details_tests;
#[cfg(test)]
mod grid_tests;

#[derive(Default, Copy, Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ExplorerMode {
    #[default]
    FileSystem,
    Database,
}

impl ExplorerMode {
    pub fn to_str(&self) -> &str {
        match self {
            ExplorerMode::FileSystem => "FileSystem",
            ExplorerMode::Database => "Database",
        }
    }
}

#[derive(Serialize)]
pub struct FileTableViewer {
    pub filter: String,
    #[serde(skip)]
    thumbnail_tx: Sender<Thumbnail>,
    #[serde(skip)]
    pub thumbs: Arc<ThumbService>,
    #[serde(skip)]
    pub scope: ScopeId,
    // Which paths have a stored thumbnail, for status counts.
    #[serde(skip)]
    pub thumb_cache: ThumbPresence,
    pub mode: ExplorerMode,
    // Flag to stop scheduling bulk description tasks
    #[serde(skip)]
    pub bulk_cancel_requested: bool,
    pub ui_settings: crate::UiSettings,
    #[serde(skip)]
    pub settings_gen: u64,
    #[serde(skip)]
    pub ai_update_tx: Sender<AIUpdate>,
    #[serde(skip)]
    pub clip_embedding_tx: Sender<crate::ClipEmbeddingRow>,
    // When showing similarity results inline, toggle this to add a scores column
    pub showing_similarity: bool,
    // Map path -> similarity score to display/sort when showing_similarity is true
    #[serde(skip)]
    pub similar_scores: HashMap<String, f32>,
    // Track which paths have a CLIP embedding
    #[serde(skip)]
    pub clip_presence: std::collections::HashSet<String>,
    // Track which hashes have CLIP embeddings (for duplicate content across paths)
    #[serde(skip)]
    pub clip_presence_hashes: std::collections::HashSet<String>,
    // Type visibility toggles (UI): affect table view filtering and scans
    pub types_show_images: bool,
    pub types_show_videos: bool,
    // Archive passwords for accessing encrypted archives
    #[serde(skip)]
    pub archive_passwords: HashMap<String, String>,
    pub types_show_dirs: bool,
    // Filter conditions beyond the search text and type toggles above
    #[serde(skip)]
    pub filters: FilterState,
    // Requests to open tabs based on user clicks in cells
    #[serde(skip)]
    pub requested_tabs: Vec<TabAction>,
    // Library and disk badges of the rows
    #[serde(skip)]
    pub badges: crate::ui::file_table::scope::Badges,
}

// Actions requested from table cells
#[derive(Clone, Debug)]
pub enum TabAction {
    OpenCategory(String),
    OpenTag(String),
    OpenSimilar(String),
    OpenArchive(String),
}

impl FileTableViewer {
    pub fn new(thumbnail_tx: Sender<Thumbnail>, ai_update_tx: Sender<AIUpdate>, clip_embedding_tx: Sender<crate::ClipEmbeddingRow>) -> Self {
        let (ui_settings, settings_gen): (UiSettings, u64) = crate::settings::new_mirror();
        let thumbs = crate::ui::file_table::thumbs::service();
        let filters = FilterState::new(&ui_settings);
        Self {
            filter: Default::default(),
            thumb_cache: ThumbPresence::new(thumbs.clone()),
            thumbs,
            scope: alloc_scope(),
            thumbnail_tx,
            mode: ExplorerMode::default(),
            ui_settings,
            settings_gen,
            bulk_cancel_requested: false,
            clip_embedding_tx,
            ai_update_tx,
            showing_similarity: false,
            similar_scores: HashMap::new(),
            clip_presence: std::collections::HashSet::new(),
            clip_presence_hashes: std::collections::HashSet::new(),
            types_show_images: true,
            types_show_videos: true,
            types_show_dirs: true,
            filters,
            requested_tabs: Vec::new(),
            archive_passwords: HashMap::new(),
            badges: Default::default(),
        }
    }

    /// Refreshes the read-only settings mirror when the store changed.
    pub fn sync_settings(&mut self) -> bool {
        let changed = crate::settings::sync_mirror(&mut self.ui_settings, &mut self.settings_gen);
        if changed {
            self.filters.follow_settings(&self.ui_settings);
        }
        changed
    }
}
