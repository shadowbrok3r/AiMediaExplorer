//! Viewer state of an explorer tab and the details view built on it.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crossbeam::channel::Sender;
use serde::Serialize;

use crate::ui::file_table::AIUpdate;
use crate::ui::file_table::rows::thumbnail::{DIR_TYPE, short_extension, with_lowercase};
use crate::ui::file_table::thumbs::{ScopeId, ThumbPresence, ThumbService, alloc_scope};
use crate::{Thumbnail, UiSettings};

pub mod actions;
pub mod cells;
pub mod columns;
pub mod details;
pub mod keys;

#[cfg(test)]
mod details_tests;

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
    // Requests to open tabs based on user clicks in cells
    #[serde(skip)]
    pub requested_tabs: Vec<TabAction>,
    // Read-only projection of the selection for readers outside the table.
    #[serde(skip)]
    pub selected: std::collections::HashSet<String>,
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
            requested_tabs: Vec::new(),
            archive_passwords: HashMap::new(),
            selected: std::collections::HashSet::new(),
        }
    }

    /// Refreshes the read-only settings mirror when the store changed.
    pub fn sync_settings(&mut self) -> bool {
        crate::settings::sync_mirror(&mut self.ui_settings, &mut self.settings_gen)
    }

    /// The row filter for the current text and type toggles, prepared once for many rows.
    pub fn row_filter(&self) -> RowFilter {
        let needle = if self.filter.trim().is_empty() { String::new() } else { self.filter.to_lowercase() };
        RowFilter { show_images: self.types_show_images, show_videos: self.types_show_videos, show_dirs: self.types_show_dirs, ascii: needle.is_ascii(), needle }
    }

    /// Whether a row passes the current type toggles and search text.
    pub fn row_passes_filter(&self, row: &Thumbnail) -> bool {
        self.row_filter().matches(row)
    }

    /// Changes whenever [`row_filter`](Self::row_filter) may answer differently for the same rows.
    pub fn filter_signature(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (&self.filter, self.types_show_images, self.types_show_videos, self.types_show_dirs, self.mode as u8).hash(&mut h);
        h.finish()
    }
}

/// Type toggles and search text, ready to test rows against.
#[derive(Clone, Debug)]
pub struct RowFilter {
    show_images: bool,
    show_videos: bool,
    show_dirs: bool,
    needle: String,
    ascii: bool,
}

fn contains_ci(hay: &str, needle: &str, ascii: bool) -> bool {
    if needle.is_empty() {
        return true;
    }
    if ascii && hay.is_ascii() {
        let (h, n) = (hay.as_bytes(), needle.as_bytes());
        return n.len() <= h.len() && h.windows(n.len()).any(|w| w.eq_ignore_ascii_case(n));
    }
    hay.to_lowercase().contains(needle)
}

impl RowFilter {
    pub fn matches(&self, row: &Thumbnail) -> bool {
        let is_dir = row.file_type == DIR_TYPE;
        if is_dir && !self.show_dirs {
            return false;
        }
        if !is_dir {
            let source = if row.path.starts_with("zip://") { &row.filename } else { &row.path };
            if let Some(ext) = short_extension(source) {
                let hidden = with_lowercase(ext, |lower| (crate::is_image(lower) && !self.show_images) || (crate::is_video(lower) && !self.show_videos));
                if hidden {
                    return false;
                }
            }
        }
        if self.needle.is_empty() {
            return true;
        }
        let (n, a) = (self.needle.as_str(), self.ascii);
        contains_ci(&row.filename, n, a)
            || contains_ci(&row.file_type, n, a)
            || contains_ci(&row.path, n, a)
            || (!row.tags.is_empty() && contains_ci(&row.tags.join(","), n, a))
            || row.category.as_deref().is_some_and(|c| contains_ci(c, n, a))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(path: &str, file_type: &str) -> Thumbnail {
        let mut t = Thumbnail::default();
        t.path = path.into();
        t.filename = path.rsplit(['/', '\\']).next().unwrap_or(path).into();
        t.file_type = file_type.into();
        t
    }

    fn filter(text: &str, images: bool, videos: bool, dirs: bool) -> RowFilter {
        let needle = if text.trim().is_empty() { String::new() } else { text.to_lowercase() };
        RowFilter { show_images: images, show_videos: videos, show_dirs: dirs, ascii: needle.is_ascii(), needle }
    }

    /// The previous filter, kept as the reference for the prepared one.
    fn legacy(text: &str, images: bool, videos: bool, dirs: bool, row: &Thumbnail) -> bool {
        if row.file_type == "<DIR>" && !dirs {
            return false;
        }
        if row.file_type != "<DIR>" {
            let ext_opt = if row.path.starts_with("zip://") {
                std::path::Path::new(&row.filename).extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase())
            } else {
                std::path::Path::new(&row.path).extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase())
            };
            if let Some(ext) = ext_opt {
                if crate::is_image(ext.as_str()) && !images {
                    return false;
                }
                if crate::is_video(ext.as_str()) && !videos {
                    return false;
                }
            }
        }
        if text.trim().is_empty() {
            return true;
        }
        let f = text.to_lowercase();
        let tags_join = row.tags.join(",");
        row.filename.to_lowercase().contains(&f)
            || row.file_type.to_lowercase().contains(&f)
            || row.path.to_lowercase().contains(&f)
            || tags_join.to_lowercase().contains(&f)
            || row.category.as_ref().map(|s| s.to_lowercase().contains(&f)).unwrap_or(false)
    }

    #[test]
    fn type_toggles_hide_images_videos_and_folders_independently() {
        let img = row("/a/x.jpg", "jpg");
        let vid = row("/a/y.mp4", "mp4");
        let dir = row("/a/d", "<DIR>");
        let txt = row("/a/t.txt", "txt");
        let arc = row("/a/z.zip", "<ARCHIVE>");
        assert!(!filter("", false, true, true).matches(&img));
        assert!(filter("", false, true, true).matches(&vid));
        assert!(!filter("", true, false, true).matches(&vid));
        assert!(!filter("", true, true, false).matches(&dir));
        for hidden in [filter("", false, false, false)] {
            assert!(hidden.matches(&txt) && hidden.matches(&arc), "other files and archives are never hidden by the type toggles");
        }
    }

    #[test]
    fn zip_entries_take_their_type_from_the_file_name() {
        let mut entry = row("zip:///x/a.zip!/inner/pic.png", "png");
        entry.filename = "pic.png".into();
        assert!(!filter("", false, true, true).matches(&entry));
        assert!(filter("", true, true, true).matches(&entry));
    }

    #[test]
    fn the_search_text_matches_name_type_path_tags_and_category_ignoring_case() {
        let mut r = row("/pics/Holiday/IMG_001.JPG", "jpg");
        r.tags = vec!["Beach".into(), "sun".into()];
        r.category = Some("Travel".into());
        for hit in ["img_001", "holiday", "JPG", "beach", "SUN", "travel", "beach,sun"] {
            assert!(filter(hit, true, true, true).matches(&r), "{hit}");
        }
        for miss in ["mountain", "img_002", "snow"] {
            assert!(!filter(miss, true, true, true).matches(&r), "{miss}");
        }
        assert!(filter("   ", true, true, true).matches(&r), "blank text is no filter");
    }

    #[test]
    fn non_ascii_text_still_matches_case_insensitively() {
        let r = row("/fotos/ÜBER/straße.jpg", "jpg");
        assert!(filter("über", true, true, true).matches(&r));
        assert!(filter("STRASSE", true, true, true).matches(&r) == legacy("STRASSE", true, true, true, &r));
        assert!(filter("ß", true, true, true).matches(&r));
    }

    #[test]
    fn the_prepared_filter_agrees_with_the_previous_one() {
        let paths = ["/a/x.jpg", "/a/Y.PNG", "/a/clip.MP4", "/a/notes.txt", "/a/d", "zip:///z.zip!/in/p.png", "C:\\pics\\q.JPEG", "/a/.hidden", "/a/noext", "/ü/é.jpg"];
        let types = ["jpg", "<DIR>", "<ARCHIVE>", "txt", ""];
        let texts = ["", "x", "PNG", "clip", "a/", "é", "tag1", "cat", "j,t", " "];
        let mut n = 0;
        for p in paths {
            for t in types {
                let mut r = row(p, t);
                r.tags = vec!["Tag1".into(), "two".into()];
                r.category = Some("Cat".into());
                for text in texts {
                    for flags in 0..8u8 {
                        let (i, v, d) = (flags & 1 != 0, flags & 2 != 0, flags & 4 != 0);
                        assert_eq!(filter(text, i, v, d).matches(&r), legacy(text, i, v, d, &r), "{p} {t} {text:?} {flags}");
                        n += 1;
                    }
                }
            }
        }
        assert!(n > 1000);
    }

    #[test]
    fn contains_ci_handles_edges() {
        assert!(contains_ci("anything", "", true));
        assert!(!contains_ci("ab", "abc", true));
        assert!(contains_ci("xxABCxx", "abc", true));
        assert!(contains_ci("ÄBC", "äbc", false));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn the_filter_signature_changes_with_every_input_of_the_filter() {
        let (a, _) = crossbeam::channel::unbounded();
        let (b, _) = crossbeam::channel::unbounded();
        let (c, _) = crossbeam::channel::unbounded();
        let mut v = FileTableViewer::new(a, b, c);
        let base = v.filter_signature();
        assert_eq!(v.filter_signature(), base);
        let mut seen = vec![base];
        v.filter = "fox".into();
        seen.push(v.filter_signature());
        v.types_show_images = false;
        seen.push(v.filter_signature());
        v.types_show_videos = false;
        seen.push(v.filter_signature());
        v.types_show_dirs = false;
        seen.push(v.filter_signature());
        v.mode = ExplorerMode::Database;
        seen.push(v.filter_signature());
        let mut unique = seen.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), seen.len());
    }
}
