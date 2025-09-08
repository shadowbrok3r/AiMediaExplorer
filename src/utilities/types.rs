use chrono::{DateTime, Local};
use once_cell::sync::Lazy;
use std::path::PathBuf;

// Supported media extensions
pub static IMAGE_EXTS: Lazy<Vec<&'static str>> = Lazy::new(|| {
    vec![
        "jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp", "heic", "heif", "avif", "arw"
    ]
});

pub static VIDEO_EXTS: Lazy<Vec<&'static str>> = Lazy::new(|| {
    vec![
        "mp4", "mov", "avi", "mkv", "webm", "wmv", "m4v", "flv", "mpeg", "mpg", "3gp",
    ]
});

// Archive containers we want to show alongside media (contents handled separately)
pub static ARCHIVE_EXTS: Lazy<Vec<&'static str>> = Lazy::new(|| {
    vec![
        "zip", // future: add "7z", "rar", "tar", "gz", etc.
    ]
});

pub fn is_image(potential_img: &str) -> bool {
    IMAGE_EXTS.iter().any(|i| *i == potential_img)
}

pub fn is_video(potential_vid: &str) -> bool {
    VIDEO_EXTS.iter().any(|v| *v == potential_vid)
}

pub fn is_supported_media_ext(ext: &str) -> bool {
    let e = ext.to_ascii_lowercase();
    is_image(&e) || is_video(&e) || is_archive(&e)
}

pub fn is_archive(potential_archive: &str) -> bool { ARCHIVE_EXTS.iter().any(|a| *a == potential_archive) }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewMode {
    Icons,
    Details,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DateField {
    Modified,
    #[allow(dead_code)]
    Created,
}

#[derive(Clone, Debug)]
pub struct Filters {
    pub root: PathBuf,
    pub include_images: bool,
    pub include_videos: bool,
    pub modified_after: Option<String>, // YYYY-MM-DD
    pub modified_before: Option<String>,
    pub date_field: DateField,
    pub only_with_thumb: bool, // UI-only filter (applied client-side) to show only items that already have a loaded thumbnail
    pub only_with_description: bool, // UI-only: only show items that have an AI description
    pub category_filter: Option<String>, // If Some(cat) show only that category
    pub category_filters: std::collections::BTreeSet<String>, // Multi-select categories (union filter); empty => all
    // Size filters (bytes). If set, file must satisfy both bounds.
    pub min_size_bytes: Option<u64>,
    pub max_size_bytes: Option<u64>,
    // Recursive scan specific settings (ignored in shallow scans unless noted)
    pub recursive_excluded_dirs: std::collections::BTreeSet<PathBuf>,
    pub recursive_excluded_exts: std::collections::BTreeSet<String>, // lowercase extensions w/out dot
    pub recursive_modified_after: Option<String>, // override date range just for recursive scans
    pub recursive_modified_before: Option<String>,
    // Exclude any file whose (lowercased) path or filename contains one of these substrings.
    pub excluded_terms: Vec<String>,
}

impl Default for Filters {
    fn default() -> Self {
        let root = std::path::absolute(std::env::current_dir().unwrap())
            .unwrap_or_else(|_| PathBuf::from("."));
        Self {
            root,
            include_images: true,
            include_videos: true,
            modified_after: None,
            modified_before: None,
            date_field: DateField::Modified,
            only_with_thumb: false,
            only_with_description: false,
            category_filter: None,
            category_filters: std::collections::BTreeSet::new(),
            min_size_bytes: None,
            max_size_bytes: None,
            recursive_excluded_dirs: std::collections::BTreeSet::new(),
            recursive_excluded_exts: std::collections::BTreeSet::new(),
            recursive_modified_after: None,
            recursive_modified_before: None,
            excluded_terms: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ScanResults {
    pub items: Vec<FoundFile>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MediaKind {
    Image,
    Video,
    Other,
}
impl Default for MediaKind {
    fn default() -> Self {
        MediaKind::Other
    }
}

impl MediaKind {
    pub fn icon_name(&self) -> &'static str {
        match self {
            MediaKind::Image => "photo",
            MediaKind::Video => "smart_display",
            MediaKind::Other => "insert_drive_file",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FoundFile {
    pub path: PathBuf,
    pub modified: Option<DateTime<Local>>,
    pub created: Option<DateTime<Local>>,
    pub size: Option<u64>,
    pub kind: MediaKind,
    pub thumb_data: Option<String>, // data URL for image thumbnails
}

impl FoundFile {
    pub fn icon_name(&self) -> &'static str {
        match self.kind {
            MediaKind::Image => "photo",
            MediaKind::Video => "smart_display",
            MediaKind::Other => "insert_drive_file",
        }
    }
}

#[derive(Clone, Debug)]
pub struct DirItem {
    pub path: PathBuf,
}

#[derive(Clone)]
pub struct QuickAccess {
    pub label: String,
    pub path: PathBuf,
    pub icon: String,
}
