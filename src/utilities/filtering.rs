use chrono::{DateTime, Local, NaiveDate};

use crate::utilities::types::{DateField, Filters, MediaKind};

pub trait FiltersExt {
    /// Returns true if the extension passes include toggles and is not excluded for the given mode
    fn ext_allowed(&self, ext: &str, recursive: bool) -> bool;
    /// Returns true if a file size satisfies min/max bounds
    fn size_ok(&self, size: u64) -> bool;
    /// Returns true if the given modified/created timestamps fall within configured range
    fn date_ok(&self, modified: Option<DateTime<Local>>, created: Option<DateTime<Local>>, recursive: bool) -> bool;
    /// When skip_icons is enabled, return false for likely icon/asset files (tiny size or .ico)
    fn skip_icons_heuristic_allows(&self, path: &std::path::Path, size: u64) -> bool;
    /// Strict variant used in UI: rejects .ico and tiny-sized images that are also tiny in dimensions (<= 64x64),
    /// but does NOT reject non-image files solely due to size.
    fn skip_icons_strict_allows(&self, path: &std::path::Path, size: u64) -> bool;
    /// Combined check using path+basic metadata; cheap and avoids allocating a full FoundFile when filtered out
    fn allow_file_attrs(&self, path: &std::path::Path, size: u64, modified: Option<DateTime<Local>>, created: Option<DateTime<Local>>, recursive: bool) -> bool;
    /// Convenience: evaluate media kind against include toggles (images/videos/archives)
    fn kind_allowed(&self, kind: &MediaKind, is_archive_file: bool) -> bool;
}

impl FiltersExt for Filters {
    fn ext_allowed(&self, ext: &str, recursive: bool) -> bool {
        let e = ext.to_ascii_lowercase();
        // Respect recursive excluded extensions only during recursive scans
        if recursive && self.recursive_excluded_exts.contains(&e) { return false; }
        // Limit to supported media families controlled by include toggles
        if crate::utilities::types::is_image(&e) { return self.include_images; }
        if crate::utilities::types::is_video(&e) { return self.include_videos; }
        if crate::utilities::types::is_archive(&e) { return self.include_archives; }
        false
    }

    fn size_ok(&self, size: u64) -> bool {
        if let Some(minb) = self.min_size_bytes { if size < minb { return false; } }
        if let Some(maxb) = self.max_size_bytes { if size > maxb { return false; } }
        true
    }

    fn date_ok(&self, modified: Option<DateTime<Local>>, created: Option<DateTime<Local>>, recursive: bool) -> bool {
        // Resolve effective after/before with recursive overrides
        let after = if recursive { self.recursive_modified_after.as_ref().or(self.modified_after.as_ref()) } else { self.modified_after.as_ref() }
            .and_then(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d").ok())
            .map(|d| d.and_hms_opt(0,0,0).unwrap());
        let before = if recursive { self.recursive_modified_before.as_ref().or(self.modified_before.as_ref()) } else { self.modified_before.as_ref() }
            .and_then(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d").ok())
            .map(|d| d.and_hms_opt(23,59,59).unwrap());

        match self.date_field {
            DateField::Modified => {
                if let Some(m) = modified {
                    if let Some(a) = after { if m.naive_local() < a { return false; } }
                    if let Some(b) = before { if m.naive_local() > b { return false; } }
                }
            }
            DateField::Created => {
                if let Some(c) = created {
                    if let Some(a) = after { if c.naive_local() < a { return false; } }
                    if let Some(b) = before { if c.naive_local() > b { return false; } }
                }
            }
        }
        true
    }

    fn skip_icons_heuristic_allows(&self, path: &std::path::Path, size: u64) -> bool {
        if !self.skip_icons { return true; }
        // quick checks: .ico or tiny size (<= min threshold or 10 KiB default)
        let is_ico = path.extension().and_then(|e| e.to_str()).map(|s| s.eq_ignore_ascii_case("ico")).unwrap_or(false);
        if is_ico { return false; }
        let tiny_thresh = self.min_size_bytes.unwrap_or(10 * 1024);
        if size <= tiny_thresh { return false; }
        true
    }

    fn skip_icons_strict_allows(&self, path: &std::path::Path, size: u64) -> bool {
        if !self.skip_icons { return true; }
        // Always reject .ico
        let is_ico = path.extension().and_then(|e| e.to_str()).map(|s| s.eq_ignore_ascii_case("ico")).unwrap_or(false);
        if is_ico { return false; }
        // If not an image, allow (UI previously only targeted image icons)
        let is_img = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .map(|ext| crate::utilities::types::is_image(ext.as_str()))
            .unwrap_or(false);
        if !is_img { return true; }
        // For images: if under tiny threshold, check dimensions and reject tiny ones
        let tiny_thresh = self.min_size_bytes.unwrap_or(10 * 1024);
        if size <= tiny_thresh {
            let tiny_dims = image::ImageReader::open(path)
                .ok()
                .and_then(|r| r.with_guessed_format().ok())
                .and_then(|r| r.into_dimensions().ok())
                .map(|(w,h)| w <= 64 && h <= 64)
                .unwrap_or(false);
            if tiny_dims { return false; }
        }
        true
    }

    fn allow_file_attrs(&self, path: &std::path::Path, size: u64, modified: Option<DateTime<Local>>, created: Option<DateTime<Local>>, recursive: bool) -> bool {
        // Extension + include toggles
        let ext_ok = path.extension().and_then(|e| e.to_str()).map(|s| self.ext_allowed(s, recursive)).unwrap_or(false);
        if !ext_ok { return false; }
        // Size
        if !self.size_ok(size) { return false; }
        // Skip icons heuristic
        if !self.skip_icons_heuristic_allows(path, size) { return false; }
        // Date range
        if !self.date_ok(modified, created, recursive) { return false; }
        true
    }

    fn kind_allowed(&self, kind: &MediaKind, is_archive_file: bool) -> bool {
        match kind {
            MediaKind::Image => self.include_images,
            MediaKind::Video => self.include_videos,
            MediaKind::Archive => self.include_archives || is_archive_file,
            MediaKind::Other => false,
        }
    }
}
