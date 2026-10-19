//! The app's row type: [`Thumbnail`] as a keyed, classified row.

use crate::database::Thumbnail;

use super::store::Keyed;
use super::view::{RowFacts, RowKind};

/// Sentinel `file_type` of directory rows.
pub const DIR_TYPE: &str = "<DIR>";
/// Sentinel `file_type` of archive container rows.
pub const ARCHIVE_TYPE: &str = "<ARCHIVE>";

/// Extension of the last path component, when there is a stem and it is short enough to be a media extension.
pub fn short_extension(path: &str) -> Option<&str> {
    let name = path.rsplit(['/', '\\']).next().unwrap_or(path);
    let (stem, ext) = name.rsplit_once('.')?;
    (!stem.is_empty() && !ext.is_empty() && ext.len() <= 8).then_some(ext)
}

/// Runs `f` on the lowercase form of an ASCII extension without allocating.
pub fn with_lowercase<T>(ext: &str, f: impl FnOnce(&str) -> T) -> T {
    let mut buf = [0u8; 8];
    for (slot, b) in buf.iter_mut().zip(ext.bytes()) {
        *slot = b.to_ascii_lowercase();
    }
    match std::str::from_utf8(&buf[..ext.len().min(8)]) {
        Ok(lower) => f(lower),
        Err(_) => f(""),
    }
}

/// Kind of a file from the extension of `path`.
pub fn kind_of_path(path: &str) -> RowKind {
    let Some(ext) = short_extension(path) else { return RowKind::Other };
    with_lowercase(ext, |lower| {
        if crate::is_image(lower) {
            RowKind::Image
        } else if crate::is_video(lower) {
            RowKind::Video
        } else if crate::is_archive(lower) {
            RowKind::Archive
        } else {
            RowKind::Other
        }
    })
}

/// Kind of a row from its `file_type` sentinel, falling back to the path's extension.
pub fn row_kind(file_type: &str, path: &str) -> RowKind {
    match file_type {
        DIR_TYPE => RowKind::Dir,
        ARCHIVE_TYPE => RowKind::Archive,
        _ => kind_of_path(path),
    }
}

impl Keyed for Thumbnail {
    fn key_path(&self) -> &str {
        &self.path
    }
}

impl RowFacts for Thumbnail {
    fn kind(&self) -> RowKind {
        row_kind(&self.file_type, &self.path)
    }

    fn size(&self) -> u64 {
        self.size
    }
}

/// Bytes a row holds on the heap, for memory estimates.
pub fn heap_bytes(row: &Thumbnail) -> usize {
    row.path.capacity()
        + row.filename.capacity()
        + row.file_type.capacity()
        + row.parent_dir.capacity()
        + row.description.as_ref().map_or(0, |s| s.capacity())
        + row.caption.as_ref().map_or(0, |s| s.capacity())
        + row.category.as_ref().map_or(0, |s| s.capacity())
        + row.hash.as_ref().map_or(0, |s| s.capacity())
        + row.thumbnail_b64.as_ref().map_or(0, |s| s.capacity())
        + row.tags.capacity() * std::mem::size_of::<String>()
        + row.tags.iter().map(|t| t.capacity()).sum::<usize>()
}

/// `count` deterministic rows for profiling: media files in numbered folders with the odd directory row.
pub fn synthetic(count: usize) -> impl Iterator<Item = Thumbnail> {
    const EXTS: [&str; 5] = ["jpg", "png", "mp4", "webp", "gif"];
    (0..count).map(|i| {
        let dir = i / 100;
        let mut t = Thumbnail::new("");
        if i % 997 == 996 {
            t.path = format!("/synthetic/d{dir:04}/sub_{i:06}");
            t.filename = format!("sub_{i:06}");
            t.file_type = DIR_TYPE.to_string();
            return t;
        }
        let ext = EXTS[i % EXTS.len()];
        t.filename = format!("img_{i:06}.{ext}");
        t.path = format!("/synthetic/d{dir:04}/{}", t.filename);
        t.parent_dir = format!("/synthetic/d{dir:04}");
        t.file_type = ext.to_string();
        t.size = 1024 + (i as u64).wrapping_mul(7919) % 9_000_000;
        t.modified = chrono::DateTime::<chrono::Utc>::from_timestamp(1_500_000_000 + i as i64 * 3_600, 0).map(Into::into);
        t.db_created = t.modified.clone();
        t.hash = (i % 2 == 0).then(|| format!("{:016x}", (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
        t.category = (i % 3 == 0).then(|| format!("category{}", i % 7));
        if i % 5 == 0 {
            t.tags = vec![format!("tag{}", i % 11), "common".to_string()];
        }
        t
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(path: &str, file_type: &str) -> Thumbnail {
        let mut t = Thumbnail::default();
        t.path = path.into();
        t.file_type = file_type.into();
        t
    }

    #[test]
    fn extensions_come_from_the_last_component_only() {
        assert_eq!(short_extension("/a.b/c.JPG"), Some("JPG"));
        assert_eq!(short_extension("C:\\x\\y.png"), Some("png"));
        assert_eq!(short_extension("/a/.hidden"), None);
        assert_eq!(short_extension("/a/noext"), None);
        assert_eq!(short_extension("/a/trailing."), None);
        assert_eq!(short_extension("/a/file.waytoolongext"), None);
        assert_eq!(short_extension("zip:///x.zip!/in/p.png"), Some("png"));
    }

    #[test]
    fn kinds_follow_the_extension_case_insensitively() {
        assert_eq!(kind_of_path("/a/b.JPG"), RowKind::Image);
        assert_eq!(kind_of_path("/a/b.Mp4"), RowKind::Video);
        assert_eq!(kind_of_path("/a/b.zip"), RowKind::Archive);
        assert_eq!(kind_of_path("/a/b.txt"), RowKind::Other);
        assert_eq!(kind_of_path("/a/b"), RowKind::Other);
    }

    #[test]
    fn the_file_type_sentinels_win_over_the_extension() {
        assert_eq!(row_kind("<DIR>", "/a/pictures.jpg"), RowKind::Dir);
        assert_eq!(row_kind("<ARCHIVE>", "/a/x.bin"), RowKind::Archive);
        assert_eq!(row_kind("jpg", "/a/x.png"), RowKind::Image);
        assert_eq!(row_kind("", "/a/x.mkv"), RowKind::Video);
    }

    #[test]
    fn thumbnails_are_keyed_and_classified() {
        let t = row("/a/b.png", "png");
        assert_eq!(t.key_path(), "/a/b.png");
        assert_eq!(t.kind(), RowKind::Image);
        let mut sized = row("/a/c.mp4", "mp4");
        sized.size = 77;
        assert_eq!(RowFacts::size(&sized), 77);
    }

    #[test]
    fn synthetic_rows_are_unique_deterministic_and_mixed() {
        let rows: Vec<Thumbnail> = synthetic(5_000).collect();
        let again: Vec<Thumbnail> = synthetic(5_000).collect();
        assert!(rows.iter().zip(&again).all(|(a, b)| a.path == b.path && a.size == b.size));
        let unique: std::collections::HashSet<&str> = rows.iter().map(|r| r.path.as_str()).collect();
        assert_eq!(unique.len(), rows.len());
        assert!(rows.iter().any(|r| r.file_type == DIR_TYPE) && rows.iter().any(|r| r.kind() == RowKind::Video) && rows.iter().any(|r| r.kind() == RowKind::Image));
        assert!(rows.iter().any(|r| !r.tags.is_empty()) && rows.iter().any(|r| r.category.is_some()));
        assert_eq!(synthetic(0).count(), 0);
    }

    #[test]
    fn heap_bytes_counts_strings_and_tags() {
        let mut t = row("/a/b.png", "png");
        let base = heap_bytes(&t);
        t.tags = vec!["alpha".into(), "beta".into()];
        t.description = Some("x".repeat(100));
        assert!(heap_bytes(&t) >= base + 100 + 2 * std::mem::size_of::<String>());
    }
}
