//! The Info tab: file facts, library status, collections and the CLIP and description status.

use egui::{Grid, RichText, Ui};
use humansize::DECIMAL;

use super::facts::{FileFacts, format_duration, format_time};
use super::state::Inspector;
use super::target::{has_description, is_dir, is_image_path, is_video_path};
use super::widgets::{chip, kv};
use crate::Thumbnail;
use crate::ui::file_table::FileExplorer;
use crate::ui::file_table::table::cells::has_clip;

/// "Library" line: whether the file has a row in the database.
pub fn library_text(insp: &Inspector, path: &str) -> String {
    match insp.library.get(path) {
        Some(lib) if lib.facts.indexed => "In the library".to_string(),
        Some(lib) if lib.error.is_some() => format!("Unavailable: {}", lib.error.as_deref().unwrap_or_default()),
        Some(_) => "Not in the library; the first metadata edit adds it".to_string(),
        None => "Checking...".to_string(),
    }
}

fn modified_text(row: &Thumbnail, facts: Option<&FileFacts>) -> String {
    if let Some(t) = facts.and_then(|f| f.modified) {
        return format_time(t);
    }
    match &row.modified {
        Some(dt) => {
            let utc: chrono::DateTime<chrono::Utc> = dt.clone().into();
            utc.with_timezone(&chrono::Local).format("%Y-%m-%d %H:%M").to_string()
        }
        None => "-".to_string(),
    }
}

fn size_text(row: &Thumbnail, facts: Option<&FileFacts>) -> String {
    match facts.and_then(|f| f.size).or((row.size > 0).then_some(row.size)) {
        Some(bytes) => format!("{} ({bytes} bytes)", humansize::format_size(bytes, DECIMAL)),
        None => "-".to_string(),
    }
}

pub fn show(ui: &mut Ui, insp: &Inspector, ex: &FileExplorer, row: &Thumbnail) {
    let facts = insp.facts.get(&row.path);
    let ext = std::path::Path::new(&row.path).extension().and_then(|e| e.to_str()).unwrap_or("").to_uppercase();
    let kind = if is_dir(row) {
        "folder"
    } else if row.file_type == "<ARCHIVE>" {
        "archive"
    } else if is_video_path(&row.path) {
        "video"
    } else if is_image_path(&row.path) {
        "image"
    } else {
        "file"
    };
    Grid::new("inspector_info").num_columns(2).spacing([14.0, 5.0]).show(ui, |ui| {
        kv(ui, "Name", row.filename.as_str());
        kv(ui, "Path", row.path.as_str());
        kv(ui, "Type", if ext.is_empty() { kind.to_string() } else { format!("{ext} {kind}") });
        if !is_dir(row) {
            kv(ui, "Size", size_text(row, facts));
        }
        kv(ui, "Modified", modified_text(row, facts));
        if let Some(created) = facts.and_then(|f| f.created) {
            kv(ui, "Created", format_time(created));
        }
        if let Some((w, h)) = facts.and_then(|f| f.dims) {
            kv(ui, "Dimensions", format!("{w} x {h} px"));
        }
        if let Some(video) = facts.and_then(|f| f.video.as_ref()) {
            if let Some(d) = video.duration_secs {
                kv(ui, "Duration", format_duration(d));
            }
            let codec = video.codec.clone().unwrap_or_default();
            let fps = video.fps.map(|f| format!("{f:.2} fps")).unwrap_or_default();
            let line = [codec, fps].into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join(", ");
            if !line.is_empty() {
                kv(ui, "Video", line);
            }
        }
        if !is_dir(row) {
            let hash = row.hash.clone().or_else(|| insp.library.get(&row.path).and_then(|l| l.facts.hash.clone()));
            kv(ui, "Hash", hash.unwrap_or_else(|| "not computed".to_string()));
            kv(ui, "Library", library_text(insp, &row.path));
            ui.label(RichText::new("Collections").weak());
            match insp.library.get(&row.path) {
                Some(lib) if !lib.collections.is_empty() => {
                    ui.horizontal_wrapped(|ui| {
                        for name in &lib.collections {
                            chip(ui, name, None, false, false);
                        }
                    });
                }
                Some(_) => {
                    ui.label("none");
                }
                None => {
                    ui.label("...");
                }
            }
            ui.end_row();
            kv(ui, "CLIP", if has_clip(&ex.viewer, row) { "Embedding present" } else { "No embedding yet" });
            kv(ui, "Description", if has_description(row) { "Described" } else { "Not described" });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_show_a_readable_and_an_exact_value() {
        let mut row = Thumbnail::new("a.jpg");
        row.size = 1_500_000;
        assert_eq!(size_text(&row, None), "1.50 MB (1500000 bytes)");
        row.size = 0;
        assert_eq!(size_text(&row, None), "-");
        let facts = FileFacts { size: Some(2000), ..Default::default() };
        assert_eq!(size_text(&row, Some(&facts)), "2 kB (2000 bytes)");
    }
}
