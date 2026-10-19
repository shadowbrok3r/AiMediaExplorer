//! Putting a result on disk and in the library: a new file, a replaced original, or a library row in a collection.

use std::path::Path;

use crate::ai::mcp::ctx::DbSource;
use crate::database::collections;
use crate::database::journal;
use crate::database::metadata_edit::{self, RowSeed};

use super::model::{is_same_file, unique_output_path};

/// The bytes of `png` in the format the extension of `path` names; PNG bytes pass through unchanged.
pub fn encode_for_path(png: &[u8], path: &Path) -> Result<Vec<u8>, String> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("png").to_ascii_lowercase();
    if ext == "png" {
        return Ok(png.to_vec());
    }
    let format = image::ImageFormat::from_extension(&ext).ok_or_else(|| format!("the file type .{ext} cannot be written"))?;
    let img = image::load_from_memory(png).map_err(|e| format!("the result could not be decoded: {e}"))?;
    let img = if matches!(format, image::ImageFormat::Jpeg | image::ImageFormat::Bmp) { image::DynamicImage::ImageRgb8(img.to_rgb8()) } else { img };
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, format).map_err(|e| format!("the result could not be encoded as .{ext}: {e}"))?;
    Ok(out.into_inner())
}

/// Writes the result to `path`; the folder is created when it is missing. Returns the path as text.
pub fn write_result(png: &[u8], path: &Path) -> Result<String, String> {
    let bytes = encode_for_path(png, path)?;
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    std::fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(path.display().to_string())
}

/// Writes the result next to the source under a name that is not taken.
pub fn save_next_to(png: &[u8], source: &str) -> Result<String, String> {
    write_result(png, &unique_output_path(source, "png", |p| p.exists()))
}

/// Replaces the source file with the result; refuses a path that is not the source.
pub fn overwrite_original(png: &[u8], source: &str, target: &str) -> Result<String, String> {
    if !is_same_file(source, target) {
        return Err("only the file the edit started from can be replaced".to_string());
    }
    write_result(png, Path::new(target))
}

/// Adds the file at `path` to the library, and to the collection `collection` (created when new); returns a sentence for the toast.
pub async fn add_to_library(db: &DbSource, path: &str, collection: Option<&str>) -> Result<String, String> {
    let handle = db.get().map_err(|e| e.message())?;
    let created = metadata_edit::ensure_rows(handle, &[RowSeed::from_path(path)]).await.map_err(|e| format!("{e:#}"))?;
    let name = Path::new(path).file_name().and_then(|n| n.to_str()).unwrap_or(path);
    let mut text = if created > 0 { format!("Added {name} to the library") } else { format!("{name} was already in the library") };
    if let Some(collection) = collection.map(str::trim).filter(|c| !c.is_empty()) {
        let (c, new) = collections::get_or_create(handle, collection).await.map_err(|e| format!("{e:#}"))?;
        journal::add_to_collection_with(handle, &c.id, &[path.to_string()], &format!("Add edit to \"{}\"", c.name)).await.map_err(|e| format!("{e:#}"))?;
        text.push_str(&format!(" and to \"{}\"{}", c.name, if new { " (new collection)" } else { "" }));
    }
    Ok(text)
}

/// Names of the collections, for the choice.
pub async fn collection_names(db: &DbSource) -> Vec<String> {
    let Ok(handle) = db.get() else { return Vec::new() };
    collections::list(handle).await.map(|list| list.into_iter().map(|c| c.name).collect()).unwrap_or_default()
}
