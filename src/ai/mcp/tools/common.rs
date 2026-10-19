//! Row loading and JSON shaping shared by the library tools.

use std::collections::HashMap;

use serde_json::{Value, json};
use surrealdb::types::{Datetime, RecordId, SurrealValue};
use surrealdb::{Connection, Surreal};

use crate::ai::mcp::error::ToolError;

/// Columns of a library row without the inline thumbnail.
pub const ITEM_COLS: &str = "id, path, filename, file_type, size, caption, description, category, tags, modified, parent_dir";

const IN_CHUNK: usize = 400;

/// Characters kept of a description in a list.
pub const LIST_DESCRIPTION_CHARS: usize = 600;
/// Characters kept of a description of a single item.
pub const ITEM_DESCRIPTION_CHARS: usize = 4000;

#[derive(Clone, Debug, SurrealValue)]
pub struct ItemRow {
    pub id: RecordId,
    pub path: String,
    pub filename: Option<String>,
    pub file_type: Option<String>,
    pub size: Option<u64>,
    pub caption: Option<String>,
    pub description: Option<String>,
    pub category: Option<String>,
    pub tags: Option<Vec<String>>,
    pub modified: Option<Datetime>,
    pub parent_dir: Option<String>,
}

#[derive(Debug, SurrealValue)]
pub struct CountRow {
    pub n: i64,
}

/// `text` without control characters and cut to `max` characters.
pub fn safe_text(text: &str, max: usize) -> String {
    let cleaned: String = text.chars().map(|c| if c == '\n' || c == '\t' { ' ' } else { c }).filter(|c| !c.is_control()).collect();
    if cleaned.chars().count() <= max {
        return cleaned;
    }
    cleaned.chars().take(max.saturating_sub(3)).collect::<String>() + "..."
}

/// A requested page size inside `1..=max`, or `default` when absent.
pub fn page_limit(requested: Option<usize>, default: usize, max: usize) -> usize {
    requested.unwrap_or(default).clamp(1, max)
}

fn iso(at: &Option<Datetime>) -> Option<String> {
    at.clone().map(|d| chrono::DateTime::<chrono::Utc>::from(d).to_rfc3339())
}

/// The JSON shape of a library row in lists and details.
pub fn item_json(row: &ItemRow, description_chars: usize) -> Value {
    json!({
        "path": row.path,
        "filename": row.filename.as_deref().map(|f| safe_text(f, 255)),
        "file_type": row.file_type,
        "size_bytes": row.size,
        "caption": row.caption.as_deref().map(|c| safe_text(c, 400)),
        "description": row.description.as_deref().map(|d| safe_text(d, description_chars)),
        "category": row.category.as_deref().map(|c| safe_text(c, 120)),
        "tags": row.tags.as_deref().unwrap_or_default().iter().take(60).map(|t| safe_text(t, 80)).collect::<Vec<_>>(),
        "modified": iso(&row.modified),
        "folder": row.parent_dir,
    })
}

/// Rows for `paths` keyed by path; paths without a row are absent.
pub async fn rows_by_paths<C: Connection>(db: &Surreal<C>, paths: &[String]) -> Result<HashMap<String, ItemRow>, ToolError> {
    let mut out = HashMap::with_capacity(paths.len());
    for chunk in paths.chunks(IN_CHUNK) {
        let rows: Vec<ItemRow> = db.query(format!("SELECT {ITEM_COLS} FROM thumbnails WHERE path IN $paths")).bind(("paths", chunk.to_vec())).await?.take(0)?;
        out.extend(rows.into_iter().map(|r| (r.path.clone(), r)));
    }
    Ok(out)
}

/// The row of one path.
pub async fn row_by_path<C: Connection>(db: &Surreal<C>, path: &str) -> Result<Option<ItemRow>, ToolError> {
    let rows: Vec<ItemRow> = db.query(format!("SELECT {ITEM_COLS} FROM thumbnails WHERE path = $p LIMIT 1")).bind(("p", path.to_string())).await?.take(0)?;
    Ok(rows.into_iter().next())
}

/// `{items, total, offset, limit, next_offset}` around a page of JSON items.
pub fn page_json(items: Vec<Value>, total: u64, offset: usize, limit: usize) -> Value {
    let next = offset + items.len();
    json!({
        "items": items,
        "returned": next - offset,
        "total": total,
        "offset": offset,
        "limit": limit,
        "next_offset": ((next as u64) < total).then_some(next),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_text_strips_controls_and_cuts() {
        assert_eq!(safe_text("a\u{0}b\u{7}c\nd\te", 100), "abc d e");
        let long = safe_text(&"x".repeat(50), 10);
        assert_eq!(long.chars().count(), 10);
        assert!(long.ends_with("..."));
        assert_eq!(safe_text("short", 10), "short");
    }

    #[test]
    fn page_sizes_are_clamped() {
        assert_eq!(page_limit(None, 25, 100), 25);
        assert_eq!(page_limit(Some(0), 25, 100), 1);
        assert_eq!(page_limit(Some(5000), 25, 100), 100);
        assert_eq!(page_limit(Some(7), 25, 100), 7);
    }

    #[test]
    fn a_page_says_where_the_next_one_starts() {
        let page = page_json(vec![json!(1), json!(2)], 5, 0, 2);
        assert_eq!(page["next_offset"], 2);
        assert_eq!(page["returned"], 2);
        let last = page_json(vec![json!(5)], 5, 4, 2);
        assert!(last["next_offset"].is_null());
    }
}
