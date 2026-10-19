//! Queries behind the scope loaders: pages of thumbnail rows per scope, generic over the connection so tests run on memory stores.

use std::collections::HashSet;

use surrealdb::types::{RecordId, SurrealValue};
use surrealdb::{Connection, Surreal};

use super::Thumbnail;
use super::collections;
use super::thumbnails::row_cols;
use crate::command::SmartView;

/// Rows read per query.
pub const PAGE: usize = 500;
/// Rows the Recent scope shows: the newest additions.
pub const RECENT_LIMIT: usize = 1_000;

/// One page of a keyset scan in path order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KeysetPage {
    pub rows: Vec<Thumbnail>,
    /// Path to continue after; `None` once the scan has read every row.
    pub next: Option<String>,
}

impl KeysetPage {
    pub fn is_last(&self) -> bool {
        self.next.is_none()
    }
}

/// A row with the columns the duplicate scan needs.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, SurrealValue)]
pub struct SizeRow {
    pub path: String,
    pub size: u64,
    pub hash: Option<String>,
}

/// Rows in the library.
pub async fn count_library<C: Connection>(db: &Surreal<C>) -> anyhow::Result<u64> {
    #[derive(serde::Deserialize, SurrealValue)]
    struct Count {
        n: i64,
    }
    let counts: Vec<Count> = db.query("SELECT count() AS n FROM thumbnails GROUP ALL").await?.take(0)?;
    Ok(counts.first().map_or(0, |c| c.n.max(0) as u64))
}

/// A condition on thumbnail rows and the value it binds.
#[derive(Clone, Copy, Debug)]
enum Condition<'a> {
    All,
    Tag(&'a str),
    Category(&'a str),
    NoDescription,
    NoTags,
}

impl Condition<'_> {
    fn clause(self) -> &'static str {
        match self {
            Self::All => "",
            Self::Tag(_) => "tags CONTAINS $value",
            Self::Category(_) => "category = $value",
            Self::NoDescription => "(description = NONE OR description = '')",
            Self::NoTags => "(tags = NONE OR tags = [])",
        }
    }

    fn value(self) -> Option<String> {
        match self {
            Self::Tag(v) | Self::Category(v) => Some(v.to_string()),
            _ => None,
        }
    }
}

async fn rows_after<C: Connection>(db: &Surreal<C>, condition: Condition<'_>, after: Option<&str>, limit: usize) -> anyhow::Result<Vec<Thumbnail>> {
    let mut clauses: Vec<&str> = Vec::new();
    if !condition.clause().is_empty() {
        clauses.push(condition.clause());
    }
    if after.is_some() {
        clauses.push("path > $after");
    }
    let filter = if clauses.is_empty() { String::new() } else { format!("WHERE {}", clauses.join(" AND ")) };
    let mut query = db.query(format!("SELECT {} FROM thumbnails {filter} ORDER BY path LIMIT $limit", row_cols())).bind(("limit", limit as i64));
    if let Some(value) = condition.value() {
        query = query.bind(("value", value));
    }
    if let Some(after) = after {
        query = query.bind(("after", after.to_string()));
    }
    Ok(query.await?.take(0)?)
}

fn page_of(rows: Vec<Thumbnail>, limit: usize) -> KeysetPage {
    let next = (rows.len() >= limit).then(|| rows.last().map(|r| r.path.clone())).flatten();
    KeysetPage { rows, next }
}

/// Rows of the whole library in path order, `limit` at a time.
pub async fn library_page<C: Connection>(db: &Surreal<C>, after: Option<&str>, limit: usize) -> anyhow::Result<KeysetPage> {
    Ok(page_of(rows_after(db, Condition::All, after, limit).await?, limit))
}

/// Rows that carry `tag`, in path order.
pub async fn tag_page<C: Connection>(db: &Surreal<C>, tag: &str, after: Option<&str>, limit: usize) -> anyhow::Result<KeysetPage> {
    Ok(page_of(rows_after(db, Condition::Tag(tag), after, limit).await?, limit))
}

/// Rows of `category`, in path order.
pub async fn category_page<C: Connection>(db: &Surreal<C>, category: &str, after: Option<&str>, limit: usize) -> anyhow::Result<KeysetPage> {
    Ok(page_of(rows_after(db, Condition::Category(category), after, limit).await?, limit))
}

/// The newest rows by creation time, newest first.
pub async fn recent_rows<C: Connection>(db: &Surreal<C>, limit: usize) -> anyhow::Result<Vec<Thumbnail>> {
    let rows: Vec<Thumbnail> = db
        .query(format!("SELECT {} FROM thumbnails ORDER BY db_created DESC, path LIMIT $limit", row_cols()))
        .bind(("limit", limit as i64))
        .await?
        .take(0)?;
    Ok(rows)
}

fn is_image_row(row: &Thumbnail) -> bool {
    std::path::Path::new(&row.path).extension().and_then(|e| e.to_str()).is_some_and(|e| crate::utilities::types::is_image(&e.to_ascii_lowercase()))
}

/// Paths among `paths` that have a CLIP embedding.
pub async fn embedded_paths<C: Connection>(db: &Surreal<C>, paths: &[String]) -> anyhow::Result<HashSet<String>> {
    let mut present = HashSet::new();
    for chunk in paths.chunks(PAGE) {
        let found: Vec<String> = db.query("SELECT VALUE path FROM clip_embeddings WHERE path IN $paths").bind(("paths", chunk.to_vec())).await?.take(0)?;
        present.extend(found);
    }
    Ok(present)
}

/// Rows of a smart view read in path order until `want` rows matched or the library ended; `NoClip` and `Undescribed` keep images only.
pub async fn smart_page<C: Connection>(db: &Surreal<C>, view: SmartView, after: Option<&str>, want: usize) -> anyhow::Result<KeysetPage> {
    let condition = match view {
        SmartView::Undescribed => Condition::NoDescription,
        SmartView::Untagged => Condition::NoTags,
        SmartView::NoClip => Condition::All,
        SmartView::FailedThumbs | SmartView::Duplicates => anyhow::bail!("{} is not a query over thumbnail rows", view.label()),
    };
    let mut kept: Vec<Thumbnail> = Vec::new();
    let mut cursor = after.map(str::to_string);
    loop {
        let page = rows_after(db, condition, cursor.as_deref(), PAGE).await?;
        let last = page.last().map(|r| r.path.clone());
        let full = page.len() >= PAGE;
        let mut batch = match view {
            SmartView::Untagged => page,
            _ => page.into_iter().filter(is_image_row).collect(),
        };
        if view == SmartView::NoClip && !batch.is_empty() {
            let paths: Vec<String> = batch.iter().map(|r| r.path.clone()).collect();
            let present = embedded_paths(db, &paths).await?;
            batch.retain(|r| !present.contains(&r.path));
        }
        kept.extend(batch);
        cursor = last;
        if !full {
            return Ok(KeysetPage { rows: kept, next: None });
        }
        if kept.len() >= want {
            return Ok(KeysetPage { rows: kept, next: cursor });
        }
    }
}

/// A page of the paths and sizes of the library for the duplicate scan, in path order.
pub async fn size_rows_page<C: Connection>(db: &Surreal<C>, after: Option<&str>, limit: usize) -> anyhow::Result<(Vec<SizeRow>, Option<String>)> {
    let filter = if after.is_some() { "WHERE size > 0 AND path > $after" } else { "WHERE size > 0" };
    let mut query = db.query(format!("SELECT path, size, hash FROM thumbnails {filter} ORDER BY path LIMIT $limit")).bind(("limit", limit as i64));
    if let Some(after) = after {
        query = query.bind(("after", after.to_string()));
    }
    let rows: Vec<SizeRow> = query.await?.take(0)?;
    let next = (rows.len() >= limit).then(|| rows.last().map(|r| r.path.clone())).flatten();
    Ok((rows, next))
}

/// Record of the cached scan with UI scan id `scan_id`; the newest when ids repeat.
pub async fn cached_scan<C: Connection>(db: &Surreal<C>, scan_id: u64) -> anyhow::Result<Option<RecordId>> {
    let found: Vec<RecordId> = db
        .query("SELECT VALUE id FROM cached_scans WHERE scan_id = $sid ORDER BY started DESC LIMIT 1")
        .bind(("sid", scan_id))
        .await?
        .take(0)?;
    Ok(found.into_iter().next())
}

/// The title a cached scan was given, or its root folder.
pub async fn cached_scan_title<C: Connection>(db: &Surreal<C>, scan: &RecordId) -> anyhow::Result<Option<String>> {
    #[derive(serde::Deserialize, SurrealValue)]
    struct Named {
        title: Option<String>,
        root: String,
    }
    let rows: Vec<Named> = db.query("SELECT title, root FROM $id").bind(("id", scan.clone())).await?.take(0)?;
    Ok(rows.into_iter().next().map(|n| n.title.filter(|t| !t.trim().is_empty()).unwrap_or(n.root)))
}

/// A page of the paths a cached scan found, in the order they were recorded.
pub async fn cached_scan_paths<C: Connection>(db: &Surreal<C>, scan: &RecordId, offset: usize, limit: usize) -> anyhow::Result<Vec<String>> {
    let paths: Vec<String> = db
        .query("SELECT VALUE path FROM cached_scan_items WHERE scan_ref = $r ORDER BY created, path LIMIT $l START $o")
        .bind(("r", scan.clone()))
        .bind(("l", limit as i64))
        .bind(("o", offset as i64))
        .await?
        .take(0)?;
    Ok(paths)
}

/// Rows of `paths` that exist in the library, in the order of `paths`.
pub async fn rows_of_paths<C: Connection>(db: &Surreal<C>, paths: &[String]) -> anyhow::Result<Vec<Thumbnail>> {
    collections::rows_for_paths(db, paths).await
}

/// The first stored CLIP embedding of `path`.
pub async fn embedding_of<C: Connection>(db: &Surreal<C>, path: &str) -> anyhow::Result<Option<Vec<f32>>> {
    let found: Vec<Vec<f32>> = db.query("SELECT VALUE embedding FROM clip_embeddings WHERE path = $path").bind(("path", path.to_string())).await?.take(0)?;
    Ok(found.into_iter().find(|e| !e.is_empty()))
}

#[cfg(test)]
pub(crate) mod tests;
