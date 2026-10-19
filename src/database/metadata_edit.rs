//! Metadata edits of the inspector: field patches that may clear a field, rows created on the first edit, and batch tag and category changes.

use std::collections::{HashMap, HashSet};

use anyhow::Context;
use chrono::Utc;
use surrealdb::types::{Datetime, RecordId, SurrealValue};
use surrealdb::{Connection, Surreal};

use super::files::Thumbnail;
use super::thumb_blobs::retry_conflicts;
use super::{LogicalGroup, THUMBNAILS};

/// Rows looked up or written per statement.
const CHUNK: usize = 400;

/// The fields needed to create the row of a file that is not in the library yet.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RowSeed {
    pub path: String,
    pub filename: String,
    pub file_type: String,
    pub size: u64,
    pub parent_dir: String,
    pub hash: Option<String>,
    pub modified: Option<Datetime>,
}

impl RowSeed {
    pub fn from_row(row: &Thumbnail) -> Self {
        let parent = std::path::Path::new(&row.path).parent().map(|p| p.to_string_lossy().to_string()).unwrap_or_default();
        Self {
            path: row.path.clone(),
            filename: row.filename.clone(),
            file_type: row.file_type.clone(),
            size: row.size,
            parent_dir: if row.parent_dir.is_empty() { parent } else { row.parent_dir.clone() },
            hash: row.hash.clone(),
            modified: row.modified.clone(),
        }
    }

    fn into_row(self) -> Thumbnail {
        Thumbnail {
            id: RecordId::new(THUMBNAILS, surrealdb_types::Uuid::new_v4().to_string()),
            db_created: Some(Utc::now().into()),
            path: self.path,
            filename: self.filename,
            file_type: self.file_type,
            size: self.size,
            description: None,
            caption: None,
            tags: Vec::new(),
            category: None,
            thumbnail_b64: None,
            modified: self.modified,
            hash: self.hash,
            parent_dir: self.parent_dir,
            logical_group: LogicalGroup::default().id,
        }
    }
}

/// New values for some fields of a row; `Some(None)` clears a text field.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MetaPatch {
    pub description: Option<Option<String>>,
    pub caption: Option<Option<String>>,
    pub category: Option<Option<String>>,
    pub tags: Option<Vec<String>>,
}

impl MetaPatch {
    pub fn is_empty(&self) -> bool {
        self.description.is_none() && self.caption.is_none() && self.category.is_none() && self.tags.is_none()
    }

    /// Names of the fields the patch writes.
    pub fn fields(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.description.is_some() {
            out.push("description");
        }
        if self.caption.is_some() {
            out.push("caption");
        }
        if self.category.is_some() {
            out.push("category");
        }
        if self.tags.is_some() {
            out.push("tags");
        }
        out
    }

    /// The patch with text trimmed (blank becomes cleared) and tags normalized.
    pub fn cleaned(&self) -> Self {
        let text = |v: &Option<Option<String>>| v.as_ref().map(|inner| inner.as_deref().map(str::trim).filter(|s| !s.is_empty()).map(str::to_string));
        Self { description: text(&self.description), caption: text(&self.caption), category: text(&self.category), tags: self.tags.as_ref().map(|t| normalize_tags(t)) }
    }
}

/// What a save did besides updating fields.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SaveOutcome {
    /// The file had no row and one was created.
    pub created: bool,
}

/// Tags to add to and remove from every row of a batch.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TagChange {
    pub add: Vec<String>,
    pub remove: Vec<String>,
}

impl TagChange {
    pub fn is_empty(&self) -> bool {
        self.add.is_empty() && self.remove.is_empty()
    }
}

/// What a batch did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BatchReport {
    /// Rows created because the file was not in the library.
    pub created: usize,
    /// Rows whose tags or category were written.
    pub updated: usize,
    /// Tags now stored per path, for rows whose tags changed.
    pub tags: HashMap<String, Vec<String>>,
}

/// A tag with whitespace trimmed and collapsed; `None` when nothing is left.
pub fn normalize_tag(raw: &str) -> Option<String> {
    let collapsed = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    (!collapsed.is_empty()).then_some(collapsed)
}

/// Normalized tags with repeats removed ignoring case; the first spelling stays.
pub fn normalize_tags<S: AsRef<str>>(tags: impl IntoIterator<Item = S>) -> Vec<String> {
    let mut seen = HashSet::new();
    tags.into_iter().filter_map(|t| normalize_tag(t.as_ref())).filter(|t| seen.insert(t.to_lowercase())).collect()
}

/// `existing` without the removed tags (ignoring case), then the added ones that were not already there.
pub fn apply_tag_change(existing: &[String], change: &TagChange) -> Vec<String> {
    let remove: HashSet<String> = change.remove.iter().filter_map(|t| normalize_tag(t)).map(|t| t.to_lowercase()).collect();
    let kept = existing.iter().filter(|t| !remove.contains(&t.trim().to_lowercase()));
    normalize_tags(kept.cloned().chain(change.add.iter().cloned()))
}

fn distinct_seeds(seeds: &[RowSeed]) -> Vec<&RowSeed> {
    let mut seen = HashSet::new();
    seeds.iter().filter(|s| !s.path.is_empty() && seen.insert(s.path.as_str())).collect()
}

async fn existing_paths<C: Connection>(db: &Surreal<C>, paths: &[&str]) -> anyhow::Result<HashSet<String>> {
    let mut found = HashSet::new();
    for chunk in paths.chunks(CHUNK) {
        let rows: Vec<String> = db
            .query("SELECT VALUE path FROM thumbnails WHERE path IN $paths")
            .bind(("paths", chunk.iter().map(|p| p.to_string()).collect::<Vec<_>>()))
            .await?
            .check()
            .context("looking up the library rows failed")?
            .take(0)?;
        found.extend(rows);
    }
    Ok(found)
}

/// Creates the rows of the seeds that have none; returns how many were created.
pub async fn ensure_rows<C: Connection>(db: &Surreal<C>, seeds: &[RowSeed]) -> anyhow::Result<usize> {
    let unique = distinct_seeds(seeds);
    let paths: Vec<&str> = unique.iter().map(|s| s.path.as_str()).collect();
    let present = existing_paths(db, &paths).await?;
    let missing: Vec<Thumbnail> = unique.into_iter().filter(|s| !present.contains(&s.path)).map(|s| s.clone().into_row()).collect();
    let mut created = 0;
    for chunk in missing.chunks(CHUNK) {
        created += retry_conflicts(|| async {
            let ids: Vec<RecordId> = db
                .query("INSERT IGNORE INTO thumbnails $rows RETURN VALUE id")
                .bind(("rows", chunk.to_vec()))
                .await?
                .check()
                .context("creating the library rows failed")?
                .take(0)?;
            Ok(ids.len())
        })
        .await?;
    }
    Ok(created)
}

/// Writes the fields of `patch` to the row of `seed.path`, creating the row first when the file is not in the library.
pub async fn save_patch<C: Connection>(db: &Surreal<C>, seed: &RowSeed, patch: &MetaPatch) -> anyhow::Result<SaveOutcome> {
    anyhow::ensure!(!seed.path.is_empty(), "the file has no path");
    let created = ensure_rows(db, std::slice::from_ref(seed)).await? > 0;
    let patch = patch.cleaned();
    if patch.is_empty() {
        return Ok(SaveOutcome { created });
    }
    let sets: Vec<String> = patch.fields().iter().map(|f| format!("{f} = ${f}")).collect();
    let sql = format!("UPDATE thumbnails SET {} WHERE path = $path RETURN VALUE id", sets.join(", "));
    let ids: Vec<RecordId> = retry_conflicts(|| async {
        let mut query = db.query(&sql).bind(("path", seed.path.clone()));
        if let Some(v) = &patch.description {
            query = query.bind(("description", v.clone()));
        }
        if let Some(v) = &patch.caption {
            query = query.bind(("caption", v.clone()));
        }
        if let Some(v) = &patch.category {
            query = query.bind(("category", v.clone()));
        }
        if let Some(v) = &patch.tags {
            query = query.bind(("tags", v.clone()));
        }
        let ids: Vec<RecordId> = query.await?.check().context("saving the metadata failed")?.take(0)?;
        Ok(ids)
    })
    .await?;
    anyhow::ensure!(!ids.is_empty(), "the library row of {} disappeared before it could be saved", seed.path);
    Ok(SaveOutcome { created })
}

#[derive(Clone, Debug, SurrealValue)]
struct TagRow {
    path: String,
    tags: Option<Vec<String>>,
}

#[derive(Clone, Debug, SurrealValue)]
struct TagUpdate {
    path: String,
    tags: Vec<String>,
}

/// Adds and removes tags on the rows of `seeds`, creating missing rows; the report lists the tags now stored.
pub async fn batch_tags<C: Connection>(db: &Surreal<C>, seeds: &[RowSeed], change: &TagChange) -> anyhow::Result<BatchReport> {
    let unique = distinct_seeds(seeds);
    anyhow::ensure!(!unique.is_empty(), "no files to change");
    let created = ensure_rows(db, seeds).await?;
    let mut report = BatchReport { created, ..Default::default() };
    if change.is_empty() {
        return Ok(report);
    }
    for chunk in unique.chunks(CHUNK) {
        let paths: Vec<String> = chunk.iter().map(|s| s.path.clone()).collect();
        let rows: Vec<TagRow> = db.query("SELECT path, tags FROM thumbnails WHERE path IN $paths").bind(("paths", paths)).await?.check()?.take(0)?;
        let updates: Vec<TagUpdate> = rows
            .into_iter()
            .filter_map(|r| {
                let before = r.tags.unwrap_or_default();
                let after = apply_tag_change(&before, change);
                (after != before).then_some(TagUpdate { path: r.path, tags: after })
            })
            .collect();
        if updates.is_empty() {
            continue;
        }
        retry_conflicts(|| async {
            db.query("FOR $u IN $updates { UPDATE thumbnails SET tags = $u.tags WHERE path = $u.path; };")
                .bind(("updates", updates.clone()))
                .await?
                .check()
                .context("saving the tags failed")?;
            Ok(())
        })
        .await?;
        report.updated += updates.len();
        report.tags.extend(updates.into_iter().map(|u| (u.path, u.tags)));
    }
    Ok(report)
}

/// Sets (or with `None` clears) the category of the rows of `seeds`, creating missing rows.
pub async fn batch_category<C: Connection>(db: &Surreal<C>, seeds: &[RowSeed], category: Option<&str>) -> anyhow::Result<BatchReport> {
    let unique = distinct_seeds(seeds);
    anyhow::ensure!(!unique.is_empty(), "no files to change");
    let category = category.map(str::trim).filter(|c| !c.is_empty()).map(str::to_string);
    let created = ensure_rows(db, seeds).await?;
    let mut report = BatchReport { created, ..Default::default() };
    for chunk in unique.chunks(CHUNK) {
        let paths: Vec<String> = chunk.iter().map(|s| s.path.clone()).collect();
        let ids: Vec<RecordId> = retry_conflicts(|| async {
            let ids: Vec<RecordId> = db
                .query("UPDATE thumbnails SET category = $category WHERE path IN $paths RETURN VALUE id")
                .bind(("category", category.clone()))
                .bind(("paths", paths.clone()))
                .await?
                .check()
                .context("saving the category failed")?
                .take(0)?;
            Ok(ids)
        })
        .await?;
        report.updated += ids.len();
    }
    Ok(report)
}

/// Distinct tags of all rows, for suggestions.
pub async fn all_tags<C: Connection>(db: &Surreal<C>) -> anyhow::Result<Vec<String>> {
    let rows: Vec<Vec<String>> = db.query("SELECT VALUE tags FROM thumbnails WHERE tags != NONE").await?.check()?.take(0)?;
    let mut all: Vec<String> = rows.into_iter().flatten().collect();
    all.sort_by_key(|t| (t.to_lowercase(), t.clone()));
    let mut tags = normalize_tags(all);
    tags.sort_by_key(|t| (t.to_lowercase(), t.clone()));
    Ok(tags)
}

/// Distinct categories of all rows, for suggestions.
pub async fn all_categories<C: Connection>(db: &Surreal<C>) -> anyhow::Result<Vec<String>> {
    let rows: Vec<String> = db.query("SELECT VALUE category FROM thumbnails WHERE category != NONE GROUP BY category").await?.check()?.take(0)?;
    let mut rows = rows;
    rows.sort_by_key(|c| (c.to_lowercase(), c.clone()));
    let mut out = normalize_tags(rows);
    out.sort_by_key(|c| (c.to_lowercase(), c.clone()));
    Ok(out)
}

/// What the library knows about one file.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LibraryFacts {
    pub indexed: bool,
    pub hash: Option<String>,
    pub description: bool,
}

#[derive(Clone, Debug, SurrealValue)]
struct FactsRow {
    hash: Option<String>,
    description: Option<String>,
}

/// Library row facts of `path`.
pub async fn library_facts<C: Connection>(db: &Surreal<C>, path: &str) -> anyhow::Result<LibraryFacts> {
    let rows: Vec<FactsRow> = db.query("SELECT hash, description FROM thumbnails WHERE path = $path LIMIT 1").bind(("path", path.to_string())).await?.check()?.take(0)?;
    Ok(match rows.into_iter().next() {
        Some(r) => LibraryFacts { indexed: true, hash: r.hash, description: r.description.is_some_and(|d| !d.trim().is_empty()) },
        None => LibraryFacts::default(),
    })
}

#[cfg(test)]
mod tests;
