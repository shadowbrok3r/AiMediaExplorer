//! Journaled forms of the metadata writers: bulk tag and category rewrites and collection membership changes.

use std::collections::{HashMap, HashSet};

use surrealdb::types::{RecordId, SurrealValue};
use surrealdb::{Connection, Surreal};

use super::{Field, Recorder};
use crate::database::collections::{self, AddReport};
use crate::database::metadata_edit::MetaPatch;
use crate::database::thumbnails::{DELETE_TAG_SQL, RENAME_TAG_SQL};

const CHUNK: usize = 400;

/// The editable metadata of a row.
#[derive(Clone, Debug, Default, PartialEq, Eq, SurrealValue)]
pub struct MetaSnapshot {
    pub path: String,
    pub caption: Option<String>,
    pub description: Option<String>,
    pub category: Option<String>,
    pub tags: Option<Vec<String>>,
}

/// The metadata of the rows of `paths`, by path; paths without a row are missing.
pub async fn snapshots_with<C: Connection>(db: &Surreal<C>, paths: &[String]) -> anyhow::Result<HashMap<String, MetaSnapshot>> {
    let mut out = HashMap::new();
    for chunk in paths.chunks(CHUNK) {
        let rows: Vec<MetaSnapshot> = db.query("SELECT path, caption, description, category, tags FROM thumbnails WHERE path IN $paths").bind(("paths", chunk.to_vec())).await?.check()?.take(0)?;
        out.extend(rows.into_iter().map(|r| (r.path.clone(), r)));
    }
    Ok(out)
}

/// Records the fields `patch` writes to `path`, given the row as it was before.
pub fn record_patch(rec: &Recorder, path: &str, before: &MetaSnapshot, patch: &MetaPatch) {
    if let Some(v) = &patch.description {
        rec.text(path, Field::Description, before.description.as_deref(), v.as_deref());
    }
    if let Some(v) = &patch.caption {
        rec.text(path, Field::Caption, before.caption.as_deref(), v.as_deref());
    }
    if let Some(v) = &patch.category {
        rec.text(path, Field::Category, before.category.as_deref(), v.as_deref());
    }
    if let Some(v) = &patch.tags {
        rec.tags(path, before.tags.as_deref().unwrap_or(&[]), v);
    }
}

#[derive(Clone, Debug, SurrealValue)]
struct TagRow {
    path: String,
    tags: Option<Vec<String>>,
}

#[derive(Clone, Debug, SurrealValue)]
struct CategoryRow {
    path: String,
    category: Option<String>,
}

async fn commit_tag_rewrite<C: Connection>(db: &Surreal<C>, source: String, before: Vec<TagRow>, after: &[TagRow]) {
    let before_by: HashMap<&str, &[String]> = before.iter().map(|r| (r.path.as_str(), r.tags.as_deref().unwrap_or(&[]))).collect();
    let rec = Recorder::new(&source);
    for a in after {
        rec.tags(&a.path, before_by.get(a.path.as_str()).copied().unwrap_or(&[]), a.tags.as_deref().unwrap_or(&[]));
    }
    rec.commit_with(db).await;
}

/// Renames or merges a tag on every row; returns the rows changed.
pub async fn rename_tag_with<C: Connection>(db: &Surreal<C>, old: &str, new: &str) -> anyhow::Result<u64> {
    let before: Vec<TagRow> = db.query("SELECT path, tags FROM thumbnails WHERE tags CONTAINS $old").bind(("old", old.to_string())).await?.check()?.take(0)?;
    let after: Vec<TagRow> = db.query(format!("{RENAME_TAG_SQL} RETURN path, tags")).bind(("old", old.to_string())).bind(("new", new.to_string())).await?.check()?.take(0)?;
    commit_tag_rewrite(db, format!("Rename tag \"{old}\" to \"{new}\""), before, &after).await;
    Ok(after.len() as u64)
}

/// Removes a tag from every row; returns the rows changed.
pub async fn delete_tag_with<C: Connection>(db: &Surreal<C>, tag: &str) -> anyhow::Result<u64> {
    let before: Vec<TagRow> = db.query("SELECT path, tags FROM thumbnails WHERE tags CONTAINS $tag").bind(("tag", tag.to_string())).await?.check()?.take(0)?;
    let after: Vec<TagRow> = db.query(format!("{DELETE_TAG_SQL} RETURN path, tags")).bind(("tag", tag.to_string())).await?.check()?.take(0)?;
    commit_tag_rewrite(db, format!("Delete tag \"{tag}\""), before, &after).await;
    Ok(after.len() as u64)
}

/// Keeps at most `limit` tags per row; returns the rows changed.
pub async fn prune_tags_with<C: Connection>(db: &Surreal<C>, limit: i64) -> anyhow::Result<u64> {
    let before: Vec<TagRow> = db.query("SELECT path, tags FROM thumbnails WHERE array::len(tags) > $limit").bind(("limit", limit)).await?.check()?.take(0)?;
    let after: Vec<TagRow> = db
        .query("UPDATE thumbnails SET tags = array::slice(tags, 0, $limit) WHERE array::len(tags) > $limit RETURN path, tags")
        .bind(("limit", limit))
        .await?
        .check()?
        .take(0)?;
    commit_tag_rewrite(db, format!("Limit tags to {limit} per file"), before, &after).await;
    Ok(after.len() as u64)
}

/// Renames or merges a category on every row; returns the rows changed.
pub async fn rename_category_with<C: Connection>(db: &Surreal<C>, old: &str, new: &str) -> anyhow::Result<u64> {
    let after: Vec<CategoryRow> = db
        .query("UPDATE thumbnails SET category = $new WHERE category = $old RETURN path, category")
        .bind(("old", old.to_string()))
        .bind(("new", new.to_string()))
        .await?
        .check()?
        .take(0)?;
    let rec = Recorder::new(&format!("Rename category \"{old}\" to \"{new}\""));
    for a in &after {
        rec.text(&a.path, Field::Category, Some(old), a.category.as_deref());
    }
    rec.commit_with(db).await;
    Ok(after.len() as u64)
}

/// Paths of `paths` that are members of the collection.
async fn members_among<C: Connection>(db: &Surreal<C>, id: &RecordId, paths: &[String]) -> anyhow::Result<HashSet<String>> {
    let mut found = HashSet::new();
    for chunk in paths.chunks(CHUNK) {
        let rows: Vec<String> = db
            .query("SELECT VALUE path FROM collection_items WHERE collection = $c AND path IN $paths")
            .bind(("c", id.clone()))
            .bind(("paths", chunk.to_vec()))
            .await?
            .check()?
            .take(0)?;
        found.extend(rows);
    }
    Ok(found)
}

fn distinct_paths(paths: &[String]) -> Vec<&String> {
    let mut seen = HashSet::new();
    paths.iter().filter(|p| !p.is_empty() && seen.insert(p.as_str())).collect()
}

/// Adds paths to a collection and journals the memberships that are new.
pub async fn add_to_collection_with<C: Connection>(db: &Surreal<C>, id: &RecordId, paths: &[String], source: &str) -> anyhow::Result<AddReport> {
    let present = members_among(db, id, paths).await.unwrap_or_default();
    let report = collections::add_paths(db, id, paths).await?;
    let key = collections::key_of(id);
    let rec = Recorder::new(source);
    for p in distinct_paths(paths).into_iter().filter(|p| !present.contains(*p)) {
        rec.member(p, &key, false, true);
    }
    rec.commit_with(db).await;
    Ok(report)
}

/// Copies every member of `from` into `to` and journals the memberships that are new; returns how many were added.
pub async fn copy_collection_with<C: Connection>(db: &Surreal<C>, from: &RecordId, to: &RecordId, source: &str) -> anyhow::Result<usize> {
    let mut paths: Vec<String> = Vec::new();
    loop {
        let page = collections::members(db, from, paths.len(), 2000).await?;
        if page.is_empty() {
            break;
        }
        paths.extend(page);
    }
    Ok(add_to_collection_with(db, to, &paths, source).await?.added)
}

/// Removes paths from a collection and journals the memberships that existed.
pub async fn remove_from_collection_with<C: Connection>(db: &Surreal<C>, id: &RecordId, paths: &[String], source: &str) -> anyhow::Result<usize> {
    let present = members_among(db, id, paths).await.unwrap_or_default();
    let removed = collections::remove_paths(db, id, paths).await?;
    let key = collections::key_of(id);
    let rec = Recorder::new(source);
    for p in distinct_paths(paths).into_iter().filter(|p| present.contains(*p)) {
        rec.member(p, &key, true, false);
    }
    rec.commit_with(db).await;
    Ok(removed)
}
