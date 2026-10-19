//! Logical group API kept for existing call sites; groups resolve to collections, and `Default` means filed nowhere.

use serde::{Deserialize, Serialize};
use surrealdb::types::{RecordId, SurrealValue};

use crate::{DB, LOGICAL_GROUPS};
use crate::database::collections::{self, UNFILED_NAME};
use crate::database::{db_activity, db_set_detail, db_set_error};

const ID_CHUNK: usize = 500;

#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct LogicalGroup {
    pub id: RecordId,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, SurrealValue)]
struct LogicalGroupNew { name: String }

impl Default for LogicalGroup {
    fn default() -> Self {
        Self {
            id: RecordId::new(LOGICAL_GROUPS, "Default"),
            name: "Default".to_string()
        }
    }
}

impl From<collections::Collection> for LogicalGroup {
    fn from(c: collections::Collection) -> Self {
        Self { id: c.id, name: c.name }
    }
}

impl LogicalGroup {
    /// True for the entry that stands for "filed nowhere".
    pub fn is_unfiled(&self) -> bool {
        self.id == Self::default().id
    }

    /// Creates the legacy `Default` group record when the legacy table is empty.
    pub async fn ensure_default() -> anyhow::Result<(), anyhow::Error> {
        let any: Vec<RecordId> = DB.query("SELECT VALUE id FROM logical_groups LIMIT 1").await?.take(0)?;
        if any.is_empty() {
            let _: Option<LogicalGroup> = DB
                .create((LOGICAL_GROUPS, UNFILED_NAME))
                .content(LogicalGroupNew { name: UNFILED_NAME.to_string() })
                .await?
                .take();
        }
        Ok(())
    }

    /// Ids of thumbnail rows that are in no collection.
    pub async fn list_unassigned_thumbnail_ids() -> anyhow::Result<Vec<RecordId>, anyhow::Error> {
        let _ga = db_activity("Select unassigned thumbnail ids");
        db_set_detail("Finding thumbnails without a collection".to_string());
        let paths = collections::unfiled_paths(&DB, usize::MAX).await?;
        thumbnail_ids_for_paths(&paths).await
    }

    /// The unfiled entry followed by every collection.
    pub async fn list_all() -> anyhow::Result<Vec<LogicalGroup>, anyhow::Error> {
        let _ga = db_activity("Select all collections");
        db_set_detail("Loading collections".to_string());
        let mut groups = vec![LogicalGroup::default()];
        groups.extend(collections::list(&DB).await?.into_iter().map(LogicalGroup::from));
        Ok(groups)
    }

    pub async fn get_by_name(name: &str) -> anyhow::Result<Option<LogicalGroup>, anyhow::Error> {
        if name == UNFILED_NAME {
            return Ok(Some(LogicalGroup::default()));
        }
        Ok(collections::get_by_name(&DB, name).await?.map(LogicalGroup::from))
    }

    pub async fn get_by_id(id: &RecordId) -> anyhow::Result<Option<LogicalGroup>, anyhow::Error> {
        let _ga = db_activity("Select collection by id");
        db_set_detail("Loading collection by id".to_string());
        if collections::is_collection_id(id) {
            return Ok(collections::get(&DB, id).await?.map(LogicalGroup::from));
        }
        let g: Option<LogicalGroup> = DB
            .query("SELECT * FROM $id")
            .bind(("id", id.clone()))
            .await?
            .take(0)?;
        Ok(g)
    }

    /// Creates a collection; the reserved `Default` name only makes sure the unfiled entry exists.
    pub async fn create(name: &str) -> anyhow::Result<LogicalGroup, anyhow::Error> {
        let _ga = db_activity("Create collection");
        db_set_detail(format!("Creating collection '{name}'"));
        if name.trim().eq_ignore_ascii_case(UNFILED_NAME) {
            Self::ensure_default().await?;
            return Ok(LogicalGroup::default());
        }
        collections::create(&DB, name).await.map(LogicalGroup::from).inspect_err(|e| db_set_error(format!("Create collection failed: {e}")))
    }

    /// Adds the files of the given thumbnail rows to a collection; adding to the unfiled entry does nothing.
    pub async fn add_thumbnails(id: &RecordId, thumbs: &[RecordId]) -> anyhow::Result<(), anyhow::Error> {
        let _ga = db_activity("Add thumbnails to collection");
        db_set_detail(format!("Adding {} files to collection", thumbs.len()));
        if !collections::is_collection_id(id) {
            return Ok(());
        }
        let paths = paths_for_thumbnail_ids(thumbs).await?;
        super::journal::add_to_collection_with(&DB, id, &paths, "Add to collection").await.map(|_| ()).inspect_err(|e| db_set_error(format!("Add thumbnails failed: {e}")))
    }

    /// Removes the file of one thumbnail row from a collection.
    pub async fn remove_thumbnail(id: &RecordId, thumb_id: &RecordId) -> anyhow::Result<(), anyhow::Error> {
        let _ga = db_activity("Remove thumbnail from collection");
        db_set_detail("Removing file from collection".to_string());
        if !collections::is_collection_id(id) {
            return Ok(());
        }
        let paths = paths_for_thumbnail_ids(std::slice::from_ref(thumb_id)).await?;
        super::journal::remove_from_collection_with(&DB, id, &paths, "Remove from collection").await.map(|_| ()).inspect_err(|e| db_set_error(format!("Remove thumbnail failed: {e}")))
    }

    pub async fn rename(id: &RecordId, new_name: &str) -> anyhow::Result<(), anyhow::Error> {
        let _ga = db_activity("Rename collection");
        db_set_detail(format!("Renaming collection to '{new_name}'"));
        if !collections::is_collection_id(id) {
            return Ok(());
        }
        collections::rename(&DB, id, new_name).await.map(|_| ()).inspect_err(|e| db_set_error(format!("Rename collection failed: {e}")))
    }

    /// Deletes a collection and its memberships; files and thumbnails are untouched.
    pub async fn delete(id: &RecordId) -> anyhow::Result<(), anyhow::Error> {
        let _ga = db_activity("Delete collection");
        db_set_detail("Deleting collection".to_string());
        if !collections::is_collection_id(id) {
            return Ok(());
        }
        collections::delete(&DB, id).await.map(|_| ()).inspect_err(|e| db_set_error(format!("Delete collection failed: {e}")))
    }
}

/// Paths of the given thumbnail rows.
async fn paths_for_thumbnail_ids(ids: &[RecordId]) -> anyhow::Result<Vec<String>> {
    let mut paths = Vec::with_capacity(ids.len());
    for chunk in ids.chunks(ID_CHUNK) {
        let found: Vec<String> = DB.query("SELECT VALUE path FROM $ids").bind(("ids", chunk.to_vec())).await?.take(0)?;
        paths.extend(found);
    }
    Ok(paths)
}

/// Ids of the thumbnail rows with the given paths.
async fn thumbnail_ids_for_paths(paths: &[String]) -> anyhow::Result<Vec<RecordId>> {
    let mut ids = Vec::with_capacity(paths.len());
    for chunk in paths.chunks(ID_CHUNK) {
        let found: Vec<RecordId> = DB.query("SELECT VALUE id FROM thumbnails WHERE path IN $paths").bind(("paths", chunk.to_vec())).await?.take(0)?;
        ids.extend(found);
    }
    Ok(ids)
}

impl crate::Thumbnail {
    /// Rows of a collection's members; other ids read the legacy `logical_group` link.
    pub async fn fetch_by_logical_group_id(id: &RecordId) -> anyhow::Result<Vec<Self>, anyhow::Error> {
        let _ga = db_activity("Select thumbnails by collection");
        db_set_detail("Loading thumbnails in collection".to_string());
        if collections::is_collection_id(id) {
            return Ok(collections::all_member_rows(&DB, id).await?.rows);
        }
        let rows: Vec<Self> = DB
            .query(format!("SELECT {} FROM thumbnails WHERE logical_group == $gid", super::thumbnails::row_cols()))
            .bind(("gid", id.clone()))
            .await?
            .take(0)?;
        Ok(rows)
    }

    pub async fn fetch_ids_by_logical_group_id(id: &RecordId) -> anyhow::Result<Vec<RecordId>, anyhow::Error> {
        let _ga = db_activity("Select thumbnail ids by collection");
        db_set_detail("Loading thumbnail ids in collection".to_string());
        if collections::is_collection_id(id) {
            let mut ids = Vec::new();
            let mut offset = 0;
            loop {
                let page = collections::members(&DB, id, offset, 2000).await?;
                if page.is_empty() {
                    return Ok(ids);
                }
                offset += page.len();
                ids.extend(thumbnail_ids_for_paths(&page).await?);
            }
        }
        let ids: Vec<RecordId> = DB
            .query("SELECT VALUE id FROM thumbnails WHERE logical_group == $gid")
            .bind(("gid", id.clone()))
            .await?
            .take(0)?;
        Ok(ids)
    }
}
