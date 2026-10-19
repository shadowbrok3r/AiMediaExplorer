//! Logical group API kept for existing call sites; groups resolve to collections, and `Default` means filed nowhere.

use serde::{Deserialize, Serialize};
use surrealdb::types::{RecordId, SurrealValue};

use crate::{DB, LOGICAL_GROUPS};
use crate::database::collections::{self, UNFILED_NAME};
use crate::database::{db_activity, db_set_detail, db_set_error};

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
