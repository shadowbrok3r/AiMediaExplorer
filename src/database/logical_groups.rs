use serde::{Deserialize, Serialize};
use surrealdb::RecordId;

use crate::DB;
use crate::database::{db_activity, db_set_detail, db_set_error};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LogicalGroup {
    pub id: Option<RecordId>,
    pub name: String,
}

impl LogicalGroup {
    pub async fn list_unassigned_thumbnail_ids() -> anyhow::Result<Vec<RecordId>, anyhow::Error> {
        // Get all thumbnail ids which are not associated with any logical group (pivoted model)
        let _ga = db_activity("Select unassigned thumbnail ids");
        db_set_detail("Finding thumbnails without a group".to_string());
        let ids: Vec<RecordId> = DB
            .query(
                r#"
                SELECT VALUE id FROM thumbnails WHERE logical_group == NONE
                "#,
            )
            .await?
            .take(0)?;
        Ok(ids)
    }
    pub async fn list_all() -> anyhow::Result<Vec<LogicalGroup>, anyhow::Error> {
        let _ga = db_activity("Select all logical_groups");
        db_set_detail("Loading logical groups".to_string());
        let groups: Vec<LogicalGroup> = DB
            .query("SELECT * FROM logical_groups ORDER BY name ASC")
            .await?
            .take(0)?;
        Ok(groups)
    }

    pub async fn get_by_name(name: &str) -> anyhow::Result<Option<LogicalGroup>, anyhow::Error> {
        let _ga = db_activity("Select logical_group by name");
        db_set_detail(format!("Loading group '{name}'"));
        let g: Option<LogicalGroup> = DB
            .query("SELECT * FROM logical_groups WHERE name = $name")
            .bind(("name", name.to_string()))
            .await?
            .take(0)?;
        Ok(g)
    }

    pub async fn get_by_id(id: &RecordId) -> anyhow::Result<Option<LogicalGroup>, anyhow::Error> {
        let _ga = db_activity("Select logical_group by id");
        db_set_detail("Loading group by id".to_string());
        let g: Option<LogicalGroup> = DB
            .query("SELECT * FROM $id")
            .bind(("id", id.clone()))
            .await?
            .take(0)?;
        Ok(g)
    }

    pub async fn create(name: &str) -> anyhow::Result<LogicalGroup, anyhow::Error> {
        let _ga = db_activity("Create logical_group");
        db_set_detail(format!("Creating group '{name}'"));
        // Use the group name as the deterministic record key for easier lookup: logical_groups:<name>
        let g: Option<LogicalGroup> = DB
            .create(("logical_groups", name))
            .content(serde_json::json!({ "name": name }))
            .await
            .map_err(|e| { db_set_error(format!("Create group failed: {e}")); e })?
            .take();
        Ok(g.unwrap_or_default())
    }

    pub async fn add_thumbnails(id: &RecordId, thumbs: &[RecordId]) -> anyhow::Result<(), anyhow::Error> {
        // Pivoted model: set thumbnails.logical_group to this group's id for all provided ids
        let _ga = db_activity("Update thumbnails logical_group -> add");
        db_set_detail(format!("Assigning {} thumbnails to group", thumbs.len()));
        DB
            .query(
                r#"
                UPDATE thumbnails SET logical_group = $gid WHERE array::find($ids, id) != NONE;
                "#,
            )
            .bind(("gid", id.clone()))
            .bind(("ids", thumbs.to_vec()))
            .await
            .map_err(|e| { db_set_error(format!("Add thumbnails failed: {e}")); e })?;
        Ok(())
    }

    pub async fn remove_thumbnail(id: &RecordId, thumb_id: &RecordId) -> anyhow::Result<(), anyhow::Error> {
        let _ga = db_activity("Update logical_group remove thumbnail");
        db_set_detail("Removing thumbnail from group".to_string());
        DB
            .query(
                r#"
                UPDATE thumbnails SET logical_group = NONE WHERE id == $tid AND logical_group == $gid;
                "#,
            )
            .bind(("gid", id.clone()))
            .bind(("tid", thumb_id.clone()))
            .await
            .map_err(|e| { db_set_error(format!("Remove thumbnail failed: {e}")); e })?;
        Ok(())
    }

    pub async fn rename(id: &RecordId, new_name: &str) -> anyhow::Result<(), anyhow::Error> {
        let _ga = db_activity("Update logical_group rename");
        db_set_detail(format!("Renaming group to '{new_name}'"));
        DB
            .query(
                r#"
                UPDATE $gid SET name = $name, updated = time::now();
                "#,
            )
            .bind(("gid", id.clone()))
            .bind(("name", new_name.to_string()))
            .await
            .map_err(|e| { db_set_error(format!("Rename group failed: {e}")); e })?;
        Ok(())
    }

    pub async fn delete(id: &RecordId) -> anyhow::Result<(), anyhow::Error> {
        let _ga = db_activity("Delete logical_group");
        db_set_detail("Deleting group".to_string());
        // Clear group association on thumbnails first, then delete group
        DB
            .query(
                r#"
                UPDATE thumbnails SET logical_group = NONE WHERE logical_group == $gid;
                DELETE $gid;
                "#,
            )
            .bind(("gid", id.clone()))
            .await
            .map_err(|e| { db_set_error(format!("Delete group failed: {e}")); e })?;
        Ok(())
    }
}

impl crate::Thumbnail {
    pub async fn fetch_by_logical_group_id(id: &RecordId) -> anyhow::Result<Vec<Self>, anyhow::Error> {
        let _ga = db_activity("Select thumbnails by logical_group");
        db_set_detail("Loading thumbnails in group".to_string());
        let rows: Vec<Self> = DB
            .query("SELECT * FROM thumbnails WHERE logical_group == $gid")
            .bind(("gid", id.clone()))
            .await?
            .take(0)?;
        Ok(rows)
    }

    pub async fn fetch_ids_by_logical_group_id(id: &RecordId) -> anyhow::Result<Vec<RecordId>, anyhow::Error> {
        let _ga = db_activity("Select thumbnail ids by logical_group");
        db_set_detail("Loading thumbnail ids in group".to_string());
        let ids: Vec<RecordId> = DB
            .query("SELECT VALUE id FROM thumbnails WHERE logical_group == $gid")
            .bind(("gid", id.clone()))
            .await?
            .take(0)?;
        Ok(ids)
    }
}
