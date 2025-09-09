use serde::{Deserialize, Serialize};
use surrealdb::RecordId;

use crate::DB;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LogicalGroup {
    pub id: Option<RecordId>,
    pub name: String,
    pub thumbnails: Vec<RecordId>,
}

impl LogicalGroup {
    pub async fn list_unassigned_thumbnail_ids() -> anyhow::Result<Vec<RecordId>, anyhow::Error> {
        // Get all thumbnail ids which are not referenced by any logical group
        let ids: Vec<RecordId> = DB
            .query(
                r#"
                LET $used = array::flatten(SELECT VALUE thumbnails FROM logical_groups);
                RETURN (
                    SELECT VALUE id FROM thumbnails WHERE $used == NONE OR array::find($used, id) == NONE
                );
                "#,
            )
            .await?
            .take(0)?;
        Ok(ids)
    }
    pub async fn list_all() -> anyhow::Result<Vec<LogicalGroup>, anyhow::Error> {
        let groups: Vec<LogicalGroup> = DB
            .query("SELECT * FROM logical_groups ORDER BY name ASC")
            .await?
            .take(0)?;
        Ok(groups)
    }

    pub async fn get_by_name(name: &str) -> anyhow::Result<Option<LogicalGroup>, anyhow::Error> {
        let g: Option<LogicalGroup> = DB
            .query("SELECT * FROM logical_groups WHERE name = $name")
            .bind(("name", name.to_string()))
            .await?
            .take(0)?;
        Ok(g)
    }

    pub async fn get_by_id(id: &RecordId) -> anyhow::Result<Option<LogicalGroup>, anyhow::Error> {
        let g: Option<LogicalGroup> = DB
            .query("SELECT * FROM $id")
            .bind(("id", id.clone()))
            .await?
            .take(0)?;
        Ok(g)
    }

    pub async fn create(name: &str) -> anyhow::Result<LogicalGroup, anyhow::Error> {
        let g: Option<LogicalGroup> = DB
            .create("logical_groups")
            .content::<LogicalGroup>(LogicalGroup { id: None, name: name.to_string(), thumbnails: vec![] })
            .await?
            .take();
        Ok(g.unwrap_or_default())
    }

    pub async fn add_thumbnails(id: &RecordId, thumbs: &[RecordId]) -> anyhow::Result<(), anyhow::Error> {
        // Merge unique
        DB
            .query(
                r#"
                UPDATE $gid MERGE { 
                    updated: time::now(),
                    thumbnails: $new_ids.group()
                }
                "#,
            )
            .bind(("gid", id.clone()))
            .bind(("new_ids", thumbs.to_vec()))
            .await?;
        Ok(())
    }

    pub async fn remove_thumbnail(id: &RecordId, thumb_id: &RecordId) -> anyhow::Result<(), anyhow::Error> {
        DB
            .query(
                r#"
                UPDATE $gid 
                    SET thumbnails = array::remove(thumbnails, $tid), 
                    updated = time::now();
                "#,
            )
            .bind(("gid", id.clone()))
            .bind(("tid", thumb_id.clone()))
            .await?;
        Ok(())
    }

    pub async fn rename(id: &RecordId, new_name: &str) -> anyhow::Result<(), anyhow::Error> {
        DB
            .query(
                r#"
                UPDATE $gid SET name = $name, updated = time::now();
                "#,
            )
            .bind(("gid", id.clone()))
            .bind(("name", new_name.to_string()))
            .await?;
        Ok(())
    }

    pub async fn delete(id: &RecordId) -> anyhow::Result<(), anyhow::Error> {
        DB
            .query("DELETE $gid")
            .bind(("gid", id.clone()))
            .await?;
        Ok(())
    }
}

impl crate::Thumbnail {
    pub async fn fetch_by_ids(ids: Vec<RecordId>) -> anyhow::Result<Vec<Self>, anyhow::Error> {
        if ids.is_empty() {
            return Ok(vec![]);
        }
        let rows: Vec<Self> = DB
            .query("SELECT * FROM $ids")
            .bind(("ids", ids))
            .await?
            .take(0)?;
        Ok(rows)
    }
}
