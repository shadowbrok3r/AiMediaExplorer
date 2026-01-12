use crate::database::{db_activity, db_set_detail};
use surrealdb::types::{RecordId, Datetime, SurrealValue};
use chrono::Utc;
use crate::DB;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, SurrealValue)]
pub struct CachedScan {
    pub id: RecordId,
    pub root: String,
    pub started: Datetime,
    pub finished: Option<Datetime>,
    pub total: Option<u64>,
    pub title: Option<String>,
    // Associate UI-side scan generation id to avoid races across tabs
    pub scan_id: Option<u64>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, SurrealValue)]
pub struct CachedScanItem {
    pub id: RecordId,
    pub scan_ref: RecordId,
    pub path: String,
    pub created: Datetime,
}

// Typed payloads for inserts/updates
#[derive(Debug, Clone, serde::Serialize, SurrealValue)]
struct CachedScanNew {
    root: String,
    started: Datetime,
    title: Option<String>,
    scan_id: u64,
}

#[derive(Debug, Clone, serde::Serialize, SurrealValue)]
struct CachedScanUpdate {
    finished: Datetime,
    total: u64,
}

#[derive(Debug, Clone, serde::Serialize, SurrealValue)]
struct CachedScanItemNew {
    scan_ref: RecordId,
    path: String,
    created: Datetime,
}

impl CachedScan {
    pub async fn create(root: &str, title: Option<String>, scan_id: u64) -> anyhow::Result<Self, anyhow::Error> {
        let _ga = db_activity("Create cached scan");
        db_set_detail(format!("start: {}", root));
        let started: Datetime = Utc::now().into();
        let row: Option<Self> = DB
            .create("cached_scans")
            .content(CachedScanNew {
                root: root.to_string(),
                started,
                title,
                scan_id,
            })
            .await?;
        row.ok_or_else(|| anyhow::anyhow!("create cached_scans returned empty"))
    }

    pub async fn mark_finished(&self, total: u64) -> anyhow::Result<Self, anyhow::Error> {
        let _ga = db_activity("Finish cached scan");
        db_set_detail("finish");
        let finished: Datetime = Utc::now().into();
        let row: Option<Self> = DB
            .update((&self.id).clone())
            .merge(CachedScanUpdate { finished, total })
            .await?;
        row.ok_or_else(|| anyhow::anyhow!("update cached_scans returned empty"))
    }

    pub async fn list_recent(limit: usize) -> anyhow::Result<Vec<Self>, anyhow::Error> {
        let _ga = db_activity("List recent scans");
        db_set_detail("recent scans");
        let q = format!("SELECT * FROM cached_scans ORDER BY started DESC LIMIT {}", limit);
        let rows: Vec<Self> = DB.query(q).await?.take(0)?;
        Ok(rows)
    }

    pub async fn get_by_scan_id(scan_id: u64) -> anyhow::Result<Option<Self>, anyhow::Error> {
        let _ga = db_activity("Get cached scan by scan_id");
        db_set_detail(format!("scan_id={}", scan_id));
        let q = "SELECT * FROM cached_scans WHERE scan_id = $sid ORDER BY started DESC LIMIT 1";
        let rows: Vec<Self> = DB
            .query(q)
            .bind(("sid", scan_id))
            .await?
            .take(0)?;
        Ok(rows.into_iter().next())
    }
}

impl CachedScanItem {
    pub async fn add_many(scan_ref: &RecordId, paths: Vec<String>) -> anyhow::Result<usize, anyhow::Error> {
        let _ga = db_activity("Add cached scan items");
        db_set_detail(format!("{} items", paths.len()));
        let created: Datetime = Utc::now().into();
        // Insert in small batches to limit payload size
        let chunk = 1000usize;
        let mut total = 0usize;
        for batch in paths.chunks(chunk) {
            let mut values: Vec<CachedScanItemNew> = Vec::with_capacity(batch.len());
            for p in batch.iter() {
                values.push(CachedScanItemNew {
                    scan_ref: scan_ref.clone(),
                    path: p.clone(),
                    created: created.clone(),
                });
            }
            let _: Option<Self> = DB.create("cached_scan_items").content(values).await?;
            total += batch.len();
        }
        Ok(total)
    }

    pub async fn list_paths(scan_ref: &RecordId, offset: usize, limit: usize) -> anyhow::Result<Vec<String>, anyhow::Error> {
        let _ga = db_activity("List cached scan items");
        db_set_detail("scan items");
        let q = "SELECT VALUE path FROM cached_scan_items WHERE scan_ref = $r LIMIT $l START $o";
        let rows: Vec<String> = DB.query(q)
            .bind(("r", scan_ref.clone()))
            .bind(("l", limit as i64))
            .bind(("o", offset as i64))
            .await?
            .take(0)?;
        Ok(rows)
    }
}
