//! Loads converted files into a SurrealDB 3.x handle and reads the target's existing rows for merge semantics.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use surrealdb::Surreal;
use surrealdb::engine::local::Db;
use surrealdb::types::{RecordId, RecordIdKey, SurrealValue, ToSql};

use super::convert::{ConvertedFile, ExistingIndex};
use super::job::{Job, PhaseId};
use crate::database::surql;

const PAGE: usize = 5000;

pub const COUNTED_TABLES: [&str; 9] = ["thumbnails", "clip_embeddings", "logical_groups", "filter_groups", "cached_scans", "cached_scan_items", "assistant_sessions", "assistant_messages", "user_settings"];

pub fn key_string(id: &RecordId) -> String {
    match &id.key {
        RecordIdKey::String(s) => s.clone(),
        RecordIdKey::Number(n) => n.to_string(),
        RecordIdKey::Uuid(u) => u.to_string(),
        other => other.to_sql(),
    }
}

#[derive(Debug, Deserialize, SurrealValue)]
struct IdPath {
    id: RecordId,
    path: String,
}

#[derive(Debug, Deserialize, SurrealValue)]
struct IdName {
    id: RecordId,
    name: String,
}

async fn ids_of(db: &Surreal<Db>, table: &str) -> Result<Vec<RecordId>> {
    Ok(db.query("SELECT VALUE id FROM type::table($t)").bind(("t", table.to_string())).await?.take(0)?)
}

/// Row counts of the tables the import touches.
pub async fn table_counts(db: &Surreal<Db>) -> Result<BTreeMap<String, u64>> {
    let mut out = BTreeMap::new();
    for t in COUNTED_TABLES {
        out.insert(t.to_string(), surql::count_rows(db, t).await?);
    }
    Ok(out)
}

async fn paged_id_path(db: &Surreal<Db>, table: &str, job: &Job, mut each: impl FnMut(IdPath)) -> Result<()> {
    let mut start = 0usize;
    loop {
        job.check_cancel()?;
        let rows: Vec<IdPath> = db
            .query("SELECT id, path FROM type::table($t) LIMIT $lim START $start")
            .bind(("t", table.to_string()))
            .bind(("lim", PAGE as i64))
            .bind(("start", start as i64))
            .await?
            .take(0)?;
        let n = rows.len();
        rows.into_iter().for_each(&mut each);
        if n < PAGE {
            return Ok(());
        }
        start += PAGE;
    }
}

/// Reads ids, paths and names of the rows already in the target store.
pub async fn load_existing(db: &Surreal<Db>, settings_key: &str, job: &Job) -> Result<ExistingIndex> {
    let mut ex = ExistingIndex::default();
    job.note("Reading the target store's existing thumbnails and embeddings");
    paged_id_path(db, "thumbnails", job, |r| {
        let k = key_string(&r.id);
        ex.thumb_ids.insert(k.clone());
        ex.thumb_by_path.insert(r.path, k);
    })
    .await?;
    paged_id_path(db, "clip_embeddings", job, |r| {
        ex.clip_ids.insert(key_string(&r.id));
        ex.clip_paths.insert(r.path);
    })
    .await?;
    let groups: Vec<IdName> = db.query("SELECT id, name FROM logical_groups").await?.take(0)?;
    for g in groups {
        ex.group_by_name.insert(g.name, key_string(&g.id));
    }
    let filters: Vec<IdName> = db.query("SELECT id, name FROM filter_groups").await?.take(0)?;
    ex.filter_names.extend(filters.into_iter().map(|f| f.name));
    ex.session_ids = ids_of(db, "assistant_sessions").await?.iter().map(key_string).collect();
    ex.message_ids = ids_of(db, "assistant_messages").await?.iter().map(key_string).collect();
    ex.settings_present = ids_of(db, "user_settings").await?.iter().any(|i| key_string(i) == settings_key);
    Ok(ex)
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct FileTiming {
    pub file: String,
    pub bytes: u64,
    pub millis: u128,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ImportStats {
    pub files: usize,
    pub bytes: u64,
    pub millis: u128,
    pub per_file: Vec<FileTiming>,
}

/// Imports the converted files in order through the engine's own import path.
pub async fn import_files(db: &Surreal<Db>, dir: &Path, files: &[ConvertedFile], job: &Arc<Job>) -> Result<ImportStats> {
    let total: u64 = files.iter().map(|f| f.bytes).sum();
    job.begin(PhaseId::Import, total, "importing into the 3.x store");
    let started = Instant::now();
    let mut stats = ImportStats::default();
    let mut done = 0u64;
    for f in files {
        job.check_cancel()?;
        job.update(PhaseId::Import, done, total, &format!("importing {}", f.file));
        let t = Instant::now();
        let path = dir.join(&f.file);
        db.import(&path).await.with_context(|| format!("importing {} failed", f.file))?;
        done += f.bytes;
        stats.per_file.push(FileTiming { file: f.file.clone(), bytes: f.bytes, millis: t.elapsed().as_millis() });
        job.update(PhaseId::Import, done, total, &format!("imported {}", f.file));
    }
    stats.files = files.len();
    stats.bytes = total;
    stats.millis = started.elapsed().as_millis();
    job.finish(PhaseId::Import);
    Ok(stats)
}
