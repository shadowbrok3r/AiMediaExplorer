//! Post-import checks: per-table counts, indexes, typed deserialization, KNN smoke queries and idempotency.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use surrealdb::Surreal;
use surrealdb::engine::local::Db;
use surrealdb::types::{RecordId, SurrealValue};

use super::convert::{self, ConvertOptions, ConvertReport};
use super::import::{self, COUNTED_TABLES};
use super::job::{Job, PhaseId};
use super::pathmap::Config;
use crate::database::surql;

const TYPECHECK_PAGE: usize = 500;
const META_COLS: &str = "id, db_created, path, filename, file_type, size, description, caption, tags, category, modified, hash, parent_dir, logical_group";
const REQUIRED_INDEXES: [(&str, &str); 4] = [("thumbnails", "path_idx"), ("clip_embeddings", "clip_path_idx"), ("logical_groups", "lg_name_idx"), ("clip_embeddings", "idx_clip_hnsw")];

pub struct VerifySpec {
    pub before: BTreeMap<String, u64>,
    pub settings_digest_before: Option<String>,
    pub knn_samples: usize,
    pub idempotent: Option<(ConvertOptions, Config)>,
    pub settings_key: String,
    pub live: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct CountCheck {
    pub before: u64,
    pub expected_added: u64,
    pub after: u64,
    pub ok: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct TypeCheck {
    pub thumbnails: u64,
    pub clip_embeddings: u64,
    pub logical_groups: u64,
    pub wrong_dimension: u64,
    pub errors: Vec<String>,
    pub millis: u128,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct KnnCheck {
    pub query: String,
    pub hits: usize,
    pub top1_is_self: bool,
    pub millis: u128,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct IdempotentCheck {
    pub new_rows: BTreeMap<String, u64>,
    pub ok: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct VerifyReport {
    pub ok: bool,
    pub problems: Vec<String>,
    pub counts: BTreeMap<String, CountCheck>,
    pub manifest_counts: BTreeMap<String, u64>,
    pub indexes_ok: bool,
    pub typecheck: TypeCheck,
    pub knn: Vec<KnnCheck>,
    pub settings_untouched: Option<bool>,
    pub idempotent: Option<IdempotentCheck>,
    pub notes: Vec<String>,
    pub millis: u128,
}

#[derive(Debug, Deserialize, SurrealValue)]
struct KnnHit {
    path: String,
    dist: f32,
}

#[derive(Debug, Deserialize, SurrealValue)]
struct SampleClip {
    path: String,
    embedding: Vec<f32>,
}

/// Blake3 of the settings record as JSON, or "none" when absent; the content is never exposed.
pub async fn settings_digest(db: &Surreal<Db>, key: &str) -> Option<String> {
    let id = RecordId::new("user_settings", key);
    let mut resp = db.query("SELECT * FROM ONLY $id").bind(("id", id)).await.ok()?;
    let v: surrealdb::types::Value = resp.take(0).ok()?;
    let j = v.into_json_value();
    if j.is_null() {
        return Some("none".into());
    }
    Some(blake3::hash(j.to_string().as_bytes()).to_hex().to_string())
}

async fn typecheck(db: &Surreal<Db>, job: &Job) -> Result<TypeCheck> {
    let started = Instant::now();
    let mut t = TypeCheck::default();
    let mut start = 0i64;
    loop {
        job.check_cancel()?;
        let res = db.query(format!("SELECT {META_COLS} FROM thumbnails LIMIT $lim START $start")).bind(("lim", TYPECHECK_PAGE as i64)).bind(("start", start)).await?.take::<Vec<crate::Thumbnail>>(0);
        match res {
            Ok(rows) => {
                t.thumbnails += rows.len() as u64;
                if rows.len() < TYPECHECK_PAGE {
                    break;
                }
            }
            Err(e) => {
                t.errors.push(format!("thumbnails at offset {start}: {e}"));
                break;
            }
        }
        start += TYPECHECK_PAGE as i64;
    }
    let dim = convert::schema_dimension(crate::database::SCHEMA_DDL);
    start = 0;
    loop {
        job.check_cancel()?;
        let res = db.query("SELECT * FROM clip_embeddings LIMIT $lim START $start").bind(("lim", 200i64)).bind(("start", start)).await?.take::<Vec<crate::ClipEmbeddingRow>>(0);
        match res {
            Ok(rows) => {
                t.clip_embeddings += rows.len() as u64;
                t.wrong_dimension += rows.iter().filter(|r| r.embedding.len() != dim).count() as u64;
                if rows.len() < 200 {
                    break;
                }
            }
            Err(e) => {
                t.errors.push(format!("clip_embeddings at offset {start}: {e}"));
                break;
            }
        }
        start += 200;
    }
    match db.query("SELECT * FROM logical_groups ORDER BY name ASC").await?.take::<Vec<crate::LogicalGroup>>(0) {
        Ok(g) => t.logical_groups = g.len() as u64,
        Err(e) => t.errors.push(format!("logical_groups: {e}")),
    }
    t.millis = started.elapsed().as_millis();
    Ok(t)
}

fn short_path(p: &str) -> String {
    let parts: Vec<&str> = p.rsplit(['/', '\\']).take(2).collect();
    parts.into_iter().rev().collect::<Vec<_>>().join("/")
}

async fn knn_smoke(db: &Surreal<Db>, n: usize, total: u64) -> Result<Vec<KnnCheck>> {
    let mut out = Vec::new();
    if total == 0 {
        return Ok(out);
    }
    for i in 0..n.min(total as usize) {
        let offset = (total as usize * i / n.max(1)) as i64;
        let rows: Vec<SampleClip> = db.query("SELECT path, embedding FROM clip_embeddings LIMIT 1 START $start").bind(("start", offset)).await?.take(0)?;
        let Some(sample) = rows.into_iter().next() else { continue };
        let started = Instant::now();
        let hits: Vec<KnnHit> = db
            .query("SELECT path, vector::distance::knn() AS dist FROM clip_embeddings WHERE embedding <|5,64|> $vec ORDER BY dist")
            .bind(("vec", sample.embedding.clone()))
            .await?
            .take(0)?;
        let top = hits.first();
        out.push(KnnCheck {
            query: short_path(&sample.path),
            hits: hits.len(),
            top1_is_self: top.is_some_and(|h| h.path == sample.path || h.dist < 1e-4),
            millis: started.elapsed().as_millis(),
        });
    }
    Ok(out)
}

/// Verifies the target after an import; `convert_report` holds the row counts the import was expected to add.
pub async fn verify_import(db: &Surreal<Db>, spec: &VerifySpec, convert_report: &ConvertReport, job: &Arc<Job>) -> Result<VerifyReport> {
    let started = Instant::now();
    job.begin(PhaseId::Verify, 5, "checking counts");
    let mut rep = VerifyReport::default();
    let manifest = surql::build_manifest_with(db, &crate::database::snapshot::no_progress(), false).await?;
    rep.manifest_counts = manifest.counts();
    for t in COUNTED_TABLES {
        let before = spec.before.get(t).copied().unwrap_or(0);
        let expected_added = if t == "user_settings" { 0 } else { convert_report.rows_out(t) };
        let after = manifest.tables.get(t).map_or(0, |d| d.count);
        let grew = spec.live && after > before + expected_added;
        let ok = after == before + expected_added || grew;
        if grew {
            rep.notes.push(format!("{t}: the open database gained {} rows from other activity while importing", after - before - expected_added));
        }
        if !ok {
            rep.problems.push(format!("{t}: expected {before} + {expected_added} rows, found {after}"));
        }
        rep.counts.insert(t.to_string(), CountCheck { before, expected_added, after, ok });
    }
    job.update(PhaseId::Verify, 1, 5, "checking indexes");
    rep.indexes_ok = true;
    for (table, idx) in REQUIRED_INDEXES {
        let present = manifest.tables.get(table).is_some_and(|d| d.indexes.iter().any(|x| x.contains(idx)));
        if !present {
            rep.indexes_ok = false;
            rep.problems.push(format!("index {idx} is missing on {table}"));
        }
    }
    job.update(PhaseId::Verify, 2, 5, "deserializing rows into the app's types");
    rep.typecheck = typecheck(db, job).await?;
    if !rep.typecheck.errors.is_empty() {
        rep.problems.extend(rep.typecheck.errors.iter().map(|e| format!("typecheck: {e}")));
    }
    if rep.typecheck.wrong_dimension > 0 {
        rep.problems.push(format!("{} embeddings have the wrong dimension", rep.typecheck.wrong_dimension));
    }
    job.update(PhaseId::Verify, 3, 5, "running similarity queries");
    let clips = rep.counts.get("clip_embeddings").map_or(0, |c| c.after);
    rep.knn = knn_smoke(db, spec.knn_samples, clips).await?;
    if clips > 0 && (rep.knn.is_empty() || rep.knn.iter().any(|k| k.hits == 0 || !k.top1_is_self)) {
        rep.problems.push("a nearest-neighbour smoke query returned no hit or did not find the query vector".into());
    }
    if let Some(before) = &spec.settings_digest_before {
        let after = settings_digest(db, &spec.settings_key).await;
        let same = after.as_deref() == Some(before.as_str());
        rep.settings_untouched = Some(same);
        let merged = !convert_report.settings.merged_fields.is_empty();
        if !same && !merged {
            rep.problems.push("the target's user settings changed although no settings fields were requested".into());
        }
    }
    job.update(PhaseId::Verify, 4, 5, "checking that a second run adds nothing");
    if let Some((opts, cfg)) = &spec.idempotent {
        let existing = import::load_existing(db, &spec.settings_key, job).await?;
        let quiet = Job::new();
        let again = convert::convert(opts, cfg, Some(&existing), &quiet)?;
        let mut new_rows = BTreeMap::new();
        for t in COUNTED_TABLES.iter().filter(|t| **t != "user_settings") {
            let n = again.rows_out(t);
            if n > 0 {
                new_rows.insert(t.to_string(), n);
            }
        }
        let ok = new_rows.is_empty();
        if !ok {
            rep.problems.push(format!("a second run would still add rows: {new_rows:?}"));
        }
        rep.idempotent = Some(IdempotentCheck { new_rows, ok });
    }
    rep.ok = rep.problems.is_empty();
    rep.millis = started.elapsed().as_millis();
    job.finish(PhaseId::Verify);
    Ok(rep)
}
