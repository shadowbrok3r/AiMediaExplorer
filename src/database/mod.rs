use crossbeam::channel::Sender;
use surrealdb::{engine::local::Db, opt::{capabilities::Capabilities, Config}, Surreal};
use std::sync::LazyLock;
pub mod settings;
pub mod files;
pub use settings::*;
pub use files::*;

pub static DB: LazyLock<Surreal<Db>> = LazyLock::new(Surreal::init);
// pub static LOCAL_DB: LazyLock<Surreal<Db>> = LazyLock::new(Surreal::init);
pub const NS: &str = "file_explorer";
pub const DB_NAME: &str = "ai_search";
pub const THUMBNAILS: &str = "thumbnails";
pub const USER_SETTINGS: &str = "user_settings";
pub const DB_DEFAULT_TABLE: &str = "./db/default.surql";
pub const DB_BACKUP_PATH: &str = "./db/backup.surql";

// #[derive(serde::Serialize)]
// struct Credentials<'a> {
//     username: &'a str,
//     password: &'a str,
// }

pub async fn new(tx: Sender<()>) -> anyhow::Result<(), anyhow::Error> {
    let capabilities = Capabilities::all().with_all_experimental_features_allowed();
    let config = Config::new().capabilities(capabilities);
    DB.connect::<surrealdb::engine::local::SurrealKv>(("./db/ai_search", config)).await?;
    DB.use_ns(NS).use_db(DB_NAME).await?;
    // DB.signin(Record {
    //     namespace: NS,
    //     database: DB_NAME,
    //     access: "admin",
    //     params: Credentials {
    //         username: "user",
    //         password: "toor"
    //     }
    // }).await?;
    // LOCAL_DB.connect::<surrealdb::engine::local::SurrealKv>(("./db/ai_search1.db", config)).await?;
    // LOCAL_DB.use_ns(NS).use_db(DB_NAME).await?;
    // DB.import(DB_DEFAULT_TABLE).await?;
    
    // DEFINE BUCKET userfiles BACKEND "memory";
    let query = r#" 
        BEGIN;
        DEFINE TABLE IF NOT EXISTS thumbnails TYPE NORMAL SCHEMAFULL PERMISSIONS FULL;
        DEFINE TABLE IF NOT EXISTS user_settings TYPE NORMAL SCHEMAFULL PERMISSIONS FULL;
        DEFINE TABLE IF NOT EXISTS clip_embeddings TYPE NORMAL SCHEMAFULL PERMISSIONS FULL;

        DEFINE FIELD IF NOT EXISTS caption ON thumbnails TYPE option<string> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS category ON thumbnails TYPE option<string> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS db_created ON thumbnails TYPE option<datetime> DEFAULT time::now() PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS description ON thumbnails TYPE option<string> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS embedding ON thumbnails TYPE option<array<float>> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS file_type ON thumbnails TYPE option<string> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS filename ON thumbnails TYPE string PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS hash ON thumbnails TYPE option<string> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS modified ON thumbnails TYPE datetime DEFAULT time::now() PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS path ON thumbnails TYPE string PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS size ON thumbnails TYPE number PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS tags ON thumbnails TYPE option<array<string>> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS thumbnail_b64 ON thumbnails TYPE option<string> PERMISSIONS FULL;

        DEFINE FIELD IF NOT EXISTS thumb_ref ON clip_embeddings TYPE option<record<thumbnails>> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS path ON clip_embeddings TYPE string PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS hash ON clip_embeddings TYPE option<string> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS embedding ON clip_embeddings TYPE array<float> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS created ON clip_embeddings TYPE datetime DEFAULT time::now() PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS updated ON clip_embeddings TYPE datetime DEFAULT time::now() PERMISSIONS FULL;

        DEFINE FIELD IF NOT EXISTS qa_collapsed ON user_settings TYPE bool PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS drives_collapsed ON user_settings TYPE bool PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS preview_collapsed ON user_settings TYPE bool PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS preview_width ON user_settings TYPE number PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS sort ON user_settings TYPE object PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS sort.by ON user_settings TYPE string PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS sort.asc ON user_settings TYPE bool PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS view_mode ON user_settings TYPE option<string> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS left_width ON user_settings TYPE number PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS ext_enabled ON user_settings TYPE option<array<any>> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS excluded_dirs ON user_settings TYPE option<array<string>> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS group_by_category ON user_settings TYPE bool PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS detail_column_widths ON user_settings TYPE option<array<number>> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS category_col_width ON user_settings TYPE option<number> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS auto_indexing ON user_settings TYPE bool PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS ai_prompt_template ON user_settings TYPE string PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS overwrite_descriptions ON user_settings TYPE bool PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS filter_modified_after ON user_settings TYPE option<string> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS filter_modified_before ON user_settings TYPE option<string> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS filter_category_multi ON user_settings TYPE option<array<string>> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS filter_only_with_thumb ON user_settings TYPE bool PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS filter_only_with_description ON user_settings TYPE bool PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS last_root ON user_settings TYPE option<string> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS show_progress_overlay ON user_settings TYPE bool PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS db_min_size_bytes ON user_settings TYPE option<number> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS db_max_size_bytes ON user_settings TYPE option<number> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS db_excluded_exts ON user_settings TYPE option<array<string>> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS auto_clip_embeddings ON user_settings TYPE bool PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS clip_augment_with_text ON user_settings TYPE bool PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS clip_model ON user_settings TYPE option<string> PERMISSIONS FULL;

        DEFINE INDEX IF NOT EXISTS category_idx ON thumbnails FIELDS category;
        DEFINE INDEX IF NOT EXISTS tags_idx ON thumbnails FIELDS tags;
        DEFINE INDEX IF NOT EXISTS path_idx ON thumbnails FIELDS path UNIQUE;
        DEFINE INDEX IF NOT EXISTS clip_path_idx ON clip_embeddings FIELDS path UNIQUE;
        DEFINE INDEX IF NOT EXISTS clip_thumb_ref_idx ON clip_embeddings FIELDS thumb_ref;
        COMMIT;
    "#;
    
    let response = DB.query(query).await?;
    let _ = response.check()?;
    let _ = tx.send(());
    Ok(())
}

pub async fn save_thumbnail_batch(thumbs: Vec<Thumbnail>) -> anyhow::Result<(), anyhow::Error> {
    log::info!("save_thumbnail_batch");
    let _: Vec<crate::Thumbnail> = DB
        .insert("thumbnails")
        .content::<Vec<crate::Thumbnail>>(thumbs)
        .await?;
    Ok(())
}



// Save (insert) a single thumbnail row (best-effort). Does not deduplicate existing rows.
pub async fn save_thumbnail_row(row: Thumbnail) -> anyhow::Result<(), anyhow::Error> {
    log::info!("SAVING: {row:?}");
    let _: Option<crate::Thumbnail> = DB
        .create("thumbnails")
        .content::<crate::Thumbnail>(row)
        .await?
        .take();
    Ok(())
}

// Fetch a single thumbnail row by exact path (leverages UNIQUE index on path)
pub async fn get_thumbnail_by_path(path: &str) -> anyhow::Result<Option<Thumbnail>, anyhow::Error> {
    let sql = "SELECT * FROM thumbnails WHERE path = $path LIMIT 1";
    let mut resp = DB
        .query(sql)
        .bind(("path", path.to_string()))
        .await?;
    let row: Option<Thumbnail> = resp.take(0)?;
    Ok(row)
}

// Fetch thumbnail record id (thumbnails:...) by path
pub async fn get_thumbnail_id_by_path(path: &str) -> anyhow::Result<Option<String>, anyhow::Error> {
    let sql = "SELECT id FROM thumbnails WHERE path = $path LIMIT 1";
    let mut resp = DB.query(sql).bind(("path", path.to_string())).await?;
    let row: Option<surrealdb::sql::Thing> = resp.take(0)?;
    Ok(row.map(|t| t.to_string()))
}

// Upsert a clip embedding row (match on path). Optionally sets thumb_ref if missing.
pub async fn upsert_clip_embedding(path: &str, hash: Option<&str>, embedding: &[f32]) -> anyhow::Result<(), anyhow::Error> {
    // Ensure thumbnail id for FK
    let thumb_id = get_thumbnail_id_by_path(path).await?;
    // UPDATE returns number or row; if none updated we INSERT
    let update_sql = "UPDATE clip_embeddings SET embedding = $embedding, hash = $hash, updated = time::now(), thumb_ref = COALESCE(thumb_ref, $thumb_ref) WHERE path = $path";
    let mut resp = DB.query(update_sql)
        .bind(("embedding", embedding.to_vec()))
        .bind(("hash", hash.map(|h| h.to_string())))
        .bind(("thumb_ref", thumb_id.clone()))
        .bind(("path", path.to_string()))
        .await?;
    let updated: Option<crate::database::ClipEmbeddingRow> = resp.take(0)?; // ignore content
    if updated.is_none() {
        // Insert new
        let _ : Option<crate::database::ClipEmbeddingRow> = DB.create("clip_embeddings")
            .content(crate::database::ClipEmbeddingRow { id: None, thumb_ref: thumb_id, path: path.to_string(), hash: hash.map(|h| h.to_string()), embedding: embedding.to_vec(), created: None, updated: None })
            .await?;
    }
    Ok(())
}

// Load all clip embedding rows and return mapping path -> (embedding, hash, thumb_ref)
pub async fn load_all_clip_embeddings() -> anyhow::Result<Vec<crate::database::ClipEmbeddingRow>, anyhow::Error> {
    let rows: Vec<crate::database::ClipEmbeddingRow> = DB.select("clip_embeddings").await?;
    Ok(rows)
}

pub async fn save_settings(s: UiSettings) -> anyhow::Result<(), anyhow::Error> {
    DB.upsert::<Option<UiSettings>>(UiSettings::default().id).content::<UiSettings>(s).await?;
    // DB.export(DB_BACKUP_PATH).await?;
    Ok(())
}

pub async fn get_settings() -> anyhow::Result<UiSettings, anyhow::Error> {
    let settings_res: Option<UiSettings> = DB.select(UiSettings::default().id).await?;
    log::info!("Got settings: {settings_res:?}");
    if let Some(settings)  = settings_res {
        return Ok(settings);
    } else {
        Ok(UiSettings::default())
    }
}

// Paged DB thumbnail retrieval without ordering. Applies optional filters.
pub async fn load_thumbnails_page(
    offset: usize,
    limit: usize,
    min_size: Option<u64>,
    max_size: Option<u64>,
    modified_after: Option<&str>,
    modified_before: Option<&str>,
    excluded_exts: Option<&[String]>,
    path_prefix: Option<&str>,
) -> anyhow::Result<Vec<Thumbnail>, anyhow::Error> {
    let mut clauses: Vec<String> = Vec::new();
    if let Some(ms) = min_size { clauses.push(format!("size >= {}", ms)); }
    if let Some(mx) = max_size { clauses.push(format!("size <= {}", mx)); }
    if let Some(a) = modified_after { clauses.push(format!("modified >= time::parse('{}')", a)); }
    if let Some(b) = modified_before { clauses.push(format!("modified <= time::parse('{}')", b)); }
    if let Some(exts) = excluded_exts { if !exts.is_empty() { let joined = exts.iter().map(|e| format!("'{}'", e)).collect::<Vec<_>>().join(","); clauses.push(format!("file_type NOT IN [{}]", joined)); } }
    if let Some(prefix) = path_prefix { if !prefix.trim().is_empty() { // simple prefix filter (string starts-with)
        // SurrealDB lacks direct STARTSWITH; use string::starts_with function if available or fallback to LIKE
        // Using LIKE with escaped %; ensure prefix sanitized (no % introduced by user). For large datasets an index on path exists.
        let safe = prefix.replace('%', "");
        clauses.push(format!("path LIKE '{}%'", safe));
    }}
    let where_sql = if clauses.is_empty() { String::new() } else { format!(" WHERE {}", clauses.join(" AND ")) };
    let sql = format!("SELECT * FROM thumbnails{} LIMIT {} START {}", where_sql, limit, offset);
    let mut resp = DB.query(sql).await?;
    let rows: Vec<Thumbnail> = resp.take(0)?;
    Ok(rows)
}




