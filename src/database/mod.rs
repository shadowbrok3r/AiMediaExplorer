use crossbeam::channel::Sender;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use crate::ui::status::{GlobalStatusIndicator, StatusState, DB_STATUS};
use surrealdb::{Surreal, engine::local::Db};
use std::fs;
use std::path::{Path, PathBuf};
pub mod clip_embeddings;
pub mod files;
pub mod settings;
pub mod thumbnails;
pub mod logical_groups;
pub mod filter_groups;
pub mod cached_scans;
pub mod assistant_chat;
pub use clip_embeddings::*;
pub use files::*;
pub use settings::*;
pub use thumbnails::*;
pub use logical_groups::*;
pub use filter_groups::*;
pub use cached_scans::*;

pub static DB: LazyLock<Surreal<Db>> = LazyLock::new(Surreal::init);
pub const NS: &str = "file_explorer";
pub const DB_NAME: &str = "ai_search";
pub const THUMBNAILS: &str = "thumbnails";
pub const USER_SETTINGS: &str = "user_settings";
pub const FILTER_GROUPS: &str = "filter_groups";
pub const LOGICAL_GROUPS: &str = "logical_groups";
pub const DB_DEFAULT_TABLE: &str = "./db/default.surql";
pub const DB_BACKUP_PATH: &str = "./db/backup.surql";
pub const DB_PATH_FILE: &str = "./db/path.txt";
pub const DB_DEFAULT_PATH: &str = "./db/ai_search";

pub async fn new(tx: Sender<()>) -> anyhow::Result<(), anyhow::Error> {
    DB_STATUS.set_state(StatusState::Initializing, "Connecting DB");
    // let capabilities = surrealdb::capabilities::Capabilities::all().with_all_experimental_features_allowed();
    // let config = surrealdb::opt::Config::new().capabilities(capabilities); // ("./db/ai_search", config)
    let db_path = get_db_path();
    DB.connect::<surrealdb::engine::local::SurrealKv>(&db_path).await?;
    DB.use_ns(NS).use_db(DB_NAME).await?;

    // DEFINE BUCKET userfiles BACKEND "memory";
    let query = r#" 
        BEGIN;
        DEFINE TABLE IF NOT EXISTS thumbnails TYPE NORMAL SCHEMAFULL PERMISSIONS FULL;
        DEFINE TABLE IF NOT EXISTS user_settings TYPE NORMAL SCHEMALESS PERMISSIONS FULL;
        DEFINE TABLE IF NOT EXISTS filter_groups TYPE NORMAL SCHEMAFULL PERMISSIONS FULL;
        DEFINE TABLE IF NOT EXISTS logical_groups TYPE NORMAL SCHEMAFULL PERMISSIONS FULL;
        DEFINE TABLE IF NOT EXISTS cached_scans TYPE NORMAL SCHEMAFULL PERMISSIONS FULL;
        DEFINE TABLE IF NOT EXISTS cached_scan_items TYPE NORMAL SCHEMAFULL PERMISSIONS FULL;
        DEFINE TABLE IF NOT EXISTS clip_embeddings TYPE NORMAL SCHEMAFULL PERMISSIONS FULL;
        DEFINE TABLE IF NOT EXISTS assistant_sessions TYPE NORMAL SCHEMAFULL PERMISSIONS FULL;
        DEFINE TABLE IF NOT EXISTS assistant_messages TYPE NORMAL SCHEMAFULL PERMISSIONS FULL;

        DEFINE FIELD IF NOT EXISTS caption ON thumbnails TYPE option<string> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS category ON thumbnails TYPE option<string> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS db_created ON thumbnails TYPE option<datetime> DEFAULT time::now() PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS description ON thumbnails TYPE option<string> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS file_type ON thumbnails TYPE option<string> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS filename ON thumbnails TYPE string PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS hash ON thumbnails TYPE option<string> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS modified ON thumbnails TYPE datetime DEFAULT time::now() PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS path ON thumbnails TYPE string PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS size ON thumbnails TYPE number PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS tags ON thumbnails TYPE option<array<string>> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS thumbnail_b64 ON thumbnails TYPE option<string> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS parent_dir ON thumbnails TYPE string;
        DEFINE FIELD IF NOT EXISTS logical_group ON thumbnails TYPE record<logical_groups> PERMISSIONS FULL;

        DEFINE FIELD IF NOT EXISTS thumb_ref ON clip_embeddings TYPE option<record<thumbnails>> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS path ON clip_embeddings TYPE string PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS hash ON clip_embeddings TYPE option<string> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS embedding ON clip_embeddings TYPE array<float> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS created ON clip_embeddings TYPE datetime DEFAULT time::now() PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS updated ON clip_embeddings TYPE datetime DEFAULT time::now() PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS similarity_score ON clip_embeddings TYPE option<float> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS clip_similarity_score ON clip_embeddings TYPE option<float> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS title ON assistant_sessions TYPE string PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS created ON assistant_sessions TYPE datetime DEFAULT time::now() PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS updated ON assistant_sessions TYPE datetime DEFAULT time::now() PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS session_ref ON assistant_messages TYPE record<assistant_sessions> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS role ON assistant_messages TYPE string PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS content ON assistant_messages TYPE string PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS attachments ON assistant_messages TYPE option<array<string>> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS created ON assistant_messages TYPE datetime DEFAULT time::now() PERMISSIONS FULL;
        
        DEFINE FIELD IF NOT EXISTS ext_enabled ON user_settings TYPE option<array<any>> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS excluded_dirs ON user_settings TYPE option<array<string>> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS group_by_category ON user_settings TYPE bool PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS auto_indexing ON user_settings TYPE bool PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS ai_prompt_template ON user_settings TYPE string PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS overwrite_descriptions ON user_settings TYPE bool PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS filter_modified_after ON user_settings TYPE option<string> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS filter_modified_before ON user_settings TYPE option<string> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS filter_skip_icons ON user_settings TYPE bool PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS db_min_size_bytes ON user_settings TYPE option<number> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS db_max_size_bytes ON user_settings TYPE option<number> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS db_excluded_exts ON user_settings TYPE option<array<string>> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS auto_clip_embeddings ON user_settings TYPE bool PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS clip_augment_with_text ON user_settings TYPE bool PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS clip_overwrite_embeddings ON user_settings TYPE bool PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS clip_model ON user_settings TYPE option<string> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS recent_paths ON user_settings TYPE array<string> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS auto_save_to_database ON user_settings TYPE bool PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS egui_preferences ON user_settings TYPE object DEFAULT {} PERMISSIONS FULL;

        DEFINE FIELD IF NOT EXISTS root ON cached_scans TYPE string PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS started ON cached_scans TYPE datetime DEFAULT time::now() PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS finished ON cached_scans TYPE option<datetime> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS total ON cached_scans TYPE option<number> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS title ON cached_scans TYPE option<string> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS scan_id ON cached_scans TYPE option<number> PERMISSIONS FULL;

        DEFINE FIELD IF NOT EXISTS scan_ref ON cached_scan_items TYPE record<cached_scans> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS path ON cached_scan_items TYPE string PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS created ON cached_scan_items TYPE datetime DEFAULT time::now() PERMISSIONS FULL;
        
        DEFINE FIELD IF NOT EXISTS name ON filter_groups TYPE string PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS include_images ON filter_groups TYPE bool PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS include_videos ON filter_groups TYPE bool PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS include_dirs ON filter_groups TYPE bool PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS skip_icons ON filter_groups TYPE bool PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS min_size_bytes ON filter_groups TYPE option<number> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS max_size_bytes ON filter_groups TYPE option<number> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS excluded_terms ON filter_groups TYPE array<string> PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS created ON filter_groups TYPE datetime DEFAULT time::now() PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS updated ON filter_groups TYPE datetime DEFAULT time::now() PERMISSIONS FULL;
        
        DEFINE FIELD IF NOT EXISTS name ON logical_groups TYPE string PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS created ON logical_groups TYPE datetime DEFAULT time::now() PERMISSIONS FULL;
        DEFINE FIELD IF NOT EXISTS updated ON logical_groups TYPE datetime DEFAULT time::now() PERMISSIONS FULL;
        
        DEFINE INDEX IF NOT EXISTS lg_name_idx ON logical_groups FIELDS name UNIQUE;
        DEFINE INDEX IF NOT EXISTS category_idx ON thumbnails FIELDS category;
        DEFINE INDEX IF NOT EXISTS tags_idx ON thumbnails FIELDS tags;
        DEFINE INDEX IF NOT EXISTS path_idx ON thumbnails FIELDS path UNIQUE;
        DEFINE INDEX IF NOT EXISTS clip_path_idx ON clip_embeddings FIELDS path UNIQUE;
        DEFINE INDEX IF NOT EXISTS clip_thumb_ref_idx ON clip_embeddings FIELDS thumb_ref;
        DEFINE INDEX IF NOT EXISTS idx_parent_dir ON thumbnails FIELDS parent_dir;
        DEFINE INDEX IF NOT EXISTS idx_thumb_logical_group ON thumbnails FIELDS logical_group;
        DEFINE INDEX IF NOT EXISTS idx_clip_hnsw ON clip_embeddings FIELDS embedding HNSW DIMENSION 1024 TYPE F32 DIST COSINE EFC 120 M 12;
        DEFINE INDEX IF NOT EXISTS idx_cached_items_scan ON cached_scan_items FIELDS scan_ref;
        DEFINE INDEX IF NOT EXISTS idx_cached_scans_started ON cached_scans FIELDS started;
    DEFINE INDEX IF NOT EXISTS idx_assistant_session ON assistant_messages FIELDS session_ref;

        COMMIT;
    "#;

    let response = DB.query(query).await?;
    let _ = response.check()?;
    // Ensure at least one logical group exists (Default)
    if crate::database::LogicalGroup::list_all().await?.is_empty() {
        let _ = crate::database::LogicalGroup::create("Default").await;
    }
    let _ = tx.send(());
    DB_STATUS.set_progress(0, 0);
    DB_STATUS.set_detail("");
    DB_STATUS.set_state(StatusState::Idle, "");
    Ok(())
}

/// Read the configured SurrealKV database folder path from disk (./db/path.txt)
/// falling back to ./db/ai_search if not present or invalid.
pub fn get_db_path() -> String {
    match fs::read_to_string(DB_PATH_FILE) {
        Ok(s) => {
            let p = s.trim();
            if p.is_empty() { DB_DEFAULT_PATH.to_string() } else { p.to_string() }
        }
        Err(_) => DB_DEFAULT_PATH.to_string(),
    }
}

/// Persist the SurrealKV database folder path to ./db/path.txt.
/// Returns Ok(()) if saved. Actual reconnection may require app restart.
pub fn set_db_path<P: AsRef<Path>>(dir: P) -> anyhow::Result<(), anyhow::Error> {
    let dir = dir.as_ref();
    if !dir.exists() {
        fs::create_dir_all(dir)?;
    }
    if !dir.is_dir() {
        anyhow::bail!("Provided path is not a directory: {}", dir.display());
    }
    let mut pb = PathBuf::new();
    pb.push(DB_PATH_FILE);
    if let Some(parent) = pb.parent() { fs::create_dir_all(parent)?; }
    fs::write(&pb, dir.as_os_str().to_string_lossy().as_ref())?;
    Ok(())
}

// --- DB activity indicator helpers ---
static ACTIVE_DB_OPS: LazyLock<AtomicUsize> = LazyLock::new(|| AtomicUsize::new(0));

pub struct DbActivityGuard(());

impl Drop for DbActivityGuard {
    fn drop(&mut self) {
        // Decrement and update status; when zero, clear to Idle and reset progress
        let remaining = ACTIVE_DB_OPS.fetch_sub(1, Ordering::SeqCst).saturating_sub(1);
        if remaining == 0 {
            // Only reset to Idle if not currently in Error state.
            if !matches!(DB_STATUS.snapshot().state, StatusState::Error) {
                DB_STATUS.set_progress(0, 0);
                DB_STATUS.set_state(StatusState::Idle, "");
                DB_STATUS.set_detail("");
            }
        } else {
            if !matches!(DB_STATUS.snapshot().state, StatusState::Error) {
                DB_STATUS.set_state(StatusState::Running, format!("Active ({remaining})"));
            }
        }
    }
}

/// Mark a DB operation as active for the duration of this guard.
pub fn db_activity(detail: impl Into<String>) -> DbActivityGuard {
    let detail = detail.into();
    let new_count = ACTIVE_DB_OPS.fetch_add(1, Ordering::SeqCst) + 1;
    let msg = if new_count > 1 {
        format!("{detail} ({new_count} active)")
    } else {
        detail
    };
    if !matches!(DB_STATUS.snapshot().state, StatusState::Error) {
        DB_STATUS.set_state(StatusState::Running, &msg);
    }
    DB_STATUS.set_detail(msg);
    DbActivityGuard(())
}

/// Convenience to set DB progress (e.g., during batch upserts).
pub fn db_set_progress(current: u64, total: u64) {
    DB_STATUS.set_progress(current, total);
}

/// Update detail text for DB operations without changing state.
pub fn db_set_detail(detail: impl Into<String>) {
    DB_STATUS.set_detail(detail);
}

/// Surface an error to the DB status card.
pub fn db_set_error(err: impl Into<String>) {
    DB_STATUS.set_error(err);
}

/// Export the current database (namespace+db) to a SurrealQL file.
pub async fn export_to<P: AsRef<Path>>(path: P) -> anyhow::Result<(), anyhow::Error> {
    let _ga = db_activity("EXPORT DB");
    db_set_detail(format!("Exporting to {}", path.as_ref().display()));
    // SurrealDB v2 exposes DB.export taking a file path or writer; use the path API.
    super::DB.export(path.as_ref()).await.map_err(|e| { db_set_error(format!("DB export failed: {e}")); e.into() })
}

/// Import a SurrealQL file into the current database (namespace+db).
pub async fn import_from<P: AsRef<Path>>(path: P) -> anyhow::Result<(), anyhow::Error> {
    let _ga = db_activity("IMPORT DB");
    db_set_detail(format!("Importing from {}", path.as_ref().display()));
    super::DB.import(path.as_ref()).await.map_err(|e| { db_set_error(format!("DB import failed: {e}")); e.into() })
}
