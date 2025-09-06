use crossbeam::channel::Sender;
use std::sync::LazyLock;
use surrealdb::{
    Surreal,
    engine::local::Db,
    opt::{Config, capabilities::Capabilities},
};
pub mod clip_embeddings;
pub mod files;
pub mod settings;
pub mod thumbnails;
pub use clip_embeddings::*;
pub use files::*;
pub use settings::*;
pub use thumbnails::*;

pub static DB: LazyLock<Surreal<Db>> = LazyLock::new(Surreal::init);
pub const NS: &str = "file_explorer";
pub const DB_NAME: &str = "ai_search";
pub const THUMBNAILS: &str = "thumbnails";
pub const USER_SETTINGS: &str = "user_settings";
pub const DB_DEFAULT_TABLE: &str = "./db/default.surql";
pub const DB_BACKUP_PATH: &str = "./db/backup.surql";

pub async fn new(tx: Sender<()>) -> anyhow::Result<(), anyhow::Error> {
    let capabilities = Capabilities::all().with_all_experimental_features_allowed();
    let config = Config::new().capabilities(capabilities);
    DB.connect::<surrealdb::engine::local::SurrealKv>(("./db/ai_search", config))
        .await?;
    DB.use_ns(NS).use_db(DB_NAME).await?;

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
