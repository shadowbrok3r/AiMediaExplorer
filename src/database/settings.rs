use crate::USER_SETTINGS;
use egui::Options;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use surrealdb::types::{RecordId, SurrealValue};
use tokio::task;
use crate::database::{db_activity, db_set_detail, db_set_error};

/// Inner struct for database storage (without egui::Options which doesn't implement SurrealValue)
#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
struct UiSettingsDb {
    pub id: RecordId,
    pub ext_enabled: Option<Vec<(String, bool)>>,
    pub excluded_dirs: Option<Vec<String>>,
    pub group_by_category: bool,
    pub auto_indexing: bool,
    pub ai_prompt_template: String,
    pub overwrite_descriptions: bool,
    pub filter_modified_after: Option<String>,
    pub filter_modified_before: Option<String>,
    pub filter_skip_icons: bool,
    pub db_min_size_bytes: Option<u64>,
    pub db_max_size_bytes: Option<u64>,
    pub db_excluded_exts: Option<Vec<String>>,
    pub auto_clip_embeddings: bool,
    pub clip_augment_with_text: bool,
    pub clip_overwrite_embeddings: bool,
    pub clip_model: Option<String>,
    pub recent_paths: Vec<String>,
    pub recent_models: Vec<String>,
    pub last_used_model: Option<String>,
    pub auto_save_to_database: bool,
    pub joycaption_model_dir: Option<String>,
    pub reranker_model: Option<String>,
    pub jina_max_length: Option<usize>,
    pub jina_doc_type: Option<String>,
    pub jina_query_type: Option<String>,
    pub jina_enable_multimodal: Option<bool>,
    pub scan_found_batch_max: Option<usize>,
    pub ai_chat_provider: Option<String>,
    pub openai_api_key: Option<String>,
    pub grok_api_key: Option<String>,
    pub gemini_api_key: Option<String>,
    pub groq_api_key: Option<String>,
    pub openrouter_api_key: Option<String>,
    pub openai_base_url: Option<String>,
    pub openai_default_model: Option<String>,
    pub openai_organization: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiSettings {
    pub id: RecordId,
    // "icons" | "details"
    pub ext_enabled: Option<Vec<(String, bool)>>,
    pub excluded_dirs: Option<Vec<String>>,
    pub group_by_category: bool,
    // Name, Path, Size, Modified, Created, Type
    pub auto_indexing: bool,
    pub ai_prompt_template: String,
    pub overwrite_descriptions: bool,
    // Persisted filter state
    pub filter_modified_after: Option<String>,
    pub filter_modified_before: Option<String>,
    // Skip likely icon/asset images in scans (small sizes/dimensions, .ico, etc.)
    pub filter_skip_icons: bool,
    pub db_min_size_bytes: Option<u64>,
    pub db_max_size_bytes: Option<u64>,
    pub db_excluded_exts: Option<Vec<String>>, // lowercase extensions (without dot)
    // --- New CLIP settings ---
    pub auto_clip_embeddings: bool, // automatically generate CLIP embeddings during indexing
    pub clip_augment_with_text: bool, // blend image + textual metadata into a joint CLIP vector
    // Overwrite existing CLIP embeddings when auto generation runs
    pub clip_overwrite_embeddings: bool,
    // Selected CLIP/SigLIP model key
    pub clip_model: Option<String>,
    pub recent_paths: Vec<String>,
    // --- AI Assistant model history ---
    pub recent_models: Vec<String>,
    pub last_used_model: Option<String>,
    // When true, new scan results and thumbnail updates are saved into the DB automatically (if a logical group is active)
    pub auto_save_to_database: bool,
    // Optional: user-specified folder containing the vision (LLaVA/JoyCaption) model
    pub joycaption_model_dir: Option<String>,
    // Hugging Face repo for the reranker model (e.g., "jinaai/jina-reranker-m0")
    pub reranker_model: Option<String>,
    // --- Jina M0 settings ---
    // Max token length for text reranking (query+document)
    pub jina_max_length: Option<usize>,
    // Document type: "text" or "image" (for future multimodal)
    pub jina_doc_type: Option<String>,
    // Query type: "text" or "image"
    pub jina_query_type: Option<String>,
    // Enable multimodal reranking when available (uses image evidence)
    pub jina_enable_multimodal: Option<bool>,
    // Batch size for sending FoundBatch messages during scans
    pub scan_found_batch_max: Option<usize>,
    // ---------------- AI Assistant (OpenAI-compatible) ----------------
    // Which provider to use for the Assistant when not using local JoyCaption.
    // One of: "local-joycaption" | "openai" | "grok" | "gemini" | "groq" | "openrouter" | "custom"
    pub ai_chat_provider: Option<String>,
    // API keys per provider (stored locally). Optional, only used when provider selected.
    pub openai_api_key: Option<String>,
    pub grok_api_key: Option<String>,
    pub gemini_api_key: Option<String>,
    pub groq_api_key: Option<String>,
    pub openrouter_api_key: Option<String>,
    // OpenAI-compatible endpoint overrides (for OpenWebUI/LocalAI/vLLM/etc.)
    // Example: http://hostname:port/v1
    pub openai_base_url: Option<String>,
    // Default model name to use with OpenAI(-compatible) endpoints
    pub openai_default_model: Option<String>,
    // Optional OpenAI organization header
    pub openai_organization: Option<String>,
    // egui preferences - kept in memory only, not persisted to DB
    #[serde(skip)]
    pub egui_preferences: Options
}

impl From<UiSettingsDb> for UiSettings {
    fn from(db: UiSettingsDb) -> Self {
        Self {
            id: db.id,
            ext_enabled: db.ext_enabled,
            excluded_dirs: db.excluded_dirs,
            group_by_category: db.group_by_category,
            auto_indexing: db.auto_indexing,
            ai_prompt_template: db.ai_prompt_template,
            overwrite_descriptions: db.overwrite_descriptions,
            filter_modified_after: db.filter_modified_after,
            filter_modified_before: db.filter_modified_before,
            filter_skip_icons: db.filter_skip_icons,
            db_min_size_bytes: db.db_min_size_bytes,
            db_max_size_bytes: db.db_max_size_bytes,
            db_excluded_exts: db.db_excluded_exts,
            auto_clip_embeddings: db.auto_clip_embeddings,
            clip_augment_with_text: db.clip_augment_with_text,
            clip_overwrite_embeddings: db.clip_overwrite_embeddings,
            clip_model: db.clip_model,
            recent_paths: db.recent_paths,
            recent_models: db.recent_models,
            last_used_model: db.last_used_model,
            auto_save_to_database: db.auto_save_to_database,
            joycaption_model_dir: db.joycaption_model_dir,
            reranker_model: db.reranker_model,
            jina_max_length: db.jina_max_length,
            jina_doc_type: db.jina_doc_type,
            jina_query_type: db.jina_query_type,
            jina_enable_multimodal: db.jina_enable_multimodal,
            scan_found_batch_max: db.scan_found_batch_max,
            ai_chat_provider: db.ai_chat_provider,
            openai_api_key: db.openai_api_key,
            grok_api_key: db.grok_api_key,
            gemini_api_key: db.gemini_api_key,
            groq_api_key: db.groq_api_key,
            openrouter_api_key: db.openrouter_api_key,
            openai_base_url: db.openai_base_url,
            openai_default_model: db.openai_default_model,
            openai_organization: db.openai_organization,
            egui_preferences: Options::default(),
        }
    }
}

impl From<&UiSettings> for UiSettingsDb {
    fn from(s: &UiSettings) -> Self {
        Self {
            id: s.id.clone(),
            ext_enabled: s.ext_enabled.clone(),
            excluded_dirs: s.excluded_dirs.clone(),
            group_by_category: s.group_by_category,
            auto_indexing: s.auto_indexing,
            ai_prompt_template: s.ai_prompt_template.clone(),
            overwrite_descriptions: s.overwrite_descriptions,
            filter_modified_after: s.filter_modified_after.clone(),
            filter_modified_before: s.filter_modified_before.clone(),
            filter_skip_icons: s.filter_skip_icons,
            db_min_size_bytes: s.db_min_size_bytes,
            db_max_size_bytes: s.db_max_size_bytes,
            db_excluded_exts: s.db_excluded_exts.clone(),
            auto_clip_embeddings: s.auto_clip_embeddings,
            clip_augment_with_text: s.clip_augment_with_text,
            clip_overwrite_embeddings: s.clip_overwrite_embeddings,
            clip_model: s.clip_model.clone(),
            recent_paths: s.recent_paths.clone(),
            recent_models: s.recent_models.clone(),
            last_used_model: s.last_used_model.clone(),
            auto_save_to_database: s.auto_save_to_database,
            joycaption_model_dir: s.joycaption_model_dir.clone(),
            reranker_model: s.reranker_model.clone(),
            jina_max_length: s.jina_max_length,
            jina_doc_type: s.jina_doc_type.clone(),
            jina_query_type: s.jina_query_type.clone(),
            jina_enable_multimodal: s.jina_enable_multimodal,
            scan_found_batch_max: s.scan_found_batch_max,
            ai_chat_provider: s.ai_chat_provider.clone(),
            openai_api_key: s.openai_api_key.clone(),
            grok_api_key: s.grok_api_key.clone(),
            gemini_api_key: s.gemini_api_key.clone(),
            groq_api_key: s.groq_api_key.clone(),
            openrouter_api_key: s.openrouter_api_key.clone(),
            openai_base_url: s.openai_base_url.clone(),
            openai_default_model: s.openai_default_model.clone(),
            openai_organization: s.openai_organization.clone(),
        }
    }
}

impl Default for UiSettings {
    fn default() -> Self {
        Self {
            id: RecordId::new(USER_SETTINGS, "ShadowbrokerPC"),
            ext_enabled: None,
            excluded_dirs: None,
            group_by_category: false,
            auto_indexing: false,
            ai_prompt_template: "Analyze the supplied image and return JSON with keys: description (detailed multi-sentence), caption (short), tags (array of lowercase single words), category (single general category). Return ONLY JSON.".into(),
            overwrite_descriptions: false,
            filter_modified_after: None,
            filter_modified_before: None,
            filter_skip_icons: false,
            // Default lower bound: 10 KiB to avoid tiny files
            db_min_size_bytes: Some(10 * 1024),
            db_max_size_bytes: None,
            db_excluded_exts: None,
            auto_clip_embeddings: false,
            clip_augment_with_text: true,
            clip_overwrite_embeddings: false,
            clip_model: Some("siglip2-large-patch16-512".into()),
            recent_paths: Vec::new(),
            recent_models: Vec::new(),
            last_used_model: None,
            auto_save_to_database: false,
            joycaption_model_dir: None,
            reranker_model: Some("jinaai/jina-reranker-m0".into()),
            jina_max_length: Some(1024),
            jina_doc_type: Some("text".into()),
            jina_query_type: Some("text".into()),
            jina_enable_multimodal: Some(false),
            scan_found_batch_max: Some(128),
            ai_chat_provider: Some("local-joycaption".into()),
            openai_api_key: std::env::var("OPENAI_API_KEY").ok(),
            grok_api_key: std::env::var("GROK_API_KEY").ok(),
            gemini_api_key: std::env::var("GEMINI_API_KEY").ok(),
            groq_api_key: std::env::var("GROQ_API_KEY").ok(),
            openrouter_api_key: std::env::var("OPENROUTER_API_KEY").ok(),
            openai_base_url: None,
            openai_default_model: Some("gpt-5-mini".into()),
            openai_organization: None,
            egui_preferences: Options::default()

        }
    }
}

impl UiSettings {
    pub fn push_recent_path(&mut self, p: String) {
        if p.trim().is_empty() { return; }
        // Move to front if exists
        if let Some(idx) = self.recent_paths.iter().position(|x| *x == p) {
            self.recent_paths.remove(idx);
        }
        self.recent_paths.insert(0, p);
        // Cap at 20
        if self.recent_paths.len() > 10 { self.recent_paths.truncate(10); }
    }

    pub fn push_recent_model(&mut self, model: &str) {
        if model.trim().is_empty() { return; }
        if let Some(idx) = self.recent_models.iter().position(|x| x == model) {
            self.recent_models.remove(idx);
        }
        self.recent_models.insert(0, model.to_string());
        if self.recent_models.len() > 8 { self.recent_models.truncate(8); }
        self.last_used_model = Some(model.to_string());
    }
}
// In-memory snapshot (optional) to avoid extra DB selects for callers that load early.
pub static SETTINGS_CACHE: Lazy<Mutex<Option<UiSettings>>> = Lazy::new(|| Mutex::new(None));

pub fn load_settings() -> Option<UiSettings> {
    // Do not fetch or fabricate defaults here. Until DB is ready and cache is hydrated,
    // return None so callers can gate behavior appropriately.
    SETTINGS_CACHE.lock().unwrap().clone()
}

pub fn save_settings(s: &UiSettings) {
    // Update cache immediately and persist asynchronously.
    *SETTINGS_CACHE.lock().unwrap() = Some(s.clone());
    let to_save = s.clone();
    task::spawn(async move {
        if let Err(e) = save_settings_in_db(to_save).await {
            log::error!("[settings] save_settings failed: {e}");
        } else {
            // After a successful save, refresh the cache from DB to ensure it's in sync.
            match get_settings().await {
                Ok(svr) => {
                    *SETTINGS_CACHE.lock().unwrap() = Some(svr);
                }
                Err(e) => log::warn!("[settings] get_settings after save failed: {e}"),
            }
        }
    });
}

pub async fn save_settings_in_db(s: UiSettings) -> anyhow::Result<(), anyhow::Error> {
    let _ga = db_activity("UPSERT user settings");
    db_set_detail("Saving user settings".to_string());
    let db_settings: UiSettingsDb = (&s).into();
    super::DB
        .upsert::<Option<UiSettingsDb>>(UiSettings::default().id)
        .content::<UiSettingsDb>(db_settings)
        .await
        .map_err(|e| { db_set_error(format!("Save settings failed: {e}")); e })?;
    Ok(())
}

pub async fn get_settings() -> anyhow::Result<UiSettings, anyhow::Error> {
    let _ga = db_activity("SELECT user settings");
    db_set_detail("Loading user settings".to_string());
    let settings_res: Option<UiSettingsDb> = super::DB.select(UiSettings::default().id).await?;
    if let Some(settings) = settings_res {
        return Ok(settings.into());
    } else {
        Ok(UiSettings::default())
    }
}
