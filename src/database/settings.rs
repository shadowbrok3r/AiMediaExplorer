use crate::USER_SETTINGS;
use egui::Options;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use surrealdb::RecordId;
use tokio::task;
use crate::database::{db_activity, db_set_detail, db_set_error};

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
    pub egui_preferences: Options
}

impl Default for UiSettings {
    fn default() -> Self {
        Self {
            id: RecordId::from_table_key(USER_SETTINGS, "ShadowbrokerPC"),
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
            openai_default_model: Some("gpt-4o-mini".into()),
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
    super::DB
        .upsert::<Option<UiSettings>>(UiSettings::default().id)
        .content::<UiSettings>(s)
        .await
        .map_err(|e| { db_set_error(format!("Save settings failed: {e}")); e })?;
    Ok(())
}

pub async fn get_settings() -> anyhow::Result<UiSettings, anyhow::Error> {
    let _ga = db_activity("SELECT user settings");
    db_set_detail("Loading user settings".to_string());
    let settings_res: Option<UiSettings> = super::DB.select(UiSettings::default().id).await?;
    // log::info!("Got settings: {:?}", settings_res.is_some());
    if let Some(settings) = settings_res {
        return Ok(settings);
    } else {
        Ok(UiSettings::default())
    }
}
