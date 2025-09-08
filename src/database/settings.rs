use crate::USER_SETTINGS;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use surrealdb::RecordId;
use tokio::task;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiSettings {
    pub id: RecordId,
    pub qa_collapsed: bool,
    pub drives_collapsed: bool,
    pub preview_collapsed: bool,
    pub preview_width: u32,
    pub sort: Option<SortSetting>,
    // "icons" | "details"
    pub view_mode: Option<String>,
    pub left_width: u32,
    pub ext_enabled: Option<Vec<(String, bool)>>,
    pub excluded_dirs: Option<Vec<String>>,
    #[serde(default)]
    pub group_by_category: bool,
    // Name, Path, Size, Modified, Created, Type
    #[serde(default)]
    pub detail_column_widths: Option<[f32; 6]>,
    #[serde(default)]
    pub category_col_width: Option<f32>,
    #[serde(default)]
    pub auto_indexing: bool,
    #[serde(default)]
    pub ai_prompt_template: String,
    #[serde(default)]
    pub overwrite_descriptions: bool,
    // Persisted filter state
    #[serde(default)]
    pub filter_modified_after: Option<String>,
    #[serde(default)]
    pub filter_modified_before: Option<String>,
    // multi-select categories
    #[serde(default)]
    pub filter_category_multi: Option<Vec<String>>,
    #[serde(default)]
    pub filter_only_with_thumb: bool,
    #[serde(default)]
    pub filter_only_with_description: bool,
    // Skip likely icon/asset images in scans (small sizes/dimensions, .ico, etc.)
    #[serde(default)]
    pub filter_skip_icons: bool,
    #[serde(default)]
    pub last_root: Option<String>,
    #[serde(default)]
    pub show_progress_overlay: bool,
    #[serde(default)]
    pub db_min_size_bytes: Option<u64>,
    #[serde(default)]
    pub db_max_size_bytes: Option<u64>,
    #[serde(default)]
    pub db_excluded_exts: Option<Vec<String>>, // lowercase extensions (without dot)
    // --- New CLIP settings ---
    #[serde(default)]
    pub auto_clip_embeddings: bool, // automatically generate CLIP embeddings during indexing
    #[serde(default)]
    pub clip_augment_with_text: bool, // blend image + textual metadata into a joint CLIP vector
    // Overwrite existing CLIP embeddings when auto generation runs
    #[serde(default)]
    pub clip_overwrite_embeddings: bool,
    // Selected CLIP/SigLIP model key
    #[serde(default)]
    pub clip_model: Option<String>,
    #[serde(default)]
    pub recent_paths: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SortSetting {
    pub by: SortBy,
    pub asc: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SortBy {
    Name,
    Category,
    Modified,
    Created,
    Size,
    Type,
}

impl Default for UiSettings {
    fn default() -> Self {
        Self {
            id: RecordId::from_table_key(USER_SETTINGS, "ShadowbrokerPC"),
            qa_collapsed: false,
            drives_collapsed: false,
            preview_collapsed: false,
            preview_width: 320,
            sort: Some(SortSetting {
                by: SortBy::Name,
                asc: true,
            }),
            view_mode: Some("list".into()),
            left_width: 240,
            ext_enabled: None,
            excluded_dirs: None,
            group_by_category: false,
            detail_column_widths: None,
            category_col_width: None,
            auto_indexing: false,
            ai_prompt_template: "Analyze the supplied image and return JSON with keys: description (detailed multi-sentence), caption (short), tags (array of lowercase single words), category (single general category). Return ONLY JSON.".into(),
            overwrite_descriptions: false,
            filter_modified_after: None,
            filter_modified_before: None,
            filter_category_multi: None,
            filter_only_with_thumb: false,
            filter_only_with_description: false,
            filter_skip_icons: false,
            last_root: None,
            show_progress_overlay: true,
            // Default lower bound: 10 KiB to avoid tiny files
            db_min_size_bytes: Some(10 * 1024),
            db_max_size_bytes: None,
            db_excluded_exts: None,
            auto_clip_embeddings: false,
            clip_augment_with_text: true,
            clip_overwrite_embeddings: false,
            clip_model: Some("unicom-vit-b32".into()),
            recent_paths: Vec::new(),
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
        if self.recent_paths.len() > 20 { self.recent_paths.truncate(20); }
    }
}
// In-memory snapshot (optional) to avoid extra DB selects for callers that load early.
static SETTINGS_CACHE: Lazy<Mutex<Option<UiSettings>>> = Lazy::new(|| Mutex::new(None));

pub fn load_settings() -> UiSettings {
    // Kick off async fetch; return default immediately (will hydrate later)
    task::spawn(async {
        if let Ok(s) = super::get_settings().await {
            *SETTINGS_CACHE.lock().unwrap() = Some(s);
        }
    });
    // Return cached if set
    if let Some(cached) = SETTINGS_CACHE.lock().unwrap().clone() {
        // log::error!("Got cached ui settings: {:?}", cached.clip_model);
        return cached;
    } else {
        log::error!("Using UiSettings::default()");
        UiSettings::default()
    }
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
    super::DB
        .upsert::<Option<UiSettings>>(UiSettings::default().id)
        .content::<UiSettings>(s)
        .await?;
    Ok(())
}

pub async fn get_settings() -> anyhow::Result<UiSettings, anyhow::Error> {
    let settings_res: Option<UiSettings> = super::DB.select(UiSettings::default().id).await?;
    // log::info!("Got settings: {:?}", settings_res.is_some());
    if let Some(settings) = settings_res {
        return Ok(settings);
    } else {
        Ok(UiSettings::default())
    }
}
