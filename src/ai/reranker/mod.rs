use super::jina_m0::engine::JinaM0Engine;
use super::model::HFAiModel;
use crate::ui::status::GlobalStatusIndicator;
use anyhow::Result;
use candle_core::Device;
use once_cell::sync::Lazy;
use tokenizers::Tokenizer;

pub static GLOBAL_RERANKER: Lazy<std::sync::Arc<tokio::sync::Mutex<Option<JinaRerankerEngine>>>> =
    Lazy::new(|| std::sync::Arc::new(tokio::sync::Mutex::new(None)));

/// Map a friendly key or full repo string to HF repo for reranker
pub fn map_reranker_key_to_repo(key: &str) -> String {
    // If key looks like an owner/repo, pass-through
    if key.contains('/') {
        return key.to_string();
    }
    let repo = match key {
        // Aliases can be added here
        "jina-m0" | "jina-reranker-m0" => "jinaai/jina-reranker-m0".to_string(),
        other => other.to_string(),
    };
    log::info!("repo: {repo}");
    repo
}

pub struct JinaRerankerEngine {
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub engine: Option<JinaM0Engine>,
}

impl JinaRerankerEngine {
    pub fn new(model_key: &str) -> Result<Self> {
        let hf_repo = map_reranker_key_to_repo(model_key);
        let device = crate::ai::hf::pick_device_cuda0_or_cpu();
        let mdl = crate::ai::hf::hf_model(&hf_repo)?;
        // Common filenames; some rerankers may ship multiple shards or different names.
        // We'll try a few typical ones.
        let config_file = mdl.get("config.json")?;
        // Prefer single-file safetensors when available
        let model_file = match mdl.get("model.safetensors") {
            Ok(p) => Some(p),
            Err(_) => None,
        };
        let tokenizer = match crate::ai::hf::load_tokenizer(&hf_repo, None) {
            Ok(t) => t,
            Err(e) => {
                log::warn!("[Reranker] tokenizer.json not found in repo {hf_repo}: {e:?}");
                // Fall back to bert-base-uncased tokenizer as a generic choice, if needed
                crate::ai::hf::load_tokenizer("bert-base-uncased", None)?
            }
        };
        let _cfg_bytes = std::fs::read(config_file)?;
        // Map weights to memory (dtype chosen conservatively for CPU compatibility)
        let dtype = candle_core::DType::F32;
        if let Some(w) = model_file.as_ref() {
            // Safety: mmap is read-only; candle handles tensor views lazily
            let _ = unsafe {
                crate::ai::hf::with_mmap_varbuilder_single(w, dtype, &device, |vb| {
                    let _ = vb; // placeholder
                    Ok(())
                })
            }?;
        } else {
            // Try sharded index if single-file missing
            let index = mdl.get("model.safetensors.index.json").ok();
            if let Some(idx) = index {
                let files = candle_examples::hub_load_local_safetensors(
                    idx.parent().unwrap(),
                    "model.safetensors.index.json",
                )?;
                let _ = unsafe {
                    crate::ai::hf::with_mmap_varbuilder_multi(&files, dtype, &device, |vb| {
                        let _ = vb; // placeholder
                        Ok(())
                    })
                }?;
            } else {
                log::warn!(
                    "[Reranker] No model weights found in repo {hf_repo}; engine will be inert until implemented"
                );
            }
        }
        // Build our local JinaM0Engine wrapper
        let engine = JinaM0Engine::load_from_repo(&hf_repo).ok();
        Ok(Self {
            tokenizer,
            device,
            engine,
        })
    }
}

impl HFAiModel for JinaRerankerEngine {
    fn load_from_hub(model_key: &str) -> anyhow::Result<Self> {
        Self::new(model_key)
    }
}

pub async fn ensure_reranker_from_settings() -> anyhow::Result<()> {
    // Fast check without long work while holding the lock
    if GLOBAL_RERANKER.lock().await.is_some() {
        return Ok(());
    }
    // Prefer cached settings to avoid an async round-trip if available
    let cached = crate::database::settings::load_settings();
    let model_key = if let Some(s) = cached.and_then(|c| c.reranker_model.clone()) {
        s
    } else {
        match crate::database::get_settings().await {
            Ok(s) => s
                .reranker_model
                .unwrap_or_else(|| "jinaai/jina-reranker-m0".to_string()),
            Err(_) => "jinaai/jina-reranker-m0".to_string(),
        }
    };
    log::info!("[Reranker] Loading model: {model_key}");
    // Update status indicator
    crate::ui::status::RERANK_STATUS.set_model(&model_key);
    crate::ui::status::RERANK_STATUS.set_state(
        crate::ui::status::StatusState::Initializing,
        "Loading reranker",
    );
    let res = JinaRerankerEngine::new(&model_key);
    match res {
        Ok(engine) => {
            let mut guard = GLOBAL_RERANKER.lock().await;
            *guard = Some(engine);
            crate::ui::status::RERANK_STATUS
                .set_state(crate::ui::status::StatusState::Running, "Ready");
            Ok(())
        }
        Err(e) => {
            crate::ui::status::RERANK_STATUS.set_error(format!("Init failed: {e}"));
            Err(e)
        }
    }
}

pub async fn clear_global_reranker() {
    let mut guard = GLOBAL_RERANKER.lock().await;
    *guard = None;
}

/// Compute text-only scores using the loaded Jina M0 engine (temporary call-through).
pub async fn jina_score_text_pairs(pairs: &[(String, String)]) -> anyhow::Result<Vec<f32>> {
    // Ensure engine exists (drop lock before awaiting heavy init)
    if GLOBAL_RERANKER.lock().await.is_none() {
        ensure_reranker_from_settings().await?;
    }
    let guard = GLOBAL_RERANKER.lock().await;
    let engine = guard
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Reranker not initialized"))?;
    // Pull settings for max_length
    let max_len = if let Some(s) = crate::database::settings::load_settings() {
        s.jina_max_length.unwrap_or(1024)
    } else {
        1024
    };
    if let Some(j) = &engine.engine {
        crate::ui::status::RERANK_STATUS.set_state(
            crate::ui::status::StatusState::Running,
            format!("Scoring {} pairs", pairs.len()),
        );
        crate::ui::status::RERANK_STATUS.set_detail(format!(
            "Batch {} len {}",
            pairs.len(),
            max_len
        ));
        let res = j.compute_score_text(pairs, max_len);
        match res {
            Ok(v) => {
                crate::ui::status::RERANK_STATUS
                    .set_state(crate::ui::status::StatusState::Idle, "Ready");
                Ok(v)
            }
            Err(e) => {
                crate::ui::status::RERANK_STATUS.set_error(format!("Scoring failed: {e}"));
                Err(e)
            }
        }
    } else {
        anyhow::bail!("JinaM0Engine missing")
    }
}

/// Synchronous wrapper around `jina_score_text_pairs` for use in non-async contexts (e.g., egui trait impls).
/// It will use the current Tokio runtime if present; otherwise, it spins up a lightweight runtime temporarily.
pub fn jina_score_text_pairs_blocking(pairs: &[(String, String)]) -> Vec<f32> {
    // Try using existing runtime
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        return tokio::task::block_in_place(|| {
            match handle.block_on(jina_score_text_pairs(pairs)) {
                Ok(v) => v,
                Err(_) => Vec::new(),
            }
        });
    } else {
        return Vec::new();
    }
}
