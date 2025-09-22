pub mod ai_search;
pub mod candle_llava;
pub mod clip;
pub mod flux_transformer;
pub mod hf;
pub mod jina_m0;
pub mod joycap;
pub mod model;
pub mod qwen2_5_vl;
pub mod qwen_image_transformer;
pub mod refine;
pub mod reranker;
pub mod siglip;
pub mod vae;
pub mod openai_compat;

pub use ai_search::*;
pub use jina_m0::*;
pub use joycap as joycaption_adapter;
pub use model::*;
pub use refine::*;
pub use reranker::*;

use crate::database::Thumbnail;
use crate::ui::status::{CLIP_STATUS, GlobalStatusIndicator, StatusState, VISION_STATUS};
use once_cell::sync::Lazy;
pub mod qwen_image_edit;
// Global lazy AI engine accessor. Initialized on first use.
pub static GLOBAL_AI_ENGINE: Lazy<AISearchEngine> = Lazy::new(|| AISearchEngine::new());

pub async fn init_global_ai_engine_async() {
    VISION_STATUS.set_state(StatusState::Initializing, "Starting workers");
    use crate::ui::status::DeviceKind;
    VISION_STATUS.set_device(if candle_core::Device::new_cuda(0).is_ok() { DeviceKind::GPU } else { DeviceKind::CPU });
    GLOBAL_AI_ENGINE.ensure_index_worker().await;
    VISION_STATUS.set_state(StatusState::Initializing, "Loading vision model");
    if let Err(e) = crate::ai::joycap::ensure_worker_started().await {
        log::warn!("[AI] vision model init failed: {e}");
        VISION_STATUS.set_state(StatusState::Error, "Vision model failed");
    }
    CLIP_STATUS.set_state(StatusState::Initializing, "CLIP engine pending");
    let loaded = GLOBAL_AI_ENGINE.load_cached().await;
    log::info!("[AI] global engine initialized (cached {loaded} rows)");
    VISION_STATUS.set_state(StatusState::Running, format!("Loading Cache"));
    CLIP_STATUS.set_state(StatusState::Idle, "Idle");
    VISION_STATUS.set_state(StatusState::Idle, format!("Cached {loaded}"));
}

// Lightweight lifecycle helpers to avoid keeping all heavy models resident simultaneously.
pub async fn unload_heavy_models_except(keep: &str) {
    // keep is one of: "CLIP", "VISION", "RERANK", or "" for none
    match keep {
        "CLIP" => {
            // Stop JOYCAP worker
            crate::ai::joycap::stop_worker().await;
        }
        "VISION" => {
            // Drop CLIP backend
            crate::ai::clip::clear_clip_engine(&GLOBAL_AI_ENGINE.clip_engine).await;
        }
        "RERANK" => {
            // Prefer to release JOYCAP and CLIP for memory headroom
            crate::ai::joycap::stop_worker().await;
            crate::ai::clip::clear_clip_engine(&GLOBAL_AI_ENGINE.clip_engine).await;
        }
        _ => {
            // Unload all heavy backends
            crate::ai::joycap::stop_worker().await;
            crate::ai::clip::clear_clip_engine(&GLOBAL_AI_ENGINE.clip_engine).await;
            crate::ai::reranker::clear_global_reranker().await;
        }
    }
}
