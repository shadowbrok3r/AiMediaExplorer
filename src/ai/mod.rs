pub mod ai_search;
pub mod candle_llava;
pub mod clip;
pub mod joycap;
pub mod siglip;

pub use ai_search::*;
pub use joycap as joycaption_adapter;

use crate::database::Thumbnail;
use crate::ui::status::{CLIP_STATUS, GlobalStatusIndicator, JOY_STATUS, StatusState};
use once_cell::sync::Lazy;

// Global lazy AI engine accessor. Initialized on first use. 
pub static GLOBAL_AI_ENGINE: Lazy<AISearchEngine> = Lazy::new(|| AISearchEngine::new());

pub async fn init_global_ai_engine_async() {
    JOY_STATUS.set_state(StatusState::Initializing, "Starting workers");
    GLOBAL_AI_ENGINE.ensure_index_worker().await;
    JOY_STATUS.set_state(StatusState::Initializing, "Loading vision model");
    if let Err(e) = crate::ai::joycap::ensure_worker_started().await {
        log::warn!("[AI] vision model init failed: {e}");
        JOY_STATUS.set_state(StatusState::Error, "Vision model failed");
    }
    CLIP_STATUS.set_state(StatusState::Initializing, "CLIP engine pending");
    let loaded = GLOBAL_AI_ENGINE.load_cached().await;
    log::info!("[AI] global engine initialized (cached {loaded} rows)");
    JOY_STATUS.set_state(StatusState::Running, format!("Loading Cache"));
    CLIP_STATUS.set_state(StatusState::Idle, "Idle");
    JOY_STATUS.set_state(StatusState::Idle, format!("Cached {loaded}"));
}
