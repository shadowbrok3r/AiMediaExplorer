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

// Global lazy AI engine accessor. Initialized on first use. Heavy components (vision model)
// are loaded explicitly via `init_global_ai_engine_async()` to avoid blocking UI thread.
pub static GLOBAL_AI_ENGINE: Lazy<AISearchEngine> = Lazy::new(|| AISearchEngine::new());

/// Safe to call multiple times; subsequent calls become no-ops.
pub async fn init_global_ai_engine_async() {
    // Ensure background index worker & model; load cached metadata.
    // We purposefully do these sequentially to minimize concurrent model load attempts.
    JOY_STATUS.set_state(StatusState::Initializing, "Starting workers");
    GLOBAL_AI_ENGINE.ensure_index_worker().await;
    JOY_STATUS.set_state(StatusState::Initializing, "Loading vision model");
    // Model label is set when the JoyCaption worker resolves its path; avoid using unrelated settings here.
    if let Err(e) = crate::ai::joycap::ensure_worker_started().await {
        log::warn!("[AI] vision model init failed: {e}");
        JOY_STATUS.set_state(StatusState::Error, "Vision model failed");
    }
    CLIP_STATUS.set_state(StatusState::Initializing, "CLIP engine pending");
    let loaded = GLOBAL_AI_ENGINE.load_cached().await;
    log::info!("[AI] global engine initialized (cached {loaded} rows)");
    JOY_STATUS.set_state(StatusState::Running, format!("Cached {loaded}"));
    CLIP_STATUS.set_state(StatusState::Idle, "Idle");
}
