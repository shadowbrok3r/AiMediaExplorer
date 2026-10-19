//! Work the engines do after settings change, and the buttons that load or free models.

use std::sync::atomic::Ordering;

use super::draft::Effect;
use crate::UiSettings;
use crate::status::{CLIP_STATUS, GlobalStatusIndicator, StatusState, VISION_STATUS};

/// What the model buttons ask for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineAction {
    LoadVision,
    ReloadVision,
    ReloadClip,
    ReloadReranker,
    BackfillClip,
    /// Free every model except the named one; `None` frees all.
    Unload(Option<Keep>),
}

/// A model kept resident when the others are freed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Keep {
    Clip,
    Vision,
    Reranker,
}

impl Keep {
    pub fn key(self) -> &'static str {
        match self {
            Self::Clip => "CLIP",
            Self::Vision => "VISION",
            Self::Reranker => "RERANK",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Clip => "Keep CLIP",
            Self::Vision => "Keep vision",
            Self::Reranker => "Keep reranker",
        }
    }
}

/// Switches the application flags that mirror the AI settings.
pub struct AiFlags<'a> {
    pub ready: &'a mut bool,
    pub initializing: &'a mut bool,
}

/// The default CLIP model key.
pub const DEFAULT_CLIP: &str = "siglip2-large-patch16-512";

/// Applies the engine side of the changed settings.
pub fn run_effects(effects: &[Effect], settings: &UiSettings, flags: &mut AiFlags) {
    for effect in effects {
        match effect {
            Effect::VisionFolder => {
                let path = settings.joycaption_model_dir.clone().unwrap_or_else(|| crate::app::DEFAULT_JOYCAPTION_PATH.to_string());
                VISION_STATUS.set_model(&path);
                VISION_STATUS.set_state(StatusState::Idle, "Model path set (reload on next use)");
            }
            Effect::ClipEngine => {
                let key = settings.clip_model.clone().unwrap_or_else(|| DEFAULT_CLIP.to_string());
                CLIP_STATUS.set_model(&key);
                CLIP_STATUS.set_state(StatusState::Idle, format!("Model set: {key} (reload on next use)"));
                let engine = crate::ai::GLOBAL_AI_ENGINE.clip_engine.clone();
                spawn(async move { crate::ai::clip::clear_clip_engine(&engine).await });
            }
            Effect::Reranker => spawn(async { crate::ai::reranker::clear_global_reranker().await }),
            Effect::AutoFlags => {
                let engine = &crate::ai::GLOBAL_AI_ENGINE;
                engine.auto_descriptions_enabled.store(settings.auto_indexing, Ordering::Relaxed);
                engine.auto_clip_enabled.store(settings.auto_clip_embeddings, Ordering::Relaxed);
                if settings.auto_indexing && !*flags.ready {
                    *flags.initializing = true;
                    spawn(async { crate::ai::init_global_ai_engine_async().await });
                }
                if settings.auto_clip_embeddings {
                    run(EngineAction::BackfillClip);
                }
            }
        }
    }
}

fn spawn<F: std::future::Future<Output = ()> + Send + 'static>(future: F) {
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        handle.spawn(future);
    }
}

/// Runs what a model button asked for.
pub fn run(action: EngineAction) {
    match action {
        EngineAction::LoadVision => {
            VISION_STATUS.set_state(StatusState::Initializing, "Loading vision model");
            spawn(async {
                crate::ai::init_global_ai_engine_async().await;
                VISION_STATUS.set_state(StatusState::Idle, "Ready");
            });
        }
        EngineAction::ReloadVision => {
            VISION_STATUS.set_state(StatusState::Initializing, "Reloading vision model");
            spawn(async {
                crate::ai::joycap::stop_worker().await;
                if let Err(e) = crate::ai::joycap::ensure_worker_started().await {
                    log::warn!("[AI] vision model reload failed: {e}");
                    VISION_STATUS.set_state(StatusState::Error, "Vision model failed");
                } else {
                    VISION_STATUS.set_state(StatusState::Idle, "Ready");
                }
            });
        }
        EngineAction::ReloadClip => {
            let key = crate::settings::store().read(|s| s.clip_model.clone()).unwrap_or_else(|| DEFAULT_CLIP.to_string());
            CLIP_STATUS.set_model(&key);
            CLIP_STATUS.set_state(StatusState::Initializing, format!("Reloading: {key}"));
            spawn(async {
                crate::ai::clip::clear_clip_engine(&crate::ai::GLOBAL_AI_ENGINE.clip_engine).await;
                match crate::ai::GLOBAL_AI_ENGINE.ensure_clip_engine().await {
                    Ok(_) => log::info!("[CLIP] Engine reloaded successfully"),
                    Err(e) => log::error!("[CLIP] Engine reload failed: {e}"),
                }
            });
        }
        EngineAction::ReloadReranker => spawn(async {
            crate::ai::reranker::clear_global_reranker().await;
            match crate::ai::reranker::ensure_reranker_from_settings().await {
                Ok(()) => log::info!("[Reranker] Engine reloaded successfully"),
                Err(e) => log::error!("[Reranker] Reload failed: {e}"),
            }
        }),
        EngineAction::BackfillClip => spawn(async {
            match crate::ai::GLOBAL_AI_ENGINE.clip_generate_recursive().await {
                Ok(added) => log::info!("[CLIP] Backfill added {added}"),
                Err(e) => log::warn!("[CLIP] Backfill failed: {e}"),
            }
        }),
        EngineAction::Unload(keep) => {
            let key = keep.map(Keep::key).unwrap_or("");
            spawn(async move { crate::ai::unload_heavy_models_except(key).await });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_keep_buttons_name_the_keys_the_unloader_knows() {
        for keep in [Keep::Clip, Keep::Vision, Keep::Reranker] {
            assert!(["CLIP", "VISION", "RERANK"].contains(&keep.key()));
            assert!(keep.label().starts_with("Keep"));
        }
    }

    #[test]
    fn effects_set_the_engine_flags_from_the_settings() {
        let (mut ready, mut initializing) = (true, false);
        let mut settings = UiSettings::default();
        settings.auto_indexing = true;
        settings.auto_clip_embeddings = false;
        run_effects(&[Effect::AutoFlags], &settings, &mut AiFlags { ready: &mut ready, initializing: &mut initializing });
        let engine = &crate::ai::GLOBAL_AI_ENGINE;
        assert!(engine.auto_descriptions_enabled.load(Ordering::Relaxed));
        assert!(!engine.auto_clip_enabled.load(Ordering::Relaxed));
        assert!(!initializing, "an engine that is ready is not started again");
        settings.auto_indexing = false;
        run_effects(&[Effect::AutoFlags], &settings, &mut AiFlags { ready: &mut ready, initializing: &mut initializing });
        assert!(!engine.auto_descriptions_enabled.load(Ordering::Relaxed));
    }

    #[test]
    fn a_missing_runtime_makes_the_buttons_do_nothing_instead_of_panicking() {
        run(EngineAction::BackfillClip);
        run(EngineAction::Unload(None));
    }
}
