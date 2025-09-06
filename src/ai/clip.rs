
use fastembed::{ImageEmbedding, ImageInitOptions, ImageEmbeddingModel, TextEmbedding, TextInitOptions, EmbeddingModel};
use crate::ui::status::{CLIP_STATUS, StatusState, GlobalStatusIndicator};

fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 { for x in &mut v { *x /= norm; } }
    v
}

pub enum ClipBackend {
    FastEmbed { image_model: ImageEmbedding, text_model: TextEmbedding },
    Siglip(crate::ai::siglip::SiglipEngine),
}

pub struct ClipEngine { backend: ClipBackend }


impl ClipEngine {
    pub fn new_with_model_key(model_key: &str) -> anyhow::Result<Self> {
        // Map simple keys -> underlying backends
        let be = match model_key {
            // FastEmbed
            "unicom-vit-b32" => {
                let img = ImageEmbedding::try_new(ImageInitOptions::new(ImageEmbeddingModel::UnicomVitB32))?;
                let txt = TextEmbedding::try_new(TextInitOptions::new(EmbeddingModel::ClipVitB32))?;
                ClipBackend::FastEmbed { image_model: img, text_model: txt }
            }
            "clip-vit-b32" => {
                let img = ImageEmbedding::try_new(ImageInitOptions::new(ImageEmbeddingModel::ClipVitB32))?;
                let txt = TextEmbedding::try_new(TextInitOptions::new(EmbeddingModel::ClipVitB32))?;
                ClipBackend::FastEmbed { image_model: img, text_model: txt }
            }
            // SigLIP family via candle
            key if key.starts_with("siglip") => {
                let sig = crate::ai::siglip::SiglipEngine::new(model_key)?;
                ClipBackend::Siglip(sig)
            }
            _ => {
                let img = ImageEmbedding::try_new(ImageInitOptions::new(ImageEmbeddingModel::UnicomVitB32))?;
                let txt = TextEmbedding::try_new(TextInitOptions::new(EmbeddingModel::ClipVitB32))?;
                ClipBackend::FastEmbed { image_model: img, text_model: txt }
            }
        };
        Ok(Self { backend: be })
    }

    pub fn embed_image_path(&mut self, path: &str) -> anyhow::Result<Vec<f32>> {
        match &mut self.backend {
            ClipBackend::FastEmbed { image_model, .. } => {
                let out = image_model.embed(vec![path.to_string()], None)?;
                Ok(l2_normalize(out.into_iter().next().unwrap()))
            }
            ClipBackend::Siglip(sig) => Ok(sig.embed_image_path(path)?),
        }
    }

    pub fn embed_text(&mut self, text: &str) -> anyhow::Result<Vec<f32>> {
        match &mut self.backend {
            ClipBackend::FastEmbed { text_model, .. } => {
                let out = text_model.embed(vec![text.to_string()], None)?;
                Ok(l2_normalize(out.into_iter().next().unwrap()))
            }
            ClipBackend::Siglip(sig) => Ok(sig.embed_text(text)?),
        }
    }

    pub fn zero_shot_tags(&mut self, _image_vec: &[f32], _top_k: usize) -> Vec<String> { Vec::new() }

    pub fn zero_shot_category(&mut self, _image_vec: &[f32]) -> Option<String> { None }

}


fn dot(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x,y)| x*y).sum() }

impl ClipEngine {
    pub fn siglip_logits_image_vs_texts(&mut self, image_path: &str, texts: &[String]) -> anyhow::Result<Vec<f32>> {
        match &mut self.backend {
            ClipBackend::Siglip(sig) => sig.logits_image_vs_texts(image_path, texts),
            _ => anyhow::bail!("siglip_logits_image_vs_texts is only available for the SigLIP backend"),
        }
    }
}

pub(crate) async fn ensure_clip_engine(engine_slot: &std::sync::Arc<tokio::sync::Mutex<Option<ClipEngine>>>) -> anyhow::Result<()> {
    let mut guard = engine_slot.lock().await;
    if guard.is_none() {
        CLIP_STATUS.set_state(StatusState::Initializing, "Loading model");
        // Prefer DB-backed settings to avoid picking the default due to async cache hydration.
        let model_key = match crate::database::get_settings().await {
            Ok(s) => s.clip_model.unwrap_or_else(|| "unicom-vit-b32".to_string()),
            Err(e) => {
                log::warn!("[CLIP] get_settings() failed: {e}. Falling back to cached settings/default.");
                crate::database::settings::load_settings().clip_model.unwrap_or_else(|| "unicom-vit-b32".to_string())
            }
        };
        CLIP_STATUS.set_model(&model_key);
        log::info!("[CLIP] Loading model key: {}", model_key);
        match ClipEngine::new_with_model_key(&model_key) {
            Ok(c) => { *guard = Some(c); log::info!("[CLIP] Loaded."); CLIP_STATUS.set_state(StatusState::Idle, "Ready"); },
            Err(e) => { log::error!("[CLIP] Failed to init: {e}"); CLIP_STATUS.set_state(StatusState::Error, "Init failed"); return Err(e); }
        }
    }
    Ok(())
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 { dot(a,b) }

impl crate::ai::AISearchEngine {
    pub async fn ensure_clip_engine(&self) -> anyhow::Result<()> { super::clip::ensure_clip_engine(&self.clip_engine).await }

    // Count images missing a CLIP embedding (for UI diagnostics)
    pub async fn clip_missing_count(&self) -> usize {
        let files = self.files.lock().await;
        files.iter().filter(|f| f.file_type == "image" && f.clip_embedding.is_none()).count()
    }

    pub async fn clip_search_text(&self, query: &str, top_k: usize) -> Vec<crate::database::FileMetadata> {
        if self.ensure_clip_engine().await.is_err() { return Vec::new(); }
        let query_vec = {
            let mut guard = self.clip_engine.lock().await;
            if let Some(engine) = guard.as_mut() {
                match engine.embed_text(query) { Ok(v) => v, Err(e) => { log::error!("[CLIP] text embed failed: {e}"); return Vec::new(); } }
            } else { return Vec::new(); }
        };
        let mut scored: Vec<(f32, crate::database::FileMetadata)> = {
            let files = self.files.lock().await;
            files.iter().filter_map(|f| f.clip_embedding.as_ref().map(|emb| (dot(&query_vec, emb), f.clone()))).collect()
        };
        scored.sort_by(|a,b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(top_k).map(|(s, mut m)| { m.clip_similarity_score = Some(s); m }).collect()
    }

    pub async fn clip_search_image(&self, image_path: &str, top_k: usize) -> Vec<crate::database::FileMetadata> {
        if self.ensure_clip_engine().await.is_err() { return Vec::new(); }
        let image_vec = {
            let mut guard = self.clip_engine.lock().await;
            if let Some(engine) = guard.as_mut() {
                match engine.embed_image_path(image_path) { Ok(v) => v, Err(e) => { log::error!("[CLIP] image embed failed: {e}"); return Vec::new(); } }
            } else { return Vec::new(); }
        };
        let mut scored: Vec<(f32, crate::database::FileMetadata)> = {
            let files = self.files.lock().await;
            files.iter().filter_map(|f| f.clip_embedding.as_ref().map(|emb| (dot(&image_vec, emb), f.clone()))).collect()
        };
        scored.sort_by(|a,b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(top_k).map(|(s, mut m)| { m.clip_similarity_score = Some(s); m }).collect()
    }

    // Persist and attach CLIP embedding for list of paths
    pub async fn clip_generate_for_paths(&self, paths: &[String]) -> usize {
        // Ensure engine is loaded and matches selected model family (fastembed vs SigLIP)
        if self.ensure_clip_engine().await.is_err() { return 0; }
        // Check desired model from DB and reconcile backend if needed
        let desired_key = match crate::database::get_settings().await {
            Ok(s) => s.clip_model.unwrap_or_else(|| "unicom-vit-b32".to_string()),
            Err(_) => crate::database::settings::load_settings().clip_model.unwrap_or_else(|| "unicom-vit-b32".to_string()),
        };
        let want_siglip = desired_key.starts_with("siglip");
        let mut must_reinit = false;
        {
            let mut guard = self.clip_engine.lock().await;
            if let Some(engine) = guard.as_ref() {
                let is_siglip = matches!(engine.backend, crate::ai::clip::ClipBackend::Siglip(_));
                if is_siglip != want_siglip {
                    log::warn!("[CLIP] Backend mismatch (loaded: {} desired: {}), reinitializing", if is_siglip {"SigLIP"} else {"FastEmbed"}, if want_siglip {"SigLIP"} else {"FastEmbed"});
                    must_reinit = true;
                }
            } else {
                must_reinit = true;
            }
            if must_reinit {
                *guard = None; // clear existing
            }
        }
        if must_reinit {
            // Recreate engine with desired key immediately
            if let Err(e) = super::clip::ensure_clip_engine(&self.clip_engine).await { log::error!("[CLIP] Re-init failed: {e}"); return 0; }
            // Log backend after reinit
            let guard = self.clip_engine.lock().await;
            if let Some(engine) = guard.as_ref() {
                let is_siglip = matches!(engine.backend, crate::ai::clip::ClipBackend::Siglip(_));
                log::info!("[CLIP] Active backend after reinit: {}", if is_siglip {"SigLIP"} else {"FastEmbed"});
            }
        } else {
            // Log currently active backend for visibility
            let guard = self.clip_engine.lock().await;
            if let Some(engine) = guard.as_ref() {
                let is_siglip = matches!(engine.backend, crate::ai::clip::ClipBackend::Siglip(_));
                log::info!("[CLIP] Active backend: {} (model: {})", if is_siglip {"SigLIP"} else {"FastEmbed"}, desired_key);
            }
        }
        let mut added = 0usize;
        log::info!("[CLIP] Starting generation for {} path(s)", paths.len());
        CLIP_STATUS.set_state(StatusState::Running, format!("Embedding {} images", paths.len()));
        CLIP_STATUS.set_progress(0, paths.len() as u64);
        for p in paths {
            let pb = std::path::Path::new(p);
            if !pb.exists() { log::warn!("[CLIP] Skip missing path {p}"); continue; }
            let maybe_meta = self.get_file_metadata(p).await;
            if maybe_meta.as_ref().map(|m| m.clip_embedding.is_some()).unwrap_or(false) { log::warn!("[CLIP] Skip already embedded {p}"); continue; }
            log::warn!("[CLIP] Embedding image {p}");
            let mut emb_opt = {
                let mut guard = self.clip_engine.lock().await;
                if let Some(engine) = guard.as_mut() { engine.embed_image_path(p).ok() } else { None }
            };

            if emb_opt.is_some() {
                let settings = crate::database::settings::load_settings();
                if settings.clip_augment_with_text {
                    if let Some(meta) = &maybe_meta {
                        log::warn!("[CLIP] Augmenting with text for {p}");
                        let mut text_pieces: Vec<String> = Vec::new();
                        if let Some(desc) = &meta.description { if desc.len() > 12 { text_pieces.push(desc.clone()); } }
                        if let Some(caption) = &meta.caption { text_pieces.push(caption.clone()); }
                        if !meta.tags.is_empty() { text_pieces.push(meta.tags.join(", ")); }
                        if let Some(cat) = &meta.category { text_pieces.push(cat.clone()); }
                        if !text_pieces.is_empty() {
                            let joined = text_pieces.join(" | ");
                            log::warn!("[CLIP] Text blend input for {p}: {}", joined);
                            let text_vec = {
                                let mut guard = self.clip_engine.lock().await;
                                if let Some(engine) = guard.as_mut() { engine.embed_text(&joined).ok() } else { None }
                            };
                            if let (Some(img_vec), Some(txt_vec)) = (emb_opt.as_ref(), text_vec.as_ref()) {
                                if img_vec.len() == txt_vec.len() {
                                    let blended: Vec<f32> = img_vec.iter().zip(txt_vec.iter()).map(|(a,b)| (a + b) * 0.5).collect();
                                    log::warn!("[CLIP] Blended image+text embedding for {p}");
                                    emb_opt = Some(blended);
                                }
                            }
                        }
                    }
                }
            }
            if let Some(vec) = emb_opt {
                {
                    let mut files = self.files.lock().await;
                    if let Some(fm) = files.iter_mut().find(|f| f.path == *p) {
                        fm.clip_embedding = Some(vec.clone());
                        if let Some(engine) = self.clip_engine.lock().await.as_mut() {
                            if fm.tags.len() < 2 { let mut tags = engine.zero_shot_tags(fm.clip_embedding.as_ref().unwrap(), 3); for t in tags.drain(..) { if !fm.tags.iter().any(|et| et == &t) { fm.tags.push(t); } } }
                            if fm.category.is_none() { fm.category = engine.zero_shot_category(fm.clip_embedding.as_ref().unwrap()); }
                        }
                    }
                }
                // Persist embedding row
                if let Some(meta2) = self.get_file_metadata(p).await { let _ = crate::database::upsert_clip_embedding(&meta2.path, meta2.hash.as_deref(), &vec).await; }
                // Best-effort persist updated tags/category/description if we synthesized them
                if let Some(updated) = self.get_file_metadata(p).await { let _ = self.cache_thumbnail_and_metadata(&updated).await; }
                added += 1;
                log::info!("[CLIP] Embedded {p}");
            }
            // progress update
            CLIP_STATUS.set_progress(added as u64, paths.len() as u64);
        }
        log::info!("[CLIP] Generation complete. Added {} new embeddings", added);
        CLIP_STATUS.set_state(StatusState::Idle, format!("Added {added}"));
        added
    }

    pub async fn clip_generate_recursive(&self) -> usize {
        let (targets, total, image_total, missing) = {
            let files = self.files.lock().await;
            let total = files.len();
            let image_total = files.iter().filter(|f| f.file_type == "image").count();
            let missing_vec: Vec<String> = files.iter().filter(|f| f.file_type == "image" && f.clip_embedding.is_none()).map(|f| f.path.clone()).collect();
            let missing = missing_vec.len();
            (missing_vec, total, image_total, missing)
        };
        log::info!("[CLIP] recursive: engine_files={total} images={image_total} missing_clip={missing}");
        self.clip_generate_for_paths(&targets).await
    }
}
