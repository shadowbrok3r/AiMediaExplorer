
use fastembed::{ImageEmbedding, ImageInitOptions, ImageEmbeddingModel, TextEmbedding, TextInitOptions, EmbeddingModel};

fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 { for x in &mut v { *x /= norm; } }
    v
}

pub struct ClipEngine {
    pub(crate) image_model: ImageEmbedding,
    pub(crate) text_model: TextEmbedding,
}


impl ClipEngine {
    pub fn new_default() -> anyhow::Result<Self> {
        // Use matching CLIP model variant for image & text
        let img_model = ImageEmbedding::try_new(ImageInitOptions::new(ImageEmbeddingModel::UnicomVitB32))?;
        let txt_model = TextEmbedding::try_new(TextInitOptions::new(EmbeddingModel::ClipVitB32))?;
        Ok(Self { image_model: img_model, text_model: txt_model })
    }

    pub fn embed_image_path(&mut self, path: &str) -> anyhow::Result<Vec<f32>> {
        let out = self.image_model.embed(vec![path.to_string()], None)?; // single
        Ok(l2_normalize(out.into_iter().next().unwrap()))
    }

    pub fn embed_text(&mut self, text: &str) -> anyhow::Result<Vec<f32>> {
        let out = self.text_model.embed(vec![text.to_string()], None)?;
        Ok(l2_normalize(out.into_iter().next().unwrap()))
    }

    pub fn zero_shot_tags(&mut self, _image_vec: &[f32], _top_k: usize) -> Vec<String> { Vec::new() }

    pub fn zero_shot_category(&mut self, _image_vec: &[f32]) -> Option<String> { None }

}


fn dot(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x,y)| x*y).sum() }

pub(crate) async fn ensure_clip_engine(engine_slot: &std::sync::Arc<tokio::sync::Mutex<Option<ClipEngine>>>) -> anyhow::Result<()> {
    let mut guard = engine_slot.lock().await;
    if guard.is_none() {
        log::info!("[CLIP] Loading fastembed CLIP models (ViT-B/32)...");
        match ClipEngine::new_default() {
            Ok(c) => { *guard = Some(c); log::info!("[CLIP] Loaded."); },
            Err(e) => { log::error!("[CLIP] Failed to init: {e}"); return Err(e); }
        }
    }
    Ok(())
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 { dot(a,b) }

// -----------------------------------------------------------------------------
// AISearchEngine extension methods (placed here for proximity to CLIP logic)
// -----------------------------------------------------------------------------
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
        if self.ensure_clip_engine().await.is_err() { return 0; }
        let mut added = 0usize;
        log::info!("[CLIP] Starting generation for {} path(s)", paths.len());
        for p in paths {
            let pb = std::path::Path::new(p);
            if !pb.exists() { log::warn!("[CLIP] Skip missing path {p}"); continue; }
            let maybe_meta = self.get_file_metadata(p).await;
            if maybe_meta.as_ref().map(|m| m.clip_embedding.is_some()).unwrap_or(false) { log::debug!("[CLIP] Skip already embedded {p}"); continue; }
            log::debug!("[CLIP] Embedding image {p}");
            let mut emb_opt = {
                let mut guard = self.clip_engine.lock().await;
                if let Some(engine) = guard.as_mut() { engine.embed_image_path(p).ok() } else { None }
            };
            // Optional augmentation: if enabled, average with text embedding derived from existing metadata
            if emb_opt.is_some() {
                let settings = crate::database::settings::load_settings();
                if settings.clip_augment_with_text {
                    if let Some(meta) = &maybe_meta {
                        log::debug!("[CLIP] Augmenting with text for {p}");
                        let mut text_pieces: Vec<String> = Vec::new();
                        if let Some(desc) = &meta.description { if desc.len() > 12 { text_pieces.push(desc.clone()); } }
                        if let Some(caption) = &meta.caption { text_pieces.push(caption.clone()); }
                        if !meta.tags.is_empty() { text_pieces.push(meta.tags.join(", ")); }
                        if let Some(cat) = &meta.category { text_pieces.push(cat.clone()); }
                        if !text_pieces.is_empty() {
                            let joined = text_pieces.join(" | ");
                            log::trace!("[CLIP] Text blend input for {p}: {}", joined);
                            let text_vec = {
                                let mut guard = self.clip_engine.lock().await;
                                if let Some(engine) = guard.as_mut() { engine.embed_text(&joined).ok() } else { None }
                            };
                            if let (Some(img_vec), Some(txt_vec)) = (emb_opt.as_ref(), text_vec.as_ref()) {
                                if img_vec.len() == txt_vec.len() {
                                    let blended: Vec<f32> = img_vec.iter().zip(txt_vec.iter()).map(|(a,b)| (a + b) * 0.5).collect();
                                    log::debug!("[CLIP] Blended image+text embedding for {p}");
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
        }
        log::info!("[CLIP] Generation complete. Added {} new embeddings", added);
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
