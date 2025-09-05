// (Removed unused base64 imports after pruning legacy vision code.)

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, Default)]
pub struct VisionDescription {
    pub description: String,
    pub caption: String,
    pub tags: Vec<String>,
    pub category: String,
}

impl super::AISearchEngine {
    // AI vision model description generation
    pub async fn generate_vision_description(
        &self,
        image_path: &std::path::PathBuf,
    ) -> Option<VisionDescription> {
        if !image_path.exists() {
            log::warn!("Image file does not exist: {:?}", image_path);
            return None;
        }
    // Prefer JoyCaption adapter when compiled + configured
        #[cfg(feature = "joycaption")]
        {
            // Offload to worker thread instead of running heavy model on the async/UI thread.
            if crate::ai::joycaption_adapter::is_enabled() {
                match tokio::fs::read(image_path).await {
                    Ok(bytes) if !bytes.is_empty() => {
                        let instruction = "Analyze the supplied image and return JSON with keys: description, caption, tags (array), category.";
                        match crate::ai::joycaption_adapter::stream_describe_bytes(bytes, instruction).await {
                            Ok(full) => {
                                log::info!("[joycaption.stream] collected {} chars", full.len());
                                if let Some(vd) = super::joycaption_adapter::extract_json_vision(&full)
                                    .and_then(|v| serde_json::from_value::<VisionDescription>(v).ok()) {
                                    return Some(vd);
                                } else {
                                    match crate::ai::joycaption_adapter::describe_image(image_path).await {
                                        Ok(vd2) => return Some(vd2),
                                        Err(e) => log::warn!("JoyCaption fallback describe failed: {e}"),
                                    }
                                }
                            }
                            Err(e) => {
                                log::error!("JoyCaption stream_describe_bytes failed: {e}");
                                if let Ok(vd) = crate::ai::joycaption_adapter::describe_image(image_path).await { return Some(vd); }
                            }
                        }
                    }
                    Ok(_) | Err(_) => { /* fall through to None */ }
                }
            }
        }
        // No other vision backend present; JoyCaption already attempted if enabled.
        log::info!("[AI] Vision description unavailable (no active vision backend) for {:?}", image_path);
        return None;
    }

    // Generate (or regenerate if force) description for a single path without re-indexing document table.
    pub async fn _generate_description_for_path(
        &self,
        path: &str,
        force: bool,
    ) -> anyhow::Result<Option<String>, anyhow::Error> {
        let pb = std::path::PathBuf::from(path);
        if !pb.exists() {
            return Ok(None);
        }

        {
            let files = self.files.lock().await;
            if !force {
                if let Some(f) = files.iter().find(|f| f.path == path) {
                    if f.description.is_some()
                        && f.description
                            .as_ref()
                            .map(|d| d.trim().len() >= 12)
                            .unwrap_or(false)
                    {
                        return Ok(f.description.clone());
                    }
                }
            }
        }

        if let Some(vd) = self.generate_vision_description(&pb).await {
            // Update & persist
            if let Some(mut meta_inner) = self.get_file_metadata(path).await {
                meta_inner.description = Some(vd.description.clone());
                meta_inner.caption = Some(vd.caption.clone());
                // Use tags directly from structured vision response
                meta_inner.tags = vd.tags.clone();
                meta_inner.category = if vd.category.trim().is_empty() { None } else { Some(vd.category.clone()) };
                // Replace existing metadata in-memory
                {
                    let mut files = self.files.lock().await;
                    if let Some(idx) = files.iter().position(|f| f.path == path) {
                        files[idx] = meta_inner.clone();
                    }
                }
                if let Err(e) = self.cache_thumbnail_and_metadata(&meta_inner).await {
                    log::warn!("Failed to persist updated description for {}: {}", path, e);
                }
                return Ok(meta_inner.description.clone());
            }
            Ok(None)
        } else {
            Ok(None)
        }
    }

    // Generate an image from a prompt (placeholder implementation creates blank image w/ metadata header file)
    #[allow(dead_code)]
    pub async fn generate_image(
        &self,
        prompt: &str,
    ) -> anyhow::Result<std::path::PathBuf, anyhow::Error> {
        use image::{ImageBuffer, Rgba};
        // Ensure directory
        let out_dir = std::path::PathBuf::from("./generated");
        if !out_dir.exists() {
            std::fs::create_dir_all(&out_dir)?;
        }
        // Create slug filename
        let slug = prompt
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == ' ')
            .collect::<String>()
            .split_whitespace()
            .take(6)
            .collect::<Vec<_>>()
            .join("_")
            .to_lowercase();
        let ts = chrono::Local::now().format("%Y%m%d%H%M%S");
        let filename = if slug.is_empty() {
            format!("gen_{}.png", ts)
        } else {
            format!("{}_{}.png", slug, ts)
        };
        let out_path = out_dir.join(filename);
        // Simple gradient image as placeholder
        let imgx = 512;
        let imgy = 512;
        let mut imgbuf: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(imgx, imgy);
        for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
            let r = (x as f32 / imgx as f32 * 255.0) as u8;
            let g = (y as f32 / imgy as f32 * 255.0) as u8;
            let b = 128u8;
            *pixel = Rgba([r, g, b, 255]);
        }
        imgbuf.save(&out_path)?;
        log::info!(
            "Generated placeholder image: {:?} for prompt '{}'",
            out_path,
            prompt
        );

        // Index generated image metadata
        let meta = super::FileMetadata {
            id: None,
            path: out_path.to_string_lossy().to_string(),
            filename: out_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string(),
            file_type: "image".to_string(),
            size: std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0),
            modified: Some(chrono::Local::now()),
            created: Some(chrono::Local::now()),
            thumbnail_path: None,
            thumb_b64: None,
            hash: self.compute_file_hash(&out_path).ok(),
            description: Some(format!("Placeholder generated for prompt: {}", prompt)),
            caption: Some(format!("generated image: {}", prompt)),
            tags: vec!["generated".into()],
            category: Some("generated".into()),
            embedding: None,
            similarity_score: None,
            clip_embedding: None,
            clip_similarity_score: None,
        };
        // Ignore errors silently for now
        let _ = self.index_file(meta).await;

        Ok(out_path)
    }

}

// NOTE: Direct conversion to Thumbnail removed; callers now construct persistence rows via cache layer merge.