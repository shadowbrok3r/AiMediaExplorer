use eframe::egui::*;

#[derive(Default)]
pub struct AssistantPanel {
    pub prompt: String,
    pub progress: Option<String>,
    pub last_reply: String,
    pub attach_current_image: bool,
    // Refinement UI state
    pub ref_old_category: String,
    pub ref_new_category: String,
    pub ref_old_tag: String,
    pub ref_new_tag: String,
    pub ref_delete_tag: String,
    pub ref_limit_tags: i32,
}

impl AssistantPanel {
    pub fn ui(&mut self, ui: &mut Ui, explorer: &mut crate::ui::file_table::FileExplorer) {
        ui.heading("AI Assistant");
        ui.label("Ask about your images. When an image is selected, we can send it to the local multimodal model (JoyCaption / LLaVA).");
        ui.separator();

        ui.horizontal(|ui| {
            ui.checkbox(&mut self.attach_current_image, "Use selected image");
            if ui.button("Open Vision Model").clicked() {
                tokio::spawn(async {
                    let _ = crate::ai::joycap::ensure_worker_started().await;
                });
            }
        });

        ui.add_sized([ui.available_width(), 120.0], TextEdit::multiline(&mut self.prompt).hint_text("Describe what you're looking for, or ask about the selected image..."));
        ui.horizontal(|ui| {
            if ui.button("Ask").clicked() {
                self.last_reply.clear();
                self.progress = Some(String::new());
                let prompt = self.prompt.clone();
                let maybe_path = if self.attach_current_image { Some(explorer.current_thumb.path.clone()) } else { None };
                let tx_updates = explorer.viewer.ai_update_tx.clone();
                // Stream via joycap (LLaVA). If an image is provided, attach it; else do text-only behavior.
                if let Some(path) = maybe_path {
                    if std::path::Path::new(&path).is_file() {
                        let instruction = prompt.clone();
                        tokio::spawn(async move {
                            match tokio::fs::read(&path).await {
                                Ok(bytes) => {
                                    let _ = crate::ai::joycap::ensure_worker_started().await;
                                    let _ = crate::ai::joycap::stream_describe_bytes_with_callback(bytes, &instruction, |tok| {
                                        let _ = tx_updates.try_send(crate::ui::file_table::AIUpdate::Interim { path: path.clone(), text: tok.to_string() });
                                    }).await;
                                }
                                Err(e) => log::error!("read image failed: {e}"),
                            }
                        });
                    }
                } else {
                    // Text-only: use CLIP text embedding to find candidates similar to the instruction
                    let query = prompt.clone();
                    tokio::spawn(async move {
                        let _ = crate::ai::GLOBAL_AI_ENGINE.ensure_clip_engine().await;
                        let mut results: Vec<crate::ui::file_table::SimilarResult> = Vec::new();
                        let q_vec_opt = {
                            let mut guard = crate::ai::GLOBAL_AI_ENGINE.clip_engine.lock().await;
                            if let Some(engine) = guard.as_mut() { engine.embed_text(&query).ok() } else { None }
                        };
                        if let Some(q) = q_vec_opt {
                            match crate::database::ClipEmbeddingRow::find_similar_by_embedding(&q, 64, 128).await {
                                Ok(hits) => {
                                    for hit in hits.into_iter() {
                                        let thumb = if let Some(t) = hit.thumb_ref { t } else { crate::Thumbnail::get_thumbnail_by_path(&hit.path).await.unwrap_or(None).unwrap_or_default() };
                                        results.push(crate::ui::file_table::SimilarResult { thumb, created: None, updated: None, similarity_score: None, clip_similarity_score: Some(hit.dist) });
                                    }
                                }
                                Err(e) => log::error!("text search knn failed: {e}"),
                            }
                        }
                        let _ = tx_updates.try_send(crate::ui::file_table::AIUpdate::SimilarResults { origin_path: format!("query:{query}"), results });
                    });
                }
            }
            if ui.button("Find by Text").on_hover_text("Use CLIP/SigLIP to search images by this prompt").clicked() {
                let query = self.prompt.clone();
                let tx_updates = explorer.viewer.ai_update_tx.clone();
                tokio::spawn(async move {
                    let _ = crate::ai::GLOBAL_AI_ENGINE.ensure_clip_engine().await;
                    let mut results: Vec<crate::ui::file_table::SimilarResult> = Vec::new();
                    let q_vec_opt = {
                        let mut guard = crate::ai::GLOBAL_AI_ENGINE.clip_engine.lock().await;
                        if let Some(engine) = guard.as_mut() { engine.embed_text(&query).ok() } else { None }
                    };
                    if let Some(q) = q_vec_opt {
                        match crate::database::ClipEmbeddingRow::find_similar_by_embedding(&q, 64, 128).await {
                            Ok(hits) => {
                                for hit in hits.into_iter() {
                                    let thumb = if let Some(t) = hit.thumb_ref { t } else { crate::Thumbnail::get_thumbnail_by_path(&hit.path).await.unwrap_or(None).unwrap_or_default() };
                                    results.push(crate::ui::file_table::SimilarResult { thumb, created: None, updated: None, similarity_score: None, clip_similarity_score: Some(hit.dist) });
                                }
                            }
                            Err(e) => log::error!("text search knn failed: {e}"),
                        }
                    }
                    let _ = tx_updates.try_send(crate::ui::file_table::AIUpdate::SimilarResults { origin_path: format!("query:{query}"), results });
                });
            }
        });

        ui.separator();
        CollapsingHeader::new("Refine Categories and Tags")
            .default_open(false)
            .show(ui, |ui| {
                ui.label("Rename / merge Category");
                ui.horizontal(|ui| {
                    ui.add_sized([220.0, 20.0], TextEdit::singleline(&mut self.ref_old_category).hint_text("Old category"));
                    ui.add_sized([220.0, 20.0], TextEdit::singleline(&mut self.ref_new_category).hint_text("New category"));
                    if ui.button("Apply").clicked() {
                        let old = self.ref_old_category.clone();
                        let newc = self.ref_new_category.clone();
                        tokio::spawn(async move {
                            match crate::Thumbnail::rename_category(&old, &newc).await {
                                Ok(n) => log::info!("Renamed category '{old}' -> '{newc}' ({n} rows)"),
                                Err(e) => log::error!("Category rename failed: {e}"),
                            }
                        });
                    }
                });
                ui.separator();
                ui.label("Rename / merge Tag");
                ui.horizontal(|ui| {
                    ui.add_sized([220.0, 20.0], TextEdit::singleline(&mut self.ref_old_tag).hint_text("Old tag"));
                    ui.add_sized([220.0, 20.0], TextEdit::singleline(&mut self.ref_new_tag).hint_text("New tag"));
                    if ui.button("Apply").clicked() {
                        let old = self.ref_old_tag.clone();
                        let newt = self.ref_new_tag.clone();
                        tokio::spawn(async move {
                            match crate::Thumbnail::rename_tag(&old, &newt).await {
                                Ok(n) => log::info!("Renamed tag '{old}' -> '{newt}' ({n} rows)"),
                                Err(e) => log::error!("Tag rename failed: {e}"),
                            }
                        });
                    }
                });
                ui.separator();
                ui.label("Delete Tag");
                ui.horizontal(|ui| {
                    ui.add_sized([220.0, 20.0], TextEdit::singleline(&mut self.ref_delete_tag).hint_text("Tag to delete"));
                    if ui.button("Delete").clicked() {
                        let tag = self.ref_delete_tag.clone();
                        tokio::spawn(async move {
                            match crate::Thumbnail::delete_tag(&tag).await {
                                Ok(n) => log::info!("Deleted tag '{tag}' from {n} rows"),
                                Err(e) => log::error!("Delete tag failed: {e}"),
                            }
                        });
                    }
                });
                ui.separator();
                ui.label("Limit number of tags per item");
                ui.horizontal(|ui| {
                    ui.add(DragValue::new(&mut self.ref_limit_tags).range(0..=64));
                    if ui.button("Prune").clicked() {
                        let limit = self.ref_limit_tags as i64;
                        tokio::spawn(async move {
                            match crate::Thumbnail::prune_tags(limit).await {
                                Ok(n) => log::info!("Pruned tags to {limit} on {n} rows"),
                                Err(e) => log::error!("Prune tags failed: {e}"),
                            }
                        });
                    }
                });
            });
        if let Some(p) = &self.progress {
            ui.colored_label(Color32::LIGHT_BLUE, "Streaming…");
            ScrollArea::vertical().max_height(200.0).show(ui, |ui| {
                ui.label(p.as_str());
            });
        }
        if !self.last_reply.is_empty() {
            ui.separator();
            ui.heading("Last Reply");
            ui.label(&self.last_reply);
        }
    }
}
