use eframe::egui::*;
use humansize::DECIMAL;

impl super::FileExplorer {
    /// Draws the selected row's picture: the cached thumbnail at once, replaced by the larger preview when it is ready.
    fn preview_image(&self, ui: &mut Ui) {
        let (thumbs, scope, path) = (&self.viewer.thumbs, self.viewer.scope, self.current_thumb.path.as_str());
        let view = if path.is_empty() || self.current_thumb.file_type == "<DIR>" || self.current_thumb.file_type == "<ARCHIVE>" {
            super::thumbs::ThumbView::Missing
        } else {
            let thumb = thumbs.request(scope, path, super::thumbs::Tier::Thumb, 0.0);
            let preview = if super::thumbs::can_preview(path) { thumbs.request(scope, path, super::thumbs::Tier::Preview, 0.0) } else { super::thumbs::ThumbView::Missing };
            if preview.is_ready() {
                preview
            } else if thumb.is_ready() {
                thumb
            } else if thumb == super::thumbs::ThumbView::Pending || preview == super::thumbs::ThumbView::Pending {
                super::thumbs::ThumbView::Pending
            } else {
                super::thumbs::ThumbView::Missing
            }
        };
        match view {
            super::thumbs::ThumbView::Ready { id, size } => {
                Image::from_texture(load::SizedTexture::new(id, size)).max_size(ui.available_size() / 1.3).ui(ui);
            }
            super::thumbs::ThumbView::Pending => {
                ui.weak("Loading...");
            }
            super::thumbs::ThumbView::Missing => {
                ui.label("No Image");
            }
        }
    }

    /// Plays the selected video in place, or draws the still picture for any other file.
    #[cfg(ame_ffmpeg)]
    fn video_or_image(&self, ui: &mut Ui) {
        let is_video = self.current_thumb.file_type != "<DIR>"
            && self.current_thumb.file_type != "<ARCHIVE>"
            && std::path::Path::new(&self.current_thumb.path).extension().and_then(|e| e.to_str()).is_some_and(|ext| crate::is_video(&ext.to_ascii_lowercase()));
        let slot = crate::video::view::slot(ui.ctx(), ("preview", self.viewer.scope));
        let mut view = slot.lock().unwrap_or_else(|e| e.into_inner());
        if is_video {
            let max = ui.available_size() / 1.3;
            view.ui(ui, std::path::Path::new(&self.current_thumb.path), max, |ui| self.preview_image(ui));
        } else {
            view.reset();
            self.preview_image(ui);
        }
    }

    pub fn preview_pane(&mut self, ui: &mut Ui) {
        let mut open = self.open_preview_pane;
        let mut close = false;
        Panel::right("MainPageRightPanel")
            .default_size(400.)
            .max_size(800.)
            .show_collapsible(ui, &mut open, |ui| {
                ui.vertical_centered(|ui| {
                    let name = &self.current_thumb.filename;
                    ui.horizontal(|ui| {
                        let btn = Button::new(
                            RichText::new(name).heading().color(ui.style().visuals.error_fg_color)
                        )
                        .ui(ui)
                        .on_hover_text("Left Click: Open File\nDouble Click: Open Folder\nRight Click: Open Folder");
                        
                        if btn.clicked() {
                            let _ = open::that(self.current_thumb.path.clone());
                        } else if btn.secondary_clicked() || btn.double_clicked() {
                            let pb = std::path::PathBuf::from(self.current_thumb.path.clone());
                            if let Some(parent) = pb.parent() { 
                                let _ = open::that(parent); 
                            }
                        }
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            if ui.button(RichText::new("✖").color(ui.style().visuals.error_fg_color)).on_hover_text("Close panel").clicked() { 
                                close = true; 
                            }
                            let can_edit = self.current_thumb.file_type != "<DIR>";
                            if ui.add_enabled(can_edit, Button::new("Edit Image")).on_hover_text("Open image editing tab for this image").clicked() {
                                // Open the Image Edit tab with this image preloaded
                                let path = self.current_thumb.path.clone();
                                crate::app::OPEN_TAB_REQUESTS.lock().unwrap()
                                    .push(crate::ui::file_table::FilterRequest::NewTab {
                                        title: "Image Edit".to_string(),
                                        rows: vec![],
                                        showing_similarity: false,
                                        similar_scores: None,
                                        origin_path: Some(path.clone()),
                                        background: false,
                                    });
                            }
                        });
                    });
                    ui.separator();
                    #[cfg(ame_ffmpeg)]
                    self.video_or_image(ui);
                    #[cfg(not(ame_ffmpeg))]
                    self.preview_image(ui);

                     ui.separator();

                     ui.horizontal(|ui| {
                         ui.label(RichText::new("Size:").underline().strong());
                         ui.label(humansize::format_size(self.current_thumb.size, DECIMAL));
                         ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                             ui.label(&self.current_thumb.file_type.to_uppercase());
                             // CLIP badge considers both path and content-hash presence (duplicates across paths)
                             let has_clip = {
                                 let by_path = self.clip_presence.contains(&self.current_thumb.path);
                                 let by_hash = self
                                     .current_thumb
                                     .hash
                                     .as_ref()
                                     .map(|h| self.viewer.clip_presence_hashes.contains(h))
                                     .unwrap_or(false);
                                 by_path || by_hash
                             };
                             let badge = if has_clip { 
                                 RichText::new("CLIP").color(crate::theme::success(ui)) 
                             } else { 
                                 RichText::new("CLIP").color(ui.style().visuals.error_fg_color) 
                             };

                             ui.label(badge);
                             ui.label(RichText::new("Type:").underline().strong());
                         });
                     });

                     ScrollArea::vertical()
                     .auto_shrink(false)
                     .show(ui, |ui| {
                         // -------- AI Description & Editable Metadata --------
                         ui.heading("Ai Description");
                         ui.group(|ui| {
                             let path_key = self.current_thumb.path.clone();

                             // Streaming interim text (if any)
                             if let Some(interim) = self.streaming_interim.get(&path_key) {
                                 ui.colored_label(crate::theme::info(ui), RichText::new("(Streaming…)"));
                                 ScrollArea::vertical().max_height(160.).show(ui, |ui| {
                                     ui.label(interim);
                                 });
                                 ui.separator();
                             }

                             // Description editor
                             if let Some(desc) = &mut self.current_thumb.description {
                                 let resp = TextEdit::multiline(desc)
                                 .desired_width(ui.available_width())
                                 .desired_rows(6)
                                 .ui(ui);
                            
                                 if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                                     let path_clone = path_key.clone();
                                     let desc_clone = desc.clone();
                                     tokio::spawn(async move {
                                         if let Some(meta) = crate::ai::GLOBAL_AI_ENGINE.get_file_metadata(&path_clone).await {
                                             let mut new_meta = meta.clone();
                                             new_meta.description = Some(desc_clone);
                                             if let Err(e) = crate::ai::GLOBAL_AI_ENGINE.cache_thumbnail_and_metadata(&new_meta).await { log::warn!("Persist edited description failed: {e}"); }
                                         }
                                     });
                                 }
                             } else if self.streaming_interim.get(&path_key).is_none() {
                                 ui.label("No Description yet");
                                 ui.add_space(30.);
                             }

                             ui.add_space(8.);
                             ui.label(RichText::new("Caption:").underline().strong());
                             let mut cap = self.current_thumb.caption.clone().unwrap_or_default();
                             let cap_resp = TextEdit::singleline(&mut cap).desired_width(ui.available_width()).ui(ui);
                             if cap_resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                                 self.current_thumb.caption = if cap.trim().is_empty() { None } else { Some(cap.clone()) };
                                 let path_clone = path_key.clone();
                                 let cap_opt = self.current_thumb.caption.clone();
                                 tokio::spawn(async move {
                                     if let Some(meta) = crate::ai::GLOBAL_AI_ENGINE.get_file_metadata(&path_clone).await {
                                         let mut new_meta = meta.clone();
                                         new_meta.caption = cap_opt.clone();
                                         if let Err(e) = crate::ai::GLOBAL_AI_ENGINE.cache_thumbnail_and_metadata(&new_meta).await { log::warn!("Persist edited caption failed: {e}"); }
                                     }
                                 });
                             }

                             ui.add_space(8.);
                             ui.label(RichText::new("Category:").underline().strong());
                             let mut cat = self.current_thumb.category.clone().unwrap_or_default();
                             let cat_resp = TextEdit::singleline(&mut cat).desired_width(ui.available_width()).ui(ui);
                             if cat_resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                                 self.current_thumb.category = if cat.trim().is_empty() { None } else { Some(cat.clone()) };
                                 let path_clone = path_key.clone();
                                 let cat_opt = self.current_thumb.category.clone();
                                 tokio::spawn(async move {
                                     if let Some(meta) = crate::ai::GLOBAL_AI_ENGINE.get_file_metadata(&path_clone).await {
                                         let mut new_meta = meta.clone();
                                         new_meta.category = cat_opt.clone();
                                         if let Err(e) = crate::ai::GLOBAL_AI_ENGINE.cache_thumbnail_and_metadata(&new_meta).await { log::warn!("Persist edited category failed: {e}"); }
                                     }
                                 });
                             }

                             ui.add_space(8.);
                             ui.label(RichText::new("Tags (comma-separated):").underline().strong());
                             let mut tags_line = self.current_thumb.tags.join(", ");
                             let tags_resp = TextEdit::singleline(&mut tags_line).desired_width(ui.available_width()).ui(ui);
                             if tags_resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                                 let new_tags: Vec<String> = tags_line.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
                                 self.current_thumb.tags = new_tags.clone();
                                 let path_clone = path_key.clone();
                                 tokio::spawn(async move {
                                     if let Some(meta) = crate::ai::GLOBAL_AI_ENGINE.get_file_metadata(&path_clone).await {
                                         let mut new_meta = meta.clone();
                                         new_meta.tags = new_tags.clone();
                                         if let Err(e) = crate::ai::GLOBAL_AI_ENGINE.cache_thumbnail_and_metadata(&new_meta).await { 
                                             log::warn!("Persist edited tags failed: {e}"); 
                                         }
                                     }
                                 });
                             }
                             ui.horizontal_wrapped(|ui| {
                                 for tag in &self.current_thumb.tags { ui.label(RichText::new(tag)); }
                             });

                             // Completeness warning
                             let have_all = self.current_thumb.caption.is_some()
                                 && self.current_thumb.category.is_some()
                                 && !self.current_thumb.tags.is_empty()
                                 && self.current_thumb.description.as_ref().map(|d| !d.trim().is_empty()).unwrap_or(false);
                             if !have_all && self.current_thumb.description.is_some() && self.streaming_interim.get(&path_key).is_none() {
                                 ui.colored_label(crate::theme::warn(ui), "Parsed AI metadata incomplete (expected caption, category, tags, description)");
                             }

                             ui.add_space(10.);
                             ui.horizontal(|ui| {
                                 let overwrite_allowed = self.viewer.ui_settings.overwrite_descriptions;
                                 let already_has = self.current_thumb.description.is_some();
                                 let can_generate = !self.streaming_interim.contains_key(&path_key) && (!already_has || overwrite_allowed);
                                 let btn_text = if already_has { if overwrite_allowed { "Regenerate (Overwrite)" } else { "Generated" } } else { "Generate Description" };
                             
                                 let btn = ui.add_enabled(can_generate, Button::new(btn_text));
                                 if self.streaming_interim.contains_key(&path_key) {
                                     ui.label(RichText::new("Generating…").weak());
                                     Spinner::new().ui(ui);
                                 }
                                 if btn.clicked() && can_generate {
                                     // Start streaming generation for this single file
                                     let engine = std::sync::Arc::new(crate::ai::GLOBAL_AI_ENGINE.clone());
                                     let path_for_stream = path_key.clone();
                                     self.streaming_interim.insert(path_for_stream.clone(), String::new());
                                     let tx_updates = self.viewer.ai_update_tx.clone();
                                     let prompt = self.viewer.ui_settings.ai_prompt_template.clone();
                                     tokio::spawn(async move {
                                         let path_string = path_for_stream;
                                         let arc_path = std::sync::Arc::new(path_string);
                                         let engine_call_path = (*arc_path).clone();
                                         let cb_path = arc_path.clone();
                                         engine.stream_vision_description(std::path::Path::new(&engine_call_path), &prompt, move |interim, final_opt| {
                                             let p_clone = (*cb_path).clone();
                                             if let Some(vd) = final_opt {
                                                 let _ = tx_updates.try_send(super::AIUpdate::Final {
                                                     path: p_clone,
                                                     description: vd.description.clone(),
                                                     caption: Some(vd.caption.clone()),
                                                     category: if vd.category.trim().is_empty() { None } else { Some(vd.category.clone()) },
                                                     tags: vd.tags.clone(),
                                                 });
                                             } else {
                                                 let _ = tx_updates.try_send(super::AIUpdate::Interim { path: p_clone, text: interim.to_string() });
                                             }
                                         }).await;
                                     });
                                 }
                               
                                // Similarity search (use DB-wide KNN via clip_embeddings)
                                if ui.button("Find Similar").clicked() {
                                    log::info!("Finding similar");
                                    let sel_path = path_key.clone();
                                    let engine = std::sync::Arc::new(crate::ai::GLOBAL_AI_ENGINE.clone());
                                    let tx_updates = self.viewer.ai_update_tx.clone();
                                    tokio::spawn(async move {
                                        // Compute query embedding once
                                        let q_vec_opt: Option<Vec<f32>> = {
                                            let mut guard = engine.clip_engine.lock().await;
                                            if let Some(eng) = guard.as_mut() { eng.embed_image_path(&sel_path).ok() } else { None }
                                        };
                                        let mut results: Vec<super::SimilarResult> = Vec::new();
                                        if let Some(q) = q_vec_opt {
                                            
                                            // Try DB-side KNN first
                                            match crate::database::ClipEmbeddingRow::find_similar_by_embedding(&q, 24, 64, 0).await {
                                                Ok(hits) => {
                                                    for hit in hits.into_iter() {
                                                        // Get thumbnail record (prefer embedded thumb_ref)
                                                        let thumb = if let Some(t) = hit.thumb_ref { t } else { crate::Thumbnail::get_thumbnail_by_path(&hit.path).await.unwrap_or(None).unwrap_or_default() };
                                                        // Enrich with created/updated and stored sims from this path's embedding row
                                                        let (mut created, mut updated, mut stored_sim, mut clip_sim) = (None, None, None, None);
                                                        if let Ok(rows) = crate::database::ClipEmbeddingRow::load_clip_embeddings_for_path(&hit.path).await {
                                                            if let Some(row) = rows.into_iter().next() {
                                                                created = row.created;
                                                                updated = row.updated;
                                                                stored_sim = row.similarity_score.or(row.clip_similarity_score);
                                                                if !row.embedding.is_empty() && row.embedding.len() == q.len() {
                                                                    clip_sim = Some(crate::ai::clip::dot(&q, &row.embedding));
                                                                }
                                                            }
                                                        }
                                                        results.push(super::SimilarResult { thumb, created, updated, similarity_score: stored_sim, clip_similarity_score: clip_sim });
                                                    }
                                                }
                                                Err(e) => {
                                                    log::error!("find_similar_by_embedding failed: {e:?}");
                                                }
                                            }
                                        }
                                        let _ = tx_updates.try_send(super::AIUpdate::SimilarResults { origin_path: sel_path.clone(), results });
                                    });
                                }
                                // Alternate in-memory engine similarity (for comparison)
                                if ui.button("Find Similar (Engine)").on_hover_text("Compare with in-memory engine scoring").clicked() {
                                    let sel_path = path_key.clone();
                                    let tx_updates = self.viewer.ai_update_tx.clone();
                                    tokio::spawn(async move {
                                        let top = 24usize;
                                        // Use the existing engine helper
                                        let hits = crate::ai::GLOBAL_AI_ENGINE.clip_search_image(&sel_path, top).await;
                                        // Build SimilarResults with score computed vs each hit's embedding
                                        let mut results: Vec<super::SimilarResult> = Vec::new();
                                        // Compute the query embedding once
                                        let q_vec_opt: Option<Vec<f32>> = {
                                            let mut guard = crate::ai::GLOBAL_AI_ENGINE.clip_engine.lock().await;
                                            if let Some(eng) = guard.as_mut() { eng.embed_image_path(&sel_path).ok() } else { None }
                                        };
                                        for thumb in hits.into_iter() {
                                            let mut created = None;
                                            let mut updated = None;
                                            let mut sim = None;
                                            let mut clip_sim = None;
                                            if let Ok(rows) = crate::database::ClipEmbeddingRow::load_clip_embeddings_for_path(&thumb.path).await {
                                                if let Some(row) = rows.into_iter().next() {
                                                    created = row.created;
                                                    updated = row.updated;
                                                    sim = row.similarity_score.or(row.clip_similarity_score);
                                                    if let (Some(q), emb) = (q_vec_opt.as_ref(), row.embedding) {
                                                        if !emb.is_empty() && emb.len() == q.len() {
                                                            clip_sim = Some(crate::ai::clip::dot(q, &emb));
                                                        }
                                                    }
                                                }
                                            }
                                            results.push(super::SimilarResult { thumb, created, updated, similarity_score: sim, clip_similarity_score: clip_sim });
                                        }
                                        let _ = tx_updates.try_send(super::AIUpdate::SimilarResults { origin_path: sel_path.clone(), results });
                                    });
                                }
                                
                                if ui.button("Generate CLIP").clicked() {
                                    let path = self.current_thumb.path.clone();
                                    tokio::spawn(async move {
                                        // Ensure engine and model are ready
                                        let _ = crate::ai::GLOBAL_AI_ENGINE.ensure_clip_engine().await;
                                        match crate::ai::GLOBAL_AI_ENGINE.clip_generate_for_paths(&[path.clone()]).await {
                                            Ok(added) => log::info!("[CLIP] Manual per-item generation: added {added} for {path}"),
                                            Err(e) => log::error!("engine.clip_generate_for_paths: {e:?}")
                                        }
                                        Ok::<(), anyhow::Error>(())
                                    });
                                }
                            });
                        });
                    });
                });
            });
        self.open_preview_pane = open && !close;
    }
}

#[cfg(all(test, ame_ffmpeg))]
mod tests {
    use super::super::FileExplorer;
    use crate::video::fixtures as fx;
    use crate::video::player::PlayerOptions;
    use eframe::egui;
    use std::path::Path;
    use std::time::{Duration, Instant};

    fn frame(ctx: &egui::Context, ex: &mut FileExplorer) {
        let input = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1100.0, 800.0))), ..Default::default() };
        ctx.run_ui(input, |ui| {
            egui::CentralPanel::default().show(ui, |ui| ex.preview_pane(ui));
        })
        .drop_without_applying_deltas();
    }

    fn select(ex: &mut FileExplorer, path: &Path, file_type: &str) {
        ex.current_thumb.path = path.display().to_string();
        ex.current_thumb.filename = path.file_name().unwrap().to_string_lossy().to_string();
        ex.current_thumb.file_type = file_type.to_string();
    }

    fn wait_frames(ctx: &egui::Context, ex: &mut FileExplorer, timeout: Duration, mut done: impl FnMut() -> bool) -> bool {
        let end = Instant::now() + timeout;
        while Instant::now() < end {
            frame(ctx, ex);
            if done() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        false
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn the_pane_plays_the_selected_video_and_stops_when_the_selection_or_pane_changes() {
        if !fx::available() {
            return fx::skip("the_pane_plays_the_selected_video");
        }
        let dir = tempfile::tempdir().unwrap();
        let (Some(a), Some(b)) = (fx::h264_mp4(dir.path(), "a.mp4", 3), fx::h264_mp4(dir.path(), "b.mp4", 3)) else { return fx::skip("pane clips") };
        let png = dir.path().join("still.png");
        image::RgbImage::from_pixel(32, 32, image::Rgb([9, 9, 9])).save(&png).unwrap();
        let ctx = crate::ui::test_support::test_ctx();
        let mut ex = FileExplorer::new(true);
        ex.open_preview_pane = true;
        select(&mut ex, &a, "mp4");
        frame(&ctx, &mut ex);
        let slot = crate::video::view::slot(&ctx, ("preview", ex.viewer.scope));
        assert!(!slot.lock().unwrap().is_active(), "playback waits for the play button");

        slot.lock().unwrap().start_with(&a, PlayerOptions { ctx: Some(ctx.clone()), idle_timeout: Duration::from_millis(500), ..Default::default() });
        let shown = wait_frames(&ctx, &mut ex, Duration::from_secs(15), || slot.lock().unwrap().frame_size().is_some());
        assert!(shown, "the decoded frame reaches the pane");
        assert_eq!(slot.lock().unwrap().frame_size(), Some([320, 180]));
        let first_done = slot.lock().unwrap().watcher().unwrap();
        assert!(!first_done());

        select(&mut ex, &b, "mp4");
        frame(&ctx, &mut ex);
        assert!(!slot.lock().unwrap().is_active() && slot.lock().unwrap().frame_size().is_none(), "another video resets the player");
        assert!(wait_frames(&ctx, &mut ex, Duration::from_secs(3), first_done), "the old decode thread ends");

        slot.lock().unwrap().start_with(&b, PlayerOptions { ctx: Some(ctx.clone()), idle_timeout: Duration::from_millis(500), ..Default::default() });
        assert!(wait_frames(&ctx, &mut ex, Duration::from_secs(15), || slot.lock().unwrap().frame_size().is_some()));
        let second_done = slot.lock().unwrap().watcher().unwrap();
        select(&mut ex, &png, "png");
        frame(&ctx, &mut ex);
        assert!(!slot.lock().unwrap().is_active(), "a non-video selection stops playback");
        assert!(wait_frames(&ctx, &mut ex, Duration::from_secs(3), second_done));

        select(&mut ex, &a, "mp4");
        frame(&ctx, &mut ex);
        slot.lock().unwrap().start_with(&a, PlayerOptions { ctx: Some(ctx.clone()), idle_timeout: Duration::from_millis(500), ..Default::default() });
        assert!(wait_frames(&ctx, &mut ex, Duration::from_secs(15), || slot.lock().unwrap().frame_size().is_some()));
        let third_done = slot.lock().unwrap().watcher().unwrap();
        ex.open_preview_pane = false;
        let end = Instant::now() + Duration::from_secs(4);
        while !third_done() && Instant::now() < end {
            frame(&ctx, &mut ex);
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(third_done(), "a closed pane stops the decode thread");
    }
}