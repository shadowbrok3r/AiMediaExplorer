use std::path::PathBuf;

use eframe::egui::Context;
use crate::ui::file_table::AIUpdate;

use crate::ui::file_table::rows::key::{KeyPolicy, paths_equivalent};

fn find_row_loose<'a>(rows: &'a [crate::database::Thumbnail], path: &str) -> Option<&'a crate::database::Thumbnail> {
    // 1) Exact match (fast path)
    if let Some(r) = rows.iter().find(|r| r.path == path) { return Some(r); }
    // 2) Platform-equivalent path match
    if let Some(r) = rows.iter().find(|r| paths_equivalent(&r.path, path, KeyPolicy::NATIVE)) { return Some(r); }
    // 3) Fallback by filename + size (helps with duplicates across mounts when streaming source path differs)
    let fname = std::path::Path::new(path).file_name().and_then(|n| n.to_str()).unwrap_or("");
    rows.iter().find(|r| r.filename.eq_ignore_ascii_case(fname) && r.size > 0)
}

impl crate::ui::file_table::FileExplorer {
    // Table row for `path`: exact index hit first, then the loose scan without cloning the table.
    fn follow_row(&self, path: &str) -> Option<crate::database::Thumbnail> {
        if let Some(row) = self.table.get_by_path(path) {
            return Some(row.clone());
        }
        find_row_loose(self.table.rows(), path).cloned()
    }

    pub fn receive_ai_update(&mut self, ctx: &Context) {
        // Process AI streaming updates
        while let Ok(update) = self.ai_update_rx.try_recv() {
            ctx.request_repaint();
            match update {
                AIUpdate::Interim { path, text } => {
                    // Auto-follow: always advance to the currently streaming image if enabled.
                    if self.follow_active_vision && !path.is_empty() {
                        let need_follow = !paths_equivalent(&self.current_thumb.path, &path, KeyPolicy::NATIVE);
                        if need_follow {
                            if let Some(row) = self.follow_row(&path) {
                                self.current_thumb = row;
                                self.open_preview_pane = true; // ensure visible
                            }
                        }
                    }
                    self.streaming_interim.insert(path, text);
                }
                AIUpdate::Final {
                    path,
                    description,
                    caption,
                    category,
                    tags,
                } => {
                    if self.follow_active_vision && !path.is_empty() {
                        let need_follow = !paths_equivalent(&self.current_thumb.path, &path, KeyPolicy::NATIVE);
                        if need_follow {
                            if let Some(row) = self.follow_row(&path) {
                                self.current_thumb = row;
                                self.open_preview_pane = true;
                            }
                        }
                    }
                    self.streaming_interim.remove(&path);
                    // Update counters: a pending item finished.
                    if self.vision_pending > 0 {
                        self.vision_pending -= 1;
                    }
                    self.vision_completed += 1;
                    let desc_clone_for_row = description.clone();
                    let caption_clone_for_row = caption.clone();
                    let category_clone_for_row = category.clone();
                    let tags_clone_for_row = tags.clone();
                    let final_pos = self.table.position(&path);
                    if let Some(mut row) = final_pos.and_then(|idx| self.table.get_mut(idx)) {
                        if !desc_clone_for_row.trim().is_empty() {
                            row.description = Some(desc_clone_for_row.clone());
                        }
                        if let Some(c) = caption_clone_for_row.clone() {
                            if !c.trim().is_empty() {
                                row.caption = Some(c);
                            }
                        }
                        if let Some(cat) = category_clone_for_row.clone() {
                            if !cat.trim().is_empty() {
                                row.category = Some(cat);
                            }
                        }
                        if !tags_clone_for_row.is_empty() {
                            row.tags = tags_clone_for_row.clone();
                        }
                    }
                    if self.current_thumb.path == path {
                        if !description.trim().is_empty() {
                            self.current_thumb.description = Some(description.clone());
                        }
                        if let Some(c) = caption.clone() {
                            if !c.trim().is_empty() {
                                self.current_thumb.caption = Some(c);
                            }
                        }
                        if let Some(cat) = category.clone() {
                            if !cat.trim().is_empty() {
                                self.current_thumb.category = Some(cat);
                            }
                        }
                        if !tags.is_empty() {
                            self.current_thumb.tags = tags.clone();
                        }
                    }
                    // Defensive persistence: ensure final AI metadata is saved to DB (idempotent if already saved by engine)
                    {
                        let persist_path = path.clone();
                        let description_clone = description.clone();
                        let caption_clone = caption.clone();
                        let category_clone = category.clone();
                        let tags_clone = tags.clone();
                        // Prefer current_thumb if active, else table row, else the cloned incoming values.
                        let (desc_final, cap_final, cat_final, tags_final) =
                            if self.current_thumb.path == persist_path {
                                (
                                    self.current_thumb.description.clone().unwrap_or_default(),
                                    self.current_thumb.caption.clone().unwrap_or_default(),
                                    self.current_thumb
                                        .category
                                        .clone()
                                        .unwrap_or_else(|| "general".into()),
                                    self.current_thumb.tags.clone(),
                                )
                            } else if let Some(row) =
                                final_pos.and_then(|idx| self.table.get(idx))
                            {
                                (
                                    row.description.clone().unwrap_or_default(),
                                    row.caption.clone().unwrap_or_default(),
                                    row.category.clone().unwrap_or_else(|| "general".into()),
                                    row.tags.clone(),
                                )
                            } else {
                                (
                                    description_clone,
                                    caption_clone.unwrap_or_default(),
                                    category_clone.unwrap_or_else(|| "general".into()),
                                    tags_clone,
                                )
                            };
                        tokio::spawn(async move {
                            let vd = crate::ai::VisionDescription {
                                description: desc_final,
                                caption: cap_final,
                                category: cat_final,
                                tags: tags_final,
                            };
                            if let Err(e) = crate::ai::GLOBAL_AI_ENGINE
                                .apply_vision_description(&persist_path, &vd)
                                .await
                            {
                                log::warn!(
                                    "[AI] UI final persist failed for {}: {}",
                                    persist_path,
                                    e
                                );
                            }
                        });
                    }
                }
                AIUpdate::SimilarResults { origin_path, results } => {
                    if results.is_empty() { continue; }
                    let same_origin = self.viewer.showing_similarity && (self.similarity_origin_path.as_deref() == Some(origin_path.as_str()) || self.current_path == origin_path);
                    if same_origin {
                        for r in results.into_iter() {
                            if self.table.insert_if_absent(r.thumb.clone()).is_none() { continue; }
                            if let Some(s) = r.clip_similarity_score.or(r.similarity_score) { self.viewer.similar_scores.insert(r.thumb.path.clone(), s); }
                        }
                        self.similarity_origin_path.get_or_insert(origin_path);
                    } else {
                        let mut rows: Vec<crate::database::Thumbnail> = Vec::with_capacity(results.len());
                        let mut scores: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
                        for r in results.into_iter() {
                            if let Some(s) = r.clip_similarity_score.or(r.similarity_score) { scores.insert(r.thumb.path.clone(), s); }
                            rows.push(r.thumb);
                        }
                        let title = format!("Similar to {}", PathBuf::from(&origin_path).file_name().unwrap_or_default().to_str().unwrap_or(&origin_path));
                        crate::app::OPEN_TAB_REQUESTS.lock().unwrap().push(crate::ui::file_table::FilterRequest::NewTab {
                            title,
                            rows,
                            showing_similarity: true,
                            similar_scores: Some(scores),
                            origin_path: Some(origin_path.clone()),
                            background: false,
                        });
                    }
                }
            }
        }
    }
}