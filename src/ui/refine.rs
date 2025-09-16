use eframe::egui::*;
use crate::ui::status::GlobalStatusIndicator;
use crossbeam::channel::Sender;

pub struct RefinementsPanel {
    pub generating: bool,
    pub proposals: Vec<RefinementProposal>,
    pub filter_text: String,
    pub reranker_choice: crate::ai::refine::RerankerType,
    pub updates_tx: Sender<Vec<RefinementProposal>>,
    pub toast_tx: Sender<(egui_toast::ToastKind, String)>,
    
}

impl RefinementsPanel {
    pub fn new(updates_tx: Sender<Vec<RefinementProposal>>, toast_tx: Sender<(egui_toast::ToastKind, String)>) -> Self {
        Self { 
            generating: false, 
            proposals: Vec::new(), 
            filter_text: String::new(), 
            reranker_choice: Default::default(), 
            updates_tx, 
            toast_tx
        }
    }
}

#[derive(Clone, Debug)]
pub struct RefinementProposal {
    pub path: String,
    pub current_category: Option<String>,
    pub current_tags: Vec<String>,
    pub current_caption: Option<String>,
    pub current_description: Option<String>,
    pub new_category: Option<String>,
    pub new_tags: Vec<String>,
    pub new_caption: Option<String>,
    pub new_description: Option<String>,
    pub selected: bool,
    pub generator: String,
}

impl RefinementsPanel {
    pub fn ui(&mut self, ui: &mut Ui) {
        ui.heading("AI Refinements (DB-only)");
        ui.label("Generate proposed improvements for categories/tags/captions/descriptions from the database.");
        ui.separator();
        ui.horizontal(|ui| {
            egui::ComboBox::new("reranker-choice", "Reranker")
            .selected_text(match self.reranker_choice { 
                crate::ai::refine::RerankerType::Heuristic => "Heuristic",
                crate::ai::refine::RerankerType::JinaM0 => "Jina M0",
                crate::ai::refine::RerankerType::QwenReranker => "Qwen (placeholder)",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.reranker_choice, crate::ai::refine::RerankerType::Heuristic, "Heuristic");
                ui.selectable_value(&mut self.reranker_choice, crate::ai::refine::RerankerType::JinaM0, "Jina M0");
                ui.selectable_value(&mut self.reranker_choice, crate::ai::refine::RerankerType::QwenReranker, "Qwen (placeholder)");
            });
            
            if ui.add_enabled(!self.generating, Button::new("Generate Proposals")).clicked() {
                self.generating = true;
                self.proposals.clear();
                let sel = self.reranker_choice; // capture
                let tx = self.updates_tx.clone();
                crate::ui::status::RERANK_STATUS.set_state(crate::ui::status::StatusState::Running, "Generating proposals");
                tokio::spawn(async move {
                    // Free memory from other heavy models to ensure reranker can load comfortably
                    crate::ai::unload_heavy_models_except("RERANK").await;
                    let generator = crate::ai::refine::ProposalGenerator::new(&crate::ai::GLOBAL_AI_ENGINE);
                    // Build reranker based on selection
                    // Choose reranker impl
                    enum Rk { Heu, Jina }
                    let mode = match sel { 
                        crate::ai::refine::RerankerType::Heuristic => Rk::Heu,
                        crate::ai::refine::RerankerType::JinaM0 => Rk::Jina,
                        crate::ai::refine::RerankerType::QwenReranker => Rk::Heu,
                    };

                    // Prepare batch buffer for smoother UI updates
                    let mut batch: Vec<RefinementProposal> = Vec::with_capacity(16);
                    let flush = |tx: &Sender<Vec<RefinementProposal>>, buf: &mut Vec<RefinementProposal>| {
                        if buf.is_empty() { return; }
                        let to_send = std::mem::take(buf);
                        if let Err(e) = tx.send(to_send) { log::error!("[Refine/UI] send batch failed: {e}"); }
                    };

                    match mode {
                        Rk::Heu => {
                            let mut rer = crate::ai::refine::HeuristicReranker;
                            generator.stream_proposals(200, &mut rer, |p| {
                                if p.new_category.is_none() && p.new_tags.is_empty() && p.new_caption.is_none() && p.new_description.is_none() { return; }
                                batch.push(RefinementProposal {
                                    path: p.path,
                                    current_category: p.current_category,
                                    current_tags: p.current_tags,
                                    current_caption: p.current_caption,
                                    current_description: p.current_description,
                                    new_category: p.new_category,
                                    new_tags: p.new_tags,
                                    new_caption: p.new_caption,
                                    new_description: p.new_description,
                                    selected: true,
                                    generator: p.generator,
                                });
                                if batch.len() >= 12 { flush(&tx, &mut batch); }
                            }).await;
                            flush(&tx, &mut batch);
                        }
                        Rk::Jina => {
                            // Ensure model engine exists/cached
                            let _ = crate::ai::reranker::ensure_reranker_from_settings().await;
                            struct Adapter;
                            impl crate::ai::refine::Reranker for Adapter {
                                fn name(&self) -> &'static str { "Jina M0" }
                                fn rerank_tags(&mut self, query: &str, candidates: &[String]) -> Vec<String> {
                                    if candidates.is_empty() { return Vec::new(); }
                                    let mut set = std::collections::BTreeSet::new();
                                    for c in candidates { let n = c.trim().to_ascii_lowercase(); if !n.is_empty() { set.insert(n); } }
                                    let cands: Vec<String> = set.into_iter().collect();
                                    let pairs: Vec<(String,String)> = cands.iter().map(|d| (query.to_string(), d.clone())).collect();
                                    let scores = crate::ai::reranker::jina_score_text_pairs_blocking(&pairs);
                                    if scores.len() != pairs.len() || scores.is_empty() {
                                        log::warn!("[Refine] Jina returned {} scores for {} pairs; falling back to heuristic", scores.len(), pairs.len());
                                        // Heuristic fallback
                                        let mut map: std::collections::BTreeMap<String, i64> = std::collections::BTreeMap::new();
                                        for (_, cand) in pairs.iter() { *map.entry(cand.clone()).or_default() += 1; }
                                        return map.into_iter().map(|(s, _)| s).take(12).collect();
                                    }
                                    let mut zipped: Vec<(String, f32)> = cands.into_iter().zip(scores.into_iter()).collect();
                                    zipped.sort_by(|a,b| b.1.total_cmp(&a.1));
                                    zipped.into_iter().map(|(s,_)| s).take(12).collect()
                                }
                                fn rerank_category(&mut self, query: &str, candidates: &[String]) -> Option<String> {
                                    if candidates.is_empty() { return None; }
                                    let mut set = std::collections::BTreeSet::new();
                                    for c in candidates { let n = c.trim().to_ascii_lowercase(); if !n.is_empty() { set.insert(n); } }
                                    let cands: Vec<String> = set.into_iter().collect();
                                    let pairs: Vec<(String,String)> = cands.iter().map(|d| (query.to_string(), d.clone())).collect();
                                    let scores = crate::ai::reranker::jina_score_text_pairs_blocking(&pairs);
                                    if scores.len() != pairs.len() || scores.is_empty() { return cands.into_iter().next(); }
                                    cands.into_iter().zip(scores.into_iter()).max_by(|a,b| a.1.total_cmp(&b.1)).map(|(s,_)| s)
                                }
                                fn rerank_caption(&mut self, query: &str, candidates: &[String]) -> Option<String> {
                                    if candidates.is_empty() { return None; }
                                    let cands: Vec<String> = candidates.iter().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
                                    if cands.is_empty() { return None; }
                                    let pairs: Vec<(String,String)> = cands.iter().map(|d| (query.to_string(), d.clone())).collect();
                                    let scores = crate::ai::reranker::jina_score_text_pairs_blocking(&pairs);
                                    if scores.len() != pairs.len() || scores.is_empty() { return cands.into_iter().next(); }
                                    cands.into_iter().zip(scores.into_iter()).max_by(|a,b| a.1.total_cmp(&b.1)).map(|(s,_)| s)
                                }
                                fn rerank_description(&mut self, query: &str, candidates: &[String]) -> Option<String> { self.rerank_caption(query, candidates) }
                            }
                            let mut rer = Adapter;
                            generator.stream_proposals(200, &mut rer, |p| {
                                if p.new_category.is_none() && p.new_tags.is_empty() && p.new_caption.is_none() && p.new_description.is_none() { return; }
                                batch.push(RefinementProposal {
                                    path: p.path,
                                    current_category: p.current_category,
                                    current_tags: p.current_tags,
                                    current_caption: p.current_caption,
                                    current_description: p.current_description,
                                    new_category: p.new_category,
                                    new_tags: p.new_tags,
                                    new_caption: p.new_caption,
                                    new_description: p.new_description,
                                    selected: true,
                                    generator: p.generator,
                                });
                                if batch.len() >= 12 { flush(&tx, &mut batch); }
                            }).await;
                            flush(&tx, &mut batch);
                        }
                    }
                    crate::ui::status::RERANK_STATUS.set_state(crate::ui::status::StatusState::Idle, "Done");
                });
            }
            if ui.button("Clear").clicked() {
                self.proposals.clear();
                self.generating = false;
            }
            ui.add_sized([240.0, 20.0], TextEdit::singleline(&mut self.filter_text).hint_text("Filter by tag/category/path"));
        
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui.button("Accept All").clicked() {
                    let to_apply = self.proposals.clone();
                    let toast_tx = self.toast_tx.clone();
                    tokio::spawn(async move {
                        let mut ok = 0usize; let total = to_apply.len();
                        for p in to_apply {
                            if let Some(thumb) = crate::Thumbnail::get_thumbnail_by_path(&p.path).await.ok().flatten() {
                                let meta = super::super::Thumbnail { id: thumb.id.clone(), db_created: thumb.db_created.clone(), path: thumb.path.clone(), filename: thumb.filename.clone(), file_type: thumb.file_type.clone(), size: thumb.size, description: p.new_description.clone(), caption: p.new_caption.clone(), tags: p.new_tags.clone(), category: p.new_category.clone(), thumbnail_b64: thumb.thumbnail_b64.clone(), modified: thumb.modified.clone(), hash: thumb.hash.clone(), parent_dir: thumb.parent_dir.clone(), logical_group: thumb.logical_group.clone(), };
                                if thumb.update_or_create_thumbnail(&meta, meta.thumbnail_b64.clone()).await.is_ok() { ok += 1; }
                            }
                        }
                        let _ = toast_tx.send((egui_toast::ToastKind::Success, format!("Accepted {ok}/{total} proposals")));
                    });
                }
                if ui.button("Deselect All").clicked() { for p in &mut self.proposals { p.selected = false; } }
                if ui.button("Select All").clicked() { for p in &mut self.proposals { p.selected = true; } }
                if ui.button("Accept Selected").clicked() {
                    let to_apply: Vec<_> = self.proposals.iter().filter(|p| p.selected).cloned().collect();
                    let toast_tx = self.toast_tx.clone();
                    tokio::spawn(async move {
                        let mut ok = 0usize; let total = to_apply.len();
                        for p in to_apply {
                            if let Some(thumb) = crate::Thumbnail::get_thumbnail_by_path(&p.path).await.ok().flatten() {
                                let meta = super::super::Thumbnail { id: thumb.id.clone(), db_created: thumb.db_created.clone(), path: thumb.path.clone(), filename: thumb.filename.clone(), file_type: thumb.file_type.clone(), size: thumb.size, description: p.new_description.clone(), caption: p.new_caption.clone(), tags: p.new_tags.clone(), category: p.new_category.clone(), thumbnail_b64: thumb.thumbnail_b64.clone(), modified: thumb.modified.clone(), hash: thumb.hash.clone(), parent_dir: thumb.parent_dir.clone(), logical_group: thumb.logical_group.clone(), };
                                if thumb.update_or_create_thumbnail(&meta, meta.thumbnail_b64.clone()).await.is_ok() { ok += 1; }
                            }
                        }
                        let _ = toast_tx.send((egui_toast::ToastKind::Success, format!("Accepted {ok}/{total} proposals")));
                    });
                }
            });
        });

        ui.separator();

        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            if self.proposals.is_empty() {
                let label = if self.generating { "Generating proposals…" } else { "No proposals yet." };
                ui.weak(label);
                return;
            }
                for (idx, p) in self.proposals.iter_mut().enumerate() {
                if !self.filter_text.is_empty() {
                    let f = self.filter_text.to_ascii_lowercase();
                    let hay = format!("{} {:?} {:?}", p.path, p.current_category, p.current_tags).to_ascii_lowercase();
                    if !hay.contains(&f) { continue; }
                }
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                            ui.checkbox(&mut p.selected, "");
                        ui.label(RichText::new(format!("#{}", idx+1)).strong().underline());
                        ui.monospace(&p.path);
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            ui.weak(format!("by {}", p.generator));
                        });
                    });
                    ui.separator();

                    ui.columns(2, |ui| {
                        ui[0].vertical_centered(|ui| {
                            ui.heading("Current");
                            ui.separator();
                        });
                        ui[1].vertical_centered(|ui| {
                            ui.heading("Proposed");
                            ui.separator();
                        });
                    });

                    ui.vertical_centered(|ui| ui.heading("Category"));
                    if p.new_category.is_some() {
                        ui.columns(2, |ui| {
                            ui[0].label(format!("{:?}", p.current_category));
                            ui[1].label(format!("{:?}", p.new_category));
                        });
                        ui.separator();
                    }
                    if !p.new_tags.is_empty() {
                        ui.columns(2, |ui| {
                            ui[0].label(format!("{:#?}", p.current_tags));
                            ui[1].label(format!("{:#?}", p.new_tags));
                        });
                    }

                    ui.vertical_centered(|ui| ui.heading("Caption"));
                    if p.current_caption.is_some() || p.new_caption.is_some() {
                        ui.separator();
                        ui.columns(2, |ui| {
                            ui[0].label(format!("{:?}", p.current_caption));
                            ui[1].label(format!("{:?}", p.new_caption));
                        });
                    }

                    ui.vertical_centered(|ui| ui.heading("Description"));
                    if p.current_description.is_some() || p.new_description.is_some() {
                        ui.separator();
                        ui.columns(2, |ui| {
                            ui[0].label(format!("{:?}", p.current_description));
                            ui[1].label(format!("{:?}", p.new_description));
                        });
                    }

                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button("Accept").clicked() {
                            let path = p.path.clone();
                            let cat = p.new_category.clone();
                            let tags = p.new_tags.clone();
                            let cap = p.new_caption.clone();
                            let desc = p.new_description.clone();
                            tokio::spawn(async move {
                                if let Some(thumb) = crate::Thumbnail::get_thumbnail_by_path(&path).await.ok().flatten() {
                                    let meta = super::super::Thumbnail {
                                        id: thumb.id.clone(),
                                        db_created: thumb.db_created.clone(),
                                        path: thumb.path.clone(),
                                        filename: thumb.filename.clone(),
                                        file_type: thumb.file_type.clone(),
                                        size: thumb.size,
                                        description: desc.clone(),
                                        caption: cap.clone(),
                                        tags: tags.clone(),
                                        category: cat.clone(),
                                        thumbnail_b64: thumb.thumbnail_b64.clone(),
                                        modified: thumb.modified.clone(),
                                        hash: thumb.hash.clone(),
                                        parent_dir: thumb.parent_dir.clone(),
                                        logical_group: thumb.logical_group.clone(),
                                    };
                                    let _ = thumb.update_or_create_thumbnail(&meta, meta.thumbnail_b64.clone()).await;
                                }
                            });
                        }
                        ui.add_enabled(false, Button::new("Reject"));
                    });
                });
                ui.separator();
            }
        });
    }
}
