use eframe::egui::*;
use crate::icons::{self, svg};
use crate::ui::status::GlobalStatusIndicator;
use crossbeam::channel::Sender;
use once_cell::sync::Lazy;
use std::sync::Mutex;

pub struct RefinementsPanel {
    pub generating: bool,
    pub proposals: Vec<RefinementProposal>,
    pub filter_text: String,
    pub reranker_choice: crate::ai::refine::RerankerType,
    pub updates_tx: Sender<Vec<RefinementProposal>>,
    pub toast_tx: Sender<(egui_toast::ToastKind, String)>,
    // Cloud (OpenAI-compatible) refinement options
    pub use_cloud_model: bool,
    pub cloud_model_name: String,
    pub cloud_provider: String,
    // Optional restriction: only refine these paths when present
    limit_paths: Option<std::collections::HashSet<String>>,
    
}

impl RefinementsPanel {
    pub fn new(updates_tx: Sender<Vec<RefinementProposal>>, toast_tx: Sender<(egui_toast::ToastKind, String)>) -> Self {
        Self { 
            generating: false, 
            proposals: Vec::new(), 
            filter_text: String::new(), 
            reranker_choice: Default::default(), 
            updates_tx, 
            toast_tx,
            use_cloud_model: false,
            cloud_model_name: String::new(),
            cloud_provider: "openrouter".into(),
            limit_paths: None,
        }
    }
}

// Paths the command dispatcher hands to the refinements panel until it is shown
pub static REFINE_SELECTION_REQUESTS: Lazy<Mutex<Vec<Vec<String>>>> = Lazy::new(|| Mutex::new(Vec::new()));

pub fn request_refine_for_paths(paths: Vec<String>) {
    if paths.is_empty() { return; }
    crate::command::send(crate::command::AppCommand::Refine { paths });
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

/// True when `filter` is empty or occurs in the proposal's path, category or tags.
pub fn matches_filter(p: &RefinementProposal, filter: &str) -> bool {
    if filter.is_empty() {
        return true;
    }
    let hay = format!("{} {:?} {:?}", p.path, p.current_category, p.current_tags).to_ascii_lowercase();
    hay.contains(&filter.to_ascii_lowercase())
}

/// Indices of flagged proposals; with `visible_only` only those the filter shows.
pub fn accept_indices(proposals: &[RefinementProposal], filter: &str, visible_only: bool) -> Vec<usize> {
    proposals
        .iter()
        .enumerate()
        .filter(|(_, p)| p.selected && (!visible_only || matches_filter(p, filter)))
        .map(|(i, _)| i)
        .collect()
}

fn text_or_none(value: &Option<String>) -> String {
    value.as_deref().filter(|v| !v.is_empty()).map(str::to_string).unwrap_or_else(|| "(none)".into())
}

fn tags_text(tags: &[String]) -> String {
    if tags.is_empty() { "(none)".into() } else { tags.join(", ") }
}

/// Writes the proposals to the database and toasts how many were applied.
fn accept_proposals(batch: Vec<RefinementProposal>, toast_tx: Sender<(egui_toast::ToastKind, String)>) {
    if batch.is_empty() {
        return;
    }
    tokio::spawn(async move {
        let total = batch.len();
        let mut ok = 0usize;
        for p in batch {
            if let Some(thumb) = crate::Thumbnail::get_thumbnail_by_path(&p.path).await.ok().flatten() {
                let meta = super::super::Thumbnail { id: thumb.id.clone(), db_created: thumb.db_created.clone(), path: thumb.path.clone(), filename: thumb.filename.clone(), file_type: thumb.file_type.clone(), size: thumb.size, description: p.new_description.clone(), caption: p.new_caption.clone(), tags: p.new_tags.clone(), category: p.new_category.clone(), thumbnail_b64: thumb.thumbnail_b64.clone(), modified: thumb.modified.clone(), hash: thumb.hash.clone(), parent_dir: thumb.parent_dir.clone(), logical_group: thumb.logical_group.clone(), };
                if thumb.update_or_create_thumbnail(&meta, meta.thumbnail_b64.clone()).await.is_ok() { ok += 1; }
            }
        }
        let _ = toast_tx.send((egui_toast::ToastKind::Success, format!("Accepted {ok}/{total} proposals")));
    });
}

impl RefinementsPanel {
    /// Applies the proposals at `indices` and removes them from the list.
    fn accept_at(&mut self, indices: &[usize]) {
        let batch: Vec<RefinementProposal> = indices.iter().filter_map(|i| self.proposals.get(*i).cloned()).collect();
        let accepted: std::collections::HashSet<usize> = indices.iter().copied().collect();
        accept_proposals(batch, self.toast_tx.clone());
        self.remove_at(&accepted);
    }

    fn remove_at(&mut self, indices: &std::collections::HashSet<usize>) {
        let mut index = 0;
        self.proposals.retain(|_| {
            let keep = !indices.contains(&index);
            index += 1;
            keep
        });
    }

    pub fn ui(&mut self, ui: &mut Ui) {
        // Drain queued selection scope requests
        {
            let mut q = REFINE_SELECTION_REQUESTS.lock().unwrap();
            if !q.is_empty() {
                let mut set: std::collections::HashSet<String> = self.limit_paths.take().unwrap_or_default();
                for batch in q.drain(..) { for p in batch { set.insert(p); } }
                if set.is_empty() { self.limit_paths = None; } else { self.limit_paths = Some(set); }
            }
        }
        icons::heading(ui, svg::AME_REFINE, "AI Refinements");
        ui.label("Generate proposed improvements for categories/tags/captions/descriptions. If a selection scope is active, only those files are processed.");
        if let Some(set) = self.limit_paths.as_ref() {
            let scope_len = set.len();
            let mut clear = false;
            ui.horizontal(|ui| {
                ui.colored_label(crate::theme::warn(ui), format!("Selection scope: {} items", scope_len));
                if icons::button(ui, svg::X_CIRCLE, "Clear Scope").clicked() { clear = true; }
            });
            if clear { self.limit_paths = None; }
        }
        ui.separator();
        CollapsingHeader::new("Cloud Model Options").show(ui, |ui| {
            ui.checkbox(&mut self.use_cloud_model, "Use cloud model (OpenAI-compatible)");
            ui.horizontal(|ui| {
                ui.label("Provider");
                egui::ComboBox::new("cloud-refine-provider", "")
                    .selected_text(&self.cloud_provider)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.cloud_provider, "openai".into(), "openai");
                        ui.selectable_value(&mut self.cloud_provider, "openrouter".into(), "openrouter");
                        ui.selectable_value(&mut self.cloud_provider, "groq".into(), "groq");
                        ui.selectable_value(&mut self.cloud_provider, "gemini".into(), "gemini");
                        ui.selectable_value(&mut self.cloud_provider, "custom".into(), "custom");
                    });
            });
            ui.horizontal(|ui| {
                ui.label("Model");
                ui.text_edit_singleline(&mut self.cloud_model_name);
            });
            ui.label(RichText::new("Will fall back to app default model if blank").weak());
        });
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
            
            if ui.add_enabled(!self.generating, icons::button_widget(ui, svg::AME_PROPOSAL, "Generate Proposals")).clicked() {
                self.generating = true;
                self.proposals.clear();
                let sel = self.reranker_choice; // capture
                let tx = self.updates_tx.clone();
                let use_cloud = self.use_cloud_model;
                let cloud_model = self.cloud_model_name.clone();
                let provider = self.cloud_provider.clone();
                let limit_paths = self.limit_paths.clone();
                crate::ui::status::RERANK_STATUS.set_state(crate::ui::status::StatusState::Running, "Generating proposals");
                tokio::spawn(async move {
                    // Free memory from other heavy models to ensure reranker can load comfortably
                    crate::ai::unload_heavy_models_except("RERANK").await;
                    let generator = crate::ai::refine::ProposalGenerator::new(&crate::ai::GLOBAL_AI_ENGINE);
                    if use_cloud {
                        log::info!("[Refine/Cloud] Enter cloud refinement path (provider={provider}, selection_scope={} items)", limit_paths.as_ref().map(|s| s.len()).unwrap_or(0));
                        // Collect candidate rows (small cap)
                        let mut batch: Vec<RefinementProposal> = Vec::new();
                        // If we have a small explicit selection (<= 16), bypass heuristic streaming & embedding KNN to avoid log spam
                        if let Some(sel) = limit_paths.as_ref() {
                            if sel.len() > 0 && sel.len() <= 16 {
                                log::info!("[Refine/Cloud] Small selection ({}); bypassing heuristic stream", sel.len());
                                for path in sel.iter() {
                                    if let Ok(opt_thumb) = crate::Thumbnail::get_thumbnail_by_path(path).await { if let Some(t) = opt_thumb {
                                        if t.caption.is_none() && t.description.is_none() && t.category.is_none() && t.tags.is_empty() { continue; }
                                        batch.push(RefinementProposal { path: t.path.clone(), current_category: t.category.clone(), current_tags: t.tags.clone(), current_caption: t.caption.clone(), current_description: t.description.clone(), new_category: None, new_tags: Vec::new(), new_caption: None, new_description: None, selected: true, generator: "cloud-direct".into() });
                                    }}
                                }
                            }
                        }
                        if batch.is_empty() {
                        // For simplicity reuse heuristic generation to gather raw data, then send each to cloud model for enhancement
                        let mut rer = crate::ai::refine::HeuristicReranker;
                        generator.stream_proposals(120, &mut rer, |p| {
                            if let Some(limit) = limit_paths.as_ref() { if !limit.contains(&p.path) { return; } }
                            batch.push(RefinementProposal {
                                path: p.path.clone(),
                                current_category: p.current_category.clone(),
                                current_tags: p.current_tags.clone(),
                                current_caption: p.current_caption.clone(),
                                current_description: p.current_description.clone(),
                                new_category: None,
                                new_tags: Vec::new(),
                                new_caption: None,
                                new_description: None,
                                selected: true,
                                generator: "cloud-seed".into(),
                            });
                        }).await; }
                        // Fallback: if heuristic stream produced no seeds (e.g. no local changes), still seed using raw selected rows
                        if batch.is_empty() {
                            log::info!("[Refine/Cloud] No heuristic seeds; falling back to direct row enumeration");
                            if let Some(limit) = limit_paths.as_ref() {
                                // Enumerate each selected path; load thumbnail metadata
                                for path in limit.iter().take(240) { // hard cap safety
                                    if let Ok(opt_thumb) = crate::Thumbnail::get_thumbnail_by_path(path).await { if let Some(t) = opt_thumb {
                                        // Require at least one piece of base metadata so prompt has context
                                        if t.caption.is_some() || t.description.is_some() || t.category.is_some() || !t.tags.is_empty() {
                                            batch.push(RefinementProposal {
                                                path: t.path.clone(),
                                                current_category: t.category.clone(),
                                                current_tags: t.tags.clone(),
                                                current_caption: t.caption.clone(),
                                                current_description: t.description.clone(),
                                                new_category: None,
                                                new_tags: Vec::new(),
                                                new_caption: None,
                                                new_description: None,
                                                selected: true,
                                                generator: "cloud-seed-fallback".into(),
                                            });
                                        }
                                    }}
                                }
                            } else {
                                // No explicit selection; enumerate rich rows directly (already filtered by DB query)
                                let rows = generator.engine.list_rich_thumbnail_rows(120).await;
                                for r in rows.into_iter() { batch.push(RefinementProposal {
                                    path: r.path.clone(),
                                    current_category: r.category.clone(),
                                    current_tags: r.tags.clone(),
                                    current_caption: r.caption.clone(),
                                    current_description: r.description.clone(),
                                    new_category: None,
                                    new_tags: Vec::new(),
                                    new_caption: None,
                                    new_description: None,
                                    selected: true,
                                    generator: "cloud-seed-fallback".into(),
                                }); }
                            }
                            log::info!("[Refine/Cloud] Fallback produced {} seed rows", batch.len());
                        }
                        // Prepare provider config (reuse assistant logic-lite)
                        let settings = crate::settings::snapshot();
                        let model_to_use = if cloud_model.trim().is_empty() {
                            settings.openai_default_model.clone().unwrap_or_else(|| "gpt-5-mini".into())
                        } else { cloud_model.clone() };
                        let env_or = |opt: &Option<String>, key: &str| opt.clone().or_else(|| std::env::var(key).ok());
                        let (api_key, base_url, org) = match provider.as_str() {
                            "openai" => (env_or(&settings.openai_api_key, "OPENAI_API_KEY"), settings.openai_base_url.clone(), settings.openai_organization.clone()),
                            "openrouter" => (env_or(&settings.openrouter_api_key, "OPENROUTER_API_KEY"), Some("https://openrouter.ai/api/v1".into()), None),
                            "groq" => (env_or(&settings.groq_api_key, "GROQ_API_KEY"), settings.openai_base_url.clone(), None),
                            "gemini" => (env_or(&settings.gemini_api_key, "GEMINI_API_KEY"), settings.openai_base_url.clone(), None),
                            _ => (env_or(&settings.openai_api_key, "OPENAI_API_KEY"), settings.openai_base_url.clone(), settings.openai_organization.clone()),
                        };
                        let cfg = crate::ai::openai_compat::ProviderConfig { provider: provider.clone(), api_key, base_url, model: model_to_use.clone(), organization: org, temperature: Some(0.2), zdr: true };
                        crate::ui::status::RERANK_STATUS.set_model(&format!("{}:{}", provider, model_to_use));
                        log::info!("[Refine/Cloud] Enriching {} seed rows with {}:{}", batch.len(), provider, model_to_use);
                        // For each proposal seed, ask model for improvements (batch sequentially for now)
                        let mut enriched: Vec<RefinementProposal> = Vec::new();
                        for seed in batch.into_iter().take(64) { // cap to avoid huge cost
                            let prompt = format!(
                                "You are an assistant that refines existing media metadata.\nReturn ONLY a compact JSON object with exactly these keys: new_category (string or null), new_tags (array of strings), new_caption (string or null), new_description (string or null).\nIf you cannot confidently improve a field, set it to null (or empty array for tags). Do not include explanatory text.\nCurrent data:\nPath: {path}\nCategory: {cat}\nTags: {tags}\nCaption: {cap}\nDescription: {desc}\nJSON:",
                                path = seed.path,
                                cat = seed.current_category.clone().unwrap_or_default(),
                                tags = seed.current_tags.join(", "),
                                cap = seed.current_caption.clone().unwrap_or_default(),
                                desc = seed.current_description.clone().unwrap_or_default()
                            );
                            let reply_res = crate::ai::openai_compat::simple_text_completion(cfg.clone(), &prompt).await;
                            match reply_res {
                                Ok(text) => {
                                    log::debug!("[Refine/Cloud] Raw model reply for {}: {}", seed.path, text);
                                    // Robust JSON extraction: try direct parse, fenced blocks, or braces slice
                                    let mut rp = seed.clone();
                                    if let Some(val) = parse_jsonish(&text) {
                                        if let Some(s) = val.get("new_category").and_then(|v| v.as_str()) { rp.new_category = Some(s.to_string()); }
                                        if let Some(arr) = val.get("new_tags").and_then(|v| v.as_array()) { rp.new_tags = arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect(); }
                                        if let Some(s) = val.get("new_caption").and_then(|v| v.as_str()) { rp.new_caption = Some(s.to_string()); }
                                        if let Some(s) = val.get("new_description").and_then(|v| v.as_str()) { rp.new_description = Some(s.to_string()); }
                                        rp.generator = format!("cloud:{provider}");
                                        enriched.push(rp);
                                    } else {
                                        log::warn!("[Refine/Cloud] Failed to parse JSON for {}", seed.path);
                                    }
                                }
                                Err(e) => log::warn!("[Refine/Cloud] completion failed: {e}"),
                            }
                        }
                        if !enriched.is_empty() { let _ = tx.send(enriched); }
                        else { log::info!("[Refine/Cloud] No enriched proposals produced"); }
                        crate::ui::status::RERANK_STATUS.set_state(crate::ui::status::StatusState::Idle, "Done");
                        return;
                    }
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
                                if let Some(limit) = limit_paths.as_ref() { if !limit.contains(&p.path) { return; } }
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
                                if let Some(limit) = limit_paths.as_ref() { if !limit.contains(&p.path) { return; } }
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
            if icons::button(ui, svg::BROOM, "Clear").clicked() {
                self.proposals.clear();
                self.generating = false;
            }
            ui.add_sized([240.0, 20.0], TextEdit::singleline(&mut self.filter_text).hint_text("Filter by tag/category/path"));
        
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if icons::button(ui, svg::CHECKS, "Accept All").on_hover_text("Apply every flagged proposal that the filter shows").clicked() {
                    let indices = accept_indices(&self.proposals, &self.filter_text, true);
                    self.accept_at(&indices);
                }
                if icons::button(ui, svg::MINUS_CIRCLE, "Deselect All").clicked() { for p in &mut self.proposals { p.selected = false; } }
                if icons::button(ui, svg::LIST_CHECKS, "Select All").clicked() { for p in &mut self.proposals { p.selected = true; } }
                if icons::button(ui, svg::CHECK_CIRCLE, "Accept Selected").on_hover_text("Apply every flagged proposal, including ones the filter hides").clicked() {
                    let indices = accept_indices(&self.proposals, &self.filter_text, false);
                    self.accept_at(&indices);
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
            let mut accept_row: Option<usize> = None;
            let mut reject_row: Option<usize> = None;
            for (idx, p) in self.proposals.iter_mut().enumerate() {
                if !matches_filter(p, &self.filter_text) { continue; }
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
                            icons::heading(ui, svg::FILE_TEXT, "Current");
                            ui.separator();
                        });
                        ui[1].vertical_centered(|ui| {
                            icons::heading(ui, svg::SPARKLE, "Proposed");
                            ui.separator();
                        });
                    });

                    ui.vertical_centered(|ui| icons::heading(ui, svg::AME_CATEGORY, "Category"));
                    if p.new_category.is_some() {
                        ui.columns(2, |ui| {
                            ui[0].label(text_or_none(&p.current_category));
                            ui[1].label(text_or_none(&p.new_category));
                        });
                        ui.separator();
                    }
                    if !p.new_tags.is_empty() {
                        ui.columns(2, |ui| {
                            ui[0].label(tags_text(&p.current_tags));
                            ui[1].label(tags_text(&p.new_tags));
                        });
                    }

                    ui.vertical_centered(|ui| icons::heading(ui, svg::CHAT_TEXT, "Caption"));
                    if p.current_caption.is_some() || p.new_caption.is_some() {
                        ui.separator();
                        ui.columns(2, |ui| {
                            ui[0].label(text_or_none(&p.current_caption));
                            ui[1].label(text_or_none(&p.new_caption));
                        });
                    }

                    ui.vertical_centered(|ui| icons::heading(ui, svg::AME_AI_DESCRIBE, "Description"));
                    if p.current_description.is_some() || p.new_description.is_some() {
                        ui.separator();
                        ui.columns(2, |ui| {
                            ui[0].label(text_or_none(&p.current_description));
                            ui[1].label(text_or_none(&p.new_description));
                        });
                    }

                    ui.separator();
                    ui.horizontal(|ui| {
                        if icons::button(ui, svg::CHECK, "Accept").on_hover_text("Apply this proposal").clicked() {
                            accept_row = Some(idx);
                        }
                        if icons::button(ui, svg::X, "Reject").on_hover_text("Drop this proposal without applying it").clicked() {
                            reject_row = Some(idx);
                        }
                    });
                });
                ui.separator();
            }
            if let Some(idx) = accept_row {
                self.accept_at(&[idx]);
            } else if let Some(idx) = reject_row {
                self.remove_at(&std::collections::HashSet::from([idx]));
            }
        });
    }
}

/// Try to parse a JSON object from a potentially noisy LLM reply.
fn parse_jsonish(text: &str) -> Option<serde_json::Value> {
    // 1) Direct parse
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(text) { return Some(v); }
    let t = text.trim();
    // 2) Fenced code block ```json ... ``` or ``` ... ```
    if let Some(start) = t.find("```") {
        let rest = &t[start + 3..];
        // Skip optional language tag
        let rest = if rest.to_ascii_lowercase().starts_with("json") { &rest[4..] } else { rest };
        if let Some(end) = rest.find("```") {
            let block = &rest[..end];
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(block) { return Some(v); }
        }
    }
    // 3) Extract substring between first '{' and last '}'
    if let (Some(i), Some(j)) = (t.find('{'), t.rfind('}')) {
        if i < j {
            let slice = &t[i..=j];
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(slice) { return Some(v); }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proposal(path: &str, selected: bool, category: Option<&str>) -> RefinementProposal {
        RefinementProposal {
            path: path.into(),
            current_category: category.map(str::to_string),
            current_tags: vec!["beach".into()],
            current_caption: None,
            current_description: None,
            new_category: None,
            new_tags: Vec::new(),
            new_caption: None,
            new_description: None,
            selected,
            generator: "test".into(),
        }
    }

    fn sample() -> Vec<RefinementProposal> {
        vec![
            proposal("/a/sunset.jpg", true, Some("nature")),
            proposal("/a/city.jpg", false, Some("urban")),
            proposal("/b/sunrise.jpg", true, Some("nature")),
        ]
    }

    #[test]
    fn accept_all_honors_per_row_flags() {
        assert_eq!(accept_indices(&sample(), "", true), vec![0, 2]);
    }

    #[test]
    fn accept_all_honors_the_filter_but_accept_selected_does_not() {
        let p = sample();
        assert_eq!(accept_indices(&p, "sunset", true), vec![0]);
        assert_eq!(accept_indices(&p, "sunset", false), vec![0, 2]);
    }

    #[test]
    fn nothing_flagged_accepts_nothing() {
        let mut p = sample();
        p.iter_mut().for_each(|x| x.selected = false);
        assert!(accept_indices(&p, "", true).is_empty());
        assert!(accept_indices(&p, "", false).is_empty());
    }

    #[test]
    fn the_filter_matches_path_category_and_tags_case_insensitively() {
        let p = proposal("/a/Sunset.jpg", true, Some("Nature"));
        assert!(matches_filter(&p, "SUNSET"));
        assert!(matches_filter(&p, "nature"));
        assert!(matches_filter(&p, "beach"));
        assert!(!matches_filter(&p, "forest"));
    }

    #[test]
    fn proposals_render_as_plain_text_not_debug_output() {
        assert_eq!(text_or_none(&Some("x".into())), "x");
        assert_eq!(text_or_none(&None), "(none)");
        assert_eq!(text_or_none(&Some(String::new())), "(none)");
        assert_eq!(tags_text(&["a".into(), "b".into()]), "a, b");
        assert_eq!(tags_text(&[]), "(none)");
    }

    #[test]
    fn reject_removes_only_the_chosen_row() {
        let (updates_tx, _updates_rx) = crossbeam::channel::unbounded();
        let (toast_tx, _toast_rx) = crossbeam::channel::unbounded();
        let mut panel = RefinementsPanel::new(updates_tx, toast_tx);
        panel.proposals = sample();
        panel.remove_at(&std::collections::HashSet::from([1]));
        let paths: Vec<_> = panel.proposals.iter().map(|p| p.path.as_str()).collect();
        assert_eq!(paths, vec!["/a/sunset.jpg", "/b/sunrise.jpg"]);
    }
}
