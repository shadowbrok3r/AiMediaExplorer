use crate::{
    app::PrefsDraft,
    database::settings::{changed_fields, copy_fields},
    icons::{self, svg},
    ui::{
        dialogs::{self, DialogResult},
        status::{GlobalStatusIndicator, StatusState, CLIP_STATUS},
    },
    UiSettings,
};
use eframe::egui::*;
use egui_toast::ToastKind;

/// What the Preferences window asked for this frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrefsAction {
    None,
    Save,
    Revert,
    ReloadClip,
    ReloadReranker,
}

fn key_row(ui: &mut Ui, label: &str, slot: &mut Option<String>, env: Option<&str>) {
    ui.horizontal(|ui| {
        ui.label(label);
        let mut v = slot.clone().or_else(|| env.and_then(|k| std::env::var(k).ok())).unwrap_or_default();
        let before = v.clone();
        let resp = ui.add(TextEdit::singleline(&mut v).password(true).desired_width(260.));
        if resp.changed() && v != before {
            *slot = if v.trim().is_empty() { None } else { Some(v) };
        }
    });
}

fn text_row(ui: &mut Ui, label: &str, slot: &mut Option<String>, default: &str) {
    ui.horizontal(|ui| {
        ui.label(label);
        let mut v = slot.clone().unwrap_or_else(|| default.to_string());
        let before = v.clone();
        if ui.text_edit_singleline(&mut v).changed() && v != before {
            *slot = if v.trim().is_empty() { None } else { Some(v) };
        }
    });
}

fn toggle_row(ui: &mut Ui, label: &str, hover: Option<&str>, value: &mut bool) {
    ui.horizontal(|ui| {
        let l = ui.label(label);
        if let Some(h) = hover {
            l.on_hover_text(h);
        }
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.checkbox(value, "Enable");
        });
    });
}

impl crate::app::SmartMediaContext {
    pub fn receive(&mut self, ctx: &eframe::egui::Context) {
        Area::new(Id::unique("toasts_overlay"))
            .fixed_pos(Pos2::ZERO)
            .show(ctx, |ui| {
                self.toasts.show(ui);
            });

        if self.first_run {
            egui_extras::install_image_loaders(ctx);

            // Check if a database exists at the configured path
            let db_path = crate::database::get_db_path();
            let db_exists = std::path::Path::new(&db_path).exists();

            if db_exists {
                self.connect_database();
            } else {
                self.show_db_select_modal = true;
                self.db_local_path = db_path;
                log::info!("No database found at configured path, showing selection modal");
            }
            self.first_run = false;
        }

        for request in crate::theme::debug_panel::take_toast_requests() {
            let _ = self.toast_tx.try_send(request);
        }
        for request in dialogs::take_toasts() {
            let _ = self.toast_tx.try_send(request);
        }

        while let Ok((kind, msg)) = self.toast_rx.try_recv() {
            self.toasts.add(egui_toast::Toast {
                kind,
                text: eframe::egui::RichText::new(msg).into(),
                ..Default::default()
            });
        }

        if let Some(dir) = dialogs::take(|r| match r {
            DialogResult::DbSetupFolder(p) => Some(p.clone()),
            _ => None,
        }) {
            self.db_local_path = dir.display().to_string();
        }
        if let Some(dir) = dialogs::take(|r| match r {
            DialogResult::PrefsModelFolder(p) => Some(p.clone()),
            _ => None,
        }) {
            if let Some(prefs) = self.prefs.as_mut() {
                let model_dir = dir.display().to_string();
                crate::ui::status::VISION_STATUS.set_model(&model_dir);
                crate::ui::status::VISION_STATUS.set_state(StatusState::Idle, "Model path set (reload on next use)");
                prefs.draft.joycaption_model_dir = Some(model_dir);
            }
        }

        if crate::settings::store().take_load_event() {
            self.on_settings_loaded();
        }

        if let Ok(_) = self.db_ready_rx.try_recv() {
            self.db_ready = true;
            log::info!("DB is ready");
            crate::ui::status::DB_STATUS.set_state(crate::ui::status::StatusState::Idle, "Ready");
            let store = std::sync::Arc::clone(crate::settings::store());
            tokio::spawn(async move {
                if let Err(e) = store.reload().await {
                    log::error!("Settings could not be loaded: {e:#}");
                }
            });
        }

        // Drain refinement proposal updates
        while let Ok(batch) = self.refine_rx.try_recv() {
            let n = batch.len();
            self.refinements.proposals = batch;
            self.refinements.generating = false;
            if n == 0 {
                log::warn!("[Refine/UI] Received empty proposals batch");
                let _ = self.toast_tx.try_send((egui_toast::ToastKind::Warning, "No proposals generated".into()));
            } else {
                let _ = self.toast_tx.try_send((egui_toast::ToastKind::Info, format!("Proposals ready ({n})")));
            }
        }

        // Poll global AI engine readiness (cheap lock attempt)
        if !self.ai_ready && self.ai_initializing {
            use std::sync::atomic::Ordering;
            let completed = crate::ai::GLOBAL_AI_ENGINE.index_completed.load(Ordering::Relaxed);
            let active = crate::ai::GLOBAL_AI_ENGINE.index_active.load(Ordering::Relaxed);
            if completed > 0 || active > 0 {
                self.ai_ready = true;
                self.ai_initializing = false;
                crate::ui::status::VISION_STATUS.set_state(crate::ui::status::StatusState::Idle, "Ready");
            }
        }

        self.show_preferences(ctx);
        self.show_egui_settings(ctx);
        self.show_db_modal(ctx);
        self.show_assistant_window(ctx);
    }

    /// Opens the database at the configured folder as a tracked task; returns the folder.
    pub fn connect_database(&self) -> String {
        let path = crate::database::get_db_path();
        tokio::spawn(connect_database_task(self.db_ready_tx.clone(), path.clone()));
        path
    }

    /// One-time work after the stored settings were read.
    fn on_settings_loaded(&mut self) {
        let (auto_indexing, enable_clip, mcp) = crate::settings::store()
            .read(|s| (s.auto_indexing, s.auto_clip_embeddings, s.mcp_stdio_enabled == Some(true)));
        crate::ai::GLOBAL_AI_ENGINE.auto_descriptions_enabled.store(auto_indexing, std::sync::atomic::Ordering::Relaxed);
        crate::ai::GLOBAL_AI_ENGINE.auto_clip_enabled.store(enable_clip, std::sync::atomic::Ordering::Relaxed);
        tokio::spawn(async move {
            let _ = crate::ai::GLOBAL_AI_ENGINE.ensure_clip_engine().await;
            if enable_clip {
                let added = crate::ai::GLOBAL_AI_ENGINE.clip_generate_recursive().await?;
                log::info!("[CLIP] Auto backfill scheduled on settings load: added {added}");
            }
            Ok::<(), anyhow::Error>(())
        });
        if auto_indexing && !self.ai_ready {
            self.ai_initializing = true;
            tokio::spawn(async move { crate::ai::init_global_ai_engine_async().await; });
        }
        if mcp || crate::settings::mcp::requested_by_flag() {
            crate::settings::mcp::start_once();
        }
    }

    /// Applies the edited preferences as a patch of the fields changed since the window opened.
    fn apply_prefs(&mut self, prefs: &mut PrefsDraft) {
        let fields = changed_fields(&prefs.baseline, &prefs.draft);
        let store = crate::settings::store();
        if !fields.is_empty() {
            let draft = prefs.draft.clone();
            store.update("preferences:save", |cur| copy_fields(cur, &draft, &fields));
            let _ = self.toast_tx.try_send(if store.is_loaded() {
                (ToastKind::Success, "Settings saved".into())
            } else {
                (ToastKind::Warning, "Settings are not loaded yet; changes are kept in memory only".into())
            });
        } else {
            let _ = self.toast_tx.try_send((ToastKind::Info, "No settings changed".into()));
        }
        let (baseline, _) = store.snapshot_with_generation();
        prefs.baseline = baseline.clone();
        prefs.draft = baseline;

        let model_key = prefs.draft.clip_model.clone().unwrap_or_else(|| "siglip2-large-patch16-512".to_string());
        CLIP_STATUS.set_model(&model_key);
        CLIP_STATUS.set_state(StatusState::Idle, format!("Model set: {} (reload on next use)", model_key));
        let enable_clip = prefs.draft.auto_clip_embeddings;
        crate::ai::GLOBAL_AI_ENGINE.auto_descriptions_enabled.store(prefs.draft.auto_indexing, std::sync::atomic::Ordering::Relaxed);
        crate::ai::GLOBAL_AI_ENGINE.auto_clip_enabled.store(enable_clip, std::sync::atomic::Ordering::Relaxed);
        {
            let clip = crate::ai::GLOBAL_AI_ENGINE.clip_engine.clone();
            tokio::spawn(async move { *clip.lock().await = None; });
        }
        if enable_clip {
            tokio::spawn(async move {
                let added = crate::ai::GLOBAL_AI_ENGINE.clip_generate_recursive().await?;
                log::info!("[CLIP] Auto backfill after settings save: added {added}");
                Ok::<(), anyhow::Error>(())
            });
        }
        if prefs.draft.auto_indexing && !self.ai_ready {
            self.ai_initializing = true;
            tokio::spawn(async move { crate::ai::init_global_ai_engine_async().await; });
        }
        if prefs.draft.mcp_stdio_enabled == Some(true) {
            crate::settings::mcp::start_once();
        }
        tokio::spawn(async move { crate::ai::reranker::clear_global_reranker().await; });
    }

    fn show_preferences(&mut self, ctx: &Context) {
        if !self.open_settings_modal {
            self.prefs = None;
            return;
        }
        let mut prefs = self.prefs.take().unwrap_or_else(|| {
            let snapshot = crate::settings::snapshot();
            PrefsDraft { baseline: snapshot.clone(), draft: snapshot }
        });
        let mut open = true;
        let mut action = PrefsAction::None;
        let result = Window::new("User Preferences")
            .open(&mut open)
            .default_width(520.0)
            .scroll(true)
            .title_bar(true)
            .show(ctx, |ui| {
                action = self.preferences_ui(ui, &mut prefs.draft);
            });
        if let Some(res) = result {
            if res.response.should_close() {
                open = false;
            }
        }
        match action {
            PrefsAction::None => {}
            PrefsAction::Save => self.apply_prefs(&mut prefs),
            PrefsAction::Revert => {
                let snapshot = crate::settings::snapshot();
                prefs = PrefsDraft { baseline: snapshot.clone(), draft: snapshot };
            }
            PrefsAction::ReloadClip => {
                self.apply_prefs(&mut prefs);
                let model_key = prefs.baseline.clip_model.clone().unwrap_or_else(|| "siglip2-large-patch16-512".to_string());
                CLIP_STATUS.set_model(&model_key);
                CLIP_STATUS.set_state(StatusState::Initializing, format!("Reloading: {}", model_key));
                tokio::spawn(async move {
                    match crate::ai::GLOBAL_AI_ENGINE.ensure_clip_engine().await {
                        Ok(_) => log::info!("[CLIP] Engine reloaded successfully"),
                        Err(e) => log::error!("[CLIP] Engine reload failed: {e}"),
                    }
                });
            }
            PrefsAction::ReloadReranker => {
                self.apply_prefs(&mut prefs);
                tokio::spawn(async move {
                    crate::ai::reranker::clear_global_reranker().await;
                    if let Err(e) = crate::ai::reranker::ensure_reranker_from_settings().await {
                        log::error!("[Reranker] Reload failed: {e}");
                    } else {
                        log::info!("[Reranker] Engine reloaded successfully");
                    }
                });
            }
        }
        self.open_settings_modal = open;
        self.prefs = open.then_some(prefs);
    }

    fn preferences_ui(&mut self, ui: &mut Ui, d: &mut UiSettings) -> PrefsAction {
        let mut action = PrefsAction::None;
        ui.horizontal(|ui| {
            ui.label("Appearance:");
            if icons::button(ui, svg::PALETTE, "Theme settings…").on_hover_text("Open the Appearance window").clicked() {
                self.appearance.open();
            }
        });
        ui.collapsing("Database", |ui| {
            let current_path = crate::database::get_db_path();
            ui.horizontal(|ui| {
                ui.label("Database folder:");
                ui.monospace(current_path.clone());
                if ui.button("Change…").clicked() {
                    dialogs::pick_folder(ui.ctx(), "Choose database folder", move |dir| {
                        let p = dir.display().to_string();
                        if let Err(e) = crate::database::set_db_path(&p) {
                            log::error!("set_db_path failed: {e}");
                            return;
                        }
                        if crate::database::is_connected() {
                            dialogs::toast(ToastKind::Info, format!("Database folder set to {p}. Restart the app to open it."));
                        } else {
                            crate::command::send(crate::command::AppCommand::ConnectDatabase);
                        }
                    });
                }
                if ui.button("Setup…").on_hover_text("Choose or create the database folder").clicked() {
                    self.db_local_path = crate::database::get_db_path();
                    self.show_db_select_modal = true;
                }
            });
            ui.separator();
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
        ui.collapsing("AI", |ui| {
            ui.separator();
            ui.label("Vision (LLaVA/JoyCaption) Model Folder:");
            ui.horizontal(|ui| {
                let mut jc_dir_text = d.joycaption_model_dir.clone().unwrap_or_default();
                let resp = ui.text_edit_singleline(&mut jc_dir_text);
                if ui.button("Browse…").clicked() {
                    let ctx = ui.ctx().clone();
                    dialogs::pick_folder(&ctx, "Choose JoyCaption/LLaVA model folder", {
                        let ctx = ctx.clone();
                        move |dir| dialogs::deliver(&ctx, DialogResult::PrefsModelFolder(dir))
                    });
                }
                let use_this_clicked = ui
                    .button("Use This")
                    .on_hover_text("Set the directory and reload on next use")
                    .clicked();

                if resp.changed() || use_this_clicked {
                    let normalized = jc_dir_text.trim();
                    d.joycaption_model_dir = if normalized.is_empty() { None } else { Some(normalized.to_string()) };
                    let model_path = d.joycaption_model_dir.as_deref().unwrap_or(crate::app::DEFAULT_JOYCAPTION_PATH);
                    crate::ui::status::VISION_STATUS.set_model(model_path);
                    crate::ui::status::VISION_STATUS.set_state(StatusState::Idle, "Model path set (reload on next use)");
                }
            });

            ui.separator();
            ui.label("Assistant Provider (OpenAI-compatible)");
            let mut provider = d.ai_chat_provider.clone().unwrap_or_else(|| "local-joycaption".to_string());
            let prov_before = provider.clone();
            ComboBox::new("ai-chat-provider", "Provider")
                .selected_text(match provider.as_str() {
                    "local-joycaption" => "Local (JoyCaption/LLaVA)",
                    "openai" => "OpenAI",
                    "grok" => "Grok (xAI)",
                    "gemini" => "Google Gemini",
                    "groq" => "Groq",
                    "openrouter" => "OpenRouter",
                    "custom" => "Custom (OpenAI-compatible)",
                    _ => provider.as_str(),
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut provider, "local-joycaption".into(), "Local (JoyCaption/LLaVA)");
                    ui.separator();
                    ui.selectable_value(&mut provider, "openai".into(), "OpenAI");
                    ui.selectable_value(&mut provider, "grok".into(), "Grok (xAI)");
                    ui.selectable_value(&mut provider, "gemini".into(), "Google Gemini");
                    ui.selectable_value(&mut provider, "groq".into(), "Groq");
                    ui.selectable_value(&mut provider, "openrouter".into(), "OpenRouter");
                    ui.separator();
                    ui.selectable_value(&mut provider, "custom".into(), "Custom (OpenAI-compatible)");
                });
            if provider != prov_before {
                d.ai_chat_provider = Some(provider.clone());
            }

            match provider.as_str() {
                "openai" => {
                    key_row(ui, "OpenAI API Key", &mut d.openai_api_key, Some("OPENAI_API_KEY"));
                    text_row(ui, "Organization (optional)", &mut d.openai_organization, "");
                    text_row(ui, "Model", &mut d.openai_default_model, "gpt-5-mini");
                }
                "grok" => {
                    key_row(ui, "Grok (xAI) API Key", &mut d.grok_api_key, Some("GROK_API_KEY"));
                    text_row(ui, "Model", &mut d.openai_default_model, "grok-4-fast-non-reasoning");
                }
                "gemini" => {
                    key_row(ui, "Gemini API Key", &mut d.gemini_api_key, Some("GEMINI_API_KEY"));
                    text_row(ui, "Model", &mut d.openai_default_model, "gemma-3-27b-it");
                }
                "groq" => {
                    key_row(ui, "Groq API Key", &mut d.groq_api_key, Some("GROQ_API_KEY"));
                    text_row(ui, "Model", &mut d.openai_default_model, "llama-3.1-70b-versatile");
                }
                "openrouter" => {
                    key_row(ui, "OpenRouter API Key", &mut d.openrouter_api_key, Some("OPENROUTER_API_KEY"));
                    text_row(ui, "Model", &mut d.openai_default_model, "meta-llama/llama-3.1-70b-instruct");
                }
                "custom" => {
                    ui.horizontal(|ui| {
                        ui.label("Base URL").on_hover_text("OpenAI-compatible endpoint, e.g., http://host:port/v1 (OpenWebUI/LocalAI/vLLM)");
                        let mut v = d.openai_base_url.clone().unwrap_or_else(|| "http://localhost:11434/v1".into());
                        let before = v.clone();
                        if ui.text_edit_singleline(&mut v).changed() && v != before {
                            d.openai_base_url = if v.trim().is_empty() { None } else { Some(v) };
                        }
                    });
                    key_row(ui, "API Key (optional)", &mut d.openai_api_key, None);
                    text_row(ui, "Model", &mut d.openai_default_model, "llama3.1");
                }
                _ => { /* local-joycaption: nothing extra */ }
            }
            ui.separator();
            ui.label("Reranker (HF) Model Repo:").on_hover_text("Hugging Face repo id for reranker model, e.g., jinaai/jina-reranker-m0");
            ui.horizontal(|ui| {
                let mut rer_text = d.reranker_model.clone().unwrap_or_else(|| "jinaai/jina-reranker-m0".to_string());
                let before = rer_text.clone();
                let resp = ui.text_edit_singleline(&mut rer_text);
                if ui.button("Reset").on_hover_text("Reset to default: jinaai/jina-reranker-m0").clicked() {
                    rer_text = "jinaai/jina-reranker-m0".to_string();
                }
                if resp.changed() || rer_text != before {
                    let val = rer_text.trim();
                    d.reranker_model = if val.is_empty() { None } else { Some(val.to_string()) };
                }
            });
            ui.collapsing("Jina M0 Settings", |ui| {
                ui.horizontal(|ui| {
                    ui.label("Max tokens (query+doc)");
                    let mut val = d.jina_max_length.unwrap_or(1024);
                    if ui.add(DragValue::new(&mut val).range(128..=8192)).changed() {
                        d.jina_max_length = Some(val.max(128).min(8192));
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Query type");
                    let mut qt = d.jina_query_type.clone().unwrap_or_else(|| "text".into());
                    let before = qt.clone();
                    ComboBox::new("jina-query-type", "Query type").selected_text(qt.as_str()).show_ui(ui, |ui| {
                        ui.selectable_value(&mut qt, "text".into(), "text");
                        ui.selectable_value(&mut qt, "image".into(), "image");
                    });
                    if qt != before {
                        d.jina_query_type = Some(qt);
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Document type");
                    let mut dt = d.jina_doc_type.clone().unwrap_or_else(|| "text".into());
                    let before = dt.clone();
                    ComboBox::new("jina-doc-type", "Document type").selected_text(dt.as_str()).show_ui(ui, |ui| {
                        ui.selectable_value(&mut dt, "text".into(), "text");
                        ui.selectable_value(&mut dt, "image".into(), "image");
                    });
                    if dt != before {
                        d.jina_doc_type = Some(dt);
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Enable multimodal reranking");
                    let mut en = d.jina_enable_multimodal.unwrap_or(false);
                    if ui.checkbox(&mut en, "Enable").changed() {
                        d.jina_enable_multimodal = Some(en);
                    }
                });
            });
            toggle_row(ui, "Auto Indexing", None, &mut d.auto_indexing);
            toggle_row(ui, "Auto CLIP Embeddings", None, &mut d.auto_clip_embeddings);
            toggle_row(ui, "Blend Image + Text Embeddings", None, &mut d.clip_augment_with_text);
            toggle_row(ui, "Overwrite existing CLIP embeddings (auto)", None, &mut d.clip_overwrite_embeddings);
            toggle_row(ui, "Skip likely icons in scans", None, &mut d.filter_skip_icons);
            toggle_row(
                ui,
                "Auto save files to database",
                Some("When enabled, newly discovered files will be saved to the database automatically (requires an active logical group)"),
                &mut d.auto_save_to_database,
            );
            let mut mcp = d.mcp_stdio_enabled.unwrap_or(false);
            toggle_row(
                ui,
                "Start MCP stdio server",
                Some("Serves MCP over this process's stdin/stdout, which only makes sense when a client launched the app. Turning it off takes effect on next launch."),
                &mut mcp,
            );
            if mcp != d.mcp_stdio_enabled.unwrap_or(false) {
                d.mcp_stdio_enabled = Some(mcp);
            }

            ui.separator();
            ui.label("Excluded Directories (for recursive scans):");
            ui.horizontal(|ui| {
                ui.label("Add new:");
                ui.text_edit_singleline(&mut self.new_excluded_dir);
                if ui.button("Add").clicked() && !self.new_excluded_dir.trim().is_empty() {
                    let new_dir = self.new_excluded_dir.trim().to_string();
                    let dirs = d.excluded_dirs.get_or_insert_with(Vec::new);
                    if !dirs.contains(&new_dir) {
                        dirs.push(new_dir);
                    }
                    self.new_excluded_dir.clear();
                }
            });

            if let Some(ref mut excluded_dirs) = d.excluded_dirs {
                let mut to_remove = None;
                for (idx, dir) in excluded_dirs.iter().enumerate() {
                    ui.horizontal(|ui| {
                        icons::label(ui, svg::FOLDER, dir.as_str());
                        if icons::icon_button(ui, svg::X, "Remove").clicked() {
                            to_remove = Some(idx);
                        }
                    });
                }
                if let Some(idx) = to_remove {
                    excluded_dirs.remove(idx);
                }
                if excluded_dirs.is_empty() {
                    d.excluded_dirs = None;
                }
            } else {
                ui.label("No excluded directories set.");
            }

            toggle_row(ui, "Overwrite existing descriptions", None, &mut d.overwrite_descriptions);
            ui.separator();
            ui.label("CLIP Model:");
            let mut selected = d.clip_model.clone().unwrap_or_else(|| "siglip2-large-patch16-512".to_string());
            let before = selected.clone();
            const CLIP_MODELS: [(&str, &str); 10] = [
                ("unicom-vit-b32", "Unicom ViT-B/32 (fastembed)"),
                ("clip-vit-b32", "OpenAI CLIP ViT-B/32 (fastembed)"),
                ("siglip-base-patch16-224", "SigLIP v1 base 224 (experimental)"),
                ("siglip2-base-patch16-224", "SigLIP v2 base 224 (experimental)"),
                ("siglip2-base-patch16-256", "SigLIP v2 base 256 (experimental)"),
                ("siglip2-base-patch16-384", "SigLIP v2 base 384 (experimental)"),
                ("siglip2-base-patch16-512", "SigLIP v2 base 512 (experimental)"),
                ("siglip2-large-patch16-256", "SigLIP v2 large 256 (experimental)"),
                ("siglip2-large-patch16-384", "SigLIP v2 large 384 (experimental)"),
                ("siglip2-large-patch16-512", "SigLIP v2 large 512 (experimental)"),
            ];
            let shown = CLIP_MODELS.iter().find(|(key, _)| *key == selected).map(|(_, label)| *label).unwrap_or(selected.as_str());
            ComboBox::new("clip-model-select", "Model").selected_text(shown).show_ui(ui, |ui| {
                for (i, (key, label)) in CLIP_MODELS.iter().enumerate() {
                    if i == 2 {
                        ui.separator();
                    }
                    ui.selectable_value(&mut selected, key.to_string(), *label);
                }
            });
            if selected != before {
                log::info!("[Prefs] Changed CLIP model to: {selected}");
                d.clip_model = Some(selected);
            }
            ui.vertical_centered_justified(|ui| {
                ui.label("Prompt Template:");
                ui.text_edit_multiline(&mut d.ai_prompt_template);
            });
        });
        ui.separator();
        ui.collapsing("Filters", |ui| {
            ui.label("Date Range filter");
            ui.horizontal(|ui| {
                let today = jiff::Zoned::now().date();
                let mut after_date: jiff::civil::Date = d
                    .filter_modified_after
                    .as_deref()
                    .and_then(|s| s.parse::<jiff::civil::Date>().ok())
                    .unwrap_or(today);
                let mut before_date: jiff::civil::Date = d
                    .filter_modified_before
                    .as_deref()
                    .and_then(|s| s.parse::<jiff::civil::Date>().ok())
                    .unwrap_or(today);

                ui.label("From:");
                let id_start_salt = format!("{after_date} Start date");
                let resp_after = egui_extras::DatePickerButton::new(&mut after_date).id_salt(&id_start_salt).ui(ui);
                if resp_after.changed() {
                    d.filter_modified_after = Some(after_date.to_string());
                }
                if icons::icon_button(ui, svg::X, "Clear start date").clicked() {
                    d.filter_modified_after = None;
                }

                ui.label("  to  ");

                let id_end_salt = format!("{before_date} End date");
                let resp_before = egui_extras::DatePickerButton::new(&mut before_date).id_salt(&id_end_salt).ui(ui);
                if resp_before.changed() {
                    d.filter_modified_before = Some(before_date.to_string());
                }
                if icons::icon_button(ui, svg::X, "Clear end date").clicked() {
                    d.filter_modified_before = None;
                }
            });
        });
        ui.separator();
        ui.horizontal_wrapped(|ui| {
            if ui.add(crate::theme::primary_button("Save")).clicked() {
                action = PrefsAction::Save;
            }
            if ui.button("Revert").on_hover_text("Discard edits and show the current settings").clicked() {
                action = PrefsAction::Revert;
            }
            if ui.button("Reload CLIP Now").on_hover_text("Save the edits, then re-initialise the CLIP engine").clicked() {
                action = PrefsAction::ReloadClip;
            }
            if ui.button("Reload Reranker Now").on_hover_text("Save the edits, then reload the reranker with the current HF repo setting").clicked() {
                action = PrefsAction::ReloadReranker;
            }
            if ui.button("Generate Missing CLIP Embeddings Now").clicked() {
                tokio::spawn(async move {
                    let count = crate::ai::GLOBAL_AI_ENGINE.clip_generate_recursive().await?;
                    log::info!("[CLIP] Manual generation completed for {count} images");
                    Ok::<(), anyhow::Error>(())
                });
            }
        });
        action
    }

    fn show_egui_settings(&mut self, ctx: &Context) {
        let mut save_requested = false;
        let settings_res = Window::new("Egui Settings")
            .open(&mut self.open_ui_settings)
            .default_size([420.0, 560.0])
            .title_bar(true)
            .show(ctx, |ui| {
                Panel::bottom("Egui Settings bottom panel")
                .exact_size(28.)
                .show(ui, |ui| {
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui.button("Save").on_hover_text("Keep these options across restarts").clicked() {
                            save_requested = true;
                        }
                    });
                });
                ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| ctx.settings_ui(ui));
            });
        if save_requested {
            self.egui_prefs_save_requested = true;
            self.open_ui_settings = false;
        }
        if let Some(res) = settings_res {
            if res.response.should_close() {
                self.open_ui_settings = false;
            }
        }
    }

    fn show_db_modal(&mut self, ctx: &Context) {
        if !self.show_db_select_modal {
            return;
        }
        Window::new("Database Setup")
            .collapsible(false)
            .resizable(false)
            .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.set_min_width(450.0);
                ui.vertical_centered(|ui| {
                    ui.heading("Welcome to Smart Media Explorer");
                    ui.add_space(8.0);
                    ui.label("No database was found. Please choose how to connect:");
                    ui.add_space(16.0);
                });

                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.db_select_mode, crate::app::DbSelectMode::Local, "Local Database");
                    ui.selectable_value(&mut self.db_select_mode, crate::app::DbSelectMode::WebSocket, "WebSocket Server");
                });
                ui.separator();

                match self.db_select_mode {
                    crate::app::DbSelectMode::Local => {
                        ui.label("Database folder path:");
                        ui.horizontal(|ui| {
                            ui.text_edit_singleline(&mut self.db_local_path);
                            if ui.button("Browse…").clicked() {
                                let ctx = ui.ctx().clone();
                                dialogs::pick_folder(&ctx, "Choose database folder", {
                                    let ctx = ctx.clone();
                                    move |dir| dialogs::deliver(&ctx, DialogResult::DbSetupFolder(dir))
                                });
                            }
                        });
                        ui.add_space(8.0);
                        ui.label(RichText::new("A new database will be created if the folder is empty.").weak().italics());
                    }
                    crate::app::DbSelectMode::WebSocket => {
                        ui.label("WebSocket URL:");
                        ui.text_edit_singleline(&mut self.db_websocket_url);
                        ui.add_space(8.0);
                        ui.label(RichText::new("Example: ws://localhost:8000 or wss://your-server.com").weak().italics());
                    }
                }

                if let Some(ref err) = self.db_connection_error {
                    ui.add_space(8.0);
                    ui.colored_label(crate::theme::error(ui), format!("Connection failed: {}", err));
                }

                ui.add_space(16.0);
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Connect").clicked() {
                        self.db_connection_error = None;
                        match self.db_select_mode {
                            crate::app::DbSelectMode::Local => {
                                let path = self.db_local_path.clone();
                                if let Err(e) = crate::database::set_db_path(&path) {
                                    self.db_connection_error = Some(e.to_string());
                                } else {
                                    self.show_db_select_modal = false;
                                    if crate::database::is_connected() {
                                        dialogs::toast(ToastKind::Info, format!("Database folder set to {path}. Restart the app to open it."));
                                    } else {
                                        self.connect_database();
                                    }
                                }
                            }
                            crate::app::DbSelectMode::WebSocket => {
                                self.db_connection_error = Some(
                                    "WebSocket connections require SurrealDB server running. This feature is coming soon.".to_string()
                                );
                            }
                        }
                    }
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui.button("Use Default").on_hover_text("Create a new database in the default location").clicked() {
                            self.db_local_path = crate::database::DB_DEFAULT_PATH.to_string();
                            self.show_db_select_modal = false;
                            if !crate::database::is_connected() {
                                self.connect_database();
                            }
                        }
                    });
                });
            });
    }

    /// Shows the AI Assistant in its own OS window while the View menu toggle is on.
    fn show_assistant_window(&mut self, ctx: &Context) {
        if !self.assistant_window_open {
            return;
        }
        let vp_id = ViewportId::from_hash_of("ai-assistant-viewport");
        let mut close_requested = false;
        let explorer = crate::app::explorer_for(&mut self.file_explorer, &mut self.filtered_tabs, self.active_tab.as_ref());
        let assistant = &mut self.assistant;
        ctx.show_viewport_immediate(
            vp_id,
            ViewportBuilder::default().with_title("AI Assistant").with_inner_size([1000.0, 900.0]),
            |ui, _class| {
                if ui.ctx().input(|i| i.viewport().close_requested()) {
                    close_requested = true;
                }
                assistant.ui(ui, explorer);
            },
        );
        if close_requested {
            self.assistant_window_open = false;
        }
    }
}

/// Name the failed database start is listed under in the needs-attention list.
pub const DB_ATTENTION_SOURCE: &str = "database-connect";

/// What the needs-attention list says when the database could not be opened.
pub fn database_attention(path: &str, error: &str) -> crate::tasks::AttentionItem {
    crate::tasks::AttentionItem {
        key: "database-connect".to_string(),
        kind: crate::tasks::AttentionKind::Database,
        title: "Database not connected".to_string(),
        detail: format!("{path}: {error}"),
        count: None,
        action: Some(crate::command::AppCommand::ConnectDatabase),
    }
}

/// Opens the database as a visible task; a failure is listed under Needs attention with a retry until a later attempt succeeds.
pub async fn connect_database_task(tx: crossbeam::channel::Sender<()>, path: String) {
    if crate::database::is_connected() {
        return;
    }
    track_database_open(&path, crate::database::new(tx)).await;
}

/// Runs `open` as the Database task of `path` and keeps the needs-attention list in step with how it ended.
pub async fn track_database_open(path: &str, open: impl std::future::Future<Output = anyhow::Result<()>>) -> bool {
    use crate::tasks::{TaskKind, TaskSpec};
    let spec = TaskSpec::new(TaskKind::Database, format!("Open database {path}"))
        .retry(crate::command::AppCommand::ConnectDatabase)
        .user_initiated()
        .toast_after(std::time::Duration::from_secs(24 * 3600));
    crate::tasks::with(|r| r.set_attention(DB_ATTENTION_SOURCE, Vec::new()));
    match crate::tasks::track(spec, |_| open).await {
        Ok(()) => {
            log::info!("[DB] opened {path}");
            true
        }
        Err(e) => {
            log::error!("[DB] opening {path} failed: {e:#}");
            let item = database_attention(path, &format!("{e:#}"));
            crate::tasks::with(|r| r.set_attention(DB_ATTENTION_SOURCE, vec![item]));
            crate::command::request_repaint();
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::{AttentionKind, TaskKind, TaskState};

    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn attention() -> Vec<crate::tasks::AttentionItem> {
        crate::tasks::with(|r| r.attention()).into_iter().filter(|i| i.key == "database-connect").collect()
    }

    fn task_of(path: &str) -> Option<crate::tasks::Task> {
        crate::tasks::with(|r| r.tasks().iter().rev().find(|t| t.kind == TaskKind::Database && t.title == format!("Open database {path}")).cloned())
    }

    #[tokio::test]
    async fn a_failed_start_is_a_failed_database_task_and_waits_in_needs_attention_with_a_retry() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let opened = track_database_open("/db/failing", async { Err(anyhow::anyhow!("lock held by another process")) }).await;
        assert!(!opened);
        let task = task_of("/db/failing").expect("the start is a task");
        assert_eq!(task.state, TaskState::Failed);
        assert_eq!(task.error.as_deref(), Some("lock held by another process"));
        assert_eq!(task.retry.as_ref().map(|c| c.name()), Some("connect_database"));
        let items = attention();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].kind, AttentionKind::Database);
        assert!(items[0].detail.contains("/db/failing") && items[0].detail.contains("lock held"));
        assert_eq!(items[0].action.as_ref().map(|c| c.name()), Some("connect_database"));
    }

    #[tokio::test]
    async fn a_later_successful_start_clears_the_attention_item() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        track_database_open("/db/first", async { Err(anyhow::anyhow!("nope")) }).await;
        assert_eq!(attention().len(), 1);
        let opened = track_database_open("/db/second", async { Ok(()) }).await;
        assert!(opened);
        assert!(attention().is_empty());
        assert_eq!(task_of("/db/second").map(|t| t.state), Some(TaskState::Completed));
    }

    #[test]
    fn the_attention_item_names_the_folder_and_the_reason() {
        let item = database_attention("/data/db", "disk full");
        assert_eq!((item.title.as_str(), item.kind), ("Database not connected", AttentionKind::Database));
        assert_eq!(item.detail, "/data/db: disk full");
    }
}
