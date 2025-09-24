use eframe::egui::*;
use egui_extras::syntax_highlighting::{highlight, CodeTheme};
use base64::Engine as _;
use crate::ui::status::GlobalStatusIndicator;
use crossbeam::channel::{unbounded, Receiver, Sender};
use tokio::task::JoinHandle;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChatRole { User, Assistant }

#[derive(Clone, Debug)]
pub struct ChatMessage { pub role: ChatRole, pub content: String }

pub struct AssistantPanel {
    pub prompt: String,
    pub progress: Option<String>,
    pub last_reply: String,
    // If true, temporarily hide the auto-selected image tag for this prompt
    pub suppress_selected_attachment: bool,
    // Attachments chosen by the user (absolute file paths)
    pub attachments: Vec<String>,
    // Chat state
    pub messages: Vec<ChatMessage>,
    chat_rx: Receiver<String>,
    chat_tx: Sender<String>,
    chat_streaming: bool,
    // Sessions and toolbar
    pub sessions: Vec<(String, Vec<ChatMessage>)>,
    pub active_session: usize,
    // DB session ids parallel to sessions vector
    pub session_ids: Vec<surrealdb::RecordId>,
    pub temp: f32,
    pub model_override: Option<String>,
    // Handle for the currently running assistant generation (to allow Stop)
    assistant_task: Option<JoinHandle<()>>,
    // Refinement UI state
    pub ref_old_category: String,
    pub ref_new_category: String,
    pub ref_old_tag: String,
    pub ref_new_tag: String,
    pub ref_delete_tag: String,
    pub ref_limit_tags: i32,
    // DB picker state
    pub show_db_picker: bool,
    db_rx: Receiver<Vec<(String, String, Option<String>)>>, // (path, filename, thumbnail_b64)
    db_tx: Sender<Vec<(String, String, Option<String>)>>,
    pub db_items: Vec<(String, String)>,
    pub db_filter: String,
    // In-memory tiny texture cache for attachment chips
    thumb_cache: HashMap<String, egui::TextureHandle>,
    db_thumb_map: HashMap<String, String>,
    // Track collapsed message indices for very long messages
    collapsed: HashSet<usize>,
    // OpenRouter models UI state
    show_models_modal: bool,
    models_list: Vec<String>,
    models_filter: String,
    models_rx: Option<crossbeam::channel::Receiver<Result<Vec<String>, String>>>,
    models_loading: bool,
    models_error: Option<String>,
}
// API key fallback to env vars
pub fn env_or(opt: &Option<String>, key: &str) -> Option<String> {
    opt.clone().or_else(|| std::env::var(key).ok())
}

impl AssistantPanel {
    pub fn new() -> Self {
        let (chat_tx, chat_rx) = unbounded::<String>();
        let (db_tx, db_rx) = unbounded::<Vec<(String, String, Option<String>)>>();
        Self {
            prompt: String::new(),
            progress: None,
            last_reply: String::new(),
            suppress_selected_attachment: false,
            attachments: Vec::new(),
            messages: Vec::new(),
            chat_rx,
            chat_tx,
            chat_streaming: false,
            sessions: Vec::new(),
            active_session: 0,
            session_ids: Vec::new(),
            temp: 0.2,
            model_override: None,
            assistant_task: None,
            ref_old_category: String::new(),
            ref_new_category: String::new(),
            ref_old_tag: String::new(),
            ref_new_tag: String::new(),
            ref_delete_tag: String::new(),
            ref_limit_tags: 0,
            show_db_picker: false,
            db_rx,
            db_tx,
            db_items: Vec::new(),
            db_filter: String::new(),
            thumb_cache: HashMap::new(),
            db_thumb_map: HashMap::new(),
            collapsed: HashSet::new(),
            show_models_modal: false,
            models_list: Vec::new(),
            models_filter: String::new(),
            models_rx: None,
            models_loading: false,
            models_error: None,
        }
    }
    
    pub fn ui(&mut self, ctx: &Context, explorer: &mut crate::ui::file_table::FileExplorer) {
        // Initialize a default session and defaults
        if self.sessions.is_empty() {
            // Load sessions from DB
            let (tx, rx) = unbounded::<(Vec<(String, Vec<ChatMessage>)>, Vec<surrealdb::RecordId>)>();
            tokio::spawn(async move {
                let mut items: Vec<(String, Vec<ChatMessage>)> = Vec::new();
                let mut ids: Vec<surrealdb::RecordId> = Vec::new();
                match crate::database::assistant_chat::list_sessions().await {
                    Ok(list) => {
                        for (id, title) in list.into_iter() {
                            let mut msgs: Vec<ChatMessage> = Vec::new();
                            if let Ok(rows) = crate::database::assistant_chat::load_messages(&id).await {
                                for r in rows.into_iter() {
                                    let role = if r.role == "assistant" { ChatRole::Assistant } else { ChatRole::User };
                                    msgs.push(ChatMessage { role, content: r.content });
                                }
                            }
                            items.push((title, msgs));
                            ids.push(id);
                        }
                    }
                    Err(e) => log::warn!("load sessions failed: {e}"),
                }
                if items.is_empty() {
                    // bootstrap
                    if let Ok(id) = crate::database::assistant_chat::create_session("Chat 1").await {
                        items.push(("Chat 1".into(), Vec::new()));
                        ids.push(id);
                    }
                }
                let _ = tx.send((items, ids));
            });
            if let Ok((items, ids)) = rx.recv() {
                self.sessions = items;
                self.session_ids = ids;
                self.active_session = 0;
                if let Some((_t, msgs)) = self.sessions.get(self.active_session) { self.messages = msgs.clone(); }
            }
        }

        if self.temp <= 0.0 { self.temp = 0.2; }

        // TOP PANEL
        TopBottomPanel::top("assistant_top").show(&ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("AI Assistant");
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.small(RichText::new("Chat").italics());
                });
            });
        });

        // LEFT: Provider/model + Sessions list
        SidePanel::left("assistant_left").exact_width(200.).show(&ctx, |ui| {
            let settings = crate::database::settings::load_settings().unwrap_or_default();
            ui.vertical_centered(|ui| ui.heading(RichText::new("Model Options").color(ui.style().visuals.error_fg_color).strong()));
            ui.separator();
            let providers = ["local-joycaption","openai","gemini","groq","grok","openrouter","custom"];
            let current_provider = settings.ai_chat_provider.clone().unwrap_or_else(|| "local-joycaption".into());
            let mut selected_provider = current_provider.clone();
            ui.label("Provider");
            ComboBox::from_id_salt("provider_combo").selected_text(&selected_provider).show_ui(ui, |ui| {
                for p in providers { ui.selectable_value(&mut selected_provider, p.to_string(), p); }
            });
            if selected_provider != current_provider {
                let mut s2 = settings.clone();
                s2.ai_chat_provider = Some(selected_provider.clone());
                crate::database::settings::save_settings(&s2);
            }
            let current_model = self.model_override.clone().unwrap_or_else(|| settings.openai_default_model.clone().unwrap_or_else(|| "gpt-4o-mini".into()));
            ui.label("Model");
            ui.add_sized([200.0, 22.0], TextEdit::singleline(self.model_override.get_or_insert(current_model)));
            if selected_provider == "openrouter" {
                if ui.small_button("List OpenRouter models").on_hover_text("Fetch available models via async-openai").clicked() {
                    self.show_models_modal = true;
                    self.models_list.clear();
                    self.models_loading = true;
                    let api_key = super::assistant::env_or(&settings.openrouter_api_key, "OPENROUTER_API_KEY");
                    // Always use OpenRouter base (do not reuse generic OpenAI base URL)
                    let base = Some("https://openrouter.ai/api/v1".into());
                    // Launch async fetch and keep receiver
                    let (tx, rx) = crossbeam::channel::unbounded::<Result<Vec<String>, String>>();
                    self.models_rx = Some(rx);
                    tokio::spawn(async move {
                        let res = crate::ai::openai_compat::list_models_via_async_openai("openrouter", api_key, base)
                            .await
                            .map_err(|e| e.to_string());
                        let _ = tx.send(res);
                    });
                }
            }
            ui.add_space(8.0);
            ui.separator();
            ui.label(RichText::new("Sessions").strong());
            ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                for (i, (title, _)) in self.sessions.iter().enumerate() {
                    let selected = i == self.active_session;
                    if ui.selectable_label(selected, title).clicked() {
                        self.active_session = i;
                        self.messages = self.sessions[i].1.clone();
                    }
                }
            });
            ui.horizontal(|ui| {
                if ui.button("+ New").clicked() {
                    let new_title = format!("Chat {}", self.sessions.len()+1);
                    let new_title_clone = new_title.clone();
                    // Create in DB
                    let (tx, rx) = unbounded::<Option<surrealdb::RecordId>>();
                    tokio::spawn(async move {
                        let id = crate::database::assistant_chat::create_session(&new_title_clone).await.ok();
                        let _ = tx.send(id);
                    });
                    if let Ok(Some(id)) = rx.recv() {
                        self.sessions.push((new_title, Vec::new()));
                        self.session_ids.push(id);
                        self.active_session = self.sessions.len()-1;
                        self.messages.clear();
                    }
                }
                if ui.button("Clear").clicked() {
                    self.messages.clear();
                    if let Some((title, store)) = self.sessions.get_mut(self.active_session) {
                        *store = self.messages.clone();
                        *title = format!("Chat {}", self.active_session+1);
                    }
                }
            });
            if self.sessions.len() > 1 {
                if ui.button("Delete").clicked() {
                    self.sessions.remove(self.active_session);
                    if self.active_session < self.session_ids.len() { self.session_ids.remove(self.active_session); }
                    self.active_session = self.active_session.saturating_sub(1);
                    if let Some((_t, msgs)) = self.sessions.get(self.active_session) { self.messages = msgs.clone(); } else { self.messages.clear(); }
                }
            }
        });

        // RIGHT: Inference settings
        SidePanel::right("assistant_right").exact_width(280.).show(&ctx, |ui| {
            ui.vertical_centered(|ui| ui.heading(RichText::new("Inference Settings").color(ui.style().visuals.error_fg_color).strong()));
            ui.separator();
            let mut settings = crate::database::settings::load_settings().unwrap_or_default();
            ui.label("Endpoint (OpenAI-compatible base URL)");
            ui.add(TextEdit::singleline(settings.openai_base_url.get_or_insert(String::new())).hint_text("e.g. http://localhost:11434/v1"));
            ui.label("Organization (optional)");
            ui.add(TextEdit::singleline(settings.openai_organization.get_or_insert(String::new())));
            ui.label("Temperature");
            ui.add(Slider::new(&mut self.temp, 0.0..=1.5).logarithmic(false));
            ui.add_space(6.0);
            if ui.button("Save Settings").clicked() {
                crate::database::settings::save_settings(&settings);
            }
            ui.separator();
            CollapsingHeader::new("Refine Categories and Tags").show(ui, |ui| {
                // Moved existing refine tools here
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
        });

        // BOTTOM: Input bar
        TopBottomPanel::bottom("assistant_bottom").show(&ctx, |ui| {
            // We'll track UI intents and apply to self after the closure to avoid borrow conflicts
            let mut suppress_auto_selected = false;
            let mut remove_index: Option<usize> = None;
            // Attachments row (chips)
            ui.horizontal_wrapped(|ui| {
                // Auto-attachment tag for current selection
                let selected_path = explorer.current_thumb.path.clone();
                if !selected_path.is_empty() && !self.suppress_selected_attachment {
                    // show auto-selected tag by default; user can hide via ✕
                    Frame::new().show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(format!("📎 {}", std::path::Path::new(&selected_path).file_name().and_then(|s| s.to_str()).unwrap_or("selected"))).monospace());
                            if ui.small_button("✕").on_hover_text("Remove").clicked() { suppress_auto_selected = true; }
                        });
                    });
                }
                // User-added attachments (iterate over a snapshot to avoid borrowing self immutably while needing &mut self)
                let attachments_snapshot: Vec<String> = self.attachments.clone();
                for (i, p) in attachments_snapshot.iter().enumerate() {
                    Frame::new().show(ui, |ui| {
                        ui.horizontal(|ui| {
                            if let Some(tex) = self.try_thumbnail(ui, p) {
                                let size = vec2(20.0, 20.0);
                                let src = eframe::egui::ImageSource::Texture((tex, size).into());
                                ui.add(eframe::egui::Image::new(src));
                            }
                            let fname = std::path::Path::new(p).file_name().and_then(|s| s.to_str()).unwrap_or(p);
                            ui.label(RichText::new(format!(" {}", fname)).monospace());
                            if ui.small_button("✕").clicked() { remove_index = Some(i); }
                        });
                    });
                }
            });
            // Apply UI intents after the closure
            if suppress_auto_selected { self.suppress_selected_attachment = true; }
            if let Some(i) = remove_index { self.attachments.remove(i); }

            TopBottomPanel::bottom("Bottom panel chat message").exact_height(25.).show_inside(ui, |ui| {
                // Plus menu to add attachments
                ui.menu_button("✚", |ui| {
                    if ui.button("Attach current selection").clicked() {
                        if !explorer.current_thumb.path.is_empty() { self.suppress_selected_attachment = false; }
                        ui.close_kind(UiKind::Menu);
                    }
                    if ui.button("Attach file…").clicked() {
                        if let Some(file) = rfd::FileDialog::new().set_title("Attach image or file").pick_file() {
                            self.attachments.push(file.display().to_string());
                        }
                        ui.close_kind(UiKind::Menu);
                    }
                    if ui.button("Pick from database…").clicked() {
                        self.show_db_picker = true;
                        let tx = self.db_tx.clone();
                        tokio::spawn(async move {
                            let rows = crate::Thumbnail::get_all_thumbnails().await.unwrap_or_default();
                            let mut items: Vec<(String,String,Option<String>)> = rows.into_iter().map(|t| (t.path, t.filename, t.thumbnail_b64)).collect();
                            if items.len() > 200 { items.truncate(200); }
                            let _ = tx.send(items);
                        });
                        ui.close_kind(UiKind::Menu);
                    }
                });

                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    // Send / Stop / Regenerate icons
                    if ui.button("⬈").on_hover_text("Send").clicked() && !self.prompt.trim().is_empty() {
                        self.send_current_prompt(explorer);
                    }
                    if ui.button("⏹").on_hover_text("Stop").clicked() {
                        self.chat_streaming = false;
                        if let Some(h) = self.assistant_task.take() { h.abort(); }
                    }
                    if ui.button("↻").on_hover_text("Regenerate").clicked() {
                        if let Some(last_user) = self.messages.iter().rev().find(|m| matches!(m.role, ChatRole::User)).cloned() { self.prompt = last_user.content; self.send_current_prompt(explorer); }
                    }
                });
            });
            ui.horizontal(|ui| {
                // Input field
                let te = TextEdit::multiline(&mut self.prompt)
                .desired_rows(5)
                .desired_width(ui.available_width())
                .hint_text("Type your message… Shift+Enter = newline, Enter = send")
                .ui(ui);

                let enter_send = te.has_focus() && ui.input(|i| i.key_pressed(Key::Enter) && !i.modifiers.shift);
                if enter_send && !self.prompt.trim().is_empty() {
                    self.send_current_prompt(explorer);
                }
            });
        });

        // CENTER: Messages list
        CentralPanel::default().show(&ctx, |ui| {
            ScrollArea::vertical().stick_to_bottom(true).auto_shrink([false, false]).show(ui, |ui| {
                for (idx, msg) in self.messages.iter().enumerate() {
                    ui.horizontal(|ui| {
                        match msg.role { ChatRole::User => { ui.strong("You"); }, ChatRole::Assistant => { ui.strong("Assistant"); } }
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            let is_long = msg.content.len() > 1200 || msg.content.lines().count() > 20;
                            if is_long {
                                let is_collapsed = self.collapsed.contains(&idx);
                                let label = if is_collapsed { "Expand" } else { "Collapse" };
                                if ui.small_button(label).clicked() {
                                    if is_collapsed { self.collapsed.remove(&idx); } else { self.collapsed.insert(idx); }
                                }
                                ui.separator();
                            }
                            if ui.small_button("Copy").clicked() { ui.ctx().copy_text(msg.content.clone()); }
                        });
                    });
                    ui.add_space(2.0);
                    let is_long = msg.content.len() > 1200 || msg.content.lines().count() > 20;
                    let is_collapsed = is_long && self.collapsed.contains(&idx);
                    if is_collapsed {
                        self.render_bubble_preview(ui, &msg.content);
                    } else {
                        self.render_bubble(ui, msg);
                    }
                    ui.add_space(8.0);
                    ui.separator();
                }
            });
        });

        // Poll streaming chat updates (if any)
        {
            // Drain any streamed tokens
            while let Ok(tok) = self.chat_rx.try_recv() {
                if let Some(last) = self.messages.iter_mut().rev().find(|m| matches!(m.role, ChatRole::Assistant)) {
                    last.content.push_str(&tok);
                }
                ctx.request_repaint();
            }
            if self.chat_rx.is_empty() && self.chat_streaming {
                // Keep UI lively while streaming
                ctx.request_repaint_after(std::time::Duration::from_millis(50));
            }
        }


        // Poll async OpenRouter models result
        if let Some(rx) = self.models_rx.as_ref() {
            if let Ok(res) = rx.try_recv() {
                match res {
                    Ok(list) => {
                        log::info!("[Assistant] received {} OpenRouter models", list.len());
                        self.models_list = list;
                        self.models_error = None;
                    }
                    Err(e) => {
                        log::warn!("OpenRouter models fetch failed: {e}");
                        self.models_error = Some(e);
                    }
                }
                self.models_rx = None;
                self.models_loading = false;
                ctx.request_repaint();
            } else if self.models_loading {
                // Keep the UI repainting while we wait so the spinner/text updates
                ctx.request_repaint_after(std::time::Duration::from_millis(200));
            }
        }

        // Database picker modal
        if self.show_db_picker {
            while let Ok(items) = self.db_rx.try_recv() {
                self.db_items = items.iter().map(|(p,f,_b)| (p.clone(), f.clone())).collect();
                for (p,_f,b) in items { if let Some(b64) = b { self.db_thumb_map.insert(p, b64); } }
            }
            Window::new("Pick from database")
                .collapsible(true)
                .resizable(true)
                .open(&mut self.show_db_picker)
                .default_size(vec2(520.0, 420.0))
                .show(&ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Filter");
                        ui.add(TextEdit::singleline(&mut self.db_filter).hint_text("filename contains…"));
                    });
                    ui.add_space(4.0);
                    ScrollArea::vertical().show(ui, |ui| {
                        for (path, fname) in self.db_items.iter().filter(|(_p, f)| self.db_filter.is_empty() || f.to_lowercase().contains(&self.db_filter.to_lowercase())) {
                            ui.horizontal(|ui| {
                                ui.label(fname);
                                if ui.small_button("Attach").clicked() {
                                    self.attachments.push(path.clone());
                                    if let Some(b64) = self.db_thumb_map.get(path).cloned() {
                                        if let Some(tex) = Self::decode_b64_to_texture(ui.ctx(), path, &b64) {
                                            self.thumb_cache.insert(path.clone(), tex);
                                        }
                                    }
                                }
                            });
                            ui.separator();
                        }
                    });
                });
        }

        // OpenRouter models modal
        if self.show_models_modal {
            // No eager refresh here; the fetch starts when button is clicked or when Refresh is pressed below.
            Window::new("OpenRouter Models")
                .collapsible(true)
                .resizable(true)
                .open(&mut self.show_models_modal)
                .default_size(vec2(520.0, 420.0))
                .show(&ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Filter");
                        ui.add(TextEdit::singleline(&mut self.models_filter).hint_text("name contains…"));
                        if ui.small_button("Refresh").clicked() {
                            let settings = crate::database::settings::load_settings().unwrap_or_default();
                            let api_key = super::assistant::env_or(&settings.openrouter_api_key, "OPENROUTER_API_KEY");
                            // Always use OpenRouter base (do not reuse generic OpenAI base URL)
                            let base = Some("https://openrouter.ai/api/v1".into());
                            let (tx, rx) = crossbeam::channel::unbounded::<Result<Vec<String>, String>>();
                            self.models_rx = Some(rx);
                            self.models_loading = true;
                            self.models_error = None;
                            tokio::spawn(async move {
                                let res = crate::ai::openai_compat::list_models_via_async_openai("openrouter", api_key, base).await.map_err(|e| e.to_string());
                                let _ = tx.send(res);
                            });
                        }
                    });
                    ui.separator();
                    let matches_count = if self.models_filter.is_empty() { self.models_list.len() } else { self.models_list.iter().filter(|m| m.to_lowercase().contains(&self.models_filter.to_lowercase())).count() };
                    if !self.models_loading { ui.label(RichText::new(format!("{} models ({} shown)", self.models_list.len(), matches_count)).weak()); }
                    if self.models_loading {
                        ui.label(RichText::new("Loading models…").weak());
                    }
                    if let Some(err) = &self.models_error { ui.colored_label(ui.visuals().warn_fg_color, format!("Error: {err}")); }
                    if !self.models_loading && !self.models_list.is_empty() && self.models_filter.is_empty() {
                        egui::CollapsingHeader::new("Preview (first 20)").show(ui, |ui| {
                            for m in self.models_list.iter().take(20) { ui.label(m); }
                        });
                    }
                    ScrollArea::vertical().show(ui, |ui| {
                        for m in self.models_list.iter().filter(|m| self.models_filter.is_empty() || m.to_lowercase().contains(&self.models_filter.to_lowercase())) {
                            ui.horizontal(|ui| {
                                ui.label(m);
                                if ui.small_button("Use").clicked() {
                                    self.model_override = Some(m.clone());
                                    let mut settings = crate::database::settings::load_settings().unwrap_or_default();
                                    settings.openai_default_model = Some(m.clone());
                                    crate::database::settings::save_settings(&settings);
                                }
                                if ui.small_button("Copy").clicked() {
                                    ui.ctx().copy_text(m.clone());
                                }
                            });
                            ui.separator();
                        }
                        if !self.models_loading && self.models_list.is_empty() {
                            ui.label(RichText::new("No models found yet. Ensure API key and try Refresh.").weak());
                        }
                    });
                });
        }
    }
}

impl AssistantPanel {
    fn send_current_prompt(&mut self, explorer: &mut crate::ui::file_table::FileExplorer) {
        self.last_reply.clear();
        self.progress = Some(String::new());
    let prompt = self.prompt.clone();
        // Choose attachments to send: user-picked attachments (all), or fallback to current selection if none (and not suppressed)
        let mut paths: Vec<String> = self.attachments.clone();
        if paths.is_empty() && !explorer.current_thumb.path.is_empty() && !self.suppress_selected_attachment {
            paths.push(explorer.current_thumb.path.clone());
        }
        self.messages.push(ChatMessage { role: ChatRole::User, content: prompt.clone() });
        self.messages.push(ChatMessage { role: ChatRole::Assistant, content: String::new() });
    let chat_tx = self.chat_tx.clone();
    self.chat_streaming = true;
        let tx_updates = explorer.viewer.ai_update_tx.clone();
        let settings = crate::database::settings::load_settings().unwrap_or_default();
        let provider = settings.ai_chat_provider.clone().unwrap_or_else(|| "local-joycaption".into());
        let use_cloud = provider != "local-joycaption";
        crate::ui::status::ASSIST_STATUS.set_state(crate::ui::status::StatusState::Running, if use_cloud { "Cloud request" } else { "Local request" });
        crate::ui::status::ASSIST_STATUS.set_model(&provider);
    if use_cloud {
            let model = self.model_override.clone().unwrap_or_else(|| settings.openai_default_model.clone().unwrap_or_else(|| "gpt-4o-mini".into()));
            let (api_key, base_url, org) = match provider.as_str() {
                "openai" => (super::assistant::env_or(&settings.openai_api_key, "OPENAI_API_KEY"), settings.openai_base_url.clone(), settings.openai_organization.clone()),
                "grok" => (super::assistant::env_or(&settings.grok_api_key, "GROK_API_KEY"), settings.openai_base_url.clone(), None),
                "gemini" => (super::assistant::env_or(&settings.gemini_api_key, "GEMINI_API_KEY"), settings.openai_base_url.clone(), None),
                "groq" => (super::assistant::env_or(&settings.groq_api_key, "GROQ_API_KEY"), settings.openai_base_url.clone(), None),
                // Always use the official OpenRouter base for this provider
                "openrouter" => (super::assistant::env_or(&settings.openrouter_api_key, "OPENROUTER_API_KEY"), Some("https://openrouter.ai/api/v1".into()), None),
                _ => (super::assistant::env_or(&settings.openai_api_key, "OPENAI_API_KEY"), settings.openai_base_url.clone(), None),
            };
            let txu = tx_updates.clone();
            let provider_clone = provider.clone();
            let path_for_updates = paths.get(0).cloned().unwrap_or_default();
            let session_id_for_async = self.session_ids.get(self.active_session).cloned();
            let temp = self.temp;
            let prompt_owned = prompt.clone();
            let paths_for_async = paths.clone();
            let handle = tokio::spawn(async move {
                let cfg = crate::ai::openai_compat::ProviderConfig { provider: provider_clone, api_key, base_url, model, organization: org, temperature: Some(temp) };
                let transcript = format!("You: {}\nAssistant:", prompt_owned);
                // Read all attachments (if any)
                let mut imgs: Vec<Vec<u8>> = Vec::new();
                for p in paths_for_async.iter() {
                    if let Ok(b) = tokio::fs::read(p).await { imgs.push(b); }
                }
                let stream_res = crate::ai::openai_compat::stream_multimodal_reply(cfg, &transcript, &imgs, |tok| { let _ = chat_tx.try_send(tok.to_string()); }).await;
                match stream_res {
                    Ok(full) => {
                        if !path_for_updates.is_empty() { let _ = txu.try_send(crate::ui::file_table::AIUpdate::Interim { path: path_for_updates.clone(), text: full.clone() }); }
                        // persist assistant reply
                        if let Some(id) = session_id_for_async {
                            let _ = crate::database::assistant_chat::append_message(&id, "assistant", &full, None).await;
                        }
                        crate::ui::status::ASSIST_STATUS.set_state(crate::ui::status::StatusState::Idle, "Idle");
                    }
                    Err(e) => { log::error!("cloud assistant error: {e}"); crate::ui::status::ASSIST_STATUS.set_error(format!("{e}")); },
                }
            });
            self.assistant_task = Some(handle);
        } else {
            if let Some(path) = paths.get(0).cloned() {
                if std::path::Path::new(&path).is_file() {
                    let instruction = prompt.clone();
                    let chat_tx2 = chat_tx.clone();
                    let handle = tokio::spawn(async move {
                        match tokio::fs::read(&path).await {
                            Ok(bytes) => {
                                let _ = crate::ai::joycap::ensure_worker_started().await;
                                let _ = crate::ai::joycap::stream_describe_bytes_with_callback(bytes, &instruction, |tok| {
                                    let s = tok.to_string();
                                    let _ = tx_updates.try_send(crate::ui::file_table::AIUpdate::Interim { path: path.clone(), text: s.clone() });
                                    let _ = chat_tx2.try_send(s);
                                }).await;
                                crate::ui::status::ASSIST_STATUS.set_state(crate::ui::status::StatusState::Idle, "Idle");
                            }
                            Err(e) => { log::error!("read image failed: {e}"); crate::ui::status::ASSIST_STATUS.set_error(format!("{e}")); },
                        }
                    });
                    self.assistant_task = Some(handle);
                }
            } else {
                let query = prompt.clone();
                let handle = tokio::spawn(async move {
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
                    crate::ui::status::ASSIST_STATUS.set_state(crate::ui::status::StatusState::Idle, "Idle");
                });
                self.assistant_task = Some(handle);
            }
        }
        if let Some((title, store)) = self.sessions.get_mut(self.active_session) {
            *title = self.messages.iter().rev().find(|m| matches!(m.role, ChatRole::User)).map(|m| m.content.chars().take(32).collect::<String>()).unwrap_or_else(|| format!("Chat {}", self.active_session+1));
            *store = self.messages.clone();
        }
        // Persist messages
        if let Some(id) = self.session_ids.get(self.active_session).cloned() {
            let user_attachments = if paths.is_empty() { None } else { Some(paths.clone()) };
            let (_tx, _rx) = (self.db_tx.clone(), &self.db_rx);
            // Spawn two inserts: user prompt and assistant response stream
            let prompt_copy = prompt.clone();
            tokio::spawn(async move {
                let _ = crate::database::assistant_chat::append_message(&id, "user", &prompt_copy, user_attachments).await;
            });
        }
        self.prompt.clear();
        // Reset per-message suppression so the selected image shows up again for next prompt by default
        self.suppress_selected_attachment = false;
    }

    fn render_bubble(&self, ui: &mut Ui, msg: &ChatMessage) {
        let is_user = matches!(msg.role, ChatRole::User);
        let bg = if is_user { Color32::from_rgb(32, 56, 88) } else { Color32::from_gray(28) };
        let stroke = Stroke::new(1.0, if is_user { Color32::from_rgb(48, 88, 140) } else { Color32::DARK_GRAY });
        Frame::new().fill(bg).stroke(stroke).show(ui, |ui| {
            self.render_message_inner(ui, &msg.content);
        });
    }

    fn render_bubble_preview(&self, ui: &mut Ui, content: &str) {
        let bg = Color32::from_gray(28);
        let stroke = Stroke::new(1.0, Color32::DARK_GRAY);
        let preview = {
            let mut s = String::new();
            for (i, line) in content.lines().take(8).enumerate() {
                if i > 0 { s.push('\n'); }
                s.push_str(line);
            }
            if content.lines().count() > 8 { s.push_str("\n…"); }
            s
        };
        Frame::new().fill(bg).stroke(stroke).show(ui, |ui| {
            ui.label(preview);
        });
    }

    fn render_message_inner(&self, ui: &mut Ui, text: &str) {
        // Parse fenced code blocks with optional language: ```lang\ncode...\n```
        #[derive(Debug)]
        enum Seg { Text(String), Code { lang: String, code: String } }
        let mut segs: Vec<Seg> = Vec::new();
        let mut in_code = false;
        let mut cur_lang = String::new();
        let mut buf = String::new();
        for line in text.lines() {
            let trimmed = line.trim_end_matches('\r');
            if trimmed.starts_with("```") {
                if in_code {
                    // close code block
                    segs.push(Seg::Code { lang: cur_lang.clone(), code: buf.trim_end().to_string() });
                    buf.clear();
                    cur_lang.clear();
                    in_code = false;
                } else {
                    // flush text
                    if !buf.is_empty() { segs.push(Seg::Text(buf.clone())); buf.clear(); }
                    // open code
                    cur_lang = trimmed.trim_start_matches('`').trim().to_string();
                    in_code = true;
                }
                continue;
            }
            buf.push_str(line);
            buf.push('\n');
        }
        if !buf.is_empty() {
            if in_code { segs.push(Seg::Code { lang: cur_lang, code: buf.trim_end().to_string() }); }
            else { segs.push(Seg::Text(buf.clone())); }
        }

        for seg in segs {
            match seg {
                Seg::Text(s) => {
                    let s = s.trim();
                    if !s.is_empty() { ui.label(s); }
                }
                Seg::Code { lang, code } => {
                    // Header with copy + language
                    ui.horizontal(|ui| {
                        if ui.small_button("Copy").clicked() { ui.ctx().copy_text(code.clone()); }
                        if !lang.is_empty() { ui.label(RichText::new(lang.clone()).weak()); }
                        else { ui.label(RichText::new("code").weak()); }
                    });
                    // Syntax-highlighted, non-interactive code block
                    ui.group(|ui| {
                        let theme = CodeTheme::from_style(ui.style().as_ref());
                        let style_clone = ui.style().clone();
                        let ctx_clone = ui.ctx().clone();
                        let lang_clone = lang.clone();
                        let mut layouter = move |ui: &Ui, text: &dyn egui::TextBuffer, wrap_width: f32| {
                            let s: &str = text.as_str();
                            let mut job = highlight(&ctx_clone, &style_clone, &theme, &lang_clone, s);
                            job.wrap.max_width = wrap_width;
                            ui.fonts(|f| f.layout_job(job))
                        };
            let rows = (code.lines().count().max(3) as f32).min(18.0);
                        let mut code_mut = code.clone();
                        ui.add(
                            TextEdit::multiline(&mut code_mut)
                                .desired_rows(rows as usize)
                .desired_width(ui.available_width())
                                .font(FontId::monospace(14.0))
                                .interactive(false)
                                .layouter(&mut layouter)
                        );
                    });
                }
            }
        }
    }

    fn try_thumbnail(&mut self, ui: &mut Ui, path: &str) -> Option<egui::TextureId> {
        if let Some(h) = self.thumb_cache.get(path) {
            return Some(h.id());
        }
        // Prefer cached DB thumbnail if known
        if let Some(b64) = self.db_thumb_map.get(path).cloned() {
            if let Some(tex) = Self::decode_b64_to_texture(ui.ctx(), path, &b64) {
                let id = tex.id();
                self.thumb_cache.insert(path.to_string(), tex);
                return Some(id);
            }
        }
        // Fallback to file read if no DB thumbnail available
        if let Ok(bytes) = std::fs::read(path) {
            if let Ok(img) = image::load_from_memory(&bytes) {
                let rgba = img.to_rgba8();
                let (w, h) = rgba.dimensions();
                let color = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &rgba);
                let name = format!("assistant_attach:{}", path);
                let tex = ui.ctx().load_texture(name, color, egui::TextureOptions::LINEAR);
                let id = tex.id();
                self.thumb_cache.insert(path.to_string(), tex);
                return Some(id);
            }
        }
        None
    }

    fn decode_b64_to_texture(ctx: &egui::Context, path: &str, b64: &str) -> Option<egui::TextureHandle> {
        let cleaned = b64.split_once(',').map(|(_, v)| v).unwrap_or(b64);
        if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(cleaned.as_bytes()) {
            if let Ok(img) = image::load_from_memory(&bytes) {
                let rgba = img.to_rgba8();
                let (w, h) = rgba.dimensions();
                let color = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &rgba);
                let name = format!("assistant_attach:{}", path);
                return Some(ctx.load_texture(name, color, egui::TextureOptions::LINEAR));
            }
        }
        None
    }
}
