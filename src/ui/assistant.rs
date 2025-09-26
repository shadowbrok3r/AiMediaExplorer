use eframe::egui::*;
use egui_extras::syntax_highlighting::{highlight, CodeTheme};
use egui_json_tree::{JsonTree, JsonTreeStyle, JsonTreeVisuals, ToggleButtonsState, DefaultExpand};
use base64::Engine as _;
use crate::ui::status::GlobalStatusIndicator;
use crossbeam::channel::{unbounded, Receiver, Sender};
use tokio::task::JoinHandle;
use std::collections::{HashMap, HashSet};
use crate::ai::openai_compat::OpenRouterModel;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChatRole { User, Assistant }

#[derive(Clone, Debug)]
pub struct ChatMessage { pub role: ChatRole, pub content: String, pub attachments: Option<Vec<String>> }

pub struct AssistantPanel {
    pub prompt: String,
    pub progress: Option<String>,
    pub last_reply: String,
    pub suppress_selected_attachment: bool,
    pub attachments: Vec<String>,
    pub messages: Vec<ChatMessage>,
    chat_rx: Receiver<String>,
    chat_tx: Sender<String>,
    chat_streaming: bool,
    pub sessions: Vec<(String, Vec<ChatMessage>)>,
    pub active_session: usize,
    pub session_ids: Vec<surrealdb::RecordId>,
    pub temp: f32,
    pub model_override: Option<String>,
    assistant_task: Option<JoinHandle<()>>,
    pub ref_old_category: String,
    pub ref_new_category: String,
    pub ref_old_tag: String,
    pub ref_new_tag: String,
    pub ref_delete_tag: String,
    pub ref_limit_tags: i32,
    pub show_db_picker: bool,
    db_rx: Receiver<Vec<(String, String, Option<String>)>>,
    db_tx: Sender<Vec<(String, String, Option<String>)>>,
    pub db_items: Vec<(String, String)>,
    pub db_filter: String,
    thumb_cache: HashMap<String, egui::TextureHandle>,
    db_thumb_map: HashMap<String, String>,
    collapsed: HashSet<usize>,
    show_models_modal: bool,
    models_list: Vec<String>,
    models_filter: String,
    models_rx: crossbeam::channel::Receiver<Result<Vec<OpenRouterModel>, String>>,
    models_tx: crossbeam::channel::Sender<Result<Vec<OpenRouterModel>, String>>,
    models_json: Vec<OpenRouterModel>,
    models_open: HashSet<String>,
    models_loading: bool,
    models_error: Option<String>,
    mcp_ping_text: String,
    mcp_search_text: String,
    model_sort: ModelSort,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModelSort { NameAsc, NameDesc, PriceAsc, PriceDesc }

// helper to read optional key or env
fn env_or(opt: &Option<String>, key: &str) -> Option<String> { opt.clone().or_else(|| std::env::var(key).ok()) }

impl AssistantPanel {
    pub fn new() -> Self {
        let (chat_tx, chat_rx) = unbounded::<String>();
        let (db_tx, db_rx) = unbounded::<Vec<(String, String, Option<String>)>>();
        let (models_tx, models_rx) = crossbeam::channel::unbounded::<Result<Vec<OpenRouterModel>, String>>();
        tokio::spawn(async move {
            if let Err(e) = crate::ai::mcp::serve_stdio_background().await {
                log::warn!("[MCP] stdio server not started: {e}");
            }
        });
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
            models_rx,
            models_tx,
            models_json: Vec::new(),
            models_open: HashSet::new(),
            models_loading: false,
            models_error: None,
            mcp_ping_text: String::new(),
            mcp_search_text: String::new(),
            model_sort: ModelSort::NameAsc,
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
                                    msgs.push(ChatMessage { role, content: r.content, attachments: r.attachments });
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
            let providers = [
                "local-joycaption",
                "openai",
                "gemini",
                "groq",
                "grok",
                "openrouter",
                "custom"
                ];
            let current_provider = settings.ai_chat_provider.clone().unwrap_or_else(|| "local-joycaption".into());
            let mut selected_provider = current_provider.clone();
            ui.vertical_centered(|ui| ui.heading(RichText::new("Provider").strong()));
            ComboBox::from_id_salt("provider_combo").selected_text(&selected_provider).show_ui(ui, |ui| {
                for p in providers { ui.selectable_value(&mut selected_provider, p.to_string(), p); }
            });
            if selected_provider != current_provider {
                let mut s2 = settings.clone();
                s2.ai_chat_provider = Some(selected_provider.clone());
                crate::database::settings::save_settings(&s2);
            }
            let current_model = self.model_override.clone().unwrap_or_else(|| settings.openai_default_model.clone().unwrap_or_else(|| "gpt-4o-mini".into()));
            ui.vertical_centered(|ui| ui.heading(RichText::new("Model").strong()));
            TextEdit::singleline(self.model_override.get_or_insert(current_model)).desired_width(ui.available_width()).ui(ui);
            if selected_provider == "openrouter" {
                if ui.button("List OpenRouter models").on_hover_text("Fetch available models via OpenRouter").clicked() {
                    self.show_models_modal = true;
                    self.models_list.clear();
                    self.models_json.clear();
                    self.models_open.clear();
                    self.models_error = None;
                    self.models_loading = true;
                    let api_key = env_or(&settings.openrouter_api_key, "OPENROUTER_API_KEY");
                    let tx = self.models_tx.clone();
                    tokio::spawn(async move {
                        let res = crate::ai::openai_compat::fetch_openrouter_models_json(api_key, Some("https://openrouter.ai/api/v1".into()))
                            .await
                            .map_err(|e| e.to_string());
                        let _ = tx.send(res);
                    });
                }
            }
            ui.add_space(8.0);
            ui.separator();
            ui.vertical_centered(|ui| ui.heading(RichText::new("Sessions").strong()));
            ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                ui.vertical_centered_justified(|ui| {
                    for (i, (title, _)) in self.sessions.iter().enumerate() {
                        let selected = i == self.active_session;
                        if ui.selectable_label(selected, title).clicked() {
                            self.active_session = i;
                            self.messages = self.sessions[i].1.clone();
                        }
                    }
                });
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

            ui.separator();
            CollapsingHeader::new("MCP Tools").show(ui, |ui| {
                // Ping
                ui.label(RichText::new("Ping (util.ping)").strong());
                ui.horizontal(|ui| {
                    ui.add(TextEdit::singleline(&mut self.mcp_ping_text).hint_text("message"));
                    if ui.button("Send").clicked() {
                        let msg = self.mcp_ping_text.clone();
                        tokio::spawn(async move {
                            match crate::ai::mcp::ping_tool(msg).await {
                                Ok(echo) => log::info!("[MCP] ping: {}", echo),
                                Err(e) => log::warn!("[MCP] ping failed: {e}"),
                            }
                        });
                    }
                });

                ui.separator();
                // Search
                ui.label(RichText::new("Media Search (media.search)").strong());
                ui.horizontal(|ui| {
                    ui.add(TextEdit::singleline(&mut self.mcp_search_text).hint_text("query"));
                    if ui.button("Run").clicked() {
                        let q = self.mcp_search_text.clone();
                        let tx_updates = explorer.viewer.ai_update_tx.clone();
                        tokio::spawn(async move {
                            match crate::ai::mcp::media_search_tool(q, Some(32)).await {
                                Ok(hits) => {
                                    let mut results = Vec::new();
                                    for h in hits {
                                        let thumb = crate::Thumbnail::get_thumbnail_by_path(&h.path).await.unwrap_or(None).unwrap_or_default();
                                        results.push(crate::ui::file_table::SimilarResult { thumb, created: None, updated: None, similarity_score: None, clip_similarity_score: Some(h.score) });
                                    }
                                    let _ = tx_updates.try_send(crate::ui::file_table::AIUpdate::SimilarResults { origin_path: "mcp:media.search".into(), results });
                                }
                                Err(e) => log::warn!("[MCP] media.search failed: {e}"),
                            }
                        });
                    }
                });

                ui.separator();
                // Describe
                ui.label(RichText::new("Describe Selected (media.describe)").strong());
                if ui.button("Describe current selection").clicked() {
                    let path = explorer.current_thumb.path.clone();
                    if !path.is_empty() {
                        tokio::spawn(async move {
                            match crate::ai::mcp::media_describe_tool(path.clone()).await {
                                Ok(resp) => log::info!("[MCP] describe ok: {} ({} tags)", resp.caption, resp.tags.len()),
                                Err(e) => log::warn!("[MCP] describe failed: {e}"),
                            }
                        });
                    }
                }
            });
        });

        // BOTTOM: Input bar
        TopBottomPanel::bottom("assistant_bottom").show(&ctx, |ui| {
            if !self.attachments.is_empty() {
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
                                if ui.button("✕").on_hover_text("Remove").clicked() { suppress_auto_selected = true; }
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
                                if ui.button("🗙").clicked() { remove_index = Some(i); }
                            });
                        });
                    }
                });
                // Apply UI intents after the closure
                if suppress_auto_selected { self.suppress_selected_attachment = true; }
                if let Some(i) = remove_index { self.attachments.remove(i); }
            }

            ui.horizontal(|ui| {
                // Plus menu to add attachments
                ui.menu_button("➕", |ui| {
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

            ui.separator();

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
                let total_messages = self.messages.len();
                for idx in 0..total_messages {
                    let msg = &self.messages[idx];
                    // clone minimal data to avoid borrow issues
                    let role_clone = msg.role.clone();
                    let content_clone = msg.content.clone();
                    let atts_clone = msg.attachments.clone();
                    ui.horizontal(|ui| {
                        match role_clone { ChatRole::User => { ui.strong("You"); }, ChatRole::Assistant => { ui.strong("Assistant"); } }
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            let is_long = content_clone.len() > 1200 || content_clone.lines().count() > 20;
                            if is_long {
                                let is_collapsed = self.collapsed.contains(&idx);
                                let label = if is_collapsed { "Expand" } else { "Collapse" };
                                if ui.button(label).clicked() {
                                    if is_collapsed { self.collapsed.remove(&idx); } else { self.collapsed.insert(idx); }
                                }
                                ui.separator();
                            }
                            if ui.button("Copy").clicked() { ui.ctx().copy_text(content_clone.clone()); }
                        });
                    });
                    ui.add_space(2.0);
                    let is_long = content_clone.len() > 1200 || content_clone.lines().count() > 20;
                    let is_collapsed = is_long && self.collapsed.contains(&idx);
                    if is_collapsed {
                        self.render_bubble_preview(ui, &content_clone);
                    } else {
                        let temp = ChatMessage { role: role_clone.clone(), content: content_clone.clone(), attachments: atts_clone.clone() };
                        self.render_bubble(ui, &temp);
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
        if let Ok(res) = self.models_rx.try_recv() {
            match res {
                Ok(json_list) => {
                    log::info!("[Assistant] received {} OpenRouter models", json_list.len());
                    self.models_json = json_list;
                    self.models_list = self.models_json.iter().map(|m| m.id.clone()).collect();
                    self.models_error = None;
                }
                Err(e) => {
                    log::warn!("OpenRouter models fetch failed: {e}");
                    self.models_error = Some(e);
                    self.models_json.clear();
                    self.models_list.clear();
                }
            }
            self.models_loading = false;
            ctx.request_repaint();
        } else if self.models_loading {
            ctx.request_repaint_after(std::time::Duration::from_millis(200));
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
                        TextEdit::singleline(&mut self.db_filter).hint_text("filename contains…").ui(ui);
                    });
                    ui.add_space(4.0);
                    ScrollArea::vertical().show(ui, |ui| {
                        for (path, fname) in self.db_items.iter().filter(|(_p, f)| self.db_filter.is_empty() || f.to_lowercase().contains(&self.db_filter.to_lowercase())) {
                            ui.separator();
                            ui.horizontal(|ui| {
                                // Thumbnail preview if available
                                if let Some(tex) = self.thumb_cache.get(path) {
                                    let size = egui::vec2(48.0, 48.0);
                                    ui.image((tex.id(), size));
                                } else if let Some(b64) = self.db_thumb_map.get(path).cloned() {
                                    if let Some(tex) = Self::decode_b64_to_texture(ui.ctx(), path, &b64) {
                                        let id = tex.id();
                                        self.thumb_cache.insert(path.clone(), tex);
                                        ui.image((id, egui::vec2(48.0, 48.0)));
                                    } else {
                                        ui.label("(no thumb)");
                                    }
                                } else {
                                    ui.label("(no thumb)");
                                }
                                ui.vertical(|ui| {
                                    ui.label(fname);
                                    ui.small(RichText::new(path).weak());
                                });
                                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                    if ui.button(RichText::new("Attach").color(ui.style().visuals.error_fg_color)).clicked() {
                                        self.attachments.push(path.clone());
                                        if let Some(b64) = self.db_thumb_map.get(path).cloned() {
                                            if let Some(tex) = Self::decode_b64_to_texture(ui.ctx(), path, &b64) {
                                                self.thumb_cache.insert(path.clone(), tex);
                                            }
                                        }
                                    }
                                });
                            });
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
                        ui.separator();
                        ui.label("Sort");
                        let sort_label = match self.model_sort { ModelSort::NameAsc=>"Name ⬆", ModelSort::NameDesc=>"Name ⬇", ModelSort::PriceAsc=>"Price ⬆", ModelSort::PriceDesc=>"Price ⬇" };
                        ComboBox::from_id_salt("model_sort_combo").selected_text(sort_label).show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.model_sort, ModelSort::NameAsc, "Name ⬆");
                            ui.selectable_value(&mut self.model_sort, ModelSort::NameDesc, "Name ⬇");
                            ui.selectable_value(&mut self.model_sort, ModelSort::PriceAsc, "Price ⬆");
                            ui.selectable_value(&mut self.model_sort, ModelSort::PriceDesc, "Price ⬇");
                        });
                        if ui.button("Refresh").clicked() {
                            let settings = crate::database::settings::load_settings().unwrap_or_default();
                            let api_key = env_or(&settings.openrouter_api_key, "OPENROUTER_API_KEY");
                            self.models_loading = true;
                            self.models_error = None;
                            let tx = self.models_tx.clone();
                            tokio::spawn(async move {
                                let res = crate::ai::openai_compat::fetch_openrouter_models_json(api_key, Some("https://openrouter.ai/api/v1".into()))
                                    .await
                                    .map_err(|e| e.to_string());
                                let _ = tx.send(res);
                            });
                        }
                    });
                    ui.separator();
                    // Build filtered + sorted list of model refs
                    let mut shown: Vec<&OpenRouterModel> = self.models_json.iter()
                        .filter(|m| self.models_filter.is_empty() || m.id.to_lowercase().contains(&self.models_filter.to_lowercase()))
                        .collect();
                    // helper to parse numeric price (prompt) as f64
                    let parse_price = |s: &str| -> Option<f64> {
                        if s.is_empty() { return None; }
                        let trimmed = s.trim().trim_start_matches('$');
                        let mut num_str = String::new();
                        for ch in trimmed.chars() { if ch.is_ascii_digit() || ch=='.' { num_str.push(ch); } else { break; } }
                        if num_str.is_empty() { None } else { num_str.parse::<f64>().ok() }
                    };
                    match self.model_sort {
                        ModelSort::NameAsc => shown.sort_by(|a,b| a.id.cmp(&b.id)),
                        ModelSort::NameDesc => shown.sort_by(|a,b| b.id.cmp(&a.id)),
                        ModelSort::PriceAsc => shown.sort_by(|a,b| {
                            let ap = a.pricing.as_ref().and_then(|p| p.prompt.as_deref()).and_then(parse_price).unwrap_or(f64::INFINITY);
                            let bp = b.pricing.as_ref().and_then(|p| p.prompt.as_deref()).and_then(parse_price).unwrap_or(f64::INFINITY);
                            ap.partial_cmp(&bp).unwrap_or(std::cmp::Ordering::Equal)
                        }),
                        ModelSort::PriceDesc => shown.sort_by(|a,b| {
                            let ap = a.pricing.as_ref().and_then(|p| p.prompt.as_deref()).and_then(parse_price).unwrap_or(-1.0);
                            let bp = b.pricing.as_ref().and_then(|p| p.prompt.as_deref()).and_then(parse_price).unwrap_or(-1.0);
                            bp.partial_cmp(&ap).unwrap_or(std::cmp::Ordering::Equal)
                        }),
                    }
                    if !self.models_loading { ui.label(RichText::new(format!("{} models ({} shown)", self.models_json.len(), shown.len())).weak()); }
                    if self.models_loading { ui.label(RichText::new("Loading models…").weak()); }
                    if let Some(err) = &self.models_error { ui.colored_label(ui.visuals().warn_fg_color, format!("Error: {err}")); }
                    ScrollArea::vertical().show(ui, |ui| {
                        for model in shown {
                            let id = &model.id;
                            let is_open = self.models_open.contains(id);
                                let toggle = if is_open { "⏷" } else { "▶" };
                                let arch = model.architecture.as_ref();
                                let mut icons = String::new();
                                if let Some(a) = arch {
                                    if a.input_modalities.iter().any(|m| m.contains("text")) { icons.push_str("🖹 "); }
                                    if a.input_modalities.iter().any(|m| m.contains("image")) { icons.push_str("🖼 "); }
                                    if a.input_modalities.iter().any(|m| m.contains("audio")) { icons.push_str("🎵 "); }
                                    if a.input_modalities.iter().any(|m| m.contains("video")) { icons.push_str("🎬 "); }
                                }
                                let price_suffix = if let Some(pr) = &model.pricing {
                                    let prompt_p = pr.prompt.as_deref().unwrap_or("");
                                    let completion_p = pr.completion.as_deref().unwrap_or("");
                                    if !prompt_p.is_empty() || !completion_p.is_empty() {
                                        let pp = if prompt_p.is_empty() {"?"} else {prompt_p};
                                        let cp = if completion_p.is_empty() {"?"} else {completion_p};
                                        format!(" (${}/{})", pp, cp)
                                    } else { String::new() }
                                } else { String::new() };
                                let label_text = format!("{toggle} {id}");
                                let prices = format!("{price_suffix} {icons}");
                                ui.horizontal(|ui| {
                                    let btn = Button::selectable(is_open, label_text)
                                    .min_size([ui.available_width()/1.1, 15.].into())
                                    .right_text(
                                        RichText::new(prices).color(ui.style().visuals.error_fg_color)
                                    ).ui(ui);
                                    if btn.clicked() {
                                        if is_open { self.models_open.remove(id); } else { self.models_open.insert(id.clone()); }
                                    }
                                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                        if ui.button("Use").on_hover_text("Set as current model").clicked() {
                                            self.model_override = Some(id.clone());
                                            let mut settings = crate::database::settings::load_settings().unwrap_or_default();
                                            settings.openai_default_model = Some(id.clone());
                                            crate::database::settings::save_settings(&settings);
                                        }
                                        if ui.button("Copy").on_hover_text("Copy model id").clicked() { ui.ctx().copy_text(id.clone()); }
                                    });
                                });
                                if is_open {
                                    ui.add_space(4.0);
                                    if let Ok(val) = serde_json::to_value(model) {
                                        let style = JsonTreeStyle::new()
                                            .abbreviate_root(true)
                                            .toggle_buttons_state(ToggleButtonsState::VisibleDisabled)
                                            .visuals(JsonTreeVisuals { bool_color: Color32::LIGHT_BLUE, ..Default::default() });
                                        JsonTree::new(format!("model-tree-{}", id), &val)
                                            .style(style)
                                            .default_expand(DefaultExpand::All)
                                            .show(ui);
                                    } else {
                                        ui.colored_label(ui.visuals().error_fg_color, "(failed to serialize model)");
                                    }
                                    ui.separator();
                                }
                        }
                    });
                });
        }
    }

    fn send_current_prompt(&mut self, explorer: &mut crate::ui::file_table::FileExplorer) {
        self.last_reply.clear();
        self.progress = Some(String::new());
        let prompt = self.prompt.clone();
        let prompt_trim = prompt.trim().to_string();
        // Choose attachments to send: user-picked attachments (all), or fallback to current selection if none (and not suppressed)
        let mut paths: Vec<String> = self.attachments.clone();
        if paths.is_empty() && !explorer.current_thumb.path.is_empty() && !self.suppress_selected_attachment {
            paths.push(explorer.current_thumb.path.clone());
        }
            let user_atts = if paths.is_empty() { None } else { Some(paths.clone()) };
            self.messages.push(ChatMessage { role: ChatRole::User, content: prompt.clone(), attachments: user_atts.clone() });
            self.messages.push(ChatMessage { role: ChatRole::Assistant, content: String::new(), attachments: None });
        let chat_tx = self.chat_tx.clone();
        self.chat_streaming = true;
        let tx_updates = explorer.viewer.ai_update_tx.clone();
        let settings = crate::database::settings::load_settings().unwrap_or_default();
        let provider = settings.ai_chat_provider.clone().unwrap_or_else(|| "local-joycaption".into());
        let use_cloud = provider != "local-joycaption";
        crate::ui::status::ASSIST_STATUS.set_state(crate::ui::status::StatusState::Running, if use_cloud { "Cloud request" } else { "Local request" });
        crate::ui::status::ASSIST_STATUS.set_model(&provider);
        // Slash commands: /ping, /search, /describe
        let is_command = prompt_trim.starts_with('/');
        if is_command {
            crate::ui::status::ASSIST_STATUS.set_state(crate::ui::status::StatusState::Running, "MCP command");
            crate::ui::status::ASSIST_STATUS.set_model("mcp");
            let session_id_for_async = self.session_ids.get(self.active_session).cloned();
            let cmdline = prompt_trim.clone();
            // Snapshot current selection path for /describe fallback
            let selected_path_snapshot = explorer.current_thumb.path.clone();
            let handle = tokio::spawn(async move {
                let mut parts = cmdline[1..].splitn(2, ' ');
                let cmd = parts.next().unwrap_or("").to_lowercase();
                let arg = parts.next().unwrap_or("").trim().to_string();
                let reply: String = match cmd.as_str() {
                    "ping" => {
                        match crate::ai::mcp::ping_tool(arg.clone()).await {
                            Ok(echo) => format!("pong: {}", echo),
                            Err(e) => format!("Ping failed: {e}"),
                        }
                    }
                    "search" => {
                        match crate::ai::mcp::media_search_tool(arg.clone(), Some(32)).await {
                            Ok(hits) => {
                                // Send results to the explorer panel
                                let mut results = Vec::new();
                                for h in hits.iter() {
                                    let thumb = crate::Thumbnail::get_thumbnail_by_path(&h.path).await.unwrap_or(None).unwrap_or_default();
                                    results.push(crate::ui::file_table::SimilarResult { thumb, created: None, updated: None, similarity_score: None, clip_similarity_score: Some(h.score) });
                                }
                                let _ = tx_updates.try_send(crate::ui::file_table::AIUpdate::SimilarResults { origin_path: format!("mcp:media.search:{}", arg), results });
                                format!("Found {} similar items for '{}'. See results panel.", hits.len(), arg)
                            }
                            Err(e) => format!("Search failed: {e}"),
                        }
                    }
                    "describe" => {
                        // Use explicit argument or fallback to the selected item snapshot
                        let chosen = if !arg.is_empty() { arg.clone() } else { selected_path_snapshot.clone() };
                        if chosen.is_empty() {
                            "No file selected or path specified.".to_string()
                        } else {
                            match crate::ai::mcp::media_describe_tool(chosen.clone()).await {
                                Ok(resp) => format!("Caption: {}\nCategory: {}\nTags: {}", resp.caption, resp.category, resp.tags.join(", ")),
                                Err(e) => format!("Describe failed: {e}"),
                            }
                        }
                    }
                    _ => format!("Unknown command: /{}", cmd),
                };
                let _ = chat_tx.try_send(reply.clone());
                // Persist assistant reply
                if let Some(id) = session_id_for_async {
                    let _ = crate::database::assistant_chat::append_message(&id, "assistant", &reply, None).await;
                }
                crate::ui::status::ASSIST_STATUS.set_state(crate::ui::status::StatusState::Idle, "Idle");
            });
            self.assistant_task = Some(handle);
            // Not streaming for commands
            self.chat_streaming = false;
            // Persist user prompt is handled below; then early return to avoid cloud/local flow
        } else if use_cloud {
            let model = self.model_override.clone().unwrap_or_else(|| settings.openai_default_model.clone().unwrap_or_else(|| "gpt-4o-mini".into()));
            let (api_key, base_url, org) = match provider.as_str() {
                "openai" => (env_or(&settings.openai_api_key, "OPENAI_API_KEY"), settings.openai_base_url.clone(), settings.openai_organization.clone()),
                "grok" => (env_or(&settings.grok_api_key, "GROK_API_KEY"), settings.openai_base_url.clone(), None),
                "gemini" => (env_or(&settings.gemini_api_key, "GEMINI_API_KEY"), settings.openai_base_url.clone(), None),
                "groq" => (env_or(&settings.groq_api_key, "GROQ_API_KEY"), settings.openai_base_url.clone(), None),
                "openrouter" => (env_or(&settings.openrouter_api_key, "OPENROUTER_API_KEY"), Some("https://openrouter.ai/api/v1".into()), None),
                _ => (env_or(&settings.openai_api_key, "OPENAI_API_KEY"), settings.openai_base_url.clone(), None),
            };
            let txu = tx_updates.clone();
            let provider_clone = provider.clone();
            let path_for_updates = paths.get(0).cloned().unwrap_or_default();
            let session_id_for_async = self.session_ids.get(self.active_session).cloned();
            let temp = self.temp;
            let prompt_owned = prompt.clone();
            let paths_for_async = paths.clone();
            // Capture thumbnail base64 map for fallback (if file read fails)
            let mut thumb_b64_map: std::collections::HashMap<String, String> = std::collections::HashMap::new();
            for p in &paths_for_async { if let Some(b) = self.db_thumb_map.get(p).cloned() { thumb_b64_map.insert(p.clone(), b); } }
            let handle = tokio::spawn(async move {
                let cfg = crate::ai::openai_compat::ProviderConfig { 
                    provider: provider_clone, 
                    api_key, 
                    base_url, 
                    model, 
                    organization: org, 
                    temperature: Some(temp),
                };
                
                // Use the raw prompt; avoid adding artificial 'You:'/'Assistant:' prefixes which can
                // confuse some provider instruction tuning (especially OpenRouter multimodal models).
                let transcript = prompt_owned.clone();
                // Read all attachments (if any)
                let mut imgs: Vec<Vec<u8>> = Vec::new();
                for p in paths_for_async.iter() {
                    match tokio::fs::read(p).await {
                        Ok(b) => imgs.push(b),
                        Err(_) => {
                            if let Some(b64) = thumb_b64_map.get(p) {
                                let cleaned = b64.split_once(',').map(|(_,v)| v).unwrap_or(b64);
                                if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(cleaned.as_bytes()) {
                                    imgs.push(bytes);
                                }
                            }
                        }
                    }
                }
                let stream_res = crate::ai::openai_compat::stream_multimodal_reply(cfg, &transcript, &imgs, |tok| { let _ = chat_tx.try_send(tok.to_string()); }).await;
                match stream_res {
                    Ok(full) => {
                        if !path_for_updates.is_empty() { let _ = txu.try_send(crate::ui::file_table::AIUpdate::Interim { path: path_for_updates.clone(), text: full.clone() }); }
                        if let Some(id) = session_id_for_async {
                            let _ = crate::database::assistant_chat::append_message(&id, "assistant", &full, None).await;
                        }
                        crate::ui::status::ASSIST_STATUS.set_state(crate::ui::status::StatusState::Idle, "Idle");
                    }
                    Err(e) => { 
                        let err_text = format!("Error: {e}");
                        log::error!("cloud assistant error: {e}");
                        let _ = chat_tx.try_send(err_text.clone());
                        if let Some(id) = session_id_for_async {
                            let _ = crate::database::assistant_chat::append_message(&id, "assistant", &err_text, None).await;
                        }
                        crate::ui::status::ASSIST_STATUS.set_error(e.to_string()); 
                    },
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
                            Err(e) => { log::error!("read image failed: {e}"); crate::ui::status::ASSIST_STATUS.set_error(e.to_string()); },
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
        // Clear attachments after sending so next message starts fresh
        self.attachments.clear();
        // Reset per-message suppression so the selected image shows up again for next prompt by default
        self.suppress_selected_attachment = false;
    }

    fn render_bubble(&mut self, ui: &mut Ui, msg: &ChatMessage) {
        let is_user = matches!(msg.role, ChatRole::User);
        let bg = if is_user { Color32::from_rgb(32, 56, 88) } else { Color32::from_gray(28) };
        let stroke = Stroke::new(1.0, if is_user { Color32::from_rgb(48, 88, 140) } else { Color32::DARK_GRAY });
        let width = ui.available_width();
        Frame::new()
            .fill(bg)
            .stroke(stroke)
            .inner_margin(egui::Margin { left: 10, right: 10, top: 6, bottom: 8 })
            .show(ui, |ui| {
                ui.set_width(width / 1.1);
                self.render_message_inner(ui, &msg.content);
                if let Some(atts) = &msg.attachments {
                    if !atts.is_empty() {
                        ui.add_space(6.0);
                        ui.horizontal_wrapped(|ui| {
                            for p in atts.iter() {
                                if let Some(tex) = self.thumb_cache.get(p) {
                                    ui.image((tex.id(), egui::vec2(40.0, 40.0))).on_hover_text(p);
                                } else if let Some(b64) = self.db_thumb_map.get(p).cloned() {
                                    if let Some(tex) = Self::decode_b64_to_texture(ui.ctx(), p, &b64) {
                                        let id = tex.id();
                                        self.thumb_cache.insert(p.clone(), tex);
                                        ui.image((id, egui::vec2(40.0, 40.0))).on_hover_text(p);
                                    } else {
                                        ui.label(RichText::new("[att]").small()).on_hover_text(p);
                                    }
                                } else {
                                    ui.label(RichText::new("[att]").small()).on_hover_text(p);
                                }
                            }
                        });
                    }
                }
            });
    }

    fn render_bubble_preview(&mut self, ui: &mut Ui, content: &str) {
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
        Frame::new().fill(bg).stroke(stroke).inner_margin(egui::Margin{left:8,right:8,top:6,bottom:6}).show(ui, |ui| {
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
                        if ui.button("Copy").clicked() { ui.ctx().copy_text(code.clone()); }
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
