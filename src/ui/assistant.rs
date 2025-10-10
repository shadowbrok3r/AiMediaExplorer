use egui_json_tree::{DefaultExpand, JsonTree, JsonTreeStyle, JsonTreeVisuals, ToggleButtonsState};
use egui::{style::StyleModifier, containers::menu::{MenuButton, MenuConfig}};
use crossbeam::channel::{Receiver, Sender, unbounded, bounded};
use egui_extras::syntax_highlighting::{CodeTheme, highlight};
use crate::{UiSettings, ui::status::GlobalStatusIndicator};
use crate::ai::openai_compat::OpenRouterModel;
use std::collections::{HashMap, HashSet};
use tokio::task::JoinHandle;
use once_cell::sync::Lazy;
use base64::Engine as _;
use std::sync::Mutex;
use eframe::egui::*;
use chrono::Local;

// Global queue to request attaching file paths to the chat window from other UI modules
pub static ATTACH_REQUESTS: Lazy<Mutex<Vec<Vec<String>>>> = Lazy::new(|| Mutex::new(Vec::new()));

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChatRole {
    User,
    Assistant,
}

#[derive(Clone, Debug)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    pub attachments: Option<Vec<String>>,
    pub created: Option<surrealdb::sql::Datetime>,
}

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

    thumb_cache: HashMap<String, egui::TextureHandle>,
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
    settings: UiSettings,
    // UI-bound copies of option fields that need stable mutable references
    selected_provider: String,
    model_name: String,
    open_model_opts: bool,
    // Async new-session creation pipe (persistent channel)
    new_session_tx: crossbeam::channel::Sender<(Option<surrealdb::RecordId>, String)>,
    new_session_rx: crossbeam::channel::Receiver<(Option<surrealdb::RecordId>, String)>,
    // OpenRouter filters
    filter_text: bool,
    filter_image: bool,
    filter_audio: bool,
    filter_free: bool,
    // Output modality filters
    filter_out_text: bool,
    filter_out_image: bool,
    filter_out_audio: bool,
    // One-shot initial load control and channel
    first_run: bool,
    initial_sessions_tx: crossbeam::channel::Sender<(Vec<(String, Vec<ChatMessage>)>, Vec<surrealdb::RecordId>)>,
    initial_sessions_rx: crossbeam::channel::Receiver<(Vec<(String, Vec<ChatMessage>)>, Vec<surrealdb::RecordId>)>,
    // Async session load upon click (index, messages)
    session_loaded_tx: crossbeam::channel::Sender<(usize, Vec<ChatMessage>)>,
    session_loaded_rx: crossbeam::channel::Receiver<(usize, Vec<ChatMessage>)>,
    // Async DB thumbnail loader: (path, Some((rgba,w,h))) or None on miss
    db_thumb_tx: crossbeam::channel::Sender<(String, Option<(Vec<u8>, u32, u32)>)>,
    db_thumb_rx: crossbeam::channel::Receiver<(String, Option<(Vec<u8>, u32, u32)>)>,
    // Prevent duplicate thumb fetches
    pending_thumb_fetch: std::collections::HashSet<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModelSort {
    NameAsc,
    NameDesc,
    PriceAsc,
    PriceDesc,
}

// helper to read optional key or env
fn env_or(opt: &Option<String>, key: &str) -> Option<String> {
    opt.clone().or_else(|| std::env::var(key).ok())
}

/// Request to attach file paths to the chat window. Safe to call from any UI code.
pub fn request_attach_to_chat(paths: Vec<String>) {
    if paths.is_empty() {
        return;
    }
    ATTACH_REQUESTS.lock().unwrap().push(paths);
}

impl AssistantPanel {
    pub fn new() -> Self {
        let (chat_tx, chat_rx) = unbounded::<String>();
        let (models_tx, models_rx) = unbounded::<Result<Vec<OpenRouterModel>, String>>();
        let (new_session_tx, new_session_rx) = unbounded::<(Option<surrealdb::RecordId>, String)>();
        let (initial_sessions_tx, initial_sessions_rx) = bounded::<(Vec<(String, Vec<ChatMessage>)>, Vec<surrealdb::RecordId>)>(1);
        let (session_loaded_tx, session_loaded_rx) = unbounded::<(usize, Vec<ChatMessage>)>();
        let (db_thumb_tx, db_thumb_rx) = unbounded::<(String, Option<(Vec<u8>, u32, u32)>)>();
        
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
            thumb_cache: HashMap::new(),
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
            settings: UiSettings::default(),
            selected_provider: "local-joycaption".into(),
            model_name: String::new(),
            open_model_opts: true,
            new_session_tx,
            new_session_rx,
            filter_text: false,
            filter_image: false,
            filter_audio: false,
            filter_free: false,
            filter_out_text: false,
            filter_out_image: false,
            filter_out_audio: false,
            first_run: true,
            initial_sessions_tx,
            initial_sessions_rx,
            session_loaded_tx,
            session_loaded_rx,
            db_thumb_tx,
            db_thumb_rx,
            pending_thumb_fetch: HashSet::new(),
        }
    }

    pub fn set_ui_settings(&mut self, settings: UiSettings) {
        // Merge new settings and refresh UI-bound copies
        self.settings = settings.clone();
        self.selected_provider = settings
            .ai_chat_provider
            .clone()
            .unwrap_or_else(|| self.selected_provider.clone());
        self.model_name = self
            .model_override
            .clone()
            .or_else(|| settings.openai_default_model.clone())
            .unwrap_or_else(|| self.model_name.clone());
    }

    pub fn ui(&mut self, ctx: &Context, explorer: &mut crate::ui::file_table::FileExplorer) {
        self.receive(ctx);

        // TOP PANEL
        TopBottomPanel::top("assistant_top").show(&ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Model Options").clicked() {
                    self.open_model_opts = !self.open_model_opts;
                }

                ui.with_layout(Layout::right_to_left(Align::Center), |_ui| {

                });
            });
        });

        // BOTTOM: Input bar
        TopBottomPanel::bottom("assistant_bottom").show(&ctx, |ui| {
            let style = StyleModifier::default();
            style.apply(ui.style_mut());
            let menu_cfg = MenuConfig::default().close_behavior(PopupCloseBehavior::CloseOnClickOutside).style(style.clone());
            ui.add_space(4.);
            MenuBar::new().config(menu_cfg).style(style).ui(ui, |ui| {
                
                // Plus menu to add attachments
                ui.menu_button("➕", |ui| {
                    if ui.button("Attach current selection").clicked() {
                        if !explorer.current_thumb.path.is_empty() {
                            self.suppress_selected_attachment = false;
                        }
                        ui.close_kind(UiKind::Menu);
                    }
                    if ui.button("Attach file…").clicked() {
                        if let Some(file) = rfd::FileDialog::new()
                            .set_title("Attach image or file")
                            .pick_file()
                        {
                            self.attachments.push(file.display().to_string());
                        }
                        ui.close_kind(UiKind::Menu);
                    }
                });

                ui.add_space(4.);
                ui.separator();
                ui.add_space(4.);

                ui.menu_button("MCP Tools", |ui| {
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
                                            let thumb =
                                                crate::Thumbnail::get_thumbnail_by_path(&h.path)
                                                    .await
                                                    .unwrap_or(None)
                                                    .unwrap_or_default();
                                            results.push(crate::ui::file_table::SimilarResult {
                                                thumb,
                                                created: None,
                                                updated: None,
                                                similarity_score: None,
                                                clip_similarity_score: Some(h.score),
                                            });
                                        }
                                        let _ = tx_updates.try_send(
                                            crate::ui::file_table::AIUpdate::SimilarResults {
                                                origin_path: "mcp:media.search".into(),
                                                results,
                                            },
                                        );
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
                                    Ok(resp) => log::info!(
                                        "[MCP] describe ok: {} ({} tags)",
                                        resp.caption,
                                        resp.tags.len()
                                    ),
                                    Err(e) => log::warn!("[MCP] describe failed: {e}"),
                                }
                            });
                        }
                    }
                });

                ui.add_space(4.);
                ui.separator();
                ui.add_space(4.);

                let menu_txt = RichText::new("Inference").color(ui.style().visuals.error_fg_color).strong();
                let style = StyleModifier::default();
                style.apply(ui.style_mut());
                let menu_cfg = MenuConfig::default().close_behavior(PopupCloseBehavior::CloseOnClickOutside).style(style);
                MenuButton::new(menu_txt).config(menu_cfg).ui(ui, |ui| {
                    let mut settings = self.settings.clone();
                    ui.label("Endpoint (OpenAI-compatible base URL)");
                    ui.add(
                        TextEdit::singleline(settings.openai_base_url.get_or_insert(String::new()))
                            .hint_text("e.g. http://localhost:11434/v1"),
                    );
                    ui.label("Organization (optional)");
                    ui.add(TextEdit::singleline(
                        settings.openai_organization.get_or_insert(String::new()),
                    ));
                    ui.label("Temperature");
                    ui.add(Slider::new(&mut self.temp, 0.0..=1.5).logarithmic(false));
                    ui.add_space(6.0);
                    if ui.button("Save Settings").clicked() {
                        crate::database::settings::save_settings(&settings);
                    }
                });

                ui.add_space(4.);
                ui.separator();
                ui.add_space(4.);

                let providers = [
                    "local-joycaption",
                    "openai",
                    "gemini",
                    "groq",
                    "grok",
                    "openrouter",
                    "custom",
                ];

                let current_provider = self.selected_provider.clone();
                let mut selected_provider = current_provider.clone();

                ComboBox::from_id_salt("provider_combo")
                .selected_text(&selected_provider)
                .width(135.)
                .show_ui(ui, |ui| {
                    for p in providers {
                        ui.selectable_value(&mut selected_provider, p.to_string(), p);
                    }
                });

                if self.selected_provider == "openrouter" {
                    if Button::new("List OpenRouter models").ui(ui).on_hover_text("Fetch available models via OpenRouter").clicked() {
                        self.show_models_modal = true;
                        self.models_list.clear();
                        self.models_json.clear();
                        self.models_open.clear();
                        self.models_error = None;
                        self.models_loading = true;
                        let api_key = env_or(&self.settings.openrouter_api_key, "OPENROUTER_API_KEY");
                        let tx = self.models_tx.clone();
                        tokio::spawn(async move {
                            let res = crate::ai::openai_compat::fetch_openrouter_models_json(
                                api_key,
                                Some("https://openrouter.ai/api/v1".into()),
                            )
                            .await
                            .map_err(|e| e.to_string());
                            let _ = tx.try_send(res);
                        });
                    }
                }

                if selected_provider != current_provider {
                    self.selected_provider = selected_provider.clone();
                    let mut s2 = self.settings.clone();
                    s2.ai_chat_provider = Some(selected_provider.clone());
                    crate::database::settings::save_settings(&s2);
                    self.settings = s2;
                }

                let current_model = if self.model_name.is_empty() {
                    self.model_override.clone().unwrap_or_else(|| {
                        self.settings
                            .openai_default_model
                            .clone()
                            .unwrap_or_else(|| "gpt-5-mini".into())
                    })
                } else { self.model_name.clone() };

                let te_resp = TextEdit::singleline(&mut self.model_name).desired_width(250.).ui(ui);

                // Only update the default model on edit; do NOT create a new session or mark it as recently used here.
                if te_resp.lost_focus() && self.model_name != current_model && !self.model_name.trim().is_empty() {
                    self.model_override = Some(self.model_name.clone());
                    let mut s2 = self.settings.clone();
                    s2.openai_default_model = Some(self.model_name.clone());
                    crate::database::settings::save_settings(&s2);
                    self.settings = s2;
                }
                
                if !self.attachments.is_empty() {
                    // We'll track UI intents and apply to self after the closure to avoid borrow conflicts
                    let mut suppress_auto_selected = false;
                    let mut remove_index: Option<usize> = None;
                    // Auto-attachment tag for current selection
                    let selected_path = explorer.current_thumb.path.clone();
                    if !selected_path.is_empty() && !self.suppress_selected_attachment {
                        // show auto-selected tag by default; user can hide via ✕
                        Frame::new().show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label(
                                    RichText::new(format!(
                                        "📎 {}",
                                        std::path::Path::new(&selected_path)
                                            .file_name()
                                            .and_then(|s| s.to_str())
                                            .unwrap_or("selected")
                                    ))
                                    .monospace(),
                                );
                                if ui.button("✕").on_hover_text("Remove").clicked() {
                                    suppress_auto_selected = true;
                                }
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
                                    let src =
                                        eframe::egui::ImageSource::Texture((tex, size).into());
                                    ui.add(eframe::egui::Image::new(src));
                                }
                                let fname = std::path::Path::new(p)
                                    .file_name()
                                    .and_then(|s| s.to_str())
                                    .unwrap_or(p);
                                ui.label(RichText::new(format!(" {}", fname)).monospace());
                                if ui.button("🗙").clicked() {
                                    remove_index = Some(i);
                                }
                            });
                        });
                    }
                    // Apply UI intents after the closure
                    if suppress_auto_selected {
                        self.suppress_selected_attachment = true;
                    }
                    if let Some(i) = remove_index {
                        self.attachments.remove(i);
                    }
                }

                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    // Send / Stop / Regenerate icons
                    if ui.button("⬈").on_hover_text("Send").clicked()
                        && !self.prompt.trim().is_empty()
                    {
                        self.send_current_prompt(explorer);
                    }
                    if ui.button("⏹").on_hover_text("Stop").clicked() {
                        self.chat_streaming = false;
                        if let Some(h) = self.assistant_task.take() {
                            h.abort();
                        }
                    }
                    if ui.button("↻").on_hover_text("Regenerate").clicked() {
                        if let Some(last_user) = self
                            .messages
                            .iter()
                            .rev()
                            .find(|m| matches!(m.role, ChatRole::User))
                            .cloned()
                        {
                            self.prompt = last_user.content;
                            self.send_current_prompt(explorer);
                        }
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

                let enter_send =
                    te.has_focus() && ui.input(|i| i.key_pressed(Key::Enter) && !i.modifiers.shift);
                if enter_send && !self.prompt.trim().is_empty() {
                    self.send_current_prompt(explorer);
                }
            });
        });

        // LEFT: Provider/model + Sessions list
        SidePanel::left("assistant_left")
        .max_width(250.)
        .min_width(200.)
        .show_animated(ctx, self.open_model_opts, |ui| {
            // --- Recent Models UI ---
            let recent_models_vec = self.settings.recent_models.clone();
            let last_used_model_opt = self.settings.last_used_model.clone();
            if !recent_models_vec.is_empty() || last_used_model_opt.is_some() {
                ui.vertical_centered(|ui| ui.heading(RichText::new("Recent Models").color(ui.style().visuals.error_fg_color)));
                ui.horizontal(|ui| {
                    if ui.button("Clear Recent Models").on_hover_text("Remove all stored recent models").clicked() {
                        let mut s2 = self.settings.clone();
                        s2.recent_models.clear();
                        // Optionally also clear last_used_model; keep it if you want separate
                        s2.last_used_model = None;
                        crate::database::settings::save_settings(&s2);
                        self.settings = s2;
                    }
                });
                if let Some(last) = last_used_model_opt.as_deref() {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Last used:").weak());
                        if ui.button(last).clicked() {
                            self.model_override = Some(last.to_string());
                            self.model_name = last.to_string();
                            let mut s2 = self.settings.clone();
                            s2.openai_default_model = Some(last.to_string());
                            crate::database::settings::save_settings(&s2);
                        }
                    });
                }
                if !recent_models_vec.is_empty() {
                    ScrollArea::vertical().id_salt("Recent Models").max_height(150.0).show(ui, |ui| {
                        ui.vertical_centered_justified(|ui| {
                            for m in &recent_models_vec {
                                if ui.button(m).clicked() {
                                    self.model_override = Some(m.clone());
                                    self.model_name = m.clone();
                                    let mut s2 = self.settings.clone();
                                    s2.openai_default_model = Some(m.clone());
                                    crate::database::settings::save_settings(&s2);
                                }
                            }
                        });
                    });
                }
            }
            
            ui.add_space(6.0);
            ui.separator();
            ui.add_space(6.0);

            // --- Sessions list ---
            ui.horizontal(|ui| {
                ui.heading(RichText::new("Sessions").color(ui.style().visuals.error_fg_color));
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui.button(RichText::new("🗙").color(ui.style().visuals.error_fg_color)).on_hover_text("Double Click to Clear ALL Sessions").double_clicked() {
                        self.messages.clear();
                        if let Some((title, store)) = self.sessions.get_mut(self.active_session) {
                            *store = self.messages.clone();
                            *title = format!("Chat {}", self.active_session + 1);
                        }
                    }
                    if ui.button("➕").on_hover_text("Create a new chat session").clicked() {
                        let new_title = format!("Chat {}", self.sessions.len() + 1);
                        tokio::spawn({
                            let title_clone = new_title.clone();
                            let tx_clone = self.new_session_tx.clone();
                            async move {
                                let id = crate::database::assistant_chat::create_session(&title_clone).await.ok();
                                let _ = tx_clone.try_send((id, title_clone));
                            }
                        });
                    }
                });
            });
            let pending_switch: &mut Option<usize> = &mut None;
            ScrollArea::vertical()
            .id_salt("Sessions")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.vertical_centered_justified(|ui| {
                    for (i, (title, _)) in self.sessions.iter().enumerate() {
                        ui.horizontal(|ui| {
                            let selected = i == self.active_session;
                            if Button::selectable(selected, title).min_size(vec2(ui.available_width(), 25.)).ui(ui).clicked() {
                                *pending_switch = Some(i);
                                log::info!("Switching to session {i} '{title}'");
                            }
                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                if ui.button("🗙").on_hover_text("Delete this session").clicked() {
                                    // Defer deletion until after the list iteration
                                    // We can't mutate self.sessions while iterating; mark index
                                    // We'll handle it below the list rendering
                                    // Use a unique id salt to stash requested deletion index in ui memory
                                    ui.data_mut(|d| d.insert_temp::<usize>(Id::new("assistant_delete_idx"), i));
                                }
                            });
                        });
                    }
                });
            });

            if let Some(i) = *pending_switch {
                log::info!("got a pending switch");
                self.active_session = i;
                self.messages.clear();
                // Spawn DB load for this session's messages
                if let Some(session_id) = self.session_ids.get(i).cloned() {
                    log::info!("got a session_id");
                    let tx = self.session_loaded_tx.clone();
                    tokio::spawn(async move {
                        let mut msgs: Vec<ChatMessage> = Vec::new();
                        match crate::database::assistant_chat::load_messages(&session_id).await {
                            Ok(rows) => {
                                log::error!("Got {} rows", rows.len());
                                for r in rows.into_iter() {
                                    let role = if r.role == "assistant" { ChatRole::Assistant } else { ChatRole::User };
                                    let mut atts = r.attachments.clone();
                                    if (atts.is_none() || atts.as_ref().map(|v| v.is_empty()).unwrap_or(true)) && r.attachments_refs.is_some() {
                                        let mut paths: Vec<String> = Vec::new();
                                        if let Some(refs) = r.attachments_refs.clone() {
                                            for rid in refs {
                                                if let Ok(Some(t)) = crate::Thumbnail::get_by_id(&rid).await { paths.push(t.path); }
                                            }
                                        }
                                        if !paths.is_empty() { atts = Some(paths); }
                                    }
                                    msgs.push(ChatMessage { role, content: r.content, attachments: atts, created: r.created });
                                }
                            }
                            Err(e) => log::error!("Failed to load messages for session {session_id}: {e}")
                        }
                        let _ = tx.try_send((i, msgs));
                    });
                }
            }
            
            // Apply a pending deletion if requested
            if let Some(del_idx) = ctx.data(|d| d.get_temp::<usize>(Id::new("assistant_delete_idx"))) {
                if del_idx < self.sessions.len() && del_idx < self.session_ids.len() {
                    let to_delete_id = self.session_ids[del_idx].clone();
                    // Remove from UI state first for snappy UX
                    self.sessions.remove(del_idx);
                    self.session_ids.remove(del_idx);
                    // Adjust active index
                    if self.sessions.is_empty() {
                        self.active_session = 0;
                        self.messages.clear();
                    } else {
                        if self.active_session >= self.sessions.len() { self.active_session = self.sessions.len() - 1; }
                        self.messages = self.sessions[self.active_session].1.clone();
                    }
                    // Fire-and-forget DB deletion
                    tokio::spawn(async move {
                        let _ = crate::database::assistant_chat::delete_session(&to_delete_id).await;
                    });
                }
                // Clear the temp marker
                ctx.data_mut(|d| d.remove::<usize>(Id::new("assistant_delete_idx")));
            }
        });

        // CENTER: Messages list
        CentralPanel::default().show(&ctx, |ui| {
            ScrollArea::vertical()
            .stick_to_bottom(true)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let total_messages = self.messages.len();
                let max_msg_width: f32 = 376.0;
                for idx in 0..total_messages {
                    let msg = &self.messages[idx];
                    // clone minimal data to avoid borrow issues
                    let role_clone = msg.role.clone();
                    let content_clone = msg.content.clone();
                    let atts_clone = msg.attachments.clone();

                    let is_user = matches!(role_clone, ChatRole::User);
                    let created_clone = msg.created.clone();
                    let layout = if is_user { Layout::top_down(Align::Max) } else { Layout::top_down(Align::Min) };
                    ui.with_layout(layout, |ui| {
                        ui.set_max_width(max_msg_width);
                        let is_long = content_clone.len() > 1200 || content_clone.lines().count() > 20;
                        let is_collapsed = is_long && self.collapsed.contains(&idx);
                        if is_collapsed { self.render_bubble_preview(ui, &content_clone); }
                        else {
                            let temp = ChatMessage { role: role_clone.clone(), content: content_clone.clone(), attachments: atts_clone.clone(), created: None };
                            self.render_bubble(ui, &temp, created_clone, idx, content_clone);
                        }
                    });
                    ui.add_space(8.0);
                }
            });
        });
    }

    pub fn receive(&mut self, ctx: &Context) {
        // Drain pending external attach requests
        {
            let mut q = ATTACH_REQUESTS.lock().unwrap();
            if !q.is_empty() {
                for batch in q.drain(..) {
                    for p in batch {
                        if !self.attachments.contains(&p) {
                            self.attachments.push(p);
                        }
                    }
                }
            }
        }
        // One-time initial session load (non-blocking, using bounded(1) channel stored on self)
        if self.first_run {
            self.first_run = false;
            let tx_init = self.initial_sessions_tx.clone();
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
                                    let mut atts = r.attachments.clone();
                                    if (atts.is_none() || atts.as_ref().map(|v| v.is_empty()).unwrap_or(true)) && r.attachments_refs.is_some() {
                                        let mut paths: Vec<String> = Vec::new();
                                        if let Some(refs) = r.attachments_refs.clone() {
                                            for rid in refs {
                                                if let Ok(Some(t)) = crate::Thumbnail::get_by_id(&rid).await { paths.push(t.path); }
                                            }
                                        }
                                        if !paths.is_empty() { atts = Some(paths); }
                                    }
                                    msgs.push(ChatMessage { role, content: r.content, attachments: atts, created: r.created });
                                }
                            }
                            items.push((title, msgs));
                            ids.push(id);
                        }
                    }
                    Err(e) => log::warn!("load sessions failed: {e}"),
                }
                if items.is_empty() {
                    if let Ok(id) = crate::database::assistant_chat::create_session("Chat 1").await {
                        items.push(("Chat 1".into(), Vec::new()));
                        ids.push(id);
                    }
                }
                let _ = tx_init.try_send((items, ids));
            });
        }
        
        if let Ok((items, ids)) = self.initial_sessions_rx.try_recv() {
            self.sessions = items;
            self.session_ids = ids;
            self.active_session = 0;
            if let Some((_t, msgs)) = self.sessions.get(self.active_session) {
                self.messages = msgs.clone();
                // Kick off DB thumb requests for all attachments in the initially active session
                let all_paths: Vec<String> = self
                    .messages
                    .iter()
                    .filter_map(|m| m.attachments.clone())
                    .flatten()
                    .collect();
                for p in all_paths { self.request_db_thumb(&p); }
            }
        }

        // Drain any new session creations
        while let Ok((opt_id, title)) = self.new_session_rx.try_recv() {
            if let Some(id) = opt_id {
                self.sessions.push((title, Vec::new()));
                self.session_ids.push(id);
                self.active_session = self.sessions.len() - 1;
                self.messages.clear();
                // Nothing to warm yet; new chat has no messages
            }
        }

        // Drain loaded session results
        while let Ok((idx, msgs)) = self.session_loaded_rx.try_recv() {
            if idx < self.sessions.len() {
                self.sessions[idx].1 = msgs.clone();
            }
            if idx == self.active_session {
                self.messages = msgs.clone();
                // Kick off DB thumb requests for all attachments in this session
                let all_paths: Vec<String> = self
                    .messages
                    .iter()
                    .filter_map(|m| m.attachments.clone())
                    .flatten()
                    .collect();
                for p in all_paths { self.request_db_thumb(&p); }
            }
            ctx.request_repaint();
        }

        // Drain DB thumbnail results and create textures
        while let Ok((path, opt_rgba)) = self.db_thumb_rx.try_recv() {
            self.pending_thumb_fetch.remove(&path);
            if let Some((rgba, w, h)) = opt_rgba {
                let color = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &rgba);
                let name = format!("assistant_attach:{}", path);
                let tex = ctx.load_texture(name, color, egui::TextureOptions::LINEAR);
                self.thumb_cache.insert(path, tex);
                ctx.request_repaint();
            }
        }


        // Poll streaming chat updates (if any)
        {
            // Drain any streamed tokens
            while let Ok(tok) = self.chat_rx.try_recv() {
                if let Some(last) = self
                    .messages
                    .iter_mut()
                    .rev()
                    .find(|m| matches!(m.role, ChatRole::Assistant))
                {
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
                        ui.add(
                            TextEdit::singleline(&mut self.models_filter)
                                .hint_text("name contains…"),
                        );
                        ui.separator();
                        ui.label("Input");
                        ui.toggle_value(&mut self.filter_text, "text");
                        ui.toggle_value(&mut self.filter_image, "image");
                        ui.toggle_value(&mut self.filter_audio, "audio");
                        ui.separator();
                        ui.label("Output");
                        ui.toggle_value(&mut self.filter_out_text, "text");
                        ui.toggle_value(&mut self.filter_out_image, "image");
                        ui.toggle_value(&mut self.filter_out_audio, "audio");
                        ui.separator();
                        ui.toggle_value(&mut self.filter_free, "free");
                        ui.separator();
                        ui.label("Sort");
                        let sort_label = match self.model_sort {
                            ModelSort::NameAsc => "Name ⬆",
                            ModelSort::NameDesc => "Name ⬇",
                            ModelSort::PriceAsc => "Price ⬆",
                            ModelSort::PriceDesc => "Price ⬇",
                        };
                        ComboBox::from_id_salt("model_sort_combo")
                            .selected_text(sort_label)
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut self.model_sort,
                                    ModelSort::NameAsc,
                                    "Name ⬆",
                                );
                                ui.selectable_value(
                                    &mut self.model_sort,
                                    ModelSort::NameDesc,
                                    "Name ⬇",
                                );
                                ui.selectable_value(
                                    &mut self.model_sort,
                                    ModelSort::PriceAsc,
                                    "Price ⬆",
                                );
                                ui.selectable_value(
                                    &mut self.model_sort,
                                    ModelSort::PriceDesc,
                                    "Price ⬇",
                                );
                            });
                        if ui.button("Refresh").clicked() {
                            let settings = self.settings.clone();
                            let api_key =
                                env_or(&settings.openrouter_api_key, "OPENROUTER_API_KEY");
                            self.models_loading = true;
                            self.models_error = None;
                            let tx = self.models_tx.clone();
                            tokio::spawn(async move {
                                let res = crate::ai::openai_compat::fetch_openrouter_models_json(
                                    api_key,
                                    Some("https://openrouter.ai/api/v1".into()),
                                )
                                .await
                                .map_err(|e| e.to_string());
                                let _ = tx.try_send(res);
                            });
                        }
                    });
                    ui.separator();
                    // Build filtered + sorted list of model refs
                    let mut shown: Vec<&OpenRouterModel> = self
                        .models_json
                        .iter()
                        .filter(|m| {
                            self.models_filter.is_empty()
                                || m.id
                                    .to_lowercase()
                                    .contains(&self.models_filter.to_lowercase())
                        })
                        .collect();
                    // Apply modality filters if any are enabled
                    if self.filter_text || self.filter_image || self.filter_audio || self.filter_out_text || self.filter_out_image || self.filter_out_audio {
                        shown.retain(|m| {
                            let arch = m.architecture.as_ref();
                            let in_mods = arch.map(|a| a.input_modalities.clone()).unwrap_or_default();
                            let out_mods = arch.map(|a| a.output_modalities.clone()).unwrap_or_default();
                            let mut ok = true;
                            if self.filter_text { ok &= in_mods.iter().any(|v| v.contains("text")); }
                            if self.filter_image { ok &= in_mods.iter().any(|v| v.contains("image")); }
                            if self.filter_audio { ok &= in_mods.iter().any(|v| v.contains("audio")); }
                            if self.filter_out_text { ok &= out_mods.iter().any(|v| v.contains("text")); }
                            if self.filter_out_image { ok &= out_mods.iter().any(|v| v.contains("image")); }
                            if self.filter_out_audio { ok &= out_mods.iter().any(|v| v.contains("audio")); }
                            ok
                        });
                    }
                    // Apply free filter naive heuristic: pricing.prompt/completion empty or starts with $0
                    if self.filter_free {
                        shown.retain(|m| {
                            if let Some(p) = &m.pricing {
                                let pp = p.prompt.as_deref().unwrap_or("");
                                let cp = p.completion.as_deref().unwrap_or("");
                                (pp.is_empty() || pp.trim_start_matches('$').starts_with('0'))
                                    && (cp.is_empty() || cp.trim_start_matches('$').starts_with('0'))
                            } else {
                                true
                            }
                        });
                    }
                    // helper to parse numeric price (prompt) as f64
                    let parse_price = |s: &str| -> Option<f64> {
                        if s.is_empty() {
                            return None;
                        }
                        let trimmed = s.trim().trim_start_matches('$');
                        let mut num_str = String::new();
                        for ch in trimmed.chars() {
                            if ch.is_ascii_digit() || ch == '.' {
                                num_str.push(ch);
                            } else {
                                break;
                            }
                        }
                        if num_str.is_empty() {
                            None
                        } else {
                            num_str.parse::<f64>().ok()
                        }
                    };
                    match self.model_sort {
                        ModelSort::NameAsc => shown.sort_by(|a, b| a.id.cmp(&b.id)),
                        ModelSort::NameDesc => shown.sort_by(|a, b| b.id.cmp(&a.id)),
                        ModelSort::PriceAsc => shown.sort_by(|a, b| {
                            let ap = a
                                .pricing
                                .as_ref()
                                .and_then(|p| p.prompt.as_deref())
                                .and_then(parse_price)
                                .unwrap_or(f64::INFINITY);
                            let bp = b
                                .pricing
                                .as_ref()
                                .and_then(|p| p.prompt.as_deref())
                                .and_then(parse_price)
                                .unwrap_or(f64::INFINITY);
                            ap.partial_cmp(&bp).unwrap_or(std::cmp::Ordering::Equal)
                        }),
                        ModelSort::PriceDesc => shown.sort_by(|a, b| {
                            let ap = a
                                .pricing
                                .as_ref()
                                .and_then(|p| p.prompt.as_deref())
                                .and_then(parse_price)
                                .unwrap_or(-1.0);
                            let bp = b
                                .pricing
                                .as_ref()
                                .and_then(|p| p.prompt.as_deref())
                                .and_then(parse_price)
                                .unwrap_or(-1.0);
                            bp.partial_cmp(&ap).unwrap_or(std::cmp::Ordering::Equal)
                        }),
                    }
                    if !self.models_loading {
                        ui.label(
                            RichText::new(format!(
                                "{} models ({} shown)",
                                self.models_json.len(),
                                shown.len()
                            ))
                            .weak(),
                        );
                    }
                    if self.models_loading {
                        ui.label(RichText::new("Loading models…").weak());
                    }
                    if let Some(err) = &self.models_error {
                        ui.colored_label(ui.visuals().warn_fg_color, format!("Error: {err}"));
                    }
                    ScrollArea::vertical().show(ui, |ui| {
                        for model in shown {
                            let id = &model.id;
                            let is_open = self.models_open.contains(id);
                            let toggle = if is_open { "⏷" } else { "▶" };
                            let arch = model.architecture.as_ref();
                            let mut icons = String::new();
                            if let Some(a) = arch {
                                if a.input_modalities.iter().any(|m| m.contains("text")) {
                                    icons.push_str("🖹 ");
                                }
                                if a.input_modalities.iter().any(|m| m.contains("image")) {
                                    icons.push_str("🖼 ");
                                }
                                if a.input_modalities.iter().any(|m| m.contains("audio")) {
                                    icons.push_str("🎵 ");
                                }
                                if a.input_modalities.iter().any(|m| m.contains("video")) {
                                    icons.push_str("🎬 ");
                                }
                            }
                            let price_suffix = if let Some(pr) = &model.pricing {
                                let prompt_p = pr.prompt.as_deref().unwrap_or("");
                                let completion_p = pr.completion.as_deref().unwrap_or("");
                                if !prompt_p.is_empty() || !completion_p.is_empty() {
                                    let pp = if prompt_p.is_empty() { "?" } else { prompt_p };
                                    let cp = if completion_p.is_empty() {
                                        "?"
                                    } else {
                                        completion_p
                                    };
                                    format!(" (${}/{})", pp, cp)
                                } else {
                                    String::new()
                                }
                            } else {
                                String::new()
                            };
                            let label_text = format!("{toggle} {id}");
                            let prices = format!("{price_suffix} {icons}");
                            ui.horizontal(|ui| {
                                let btn = Button::selectable(is_open, label_text)
                                    .min_size([ui.available_width() / 1.1, 15.].into())
                                    .right_text(
                                        RichText::new(prices).color(ui.style().visuals.error_fg_color),
                                    )
                                    .ui(ui);
                                if btn.clicked() {
                                    if is_open {
                                        self.models_open.remove(id);
                                    } else {
                                        self.models_open.insert(id.clone());
                                    }
                                }
                                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                    if ui
                                        .button("Use")
                                        .on_hover_text("Set as current model")
                                        .clicked()
                                    {
                                        self.model_override = Some(id.clone());
                                        self.model_name = id.clone();
                                        let mut settings = self.settings.clone();
                                        settings.openai_default_model = Some(id.clone());
                                        settings.push_recent_model(&id);
                                        crate::database::settings::save_settings(&settings);
                                        self.settings = settings;
                                        // Create a new chat session named after the model and switch to it
                                        let model_title = id.clone();
                                        let tx_clone = self.new_session_tx.clone();
                                        tokio::spawn(async move {
                                            let id = crate::database::assistant_chat::create_session(&model_title).await.ok();
                                            let _ = tx_clone.try_send((id, model_title));
                                        });
                                    }
                                    if ui.button("Copy").on_hover_text("Copy model id").clicked() {
                                        ui.ctx().copy_text(id.clone());
                                    }
                                });
                            });
                            if is_open {
                                ui.add_space(4.0);
                                if let Ok(val) = serde_json::to_value(model) {
                                    let style = JsonTreeStyle::new()
                                        .abbreviate_root(true)
                                        .toggle_buttons_state(ToggleButtonsState::VisibleDisabled)
                                        .visuals(JsonTreeVisuals {
                                            bool_color: Color32::LIGHT_BLUE,
                                            ..Default::default()
                                        });
                                    JsonTree::new(format!("model-tree-{}", id), &val)
                                        .style(style)
                                        .default_expand(DefaultExpand::All)
                                        .show(ui);
                                } else {
                                    ui.colored_label(
                                        ui.visuals().error_fg_color,
                                        "(failed to serialize model)",
                                    );
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
        if paths.is_empty()
            && !explorer.current_thumb.path.is_empty()
            && !self.suppress_selected_attachment
        {
            paths.push(explorer.current_thumb.path.clone());
        }
        let user_atts = if paths.is_empty() {
            None
        } else {
            Some(paths.clone())
        };
        self.messages.push(ChatMessage {
            role: ChatRole::User,
            content: prompt.clone(),
            attachments: user_atts.clone(),
            created: Some(chrono::Utc::now().into()),
        });
        self.messages.push(ChatMessage {
            role: ChatRole::Assistant,
            content: String::new(),
            attachments: None,
            created: Some(chrono::Utc::now().into()),
        });
        let chat_tx = self.chat_tx.clone();
        self.chat_streaming = true;
        let tx_updates = explorer.viewer.ai_update_tx.clone();
        let settings = self.settings.clone();
        let provider = settings
            .ai_chat_provider
            .clone()
            .unwrap_or_else(|| "local-joycaption".into());
        let use_cloud = provider != "local-joycaption";
        crate::ui::status::ASSIST_STATUS.set_state(
            crate::ui::status::StatusState::Running,
            if use_cloud {
                "Cloud request"
            } else {
                "Local request"
            },
        );
        crate::ui::status::ASSIST_STATUS.set_model(&provider);
        // Slash commands: /ping, /search, /describe
        let is_command = prompt_trim.starts_with('/');
        if is_command {
            crate::ui::status::ASSIST_STATUS
                .set_state(crate::ui::status::StatusState::Running, "MCP command");
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
                    "ping" => match crate::ai::mcp::ping_tool(arg.clone()).await {
                        Ok(echo) => format!("pong: {}", echo),
                        Err(e) => format!("Ping failed: {e}"),
                    },
                    "search" => {
                        match crate::ai::mcp::media_search_tool(arg.clone(), Some(32)).await {
                            Ok(hits) => {
                                // Send results to the explorer panel
                                let mut results = Vec::new();
                                for h in hits.iter() {
                                    let thumb = crate::Thumbnail::get_thumbnail_by_path(&h.path)
                                        .await
                                        .unwrap_or(None)
                                        .unwrap_or_default();
                                    results.push(crate::ui::file_table::SimilarResult {
                                        thumb,
                                        created: None,
                                        updated: None,
                                        similarity_score: None,
                                        clip_similarity_score: Some(h.score),
                                    });
                                }
                                let _ = tx_updates.try_send(
                                    crate::ui::file_table::AIUpdate::SimilarResults {
                                        origin_path: format!("mcp:media.search:{}", arg),
                                        results,
                                    },
                                );
                                format!(
                                    "Found {} similar items for '{}'. See results panel.",
                                    hits.len(),
                                    arg
                                )
                            }
                            Err(e) => format!("Search failed: {e}"),
                        }
                    }
                    "describe" => {
                        // Use explicit argument or fallback to the selected item snapshot
                        let chosen = if !arg.is_empty() {
                            arg.clone()
                        } else {
                            selected_path_snapshot.clone()
                        };
                        if chosen.is_empty() {
                            "No file selected or path specified.".to_string()
                        } else {
                            match crate::ai::mcp::media_describe_tool(chosen.clone()).await {
                                Ok(resp) => format!(
                                    "Caption: {}\nCategory: {}\nTags: {}",
                                    resp.caption,
                                    resp.category,
                                    resp.tags.join(", ")
                                ),
                                Err(e) => format!("Describe failed: {e}"),
                            }
                        }
                    }
                    _ => format!("Unknown command: /{}", cmd),
                };
                let _ = chat_tx.try_send(reply.clone());
                // Persist assistant reply
                if let Some(id) = session_id_for_async {
                    let _ = crate::database::assistant_chat::append_message(
                        &id,
                        "assistant",
                        &reply,
                        None,
                    )
                    .await;
                }
                crate::ui::status::ASSIST_STATUS
                    .set_state(crate::ui::status::StatusState::Idle, "Idle");
            });
            self.assistant_task = Some(handle);
            // Not streaming for commands
            self.chat_streaming = false;
            // Persist user prompt is handled below; then early return to avoid cloud/local flow
        } else if use_cloud {
            let model = self.model_override.clone().unwrap_or_else(|| {
                settings
                    .openai_default_model
                    .clone()
                    .unwrap_or_else(|| "gpt-5-mini".into())
            });
            let (api_key, base_url, org) = match provider.as_str() {
                "openai" => (
                    env_or(&settings.openai_api_key, "OPENAI_API_KEY"),
                    settings.openai_base_url.clone(),
                    settings.openai_organization.clone(),
                ),
                "grok" => (
                    env_or(&settings.grok_api_key, "GROK_API_KEY"),
                    settings.openai_base_url.clone(),
                    None,
                ),
                "gemini" => (
                    env_or(&settings.gemini_api_key, "GEMINI_API_KEY"),
                    settings.openai_base_url.clone(),
                    None,
                ),
                "groq" => (
                    env_or(&settings.groq_api_key, "GROQ_API_KEY"),
                    settings.openai_base_url.clone(),
                    None,
                ),
                "openrouter" => (
                    env_or(&settings.openrouter_api_key, "OPENROUTER_API_KEY"),
                    Some("https://openrouter.ai/api/v1".into()),
                    None,
                ),
                _ => (
                    env_or(&settings.openai_api_key, "OPENAI_API_KEY"),
                    settings.openai_base_url.clone(),
                    None,
                ),
            };
            let txu = tx_updates.clone();
            let provider_clone = provider.clone();
            let path_for_updates = paths.get(0).cloned().unwrap_or_default();
            let session_id_for_async = self.session_ids.get(self.active_session).cloned();
            let temp = self.temp;
            let prompt_owned = prompt.clone();
            let paths_for_async = paths.clone();
            // If file read fails later, we'll try to fetch DB thumbnail on-demand.
            let model_id_for_usage = model.clone();
            let handle = tokio::spawn(async move {
                let cfg = crate::ai::openai_compat::ProviderConfig {
                    provider: provider_clone,
                    api_key,
                    base_url,
                    model,
                    organization: org,
                    temperature: Some(temp),
                    zdr: true
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
                            if let Ok(Some(thumb)) =
                                crate::Thumbnail::get_thumbnail_by_path(p).await
                            {
                                if let Some(b64) = thumb.thumbnail_b64 {
                                    let cleaned =
                                        b64.split_once(',').map(|(_, v)| v).unwrap_or(&b64);
                                    if let Ok(bytes) = base64::engine::general_purpose::STANDARD
                                        .decode(cleaned.as_bytes())
                                    {
                                        imgs.push(bytes);
                                    }
                                }
                            }
                        }
                    }
                }
                let stream_res = crate::ai::openai_compat::stream_multimodal_reply(
                    cfg,
                    &transcript,
                    &imgs,
                    |tok| {
                        let _ = chat_tx.try_send(tok.to_string());
                    },
                )
                .await;
                match stream_res {
                    Ok(full) => {
                        // On successful assistant reply, record model usage as recently used
                        if let Some(mut s) = Some(settings.clone()) {
                            s.push_recent_model(&model_id_for_usage);
                            crate::database::settings::save_settings(&s);
                        }
                        if !path_for_updates.is_empty() {
                            let _ = txu.try_send(crate::ui::file_table::AIUpdate::Interim {
                                path: path_for_updates.clone(),
                                text: full.clone(),
                            });
                        }
                        if let Some(id) = session_id_for_async {
                            let _ = crate::database::assistant_chat::append_message(
                                &id,
                                "assistant",
                                &full,
                                None,
                            )
                            .await;
                        }
                        crate::ui::status::ASSIST_STATUS
                            .set_state(crate::ui::status::StatusState::Idle, "Idle");
                    }
                    Err(e) => {
                        let err_text = format!("Error: {e}");
                        log::error!("cloud assistant error: {e}");
                        let _ = chat_tx.try_send(err_text.clone());
                        if let Some(id) = session_id_for_async {
                            let _ = crate::database::assistant_chat::append_message(
                                &id,
                                "assistant",
                                &err_text,
                                None,
                            )
                            .await;
                        }
                        crate::ui::status::ASSIST_STATUS.set_error(e.to_string());
                    }
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
                                let _ = crate::ai::joycap::stream_describe_bytes_with_callback(
                                    bytes,
                                    &instruction,
                                    |tok| {
                                        let s = tok.to_string();
                                        let _ = tx_updates.try_send(
                                            crate::ui::file_table::AIUpdate::Interim {
                                                path: path.clone(),
                                                text: s.clone(),
                                            },
                                        );
                                        let _ = chat_tx2.try_send(s);
                                    },
                                )
                                .await;
                                // Mark the local model as recently used (use provider name)
                                let mut s = settings.clone();
                                let mname = s.openai_default_model.clone().unwrap_or_else(|| "local-joycaption".into());
                                s.push_recent_model(&mname);
                                crate::database::settings::save_settings(&s);
                                crate::ui::status::ASSIST_STATUS
                                    .set_state(crate::ui::status::StatusState::Idle, "Idle");
                            }
                            Err(e) => {
                                log::error!("read image failed: {e}");
                                crate::ui::status::ASSIST_STATUS.set_error(e.to_string());
                            }
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
                        if let Some(engine) = guard.as_mut() {
                            engine.embed_text(&query).ok()
                        } else {
                            None
                        }
                    };
                    if let Some(q) = q_vec_opt {
                        match crate::database::ClipEmbeddingRow::find_similar_by_embedding(
                            &q, 64, 128, 0,
                        )
                        .await
                        {
                            Ok(hits) => {
                                for hit in hits.into_iter() {
                                    let thumb = if let Some(t) = hit.thumb_ref {
                                        t
                                    } else {
                                        crate::Thumbnail::get_thumbnail_by_path(&hit.path)
                                            .await
                                            .unwrap_or(None)
                                            .unwrap_or_default()
                                    };
                                    results.push(crate::ui::file_table::SimilarResult {
                                        thumb,
                                        created: None,
                                        updated: None,
                                        similarity_score: None,
                                        clip_similarity_score: Some(hit.dist),
                                    });
                                }
                            }
                            Err(e) => log::error!("text search knn failed: {e}"),
                        }
                    }
                    let _ = tx_updates.try_send(crate::ui::file_table::AIUpdate::SimilarResults {
                        origin_path: format!("query:{query}"),
                        results,
                    });
                    // On successful local text search, treat the current default model as used
                    let mut s = settings.clone();
                    let mname = s.openai_default_model.clone().unwrap_or_else(|| "local-joycaption".into());
                    s.push_recent_model(&mname);
                    crate::database::settings::save_settings(&s);
                    crate::ui::status::ASSIST_STATUS
                        .set_state(crate::ui::status::StatusState::Idle, "Idle");
                });
                self.assistant_task = Some(handle);
            }
        }
        if let Some((title, store)) = self.sessions.get_mut(self.active_session) {
            *title = self
                .messages
                .iter()
                .rev()
                .find(|m| matches!(m.role, ChatRole::User))
                .map(|m| m.content.chars().take(32).collect::<String>())
                .unwrap_or_else(|| format!("Chat {}", self.active_session + 1));
            *store = self.messages.clone();
        }
        // Persist messages
        if let Some(id) = self.session_ids.get(self.active_session).cloned() {
            let user_attachments = if paths.is_empty() {
                None
            } else {
                Some(paths.clone())
            };
            // Spawn user message insert
            let prompt_copy = prompt.clone();
            tokio::spawn(async move {
                let _ = crate::database::assistant_chat::append_message(
                    &id,
                    "user",
                    &prompt_copy,
                    user_attachments,
                )
                .await;
            });
        }
        self.prompt.clear();
        // Clear attachments after sending so next message starts fresh
        self.attachments.clear();
        // Reset per-message suppression so the selected image shows up again for next prompt by default
        self.suppress_selected_attachment = false;
    }

    fn render_bubble(&mut self, ui: &mut Ui, msg: &ChatMessage, created_clone: Option<surrealdb::sql::Datetime>, idx: usize, content_clone: String) {
        let is_user = matches!(msg.role, ChatRole::User);
        let rounding = 8.0;
        let rnd = egui::CornerRadius {
            ne: if is_user { 0 } else { rounding as u8 },
            nw: if is_user { rounding as u8 } else { 0 },
            se: rounding as u8,
            sw: rounding as u8,
        };
        let style = ui.style().clone();
        let base_fill = if is_user {
            style.visuals.widgets.active.bg_fill
        } else {
            style.visuals.widgets.active.weak_bg_fill
        };
        let (fill, stroke, shadow) = (
            base_fill,
            style.visuals.widgets.open.bg_stroke,
            egui::epaint::Shadow::default(),
        );

        let outer_margin = egui::Margin { left: 1, right: 1, top: 4, bottom: 1 };
        let inner_margin = egui::Margin { left: 6, right: 6, top: 6, bottom: 6 };
        Frame::new()
            .corner_radius(rnd)
            .inner_margin(inner_margin)
            .outer_margin(outer_margin)
            .fill(fill)
            .shadow(shadow)
            .stroke(stroke)
            .show(ui, |ui| {
                // Header row: name + actions
                ui.horizontal(|ui| {
                    let who = if is_user { "You" } else { "Assistant" };
                    let layout = if is_user {
                        Layout::left_to_right(Align::Center)
                    } else {
                        Layout::right_to_left(Align::Center)
                    };
                    ui.strong(who);
                    ui.with_layout(layout, |ui| {
                        if let Some(ts) = created_clone.clone() {
                            let s = chrono::DateTime::<chrono::Utc>::from(ts).with_timezone(&Local).format("%m/%d @ %I:%M%p").to_string();
                            ui.label(RichText::new(s).weak());
                            ui.separator();
                        }
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
                        if let Some(atts) = &msg.attachments {
                            if !atts.is_empty() {
                                ui.add_space(6.0);
                                ui.horizontal_wrapped(|ui| {
                                    for p in atts.iter() {
                                        if let Some(tex) = self.thumb_cache.get(p) {
                                            ui.image((tex.id(), egui::vec2(40.0, 40.0))).on_hover_text(p);
                                        } else {
                                            if ui.button(RichText::new("[att]").small()).on_hover_text(p).clicked() {
                                                self.request_db_thumb(p);
                                            }
                                        }
                                    }
                                });
                            }
                        }
                    });
                });
                ui.add_space(2.0);
                // Header row: icon/privacy and timestamp could go here if needed
                // Content frame similar to provided sample
                Frame::new()
                    .fill(Color32::from_rgb(10, 10, 12))
                    .stroke(style.visuals.widgets.inactive.bg_stroke)
                    .outer_margin(egui::Margin { top: 3, ..Default::default() })
                    .inner_margin(egui::Margin { left: 10, right: 10, top: 6, bottom: 6 })
                    .corner_radius(rnd)
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        self.render_message_inner(ui, &msg.content);
                    });
            });
    }

    fn render_bubble_preview(&mut self, ui: &mut Ui, content: &str) {
        let bg = Color32::from_gray(28);
        let stroke = Stroke::new(1.0, Color32::DARK_GRAY);
        let preview = {
            let mut s = String::new();
            for (i, line) in content.lines().take(10).enumerate() {
                if i > 0 {
                    s.push('\n');
                }
                s.push_str(line);
            }
            if content.lines().count() > 10 {
                s.push_str("\n…");
            }
            s
        };
        Frame::new()
            .fill(bg)
            .stroke(stroke)
            .inner_margin(egui::Margin {
                left: 8,
                right: 8,
                top: 6,
                bottom: 6,
            })
            .show(ui, |ui| {
                ui.label(preview);
            });
    }

    fn render_message_inner(&self, ui: &mut Ui, text: &str) {
        // Parse fenced code blocks with optional language: ```lang\ncode...\n```
        #[derive(Debug)]
        enum Seg {
            Text(String),
            Code { lang: String, code: String },
        }
        let mut segs: Vec<Seg> = Vec::new();
        let mut in_code = false;
        let mut cur_lang = String::new();
        let mut buf = String::new();
        for line in text.lines() {
            let trimmed = line.trim_end_matches('\r');
            if trimmed.starts_with("```") {
                if in_code {
                    // close code block
                    segs.push(Seg::Code {
                        lang: cur_lang.clone(),
                        code: buf.trim_end().to_string(),
                    });
                    buf.clear();
                    cur_lang.clear();
                    in_code = false;
                } else {
                    // flush text
                    if !buf.is_empty() {
                        segs.push(Seg::Text(buf.clone()));
                        buf.clear();
                    }
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
            if in_code {
                segs.push(Seg::Code {
                    lang: cur_lang,
                    code: buf.trim_end().to_string(),
                });
            } else {
                segs.push(Seg::Text(buf.clone()));
            }
        }

        for seg in segs {
            match seg {
                Seg::Text(s) => {
                    let s = s.trim();
                    if !s.is_empty() {
                        ui.label(s);
                    }
                }
                Seg::Code { lang, code } => {
                    // Header with copy + language
                    ui.horizontal(|ui| {
                        if ui.button("Copy").clicked() {
                            ui.ctx().copy_text(code.clone());
                        }
                        if !lang.is_empty() {
                            ui.label(RichText::new(lang.clone()).weak());
                        } else {
                            ui.label(RichText::new("code").weak());
                        }
                    });
                    // Syntax-highlighted, non-interactive code block
                    ui.group(|ui| {
                        let theme = CodeTheme::from_style(ui.style().as_ref());
                        let style_clone = ui.style().clone();
                        let ctx_clone = ui.ctx().clone();
                        let lang_clone = lang.clone();
                        let mut layouter =
                            move |ui: &Ui, text: &dyn egui::TextBuffer, wrap_width: f32| {
                                let s: &str = text.as_str();
                                let mut job =
                                    highlight(&ctx_clone, &style_clone, &theme, &lang_clone, s);
                                job.wrap.max_width = wrap_width;
                                ui.fonts_mut(|f| f.layout_job(job))
                            };
                        let rows = (code.lines().count().max(3) as f32).min(18.0);
                        let mut code_mut = code.clone();
                        ui.add(
                            TextEdit::multiline(&mut code_mut)
                                .desired_rows(rows as usize)
                                .desired_width(ui.available_width())
                                .font(FontId::monospace(14.0))
                                .interactive(false)
                                .layouter(&mut layouter),
                        );
                    });
                }
            }
        }
    }

    fn try_thumbnail(&mut self, _ui: &mut Ui, path: &str) -> Option<egui::TextureId> {
        if let Some(h) = self.thumb_cache.get(path) { return Some(h.id()); }
        // If not cached, request from DB asynchronously and return None for now
        self.request_db_thumb(path);
        None
    }
}

impl AssistantPanel {
    fn request_db_thumb(&mut self, path: &str) {
        if self.thumb_cache.contains_key(path) { return; }
        if self.pending_thumb_fetch.contains(path) { return; }
        self.pending_thumb_fetch.insert(path.to_string());
        let path_s = path.to_string();
        let tx = self.db_thumb_tx.clone();
        tokio::spawn(async move {
            let res = match crate::Thumbnail::get_thumbnail_by_path(&path_s).await {
                Ok(Some(t)) => {
                    if let Some(b64) = t.thumbnail_b64 {
                        match base64::engine::general_purpose::STANDARD.decode(b64) {
                            Ok(bytes) => match image::load_from_memory(&bytes) {
                                Ok(img) => {
                                    let rgba = img.to_rgba8();
                                    let (w, h) = rgba.dimensions();
                                    Some((rgba.into_raw(), w, h))
                                }
                                Err(_) => None,
                            },
                            Err(_) => None,
                        }
                    } else { None }
                }
                _ => None,
            };
            let _ = tx.try_send((path_s, res));
        });
    }
}
