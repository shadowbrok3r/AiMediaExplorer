use crate::ui::dialogs;
use crate::status::GlobalStatusIndicator;
use eframe::egui::*;

impl crate::app::SmartMediaContext {
    pub fn receive(&mut self, ctx: &eframe::egui::Context) {
        Area::new(Id::unique("toasts_overlay"))
            .fixed_pos(Pos2::ZERO)
            .show(ctx, |ui| {
                self.toasts.show(ui);
            });

        if self.first_run {
            egui_extras::install_image_loaders(ctx);

            let db_path = crate::database::get_db_path();
            if std::path::Path::new(&db_path).exists() {
                crate::ui::wizard::mark_location_done_if_present();
                self.connect_database();
            } else {
                self.wizard.open_first_run();
                log::info!("No database found at {db_path}, showing the first-run wizard");
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
            self.toasts.add(app_toast(kind, msg));
        }

        if crate::ui::settings::open_key_pressed(ctx) {
            crate::command::send(crate::command::AppCommand::ShowSettings { section: None });
        }

        if crate::settings::store().take_load_event() {
            self.on_settings_loaded();
        }

        if let Ok(_) = self.db_ready_rx.try_recv() {
            self.db_ready = true;
            log::info!("DB is ready");
            crate::status::DB_STATUS.set_state(crate::status::StatusState::Idle, "Ready");
            let store = std::sync::Arc::clone(crate::settings::store());
            tokio::spawn(async move {
                if let Err(e) = store.reload().await {
                    log::error!("Settings could not be loaded: {e:#}");
                }
            });
        }

        // Poll global AI engine readiness (cheap lock attempt)
        if !self.ai_ready && self.ai_initializing {
            use std::sync::atomic::Ordering;
            let completed = crate::ai::GLOBAL_AI_ENGINE.index_completed.load(Ordering::Relaxed);
            let active = crate::ai::GLOBAL_AI_ENGINE.index_active.load(Ordering::Relaxed);
            if completed > 0 || active > 0 {
                self.ai_ready = true;
                self.ai_initializing = false;
                crate::status::VISION_STATUS.set_state(crate::status::StatusState::Idle, "Ready");
            }
        }

        crate::zc::tick(ctx);
        self.assistant.receive(ctx);
        crate::ui::wizard::show(ctx, &mut self.wizard, crate::database::is_connected());
        self.run_wizard_devtools(ctx);
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
        let (auto_indexing, enable_clip) = crate::settings::store().read(|s| (s.auto_indexing, s.auto_clip_embeddings));
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
    }

    /// Runs the buttons pressed in the Wizard devtools window and publishes its state.
    pub(crate) fn run_wizard_devtools(&mut self, ctx: &Context) {
        use crate::ui::devtools::wizard::{Request, publish, take_requests};
        for request in take_requests(ctx) {
            match request {
                Request::Close => self.wizard.close(),
                Request::SimulateFirstRun => {
                    crate::settings::ui_prefs::update(|p| p.checklist = Default::default());
                    self.wizard.open_first_run();
                }
            }
        }
        publish(ctx, &self.wizard);
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
            ViewportBuilder::default().with_title("AI Assistant").with_inner_size([760.0, 900.0]),
            |ui, _class| {
                if ui.ctx().input(|i| i.viewport().close_requested()) {
                    close_requested = true;
                }
                if assistant.ui(ui, explorer).dock {
                    close_requested = true;
                }
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

/// A toast that stays until it is clicked and draws no progress bar.
fn app_toast(kind: egui_toast::ToastKind, text: impl Into<String>) -> egui_toast::Toast {
    egui_toast::Toast { kind, text: eframe::egui::RichText::new(text.into()).into(), options: egui_toast::ToastOptions::default().show_progress(false), ..Default::default() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::{AttentionKind, TaskKind, TaskState};

    // The delay before the next frame that the last of thirty frames showing `toast` asks for.
    fn repaint_delay_with(toast: egui_toast::Toast) -> std::time::Duration {
        let ctx = crate::ui::test_support::test_ctx();
        let mut toasts = egui_toast::Toasts::new().anchor(Align2::RIGHT_TOP, (-10.0, 10.0));
        toasts.add(toast);
        let mut delay = std::time::Duration::ZERO;
        for frame in 0..30 {
            let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(800.0, 600.0))), time: Some(frame as f64 * 0.1), ..Default::default() };
            let out = ctx.run_ui(input, |ui| toasts.show(ui));
            delay = out.viewport_output.get(&ViewportId::ROOT).map(|v| v.repaint_delay).unwrap();
            out.drop_without_applying_deltas();
        }
        delay
    }

    #[test]
    fn a_waiting_toast_asks_for_no_frames_until_it_is_clicked() {
        for kind in [egui_toast::ToastKind::Info, egui_toast::ToastKind::Success, egui_toast::ToastKind::Error] {
            let delay = repaint_delay_with(app_toast(kind, "done"));
            assert!(delay > std::time::Duration::from_secs(60), "{kind:?} keeps the application drawing: {delay:?}");
        }
    }

    #[test]
    fn the_default_toast_keeps_every_frame_coming_which_is_why_the_application_does_not_use_it() {
        let default = egui_toast::Toast { text: "x".into(), ..Default::default() };
        assert!(repaint_delay_with(default) < std::time::Duration::from_millis(100));
    }

    #[test]
    fn an_application_toast_keeps_its_kind_and_text() {
        let toast = app_toast(egui_toast::ToastKind::Warning, "careful");
        assert_eq!(toast.kind, egui_toast::ToastKind::Warning);
        assert_eq!(toast.text.text(), "careful");
        assert!(!toast.options.show_progress);
    }

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
