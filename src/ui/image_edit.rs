//! The Image Edit tab: prompt presets, a tracked edit job with progress and cancel, a compare view of the source and the result, session history, and saving without touching the original unless asked.

mod backend;
mod debug;
mod job;
pub mod model;
mod qwen;
mod save;
#[cfg(test)]
mod tests;
mod view;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::SystemTime;

use crossbeam::channel::{Receiver, Sender, unbounded};
use eframe::egui::{ColorImage, Context, TextureHandle, TextureOptions};

use crate::ai::mcp::ctx::DbSource;
use crate::command::{AppCommand, ToastLevel};
use crate::ui::file_table::thumbs::{ScopeId, ThumbService, ThumbView, Tier, can_preview};

pub use backend::{EditBackend, EditRequest, RunCtl};
pub use job::{Decoded, Msg};
pub use model::{CompareMode, EditError, ErrorKind, ModelSettings, Params, Phase};

/// Results kept for the session.
const HISTORY_MAX: usize = 12;

/// One finished edit of this session.
pub struct EditEntry {
    pub id: u64,
    pub source: String,
    pub prompt: String,
    pub png: Arc<[u8]>,
    /// Pixels of the saved image.
    pub full: [u32; 2],
    pub texture: TextureHandle,
    pub created: SystemTime,
    /// The file the result was written to, once saved.
    pub saved: Option<String>,
    pub in_library: bool,
}

/// State of the GGUF download.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GgufState {
    pub fetching: bool,
    pub status: Option<String>,
}

pub struct ImageEditPanel {
    pub current_path: Option<String>,
    pub params: Params,
    pub settings: ModelSettings,
    phase: Phase,
    cancel_flag: Option<Arc<AtomicBool>>,
    backend: Arc<dyn EditBackend>,
    history: Vec<EditEntry>,
    shown: Option<u64>,
    live: Option<TextureHandle>,
    compare: CompareMode,
    swipe: f32,
    flipped: bool,
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
    thumbs: Arc<ThumbService>,
    thumb_scope: ScopeId,
    db: DbSource,
    next_id: u64,
    notice: Option<(bool, String)>,
    gguf: GgufState,
    model_busy: bool,
    model_error: Option<EditError>,
    confirm_overwrite: Option<u64>,
    collections: Vec<String>,
    collection_pick: Option<String>,
    new_collection: String,
    collections_loaded: bool,
    debug_backend: bool,
}

impl Default for ImageEditPanel {
    fn default() -> Self {
        let mut panel = Self::with_backend(Arc::new(qwen::QwenBackend::default()), DbSource::Global, crate::ui::file_table::thumbs::service());
        if debug::requested_by_env() {
            panel.use_debug_backend(true);
        }
        panel
    }
}

fn toast(level: ToastLevel, text: impl Into<String>) {
    crate::command::send(AppCommand::Toast { kind: level, text: text.into() });
}

impl ImageEditPanel {
    /// A panel over a given model backend, database and thumbnail service, for tests and tools.
    pub fn with_backend(backend: Arc<dyn EditBackend>, db: DbSource, thumbs: Arc<ThumbService>) -> Self {
        let (tx, rx) = unbounded();
        Self {
            current_path: None,
            params: Params::default(),
            settings: ModelSettings::default(),
            phase: Phase::Idle,
            cancel_flag: None,
            backend,
            history: Vec::new(),
            shown: None,
            live: None,
            compare: CompareMode::default(),
            swipe: 0.5,
            flipped: false,
            tx,
            rx,
            thumbs,
            thumb_scope: crate::ui::file_table::thumbs::alloc_scope(),
            db,
            next_id: 1,
            notice: None,
            gguf: GgufState::default(),
            model_busy: false,
            model_error: None,
            confirm_overwrite: None,
            collections: Vec::new(),
            collection_pick: None,
            new_collection: String::new(),
            collections_loaded: false,
            debug_backend: false,
        }
    }

    /// Switches between the Qwen pipeline and the test backend that edits in a moment; ignored while an edit runs.
    pub fn use_debug_backend(&mut self, on: bool) {
        if self.phase.is_busy() || self.debug_backend == on {
            return;
        }
        self.backend = if on { Arc::new(debug::DebugBackend::default()) } else { Arc::new(qwen::QwenBackend::default()) };
        self.debug_backend = on;
        self.model_error = None;
    }

    pub fn uses_debug_backend(&self) -> bool {
        self.debug_backend
    }

    /// Points the tab at `path`; the results of earlier edits stay in the history.
    pub fn open_with_path(&mut self, path: &str) {
        if self.current_path.as_deref() != Some(path) {
            self.shown = None;
            self.live = None;
            self.flipped = false;
        }
        self.current_path = Some(path.to_string());
    }

    /// Stops the edit that was started last; `None` before the first edit.
    pub fn cancel_handle(&self) -> Option<crate::tasks::CancelHandle> {
        self.cancel_flag.clone().map(crate::tasks::CancelHandle::from_flag)
    }

    pub fn phase(&self) -> &Phase {
        &self.phase
    }

    pub fn history(&self) -> &[EditEntry] {
        &self.history
    }

    pub fn is_busy(&self) -> bool {
        self.phase.is_busy()
    }

    /// Sender that jobs report to; tests use it to feed messages.
    #[cfg(test)]
    pub fn sender(&self) -> Sender<Msg> {
        self.tx.clone()
    }

    fn note(&mut self, ok: bool, text: impl Into<String>) {
        self.notice = Some((ok, text.into()));
    }

    fn spawn<F: std::future::Future<Output = ()> + Send + 'static>(&self, fut: F) -> bool {
        match tokio::runtime::Handle::try_current() {
            Ok(rt) => {
                rt.spawn(fut);
                true
            }
            Err(_) => false,
        }
    }

    /// Why an edit cannot start now, if it cannot.
    pub fn blocked(&self) -> Option<&'static str> {
        if self.phase.is_busy() {
            return Some("An edit is already running");
        }
        if self.current_path.is_none() {
            return Some("Choose an image first");
        }
        self.params.problem()
    }

    /// Starts the edit of the source image; the model is loaded first when it is not.
    pub fn start(&mut self) -> Result<(), String> {
        if let Some(why) = self.blocked() {
            return Err(why.to_string());
        }
        let source = std::path::PathBuf::from(self.current_path.clone().unwrap_or_default());
        let cancel = Arc::new(AtomicBool::new(false));
        let req = EditRequest { source, params: self.params.clone(), settings: self.settings.clone() };
        let (backend, tx, flag) = (self.backend.clone(), self.tx.clone(), cancel.clone());
        if !self.spawn(job::run_job(backend, req, flag, tx)) {
            return Err("no async runtime is available".to_string());
        }
        self.phase = if self.backend.is_loaded() { Phase::Running { done: 0, total: self.params.steps as u64, detail: "Starting".to_string() } } else { Phase::LoadingModel };
        self.cancel_flag = Some(cancel);
        self.model_error = None;
        self.notice = None;
        self.live = None;
        Ok(())
    }

    /// Asks the running edit to stop at its next step.
    pub fn cancel(&mut self) {
        if let Some(flag) = &self.cancel_flag {
            flag.store(true, Ordering::Relaxed);
        }
        if self.phase.is_busy() {
            self.phase = Phase::Cancelling;
        }
    }

    /// Runs the failed edit again with the lower precision; for out-of-memory errors.
    pub fn retry_lower_precision(&mut self) -> Result<(), String> {
        self.params.prefer_full_precision = false;
        self.params.steps = self.params.steps.min(20);
        self.phase = Phase::Idle;
        self.start()
    }

    /// Takes in what the background jobs reported.
    pub fn pump(&mut self, ctx: &Context) {
        let msgs: Vec<Msg> = self.rx.try_iter().collect();
        for msg in msgs {
            self.ingest(msg, ctx);
        }
    }

    /// Applies one report of a job.
    pub fn ingest(&mut self, msg: Msg, ctx: &Context) {
        match msg {
            Msg::Phase(p) => {
                if self.phase != Phase::Cancelling {
                    self.phase = p;
                }
            }
            Msg::Preview(decoded) => self.live = Some(upload(ctx, "image_edit_live", &decoded)),
            Msg::Finished { png, decoded, source, prompt } => self.finish(png, decoded, source, prompt, ctx),
            Msg::Failed(e) => {
                self.live = None;
                toast(ToastLevel::Error, format!("Image edit failed: {}", e.message));
                self.phase = Phase::Failed(e);
            }
            Msg::Cancelled => {
                self.live = None;
                self.phase = Phase::Cancelled;
            }
            Msg::ModelLoaded(result) => {
                self.model_busy = false;
                match result {
                    Ok(()) => {
                        self.model_error = None;
                        self.note(true, "The model is loaded");
                    }
                    Err(e) => {
                        toast(ToastLevel::Error, format!("Model load failed: {}", e.message));
                        self.model_error = Some(e);
                    }
                }
            }
            Msg::Gguf(result) => {
                self.gguf.fetching = false;
                match result {
                    Ok(path) => {
                        self.gguf.status = Some(format!("Ready: {path}"));
                        self.settings.gguf_path = Some(path);
                    }
                    Err(e) => self.gguf.status = Some(format!("Error: {e}")),
                }
            }
            Msg::Saved { id, result } => match result {
                Ok(path) => {
                    toast(ToastLevel::Success, format!("Saved {path}"));
                    if let Some(e) = self.history.iter_mut().find(|e| e.id == id) {
                        e.saved = Some(path.clone());
                    }
                    self.note(true, format!("Saved {path}"));
                }
                Err(e) => {
                    toast(ToastLevel::Error, format!("Save failed: {e}"));
                    self.note(false, format!("Save failed: {e}"));
                }
            },
            Msg::Library { id, result } => match result {
                Ok(text) => {
                    if let Some(e) = self.history.iter_mut().find(|e| e.id == id) {
                        e.in_library = true;
                    }
                    toast(ToastLevel::Success, text.clone());
                    self.note(true, text);
                    self.collections_loaded = false;
                }
                Err(e) => {
                    toast(ToastLevel::Error, format!("Adding to the library failed: {e}"));
                    self.note(false, format!("Adding to the library failed: {e}"));
                }
            },
            Msg::Collections(names) => {
                self.collections = names;
                self.collections_loaded = true;
            }
        }
    }

    fn finish(&mut self, png: Arc<[u8]>, decoded: Decoded, source: String, prompt: String, ctx: &Context) {
        let id = self.next_id;
        self.next_id += 1;
        let texture = upload(ctx, &format!("image_edit_result_{id}"), &decoded);
        self.history.push(EditEntry { id, source, prompt, png, full: decoded.full, texture, created: SystemTime::now(), saved: None, in_library: false });
        if self.history.len() > HISTORY_MAX {
            let dropped = self.history.remove(0);
            if self.shown == Some(dropped.id) {
                self.shown = None;
            }
        }
        self.shown = Some(id);
        self.live = None;
        self.flipped = false;
        self.phase = Phase::Done;
        toast(ToastLevel::Success, "Image edit finished");
    }

    /// The result on screen: the one picked in the history, else the newest for the source image.
    pub fn shown_entry(&self) -> Option<&EditEntry> {
        match self.shown {
            Some(id) => self.history.iter().find(|e| e.id == id),
            None => self.history.iter().rev().find(|e| Some(&e.source) == self.current_path.as_ref()),
        }
    }

    /// Shows a result of the history, with the source image it came from.
    pub fn show_entry(&mut self, id: u64) {
        if let Some(e) = self.history.iter().find(|e| e.id == id) {
            self.current_path = Some(e.source.clone());
            self.shown = Some(id);
            self.flipped = false;
        }
    }

    /// Drops a result from the history.
    pub fn remove_entry(&mut self, id: u64) {
        self.history.retain(|e| e.id != id);
        if self.shown == Some(id) {
            self.shown = None;
        }
        if self.confirm_overwrite == Some(id) {
            self.confirm_overwrite = None;
        }
    }

    /// Loads the model with the current settings, without starting an edit.
    pub fn load_model(&mut self) {
        if self.model_busy || self.phase.is_busy() {
            return;
        }
        let (backend, settings, params, tx) = (self.backend.clone(), self.settings.clone(), self.params.clone(), self.tx.clone());
        if self.spawn(async move {
            let result = backend.load(&settings, &params).await;
            let _ = tx.send(Msg::ModelLoaded(result));
            crate::command::request_repaint();
        }) {
            self.model_busy = true;
            self.model_error = None;
        }
    }

    /// Frees the memory of the model.
    pub fn unload_model(&mut self) {
        if self.phase.is_busy() {
            return;
        }
        let backend = self.backend.clone();
        self.spawn(async move { backend.unload().await });
    }

    /// Downloads the GGUF file named in the settings.
    pub fn fetch_gguf(&mut self) {
        if self.gguf.fetching || self.settings.gguf_repo.trim().is_empty() || self.settings.gguf_file.trim().is_empty() {
            return;
        }
        let (repo, file, tx) = (self.settings.gguf_repo.clone(), self.settings.gguf_file.clone(), self.tx.clone());
        if self.spawn(async move {
            let fetched = tokio::task::spawn_blocking(move || {
                let repo = crate::ai::hf::hf_model(&repo).map_err(|e| e.to_string())?;
                crate::ai::hf::hf_get_file(&repo, &file).map(|p| p.to_string_lossy().to_string()).map_err(|e| e.to_string())
            })
            .await
            .unwrap_or_else(|e| Err(e.to_string()));
            let _ = tx.send(Msg::Gguf(fetched));
            crate::command::request_repaint();
        }) {
            self.gguf = GgufState { fetching: true, status: Some("Downloading...".to_string()) };
        }
    }

    /// Writes the result next to the source under a new name.
    pub fn save_next_to_original(&mut self, id: u64) {
        let Some(e) = self.history.iter().find(|e| e.id == id) else { return };
        let (png, source, tx) = (e.png.clone(), e.source.clone(), self.tx.clone());
        self.spawn(async move {
            let result = tokio::task::spawn_blocking(move || save::save_next_to(&png, &source)).await.unwrap_or_else(|e| Err(e.to_string()));
            let _ = tx.send(Msg::Saved { id, result });
            crate::command::request_repaint();
        });
    }

    /// Writes the result to `path`, chosen by the user.
    pub fn save_to(&mut self, id: u64, path: std::path::PathBuf) {
        let Some(e) = self.history.iter().find(|e| e.id == id) else { return };
        let (png, tx) = (e.png.clone(), self.tx.clone());
        self.spawn(async move {
            let result = tokio::task::spawn_blocking(move || save::write_result(&png, &path)).await.unwrap_or_else(|e| Err(e.to_string()));
            let _ = tx.send(Msg::Saved { id, result });
            crate::command::request_repaint();
        });
    }

    /// Asks to replace the original; nothing is written until `confirm_replace`.
    pub fn ask_replace(&mut self, id: u64) {
        if self.history.iter().any(|e| e.id == id) {
            self.confirm_overwrite = Some(id);
        }
    }

    pub fn replace_target(&self) -> Option<(&EditEntry, &str)> {
        let id = self.confirm_overwrite?;
        self.history.iter().find(|e| e.id == id).map(|e| (e, e.source.as_str()))
    }

    /// Replaces the original file with the result that was asked about.
    pub fn confirm_replace(&mut self) {
        let Some(id) = self.confirm_overwrite.take() else { return };
        let Some(e) = self.history.iter().find(|e| e.id == id) else { return };
        let (png, source, tx) = (e.png.clone(), e.source.clone(), self.tx.clone());
        self.spawn(async move {
            let result = tokio::task::spawn_blocking(move || save::overwrite_original(&png, &source, &source)).await.unwrap_or_else(|e| Err(e.to_string()));
            let _ = tx.send(Msg::Saved { id, result });
            crate::command::request_repaint();
        });
    }

    pub fn cancel_replace(&mut self) {
        self.confirm_overwrite = None;
    }

    /// Adds the result to the library, saving it next to the original first when it has no file yet.
    pub fn add_to_library(&mut self, id: u64) {
        let Some(e) = self.history.iter().find(|e| e.id == id) else { return };
        let (png, source, saved, tx, db) = (e.png.clone(), e.source.clone(), e.saved.clone(), self.tx.clone(), self.db.clone());
        let collection = if self.new_collection.trim().is_empty() { self.collection_pick.clone() } else { Some(self.new_collection.trim().to_string()) };
        self.spawn(async move {
            let path = match saved {
                Some(p) => Ok(p),
                None => tokio::task::spawn_blocking(move || save::save_next_to(&png, &source)).await.unwrap_or_else(|e| Err(e.to_string())),
            };
            let result = match path {
                Ok(p) => {
                    let _ = tx.send(Msg::Saved { id, result: Ok(p.clone()) });
                    save::add_to_library(&db, &p, collection.as_deref()).await
                }
                Err(e) => Err(e),
            };
            let _ = tx.send(Msg::Library { id, result });
            crate::command::request_repaint();
        });
    }

    /// Reads the collection names for the choice, once.
    pub fn load_collections_if_needed(&mut self) {
        if self.collections_loaded {
            return;
        }
        self.collections_loaded = true;
        let (db, tx) = (self.db.clone(), self.tx.clone());
        self.spawn(async move {
            let names = save::collection_names(&db).await;
            let _ = tx.send(Msg::Collections(names));
            crate::command::request_repaint();
        });
    }

    /// The picture of the source image: the larger preview when it can be decoded, else the stored thumbnail.
    pub fn source_view(&self) -> ThumbView {
        let Some(path) = &self.current_path else { return ThumbView::Missing };
        let preview = if can_preview(path) { self.thumbs.request(self.thumb_scope, path, Tier::Preview, 0.0) } else { ThumbView::Missing };
        if preview.is_ready() {
            return preview;
        }
        let thumb = self.thumbs.request(self.thumb_scope, path, Tier::Thumb, 0.0);
        if thumb.is_ready() || preview == ThumbView::Missing { thumb } else { preview }
    }
}

/// Uploads a decoded picture as a texture.
fn upload(ctx: &Context, name: &str, decoded: &Decoded) -> TextureHandle {
    ctx.load_texture(name, ColorImage::from_rgba_unmultiplied(decoded.size, &decoded.rgba), TextureOptions::LINEAR)
}
