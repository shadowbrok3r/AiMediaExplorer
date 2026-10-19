//! Headless frames through the real rail: layout at three widths, every target state, autosave, batch editing, the viewer and the browser's toggle.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use egui::epaint::ClippedShape;
use egui::{CentralPanel, Context, Event, Key, Modifiers, PointerButton, Pos2, RawInput, Rect, vec2};
use surrealdb::Surreal;
use surrealdb::engine::local::{Db, Mem};

use super::backend::DbBackend;
use super::backend::testing::ScriptedBackend;
use super::host::{self, HostArgs, HostOutput};
use super::prefs::{DetailTab, PrefsStore};
use super::state::{Inspector, apply_edit};
use crate::Thumbnail;
use crate::ui::file_table::FileExplorer;
use crate::ui::file_table::thumbs::service::{Producer, ServiceConfig, ThumbService};
use crate::ui::file_table::thumbs::store::{Blob, BlobFormat, MemStore};
use crate::ui::file_table::thumbs::ThumbPresence;
use crate::ui::test_support::{test_ctx, texts};

mod commands;
mod flows;
mod frames;

struct NoProducer;

impl Producer for NoProducer {
    fn request(&self, _path: &Path) -> bool {
        true
    }
}

pub(super) fn jpeg() -> Blob {
    let img: image::RgbImage = image::ImageBuffer::from_fn(256, 171, |x, y| image::Rgb([(x * 7 + y * 3) as u8, (x ^ y) as u8, ((x * y) >> 3) as u8]));
    let mut bytes = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Jpeg).unwrap();
    Blob { format: BlobFormat::Jpeg, w: 256, h: 171, bytes }
}

pub(super) async fn mem_db() -> Surreal<Db> {
    let db = Surreal::new::<Mem>(()).await.unwrap();
    db.use_ns(crate::database::NS).use_db(crate::database::DB_NAME).await.unwrap();
    crate::database::define_schema(&db).await.unwrap();
    crate::database::collections::ensure_schema(&db).await.unwrap();
    db
}

/// A row for a file in `dir`.
pub(super) fn file_row(dir: &Path, name: &str, file_type: &str) -> Thumbnail {
    let path = dir.join(name);
    let mut t = Thumbnail::new(name);
    t.path = path.display().to_string();
    t.file_type = file_type.to_string();
    t.size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    t.parent_dir = dir.display().to_string();
    t
}

/// The files of the rig, in table order: three pictures, a video, a text file and a folder.
pub(super) const NAMES: [&str; 6] = ["a.png", "b.png", "c.png", "clip.mp4", "notes.txt", "sub"];

pub(super) struct Rig {
    pub ctx: Context,
    pub ex: FileExplorer,
    pub insp: Inspector,
    pub svc: Arc<ThumbService>,
    pub dir: tempfile::TempDir,
    pub db: Surreal<Db>,
    pub backend: Arc<ScriptedBackend<DbBackend<Db>>>,
    pub out: HostOutput,
    pub shapes: Vec<ClippedShape>,
    pub assistant: bool,
    pub panel: crate::ui::assistant::AssistantPanel,
    pub notices: Vec<(crate::command::ToastLevel, String)>,
    pub width: f32,
    pub height: f32,
    clock: f64,
}

impl Rig {
    pub async fn new(width: f32) -> Self {
        let dir = tempfile::tempdir().unwrap();
        for n in ["a.png", "b.png", "c.png"] {
            image::RgbImage::from_pixel(40, 30, image::Rgb([200, 30, 30])).save(dir.path().join(n)).unwrap();
        }
        std::fs::write(dir.path().join("clip.mp4"), b"not a real video").unwrap();
        std::fs::write(dir.path().join("notes.txt"), b"hello").unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        let store = MemStore::new();
        let svc = ThumbService::new(store.clone(), tokio::runtime::Handle::current(), ServiceConfig::default(), Arc::new(NoProducer));
        let mut ex = FileExplorer::new(true);
        ex.viewer.thumbs = svc.clone();
        ex.viewer.thumb_cache = ThumbPresence::new(svc.clone());
        let types = ["png", "png", "png", "mp4", "txt", "<DIR>"];
        let rows: Vec<Thumbnail> = NAMES.iter().zip(types).map(|(n, t)| file_row(dir.path(), n, t)).collect();
        for r in rows.iter().filter(|r| r.file_type == "png") {
            store.insert(&r.path, jpeg());
        }
        ex.table.bulk_load(rows);
        ex.sync_view();
        let db = mem_db().await;
        let backend = Arc::new(ScriptedBackend::new(DbBackend::new(db.clone())));
        let prefs = PrefsStore::open(Some(dir.path().join("cfg").join("inspector_prefs.json")));
        let insp = Inspector::new(backend.clone(), prefs);
        let ctx = test_ctx();
        egui_extras::install_image_loaders(&ctx);
        Self { ctx, ex, insp, svc, dir, db, backend, out: HostOutput::default(), shapes: Vec::new(), assistant: false, panel: crate::ui::assistant::AssistantPanel::with_runtime(crate::zc::ZcRuntime::new(tokio::runtime::Handle::current(), None)), notices: Vec::new(), width, height: 700.0, clock: 0.0 }
    }

    pub fn path(&self, name: &str) -> String {
        self.dir.path().join(name).display().to_string()
    }

    pub fn select(&mut self, names: &[&str]) {
        let paths: Vec<String> = names.iter().map(|n| self.path(n)).collect();
        self.ex.select_paths(paths.iter().map(String::as_str));
    }

    pub fn frame_with(&mut self, events: Vec<Event>, modifiers: Modifiers) {
        let mut all = vec![Event::ModifiersChanged(modifiers)];
        all.extend(events);
        self.clock += 0.016;
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(self.width, self.height))), events: all, time: Some(self.clock), ..Default::default() };
        let Self { ctx, ex, insp, svc, out, assistant, panel, .. } = self;
        let mut host_out = HostOutput::default();
        let full = ctx.run_ui(input, |ui| {
            svc.poll(ui.ctx());
            host_out = host::show(ui, insp, HostArgs { ex, tab: "File Explorer", assistant_open: assistant, assistant: panel });
            CentralPanel::default().show(ui, |ui| ex.details_panel(ui));
        });
        self.shapes = full.shapes.clone();
        full.drop_without_applying_deltas();
        for edit in &host_out.edits {
            apply_edit(&mut self.ex, edit);
        }
        self.notices.extend(host_out.notices.iter().cloned());
        *out = host_out;
    }

    pub fn frame(&mut self) {
        self.frame_with(Vec::new(), Modifiers::NONE);
    }

    pub async fn settle(&mut self, frames: usize) {
        for _ in 0..frames {
            self.frame();
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// Runs frames until `done` holds; panics after five seconds.
    pub async fn until(&mut self, what: &str, mut done: impl FnMut(&mut Rig) -> bool) {
        let end = std::time::Instant::now() + Duration::from_secs(5);
        while std::time::Instant::now() < end {
            self.frame();
            if done(self) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(8)).await;
        }
        panic!("timed out waiting for {what}; state {:?}, texts {:?}", self.insp.meta.state(), self.texts());
    }

    pub fn texts(&self) -> Vec<String> {
        texts(&self.shapes).into_iter().map(|(t, _)| t).collect()
    }

    pub fn has_text(&self, text: &str) -> bool {
        texts(&self.shapes).iter().any(|(t, _)| t == text)
    }

    pub fn text_pos(&self, text: &str) -> Pos2 {
        match texts(&self.shapes).into_iter().find(|(t, _)| t == text) {
            Some((_, p)) => p,
            None => panic!("no {text:?} drawn: {:?}", self.texts()),
        }
    }

    pub fn click_at(&mut self, p: Pos2) {
        let button = |pressed| Event::PointerButton { pos: p, button: PointerButton::Primary, pressed, modifiers: Modifiers::NONE };
        self.frame_with(vec![Event::PointerMoved(p), button(true), button(false)], Modifiers::NONE);
        self.frame();
    }

    pub fn click_text(&mut self, text: &str) {
        let p = self.text_pos(text) + vec2(8.0, 6.0);
        self.click_at(p);
    }

    /// Clicks into the text field drawn `below` points under the label `label`.
    pub fn click_field(&mut self, label: &str, below: f32) {
        let p = self.text_pos(label) + vec2(30.0, below);
        self.click_at(p);
    }

    pub fn type_text(&mut self, text: &str) {
        self.frame_with(vec![Event::Text(text.to_string())], Modifiers::NONE);
        self.frame();
    }

    pub fn press(&mut self, key: Key) {
        let ev = Event::Key { key, physical_key: Some(key), pressed: true, repeat: false, modifiers: Modifiers::NONE };
        self.frame_with(vec![ev], Modifiers::NONE);
        self.frame();
    }

    pub fn open_tab(&mut self, tab: DetailTab) {
        self.insp.edit_prefs(|p| p.detail_tab = tab);
        self.frame();
    }

    pub fn row(&self, name: &str) -> Thumbnail {
        self.ex.table.get_by_path(&self.path(name)).cloned().expect("row")
    }

    pub async fn db_row(&self, name: &str) -> Option<Thumbnail> {
        let cols = "id, db_created, path, filename, file_type, size, description, caption, tags, category, thumbnail_b64, modified, hash, parent_dir, logical_group";
        self.db.query(format!("SELECT {cols} FROM thumbnails WHERE path = $p")).bind(("p", self.path(name))).await.unwrap().take::<Vec<Thumbnail>>(0).unwrap().into_iter().next()
    }
}
