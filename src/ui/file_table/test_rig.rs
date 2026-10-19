//! A headless explorer drawn through its real `ui`, for the toolbar, action bar, shortcut and dialog tests.

use eframe::egui::epaint::ClippedShape;
use eframe::egui::*;

use super::FileExplorer;
use crate::Thumbnail;
use crate::database::ui_themes::tests::RT;
use crate::ui::test_support::{press_release, test_ctx, texts};

/// Number of image rows the rig starts with; a folder row comes first.
pub(crate) const ROWS: usize = 12;

pub(crate) fn path_of(i: usize) -> String {
    format!("/rig/img_{i:02}.jpg")
}

pub(crate) fn image_row(i: usize) -> Thumbnail {
    let mut t = Thumbnail::new(&format!("img_{i:02}.jpg"));
    t.path = path_of(i);
    t.file_type = "jpg".into();
    t.size = 20_000 + i as u64 * 1_000;
    t.parent_dir = "/rig".into();
    t
}

pub(crate) fn dir_row() -> Thumbnail {
    let mut t = Thumbnail::new("sub");
    t.path = "/rig/sub".into();
    t.file_type = "<DIR>".into();
    t
}

pub(crate) struct Rig {
    pub ctx: Context,
    pub ex: FileExplorer,
    pub width: f32,
    pub height: f32,
    pub shapes: Vec<ClippedShape>,
    pub copied: Vec<String>,
    /// Also draws the devtools windows after the explorer.
    pub devtools: bool,
    clock: f64,
    _runtime: tokio::runtime::EnterGuard<'static>,
}

impl Rig {
    pub fn new(width: f32, height: f32) -> Self {
        let runtime = RT.enter();
        let mut ex = FileExplorer::new(true);
        let mut rows = vec![dir_row()];
        rows.extend((0..ROWS).map(image_row));
        ex.table.bulk_load(rows);
        let ctx = test_ctx();
        egui_extras::install_image_loaders(&ctx);
        let mut rig = Self { ctx, ex, width, height, shapes: Vec::new(), copied: Vec::new(), devtools: false, clock: 0.0, _runtime: runtime };
        rig.settle(3);
        rig
    }

    pub fn frame_with(&mut self, events: Vec<Event>, modifiers: Modifiers) {
        let mut all = vec![Event::ModifiersChanged(modifiers)];
        all.extend(events);
        self.clock += 0.016;
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(self.width, self.height))), events: all, time: Some(self.clock), ..Default::default() };
        let ex = &mut self.ex;
        let devtools = self.devtools;
        let out = self.ctx.run_ui(input, |ui| {
            ex.ui(ui);
            if devtools {
                crate::ui::devtools::show(ui.ctx());
            }
        });
        self.shapes = out.shapes.clone();
        self.copied.extend(out.platform_output.commands.iter().filter_map(|c| if let OutputCommand::CopyText(t) = c { Some(t.clone()) } else { None }));
        out.drop_without_applying_deltas();
    }

    pub fn frame(&mut self, events: Vec<Event>) {
        self.frame_with(events, Modifiers::NONE);
    }

    pub fn settle(&mut self, frames: usize) {
        for _ in 0..frames {
            self.frame_with(Vec::new(), Modifiers::NONE);
        }
    }

    pub fn click(&mut self, pos: Pos2) {
        self.clock += 1.0;
        self.frame(press_release(pos));
        self.settle(3);
    }

    pub fn key(&mut self, key: Key, modifiers: Modifiers) {
        let event = |pressed| Event::Key { key, physical_key: Some(key), pressed, repeat: false, modifiers };
        self.frame_with(vec![event(true), event(false)], modifiers);
        self.settle(2);
    }

    pub fn type_text(&mut self, text: &str) {
        self.frame(vec![Event::Text(text.to_string())]);
        self.settle(2);
    }

    /// Selects exactly the rows of `paths`.
    pub fn select(&mut self, paths: &[String]) {
        self.ex.sync_view();
        self.ex.select_paths(paths.iter().map(String::as_str));
        self.settle(3);
    }

    pub fn select_images(&mut self, range: std::ops::Range<usize>) {
        let paths: Vec<String> = range.map(path_of).collect();
        self.select(&paths);
    }

    pub fn has_text(&self, text: &str) -> bool {
        texts(&self.shapes).iter().any(|(t, _)| t == text)
    }

    pub fn text_pos(&self, text: &str) -> Option<Pos2> {
        texts(&self.shapes).into_iter().find(|(t, _)| t == text).map(|(_, p)| p)
    }
}

/// Text rectangles of a frame, leaving out text clipped away entirely; the rectangles are in screen space.
pub(crate) fn text_rects(shapes: &[ClippedShape]) -> Vec<(String, Rect)> {
    fn walk(shape: &Shape, clip: Rect, out: &mut Vec<(String, Rect)>) {
        match shape {
            Shape::Text(t) if clip.width() > 0.0 && clip.height() > 0.0 => out.push((t.galley.text().to_string(), t.galley.rect.translate(t.pos.to_vec2()))),
            Shape::Vec(v) => v.iter().for_each(|s| walk(s, clip, out)),
            _ => {}
        }
    }
    let mut out = Vec::new();
    shapes.iter().for_each(|s| walk(&s.shape, s.clip_rect, &mut out));
    out
}

/// The whole application drawn headless: the dock with the main explorer, the shell overlays and the command pump.
pub(crate) struct AppRig {
    pub app: crate::app::SmartMediaApp,
    pub ctx: Context,
    pub shapes: Vec<ClippedShape>,
    width: f32,
    height: f32,
    clock: f64,
    _fixture: crate::database::ui_themes::tests::Fixture,
    _runtime: tokio::runtime::EnterGuard<'static>,
}

impl AppRig {
    pub fn new(width: f32, height: f32) -> Self {
        let fixture = crate::database::ui_themes::tests::Fixture::new();
        let runtime = RT.enter();
        let ctx = test_ctx();
        egui_extras::install_image_loaders(&ctx);
        let appearance = crate::theme::Appearance::attach(&ctx, None, std::sync::Arc::new(fixture.store()));
        let mut explorer = FileExplorer::new(true);
        let mut rows = vec![dir_row()];
        rows.extend((0..ROWS).map(image_row));
        explorer.table.bulk_load(rows);
        let mut app = crate::app::SmartMediaApp::assemble(ctx.clone(), appearance, crate::theme::chrome::Chrome { frameless: true, available: true }, explorer);
        app.context.db_ready = true;
        let mut rig = Self { app, ctx, shapes: Vec::new(), width, height, clock: 0.0, _fixture: fixture, _runtime: runtime };
        rig.settle(3);
        rig
    }

    pub fn frame(&mut self, events: Vec<Event>) {
        self.clock += 0.016;
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(self.width, self.height))), events, time: Some(self.clock), ..Default::default() };
        let app = &mut self.app;
        let out = self.ctx.run_ui(input, |ui| {
            let ctx = ui.ctx().clone();
            app.pump_commands(&ctx);
            app.side_panels(ui);
            CentralPanel::default().show(ui, |ui| {
                egui_dock::DockArea::new(&mut app.tree).show_inside(ui, &mut app.context);
            });
            crate::ui::devtools::show(&ctx);
            app.shell_overlays(&ctx);
            app.pump_commands(&ctx);
        });
        self.shapes = out.shapes.clone();
        out.drop_without_applying_deltas();
    }

    pub fn settle(&mut self, frames: usize) {
        for _ in 0..frames {
            self.frame(Vec::new());
        }
    }

    pub fn key(&mut self, key: Key, modifiers: Modifiers) {
        let event = |pressed| Event::Key { key, physical_key: Some(key), pressed, repeat: false, modifiers };
        self.frame(vec![Event::ModifiersChanged(modifiers), event(true), event(false), Event::ModifiersChanged(Modifiers::NONE)]);
        self.settle(2);
    }

    pub fn ex(&mut self) -> &mut FileExplorer {
        self.app.context.active_explorer()
    }
}

/// Runs `f` on the test runtime from a thread that is not inside one, with room for the database's recursion.
pub(crate) fn db<T: Send>(f: impl std::future::Future<Output = T> + Send) -> T {
    std::thread::scope(|s| std::thread::Builder::new().stack_size(64 << 20).spawn_scoped(s, || RT.block_on(f)).unwrap().join().unwrap())
}

/// Connects the process-wide database to memory and creates the tables the library edits read; the library tables stay schemaless like the other tests expect.
pub(crate) fn global_db() {
    crate::settings::testing::mem_db();
    static TABLES: std::sync::Once = std::sync::Once::new();
    TABLES.call_once(|| {
        db(async {
            crate::database::DB.query("DEFINE TABLE IF NOT EXISTS thumbnails SCHEMALESS; DEFINE TABLE IF NOT EXISTS clip_embeddings SCHEMALESS;").await.unwrap().check().unwrap();
            crate::database::collections::ensure_schema(&crate::database::DB).await.unwrap();
        })
    });
}
