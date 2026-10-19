//! The Process dialog: describe and CLIP embed the selection, this view or a folder, with counts, overwrite flags and a confirmation for large runs.

use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crossbeam::channel::{Receiver, Sender, unbounded};
use eframe::egui::*;

use crate::Thumbnail;
use crate::command::actions::BULK_CONFIRM_ABOVE;
use crate::icons::{self, svg};
use crate::tasks::{TaskKind, TaskSpec};
use crate::ui::file_table::table::cells;
use crate::ui::file_table::{AIUpdate, FileExplorer};

/// Paths asked about the library per query.
const QUERY_CHUNK: usize = 400;
/// Images embedded per engine call.
const CLIP_CHUNK: usize = 16;
/// Folders deeper than this are not entered.
const WALK_DEPTH: usize = 64;

/// Which files a run covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessScope {
    Selection,
    View,
    Folder,
}

impl ProcessScope {
    pub const ALL: [ProcessScope; 3] = [Self::Selection, Self::View, Self::Folder];

    pub fn label(self) -> &'static str {
        match self {
            Self::Selection => "Selection",
            Self::View => "This view",
            Self::Folder => "Folder",
        }
    }
}

/// Image counts of a set of files.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub images: usize,
    pub described: usize,
    pub embedded: usize,
}

impl Counts {
    /// Images a describe run would work on.
    pub fn describe_todo(&self, overwrite: bool) -> usize {
        if overwrite { self.images } else { self.images - self.described }
    }

    /// Images a CLIP run would work on.
    pub fn clip_todo(&self, regenerate: bool) -> usize {
        if regenerate { self.images } else { self.images - self.embedded }
    }
}

fn is_image_path(path: &str) -> bool {
    Path::new(path).extension().and_then(|e| e.to_str()).is_some_and(|e| crate::is_image(&e.to_ascii_lowercase()))
}

/// Whether the vision model and CLIP can work on this row: an image file on disk.
pub fn processable(row: &Thumbnail) -> bool {
    row.file_type != "<DIR>" && !row.path.contains("://") && is_image_path(&row.path)
}

/// Counts over the processable rows of `rows`; `has_clip` says whether a row has an embedding.
pub fn count_rows<'a>(rows: impl Iterator<Item = &'a Thumbnail>, has_clip: impl Fn(&Thumbnail) -> bool) -> Counts {
    let mut c = Counts::default();
    for r in rows.filter(|r| processable(r)) {
        c.images += 1;
        if r.caption.is_some() || r.description.is_some() {
            c.described += 1;
        }
        if has_clip(r) {
            c.embedded += 1;
        }
    }
    c
}

/// Image files in `dir`, and below it when `recursive`, in walking order.
pub fn images_in(dir: &Path, recursive: bool) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![(dir.to_path_buf(), 0usize)];
    while let Some((folder, depth)) = stack.pop() {
        let Ok(read) = std::fs::read_dir(&folder) else { continue };
        let mut entries: Vec<_> = read.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            let path = entry.path();
            let Ok(kind) = entry.file_type() else { continue };
            if kind.is_dir() {
                if recursive && depth < WALK_DEPTH {
                    stack.push((path, depth + 1));
                }
            } else if kind.is_file() {
                let text = path.to_string_lossy().to_string();
                if is_image_path(&text) {
                    out.push(text);
                }
            }
        }
    }
    out
}

/// What a run starts from.
#[derive(Clone, Debug)]
pub enum Source {
    Paths(Vec<String>),
    Folder { path: String, recursive: bool },
}

fn images_noun(n: usize) -> String {
    format!("{n} image{}", if n == 1 { "" } else { "s" })
}

impl Source {
    /// What a task over this source is called.
    fn title(&self, what: &str) -> String {
        match self {
            Source::Paths(paths) => format!("{what} {}", images_noun(paths.len())),
            Source::Folder { path, .. } => format!("{what} images in {path}"),
        }
    }

    async fn gather(self) -> Vec<String> {
        match self {
            Source::Paths(paths) => paths,
            Source::Folder { path, recursive } => tokio::task::spawn_blocking(move || images_in(Path::new(&path), recursive)).await.unwrap_or_default(),
        }
    }
}

/// Flags of one run.
#[derive(Clone, Debug)]
pub struct RunOptions {
    pub describe: bool,
    pub clip: bool,
    pub overwrite_descriptions: bool,
    pub regenerate_clip: bool,
}

/// Image count of a folder, read in the background.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FolderCount {
    pub folder: String,
    pub recursive: bool,
    pub images: usize,
}

pub struct ProcessDialog {
    pub open: bool,
    pub scope: ProcessScope,
    pub folder: String,
    pub recursive: bool,
    pub describe: bool,
    pub clip: bool,
    pub overwrite_descriptions: bool,
    pub regenerate_clip: bool,
    /// Counts of the run waiting for the second confirmation.
    pub confirming: Option<(usize, usize)>,
    picked: Arc<Mutex<Option<String>>>,
    cache: Option<((u64, u64), Counts, Counts)>,
    folder_count: Option<FolderCount>,
    counting: Option<(String, bool)>,
    count_tx: Sender<FolderCount>,
    count_rx: Receiver<FolderCount>,
    pub starts: u64,
    pub last: Option<String>,
}

impl Default for ProcessDialog {
    fn default() -> Self {
        let (count_tx, count_rx) = unbounded();
        Self {
            open: false,
            scope: ProcessScope::Selection,
            folder: String::new(),
            recursive: true,
            describe: true,
            clip: true,
            overwrite_descriptions: false,
            regenerate_clip: false,
            confirming: None,
            picked: Arc::default(),
            cache: None,
            folder_count: None,
            counting: None,
            count_tx,
            count_rx,
            starts: 0,
            last: None,
        }
    }
}

impl ProcessDialog {
    /// Opens the dialog; `folder` fills the folder field and chooses the Folder scope.
    pub fn open_for(&mut self, folder: Option<String>) {
        self.open = true;
        self.confirming = None;
        if let Some(folder) = folder.filter(|f| !f.trim().is_empty()) {
            self.folder = folder;
            self.scope = ProcessScope::Folder;
        }
    }

    pub fn options(&self) -> RunOptions {
        RunOptions { describe: self.describe, clip: self.clip, overwrite_descriptions: self.overwrite_descriptions, regenerate_clip: self.regenerate_clip }
    }

    /// Takes in the count of a folder that finished in the background.
    fn poll_count(&mut self) {
        while let Ok(count) = self.count_rx.try_recv() {
            if self.counting.as_ref() == Some(&(count.folder.clone(), count.recursive)) {
                self.counting = None;
            }
            self.folder_count = Some(count);
        }
    }

    /// Starts counting the images of the folder field unless that count is known or under way.
    fn want_count(&mut self) {
        let folder = self.folder.trim().to_string();
        if folder.is_empty() || !Path::new(&folder).is_dir() {
            return;
        }
        let key = (folder.clone(), self.recursive);
        let known = self.folder_count.as_ref().is_some_and(|c| c.folder == folder && c.recursive == self.recursive);
        if known || self.counting.as_ref() == Some(&key) {
            return;
        }
        self.counting = Some(key);
        let (tx, recursive) = (self.count_tx.clone(), self.recursive);
        tokio::spawn(async move {
            let path = folder.clone();
            let images = tokio::task::spawn_blocking(move || images_in(Path::new(&path), recursive).len()).await.unwrap_or(0);
            let _ = tx.send(FolderCount { folder, recursive, images });
            crate::command::request_repaint();
        });
    }

    /// Image count of the folder field, when it is known.
    pub fn folder_images(&self) -> Option<usize> {
        let folder = self.folder.trim();
        self.folder_count.as_ref().filter(|c| c.folder == folder && c.recursive == self.recursive).map(|c| c.images)
    }
}

impl FileExplorer {
    /// Counts of the selection and of the whole view, recomputed when either changed.
    pub fn process_counts(&mut self) -> (Counts, Counts) {
        self.sync_view();
        let key = (self.selection.revision().wrapping_add(self.view.revision() << 32), self.table.versions().revision);
        if let Some((k, selected, view)) = &self.toolbar.process.cache {
            if *k == key {
                return (*selected, *view);
            }
        }
        let viewer = &self.viewer;
        let has_clip = |r: &Thumbnail| cells::has_clip(viewer, r);
        let selected = count_rows(self.selected_rows().into_iter(), has_clip);
        let view = count_rows(self.view.rows(&self.table), |r: &Thumbnail| cells::has_clip(viewer, r));
        self.toolbar.process.cache = Some((key, selected, view));
        (selected, view)
    }

    /// The source and image count of the scope chosen in the dialog.
    fn process_source(&mut self) -> (Source, Option<usize>) {
        match self.toolbar.process.scope {
            ProcessScope::Selection => {
                self.sync_view();
                let paths: Vec<String> = self.selected_rows().into_iter().filter(|r| processable(r)).map(|r| r.path.clone()).collect();
                let n = paths.len();
                (Source::Paths(paths), Some(n))
            }
            ProcessScope::View => {
                self.sync_view();
                let paths: Vec<String> = self.view.rows(&self.table).filter(|r| processable(r)).map(|r| r.path.clone()).collect();
                let n = paths.len();
                (Source::Paths(paths), Some(n))
            }
            ProcessScope::Folder => {
                let d = &self.toolbar.process;
                (Source::Folder { path: d.folder.trim().to_string(), recursive: d.recursive }, d.folder_images())
            }
        }
    }

    /// Starts the runs the dialog describes as tasks; returns a line telling what started.
    pub fn start_process(&mut self, confirmed: bool) -> Result<String, String> {
        let options = self.toolbar.process.options();
        if !options.describe && !options.clip {
            return Err("Choose describe, CLIP or both".to_string());
        }
        if self.toolbar.process.scope == ProcessScope::Folder && !Path::new(self.toolbar.process.folder.trim()).is_dir() {
            return Err("Choose a folder that exists".to_string());
        }
        let (source, count) = self.process_source();
        if count == Some(0) {
            return Err("There are no images to process in that scope".to_string());
        }
        let n = count.unwrap_or(0);
        if !confirmed && count.is_none_or(|n| n > BULK_CONFIRM_ABOVE) {
            self.toolbar.process.confirming = Some((n, usize::from(options.describe) + usize::from(options.clip)));
            return Ok("waiting for confirmation".to_string());
        }
        self.toolbar.process.confirming = None;
        let tx = self.viewer.ai_update_tx.clone();
        let prompt = self.viewer.ui_settings.ai_prompt_template.clone();
        let mut started = Vec::new();
        if options.describe {
            spawn_describe(source.clone(), options.overwrite_descriptions, prompt, tx);
            started.push("describe");
        }
        if options.clip {
            spawn_clip(source, options.regenerate_clip);
            started.push("CLIP");
        }
        self.toolbar.process.starts += 1;
        let what = format!("Started {} for {}", started.join(" and "), if count.is_some() { format!("{n} images") } else { "the folder".to_string() });
        self.toolbar.process.last = Some(what.clone());
        Ok(what)
    }
}

async fn described_paths(paths: &[String]) -> HashSet<String> {
    let mut found = HashSet::new();
    for chunk in paths.chunks(QUERY_CHUNK) {
        let rows: Vec<String> = match crate::database::DB.query("SELECT VALUE path FROM thumbnails WHERE path IN $paths AND description != NONE AND description != ''").bind(("paths", chunk.to_vec())).await {
            Ok(mut r) => r.take(0).unwrap_or_default(),
            Err(e) => {
                log::warn!("[process] reading described paths failed: {e}");
                Vec::new()
            }
        };
        found.extend(rows);
    }
    found
}

async fn embedded_paths(paths: &[String]) -> HashSet<String> {
    let mut found = HashSet::new();
    for chunk in paths.chunks(QUERY_CHUNK) {
        let rows: Vec<String> = match crate::database::DB.query("SELECT VALUE path FROM clip_embeddings WHERE path IN $paths").bind(("paths", chunk.to_vec())).await {
            Ok(mut r) => r.take(0).unwrap_or_default(),
            Err(e) => {
                log::warn!("[process] reading embedded paths failed: {e}");
                Vec::new()
            }
        };
        found.extend(rows);
    }
    found
}

async fn forget_embeddings(paths: &[String]) {
    if let Err(e) = crate::database::DB.query("DELETE clip_embeddings WHERE path IN $paths").bind(("paths", paths.to_vec())).await {
        log::warn!("[process] clearing old embeddings failed: {e}");
    }
}

async fn describe_path(path: String, prompt: String, tx: Sender<AIUpdate>) {
    let shown = path.clone();
    crate::ai::GLOBAL_AI_ENGINE
        .stream_vision_description(Path::new(&shown), &prompt, move |interim, final_opt| {
            if let Some(vd) = final_opt {
                let _ = tx.try_send(AIUpdate::Final {
                    path: path.clone(),
                    description: vd.description.clone(),
                    caption: Some(vd.caption.clone()),
                    category: if vd.category.trim().is_empty() { None } else { Some(vd.category.clone()) },
                    tags: vd.tags.clone(),
                });
            } else {
                let _ = tx.try_send(AIUpdate::Interim { path: path.clone(), text: interim.to_string() });
            }
        })
        .await;
}

/// Describes the files of `source` one after the other as a task that can be cancelled.
pub fn spawn_describe(source: Source, overwrite: bool, prompt: String, tx: Sender<AIUpdate>) {
    let stop = Arc::new(AtomicBool::new(false));
    let spec = TaskSpec::new(TaskKind::Describe, source.title("Describe")).cancel_flag(stop.clone()).user_initiated();
    tokio::spawn(async move {
        let outcome = crate::tasks::track(spec, |reporter| async move {
            let paths = source.gather().await;
            let all = paths.len();
            let known = if overwrite { HashSet::new() } else { described_paths(&paths).await };
            let todo: Vec<String> = paths.into_iter().filter(|p| !known.contains(p)).collect();
            let total = todo.len();
            reporter.detail(format!("{total} of {all} images"));
            reporter.progress(0, total as u64);
            crate::ai::GLOBAL_AI_ENGINE.reset_bulk_cancel();
            let mut done = 0usize;
            for path in todo {
                if stop.load(Ordering::Relaxed) || reporter.cancel_requested() {
                    crate::ai::GLOBAL_AI_ENGINE.cancel_bulk_descriptions();
                    break;
                }
                describe_path(path, prompt.clone(), tx.clone()).await;
                done += 1;
                reporter.progress(done as u64, total as u64);
            }
            anyhow::Ok((done, total, all - total))
        })
        .await;
        let (kind, text) = match outcome {
            Ok((done, total, _)) if done < total => (egui_toast::ToastKind::Warning, format!("Stopped after describing {done} of {total} images")),
            Ok((done, _, 0)) => (egui_toast::ToastKind::Success, format!("Described {done} images")),
            Ok((done, _, skipped)) => (egui_toast::ToastKind::Success, format!("Described {done} images; {skipped} already had a description")),
            Err(e) => (egui_toast::ToastKind::Error, format!("Describing failed: {e:#}")),
        };
        crate::ui::dialogs::toast(kind, text);
        crate::ui::sidebar::request_refresh();
    });
}

/// Generates CLIP embeddings for the files of `source` as a task that can be cancelled.
pub fn spawn_clip(source: Source, regenerate: bool) {
    let stop = Arc::new(AtomicBool::new(false));
    let spec = TaskSpec::new(TaskKind::Clip, source.title("Embed")).cancel_flag(stop.clone()).user_initiated();
    tokio::spawn(async move {
        let outcome = crate::tasks::track(spec, |reporter| async move {
            let paths = source.gather().await;
            let all = paths.len();
            let known = if regenerate { HashSet::new() } else { embedded_paths(&paths).await };
            let todo: Vec<String> = paths.into_iter().filter(|p| !known.contains(p)).collect();
            let total = todo.len();
            reporter.detail(format!("{total} of {all} images"));
            reporter.progress(0, total as u64);
            if total == 0 {
                return anyhow::Ok((0, 0, all));
            }
            crate::ai::GLOBAL_AI_ENGINE.ensure_clip_engine().await?;
            let mut done = 0usize;
            for chunk in todo.chunks(CLIP_CHUNK) {
                if stop.load(Ordering::Relaxed) || reporter.cancel_requested() {
                    break;
                }
                if regenerate {
                    forget_embeddings(chunk).await;
                }
                crate::ai::GLOBAL_AI_ENGINE.clip_generate_for_paths(chunk).await?;
                done += chunk.len();
                reporter.progress(done as u64, total as u64);
            }
            anyhow::Ok((done, total, all - total))
        })
        .await;
        let (kind, text) = match outcome {
            Ok((done, total, _)) if done < total => (egui_toast::ToastKind::Warning, format!("Stopped after embedding {done} of {total} images")),
            Ok((0, 0, skipped)) => (egui_toast::ToastKind::Info, format!("Nothing to embed: {skipped} images already have an embedding")),
            Ok((done, _, 0)) => (egui_toast::ToastKind::Success, format!("Generated CLIP embeddings for {done} images")),
            Ok((done, _, skipped)) => (egui_toast::ToastKind::Success, format!("Generated CLIP embeddings for {done} images; {skipped} already had one")),
            Err(e) => (egui_toast::ToastKind::Error, format!("CLIP generation failed: {e:#}")),
        };
        crate::ui::dialogs::toast(kind, text);
    });
}

fn counts_line(label: &str, todo: usize, images: usize, skipped_noun: &str) -> String {
    let skipped = images - todo;
    if skipped == 0 { format!("{label}: {todo} of {images} images") } else { format!("{label}: {todo} of {images} images ({skipped} {skipped_noun})") }
}

/// Draws the dialog when it is open and applies what the user chose.
pub fn show(ctx: &Context, ex: &mut FileExplorer) {
    if !ex.toolbar.process.open {
        return;
    }
    ex.toolbar.process.poll_count();
    if let Some(folder) = ex.toolbar.process.picked.lock().unwrap_or_else(|e| e.into_inner()).take() {
        ex.toolbar.process.folder = folder;
        ex.toolbar.process.scope = ProcessScope::Folder;
    }
    if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Escape)) {
        ex.toolbar.process.open = false;
        return;
    }
    if ex.toolbar.process.scope == ProcessScope::Folder {
        ex.toolbar.process.want_count();
    }
    let (selected, view) = ex.process_counts();
    let current_folder = ex.nav.scope.is_filesystem().then(|| ex.current_path.clone());
    let ai = crate::ui::file_table::action_bar::ai_ready();
    let mut open = true;
    let (mut start, mut browse, mut confirm, mut back) = (false, false, false, false);
    let id = Id::unique(("ame_process_dialog", ex.viewer.scope));
    let folder_images = ex.toolbar.process.folder_images();
    Window::new("Process files").id(id).open(&mut open).collapsible(false).resizable(false).anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0)).show(ctx, |ui| {
        ui.set_width(430.0);
        let d = &mut ex.toolbar.process;
        if let Some((images, ops)) = d.confirming {
            let what = if images > 0 { format!("{images} images") } else { "every image in the folder".to_string() };
            ui.label(RichText::new("Start this run?").strong());
            ui.add(Label::new(format!("{} on {what}. This can take a long time; you can stop it from the task list.", if ops == 2 { "Describe and CLIP" } else if d.describe { "Describe" } else { "CLIP" })).wrap());
            ui.horizontal(|ui| {
                confirm = ui.button(if images > 0 { format!("Start on {images} images") } else { "Start".to_string() }).clicked();
                back = ui.button("Back").clicked();
            });
            return;
        }
        ui.label(RichText::new("Files").strong());
        let (sel_n, view_n) = (selected.images, view.images);
        for scope in ProcessScope::ALL {
            let text = match scope {
                ProcessScope::Selection => format!("Selection ({sel_n} images)"),
                ProcessScope::View => format!("This view ({view_n} images)"),
                ProcessScope::Folder => "Folder".to_string(),
            };
            ui.radio_value(&mut d.scope, scope, text);
        }
        if d.scope == ProcessScope::Folder {
            ui.horizontal(|ui| {
                let width = (ui.available_width() - 96.0).max(120.0);
                ui.add(TextEdit::singleline(&mut d.folder).id_salt("ame_process_folder").hint_text("Folder").desired_width(width));
                browse = icons::button(ui, svg::FOLDER_OPEN, "Browse").clicked();
            });
            ui.horizontal(|ui| {
                ui.checkbox(&mut d.recursive, "Include subfolders");
                if let Some(folder) = &current_folder {
                    if ui.small_button("Use the folder shown").clicked() {
                        d.folder = folder.clone();
                    }
                }
            });
            match (d.folder.trim().is_empty(), Path::new(d.folder.trim()).is_dir(), folder_images) {
                (true, _, _) => ui.weak("Choose a folder"),
                (_, false, _) => ui.colored_label(crate::theme::warn(ui), "That folder does not exist"),
                (_, true, Some(n)) => ui.weak(format!("{n} images found")),
                _ => ui.weak("Counting images..."),
            };
        }
        ui.separator();
        ui.label(RichText::new("What to do").strong());
        let counts = match d.scope {
            ProcessScope::Selection => Some(selected),
            ProcessScope::View => Some(view),
            ProcessScope::Folder => None,
        };
        ui.checkbox(&mut d.describe, "Describe with the vision model");
        if d.describe {
            ui.indent("ame_process_describe", |ui| {
                ui.checkbox(&mut d.overwrite_descriptions, "Replace existing descriptions");
                ui.checkbox(&mut ex.follow_active_vision, "Show each image in the preview while it is described");
                if let Some(c) = counts {
                    ui.weak(counts_line("Describe", c.describe_todo(d.overwrite_descriptions), c.images, "already described"));
                }
                if !ai {
                    ui.colored_label(crate::theme::warn(ui), "Choose a vision model folder in the settings first");
                }
            });
        }
        ui.checkbox(&mut d.clip, "Generate CLIP embeddings");
        if d.clip {
            ui.indent("ame_process_clip", |ui| {
                ui.checkbox(&mut d.regenerate_clip, "Replace existing embeddings").on_hover_text("Existing embeddings of these images are deleted before new ones are made");
                if let Some(c) = counts {
                    ui.weak(counts_line("CLIP", c.clip_todo(d.regenerate_clip), c.images, "already embedded"));
                }
            });
        }
        ui.separator();
        ui.horizontal(|ui| {
            let ready = (d.describe || d.clip) && (d.scope != ProcessScope::Folder || Path::new(d.folder.trim()).is_dir());
            start = ui.add_enabled(ready, Button::new("Start")).clicked();
            if ui.button("Close").clicked() {
                d.open = false;
            }
        });
    });
    if browse {
        let slot = ex.toolbar.process.picked.clone();
        crate::ui::dialogs::pick_folder(ctx, "Folder to process", move |dir| {
            *slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(dir.display().to_string());
        });
    }
    if back {
        ex.toolbar.process.confirming = None;
    }
    if start || confirm {
        match ex.start_process(confirm) {
            Ok(_) if ex.toolbar.process.confirming.is_none() => ex.toolbar.process.open = false,
            Ok(_) => {}
            Err(reason) => crate::ui::dialogs::toast(egui_toast::ToastKind::Warning, reason),
        }
    }
    if !open {
        ex.toolbar.process.open = false;
    }
}

#[cfg(test)]
mod tests;
