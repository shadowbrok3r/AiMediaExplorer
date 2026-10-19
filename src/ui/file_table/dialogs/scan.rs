//! The Scan dialog: the folder, whether to go below it, whether the files found join the library, and the filters that decide what is listed.

use std::sync::{Arc, Mutex};

use eframe::egui::*;

use crate::Thumbnail;
use crate::command::{AppCommand, Scope, Target};
use crate::icons::{self, svg};
use crate::tasks::{TaskKind, TaskSpec};
use crate::ui::file_table::FileExplorer;
use crate::ui::file_table::explorer::persist::{Persisted, persist_scanned, saved_summary};
use crate::ui::file_table::rows::FilterSpec;
use crate::ui::file_table::scope::nav::check_openable;

/// Rows written to the library per step of the saving task.
const SAVE_CHUNK: usize = 400;

/// Where the scan shows its results.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanTarget {
    NewTab,
    ThisTab,
}

pub struct ScanDialog {
    pub open: bool,
    pub folder: String,
    pub recursive: bool,
    /// Add the files the scan finds to the library when it is done.
    pub library: bool,
    pub target: ScanTarget,
    /// Folder chosen in the file dialog, waiting to be taken in.
    picked: Arc<Mutex<Option<String>>>,
    pub starts: u64,
    pub last: Option<String>,
}

impl Default for ScanDialog {
    fn default() -> Self {
        Self { open: false, folder: String::new(), recursive: true, library: false, target: ScanTarget::NewTab, picked: Arc::default(), starts: 0, last: None }
    }
}

impl ScanDialog {
    /// Opens the dialog; `folder` fills the folder field, else a folder the tab shows does when the field is empty.
    pub fn open_for(&mut self, folder: Option<String>, scope: &Scope) {
        self.open = true;
        if let Some(folder) = folder.filter(|f| !f.trim().is_empty()) {
            self.folder = folder;
        } else if self.folder.trim().is_empty() {
            if let Scope::Folder(path) = scope {
                self.folder = path.clone();
            }
        }
    }

    /// The scope the dialog would scan, or why it cannot.
    pub fn scope(&self) -> Result<Scope, String> {
        let folder = self.folder.trim();
        if folder.is_empty() {
            return Err("Choose a folder to scan".to_string());
        }
        let scope = Scope::Folder(folder.to_string());
        check_openable(&scope)?;
        Ok(scope)
    }
}

/// Labels of the conditions in `spec`, as the strip shows them.
pub fn filter_labels(spec: &FilterSpec) -> Vec<String> {
    spec.chips().into_iter().map(|c| c.label).collect()
}

static SAVE_ON_DONE: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Asks for the files of the scan of `root` to be added to the library once it is done.
pub fn request_library_save(root: &str) {
    SAVE_ON_DONE.lock().unwrap_or_else(|e| e.into_inner()).push(root.to_string());
}

/// True once for each request made for `root`, and removes it.
pub fn take_library_save(root: &str) -> bool {
    let mut list = SAVE_ON_DONE.lock().unwrap_or_else(|e| e.into_inner());
    match list.iter().position(|r| r == root) {
        Some(i) => {
            list.remove(i);
            true
        }
        None => false,
    }
}

impl FileExplorer {
    /// Starts the scan the dialog describes.
    pub fn start_scan(&mut self) -> Result<String, String> {
        let dialog = &self.toolbar.scan;
        let scope = dialog.scope()?;
        let (recursive, library, target) = (dialog.recursive, dialog.library, dialog.target);
        let Scope::Folder(root) = &scope else { return Err("Choose a folder to scan".to_string()) };
        if library {
            request_library_save(root);
        }
        let text = format!("Scanning {root}{}{}", if recursive { " and its subfolders" } else { "" }, if library { "; the files found join the library" } else { "" });
        match target {
            ScanTarget::NewTab => crate::command::send(AppCommand::OpenScope { scope, target: Target::NewTab, title: None, recursive }),
            ScanTarget::ThisTab => self.open_scope_recursive(scope, recursive)?,
        }
        self.toolbar.scan.starts += 1;
        self.toolbar.scan.last = Some(text.clone());
        Ok(text)
    }

    /// Adds the files the scan of this tab found to the library as a task; used when a scan was started with the library option.
    pub fn save_scanned_to_library(&mut self) {
        let rows: Vec<Thumbnail> = self.last_scan_rows.iter().filter(|r| r.file_type != "<DIR>").cloned().collect();
        if rows.is_empty() {
            return;
        }
        self.viewer.badges.mark_indexed(rows.iter().map(|r| r.path.as_str()));
        save_rows_tracked(rows);
    }
}

/// Writes `rows` to the library and the active collection as a task with progress, and toasts the counts.
pub fn save_rows_tracked(rows: Vec<Thumbnail>) {
    let total = rows.len();
    let collection = crate::command::context::active_collection();
    let spec = TaskSpec::new(TaskKind::Database, format!("Add {total} scanned files to the library")).total(total as u64).user_initiated();
    tokio::spawn(async move {
        let named = collection.clone();
        let outcome = crate::tasks::track(spec, |reporter| async move {
            let mut sum = Persisted::default();
            let mut done = 0;
            for chunk in rows.chunks(SAVE_CHUNK) {
                let part = persist_scanned(&crate::database::DB, chunk.to_vec(), collection.as_deref()).await?;
                sum.inserted += part.inserted;
                sum.existing += part.existing;
                sum.filed += part.filed;
                sum.collection_missing |= part.collection_missing;
                done += chunk.len();
                reporter.progress(done as u64, total as u64);
            }
            anyhow::Ok(sum)
        })
        .await;
        let (kind, text) = match outcome {
            Ok(sum) => (egui_toast::ToastKind::Success, saved_summary(&sum, named.as_deref())),
            Err(e) => (egui_toast::ToastKind::Error, format!("Adding the scanned files to the library failed: {e:#}")),
        };
        crate::ui::dialogs::toast(kind, text);
        crate::ui::sidebar::request_refresh();
    });
}

/// Draws the dialog when it is open and applies what the user chose.
pub fn show(ctx: &Context, ex: &mut FileExplorer) {
    if !ex.toolbar.scan.open {
        return;
    }
    if let Some(folder) = ex.toolbar.scan.picked.lock().unwrap_or_else(|e| e.into_inner()).take() {
        ex.toolbar.scan.folder = folder;
    }
    if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Escape)) {
        ex.toolbar.scan.open = false;
        return;
    }
    let preview = match ex.toolbar.scan.target {
        ScanTarget::ThisTab => ex.filter_spec(),
        ScanTarget::NewTab => FilterSpec::from_settings(&ex.viewer.ui_settings),
    };
    let labels = filter_labels(&preview);
    let verdict = ex.toolbar.scan.scope();
    let mut open = true;
    let (mut start, mut browse, mut current) = (false, false, false);
    let current_folder = ex.nav.scope.is_filesystem().then(|| ex.current_path.clone());
    let id = Id::unique(("ame_scan_dialog", ex.viewer.scope));
    Window::new("Scan folder").id(id).open(&mut open).collapsible(false).resizable(false).anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0)).show(ctx, |ui| {
        ui.set_width(430.0);
        let dialog = &mut ex.toolbar.scan;
        ui.horizontal(|ui| {
            ui.label("Folder");
            let width = (ui.available_width() - 96.0).max(120.0);
            ui.add(TextEdit::singleline(&mut dialog.folder).id_salt("ame_scan_folder").hint_text("Folder to scan").desired_width(width));
            browse = icons::button(ui, svg::FOLDER_OPEN, "Browse").clicked();
        });
        if let Some(folder) = &current_folder {
            if ui.small_button(format!("Use the folder shown ({})", crate::ui::file_table::menus::breadcrumbs::crumbs(folder).last().map_or("root", |c| c.label.as_str()))).clicked() {
                current = true;
            }
        }
        ui.add_space(crate::theme::metrics(ui.ctx()).gap.y);
        ui.checkbox(&mut dialog.recursive, "Include subfolders").on_hover_text("Scan everything below the folder, not only the files in it");
        ui.checkbox(&mut dialog.library, "Add the files found to the library").on_hover_text("When the scan is done, save every file it found to the library and the active collection");
        ui.horizontal(|ui| {
            ui.label("Show the results in");
            ui.selectable_value(&mut dialog.target, ScanTarget::NewTab, "A new tab");
            ui.selectable_value(&mut dialog.target, ScanTarget::ThisTab, "This tab");
        });
        ui.separator();
        ui.label(RichText::new("What the results list").strong());
        ui.label(RichText::new("The scan loads every image, video and archive it finds. These filters decide what is listed; change them in the filter strip at any time.").weak().small());
        if labels.is_empty() {
            ui.label("No filters: everything found is listed.");
        } else {
            ui.horizontal_wrapped(|ui| {
                for label in &labels {
                    ui.add(Button::new(RichText::new(label).small()).small().sense(Sense::hover()));
                }
            });
        }
        ui.separator();
        if let Err(reason) = &verdict {
            ui.colored_label(crate::theme::warn(ui), reason);
        }
        ui.horizontal(|ui| {
            start = ui.add_enabled(verdict.is_ok(), Button::new("Start scan")).clicked();
            if ui.button("Close").clicked() {
                dialog.open = false;
            }
        });
    });
    if browse {
        let slot = ex.toolbar.scan.picked.clone();
        crate::ui::dialogs::pick_folder(ctx, "Folder to scan", move |dir| {
            *slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(dir.display().to_string());
        });
    }
    if current {
        if let Some(folder) = current_folder {
            ex.toolbar.scan.folder = folder;
        }
    }
    if start {
        match ex.start_scan() {
            Ok(_) => ex.toolbar.scan.open = false,
            Err(reason) => crate::ui::dialogs::toast(egui_toast::ToastKind::Error, reason),
        }
    }
    if !open {
        ex.toolbar.scan.open = false;
    }
}

#[cfg(test)]
mod tests;
