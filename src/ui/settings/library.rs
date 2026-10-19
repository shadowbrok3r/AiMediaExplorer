//! Library section: where the library lives, backups, watched folders and the defaults of a scan.

use std::path::{Path, PathBuf};

use eframe::egui::{DragValue, ProgressBar, RichText, TextEdit, Ui};
use egui_toast::ToastKind;

use super::backup::{self, ExportChoice};
use super::draft::Draft;
use super::events::{Event, Tx};
use super::sections::name;
use super::widgets::{Focus, caption, group, human, path_field, size_limit, string_list, titled, toggle};
use crate::command::{AppCommand, Section};
use crate::database::snapshot::{self, SnapshotInfo, SnapshotKind};
use crate::icons::{self, svg};
use crate::settings::location::{self, Check};
use crate::settings::ui_prefs::{self, Confirm};
use crate::ui::dialogs;

const SLOT_FOLDER: &str = "settings-library-folder";
const SLOT_COPY: &str = "settings-library-copy-parent";
const SLOT_WATCH: &str = "settings-library-watch";
const SLOT_IMPORT: &str = "settings-library-import";
const SLOT_RESTORE_FILE: &str = "settings-library-restore-file";
const SLOT_RESTORE_DIR: &str = "settings-library-restore-dir";

/// A question waiting for an answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Pending {
    /// Run this .surql file against the open library.
    Import(PathBuf),
    /// Build a new library from this .surql file; the folder is the one in [`Library::restore_dir`].
    RestoreInto(PathBuf),
    /// Restore this snapshot folder.
    Restore(PathBuf),
    /// Switch to the copy at this folder.
    UseFolder(PathBuf),
}

/// State of the Library section.
#[derive(Default)]
pub struct Library {
    /// Folder text of the location field.
    pub folder: String,
    /// The configured folder the field showed last, so a change elsewhere reaches an unedited field.
    shown_location: String,
    folder_checked: Option<(String, Check)>,
    pub copied: Option<Result<PathBuf, String>>,
    /// Folder text of the restore dialog.
    pub restore_dir: String,
    finished_seen: usize,
    pub choice: ExportChoice,
    pub pending: Option<Pending>,
    pub snapshots: Option<Vec<SnapshotInfo>>,
    snapshots_loading: bool,
    pub notice: Option<(bool, String)>,
    dir_input: String,
    ext_input: String,
    watch_input: String,
}

/// What the page needs from the application.
pub struct LibraryEnv<'a> {
    pub db_ready: bool,
    pub tx: &'a Tx,
}

impl Library {
    pub fn new() -> Self {
        Self { choice: ExportChoice::default(), ..Default::default() }
    }

    pub fn on_event(&mut self, event: &Event) -> bool {
        match event {
            Event::NewLibrary(result) => {
                self.copied = Some(result.clone());
                true
            }
            Event::Snapshots(list) => {
                self.snapshots = Some(list.clone());
                self.snapshots_loading = false;
                true
            }
            _ => false,
        }
    }

    /// Takes the paths the pickers delivered this frame.
    pub fn take_picked(&mut self, env: &LibraryEnv) {
        if let Some(dir) = dialogs::take_picked(SLOT_FOLDER) {
            self.folder = dir.display().to_string();
            self.folder_checked = None;
        }
        if let Some(parent) = dialogs::take_picked(SLOT_COPY) {
            self.copied = None;
            let (tx, store) = (env.tx.clone(), PathBuf::from(crate::database::get_db_path()));
            tokio::spawn(async move {
                let result = backup::copy_store_task(store, parent).await.map_err(|e| format!("{e:#}"));
                let _ = tx.send(Event::NewLibrary(result));
                crate::command::request_repaint();
            });
        }
        if let Some(dir) = dialogs::take_picked(SLOT_WATCH) {
            ui_prefs::update(|p| {
                p.watch(&dir.display().to_string());
            });
        }
        if let Some(file) = dialogs::take_picked(SLOT_IMPORT) {
            self.pending = Some(Pending::Import(file));
        }
        if let Some(file) = dialogs::take_picked(SLOT_RESTORE_FILE) {
            self.restore_dir = backup::default_restore_dir().display().to_string();
            self.pending = Some(Pending::RestoreInto(file));
        }
        if let Some(dir) = dialogs::take_picked(SLOT_RESTORE_DIR) {
            self.restore_dir = dir.display().to_string();
        }
    }

    fn refresh_snapshots(&mut self, tx: &Tx) {
        if self.snapshots_loading {
            return;
        }
        self.snapshots_loading = true;
        let (tx, store) = (tx.clone(), PathBuf::from(crate::database::get_db_path()));
        tokio::spawn(async move {
            let list = tokio::task::spawn_blocking(move || snapshot::list_snapshots_for(&store)).await.unwrap_or_default();
            let _ = tx.send(Event::Snapshots(list));
            crate::command::request_repaint();
        });
    }

    fn check(&mut self) -> Option<&Check> {
        let text = self.folder.trim().to_string();
        if text.is_empty() {
            return None;
        }
        if self.folder_checked.as_ref().map(|(t, _)| t != &text).unwrap_or(true) {
            let check = location::check_location(Path::new(&text));
            self.folder_checked = Some((text, check));
        }
        self.folder_checked.as_ref().map(|(_, c)| c)
    }

    pub fn ui(&mut self, ui: &mut Ui, focus: &Focus, draft: &mut Draft, env: &LibraryEnv) {
        self.take_picked(env);
        self.location_ui(ui, focus, env);
        self.backup_ui(ui, focus, env);
        self.watched_ui(ui, focus);
        self.scan_defaults_ui(ui, focus, draft);
    }

    fn location_ui(&mut self, ui: &mut Ui, focus: &Focus, env: &LibraryEnv) {
        titled(ui, focus, name::DB_LOCATION);
        let current = crate::database::get_db_path();
        if self.folder.is_empty() || (self.folder == self.shown_location && self.shown_location != current) {
            self.folder = current.clone();
            self.folder_checked = None;
        }
        self.shown_location = current.clone();
        ui.horizontal_wrapped(|ui| {
            ui.label("Folder").on_hover_text("The folder that holds the library database");
            ui.monospace(&current);
            let status = if crate::database::is_connected() { ("connected", crate::theme::success_ctx(ui.ctx())) } else { ("not connected", ui.visuals().warn_fg_color) };
            ui.label(RichText::new(status.0).color(status.1));
        });
        ui.horizontal_wrapped(|ui| {
            path_field(ui, &mut self.folder, "Folder of the library", "Choose the library folder", SLOT_FOLDER, true);
        });
        match self.check().cloned() {
            Some(Check::Existing) => {
                ui.label("This folder holds a library. It is opened as it is.");
            }
            Some(Check::New) => {
                ui.label("This folder is empty or does not exist yet. A new library is created in it.");
            }
            Some(Check::Rejected(why)) => {
                ui.colored_label(ui.visuals().error_fg_color, why);
            }
            None => {}
        }
        let usable = self.folder_checked.as_ref().is_some_and(|(_, c)| c.is_usable()) && self.folder.trim() != current;
        ui.horizontal_wrapped(|ui| {
            if ui.add_enabled(usable, crate::theme::primary_button("Use this folder")).clicked() {
                self.use_folder(PathBuf::from(self.folder.trim()));
            }
            let can_copy = crate::database::is_connected() && !backup::busy();
            if ui.add_enabled(can_copy, eframe::egui::Button::new("Copy the library to a new folder...")).on_hover_text("Copies the library files into a new folder inside the folder you choose; the old library stays where it is").clicked() {
                dialogs::pick_folder_into(ui.ctx(), "Choose the folder the copy goes into", SLOT_COPY);
            }
            if ui.button("Open folder").clicked() {
                let _ = open::that_detached(&current);
            }
        });
        match &self.copied {
            Some(Ok(path)) => {
                let path = path.clone();
                ui.horizontal_wrapped(|ui| {
                    ui.label(format!("New library at {}", path.display()));
                    if ui.button("Use it").clicked() {
                        self.pending = Some(Pending::UseFolder(path));
                    }
                });
            }
            Some(Err(e)) => {
                ui.colored_label(ui.visuals().error_fg_color, format!("The new library could not be made: {e}"));
            }
            None => {}
        }
        if let Some(Pending::UseFolder(path)) = self.pending.clone() {
            self.confirm_box(ui, &format!("Open the library at {} the next time the application starts?", path.display()), "Use it", env);
        }
        if !env.db_ready {
            ui.weak("The library is not open. Settings that live in it stay in memory until it is.");
        }
    }

    fn use_folder(&mut self, path: PathBuf) {
        match location::check_location(&path) {
            Check::Rejected(why) => self.notice = Some((false, why)),
            _ => match crate::database::set_db_path(&path) {
                Ok(()) if crate::database::is_connected() => {
                    self.notice = Some((true, format!("The library folder is now {}. Restart the application to open it.", path.display())));
                }
                Ok(()) => {
                    crate::command::send(AppCommand::ConnectDatabase);
                    self.notice = Some((true, format!("Opening the library at {}", path.display())));
                }
                Err(e) => self.notice = Some((false, format!("Could not set the library folder: {e}"))),
            },
        }
    }

    /// A one-line question with Confirm and Cancel; Confirm runs the pending action.
    fn confirm_box(&mut self, ui: &mut Ui, question: &str, confirm: &str, env: &LibraryEnv) {
        let Some(pending) = self.pending.clone() else { return };
        eframe::egui::Frame::group(ui.style()).stroke(eframe::egui::Stroke::new(1.0, ui.visuals().warn_fg_color)).show(ui, |ui| {
            ui.label(RichText::new(question).color(ui.visuals().warn_fg_color));
            ui.horizontal(|ui| {
                if ui.add(crate::theme::primary_button(confirm)).clicked() {
                    self.pending = None;
                    self.run_pending(pending, env);
                }
                if ui.button("Cancel").clicked() {
                    self.pending = None;
                }
            });
        });
    }

    fn run_pending(&mut self, pending: Pending, env: &LibraryEnv) {
        match pending {
            Pending::UseFolder(path) => self.use_folder(path),
            Pending::RestoreInto(file) => {
                let (dir, tx) = (PathBuf::from(self.restore_dir.trim()), env.tx.clone());
                tokio::spawn(async move {
                    let result = backup::restore_task(file, dir).await.map(|report| report.dir).map_err(|e| format!("{e:#}"));
                    let _ = tx.send(Event::NewLibrary(result));
                    crate::command::request_repaint();
                });
            }
            Pending::Import(file) => {
                tokio::spawn(async move {
                    match backup::import_task(&crate::database::DB, file).await {
                        Ok(()) => dialogs::toast(ToastKind::Success, "Backup imported"),
                        Err(e) => dialogs::toast(ToastKind::Error, format!("Import failed: {e:#}")),
                    }
                });
            }
            Pending::Restore(dir) => {
                let store = PathBuf::from(crate::database::get_db_path());
                if crate::database::is_connected() {
                    match snapshot::stage_restore(&store, &dir) {
                        Ok(()) => self.notice = Some((true, "Restore staged. Restart the application to apply it; the current library is moved aside, never deleted.".to_string())),
                        Err(e) => self.notice = Some((false, format!("Could not stage the restore: {e}"))),
                    }
                } else {
                    tokio::spawn(async move {
                        let spec = crate::tasks::TaskSpec::new(crate::tasks::TaskKind::Database, format!("Restore snapshot {}", dir.display())).user_initiated();
                        let result = crate::tasks::track(spec, |reporter| async move {
                            snapshot::restore_snapshot_to(dir, store, backup::progress_to(reporter)).await.map_err(|e| anyhow::anyhow!(e.user_message()))
                        })
                        .await;
                        match result {
                            Ok(_) => {
                                crate::database::reopen_with_last_tx();
                                dialogs::toast(ToastKind::Success, "Snapshot restored");
                            }
                            Err(e) => dialogs::toast(ToastKind::Error, format!("Restore failed: {e:#}")),
                        }
                    });
                }
            }
        }
    }

    fn backup_ui(&mut self, ui: &mut Ui, focus: &Focus, env: &LibraryEnv) {
        titled(ui, focus, name::BACKUP);
        let connected = crate::database::is_connected();
        let busy = backup::busy();
        toggle(ui, focus, name::REDACT_KEYS, "The exported settings record leaves out the API keys. A snapshot copies the library files and always contains them.", &mut self.choice.redact_keys);
        ui.checkbox(&mut self.choice.include_thumb_blobs, "Include thumbnails").on_hover_text("Makes the file much larger; thumbnails can be made again");
        ui.horizontal_wrapped(|ui| {
            if ui.add_enabled(connected && !busy, eframe::egui::Button::new("Export library...")).on_hover_text("Writes a .surql file with everything in the library").clicked() {
                let choice = self.choice;
                dialogs::save_file(ui.ctx(), "Export the library", &backup::export_file_name(choice.redact_keys, chrono::Local::now()), Some(("SurrealQL", &["surql"])), move |file| {
                    tokio::spawn(async move {
                        match backup::export_task(&crate::database::DB, file, choice).await {
                            Ok(report) => dialogs::toast(ToastKind::Success, format!("Exported {} to {}", human(report.bytes), report.path.display())),
                            Err(e) => dialogs::toast(ToastKind::Error, format!("Export failed: {e:#}")),
                        }
                    });
                });
            }
            if ui.add_enabled(!busy, eframe::egui::Button::new("Restore from a .surql file...")).on_hover_text("Builds a new library folder from an export file; the open library is not touched").clicked() {
                dialogs::pick_file_into(ui.ctx(), "Choose a .surql backup", SLOT_RESTORE_FILE);
            }
            if ui.add_enabled(connected && !busy, eframe::egui::Button::new("Import into this library...")).on_hover_text("Runs an export file against the open library; records that exist already make it fail").clicked() {
                dialogs::pick_file_into(ui.ctx(), "Choose a .surql file", SLOT_IMPORT);
            }
            if ui.add_enabled(!busy, eframe::egui::Button::new("Snapshot now")).on_hover_text("Copies the library files into the snapshots folder").clicked() {
                let tx = env.tx.clone();
                tokio::spawn(async move {
                    match backup::snapshot_task().await {
                        Ok(info) => dialogs::toast(ToastKind::Success, format!("Snapshot {} made", info.name)),
                        Err(e) => dialogs::toast(ToastKind::Error, format!("Snapshot failed: {e:#}")),
                    }
                    let store = PathBuf::from(crate::database::get_db_path());
                    let list = tokio::task::spawn_blocking(move || snapshot::list_snapshots_for(&store)).await.unwrap_or_default();
                    let _ = tx.send(Event::Snapshots(list));
                    crate::command::request_repaint();
                });
            }
            if ui.button("More tools...").on_hover_text("Verify, delete and tune snapshots in Devtools").clicked() {
                crate::command::send(AppCommand::OpenDevtools { window: crate::command::DevtoolsWindow::DbMaintenance });
            }
        });
        if !connected {
            ui.weak("Export, import and copy need the library to be open.");
        }

        if let Some(Pending::RestoreInto(file)) = self.pending.clone() {
            self.restore_ui(ui, &file, env);
        }

        if let Some(Pending::Import(file)) = self.pending.clone() {
            if ui_prefs::confirms(Confirm::ImportBackup) {
                self.confirm_box(ui, &format!("Run {} against the open library? Existing records may be changed. Take a snapshot first.", file.display()), "Restore", env);
            } else {
                self.pending = None;
                self.run_pending(Pending::Import(file), env);
            }
        }
        if let Some(Pending::Restore(dir)) = self.pending.clone() {
            if ui_prefs::confirms(Confirm::RestoreSnapshot) {
                self.confirm_box(ui, &format!("Restore {}? The current library is moved aside, never deleted.", dir.display()), "Restore", env);
            } else {
                self.pending = None;
                self.run_pending(Pending::Restore(dir), env);
            }
        }

        self.tasks_ui(ui);

        if let Some((ok, text)) = &self.notice {
            let color = if *ok { ui.visuals().text_color() } else { ui.visuals().error_fg_color };
            ui.colored_label(color, text);
        }
        self.snapshots_ui(ui, env);
    }

    fn restore_ui(&mut self, ui: &mut Ui, file: &Path, env: &LibraryEnv) {
        eframe::egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.label(RichText::new(format!("Restore {} into a new library folder", file.display())).strong());
            ui.horizontal_wrapped(|ui| {
                path_field(ui, &mut self.restore_dir, "Folder for the new library", "Choose the folder for the new library", SLOT_RESTORE_DIR, true);
            });
            let problem = backup::restore_target_problem(Path::new(self.restore_dir.trim()));
            match &problem {
                Some(why) => {
                    ui.colored_label(ui.visuals().error_fg_color, why.as_str());
                }
                None => {
                    ui.label("The folder is new or empty. Nothing that exists is overwritten, and the open library is not touched.");
                }
            }
            ui.horizontal(|ui| {
                if ui.add_enabled(problem.is_none(), crate::theme::primary_button("Restore")).clicked() {
                    self.pending = None;
                    self.run_pending(Pending::RestoreInto(file.to_path_buf()), env);
                }
                if ui.button("Cancel").clicked() {
                    self.pending = None;
                }
            });
        });
    }

    fn tasks_ui(&self, ui: &mut Ui) {
        let tasks = backup::recent();
        for task in tasks.iter().rev().take(4) {
            ui.horizontal(|ui| {
                match task.state {
                    crate::tasks::TaskState::Failed => {
                        ui.colored_label(ui.visuals().error_fg_color, format!("{}: {}", task.title, task.error.clone().unwrap_or_default()));
                    }
                    crate::tasks::TaskState::Completed => {
                        icons::label(ui, svg::CHECK, task.title.as_str());
                    }
                    crate::tasks::TaskState::Cancelled => {
                        ui.weak(format!("{} (cancelled)", task.title));
                    }
                    _ => {
                        ui.spinner();
                        ui.label(task.title.as_str());
                        if let Some(fraction) = task.fraction() {
                            ui.add(ProgressBar::new(fraction).desired_width(160.0).show_percentage());
                        }
                        if !task.detail.is_empty() {
                            ui.weak(task.detail.as_str());
                        }
                        ui.ctx().request_repaint_after(std::time::Duration::from_millis(250));
                    }
                }
            });
        }
    }

    fn snapshots_ui(&mut self, ui: &mut Ui, env: &LibraryEnv) {
        ui.add_space(4.0);
        let mut load = false;
        ui.horizontal(|ui| {
            ui.label(RichText::new("Snapshots").strong());
            if ui.small_button("Refresh").clicked() {
                load = true;
            }
        });
        let finished = backup::recent().iter().filter(|t| !t.state.is_active()).count();
        if self.snapshots.is_none() || std::mem::replace(&mut self.finished_seen, finished) != finished {
            load = true;
        }
        if load {
            self.refresh_snapshots(env.tx);
        }
        let Some(list) = self.snapshots.clone() else {
            ui.weak("Looking for snapshots...");
            return;
        };
        if list.is_empty() {
            ui.weak("No snapshots yet.");
            return;
        }
        for info in list.iter().take(12) {
            ui.horizontal_wrapped(|ui| {
                ui.label(info.name.as_str());
                ui.weak(format!("{} - {}", info.kind.label(), human(info.size_bytes)));
                if ui.small_button("Show").clicked() {
                    let target = if info.kind == SnapshotKind::Dir { info.path.clone() } else { info.path.parent().map(Path::to_path_buf).unwrap_or_else(|| info.path.clone()) };
                    let _ = open::that_detached(target);
                }
                if ui.add_enabled(!backup::busy(), eframe::egui::Button::new("Restore").small()).clicked() {
                    if info.kind == SnapshotKind::Dir {
                        self.pending = Some(Pending::Restore(info.path.clone()));
                    } else {
                        self.restore_dir = backup::default_restore_dir().display().to_string();
                        self.pending = Some(Pending::RestoreInto(info.path.clone()));
                    }
                }
            });
        }
    }

    fn watched_ui(&mut self, ui: &mut Ui, focus: &Focus) {
        titled(ui, focus, name::WATCHED);
        let watched = ui_prefs::get().library.watched;
        if watched.is_empty() {
            ui.weak("No folders yet. Add the folders the library is built from.");
        }
        let mut remove = None;
        for path in &watched {
            ui.horizontal_wrapped(|ui| {
                icons::label(ui, svg::FOLDER, path.as_str());
                if ui.small_button("Scan").on_hover_text("Scan the folder and everything in it").clicked() {
                    crate::command::send(crate::ui::shell::open_folder_command(Path::new(path), true));
                }
                if icons::icon_button(ui, svg::X, "Remove from the list").clicked() {
                    remove = Some(path.clone());
                }
            });
        }
        if let Some(path) = remove {
            ui_prefs::update(|p| {
                p.unwatch(&path);
            });
        }
        ui.horizontal_wrapped(|ui| {
            if ui.button("Add folder...").clicked() {
                dialogs::pick_folder_into(ui.ctx(), "Add a folder to the library", SLOT_WATCH);
            }
            ui.add(TextEdit::singleline(&mut self.watch_input).hint_text("or type a path").desired_width(220.0));
            if ui.add_enabled(!self.watch_input.trim().is_empty(), eframe::egui::Button::new("Add")).clicked() {
                let text = std::mem::take(&mut self.watch_input);
                ui_prefs::update(|p| {
                    p.watch(&text);
                });
            }
            if ui.add_enabled(!watched.is_empty(), eframe::egui::Button::new("Scan all")).clicked() {
                for path in &watched {
                    crate::command::send(crate::ui::shell::open_folder_command(Path::new(path), true));
                }
            }
        });
        ui.weak("Adding and removing folders applies at once.");
    }

    fn scan_defaults_ui(&mut self, ui: &mut Ui, focus: &Focus, draft: &mut Draft) {
        group(ui, "Scan defaults");
        let s = &mut draft.value;
        toggle(ui, focus, name::SKIP_ICONS, "Leave out small images that look like icons or assets", &mut s.filter_skip_icons);
        ui.horizontal_wrapped(|ui| {
            caption(ui, focus, name::MIN_SIZE);
            size_limit(ui, "Files smaller than this are left out", &mut s.db_min_size_bytes);
        });
        ui.horizontal_wrapped(|ui| {
            caption(ui, focus, name::MAX_SIZE);
            size_limit(ui, "Files larger than this are left out", &mut s.db_max_size_bytes);
        });
        ui.horizontal_wrapped(|ui| {
            caption(ui, focus, name::BATCH);
            let mut batch = s.scan_found_batch_max.unwrap_or(128);
            if ui.add(DragValue::new(&mut batch).range(16..=4096)).on_hover_text("Rows per scan batch").changed() {
                s.scan_found_batch_max = Some(batch);
            }
        });
        toggle(ui, focus, name::AUTO_SAVE, "New scan results are saved into the library as they are found", &mut s.auto_save_to_database);

        caption(ui, focus, name::EXCLUDED_DIRS);
        let mut dirs = s.excluded_dirs.clone().unwrap_or_default();
        if string_list(ui, "settings-excluded-dirs", &mut dirs, &mut self.dir_input, "folder to skip", true) {
            s.excluded_dirs = if dirs.is_empty() { None } else { Some(dirs) };
        }
        caption(ui, focus, name::EXCLUDED_EXTS);
        let mut exts = s.db_excluded_exts.clone().unwrap_or_default();
        if string_list(ui, "settings-excluded-exts", &mut exts, &mut self.ext_input, "extension without the dot", false) {
            for ext in exts.iter_mut() {
                *ext = ext.trim_start_matches('.').to_lowercase();
            }
            s.db_excluded_exts = if exts.is_empty() { None } else { Some(exts) };
        }

        ui.horizontal_wrapped(|ui| {
            caption(ui, focus, name::DATE_RANGE);
            date_range(ui, s);
        });
    }
}

fn date_range(ui: &mut Ui, s: &mut crate::UiSettings) {
    let today = jiff::Zoned::now().date();
    let parse = |v: &Option<String>| v.as_deref().and_then(|t| t.parse::<jiff::civil::Date>().ok()).unwrap_or(today);
    let (mut after, mut before) = (parse(&s.filter_modified_after), parse(&s.filter_modified_before));
    ui.label("From");
    let salt_after = format!("{after} settings start date");
    if ui.add(egui_extras::DatePickerButton::new(&mut after).id_salt(&salt_after)).changed() {
        s.filter_modified_after = Some(after.to_string());
    }
    if icons::icon_button(ui, svg::X, "Clear the start date").clicked() {
        s.filter_modified_after = None;
    }
    ui.label("to");
    let salt_before = format!("{before} settings end date");
    if ui.add(egui_extras::DatePickerButton::new(&mut before).id_salt(&salt_before)).changed() {
        s.filter_modified_before = Some(before.to_string());
    }
    if icons::icon_button(ui, svg::X, "Clear the end date").clicked() {
        s.filter_modified_before = None;
    }
    if s.filter_modified_after.is_none() && s.filter_modified_before.is_none() {
        ui.weak("any date");
    }
}

/// The command that shows this section.
pub fn show_command() -> AppCommand {
    AppCommand::ShowSettings { section: Some(Section::Library) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_tx() -> (Tx, super::super::events::Rx) {
        super::super::events::channel()
    }

    #[test]
    fn the_scan_buttons_send_a_recursive_scan_in_a_new_tab() {
        let cmd = crate::ui::shell::open_folder_command(Path::new("/pics"), true);
        assert_eq!(serde_json::to_value(&cmd).unwrap(), serde_json::json!({"cmd": "open_scope", "scope": "fs:/pics", "target": "new_tab", "recursive": true}));
    }

    #[test]
    fn events_fill_the_section_state() {
        let (_tx, _rx) = env_tx();
        let mut lib = Library::new();
        assert!(lib.on_event(&Event::NewLibrary(Ok(PathBuf::from("/copy")))));
        assert_eq!(lib.copied, Some(Ok(PathBuf::from("/copy"))));
        assert!(lib.on_event(&Event::Snapshots(Vec::new())));
        assert_eq!(lib.snapshots.as_ref().map(Vec::len), Some(0));
        assert!(!lib.on_event(&Event::Tested("openai".into(), Ok(1))));
    }

    #[test]
    fn a_folder_text_is_checked_once_per_change() {
        let dir = tempfile::tempdir().unwrap();
        let mut lib = Library::new();
        assert!(lib.check().is_none());
        lib.folder = dir.path().display().to_string();
        assert_eq!(lib.check().cloned(), Some(Check::New));
        std::fs::write(dir.path().join("x.txt"), "x").unwrap();
        assert_eq!(lib.check().cloned(), Some(Check::New), "the answer is kept until the text changes");
        lib.folder = format!("{} ", dir.path().display());
        assert_eq!(lib.check().cloned(), Some(Check::New));
        lib.folder = dir.path().join("sub").display().to_string();
        lib.folder_checked = None;
        assert!(lib.check().is_some());
    }

    #[test]
    fn the_show_command_names_the_library_section() {
        assert_eq!(serde_json::to_value(show_command()).unwrap(), serde_json::json!({"cmd": "show_settings", "section": "library"}));
    }
}
