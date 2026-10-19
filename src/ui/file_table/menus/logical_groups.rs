use std::collections::HashMap;
use std::future::Future;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

use crossbeam::channel::Sender;
use eframe::egui::*;
use egui::containers::menu::{MenuButton, MenuConfig};
use surrealdb::Surreal;
use surrealdb::engine::local::Db;
use surrealdb::types::RecordId;

use crate::database::collections::{self, UNFILED_NAME};
use crate::database::collections_migration::{self, CopyOptions, FinishOptions, Migrator};
use crate::database::{DB, LogicalGroup, Thumbnail};
use crate::icons::{self, svg};
use crate::ui::dialogs;
use crate::ui::file_table::explorer::persist::persist_scanned;

/// Longest collection name shown on the menu button before it is cut with an ellipsis.
const GROUP_LABEL_MAX_CHARS: usize = 18;
/// How long counts are reused before the open menu reads them again.
const COUNTS_TTL: Duration = Duration::from_secs(4);

#[derive(Default)]
struct MenuState {
    counts: HashMap<String, u64>,
    refreshed: Option<Instant>,
    refreshing: bool,
    pending_migration: bool,
    migrating: bool,
    message: Option<(bool, String)>,
    confirm_delete: Option<String>,
}

static MENU: LazyLock<Mutex<MenuState>> = LazyLock::new(|| Mutex::new(MenuState::default()));

fn menu<R>(f: impl FnOnce(&mut MenuState) -> R) -> R {
    f(&mut MENU.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Text of the collections menu button for the active collection, if any.
pub fn menu_label(active: Option<&str>) -> String {
    match active {
        Some(name) => {
            let short: String = if name.chars().count() > GROUP_LABEL_MAX_CHARS {
                name.chars().take(GROUP_LABEL_MAX_CHARS - 3).collect::<String>() + "..."
            } else {
                name.to_string()
            };
            format!("Collections ({short})")
        }
        None => "Collections".to_string(),
    }
}

/// The active group name as a collection name; the unfiled marker means none.
pub fn collection_of_active(active: Option<&str>) -> Option<&str> {
    active.filter(|n| *n != UNFILED_NAME)
}

/// Adds `paths` to the collection called `name`, creating it when absent; returns a one-line summary.
pub async fn add_to_named(db: &Surreal<Db>, name: &str, paths: &[String]) -> anyhow::Result<String> {
    let (c, created) = collections::get_or_create(db, name).await?;
    let r = crate::database::journal::add_to_collection_with(db, &c.id, paths, &format!("Add to \"{}\"", c.name)).await?;
    let made = if created { " (new collection)" } else { "" };
    Ok(format!("Added {} files to \"{}\"{made}; {} were already there", r.added, c.name, r.already))
}

/// Removes `paths` from a collection; returns a one-line summary.
pub async fn remove_from(db: &Surreal<Db>, id: &RecordId, name: &str, paths: &[String]) -> anyhow::Result<String> {
    let n = crate::database::journal::remove_from_collection_with(db, id, paths, &format!("Remove from \"{name}\"")).await?;
    Ok(format!("Removed {n} files from \"{name}\"; {} were not in it", paths.len().saturating_sub(n)))
}

/// Copies the members of one collection into another, creating the destination when absent.
pub async fn copy_between(db: &Surreal<Db>, src: &str, dst: &str) -> anyhow::Result<String> {
    let from = collections::get_by_name(db, src).await?.ok_or_else(|| collections::CollectionError::NotFound(src.to_string()))?;
    let (to, _) = collections::get_or_create(db, dst).await?;
    anyhow::ensure!(from.id != to.id, "choose two different collections");
    let n = crate::database::journal::copy_collection_with(db, &from.id, &to.id, &format!("Copy \"{}\" to \"{}\"", from.name, to.name)).await?;
    Ok(format!("Copied {n} files from \"{}\" to \"{}\"", from.name, to.name))
}

/// Adds every indexed file that is in no collection to the collection called `name`.
pub async fn add_unfiled(db: &Surreal<Db>, name: &str) -> anyhow::Result<String> {
    let paths = collections::unfiled_paths(db, usize::MAX).await?;
    if paths.is_empty() {
        return Ok("Every indexed file is already in a collection".to_string());
    }
    add_to_named(db, name, &paths).await
}

/// What the add and remove actions do with the rows they are given.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupAction {
    Add,
    Remove,
}

/// The collection to act on and the files among `rows`; selecting nothing or having no active collection plans nothing.
pub fn plan_group_action(rows: &[&Thumbnail], active: Option<&str>) -> Result<(String, Vec<Thumbnail>), String> {
    let files: Vec<Thumbnail> = rows.iter().filter(|r| r.file_type != "<DIR>" && !r.path.is_empty()).map(|r| (*r).clone()).collect();
    if files.is_empty() {
        return Err("Select files first: this acts on the selection only".to_string());
    }
    match collection_of_active(active) {
        Some(name) => Ok((name.to_string(), files)),
        None => Err("Choose an active collection in the Collections menu first".to_string()),
    }
}

/// Adds the files to the collection `name`, saving them to the library first, or removes them from it.
pub async fn apply_group_action(db: &Surreal<Db>, action: GroupAction, name: &str, files: Vec<Thumbnail>) -> anyhow::Result<String> {
    let paths: Vec<String> = files.iter().map(|r| r.path.clone()).collect();
    match action {
        GroupAction::Add => {
            persist_scanned(db, files, None).await?;
            add_to_named(db, name, &paths).await
        }
        GroupAction::Remove => {
            let found = collections::get_by_name(db, name).await?.ok_or_else(|| collections::CollectionError::NotFound(name.to_string()))?;
            remove_from(db, &found.id, &found.name, &paths).await
        }
    }
}

/// Runs the action on the selected rows in the background and toasts the counts, or why nothing happened.
pub fn run_group_action(action: GroupAction, rows: &[&Thumbnail]) {
    let active = crate::command::context::active_collection();
    match plan_group_action(rows, active.as_deref()) {
        Err(reason) => dialogs::toast(egui_toast::ToastKind::Warning, reason),
        Ok((name, files)) => {
            tokio::spawn(async move {
                let (kind, text) = match apply_group_action(&DB, action, &name, files).await {
                    Ok(summary) => (egui_toast::ToastKind::Success, summary),
                    Err(e) => (egui_toast::ToastKind::Error, format!("{e:#}")),
                };
                dialogs::toast(kind, text);
            });
        }
    }
}

fn run_op<F>(ctx: &Context, tx: Sender<Vec<LogicalGroup>>, fut: F)
where
    F: Future<Output = anyhow::Result<String>> + Send + 'static,
{
    let ctx = ctx.clone();
    tokio::spawn(async move {
        let message = match fut.await {
            Ok(text) => (true, text),
            Err(e) => (false, format!("{e:#}")),
        };
        dialogs::toast(if message.0 { egui_toast::ToastKind::Success } else { egui_toast::ToastKind::Error }, message.1.clone());
        menu(|m| m.message = Some(message));
        refresh_now(&ctx, tx).await;
    });
}

async fn refresh_now(ctx: &Context, tx: Sender<Vec<LogicalGroup>>) {
    let rows = collections::list_with_counts(&DB).await;
    let pending = collections_migration::legacy_pending(&DB).await.unwrap_or(false);
    match rows {
        Ok(rows) => {
            let mut groups = vec![LogicalGroup::default()];
            groups.extend(rows.iter().map(|c| LogicalGroup::from(c.collection.clone())));
            menu(|m| {
                m.counts = rows.iter().map(|c| (c.collection.key(), c.count)).collect();
                m.pending_migration = pending;
                m.refreshed = Some(Instant::now());
            });
            let _ = tx.try_send(groups);
        }
        Err(e) => {
            log::warn!("refresh collections failed: {e:?}");
            menu(|m| {
                m.message = Some((false, format!("Reading collections failed: {e:#}")));
                m.refreshed = Some(Instant::now());
            });
        }
    }
    menu(|m| m.refreshing = false);
    ctx.request_repaint();
}

fn refresh(ctx: &Context, tx: Sender<Vec<LogicalGroup>>) {
    if menu(|m| std::mem::replace(&mut m.refreshing, true)) {
        return;
    }
    let ctx = ctx.clone();
    tokio::spawn(async move { refresh_now(&ctx, tx).await });
}

fn refresh_if_stale(ctx: &Context, tx: Sender<Vec<LogicalGroup>>) {
    if menu(|m| m.refreshed.is_none_or(|t| t.elapsed() > COUNTS_TTL)) {
        refresh(ctx, tx);
    }
}

fn migrate_now(ctx: &Context, tx: Sender<Vec<LogicalGroup>>) {
    if menu(|m| std::mem::replace(&mut m.migrating, true)) {
        return;
    }
    let ctx = ctx.clone();
    tokio::spawn(async move {
        let m = Migrator::global();
        let result = async {
            let copy = m.run_copy(CopyOptions::default()).await?;
            let finish = m.run_finish(FinishOptions::default()).await?;
            anyhow::Ok(format!("Migrated {} collections and {} memberships; {}", copy.collections_created, copy.memberships_added, finish.summary))
        }
        .await;
        let message = match result {
            Ok(text) => (true, text),
            Err(e) => (false, format!("Migration failed: {e:#}")),
        };
        menu(|s| {
            s.message = Some(message);
            s.migrating = false;
        });
        refresh_now(&ctx, tx).await;
    });
}

/// Name field with an action button; true when the button is clicked or Enter is pressed in the field.
fn name_row(ui: &mut Ui, label: &str, hint: &str, text: &mut String, button: &str, enabled: bool) -> bool {
    let mut go = false;
    ui.horizontal(|ui| {
        ui.label(label);
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            go |= ui.add_enabled(enabled, Button::new(button)).clicked();
            let response = TextEdit::singleline(text).hint_text(hint).desired_width(170.).ui(ui);
            go |= enabled && response.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter));
        });
    });
    go && !text.trim().is_empty()
}

impl crate::ui::file_table::FileExplorer {
    /// The collection scanned files are added to, if one is chosen.
    pub fn active_collection(&self) -> Option<String> {
        crate::command::context::active_collection().filter(|n| collection_of_active(Some(n)).is_some())
    }

    /// Text of the Collections menu button.
    pub fn logical_group_label(&self) -> String {
        menu_label(self.active_collection().as_deref())
    }

    /// Selected rows that are files, in view order.
    fn selected_file_rows(&self) -> Vec<Thumbnail> {
        self.selected_rows().into_iter().filter(|r| r.file_type != "<DIR>").cloned().collect()
    }

    fn selected_file_count(&self) -> usize {
        self.selected_rows().iter().filter(|r| r.file_type != "<DIR>").count()
    }

    fn view_file_count(&self) -> usize {
        self.view.rows(&self.table).filter(|r| r.file_type != "<DIR>").count()
    }

    fn add_rows_to(&self, ctx: &Context, name: String, rows: Vec<Thumbnail>) {
        if rows.is_empty() || name.trim().is_empty() {
            return;
        }
        let paths: Vec<String> = rows.iter().map(|r| r.path.clone()).collect();
        run_op(ctx, self.logical_groups_tx.clone(), async move {
            let summary = add_to_named(&DB, &name, &paths).await?;
            if let Err(e) = persist_scanned(&DB, rows, None).await {
                log::error!("Add to collection: saving the rows failed: {e:#}");
            }
            Ok(summary)
        });
    }

    /// Menu button listing collections with counts and the operations on them.
    pub fn logical_group_menu(&mut self, ui: &mut Ui) {
        let label = self.logical_group_label();
        let txt = if self.active_collection().is_some() { RichText::new(label).color(crate::theme::success(ui)) } else { RichText::new(label) };
        MenuButton::new(txt)
            .config(MenuConfig::new().close_behavior(PopupCloseBehavior::CloseOnClickOutside))
            .ui(ui, |ui| self.collections_menu_body(ui));
    }

    /// Contents of the Collections menu.
    pub fn collections_menu_body(&mut self, ui: &mut Ui) {
        let tx = self.logical_groups_tx.clone();
        refresh_if_stale(ui.ctx(), tx.clone());
        ui.vertical_centered(|ui| {
            ui.set_width(520.);
            self.collections_header(ui);
            self.collections_hints(ui);
            ui.separator();
            self.collections_operations(ui);
            ui.separator();
            self.collections_list(ui);
        });
    }

    fn collections_header(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            match self.active_collection() {
                Some(name) => {
                    let known = self.logical_groups.is_empty() || self.logical_groups.iter().any(|g| g.name == name);
                    if known {
                        ui.colored_label(crate::theme::success(ui), format!("Active collection: {name}"));
                    } else {
                        ui.colored_label(crate::theme::warn(ui), format!("Active collection \"{name}\" no longer exists."));
                    }
                    if ui.small_button("Clear").on_hover_text("Stop adding scanned files to this collection").clicked() {
                        crate::command::context::set_active_collection(None);
                    }
                }
                None => {
                    ui.label(RichText::new("No active collection: scanned files are not added to one.").weak());
                }
            }
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if icons::button(ui, svg::ARROWS_CLOCKWISE, "Refresh").clicked() {
                    refresh(ui.ctx(), self.logical_groups_tx.clone());
                }
            });
        });
    }

    fn collections_hints(&mut self, ui: &mut Ui) {
        let (pending, migrating, message) = menu(|m| (m.pending_migration, m.migrating, m.message.clone()));
        if pending {
            ui.horizontal_wrapped(|ui| {
                ui.colored_label(crate::theme::warn(ui), "Logical groups from the old database are not collections yet.");
                if ui.add_enabled(!migrating, Button::new("Migrate now")).on_hover_text("Copies groups into collections after taking a store snapshot; details in Devtools > Collections").clicked() {
                    migrate_now(ui.ctx(), self.logical_groups_tx.clone());
                }
            });
        }
        if let Some((ok, text)) = message {
            ui.horizontal_wrapped(|ui| {
                let color = if ok { crate::theme::success(ui) } else { crate::theme::error(ui) };
                ui.colored_label(color, text);
                if ui.small_button("Dismiss").clicked() {
                    menu(|m| m.message = None);
                }
            });
        }
    }

    fn collections_operations(&mut self, ui: &mut Ui) {
        let selected = self.selected_file_count();
        let in_view = self.view_file_count();
        ui.heading(RichText::new("Operations").underline().color(crate::theme::primary(ui)));
        ui.label(format!("Selected files: {selected}"));
        ui.add_space(3.);

        let mut target = std::mem::take(&mut self.group_add_target);
        let go = name_row(ui, &format!("Add selection ({selected}) to:"), "Collection name", &mut target, "Add", selected > 0);
        if go {
            let rows = self.selected_file_rows();
            self.add_rows_to(ui.ctx(), target.trim().to_string(), rows);
            target.clear();
        }
        self.group_add_target = target;

        let mut view_target = std::mem::take(&mut self.group_add_view_target);
        let go = name_row(ui, &format!("Add current view ({in_view}) to:"), "Collection name", &mut view_target, "Add", in_view > 0);
        if go {
            let rows: Vec<Thumbnail> = self.view.rows(&self.table).filter(|r| r.file_type != "<DIR>").cloned().collect();
            self.add_rows_to(ui.ctx(), view_target.trim().to_string(), rows);
            view_target.clear();
        }
        self.group_add_view_target = view_target;

        ui.horizontal(|ui| {
            ui.label("Copy members from:");
            TextEdit::singleline(&mut self.group_copy_src).hint_text("Source").desired_width(130.).ui(ui);
            ui.label("to");
            TextEdit::singleline(&mut self.group_copy_dst).hint_text("Destination").desired_width(130.).ui(ui);
            let ready = !self.group_copy_src.trim().is_empty() && !self.group_copy_dst.trim().is_empty() && self.group_copy_src.trim() != self.group_copy_dst.trim();
            if ui.add_enabled(ready, Button::new("Copy")).clicked() {
                let (src, dst) = (self.group_copy_src.trim().to_string(), self.group_copy_dst.trim().to_string());
                run_op(ui.ctx(), self.logical_groups_tx.clone(), async move { copy_between(&DB, &src, &dst).await });
            }
        });

        let mut unfiled = std::mem::take(&mut self.group_add_unassigned_target);
        if name_row(ui, "Add files in no collection to:", "Collection name", &mut unfiled, "Add", true) {
            let name = unfiled.trim().to_string();
            run_op(ui.ctx(), self.logical_groups_tx.clone(), async move { add_unfiled(&DB, &name).await });
            unfiled.clear();
        }
        self.group_add_unassigned_target = unfiled;

        ui.separator();
        ui.heading(RichText::new("Create New Collection").underline().color(crate::theme::primary(ui)));
        let mut name = std::mem::take(&mut self.group_create_name_input);
        if name_row(ui, "Name:", "Collection name", &mut name, "Create", true) {
            let wanted = name.trim().to_string();
            run_op(ui.ctx(), self.logical_groups_tx.clone(), async move {
                let c = collections::create(&DB, &wanted).await?;
                Ok(format!("Created \"{}\"", c.name))
            });
            name.clear();
        }
        self.group_create_name_input = name;
    }

    fn collections_list(&mut self, ui: &mut Ui) {
        let groups: Vec<LogicalGroup> = self.logical_groups.iter().filter(|g| collections::is_collection_id(&g.id)).cloned().collect();
        if groups.is_empty() {
            ui.label(RichText::new("No collections yet.").weak());
            return;
        }
        let counts = menu(|m| m.counts.clone());
        ScrollArea::vertical().max_height(220.).show(ui, |ui| {
            for g in groups {
                self.collection_row(ui, &g, counts.get(&collections::key_of(&g.id)).copied());
            }
        });
    }

    fn collection_row(&mut self, ui: &mut Ui, g: &LogicalGroup, count: Option<u64>) {
        let selected = self.selected_file_count();
        let renaming = self.group_rename_target.as_deref() == Some(g.name.as_str());
        let armed = menu(|m| m.confirm_delete.as_deref() == Some(g.name.as_str()));
        ui.horizontal(|ui| {
            if renaming {
                ui.add_sized([170., 22.], TextEdit::singleline(&mut self.group_rename_input));
                if ui.button("Save").on_hover_text("Rename collection").clicked() {
                    self.rename_collection(ui.ctx(), g);
                }
                if ui.button("Cancel").clicked() {
                    self.group_rename_target = None;
                    self.group_rename_input.clear();
                }
                return;
            }
            let active = self.active_collection().as_deref() == Some(g.name.as_str());
            let name = if active { RichText::new(&g.name).strong() } else { RichText::new(&g.name) };
            ui.label(name);
            ui.label(RichText::new(count.map_or("-".to_string(), |n| format!("{n} files"))).weak());
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if armed {
                    if ui.button(RichText::new("Confirm delete").color(crate::theme::error(ui))).on_hover_text("Removes the collection; files are untouched").clicked() {
                        self.delete_collection(ui.ctx(), g);
                    }
                    if ui.small_button("Keep").clicked() {
                        menu(|m| m.confirm_delete = None);
                    }
                    return;
                }
                if icons::icon_button(ui, svg::TRASH, "Delete this collection (files are untouched)").clicked() {
                    menu(|m| m.confirm_delete = Some(g.name.clone()));
                }
                if icons::icon_button(ui, svg::PENCIL_SIMPLE, "Rename").clicked() {
                    self.group_rename_target = Some(g.name.clone());
                    self.group_rename_input = g.name.clone();
                }
                let can = selected > 0;
                let remove = ui.add_enabled_ui(can, |ui| icons::icon_button(ui, svg::MINUS, &format!("Remove the {selected} selected files from this collection"))).inner;
                if remove.clicked() {
                    let paths: Vec<String> = self.selected_file_rows().into_iter().map(|r| r.path).collect();
                    let (id, name) = (g.id.clone(), g.name.clone());
                    run_op(ui.ctx(), self.logical_groups_tx.clone(), async move { remove_from(&DB, &id, &name, &paths).await });
                }
                let add = ui.add_enabled_ui(can, |ui| icons::icon_button(ui, svg::PLUS, &format!("Add the {selected} selected files to this collection"))).inner;
                if add.clicked() {
                    let rows = self.selected_file_rows();
                    self.add_rows_to(ui.ctx(), g.name.clone(), rows);
                }
                if icons::icon_button(ui, svg::FOLDER_OPEN, "Show the indexed files of this collection").clicked() {
                    self.open_collection(&g.name);
                }
                if icons::toggle_button(ui, svg::CHECK, active, "Add scanned files to this collection").clicked() {
                    crate::command::context::set_active_collection((!active).then(|| g.name.clone()));
                }
            });
        });
    }

    fn rename_collection(&mut self, ctx: &Context, g: &LogicalGroup) {
        let new_name = self.group_rename_input.trim().to_string();
        self.group_rename_target = None;
        self.group_rename_input.clear();
        let Ok((display, _)) = collections::normalize(&new_name) else {
            menu(|m| m.message = Some((false, format!("\"{new_name}\" is not a usable collection name"))));
            return;
        };
        if self.active_collection().as_deref() == Some(g.name.as_str()) {
            crate::command::context::set_active_collection(Some(display));
        }
        let id = g.id.clone();
        run_op(ctx, self.logical_groups_tx.clone(), async move {
            let r = collections::rename(&DB, &id, &new_name).await?;
            Ok(format!("Renamed to \"{}\"; {} memberships moved", r.collection.name, r.moved))
        });
    }

    fn delete_collection(&mut self, ctx: &Context, g: &LogicalGroup) {
        menu(|m| m.confirm_delete = None);
        if self.active_collection().as_deref() == Some(g.name.as_str()) {
            crate::command::context::set_active_collection(None);
        }
        let (id, name) = (g.id.clone(), g.name.clone());
        run_op(ctx, self.logical_groups_tx.clone(), async move {
            let n = collections::delete(&DB, &id).await?;
            Ok(format!("Deleted \"{name}\" and its {n} memberships; files were not touched"))
        });
    }

    /// Compact collection list for the Quick Access pane.
    pub fn collections_quick_list(&mut self, ui: &mut Ui) {
        refresh_if_stale(ui.ctx(), self.logical_groups_tx.clone());
        ui.horizontal(|ui| {
            match self.active_collection() {
                Some(name) => ui.label(format!("Active: {name}")),
                None => ui.label("Active: (none)"),
            };
            if ui.button("Refresh").clicked() {
                refresh(ui.ctx(), self.logical_groups_tx.clone());
            }
        });
        if menu(|m| m.pending_migration) {
            ui.colored_label(crate::theme::warn(ui), "Old logical groups are not migrated; use the Collections menu.");
        }
        ui.separator();
        let groups: Vec<LogicalGroup> = self.logical_groups.iter().filter(|g| collections::is_collection_id(&g.id)).cloned().collect();
        if groups.is_empty() {
            ui.label(RichText::new("No collections yet.").weak());
            return;
        }
        let counts = menu(|m| m.counts.clone());
        ScrollArea::vertical().max_height(160.).show(ui, |ui| {
            for g in groups {
                ui.horizontal(|ui| {
                    let active = self.active_collection().as_deref() == Some(g.name.as_str());
                    let count = counts.get(&collections::key_of(&g.id)).map_or(String::new(), |n| format!(" ({n})"));
                    ui.label(if active { RichText::new(format!("{}{count}", g.name)).strong() } else { RichText::new(format!("{}{count}", g.name)) });
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if icons::icon_button(ui, svg::FOLDER_OPEN, "Show the indexed files of this collection").clicked() {
                            self.open_collection(&g.name);
                        }
                        if icons::toggle_button(ui, svg::CHECK, active, "Add scanned files to this collection").clicked() {
                            crate::command::context::set_active_collection((!active).then(|| g.name.clone()));
                        }
                    });
                });
            }
        });
    }
}

#[cfg(test)]
mod tests;
