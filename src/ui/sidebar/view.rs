//! Drawing of the sidebar: section headers, link rows, inline editors and the icon rail.

use eframe::egui::containers::menu::{MenuButton, MenuConfig};
use eframe::egui::{Button, Key, PopupCloseBehavior, Rect, Response, RichText, ScrollArea, Sense, TextEdit, Ui, pos2, vec2};

use super::links::{self, CollectionEntry, CountEntry, Link, click_target};
use super::{RenameState, SectionId, SidebarAction, SidebarInput, SidebarState, Source, UiLayout};
use crate::command::Target;
use crate::icons::{self, IconSize, svg};
use crate::theme::controls::{self, fold_header, loading_row, note, search_field};
use crate::ui::titlebar::kit_icon;

const HINT_OPEN: &str = "Click: open here. Ctrl or middle click: new tab.";
const HINT_FOLDER: &str = " Shift: scan recursively.";
/// Width of a rail flyout.
const FLYOUT_WIDTH: f32 = 260.0;
/// Tallest rail flyout.
const FLYOUT_HEIGHT: f32 = 460.0;
/// Side of the action button at the end of a section header.
const HEADER_ACTION: f32 = 20.0;

/// What a section draws with.
struct Env<'a> {
    input: &'a SidebarInput<'a>,
    connected: bool,
    /// Close the enclosing menu after a link is clicked.
    flyout: bool,
}

/// The library is usable: the database is ready and the debug switch is off.
pub fn is_connected(state: &SidebarState, input: &SidebarInput) -> bool {
    input.db_ready && !state.force_disconnected
}

fn visible_sections(connected: bool) -> impl Iterator<Item = SectionId> {
    SectionId::ALL.into_iter().filter(move |s| connected || !s.needs_library())
}

/// Draws the full sidebar; returns what the user asked for.
pub fn show_full(ui: &mut Ui, state: &mut SidebarState, layout: &mut UiLayout, input: &SidebarInput) -> Vec<SidebarAction> {
    let env = Env { input, connected: is_connected(state, input), flyout: false };
    let mut actions = Vec::new();
    state.data.poll(ui.ctx(), Source::Places, true, input.now);
    ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        for section in visible_sections(env.connected) {
            let open = layout.is_section_open(section.key(), section.default_open());
            let header = fold_header(ui, open, kit_icon(section.icon()), section.title());
            let acted = header_action(ui, state, section, header.rect, &env);
            if header.clicked() && !acted {
                layout.set_section_open(section.key(), !open);
            }
            header.context_menu(|ui| {
                if ui.button("Refresh").clicked() {
                    section_sources(section).iter().for_each(|s| state.data.invalidate(*s));
                    ui.close();
                }
            });
            if open {
                section_body(ui, state, section, &env, &mut actions);
            }
            if section == SectionId::Places && !env.connected {
                disconnected_card(ui, &mut actions);
            }
        }
    });
    actions
}

/// Draws the narrow sidebar: one icon per section that opens its links in a flyout.
pub fn show_rail(ui: &mut Ui, state: &mut SidebarState, input: &SidebarInput) -> Vec<SidebarAction> {
    let env = Env { input, connected: is_connected(state, input), flyout: true };
    let mut actions = Vec::new();
    state.data.poll(ui.ctx(), Source::Places, true, input.now);
    ui.vertical_centered(|ui| {
        for section in visible_sections(env.connected) {
            let button = icons::icon_button_widget(ui, section.icon(), IconSize::Md);
            let config = MenuConfig::new().close_behavior(PopupCloseBehavior::CloseOnClickOutside);
            let (response, _) = MenuButton::from_button(button).config(config).ui(ui, |ui| {
                ui.set_min_width(FLYOUT_WIDTH);
                ui.label(RichText::new(section.title()).strong());
                ScrollArea::vertical().max_height(FLYOUT_HEIGHT).show(ui, |ui| section_body(ui, state, section, &env, &mut actions));
            });
            response.on_hover_text(section.title());
        }
        if !env.connected && icons::icon_button(ui, svg::PLUGS_CONNECTED, "Library not connected. Click to connect.").clicked() {
            actions.push(SidebarAction::Connect);
        }
    });
    actions
}

fn section_sources(section: SectionId) -> &'static [Source] {
    match section {
        SectionId::Places => &[Source::Places],
        SectionId::Library => &[Source::Library],
        SectionId::Collections => &[Source::Collections],
        SectionId::Tags => &[Source::Tags],
        SectionId::Categories => &[Source::Categories],
        SectionId::Scans => &[Source::Scans],
    }
}

/// Draws the hover action at the end of a section header; true when it was clicked.
fn header_action(ui: &mut Ui, state: &mut SidebarState, section: SectionId, header: Rect, env: &Env) -> bool {
    let (icon, tip) = match section {
        SectionId::Collections if env.connected => (svg::PLUS, "New collection"),
        SectionId::Places | SectionId::Tags | SectionId::Categories | SectionId::Scans => (svg::ARROWS_CLOCKWISE, "Refresh"),
        _ => return false,
    };
    let spot = Rect::from_center_size(pos2(header.right() - HEADER_ACTION / 2.0 - 4.0, header.center().y), vec2(HEADER_ACTION, HEADER_ACTION));
    let response = ui.interact(spot, ui.id().with(("sidebar_header_action", section.key())), Sense::click());
    if !(ui.rect_contains_pointer(header) || response.has_focus()) {
        return false;
    }
    let tokens = crate::theme::tokens(ui.ctx());
    if response.hovered() {
        ui.painter().rect_filled(spot, tokens.radius_sm, tokens.hover);
    }
    kit_icon(icon).paint(ui, spot, crate::theme::metrics(ui.ctx()).icon_sm, if response.hovered() { tokens.text } else { tokens.text_dim });
    let response = response.on_hover_text(tip);
    if !response.clicked() {
        return false;
    }
    match section {
        SectionId::Collections => state.new_collection = Some(String::new()),
        other => section_sources(other).iter().for_each(|s| state.data.invalidate(*s)),
    }
    true
}

fn section_body(ui: &mut Ui, state: &mut SidebarState, section: SectionId, env: &Env, actions: &mut Vec<SidebarAction>) {
    match section {
        SectionId::Places => places(ui, state, env, actions),
        SectionId::Library => library(ui, state, env, actions),
        SectionId::Collections => collections(ui, state, env, actions),
        SectionId::Tags => counted(ui, state, env, actions, CountKind::Tags),
        SectionId::Categories => counted(ui, state, env, actions, CountKind::Categories),
        SectionId::Scans => scans(ui, state, env, actions),
    }
}

/// A one-line problem or a loading hint for `source`; true when the list should not be drawn yet.
fn status_line(ui: &mut Ui, state: &SidebarState, source: Source, empty: bool) -> bool {
    let meta = state.data.meta(source);
    if let Some(error) = &meta.error {
        let tone = crate::theme::error(ui);
        note(ui, controls::icons::WARNING, &format!("Could not load {}: {error}", source.name()), tone);
        if ui.small_button("Retry").clicked() {
            state.data.invalidate(source);
        }
        return empty;
    }
    if meta.loaded_at.is_none() {
        if state.data.auto_load() {
            loading_row(ui, "Loading");
        }
        return true;
    }
    false
}

fn places(ui: &mut Ui, state: &mut SidebarState, env: &Env, actions: &mut Vec<SidebarAction>) {
    let raw = state.data.places();
    let model = links::places_links(&raw.quick, &raw.drives, &raw.wsl, env.input.recent_paths);
    if model.quick.is_empty() && state.data.meta(Source::Places).loaded_at.is_none() && state.data.auto_load() {
        loading_row(ui, "Loading");
    }
    model.quick.iter().for_each(|l| link_row(ui, l, env, actions));
    if !model.drives.is_empty() {
        controls::section_label(ui, "Drives");
        model.drives.iter().for_each(|l| link_row(ui, l, env, actions));
    }
    if !model.wsl.is_empty() {
        controls::section_label(ui, "WSL");
        model.wsl.iter().for_each(|l| link_row(ui, l, env, actions));
    }
    wsl_drives(ui, &raw.physical);
    if !model.recent.is_empty() {
        controls::section_label(ui, "Recent folders");
        model.recent.iter().for_each(|l| link_row(ui, l, env, actions));
    }
}

/// Mount and unmount buttons of the physical drives WSL can attach.
fn wsl_drives(ui: &mut Ui, drives: &[crate::utilities::explorer::PhysicalDrive]) {
    if drives.is_empty() {
        return;
    }
    controls::section_label(ui, "Physical drives (WSL mount)");
    for drive in drives {
        ui.label(format!("{} ({:.1} GB)", drive.model, drive.size as f64 / (1024.0 * 1024.0 * 1024.0)));
        ui.horizontal_wrapped(|ui| {
            for partition in 0..drive.partitions {
                if ui.small_button(format!("Mount P{partition}")).on_hover_text(format!("Mount partition {partition} of {}", drive.device_id)).clicked() {
                    let device = drive.device_id.clone();
                    tokio::spawn(async move { report_wsl(crate::utilities::explorer::mount_wsl_drive(&device, partition)) });
                }
            }
            if ui.small_button("Unmount").on_hover_text(format!("Unmount all partitions of {}", drive.device_id)).clicked() {
                let device = drive.device_id.clone();
                tokio::spawn(async move { report_wsl(crate::utilities::explorer::unmount_wsl_drive(&device)) });
            }
        });
    }
}

fn report_wsl(result: Result<String, String>) {
    match result {
        Ok(text) => log::info!("WSL: {text}"),
        Err(e) => log::error!("WSL: {e}"),
    }
}

fn library(ui: &mut Ui, state: &mut SidebarState, env: &Env, actions: &mut Vec<SidebarAction>) {
    state.data.poll(ui.ctx(), Source::Library, true, env.input.now);
    for link in links::library_links(state.data.total(), env.input.failed_thumbs) {
        link_row(ui, &link, env, actions);
    }
}

fn collections(ui: &mut Ui, state: &mut SidebarState, env: &Env, actions: &mut Vec<SidebarAction>) {
    state.data.poll(ui.ctx(), Source::Collections, true, env.input.now);
    new_collection_field(ui, state);
    let entries = state.data.collections();
    if status_line(ui, state, Source::Collections, entries.is_empty()) {
        return;
    }
    if entries.is_empty() && state.new_collection.is_none() {
        ui.label(RichText::new("No collections yet").weak());
        return;
    }
    let all = links::collection_links(&entries);
    for (entry, link) in entries.iter().zip(&all) {
        collection_row(ui, state, entry, link, env, actions);
    }
}

fn new_collection_field(ui: &mut Ui, state: &mut SidebarState) {
    let Some(text) = state.new_collection.as_mut() else { return };
    let edit = ui.add(TextEdit::singleline(text).hint_text("New collection name").desired_width(f32::INFINITY));
    if !edit.has_focus() && !edit.lost_focus() {
        edit.request_focus();
    }
    let enter = edit.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter));
    let escape = ui.input(|i| i.key_pressed(Key::Escape));
    if enter && !text.trim().is_empty() {
        let name = text.trim().to_string();
        state.new_collection = None;
        state.data.create_collection(ui.ctx(), name);
    } else if escape || edit.lost_focus() {
        state.new_collection = None;
    }
}

fn collection_row(ui: &mut Ui, state: &mut SidebarState, entry: &CollectionEntry, link: &Link, env: &Env, actions: &mut Vec<SidebarAction>) {
    if state.rename.as_ref().is_some_and(|r| r.id == entry.id) {
        rename_field(ui, state, entry);
        return;
    }
    if state.confirm_delete.as_deref() == Some(entry.name.as_str()) {
        delete_confirm(ui, state, entry);
        return;
    }
    let mut rename = false;
    let mut delete = false;
    let response = row(ui, link, env, actions);
    response.context_menu(|ui| {
        link_menu(ui, link, actions);
        ui.separator();
        if ui.button("Rename…").clicked() {
            rename = true;
            ui.close();
        }
        if ui.button("Delete…").clicked() {
            delete = true;
            ui.close();
        }
    });
    if rename {
        state.rename = Some(RenameState { id: entry.id.clone(), input: entry.name.clone() });
    }
    if delete {
        state.confirm_delete = Some(entry.name.clone());
    }
}

fn rename_field(ui: &mut Ui, state: &mut SidebarState, entry: &CollectionEntry) {
    let Some(rename) = state.rename.as_mut() else { return };
    let edit = ui.add(TextEdit::singleline(&mut rename.input).desired_width(f32::INFINITY));
    if !edit.has_focus() && !edit.lost_focus() {
        edit.request_focus();
    }
    let enter = edit.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter));
    let escape = ui.input(|i| i.key_pressed(Key::Escape));
    if enter {
        let input = rename.input.trim().to_string();
        state.rename = None;
        if !input.is_empty() && input != entry.name {
            state.data.rename_collection(ui.ctx(), entry.id.clone(), input);
        }
    } else if escape || edit.lost_focus() {
        state.rename = None;
    }
}

fn delete_confirm(ui: &mut Ui, state: &mut SidebarState, entry: &CollectionEntry) {
    ui.label(format!("Delete \"{}\"? Files are not touched.", entry.name));
    ui.horizontal(|ui| {
        if ui.add(Button::new(RichText::new("Delete").color(crate::theme::error(ui)))).clicked() {
            state.confirm_delete = None;
            state.data.delete_collection(ui.ctx(), entry.id.clone(), entry.name.clone());
        }
        if ui.button("Keep").clicked() {
            state.confirm_delete = None;
        }
    });
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CountKind {
    Tags,
    Categories,
}

fn counted(ui: &mut Ui, state: &mut SidebarState, env: &Env, actions: &mut Vec<SidebarAction>, kind: CountKind) {
    let (source, entries): (Source, std::sync::Arc<Vec<CountEntry>>) = match kind {
        CountKind::Tags => (Source::Tags, state.data.tags()),
        CountKind::Categories => (Source::Categories, state.data.categories()),
    };
    state.data.poll(ui.ctx(), source, true, env.input.now);
    if status_line(ui, state, source, entries.is_empty()) {
        return;
    }
    if entries.is_empty() {
        ui.label(RichText::new(if kind == CountKind::Tags { "No tags yet" } else { "No categories yet" }).weak());
        return;
    }
    let (filter, show_all) = match kind {
        CountKind::Tags => (&mut state.tag_filter, &mut state.show_all_tags),
        CountKind::Categories => (&mut state.category_filter, &mut state.show_all_categories),
    };
    if entries.len() > links::TOP_N {
        let hint = if kind == CountKind::Tags { "Filter tags" } else { "Filter categories" };
        search_field(ui, filter, hint, ui.available_width());
    }
    let (shown, hidden) = links::top_entries(&entries, filter, *show_all);
    for entry in shown {
        let link = if kind == CountKind::Tags { links::tag_link(entry) } else { links::category_link(entry) };
        link_row(ui, &link, env, actions);
    }
    if hidden > 0 && ui.small_button(format!("Show {hidden} more")).clicked() {
        *show_all = true;
    } else if *show_all && filter.trim().is_empty() && ui.small_button("Show fewer").clicked() {
        *show_all = false;
    }
}

fn scans(ui: &mut Ui, state: &mut SidebarState, env: &Env, actions: &mut Vec<SidebarAction>) {
    state.data.poll(ui.ctx(), Source::Scans, true, env.input.now);
    let entries = state.data.scans();
    if status_line(ui, state, Source::Scans, entries.is_empty()) {
        return;
    }
    if entries.is_empty() {
        ui.label(RichText::new("No saved scans").weak());
        return;
    }
    for link in links::scan_links(&entries) {
        link_row(ui, &link, env, actions);
    }
}

fn disconnected_card(ui: &mut Ui, actions: &mut Vec<SidebarAction>) {
    ui.add_space(6.0);
    let tone = crate::theme::warn(ui);
    note(ui, controls::icons::WARNING, "Library not connected", tone);
    if icons::button(ui, svg::PLUGS_CONNECTED, "Connect").on_hover_text("Open the database setup").clicked() {
        actions.push(SidebarAction::Connect);
    }
}

fn link_row(ui: &mut Ui, link: &Link, env: &Env, actions: &mut Vec<SidebarAction>) {
    let response = row(ui, link, env, actions);
    response.context_menu(|ui| link_menu(ui, link, actions));
}

/// One link row; a click pushes the open command.
fn row(ui: &mut Ui, link: &Link, env: &Env, actions: &mut Vec<SidebarAction>) -> Response {
    let selected = link.is_active(env.input.active);
    let response = controls::ListItem::new(&link.label).icon(kit_icon(link.icon)).detail(&link.detail).selected(selected).show(ui);
    let hint = if link.is_folder() { format!("{HINT_OPEN}{HINT_FOLDER}") } else { HINT_OPEN.to_string() };
    let response = response.on_hover_text(format!("{}\n{hint}", link.tooltip));
    let middle = response.middle_clicked();
    if response.clicked() || middle {
        let (ctrl, shift) = ui.input(|i| (i.modifiers.command || i.modifiers.ctrl, i.modifiers.shift));
        let (target, recursive) = click_target(ctrl, middle, shift);
        actions.push(SidebarAction::Command(link.command(target, recursive)));
        if env.flyout {
            ui.close();
        }
    }
    response
}

fn link_menu(ui: &mut Ui, link: &Link, actions: &mut Vec<SidebarAction>) {
    let mut open = |ui: &mut Ui, label: &str, target: Target, recursive: bool| {
        if ui.button(label).clicked() {
            actions.push(SidebarAction::Command(link.command(target, recursive)));
            ui.close();
        }
    };
    open(ui, "Open here", Target::Current, false);
    open(ui, "Open in new tab", Target::NewTab, false);
    open(ui, "Open in background tab", Target::Background, false);
    if link.is_folder() {
        open(ui, "Scan recursively here", Target::Current, true);
        open(ui, "Scan recursively in new tab", Target::NewTab, true);
    }
    if ui.button("Copy scope").on_hover_text(link.scope.format()).clicked() {
        ui.ctx().copy_text(link.scope.format());
        ui.close();
    }
}
