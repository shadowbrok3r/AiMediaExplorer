//! The rail contents: tabs, header, preview and the Info, Metadata, AI and Edit tabs for the focused target.

use egui::{Label, RichText, ScrollArea, Ui};

use super::prefs::{DetailTab, RailTab};
use super::preview::{PreviewEvent, controls, picture};
use super::state::Inspector;
use super::target::{Target, is_dir, is_video_path, navigate};
use super::{ai_tab, batch_tab, info_tab, meta_tab, video};
use crate::Thumbnail;
use crate::command::{ActionId, AppCommand, Selector};
use crate::icons::{self, svg};
use crate::ui::file_table::FileExplorer;

/// What the host offers the rail beyond the explorer.
pub struct RailEnv<'a> {
    pub assistant_open: &'a mut bool,
    /// Set when the user pressed the close button.
    pub close: bool,
}

fn run_on(action: ActionId, path: &str) {
    crate::command::send(AppCommand::Run { action, on: Selector::Paths(vec![path.to_string()]) });
}

/// Whether the file is known to be gone from disk.
pub fn is_offline(facts: Option<&super::facts::FileFacts>) -> bool {
    facts.is_some_and(|f| !f.exists)
}

fn tabs_row(ui: &mut Ui, insp: &mut Inspector, env: &mut RailEnv) {
    ui.horizontal(|ui| {
        for (tab, label) in [(RailTab::Inspect, "Inspect"), (RailTab::Ask, "Ask")] {
            if ui.selectable_label(insp.prefs.prefs.rail_tab == tab, label).clicked() {
                insp.edit_prefs(|p| p.rail_tab = tab);
            }
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if icons::icon_button(ui, svg::X, "Close the inspector (Ctrl+I)").clicked() {
                env.close = true;
            }
        });
    });
    ui.separator();
}

fn ask_tab(ui: &mut Ui, env: &mut RailEnv) {
    ui.add_space(24.0);
    ui.vertical_centered(|ui| {
        ui.label(RichText::new("Assistant").heading());
        ui.label(RichText::new("The assistant will live in this rail.").weak());
        ui.add_space(6.0);
        if icons::button(ui, svg::AME_ASSISTANT, "Open the assistant").clicked() {
            *env.assistant_open = true;
        }
    });
}

fn empty_state(ui: &mut Ui) {
    ui.add_space(40.0);
    ui.vertical_centered(|ui| {
        ui.label(RichText::new("Select a file").heading());
        ui.label(RichText::new("Its preview and details appear here.").weak());
    });
}

fn header(ui: &mut Ui, ex: &mut FileExplorer, row: &Thumbnail, offline: bool, folder: bool) {
    ui.horizontal(|ui| {
        if offline {
            icons::label(ui, svg::CLOUD_SLASH, RichText::new("Offline").color(crate::theme::warn(ui))).on_hover_text("The file is not on disk any more");
        }
        ui.add(Label::new(RichText::new(&row.filename).heading()).truncate());
    });
    ui.horizontal_wrapped(|ui| {
        let open_tip = if folder { "Open the folder in the browser" } else { "Open with the default application" };
        if icons::button(ui, svg::ARROW_SQUARE_OUT, "Open").on_hover_text(open_tip).clicked() {
            if folder {
                let key = ex.table.key_for(&row.path);
                ex.open_row(&key);
            } else {
                run_on(ActionId::Open, &row.path);
            }
        }
        if icons::button(ui, svg::AME_REVEAL_IN_EXPLORER, "Reveal").on_hover_text("Show in the file manager").clicked() {
            run_on(ActionId::Reveal, &row.path);
        }
        if icons::button(ui, svg::COPY, "Copy path").clicked() {
            run_on(ActionId::CopyPath, &row.path);
        }
    });
}

fn apply_preview(insp: &mut Inspector, ex: &mut FileExplorer, event: PreviewEvent) {
    match event {
        PreviewEvent::None => {}
        PreviewEvent::OpenViewer => insp.lightbox.show(),
        PreviewEvent::Navigate { forward } => {
            navigate(ex, forward);
        }
    }
}

fn file_view(ui: &mut Ui, insp: &mut Inspector, ex: &mut FileExplorer, row: &Thumbnail) {
    let facts = insp.facts.get(&row.path).cloned();
    header(ui, ex, row, is_offline(facts.as_ref()), false);
    ui.add_space(4.0);
    let video_file = is_video_path(&row.path);
    let mut event = PreviewEvent::None;
    if video_file {
        video::show(ui, insp, ex, row, facts.as_ref());
    } else {
        video::stop(ui.ctx(), ex.viewer.scope);
        if picture(ui, insp, ex, row, facts.as_ref()) {
            event = PreviewEvent::OpenViewer;
        }
    }
    let strip = controls(ui, insp, ex, row, !video_file);
    if strip != PreviewEvent::None {
        event = strip;
    }
    apply_preview(insp, ex, event);
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        for tab in DetailTab::ALL {
            if ui.selectable_label(insp.prefs.prefs.detail_tab == tab, tab.label()).clicked() {
                insp.edit_prefs(|p| p.detail_tab = tab);
                insp.suggest_field = None;
            }
        }
    });
    ui.separator();
    match insp.prefs.prefs.detail_tab {
        DetailTab::Info => info_tab::show(ui, insp, ex, row),
        DetailTab::Metadata => meta_tab::show(ui, insp, ex, row),
        DetailTab::Ai => ai_tab::show(ui, ex, row),
        DetailTab::Edit => ai_tab::edit_tab(ui, row),
    }
}

fn folder_view(ui: &mut Ui, insp: &mut Inspector, ex: &mut FileExplorer, row: &Thumbnail) {
    let facts = insp.facts.get(&row.path).cloned();
    header(ui, ex, row, is_offline(facts.as_ref()), true);
    ui.add_space(6.0);
    info_tab::show(ui, insp, ex, row);
}

fn multi_view(ui: &mut Ui, insp: &mut Inspector, ex: &mut FileExplorer, count: usize, focus: Option<&Thumbnail>) {
    batch_tab::show(ui, insp, ex, count);
    if let Some(row) = focus.filter(|r| !is_dir(r)) {
        ui.add_space(8.0);
        ui.separator();
        ui.label(RichText::new(format!("Focus: {}", row.filename)).weak());
        let facts = insp.facts.get(&row.path).cloned();
        picture(ui, insp, ex, row, facts.as_ref());
        let strip = controls(ui, insp, ex, row, true);
        apply_preview(insp, ex, strip);
    }
}

/// Draws the rail for `target`.
pub fn show(ui: &mut Ui, insp: &mut Inspector, ex: &mut FileExplorer, env: &mut RailEnv, target: &Target) {
    tabs_row(ui, insp, env);
    if insp.prefs.prefs.rail_tab == RailTab::Ask {
        ask_tab(ui, env);
        return;
    }
    if !matches!(target, Target::Single(r) if is_video_path(&r.path) && !is_dir(r)) {
        video::stop(ui.ctx(), ex.viewer.scope);
    }
    ScrollArea::vertical().id_salt("inspector_scroll").auto_shrink([false, false]).show(ui, |ui| match target {
        Target::Empty => {
            insp.preview = Default::default();
            empty_state(ui)
        }
        Target::Single(row) if is_dir(row) => {
            insp.preview = Default::default();
            folder_view(ui, insp, ex, row)
        }
        Target::Single(row) => file_view(ui, insp, ex, row),
        Target::Multi { count, focus } => multi_view(ui, insp, ex, *count, focus.as_deref()),
    });
}
