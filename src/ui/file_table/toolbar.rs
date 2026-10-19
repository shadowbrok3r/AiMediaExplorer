//! The browser toolbar: one row with navigation, the path area, the query bar and the dialog buttons, folding into a menu as the row narrows.

use eframe::egui::containers::menu::MenuButton;
use eframe::egui::*;

use super::FileExplorer;
use super::action_bar::BarState;
use super::dialogs::{process::ProcessDialog, scan::ScanDialog};
use super::menus::breadcrumbs::PathBar;
use super::query::{ModeStyle, QueryBar};
use crate::command::DialogKind;
use crate::icons::{self, IconSize, svg};

/// Narrowest path area the plans keep.
pub const PATH_MIN: f32 = 120.0;

/// State of the toolbar and everything that hangs off it in one tab.
pub struct Toolbar {
    pub query: QueryBar,
    pub path: PathBar,
    pub bar: BarState,
    pub scan: ScanDialog,
    pub process: ProcessDialog,
    /// Where the parts were drawn in the last frame.
    pub layout: Option<ToolbarLayout>,
}

impl Default for Toolbar {
    fn default() -> Self {
        Self { query: QueryBar::default(), path: PathBar::default(), bar: BarState::default(), scan: ScanDialog::default(), process: ProcessDialog::default(), layout: None }
    }
}

/// How a button of the row is drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Btn {
    /// Only in the More menu.
    Menu,
    Icon,
    Text,
}

/// Sizes the plan needs, read from the style.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sizes {
    /// Height of the controls.
    pub side: f32,
    /// Width of an icon button.
    pub icon_w: f32,
    pub gap: f32,
    pub scan_text: f32,
    pub process_text: f32,
    pub mode_labels: f32,
    pub mode_icons: f32,
}

/// What the row shows at a given width.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plan {
    /// Navigation buttons drawn: Back, Forward, Up, Refresh, Home in this order; the rest are in the More menu.
    pub nav: usize,
    pub mode: ModeStyle,
    pub text_w: f32,
    pub scan: Btn,
    pub process: Btn,
}

impl Plan {
    /// Width of the buttons at the right end, More included.
    pub fn actions_w(&self, s: &Sizes) -> f32 {
        let width = |b: Btn, text: f32| match b {
            Btn::Menu => 0.0,
            Btn::Icon => s.icon_w,
            Btn::Text => text,
        };
        let shown: Vec<f32> = [width(self.scan, s.scan_text), width(self.process, s.process_text)].into_iter().filter(|w| *w > 0.0).collect();
        s.icon_w + shown.iter().map(|w| w + s.gap).sum::<f32>()
    }

    /// Width of the query bar: the mode control, the field and its clear button.
    pub fn query_w(&self, s: &Sizes) -> f32 {
        let mode = match self.mode {
            ModeStyle::Labels => s.mode_labels,
            ModeStyle::Icons => s.mode_icons,
            ModeStyle::Menu => s.icon_w,
        };
        mode + s.gap + self.text_w + s.gap + s.icon_w
    }

    /// Width of the navigation buttons.
    pub fn nav_w(&self, s: &Sizes) -> f32 {
        self.nav as f32 * s.icon_w + self.nav.saturating_sub(1) as f32 * s.gap
    }

    /// Width the row needs for this plan with the narrowest path area.
    pub fn needs(&self, s: &Sizes) -> f32 {
        self.nav_w(s) + s.gap + PATH_MIN + s.gap + self.query_w(s) + s.gap + self.actions_w(s)
    }
}

/// The richest plan that fits `width`, else the tightest.
pub fn plan_for(width: f32, s: &Sizes) -> Plan {
    let tiers = [
        Plan { nav: 5, mode: ModeStyle::Labels, text_w: (width * 0.2).clamp(180.0, 280.0), scan: Btn::Text, process: Btn::Text },
        Plan { nav: 5, mode: ModeStyle::Icons, text_w: 170.0, scan: Btn::Icon, process: Btn::Icon },
        Plan { nav: 4, mode: ModeStyle::Menu, text_w: 150.0, scan: Btn::Icon, process: Btn::Menu },
        Plan { nav: 4, mode: ModeStyle::Menu, text_w: 110.0, scan: Btn::Menu, process: Btn::Menu },
        Plan { nav: 3, mode: ModeStyle::Menu, text_w: 90.0, scan: Btn::Menu, process: Btn::Menu },
    ];
    tiers.into_iter().find(|p| p.needs(s) <= width).unwrap_or(tiers[4])
}

/// Where the parts of the row were drawn, for tests.
#[derive(Clone, Debug)]
pub struct ToolbarLayout {
    pub row: Rect,
    pub nav: Vec<(&'static str, Rect)>,
    pub path: Rect,
    pub mode: Rect,
    pub text: Rect,
    pub actions: Vec<(&'static str, Rect)>,
    pub more: Rect,
}

impl Default for ToolbarLayout {
    fn default() -> Self {
        Self { row: Rect::ZERO, nav: Vec::new(), path: Rect::ZERO, mode: Rect::ZERO, text: Rect::ZERO, actions: Vec::new(), more: Rect::ZERO }
    }
}

impl ToolbarLayout {
    /// Every rectangle drawn, with a name.
    pub fn parts(&self) -> Vec<(String, Rect)> {
        let mut out: Vec<(String, Rect)> = self.nav.iter().map(|(n, r)| (n.to_string(), *r)).collect();
        out.push(("path".to_string(), self.path));
        out.push(("mode".to_string(), self.mode));
        out.push(("text".to_string(), self.text));
        out.extend(self.actions.iter().map(|(n, r)| (n.to_string(), *r)));
        out.push(("more".to_string(), self.more));
        out
    }
}

fn text_width(ui: &Ui, text: &str) -> f32 {
    let font = TextStyle::Button.resolve(ui.style());
    ui.painter().layout_no_wrap(text.to_string(), font, Color32::PLACEHOLDER).size().x
}

/// Width of an icon-then-text button.
fn button_width(ui: &Ui, text: &str) -> f32 {
    IconSize::Md.px(ui) + 5.0 + text_width(ui, text) + ui.spacing().button_padding.x * 2.0 + 4.0
}

/// The sizes the row works with in `ui`'s style.
pub fn sizes(ui: &Ui) -> Sizes {
    let side = ui.spacing().interact_size.y.max(IconSize::Md.px(ui) + 4.0);
    let icon_w = side.max(IconSize::Md.px(ui) + 2.0 * ui.spacing().button_padding.x);
    let gap = ui.spacing().item_spacing.x;
    let segment = |label: &str| text_width(ui, label) + IconSize::Sm.px(ui) + 4.0 + 22.0;
    Sizes {
        side,
        icon_w,
        gap,
        scan_text: button_width(ui, "Scan"),
        process_text: button_width(ui, "Process"),
        mode_labels: crate::command::QueryMode::ALL.iter().map(|m| segment(m.label())).sum::<f32>() + 8.0,
        mode_icons: 4.0 * (IconSize::Sm.px(ui) + 24.0) + 8.0,
    }
}

impl FileExplorer {
    /// Draws the toolbar row and applies what the user did in it.
    pub fn toolbar(&mut self, ui: &mut Ui) {
        self.run_shortcuts(ui);
        self.run_query_dev(ui.ctx());
        self.run_actions_dev(ui.ctx());
        let s = sizes(ui);
        let metrics = crate::theme::metrics(ui.ctx());
        let height = s.side.max(metrics.control_h) + 6.0;
        Panel::top("file_explorer_toolbar").exact_size(height).show_separator_line(false).frame(Frame::NONE.inner_margin(Margin::symmetric(4, 3))).show(ui, |ui| {
            let row = ui.available_rect_before_wrap();
            let plan = plan_for(row.width(), &s);
            let layout = self.draw_row(ui, row, &plan, &s);
            self.toolbar.layout = Some(layout);
        });
    }

    fn draw_row(&mut self, ui: &mut Ui, row: Rect, plan: &Plan, s: &Sizes) -> ToolbarLayout {
        let mut layout = ToolbarLayout { row, ..Default::default() };
        let gap = s.gap;
        let actions_w = plan.actions_w(s);
        let actions_rect = Rect::from_min_max(pos2(row.right() - actions_w, row.top()), row.right_bottom());
        let query_rect = Rect::from_min_max(pos2(actions_rect.left() - gap - plan.query_w(s), row.top()), pos2(actions_rect.left() - gap, row.bottom()));
        let left_rect = Rect::from_min_max(row.left_top(), pos2((query_rect.left() - gap).max(row.left()), row.bottom()));

        let child = |rect: Rect, layout: Layout| UiBuilder::new().max_rect(rect).layout(layout);

        ui.scope_builder(child(actions_rect, Layout::right_to_left(Align::Center)), |ui| self.draw_actions(ui, plan, &mut layout));
        ui.scope_builder(child(query_rect, Layout::left_to_right(Align::Center)), |ui| {
            let rects = super::query::show(ui, self, plan.mode);
            layout.mode = rects.mode;
            layout.text = rects.text;
        });
        ui.scope_builder(child(left_rect, Layout::left_to_right(Align::Center)), |ui| {
            self.draw_nav(ui, plan, &mut layout);
            let path = Rect::from_min_max(pos2(ui.cursor().left(), row.top()), pos2(left_rect.right(), row.bottom()));
            ui.scope_builder(child(path, Layout::left_to_right(Align::Center)), |ui| {
                layout.path = self.path_area(ui);
            });
        });
        layout
    }

    fn draw_nav(&mut self, ui: &mut Ui, plan: &Plan, layout: &mut ToolbarLayout) {
        let (can_up, can_back, can_forward) = (self.can_go_up(), self.can_go_back(), self.can_go_forward());
        let steps: [(&'static str, icons::Icon, &str, bool); 5] = [
            ("back", svg::ARROW_LEFT, "Back (Alt+Left)", can_back),
            ("forward", svg::ARROW_RIGHT, "Forward (Alt+Right)", can_forward),
            ("up", svg::ARROW_UP, "Up (Alt+Up)", can_up),
            ("refresh", svg::ARROW_CLOCKWISE, "Refresh (F5)", true),
            ("home", svg::HOUSE_SIMPLE, "Home", true),
        ];
        for (name, icon, tip, enabled) in steps.into_iter().take(plan.nav) {
            let response = ui.add_enabled_ui(enabled, |ui| icons::icon_button(ui, icon, tip)).inner;
            layout.nav.push((name, response.rect));
            if response.clicked() {
                match name {
                    "back" => self.nav_back(),
                    "forward" => self.nav_forward(),
                    "up" => self.nav_up(),
                    "refresh" => self.refresh(),
                    _ => self.nav_home(),
                }
            }
        }
    }

    /// Draws the buttons at the right end, rightmost first.
    fn draw_actions(&mut self, ui: &mut Ui, plan: &Plan, layout: &mut ToolbarLayout) {
        let more = icons::icon_menu_button(ui, svg::DOTS_THREE, "More", |ui| self.more_items(ui, plan));
        layout.more = more.response.rect;
        for (name, kind, icon, text, tip, dialog) in [
            ("process", plan.process, svg::AME_MODEL, "Process", "Describe or generate CLIP embeddings for the selection, this view or a folder", DialogKind::Process),
            ("scan", plan.scan, svg::AME_SCAN, "Scan", "Scan a folder and its subfolders", DialogKind::Scan),
        ] {
            let response = match kind {
                Btn::Menu => continue,
                Btn::Icon => icons::icon_button(ui, icon, &format!("{text}...: {tip}")),
                Btn::Text => icons::button(ui, icon, text).on_hover_text(tip),
            };
            layout.actions.push((name, response.rect));
            if response.clicked() {
                let ctx = ui.ctx().clone();
                let _ = self.open_dialog(&ctx, dialog, None);
            }
        }
    }

    fn more_items(&mut self, ui: &mut Ui, plan: &Plan) {
        let ctx = ui.ctx().clone();
        if plan.nav < 5 {
            if plan.nav < 4 && icons::button(ui, svg::ARROW_CLOCKWISE, "Refresh").clicked() {
                self.refresh();
                ui.close();
            }
            if icons::button(ui, svg::HOUSE_SIMPLE, "Home").clicked() {
                self.nav_home();
                ui.close();
            }
        }
        if plan.scan == Btn::Menu && icons::button(ui, svg::AME_SCAN, "Scan folder...").clicked() {
            let _ = self.open_dialog(&ctx, DialogKind::Scan, None);
            ui.close();
        }
        if plan.process == Btn::Menu && icons::button(ui, svg::AME_MODEL, "Process files...").clicked() {
            let _ = self.open_dialog(&ctx, DialogKind::Process, None);
            ui.close();
        }
        if plan.nav < 5 || plan.scan == Btn::Menu || plan.process == Btn::Menu {
            ui.separator();
        }
        if icons::button(ui, svg::AME_SCAN, "Scan this folder...").on_hover_text("Open the scan dialog for the folder shown").clicked() {
            let folder = self.nav.scope.is_filesystem().then(|| self.current_path.clone());
            let _ = self.open_dialog(&ctx, DialogKind::Scan, folder);
            ui.close();
        }
        if icons::button(ui, svg::CLIPBOARD_TEXT, "Copy address").clicked() {
            ctx.copy_text(if self.nav.scope.is_filesystem() { self.current_path.clone() } else { self.nav.scope.format() });
            ui.close();
        }
        let on_disk = self.nav.scope.is_filesystem() && !self.current_path.contains("://");
        if ui.add_enabled(on_disk, icons::button_widget(ui, svg::FOLDER_OPEN, "Open in the file manager")).clicked() {
            let path = self.current_path.clone();
            if let Err(e) = open::that_detached(&path) {
                crate::ui::dialogs::toast(egui_toast::ToastKind::Error, format!("Could not open {path}: {e}"));
            }
            ui.close();
        }
        if icons::button(ui, svg::MAGNIFYING_GLASS, "Search (Ctrl+F)").clicked() {
            self.toolbar.query.focus = true;
            ui.close();
        }
    }

    /// Text of the status bar menu that opens the scan and process dialogs.
    pub const DIALOGS_LABEL: &'static str = "Process / Scan";

    /// Menu button for the status bar: the scan and process dialogs.
    pub fn dialogs_menu(&mut self, ui: &mut Ui) {
        let ctx = ui.ctx().clone();
        MenuButton::new(Self::DIALOGS_LABEL).ui(ui, |ui| {
            if icons::button(ui, svg::AME_SCAN, "Scan folder...").clicked() {
                let _ = self.open_dialog(&ctx, DialogKind::Scan, None);
                ui.close();
            }
            if icons::button(ui, svg::AME_MODEL, "Process files...").clicked() {
                let _ = self.open_dialog(&ctx, DialogKind::Process, None);
                ui.close();
            }
        });
    }

    /// Opens the scan or process dialog of this tab; `folder` fills the folder field.
    pub fn open_dialog(&mut self, ctx: &Context, dialog: DialogKind, folder: Option<String>) -> crate::command::CommandResult {
        match dialog {
            DialogKind::Scan => {
                let start = folder.clone().or_else(|| self.nav.scope.is_filesystem().then(|| self.current_path.clone()));
                let scope = self.nav.scope.clone();
                self.toolbar.scan.open_for(start, &scope);
            }
            DialogKind::Process => {
                let start = folder.clone();
                self.toolbar.process.open_for(start);
            }
        }
        ctx.request_repaint();
        Ok(format!("{} dialog opened", dialog.label()))
    }
}

#[cfg(test)]
mod tests;
