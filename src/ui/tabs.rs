use egui_dock::{DockState, NodePath, SurfaceIndex, tab_viewer::OnCloseResponse};
use eframe::egui::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::command::Tool;

/// Identity of one browser tab.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BrowserId(pub u64);

static NEXT_BROWSER: AtomicU64 = AtomicU64::new(1);

impl BrowserId {
    /// The browser that is always there at first.
    pub const MAIN: BrowserId = BrowserId(0);

    /// A fresh id.
    pub fn next() -> BrowserId {
        BrowserId(NEXT_BROWSER.fetch_add(1, Ordering::Relaxed))
    }
}

/// What a dock tab holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TabKind {
    Browser { id: BrowserId },
    Refine,
    ImageEdit,
    Settings,
    Devtools,
    Logs,
}

impl TabKind {
    /// The first browser.
    pub const MAIN: TabKind = TabKind::Browser { id: BrowserId::MAIN };

    /// Tabs the View menu can switch on and off.
    pub const TOGGLED: [TabKind; 3] = [TabKind::MAIN, TabKind::Logs, TabKind::Refine];

    pub fn browser(id: BrowserId) -> TabKind {
        TabKind::Browser { id }
    }

    pub fn browser_id(&self) -> Option<BrowserId> {
        match self {
            TabKind::Browser { id } => Some(*id),
            _ => None,
        }
    }

    /// Title of tabs that do not show a scope.
    pub fn fixed_title(&self) -> &'static str {
        match self {
            TabKind::Browser { .. } => "Browser",
            TabKind::Refine => "AI Refinements",
            TabKind::ImageEdit => "Image Edit",
            TabKind::Settings => "Settings",
            TabKind::Devtools => "Devtools",
            TabKind::Logs => "Logs",
        }
    }

    /// Name in the View menu.
    pub fn menu_label(&self) -> &'static str {
        match self {
            TabKind::Browser { .. } => "File Explorer",
            other => other.fixed_title(),
        }
    }
}

impl From<Tool> for TabKind {
    fn from(tool: Tool) -> Self {
        match tool {
            Tool::ImageEdit => TabKind::ImageEdit,
            Tool::Refinements => TabKind::Refine,
            Tool::Logs => TabKind::Logs,
        }
    }
}

impl egui_dock::TabViewer for crate::app::SmartMediaContext {
    type Tab = TabKind;

    fn id(&mut self, tab: &mut Self::Tab) -> Id {
        Id::unique(("ame_dock_tab", *tab))
    }

    fn title(&mut self, tab: &mut Self::Tab) -> WidgetText {
        match tab {
            TabKind::Browser { id } => self.explorer_mut(*id).map(|ex| ex.tab_title()).unwrap_or_else(|| tab.fixed_title().to_string()).into(),
            TabKind::Settings if self.settings_page.is_dirty() => "Settings (unsaved)".into(),
            other => other.fixed_title().into(),
        }
    }

    fn ui(&mut self, ui: &mut Ui, tab: &mut Self::Tab) {
        match *tab {
            TabKind::Browser { id } => match self.explorer_mut(id) {
                Some(ex) => ex.ui(ui),
                None => {
                    ui.label("This tab has no content.");
                }
            },
            TabKind::Refine => self.refinements.ui(ui),
            TabKind::ImageEdit => self.image_edit.ui(ui),
            TabKind::Settings => crate::ui::settings::show(self, ui),
            TabKind::Devtools => {
                ui.add_space(8.0);
                ui.label("Debug windows open from View > Devtools.");
            }
            TabKind::Logs => {
                egui_logger::logger_ui()
                    .warn_color(crate::theme::warn(ui))
                    .error_color(crate::theme::error(ui))
                    .log_levels([true, true, true, false, false])
                    .enable_category("eframe".to_string(), false)
                    .enable_category("eframe::native::glow_integration".to_string(), false)
                    .enable_category("egui_glow::shader_version".to_string(), false)
                    .enable_category("egui_glow::painter".to_string(), false)
                    .max_log_length(200)
                    .show(ui);
            }
        }
    }

    fn context_menu(&mut self, _ui: &mut Ui, _tab: &mut Self::Tab, _path: NodePath) {}

    fn is_closeable(&self, _: &Self::Tab) -> bool { true }

    fn on_add(&mut self, path: NodePath) {
        self.pending_tab_adds.push(path);
    }

    // File explorer / logs / etc. own their scroll; dock-level scroll hides the path bar.
    fn scroll_bars(&self, _tab: &Self::Tab) -> [bool; 2] {
        [false, false]
    }

    fn on_close(&mut self, tab: &mut Self::Tab) -> OnCloseResponse {
        self.open_tabs.remove(tab);
        if let Some(id) = tab.browser_id() {
            self.filtered_tabs.remove(&id);
        }
        OnCloseResponse::Close
    }
}

/// Focuses `kind` when it is open, otherwise opens it in the focused leaf.
pub fn focus_or_open_tab(tree: &mut DockState<TabKind>, open_tabs: &mut HashSet<TabKind>, kind: TabKind) {
    if let Some(path) = tree.find_tab(&kind) {
        let _ = tree.set_active_tab(path);
        tree.set_focused_node_and_surface(path.node_path());
    } else {
        tree[SurfaceIndex::main()].push_to_focused_leaf(kind);
        open_tabs.insert(kind);
    }
}

/// Opens `kind` in the focused leaf; with `background` the previously focused tab stays in front.
pub fn open_tab(tree: &mut DockState<TabKind>, open_tabs: &mut HashSet<TabKind>, kind: TabKind, background: bool) {
    let previous = background.then(|| tree.find_active_focused().map(|(_, tab)| *tab)).flatten();
    tree[SurfaceIndex::main()].push_to_focused_leaf(kind);
    open_tabs.insert(kind);
    if let Some(path) = previous.and_then(|tab| tree.find_tab(&tab)) {
        let _ = tree.set_active_tab(path);
        tree.set_focused_node_and_surface(path.node_path());
    }
}

/// The target of a tab-opening request: its own when it names one, a new tab for Ctrl or middle click, else the focused tab.
pub fn request_target(explicit: Option<crate::command::Target>, ctrl_or_middle: bool) -> crate::command::Target {
    match (explicit, ctrl_or_middle) {
        (Some(target), _) => target,
        (None, true) => crate::command::Target::NewTab,
        (None, false) => crate::command::Target::Current,
    }
}

/// Switches keyboard navigation: Alt+Left and Alt+Right move through history, Alt+Up goes up, and the mouse back and forward buttons do the same.
pub fn read_nav_keys(ctx: &Context) -> Option<crate::command::NavOp> {
    use crate::command::NavOp;
    if ctx.egui_wants_keyboard_input() {
        return None;
    }
    ctx.input_mut(|i| {
        if i.consume_key(Modifiers::ALT, Key::ArrowLeft) || i.pointer.button_pressed(PointerButton::Extra1) {
            Some(NavOp::Back)
        } else if i.consume_key(Modifiers::ALT, Key::ArrowRight) || i.pointer.button_pressed(PointerButton::Extra2) {
            Some(NavOp::Forward)
        } else if i.consume_key(Modifiers::ALT, Key::ArrowUp) {
            Some(NavOp::Up)
        } else {
            None
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::Target;
    use egui_dock::NodeIndex;

    fn dock() -> (DockState<TabKind>, HashSet<TabKind>) {
        let mut tree = DockState::new(vec![TabKind::MAIN, TabKind::Logs]);
        tree.set_focused_node_and_surface(NodePath { surface: SurfaceIndex::main(), node: NodeIndex::root() });
        (tree, HashSet::from([TabKind::MAIN, TabKind::Logs]))
    }

    fn kinds(tree: &DockState<TabKind>) -> Vec<TabKind> {
        tree.iter_all_tabs().map(|(_, t)| *t).collect()
    }

    fn active(tree: &mut DockState<TabKind>) -> Option<TabKind> {
        tree.find_active_focused().map(|(_, t)| *t)
    }

    #[test]
    fn a_second_open_of_a_tool_tab_focuses_the_existing_tab() {
        let (mut tree, mut open) = dock();
        focus_or_open_tab(&mut tree, &mut open, TabKind::ImageEdit);
        assert_eq!(kinds(&tree).iter().filter(|t| **t == TabKind::ImageEdit).count(), 1);
        focus_or_open_tab(&mut tree, &mut open, TabKind::MAIN);
        assert_eq!(active(&mut tree), Some(TabKind::MAIN));
        focus_or_open_tab(&mut tree, &mut open, TabKind::ImageEdit);
        assert_eq!(kinds(&tree).iter().filter(|t| **t == TabKind::ImageEdit).count(), 1, "no duplicate or stub tab");
        assert_eq!(active(&mut tree), Some(TabKind::ImageEdit), "the existing tab takes focus");
    }

    #[test]
    fn a_foreground_open_takes_focus_and_a_background_open_does_not() {
        let (mut tree, mut open) = dock();
        let (a, b) = (TabKind::browser(BrowserId::next()), TabKind::browser(BrowserId::next()));
        open_tab(&mut tree, &mut open, a, false);
        assert_eq!(active(&mut tree), Some(a));
        open_tab(&mut tree, &mut open, b, true);
        assert_eq!(active(&mut tree), Some(a), "background open leaves focus alone");
        assert!(kinds(&tree).contains(&b));
        assert!(open.contains(&b));
    }

    #[test]
    fn browser_ids_are_unique_and_the_main_one_is_reserved() {
        let (a, b) = (BrowserId::next(), BrowserId::next());
        assert_ne!(a, b);
        assert_ne!(a, BrowserId::MAIN);
        assert_ne!(b, BrowserId::MAIN);
        assert_eq!(TabKind::MAIN.browser_id(), Some(BrowserId::MAIN));
        assert_eq!(TabKind::Logs.browser_id(), None);
    }

    #[test]
    fn tool_tabs_have_fixed_titles_and_the_tools_map_onto_them() {
        assert_eq!(TabKind::from(Tool::ImageEdit), TabKind::ImageEdit);
        assert_eq!(TabKind::from(Tool::Refinements), TabKind::Refine);
        assert_eq!(TabKind::from(Tool::Logs), TabKind::Logs);
        for kind in [TabKind::Refine, TabKind::ImageEdit, TabKind::Settings, TabKind::Devtools, TabKind::Logs, TabKind::MAIN] {
            assert!(!kind.fixed_title().is_empty() && !kind.menu_label().is_empty());
        }
        assert_eq!(TabKind::Refine.fixed_title(), Tool::Refinements.tab_title());
        assert_eq!(TabKind::ImageEdit.fixed_title(), Tool::ImageEdit.tab_title());
        assert_eq!(TabKind::Logs.fixed_title(), Tool::Logs.tab_title());
    }

    #[test]
    fn a_plain_request_replaces_the_tab_and_a_modified_click_opens_one() {
        assert_eq!(request_target(None, false), Target::Current);
        assert_eq!(request_target(None, true), Target::NewTab);
        assert_eq!(request_target(Some(Target::Background), false), Target::Background, "a named target is kept");
        assert_eq!(request_target(Some(Target::Current), true), Target::Current);
    }

    #[test]
    fn tab_kinds_round_trip_through_json() {
        for kind in [TabKind::MAIN, TabKind::browser(BrowserId(7)), TabKind::Logs, TabKind::Settings] {
            let json = serde_json::to_string(&kind).unwrap();
            assert_eq!(serde_json::from_str::<TabKind>(&json).unwrap(), kind);
        }
    }

    #[test]
    fn alt_arrows_and_mouse_buttons_read_as_navigation() {
        use crate::command::NavOp;
        let ctx = crate::ui::test_support::test_ctx();
        let press = |key: Key| {
            let ev = Event::Key { key, physical_key: Some(key), pressed: true, repeat: false, modifiers: Modifiers::ALT };
            let mut raw = RawInput::default();
            raw.events.push(Event::ModifiersChanged(Modifiers::ALT));
            raw.events.push(ev);
            let mut got = None;
            let out = ctx.run_ui(raw, |_| got = read_nav_keys(&ctx));
            out.drop_without_applying_deltas();
            got
        };
        assert_eq!(press(Key::ArrowLeft), Some(NavOp::Back));
        assert_eq!(press(Key::ArrowRight), Some(NavOp::Forward));
        assert_eq!(press(Key::ArrowUp), Some(NavOp::Up));
        assert_eq!(press(Key::ArrowDown), None);
    }
}
