use egui_dock::{DockState, NodePath, SurfaceIndex, tab_viewer::OnCloseResponse};
use eframe::egui::*;
use std::collections::HashSet;

pub const TABS: [&str; 3] = [
    "File Explorer",
    "Logs",
    "AI Refinements",
];

impl egui_dock::TabViewer for crate::app::SmartMediaContext {
    type Tab = String;

    fn id(&mut self, tab: &mut Self::Tab) -> Id {
        Id::unique(tab.as_str())
    }

    fn title(&mut self, tab: &mut Self::Tab) -> WidgetText {
        tab.as_str().into()
    }

    fn ui(&mut self, ui: &mut Ui, tab: &mut Self::Tab) {
        match tab.as_str() {
            "File Explorer" => self.file_explorer.ui(ui),
            "AI Refinements" => self.refinements.ui(ui),
            "Image Edit" => self.image_edit.ui(ui),
            "Logs" => egui_logger::logger_ui()
                .warn_color(crate::theme::warn(ui))
                .error_color(crate::theme::error(ui))
                .log_levels([true, true, true, false, false])
                .enable_category("eframe".to_string(), false)
                .enable_category("eframe::native::glow_integration".to_string(), false)
                .enable_category("egui_glow::shader_version".to_string(), false)
                .enable_category("egui_glow::painter".to_string(), false)
                .max_log_length(200)
                .show(ui),
            _ => {
                // Dynamic tabs: delegate to filtered explorers if present
                if let Some(ex) = self.filtered_tabs.get_mut(tab) {
                    ex.ui(ui);
                } else {
                    ui.label(tab.as_str());
                }
            }
        }
    }

    fn context_menu(
        &mut self,
        _ui: &mut Ui,
        tab: &mut Self::Tab,
        _path: NodePath,
    ) {
        match tab.as_str() {

            _ => {

            }
        }
    }

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
    self.filtered_tabs.remove(tab);
        OnCloseResponse::Close
    }
}

/// Focuses `title` when it is open, otherwise opens it in the focused leaf.
pub fn focus_or_open_tab(tree: &mut DockState<String>, open_tabs: &mut HashSet<String>, title: &str) {
    let tab = title.to_string();
    if let Some(path) = tree.find_tab(&tab) {
        let _ = tree.set_active_tab(path);
        tree.set_focused_node_and_surface(path.node_path());
    } else {
        tree[SurfaceIndex::main()].push_to_focused_leaf(tab.clone());
        open_tabs.insert(tab);
    }
}

/// Opens `title` in the focused leaf; with `background` the previously focused tab stays in front.
pub fn open_tab(tree: &mut DockState<String>, open_tabs: &mut HashSet<String>, title: &str, background: bool) {
    let previous = background.then(|| tree.find_active_focused().map(|(_, tab)| tab.clone())).flatten();
    tree[SurfaceIndex::main()].push_to_focused_leaf(title.to_string());
    open_tabs.insert(title.to_string());
    if let Some(path) = previous.and_then(|tab| tree.find_tab(&tab)) {
        let _ = tree.set_active_tab(path);
        tree.set_focused_node_and_surface(path.node_path());
    }
}

/// `base`, or `base (2)`, `base (3)`, ... when the title is already open.
pub fn uniquify_title(open_tabs: &HashSet<String>, base: &str) -> String {
    if !open_tabs.contains(base) {
        return base.to_string();
    }
    let mut idx = 2;
    loop {
        let cand = format!("{} ({})", base, idx);
        if !open_tabs.contains(&cand) {
            return cand;
        }
        idx += 1;
        if idx > 9999 { return format!("{} ({})", base, idx); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui_dock::NodeIndex;

    fn dock() -> (DockState<String>, HashSet<String>) {
        let mut tree = DockState::new(vec!["File Explorer".to_owned(), "Logs".to_owned()]);
        tree.set_focused_node_and_surface(NodePath { surface: SurfaceIndex::main(), node: NodeIndex::root() });
        let open = ["File Explorer", "Logs"].iter().map(|t| t.to_string()).collect();
        (tree, open)
    }

    fn titles(tree: &DockState<String>) -> Vec<String> {
        tree.iter_all_tabs().map(|(_, t)| t.clone()).collect()
    }

    fn active(tree: &mut DockState<String>) -> String {
        tree.find_active_focused().map(|(_, t)| t.clone()).unwrap_or_default()
    }

    #[test]
    fn a_second_open_of_a_tool_tab_focuses_the_existing_tab() {
        let (mut tree, mut open) = dock();
        focus_or_open_tab(&mut tree, &mut open, "Image Edit");
        assert_eq!(titles(&tree).iter().filter(|t| t.starts_with("Image Edit")).count(), 1);
        focus_or_open_tab(&mut tree, &mut open, "File Explorer");
        assert_eq!(active(&mut tree), "File Explorer");
        focus_or_open_tab(&mut tree, &mut open, "Image Edit");
        assert_eq!(titles(&tree).iter().filter(|t| t.starts_with("Image Edit")).count(), 1, "no duplicate or stub tab");
        assert_eq!(active(&mut tree), "Image Edit", "the existing tab takes focus");
    }

    #[test]
    fn a_foreground_open_takes_focus_and_a_background_open_does_not() {
        let (mut tree, mut open) = dock();
        open_tab(&mut tree, &mut open, "Scan: a", false);
        assert_eq!(active(&mut tree), "Scan: a");
        open_tab(&mut tree, &mut open, "Scan: b", true);
        assert_eq!(active(&mut tree), "Scan: a", "background open leaves focus alone");
        assert!(titles(&tree).contains(&"Scan: b".to_string()));
        assert!(open.contains("Scan: b"));
    }

    #[test]
    fn titles_gain_a_counter_only_when_taken() {
        let open: HashSet<String> = ["Scan: a", "Scan: a (2)"].iter().map(|t| t.to_string()).collect();
        assert_eq!(uniquify_title(&open, "Scan: b"), "Scan: b");
        assert_eq!(uniquify_title(&open, "Scan: a"), "Scan: a (3)");
    }
}
