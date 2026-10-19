//! Brings a [`Workspace`] back: the dock tree, one explorer per saved browser and the focus.

use std::collections::{HashMap, HashSet};

use eframe::egui::{pos2, vec2};
use egui_dock::{Node, NodeIndex, Split, Surface, SurfaceIndex, TabIndex, Tree};

use crate::app::SmartMediaApp;
use crate::settings::workspace::{MIN_FRACTION, SavedNode, SavedTab, SplitDir, Workspace};
use crate::ui::file_table::FileExplorer;
use crate::ui::tabs::{BrowserId, TabKind};

/// What a restore did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Applied {
    pub browsers: usize,
    pub tabs: usize,
    /// Scope of the browser that has the focus.
    pub focused: Option<String>,
}

impl Applied {
    pub fn text(&self) -> String {
        format!("{} tabs, {} browsers{}", self.tabs, self.browsers, self.focused.as_ref().map(|f| format!(", focus on {f}")).unwrap_or_default())
    }
}

/// The dock tab a saved tab stands for; `ids` maps the browsers of the file to live ones.
pub fn tab_kind(tab: &SavedTab, ids: &dyn Fn(u64) -> Option<BrowserId>) -> Option<TabKind> {
    Some(match tab {
        SavedTab::Browser { id } => TabKind::browser(ids(*id)?),
        SavedTab::Refine => TabKind::Refine,
        SavedTab::ImageEdit => TabKind::ImageEdit,
        SavedTab::Settings => TabKind::Settings,
        SavedTab::Devtools => TabKind::Devtools,
        SavedTab::Logs => TabKind::Logs,
    })
}

/// The dock tree of `node`; tabs `kind` cannot name are left out.
pub fn build_tree(node: &SavedNode, kind: &dyn Fn(&SavedTab) -> Option<TabKind>) -> Option<Tree<TabKind>> {
    let seed: Vec<TabKind> = node.seed().iter().filter_map(kind).collect();
    if seed.is_empty() {
        return None;
    }
    let mut tree = Tree::new(seed);
    fill(&mut tree, NodeIndex::root(), node, kind);
    Some(tree)
}

/// Makes the leaf at `at`, which holds the first leaf of `node`, into `node`.
fn fill(tree: &mut Tree<TabKind>, at: NodeIndex, node: &SavedNode, kind: &dyn Fn(&SavedTab) -> Option<TabKind>) {
    match node {
        SavedNode::Leaf { tabs, active } => {
            let kinds: Vec<TabKind> = tabs.iter().filter_map(kind).collect();
            if let Node::Leaf(leaf) = &mut tree[at] {
                leaf.active = TabIndex((*active).min(kinds.len().saturating_sub(1)));
                leaf.tabs = kinds;
            }
        }
        SavedNode::Split { dir, fraction, first, second } => {
            let seed: Vec<TabKind> = second.seed().iter().filter_map(kind).collect();
            if seed.is_empty() {
                fill(tree, at, first, kind);
                return;
            }
            let side = match dir {
                SplitDir::Horizontal => Split::Right,
                SplitDir::Vertical => Split::Below,
            };
            let fraction = if fraction.is_finite() { fraction.clamp(MIN_FRACTION, 1.0 - MIN_FRACTION) } else { 0.5 };
            let [old, new] = tree.split(at, side, fraction, Node::leaf_with(seed));
            fill(tree, old, first, kind);
            fill(tree, new, second, kind);
        }
    }
}

/// Stops what the explorers of `app` are doing before they are replaced.
fn stop_explorers(app: &mut SmartMediaApp) {
    app.context.file_explorer.stop_work();
    app.context.filtered_tabs.values_mut().for_each(FileExplorer::stop_work);
}

/// Replaces the tabs of `app` with those of `workspace`: the dock layout, one explorer per saved browser (read when its tab is first shown) and the focus. The window and the sidebar are not touched.
pub fn apply_full(app: &mut SmartMediaApp, workspace: &Workspace) -> Result<Applied, String> {
    if !workspace.has_tabs() {
        return Err("the workspace holds no tabs".to_string());
    }
    let mut ids: HashMap<u64, BrowserId> = HashMap::new();
    for browser in &workspace.browsers {
        ids.insert(browser.id, if browser.id == 0 { BrowserId::MAIN } else { BrowserId::next() });
    }
    let kind = |tab: &SavedTab| tab_kind(tab, &|label| ids.get(&label).copied());
    let tree = workspace.dock.as_ref().and_then(|dock| build_tree(dock, &kind)).unwrap_or_else(|| Tree::new(Vec::new()));

    stop_explorers(app);
    app.context.filtered_tabs.clear();
    let mut main_saved = false;
    for browser in &workspace.browsers {
        let Some(id) = ids.get(&browser.id).copied() else { continue };
        if id == BrowserId::MAIN {
            app.context.file_explorer.restore_state(browser);
            main_saved = true;
        } else {
            let mut explorer = FileExplorer::new(true);
            explorer.restore_state(browser);
            app.context.filtered_tabs.insert(id, explorer);
        }
    }
    if !main_saved {
        app.context.file_explorer.defer_load(Vec::new());
    }

    *app.tree.main_surface_mut() = tree;
    for index in (1..app.tree.surfaces_count()).rev() {
        app.tree.remove_surface(SurfaceIndex(index));
    }
    for window in &workspace.floating {
        let Some(tree) = build_tree(&window.dock, &kind) else { continue };
        let index = app.tree.add_window(Vec::new());
        if let Some(Surface::Window(slot, state)) = app.tree.get_surface_mut(index) {
            *slot = tree;
            if let Some([x, y]) = window.position {
                state.set_position(pos2(x, y));
            }
            if let Some([w, h]) = window.size {
                state.set_size(vec2(w, h));
            }
        }
    }
    let tabs = app.tree.iter_all_tabs().count();
    app.context.open_tabs = app.tree.iter_all_tabs().map(|(_, tab)| *tab).collect::<HashSet<TabKind>>();
    let focus = workspace.focused.as_ref().and_then(&kind).or_else(|| workspace.focused_browser().and_then(|b| ids.get(&b.id)).map(|id| TabKind::browser(*id)));
    let focused_scope = focus.and_then(|kind| {
        if let Some(path) = app.tree.find_tab(&kind) {
            let _ = app.tree.set_active_tab(path);
            app.tree.set_focused_node_and_surface(path.node_path());
        }
        app.context.active_tab = Some(kind);
        kind.browser_id().and_then(|id| app.context.explorer_mut(id)).map(|ex| ex.scope().format())
    });
    Ok(Applied { browsers: workspace.browsers.len(), tabs, focused: focused_scope })
}

/// Shows the focused browser of `workspace` in the main tab and leaves the dock as it is.
pub fn apply_scope(app: &mut SmartMediaApp, workspace: &Workspace) -> Result<Applied, String> {
    let browser = workspace.focused_browser().ok_or_else(|| "the workspace holds no browser".to_string())?;
    app.context.file_explorer.stop_work();
    app.context.file_explorer.restore_state(browser);
    Ok(Applied { browsers: 1, tabs: 0, focused: Some(browser.scope.clone()) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::workspace::capture::{capture_node, saved_tab};

    fn identity(tab: &SavedTab) -> Option<TabKind> {
        tab_kind(tab, &|label| Some(BrowserId(label)))
    }

    fn rebuilt(node: &SavedNode) -> Option<SavedNode> {
        let tree = build_tree(node, &identity)?;
        capture_node(&tree, NodeIndex::root(), &|kind| saved_tab(kind, &|id| Some(id.0)))
    }

    fn leaf(tabs: &[SavedTab], active: usize) -> SavedNode {
        SavedNode::Leaf { tabs: tabs.to_vec(), active }
    }

    fn b(id: u64) -> SavedTab {
        SavedTab::Browser { id }
    }

    fn split(dir: SplitDir, fraction: f32, first: SavedNode, second: SavedNode) -> SavedNode {
        SavedNode::Split { dir, fraction, first: Box::new(first), second: Box::new(second) }
    }

    #[test]
    fn a_single_leaf_keeps_its_tab_order_and_front_tab() {
        let node = leaf(&[b(0), SavedTab::Settings, b(1), SavedTab::Logs], 2);
        assert_eq!(rebuilt(&node), Some(node));
    }

    #[test]
    fn splits_keep_their_direction_fraction_and_sides() {
        let node = split(SplitDir::Horizontal, 0.3, leaf(&[b(0)], 0), split(SplitDir::Vertical, 0.65, leaf(&[b(1), b(2)], 1), leaf(&[SavedTab::Logs], 0)));
        assert_eq!(rebuilt(&node), Some(node));
    }

    #[test]
    fn a_tree_that_nests_on_both_sides_survives() {
        let node = split(
            SplitDir::Vertical,
            0.5,
            split(SplitDir::Horizontal, 0.25, leaf(&[b(0)], 0), split(SplitDir::Horizontal, 0.5, leaf(&[b(1)], 0), leaf(&[b(2)], 0))),
            split(SplitDir::Horizontal, 0.75, split(SplitDir::Vertical, 0.4, leaf(&[b(3)], 0), leaf(&[SavedTab::Refine], 0)), leaf(&[SavedTab::ImageEdit, SavedTab::Devtools], 1)),
        );
        assert_eq!(rebuilt(&node), Some(node));
    }

    #[test]
    fn a_long_chain_of_splits_survives() {
        let mut node = leaf(&[b(0)], 0);
        for i in 1..12u64 {
            let dir = if i % 2 == 0 { SplitDir::Horizontal } else { SplitDir::Vertical };
            node = split(dir, 0.6, node, leaf(&[b(i)], 0));
        }
        assert_eq!(rebuilt(&node), Some(node.clone()));
        let reversed = {
            let mut n = leaf(&[b(0)], 0);
            for i in 1..12u64 {
                n = split(SplitDir::Horizontal, 0.4, leaf(&[b(i)], 0), n);
            }
            n
        };
        assert_eq!(rebuilt(&reversed), Some(reversed));
    }

    #[test]
    fn the_first_side_of_a_split_is_the_left_child_and_the_second_the_right() {
        let node = split(SplitDir::Horizontal, 0.5, leaf(&[b(5), b(6)], 0), leaf(&[b(7)], 0));
        let tree = build_tree(&node, &identity).unwrap();
        let browsers = |at: usize| tree[NodeIndex(at)].tabs().unwrap().iter().filter_map(|k| k.browser_id().map(|id| id.0)).collect::<Vec<_>>();
        assert!(tree[NodeIndex::root()].is_horizontal());
        assert_eq!((browsers(1), browsers(2)), (vec![5, 6], vec![7]));
        let stacked = build_tree(&split(SplitDir::Vertical, 0.2, leaf(&[b(1)], 0), leaf(&[b(2)], 0)), &identity).unwrap();
        assert!(stacked[NodeIndex::root()].is_vertical());
    }

    #[test]
    fn a_tab_that_cannot_be_named_is_left_out_and_the_rest_is_built() {
        let node = split(SplitDir::Horizontal, 0.5, leaf(&[b(0), b(9)], 1), leaf(&[b(9)], 0));
        let only_zero = |tab: &SavedTab| tab_kind(tab, &|label| (label == 0).then_some(BrowserId(0)));
        let tree = build_tree(&node, &only_zero).unwrap();
        assert_eq!(tree.num_tabs(), 1);
        assert!(build_tree(&leaf(&[b(9)], 0), &only_zero).is_none());
    }

    #[test]
    fn an_active_index_past_the_end_falls_back_to_the_last_tab() {
        let tree = build_tree(&leaf(&[b(0), b(1)], 9), &identity).unwrap();
        match &tree[NodeIndex::root()] {
            Node::Leaf(l) => assert_eq!(l.active.0, 1),
            other => panic!("{other:?}"),
        }
    }
}
