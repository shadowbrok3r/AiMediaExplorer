//! What is open now, as a [`Workspace`]: the dock tree, the state of every browser tab and the window.

use std::collections::HashMap;

use eframe::egui::{Context, Id, Rect};
use egui_dock::{Node, NodeIndex, Surface, SurfaceIndex, Tree};

use crate::app::SmartMediaApp;
use crate::settings::workspace::{BrowserState, FloatingWindow, SavedNode, SavedTab, SplitDir, WindowState, Workspace};
use crate::ui::tabs::{BrowserId, TabKind};

/// What egui reports of the window.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WindowInfo {
    pub inner: Option<Rect>,
    pub outer: Option<Rect>,
    pub maximized: bool,
    pub fullscreen: bool,
    pub minimized: bool,
    /// Size of the monitor the window is on.
    pub monitor: Option<[f32; 2]>,
}

impl WindowInfo {
    /// Reads the root viewport of `ctx`.
    pub fn read(ctx: &Context) -> Self {
        ctx.input(|i| {
            let v = i.viewport();
            Self {
                inner: v.inner_rect,
                outer: v.outer_rect,
                maximized: v.maximized.unwrap_or(false),
                fullscreen: v.fullscreen.unwrap_or(false),
                minimized: v.minimized.unwrap_or(false),
                monitor: v.monitor_size.map(|s| [s.x, s.y]),
            }
        })
    }
}

/// The window to save: its size and place while it is a normal window, those it had last while it is maximized, nothing new while it is minimized or fullscreen. `last` keeps the normal geometry between calls.
pub fn window_state(info: &WindowInfo, last: &mut Option<WindowState>) -> Option<WindowState> {
    if info.minimized || info.fullscreen {
        return *last;
    }
    let mut state = last.unwrap_or(WindowState { size: [0.0, 0.0], position: None, maximized: false });
    if !info.maximized || last.is_none() {
        if let Some(inner) = info.inner {
            state.size = [inner.width(), inner.height()];
        }
        if let Some(outer) = info.outer {
            state.position = Some([outer.min.x, outer.min.y]);
        }
    }
    state.maximized = info.maximized;
    if state.size[0] <= 0.0 || state.size[1] <= 0.0 {
        return *last;
    }
    *last = Some(state);
    *last
}

/// The saved form of a dock tab; `label` names a browser in the file, `None` for a browser that is not saved.
pub fn saved_tab(kind: &TabKind, label: &dyn Fn(BrowserId) -> Option<u64>) -> Option<SavedTab> {
    Some(match kind {
        TabKind::Browser { id } => SavedTab::Browser { id: label(*id)? },
        TabKind::Refine => SavedTab::Refine,
        TabKind::ImageEdit => SavedTab::ImageEdit,
        TabKind::Settings => SavedTab::Settings,
        TabKind::Devtools => SavedTab::Devtools,
        TabKind::Logs => SavedTab::Logs,
    })
}

/// The node of `tree` at `at` and everything below it, with the tabs `keep` names.
pub fn capture_node(tree: &Tree<TabKind>, at: NodeIndex, keep: &dyn Fn(&TabKind) -> Option<SavedTab>) -> Option<SavedNode> {
    if at.0 >= tree.len() {
        return None;
    }
    match &tree[at] {
        Node::Empty => None,
        Node::Leaf(leaf) => {
            let kept: Vec<(usize, SavedTab)> = leaf.tabs.iter().enumerate().filter_map(|(i, t)| keep(t).map(|s| (i, s))).collect();
            if kept.is_empty() {
                return None;
            }
            let active = kept.iter().position(|(i, _)| *i == leaf.active.0).unwrap_or(0);
            Some(SavedNode::Leaf { tabs: kept.into_iter().map(|(_, s)| s).collect(), active })
        }
        Node::Horizontal(split) | Node::Vertical(split) => {
            let dir = if tree[at].is_horizontal() { SplitDir::Horizontal } else { SplitDir::Vertical };
            match (capture_node(tree, at.left(), keep), capture_node(tree, at.right(), keep)) {
                (Some(first), Some(second)) => Some(SavedNode::Split { dir, fraction: split.fraction, first: Box::new(first), second: Box::new(second) }),
                (Some(only), None) | (None, Some(only)) => Some(only),
                (None, None) => None,
            }
        }
    }
}

/// The browser tabs of the main surface in layout order.
fn browsers_in_order(tree: &Tree<TabKind>) -> Vec<BrowserId> {
    fn walk(tree: &Tree<TabKind>, at: NodeIndex, out: &mut Vec<BrowserId>) {
        if at.0 >= tree.len() {
            return;
        }
        match &tree[at] {
            Node::Empty => {}
            Node::Leaf(leaf) => out.extend(leaf.tabs.iter().filter_map(TabKind::browser_id)),
            Node::Horizontal(_) | Node::Vertical(_) => {
                walk(tree, at.left(), out);
                walk(tree, at.right(), out);
            }
        }
    }
    let mut out = Vec::new();
    walk(tree, NodeIndex::root(), &mut out);
    out
}

/// Where the dock drew the floating window of surface `index`: its area in the context's memory, which the dock names after the surface.
fn window_area(ctx: &Context, index: SurfaceIndex) -> Option<[f32; 4]> {
    let rect = ctx.memory(|m| m.area_rect(Id::unique(format!("window {index:?}"))))?;
    rect.is_finite().then(|| [rect.min.x, rect.min.y, rect.width(), rect.height()])
}

/// The floating windows of `app`'s dock with their trees and places (x, y, width, height).
fn floating_windows(app: &SmartMediaApp) -> Vec<(&Tree<TabKind>, Option<[f32; 4]>)> {
    app.tree
        .iter_surfaces_indexed()
        .filter(|(index, _)| *index != SurfaceIndex::main())
        .filter_map(|(index, surface)| match surface {
            Surface::Window(tree, _) => Some((tree, window_area(&app.egui_ctx, index))),
            _ => None,
        })
        .collect()
}

/// The workspace of `app` as it is now.
pub fn capture(app: &SmartMediaApp, window: Option<WindowState>) -> Workspace {
    let tree = app.tree.main_surface();
    let windows = floating_windows(app);
    let mut labels: HashMap<BrowserId, u64> = HashMap::new();
    let mut browsers: Vec<BrowserState> = Vec::new();
    let mut next = 1u64;
    let order: Vec<BrowserId> = browsers_in_order(tree).into_iter().chain(windows.iter().flat_map(|(t, _)| browsers_in_order(t))).collect();
    for id in order {
        let explorer = if id == BrowserId::MAIN { Some(&app.context.file_explorer) } else { app.context.filtered_tabs.get(&id) };
        let label = if id == BrowserId::MAIN { 0 } else { next };
        if let Some(state) = explorer.and_then(|ex| ex.workspace_state(label)) {
            labels.insert(id, label);
            browsers.push(state);
            if id != BrowserId::MAIN {
                next += 1;
            }
        }
    }
    let label_of = |id: BrowserId| labels.get(&id).copied();
    let keep = |kind: &TabKind| saved_tab(kind, &label_of);
    let dock = capture_node(tree, NodeIndex::root(), &keep);
    let floating = windows
        .iter()
        .filter_map(|(tree, rect)| {
            let dock = capture_node(tree, NodeIndex::root(), &keep)?;
            Some(FloatingWindow { dock, position: rect.map(|r| [r[0], r[1]]), size: rect.map(|r| [r[2], r[3]]) })
        })
        .collect();
    let focused = app.tree.focused_leaf().and_then(|p| {
        let leaf = app.tree.leaf(p).ok()?;
        leaf.tabs.get(leaf.active.0).and_then(&keep)
    });
    let mut workspace = Workspace { window, dock, floating, focused, browsers, ..Workspace::default() };
    workspace.sanitize();
    workspace
}

#[cfg(test)]
mod tests {
    use super::*;
    use eframe::egui::{pos2, vec2};

    fn info(inner: Option<(f32, f32)>, outer: Option<(f32, f32)>) -> WindowInfo {
        WindowInfo {
            inner: inner.map(|(w, h)| Rect::from_min_size(pos2(0.0, 0.0), vec2(w, h))),
            outer: outer.map(|(x, y)| Rect::from_min_size(pos2(x, y), vec2(10.0, 10.0))),
            ..Default::default()
        }
    }

    #[test]
    fn a_normal_window_is_saved_with_its_size_and_place() {
        let mut last = None;
        let state = window_state(&info(Some((1100.0, 700.0)), Some((30.0, 40.0))), &mut last).unwrap();
        assert_eq!((state.size, state.position, state.maximized), ([1100.0, 700.0], Some([30.0, 40.0]), false));
        assert_eq!(last, Some(state));
    }

    #[test]
    fn a_maximized_window_keeps_the_last_normal_geometry_and_the_flag() {
        let mut last = None;
        window_state(&info(Some((1100.0, 700.0)), Some((30.0, 40.0))), &mut last);
        let maximized = WindowInfo { maximized: true, ..info(Some((1920.0, 1080.0)), Some((0.0, 0.0))) };
        let state = window_state(&maximized, &mut last).unwrap();
        assert_eq!((state.size, state.position, state.maximized), ([1100.0, 700.0], Some([30.0, 40.0]), true));
        let back = window_state(&info(Some((1000.0, 600.0)), Some((50.0, 60.0))), &mut last).unwrap();
        assert!(!back.maximized && back.size == [1000.0, 600.0]);
    }

    #[test]
    fn a_minimized_or_fullscreen_window_changes_nothing() {
        let mut last = Some(WindowState { size: [900.0, 600.0], position: None, maximized: false });
        let minimized = WindowInfo { minimized: true, ..info(Some((0.0, 0.0)), Some((-32_000.0, -32_000.0))) };
        assert_eq!(window_state(&minimized, &mut last), Some(WindowState { size: [900.0, 600.0], position: None, maximized: false }));
        let full = WindowInfo { fullscreen: true, ..info(Some((1920.0, 1080.0)), None) };
        assert_eq!(window_state(&full, &mut last).unwrap().size, [900.0, 600.0]);
    }

    #[test]
    fn a_window_whose_size_is_unknown_is_not_saved() {
        let mut last = None;
        assert_eq!(window_state(&info(None, None), &mut last), None);
        assert_eq!(window_state(&info(Some((0.0, 0.0)), None), &mut last), None);
        let wayland = window_state(&info(Some((800.0, 500.0)), None), &mut last).unwrap();
        assert_eq!(wayland.position, None, "the position stays unknown where the system keeps it");
    }

    #[test]
    fn tabs_are_named_by_the_label_of_their_browser() {
        let label = |id: BrowserId| (id == BrowserId(7)).then_some(3);
        assert_eq!(saved_tab(&TabKind::browser(BrowserId(7)), &label), Some(SavedTab::Browser { id: 3 }));
        assert_eq!(saved_tab(&TabKind::browser(BrowserId(8)), &label), None);
        assert_eq!(saved_tab(&TabKind::Refine, &label), Some(SavedTab::Refine));
        assert_eq!(saved_tab(&TabKind::Logs, &label), Some(SavedTab::Logs));
    }
}
