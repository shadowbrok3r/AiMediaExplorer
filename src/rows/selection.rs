//! Selection by key with an anchor for ranges and a focus that drives the preview.

use std::collections::HashSet;

use super::key::PathKey;
use super::store::RowStore;
use super::view::{RowFacts, RowKind, ViewIndex};

/// A store and the view over it, as selection operations see them.
pub struct ViewRef<'a, R> {
    pub store: &'a RowStore<R>,
    pub view: &'a ViewIndex,
}

impl<R> Clone for ViewRef<'_, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R> Copy for ViewRef<'_, R> {}

impl<'a, R> ViewRef<'a, R> {
    pub fn new(store: &'a RowStore<R>, view: &'a ViewIndex) -> Self {
        Self { store, view }
    }

    /// Number of visible rows.
    pub fn len(&self) -> usize {
        self.view.len()
    }

    pub fn is_empty(&self) -> bool {
        self.view.is_empty()
    }

    pub fn key_at(&self, pos: usize) -> Option<&'a PathKey> {
        self.store.key_at(self.view.store_index(pos)?)
    }

    pub fn row_at(&self, pos: usize) -> Option<&'a R> {
        self.store.get(self.view.store_index(pos)?)
    }

    /// View position of the row `key` names, when that row is visible.
    pub fn pos_of(&self, key: &PathKey) -> Option<usize> {
        self.view.view_pos(self.store.position_of_key(key)?)
    }

    pub fn is_visible(&self, key: &PathKey) -> bool {
        self.pos_of(key).is_some()
    }

    fn keys_between(&self, lo: usize, hi: usize) -> Vec<PathKey> {
        (lo..=hi).filter_map(|p| self.key_at(p).cloned()).collect()
    }

    fn all_keys(&self) -> Vec<PathKey> {
        (0..self.len()).filter_map(|p| self.key_at(p).cloned()).collect()
    }
}

/// How far a keyboard move goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    Up(usize),
    Down(usize),
    Home,
    End,
}

/// Counts over the selected rows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SelectionStats {
    pub count: usize,
    pub images: usize,
    pub videos: usize,
    pub dirs: usize,
    pub archives: usize,
    pub other: usize,
    /// Bytes of the selected rows that are not directories.
    pub size: u64,
}

#[derive(Clone, Debug, Default)]
pub struct Selection {
    order: Vec<PathKey>,
    members: HashSet<PathKey>,
    anchor: Option<PathKey>,
    focus: Option<PathKey>,
    revision: u64,
}

impl Selection {
    pub fn new() -> Self {
        Self::default()
    }

    /// Bumped by every operation that changed the selected set, the anchor or the focus.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    pub fn contains(&self, key: &PathKey) -> bool {
        self.members.contains(key)
    }

    /// Selected keys in selection order.
    pub fn keys(&self) -> &[PathKey] {
        &self.order
    }

    pub fn iter(&self) -> std::slice::Iter<'_, PathKey> {
        self.order.iter()
    }

    /// Fixed end of a range selection.
    pub fn anchor(&self) -> Option<&PathKey> {
        self.anchor.as_ref()
    }

    /// The row the keyboard cursor is on.
    pub fn focus(&self) -> Option<&PathKey> {
        self.focus.as_ref()
    }

    /// The row the preview shows: the focus while it is selected, otherwise the last row selected.
    pub fn preview_target(&self) -> Option<&PathKey> {
        match &self.focus {
            Some(f) if self.members.contains(f) => Some(f),
            _ => self.order.last(),
        }
    }

    fn commit(&mut self, order: Vec<PathKey>, anchor: Option<PathKey>, focus: Option<PathKey>) -> bool {
        if self.order == order && self.anchor == anchor && self.focus == focus {
            return false;
        }
        self.members = order.iter().cloned().collect();
        self.order = order;
        self.anchor = anchor;
        self.focus = focus;
        self.revision += 1;
        true
    }

    /// Deselects everything; anchor and focus stay.
    pub fn clear(&mut self) -> bool {
        if self.order.is_empty() {
            return false;
        }
        self.order.clear();
        self.members.clear();
        self.revision += 1;
        true
    }

    /// Plain click: selects only `key`.
    pub fn click(&mut self, key: PathKey) -> bool {
        self.commit(vec![key.clone()], Some(key.clone()), Some(key))
    }

    /// Ctrl click: flips `key`; anchor and focus move to it.
    pub fn toggle(&mut self, key: PathKey) -> bool {
        if self.members.remove(&key) {
            self.order.retain(|k| *k != key);
        } else {
            self.members.insert(key.clone());
            self.order.push(key.clone());
        }
        self.anchor = Some(key.clone());
        self.focus = Some(key);
        self.revision += 1;
        true
    }

    /// Shift click: selects the visible rows from the anchor (else the focus, else `target`) to `target`.
    pub fn select_range<R>(&mut self, view: &ViewRef<'_, R>, target: &PathKey, additive: bool) -> bool {
        let Some(t) = view.pos_of(target) else { return false };
        let anchor_pos = self.anchor.as_ref().and_then(|k| view.pos_of(k));
        let (a, anchor) = match anchor_pos {
            Some(a) => (a, self.anchor.clone()),
            None => match self.focus.as_ref().and_then(|k| view.pos_of(k)) {
                Some(f) => (f, self.focus.clone()),
                None => (t, Some(target.clone())),
            },
        };
        let range = view.keys_between(a.min(t), a.max(t));
        let order = if additive {
            let mut order = self.order.clone();
            order.extend(range.into_iter().filter(|k| !self.members.contains(k)));
            order
        } else {
            range
        };
        self.commit(order, anchor, Some(target.clone()))
    }

    /// Ctrl+A: every visible row.
    pub fn select_all<R>(&mut self, view: &ViewRef<'_, R>) -> bool {
        let first = view.key_at(0).cloned();
        let anchor = self.anchor.clone().filter(|k| view.is_visible(k)).or_else(|| first.clone());
        let focus = self.focus.clone().filter(|k| view.is_visible(k)).or(first);
        self.commit(view.all_keys(), anchor, focus)
    }

    /// Selects the visible rows that were not selected and deselects the rest.
    pub fn invert<R>(&mut self, view: &ViewRef<'_, R>) -> bool {
        let order = view.all_keys().into_iter().filter(|k| !self.members.contains(k)).collect();
        let anchor = self.anchor.clone().filter(|k| view.is_visible(k));
        let focus = self.focus.clone().filter(|k| view.is_visible(k));
        self.commit(order, anchor, focus)
    }

    /// Selects exactly the visible rows among `keys`, in view order.
    pub fn select_keys<R>(&mut self, view: &ViewRef<'_, R>, keys: impl IntoIterator<Item = PathKey>) -> bool {
        let wanted: HashSet<PathKey> = keys.into_iter().collect();
        let order: Vec<PathKey> = view.all_keys().into_iter().filter(|k| wanted.contains(k)).collect();
        let focus = order.last().cloned();
        let anchor = order.first().cloned();
        self.commit(order, anchor, focus)
    }

    /// Moves the focus and selects the new row, or the range from the anchor when `extend`; `None` for an empty view.
    pub fn step_focus<R>(&mut self, view: &ViewRef<'_, R>, step: Step, extend: bool) -> Option<PathKey> {
        if view.is_empty() {
            return None;
        }
        let last = view.len() - 1;
        let current = self.focus.as_ref().and_then(|k| view.pos_of(k));
        let target = match (step, current) {
            (Step::Home, _) => 0,
            (Step::End, _) => last,
            (Step::Down(_), None) => 0,
            (Step::Up(_), None) => last,
            (Step::Down(n), Some(c)) => (c + n).min(last),
            (Step::Up(n), Some(c)) => c.saturating_sub(n),
        };
        let key = view.key_at(target)?.clone();
        if extend {
            self.select_range(view, &key, false);
        } else {
            self.click(key.clone());
        }
        Some(key)
    }

    /// Moves the focus without changing what is selected.
    pub fn set_focus(&mut self, key: Option<PathKey>) -> bool {
        if self.focus == key {
            return false;
        }
        self.focus = key;
        self.revision += 1;
        true
    }

    /// Drops selected keys, the anchor and the focus that are not visible rows; returns how many keys were dropped.
    pub fn prune<R>(&mut self, view: &ViewRef<'_, R>) -> usize {
        let before = self.order.len();
        let order: Vec<PathKey> = self.order.iter().filter(|k| view.is_visible(k)).cloned().collect();
        let anchor = self.anchor.clone().filter(|k| view.is_visible(k));
        let focus = self.focus.clone().filter(|k| view.is_visible(k));
        self.commit(order, anchor, focus);
        before - self.order.len()
    }

    /// Store positions of the selected rows in view order.
    pub fn store_indices<R>(&self, view: &ViewRef<'_, R>) -> Vec<usize> {
        let mut found: Vec<(usize, usize)> = self
            .order
            .iter()
            .filter_map(|k| {
                let idx = view.store.position_of_key(k)?;
                Some((view.view.view_pos(idx)?, idx))
            })
            .collect();
        found.sort_unstable();
        found.into_iter().map(|(_, idx)| idx).collect()
    }

    /// Selected rows in view order; empty when nothing is selected.
    pub fn rows<'a, R>(&self, view: &ViewRef<'a, R>) -> Vec<&'a R> {
        self.store_indices(view).into_iter().filter_map(|i| view.store.get(i)).collect()
    }

    /// Counts over the selected rows still in `store`, visible or not.
    pub fn stats<R: RowFacts>(&self, store: &RowStore<R>) -> SelectionStats {
        let mut s = SelectionStats::default();
        for row in self.order.iter().filter_map(|k| store.get_by_key(k)) {
            s.count += 1;
            match row.kind() {
                RowKind::Dir => s.dirs += 1,
                RowKind::Archive => s.archives += 1,
                RowKind::Image => s.images += 1,
                RowKind::Video => s.videos += 1,
                RowKind::Other => s.other += 1,
            }
            if row.kind() != RowKind::Dir {
                s.size = s.size.saturating_add(row.size());
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::key::KeyPolicy;
    use super::super::testing::{TestRow, sort_key};
    use super::super::view::{SortSpec, ViewInputs};

    fn key(p: &str) -> PathKey {
        PathKey::with_policy(p, KeyPolicy::Exact)
    }

    fn all(_: &TestRow) -> bool {
        true
    }

    /// Rows `/r/00`..`/r/{n-1}` with size `n - i`, so a size sort reverses the store order.
    struct Rig {
        store: RowStore<TestRow>,
        view: ViewIndex,
    }

    impl Rig {
        fn new(n: usize) -> Self {
            let mut store = RowStore::with_policy(KeyPolicy::Exact);
            store.bulk_load((0..n).map(|i| TestRow::new(&format!("/r/{i:02}")).size((n - i) as u64)));
            let mut rig = Rig { store, view: ViewIndex::new() };
            rig.refresh(&all, 0);
            rig
        }

        fn refresh(&mut self, filter: &dyn Fn(&TestRow) -> bool, sig: u64) {
            let inputs = ViewInputs { filter_sig: sig, data_sig: 0, filter, sort_key: &sort_key };
            self.view.refresh(&self.store, &inputs);
        }

        fn sorted_by_size(mut self) -> Self {
            self.view.set_sort(SortSpec::single(1, false));
            self.refresh(&all, 0);
            self
        }

        fn v(&self) -> ViewRef<'_, TestRow> {
            ViewRef::new(&self.store, &self.view)
        }
    }

    fn paths(sel: &Selection) -> Vec<String> {
        sel.iter().map(|k| k.as_str().to_string()).collect()
    }

    fn r(i: usize) -> String {
        format!("/r/{i:02}")
    }

    #[test]
    fn click_selects_exactly_one_row_and_sets_anchor_and_focus() {
        let mut sel = Selection::new();
        assert!(sel.click(key("/r/02")));
        assert!(sel.click(key("/r/04")));
        assert_eq!(paths(&sel), ["/r/04"]);
        assert_eq!(sel.anchor().unwrap().as_str(), "/r/04");
        assert_eq!(sel.focus().unwrap().as_str(), "/r/04");
        assert_eq!(sel.len(), 1);
    }

    #[test]
    fn clicking_the_only_selected_row_again_changes_nothing() {
        let mut sel = Selection::new();
        sel.click(key("/a"));
        let rev = sel.revision();
        assert!(!sel.click(key("/a")));
        assert_eq!(sel.revision(), rev);
    }

    #[test]
    fn ctrl_click_toggles_membership_and_moves_the_anchor() {
        let mut sel = Selection::new();
        sel.click(key("/a"));
        sel.toggle(key("/c"));
        sel.toggle(key("/b"));
        assert_eq!(paths(&sel), ["/a", "/c", "/b"]);
        assert_eq!(sel.anchor().unwrap().as_str(), "/b");
        sel.toggle(key("/c"));
        assert_eq!(paths(&sel), ["/a", "/b"]);
        assert!(!sel.contains(&key("/c")));
        assert_eq!(sel.focus().unwrap().as_str(), "/c", "the cursor moves to the clicked row even when it was deselected");
    }

    #[test]
    fn shift_click_selects_the_range_from_the_anchor_in_view_order() {
        let rig = Rig::new(10);
        let mut sel = Selection::new();
        sel.click(key(&r(2)));
        assert!(sel.select_range(&rig.v(), &key(&r(5)), false));
        assert_eq!(paths(&sel), [r(2), r(3), r(4), r(5)]);
        assert_eq!(sel.anchor().unwrap().as_str(), r(2));
        assert_eq!(sel.focus().unwrap().as_str(), r(5));
    }

    #[test]
    fn a_range_follows_the_view_order_not_the_store_order() {
        let rig = Rig::new(10).sorted_by_size();
        assert_eq!(rig.v().key_at(0).unwrap().as_str(), r(9), "size sort reverses the store");
        let mut sel = Selection::new();
        sel.click(key(&r(7)));
        sel.select_range(&rig.v(), &key(&r(4)), false);
        assert_eq!(paths(&sel), [r(7), r(6), r(5), r(4)]);
    }

    #[test]
    fn extending_a_range_backwards_and_shrinking_it_keeps_the_anchor() {
        let rig = Rig::new(10);
        let mut sel = Selection::new();
        sel.click(key(&r(5)));
        sel.select_range(&rig.v(), &key(&r(2)), false);
        assert_eq!(paths(&sel), [r(2), r(3), r(4), r(5)]);
        sel.select_range(&rig.v(), &key(&r(7)), false);
        assert_eq!(paths(&sel), [r(5), r(6), r(7)]);
        assert_eq!(sel.anchor().unwrap().as_str(), r(5));
    }

    #[test]
    fn ctrl_shift_click_adds_the_range_to_the_selection() {
        let rig = Rig::new(12);
        let mut sel = Selection::new();
        sel.click(key(&r(0)));
        sel.toggle(key(&r(6)));
        sel.select_range(&rig.v(), &key(&r(9)), true);
        assert_eq!(paths(&sel), [r(0), r(6), r(7), r(8), r(9)]);
        assert_eq!(sel.anchor().unwrap().as_str(), r(6));
    }

    #[test]
    fn shift_click_without_an_anchor_uses_the_focus_then_the_target() {
        let rig = Rig::new(8);
        let mut sel = Selection::new();
        assert!(sel.select_range(&rig.v(), &key(&r(3)), false));
        assert_eq!(paths(&sel), [r(3)]);
        let mut sel = Selection::new();
        sel.set_focus(Some(key(&r(1))));
        sel.select_range(&rig.v(), &key(&r(3)), false);
        assert_eq!(paths(&sel), [r(1), r(2), r(3)]);
        assert_eq!(sel.anchor().unwrap().as_str(), r(1));
    }

    #[test]
    fn shift_click_with_a_hidden_anchor_falls_back() {
        let mut rig = Rig::new(8);
        let mut sel = Selection::new();
        sel.click(key(&r(1)));
        rig.refresh(&|row: &TestRow| row.path != r(1), 1);
        assert!(sel.select_range(&rig.v(), &key(&r(4)), false));
        assert_eq!(paths(&sel), [r(4)]);
    }

    #[test]
    fn shift_click_on_a_row_that_is_not_visible_does_nothing() {
        let mut rig = Rig::new(5);
        rig.refresh(&|row: &TestRow| row.path != r(3), 1);
        let mut sel = Selection::new();
        sel.click(key(&r(1)));
        let rev = sel.revision();
        assert!(!sel.select_range(&rig.v(), &key(&r(3)), false));
        assert!(!sel.select_range(&rig.v(), &key("/not/there"), false));
        assert_eq!(sel.revision(), rev);
    }

    #[test]
    fn a_range_skips_rows_the_filter_hides() {
        let mut rig = Rig::new(10);
        rig.refresh(&|row: &TestRow| row.size % 2 == 0, 1);
        let mut sel = Selection::new();
        sel.click(key(&r(0)));
        sel.select_range(&rig.v(), &key(&r(6)), false);
        let expected: Vec<String> = (0..=6).filter(|i| (10 - i) % 2 == 0).map(r).collect();
        assert_eq!(paths(&sel), expected);
    }

    #[test]
    fn select_all_covers_exactly_the_visible_rows() {
        let mut rig = Rig::new(10);
        rig.refresh(&|row: &TestRow| row.size > 5, 1);
        let mut sel = Selection::new();
        assert!(sel.select_all(&rig.v()));
        assert_eq!(sel.len(), rig.view.len());
        assert_eq!(sel.len(), 5);
        assert!(sel.keys().iter().all(|k| rig.v().is_visible(k)));
        assert_eq!(sel.focus().unwrap().as_str(), r(0));
        assert!(!sel.select_all(&rig.v()), "selecting all twice changes nothing");
    }

    #[test]
    fn select_all_on_an_empty_view_selects_nothing() {
        let mut rig = Rig::new(4);
        rig.refresh(&|_: &TestRow| false, 1);
        let mut sel = Selection::new();
        assert!(!sel.select_all(&rig.v()));
        assert!(sel.is_empty());
        assert!(sel.rows(&rig.v()).is_empty());
    }

    #[test]
    fn select_all_keeps_a_visible_focus() {
        let rig = Rig::new(6);
        let mut sel = Selection::new();
        sel.click(key(&r(3)));
        sel.select_all(&rig.v());
        assert_eq!(sel.focus().unwrap().as_str(), r(3));
        assert_eq!(sel.anchor().unwrap().as_str(), r(3));
    }

    #[test]
    fn invert_flips_every_visible_row() {
        let mut rig = Rig::new(8);
        let mut sel = Selection::new();
        sel.click(key(&r(1)));
        sel.toggle(key(&r(2)));
        assert!(sel.invert(&rig.v()));
        assert_eq!(sel.len(), 6);
        assert!(!sel.contains(&key(&r(1))) && !sel.contains(&key(&r(2))));
        sel.invert(&rig.v());
        assert_eq!(sel.len(), 2);
        rig.refresh(&|row: &TestRow| row.path != r(0), 1);
        let mut none = Selection::new();
        none.invert(&rig.v());
        assert_eq!(none.len(), 7, "hidden rows never become selected");
    }

    #[test]
    fn clear_deselects_everything_but_keeps_the_cursor() {
        let mut sel = Selection::new();
        sel.click(key("/a"));
        assert!(sel.clear());
        assert!(sel.is_empty());
        assert_eq!(sel.focus().unwrap().as_str(), "/a");
        assert!(!sel.clear());
    }

    #[test]
    fn an_empty_selection_never_means_all_rows() {
        let rig = Rig::new(20);
        let sel = Selection::new();
        assert!(sel.rows(&rig.v()).is_empty());
        assert!(sel.store_indices(&rig.v()).is_empty());
        assert_eq!(sel.stats(&rig.store), SelectionStats::default());
        let mut cleared = Selection::new();
        cleared.select_all(&rig.v());
        cleared.clear();
        assert!(cleared.rows(&rig.v()).is_empty());
        assert!(cleared.preview_target().is_none());
    }

    #[test]
    fn selected_rows_come_back_in_view_order_whatever_the_click_order() {
        let rig = Rig::new(10);
        let mut sel = Selection::new();
        sel.click(key(&r(7)));
        sel.toggle(key(&r(1)));
        sel.toggle(key(&r(4)));
        let rows: Vec<&str> = sel.rows(&rig.v()).iter().map(|row| row.path.as_str()).collect();
        assert_eq!(rows, [r(1), r(4), r(7)]);
        assert_eq!(paths(&sel), [r(7), r(1), r(4)], "the selection itself remembers click order");
        let idx = sel.store_indices(&rig.v());
        assert_eq!(idx, [1, 4, 7]);
    }

    #[test]
    fn prune_drops_rows_that_are_hidden_or_gone() {
        let mut rig = Rig::new(10);
        let mut sel = Selection::new();
        sel.click(key(&r(1)));
        sel.toggle(key(&r(2)));
        sel.toggle(key(&r(3)));
        rig.refresh(&|row: &TestRow| row.path != r(2), 1);
        assert_eq!(sel.prune(&rig.v()), 1);
        assert_eq!(paths(&sel), [r(1), r(3)]);
        rig.store.remove_path(&r(3));
        rig.refresh(&|row: &TestRow| row.path != r(2), 1);
        assert_eq!(sel.prune(&rig.v()), 1);
        assert_eq!(paths(&sel), [r(1)]);
        assert!(sel.focus().is_none(), "the focus was the removed row");
        assert!(sel.anchor().is_none());
    }

    #[test]
    fn prune_leaves_a_fully_visible_selection_untouched() {
        let rig = Rig::new(6);
        let mut sel = Selection::new();
        sel.click(key(&r(1)));
        sel.toggle(key(&r(2)));
        let rev = sel.revision();
        assert_eq!(sel.prune(&rig.v()), 0);
        assert_eq!(sel.revision(), rev);
    }

    #[test]
    fn arrows_move_the_focus_and_select_the_row() {
        let rig = Rig::new(6);
        let mut sel = Selection::new();
        assert_eq!(sel.step_focus(&rig.v(), Step::Down(1), false).unwrap().as_str(), r(0));
        assert_eq!(sel.step_focus(&rig.v(), Step::Down(1), false).unwrap().as_str(), r(1));
        assert_eq!(paths(&sel), [r(1)]);
        assert_eq!(sel.step_focus(&rig.v(), Step::Up(1), false).unwrap().as_str(), r(0));
        assert_eq!(sel.step_focus(&rig.v(), Step::Up(1), false).unwrap().as_str(), r(0), "stops at the top");
        assert_eq!(sel.step_focus(&rig.v(), Step::End, false).unwrap().as_str(), r(5));
        assert_eq!(sel.step_focus(&rig.v(), Step::Down(1), false).unwrap().as_str(), r(5), "stops at the bottom");
        assert_eq!(sel.step_focus(&rig.v(), Step::Home, false).unwrap().as_str(), r(0));
        assert_eq!(paths(&sel), [r(0)]);
    }

    #[test]
    fn page_moves_jump_by_the_page_size_and_clamp() {
        let rig = Rig::new(30);
        let mut sel = Selection::new();
        sel.click(key(&r(2)));
        assert_eq!(sel.step_focus(&rig.v(), Step::Down(8), false).unwrap().as_str(), r(10));
        assert_eq!(sel.step_focus(&rig.v(), Step::Down(100), false).unwrap().as_str(), r(29));
        assert_eq!(sel.step_focus(&rig.v(), Step::Up(8), false).unwrap().as_str(), r(21));
        assert_eq!(sel.step_focus(&rig.v(), Step::Up(100), false).unwrap().as_str(), r(0));
    }

    #[test]
    fn up_with_no_focus_starts_at_the_bottom_and_down_at_the_top() {
        let rig = Rig::new(5);
        let mut sel = Selection::new();
        assert_eq!(sel.step_focus(&rig.v(), Step::Up(1), false).unwrap().as_str(), r(4));
        let mut sel = Selection::new();
        assert_eq!(sel.step_focus(&rig.v(), Step::Down(5), false).unwrap().as_str(), r(0));
    }

    #[test]
    fn shift_arrows_extend_the_selection_from_the_anchor() {
        let rig = Rig::new(10);
        let mut sel = Selection::new();
        sel.click(key(&r(4)));
        sel.step_focus(&rig.v(), Step::Down(1), true);
        sel.step_focus(&rig.v(), Step::Down(1), true);
        assert_eq!(paths(&sel), [r(4), r(5), r(6)]);
        sel.step_focus(&rig.v(), Step::Up(1), true);
        assert_eq!(paths(&sel), [r(4), r(5)]);
        sel.step_focus(&rig.v(), Step::Up(3), true);
        assert_eq!(paths(&sel), [r(2), r(3), r(4)]);
        assert_eq!(sel.anchor().unwrap().as_str(), r(4));
        sel.step_focus(&rig.v(), Step::End, true);
        assert_eq!(sel.len(), 6);
        sel.step_focus(&rig.v(), Step::Home, true);
        assert_eq!(sel.len(), 5);
    }

    #[test]
    fn stepping_follows_the_view_order() {
        let rig = Rig::new(6).sorted_by_size();
        let mut sel = Selection::new();
        sel.step_focus(&rig.v(), Step::Down(1), false);
        assert_eq!(sel.focus().unwrap().as_str(), r(5));
        sel.step_focus(&rig.v(), Step::Down(1), false);
        assert_eq!(sel.focus().unwrap().as_str(), r(4));
    }

    #[test]
    fn stepping_through_an_empty_view_does_nothing() {
        let mut rig = Rig::new(3);
        rig.refresh(&|_: &TestRow| false, 1);
        let mut sel = Selection::new();
        assert!(sel.step_focus(&rig.v(), Step::Down(1), false).is_none());
        assert!(sel.is_empty());
    }

    #[test]
    fn a_hidden_focus_restarts_stepping_from_the_edge() {
        let mut rig = Rig::new(6);
        let mut sel = Selection::new();
        sel.click(key(&r(2)));
        rig.refresh(&|row: &TestRow| row.path != r(2), 1);
        assert_eq!(sel.step_focus(&rig.v(), Step::Down(1), false).unwrap().as_str(), r(0));
    }

    #[test]
    fn the_preview_follows_the_focus_while_it_is_selected() {
        let mut sel = Selection::new();
        assert!(sel.preview_target().is_none());
        sel.click(key("/a"));
        sel.toggle(key("/b"));
        assert_eq!(sel.preview_target().unwrap().as_str(), "/b");
        sel.toggle(key("/b"));
        assert_eq!(sel.focus().unwrap().as_str(), "/b");
        assert_eq!(sel.preview_target().unwrap().as_str(), "/a", "the deselected focus no longer drives the preview");
        sel.toggle(key("/c"));
        sel.set_focus(Some(key("/a")));
        assert_eq!(sel.preview_target().unwrap().as_str(), "/a");
    }

    #[test]
    fn multi_select_leaves_the_preview_on_the_focused_row() {
        let rig = Rig::new(30);
        let mut sel = Selection::new();
        sel.click(key(&r(2)));
        sel.select_range(&rig.v(), &key(&r(20)), false);
        assert_eq!(sel.len(), 19);
        assert_eq!(sel.preview_target().unwrap().as_str(), r(20), "one deterministic row, not the last of a stream of events");
    }

    #[test]
    fn select_keys_selects_only_visible_rows_among_the_wanted_keys() {
        let mut rig = Rig::new(8);
        rig.refresh(&|row: &TestRow| row.path != r(3), 1);
        let mut sel = Selection::new();
        sel.select_keys(&rig.v(), [key(&r(5)), key(&r(3)), key(&r(1)), key("/nope")]);
        assert_eq!(paths(&sel), [r(1), r(5)]);
        assert!(sel.select_keys(&rig.v(), Vec::new()));
        assert!(sel.is_empty());
    }

    #[test]
    fn stats_count_kinds_and_sizes_of_selected_rows() {
        let mut store = RowStore::with_policy(KeyPolicy::Exact);
        store.bulk_load([
            TestRow::new("/d").kind(RowKind::Dir).size(4096),
            TestRow::new("/i").kind(RowKind::Image).size(10),
            TestRow::new("/v").kind(RowKind::Video).size(20),
            TestRow::new("/z").kind(RowKind::Archive).size(30),
        ]);
        let mut sel = Selection::new();
        for p in ["/d", "/i", "/v", "/z", "/gone"] {
            sel.toggle(key(p));
        }
        let s = sel.stats(&store);
        assert_eq!((s.count, s.dirs, s.images, s.videos, s.archives, s.other), (4, 1, 1, 1, 1, 0));
        assert_eq!(s.size, 60);
    }

    #[test]
    fn the_revision_moves_exactly_when_something_changed() {
        let rig = Rig::new(5);
        let mut sel = Selection::new();
        let r0 = sel.revision();
        sel.click(key(&r(1)));
        let r1 = sel.revision();
        assert!(r1 > r0);
        sel.click(key(&r(1)));
        assert_eq!(sel.revision(), r1);
        sel.select_all(&rig.v());
        assert!(sel.revision() > r1);
        let r2 = sel.revision();
        sel.set_focus(sel.focus().cloned());
        assert_eq!(sel.revision(), r2);
    }
}
