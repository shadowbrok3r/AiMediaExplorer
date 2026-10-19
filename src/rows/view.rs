//! View over a row store: visible rows in display order with counts cached per revision.

use std::borrow::Cow;
use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

use super::store::{Keyed, RowStore, Versions};

/// Marks a store position that is not in the view.
pub const HIDDEN: u32 = u32::MAX;

/// Row category used by the cached counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowKind {
    Dir,
    Archive,
    Image,
    Video,
    Other,
}

/// What a view needs to know about a row besides its path.
pub trait RowFacts: Keyed {
    fn kind(&self) -> RowKind;
    fn size(&self) -> u64;
}

/* ----------------------------------------------- sorting ----------------------------------------------- */

/// One sort column and its direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SortKey {
    pub column: u8,
    pub descending: bool,
}

/// Ordered sort keys, most significant first.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SortSpec {
    keys: Vec<SortKey>,
}

impl SortSpec {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn single(column: u8, descending: bool) -> Self {
        Self { keys: vec![SortKey { column, descending }] }
    }

    pub fn from_keys(keys: Vec<SortKey>) -> Self {
        let mut seen = Vec::new();
        let keys = keys
            .into_iter()
            .filter(|k| {
                let fresh = !seen.contains(&k.column);
                seen.push(k.column);
                fresh
            })
            .collect();
        Self { keys }
    }

    pub fn keys(&self) -> &[SortKey] {
        &self.keys
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    pub fn uses(&self, column: u8) -> bool {
        self.keys.iter().any(|k| k.column == column)
    }

    /// Priority (0 is primary) and direction of `column`, when it is a sort key.
    pub fn position_of(&self, column: u8) -> Option<(usize, bool)> {
        self.keys.iter().position(|k| k.column == column).map(|i| (i, self.keys[i].descending))
    }

    /// Header click: cycles the column ascending, descending, off. Without `additive` it replaces the other keys.
    pub fn click(&mut self, column: u8, additive: bool) {
        let at = self.keys.iter().position(|k| k.column == column);
        if additive {
            match at {
                None => self.keys.push(SortKey { column, descending: false }),
                Some(i) if !self.keys[i].descending => self.keys[i].descending = true,
                Some(i) => {
                    self.keys.remove(i);
                }
            }
            return;
        }
        match (self.keys.len(), at) {
            (1, Some(0)) if !self.keys[0].descending => self.keys[0].descending = true,
            (1, Some(0)) => self.keys.clear(),
            _ => self.keys = vec![SortKey { column, descending: false }],
        }
    }

    pub fn retain_columns(&mut self, keep: impl Fn(u8) -> bool) {
        self.keys.retain(|k| keep(k.column));
    }
}

/// A comparable cell value; missing values order before present ones.
#[derive(Clone, Debug, PartialEq)]
pub enum SortValue<'a> {
    Missing,
    Flag(bool),
    Int(i128),
    Float(f32),
    Text(Cow<'a, str>),
}

impl SortValue<'_> {
    fn rank(&self) -> u8 {
        match self {
            SortValue::Missing => 0,
            SortValue::Flag(_) => 1,
            SortValue::Int(_) => 2,
            SortValue::Float(_) => 3,
            SortValue::Text(_) => 4,
        }
    }

    pub fn compare(&self, other: &SortValue<'_>) -> Ordering {
        match (self, other) {
            (SortValue::Flag(a), SortValue::Flag(b)) => a.cmp(b),
            (SortValue::Int(a), SortValue::Int(b)) => a.cmp(b),
            (SortValue::Float(a), SortValue::Float(b)) => a.total_cmp(b),
            (SortValue::Text(a), SortValue::Text(b)) => a.cmp(b),
            _ => self.rank().cmp(&other.rank()),
        }
    }
}

/// Cell value of `row` in `column` for sorting.
pub type SortKeyFn<'a, R> = &'a dyn for<'r> Fn(&'r R, u8) -> SortValue<'r>;

fn compare_values(spec: &[SortKey], a: &[SortValue<'_>], b: &[SortValue<'_>]) -> Ordering {
    for (j, key) in spec.iter().enumerate() {
        let ord = a[j].compare(&b[j]);
        let ord = if key.descending { ord.reverse() } else { ord };
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

/* ------------------------------------------------- stats ----------------------------------------------- */

/// Counts over the visible rows of one revision.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ViewStats {
    pub total: usize,
    pub visible: usize,
    pub hidden_by_filter: usize,
    pub images: usize,
    pub videos: usize,
    pub dirs: usize,
    pub archives: usize,
    pub other: usize,
    /// Bytes of the visible rows that are not directories.
    pub total_size: u64,
}

impl ViewStats {
    fn add(&mut self, kind: RowKind, size: u64) {
        self.visible += 1;
        match kind {
            RowKind::Dir => self.dirs += 1,
            RowKind::Archive => self.archives += 1,
            RowKind::Image => self.images += 1,
            RowKind::Video => self.videos += 1,
            RowKind::Other => self.other += 1,
        }
        if kind != RowKind::Dir {
            self.total_size = self.total_size.saturating_add(size);
        }
    }

    fn set_total(&mut self, total: usize) {
        self.total = total;
        self.hidden_by_filter = total - self.visible;
    }
}

/// What the last [`ViewIndex::refresh`] had to do.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Refresh {
    #[default]
    Unchanged,
    Appended(usize),
    Rebuilt,
}

/// How often each refresh path ran.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RefreshCounts {
    pub unchanged: u64,
    pub appended: u64,
    pub rebuilt: u64,
}

/// Everything besides the rows that decides the view.
pub struct ViewInputs<'a, R> {
    /// Changes whenever [`filter`](Self::filter) may answer differently for the same row.
    pub filter_sig: u64,
    /// Changes whenever data read by [`sort_key`](Self::sort_key) outside the row changed.
    pub data_sig: u64,
    pub filter: &'a dyn Fn(&R) -> bool,
    pub sort_key: SortKeyFn<'a, R>,
}

#[derive(Clone, Debug, Default)]
struct Applied {
    valid: bool,
    versions: Versions,
    len: usize,
    filter_sig: u64,
    data_sig: u64,
    sort: SortSpec,
}

/* -------------------------------------------------- index ---------------------------------------------- */

#[derive(Default)]
pub struct ViewIndex {
    order: Vec<u32>,
    pos_of: Vec<u32>,
    sort: SortSpec,
    applied: Applied,
    revision: u64,
    stats: ViewStats,
    last: Refresh,
    counts: RefreshCounts,
}

impl ViewIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn sort(&self) -> &SortSpec {
        &self.sort
    }

    /// Sets the sort; the next refresh reorders. Returns whether it changed.
    pub fn set_sort(&mut self, sort: SortSpec) -> bool {
        let changed = self.sort != sort;
        self.sort = sort;
        changed
    }

    /// Bumped by every refresh that changed the order, the visible set or the counts.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn stats(&self) -> &ViewStats {
        &self.stats
    }

    pub fn last_refresh(&self) -> Refresh {
        self.last
    }

    pub fn counts(&self) -> RefreshCounts {
        self.counts
    }

    /// Number of visible rows.
    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// Store position of the row shown at view position `pos`.
    pub fn store_index(&self, pos: usize) -> Option<usize> {
        self.order.get(pos).map(|&i| i as usize)
    }

    /// View position of the row at store position `idx`, when it is visible.
    pub fn view_pos(&self, idx: usize) -> Option<usize> {
        match self.pos_of.get(idx) {
            Some(&p) if p != HIDDEN => Some(p as usize),
            _ => None,
        }
    }

    pub fn is_visible(&self, idx: usize) -> bool {
        self.view_pos(idx).is_some()
    }

    pub fn order(&self) -> &[u32] {
        &self.order
    }

    /// Visible rows in display order.
    pub fn rows<'a, R>(&'a self, store: &'a RowStore<R>) -> impl Iterator<Item = &'a R> + 'a {
        self.order.iter().filter_map(move |&i| store.get(i as usize))
    }

    /// Forces the next refresh to rebuild.
    pub fn invalidate(&mut self) {
        self.applied.valid = false;
    }

    /// Whether the view reflects the store and inputs as they are now.
    pub fn is_current<R: RowFacts>(&self, store: &RowStore<R>, inputs: &ViewInputs<'_, R>) -> bool {
        self.applied.valid && self.reusable(store, inputs) && store.len() == self.applied.len
    }

    fn reusable<R: RowFacts>(&self, store: &RowStore<R>, inputs: &ViewInputs<'_, R>) -> bool {
        let v = store.versions();
        self.applied.valid
            && self.applied.versions.layout == v.layout
            && self.applied.versions.edits == v.edits
            && self.applied.filter_sig == inputs.filter_sig
            && self.applied.data_sig == inputs.data_sig
            && self.applied.sort == self.sort
            && store.len() >= self.applied.len
    }

    /// Brings the view up to date with `store` and `inputs`.
    pub fn refresh<R: RowFacts>(&mut self, store: &RowStore<R>, inputs: &ViewInputs<'_, R>) -> Refresh {
        let outcome = if !self.reusable(store, inputs) {
            self.rebuild(store, inputs);
            Refresh::Rebuilt
        } else if store.len() == self.applied.len {
            Refresh::Unchanged
        } else {
            let added = store.len() - self.applied.len;
            self.append(store, inputs);
            Refresh::Appended(added)
        };
        self.applied = Applied {
            valid: true,
            versions: store.versions(),
            len: store.len(),
            filter_sig: inputs.filter_sig,
            data_sig: inputs.data_sig,
            sort: self.sort.clone(),
        };
        match outcome {
            Refresh::Unchanged => self.counts.unchanged += 1,
            Refresh::Appended(_) => self.counts.appended += 1,
            Refresh::Rebuilt => self.counts.rebuilt += 1,
        }
        if outcome != Refresh::Unchanged {
            self.revision += 1;
        }
        self.last = outcome;
        outcome
    }

    fn rebuild<R: RowFacts>(&mut self, store: &RowStore<R>, inputs: &ViewInputs<'_, R>) {
        let rows = store.rows();
        let mut stats = ViewStats::default();
        let mut visible: Vec<u32> = Vec::with_capacity(rows.len());
        for (i, row) in rows.iter().enumerate() {
            if (inputs.filter)(row) {
                stats.add(row.kind(), row.size());
                visible.push(i as u32);
            }
        }
        stats.set_total(rows.len());
        self.order = if self.sort.is_empty() { visible } else { sorted_order(store, &visible, self.sort.keys(), inputs.sort_key) };
        self.stats = stats;
        self.pos_of.clear();
        self.pos_of.resize(rows.len(), HIDDEN);
        for (pos, &i) in self.order.iter().enumerate() {
            self.pos_of[i as usize] = pos as u32;
        }
    }

    fn append<R: RowFacts>(&mut self, store: &RowStore<R>, inputs: &ViewInputs<'_, R>) {
        let rows = store.rows();
        let from = self.applied.len;
        self.pos_of.resize(rows.len(), HIDDEN);
        let mut fresh: Vec<u32> = Vec::new();
        for (i, row) in rows.iter().enumerate().skip(from) {
            if (inputs.filter)(row) {
                self.stats.add(row.kind(), row.size());
                fresh.push(i as u32);
            }
        }
        self.stats.set_total(rows.len());
        if fresh.is_empty() {
            return;
        }
        if self.sort.is_empty() {
            for i in fresh {
                self.pos_of[i as usize] = self.order.len() as u32;
                self.order.push(i);
            }
            return;
        }
        let keys = self.sort.keys();
        let fresh = sorted_order(store, &fresh, keys, inputs.sort_key);
        let fresh_vals: Vec<Vec<SortValue<'_>>> = fresh.iter().map(|&i| row_values(&rows[i as usize], keys, inputs.sort_key)).collect();
        let cuts: Vec<usize> = fresh_vals
            .iter()
            .map(|vals| {
                self.order.partition_point(|&e| {
                    let existing = row_values(&rows[e as usize], keys, inputs.sort_key);
                    compare_values(keys, &existing, vals) != Ordering::Greater
                })
            })
            .collect();
        if cuts.iter().all(|&c| c == self.order.len()) {
            for i in fresh {
                self.pos_of[i as usize] = self.order.len() as u32;
                self.order.push(i);
            }
            return;
        }
        let mut merged = Vec::with_capacity(self.order.len() + fresh.len());
        let mut taken = 0;
        for (&i, &cut) in fresh.iter().zip(&cuts) {
            merged.extend_from_slice(&self.order[taken..cut]);
            merged.push(i);
            taken = cut;
        }
        merged.extend_from_slice(&self.order[taken..]);
        self.order = merged;
        for (pos, &i) in self.order.iter().enumerate() {
            self.pos_of[i as usize] = pos as u32;
        }
    }

    /// Compares the view against a from-scratch computation; `Err` names the first difference.
    pub fn check_invariants<R: RowFacts>(&self, store: &RowStore<R>, inputs: &ViewInputs<'_, R>) -> Result<(), String> {
        if !self.is_current(store, inputs) {
            return Err("view is stale: the store or the view inputs changed since the last refresh".into());
        }
        let mut fresh = ViewIndex::new();
        fresh.set_sort(self.sort.clone());
        fresh.refresh(store, inputs);
        if fresh.order != self.order {
            let at = fresh.order.iter().zip(&self.order).position(|(a, b)| a != b);
            return Err(format!("order differs from a rebuild (lengths {} vs {}, first difference at {:?})", self.order.len(), fresh.order.len(), at));
        }
        if fresh.pos_of != self.pos_of {
            return Err("position table differs from a rebuild".into());
        }
        if fresh.stats != self.stats {
            return Err(format!("stats differ from a rebuild: {:?} vs {:?}", self.stats, fresh.stats));
        }
        let mut seen = vec![false; store.len()];
        for (pos, &i) in self.order.iter().enumerate() {
            let i = i as usize;
            if i >= store.len() || std::mem::replace(&mut seen[i], true) {
                return Err(format!("view position {pos} holds an invalid or repeated store index {i}"));
            }
            if self.pos_of[i] as usize != pos {
                return Err(format!("store index {i} is at view position {pos} but the table says {}", self.pos_of[i]));
            }
        }
        Ok(())
    }
}

fn row_values<'r, R>(row: &'r R, keys: &[SortKey], sort_key: SortKeyFn<'_, R>) -> Vec<SortValue<'r>> {
    keys.iter().map(|k| sort_key(row, k.column)).collect()
}

/// `visible` store positions ordered by `keys`; equal keys keep store order.
fn sorted_order<R>(store: &RowStore<R>, visible: &[u32], keys: &[SortKey], sort_key: SortKeyFn<'_, R>) -> Vec<u32>
where
    R: Keyed,
{
    let k = keys.len();
    let rows = store.rows();
    let mut values: Vec<SortValue<'_>> = Vec::with_capacity(visible.len() * k);
    for &i in visible {
        for key in keys {
            values.push(sort_key(&rows[i as usize], key.column));
        }
    }
    let mut slots: Vec<u32> = (0..visible.len() as u32).collect();
    slots.sort_unstable_by(|&a, &b| {
        let (a, b) = (a as usize, b as usize);
        compare_values(keys, &values[a * k..(a + 1) * k], &values[b * k..(b + 1) * k]).then(a.cmp(&b))
    });
    slots.into_iter().map(|s| visible[s as usize]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::key::KeyPolicy;
    use super::super::testing::{TestRow, sort_key};

    fn store_of(rows: Vec<TestRow>) -> RowStore<TestRow> {
        let mut s = RowStore::with_policy(KeyPolicy::Exact);
        s.bulk_load(rows);
        s
    }

    fn numbered(n: usize) -> RowStore<TestRow> {
        store_of((0..n).map(|i| TestRow::new(&format!("/r/{i:03}")).size((i as u64 * 37) % 11).name(&format!("n{}", (i * 7) % 10))).collect())
    }

    fn all(_: &TestRow) -> bool {
        true
    }

    fn inputs<'a>(filter: &'a dyn Fn(&TestRow) -> bool, sig: u64) -> ViewInputs<'a, TestRow> {
        ViewInputs { filter_sig: sig, data_sig: 0, filter, sort_key: &sort_key }
    }

    fn visible_paths(v: &ViewIndex, s: &RowStore<TestRow>) -> Vec<String> {
        v.rows(s).map(|r| r.path.clone()).collect()
    }

    #[test]
    fn an_unsorted_unfiltered_view_mirrors_the_store() {
        let s = numbered(20);
        let mut v = ViewIndex::new();
        assert_eq!(v.refresh(&s, &inputs(&all, 0)), Refresh::Rebuilt);
        assert_eq!(v.len(), 20);
        assert_eq!(visible_paths(&v, &s), s.iter().map(|r| r.path.clone()).collect::<Vec<_>>());
        assert_eq!(v.store_index(3), Some(3));
        assert_eq!(v.view_pos(3), Some(3));
        assert_eq!(v.store_index(20), None);
        v.check_invariants(&s, &inputs(&all, 0)).unwrap();
    }

    #[test]
    fn nothing_is_recomputed_when_nothing_changed() {
        let s = numbered(50);
        let mut v = ViewIndex::new();
        v.refresh(&s, &inputs(&all, 0));
        let rev = v.revision();
        for _ in 0..100 {
            assert_eq!(v.refresh(&s, &inputs(&all, 0)), Refresh::Unchanged);
        }
        assert_eq!(v.revision(), rev);
        assert_eq!(v.counts(), RefreshCounts { unchanged: 100, appended: 0, rebuilt: 1 });
    }

    #[test]
    fn the_filter_decides_visibility_and_the_hidden_count() {
        let s = numbered(30);
        let mut v = ViewIndex::new();
        let even = |r: &TestRow| r.size % 2 == 0;
        v.refresh(&s, &inputs(&even, 1));
        let expected = s.iter().filter(|r| r.size % 2 == 0).count();
        assert_eq!(v.len(), expected);
        assert_eq!(v.stats().visible, expected);
        assert_eq!(v.stats().total, 30);
        assert_eq!(v.stats().hidden_by_filter, 30 - expected);
        v.check_invariants(&s, &inputs(&even, 1)).unwrap();
        for i in 0..s.len() {
            assert_eq!(v.is_visible(i), s.get(i).unwrap().size % 2 == 0);
        }
    }

    #[test]
    fn changing_the_filter_signature_refreshes_even_though_the_rows_did_not_change() {
        let s = numbered(30);
        let mut v = ViewIndex::new();
        let hide_none = |_: &TestRow| true;
        let hide_all = |_: &TestRow| false;
        v.refresh(&s, &inputs(&hide_none, 1));
        assert_eq!(v.len(), 30);
        assert_eq!(v.refresh(&s, &inputs(&hide_all, 2)), Refresh::Rebuilt);
        assert_eq!(v.len(), 0);
        assert_eq!(v.stats().hidden_by_filter, 30);
        assert_eq!(v.refresh(&s, &inputs(&hide_none, 3)), Refresh::Rebuilt);
        assert_eq!(v.len(), 30);
    }

    #[test]
    fn an_unchanged_signature_keeps_the_old_answer() {
        let s = numbered(10);
        let mut v = ViewIndex::new();
        v.refresh(&s, &inputs(&all, 5));
        let hide_all = |_: &TestRow| false;
        assert_eq!(v.refresh(&s, &inputs(&hide_all, 5)), Refresh::Unchanged, "the signature is the contract for filter changes");
        assert_eq!(v.len(), 10);
    }

    #[test]
    fn single_key_sorts_order_ascending_and_descending_with_store_order_for_ties() {
        let s = numbered(40);
        let mut v = ViewIndex::new();
        v.set_sort(SortSpec::single(1, false));
        v.refresh(&s, &inputs(&all, 0));
        let sizes: Vec<u64> = v.rows(&s).map(|r| r.size).collect();
        assert!(sizes.windows(2).all(|w| w[0] <= w[1]));
        let ties: Vec<&str> = v.rows(&s).filter(|r| r.size == 0).map(|r| r.path.as_str()).collect();
        assert!(ties.windows(2).all(|w| w[0] < w[1]), "ties stay in store order: {ties:?}");
        v.set_sort(SortSpec::single(1, true));
        v.refresh(&s, &inputs(&all, 0));
        let sizes: Vec<u64> = v.rows(&s).map(|r| r.size).collect();
        assert!(sizes.windows(2).all(|w| w[0] >= w[1]));
        v.check_invariants(&s, &inputs(&all, 0)).unwrap();
    }

    #[test]
    fn multi_key_sorts_use_later_keys_to_break_ties() {
        let s = numbered(60);
        let mut v = ViewIndex::new();
        v.set_sort(SortSpec::from_keys(vec![SortKey { column: 1, descending: false }, SortKey { column: 0, descending: true }]));
        v.refresh(&s, &inputs(&all, 0));
        let pairs: Vec<(u64, String)> = v.rows(&s).map(|r| (r.size, r.name.clone())).collect();
        for w in pairs.windows(2) {
            assert!(w[0].0 < w[1].0 || (w[0].0 == w[1].0 && w[0].1 >= w[1].1), "{w:?}");
        }
        v.check_invariants(&s, &inputs(&all, 0)).unwrap();
    }

    #[test]
    fn missing_values_sort_first_ascending_and_last_descending() {
        let mut rows = vec![TestRow::new("/a").tag(Some(5)), TestRow::new("/b").tag(None), TestRow::new("/c").tag(Some(2))];
        rows.push(TestRow::new("/d").tag(None));
        let s = store_of(rows);
        let mut v = ViewIndex::new();
        v.set_sort(SortSpec::single(2, false));
        v.refresh(&s, &inputs(&all, 0));
        assert_eq!(visible_paths(&v, &s), ["/b", "/d", "/c", "/a"]);
        v.set_sort(SortSpec::single(2, true));
        v.refresh(&s, &inputs(&all, 0));
        assert_eq!(visible_paths(&v, &s), ["/a", "/c", "/b", "/d"]);
    }

    #[test]
    fn changing_only_the_sort_reorders_without_touching_the_store() {
        let s = numbered(25);
        let mut v = ViewIndex::new();
        v.refresh(&s, &inputs(&all, 0));
        let rev = s.revision();
        assert!(v.set_sort(SortSpec::single(0, false)));
        assert!(!v.set_sort(SortSpec::single(0, false)));
        assert_eq!(v.refresh(&s, &inputs(&all, 0)), Refresh::Rebuilt);
        assert_eq!(s.revision(), rev);
        let names: Vec<String> = v.rows(&s).map(|r| r.name.clone()).collect();
        assert!(names.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn appending_rows_extends_an_unsorted_view_without_a_rebuild() {
        let mut s = numbered(10);
        let mut v = ViewIndex::new();
        v.refresh(&s, &inputs(&all, 0));
        s.bulk_load((10..25).map(|i| TestRow::new(&format!("/r/{i:03}")).size(i as u64)));
        assert_eq!(v.refresh(&s, &inputs(&all, 0)), Refresh::Appended(15));
        assert_eq!(v.len(), 25);
        assert_eq!(v.counts().rebuilt, 1);
        v.check_invariants(&s, &inputs(&all, 0)).unwrap();
    }

    #[test]
    fn appending_rows_into_a_sorted_view_matches_a_full_rebuild() {
        for spec in [SortSpec::single(1, false), SortSpec::single(1, true), SortSpec::single(0, false), SortSpec::from_keys(vec![SortKey { column: 1, descending: true }, SortKey { column: 0, descending: false }])] {
            let mut s = numbered(30);
            let mut v = ViewIndex::new();
            v.set_sort(spec.clone());
            v.refresh(&s, &inputs(&all, 0));
            for round in 0..5usize {
                s.bulk_load((0..17).map(|i| TestRow::new(&format!("/more/{round}/{i:02}")).size(((i * 13 + round * 5) % 9) as u64).name(&format!("m{}", (i * 3 + round) % 7))));
                assert!(matches!(v.refresh(&s, &inputs(&all, 0)), Refresh::Appended(17)));
                v.check_invariants(&s, &inputs(&all, 0)).unwrap();
            }
            assert_eq!(v.counts().rebuilt, 1, "{spec:?}");
        }
    }

    #[test]
    fn appended_rows_that_the_filter_hides_only_change_the_counts() {
        let mut s = numbered(5);
        let mut v = ViewIndex::new();
        let small = |r: &TestRow| r.size < 100;
        v.refresh(&s, &inputs(&small, 1));
        s.upsert(TestRow::new("/huge").size(1000));
        assert_eq!(v.refresh(&s, &inputs(&small, 1)), Refresh::Appended(1));
        assert_eq!(v.len(), 5);
        assert_eq!(v.stats().total, 6);
        assert_eq!(v.stats().hidden_by_filter, 1);
        v.check_invariants(&s, &inputs(&small, 1)).unwrap();
    }

    #[test]
    fn in_place_edits_removals_and_clears_rebuild_the_view() {
        let mut s = numbered(20);
        let mut v = ViewIndex::new();
        v.set_sort(SortSpec::single(1, false));
        v.refresh(&s, &inputs(&all, 0));
        s.get_mut(0).unwrap().size = 10_000;
        assert_eq!(v.refresh(&s, &inputs(&all, 0)), Refresh::Rebuilt);
        assert_eq!(v.rows(&s).last().unwrap().path, "/r/000");
        s.remove_at(5);
        assert_eq!(v.refresh(&s, &inputs(&all, 0)), Refresh::Rebuilt);
        assert_eq!(v.len(), 19);
        s.retain(|r| r.size < 5);
        assert_eq!(v.refresh(&s, &inputs(&all, 0)), Refresh::Rebuilt);
        v.check_invariants(&s, &inputs(&all, 0)).unwrap();
        s.clear();
        assert_eq!(v.refresh(&s, &inputs(&all, 0)), Refresh::Rebuilt);
        assert!(v.is_empty());
        assert_eq!(*v.stats(), ViewStats::default());
    }

    #[test]
    fn replacing_a_row_in_place_updates_the_cached_counts() {
        let mut s = store_of(vec![TestRow::new("/a").size(10), TestRow::new("/b").size(20)]);
        let mut v = ViewIndex::new();
        v.refresh(&s, &inputs(&all, 0));
        assert_eq!(v.stats().total_size, 30);
        s.upsert(TestRow::new("/b").size(5));
        v.refresh(&s, &inputs(&all, 0));
        assert_eq!(v.stats().total_size, 15);
    }

    #[test]
    fn the_data_signature_forces_a_resort() {
        let s = numbered(10);
        let mut v = ViewIndex::new();
        v.set_sort(SortSpec::single(1, false));
        v.refresh(&s, &ViewInputs { filter_sig: 0, data_sig: 1, filter: &all, sort_key: &sort_key });
        let again = ViewInputs { filter_sig: 0, data_sig: 2, filter: &all, sort_key: &sort_key };
        assert_eq!(v.refresh(&s, &again), Refresh::Rebuilt);
    }

    #[test]
    fn invalidate_forces_a_rebuild() {
        let s = numbered(10);
        let mut v = ViewIndex::new();
        v.refresh(&s, &inputs(&all, 0));
        v.invalidate();
        assert_eq!(v.refresh(&s, &inputs(&all, 0)), Refresh::Rebuilt);
    }

    #[test]
    fn the_revision_moves_only_when_the_view_changed() {
        let mut s = numbered(10);
        let mut v = ViewIndex::new();
        let r0 = v.revision();
        v.refresh(&s, &inputs(&all, 0));
        let r1 = v.revision();
        assert!(r1 > r0);
        v.refresh(&s, &inputs(&all, 0));
        assert_eq!(v.revision(), r1);
        s.upsert(TestRow::new("/new"));
        v.refresh(&s, &inputs(&all, 0));
        assert!(v.revision() > r1);
    }

    #[test]
    fn stats_count_each_kind_and_sum_sizes_without_directories() {
        let s = store_of(vec![
            TestRow::new("/d").kind(RowKind::Dir).size(4096),
            TestRow::new("/i1").kind(RowKind::Image).size(100),
            TestRow::new("/i2").kind(RowKind::Image).size(50),
            TestRow::new("/v").kind(RowKind::Video).size(1000),
            TestRow::new("/z").kind(RowKind::Archive).size(7),
            TestRow::new("/o").kind(RowKind::Other).size(3),
        ]);
        let mut v = ViewIndex::new();
        v.refresh(&s, &inputs(&all, 0));
        let st = v.stats();
        assert_eq!((st.images, st.videos, st.dirs, st.archives, st.other), (2, 1, 1, 1, 1));
        assert_eq!(st.total_size, 100 + 50 + 1000 + 7 + 3);
        assert_eq!((st.total, st.visible, st.hidden_by_filter), (6, 6, 0));
    }

    #[test]
    fn sort_click_cycles_ascending_descending_off() {
        let mut spec = SortSpec::new();
        spec.click(3, false);
        assert_eq!(spec.keys(), [SortKey { column: 3, descending: false }]);
        spec.click(3, false);
        assert_eq!(spec.keys(), [SortKey { column: 3, descending: true }]);
        spec.click(3, false);
        assert!(spec.is_empty());
    }

    #[test]
    fn a_plain_click_on_another_column_replaces_every_key() {
        let mut spec = SortSpec::from_keys(vec![SortKey { column: 1, descending: true }, SortKey { column: 2, descending: false }]);
        spec.click(4, false);
        assert_eq!(spec.keys(), [SortKey { column: 4, descending: false }]);
        let mut spec = SortSpec::from_keys(vec![SortKey { column: 1, descending: false }, SortKey { column: 2, descending: false }]);
        spec.click(1, false);
        assert_eq!(spec.keys(), [SortKey { column: 1, descending: false }], "a plain click on one of several keys restarts from it");
    }

    #[test]
    fn a_shift_click_adds_cycles_and_removes_keys_in_click_order() {
        let mut spec = SortSpec::new();
        spec.click(1, true);
        spec.click(2, true);
        spec.click(3, true);
        assert_eq!(spec.keys().iter().map(|k| k.column).collect::<Vec<_>>(), [1, 2, 3]);
        assert_eq!(spec.position_of(2), Some((1, false)));
        spec.click(2, true);
        assert_eq!(spec.position_of(2), Some((1, true)));
        spec.click(2, true);
        assert_eq!(spec.position_of(2), None);
        assert_eq!(spec.keys().iter().map(|k| k.column).collect::<Vec<_>>(), [1, 3]);
        assert!(spec.uses(3) && !spec.uses(2));
    }

    #[test]
    fn duplicate_columns_are_dropped_when_building_a_spec() {
        let spec = SortSpec::from_keys(vec![SortKey { column: 1, descending: true }, SortKey { column: 1, descending: false }, SortKey { column: 2, descending: false }]);
        assert_eq!(spec.keys().len(), 2);
        assert_eq!(spec.position_of(1), Some((0, true)));
    }

    #[test]
    fn retain_columns_drops_keys_for_unavailable_columns() {
        let mut spec = SortSpec::from_keys(vec![SortKey { column: 1, descending: true }, SortKey { column: 9, descending: false }]);
        spec.retain_columns(|c| c < 5);
        assert_eq!(spec.keys().len(), 1);
    }

    #[test]
    fn a_sort_spec_round_trips_through_json() {
        let spec = SortSpec::from_keys(vec![SortKey { column: 6, descending: true }, SortKey { column: 1, descending: false }]);
        let json = serde_json::to_string(&spec).unwrap();
        assert_eq!(serde_json::from_str::<SortSpec>(&json).unwrap(), spec);
    }

    #[test]
    fn sort_values_order_by_kind_then_content() {
        use SortValue::*;
        assert_eq!(Missing.compare(&Int(0)), Ordering::Less);
        assert_eq!(Int(5).compare(&Int(7)), Ordering::Less);
        assert_eq!(Float(f32::NAN).compare(&Float(1.0)), Ordering::Greater);
        assert_eq!(Text("a".into()).compare(&Text("b".into())), Ordering::Less);
        assert_eq!(Flag(false).compare(&Flag(true)), Ordering::Less);
        assert_eq!(Missing.compare(&Missing), Ordering::Equal);
    }

    #[test]
    fn a_stale_view_fails_the_invariant_check() {
        let mut s = numbered(5);
        let mut v = ViewIndex::new();
        v.refresh(&s, &inputs(&all, 0));
        s.upsert(TestRow::new("/late"));
        assert!(v.check_invariants(&s, &inputs(&all, 0)).unwrap_err().contains("stale"));
    }
}
