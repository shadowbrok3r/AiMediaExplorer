//! Row storage: rows in insertion order and a path-key index that always agree.

use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use super::key::{KeyPolicy, PathKey, normalize};

/// A row addressable by path.
pub trait Keyed {
    fn key_path(&self) -> &str;
}

/// What a write did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Upsert {
    Inserted(usize),
    Replaced(usize),
    Rejected,
}

impl Upsert {
    pub fn index(self) -> Option<usize> {
        match self {
            Upsert::Inserted(i) | Upsert::Replaced(i) => Some(i),
            Upsert::Rejected => None,
        }
    }
}

/// Totals of a bulk write.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoadSummary {
    pub inserted: usize,
    pub replaced: usize,
    pub rejected: usize,
}

/// Change counters of a store; a reader that saw the same values and length sees the same rows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Versions {
    /// Bumped by every change.
    pub revision: u64,
    /// Bumped when rows move, vanish or are all replaced.
    pub layout: u64,
    /// Bumped when a row is rewritten in place.
    pub edits: u64,
}

/// Memory held by a store, in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryEstimate {
    pub row_inline: usize,
    pub row_heap: usize,
    pub keys: usize,
    pub index: usize,
}

impl MemoryEstimate {
    pub fn total(&self) -> usize {
        self.row_inline + self.row_heap + self.keys + self.index
    }
}

pub struct RowStore<R> {
    rows: Vec<R>,
    keys: Vec<PathKey>,
    index: HashMap<PathKey, usize>,
    policy: KeyPolicy,
    versions: Versions,
}

impl<R> Default for RowStore<R> {
    fn default() -> Self {
        Self::with_policy(KeyPolicy::NATIVE)
    }
}

impl<R> RowStore<R> {
    pub fn with_policy(policy: KeyPolicy) -> Self {
        Self { rows: Vec::new(), keys: Vec::new(), index: HashMap::new(), policy, versions: Versions::default() }
    }

    pub fn policy(&self) -> KeyPolicy {
        self.policy
    }

    pub fn versions(&self) -> Versions {
        self.versions
    }

    pub fn revision(&self) -> u64 {
        self.versions.revision
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn get(&self, i: usize) -> Option<&R> {
        self.rows.get(i)
    }

    pub fn first(&self) -> Option<&R> {
        self.rows.first()
    }

    pub fn last(&self) -> Option<&R> {
        self.rows.last()
    }

    pub fn rows(&self) -> &[R] {
        &self.rows
    }

    pub fn iter(&self) -> std::slice::Iter<'_, R> {
        self.rows.iter()
    }

    pub fn key_at(&self, i: usize) -> Option<&PathKey> {
        self.keys.get(i)
    }

    pub fn keys(&self) -> &[PathKey] {
        &self.keys
    }

    /// Row position of `path`, found under the store's key policy.
    pub fn position(&self, path: &str) -> Option<usize> {
        match self.policy {
            KeyPolicy::Exact => self.index.get(path).copied(),
            KeyPolicy::Windows => self.index.get(normalize(path, self.policy).as_str()).copied(),
        }
    }

    pub fn position_of_key(&self, key: &PathKey) -> Option<usize> {
        self.index.get(key).copied()
    }

    pub fn contains_path(&self, path: &str) -> bool {
        self.position(path).is_some()
    }

    pub fn get_by_path(&self, path: &str) -> Option<&R> {
        self.position(path).and_then(|i| self.rows.get(i))
    }

    pub fn get_by_key(&self, key: &PathKey) -> Option<&R> {
        self.position_of_key(key).and_then(|i| self.rows.get(i))
    }

    /// Key for `path` under this store's policy.
    pub fn key_for(&self, path: &str) -> PathKey {
        PathKey::with_policy(path, self.policy)
    }

    pub fn reserve(&mut self, additional: usize) {
        self.rows.reserve(additional);
        self.keys.reserve(additional);
        self.index.reserve(additional);
    }

    fn note_append(&mut self) {
        self.versions.revision += 1;
    }

    fn note_edit(&mut self) {
        self.versions.revision += 1;
        self.versions.edits += 1;
    }

    fn note_layout(&mut self) {
        self.versions.revision += 1;
        self.versions.layout += 1;
    }

    /// Removes every row.
    pub fn clear(&mut self) {
        if self.rows.is_empty() {
            return;
        }
        self.rows.clear();
        self.keys.clear();
        self.index.clear();
        self.note_layout();
    }

    /// Memory held by the store; `row_heap` sums `heap_bytes` over the rows.
    pub fn estimate_bytes(&self, heap_bytes: impl Fn(&R) -> usize) -> MemoryEstimate {
        const ARC_HEADER: usize = 16;
        let key_text: usize = self.keys.iter().map(|k| k.as_str().len() + ARC_HEADER).sum();
        MemoryEstimate {
            row_inline: self.rows.capacity() * std::mem::size_of::<R>(),
            row_heap: self.rows.iter().map(heap_bytes).sum(),
            keys: key_text + self.keys.capacity() * std::mem::size_of::<PathKey>(),
            index: self.index.capacity() * (std::mem::size_of::<(PathKey, usize)>() + 1),
        }
    }
}

impl<R: Keyed> RowStore<R> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(policy: KeyPolicy, capacity: usize) -> Self {
        let mut store = Self::with_policy(policy);
        store.reserve(capacity);
        store
    }

    /// Writes `row`: appended when its path is new, otherwise replacing the row of that path in place.
    pub fn upsert(&mut self, row: R) -> Upsert {
        let path = row.key_path();
        if path.is_empty() {
            return Upsert::Rejected;
        }
        match self.position(path) {
            Some(i) => {
                self.rows[i] = row;
                self.note_edit();
                Upsert::Replaced(i)
            }
            None => {
                let key = PathKey::with_policy(path, self.policy);
                let i = self.rows.len();
                self.index.insert(key.clone(), i);
                self.keys.push(key);
                self.rows.push(row);
                self.note_append();
                Upsert::Inserted(i)
            }
        }
    }

    /// Same as [`upsert`](Self::upsert).
    pub fn push(&mut self, row: R) -> Upsert {
        self.upsert(row)
    }

    /// Appends `row` unless its path is present; returns the new position.
    pub fn insert_if_absent(&mut self, row: R) -> Option<usize> {
        if row.key_path().is_empty() || self.contains_path(row.key_path()) {
            return None;
        }
        self.upsert(row).index()
    }

    /// Writes every row in order.
    pub fn bulk_load(&mut self, rows: impl IntoIterator<Item = R>) -> LoadSummary {
        let rows = rows.into_iter();
        self.reserve(rows.size_hint().0);
        let mut summary = LoadSummary::default();
        for row in rows {
            match self.upsert(row) {
                Upsert::Inserted(_) => summary.inserted += 1,
                Upsert::Replaced(_) => summary.replaced += 1,
                Upsert::Rejected => summary.rejected += 1,
            }
        }
        summary
    }

    /// Drops every row, then loads `rows`.
    pub fn replace_all(&mut self, rows: impl IntoIterator<Item = R>) -> LoadSummary {
        self.clear();
        self.bulk_load(rows)
    }

    /// Removes the row at position `i`, shifting later rows down.
    pub fn remove_at(&mut self, i: usize) -> Option<R> {
        if i >= self.rows.len() {
            return None;
        }
        let row = self.rows.remove(i);
        let key = self.keys.remove(i);
        self.index.remove(&key);
        for j in i..self.rows.len() {
            if let Some(slot) = self.index.get_mut(&self.keys[j]) {
                *slot = j;
            }
        }
        self.note_layout();
        Some(row)
    }

    pub fn remove_path(&mut self, path: &str) -> Option<R> {
        let i = self.position(path)?;
        self.remove_at(i)
    }

    /// Keeps the rows `keep` accepts, in order; returns how many were removed.
    pub fn retain(&mut self, mut keep: impl FnMut(&R) -> bool) -> usize {
        let flags: Vec<bool> = self.rows.iter().map(&mut keep).collect();
        let removed = flags.iter().filter(|k| !**k).count();
        if removed == 0 {
            return 0;
        }
        let mut write = 0;
        for read in 0..flags.len() {
            if flags[read] {
                if write != read {
                    self.rows.swap(write, read);
                    self.keys.swap(write, read);
                    if let Some(slot) = self.index.get_mut(&self.keys[write]) {
                        *slot = write;
                    }
                }
                write += 1;
            } else {
                self.index.remove(&self.keys[read]);
            }
        }
        self.rows.truncate(write);
        self.keys.truncate(write);
        self.note_layout();
        removed
    }

    /// Mutable access to one row; the guard restores the invariants when dropped, even if the path was edited.
    pub fn get_mut(&mut self, i: usize) -> Option<RowMut<'_, R>> {
        if i >= self.rows.len() {
            return None;
        }
        Some(RowMut { store: self, idx: i })
    }

    pub fn get_mut_by_path(&mut self, path: &str) -> Option<RowMut<'_, R>> {
        let i = self.position(path)?;
        self.get_mut(i)
    }

    /// Runs `edit` on every row, then restores the invariants for any row whose path changed.
    pub fn for_each_mut(&mut self, mut edit: impl FnMut(&mut R)) {
        if self.rows.is_empty() {
            return;
        }
        self.rows.iter_mut().for_each(&mut edit);
        self.note_edit();
        let intact = self.rows.iter().zip(&self.keys).all(|(row, key)| self.key_matches(row.key_path(), key));
        if !intact {
            self.rebuild_from_rows();
        }
    }

    fn key_matches(&self, path: &str, key: &PathKey) -> bool {
        match self.policy {
            KeyPolicy::Exact => path == key.as_str(),
            KeyPolicy::Windows => !path.is_empty() && normalize(path, self.policy) == key.as_str(),
        }
    }

    fn rebuild_from_rows(&mut self) {
        let rows = std::mem::take(&mut self.rows);
        self.keys.clear();
        self.index.clear();
        self.reserve(rows.len());
        for row in rows {
            self.upsert(row);
        }
        self.note_layout();
    }

    fn settle(&mut self, i: usize) {
        self.note_edit();
        let intact = match (self.rows.get(i), self.keys.get(i)) {
            (Some(row), Some(key)) => !row.key_path().is_empty() && self.key_matches(row.key_path(), key),
            _ => true,
        };
        if !intact {
            if let Some(row) = self.remove_at(i) {
                self.upsert(row);
            }
        }
    }

    /// Checks that lengths agree, each row has one unique non-empty key equal to its normalized path, and the index maps it back.
    pub fn check_invariants(&self) -> Result<(), String> {
        if self.keys.len() != self.rows.len() {
            return Err(format!("{} keys for {} rows", self.keys.len(), self.rows.len()));
        }
        if self.index.len() != self.rows.len() {
            return Err(format!("{} index entries for {} rows", self.index.len(), self.rows.len()));
        }
        for (i, (row, key)) in self.rows.iter().zip(&self.keys).enumerate() {
            if key.is_empty() {
                return Err(format!("row {i} has an empty key"));
            }
            if !self.key_matches(row.key_path(), key) {
                return Err(format!("row {i} path {:?} does not match its key {:?}", row.key_path(), key.as_str()));
            }
            match self.index.get(key) {
                Some(&at) if at == i => {}
                Some(&at) => return Err(format!("key {:?} of row {i} is indexed at {at}", key.as_str())),
                None => return Err(format!("key {:?} of row {i} is not indexed", key.as_str())),
            }
        }
        Ok(())
    }
}

impl<'a, R> IntoIterator for &'a RowStore<R> {
    type Item = &'a R;
    type IntoIter = std::slice::Iter<'a, R>;

    fn into_iter(self) -> Self::IntoIter {
        self.rows.iter()
    }
}

/// Mutable handle to one row; see [`RowStore::get_mut`].
pub struct RowMut<'a, R: Keyed> {
    store: &'a mut RowStore<R>,
    idx: usize,
}

impl<R: Keyed> Deref for RowMut<'_, R> {
    type Target = R;

    fn deref(&self) -> &R {
        &self.store.rows[self.idx]
    }
}

impl<R: Keyed> DerefMut for RowMut<'_, R> {
    fn deref_mut(&mut self) -> &mut R {
        &mut self.store.rows[self.idx]
    }
}

impl<R: Keyed> Drop for RowMut<'_, R> {
    fn drop(&mut self) {
        self.store.settle(self.idx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::testing::TestRow;

    fn row(path: &str) -> TestRow {
        TestRow::new(path)
    }

    fn store(paths: &[&str]) -> RowStore<TestRow> {
        let mut s = RowStore::with_policy(KeyPolicy::Exact);
        s.bulk_load(paths.iter().map(|p| row(p)));
        s
    }

    fn paths(s: &RowStore<TestRow>) -> Vec<&str> {
        s.iter().map(|r| r.path.as_str()).collect()
    }

    #[test]
    fn upsert_appends_new_paths_and_replaces_existing_ones_in_place() {
        let mut s = RowStore::with_policy(KeyPolicy::Exact);
        assert_eq!(s.upsert(row("/a")), Upsert::Inserted(0));
        assert_eq!(s.upsert(row("/b")), Upsert::Inserted(1));
        let mut changed = row("/a");
        changed.size = 99;
        assert_eq!(s.upsert(changed), Upsert::Replaced(0));
        assert_eq!(s.len(), 2);
        assert_eq!(s.get(0).unwrap().size, 99);
        assert_eq!(paths(&s), ["/a", "/b"]);
        s.check_invariants().unwrap();
    }

    #[test]
    fn rows_without_a_path_are_rejected() {
        let mut s = RowStore::with_policy(KeyPolicy::Exact);
        assert_eq!(s.upsert(row("")), Upsert::Rejected);
        assert!(s.is_empty());
        assert_eq!(s.versions(), Versions::default());
        assert_eq!(s.insert_if_absent(row("")), None);
    }

    #[test]
    fn push_is_an_upsert() {
        let mut s = store(&["/a"]);
        assert_eq!(s.push(row("/a")), Upsert::Replaced(0));
        assert_eq!(s.push(row("/b")), Upsert::Inserted(1));
    }

    #[test]
    fn insert_if_absent_never_overwrites() {
        let mut s = store(&["/a"]);
        let mut other = row("/a");
        other.size = 5;
        assert_eq!(s.insert_if_absent(other), None);
        assert_eq!(s.get(0).unwrap().size, 0);
        assert_eq!(s.insert_if_absent(row("/b")), Some(1));
    }

    #[test]
    fn lookups_work_by_path_key_and_position() {
        let s = store(&["/a", "/b", "/c"]);
        assert_eq!(s.position("/b"), Some(1));
        assert_eq!(s.position("/nope"), None);
        assert!(s.contains_path("/c"));
        assert_eq!(s.get_by_path("/c").unwrap().path, "/c");
        let key = s.key_for("/b");
        assert_eq!(s.position_of_key(&key), Some(1));
        assert_eq!(s.get_by_key(&key).unwrap().path, "/b");
        assert_eq!(s.key_at(2).unwrap().as_str(), "/c");
        assert_eq!(s.first().unwrap().path, "/a");
        assert_eq!(s.last().unwrap().path, "/c");
    }

    #[test]
    fn remove_shifts_later_rows_and_keeps_the_index_exact() {
        let mut s = store(&["/a", "/b", "/c", "/d"]);
        assert_eq!(s.remove_path("/b").unwrap().path, "/b");
        assert_eq!(paths(&s), ["/a", "/c", "/d"]);
        assert_eq!(s.position("/c"), Some(1));
        assert_eq!(s.position("/d"), Some(2));
        assert_eq!(s.position("/b"), None);
        assert!(s.remove_path("/b").is_none());
        assert!(s.remove_at(9).is_none());
        s.check_invariants().unwrap();
        assert_eq!(s.remove_at(0).unwrap().path, "/a");
        assert_eq!(s.position("/c"), Some(0));
        s.check_invariants().unwrap();
    }

    #[test]
    fn retain_compacts_in_order_and_reindexes() {
        let mut s = store(&["/a", "/b", "/c", "/d", "/e", "/f"]);
        let removed = s.retain(|r| !matches!(r.path.as_str(), "/a" | "/c" | "/d"));
        assert_eq!(removed, 3);
        assert_eq!(paths(&s), ["/b", "/e", "/f"]);
        assert_eq!((s.position("/b"), s.position("/e"), s.position("/f")), (Some(0), Some(1), Some(2)));
        assert_eq!((s.position("/a"), s.position("/c"), s.position("/d")), (None, None, None));
        s.check_invariants().unwrap();
        assert_eq!(s.retain(|_| true), 0);
        assert_eq!(s.retain(|_| false), 3);
        assert!(s.is_empty());
        s.check_invariants().unwrap();
    }

    #[test]
    fn a_retain_predicate_that_panics_leaves_the_store_intact() {
        let mut s = store(&["/a", "/b", "/c"]);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            s.retain(|r| {
                if r.path == "/c" {
                    panic!("predicate failed");
                }
                false
            })
        }));
        assert!(result.is_err());
        assert_eq!(paths(&s), ["/a", "/b", "/c"]);
        s.check_invariants().unwrap();
    }

    #[test]
    fn clear_and_replace_all_reset_the_content() {
        let mut s = store(&["/a", "/b"]);
        let summary = s.replace_all([row("/x"), row("/y"), row("/x"), row("")]);
        assert_eq!(summary, LoadSummary { inserted: 2, replaced: 1, rejected: 1 });
        assert_eq!(paths(&s), ["/x", "/y"]);
        s.check_invariants().unwrap();
        s.clear();
        assert!(s.is_empty() && s.position("/x").is_none());
        s.check_invariants().unwrap();
    }

    #[test]
    fn bulk_load_counts_each_outcome() {
        let mut s = store(&["/a"]);
        let summary = s.bulk_load([row("/a"), row("/b"), row("")]);
        assert_eq!(summary, LoadSummary { inserted: 1, replaced: 1, rejected: 1 });
    }

    #[test]
    fn versions_tell_appends_from_edits_and_layout_changes() {
        let mut s = RowStore::with_policy(KeyPolicy::Exact);
        s.upsert(row("/a"));
        s.upsert(row("/b"));
        let v = s.versions();
        assert_eq!((v.layout, v.edits), (0, 0));
        assert_eq!(v.revision, 2);
        s.upsert(row("/a"));
        let v = s.versions();
        assert_eq!((v.layout, v.edits, v.revision), (0, 1, 3));
        s.remove_path("/b");
        let v = s.versions();
        assert_eq!((v.layout, v.edits, v.revision), (1, 1, 4));
        s.clear();
        assert_eq!(s.versions().layout, 2);
        let before = s.versions();
        s.clear();
        assert_eq!(s.versions(), before, "clearing an empty store changes nothing");
    }

    #[test]
    fn editing_through_the_guard_keeps_the_key_when_the_path_is_untouched() {
        let mut s = store(&["/a", "/b"]);
        {
            let mut r = s.get_mut(1).unwrap();
            r.size = 7;
            r.name = "renamed".into();
        }
        assert_eq!(s.get(1).unwrap().size, 7);
        assert_eq!(s.position("/b"), Some(1));
        s.check_invariants().unwrap();
        assert!(s.get_mut(5).is_none());
        assert!(s.get_mut_by_path("/zzz").is_none());
        s.get_mut_by_path("/a").unwrap().size = 3;
        assert_eq!(s.get(0).unwrap().size, 3);
    }

    #[test]
    fn editing_a_path_through_the_guard_rekeys_the_row() {
        let mut s = store(&["/a", "/b", "/c"]);
        s.get_mut(0).unwrap().path = "/renamed".into();
        s.check_invariants().unwrap();
        assert!(s.position("/a").is_none());
        assert!(s.contains_path("/renamed"));
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn a_path_edit_onto_an_existing_key_replaces_that_row() {
        let mut s = store(&["/a", "/b", "/c"]);
        {
            let mut r = s.get_mut(0).unwrap();
            r.path = "/c".into();
            r.size = 42;
        }
        s.check_invariants().unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(s.get_by_path("/c").unwrap().size, 42);
        assert!(s.position("/a").is_none());
    }

    #[test]
    fn a_path_edited_to_empty_drops_the_row() {
        let mut s = store(&["/a", "/b"]);
        s.get_mut(1).unwrap().path.clear();
        s.check_invariants().unwrap();
        assert_eq!(paths(&s), ["/a"]);
    }

    #[test]
    fn for_each_mut_edits_all_rows_and_repairs_changed_paths() {
        let mut s = store(&["/a", "/b", "/c"]);
        s.for_each_mut(|r| r.size += 1);
        assert!(s.iter().all(|r| r.size == 1));
        s.check_invariants().unwrap();
        s.for_each_mut(|r| {
            if r.path == "/b" {
                r.path = "/a".into();
                r.size = 9;
            }
        });
        s.check_invariants().unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(s.get_by_path("/a").unwrap().size, 9, "the later row wins the shared key");
        s.for_each_mut(|r| r.path.clear());
        s.check_invariants().unwrap();
        assert!(s.is_empty());
    }

    #[test]
    fn windows_policy_matches_paths_that_differ_in_case_and_separators() {
        let mut s = RowStore::with_policy(KeyPolicy::Windows);
        assert_eq!(s.upsert(row("C:/Pics/A.JPG")), Upsert::Inserted(0));
        assert_eq!(s.upsert(row("c:\\pics\\a.jpg")), Upsert::Replaced(0));
        assert_eq!(s.len(), 1);
        assert_eq!(s.position("C:\\PICS/a.Jpg"), Some(0));
        s.check_invariants().unwrap();
        s.get_mut(0).unwrap().size = 5;
        s.check_invariants().unwrap();
        assert!(s.remove_path("C:/PICS/A.jpg").is_some());
        assert!(s.is_empty());
    }

    #[test]
    fn windows_policy_keeps_archive_internals_and_virtual_names_distinct() {
        let mut s = RowStore::with_policy(KeyPolicy::Windows);
        s.bulk_load([row("zip://C:/A.zip!/In/x.png"), row("zip://c:\\a.zip!/in/x.png"), row("tag://Fox"), row("tag://fox")]);
        assert_eq!(s.len(), 4);
        s.check_invariants().unwrap();
        assert_eq!(s.position("zip://C:\\A.ZIP!/In/x.png"), Some(0));
    }

    #[test]
    fn the_invariant_check_reports_corruption() {
        let mut s = store(&["/a", "/b"]);
        s.rows[0].path = "/changed".into();
        assert!(s.check_invariants().unwrap_err().contains("does not match"));
        let mut s = store(&["/a", "/b"]);
        s.index.insert(s.keys[0].clone(), 1);
        assert!(s.check_invariants().unwrap_err().contains("indexed at"));
        let mut s = store(&["/a", "/b"]);
        s.index.remove("/b");
        assert!(s.check_invariants().is_err());
        let mut s = store(&["/a", "/b"]);
        s.keys.pop();
        assert!(s.check_invariants().is_err());
    }

    #[test]
    fn the_memory_estimate_grows_with_rows() {
        let mut s = RowStore::with_policy(KeyPolicy::Exact);
        let empty = s.estimate_bytes(|r: &TestRow| r.path.len());
        s.bulk_load((0..1000).map(|i| row(&format!("/photos/{i:05}.jpg"))));
        let full = s.estimate_bytes(|r| r.path.len());
        assert!(full.total() > empty.total() + 1000 * 20);
        assert_eq!(full.row_heap, 1000 * "/photos/00000.jpg".len());
    }

    #[test]
    fn iteration_follows_insertion_order() {
        let s = store(&["/c", "/a", "/b"]);
        let via_ref: Vec<&str> = (&s).into_iter().map(|r| r.path.as_str()).collect();
        assert_eq!(via_ref, ["/c", "/a", "/b"]);
        assert_eq!(s.rows().len(), 3);
        assert_eq!(s.keys().len(), 3);
        assert_eq!(s.revision(), 3);
    }
}
