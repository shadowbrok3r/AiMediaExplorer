use crate::Thumbnail;
use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hasher};

const MAX_INDEXES: usize = 16;

fn path_hash(path: &str) -> u64 {
    let mut h = DefaultHasher::new();
    h.write(path.as_bytes());
    h.finish()
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Snapshot {
    Scanned,
    Filtered,
}

// Path hash -> row position, extended lazily as rows are appended and rebuilt when the indexed prefix changed.
#[derive(Default)]
struct PathIndex {
    map: HashMap<u64, u32>,
    synced: usize,
    samples: [u64; 3],
}

fn sample_positions(len: usize) -> [usize; 3] {
    [0, len / 2, len.saturating_sub(1)]
}

impl PathIndex {
    fn prefix_intact(&self, rows: &[Thumbnail]) -> bool {
        if self.synced == 0 {
            return true;
        }
        if rows.len() < self.synced {
            return false;
        }
        sample_positions(self.synced).iter().zip(self.samples).all(|(&i, h)| path_hash(&rows[i].path) == h)
    }

    fn sync(&mut self, rows: &[Thumbnail]) {
        if !self.prefix_intact(rows) {
            self.map.clear();
            self.synced = 0;
        }
        for (i, r) in rows.iter().enumerate().skip(self.synced) {
            self.map.insert(path_hash(&r.path), i as u32);
        }
        self.synced = rows.len();
        self.samples = if rows.is_empty() { [0; 3] } else { sample_positions(rows.len()).map(|i| path_hash(&rows[i].path)) };
    }

    fn find(&mut self, rows: &[Thumbnail], path: &str) -> Option<usize> {
        self.sync(rows);
        let idx = *self.map.get(&path_hash(path))? as usize;
        (rows.get(idx)?.path == path).then_some(idx)
    }
}

thread_local! {
    static INDEXES: RefCell<Vec<((u64, Snapshot), PathIndex)>> = const { RefCell::new(Vec::new()) };
}

fn locate(scan_id: u64, which: Snapshot, rows: &[Thumbnail], path: &str) -> Option<usize> {
    INDEXES.with(|cell| {
        let mut entries = cell.borrow_mut();
        let key = (scan_id, which);
        match entries.iter().position(|(k, _)| *k == key) {
            Some(0) => {}
            Some(pos) => {
                let entry = entries.remove(pos);
                entries.insert(0, entry);
            }
            None => {
                if entries.len() >= MAX_INDEXES {
                    entries.pop();
                }
                entries.insert(0, (key, PathIndex::default()));
            }
        }
        entries[0].1.find(rows, path)
    })
}

impl crate::ui::file_table::FileExplorer {
    // Row position in `table` for `path`; repairs `table_index` when it no longer matches the table.
    pub(super) fn table_pos(&mut self, path: &str) -> Option<usize> {
        let valid = |this: &Self, idx: usize| this.table.get(idx).is_some_and(|r| r.path == path);
        match self.table_index.get(path).copied() {
            Some(idx) if valid(self, idx) => return Some(idx),
            Some(_) => {}
            None if self.table_index.len() == self.table.len() => return None,
            None => {}
        }
        self.rebuild_table_index();
        self.table_index.get(path).copied().filter(|&idx| valid(self, idx))
    }

    pub(super) fn rebuild_table_index(&mut self) {
        self.table_index.clear();
        for (i, r) in self.table.iter().enumerate() {
            self.table_index.insert(r.path.clone(), i);
        }
    }

    // Stores the thumbnail on the scan snapshot row and on the filtered snapshot row for `path`.
    pub(super) fn set_snapshot_thumb(&mut self, scan_id: u64, path: &str, thumb: &str) {
        if let Some(i) = locate(scan_id, Snapshot::Scanned, &self.last_scan_rows, path) {
            self.last_scan_rows[i].thumbnail_b64 = Some(thumb.to_string());
        }
        if !self.recursive_filtered_rows.is_empty() {
            if let Some(i) = locate(scan_id, Snapshot::Filtered, &self.recursive_filtered_rows, path) {
                self.recursive_filtered_rows[i].thumbnail_b64 = Some(thumb.to_string());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(path: &str) -> Thumbnail {
        let mut t = Thumbnail::new("x");
        t.path = path.to_string();
        t
    }

    fn rows(n: usize) -> Vec<Thumbnail> {
        (0..n).map(|i| row(&format!("/data/photos/dir{}/img_{i}.jpg", i % 97))).collect()
    }

    #[test]
    fn finds_appended_rows_incrementally() {
        let mut v = rows(10);
        assert_eq!(locate(9_001, Snapshot::Scanned, &v, "/data/photos/dir3/img_3.jpg"), Some(3));
        v.extend(rows(30).into_iter().skip(10));
        assert_eq!(locate(9_001, Snapshot::Scanned, &v, "/data/photos/dir25/img_25.jpg"), Some(25));
        assert_eq!(locate(9_001, Snapshot::Scanned, &v, "/missing.jpg"), None);
    }

    #[test]
    fn rebuilds_after_clear_and_refill() {
        let mut v = rows(50);
        assert_eq!(locate(9_002, Snapshot::Scanned, &v, "/data/photos/dir4/img_4.jpg"), Some(4));
        v.clear();
        v.extend((0..60).map(|i| row(&format!("/other/{i}.png"))));
        assert_eq!(locate(9_002, Snapshot::Scanned, &v, "/data/photos/dir4/img_4.jpg"), None);
        assert_eq!(locate(9_002, Snapshot::Scanned, &v, "/other/59.png"), Some(59));
        assert_eq!(locate(9_002, Snapshot::Scanned, &v, "/other/0.png"), Some(0));
    }

    #[test]
    fn rebuilds_after_replacement_with_reordered_rows() {
        let mut v = rows(40);
        assert_eq!(locate(9_003, Snapshot::Filtered, &v, "/data/photos/dir5/img_5.jpg"), Some(5));
        v.reverse();
        assert_eq!(locate(9_003, Snapshot::Filtered, &v, "/data/photos/dir5/img_5.jpg"), Some(34));
        v.remove(0);
        assert_eq!(locate(9_003, Snapshot::Filtered, &v, "/data/photos/dir5/img_5.jpg"), Some(33));
    }

    #[test]
    fn hit_is_validated_against_the_row_path() {
        let mut v = rows(20);
        assert_eq!(locate(9_004, Snapshot::Scanned, &v, "/data/photos/dir7/img_7.jpg"), Some(7));
        v[7].path = "/changed.jpg".into();
        assert_eq!(locate(9_004, Snapshot::Scanned, &v, "/data/photos/dir7/img_7.jpg"), None);
    }

    #[test]
    fn evicts_least_recently_used_index() {
        let v = rows(5);
        for id in 0..(MAX_INDEXES as u64 + 4) {
            assert_eq!(locate(10_000 + id, Snapshot::Scanned, &v, "/data/photos/dir1/img_1.jpg"), Some(1));
        }
        let count = INDEXES.with(|c| c.borrow().len());
        assert_eq!(count, MAX_INDEXES);
    }

    #[test]
    #[ignore]
    fn bench_index_vs_linear_find() {
        for n in [5_000usize, 20_000, 100_000] {
            let mut v = rows(n);
            let keys: Vec<String> = v.iter().map(|r| r.path.clone()).collect();
            let t = std::time::Instant::now();
            for k in &keys {
                if let Some(r) = v.iter_mut().find(|r| &r.path == k) {
                    r.thumbnail_b64 = Some(String::new());
                }
            }
            let linear = t.elapsed();
            let t = std::time::Instant::now();
            for k in &keys {
                if let Some(i) = locate(77_000 + n as u64, Snapshot::Scanned, &v, k) {
                    v[i].thumbnail_b64 = Some(String::new());
                }
            }
            let indexed = t.elapsed();
            println!("bench lookup n={n}: linear {linear:?} vs indexed {indexed:?}");
        }
    }
}
