//! Duplicate detection: rows grouped by size, then confirmed by blake3 content hash.

use std::collections::HashMap;
use std::io::Read;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::database::scope_queries::SizeRow;

/// Files hashed at the same time.
const HASH_PARALLEL: usize = 4;

/// Computes the content hash of a file, or `None` when it cannot be read.
pub type Hasher = Arc<dyn Fn(&str) -> Option<String> + Send + Sync>;

/// Files with identical contents.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Group {
    pub hash: String,
    pub size: u64,
    /// Paths in order.
    pub paths: Vec<String>,
}

/// The blake3 hash of the whole file as hex, the same form the indexer stores.
pub fn file_hash(path: &str) -> Option<String> {
    let mut file = std::fs::File::open(path).ok()?;
    let mut hasher = blake3::Hasher::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf).ok()?;
        if n == 0 {
            return Some(hasher.finalize().to_hex().to_string());
        }
        hasher.update(&buf[..n]);
    }
}

/// Collects rows by size across pages.
#[derive(Default)]
pub struct SizeIndex {
    by_size: HashMap<u64, Vec<SizeRow>>,
    rows: usize,
}

impl SizeIndex {
    pub fn add(&mut self, rows: impl IntoIterator<Item = SizeRow>) {
        for row in rows {
            if row.size == 0 {
                continue;
            }
            self.rows += 1;
            self.by_size.entry(row.size).or_default().push(row);
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Groups of two or more rows of one size, the biggest files first.
    pub fn candidates(self) -> Vec<Vec<SizeRow>> {
        let mut groups: Vec<Vec<SizeRow>> = self.by_size.into_values().filter(|g| g.len() > 1).collect();
        groups.sort_by(|a, b| b[0].size.cmp(&a[0].size).then_with(|| a[0].path.cmp(&b[0].path)));
        groups
    }
}

/// Splits one same-size group into groups of identical content, hashing the rows that have no stored hash.
pub async fn confirm(candidates: Vec<SizeRow>, hasher: Hasher, cancel: &AtomicBool) -> Vec<Group> {
    let size = candidates.first().map_or(0, |r| r.size);
    let mut hashed: Vec<(String, String)> = Vec::new();
    let mut unhashed: Vec<String> = Vec::new();
    for row in candidates {
        match row.hash.filter(|h| !h.is_empty()) {
            Some(hash) => hashed.push((hash, row.path)),
            None => unhashed.push(row.path),
        }
    }
    for chunk in unhashed.chunks(HASH_PARALLEL) {
        if cancel.load(Ordering::Relaxed) {
            return Vec::new();
        }
        let jobs: Vec<_> = chunk
            .iter()
            .cloned()
            .map(|path| {
                let hasher = hasher.clone();
                tokio::task::spawn_blocking(move || hasher(&path).map(|hash| (hash, path)))
            })
            .collect();
        for job in jobs {
            if let Ok(Some(found)) = job.await {
                hashed.push(found);
            }
        }
    }
    let mut by_hash: HashMap<String, Vec<String>> = HashMap::new();
    for (hash, path) in hashed {
        by_hash.entry(hash).or_default().push(path);
    }
    let mut groups: Vec<Group> = by_hash
        .into_iter()
        .filter(|(_, paths)| paths.len() > 1)
        .map(|(hash, mut paths)| {
            paths.sort();
            paths.dedup();
            Group { hash, size, paths }
        })
        .filter(|g| g.paths.len() > 1)
        .collect();
    groups.sort_by(|a, b| a.paths[0].cmp(&b.paths[0]));
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(path: &str, size: u64, hash: Option<&str>) -> SizeRow {
        SizeRow { path: path.into(), size, hash: hash.map(str::to_string) }
    }

    fn write(dir: &std::path::Path, name: &str, bytes: &[u8]) -> String {
        let p = dir.join(name);
        std::fs::write(&p, bytes).unwrap();
        p.display().to_string()
    }

    #[test]
    fn only_sizes_shared_by_two_rows_are_candidates_and_empty_files_never_are() {
        let mut index = SizeIndex::default();
        index.add([row("/a", 10, None), row("/b", 10, None), row("/c", 20, None), row("/d", 0, None), row("/e", 0, None), row("/f", 30, None), row("/g", 30, None), row("/h", 30, None)]);
        assert_eq!(index.rows(), 6);
        let groups = index.candidates();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].len(), 3, "the biggest files first");
        assert_eq!(groups[1].len(), 2);
    }

    #[tokio::test]
    async fn files_with_equal_contents_form_a_group_and_equal_sizes_alone_do_not() {
        let dir = tempfile::tempdir().unwrap();
        let a = write(dir.path(), "a.jpg", b"same bytes");
        let b = write(dir.path(), "b.jpg", b"same bytes");
        let c = write(dir.path(), "c.jpg", b"diff bytes");
        let candidates = vec![row(&b, 10, None), row(&a, 10, None), row(&c, 10, None)];
        let groups = confirm(candidates, Arc::new(file_hash), &AtomicBool::new(false)).await;
        assert_eq!(groups.len(), 1);
        let mut expected = vec![a, b];
        expected.sort();
        assert_eq!(groups[0].paths, expected);
        assert_eq!(groups[0].size, 10);
        assert_eq!(groups[0].hash, blake3::hash(b"same bytes").to_hex().to_string());
    }

    #[tokio::test]
    async fn stored_hashes_are_trusted_and_mix_with_computed_ones() {
        let dir = tempfile::tempdir().unwrap();
        let real = write(dir.path(), "real.jpg", b"payload");
        let hash = blake3::hash(b"payload").to_hex().to_string();
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = calls.clone();
        let hasher: Hasher = Arc::new(move |p| {
            counter.fetch_add(1, Ordering::Relaxed);
            file_hash(p)
        });
        let candidates = vec![row("/gone/stored.jpg", 7, Some(&hash)), row(&real, 7, None), row("/gone/other.jpg", 7, Some("different"))];
        let groups = confirm(candidates, hasher, &AtomicBool::new(false)).await;
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].paths.len(), 2);
        assert_eq!(calls.load(Ordering::Relaxed), 1, "only the row without a hash is read");
    }

    #[tokio::test]
    async fn unreadable_files_are_left_out_without_failing() {
        let dir = tempfile::tempdir().unwrap();
        let a = write(dir.path(), "a.jpg", b"xx");
        let b = write(dir.path(), "b.jpg", b"xx");
        let candidates = vec![row(&a, 2, None), row(&b, 2, None), row("/missing/c.jpg", 2, None)];
        let groups = confirm(candidates, Arc::new(file_hash), &AtomicBool::new(false)).await;
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].paths.len(), 2);
    }

    #[tokio::test]
    async fn a_cancelled_confirmation_returns_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let a = write(dir.path(), "a.jpg", b"xx");
        let b = write(dir.path(), "b.jpg", b"xx");
        let groups = confirm(vec![row(&a, 2, None), row(&b, 2, None)], Arc::new(file_hash), &AtomicBool::new(true)).await;
        assert!(groups.is_empty());
    }

    #[test]
    fn the_file_hash_matches_blake3_of_the_whole_file() {
        let dir = tempfile::tempdir().unwrap();
        let big: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
        let p = write(dir.path(), "big.bin", &big);
        assert_eq!(file_hash(&p), Some(blake3::hash(&big).to_hex().to_string()));
        assert_eq!(file_hash("/definitely/not/here"), None);
    }
}
