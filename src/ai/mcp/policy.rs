//! Which files tools may touch: indexed library roots, watched folders and the Pictures and Videos folders.

use std::path::{Component, Path, PathBuf};
use std::sync::RwLock;
use std::time::{Duration, Instant};

use super::ctx::DbSource;

/// How long the library roots stay cached.
const ROOTS_TTL: Duration = Duration::from_secs(60);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PolicyError {
    Empty,
    NotAbsolute,
    NulByte,
    Virtual,
    ParentComponent,
    Outside,
    Missing,
    NotAFile,
}

impl std::fmt::Display for PolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Empty => "the path is empty",
            Self::NotAbsolute => "the path must be absolute",
            Self::NulByte => "the path contains a NUL byte",
            Self::Virtual => "archive and other virtual paths are not accepted",
            Self::ParentComponent => "the path contains '..'",
            Self::Outside => "the path is outside the library roots, watched folders and the Pictures and Videos folders",
            Self::Missing => "the file does not exist",
            Self::NotAFile => "the path is not a regular file",
        })
    }
}

impl std::error::Error for PolicyError {}

fn comparable(p: &Path) -> PathBuf {
    if cfg!(windows) { PathBuf::from(p.to_string_lossy().to_lowercase()) } else { p.to_path_buf() }
}

fn is_under(root: &Path, path: &Path) -> bool {
    comparable(path).starts_with(comparable(root))
}

/// The directories among `dirs` that no other of them contains, in sorted order.
pub fn minimal_roots(dirs: impl IntoIterator<Item = String>) -> Vec<PathBuf> {
    let mut sorted: Vec<PathBuf> = dirs.into_iter().filter(|d| !d.is_empty()).map(PathBuf::from).filter(|p| p.is_absolute()).collect();
    sorted.sort();
    sorted.dedup();
    let mut roots: Vec<PathBuf> = Vec::new();
    for dir in sorted {
        if !roots.iter().any(|r| is_under(r, &dir)) {
            roots.push(dir);
        }
    }
    roots
}

/// Lexical checks shared by both modes: an absolute, plain path without '..'.
fn plain(path: &str) -> Result<PathBuf, PolicyError> {
    if path.trim().is_empty() {
        return Err(PolicyError::Empty);
    }
    if path.contains('\0') {
        return Err(PolicyError::NulByte);
    }
    if path.contains("://") {
        return Err(PolicyError::Virtual);
    }
    let p = PathBuf::from(path);
    if !p.is_absolute() {
        return Err(PolicyError::NotAbsolute);
    }
    if p.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(PolicyError::ParentComponent);
    }
    Ok(p)
}

/// The folders tools may use and the checks that keep a path inside them.
pub struct PathPolicy {
    fixed: RwLock<Vec<PathBuf>>,
    library: RwLock<(Vec<PathBuf>, Option<Instant>)>,
}

impl PathPolicy {
    /// A policy over fixed roots only.
    pub fn with_roots(roots: Vec<PathBuf>) -> Self {
        Self { fixed: RwLock::new(roots), library: RwLock::new((Vec::new(), None)) }
    }

    /// Watched folders from the settings plus the Pictures and Videos folders of the user.
    pub fn production(watched: &[String]) -> Self {
        let mut roots: Vec<PathBuf> = watched.iter().map(PathBuf::from).collect();
        if let Some(dirs) = directories::UserDirs::new() {
            roots.extend(dirs.picture_dir().map(Path::to_path_buf));
            roots.extend(dirs.video_dir().map(Path::to_path_buf));
        }
        Self::with_roots(roots)
    }

    pub fn set_fixed_roots(&self, roots: Vec<PathBuf>) {
        *self.fixed.write().unwrap_or_else(|e| e.into_inner()) = roots;
        self.library.write().unwrap_or_else(|e| e.into_inner()).1 = None;
    }

    /// Replaces the cached library roots.
    pub fn set_library_roots(&self, roots: Vec<PathBuf>) {
        *self.library.write().unwrap_or_else(|e| e.into_inner()) = (roots, Some(Instant::now()));
    }

    /// Every root a path may be under.
    pub fn roots(&self) -> Vec<PathBuf> {
        let mut all = self.fixed.read().unwrap_or_else(|e| e.into_inner()).clone();
        all.extend(self.library.read().unwrap_or_else(|e| e.into_inner()).0.iter().cloned());
        all
    }

    fn stale(&self) -> bool {
        self.library.read().unwrap_or_else(|e| e.into_inner()).1.is_none_or(|at| at.elapsed() > ROOTS_TTL)
    }

    /// Reloads the library roots from the folders of the indexed rows when the cache is old.
    pub async fn refresh(&self, db: &DbSource) {
        if !self.stale() {
            return;
        }
        let Ok(db) = db.get() else { return };
        let dirs: Result<Vec<String>, _> = async { db.query("SELECT VALUE parent_dir FROM thumbnails GROUP BY parent_dir").await?.take(0) }.await;
        match dirs {
            Ok(dirs) => self.set_library_roots(minimal_roots(dirs)),
            Err(e) => log::warn!("[mcp] library roots not loaded: {e}"),
        }
    }

    /// Checks `path` as written, without touching the file system; for rows that name library files.
    pub fn check_lexical(&self, path: &str) -> Result<PathBuf, PolicyError> {
        let p = plain(path)?;
        if self.roots().iter().any(|r| is_under(r, &p)) { Ok(p) } else { Err(PolicyError::Outside) }
    }

    /// Resolves symlinks and checks the real file against the real roots; returns the real path.
    pub fn check_file(&self, path: &str) -> Result<PathBuf, PolicyError> {
        let p = plain(path)?;
        let real = std::fs::canonicalize(&p).map_err(|_| PolicyError::Missing)?;
        let allowed = self.roots().iter().filter_map(|r| std::fs::canonicalize(r).ok()).any(|r| is_under(&r, &real));
        if !allowed {
            return Err(PolicyError::Outside);
        }
        if !real.is_file() {
            return Err(PolicyError::NotAFile);
        }
        Ok(real)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree() -> (tempfile::TempDir, PathBuf, PathBuf) {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("pictures");
        let outside = tmp.path().join("secret");
        std::fs::create_dir_all(root.join("sub")).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(root.join("a.jpg"), b"a").unwrap();
        std::fs::write(root.join("sub/b.jpg"), b"b").unwrap();
        std::fs::write(outside.join("key.txt"), b"k").unwrap();
        (tmp, root, outside)
    }

    fn s(p: &Path) -> String {
        p.to_string_lossy().to_string()
    }

    #[test]
    fn files_under_a_root_pass_and_others_do_not() {
        let (tmp, root, outside) = tree();
        let policy = PathPolicy::with_roots(vec![root.clone()]);
        assert!(policy.check_file(&s(&root.join("a.jpg"))).is_ok());
        assert!(policy.check_file(&s(&root.join("sub/b.jpg"))).is_ok());
        assert_eq!(policy.check_file(&s(&outside.join("key.txt"))), Err(PolicyError::Outside));
        assert_eq!(policy.check_lexical(&s(&outside.join("key.txt"))), Err(PolicyError::Outside));
        let sibling = tmp.path().join("pictures2");
        std::fs::create_dir_all(&sibling).unwrap();
        std::fs::write(sibling.join("c.jpg"), b"c").unwrap();
        assert_eq!(policy.check_file(&s(&sibling.join("c.jpg"))), Err(PolicyError::Outside), "a shared name prefix is not containment");
        assert_eq!(policy.check_lexical(&s(&sibling.join("c.jpg"))), Err(PolicyError::Outside));
    }

    #[test]
    fn parent_components_are_refused_before_anything_else() {
        let (_tmp, root, outside) = tree();
        let policy = PathPolicy::with_roots(vec![root.clone()]);
        let sneaky = format!("{}/../secret/key.txt", root.display());
        assert_eq!(policy.check_file(&sneaky), Err(PolicyError::ParentComponent));
        assert_eq!(policy.check_lexical(&sneaky), Err(PolicyError::ParentComponent));
        let _ = outside;
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_that_leave_the_roots_are_refused() {
        let (_tmp, root, outside) = tree();
        std::os::unix::fs::symlink(outside.join("key.txt"), root.join("link.jpg")).unwrap();
        std::os::unix::fs::symlink(&outside, root.join("dir_link")).unwrap();
        let policy = PathPolicy::with_roots(vec![root.clone()]);
        assert_eq!(policy.check_file(&s(&root.join("link.jpg"))), Err(PolicyError::Outside));
        assert_eq!(policy.check_file(&s(&root.join("dir_link/key.txt"))), Err(PolicyError::Outside));
        assert!(policy.check_lexical(&s(&root.join("link.jpg"))).is_ok(), "lexical mode never follows links");
    }

    #[cfg(unix)]
    #[test]
    fn a_symlinked_root_still_admits_its_real_files() {
        let (tmp, root, _outside) = tree();
        let alias = tmp.path().join("alias");
        std::os::unix::fs::symlink(&root, &alias).unwrap();
        let policy = PathPolicy::with_roots(vec![alias.clone()]);
        let real = policy.check_file(&s(&alias.join("a.jpg"))).unwrap();
        assert_eq!(real, std::fs::canonicalize(root.join("a.jpg")).unwrap());
    }

    #[test]
    fn malformed_paths_are_named() {
        let (_tmp, root, _o) = tree();
        let policy = PathPolicy::with_roots(vec![root.clone()]);
        assert_eq!(policy.check_file(""), Err(PolicyError::Empty));
        assert_eq!(policy.check_file("   "), Err(PolicyError::Empty));
        assert_eq!(policy.check_file("a.jpg"), Err(PolicyError::NotAbsolute));
        assert_eq!(policy.check_file("./a.jpg"), Err(PolicyError::NotAbsolute));
        assert_eq!(policy.check_file(&format!("{}/a\0.jpg", root.display())), Err(PolicyError::NulByte));
        assert_eq!(policy.check_file("zip:///data/a.zip!/x.jpg"), Err(PolicyError::Virtual));
        assert_eq!(policy.check_file(&s(&root.join("missing.jpg"))), Err(PolicyError::Missing));
        assert_eq!(policy.check_file(&s(&root.join("sub"))), Err(PolicyError::NotAFile));
        assert!(policy.check_lexical(&s(&root.join("missing.jpg"))).is_ok());
    }

    #[test]
    fn a_policy_without_roots_admits_nothing() {
        let (_tmp, root, _o) = tree();
        let policy = PathPolicy::with_roots(Vec::new());
        assert_eq!(policy.check_file(&s(&root.join("a.jpg"))), Err(PolicyError::Outside));
    }

    #[test]
    fn library_roots_are_the_outermost_indexed_folders() {
        let dirs = ["/p/a/b", "/p/a", "/p/c", "/p/a/b/c", "/q", "/p/ab", "", "rel/x", "/p/a"].map(String::from);
        let roots = minimal_roots(dirs);
        assert_eq!(roots, vec![PathBuf::from("/p/a"), PathBuf::from("/p/ab"), PathBuf::from("/p/c"), PathBuf::from("/q")]);
    }

    #[test]
    fn cached_library_roots_extend_the_fixed_ones_and_reset_with_them() {
        let (_tmp, root, outside) = tree();
        let policy = PathPolicy::with_roots(vec![root.clone()]);
        assert!(policy.check_lexical(&s(&outside.join("key.txt"))).is_err());
        policy.set_library_roots(vec![outside.clone()]);
        assert!(policy.check_lexical(&s(&outside.join("key.txt"))).is_ok());
        assert_eq!(policy.roots().len(), 2);
        assert!(!policy.stale());
        policy.set_fixed_roots(vec![root]);
        assert!(policy.stale());
    }
}
