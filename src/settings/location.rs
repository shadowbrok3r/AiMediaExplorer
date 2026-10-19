//! Where the library lives: the config-dir pointer, the default folder, older installs next to the program and folder checks.

use std::path::{Path, PathBuf};

/// File inside the config directory that records the chosen library folder.
pub const POINTER_FILE: &str = "db_location.txt";
/// Folder name of an older install's library, and the file that points at another one.
const LEGACY_DIR: &str = "db";
const LEGACY_STORE: &str = "ai_search";
const LEGACY_POINTER: &str = "path.txt";
/// Entries a SurrealKV store folder holds.
const STORE_MARKERS: [&str; 4] = ["manifest", "wal", "sstables", "vlog"];

/// `<config dir>/smart-media/db_location.txt`.
pub fn pointer_path() -> Option<PathBuf> {
    directories::BaseDirs::new().map(|dirs| dirs.config_dir().join(crate::ui::titlebar::APP_ID).join(POINTER_FILE))
}

/// The folder written in the pointer file at `path`, when there is one.
pub fn read_pointer(path: &Path) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    let line = text.lines().map(str::trim).find(|l| !l.is_empty())?;
    Some(line.to_string())
}

/// Writes `dir` into the pointer file at `path`.
pub fn write_pointer(path: &Path, dir: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("txt.tmp");
    std::fs::write(&tmp, dir.to_string_lossy().as_bytes())?;
    std::fs::rename(&tmp, path)
}

/// The library folder from the config-dir pointer; tests never read the real one.
pub fn configured() -> Option<String> {
    #[cfg(test)]
    {
        None
    }
    #[cfg(not(test))]
    {
        read_pointer(&pointer_path()?)
    }
}

/// Records `dir` as the library folder in the config directory; tests never write the real one.
pub fn remember(dir: &Path) {
    if cfg!(test) {
        return;
    }
    if let Some(path) = pointer_path() {
        if let Err(e) = write_pointer(&path, dir) {
            log::warn!("[location] cannot write {}: {e}", path.display());
        }
    }
}

/// The folder a new library gets by default: `<data dir>/smart-media/db`.
pub fn default_location() -> PathBuf {
    let base = directories::BaseDirs::new().map(|d| d.data_dir().to_path_buf()).unwrap_or_else(|| PathBuf::from("."));
    base.join(crate::ui::titlebar::APP_ID).join(LEGACY_DIR)
}

/// True when `dir` holds a SurrealKV store.
pub fn looks_like_store(dir: &Path) -> bool {
    dir.is_dir() && STORE_MARKERS.iter().any(|m| dir.join(m).exists())
}

/// How an older library was found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Via {
    /// A `db/path.txt` file named it.
    PointerFile,
    /// The `db/ai_search` folder next to the program or in the working directory.
    DefaultFolder,
}

/// An older library folder that can be offered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub path: PathBuf,
    pub via: Via,
    /// Where the pointer or the folder was looked up.
    pub origin: PathBuf,
}

/// Older libraries next to the executable and in the working directory, each folder once.
pub fn detect_legacy(cwd: Option<&Path>, exe_dir: Option<&Path>) -> Vec<Candidate> {
    let mut found: Vec<Candidate> = Vec::new();
    let mut push = |c: Candidate| {
        if !found.iter().any(|f| f.path == c.path) {
            found.push(c);
        }
    };
    for root in [cwd, exe_dir].into_iter().flatten() {
        let dir = root.join(LEGACY_DIR);
        if let Some(target) = read_pointer(&dir.join(LEGACY_POINTER)) {
            let path = if Path::new(&target).is_absolute() { PathBuf::from(&target) } else { root.join(&target) };
            if looks_like_store(&path) {
                push(Candidate { path, via: Via::PointerFile, origin: dir.join(LEGACY_POINTER) });
            }
        }
        let store = dir.join(LEGACY_STORE);
        if looks_like_store(&store) {
            push(Candidate { path: store, via: Via::DefaultFolder, origin: dir.clone() });
        }
    }
    found
}

/// Older libraries around the running program.
pub fn detect_here() -> Vec<Candidate> {
    let cwd = std::env::current_dir().ok();
    let exe_dir = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf));
    detect_legacy(cwd.as_deref(), exe_dir.as_deref())
}

/// What choosing a folder as the library would do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Check {
    /// The folder holds a library that is opened as it is.
    Existing,
    /// The folder is missing or empty and gets a new library.
    New,
    /// The folder cannot be used; the text says why.
    Rejected(String),
}

impl Check {
    pub fn is_usable(&self) -> bool {
        !matches!(self, Self::Rejected(_))
    }
}

/// Decides what `dir` can be: an existing library, a place for a new one, or neither. Nothing is created or changed.
pub fn check_location(dir: &Path) -> Check {
    if dir.as_os_str().is_empty() {
        return Check::Rejected("Choose a folder".to_string());
    }
    if !dir.exists() {
        let parent = nearest_existing(dir);
        return if parent.is_dir() { Check::New } else { Check::Rejected(format!("{} is a file, not a folder", parent.display())) };
    }
    if !dir.is_dir() {
        return Check::Rejected(format!("{} is a file, not a folder", dir.display()));
    }
    if looks_like_store(dir) {
        return Check::Existing;
    }
    match std::fs::read_dir(dir) {
        Ok(mut entries) => match entries.next() {
            None => Check::New,
            Some(_) => Check::Rejected("The folder is not empty and holds no library. Choose an empty folder, or a library folder to open it".to_string()),
        },
        Err(e) => Check::Rejected(format!("The folder cannot be read: {e}")),
    }
}

fn nearest_existing(dir: &Path) -> PathBuf {
    dir.ancestors().skip(1).find(|a| !a.as_os_str().is_empty() && a.exists()).map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store_in(dir: &Path) {
        std::fs::create_dir_all(dir.join("manifest")).unwrap();
        std::fs::write(dir.join("LOCK"), "").unwrap();
    }

    #[test]
    fn the_default_folder_is_inside_the_data_dir() {
        let p = default_location();
        assert!(p.ends_with("smart-media/db"), "{}", p.display());
    }

    #[test]
    fn a_pointer_file_round_trips_and_ignores_blank_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cfg").join(POINTER_FILE);
        assert_eq!(read_pointer(&path), None);
        write_pointer(&path, Path::new("/data/lib")).unwrap();
        assert_eq!(read_pointer(&path).as_deref(), Some("/data/lib"));
        std::fs::write(&path, "\n  \n /other \n").unwrap();
        assert_eq!(read_pointer(&path).as_deref(), Some("/other"));
        std::fs::write(&path, "  \n").unwrap();
        assert_eq!(read_pointer(&path), None);
    }

    #[test]
    fn tests_never_read_the_real_config_pointer() {
        assert_eq!(configured(), None);
    }

    #[test]
    fn a_legacy_default_folder_and_pointer_are_found_in_the_working_directory() {
        let cwd = tempfile::tempdir().unwrap();
        store_in(&cwd.path().join("db").join("ai_search"));
        let found = detect_legacy(Some(cwd.path()), None);
        assert_eq!(found.len(), 1);
        assert_eq!((found[0].via, found[0].path.clone()), (Via::DefaultFolder, cwd.path().join("db").join("ai_search")));

        let elsewhere = tempfile::tempdir().unwrap();
        store_in(&elsewhere.path().join("movedlib"));
        std::fs::write(cwd.path().join("db").join("path.txt"), elsewhere.path().join("movedlib").to_string_lossy().as_bytes()).unwrap();
        let found = detect_legacy(Some(cwd.path()), None);
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].via, Via::PointerFile);
        assert_eq!(found[0].path, elsewhere.path().join("movedlib"));
    }

    #[test]
    fn a_relative_pointer_resolves_against_the_folder_it_was_found_in() {
        let cwd = tempfile::tempdir().unwrap();
        store_in(&cwd.path().join("lib2"));
        std::fs::create_dir_all(cwd.path().join("db")).unwrap();
        std::fs::write(cwd.path().join("db").join("path.txt"), "lib2").unwrap();
        let found = detect_legacy(Some(cwd.path()), None);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].path, cwd.path().join("lib2"));
    }

    #[test]
    fn the_program_folder_is_searched_too_and_duplicates_collapse() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        store_in(&b.path().join("db").join("ai_search"));
        assert_eq!(detect_legacy(Some(a.path()), Some(b.path())).len(), 1);
        assert_eq!(detect_legacy(Some(b.path()), Some(b.path())).len(), 1, "the same folder is offered once");
        assert!(detect_legacy(Some(a.path()), None).is_empty());
        assert!(detect_legacy(None, None).is_empty());
    }

    #[test]
    fn a_pointer_to_a_missing_or_foreign_folder_offers_nothing() {
        let cwd = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(cwd.path().join("db")).unwrap();
        std::fs::write(cwd.path().join("db").join("path.txt"), "/does/not/exist").unwrap();
        assert!(detect_legacy(Some(cwd.path()), None).is_empty());
        let plain = tempfile::tempdir().unwrap();
        std::fs::write(plain.path().join("a.txt"), "x").unwrap();
        std::fs::write(cwd.path().join("db").join("path.txt"), plain.path().to_string_lossy().as_bytes()).unwrap();
        assert!(detect_legacy(Some(cwd.path()), None).is_empty(), "a folder without a store is not a library");
    }

    #[test]
    fn a_missing_folder_with_a_reachable_parent_gets_a_new_library() {
        let root = tempfile::tempdir().unwrap();
        assert_eq!(check_location(&root.path().join("a").join("b").join("db")), Check::New);
        assert!(!root.path().join("a").exists(), "checking creates nothing");
    }

    #[test]
    fn an_empty_folder_gets_a_new_library_and_a_store_is_opened_as_it_is() {
        let root = tempfile::tempdir().unwrap();
        assert_eq!(check_location(root.path()), Check::New);
        store_in(root.path());
        assert_eq!(check_location(root.path()), Check::Existing);
    }

    #[test]
    fn a_folder_with_other_files_is_never_taken_over() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("photo.jpg"), "x").unwrap();
        let check = check_location(root.path());
        assert!(matches!(check, Check::Rejected(ref why) if why.contains("not empty")), "{check:?}");
        assert!(!check.is_usable());
        assert!(root.path().join("photo.jpg").exists());
    }

    #[test]
    fn files_and_empty_paths_are_rejected() {
        let root = tempfile::tempdir().unwrap();
        let file = root.path().join("f");
        std::fs::write(&file, "x").unwrap();
        assert!(matches!(check_location(&file), Check::Rejected(_)));
        assert!(matches!(check_location(&file.join("child")), Check::Rejected(_)));
        assert!(matches!(check_location(Path::new("")), Check::Rejected(_)));
    }
}
