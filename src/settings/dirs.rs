//! Start-directory resolution that never panics when the OS reports no user directories.

use std::path::PathBuf;

/// Where a resolved start directory came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StartDirSource {
    Pictures,
    Home,
    CurrentDir,
    Dot,
}

/// Picks the first available of pictures, home and the current directory, else ".".
pub fn resolve_start_dir(
    pictures: Option<PathBuf>,
    home: Option<PathBuf>,
    cwd: Option<PathBuf>,
) -> (PathBuf, StartDirSource) {
    if let Some(p) = pictures {
        return (p, StartDirSource::Pictures);
    }
    if let Some(h) = home {
        return (h, StartDirSource::Home);
    }
    if let Some(c) = cwd {
        return (c, StartDirSource::CurrentDir);
    }
    (PathBuf::from("."), StartDirSource::Dot)
}

/// The Pictures directory, falling back to the home directory, then the current directory.
pub fn start_dir() -> PathBuf {
    let user = directories::UserDirs::new();
    let pictures = user.as_ref().and_then(|u| u.picture_dir().map(|p| p.to_path_buf()));
    let home = user.as_ref().map(|u| u.home_dir().to_path_buf());
    let (dir, source) = resolve_start_dir(pictures, home, std::env::current_dir().ok());
    if source != StartDirSource::Pictures {
        log::warn!("[dirs] no Pictures directory; starting in {} ({source:?})", dir.display());
    }
    dir
}

/// The home directory, falling back to the current directory.
pub fn home_dir() -> PathBuf {
    let home = directories::UserDirs::new().map(|u| u.home_dir().to_path_buf());
    let (dir, source) = resolve_start_dir(None, home, std::env::current_dir().ok());
    if source != StartDirSource::Home {
        log::warn!("[dirs] no home directory; using {} ({source:?})", dir.display());
    }
    dir
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> Option<PathBuf> {
        Some(PathBuf::from(s))
    }

    #[test]
    fn pictures_win_when_present() {
        let (dir, source) = resolve_start_dir(p("/pics"), p("/home/u"), p("/cwd"));
        assert_eq!((dir, source), (PathBuf::from("/pics"), StartDirSource::Pictures));
    }

    #[test]
    fn home_is_next_then_current_dir_then_dot() {
        assert_eq!(resolve_start_dir(None, p("/home/u"), p("/cwd")).1, StartDirSource::Home);
        assert_eq!(resolve_start_dir(None, None, p("/cwd")).1, StartDirSource::CurrentDir);
        let (dir, source) = resolve_start_dir(None, None, None);
        assert_eq!((dir, source), (PathBuf::from("."), StartDirSource::Dot));
    }
}
