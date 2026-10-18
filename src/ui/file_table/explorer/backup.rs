use std::path::{Path, PathBuf};

use egui_toast::ToastKind;

/// Outcome of a copy or move batch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BackupReport {
    pub ok: usize,
    pub failed: usize,
    pub skipped: usize,
}

impl BackupReport {
    fn summary(&self, verb: &str, dest: &Path) -> String {
        let mut text = format!("{verb} {} file(s) to {}", self.ok, dest.display());
        if self.failed > 0 {
            text.push_str(&format!(", {} failed", self.failed));
        }
        if self.skipped > 0 {
            text.push_str(&format!(", {} skipped", self.skipped));
        }
        text
    }
}

/// A free name in the same folder: `name (1).ext`, `name (2).ext`, ...
pub fn unique_dest_path(dest: PathBuf) -> PathBuf {
    if !dest.exists() { return dest; }
    let stem = dest.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let ext_opt = dest.extension().and_then(|s| s.to_str());
    let parent = dest.parent().map(|p| p.to_path_buf()).unwrap_or_default();
    let mut idx = 1u32;
    loop {
        let name = match ext_opt {
            Some(ext) => format!("{} ({}).{}", stem, idx, ext),
            None => format!("{} ({})", stem, idx),
        };
        let cand = parent.join(name);
        if !cand.exists() { return cand; }
        idx += 1;
        if idx > 10_000 { return dest; }
    }
}

fn destination_for(src: &Path, dest_dir: &Path) -> Option<PathBuf> {
    src.file_name().map(|name| unique_dest_path(dest_dir.join(name)))
}

/// Copies each existing file into `dest_dir`, renaming on collisions.
pub async fn copy_files(paths: Vec<PathBuf>, dest_dir: PathBuf) -> BackupReport {
    let mut report = BackupReport::default();
    for src in paths {
        let Some(dest) = src.is_file().then(|| destination_for(&src, &dest_dir)).flatten() else {
            report.skipped += 1;
            continue;
        };
        let (s, d) = (src.clone(), dest.clone());
        match tokio::task::spawn_blocking(move || std::fs::copy(&s, &d)).await {
            Ok(Ok(_)) => report.ok += 1,
            Ok(Err(e)) => {
                report.failed += 1;
                log::error!("Backup copy failed {} -> {}: {e:?}", src.display(), dest.display());
            }
            Err(e) => {
                report.failed += 1;
                log::error!("Backup copy join error {} -> {}: {e:?}", src.display(), dest.display());
            }
        }
    }
    report
}

/// Moves each existing file into `dest_dir`; files already in `dest_dir` are skipped.
pub async fn move_files(paths: Vec<PathBuf>, dest_dir: PathBuf) -> BackupReport {
    let mut report = BackupReport::default();
    for src in paths {
        let same_dir = src.parent().is_some_and(|p| p == dest_dir);
        let Some(dest) = (src.is_file() && !same_dir).then(|| destination_for(&src, &dest_dir)).flatten() else {
            report.skipped += 1;
            continue;
        };
        let (s, d) = (src.clone(), dest.clone());
        let renamed = matches!(tokio::task::spawn_blocking(move || std::fs::rename(&s, &d)).await, Ok(Ok(())));
        if renamed {
            report.ok += 1;
            continue;
        }
        let (s, d) = (src.clone(), dest.clone());
        let copied = matches!(tokio::task::spawn_blocking(move || std::fs::copy(&s, &d)).await, Ok(Ok(_)));
        if !copied {
            report.failed += 1;
            log::error!("Backup move failed {} -> {}", src.display(), dest.display());
            continue;
        }
        let s = src.clone();
        match tokio::task::spawn_blocking(move || std::fs::remove_file(&s)).await {
            Ok(Ok(())) => report.ok += 1,
            _ => {
                report.failed += 1;
                log::error!("Backup move copied but could not remove {}", src.display());
            }
        }
    }
    report
}

impl crate::ui::file_table::FileExplorer {
    /// Copies the given files into `dir` in the background and toasts the counts.
    pub fn backup_copy_files(paths: Vec<PathBuf>, dir: PathBuf) {
        if paths.is_empty() { return; }
        tokio::spawn(async move {
            let report = copy_files(paths, dir.clone()).await;
            log::info!("{}", report.summary("Copied", &dir));
            let kind = if report.failed > 0 { ToastKind::Warning } else { ToastKind::Success };
            crate::ui::dialogs::toast(kind, report.summary("Copied", &dir));
        });
    }

    /// Moves the given files into `dir` in the background and toasts the counts.
    pub fn backup_move_files(paths: Vec<PathBuf>, dir: PathBuf) {
        if paths.is_empty() { return; }
        tokio::spawn(async move {
            let report = move_files(paths, dir.clone()).await;
            log::info!("{}", report.summary("Moved", &dir));
            let kind = if report.failed > 0 { ToastKind::Warning } else { ToastKind::Success };
            crate::ui::dialogs::toast(kind, report.summary("Moved", &dir));
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, name.as_bytes()).unwrap();
        p
    }

    #[tokio::test]
    async fn copy_reports_counts_and_renames_collisions() {
        let (src, dst) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let a = write(src.path(), "a.jpg");
        let b = write(src.path(), "b.jpg");
        write(dst.path(), "a.jpg");
        let missing = src.path().join("gone.jpg");
        let report = copy_files(vec![a.clone(), b, missing], dst.path().to_path_buf()).await;
        assert_eq!(report, BackupReport { ok: 2, failed: 0, skipped: 1 });
        assert!(dst.path().join("a (1).jpg").exists());
        assert!(dst.path().join("b.jpg").exists());
        assert!(a.exists(), "copy leaves the source");
    }

    #[tokio::test]
    async fn move_removes_sources_and_skips_files_already_in_place() {
        let (src, dst) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let a = write(src.path(), "a.png");
        let report = move_files(vec![a.clone()], dst.path().to_path_buf()).await;
        assert_eq!(report, BackupReport { ok: 1, failed: 0, skipped: 0 });
        assert!(!a.exists());
        assert!(dst.path().join("a.png").exists());

        let stay = write(src.path(), "stay.png");
        let report = move_files(vec![stay.clone()], src.path().to_path_buf()).await;
        assert_eq!(report, BackupReport { ok: 0, failed: 0, skipped: 1 });
        assert!(stay.exists());
    }

    #[test]
    fn summary_mentions_failures_and_skips_only_when_present() {
        let ok = BackupReport { ok: 3, failed: 0, skipped: 0 }.summary("Copied", Path::new("/d"));
        assert_eq!(ok, "Copied 3 file(s) to /d");
        let mixed = BackupReport { ok: 1, failed: 2, skipped: 3 }.summary("Moved", Path::new("/d"));
        assert_eq!(mixed, "Moved 1 file(s) to /d, 2 failed, 3 skipped");
    }
}
