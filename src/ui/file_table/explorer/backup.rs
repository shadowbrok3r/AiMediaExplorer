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

/// Called after each file with the files done and the total.
pub type Progress<'a> = &'a (dyn Fn(u64, u64) + Sync);
/// Asked before each file whether to stop.
pub type Stop<'a> = &'a (dyn Fn() -> bool + Sync);

/// Copies each existing file into `dest_dir`, renaming on collisions.
pub async fn copy_files(paths: Vec<PathBuf>, dest_dir: PathBuf) -> BackupReport {
    copy_files_with(paths, dest_dir, &|_, _| {}, &|| false).await
}

/// [`copy_files`] reporting progress per file and stopping early when asked; files left over count as skipped.
pub async fn copy_files_with(paths: Vec<PathBuf>, dest_dir: PathBuf, progress: Progress<'_>, stop: Stop<'_>) -> BackupReport {
    let mut report = BackupReport::default();
    let total = paths.len() as u64;
    for (i, src) in paths.into_iter().enumerate() {
        progress(i as u64, total);
        if stop() {
            report.skipped += total as usize - i;
            break;
        }
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
    progress(total, total);
    report
}

/// Moves each existing file into `dest_dir`; files already in `dest_dir` are skipped.
pub async fn move_files(paths: Vec<PathBuf>, dest_dir: PathBuf) -> BackupReport {
    move_files_with(paths, dest_dir, &|_, _| {}, &|| false).await
}

/// [`move_files`] reporting progress per file and stopping early when asked; files left over count as skipped.
pub async fn move_files_with(paths: Vec<PathBuf>, dest_dir: PathBuf, progress: Progress<'_>, stop: Stop<'_>) -> BackupReport {
    let mut report = BackupReport::default();
    let total = paths.len() as u64;
    for (i, src) in paths.into_iter().enumerate() {
        progress(i as u64, total);
        if stop() {
            report.skipped += total as usize - i;
            break;
        }
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
    progress(total, total);
    report
}

/// Runs a copy or move as a task that shows per-file progress and can be cancelled, then toasts the counts.
fn run_tracked(moving: bool, paths: Vec<PathBuf>, dir: PathBuf) {
    if paths.is_empty() {
        return;
    }
    let (verb, past) = if moving { ("Move", "Moved") } else { ("Copy", "Copied") };
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let spec = crate::tasks::TaskSpec::new(crate::tasks::TaskKind::Backup, format!("{verb} {} files to {}", paths.len(), dir.display()))
        .total(paths.len() as u64)
        .cancel_flag(stop.clone())
        .user_initiated()
        .toast_after(std::time::Duration::from_secs(24 * 3600));
    tokio::spawn(async move {
        let shown = dir.clone();
        let outcome = crate::tasks::track(spec, |reporter| async move {
            let progress = |done: u64, total: u64| reporter.progress(done, total);
            let stopped = || stop.load(std::sync::atomic::Ordering::Relaxed);
            let report = if moving { move_files_with(paths, dir, &progress, &stopped).await } else { copy_files_with(paths, dir, &progress, &stopped).await };
            if report.ok == 0 && report.failed > 0 { Err(anyhow::anyhow!("{} files could not be {}", report.failed, past.to_lowercase())) } else { Ok(report) }
        })
        .await;
        let Ok(report) = outcome else { return };
        log::info!("{}", report.summary(past, &shown));
        let kind = if report.failed > 0 { ToastKind::Warning } else { ToastKind::Success };
        crate::ui::dialogs::toast(kind, report.summary(past, &shown));
    });
}

impl crate::ui::file_table::FileExplorer {
    /// Copies the given files into `dir` in the background as a tracked task and toasts the counts.
    pub fn backup_copy_files(paths: Vec<PathBuf>, dir: PathBuf) {
        run_tracked(false, paths, dir);
    }

    /// Moves the given files into `dir` in the background as a tracked task and toasts the counts.
    pub fn backup_move_files(paths: Vec<PathBuf>, dir: PathBuf) {
        run_tracked(true, paths, dir);
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

    #[tokio::test]
    async fn copy_reports_each_file_and_ends_at_the_total() {
        let (src, dst) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let files: Vec<PathBuf> = (0..4).map(|i| write(src.path(), &format!("f{i}.jpg"))).collect();
        let seen = std::sync::Mutex::new(Vec::new());
        let report = copy_files_with(files, dst.path().to_path_buf(), &|d, t| seen.lock().unwrap().push((d, t)), &|| false).await;
        assert_eq!(report, BackupReport { ok: 4, failed: 0, skipped: 0 });
        let seen = seen.into_inner().unwrap();
        assert_eq!(seen.first(), Some(&(0, 4)));
        assert_eq!(seen.last(), Some(&(4, 4)));
        assert!(seen.windows(2).all(|w| w[0].0 <= w[1].0));
    }

    #[tokio::test]
    async fn a_stop_request_skips_the_files_left() {
        let (src, dst) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let files: Vec<PathBuf> = (0..5).map(|i| write(src.path(), &format!("f{i}.jpg"))).collect();
        let done = std::sync::atomic::AtomicUsize::new(0);
        let report = copy_files_with(files, dst.path().to_path_buf(), &|d, _| done.store(d as usize, std::sync::atomic::Ordering::Relaxed), &|| done.load(std::sync::atomic::Ordering::Relaxed) >= 2).await;
        assert_eq!(report.ok, 2);
        assert_eq!(report.skipped, 3);
        assert_eq!(std::fs::read_dir(dst.path()).unwrap().count(), 2);
    }

    #[tokio::test]
    async fn a_tracked_move_shows_as_a_backup_task_that_completes() {
        let (src, dst) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let a = write(src.path(), "tracked-move-a.jpg");
        let b = write(src.path(), "tracked-move-b.jpg");
        crate::ui::file_table::FileExplorer::backup_move_files(vec![a.clone(), b.clone()], dst.path().to_path_buf());
        let mut state = None;
        for _ in 0..200 {
            state = crate::tasks::with(|r| r.tasks().iter().find(|t| t.kind == crate::tasks::TaskKind::Backup && t.title.contains(&dst.path().display().to_string())).map(|t| (t.state, t.done, t.total)));
            if matches!(state, Some((crate::tasks::TaskState::Completed, _, _))) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert_eq!(state, Some((crate::tasks::TaskState::Completed, 2, 2)));
        assert!(!a.exists() && !b.exists());
        assert!(dst.path().join("tracked-move-a.jpg").exists());
    }

    #[tokio::test]
    async fn a_tracked_copy_of_nothing_starts_no_task() {
        let before = crate::tasks::with(|r| r.tasks().len());
        crate::ui::file_table::FileExplorer::backup_copy_files(Vec::new(), PathBuf::from("/nowhere"));
        assert_eq!(crate::tasks::with(|r| r.tasks().len()), before);
    }
}
