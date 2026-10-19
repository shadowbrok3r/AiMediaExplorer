use super::*;

#[derive(Clone, Debug, Default, Serialize)]
pub struct PruneReport {
    pub deleted: Vec<PathBuf>,
    pub kept: usize,
    pub protected: Vec<PathBuf>,
    pub errors: Vec<String>,
}

fn remove_snapshot_path(info_path: &Path, is_dir: bool) -> io::Result<()> {
    if is_dir { fs::remove_dir_all(info_path) } else { fs::remove_file(info_path) }
}

/// Keep the newest N per kind, never deleting snapshots younger than the minimum age or the latest pre/post ones.
pub fn prune_snapshots(store: &Path, settings: &MaintenanceSettings, now: DateTime<Local>) -> PruneReport {
    let mut report = PruneReport::default();
    let me = store_name(store);
    let all: Vec<SnapshotInfo> = list_snapshots_for(store).into_iter().filter(|s| s.store_name.as_deref() == Some(me.as_str())).collect();
    let min_age = chrono::Duration::from_std(settings.min_age()).unwrap_or_else(|_| chrono::Duration::days(7));
    for (kind, protected_label) in [(SnapshotKind::Dir, "pre"), (SnapshotKind::Surql, "post")] {
        let group: Vec<&SnapshotInfo> = all.iter().filter(|s| s.kind == kind).collect();
        let protected = group.iter().find(|s| s.label.as_deref() == Some(protected_label)).map(|s| s.path.clone());
        for (i, s) in group.iter().enumerate() {
            let young = s.created.map(|c| now.signed_duration_since(c) < min_age).unwrap_or(true);
            if i < settings.retention_keep || young {
                report.kept += 1;
                continue;
            }
            if protected.as_ref() == Some(&s.path) {
                report.protected.push(s.path.clone());
                report.kept += 1;
                continue;
            }
            match remove_snapshot_path(&s.path, kind == SnapshotKind::Dir) {
                Ok(()) => {
                    log_event(log::Level::Info, format!("Retention: deleted old snapshot {}", s.path.display()));
                    report.deleted.push(s.path.clone());
                }
                Err(e) => {
                    log_event(log::Level::Warn, format!("Retention: could not delete {}: {e}", s.path.display()));
                    report.errors.push(format!("{}: {e}", s.path.display()));
                }
            }
        }
    }
    if let Ok(rd) = fs::read_dir(snapshots_dir(store)) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            let old = e.metadata().and_then(|m| m.modified()).ok().and_then(|m| m.elapsed().ok()).map(|a| a > STALE_PARTIAL_AGE).unwrap_or(false);
            if name.ends_with(PARTIAL_SUFFIX) && old {
                let is_dir = e.path().is_dir();
                match remove_snapshot_path(&e.path(), is_dir) {
                    Ok(()) => log_event(log::Level::Info, format!("Retention: deleted stale partial {}", e.path().display())),
                    Err(err) => log_event(log::Level::Warn, format!("Retention: could not delete {}: {err}", e.path().display())),
                }
            }
        }
    }
    report
}

/// Delete one snapshot or export, refusing paths outside the snapshots folder.
pub fn delete_snapshot(store: &Path, path: &Path) -> Result<(), SnapshotError> {
    let dir = snapshots_dir(store);
    let canon_dir = fs::canonicalize(&dir).map_err(|e| SnapshotError::io(format!("Cannot resolve {}", dir.display()), e))?;
    let canon = fs::canonicalize(path).map_err(|_| SnapshotError::Missing(path.to_path_buf()))?;
    if canon.parent() != Some(canon_dir.as_path()) {
        return Err(SnapshotError::Other(format!("{} is not inside the snapshots folder", path.display())));
    }
    let is_dir = canon.is_dir();
    remove_snapshot_path(&canon, is_dir).map_err(|e| SnapshotError::io(format!("Cannot delete {}", path.display()), e))?;
    log_event(log::Level::Info, format!("Deleted snapshot {}", path.display()));
    Ok(())
}
