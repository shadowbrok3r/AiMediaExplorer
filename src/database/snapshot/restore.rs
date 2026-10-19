use super::*;

#[derive(Clone, Debug, Serialize)]
pub struct RestoreReport {
    pub store: PathBuf,
    pub restored_from: PathBuf,
    pub replaced_to: Option<PathBuf>,
    pub bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PendingRestore {
    pub snapshot: PathBuf,
    pub requested_rfc3339: String,
}

/// Replace `store` with a copy of `snapshot_dir`, moving the old store aside to `<store>.replaced-<ts>`; the store must not be open.
pub fn restore_snapshot_blocking(
    snapshot_dir: &Path,
    store: &Path,
    now: DateTime<Local>,
    space: &SpaceFn,
    progress: &ProgressFn,
) -> Result<RestoreReport, SnapshotError> {
    if !snapshot_dir.is_dir() {
        return Err(SnapshotError::Missing(snapshot_dir.to_path_buf()));
    }
    if store_in_use(store) {
        return Err(SnapshotError::InUse(store.to_path_buf()));
    }
    let check = verify_snapshot_dir(snapshot_dir, progress)?;
    if !check.ok {
        return Err(SnapshotError::Corrupt(format!(
            "verification failed (missing: {}, mismatched: {}, extra: {}, notes: {})",
            check.missing.len(), check.mismatched.len(), check.extra.len(), check.notes.join("; ")
        )));
    }
    let parent = db_parent(store);
    fs::create_dir_all(&parent).map_err(|e| SnapshotError::io(format!("Cannot create {}", parent.display()), e))?;
    check_free_space(required_space(check.bytes_checked), &parent, space)?;

    let ts = now.format(NAME_TS_FORMAT).to_string();
    let name = store_name(store);
    let staging = parent.join(format!("{name}.restoring-{ts}"));
    log_event(log::Level::Info, format!("Restore: copying {} to {}", snapshot_dir.display(), staging.display()));
    let p = progress.clone();
    let outcome = copy_tree(snapshot_dir, &staging, &[LOCK_FILE, MANIFEST_FILE], 1, &mut |cur, total| p(cur, total, "Restoring snapshot files")).map_err(|e| {
        let _ = fs::remove_dir_all(&staging);
        SnapshotError::io("Restore copy failed", e)
    })?;

    let mut replaced_to = None;
    if store.exists() {
        let aside = unique_path(&parent, &format!("{name}.replaced-{ts}"), "");
        fs::rename(store, &aside).map_err(|e| {
            let _ = fs::remove_dir_all(&staging);
            SnapshotError::io(format!("Cannot move {} aside", store.display()), e)
        })?;
        log_event(log::Level::Info, format!("Restore: previous store kept at {}", aside.display()));
        replaced_to = Some(aside);
    }
    if let Err(e) = fs::rename(&staging, store) {
        if let Some(aside) = &replaced_to {
            let _ = fs::rename(aside, store);
        }
        return Err(SnapshotError::io("Cannot move restored store into place", e));
    }
    sync_dir(&parent);

    restamp_after_restore(store, snapshot_dir, now);
    super::super::thumb_migration::reset_state(store);
    log_event(log::Level::Info, format!("Restore complete: {} restored from {}", store.display(), snapshot_dir.display()));
    Ok(RestoreReport { store: store.to_path_buf(), restored_from: snapshot_dir.to_path_buf(), replaced_to, bytes: outcome.bytes })
}

/// Point the engine stamp at the engine that wrote the restored data, or drop it when that is unknown.
fn restamp_after_restore(store: &Path, snapshot_dir: &Path, now: DateTime<Local>) {
    let name = snapshot_dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let parsed = parse_snapshot_name(&name);
    let manifest: Option<SnapshotManifest> = fs::read_to_string(snapshot_dir.join(MANIFEST_FILE)).ok().and_then(|t| serde_json::from_str(&t).ok());
    let engine = match parsed.as_ref().map(|p| p.label.as_str()) {
        Some("pre") => manifest.and_then(|m| m.engine_from),
        Some(_) => parsed.map(|p| p.engine),
        None => None,
    };
    match engine {
        Some(engine) => {
            let mut stamp = stamp_for(store, engine_versions(), now);
            stamp.surrealdb_version = engine;
            stamp.surrealkv_version = None;
            if let Err(e) = write_stamp(store, &stamp) {
                log_event(log::Level::Warn, format!("Restore: could not write engine stamp: {e}"));
            }
        }
        None => {
            let _ = fs::remove_file(stamp_path(store));
        }
    }
}

/// Restore over the configured store; errors while the global DB is connected (use stage_restore and restart).
pub async fn restore_snapshot(snapshot_dir: &Path) -> Result<RestoreReport, SnapshotError> {
    if super::super::is_connected() {
        return Err(SnapshotError::Other("The database is open in this process and its handle cannot be closed. Use stage_restore() and restart the app.".into()));
    }
    let store = PathBuf::from(super::super::get_db_path());
    restore_snapshot_to(snapshot_dir.to_path_buf(), store, status_progress("Restore")).await
}

/// Restore `snapshot_dir` over an explicit store folder on a blocking thread.
pub async fn restore_snapshot_to(snapshot_dir: PathBuf, store: PathBuf, progress: ProgressFn) -> Result<RestoreReport, SnapshotError> {
    tokio::task::spawn_blocking(move || restore_snapshot_blocking(&snapshot_dir, &store, Local::now(), &available_space, &progress))
        .await
        .map_err(|e| SnapshotError::Other(format!("restore task failed: {e}")))?
}

/// Record a restore to apply the next time the database is opened (before it connects).
pub fn stage_restore(store: &Path, snapshot_dir: &Path) -> io::Result<()> {
    let pending = PendingRestore { snapshot: snapshot_dir.to_path_buf(), requested_rfc3339: Local::now().to_rfc3339() };
    let json = serde_json::to_vec_pretty(&pending).map_err(io::Error::other)?;
    write_atomic(&pending_restore_path(store), &json)?;
    log_event(log::Level::Info, format!("Restore of {} staged; it will be applied the next time the database opens", snapshot_dir.display()));
    Ok(())
}

pub fn pending_restore(store: &Path) -> Option<PendingRestore> {
    serde_json::from_str(&fs::read_to_string(pending_restore_path(store)).ok()?).ok()
}

pub fn clear_pending_restore(store: &Path) {
    let _ = fs::remove_file(pending_restore_path(store));
}
