use super::*;

static OVERRIDE_ONCE: AtomicBool = AtomicBool::new(false);
static BLOCKED: Mutex<Option<Blocked>> = Mutex::new(None);
static LAST_OPEN: Mutex<Option<OpenSummary>> = Mutex::new(None);

/// Skip the pre-open snapshot (and its free-space guard) for the next open only.
pub fn override_snapshot_once() {
    OVERRIDE_ONCE.store(true, Ordering::SeqCst);
    log_event(log::Level::Warn, "Pre-open snapshot will be skipped for the next open (user override)");
}

pub fn override_pending() -> bool {
    OVERRIDE_ONCE.load(Ordering::SeqCst)
}

#[derive(Clone, Debug)]
pub struct Blocked {
    pub message: String,
    pub overridable: bool,
}

/// Record why the last open attempt was refused before connecting.
pub fn set_blocked(err: &SnapshotError) {
    let overridable = matches!(err, SnapshotError::InsufficientSpace { .. } | SnapshotError::Io { .. });
    let message = err.user_message();
    log_event(log::Level::Error, format!("Database open blocked: {message}"));
    if let Ok(mut b) = BLOCKED.lock() { *b = Some(Blocked { message, overridable }); }
}

pub fn clear_blocked() {
    if let Ok(mut b) = BLOCKED.lock() { *b = None; }
}

/// Refusal recorded by the last open attempt, if any.
pub fn blocked() -> Option<Blocked> {
    BLOCKED.lock().ok().and_then(|b| b.clone())
}

#[derive(Clone, Debug, Serialize)]
pub struct OpenSummary {
    pub change: EngineChange,
    pub snapshot: Option<PathBuf>,
    pub note: Option<String>,
    pub auto_export: Option<PathBuf>,
    pub restored: Option<PathBuf>,
}

/// Outcome of the pre-open phase, handed to `finish_open` and `spawn_auto_export`.
#[derive(Clone, Debug)]
pub struct PreOpen {
    pub change: EngineChange,
    pub snapshot: Option<SnapshotInfo>,
    pub note: Option<String>,
    pub settings: MaintenanceSettings,
    pub restored: Option<RestoreReport>,
}

pub fn last_open_summary() -> Option<OpenSummary> {
    LAST_OPEN.lock().ok().and_then(|g| g.clone())
}

fn record_open(pre: &PreOpen) {
    if let Ok(mut g) = LAST_OPEN.lock() {
        *g = Some(OpenSummary {
            change: pre.change.clone(),
            snapshot: pre.snapshot.as_ref().map(|s| s.path.clone()),
            note: pre.note.clone(),
            auto_export: g.as_ref().and_then(|s| s.auto_export.clone()),
            restored: pre.restored.as_ref().map(|r| r.restored_from.clone()),
        });
    }
}

/// Apply a staged restore, then snapshot the store if the engine changed; call before connecting.
pub async fn prepare_open(store: &Path) -> Result<PreOpen, SnapshotError> {
    let settings = MaintenanceSettings::load(store);
    let mut restored = None;
    if let Some(pending) = pending_restore(store) {
        log_event(log::Level::Info, format!("Applying staged restore from {}", pending.snapshot.display()));
        DB_STATUS.set_state(StatusState::Initializing, "Restoring database snapshot");
        let res = restore_snapshot_to(pending.snapshot.clone(), store.to_path_buf(), status_progress("Restore")).await;
        clear_pending_restore(store);
        match res {
            Ok(r) => restored = Some(r),
            Err(e) => log_event(log::Level::Error, format!("Staged restore failed, opening the existing store: {}", e.user_message())),
        }
    }
    let mut pre = prepare_open_with(store, settings, Arc::new(available_space), Local::now(), status_progress("Snapshot")).await?;
    pre.restored = restored;
    DB_STATUS.set_progress(0, 0);
    record_open(&pre);
    Ok(pre)
}

/// Pre-open flow with injectable free-space probe, clock and progress sink.
pub async fn prepare_open_with(
    store: &Path,
    settings: MaintenanceSettings,
    space: Arc<SpaceFn>,
    now: DateTime<Local>,
    progress: ProgressFn,
) -> Result<PreOpen, SnapshotError> {
    let override_requested = OVERRIDE_ONCE.swap(false, Ordering::SeqCst);
    let has_data = store_has_data(store);
    let stamp = read_stamp(store);
    let change = classify(store, has_data, &stamp, engine_versions());
    let mut pre = PreOpen { change: change.clone(), snapshot: None, note: None, settings: settings.clone(), restored: None };
    let EngineChange::Changed { from, to, reason } = &change else {
        return Ok(pre);
    };
    log_event(log::Level::Info, format!("Engine change detected for {}: {reason}", store.display()));
    if !settings.snapshot_on_engine_change {
        pre.note = Some("pre-open snapshot disabled in settings".into());
        log_event(log::Level::Warn, "Pre-open snapshot disabled in settings; opening without a backup");
        return Ok(pre);
    }
    if override_requested {
        pre.note = Some("pre-open snapshot skipped by user override".into());
        log_event(log::Level::Warn, "Pre-open snapshot skipped by user override; opening without a backup");
        return Ok(pre);
    }
    if store_in_use(store) {
        return Err(SnapshotError::InUse(store.to_path_buf()));
    }
    DB_STATUS.set_state(StatusState::Initializing, format!("Snapshotting database before engine upgrade to {to}"));
    let store_buf = store.to_path_buf();
    let engine_from = from.clone();
    let handle = tokio::task::spawn_blocking(move || create_snapshot_blocking(&store_buf, "pre", engine_from, now, space.as_ref(), &progress));
    let info = handle.await.map_err(|e| SnapshotError::Other(format!("snapshot task failed: {e}")))??;
    pre.snapshot = Some(info);
    let report = prune_snapshots(store, &settings, now);
    if !report.deleted.is_empty() {
        log_event(log::Level::Info, format!("Retention removed {} old snapshot(s)", report.deleted.len()));
    }
    Ok(pre)
}

/// Record the running engine in the stamp; call only after connect and schema define succeeded.
pub fn finish_open(store: &Path) {
    let stamp = stamp_for(store, engine_versions(), Local::now());
    match write_stamp(store, &stamp) {
        Ok(()) => log::info!("[DB] engine stamp updated: surrealdb {}", stamp.surrealdb_version),
        Err(e) => log_event(log::Level::Warn, format!("Could not write engine stamp {}: {e}", stamp_path(store).display())),
    }
}

#[derive(Clone, Debug, Serialize)]
pub enum AutoExport {
    Skipped(String),
    Exported(PathBuf, u64),
    Failed(String),
}

/// Export the open DB to `<snapshots>/<store>-post-<engine>-<ts>.surql` when the engine changed and the store is small enough.
pub async fn auto_export_if_changed(db: &surrealdb::Surreal<surrealdb::engine::local::Db>, store: &Path, pre: &PreOpen) -> AutoExport {
    if !pre.change.is_changed() {
        return AutoExport::Skipped("engine unchanged".into());
    }
    if !pre.settings.auto_export_on_engine_change {
        return AutoExport::Skipped("auto export disabled in settings".into());
    }
    let size_store = store.to_path_buf();
    let store_size = tokio::task::spawn_blocking(move || dir_size(&size_store)).await.unwrap_or(0);
    let include_blobs = pre.settings.export_include_thumb_blobs;
    let blob_bytes = if include_blobs { 0 } else { super::super::thumb_blobs::quick_stats_with(db, 256).await.map(|q| q.stats.total_bytes).unwrap_or(0) };
    let size = store_size.saturating_sub(blob_bytes);
    if size > pre.settings.export_size_threshold_bytes {
        let msg = format!("store is {} which exceeds the auto-export threshold of {}", human(size), human(pre.settings.export_size_threshold_bytes));
        log_event(log::Level::Info, format!("Auto export skipped: {msg}"));
        return AutoExport::Skipped(msg);
    }
    let dir = snapshots_dir(store);
    if let Err(e) = fs::create_dir_all(&dir) {
        log_event(log::Level::Error, format!("Auto export skipped: cannot create {}: {e}", dir.display()));
        return AutoExport::Failed(format!("cannot create {}: {e}", dir.display()));
    }
    if let Err(e) = check_free_space(size, &dir, &available_space) {
        log_event(log::Level::Warn, format!("Auto export skipped: {}", e.user_message()));
        return AutoExport::Skipped(e.user_message());
    }
    let base = snapshot_file_name(&store_name(store), "post", &engine_versions().surrealdb, Local::now());
    let path = unique_path(&dir, &base, ".surql");
    log_event(log::Level::Info, format!("Auto export started: {}", path.display()));
    let _ga = super::super::db_activity("Auto export DB");
    let opts = super::super::surql::ExportOptions { include_thumb_blobs: include_blobs };
    match super::super::surql::export_atomic_with(db, &path, &status_progress("Export"), opts).await {
        Ok(rep) => {
            log_event(log::Level::Info, format!("Auto export done: {} ({} in {:.1}s)", path.display(), human(rep.bytes), rep.millis as f32 / 1000.0));
            if let Ok(mut g) = LAST_OPEN.lock() {
                if let Some(s) = g.as_mut() { s.auto_export = Some(path.clone()); }
            }
            prune_snapshots(store, &pre.settings, Local::now());
            AutoExport::Exported(path, rep.bytes)
        }
        Err(e) => {
            log_event(log::Level::Error, format!("Auto export failed: {e}"));
            AutoExport::Failed(e.to_string())
        }
    }
}

/// Run the post-upgrade auto export on the global DB in the background.
pub fn spawn_auto_export(store: PathBuf, pre: PreOpen) {
    if !pre.change.is_changed() {
        return;
    }
    let Ok(handle) = tokio::runtime::Handle::try_current() else { return };
    handle.spawn(async move {
        tokio::time::sleep(AUTO_EXPORT_DELAY).await;
        let _ = auto_export_if_changed(&super::super::DB, &store, &pre).await;
    });
}
