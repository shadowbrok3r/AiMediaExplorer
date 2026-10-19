//! Backup and restore of the library as tracked tasks: .surql export with optional key redaction, import, directory snapshots and folder copies.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use surrealdb::Surreal;
use surrealdb::engine::local::Db;

use crate::database::snapshot::{self, ProgressFn, SnapshotInfo};
use crate::database::surql::{self, ExportOptions, ExportReport};
use crate::settings::location::{self, Check};
use crate::tasks::{Reporter, TaskKind, TaskSpec};

/// What an export contains.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExportChoice {
    /// Leave the API keys out of the settings record.
    pub redact_keys: bool,
    /// Include the thumbnail blobs; the file gets much larger.
    pub include_thumb_blobs: bool,
}

impl Default for ExportChoice {
    fn default() -> Self {
        Self { redact_keys: true, include_thumb_blobs: false }
    }
}

/// What the `backup` command does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Op {
    /// Write a .surql file of the open library.
    Export,
    /// Run a .surql file against the open library.
    Import,
    /// Build a new library folder from a .surql file.
    Restore,
    /// Copy the library folder into the snapshots folder.
    Snapshot,
}

/// Longest wait for the library to open before a scripted export or import gives up.
pub const LIBRARY_WAIT: std::time::Duration = std::time::Duration::from_secs(120);

/// Waits until the library is open, at most `wait`.
pub async fn library_ready(wait: std::time::Duration) -> anyhow::Result<()> {
    let deadline = std::time::Instant::now() + wait;
    while !crate::database::is_ready() {
        if std::time::Instant::now() >= deadline {
            anyhow::bail!("the library did not open within {} seconds", wait.as_secs());
        }
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    }
    Ok(())
}

/// Starts the backup action `op` as a task; an export or import waits for the library to open. The text says what started.
pub fn run_command(op: Op, path: Option<&str>, dir: Option<&str>, choice: ExportChoice) -> Result<String, crate::command::dispatch::CommandError> {
    use crate::command::dispatch::CommandError;
    let file = |what: &str| path.map(str::trim).filter(|p| !p.is_empty()).map(PathBuf::from).ok_or_else(|| CommandError::invalid(format!("{what} needs a path")));
    match op {
        Op::Export => {
            let file = file("export")?;
            let shown = file.display().to_string();
            tokio::spawn(async move {
                if let Err(e) = library_ready(LIBRARY_WAIT).await {
                    crate::ui::dialogs::toast(egui_toast::ToastKind::Error, format!("Export failed: {e:#}"));
                    return;
                }
                match export_task(&crate::database::DB, file, choice).await {
                    Ok(report) => crate::ui::dialogs::toast(egui_toast::ToastKind::Success, format!("Exported {} to {}", snapshot::human(report.bytes), report.path.display())),
                    Err(e) => crate::ui::dialogs::toast(egui_toast::ToastKind::Error, format!("Export failed: {e:#}")),
                }
            });
            Ok(format!("exporting to {shown}{}", if choice.redact_keys { " without API keys" } else { "" }))
        }
        Op::Import => {
            let file = file("import")?;
            if !file.is_file() {
                return Err(CommandError::not_found(format!("{} is not a file", file.display())));
            }
            let shown = file.display().to_string();
            tokio::spawn(async move {
                if let Err(e) = library_ready(LIBRARY_WAIT).await {
                    crate::ui::dialogs::toast(egui_toast::ToastKind::Error, format!("Import failed: {e:#}"));
                    return;
                }
                match import_task(&crate::database::DB, file).await {
                    Ok(()) => crate::ui::dialogs::toast(egui_toast::ToastKind::Success, "Backup imported"),
                    Err(e) => crate::ui::dialogs::toast(egui_toast::ToastKind::Error, format!("Import failed: {e:#}")),
                }
            });
            Ok(format!("importing {shown}"))
        }
        Op::Restore => {
            let file = file("restore")?;
            if !file.is_file() {
                return Err(CommandError::not_found(format!("{} is not a file", file.display())));
            }
            let target = dir.map(str::trim).filter(|d| !d.is_empty()).map(PathBuf::from).ok_or_else(|| CommandError::invalid("restore needs a dir for the new library"))?;
            if let Some(why) = restore_target_problem(&target) {
                return Err(CommandError::invalid(why));
            }
            let shown = target.display().to_string();
            tokio::spawn(async move {
                match restore_task(file, target).await {
                    Ok(report) => crate::ui::dialogs::toast(egui_toast::ToastKind::Success, format!("Library restored to {} ({} records). Choose it in Settings > Library to use it.", report.dir.display(), report.total_records)),
                    Err(e) => crate::ui::dialogs::toast(egui_toast::ToastKind::Error, format!("Restore failed: {e:#}")),
                }
            });
            Ok(format!("restoring into {shown}"))
        }
        Op::Snapshot => {
            tokio::spawn(async {
                match snapshot_task().await {
                    Ok(info) => crate::ui::dialogs::toast(egui_toast::ToastKind::Success, format!("Snapshot {} made", info.name)),
                    Err(e) => crate::ui::dialogs::toast(egui_toast::ToastKind::Error, format!("Snapshot failed: {e:#}")),
                }
            });
            Ok("snapshot started".to_string())
        }
    }
}

/// A progress sink that reports to the task `reporter`.
pub fn progress_to(reporter: Reporter) -> ProgressFn {
    Arc::new(move |done, total, what| {
        if total > 0 {
            reporter.progress(done, total);
        }
        reporter.detail(what);
    })
}

fn spec(title: String) -> TaskSpec {
    TaskSpec::new(TaskKind::Database, title).user_initiated()
}

/// Suggested file name for an export made now.
pub fn export_file_name(redacted: bool, now: chrono::DateTime<chrono::Local>) -> String {
    let stamp = now.format("%Y%m%d-%H%M%S");
    if redacted { format!("library-{stamp}-no-keys.surql") } else { format!("library-{stamp}.surql") }
}

/// Exports `db` to `file` as a task.
pub async fn export_task(db: &Surreal<Db>, file: PathBuf, choice: ExportChoice) -> anyhow::Result<ExportReport> {
    let title = format!("Export library to {}{}", file.display(), if choice.redact_keys { " (no API keys)" } else { "" });
    crate::tasks::track(spec(title), |reporter| async move {
        let progress = progress_to(reporter);
        let opts = ExportOptions { include_thumb_blobs: choice.include_thumb_blobs };
        if choice.redact_keys {
            surql::export_atomic_redacted(db, &file, &progress, opts).await
        } else {
            surql::export_atomic_with(db, &file, &progress, opts).await
        }
    })
    .await
}

/// Why `dir` cannot take a restored library: it must be missing or empty, never a folder that holds files.
pub fn restore_target_problem(dir: &Path) -> Option<String> {
    match location::check_location(dir) {
        Check::New => None,
        Check::Existing => Some(format!("{} already holds a library; restore into a new folder", dir.display())),
        Check::Rejected(why) => Some(why),
    }
}

/// Builds a new library in the folder `dir` from the .surql `file` as a task; the open library is not touched.
pub async fn restore_task(file: PathBuf, dir: PathBuf) -> anyhow::Result<surql::NewStoreReport> {
    if let Some(why) = restore_target_problem(&dir) {
        anyhow::bail!(why);
    }
    let title = format!("Restore library from {} into {}", file.display(), dir.display());
    crate::tasks::track(spec(title), |reporter| async move { surql::import_into_new_store(&file, &dir, &progress_to(reporter)).await }).await
}

/// The folder a restored library goes to by default: next to the current library, never an existing one.
pub fn default_restore_dir() -> PathBuf {
    let store = PathBuf::from(crate::database::get_db_path());
    snapshot::unique_path(&snapshot::db_parent(&store), &format!("{}-restored", snapshot::store_name(&store)), "")
}

/// Runs the .surql `file` against `db` as a task.
pub async fn import_task(db: &Surreal<Db>, file: PathBuf) -> anyhow::Result<()> {
    let title = format!("Import library from {}", file.display());
    crate::tasks::track(spec(title), |reporter| async move {
        reporter.detail("Importing records");
        let _activity = crate::database::db_activity("IMPORT DB");
        db.import(&file).await?;
        Ok::<(), anyhow::Error>(())
    })
    .await
}

/// Copies the store folder into a snapshot as a task.
pub async fn snapshot_task() -> anyhow::Result<SnapshotInfo> {
    crate::tasks::track(spec("Snapshot the library folder".to_string()), |reporter| async move {
        snapshot::snapshot_now_with(progress_to(reporter)).await.map_err(|e| anyhow::anyhow!(e.user_message()))
    })
    .await
}

/// The folder a copy of the store at `store` goes to inside `parent`; it never exists yet.
pub fn copy_target(parent: &Path, store: &Path) -> PathBuf {
    snapshot::unique_path(parent, &format!("{}-copy", snapshot::store_name(store)), "")
}

/// Copies the store folder at `store` to a new folder inside `parent` as a task; returns the new folder.
pub async fn copy_store_task(store: PathBuf, parent: PathBuf) -> anyhow::Result<PathBuf> {
    let target = copy_target(&parent, &store);
    let title = format!("Copy library to {}", target.display());
    let dest = target.clone();
    crate::tasks::track(spec(title), |reporter| async move {
        let progress = progress_to(reporter);
        let outcome = tokio::task::spawn_blocking(move || snapshot::copy_tree(&store, &dest, &[snapshot::LOCK_FILE], 2, &mut |done, total| progress(done, total, "Copying database files"))).await??;
        if !outcome.consistent {
            log::warn!("[backup] the library changed while it was copied; open the copy and run a verify");
        }
        Ok::<PathBuf, anyhow::Error>(target)
    })
    .await
}

/// True while a Database task of the backup actions is running.
pub fn busy() -> bool {
    crate::tasks::with(|r| r.tasks().iter().any(|t| t.kind == TaskKind::Database && t.state.is_active() && is_backup_title(&t.title)))
}

/// True for the titles the tasks above use.
pub fn is_backup_title(title: &str) -> bool {
    ["Export library", "Import library", "Restore library", "Snapshot the library", "Copy library"].iter().any(|p| title.starts_with(p))
}

/// Active and finished backup tasks, newest last.
pub fn recent() -> Vec<crate::tasks::Task> {
    crate::tasks::with(|r| r.tasks().iter().filter(|t| t.kind == TaskKind::Database && is_backup_title(&t.title)).cloned().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::collections::tests::db as fresh_db;
    use crate::database::scope_queries::tests::{insert, seed};
    use crate::tasks::TaskState;
    use surrealdb::types::ToSql;

    async fn empty_db() -> Surreal<Db> {
        let db = Surreal::new::<surrealdb::engine::local::Mem>(()).await.expect("memory database");
        db.use_ns(crate::database::NS).use_db(crate::database::DB_NAME).await.expect("select");
        db
    }

    async fn seeded() -> Surreal<Db> {
        let db = fresh_db().await;
        let mut rows: Vec<_> = (0..5).map(|i| seed(i, &format!("/lib/p{i}.jpg"))).collect();
        rows[0].tags = vec!["sun".into(), "sea".into()];
        insert(&db, rows).await;
        db.query("UPSERT user_settings:ShadowbrokerPC CONTENT { auto_indexing: true, ai_prompt_template: 'Describe', overwrite_descriptions: false, filter_skip_icons: true, auto_clip_embeddings: false, clip_augment_with_text: true, clip_overwrite_embeddings: false, recent_paths: ['/a'], recent_models: [], auto_save_to_database: false, group_by_category: false, openai_api_key: 'sk-secret-one', grok_api_key: 'xai-secret-two', openrouter_api_key: 'or-secret-three', openai_default_model: 'gpt-x' }")
            .await
            .expect("settings")
            .check()
            .expect("settings written");
        db
    }

    async fn count(db: &Surreal<Db>, table: &str) -> u64 {
        surql::count_rows(db, table).await.unwrap()
    }

    #[tokio::test]
    async fn the_backup_command_checks_its_arguments_before_it_starts_anything() {
        use crate::command::dispatch::ErrorKind;
        let choice = ExportChoice::default();
        assert_eq!(run_command(Op::Export, None, None, choice).unwrap_err().kind, ErrorKind::Invalid);
        assert_eq!(run_command(Op::Import, Some("  "), None, choice).unwrap_err().kind, ErrorKind::Invalid);
        assert_eq!(run_command(Op::Import, Some("/no/such/file.surql"), None, choice).unwrap_err().kind, ErrorKind::NotFound);
    }

    #[tokio::test]
    async fn a_restore_builds_a_new_library_folder_and_refuses_folders_that_hold_anything() {
        let db = seeded().await;
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("backup.surql");
        export_task(&db, file.clone(), ExportChoice { redact_keys: true, include_thumb_blobs: false }).await.unwrap();

        let target = dir.path().join("restored").join("lib");
        assert_eq!(restore_target_problem(&target), None);
        let report = restore_task(file.clone(), target.clone()).await.unwrap();
        assert_eq!(report.dir, target);
        assert!(report.total_records >= 6, "{} records", report.total_records);
        assert!(crate::settings::location::looks_like_store(&target), "the folder holds a library now");

        assert!(restore_target_problem(&target).unwrap().contains("already holds a library"));
        assert!(restore_task(file.clone(), target.clone()).await.is_err(), "a library is never restored over another");

        let busy = dir.path().join("busy");
        std::fs::create_dir_all(&busy).unwrap();
        std::fs::write(busy.join("photo.jpg"), "x").unwrap();
        assert!(restore_target_problem(&busy).is_some());
        assert!(restore_task(file, busy.clone()).await.is_err());
        assert!(busy.join("photo.jpg").exists());
    }

    #[tokio::test]
    async fn the_restore_command_checks_the_file_and_the_folder_before_it_starts() {
        use crate::command::dispatch::ErrorKind;
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("b.surql");
        std::fs::write(&file, "x").unwrap();
        let (path, target) = (file.to_str().unwrap(), dir.path().join("new").display().to_string());
        let choice = ExportChoice::default();
        assert_eq!(run_command(Op::Restore, None, Some(&target), choice).unwrap_err().kind, ErrorKind::Invalid);
        assert_eq!(run_command(Op::Restore, Some("/no/such.surql"), Some(&target), choice).unwrap_err().kind, ErrorKind::NotFound);
        assert_eq!(run_command(Op::Restore, Some(path), None, choice).unwrap_err().kind, ErrorKind::Invalid, "a folder is needed");
        let busy = dir.path().display().to_string();
        assert_eq!(run_command(Op::Restore, Some(path), Some(&busy), choice).unwrap_err().kind, ErrorKind::Invalid, "a folder with files is refused");
    }

    #[tokio::test]
    async fn a_scripted_export_waits_for_the_library_and_gives_up_when_it_never_opens() {
        let err = library_ready(std::time::Duration::from_millis(300)).await.unwrap_err();
        assert!(format!("{err}").contains("did not open"), "{err}");
    }

    #[test]
    fn the_backup_command_json_defaults_to_redacting_the_keys() {
        let cmd: crate::command::AppCommand = serde_json::from_str(r#"{"cmd":"backup","op":"export","path":"/x.surql"}"#).unwrap();
        assert!(matches!(cmd, crate::command::AppCommand::Backup { op: Op::Export, redact_keys: true, include_thumbs: false, .. }));
        let full: crate::command::AppCommand = serde_json::from_str(r#"{"cmd":"backup","op":"export","path":"/x.surql","redact_keys":false}"#).unwrap();
        assert!(matches!(full, crate::command::AppCommand::Backup { redact_keys: false, .. }));
        assert!(serde_json::from_str::<crate::command::AppCommand>(r#"{"cmd":"backup","op":"erase"}"#).is_err());
        assert!(serde_json::from_str::<crate::command::AppCommand>(r#"{"cmd":"backup"}"#).is_err());
        let json = serde_json::to_value(&cmd).unwrap();
        assert_eq!(json, serde_json::json!({"cmd": "backup", "op": "export", "path": "/x.surql"}), "defaults stay out of the JSON");
    }

    #[test]
    fn export_names_say_whether_keys_are_left_out() {
        let now = chrono::Local::now();
        assert!(export_file_name(true, now).ends_with("-no-keys.surql"));
        assert!(!export_file_name(false, now).contains("no-keys"));
        assert!(export_file_name(false, now).starts_with("library-"));
    }

    #[test]
    fn the_defaults_redact_keys_and_leave_the_blobs_out() {
        let c = ExportChoice::default();
        assert!(c.redact_keys && !c.include_thumb_blobs);
    }

    #[test]
    fn a_copy_never_lands_on_an_existing_folder() {
        let parent = tempfile::tempdir().unwrap();
        let store = parent.path().join("ai_search");
        let first = copy_target(parent.path(), &store);
        assert_eq!(first.file_name().unwrap(), "ai_search-copy");
        std::fs::create_dir_all(&first).unwrap();
        let second = copy_target(parent.path(), &store);
        assert_ne!(second, first);
        assert!(!second.exists());
    }

    #[test]
    fn only_backup_titles_count_as_backup_tasks() {
        assert!(is_backup_title("Export library to /a.surql"));
        assert!(is_backup_title("Copy library to /b"));
        assert!(!is_backup_title("Open database /x"));
    }

    #[tokio::test]
    async fn a_redacted_export_leaves_no_key_in_the_file_and_imports_into_a_new_store() {
        let db = seeded().await;
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("out").join("backup.surql");
        let report = export_task(&db, file.clone(), ExportChoice::default()).await.unwrap();
        assert_eq!(report.path, file);
        let text = std::fs::read_to_string(&file).unwrap();
        for secret in ["sk-secret-one", "xai-secret-two", "or-secret-three"] {
            assert!(!text.contains(secret), "{secret} leaked into the backup");
        }
        assert!(text.contains("gpt-x"), "the other settings are kept");
        assert!(!dir.path().join("out").join("backup.surql.partial").exists());

        let target = empty_db().await;
        import_task(&target, file).await.unwrap();
        assert_eq!(count(&target, "thumbnails").await, 5);
        let rows: Vec<surrealdb::types::Value> = target.query("SELECT * FROM user_settings").await.unwrap().take(0).unwrap();
        assert_eq!(rows.len(), 1);
        let row = rows[0].to_sql();
        assert!(row.contains("gpt-x") && row.contains("Describe"), "{row}");
        assert!(!row.contains("api_key"), "no key field is restored: {row}");
    }

    #[tokio::test]
    async fn a_plain_export_keeps_everything_and_restores_it() {
        let db = seeded().await;
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("full.surql");
        let choice = ExportChoice { redact_keys: false, include_thumb_blobs: false };
        export_task(&db, file.clone(), choice).await.unwrap();
        assert!(std::fs::read_to_string(&file).unwrap().contains("sk-secret-one"));
        let target = empty_db().await;
        import_task(&target, file).await.unwrap();
        assert_eq!(count(&target, "thumbnails").await, 5);
        let keys: Vec<Option<String>> = target.query("SELECT VALUE openai_api_key FROM user_settings").await.unwrap().take(0).unwrap();
        assert_eq!(keys, vec![Some("sk-secret-one".to_string())]);
    }

    #[tokio::test]
    async fn tasks_report_their_title_state_and_failure() {
        let db = seeded().await;
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("t.surql");
        export_task(&db, file.clone(), ExportChoice::default()).await.unwrap();
        let ok = recent().into_iter().rev().find(|t| t.title.contains(&file.display().to_string())).expect("the export is a task");
        assert_eq!(ok.state, TaskState::Completed);
        let missing = dir.path().join("missing.surql");
        let target = empty_db().await;
        assert!(import_task(&target, missing.clone()).await.is_err());
        let failed = recent().into_iter().rev().find(|t| t.title.contains(&missing.display().to_string())).expect("the import is a task");
        assert_eq!(failed.state, TaskState::Failed);
        assert!(failed.error.is_some());
    }

    #[tokio::test]
    async fn a_store_copy_has_the_same_files_in_a_new_folder() {
        let parent = tempfile::tempdir().unwrap();
        let store = parent.path().join("lib");
        std::fs::create_dir_all(store.join("wal")).unwrap();
        std::fs::write(store.join("wal").join("0001"), b"records").unwrap();
        std::fs::write(store.join("manifest"), b"m").unwrap();
        std::fs::write(store.join("LOCK"), b"").unwrap();
        let dest_parent = tempfile::tempdir().unwrap();
        let copy = copy_store_task(store.clone(), dest_parent.path().to_path_buf()).await.unwrap();
        assert_eq!(copy.parent().unwrap(), dest_parent.path());
        assert_eq!(std::fs::read(copy.join("wal").join("0001")).unwrap(), b"records");
        assert!(!copy.join("LOCK").exists(), "the lock file is not copied");
        assert!(store.join("manifest").exists(), "the source is untouched");
    }
}
