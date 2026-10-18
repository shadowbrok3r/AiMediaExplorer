use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use super::snapshot::{self, ProgressFn, human, store_has_data, store_in_use};
use super::surql::{self, ExportOptions, VerifyOptions, VerifyTarget};
use super::thumb_blobs;
use super::thumb_migration::{CopyOptions, FinishOptions, Migrator};

pub const EXIT_OK: i32 = 0;
pub const EXIT_FAILED: i32 = 1;
pub const EXIT_USAGE: i32 = 2;
pub const EXIT_IN_USE: i32 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ThumbsAction {
    Info,
    Migrate,
    Finish,
    Missing,
}

#[derive(Debug, PartialEq, Eq)]
enum Command {
    Snapshot,
    Export { file: PathBuf, with_thumbs: bool },
    Import { file: PathBuf, force: bool },
    Verify { target: PathBuf, no_compare: bool, with_thumbs: bool },
    Restore(PathBuf),
    Info,
    Thumbs { action: ThumbsAction, clear_undecodable: bool },
}

#[derive(Debug, PartialEq, Eq)]
struct Parsed {
    command: Command,
    db_path: Option<PathBuf>,
}

const DB_FLAGS: [&str; 7] = ["--db-snapshot", "--db-export", "--db-import", "--db-verify", "--db-restore", "--db-info", "--db-thumbs"];

fn split_flag(arg: &str) -> (&str, Option<&str>) {
    match arg.split_once('=') {
        Some((k, v)) if k.starts_with("--") => (k, Some(v)),
        _ => (arg, None),
    }
}

/// Parse `--db-*` flags; None when no database flag is present.
fn parse(args: &[String]) -> Option<Result<Parsed, String>> {
    if !args.iter().any(|a| DB_FLAGS.contains(&split_flag(a).0)) {
        return None;
    }
    let mut command: Option<Command> = None;
    let mut db_path = None;
    let mut force = false;
    let mut no_compare = false;
    let mut with_thumbs = false;
    let mut clear_undecodable = false;
    let mut i = 0;
    let mut value = |i: &mut usize, inline: Option<&str>, flag: &str| -> Result<String, String> {
        if let Some(v) = inline {
            return Ok(v.to_string());
        }
        *i += 1;
        match args.get(*i) {
            Some(v) if !v.starts_with("--") => Ok(v.clone()),
            _ => Err(format!("{flag} requires a value")),
        }
    };
    while i < args.len() {
        let (flag, inline) = split_flag(&args[i]);
        let next = match flag {
            "--db-snapshot" => Some(Ok(Command::Snapshot)),
            "--db-info" => Some(Ok(Command::Info)),
            "--db-export" => Some(value(&mut i, inline, flag).map(|v| Command::Export { file: PathBuf::from(v), with_thumbs: false })),
            "--db-thumbs" => Some(value(&mut i, inline, flag).and_then(|v| match v.as_str() {
                "info" => Ok(ThumbsAction::Info),
                "migrate" => Ok(ThumbsAction::Migrate),
                "finish" => Ok(ThumbsAction::Finish),
                "missing" => Ok(ThumbsAction::Missing),
                other => Err(format!("--db-thumbs expects info, migrate, finish or missing, not '{other}'")),
            }).map(|action| Command::Thumbs { action, clear_undecodable: false })),
            "--db-import" => Some(value(&mut i, inline, flag).map(|v| Command::Import { file: PathBuf::from(v), force: false })),
            "--db-verify" => Some(value(&mut i, inline, flag).map(|v| Command::Verify { target: PathBuf::from(v), no_compare: false, with_thumbs: false })),
            "--db-restore" => Some(value(&mut i, inline, flag).map(|v| Command::Restore(PathBuf::from(v)))),
            "--db-path" => match value(&mut i, inline, flag) {
                Ok(v) => {
                    db_path = Some(PathBuf::from(v));
                    None
                }
                Err(e) => return Some(Err(e)),
            },
            "--force" => {
                force = true;
                None
            }
            "--no-compare" => {
                no_compare = true;
                None
            }
            "--with-thumbs" => {
                with_thumbs = true;
                None
            }
            "--clear-undecodable" => {
                clear_undecodable = true;
                None
            }
            _ => None,
        };
        match next {
            Some(Ok(c)) => {
                if command.is_some() {
                    return Some(Err("only one of --db-snapshot, --db-export, --db-import, --db-verify, --db-restore, --db-info, --db-thumbs may be given".into()));
                }
                command = Some(c);
            }
            Some(Err(e)) => return Some(Err(e)),
            None => {}
        }
        i += 1;
    }
    let mut command = command?;
    match &mut command {
        Command::Import { force: f, .. } => *f = force,
        Command::Verify { no_compare: n, with_thumbs: w, .. } => {
            *n = no_compare;
            *w = with_thumbs;
        }
        Command::Export { with_thumbs: w, .. } => *w = with_thumbs,
        Command::Thumbs { clear_undecodable: c, .. } => *c = clear_undecodable,
        _ => {}
    }
    Some(Ok(Parsed { command, db_path }))
}

fn usage() -> &'static str {
    "database flags:\n  --db-info                         print engine, stamp and snapshots as JSON (does not open the store)\n  --db-snapshot                     copy the store folder into <db>/snapshots (store must not be in use)\n  --db-export <file.surql> [--with-thumbs]  export the database to a .surql file (thumbnail blob records are left out unless --with-thumbs)\n  --db-import <file.surql> [--force]  import into a new or empty store (--force to import into a populated one)\n  --db-verify <file.surql|dir> [--no-compare] [--with-thumbs]  import into a scratch store and compare with the store (dir: deep-verify a snapshot copy; --with-thumbs also compares thumbnail blob records)\n  --db-thumbs <info|migrate|finish|missing> [--clear-undecodable]  thumbnail blob storage: stats, copy inline thumbnails into blobs, verify and clear the inline copies, list rows without a blob\n  --db-restore <snapshot dir>       replace the store with a snapshot (old store kept as <store>.replaced-<ts>; store must not be in use)
  --db-path <dir>                   store folder to use instead of the configured one"
}

fn cli_progress() -> ProgressFn {
    let last = std::sync::Mutex::new(Instant::now() - Duration::from_secs(5));
    Arc::new(move |cur, total, what| {
        let Ok(mut l) = last.lock() else { return };
        if l.elapsed() < Duration::from_secs(2) {
            return;
        }
        *l = Instant::now();
        if total > 0 {
            eprintln!("{what}: {} of {} ({:.0}%)", human(cur), human(total), cur as f64 * 100.0 / total as f64);
        } else {
            eprintln!("{what}: {}", human(cur));
        }
    })
}

fn print_json(v: &Value) {
    println!("{}", serde_json::to_string_pretty(v).unwrap_or_else(|_| v.to_string()));
}

fn fail(msg: impl std::fmt::Display) -> i32 {
    eprintln!("error: {msg}");
    EXIT_FAILED
}

fn in_use_exit(store: &Path) -> i32 {
    eprintln!("error: the database at {} is in use by another process; close the app and retry", store.display());
    EXIT_IN_USE
}

async fn table_counts(db: &surrealdb::Surreal<surrealdb::engine::local::Db>) -> anyhow::Result<Value> {
    let mut resp = db.query("INFO FOR DB").await?;
    let info: surrealdb::types::Value = resp.take(0)?;
    let info = info.into_json_value();
    let mut counts = serde_json::Map::new();
    if let Some(tables) = info.get("tables").and_then(|t| t.as_object()) {
        for name in tables.keys() {
            counts.insert(name.clone(), json!(surql::count_rows(db, name).await?));
        }
    }
    Ok(Value::Object(counts))
}

fn spawn_progress_reporter(m: &std::sync::Arc<Migrator>) -> tokio::task::JoinHandle<()> {
    let m = std::sync::Arc::clone(m);
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(2)).await;
            let p = m.progress();
            eprintln!(
                "{}: {} processed of {} ({} migrated, {} failed, {:.0} rows/s)",
                p.phase.label(),
                p.processed,
                p.rows_total.map_or("?".to_string(), |t| t.to_string()),
                p.migrated,
                p.failed,
                p.rows_per_sec()
            );
        }
    })
}

async fn run_thumbs(store: &Path, action: ThumbsAction, clear_undecodable: bool) -> i32 {
    if !store_has_data(store) {
        return fail(format!("no database found at {}", store.display()));
    }
    if store_in_use(store) {
        return in_use_exit(store);
    }
    let opened = match surql::open_store(store).await {
        Ok(o) => o,
        Err(e) => return fail(e),
    };
    let code = {
        let m = Migrator::new(opened.db.clone(), store.to_path_buf());
        match action {
            ThumbsAction::Info => {
                let blobs = thumb_blobs::stats_with(&opened.db).await;
                let rows = surql::count_rows(&opened.db, super::THUMBNAILS).await;
                let legacy = m.count_legacy().await;
                match (blobs, rows, legacy) {
                    (Ok(b), Ok(r), Ok(l)) => {
                        print_json(&json!({
                            "thumbnail_rows": r,
                            "rows_with_inline_thumbnail": l,
                            "blobs": { "count": b.count, "total_bytes": b.total_bytes, "avg_bytes": b.avg_bytes() },
                            "migration": m.state(),
                        }));
                        EXIT_OK
                    }
                    (b, r, l) => fail(format!("stats failed: {:?} {:?} {:?}", b.err(), r.err(), l.err())),
                }
            }
            ThumbsAction::Migrate => {
                let reporter = spawn_progress_reporter(&m);
                let res = m.run_copy(CopyOptions::default()).await;
                reporter.abort();
                match res {
                    Ok(report) => {
                        print_json(&json!({ "copy": report, "state": m.state() }));
                        EXIT_OK
                    }
                    Err(e) => fail(format!("{e:#}")),
                }
            }
            ThumbsAction::Finish => {
                let reporter = spawn_progress_reporter(&m);
                let res = m.run_finish(FinishOptions { clear_undecodable, ..Default::default() }).await;
                reporter.abort();
                match res {
                    Ok(report) => {
                        print_json(&json!({ "finish": report, "state": m.state() }));
                        if report.ok { EXIT_OK } else { EXIT_FAILED }
                    }
                    Err(e) => fail(format!("{e:#}")),
                }
            }
            ThumbsAction::Missing => match m.missing_paths(usize::MAX).await {
                Ok(report) => {
                    print_json(&json!({ "scanned": report.scanned, "missing": report.missing, "paths": report.paths }));
                    EXIT_OK
                }
                Err(e) => fail(format!("{e:#}")),
            },
        }
    };
    surql::close_store(opened, store).await;
    code
}

async fn run_command(parsed: Parsed) -> i32 {
    let store = parsed.db_path.unwrap_or_else(|| PathBuf::from(super::get_db_path()));
    let progress = cli_progress();
    match parsed.command {
        Command::Info => {
            print_json(&super::maintenance::info_json(&store));
            EXIT_OK
        }
        Command::Snapshot => {
            if !store_has_data(&store) {
                return fail(format!("no database found at {}", store.display()));
            }
            if store_in_use(&store) {
                return in_use_exit(&store);
            }
            match snapshot::create_snapshot(store.clone(), "manual", None, progress).await {
                Ok(info) => {
                    let settings = snapshot::MaintenanceSettings::load(&store);
                    snapshot::prune_snapshots(&store, &settings, chrono::Local::now());
                    print_json(&json!({ "snapshot": info }));
                    EXIT_OK
                }
                Err(e) => fail(e),
            }
        }
        Command::Export { file, with_thumbs } => {
            if !store_has_data(&store) {
                return fail(format!("no database found at {}", store.display()));
            }
            if store_in_use(&store) {
                return in_use_exit(&store);
            }
            let opened = match surql::open_store(&store).await {
                Ok(o) => o,
                Err(e) => return fail(e),
            };
            let result = surql::export_atomic_with(&opened.db, &file, &progress, ExportOptions { include_thumb_blobs: with_thumbs }).await;
            let counts = table_counts(&opened.db).await.unwrap_or(Value::Null);
            let (change, pre_snapshot) = (opened.pre.change.clone(), opened.pre.snapshot.clone());
            surql::close_store(opened, &store).await;
            match result {
                Ok(report) => {
                    print_json(&json!({ "export": report, "tables": counts, "engine_change": change, "pre_open_snapshot": pre_snapshot }));
                    EXIT_OK
                }
                Err(e) => fail(e),
            }
        }
        Command::Import { file, force } => {
            if !file.is_file() {
                return fail(format!("import file not found: {}", file.display()));
            }
            if store_has_data(&store) {
                if !force {
                    eprintln!("error: {} already contains a database; pass --force to import into it or choose another --db-path", store.display());
                    return EXIT_USAGE;
                }
                if store_in_use(&store) {
                    return in_use_exit(&store);
                }
                let opened = match surql::open_store(&store).await {
                    Ok(o) => o,
                    Err(e) => return fail(e),
                };
                let result = opened.db.import(&file).await;
                let counts = table_counts(&opened.db).await.unwrap_or(Value::Null);
                let pre_snapshot = opened.pre.snapshot.clone();
                surql::close_store(opened, &store).await;
                match result {
                    Ok(()) => {
                        print_json(&json!({ "imported_into": store, "tables": counts, "pre_open_snapshot": pre_snapshot }));
                        EXIT_OK
                    }
                    Err(e) => fail(format!("import failed: {e}")),
                }
            } else {
                match surql::import_into_new_store(&file, &store, &progress).await {
                    Ok(rep) => {
                        print_json(&json!({ "imported_into_new_store": rep.dir, "import_millis": rep.import_millis, "total_records": rep.total_records, "tables": rep.manifest.counts() }));
                        EXIT_OK
                    }
                    Err(e) => fail(e),
                }
            }
        }
        Command::Thumbs { action, clear_undecodable } => run_thumbs(&store, action, clear_undecodable).await,
        Command::Restore(dir) => {
            if store_in_use(&store) {
                return in_use_exit(&store);
            }
            match snapshot::restore_snapshot_to(dir, store.clone(), progress).await {
                Ok(report) => {
                    print_json(&json!({ "restored": report }));
                    EXIT_OK
                }
                Err(e) => fail(e),
            }
        }
        Command::Verify { target, no_compare, with_thumbs } => {
            if target.is_dir() {
                let check = match tokio::task::spawn_blocking({
                    let (t, p) = (target.clone(), progress.clone());
                    move || snapshot::verify_snapshot_dir(&t, &p)
                })
                .await
                {
                    Ok(Ok(c)) => c,
                    Ok(Err(e)) => return fail(e),
                    Err(e) => return fail(e),
                };
                let deep = match surql::digest_snapshot_copy(&target, &progress).await {
                    Ok(m) => json!({ "opens": true, "total_records": m.total_records(), "tables": m.counts() }),
                    Err(e) => json!({ "opens": false, "error": e.to_string() }),
                };
                let ok = check.ok && deep["opens"] == json!(true);
                print_json(&json!({ "snapshot": target, "integrity": check, "deep": deep, "ok": ok }));
                return if ok { EXIT_OK } else { EXIT_FAILED };
            }
            if !target.is_file() {
                return fail(format!("file not found: {}", target.display()));
            }
            let opts = VerifyOptions { include_thumb_blobs: with_thumbs, ..Default::default() };
            let report = if no_compare || !store_has_data(&store) {
                surql::verify_export(&target, VerifyTarget::FileOnly, &opts, &progress).await
            } else {
                if store_in_use(&store) {
                    return in_use_exit(&store);
                }
                let opened = match surql::open_store(&store).await {
                    Ok(o) => o,
                    Err(e) => return fail(e),
                };
                let report = surql::verify_export(&target, VerifyTarget::Live(&opened.db), &opts, &progress).await;
                surql::close_store(opened, &store).await;
                report
            };
            match report {
                Ok(r) => {
                    let ok = r.ok;
                    print_json(&serde_json::to_value(&r).unwrap_or(Value::Null));
                    if ok { EXIT_OK } else { EXIT_FAILED }
                }
                Err(e) => fail(e),
            }
        }
    }
}

/// Handle `--db-*` command line flags; Some(exit code) when one matched, None to start the GUI normally.
pub fn run(args: &[String]) -> Option<i32> {
    let parsed = match parse(args)? {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}\n{}", usage());
            return Some(EXIT_USAGE);
        }
    };
    let config = simplelog::ConfigBuilder::new().add_filter_allow_str("smart_media").add_filter_allow_str("surrealdb").add_filter_allow_str("surrealkv").build();
    let _ = simplelog::TermLogger::init(log::LevelFilter::Info, config, simplelog::TerminalMode::Stderr, simplelog::ColorChoice::Never);
    let worker = std::thread::Builder::new().name("db-cli".into()).spawn(move || {
        match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
            Ok(rt) => rt.block_on(run_command(parsed)),
            Err(e) => fail(format!("cannot start async runtime: {e}")),
        }
    });
    Some(match worker.map(|h| h.join()) {
        Ok(Ok(code)) => code,
        Ok(Err(_)) => fail("database command panicked"),
        Err(e) => fail(format!("cannot start worker thread: {e}")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn no_flags_means_gui() {
        assert!(parse(&args(&[])).is_none());
        assert!(parse(&args(&["--something-else", "x"])).is_none());
        assert_eq!(run(&args(&["--other"])), None);
    }

    #[test]
    fn parses_each_command() {
        let p = parse(&args(&["--db-export", "a.surql"])).unwrap().unwrap();
        assert_eq!(p.command, Command::Export { file: "a.surql".into(), with_thumbs: false });
        let p = parse(&args(&["--db-export", "a.surql", "--with-thumbs"])).unwrap().unwrap();
        assert_eq!(p.command, Command::Export { file: "a.surql".into(), with_thumbs: true });
        let p = parse(&args(&["--db-thumbs", "finish", "--clear-undecodable"])).unwrap().unwrap();
        assert_eq!(p.command, Command::Thumbs { action: ThumbsAction::Finish, clear_undecodable: true });
        assert!(parse(&args(&["--db-thumbs", "explode"])).unwrap().is_err());
        let p = parse(&args(&["--db-import=b.surql", "--db-path", "d", "--force"])).unwrap().unwrap();
        assert_eq!(p.command, Command::Import { file: "b.surql".into(), force: true });
        assert_eq!(p.db_path, Some("d".into()));
        let p = parse(&args(&["--db-verify", "c.surql", "--no-compare"])).unwrap().unwrap();
        assert_eq!(p.command, Command::Verify { target: "c.surql".into(), no_compare: true, with_thumbs: false });
        let p = parse(&args(&["--db-verify", "c.surql", "--with-thumbs"])).unwrap().unwrap();
        assert_eq!(p.command, Command::Verify { target: "c.surql".into(), no_compare: false, with_thumbs: true });
        assert_eq!(parse(&args(&["--db-info"])).unwrap().unwrap().command, Command::Info);
        assert_eq!(parse(&args(&["--db-snapshot"])).unwrap().unwrap().command, Command::Snapshot);
        assert_eq!(parse(&args(&["--db-restore", "snap"])).unwrap().unwrap().command, Command::Restore("snap".into()));
    }

    #[test]
    fn rejects_bad_usage() {
        assert!(parse(&args(&["--db-export"])).unwrap().is_err());
        assert!(parse(&args(&["--db-export", "--db-info"])).unwrap().is_err());
        assert!(parse(&args(&["--db-info", "--db-snapshot"])).unwrap().is_err());
        assert!(parse(&args(&["--db-info", "--db-path"])).unwrap().is_err());
        assert_eq!(run(&args(&["--db-export"])), Some(EXIT_USAGE));
    }
}
