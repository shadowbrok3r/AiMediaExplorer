use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use super::snapshot::{self, ProgressFn, human, store_has_data, store_in_use};
use super::surql::{self, VerifyOptions, VerifyTarget};

pub const EXIT_OK: i32 = 0;
pub const EXIT_FAILED: i32 = 1;
pub const EXIT_USAGE: i32 = 2;
pub const EXIT_IN_USE: i32 = 3;

#[derive(Debug, PartialEq, Eq)]
enum Command {
    Snapshot,
    Export(PathBuf),
    Import { file: PathBuf, force: bool },
    Verify { target: PathBuf, no_compare: bool },
    Restore(PathBuf),
    Info,
}

#[derive(Debug, PartialEq, Eq)]
struct Parsed {
    command: Command,
    db_path: Option<PathBuf>,
}

const DB_FLAGS: [&str; 6] = ["--db-snapshot", "--db-export", "--db-import", "--db-verify", "--db-restore", "--db-info"];

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
            "--db-export" => Some(value(&mut i, inline, flag).map(|v| Command::Export(PathBuf::from(v)))),
            "--db-import" => Some(value(&mut i, inline, flag).map(|v| Command::Import { file: PathBuf::from(v), force: false })),
            "--db-verify" => Some(value(&mut i, inline, flag).map(|v| Command::Verify { target: PathBuf::from(v), no_compare: false })),
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
            _ => None,
        };
        match next {
            Some(Ok(c)) => {
                if command.is_some() {
                    return Some(Err("only one of --db-snapshot, --db-export, --db-import, --db-verify, --db-restore, --db-info may be given".into()));
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
        Command::Verify { no_compare: n, .. } => *n = no_compare,
        _ => {}
    }
    Some(Ok(Parsed { command, db_path }))
}

fn usage() -> &'static str {
    "database flags:\n  --db-info                         print engine, stamp and snapshots as JSON (does not open the store)\n  --db-snapshot                     copy the store folder into <db>/snapshots (store must not be in use)\n  --db-export <file.surql>          export the database to a .surql file\n  --db-import <file.surql> [--force]  import into a new or empty store (--force to import into a populated one)\n  --db-verify <file.surql|dir> [--no-compare]  import into a scratch store and compare with the store (dir: deep-verify a snapshot copy)\n  --db-restore <snapshot dir>       replace the store with a snapshot (old store kept as <store>.replaced-<ts>; store must not be in use)
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
        Command::Export(file) => {
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
            let result = surql::export_atomic(&opened.db, &file, &progress).await;
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
        Command::Verify { target, no_compare } => {
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
            let opts = VerifyOptions::default();
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
        assert_eq!(p.command, Command::Export("a.surql".into()));
        let p = parse(&args(&["--db-import=b.surql", "--db-path", "d", "--force"])).unwrap().unwrap();
        assert_eq!(p.command, Command::Import { file: "b.surql".into(), force: true });
        assert_eq!(p.db_path, Some("d".into()));
        let p = parse(&args(&["--db-verify", "c.surql", "--no-compare"])).unwrap().unwrap();
        assert_eq!(p.command, Command::Verify { target: "c.surql".into(), no_compare: true });
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
