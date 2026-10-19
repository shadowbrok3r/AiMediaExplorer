use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use super::legacy::job::{Job, PhaseState};
use super::legacy::pathmap::Policy;
use super::legacy::{self, ImportRequest, Target};
use super::snapshot::{self, ProgressFn, human, store_has_data, store_in_use};
use super::surql::{self, ExportOptions, VerifyOptions, VerifyTarget};
use super::collections_migration::{self, Migrator as CollectionsMigrator};
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CollectionsAction {
    Status,
    Migrate,
    Finish,
    Rollback,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct LegacyArgs {
    old_store: PathBuf,
    path_map: Option<PathBuf>,
    merge: bool,
    into_new_store: Option<PathBuf>,
    missing: Option<Policy>,
    unmapped: Option<Policy>,
    settings: Option<String>,
    drop_b64: bool,
    dry_run: bool,
    report: Option<PathBuf>,
    work_dir: Option<PathBuf>,
    surreal_bin: Option<PathBuf>,
    keep_export: bool,
    then_migrate_thumbs: bool,
    then_migrate_collections: bool,
    old_ns: Option<String>,
    old_db: Option<String>,
    from_export: Option<PathBuf>,
    no_snapshot: bool,
    accept_suggested_map: bool,
}

impl LegacyArgs {
    fn set_flag(&mut self, flag: &str) -> bool {
        match flag {
            "--merge" => self.merge = true,
            "--drop-b64" => self.drop_b64 = true,
            "--dry-run" => self.dry_run = true,
            "--keep-export" => self.keep_export = true,
            "--then-migrate-thumbs" => self.then_migrate_thumbs = true,
            "--then-migrate-collections" => self.then_migrate_collections = true,
            "--no-snapshot" => self.no_snapshot = true,
            "--accept-suggested-map" => self.accept_suggested_map = true,
            _ => return false,
        }
        true
    }

    fn set_value(&mut self, flag: &str, v: String) -> Result<bool, String> {
        let path = || PathBuf::from(&v);
        match flag {
            "--path-map" => self.path_map = Some(path()),
            "--into-new-store" => self.into_new_store = Some(path()),
            "--report" => self.report = Some(path()),
            "--work-dir" => self.work_dir = Some(path()),
            "--surreal-bin" => self.surreal_bin = Some(path()),
            "--from-export" => self.from_export = Some(path()),
            "--missing" => self.missing = Some(v.parse().map_err(|e| format!("--missing: {e}"))?),
            "--unmapped" => self.unmapped = Some(v.parse().map_err(|e| format!("--unmapped: {e}"))?),
            "--settings" => self.settings = Some(v),
            "--old-ns" => self.old_ns = Some(v),
            "--old-db" => self.old_db = Some(v),
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn settings_fields(&self) -> Result<Vec<String>, String> {
        match self.settings.as_deref() {
            None | Some("skip") => Ok(Vec::new()),
            Some(s) => match s.strip_prefix("fields:") {
                Some(list) => {
                    let names: Vec<String> = list.split(',').map(|f| f.trim().to_string()).filter(|f| !f.is_empty()).collect();
                    if names.is_empty() { Err("--settings fields: needs at least one field name".into()) } else { Ok(names) }
                }
                None => Err(format!("--settings expects skip or fields:<name>,<name>, not '{s}'")),
            },
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Command {
    LegacyImport(Box<LegacyArgs>),
    Snapshot,
    Export { file: PathBuf, with_thumbs: bool },
    Import { file: PathBuf, force: bool },
    Verify { target: PathBuf, no_compare: bool, with_thumbs: bool },
    Restore(PathBuf),
    Info,
    Thumbs { action: ThumbsAction, clear_undecodable: bool },
    Collections { action: CollectionsAction, dry_run: bool, force: bool, no_snapshot: bool, no_repair: bool },
}

#[derive(Debug, PartialEq, Eq)]
struct Parsed {
    command: Command,
    db_path: Option<PathBuf>,
}

const DB_FLAGS: [&str; 9] = ["--db-snapshot", "--db-export", "--db-import", "--db-verify", "--db-restore", "--db-info", "--db-thumbs", "--db-import-legacy", "--db-collections"];

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
    let mut no_repair = false;
    let mut legacy = LegacyArgs::default();
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
            "--db-collections" => Some(value(&mut i, inline, flag).and_then(|v| match v.as_str() {
                "status" => Ok(CollectionsAction::Status),
                "migrate" => Ok(CollectionsAction::Migrate),
                "finish" => Ok(CollectionsAction::Finish),
                "rollback" => Ok(CollectionsAction::Rollback),
                other => Err(format!("--db-collections expects status, migrate, finish or rollback, not '{other}'")),
            }).map(|action| Command::Collections { action, dry_run: false, force: false, no_snapshot: false, no_repair: false })),
            "--db-import" => Some(value(&mut i, inline, flag).map(|v| Command::Import { file: PathBuf::from(v), force: false })),
            "--db-verify" => Some(value(&mut i, inline, flag).map(|v| Command::Verify { target: PathBuf::from(v), no_compare: false, with_thumbs: false })),
            "--db-restore" => Some(value(&mut i, inline, flag).map(|v| Command::Restore(PathBuf::from(v)))),
            "--db-import-legacy" => Some(value(&mut i, inline, flag).map(|v| Command::LegacyImport(Box::new(LegacyArgs { old_store: PathBuf::from(v), ..Default::default() })))),
            "--path-map" | "--into-new-store" | "--report" | "--work-dir" | "--surreal-bin" | "--from-export" | "--missing" | "--unmapped" | "--settings" | "--old-ns" | "--old-db" => match value(&mut i, inline, flag).and_then(|v| legacy.set_value(flag, v)) {
                Ok(_) => None,
                Err(e) => return Some(Err(e)),
            },
            f if legacy.set_flag(f) => None,
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
            "--no-repair" => {
                no_repair = true;
                None
            }
            _ => None,
        };
        match next {
            Some(Ok(c)) => {
                if command.is_some() {
                    return Some(Err("only one of --db-snapshot, --db-export, --db-import, --db-import-legacy, --db-verify, --db-restore, --db-info, --db-thumbs, --db-collections may be given".into()));
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
        Command::Collections { dry_run: d, force: f, no_snapshot: n, no_repair: r, .. } => {
            *d = legacy.dry_run;
            *f = force;
            *n = legacy.no_snapshot;
            *r = no_repair;
        }
        Command::LegacyImport(a) => {
            let old_store = std::mem::take(&mut a.old_store);
            **a = LegacyArgs { old_store, ..legacy };
            if a.merge && a.into_new_store.is_some() {
                return Some(Err("--merge and --into-new-store cannot be combined".into()));
            }
            if let Err(e) = a.settings_fields() {
                return Some(Err(e));
            }
        }
        _ => {}
    }
    Some(Ok(Parsed { command, db_path }))
}

fn usage() -> &'static str {
    "database flags:\n  --db-info                         print engine, stamp and snapshots as JSON (does not open the store)\n  --db-snapshot                     copy the store folder into <db>/snapshots (store must not be in use)\n  --db-export <file.surql> [--with-thumbs]  export the database to a .surql file (thumbnail blob records are left out unless --with-thumbs)\n  --db-import <file.surql> [--force]  import into a new or empty store (--force to import into a populated one)\n  --db-verify <file.surql|dir> [--no-compare] [--with-thumbs]  import into a scratch store and compare with the store (dir: deep-verify a snapshot copy; --with-thumbs also compares thumbnail blob records)\n  --db-thumbs <info|migrate|finish|missing> [--clear-undecodable]  thumbnail blob storage: stats, copy inline thumbnails into blobs, verify and clear the inline copies, list rows without a blob\n  --db-collections <status|migrate|finish|rollback> [--dry-run] [--no-snapshot] [--no-repair] [--force]  move single-membership logical groups into collections: status of the migration, copy (dry run lists the plan), verify and mark finished (missing memberships are copied again unless --no-repair), or remove the collections again (--force discards ones made after the migration); the legacy groups are never changed
  --db-restore <snapshot dir>       replace the store with a snapshot (old store kept as <store>.replaced-<ts>; store must not be in use)
  --db-import-legacy <old store dir> [--path-map <file.toml>] [--merge | --into-new-store <dir>] [--missing keep|drop|tag] [--unmapped keep|drop|tag]
        [--settings fields:a,b] [--drop-b64] [--dry-run] [--report <file.json>] [--work-dir <dir>] [--surreal-bin <path>] [--keep-export] [--then-migrate-thumbs] [--then-migrate-collections]
        [--old-ns <ns> --old-db <db>] [--from-export <file.surql>] [--no-snapshot] [--accept-suggested-map]
                                    import a SurrealDB 2.x library (for example from Windows) via a pinned 2.x CLI run on a copy of the old store;
                                    exit codes: 0 ok, 1 failed, 2 usage, 3 in use, 4 engine unavailable, 5 verification problems, 6 path map needs review, 7 cancelled
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

fn spawn_collections_reporter(m: &Arc<CollectionsMigrator>) -> tokio::task::JoinHandle<()> {
    let m = Arc::clone(m);
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(2)).await;
            let p = m.progress();
            eprintln!("{}: {} of {} memberships, {} of {} groups, {} added", p.step.label(), p.processed, p.memberships_total, p.groups_done, p.groups_total, p.added);
        }
    })
}

async fn run_collections(store: &Path, action: CollectionsAction, dry_run: bool, force: bool, no_snapshot: bool, no_repair: bool) -> i32 {
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
        let m = CollectionsMigrator::new(opened.db.clone(), store.to_path_buf());
        match action {
            CollectionsAction::Status => match m.status().await {
                Ok(s) => {
                    print_json(&json!({ "status": s }));
                    EXIT_OK
                }
                Err(e) => fail(format!("{e:#}")),
            },
            CollectionsAction::Migrate if dry_run => match m.plan().await {
                Ok(plan) => {
                    print_json(&json!({ "dry_run": true, "plan": plan }));
                    EXIT_OK
                }
                Err(e) => fail(format!("{e:#}")),
            },
            CollectionsAction::Migrate => {
                let reporter = spawn_collections_reporter(&m);
                let res = m.run_copy(collections_migration::CopyOptions { require_snapshot: !no_snapshot, ..Default::default() }).await;
                reporter.abort();
                match res {
                    Ok(report) => {
                        print_json(&json!({ "copy": report, "state": m.state().await.ok() }));
                        if report.cancelled { EXIT_FAILED } else { EXIT_OK }
                    }
                    Err(e) => fail(format!("{e:#}")),
                }
            }
            CollectionsAction::Finish => {
                let reporter = spawn_collections_reporter(&m);
                let res = m.run_finish(collections_migration::FinishOptions { require_snapshot: !no_snapshot, repair: !no_repair }).await;
                reporter.abort();
                match res {
                    Ok(report) => {
                        print_json(&json!({ "finish": report, "state": m.state().await.ok() }));
                        if report.ok { EXIT_OK } else { EXIT_FAILED }
                    }
                    Err(e) => fail(format!("{e:#}")),
                }
            }
            CollectionsAction::Rollback => {
                let reporter = spawn_collections_reporter(&m);
                let res = m.run_rollback(collections_migration::RollbackOptions { require_snapshot: !no_snapshot, force }).await;
                reporter.abort();
                match res {
                    Ok(report) => {
                        print_json(&json!({ "rollback": report, "state": m.state().await.ok() }));
                        if report.ok { EXIT_OK } else { EXIT_FAILED }
                    }
                    Err(e) => fail(format!("{e:#}")),
                }
            }
        }
    };
    surql::close_store(opened, store).await;
    code
}

fn fmt_duration(d: Duration) -> String {
    let s = d.as_secs();
    if s >= 3600 { format!("{}h {}m", s / 3600, s % 3600 / 60) } else if s >= 60 { format!("{}m {}s", s / 60, s % 60) } else { format!("{s}s") }
}

fn spawn_job_reporter(job: &Arc<Job>) -> tokio::task::JoinHandle<()> {
    let job = Arc::clone(job);
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(2)).await;
            for v in job.views().into_iter().filter(|v| v.state == PhaseState::Running) {
                let bytes = !matches!(v.phase, legacy::job::PhaseId::Verify | legacy::job::PhaseId::Thumbs);
                let fmt = |n: u64| if bytes { human(n) } else { n.to_string() };
                let pct = v.fraction().map(|f| format!(" ({:.0}%)", f * 100.0)).unwrap_or_default();
                let eta = v.eta().map(|d| format!(", about {} left", fmt_duration(d))).unwrap_or_default();
                let of = if v.total > 0 { format!(" of {}", fmt(v.total)) } else { String::new() };
                eprintln!("[{}] {}: {}{of}{pct}{eta}", v.phase.label().to_lowercase(), v.what, fmt(v.done));
            }
        }
    })
}

fn print_mapping_table(report: &legacy::convert::ConvertReport) {
    eprintln!("path mapping ({} thumbnail rows, {:.1}% map to an existing file):", report.thumbnail_rows, report.exists_rate() * 100.0);
    eprintln!("{:<34} {:>8} {:>8} {:>8}  {}", "rule", "rows", "exists", "case-fix", "missing prefixes");
    for r in &report.path_rules {
        let rule = if r.rule.is_none() { "(no rule matched)".to_string() } else if r.to.is_empty() { format!("{} (unmapped)", r.from) } else { format!("{} -> {}", r.from, r.to) };
        let missing = r.missing_top_prefixes.iter().take(3).map(|(p, n)| format!("{p} ({n})")).collect::<Vec<_>>().join("; ");
        eprintln!("{:<34} {:>8} {:>7.1}% {:>8}  {}", rule, r.rows, r.exists_rate * 100.0, r.case_fixed_rows, missing);
    }
}

async fn run_legacy(a: LegacyArgs, store: PathBuf) -> i32 {
    let fields = match a.settings_fields() {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: {e}");
            return EXIT_USAGE;
        }
    };
    let target = match (&a.into_new_store, a.merge) {
        (Some(dir), _) => Target::NewStore(dir.clone()),
        (None, true) => Target::Merge(store.clone()),
        (None, false) if store_has_data(&store) => Target::Merge(store.clone()),
        (None, false) => Target::NewStore(store.clone()),
    };
    let mut req = ImportRequest::new(a.old_store.clone(), target);
    req.path_map = a.path_map;
    req.accept_suggested = a.accept_suggested_map;
    req.missing = a.missing;
    req.unmapped = a.unmapped;
    req.settings_fields = fields;
    req.drop_b64 = a.drop_b64;
    req.dry_run = a.dry_run;
    req.report = a.report;
    req.work_dir = a.work_dir;
    req.surreal_bin = a.surreal_bin;
    req.keep_export = a.keep_export;
    req.then_migrate_thumbs = a.then_migrate_thumbs;
    req.then_migrate_collections = a.then_migrate_collections;
    req.old_ns = a.old_ns;
    req.old_db = a.old_db;
    req.from_export = a.from_export;
    req.skip_snapshot = a.no_snapshot;
    let job = Job::new();
    let reporter = spawn_job_reporter(&job);
    let interrupt = {
        let job = Arc::clone(&job);
        tokio::spawn(async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                eprintln!("interrupted: cancelling and removing scratch files (press Ctrl-C again to quit at once)");
                job.cancel();
                if tokio::signal::ctrl_c().await.is_ok() {
                    std::process::exit(130);
                }
            }
        })
    };
    let outcome = legacy::run(req, job).await;
    reporter.abort();
    interrupt.abort();
    if let Some(c) = &outcome.report.convert {
        print_mapping_table(c);
    }
    if let Some(m) = &outcome.report.path_map {
        eprintln!("path map: {}{}", m.file, if m.generated { " (generated from this machine's volumes; review before relying on it)" } else { "" });
        for w in &m.warnings {
            eprintln!("warning: {w}");
        }
    }
    for w in &outcome.report.warnings {
        eprintln!("warning: {w}");
    }
    for n in &outcome.report.next_steps {
        eprintln!("next: {n}");
    }
    print_json(&serde_json::to_value(&outcome.report).unwrap_or(Value::Null));
    match outcome.result {
        Ok(()) => EXIT_OK,
        Err(e) => {
            eprintln!("error: {}", e.message());
            e.exit_code()
        }
    }
}

async fn run_command(parsed: Parsed) -> i32 {
    let store = parsed.db_path.unwrap_or_else(|| PathBuf::from(super::get_db_path()));
    let progress = cli_progress();
    match parsed.command {
        Command::LegacyImport(a) => run_legacy(*a, store).await,
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
        Command::Collections { action, dry_run, force, no_snapshot, no_repair } => run_collections(&store, action, dry_run, force, no_snapshot, no_repair).await,
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
    fn parses_legacy_import_flags() {
        let p = parse(&args(&[
            "--db-import-legacy", "/old/ai_search", "--path-map", "m.toml", "--into-new-store", "/new", "--missing", "tag", "--unmapped=drop", "--settings", "fields:recent_paths,excluded_dirs", "--drop-b64", "--dry-run",
            "--report", "r.json", "--work-dir", "/w", "--surreal-bin", "/b/surreal", "--keep-export", "--then-migrate-thumbs", "--then-migrate-collections", "--old-ns", "a", "--old-db", "b", "--from-export", "e.surql", "--no-snapshot", "--accept-suggested-map", "--db-path", "d",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(p.db_path, Some("d".into()));
        let Command::LegacyImport(a) = p.command else { panic!("legacy command expected") };
        assert_eq!(a.old_store, PathBuf::from("/old/ai_search"));
        assert_eq!((a.path_map.as_deref(), a.into_new_store.as_deref()), (Some(Path::new("m.toml")), Some(Path::new("/new"))));
        assert_eq!((a.missing, a.unmapped), (Some(Policy::Tag), Some(Policy::Drop)));
        assert_eq!(a.settings_fields().unwrap(), vec!["recent_paths".to_string(), "excluded_dirs".to_string()]);
        assert!(a.drop_b64 && a.dry_run && a.keep_export && a.then_migrate_thumbs && a.then_migrate_collections && a.no_snapshot && a.accept_suggested_map && !a.merge);
        assert_eq!((a.old_ns.as_deref(), a.old_db.as_deref(), a.from_export.as_deref()), (Some("a"), Some("b"), Some(Path::new("e.surql"))));
        assert_eq!((a.report.as_deref(), a.work_dir.as_deref(), a.surreal_bin.as_deref()), (Some(Path::new("r.json")), Some(Path::new("/w")), Some(Path::new("/b/surreal"))));
        let m = parse(&args(&["--db-import-legacy=/old", "--merge"])).unwrap().unwrap();
        let Command::LegacyImport(a) = m.command else { panic!("legacy command expected") };
        assert!(a.merge && a.into_new_store.is_none() && a.settings_fields().unwrap().is_empty());
    }

    #[test]
    fn parses_the_collections_command_and_its_flags() {
        let c = |a: &[&str]| match parse(&args(a)).unwrap().unwrap().command {
            Command::Collections { action, dry_run, force, no_snapshot, no_repair } => (action, dry_run, force, no_snapshot, no_repair),
            other => panic!("collections command expected, got {other:?}"),
        };
        assert_eq!(c(&["--db-collections", "status"]), (CollectionsAction::Status, false, false, false, false));
        assert_eq!(c(&["--db-collections=migrate", "--dry-run"]), (CollectionsAction::Migrate, true, false, false, false));
        assert_eq!(c(&["--db-collections", "migrate", "--no-snapshot"]), (CollectionsAction::Migrate, false, false, true, false));
        assert_eq!(c(&["--db-collections", "finish", "--no-repair", "--db-path", "d"]), (CollectionsAction::Finish, false, false, false, true));
        assert_eq!(c(&["--force", "--db-collections", "rollback"]), (CollectionsAction::Rollback, false, true, false, false));
        assert_eq!(parse(&args(&["--db-collections", "status", "--db-path", "d"])).unwrap().unwrap().db_path, Some("d".into()));
        assert!(parse(&args(&["--db-collections", "explode"])).unwrap().is_err());
        assert!(parse(&args(&["--db-collections"])).unwrap().is_err());
        assert!(parse(&args(&["--db-collections", "status", "--db-info"])).unwrap().is_err());
        assert_eq!(run(&args(&["--db-collections", "explode"])), Some(EXIT_USAGE));
    }

    #[test]
    fn rejects_bad_legacy_usage() {
        assert!(parse(&args(&["--db-import-legacy"])).unwrap().is_err());
        assert!(parse(&args(&["--db-import-legacy", "/o", "--merge", "--into-new-store", "/n"])).unwrap().is_err());
        assert!(parse(&args(&["--db-import-legacy", "/o", "--missing", "maybe"])).unwrap().is_err());
        assert!(parse(&args(&["--db-import-legacy", "/o", "--settings", "all"])).unwrap().is_err());
        assert!(parse(&args(&["--db-import-legacy", "/o", "--settings", "fields:"])).unwrap().is_err());
        assert!(parse(&args(&["--db-import-legacy", "/o", "--db-info"])).unwrap().is_err());
        assert_eq!(run(&args(&["--db-import-legacy", "/o", "--missing", "maybe"])), Some(EXIT_USAGE));
    }

    #[test]
    fn legacy_exit_codes_are_distinct() {
        use crate::database::legacy::LegacyError;
        let codes = [
            LegacyError::Usage("u".into()).exit_code(),
            LegacyError::InUse("p".into()).exit_code(),
            LegacyError::Engine("e".into()).exit_code(),
            LegacyError::VerifyFailed.exit_code(),
            LegacyError::MapReview("m".into()).exit_code(),
            LegacyError::Cancelled.exit_code(),
            LegacyError::Failed("f".into()).exit_code(),
        ];
        let mut sorted = codes.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), codes.len());
        assert!(!codes.contains(&EXIT_OK));
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
