//! Imports legacy SurrealDB 2.x libraries via a pinned 2.x CLI export, streaming conversion and a verified 3.x import.

pub mod convert;
pub mod engine;
pub mod import;
pub mod job;
pub mod pathmap;
pub mod sq;
pub mod suggest;
pub mod verify;
#[cfg(test)]
mod tests;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use serde::Serialize;
use surrealdb::Surreal;
use surrealdb::engine::local::{Db, SurrealKv};

use self::convert::{Analysis, ConvertOptions, ConvertReport};
use self::engine::{CopyStats, Engine, EngineOptions, ExportStats, StoreKind};
use self::import::ImportStats;
use self::job::{Job, PhaseId, PhaseView, is_cancelled};
use self::pathmap::{Config, Policy, default_config_path};
use self::suggest::{Evidence, Suggestion};
use self::verify::{VerifyReport, VerifySpec};
use super::collections_migration::{self, Migrator as CollectionsMigrator};
use super::snapshot::{self, human, store_has_data, store_in_use};
use super::surql;
use super::thumb_migration::{self, CopyOptions, CopyReport, FinishOptions, FinishReport, MigrationState, Migrator};
use super::{DB, DB_NAME, NS, define_schema};

pub const REPORT_VERSION: u32 = 1;
pub const EXIT_ENGINE: i32 = 4;
pub const EXIT_VERIFY: i32 = 5;
pub const EXIT_REVIEW: i32 = 6;
pub const EXIT_CANCELLED: i32 = 7;
pub const WORK_DIR_NAME: &str = "legacy-import-work";
const EXPORT_NAME: &str = "legacy-export.surql";
const CONVERTED_DIR: &str = "converted";
const COPY_DIR: &str = "old-copy";
const SHUTDOWN_WAIT: std::time::Duration = std::time::Duration::from_secs(120);

/// Where converted rows go.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Target {
    /// An existing store folder, opened by the import (the app must be closed).
    Merge(PathBuf),
    /// The store the running app has open.
    Live,
    /// A new store folder created by the import.
    NewStore(PathBuf),
}

impl Target {
    pub fn store_path(&self) -> PathBuf {
        match self {
            Target::Merge(p) | Target::NewStore(p) => p.clone(),
            Target::Live => PathBuf::from(super::get_db_path()),
        }
    }

    pub fn mode_label(&self) -> &'static str {
        match self {
            Target::Merge(_) => "merge",
            Target::Live => "merge-live",
            Target::NewStore(_) => "new-store",
        }
    }

    fn merges(&self) -> bool {
        !matches!(self, Target::NewStore(_))
    }
}

#[derive(Clone, Debug)]
pub struct ImportRequest {
    pub old_store: PathBuf,
    pub target: Target,
    pub path_map: Option<PathBuf>,
    pub accept_suggested: bool,
    pub missing: Option<Policy>,
    pub unmapped: Option<Policy>,
    pub settings_fields: Vec<String>,
    pub drop_b64: bool,
    pub dry_run: bool,
    pub report: Option<PathBuf>,
    pub work_dir: Option<PathBuf>,
    pub surreal_bin: Option<PathBuf>,
    pub keep_export: bool,
    pub then_migrate_thumbs: bool,
    pub then_migrate_collections: bool,
    pub old_ns: Option<String>,
    pub old_db: Option<String>,
    pub from_export: Option<PathBuf>,
    pub skip_snapshot: bool,
}

impl ImportRequest {
    pub fn new(old_store: impl Into<PathBuf>, target: Target) -> Self {
        Self {
            old_store: old_store.into(),
            target,
            path_map: None,
            accept_suggested: false,
            missing: None,
            unmapped: None,
            settings_fields: Vec::new(),
            drop_b64: false,
            dry_run: false,
            report: None,
            work_dir: None,
            surreal_bin: None,
            keep_export: false,
            then_migrate_thumbs: false,
            then_migrate_collections: false,
            old_ns: None,
            old_db: None,
            from_export: None,
            skip_snapshot: false,
        }
    }

    pub fn work_dir_or_default(&self) -> PathBuf {
        self.work_dir.clone().unwrap_or_else(|| snapshot::db_parent(&self.target.store_path()).join(WORK_DIR_NAME))
    }
}

#[derive(Debug)]
pub enum LegacyError {
    Usage(String),
    InUse(PathBuf),
    Engine(String),
    MapReview(PathBuf),
    Cancelled,
    VerifyFailed,
    Failed(String),
}

impl LegacyError {
    pub fn exit_code(&self) -> i32 {
        match self {
            LegacyError::Usage(_) => super::cli::EXIT_USAGE,
            LegacyError::InUse(_) => super::cli::EXIT_IN_USE,
            LegacyError::Engine(_) => EXIT_ENGINE,
            LegacyError::VerifyFailed => EXIT_VERIFY,
            LegacyError::MapReview(_) => EXIT_REVIEW,
            LegacyError::Cancelled => EXIT_CANCELLED,
            LegacyError::Failed(_) => super::cli::EXIT_FAILED,
        }
    }

    pub fn message(&self) -> String {
        match self {
            LegacyError::Usage(m) | LegacyError::Engine(m) | LegacyError::Failed(m) => m.clone(),
            LegacyError::InUse(p) => format!("the database at {} is in use by another process; close the app and retry", p.display()),
            LegacyError::MapReview(p) => format!("a suggested path map was written to {}; review or edit it, then run again (the file is used as it stands; --accept-suggested-map skips this stop)", p.display()),
            LegacyError::Cancelled => "cancelled".into(),
            LegacyError::VerifyFailed => "the import finished but verification found problems; see the report".into(),
        }
    }
}

impl From<anyhow::Error> for LegacyError {
    fn from(e: anyhow::Error) -> Self {
        if is_cancelled(&e) { LegacyError::Cancelled } else { LegacyError::Failed(format!("{e:#}")) }
    }
}

impl From<tokio::task::JoinError> for LegacyError {
    fn from(e: tokio::task::JoinError) -> Self {
        LegacyError::Failed(format!("background task failed: {e}"))
    }
}

impl From<std::io::Error> for LegacyError {
    fn from(e: std::io::Error) -> Self {
        LegacyError::Failed(e.to_string())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct EngineReport {
    pub version: String,
    pub source: String,
    pub checksum_verified: bool,
    pub path: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ExportInfo {
    pub bytes: u64,
    pub millis: u128,
    pub kept_at: Option<String>,
}

/// Result of the copy, export and analysis steps; the export file is the input of every later step.
#[derive(Clone, Debug)]
pub struct Prepared {
    pub work: PathBuf,
    pub export: PathBuf,
    pub export_owned: bool,
    pub ns: String,
    pub db: String,
    pub analysis: Analysis,
    pub hints: Vec<String>,
    pub old_store: PathBuf,
    pub engine: Option<EngineReport>,
    pub copy: Option<CopyStats>,
    pub export_info: Option<ExportInfo>,
    pub store_kind: Option<StoreKind>,
    pub namespaces: Vec<engine::NsInfo>,
}

#[derive(Clone, Debug, Serialize)]
pub struct MapReport {
    pub file: String,
    pub generated: bool,
    pub rules: usize,
    pub warnings: Vec<String>,
    pub unresolved: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct MigrateReport {
    pub copy: CopyReport,
    pub finish: FinishReport,
}

#[derive(Clone, Debug, Serialize)]
pub struct CollectionsMigrateReport {
    pub copy: collections_migration::CopyReport,
    pub finish: collections_migration::FinishReport,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct LegacyReport {
    pub version: u32,
    pub ok: bool,
    pub dry_run: bool,
    pub mode: String,
    pub started: String,
    pub elapsed_ms: u128,
    pub old_store: String,
    pub target: String,
    pub engine: Option<EngineReport>,
    pub namespace: Option<(String, String)>,
    pub copy: Option<CopyStats>,
    pub export: Option<ExportInfo>,
    pub analysis_tables: std::collections::BTreeMap<String, u64>,
    pub path_map: Option<MapReport>,
    pub convert: Option<ConvertReport>,
    pub import: Option<ImportStats>,
    pub verify: Option<VerifyReport>,
    pub snapshot: Option<String>,
    pub thumb_migration_state_reset: bool,
    pub migrate: Option<MigrateReport>,
    pub collections: Option<CollectionsMigrateReport>,
    pub phases: Vec<PhaseView>,
    pub warnings: Vec<String>,
    pub next_steps: Vec<String>,
    pub peak_rss_mb: Option<u64>,
    pub error: Option<String>,
}

pub struct Outcome {
    pub report: LegacyReport,
    pub result: Result<(), LegacyError>,
}

fn peak_rss_mb() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = s.lines().find(|l| l.starts_with("VmHWM:"))?;
    line.split_whitespace().nth(1)?.parse::<u64>().ok().map(|kb| kb / 1024)
}

fn ensure_private_dir(dir: &Path) -> Result<(), LegacyError> {
    if !dir.exists() {
        let mut b = std::fs::DirBuilder::new();
        b.recursive(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            b.mode(0o700);
        }
        b.create(dir).map_err(|e| LegacyError::Failed(format!("cannot create {}: {e}", dir.display())))?;
    }
    Ok(())
}

fn remove_dir_quiet(dir: &Path) {
    if dir.exists() {
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// Checks the request against obviously wrong combinations before any work starts.
pub fn validate(req: &ImportRequest) -> Result<(), LegacyError> {
    if req.then_migrate_thumbs && (req.drop_b64 || req.dry_run) {
        return Err(LegacyError::Usage("--then-migrate-thumbs needs the inline thumbnails (not --drop-b64) and cannot be combined with --dry-run".into()));
    }
    if req.then_migrate_thumbs && req.target == Target::Live {
        return Err(LegacyError::Usage("run the thumbnail migration from the Thumbnail storage window when importing into the open database".into()));
    }
    if req.then_migrate_collections && req.dry_run {
        return Err(LegacyError::Usage("--then-migrate-collections cannot be combined with --dry-run".into()));
    }
    if req.then_migrate_collections && req.target == Target::Live {
        return Err(LegacyError::Usage("run the collection migration from the Collections window when importing into the open database".into()));
    }
    if req.from_export.is_none() && !req.old_store.is_dir() {
        return Err(LegacyError::Usage(format!("old store folder not found: {}", req.old_store.display())));
    }
    if let Some(e) = &req.from_export {
        if !e.is_file() {
            return Err(LegacyError::Usage(format!("export file not found: {}", e.display())));
        }
    }
    let store = req.target.store_path();
    if req.from_export.is_none() {
        let a = req.old_store.canonicalize().unwrap_or_else(|_| req.old_store.clone());
        let b = store.canonicalize().unwrap_or_else(|_| store.clone());
        if a == b || b.starts_with(&a) {
            return Err(LegacyError::Usage("the target store is the old store itself".into()));
        }
    }
    if let Target::NewStore(dir) = &req.target {
        if !req.dry_run && store_has_data(dir) {
            return Err(LegacyError::Usage(format!("{} already contains a database; choose an empty folder or use --merge", dir.display())));
        }
    }
    Ok(())
}

async fn analyze_blocking(export: PathBuf, job: &Arc<Job>) -> Result<Analysis, LegacyError> {
    let j = Arc::clone(job);
    Ok(tokio::task::spawn_blocking(move || convert::analyze(&export, &j)).await??)
}

/// Copies the old store, exports it with the 2.x CLI and analyses the export (or analyses a given export).
pub async fn prepare(req: &ImportRequest, job: &Arc<Job>) -> Result<Prepared, LegacyError> {
    let work = engine::absolute(&req.work_dir_or_default());
    ensure_private_dir(&work)?;
    let ns_given = req.old_ns.clone();
    let db_given = req.old_db.clone();
    let (ns, db) = (ns_given.clone().unwrap_or_else(|| NS.to_string()), db_given.clone().unwrap_or_else(|| DB_NAME.to_string()));
    if let Some(export) = &req.from_export {
        for p in [PhaseId::Engine, PhaseId::Copy, PhaseId::Export] {
            job.skip(p, "using an existing export");
        }
        let analysis = analyze_blocking(export.clone(), job).await?;
        let hints = Vec::new();
        return Ok(Prepared { work, export: export.clone(), export_owned: false, ns, db, analysis, hints, old_store: req.old_store.clone(), engine: None, copy: None, export_info: None, store_kind: None, namespaces: Vec::new() });
    }
    let old = req.old_store.clone();
    let kind = engine::detect_kind(&old);
    match kind {
        StoreKind::SurrealKvLegacy | StoreKind::RocksDb => {}
        StoreKind::SurrealKvCurrent => return Err(LegacyError::Usage(format!("{} already uses the SurrealDB 3.x layout; point the app at it or use --db-import with an export instead", old.display()))),
        StoreKind::Unknown => return Err(LegacyError::Usage(format!("{} does not look like a SurrealKV or RocksDB store folder", old.display()))),
    }
    let hints = {
        let o = old.clone();
        tokio::task::spawn_blocking(move || suggest::manifest_paths(&o)).await?
    };
    let eng = engine::ensure_engine(&EngineOptions { explicit: req.surreal_bin.clone(), allow_download: true, cache_dir: None }, job).await.map_err(|e| if is_cancelled(&e) { LegacyError::Cancelled } else { LegacyError::Engine(format!("{e:#}")) })?;
    let engine_report = EngineReport { version: eng.version.clone(), source: eng.source.label().to_string(), checksum_verified: eng.checksum_ok, path: eng.path.display().to_string() };
    let size = {
        let o = old.clone();
        tokio::task::spawn_blocking(move || engine::store_bytes(&o)).await?
    };
    let needed = snapshot::required_space(size) + size / 2;
    snapshot::check_free_space(needed, &work, &snapshot::available_space).map_err(|e| LegacyError::Failed(e.user_message()))?;
    job.note(format!("Old store {} ({}), {}", old.display(), human(size), kind.label()));
    let copy_dir = work.join(COPY_DIR).join(snapshot::store_name(&old));
    let copy_stats = {
        let (o, c, j) = (old.clone(), copy_dir.clone(), Arc::clone(job));
        let r = tokio::task::spawn_blocking(move || engine::copy_store_blocking(&o, &c, &j)).await;
        let r = r.map_err(LegacyError::from).and_then(|r| r.map_err(LegacyError::from));
        if r.is_err() {
            remove_dir_quiet(&work.join(COPY_DIR));
        }
        r?
    };
    job.finish(PhaseId::Copy);
    let export = work.join(EXPORT_NAME);
    let res = export_and_analyze(&eng, kind, &copy_dir, &export, size, (ns, db), ns_given.is_some() || db_given.is_some(), job).await;
    remove_dir_quiet(&work.join(COPY_DIR));
    let _ = std::fs::remove_file(work.join("history.txt"));
    let (ns, db, analysis, stats, namespaces) = res?;
    Ok(Prepared {
        work,
        export: export.clone(),
        export_owned: true,
        ns,
        db,
        analysis,
        hints,
        old_store: old,
        engine: Some(engine_report),
        copy: Some(copy_stats),
        export_info: Some(ExportInfo { bytes: stats.bytes, millis: stats.millis, kept_at: req.keep_export.then(|| export.display().to_string()) }),
        store_kind: Some(kind),
        namespaces,
    })
}

type ExportOutcome = (String, String, Analysis, ExportStats, Vec<engine::NsInfo>);

#[allow(clippy::too_many_arguments)]
async fn export_and_analyze(eng: &Engine, kind: StoreKind, copy: &Path, export: &Path, size: u64, nsdb: (String, String), explicit: bool, job: &Arc<Job>) -> Result<ExportOutcome, LegacyError> {
    let (mut ns, mut db) = nsdb;
    let est = size / 7;
    let mut stats = engine::export(eng, kind, copy, &ns, &db, export, est, job).await?;
    job.finish(PhaseId::Export);
    let mut analysis = analyze_blocking(export.to_path_buf(), job).await?;
    let mut namespaces = Vec::new();
    if analysis.tables.get("thumbnails").copied().unwrap_or(0) == 0 {
        job.note(format!("Namespace {ns} / database {db} holds no thumbnails; looking at the other namespaces"));
        namespaces = engine::detect_namespaces(eng, kind, copy, export.parent().unwrap_or(copy), job).await?;
        match engine::pick_namespace(&namespaces) {
            Ok((n, d)) if (n.as_str(), d.as_str()) != (ns.as_str(), db.as_str()) && !explicit => {
                job.note(format!("Using namespace {n} / database {d}"));
                ns = n;
                db = d;
                stats = engine::export(eng, kind, copy, &ns, &db, export, est, job).await?;
                job.finish(PhaseId::Export);
                analysis = analyze_blocking(export.to_path_buf(), job).await?;
            }
            Ok(_) => {}
            Err(cands) => {
                let _ = std::fs::remove_file(export);
                let list = cands.iter().map(|(n, d)| format!("{n}/{d}")).collect::<Vec<_>>().join(", ");
                return Err(LegacyError::Usage(format!("no single database with a thumbnails table was found (candidates: {list}); choose one with --old-ns and --old-db")));
            }
        }
    }
    if analysis.tables.get("thumbnails").copied().unwrap_or(0) == 0 && analysis.tables.get("clip_embeddings").copied().unwrap_or(0) == 0 {
        let _ = std::fs::remove_file(export);
        let found = namespaces.iter().flat_map(|n| n.dbs.iter().map(move |d| format!("{}/{}", n.ns, d.name))).collect::<Vec<_>>().join(", ");
        let hint = if found.is_empty() { String::new() } else { format!(" (databases in the old store: {found})") };
        return Err(LegacyError::Failed(format!("the export of {ns}/{db} contains no thumbnails or embeddings{hint}")));
    }
    Ok((ns, db, analysis, stats, namespaces))
}

#[derive(Clone, Debug)]
pub struct MapChoice {
    pub config: Config,
    pub file: PathBuf,
    pub generated: bool,
    pub suggestion: Option<Suggestion>,
}

/// Loads the requested or default path map, or builds and saves a suggestion when none exists.
pub fn resolve_map(explicit: Option<&Path>, prep: &Prepared, target_store: &Path) -> Result<MapChoice, LegacyError> {
    if let Some(p) = explicit {
        if !p.is_file() {
            return Err(LegacyError::Usage(format!("path map not found: {}", p.display())));
        }
        let config = Config::load(p).map_err(|e| LegacyError::Usage(format!("{e:#}")))?;
        return Ok(MapChoice { config, file: p.to_path_buf(), generated: false, suggestion: None });
    }
    let file = default_config_path(target_store);
    if file.is_file() {
        let config = Config::load(&file).map_err(|e| LegacyError::Usage(format!("{e:#}")))?;
        return Ok(MapChoice { config, file, generated: false, suggestion: None });
    }
    let s = suggest_map(prep);
    if let Some(parent) = file.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(&file, suggest::render_toml(&s)).map_err(|e| LegacyError::Failed(format!("cannot write {}: {e}", file.display())))?;
    Ok(MapChoice { config: s.config.clone(), file, generated: true, suggestion: Some(s) })
}

/// Suggested path map for the analysed export on this machine.
pub fn suggest_map(prep: &Prepared) -> Suggestion {
    let volumes = suggest::list_volumes();
    let mut hints = prep.hints.clone();
    hints.extend(prep.analysis.settings_paths.iter().cloned());
    let evidence = Evidence {
        hints,
        old_store_volume: suggest::volume_of(&prep.old_store, &volumes).map(|v| v.mount.clone()),
        home: directories::UserDirs::new().map(|u| u.home_dir().to_path_buf()),
    };
    suggest::suggest(&prep.analysis, &volumes, &evidence)
}

fn apply_overrides(cfg: &mut Config, req: &ImportRequest) {
    if let Some(p) = req.missing {
        cfg.options.missing = p;
    }
    if let Some(p) = req.unmapped {
        cfg.options.unmapped = p;
    }
}

/// Dry run: reports how every stored path would map, writing nothing.
pub async fn dry_run(prep: &Prepared, cfg: &Config, req: &ImportRequest, job: &Arc<Job>) -> Result<ConvertReport, LegacyError> {
    let mut opts = ConvertOptions::new(&prep.export);
    opts.drop_b64 = req.drop_b64;
    opts.settings_fields = req.settings_fields.clone();
    opts.merge = req.target.merges();
    let (cfg, j) = (cfg.clone(), Arc::clone(job));
    Ok(tokio::task::spawn_blocking(move || convert::convert(&opts, &cfg, None, &j)).await??)
}

struct StageOut {
    convert: ConvertReport,
    import: ImportStats,
    verify: VerifyReport,
}

enum SnapPlan {
    Covered(PathBuf),
    Skip(&'static str),
    Take { store: PathBuf, live: bool },
}

async fn take_pre_import_snapshot(store: &Path, live: bool, job: &Arc<Job>) -> Result<PathBuf, LegacyError> {
    job.begin(PhaseId::Snapshot, 0, "snapshotting the target store");
    let progress = job.progress_fn(PhaseId::Snapshot);
    let info = if live { snapshot::snapshot_now_with(progress).await } else { snapshot::create_snapshot(store.to_path_buf(), "pre", None, progress).await };
    let info = info.map_err(|e| LegacyError::Failed(e.user_message()))?;
    job.finish(PhaseId::Snapshot);
    Ok(info.path)
}

#[allow(clippy::too_many_arguments)]
async fn import_stage(req: &ImportRequest, prep: &Prepared, cfg: &Config, db: &Surreal<Db>, merge: bool, snap: SnapPlan, job: &Arc<Job>, rep: &mut LegacyReport) -> Result<StageOut, LegacyError> {
    let key = convert::SETTINGS_KEY;
    let before = import::table_counts(db).await?;
    let digest = if merge { verify::settings_digest(db, key).await } else { None };
    let existing = if merge { Some(import::load_existing(db, key, job).await?) } else { None };
    let converted = prep.work.join(CONVERTED_DIR);
    remove_dir_quiet(&converted);
    let mut opts = ConvertOptions::new(&prep.export);
    opts.out_dir = Some(converted.clone());
    opts.drop_b64 = req.drop_b64;
    opts.merge = merge;
    opts.settings_fields = req.settings_fields.clone();
    let conv = {
        let (o, c, j) = (opts.clone(), cfg.clone(), Arc::clone(job));
        tokio::task::spawn_blocking(move || convert::convert(&o, &c, existing.as_ref(), &j)).await?
    };
    let conv = match conv {
        Ok(c) => c,
        Err(e) => {
            remove_dir_quiet(&converted);
            return Err(e.into());
        }
    };
    job.note(format!("Converted: {} thumbnails, {} embeddings, {} groups ready to import", conv.rows_out("thumbnails"), conv.rows_out("clip_embeddings"), conv.rows_out("logical_groups")));
    let snapped = match snap {
        SnapPlan::Covered(p) => {
            job.skip(PhaseId::Snapshot, "the pre-open snapshot already covers this");
            Ok(Some(p))
        }
        SnapPlan::Skip(why) => {
            job.skip(PhaseId::Snapshot, why);
            Ok(None)
        }
        SnapPlan::Take { .. } if conv.files.is_empty() => {
            job.skip(PhaseId::Snapshot, "nothing to import");
            Ok(None)
        }
        SnapPlan::Take { store, live } => take_pre_import_snapshot(&store, live, job).await.map(Some),
    };
    let snapped = match snapped {
        Ok(p) => p,
        Err(e) => {
            remove_dir_quiet(&converted);
            return Err(e);
        }
    };
    rep.snapshot = snapped.map(|p| p.display().to_string());
    let imp = import::import_files(db, &converted, &conv.files, job).await;
    remove_dir_quiet(&converted);
    let imp = imp?;
    let mut idem_opts = opts.clone();
    idem_opts.out_dir = None;
    idem_opts.merge = true;
    let spec = VerifySpec { before, settings_digest_before: digest, knn_samples: 3, idempotent: Some((idem_opts, cfg.clone())), settings_key: key.to_string(), live: req.target == Target::Live };
    let ver = verify::verify_import(db, &spec, &conv, job).await?;
    Ok(StageOut { convert: conv, import: imp, verify: ver })
}

fn reset_thumb_state_if_needed(store: &Path, conv: &ConvertReport, drop_b64: bool, job: &Job) -> bool {
    if drop_b64 || conv.rows_out("thumbnails") == 0 {
        return false;
    }
    let st = MigrationState::load(store);
    if st.started.is_none() && !st.is_finished() {
        return false;
    }
    thumb_migration::reset_state(store);
    job.note("Thumbnail migration state reset: the imported rows carry inline thumbnails again");
    true
}

async fn migrate_thumbs(db: &Surreal<Db>, store: &Path, require_snapshot: bool, job: &Arc<Job>) -> Result<MigrateReport, LegacyError> {
    job.begin(PhaseId::Thumbs, 0, "copying inline thumbnails into blobs");
    let m = Migrator::new(db.clone(), store.to_path_buf());
    let reporter = {
        let (m, j) = (Arc::clone(&m), Arc::clone(job));
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                let p = m.progress();
                j.update(PhaseId::Thumbs, p.processed, p.rows_total.unwrap_or(0).max(p.processed), p.phase.label());
                if j.is_cancel_requested() {
                    m.cancel();
                }
            }
        })
    };
    let copy = m.run_copy(CopyOptions { require_snapshot, ..Default::default() }).await;
    let finish = match &copy {
        Ok(c) if !c.cancelled => Some(m.run_finish(FinishOptions { require_snapshot, ..Default::default() }).await),
        _ => None,
    };
    reporter.abort();
    let copy = copy?;
    if copy.cancelled {
        return Err(LegacyError::Cancelled);
    }
    let finish = finish.ok_or_else(|| LegacyError::Failed("thumbnail finish did not run".into()))??;
    if !finish.ok {
        job.fail(PhaseId::Thumbs, &finish.summary);
        return Err(LegacyError::Failed(format!("thumbnail migration did not finish cleanly: {}", finish.summary)));
    }
    job.finish(PhaseId::Thumbs);
    Ok(MigrateReport { copy, finish })
}

async fn migrate_collections(db: &Surreal<Db>, store: &Path, require_snapshot: bool, job: &Arc<Job>) -> Result<CollectionsMigrateReport, LegacyError> {
    job.note("Migrating logical groups to collections");
    let m = CollectionsMigrator::new(db.clone(), store.to_path_buf());
    let watcher = {
        let (m, j) = (Arc::clone(&m), Arc::clone(job));
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                if j.is_cancel_requested() {
                    m.cancel();
                }
            }
        })
    };
    let copy = m.run_copy(collections_migration::CopyOptions { require_snapshot, ..Default::default() }).await;
    let finish = match &copy {
        Ok(c) if !c.cancelled => Some(m.run_finish(collections_migration::FinishOptions { require_snapshot, repair: true }).await),
        _ => None,
    };
    watcher.abort();
    let copy = copy.map_err(|e| LegacyError::Failed(format!("collection migration failed: {e:#}")))?;
    if copy.cancelled {
        return Err(LegacyError::Cancelled);
    }
    let finish = finish.ok_or_else(|| LegacyError::Failed("collection finish did not run".into()))?.map_err(|e| LegacyError::Failed(format!("collection migration failed: {e:#}")))?;
    if !finish.ok {
        return Err(LegacyError::Failed(format!("collection migration did not finish cleanly: {}", finish.summary)));
    }
    job.note(format!("Collections: {} created, {} memberships added", copy.collections_created, copy.memberships_added));
    Ok(CollectionsMigrateReport { copy, finish })
}

fn next_steps(req: &ImportRequest, rep: &LegacyReport) -> Vec<String> {
    let mut v = Vec::new();
    if req.dry_run {
        v.push("Review the mapping table and the path map, then run the import without --dry-run.".into());
        return v;
    }
    if rep.import.is_none() {
        return v;
    }
    if rep.migrate.is_none() && !req.drop_b64 && rep.convert.as_ref().is_some_and(|c| c.rows_out("thumbnails") > 0) {
        v.push("The path map is applied. Next run `smart-media --db-thumbs migrate` and then `--db-thumbs finish` (thumbnail blobs are keyed by the stored path, so do not change the path map afterwards).".into());
    }
    if rep.collections.is_none() && rep.convert.as_ref().is_some_and(|c| c.rows_out("logical_groups") > 0) {
        v.push("Imported logical groups become collections with `smart-media --db-collections migrate` followed by `--db-collections finish`.".into());
    }
    if let Target::NewStore(dir) = &req.target {
        v.push(format!("Start the app on the new store: smart-media --db-path {} (or write that folder to ./db/path.txt).", dir.display()));
    }
    if rep.snapshot.is_some() {
        v.push("A snapshot taken before the import is the rollback copy (restore it from Database maintenance or `--db-restore`).".into());
    }
    if rep.export.as_ref().is_some_and(|e| e.kept_at.is_some()) {
        v.push("The kept export file contains the legacy settings including any API keys; delete it when you no longer need it.".into());
    }
    v
}

fn merge_snap_plan(req: &ImportRequest, store: &Path, pre: Option<PathBuf>, rep: &mut LegacyReport) -> SnapPlan {
    if let Some(p) = pre {
        SnapPlan::Covered(p)
    } else if req.skip_snapshot {
        rep.warnings.push("no snapshot was taken before importing (--no-snapshot)".into());
        SnapPlan::Skip("skipped by request")
    } else {
        SnapPlan::Take { store: store.to_path_buf(), live: req.target == Target::Live }
    }
}

async fn merge_into_open(req: &ImportRequest, prep: &Prepared, cfg: &Config, db: &Surreal<Db>, store: &Path, pre: Option<PathBuf>, job: &Arc<Job>, rep: &mut LegacyReport) -> Result<StageOut, LegacyError> {
    let plan = merge_snap_plan(req, store, pre, rep);
    let out = import_stage(req, prep, cfg, db, true, plan, job, rep).await?;
    if reset_thumb_state_if_needed(store, &out.convert, req.drop_b64, job) {
        rep.thumb_migration_state_reset = true;
    }
    if out.convert.rows_out("thumbnails") > 0 || out.convert.rows_out("logical_groups") > 0 {
        if let Err(e) = collections_migration::note_legacy_import(db).await {
            log::warn!("[collections] cannot flag the migration as stale: {e:#}");
        }
    }
    Ok(out)
}

fn record_stage(rep: &mut LegacyReport, stage: &Result<StageOut, LegacyError>) {
    if let Ok(s) = stage {
        rep.convert = Some(s.convert.clone());
        rep.import = Some(s.import.clone());
        rep.verify = Some(s.verify.clone());
    }
}

/// Imports a prepared export into the requested target and verifies the result.
pub async fn import_prepared(req: &ImportRequest, prep: &Prepared, cfg: &Config, job: &Arc<Job>, rep: &mut LegacyReport) -> Result<(), LegacyError> {
    let stage = match &req.target {
        Target::Merge(store) => {
            if !store_has_data(store) {
                return Err(LegacyError::Usage(format!("no database found at {}; use --into-new-store to create one", store.display())));
            }
            if store_in_use(store) {
                return Err(LegacyError::InUse(store.clone()));
            }
            let opened = surql::open_store(store).await.map_err(|e| LegacyError::Failed(format!("{e:#}")))?;
            let db = opened.db.clone();
            let out = merge_into_open(req, prep, cfg, &db, store, opened.pre.snapshot.as_ref().map(|s| s.path.clone()), job, rep).await;
            record_stage(rep, &out);
            let migrated = match (&out, req.then_migrate_thumbs) {
                (Ok(s), true) if s.verify.ok => Some(migrate_thumbs(&db, store, true, job).await),
                _ => {
                    job.skip(PhaseId::Thumbs, "not requested");
                    None
                }
            };
            let collections = match (&out, req.then_migrate_collections) {
                (Ok(s), true) if s.verify.ok => Some(migrate_collections(&db, store, true, job).await),
                _ => None,
            };
            snapshot::finish_open(store);
            drop(db);
            surql::close_store(opened, store).await;
            if let Some(m) = migrated {
                rep.migrate = Some(m?);
            }
            if let Some(c) = collections {
                rep.collections = Some(c?);
            }
            out?
        }
        Target::Live => {
            if !super::is_connected() {
                return Err(LegacyError::Failed("the database is not connected in this process".into()));
            }
            job.skip(PhaseId::Thumbs, "run it from the Thumbnail storage window");
            let out = merge_into_open(req, prep, cfg, &DB, &req.target.store_path(), None, job, rep).await;
            record_stage(rep, &out);
            out?
        }
        Target::NewStore(dir) => {
            std::fs::create_dir_all(dir)?;
            let needed = prep.export_info.as_ref().map_or(0, |e| e.bytes) * 3 / 2;
            snapshot::check_free_space(needed, dir, &snapshot::available_space).map_err(|e| LegacyError::Failed(e.user_message()))?;
            let db = Surreal::new::<SurrealKv>(dir.to_string_lossy().as_ref()).await.map_err(|e| LegacyError::Failed(e.to_string()))?;
            db.use_ns(NS).use_db(DB_NAME).await.map_err(|e| LegacyError::Failed(e.to_string()))?;
            define_schema(&db).await?;
            let out = import_stage(req, prep, cfg, &db, false, SnapPlan::Skip("new store"), job, rep).await;
            record_stage(rep, &out);
            let migrated = match (&out, req.then_migrate_thumbs) {
                (Ok(s), true) if s.verify.ok => Some(migrate_thumbs(&db, dir, false, job).await),
                _ => {
                    job.skip(PhaseId::Thumbs, "not requested");
                    None
                }
            };
            let collections = match (&out, req.then_migrate_collections) {
                (Ok(s), true) if s.verify.ok => Some(migrate_collections(&db, dir, false, job).await),
                _ => None,
            };
            drop(db);
            snapshot::wait_released(dir, SHUTDOWN_WAIT).await;
            if out.is_ok() {
                let stamp = snapshot::stamp_for(dir, snapshot::engine_versions(), chrono::Local::now());
                snapshot::write_stamp(dir, &stamp)?;
            }
            if let Some(m) = migrated {
                rep.migrate = Some(m?);
            }
            if let Some(c) = collections {
                rep.collections = Some(c?);
            }
            out?
        }
    };
    if !stage.verify.ok {
        return Err(LegacyError::VerifyFailed);
    }
    Ok(())
}

fn cleanup(req: &ImportRequest, prep: &Prepared) {
    remove_dir_quiet(&prep.work.join(CONVERTED_DIR));
    remove_dir_quiet(&prep.work.join(COPY_DIR));
    let _ = std::fs::remove_file(prep.work.join("history.txt"));
    if prep.export_owned && !req.keep_export {
        let _ = std::fs::remove_file(&prep.export);
    }
}

fn finish_report(rep: &mut LegacyReport, req: &ImportRequest, job: &Job, started: Instant, result: &Result<(), LegacyError>) {
    rep.elapsed_ms = started.elapsed().as_millis();
    rep.phases = job.views();
    rep.peak_rss_mb = peak_rss_mb();
    rep.ok = result.is_ok();
    rep.error = result.as_ref().err().map(|e| e.message());
    rep.next_steps = if result.is_ok() { next_steps(req, rep) } else { Vec::new() };
    if let Some(c) = &rep.convert {
        rep.warnings.extend(c.warnings.iter().cloned());
    }
}

fn new_report(req: &ImportRequest) -> LegacyReport {
    LegacyReport {
        version: REPORT_VERSION,
        dry_run: req.dry_run,
        mode: req.target.mode_label().to_string(),
        started: chrono::Local::now().to_rfc3339(),
        old_store: req.old_store.display().to_string(),
        target: req.target.store_path().display().to_string(),
        ..Default::default()
    }
}

fn absorb_prepared(rep: &mut LegacyReport, prep: &Prepared) {
    rep.engine = prep.engine.clone();
    rep.copy = prep.copy.clone();
    rep.export = prep.export_info.clone();
    rep.namespace = Some((prep.ns.clone(), prep.db.clone()));
    rep.analysis_tables = prep.analysis.tables.clone();
}

fn write_report(req: &ImportRequest, rep: &LegacyReport) {
    if let Some(p) = &req.report {
        if let Ok(text) = serde_json::to_string_pretty(rep) {
            let _ = std::fs::write(p, text);
        }
    }
}

/// Runs the whole pipeline for `req`; the report is returned even when the run fails.
pub async fn run(req: ImportRequest, job: Arc<Job>) -> Outcome {
    let started = Instant::now();
    let mut rep = new_report(&req);
    let result = run_inner(&req, &job, &mut rep).await;
    finish_report(&mut rep, &req, &job, started, &result);
    write_report(&req, &rep);
    Outcome { report: rep, result }
}

/// Imports (or dry-runs) an already prepared export with an explicit path map.
pub async fn run_prepared(req: ImportRequest, prep: Prepared, cfg: Config, job: Arc<Job>) -> Outcome {
    let started = Instant::now();
    let mut rep = new_report(&req);
    absorb_prepared(&mut rep, &prep);
    let mut cfg = cfg;
    apply_overrides(&mut cfg, &req);
    for p in [PhaseId::Engine, PhaseId::Copy, PhaseId::Export] {
        if job.view(p).state == job::PhaseState::Pending {
            job.skip(p, "prepared earlier");
        }
    }
    let result = match validate(&req) {
        Ok(()) if req.dry_run => dry_run(&prep, &cfg, &req, &job).await.map(|c| rep.convert = Some(c)),
        Ok(()) => import_prepared(&req, &prep, &cfg, &job, &mut rep).await,
        Err(e) => Err(e),
    };
    if !req.dry_run {
        cleanup(&req, &prep);
    }
    finish_report(&mut rep, &req, &job, started, &result);
    write_report(&req, &rep);
    Outcome { report: rep, result }
}

/// Removes the export and scratch folders of a prepared run.
pub fn discard_prepared(prep: &Prepared) {
    remove_dir_quiet(&prep.work.join(CONVERTED_DIR));
    remove_dir_quiet(&prep.work.join(COPY_DIR));
    if prep.export_owned {
        let _ = std::fs::remove_file(&prep.export);
    }
}

async fn run_inner(req: &ImportRequest, job: &Arc<Job>, rep: &mut LegacyReport) -> Result<(), LegacyError> {
    validate(req)?;
    let prep = prepare(req, job).await?;
    absorb_prepared(rep, &prep);
    let result = run_after_prepare(req, &prep, job, rep).await;
    cleanup(req, &prep);
    result
}

async fn run_after_prepare(req: &ImportRequest, prep: &Prepared, job: &Arc<Job>, rep: &mut LegacyReport) -> Result<(), LegacyError> {
    let store = req.target.store_path();
    let choice = {
        let (explicit, p, s) = (req.path_map.clone(), prep.clone(), store.clone());
        tokio::task::spawn_blocking(move || resolve_map(explicit.as_deref(), &p, &s)).await??
    };
    let (warnings, unresolved) = choice.suggestion.as_ref().map(|s| (s.warnings.clone(), s.unresolved.clone())).unwrap_or_default();
    rep.path_map = Some(MapReport { file: choice.file.display().to_string(), generated: choice.generated, rules: choice.config.rules.len(), warnings: warnings.clone(), unresolved });
    for w in &warnings {
        job.note(format!("Path map: {w}"));
    }
    let mut cfg = choice.config.clone();
    apply_overrides(&mut cfg, req);
    if choice.generated && !req.accept_suggested && !req.dry_run {
        rep.convert = Some(dry_run(prep, &cfg, req, job).await?);
        return Err(LegacyError::MapReview(choice.file));
    }
    if req.dry_run {
        for p in [PhaseId::Snapshot, PhaseId::Import, PhaseId::Verify, PhaseId::Thumbs] {
            job.skip(p, "dry run");
        }
        rep.convert = Some(dry_run(prep, &cfg, req, job).await?);
        return Ok(());
    }
    import_prepared(req, prep, &cfg, job, rep).await
}
