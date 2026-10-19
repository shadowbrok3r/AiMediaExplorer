use std::collections::BTreeMap;
use std::future::IntoFuture;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use surrealdb::Surreal;
use surrealdb::engine::local::{Db, Mem, SurrealKv};
use surrealdb::types::{RecordId, SurrealValue, ToSql};

use super::snapshot::{self, PreOpen, ProgressFn, human, log_event, store_has_data};
use super::thumb_blobs;
use super::{DB_NAME, NS, define_schema};

const ID_PAGE: usize = 50_000;
const DEFAULT_MEM_LIMIT: u64 = 768 * 1024 * 1024;
const SHUTDOWN_WAIT: Duration = Duration::from_secs(120);

#[derive(Clone, Debug, Serialize)]
pub struct ExportReport {
    pub path: PathBuf,
    pub bytes: u64,
    pub millis: u128,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ExportOptions {
    pub include_thumb_blobs: bool,
}

/// Await an export future while reporting the growth of `watch`.
async fn drive_export<F>(fut: F, watch: &Path, progress: &ProgressFn) -> anyhow::Result<()>
where
    F: IntoFuture<Output = surrealdb::Result<()>>,
{
    let fut = fut.into_future();
    tokio::pin!(fut);
    let result = loop {
        tokio::select! {
            r = &mut fut => break r,
            _ = tokio::time::sleep(Duration::from_millis(250)) => {
                if let Ok(md) = tokio::fs::metadata(watch).await {
                    progress(md.len(), 0, "Exporting records");
                }
            }
        }
    };
    Ok(result?)
}

async fn table_names(db: &Surreal<Db>) -> anyhow::Result<Vec<String>> {
    let info = info_json(db, "INFO FOR DB").await?;
    Ok(sorted_keys(info.get("tables")))
}

/// Export every table except `thumb_blobs`, then append that table's definition without records.
async fn export_without_blobs(db: &Surreal<Db>, partial: &Path, progress: &ProgressFn) -> anyhow::Result<()> {
    let all = table_names(db).await?;
    let has_blobs = all.iter().any(|t| t == thumb_blobs::TABLE);
    let others: Vec<String> = all.into_iter().filter(|t| t != thumb_blobs::TABLE).collect();
    drive_export(db.export(partial).with_config().tables(others), partial, progress).await?;
    if !has_blobs {
        return Ok(());
    }
    append_table_definition(db, partial, thumb_blobs::TABLE).await
}

/// Append the definition of `table`, without its records, to the export at `partial`.
async fn append_table_definition(db: &Surreal<Db>, partial: &Path, table: &str) -> anyhow::Result<()> {
    let mut schema_path = partial.as_os_str().to_owned();
    schema_path.push(".schema");
    let schema_path = PathBuf::from(schema_path);
    let result = async {
        db.export(&schema_path)
            .with_config()
            .database_definition(false)
            .users(false)
            .accesses(false)
            .params(false)
            .functions(false)
            .analyzers(false)
            .apis(false)
            .buckets(false)
            .modules(false)
            .configs(false)
            .sequences(false)
            .records(false)
            .tables(vec![table.to_string()])
            .await?;
        let schema = String::from_utf8_lossy(&tokio::fs::read(&schema_path).await?).into_owned();
        let section = schema.find("-- TABLE: ").map(|i| schema[..i].rfind("-- ------------------------------").unwrap_or(i)).map(|start| &schema[start..]);
        if let Some(section) = section {
            let mut main = tokio::fs::OpenOptions::new().append(true).open(partial).await?;
            tokio::io::AsyncWriteExt::write_all(&mut main, b"\n").await?;
            tokio::io::AsyncWriteExt::write_all(&mut main, section.as_bytes()).await?;
            main.sync_all().await?;
        }
        anyhow::Ok(())
    }
    .await;
    let _ = tokio::fs::remove_file(&schema_path).await;
    result
}

/// Export the NS/DB to `path` without thumbnail blob records.
pub async fn export_atomic(db: &Surreal<Db>, path: &Path, progress: &ProgressFn) -> anyhow::Result<ExportReport> {
    export_atomic_with(db, path, progress, ExportOptions::default()).await
}

/// Export the NS/DB to `path` via a `.partial` file that is renamed on success.
pub async fn export_atomic_with(db: &Surreal<Db>, path: &Path, progress: &ProgressFn, opts: ExportOptions) -> anyhow::Result<ExportReport> {
    export_atomic_via(path, progress, |partial| async move {
        if opts.include_thumb_blobs {
            drive_export(db.export(&partial), &partial, progress).await
        } else {
            export_without_blobs(db, &partial, progress).await
        }
    })
    .await
}

/// Export the NS/DB like [`export_atomic_with`], with the API keys left out of the settings records.
pub async fn export_atomic_redacted(db: &Surreal<Db>, path: &Path, progress: &ProgressFn, opts: ExportOptions) -> anyhow::Result<ExportReport> {
    export_atomic_via(path, progress, |partial| async move { export_redacted(db, &partial, opts, progress).await }).await
}

/// Every table but the settings, then the settings records without their credential fields.
async fn export_redacted(db: &Surreal<Db>, partial: &Path, opts: ExportOptions, progress: &ProgressFn) -> anyhow::Result<()> {
    let settings = crate::database::USER_SETTINGS;
    let all = table_names(db).await?;
    let others: Vec<String> = all.iter().filter(|t| t.as_str() != settings && (opts.include_thumb_blobs || t.as_str() != thumb_blobs::TABLE)).cloned().collect();
    drive_export(db.export(partial).with_config().tables(others), partial, progress).await?;
    if !opts.include_thumb_blobs && all.iter().any(|t| t == thumb_blobs::TABLE) {
        append_table_definition(db, partial, thumb_blobs::TABLE).await?;
    }
    if !all.iter().any(|t| t == settings) {
        return Ok(());
    }
    append_table_definition(db, partial, settings).await?;
    let omit: Vec<&str> = crate::database::settings::PERSISTED_FIELDS.iter().copied().filter(|f| crate::settings::is_secret_field(f)).collect();
    let rows: Vec<surrealdb::types::Value> = db.query(format!("SELECT * OMIT {} FROM {settings}", omit.join(", "))).await?.take(0)?;
    let records: Vec<String> = rows.iter().map(|r| r.to_sql()).collect();
    let text = format!("\n-- ------------------------------\n-- TABLE DATA: {settings} (API keys left out)\n-- ------------------------------\n\nINSERT [ {} ];\n", records.join(", "));
    let mut main = tokio::fs::OpenOptions::new().append(true).open(partial).await?;
    tokio::io::AsyncWriteExt::write_all(&mut main, text.as_bytes()).await?;
    main.sync_all().await?;
    Ok(())
}

/// Runs `write` against a `.partial` file next to `path` and renames it over `path` once it holds data.
async fn export_atomic_via<F, Fut>(path: &Path, progress: &ProgressFn, write: F) -> anyhow::Result<ExportReport>
where
    F: FnOnce(PathBuf) -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<()>>,
{
    let started = Instant::now();
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        tokio::fs::create_dir_all(parent).await?;
    }
    let mut partial = path.as_os_str().to_owned();
    partial.push(".partial");
    let partial = PathBuf::from(partial);
    let result = write(partial.clone()).await;
    if let Err(e) = result {
        let _ = tokio::fs::remove_file(&partial).await;
        return Err(e);
    }
    let bytes = tokio::fs::metadata(&partial).await.map(|m| m.len()).unwrap_or(0);
    if bytes == 0 {
        let _ = tokio::fs::remove_file(&partial).await;
        anyhow::bail!("export produced an empty file");
    }
    if let Ok(f) = tokio::fs::File::open(&partial).await {
        let _ = f.sync_all().await;
    }
    tokio::fs::rename(&partial, path).await?;
    progress(bytes, bytes, "Export written");
    Ok(ExportReport { path: path.to_path_buf(), bytes, millis: started.elapsed().as_millis() })
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TableDigest {
    pub count: u64,
    pub id_checksum: String,
    pub fields: Vec<String>,
    pub indexes: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Manifest {
    pub tables: BTreeMap<String, TableDigest>,
}

impl Manifest {
    pub fn total_records(&self) -> u64 {
        self.tables.values().map(|t| t.count).sum()
    }

    pub fn counts(&self) -> BTreeMap<String, u64> {
        self.tables.iter().map(|(k, v)| (k.clone(), v.count)).collect()
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct TableCheck {
    pub table: String,
    pub expected: Option<TableDigest>,
    pub actual: Option<TableDigest>,
    pub ok: bool,
    pub problems: Vec<String>,
}

fn diff_list(label: &str, expected: &[String], actual: &[String], problems: &mut Vec<String>) {
    let missing: Vec<&String> = expected.iter().filter(|e| !actual.contains(e)).collect();
    let extra: Vec<&String> = actual.iter().filter(|a| !expected.contains(a)).collect();
    if !missing.is_empty() {
        problems.push(format!("{label} missing from file: {}", missing.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" | ")));
    }
    if !extra.is_empty() {
        problems.push(format!("{label} only in file: {}", extra.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" | ")));
    }
}

/// Compare an expected manifest (live DB or saved) with the manifest of an imported file.
pub fn compare_manifests(expected: &Manifest, actual: &Manifest) -> Vec<TableCheck> {
    let mut names: Vec<&String> = expected.tables.keys().chain(actual.tables.keys()).collect();
    names.sort();
    names.dedup();
    names
        .into_iter()
        .map(|name| {
            let (e, a) = (expected.tables.get(name), actual.tables.get(name));
            let mut problems = Vec::new();
            match (e, a) {
                (Some(e), Some(a)) => {
                    if e.count != a.count {
                        problems.push(format!("record count: expected {}, file has {}", e.count, a.count));
                    }
                    if e.id_checksum != a.id_checksum {
                        problems.push("record id checksum differs".into());
                    }
                    diff_list("fields", &e.fields, &a.fields, &mut problems);
                    diff_list("indexes", &e.indexes, &a.indexes, &mut problems);
                }
                (Some(_), None) => problems.push("table missing from file".into()),
                (None, Some(_)) => problems.push("table only in file".into()),
                (None, None) => {}
            }
            TableCheck { table: name.clone(), expected: e.cloned(), actual: a.cloned(), ok: problems.is_empty(), problems }
        })
        .collect()
}

fn quote_ident(name: &str) -> String {
    format!("`{}`", name.replace('`', "\\`"))
}

async fn info_json(db: &Surreal<Db>, query: &str) -> anyhow::Result<serde_json::Value> {
    let mut resp = db.query(query).await?;
    let v: surrealdb::types::Value = resp.take(0)?;
    Ok(v.into_json_value())
}

fn sorted_keys(v: Option<&serde_json::Value>) -> Vec<String> {
    let mut keys: Vec<String> = v.and_then(|v| v.as_object()).map(|o| o.keys().cloned().collect()).unwrap_or_default();
    keys.sort();
    keys
}

fn sorted_values(v: Option<&serde_json::Value>) -> Vec<String> {
    let mut vals: Vec<String> = v
        .and_then(|v| v.as_object())
        .map(|o| o.values().map(|x| x.as_str().map(str::to_string).unwrap_or_else(|| x.to_string())).collect())
        .unwrap_or_default();
    vals.sort();
    vals
}

fn id_hash(id: &RecordId, sum: &mut u64, xor: &mut u64) {
    let h = blake3::hash(id.to_sql().as_bytes());
    let b = h.as_bytes();
    *sum = sum.wrapping_add(u64::from_le_bytes(b[0..8].try_into().unwrap()));
    *xor ^= u64::from_le_bytes(b[8..16].try_into().unwrap());
}

/// Count, order-independent id checksum, field names and index definitions of one table.
pub async fn table_digest(db: &Surreal<Db>, table: &str) -> anyhow::Result<TableDigest> {
    let (mut sum, mut xor, mut count) = (0u64, 0u64, 0u64);
    let mut start = 0usize;
    loop {
        let ids: Vec<RecordId> = db
            .query("SELECT VALUE id FROM type::table($t) LIMIT $lim START $start")
            .bind(("t", table.to_string()))
            .bind(("lim", ID_PAGE as i64))
            .bind(("start", start as i64))
            .await?
            .take(0)?;
        for id in &ids {
            id_hash(id, &mut sum, &mut xor);
        }
        count += ids.len() as u64;
        if ids.len() < ID_PAGE {
            break;
        }
        start += ID_PAGE;
    }
    let info = info_json(db, &format!("INFO FOR TABLE {}", quote_ident(table))).await?;
    Ok(TableDigest {
        count,
        id_checksum: format!("{sum:016x}{xor:016x}"),
        fields: sorted_keys(info.get("fields")),
        indexes: sorted_values(info.get("indexes")),
    })
}

/// Digest every table of the selected NS/DB.
pub async fn build_manifest(db: &Surreal<Db>, progress: &ProgressFn) -> anyhow::Result<Manifest> {
    build_manifest_with(db, progress, true).await
}

/// Digest the tables of the selected NS/DB, leaving out `thumb_blobs` unless `include_thumb_blobs`.
pub async fn build_manifest_with(db: &Surreal<Db>, progress: &ProgressFn, include_thumb_blobs: bool) -> anyhow::Result<Manifest> {
    let info = info_json(db, "INFO FOR DB").await?;
    let tables: Vec<String> = sorted_keys(info.get("tables")).into_iter().filter(|t| include_thumb_blobs || t != thumb_blobs::TABLE).collect();
    let mut manifest = Manifest::default();
    let total = tables.len() as u64;
    for (i, t) in tables.iter().enumerate() {
        progress(i as u64, total, &format!("Reading table {t}"));
        manifest.tables.insert(t.clone(), table_digest(db, t).await?);
    }
    progress(total, total, "Tables read");
    Ok(manifest)
}

pub enum VerifyTarget<'a> {
    FileOnly,
    Manifest(&'a Manifest),
    Live(&'a Surreal<Db>),
}

#[derive(Clone, Debug)]
pub struct VerifyOptions {
    pub mem_limit_bytes: u64,
    pub scratch_dir: Option<PathBuf>,
    pub include_thumb_blobs: bool,
}

impl Default for VerifyOptions {
    fn default() -> Self {
        Self { mem_limit_bytes: DEFAULT_MEM_LIMIT, scratch_dir: None, include_thumb_blobs: false }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct VerifyReport {
    pub file: PathBuf,
    pub file_bytes: u64,
    pub backend: String,
    pub import_millis: u128,
    pub import_error: Option<String>,
    pub imported: Manifest,
    pub expected: Option<Manifest>,
    pub tables: Vec<TableCheck>,
    pub hnsw_defined: bool,
    pub live_changed_during_check: bool,
    pub ok: bool,
    pub problems: Vec<String>,
}

enum Scratch {
    Mem(Surreal<Db>),
    Disk(Surreal<Db>, tempfile::TempDir),
}

impl Scratch {
    fn db(&self) -> &Surreal<Db> {
        match self {
            Self::Mem(d) | Self::Disk(d, _) => d,
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Self::Mem(_) => "kv-mem",
            Self::Disk(..) => "surrealkv-temp",
        }
    }
}

async fn open_scratch(file_bytes: u64, opts: &VerifyOptions) -> anyhow::Result<Scratch> {
    let db = if file_bytes > opts.mem_limit_bytes {
        let tmp = match &opts.scratch_dir {
            Some(d) => {
                std::fs::create_dir_all(d)?;
                tempfile::Builder::new().prefix("verify-").tempdir_in(d)?
            }
            None => tempfile::Builder::new().prefix("smart-media-verify-").tempdir()?,
        };
        let path = tmp.path().join("store");
        let db = Surreal::new::<SurrealKv>(path.to_string_lossy().as_ref()).await?;
        Scratch::Disk(db, tmp)
    } else {
        Scratch::Mem(Surreal::new::<Mem>(()).await?)
    };
    db.db().use_ns(NS).use_db(DB_NAME).await?;
    Ok(db)
}

/// Import a .surql file into a throwaway store and compare it with the live DB or a manifest.
pub async fn verify_export(path: &Path, target: VerifyTarget<'_>, opts: &VerifyOptions, progress: &ProgressFn) -> anyhow::Result<VerifyReport> {
    let file_bytes = tokio::fs::metadata(path).await.map_err(|e| anyhow::anyhow!("cannot read {}: {e}", path.display()))?.len();
    log_event(log::Level::Info, format!("Verify: importing {} ({}) into a scratch store", path.display(), human(file_bytes)));
    let scratch = open_scratch(file_bytes, opts).await?;
    let started = Instant::now();
    progress(0, 0, "Importing file into scratch store");
    let import = scratch.db().import(path).await;
    let import_millis = started.elapsed().as_millis();

    let mut report = VerifyReport {
        file: path.to_path_buf(),
        file_bytes,
        backend: scratch.label().to_string(),
        import_millis,
        import_error: None,
        imported: Manifest::default(),
        expected: None,
        tables: Vec::new(),
        hnsw_defined: false,
        live_changed_during_check: false,
        ok: false,
        problems: Vec::new(),
    };
    if let Err(e) = import {
        report.import_error = Some(e.to_string());
        report.problems.push(format!("import failed: {e}"));
        log_event(log::Level::Error, format!("Verify: {} failed to import: {e}", path.display()));
        return Ok(report);
    }
    let with_blobs = opts.include_thumb_blobs;
    report.imported = build_manifest_with(scratch.db(), progress, with_blobs).await?;
    report.hnsw_defined = report.imported.tables.values().any(|t| t.indexes.iter().any(|d| d.contains("HNSW")));

    let expected = match target {
        VerifyTarget::FileOnly => None,
        VerifyTarget::Manifest(m) => {
            let mut m = m.clone();
            if !with_blobs {
                m.tables.remove(thumb_blobs::TABLE);
            }
            Some(m)
        }
        VerifyTarget::Live(live) => {
            progress(0, 0, "Reading live database");
            let first = build_manifest_with(live, progress, with_blobs).await?;
            let second = build_manifest_with(live, progress, with_blobs).await?;
            report.live_changed_during_check = first != second;
            if report.live_changed_during_check {
                report.problems.push("the live database changed while it was being read; differences may be expected".into());
            }
            Some(second)
        }
    };
    if let Some(expected) = &expected {
        report.tables = compare_manifests(expected, &report.imported);
        for t in report.tables.iter().filter(|t| !t.ok) {
            report.problems.push(format!("{}: {}", t.table, t.problems.join("; ")));
        }
    }
    report.ok = report.import_error.is_none() && report.tables.iter().all(|t| t.ok);
    report.expected = expected;
    log_event(
        if report.ok { log::Level::Info } else { log::Level::Warn },
        format!(
            "Verify {}: {} ({} records in {} tables, {} backend, import {:.1}s)",
            path.display(),
            if report.ok { "OK" } else { "PROBLEMS FOUND" },
            report.imported.total_records(),
            report.imported.tables.len(),
            report.backend,
            import_millis as f32 / 1000.0
        ),
    );
    Ok(report)
}

#[derive(Clone, Debug, Serialize)]
pub struct NewStoreReport {
    pub dir: PathBuf,
    pub import_millis: u128,
    pub manifest: Manifest,
    pub total_records: u64,
}

/// Create a fresh SurrealKV store at `new_dir` on the running engine and load a .surql file into it.
pub async fn import_into_new_store(surql: &Path, new_dir: &Path, progress: &ProgressFn) -> anyhow::Result<NewStoreReport> {
    anyhow::ensure!(surql.is_file(), "import file not found: {}", surql.display());
    anyhow::ensure!(!store_has_data(new_dir), "target folder {} already contains a database", new_dir.display());
    std::fs::create_dir_all(new_dir)?;
    log_event(log::Level::Info, format!("Importing {} into new store {}", surql.display(), new_dir.display()));
    let db = Surreal::new::<SurrealKv>(new_dir.to_string_lossy().as_ref()).await?;
    db.use_ns(NS).use_db(DB_NAME).await?;
    let started = Instant::now();
    progress(0, 0, "Importing records");
    db.import(surql).await?;
    define_schema(&db).await?;
    let import_millis = started.elapsed().as_millis();
    let manifest = build_manifest(&db, progress).await?;
    drop(db);
    snapshot::wait_released(new_dir, SHUTDOWN_WAIT).await;
    let total_records = manifest.total_records();
    log_event(log::Level::Info, format!("New store ready at {}: {total_records} records in {:.1}s", new_dir.display(), import_millis as f32 / 1000.0));
    Ok(NewStoreReport { dir: new_dir.to_path_buf(), import_millis, manifest, total_records })
}

pub struct OpenedStore {
    pub db: Surreal<Db>,
    pub pre: PreOpen,
}

/// Open a store on a private handle with the same pre-open snapshot and stamp handling as `database::new`.
pub async fn open_store(store: &Path) -> anyhow::Result<OpenedStore> {
    let pre = snapshot::prepare_open(store).await?;
    let db = Surreal::new::<SurrealKv>(store.to_string_lossy().as_ref()).await?;
    db.use_ns(NS).use_db(DB_NAME).await?;
    define_schema(&db).await?;
    super::collections::ensure_schema(&db).await?;
    snapshot::finish_open(store);
    Ok(OpenedStore { db, pre })
}

/// Open a copy of a directory snapshot in a temp folder and digest it; the original is never opened.
pub async fn digest_snapshot_copy(snapshot_dir: &Path, progress: &ProgressFn) -> anyhow::Result<Manifest> {
    let tmp = tempfile::Builder::new().prefix("smart-media-deepverify-").tempdir()?;
    let copy = tmp.path().join("store");
    let src = snapshot_dir.to_path_buf();
    let dst = copy.clone();
    tokio::task::spawn_blocking(move || snapshot::copy_tree(&src, &dst, &[snapshot::LOCK_FILE, snapshot::MANIFEST_FILE], 1, &mut |_, _| {}))
        .await??;
    let db = Surreal::new::<SurrealKv>(copy.to_string_lossy().as_ref()).await?;
    db.use_ns(NS).use_db(DB_NAME).await?;
    let manifest = build_manifest(&db, progress).await;
    drop(db);
    snapshot::wait_released(&copy, SHUTDOWN_WAIT).await;
    manifest
}

/// Drop a private store handle and wait for the engine to flush and release the store lock.
pub async fn close_store(opened: OpenedStore, store: &Path) {
    drop(opened);
    snapshot::wait_released(store, SHUTDOWN_WAIT).await;
}

#[derive(Debug, Deserialize, SurrealValue)]
struct CountRow {
    count: i64,
}

/// Row count of one table on the given handle.
pub async fn count_rows(db: &Surreal<Db>, table: &str) -> anyhow::Result<u64> {
    let rows: Vec<CountRow> = db
        .query("SELECT count() AS count FROM type::table($t) GROUP ALL")
        .bind(("t", table.to_string()))
        .await?
        .take(0)?;
    Ok(rows.first().map(|r| r.count as u64).unwrap_or(0))
}
