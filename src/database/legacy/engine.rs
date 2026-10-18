//! Old-engine access: locates, downloads and verifies the pinned SurrealDB 2.x CLI and drives it as an external process.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use futures_util::StreamExt;
use serde::Serialize;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use super::job::{EngineSource, EngineStatus, Job, PhaseId};
use crate::database::snapshot::human;

pub const PINNED_VERSION: &str = "2.3.10";
pub const ENV_BIN: &str = "AME_SURREAL_BIN";
pub const ENV_DIR: &str = "AME_LEGACY_ENGINE_DIR";
const RELEASE_BASE: &str = "https://github.com/surrealdb/surrealdb/releases/download";
const VERSION_TIMEOUT: Duration = Duration::from_secs(20);
const QUERY_TIMEOUT: Duration = Duration::from_secs(900);
const COPY_BUF: usize = 4 << 20;
const STDERR_KEEP: usize = 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssetKind {
    Tgz,
    Exe,
}

#[derive(Clone, Debug)]
pub struct Asset {
    pub name: String,
    pub url: String,
    pub sha256: String,
    pub kind: AssetKind,
    pub binary_sha256: Option<String>,
}

struct Pin {
    os: &'static str,
    arch: &'static str,
    suffix: &'static str,
    kind: AssetKind,
    sha256: &'static str,
    binary_sha256: Option<&'static str>,
}

const PINS: &[Pin] = &[
    Pin {
        os: "linux",
        arch: "x86_64",
        suffix: "linux-amd64.tgz",
        kind: AssetKind::Tgz,
        sha256: "cb3ccfa9906684de8cad4e47326056617a36f4b049231d4aac53daadfce41dc2",
        binary_sha256: Some("bc916c7c775ed32aa40c33fd57bc9b477128c8ad9118bb1620696a05247bb0a7"),
    },
    Pin { os: "linux", arch: "aarch64", suffix: "linux-arm64.tgz", kind: AssetKind::Tgz, sha256: "ef0848a00ec881af4bd73d8817de79b9fbe07fe7049a88b75d53e33f6937deb4", binary_sha256: None },
    Pin { os: "macos", arch: "x86_64", suffix: "darwin-amd64.tgz", kind: AssetKind::Tgz, sha256: "3b235838d31668f7d9d303684067cadc92411a028105e3f3da6c61c41dc7e402", binary_sha256: None },
    Pin { os: "macos", arch: "aarch64", suffix: "darwin-arm64.tgz", kind: AssetKind::Tgz, sha256: "291daeebc15a98a9957182fc2e373aaabc700261d07cddf1bfe9de533342f9d4", binary_sha256: None },
    Pin { os: "windows", arch: "x86_64", suffix: "windows-amd64.exe", kind: AssetKind::Exe, sha256: "9f80f5b4c361b5274e17574822bddc7408f3a9c7b0d6a83a88a4c81f582ea9db", binary_sha256: None },
];

/// Pinned release asset for an OS and architecture as named by `std::env::consts`.
pub fn pinned_asset_for(os: &str, arch: &str) -> Option<Asset> {
    PINS.iter().find(|p| p.os == os && p.arch == arch).map(|p| {
        let name = format!("surreal-v{PINNED_VERSION}.{}", p.suffix);
        Asset {
            url: format!("{RELEASE_BASE}/v{PINNED_VERSION}/{name}"),
            name,
            sha256: p.sha256.to_string(),
            kind: p.kind,
            binary_sha256: p.binary_sha256.map(str::to_string),
        }
    })
}

pub fn pinned_asset() -> Option<Asset> {
    pinned_asset_for(std::env::consts::OS, std::env::consts::ARCH)
}

/// Absolute form of `p` without touching the file system.
pub fn absolute(p: &Path) -> PathBuf {
    std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf())
}

pub fn bin_name() -> &'static str {
    if cfg!(windows) { "surreal.exe" } else { "surreal" }
}

/// Folder holding downloaded engines, below the per-user data directory.
pub fn cache_dir() -> PathBuf {
    if let Some(d) = std::env::var_os(ENV_DIR).filter(|d| !d.is_empty()) {
        return PathBuf::from(d);
    }
    directories::ProjectDirs::from("", "", "smart-media").map(|d| d.data_local_dir().join("legacy-engine")).unwrap_or_else(|| PathBuf::from("./db/legacy-engine"))
}

pub fn cached_binary() -> PathBuf {
    cache_dir().join(PINNED_VERSION).join(bin_name())
}

#[derive(Clone, Debug)]
pub struct Engine {
    pub path: PathBuf,
    pub version: String,
    pub source: EngineSource,
    pub checksum_ok: bool,
}

#[derive(Clone, Debug, Default)]
pub struct EngineOptions {
    pub explicit: Option<PathBuf>,
    pub allow_download: bool,
    pub cache_dir: Option<PathBuf>,
}

/// Command for the old CLI with every SURREAL_* variable removed from its environment.
pub fn command(bin: &Path) -> tokio::process::Command {
    let mut c = tokio::process::Command::new(bin);
    for (k, _) in std::env::vars_os() {
        if k.to_string_lossy().to_ascii_uppercase().starts_with("SURREAL") {
            c.env_remove(&k);
        }
    }
    c.stdin(Stdio::null()).kill_on_drop(true);
    c
}

fn sha256_file(path: &Path) -> Result<String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).with_context(|| format!("cannot read {}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Version string reported by `<bin> version`; fails unless it is a 2.x CLI.
pub async fn probe_version(bin: &Path) -> Result<String> {
    let mut cmd = command(bin);
    cmd.arg("version");
    let out = tokio::time::timeout(VERSION_TIMEOUT, cmd.output()).await.map_err(|_| anyhow!("{} did not answer `version` within {:?}", bin.display(), VERSION_TIMEOUT))?.with_context(|| format!("cannot run {}", bin.display()))?;
    if !out.status.success() {
        bail!("{} version failed: {}", bin.display(), String::from_utf8_lossy(&out.stderr).trim());
    }
    parse_version(&String::from_utf8_lossy(&out.stdout))
}

pub fn parse_version(text: &str) -> Result<String> {
    let token = text.split_whitespace().next().unwrap_or("");
    let token = token.strip_prefix("surreal-").unwrap_or(token).trim_start_matches('v');
    let major: u32 = token.split('.').next().and_then(|m| m.parse().ok()).ok_or_else(|| anyhow!("unrecognised version output: {:?}", text.trim()))?;
    if major != 2 {
        bail!("found SurrealDB {token}; the legacy engine must be a 2.x CLI ({PINNED_VERSION} recommended)");
    }
    Ok(token.to_string())
}

fn verify_cached(bin: &Path, asset: Option<&Asset>) -> Result<bool> {
    let actual = sha256_file(bin)?;
    if let Some(known) = asset.and_then(|a| a.binary_sha256.as_deref()) {
        if actual != known {
            bail!("cached engine checksum {actual} does not match the pinned {known}");
        }
        return Ok(true);
    }
    let sidecar = bin.with_file_name("surreal.sha256");
    let recorded = std::fs::read_to_string(&sidecar).with_context(|| format!("missing checksum record {}", sidecar.display()))?;
    if recorded.trim() != actual {
        bail!("cached engine checksum {actual} does not match the recorded {}", recorded.trim());
    }
    Ok(true)
}

/// Finds a usable SurrealDB 2.x CLI: explicit path, cached download, then a verified download.
pub async fn ensure_engine(opts: &EngineOptions, job: &Arc<Job>) -> Result<Engine> {
    job.begin(PhaseId::Engine, 0, "locating the SurrealDB 2.x CLI");
    job.set_engine(EngineStatus::Checking);
    let res = ensure_inner(opts, job).await;
    match &res {
        Ok(e) => {
            job.set_engine(EngineStatus::Found { path: e.path.display().to_string(), version: e.version.clone(), source: e.source.clone(), checksum_ok: e.checksum_ok });
            job.finish(PhaseId::Engine);
        }
        Err(e) => {
            job.set_engine(EngineStatus::Failed(format!("{e:#}")));
            job.fail(PhaseId::Engine, &format!("{e:#}"));
        }
    }
    res
}

async fn ensure_inner(opts: &EngineOptions, job: &Arc<Job>) -> Result<Engine> {
    let asset = pinned_asset();
    let given = opts.explicit.clone().map(|p| (p, EngineSource::Explicit)).or_else(|| std::env::var_os(ENV_BIN).filter(|v| !v.is_empty()).map(|v| (PathBuf::from(v), EngineSource::Environment)));
    if let Some((path, source)) = given {
        let path = absolute(&path);
        if !path.is_file() {
            bail!("engine binary not found: {}", path.display());
        }
        let version = probe_version(&path).await?;
        let pinned_ok = asset.as_ref().and_then(|a| a.binary_sha256.clone()).is_some_and(|k| sha256_file(&path).is_ok_and(|h| h == k));
        job.note(format!("Using SurrealDB {version} CLI at {}", path.display()));
        return Ok(Engine { path, version, source, checksum_ok: pinned_ok });
    }
    let dir = opts.cache_dir.clone().unwrap_or_else(cache_dir).join(PINNED_VERSION);
    let cached = dir.join(bin_name());
    if cached.is_file() {
        match verify_cached(&cached, asset.as_ref()) {
            Ok(_) => {
                let version = probe_version(&cached).await?;
                job.note(format!("Using cached SurrealDB {version} CLI at {}", cached.display()));
                return Ok(Engine { path: cached, version, source: EngineSource::Cache, checksum_ok: true });
            }
            Err(e) => job.note(format!("Cached engine rejected ({e:#}); downloading again")),
        }
    }
    let Some(asset) = asset else {
        bail!("no pinned SurrealDB {PINNED_VERSION} release exists for {}-{}; install a 2.x CLI yourself and pass --surreal-bin", std::env::consts::OS, std::env::consts::ARCH);
    };
    if !opts.allow_download {
        bail!("no SurrealDB 2.x CLI available and downloading is disabled; pass --surreal-bin <path> or set {ENV_BIN}");
    }
    job.note(format!("Downloading {} ({PINNED_VERSION}) into {}", asset.name, dir.display()));
    download_asset(&asset, &cached, job).await?;
    let version = probe_version(&cached).await?;
    Ok(Engine { path: cached, version, source: EngineSource::Downloaded, checksum_ok: true })
}

/// Downloads `asset`, checks its SHA-256 and installs the CLI binary at `dest`.
pub async fn download_asset(asset: &Asset, dest: &Path, job: &Arc<Job>) -> Result<()> {
    let dir = dest.parent().ok_or_else(|| anyhow!("bad engine destination"))?;
    tokio::fs::create_dir_all(dir).await?;
    let partial = dir.join(format!("{}.partial", asset.name));
    let result = download_to(asset, &partial, job).await;
    if let Err(e) = result {
        let _ = tokio::fs::remove_file(&partial).await;
        return Err(e);
    }
    job.set_engine(EngineStatus::Verifying);
    let (asset2, partial2, dest2) = (asset.clone(), partial.clone(), dest.to_path_buf());
    let installed = tokio::task::spawn_blocking(move || install_binary(&asset2, &partial2, &dest2)).await?;
    let _ = tokio::fs::remove_file(&partial).await;
    installed
}

async fn download_to(asset: &Asset, partial: &Path, job: &Arc<Job>) -> Result<()> {
    let client = reqwest::Client::builder().user_agent("smart-media-legacy-import").connect_timeout(Duration::from_secs(20)).build()?;
    let resp = client.get(&asset.url).send().await.with_context(|| format!("cannot reach {}", asset.url))?;
    if !resp.status().is_success() {
        bail!("download of {} failed: HTTP {}", asset.url, resp.status());
    }
    let total = resp.content_length().unwrap_or(0);
    let mut file = tokio::fs::File::create(partial).await?;
    let mut hasher = Sha256::new();
    let mut done = 0u64;
    let mut stream = resp.bytes_stream();
    job.set_engine(EngineStatus::Downloading { done, total });
    while let Some(chunk) = stream.next().await {
        job.check_cancel()?;
        let chunk = chunk.context("download interrupted")?;
        hasher.update(&chunk);
        file.write_all(&chunk).await?;
        done += chunk.len() as u64;
        job.set_engine(EngineStatus::Downloading { done, total });
        job.update(PhaseId::Engine, done, total, "downloading the engine");
    }
    file.flush().await?;
    drop(file);
    let actual = hex(&hasher.finalize());
    if actual != asset.sha256 {
        bail!("checksum mismatch for {}: expected {}, got {actual}; the download was discarded", asset.name, asset.sha256);
    }
    job.note(format!("Downloaded {} ({}), SHA-256 verified", asset.name, human(done)));
    Ok(())
}

fn install_binary(asset: &Asset, partial: &Path, dest: &Path) -> Result<()> {
    let tmp = dest.with_file_name(format!("{}.new", bin_name()));
    match asset.kind {
        AssetKind::Exe => {
            std::fs::copy(partial, &tmp)?;
        }
        AssetKind::Tgz => {
            let gz = flate2::read::GzDecoder::new(std::fs::File::open(partial)?);
            let mut archive = tar::Archive::new(gz);
            let mut found = false;
            for entry in archive.entries()? {
                let mut entry = entry?;
                let is_cli = entry.header().entry_type().is_file() && entry.path()?.file_name().is_some_and(|n| n == "surreal" || n == "surreal.exe");
                if is_cli {
                    let mut out = std::fs::File::create(&tmp)?;
                    std::io::copy(&mut entry, &mut out)?;
                    found = true;
                    break;
                }
            }
            if !found {
                bail!("{} does not contain a surreal binary", asset.name);
            }
        }
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o755))?;
    }
    let digest = sha256_file(&tmp)?;
    if let Some(known) = &asset.binary_sha256 {
        if &digest != known {
            let _ = std::fs::remove_file(&tmp);
            bail!("extracted engine checksum {digest} does not match the pinned {known}");
        }
    }
    std::fs::write(dest.with_file_name("surreal.sha256"), &digest)?;
    std::fs::rename(&tmp, dest)?;
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum StoreKind {
    SurrealKvLegacy,
    SurrealKvCurrent,
    RocksDb,
    Unknown,
}

impl StoreKind {
    pub fn label(self) -> &'static str {
        match self {
            StoreKind::SurrealKvLegacy => "SurrealKV 0.9 (clog and manifest layout)",
            StoreKind::SurrealKvCurrent => "SurrealKV with sstables (SurrealDB 3.x layout)",
            StoreKind::RocksDb => "RocksDB",
            StoreKind::Unknown => "unrecognised",
        }
    }

    fn scheme(self) -> &'static str {
        match self {
            StoreKind::RocksDb => "rocksdb",
            _ => "surrealkv",
        }
    }
}

pub fn detect_kind(dir: &Path) -> StoreKind {
    let has = |n: &str| dir.join(n).exists();
    if has("clog") && has("manifest") {
        StoreKind::SurrealKvLegacy
    } else if has("sstables") || has("vlog") || has("wal") {
        StoreKind::SurrealKvCurrent
    } else if has("CURRENT") && std::fs::read_dir(dir).is_ok_and(|rd| rd.flatten().any(|e| e.path().extension().is_some_and(|x| x == "sst"))) {
        StoreKind::RocksDb
    } else {
        StoreKind::Unknown
    }
}

pub fn endpoint(kind: StoreKind, store: &Path) -> String {
    format!("{}://{}", kind.scheme(), store.display())
}

fn stderr_collector(child: &mut tokio::process::Child) -> Arc<Mutex<VecDeque<String>>> {
    let tail = Arc::new(Mutex::new(VecDeque::new()));
    if let Some(err) = child.stderr.take() {
        let sink = Arc::clone(&tail);
        tokio::spawn(async move {
            let mut lines = BufReader::new(err).lines();
            while let Ok(Some(l)) = lines.next_line().await {
                if let Ok(mut q) = sink.lock() {
                    if q.len() >= STDERR_KEEP {
                        q.pop_front();
                    }
                    q.push_back(l);
                }
            }
        });
    }
    tail
}

fn tail_text(tail: &Arc<Mutex<VecDeque<String>>>) -> String {
    tail.lock().map(|q| q.iter().rev().take(8).rev().cloned().collect::<Vec<_>>().join(" | ")).unwrap_or_default()
}

#[derive(Clone, Debug, Serialize)]
pub struct ExportStats {
    pub bytes: u64,
    pub millis: u128,
    pub ns: String,
    pub db: String,
}

/// Runs `surreal export` against `store` into `out` (created with mode 0600 before the CLI writes to it).
pub async fn export(engine: &Engine, kind: StoreKind, store: &Path, ns: &str, db: &str, out: &Path, est_bytes: u64, job: &Arc<Job>) -> Result<ExportStats> {
    let started = std::time::Instant::now();
    let (store, out) = (absolute(store), absolute(out));
    let (store, out) = (store.as_path(), out.as_path());
    create_private_file(out)?;
    let mut cmd = command(&absolute(&engine.path));
    cmd.args(["export", "--endpoint", &endpoint(kind, store), "--ns", ns, "--db", db, "--log", "error"]).arg(out);
    cmd.stdout(Stdio::null()).stderr(Stdio::piped());
    if let Some(dir) = out.parent().filter(|d| d.is_dir()) {
        cmd.current_dir(dir);
    }
    let mut child = cmd.spawn().with_context(|| format!("cannot start {}", engine.path.display()))?;
    let tail = stderr_collector(&mut child);
    job.begin(PhaseId::Export, est_bytes, "exporting with the SurrealDB 2.x CLI");
    let status = loop {
        tokio::select! {
            r = child.wait() => break r?,
            _ = tokio::time::sleep(Duration::from_millis(250)) => {
                let size = tokio::fs::metadata(out).await.map(|m| m.len()).unwrap_or(0);
                job.update(PhaseId::Export, size, est_bytes.max(size + 1), "exporting with the SurrealDB 2.x CLI");
                if job.is_cancel_requested() {
                    let _ = child.kill().await;
                    let _ = tokio::fs::remove_file(out).await;
                    return Err(anyhow::Error::new(super::job::Cancelled));
                }
            }
        }
    };
    if !status.success() {
        let _ = tokio::fs::remove_file(out).await;
        bail!("surreal export failed ({status}): {}", tail_text(&tail));
    }
    let bytes = tokio::fs::metadata(out).await.map(|m| m.len()).unwrap_or(0);
    Ok(ExportStats { bytes, millis: started.elapsed().as_millis(), ns: ns.to_string(), db: db.to_string() })
}

/// Creates an empty file readable only by the current user.
pub fn create_private_file(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut o = std::fs::OpenOptions::new();
    o.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.mode(0o600);
    }
    o.open(path).with_context(|| format!("cannot create {}", path.display()))?;
    Ok(())
}

/// Restricts an existing file to the current user.
pub fn make_private(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
    #[cfg(not(unix))]
    let _ = path;
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct DbInfo {
    pub name: String,
    pub tables: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct NsInfo {
    pub ns: String,
    pub dbs: Vec<DbInfo>,
}

fn keys_of(v: &serde_json::Value, field: &str) -> Vec<String> {
    let mut keys: Vec<String> = v.get(field).and_then(|o| o.as_object()).map(|o| o.keys().cloned().collect()).unwrap_or_default();
    keys.sort();
    keys
}

struct Session {
    child: tokio::process::Child,
    stdin: tokio::process::ChildStdin,
    out: tokio::io::Lines<BufReader<tokio::process::ChildStdout>>,
    tail: Arc<Mutex<VecDeque<String>>>,
}

impl Session {
    async fn ask(&mut self, q: &str) -> Result<serde_json::Value> {
        self.stdin.write_all(q.as_bytes()).await?;
        self.stdin.write_all(b"\n").await?;
        self.stdin.flush().await?;
        loop {
            let line = tokio::time::timeout(QUERY_TIMEOUT, self.out.next_line()).await.map_err(|_| anyhow!("the engine did not answer within {:?}", QUERY_TIMEOUT))??;
            let Some(line) = line else { bail!("the engine closed the session: {}", tail_text(&self.tail)) };
            if line.trim().is_empty() {
                continue;
            }
            return serde_json::from_str(&line).with_context(|| format!("unreadable engine answer to {q:?}"));
        }
    }
}

/// Lists namespaces, databases and tables of a store copy through one `surreal sql` session.
pub async fn detect_namespaces(engine: &Engine, kind: StoreKind, store: &Path, cwd: &Path, job: &Arc<Job>) -> Result<Vec<NsInfo>> {
    let mut cmd = command(&absolute(&engine.path));
    cmd.current_dir(absolute(cwd));
    cmd.args(["sql", "--endpoint", &endpoint(kind, &absolute(store)), "--hide-welcome", "--json", "--log", "error"]);
    cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn().with_context(|| format!("cannot start {}", engine.path.display()))?;
    let tail = stderr_collector(&mut child);
    let stdin = child.stdin.take().ok_or_else(|| anyhow!("no stdin"))?;
    let out = BufReader::new(child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?).lines();
    let mut s = Session { child, stdin, out, tail };
    job.note("Inspecting namespaces and databases of the store copy");
    let result = async {
        let root = s.ask("INFO FOR ROOT;").await?;
        let mut found = Vec::new();
        for ns in keys_of(root.get(0).unwrap_or(&serde_json::Value::Null), "namespaces") {
            let info = s.ask(&format!("USE NS `{ns}`; INFO FOR NS;")).await?;
            let mut dbs = Vec::new();
            for db in keys_of(info.get(1).unwrap_or(&serde_json::Value::Null), "databases") {
                let dbinfo = s.ask(&format!("USE NS `{ns}` DB `{db}`; INFO FOR DB;")).await?;
                dbs.push(DbInfo { name: db, tables: keys_of(dbinfo.get(1).unwrap_or(&serde_json::Value::Null), "tables") });
            }
            found.push(NsInfo { ns, dbs });
        }
        anyhow::Ok(found)
    }
    .await;
    drop(s.stdin);
    if tokio::time::timeout(Duration::from_secs(20), s.child.wait()).await.is_err() {
        let _ = s.child.kill().await;
    }
    result
}

/// Picks the single (ns, db) pair that holds a `thumbnails` table.
pub fn pick_namespace(found: &[NsInfo]) -> std::result::Result<(String, String), Vec<(String, String)>> {
    let mut with_thumbs = Vec::new();
    let mut any = Vec::new();
    for n in found {
        for d in &n.dbs {
            any.push((n.ns.clone(), d.name.clone()));
            if d.tables.iter().any(|t| t == "thumbnails") {
                with_thumbs.push((n.ns.clone(), d.name.clone()));
            }
        }
    }
    match (with_thumbs.len(), any.len()) {
        (1, _) => Ok(with_thumbs.remove(0)),
        (0, 1) => Ok(any.remove(0)),
        (0, _) => Err(any),
        _ => Err(with_thumbs),
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CopyStats {
    pub files: usize,
    pub bytes: u64,
    pub changed_during_copy: bool,
}

struct SrcFile {
    rel: PathBuf,
    len: u64,
    modified: Option<std::time::SystemTime>,
}

fn scan(root: &Path) -> std::io::Result<(Vec<PathBuf>, Vec<SrcFile>)> {
    let mut dirs = Vec::new();
    let mut files = Vec::new();
    let mut stack = vec![PathBuf::new()];
    while let Some(rel_dir) = stack.pop() {
        for e in std::fs::read_dir(root.join(&rel_dir))? {
            let e = e?;
            if e.file_name() == "LOCK" {
                continue;
            }
            let rel = rel_dir.join(e.file_name());
            let ft = e.file_type()?;
            if ft.is_dir() {
                dirs.push(rel.clone());
                stack.push(rel);
            } else if ft.is_file() {
                let md = e.metadata()?;
                files.push(SrcFile { rel, len: md.len(), modified: md.modified().ok() });
            }
        }
    }
    files.sort_by(|a, b| a.rel.cmp(&b.rel));
    Ok((dirs, files))
}

fn within(inner: &Path, outer: &Path) -> bool {
    let canon = |p: &Path| {
        let mut base = p.to_path_buf();
        let mut tail = Vec::new();
        while !base.exists() {
            match base.file_name() {
                Some(n) => tail.push(n.to_os_string()),
                None => break,
            }
            if !base.pop() {
                break;
            }
        }
        let mut full = base.canonicalize().unwrap_or(base);
        for t in tail.into_iter().rev() {
            full.push(t);
        }
        full
    };
    canon(inner).starts_with(canon(outer))
}

/// Total size of the files a copy of `src` would contain.
pub fn store_bytes(src: &Path) -> u64 {
    scan(src).map(|(_, f)| f.iter().map(|x| x.len).sum()).unwrap_or(0)
}

/// Copies an old store folder (read-only on the source) into `dst`, skipping LOCK files.
pub fn copy_store_blocking(src: &Path, dst: &Path, job: &Job) -> Result<CopyStats> {
    use std::io::{Read, Write};
    if within(dst, src) || within(src, dst) {
        bail!("the work folder {} overlaps the old store {}", dst.display(), src.display());
    }
    let (dirs, files) = scan(src).with_context(|| format!("cannot read {}", src.display()))?;
    let total: u64 = files.iter().map(|f| f.len).sum();
    if dst.exists() {
        std::fs::remove_dir_all(dst).with_context(|| format!("cannot clear {}", dst.display()))?;
    }
    std::fs::create_dir_all(dst)?;
    for d in &dirs {
        std::fs::create_dir_all(dst.join(d))?;
    }
    job.begin(PhaseId::Copy, total, "copying the old store");
    let mut copied = 0u64;
    let mut buf = vec![0u8; COPY_BUF];
    for f in &files {
        let mut r = std::fs::File::open(src.join(&f.rel)).with_context(|| format!("cannot read {}", src.join(&f.rel).display()))?;
        let target = dst.join(&f.rel);
        let mut w = std::fs::File::create(&target)?;
        let mut file_bytes = 0u64;
        loop {
            job.check_cancel()?;
            let n = r.read(&mut buf)?;
            if n == 0 {
                break;
            }
            w.write_all(&buf[..n])?;
            file_bytes += n as u64;
            copied += n as u64;
            job.update(PhaseId::Copy, copied, total.max(copied), "copying the old store");
        }
        let _ = w.sync_all();
        if file_bytes != f.len {
            bail!("{} changed while copying ({} of {} bytes)", f.rel.display(), file_bytes, f.len);
        }
    }
    let changed = scan(src).map(|(_, after)| after.len() != files.len() || after.iter().zip(&files).any(|(a, b)| a.rel != b.rel || a.len != b.len || a.modified != b.modified)).unwrap_or(true);
    if changed {
        job.note("The old store changed on disk while it was being copied");
    }
    Ok(CopyStats { files: files.len(), bytes: copied, changed_during_copy: changed })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_parsing_accepts_only_two_x() {
        assert_eq!(parse_version("2.3.10 for linux on x86_64\n").unwrap(), "2.3.10");
        assert_eq!(parse_version("surreal-v2.4.1 for macos").unwrap(), "2.4.1");
        assert!(parse_version("3.3.4 for linux").is_err());
        assert!(parse_version("garbage").is_err());
    }

    #[test]
    fn pinned_assets_cover_the_release_platforms() {
        let linux = pinned_asset_for("linux", "x86_64").unwrap();
        assert_eq!(linux.name, "surreal-v2.3.10.linux-amd64.tgz");
        assert_eq!(linux.sha256, "cb3ccfa9906684de8cad4e47326056617a36f4b049231d4aac53daadfce41dc2");
        assert!(linux.url.ends_with("/v2.3.10/surreal-v2.3.10.linux-amd64.tgz"));
        assert_eq!(pinned_asset_for("windows", "x86_64").unwrap().kind, AssetKind::Exe);
        assert!(pinned_asset_for("macos", "aarch64").is_some());
        assert!(pinned_asset_for("linux", "aarch64").is_some());
        assert!(pinned_asset_for("freebsd", "x86_64").is_none());
    }

    #[test]
    fn store_kind_detection_follows_the_layout() {
        let d = tempfile::tempdir().unwrap();
        assert_eq!(detect_kind(d.path()), StoreKind::Unknown);
        std::fs::create_dir_all(d.path().join("clog")).unwrap();
        std::fs::create_dir_all(d.path().join("manifest")).unwrap();
        assert_eq!(detect_kind(d.path()), StoreKind::SurrealKvLegacy);
        let n = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(n.path().join("sstables")).unwrap();
        assert_eq!(detect_kind(n.path()), StoreKind::SurrealKvCurrent);
        assert!(endpoint(StoreKind::SurrealKvLegacy, Path::new("/x/y")).starts_with("surrealkv:///x"));
    }

    #[test]
    fn namespace_pick_prefers_the_thumbnails_database() {
        let a = NsInfo { ns: "file_explorer".into(), dbs: vec![DbInfo { name: "ai_search".into(), tables: vec!["thumbnails".into()] }] };
        let b = NsInfo { ns: "other".into(), dbs: vec![DbInfo { name: "zzz".into(), tables: vec!["foo".into()] }] };
        assert_eq!(pick_namespace(&[b.clone(), a.clone()]).unwrap(), ("file_explorer".into(), "ai_search".into()));
        let c = NsInfo { ns: "x".into(), dbs: vec![DbInfo { name: "y".into(), tables: vec!["thumbnails".into()] }] };
        assert_eq!(pick_namespace(&[a, c]).unwrap_err().len(), 2);
        assert_eq!(pick_namespace(&[b]).unwrap(), ("other".into(), "zzz".into()));
    }

    #[test]
    fn cached_engines_are_checked_against_the_pinned_or_recorded_digest() {
        let d = tempfile::tempdir().unwrap();
        let bin = d.path().join("surreal");
        std::fs::write(&bin, b"fake engine").unwrap();
        let digest = sha256_file(&bin).unwrap();
        assert!(verify_cached(&bin, None).is_err(), "no record, no trust");
        std::fs::write(d.path().join("surreal.sha256"), &digest).unwrap();
        assert!(verify_cached(&bin, None).unwrap());
        let pinned = Asset { name: "a".into(), url: "u".into(), sha256: "x".into(), kind: AssetKind::Tgz, binary_sha256: Some(digest.clone()) };
        assert!(verify_cached(&bin, Some(&pinned)).unwrap());
        std::fs::write(&bin, b"tampered engine").unwrap();
        assert!(verify_cached(&bin, None).is_err());
        assert!(verify_cached(&bin, Some(&pinned)).is_err());
    }

    #[test]
    fn pinned_digests_are_well_formed() {
        for (os, arch) in [("linux", "x86_64"), ("linux", "aarch64"), ("macos", "x86_64"), ("macos", "aarch64"), ("windows", "x86_64")] {
            let a = pinned_asset_for(os, arch).unwrap();
            assert_eq!(a.sha256.len(), 64);
            assert!(a.sha256.bytes().all(|b| b.is_ascii_hexdigit()));
            assert!(a.url.starts_with("https://github.com/surrealdb/surrealdb/releases/download/v2.3.10/"));
        }
        assert_eq!(pinned_asset_for("linux", "x86_64").unwrap().binary_sha256.unwrap().len(), 64);
    }

    #[test]
    fn copy_skips_lock_and_never_touches_the_source() {
        let src = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(src.path().join("clog")).unwrap();
        std::fs::create_dir_all(src.path().join("manifest")).unwrap();
        std::fs::write(src.path().join("clog/00"), vec![7u8; 10_000]).unwrap();
        std::fs::write(src.path().join("manifest/0.manifest"), b"m").unwrap();
        std::fs::write(src.path().join("LOCK"), b"x").unwrap();
        let work = tempfile::tempdir().unwrap();
        let dst = work.path().join("copy");
        let job = Job::new();
        let stats = copy_store_blocking(src.path(), &dst, &job).unwrap();
        assert_eq!((stats.files, stats.bytes, stats.changed_during_copy), (2, 10_001, false));
        assert!(!dst.join("LOCK").exists());
        assert_eq!(std::fs::read(dst.join("clog/00")).unwrap().len(), 10_000);
        assert!(copy_store_blocking(src.path(), &src.path().join("inside"), &job).is_err());
    }
}
