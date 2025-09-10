use crate::database::Thumbnail;
use std::path::Path;

// Unified representation of an item inside an archive listing
#[derive(Clone, Debug)]
pub struct ArchiveEntry {
    pub is_dir: bool,
    pub name: String,         // immediate child name (no path separators)
    pub size: u64,
}

// Input to list request
#[derive(Clone, Debug)]
pub struct ListParams<'a> {
    pub archive_fs_path: &'a str,
    pub internal_path: &'a str, // logical path inside the archive
    pub password: Option<&'a str>,
}

// Trait for archive handlers
pub trait ArchiveHandler: Send + Sync {
    fn scheme(&self) -> &'static str; // e.g. "zip", "tar", "7z"
    fn handles_extension(&self, lower_ext_or_name: &str) -> bool; // quick detection from file name
    fn list(&self, params: &ListParams) -> anyhow::Result<Vec<ArchiveEntry>>;
}

// -------- ZIP handler --------
pub struct ZipHandler;

impl ArchiveHandler for ZipHandler {
    fn scheme(&self) -> &'static str { "zip" }
    fn handles_extension(&self, lower: &str) -> bool { lower.ends_with(".zip") || lower == "zip" }
    fn list(&self, params: &ListParams) -> anyhow::Result<Vec<ArchiveEntry>> {
        let f = std::fs::File::open(params.archive_fs_path)?;
        let mut zip = zip::ZipArchive::new(f)?;
        let mut dirs: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let mut files: Vec<ArchiveEntry> = Vec::new();
        let prefix = normalize_prefix(params.internal_path);
        for i in 0..zip.len() {
            let entry = zip.by_index(i)?;
            let full_name = entry.name().to_string();
            if !full_name.starts_with(&prefix) { continue; }
            let rest = &full_name[prefix.len()..];
            if rest.is_empty() { continue; }
            let parts: Vec<&str> = rest.split('/').filter(|s| !s.is_empty()).collect();
            if parts.is_empty() { continue; }
            if parts.len() > 1 || full_name.ends_with('/') {
                dirs.insert(parts[0].to_string());
            } else {
                files.push(ArchiveEntry { is_dir: false, name: parts[0].to_string(), size: entry.size() as u64 });
            }
        }
        let mut out: Vec<ArchiveEntry> = Vec::with_capacity(dirs.len() + files.len());
        out.extend(dirs.into_iter().map(|d| ArchiveEntry { is_dir: true, name: d, size: 0 }));
        out.extend(files.into_iter());
        Ok(out)
    }
}

// -------- TAR handler (supports tar, tar.gz, tgz, tar.bz2, tbz2, tar.xz, txz) --------
pub struct TarHandler;

impl ArchiveHandler for TarHandler {
    fn scheme(&self) -> &'static str { "tar" }
    fn handles_extension(&self, name: &str) -> bool {
        let n = name.to_ascii_lowercase();
        n.ends_with(".tar") || n.ends_with(".tgz") || n.ends_with(".tar.gz")
            || n.ends_with(".tbz") || n.ends_with(".tbz2") || n.ends_with(".tar.bz2")
            || n.ends_with(".txz") || n.ends_with(".tar.xz") || n == "tar" || n == "tgz" || n == "tbz" || n == "tbz2" || n == "txz"
    }
    fn list(&self, params: &ListParams) -> anyhow::Result<Vec<ArchiveEntry>> {
        use std::io::Read;
        let ext_l = Path::new(params.archive_fs_path)
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();
        let file = std::fs::File::open(params.archive_fs_path)?;
        let reader: tar::Archive<Box<dyn std::io::Read>> = if ext_l == "tar" {
            tar::Archive::new(Box::new(file) as Box<dyn Read>)
        } else if ext_l == "gz" || params.archive_fs_path.ends_with(".tgz") || params.archive_fs_path.ends_with(".tar.gz") {
            let gz = flate2::read::GzDecoder::new(file);
            tar::Archive::new(Box::new(gz) as Box<dyn Read>)
        } else if ext_l == "bz" || ext_l == "bz2" || params.archive_fs_path.ends_with(".tbz") || params.archive_fs_path.ends_with(".tbz2") || params.archive_fs_path.ends_with(".tar.bz2") {
            let bz = bzip2::read::BzDecoder::new(file);
            tar::Archive::new(Box::new(bz) as Box<dyn Read>)
        } else if ext_l == "xz" || params.archive_fs_path.ends_with(".txz") || params.archive_fs_path.ends_with(".tar.xz") {
            let xz = xz2::read::XzDecoder::new(file);
            tar::Archive::new(Box::new(xz) as Box<dyn Read>)
        } else {
            tar::Archive::new(Box::new(file) as Box<dyn Read>)
        };

        let mut tar = reader;
        let mut dirs: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let mut files: Vec<ArchiveEntry> = Vec::new();
        let prefix = normalize_prefix(params.internal_path);
        let iter = tar.entries()?;
        for entry_res in iter {
            let entry = match entry_res { Ok(e) => e, Err(_) => continue };
            let path = match entry.path() { Ok(p) => p, Err(_) => continue };
            let full_name = path.to_string_lossy().replace('\\', "/");
            if !full_name.starts_with(&prefix) { continue; }
            let rest = &full_name[prefix.len()..];
            if rest.is_empty() { continue; }
            let parts: Vec<&str> = rest.split('/').filter(|s| !s.is_empty()).collect();
            if parts.is_empty() { continue; }
            let is_dir = entry.header().entry_type().is_dir();
            if parts.len() > 1 || is_dir {
                dirs.insert(parts[0].to_string());
            } else {
                files.push(ArchiveEntry { is_dir: false, name: parts[0].to_string(), size: entry.size() as u64 });
            }
        }
        let mut out: Vec<ArchiveEntry> = Vec::with_capacity(dirs.len() + files.len());
        out.extend(dirs.into_iter().map(|d| ArchiveEntry { is_dir: true, name: d, size: 0 }));
        out.extend(files.into_iter());
        Ok(out)
    }
}

// -------- 7z handler via external 7z.exe --------
pub struct SevenZHandler;

impl ArchiveHandler for SevenZHandler {
    fn scheme(&self) -> &'static str { "7z" }
    fn handles_extension(&self, ext: &str) -> bool { ext.ends_with(".7z") || ext == "7z" }
    fn list(&self, params: &ListParams) -> anyhow::Result<Vec<ArchiveEntry>> {
        use std::collections::BTreeSet;
    let pw: &str = params.password.unwrap_or("");
    let mut sz = sevenz_rust2::ArchiveReader::open(params.archive_fs_path, pw.into())?;
        let mut dirs: BTreeSet<String> = BTreeSet::new();
        let mut files: Vec<ArchiveEntry> = Vec::new();
        let prefix = normalize_prefix(params.internal_path);
        for entry in sz.archive().files.iter() {
            let mut full_name = entry.name().replace('\\', "/");
            if full_name.starts_with("./") { full_name = full_name[2..].to_string(); }
            if !full_name.starts_with(&prefix) { continue; }
            let rest = &full_name[prefix.len()..];
            if rest.is_empty() { continue; }
            let parts: Vec<&str> = rest.split('/').filter(|s| !s.is_empty()).collect();
            if parts.is_empty() { continue; }
            let is_dir = !entry.has_stream() || full_name.ends_with('/');
            if parts.len() > 1 || is_dir {
                dirs.insert(parts[0].to_string());
            } else {
                let size = entry.size();
                files.push(ArchiveEntry { is_dir: false, name: parts[0].to_string(), size });
            }
        }
        let mut out_list: Vec<ArchiveEntry> = Vec::with_capacity(dirs.len() + files.len());
        out_list.extend(dirs.into_iter().map(|d| ArchiveEntry { is_dir: true, name: d, size: 0 }));
        out_list.extend(files.into_iter());
        Ok(out_list)
    }
}

// Normalize internal path to a directory prefix ("foo/bar" -> "foo/bar/")
fn normalize_prefix(internal: &str) -> String {
    let t = internal.trim_matches('/');
    if t.is_empty() { String::new() } else { format!("{}/", t) }
}

// Helper to turn ArchiveEntry into a Thumbnail for a given scheme
pub fn entry_to_thumbnail(scheme: &str, archive_fs_path: &str, internal_path: &str, e: ArchiveEntry) -> Thumbnail {
    if e.is_dir {
        let child_vpath = if internal_path.trim_matches('/').is_empty() {
            format!("{}://{}!/{}", scheme, archive_fs_path, e.name)
        } else {
            format!("{}://{}!/{}/{}", scheme, archive_fs_path, internal_path.trim_matches('/'), e.name)
        };
        let mut t = Thumbnail::default();
        t.path = child_vpath.clone();
        t.filename = e.name;
        t.file_type = "<DIR>".into();
        return t;
    } else {
        let vpath = format!("{}://{}!/{}/{}",
            scheme,
            archive_fs_path,
            normalize_prefix(internal_path).trim_end_matches('/'),
            e.name
        ).replace("//!", "/!").replace("!//", "!/");
        Thumbnail {
            id: None,
            db_created: None,
            path: vpath,
            filename: e.name.clone(),
            file_type: e.name.split('.').last().unwrap_or("").to_ascii_lowercase(),
            size: e.size,
            description: None,
            caption: None,
            tags: Vec::new(),
            category: None,
            thumbnail_b64: None,
            modified: None,
            hash: None,
            parent_dir: format!("{}://{}!/{}", scheme, archive_fs_path, normalize_prefix(internal_path).trim_end_matches('/')),
        }
    }
}

// A simple registry to select a handler
pub struct ArchiveRegistry {
    handlers: Vec<Box<dyn ArchiveHandler>>,
}

impl Default for ArchiveRegistry {
    fn default() -> Self {
        Self {
            handlers: vec![
                Box::new(ZipHandler),
                Box::new(TarHandler),
                Box::new(SevenZHandler),
            ],
        }
    }
}

impl ArchiveRegistry {
    pub fn by_scheme<'a>(&'a self, scheme: &str) -> Option<&'a dyn ArchiveHandler> {
        self.handlers.iter().map(|b| &**b).find(|h| h.scheme() == scheme)
    }
    pub fn by_extension<'a>(&'a self, name_or_ext: &str) -> Option<&'a dyn ArchiveHandler> {
        self.handlers.iter().map(|b| &**b).find(|h| h.handles_extension(name_or_ext))
    }
}
