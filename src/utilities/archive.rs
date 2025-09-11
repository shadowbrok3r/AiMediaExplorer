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
        
        let mut valid_entries = Vec::new();
        
        // First pass: collect entry information without holding mutable borrows
        for i in 0..zip.len() {
            match zip.by_index(i) {
                Ok(entry) => {
                    let full_name = entry.name().to_string();
                    let size = entry.size() as u64;
            let encrypted = entry.encrypted();
                    drop(entry); // Explicitly drop to release the borrow
                    
                    if !full_name.starts_with(&prefix) { continue; }
                    let rest = &full_name[prefix.len()..];
                    if rest.is_empty() { continue; }
                    let parts: Vec<&str> = rest.split('/').filter(|s| !s.is_empty()).collect();
                    if parts.is_empty() { continue; }
                    
                    valid_entries.push((full_name, size, encrypted));
                }
                Err(zip::result::ZipError::UnsupportedArchive(_)) | 
                Err(zip::result::ZipError::InvalidPassword) => {
                    // Skip encrypted files when no password is provided initially
                    continue;
                }
                Err(e) => return Err(e.into()),
            }
        }
        
        // Second pass: process collected entries
        for (full_name, size, encrypted) in valid_entries {
            if encrypted && params.password.is_none() {
                // No password provided for encrypted entry, skip
                continue;
            }
            
            // Process the entry
            let rest = &full_name[prefix.len()..];
            let parts: Vec<&str> = rest.split('/').filter(|s| !s.is_empty()).collect();
            if parts.len() > 1 || full_name.ends_with('/') {
                dirs.insert(parts[0].to_string());
            } else {
                files.push(ArchiveEntry { is_dir: false, name: parts[0].to_string(), size });
            }
        }
        
        let mut out: Vec<ArchiveEntry> = Vec::with_capacity(dirs.len() + files.len());
        out.extend(dirs.into_iter().map(|d| ArchiveEntry { is_dir: true, name: d, size: 0 }));
        out.extend(files.into_iter());
    // Allow listing even if encrypted files exist; thumbnail extraction will request password
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
    let sz = sevenz_rust2::ArchiveReader::open(params.archive_fs_path, pw.into())?;
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
        
        // Thumbnails for media files will be generated asynchronously in the explorer
        
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
            logical_group: None,
        }
    }
}

// Extract a file from an archive temporarily and generate a thumbnail
pub fn extract_and_generate_thumbnail(
    scheme: &str, 
    archive_fs_path: &str, 
    internal_path: &str, 
    filename: &str,
    password: Option<&str>
) -> anyhow::Result<Option<String>> {
    let full_internal_path = if internal_path.trim_matches('/').is_empty() {
        filename.to_string()
    } else {
        format!("{}/{}", internal_path.trim_matches('/'), filename)
    };
    log::info!("extracting thumbnail from {full_internal_path}");
    
    match scheme {
        "zip" => extract_zip_file_and_generate_thumbnail(archive_fs_path, &full_internal_path, password),
        "7z" => extract_7z_file_and_generate_thumbnail(archive_fs_path, &full_internal_path, password),
        "tar" => Ok(extract_tar_file_and_generate_thumbnail(archive_fs_path, &full_internal_path)),
        _ => Ok(None),
    }
}

fn extract_zip_file_and_generate_thumbnail(archive_path: &str, internal_path: &str, password: Option<&str>) -> anyhow::Result<Option<String>> {
    use std::io::Write;
    let file = std::fs::File::open(archive_path)?;
    let mut zip = zip::ZipArchive::new(file)?;
    // Find the file in the archive
    let mut zip_file = if let Some(pw) = password {
        let pw_bytes = pw.as_bytes();
        match zip.by_name_decrypt(internal_path, pw_bytes) {
            Ok(f) => f,
            Err(zip::result::ZipError::InvalidPassword) => {
                return Err(anyhow::anyhow!("PasswordRequired"));
            }
            Err(e) => return Err(e.into()),
        }
    } else {
        match zip.by_name(internal_path) {
            Ok(f) => {
                if f.encrypted() {
                    return Err(anyhow::anyhow!("PasswordRequired"));
                }
                f
            }
            Err(e) => return Err(e.into()),
        }
    };
    // Create a temporary file with the original extension (helps decoders and Windows Shell)
    let ext = std::path::Path::new(internal_path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    let mut builder = tempfile::Builder::new();
    let ex = format!(".{}", ext);
    if !ext.is_empty() { builder.suffix(&ex); }
    let mut temp_file = builder.tempfile()?;
    std::io::copy(&mut zip_file, &mut temp_file)?;
    let _ = temp_file.as_file_mut().flush();
    let temp_path = temp_file.path();
    // Generate thumbnail based on file extension
    let ext = std::path::Path::new(internal_path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    let result = if crate::utilities::types::is_image(&ext) {
        crate::utilities::thumbs::generate_image_thumb_data(temp_path).ok()
    } else if crate::utilities::types::is_video(&ext) {
        crate::utilities::thumbs::generate_video_thumb_data(temp_path).ok()
    } else {
        None
    };
    Ok(result)
}

fn extract_7z_file_and_generate_thumbnail(archive_path: &str, internal_path: &str, password: Option<&str>) -> anyhow::Result<Option<String>> {
    use std::io::Write;
    
    let pw = password.unwrap_or("");
    let mut sz = match sevenz_rust2::ArchiveReader::open(archive_path, pw.into()) {
        Ok(r) => r,
        Err(e) => {
            let es = format!("{e:?}");
            if es.contains("PasswordRequired") || es.contains("WrongPassword") || es.contains("Wrong password") {
                return Err(anyhow::anyhow!("PasswordRequired"));
            }
            return Err(anyhow::anyhow!(es));
        }
    };
    
    // Read the file data directly
    match sz.read_file(internal_path) {
        Ok(file_data) => {
            // Create a temporary file with the original extension and write the data
            let ext = std::path::Path::new(internal_path)
                .extension()
                .and_then(|e| e.to_str())
                .map(|s| s.to_ascii_lowercase())
                .unwrap_or_default();
            let mut builder = tempfile::Builder::new();
            let ex = format!(".{}", ext);
            if !ext.is_empty() { builder.suffix(&ex); }
            let mut temp_file = builder.tempfile()?;
            temp_file.write_all(&file_data)?;
            let _ = temp_file.as_file_mut().flush();
            let temp_path = temp_file.path();
            
            // Generate thumbnail based on file extension
            let ext = std::path::Path::new(internal_path)
                .extension()
                .and_then(|e| e.to_str())
                .map(|s| s.to_ascii_lowercase())
                .unwrap_or_default();
            let result = if crate::utilities::types::is_image(&ext) {
                crate::utilities::thumbs::generate_image_thumb_data(temp_path).ok()
            } else if crate::utilities::types::is_video(&ext) {
                crate::utilities::thumbs::generate_video_thumb_data(temp_path).ok()
            } else {
                None
            };
            return Ok(result);
        },
        Err(e) => {
            let es = format!("{e:?}");
            log::error!("Error reading 7z file: {es}");
            if es.contains("PasswordRequired") || es.contains("WrongPassword") || es.contains("Wrong password") || es.contains("MaybeBadPassword") || es.contains("corrupted input data") {
                return Err(anyhow::anyhow!("PasswordRequired"));
            }
            return Err(anyhow::anyhow!(es));
        },
    }
}

fn extract_tar_file_and_generate_thumbnail(archive_path: &str, internal_path: &str) -> Option<String> {
    use std::io::{Read, Write};
    
    let ext_l = Path::new(archive_path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    
    let file = std::fs::File::open(archive_path).ok()?;
    let reader: tar::Archive<Box<dyn std::io::Read>> = if ext_l == "tar" {
        tar::Archive::new(Box::new(file) as Box<dyn Read>)
    } else if ext_l == "gz" || archive_path.ends_with(".tgz") || archive_path.ends_with(".tar.gz") {
        let gz = flate2::read::GzDecoder::new(file);
        tar::Archive::new(Box::new(gz) as Box<dyn Read>)
    } else if ext_l == "bz" || ext_l == "bz2" || archive_path.ends_with(".tbz") || archive_path.ends_with(".tbz2") || archive_path.ends_with(".tar.bz2") {
        let bz = bzip2::read::BzDecoder::new(file);
        tar::Archive::new(Box::new(bz) as Box<dyn Read>)
    } else if ext_l == "xz" || archive_path.ends_with(".txz") || archive_path.ends_with(".tar.xz") {
        let xz = xz2::read::XzDecoder::new(file);
        tar::Archive::new(Box::new(xz) as Box<dyn Read>)
    } else {
        tar::Archive::new(Box::new(file) as Box<dyn Read>)
    };
    
    let mut tar = reader;
    let entries = tar.entries().ok()?;
    
    for entry_res in entries {
        let mut entry = entry_res.ok()?;
        let path = entry.path().ok()?;
        let entry_name = path.to_string_lossy().replace('\\', "/");
        
        if entry_name == internal_path {
            // Extract to temporary file with the original extension
            let ext = std::path::Path::new(internal_path)
                .extension()
                .and_then(|e| e.to_str())
                .map(|s| s.to_ascii_lowercase())
                .unwrap_or_default();
            let mut builder = tempfile::Builder::new();
            let ex = format!(".{}", ext);
            if !ext.is_empty() { builder.suffix(&ex); }
            let mut temp_file = builder.tempfile().ok()?;
            std::io::copy(&mut entry, &mut temp_file).ok()?;
            let _ = temp_file.as_file_mut().flush();
            let temp_path = temp_file.path();
            
            // Generate thumbnail based on file extension
            let ext = std::path::Path::new(internal_path)
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();
        
            return if crate::utilities::types::is_image(&ext) {
                crate::utilities::thumbs::generate_image_thumb_data(temp_path).ok()
            } else if crate::utilities::types::is_video(&ext) {
                crate::utilities::thumbs::generate_video_thumb_data(temp_path).ok()
            } else {
                None
            };
        }
    }
    
    None
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
