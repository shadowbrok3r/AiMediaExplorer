//! The gateway token: one mode 0600 file in the settings folder; never the database, never a log line.

use std::io::Write;
use std::path::{Path, PathBuf};

pub const TOKEN_FILE: &str = "zeroclaw_token";

/// Where the gateway token is kept.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenStore {
    dir: PathBuf,
}

impl TokenStore {
    /// The store of the settings folder (`<config>/smart-media`), when the platform has one.
    pub fn standard() -> Option<Self> {
        crate::ai::mcp::config::default_dir().map(Self::at)
    }

    pub fn at(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn path(&self) -> PathBuf {
        self.dir.join(TOKEN_FILE)
    }

    /// The stored token; `None` when the file is missing, empty or unreadable.
    pub fn load(&self) -> Option<String> {
        let text = std::fs::read_to_string(self.path()).ok()?;
        let token = text.trim();
        (!token.is_empty()).then(|| token.to_string())
    }

    pub fn exists(&self) -> bool {
        self.load().is_some()
    }

    /// Writes the token through a temporary file so a crash never leaves half of it.
    pub fn save(&self, token: &str) -> std::io::Result<()> {
        let token = token.trim();
        if token.is_empty() {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "the token is empty"));
        }
        crate::ai::mcp::auth::ensure_private_dir(&self.dir)?;
        let tmp = self.dir.join(format!("{TOKEN_FILE}.tmp"));
        write_private(&tmp, token)?;
        std::fs::rename(&tmp, self.path())
    }

    /// Deletes the stored token; a missing file is fine.
    pub fn clear(&self) -> std::io::Result<()> {
        match std::fs::remove_file(self.path()) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e),
            _ => Ok(()),
        }
    }
}

fn write_private(path: &Path, text: &str) -> std::io::Result<()> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    }
    file.write_all(text.as_bytes())?;
    file.sync_all()
}

/// First characters of a token and a row of stars, for display.
pub fn mask(token: &str) -> String {
    crate::ai::mcp::auth::mask(token)
}

/// Reads a token from a file for the command line and the import path; the text is trimmed.
pub fn read_token_file(path: &Path) -> Result<String, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let token = text.trim();
    if token.is_empty() { Err(format!("{} is empty", path.display())) } else { Ok(token.to_string()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ame-zc-secrets-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn a_saved_token_reads_back_trimmed() {
        let store = TokenStore::at(dir("roundtrip"));
        assert_eq!(store.load(), None);
        store.save("  tok-123\n").unwrap();
        assert_eq!(store.load().as_deref(), Some("tok-123"));
        assert!(store.exists());
        let _ = std::fs::remove_dir_all(store.dir());
    }

    #[cfg(unix)]
    #[test]
    fn the_token_file_is_private_to_the_owner() {
        use std::os::unix::fs::PermissionsExt;
        let store = TokenStore::at(dir("mode").join("nested"));
        store.save("secret-value").unwrap();
        let mode = std::fs::metadata(store.path()).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
        store.save("replaced").unwrap();
        assert_eq!(std::fs::metadata(store.path()).unwrap().permissions().mode() & 0o777, 0o600);
        assert!(!store.dir().join(format!("{TOKEN_FILE}.tmp")).exists());
        let _ = std::fs::remove_dir_all(store.dir().parent().unwrap());
    }

    #[test]
    fn an_empty_token_is_refused_and_clear_removes_the_file() {
        let store = TokenStore::at(dir("clear"));
        assert!(store.save("   ").is_err());
        store.save("abc").unwrap();
        store.clear().unwrap();
        assert!(!store.exists());
        store.clear().unwrap();
        let _ = std::fs::remove_dir_all(store.dir());
    }

    #[test]
    fn the_mask_hides_all_but_the_first_characters() {
        let m = mask("abcdefghijklmnop");
        assert!(m.starts_with("abcd") && !m.contains("efgh"));
    }

    #[test]
    fn a_token_file_is_trimmed_and_must_not_be_empty() {
        let d = dir("file");
        std::fs::create_dir_all(&d).unwrap();
        let path = d.join("t");
        std::fs::write(&path, "  tok \n").unwrap();
        assert_eq!(read_token_file(&path).unwrap(), "tok");
        std::fs::write(&path, "\n").unwrap();
        assert!(read_token_file(&path).is_err());
        assert!(read_token_file(&d.join("missing")).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }
}
