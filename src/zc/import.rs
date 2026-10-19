//! Imports a gateway from a ZC Codex `config.json`: the selected ZeroClaw machine, else the first one.

use std::fmt;
use std::path::{Path, PathBuf};

use serde_json::Value;

/// A machine read from the config; the token is kept out of `Debug` and `Display`.
#[derive(Clone, PartialEq, Eq)]
pub struct Imported {
    pub name: String,
    pub host: String,
    pub agent: String,
    token: String,
}

impl Imported {
    /// The token, for the caller that stores it.
    pub fn into_token(self) -> String {
        self.token
    }

    pub fn token(&self) -> &str {
        &self.token
    }
}

impl fmt::Debug for Imported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Imported").field("name", &self.name).field("host", &self.host).field("agent", &self.agent).field("token", &"<redacted>").finish()
    }
}

/// `zc-codex/config.json` inside the config folder `base`.
pub fn config_path_in(base: &Path) -> PathBuf {
    base.join("zc-codex").join("config.json")
}

/// `~/.config/zc-codex/config.json` (the config folder of the platform). Under test nothing is read from the real folder: the path is a missing file unless a test sets one.
pub fn default_path() -> Option<PathBuf> {
    #[cfg(test)]
    {
        return Some(test_path::get().unwrap_or_else(|| std::env::temp_dir().join(format!("ame-zc-no-codex-config-{}", std::process::id())).join("config.json")));
    }
    #[cfg(not(test))]
    directories::BaseDirs::new().map(|d| config_path_in(d.config_dir()))
}

#[cfg(test)]
pub(crate) mod test_path {
    use std::path::PathBuf;
    use std::sync::Mutex;

    static PATH: Mutex<Option<PathBuf>> = Mutex::new(None);

    pub fn get() -> Option<PathBuf> {
        PATH.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Makes [`default_path`](super::default_path) return `path` (or the missing file again with `None`).
    pub fn set(path: Option<PathBuf>) {
        *PATH.lock().unwrap_or_else(|e| e.into_inner()) = path;
    }
}

/// Reads the config at `path`.
pub fn import_from(path: &Path) -> Result<Imported, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    parse(&text)
}

fn text_of<'a>(m: &'a Value, key: &str) -> &'a str {
    m.get(key).and_then(Value::as_str).unwrap_or("").trim()
}

fn usable(m: &Value) -> Option<Imported> {
    let (host, token) = (text_of(m, "host"), text_of(m, "token"));
    (text_of(m, "kind") == "ZeroClaw" && !host.is_empty() && !token.is_empty()).then(|| {
        let name = text_of(m, "name");
        Imported {
            name: if name.is_empty() { host.to_string() } else { name.to_string() },
            host: host.to_string(),
            agent: text_of(m, "agent").to_string(),
            token: token.to_string(),
        }
    })
}

/// The selected machine when it is a ZeroClaw entry, otherwise the first usable ZeroClaw entry.
pub fn parse(text: &str) -> Result<Imported, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| format!("not a ZC Codex config: {e}"))?;
    let machines = v.get("machines").and_then(Value::as_array).ok_or("the config lists no machines")?;
    let selected = match v.get("selected") {
        Some(Value::Number(n)) => n.as_u64().and_then(|i| machines.get(i as usize)),
        Some(Value::String(name)) => machines.iter().find(|m| m.get("name").and_then(Value::as_str) == Some(name.as_str())),
        _ => None,
    };
    selected.and_then(usable).or_else(|| machines.iter().find_map(usable)).ok_or_else(|| "the config has no ZeroClaw machine with a host and a token".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("fixtures/zc-codex-config.json");

    fn fixture_file(name: &str, text: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ame-zc-import-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        std::fs::write(&path, text).unwrap();
        path
    }

    #[test]
    fn the_selected_zeroclaw_machine_wins() {
        let path = fixture_file("selected", FIXTURE);
        let m = import_from(&path).unwrap();
        assert_eq!((m.name.as_str(), m.host.as_str(), m.agent.as_str()), ("claw (public)", "wss://claw.example.test", "media"));
        assert_eq!(m.token(), "fixture-token-selected-public");
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn a_selected_codex_machine_falls_back_to_the_first_zeroclaw_one() {
        let text = FIXTURE.replace("\"selected\": 2", "\"selected\": 0");
        let m = parse(&text).unwrap();
        assert_eq!((m.name.as_str(), m.host.as_str(), m.agent.as_str()), ("zeroclaw", "100.96.121.37:42617", "companion"));
    }

    #[test]
    fn a_machine_can_be_selected_by_name() {
        let text = FIXTURE.replace("\"selected\": 2", "\"selected\": \"zeroclaw\"");
        assert_eq!(parse(&text).unwrap().agent, "companion");
    }

    #[test]
    fn no_selection_takes_the_first_zeroclaw_machine() {
        let text = FIXTURE.replace("\"selected\": 2,", "");
        assert_eq!(parse(&text).unwrap().name, "zeroclaw");
    }

    #[test]
    fn entries_without_a_host_or_token_are_skipped() {
        let text = r#"{"machines":[{"name":"a","host":"","token":"t","kind":"ZeroClaw"},{"name":"b","host":"h","token":"","kind":"ZeroClaw"}]}"#;
        assert!(parse(text).unwrap_err().contains("no ZeroClaw machine"));
    }

    #[test]
    fn bad_files_are_errors() {
        assert!(parse("not json").is_err());
        assert!(parse("{}").unwrap_err().contains("no machines"));
        assert!(import_from(&std::env::temp_dir().join("ame-zc-import-missing-file.json")).is_err());
    }

    #[test]
    fn the_token_is_never_printed() {
        let m = parse(FIXTURE).unwrap();
        let shown = format!("{m:?}");
        assert!(!shown.contains("fixture-token"), "{shown}");
        assert!(shown.contains("<redacted>"));
        assert_eq!(m.clone().into_token(), "fixture-token-selected-public");
    }

    #[test]
    fn the_config_path_is_inside_the_config_folder() {
        assert_eq!(config_path_in(Path::new("/home/u/.config")), PathBuf::from("/home/u/.config/zc-codex/config.json"));
    }

    #[test]
    fn under_test_the_default_path_never_points_at_the_real_config() {
        let real = directories::BaseDirs::new().map(|d| config_path_in(d.config_dir()));
        assert_ne!(default_path(), real);
    }
}
