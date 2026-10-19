//! Settings of the MCP server: a JSON file next to the token, command-line overrides and the address rules.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

pub const DEFAULT_PORT: u16 = 9871;
pub const CONFIG_FILE: &str = "mcp_server.json";
const LOOPBACK_HOSTS: [&str; 3] = ["localhost", "127.0.0.1", "::1"];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct McpConfig {
    /// Start the server with the application.
    pub enabled: bool,
    pub port: u16,
    /// Bind `bind_addr` instead of loopback; needs `allowed_hosts`.
    pub allow_tailnet: bool,
    /// Interface address to bind when `allow_tailnet` is set.
    pub bind_addr: String,
    /// `Host` values accepted besides loopback when `allow_tailnet` is set.
    pub allowed_hosts: Vec<String>,
    /// Browser origins accepted; empty rejects every request that carries an `Origin`.
    pub allowed_origins: Vec<String>,
    /// Folders tools may read besides the library roots and the Pictures and Videos folders.
    pub watched_folders: Vec<String>,
    pub rate_per_minute: u32,
    pub rate_burst: u32,
    pub max_body_bytes: usize,
    /// Jobs allowed to wait for the GPU.
    pub gpu_queue_depth: usize,
    pub gpu_job_timeout_secs: u64,
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            port: DEFAULT_PORT,
            allow_tailnet: false,
            bind_addr: String::new(),
            allowed_hosts: Vec::new(),
            allowed_origins: Vec::new(),
            watched_folders: Vec::new(),
            rate_per_minute: 240,
            rate_burst: 60,
            max_body_bytes: 1024 * 1024,
            gpu_queue_depth: 8,
            gpu_job_timeout_secs: 300,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConfigError {
    BadBindAddr(String),
    Unspecified,
    NoAllowedHosts,
    BadHost(String),
    BadOrigin(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadBindAddr(a) => write!(f, "'{a}' is not an interface address"),
            Self::Unspecified => write!(f, "bind a specific interface address, not 0.0.0.0 or ::"),
            Self::NoAllowedHosts => write!(f, "tailnet access needs at least one allowed host name or address"),
            Self::BadHost(h) => write!(f, "'{h}' is not a host name or address"),
            Self::BadOrigin(o) => write!(f, "'{o}' is not an origin like https://host:port"),
        }
    }
}

impl std::error::Error for ConfigError {}

/// Where the server listens and which `Host` and `Origin` values it accepts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Binding {
    pub addr: SocketAddr,
    pub allowed_hosts: Vec<String>,
    pub allowed_origins: Vec<String>,
}

fn valid_host(host: &str) -> bool {
    let h = host.trim();
    !h.is_empty() && h.len() <= 255 && h.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | ':' | '[' | ']' | '_'))
}

fn valid_origin(origin: &str) -> bool {
    let rest = origin.strip_prefix("https://").or_else(|| origin.strip_prefix("http://"));
    rest.is_some_and(|r| !r.is_empty() && !r.contains('/') && !r.contains(' '))
}

impl McpConfig {
    /// The listening address and accepted names; loopback unless tailnet access is on.
    pub fn binding(&self) -> Result<Binding, ConfigError> {
        for o in &self.allowed_origins {
            if !valid_origin(o) {
                return Err(ConfigError::BadOrigin(o.clone()));
            }
        }
        let mut hosts: Vec<String> = LOOPBACK_HOSTS.iter().map(|h| h.to_string()).collect();
        let ip = if self.allow_tailnet {
            let ip: IpAddr = self.bind_addr.trim().parse().map_err(|_| ConfigError::BadBindAddr(self.bind_addr.clone()))?;
            if ip.is_unspecified() {
                return Err(ConfigError::Unspecified);
            }
            let extra: Vec<String> = self.allowed_hosts.iter().map(|h| h.trim().to_string()).filter(|h| !h.is_empty()).collect();
            if extra.is_empty() {
                return Err(ConfigError::NoAllowedHosts);
            }
            if let Some(bad) = extra.iter().find(|h| !valid_host(h)) {
                return Err(ConfigError::BadHost(bad.clone()));
            }
            hosts.extend(extra);
            ip
        } else {
            IpAddr::V4(Ipv4Addr::LOCALHOST)
        };
        Ok(Binding { addr: SocketAddr::new(ip, self.port), allowed_hosts: hosts, allowed_origins: self.allowed_origins.clone() })
    }
}

/// The folder holding the token and the settings file.
pub fn default_dir() -> Option<PathBuf> {
    directories::BaseDirs::new().map(|d| d.config_dir().join("smart-media"))
}

/// Settings read from `dir`; a missing file gives the defaults and an unreadable one the defaults plus a warning.
pub fn load(dir: &Path) -> (McpConfig, Option<String>) {
    let path = dir.join(CONFIG_FILE);
    match std::fs::read_to_string(&path) {
        Ok(text) => match serde_json::from_str::<McpConfig>(&text) {
            Ok(cfg) => (cfg, None),
            Err(e) => (McpConfig::default(), Some(format!("{} is not valid, using defaults: {e}", path.display()))),
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (McpConfig::default(), None),
        Err(e) => (McpConfig::default(), Some(format!("{} cannot be read, using defaults: {e}", path.display()))),
    }
}

/// Writes the settings through a temporary file so a crash never leaves half a file.
pub fn save(dir: &Path, cfg: &McpConfig) -> std::io::Result<()> {
    super::auth::ensure_private_dir(dir)?;
    let tmp = dir.join(format!("{CONFIG_FILE}.tmp"));
    std::fs::write(&tmp, serde_json::to_vec_pretty(cfg).map_err(std::io::Error::other)?)?;
    std::fs::rename(&tmp, dir.join(CONFIG_FILE))
}

/// Start-up choices from the command line and the environment.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CliOverrides {
    /// `--mcp-port N` (or `SMART_MEDIA_MCP_PORT`): serve on this port for this run.
    pub port: Option<u16>,
    /// `--mcp-config-dir <dir>`: read and write the token and settings there.
    pub config_dir: Option<PathBuf>,
}

fn flag_value<'a>(args: &'a [String], name: &str) -> Result<Option<&'a str>, String> {
    let mut found = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        if let Some(v) = arg.strip_prefix(&format!("{name}=")) {
            found = Some(v);
        } else if arg == name {
            found = Some(it.next().map(String::as_str).ok_or_else(|| format!("{name} needs a value"))?);
        }
    }
    Ok(found)
}

/// Reads `--mcp-port` and `--mcp-config-dir` from `args` and the port from `env_port`.
pub fn parse_cli(args: &[String], env_port: Option<&str>) -> Result<CliOverrides, String> {
    let port_text = flag_value(args, "--mcp-port")?.or(env_port);
    let port = port_text.map(|t| t.trim().parse::<u16>().map_err(|_| format!("--mcp-port: '{t}' is not a port number"))).transpose()?;
    let config_dir = flag_value(args, "--mcp-config-dir")?.map(PathBuf::from);
    Ok(CliOverrides { port, config_dir })
}

impl CliOverrides {
    /// The settings with the command-line port applied; a port turns the server on.
    pub fn apply(&self, mut cfg: McpConfig) -> McpConfig {
        if let Some(port) = self.port {
            cfg.port = port;
            cfg.enabled = true;
        }
        cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|a| a.to_string()).collect()
    }

    #[test]
    fn the_default_binds_loopback_on_the_default_port() {
        let b = McpConfig::default().binding().unwrap();
        assert_eq!(b.addr, "127.0.0.1:9871".parse::<SocketAddr>().unwrap());
        assert_eq!(b.allowed_hosts, vec!["localhost", "127.0.0.1", "::1"]);
        assert!(b.allowed_origins.is_empty());
    }

    #[test]
    fn extra_hosts_and_a_bind_address_do_nothing_without_the_tailnet_opt_in() {
        let cfg = McpConfig { bind_addr: "100.64.1.2".into(), allowed_hosts: vec!["desk.tail.ts.net".into()], ..Default::default() };
        let b = cfg.binding().unwrap();
        assert!(b.addr.ip().is_loopback());
        assert_eq!(b.allowed_hosts.len(), 3);
    }

    #[test]
    fn tailnet_access_binds_the_interface_and_adds_the_hosts() {
        let cfg = McpConfig { allow_tailnet: true, bind_addr: "100.64.1.2".into(), allowed_hosts: vec!["desk.tail.ts.net".into(), "100.64.1.2:9871".into()], port: 9014, ..Default::default() };
        let b = cfg.binding().unwrap();
        assert_eq!(b.addr, "100.64.1.2:9014".parse::<SocketAddr>().unwrap());
        assert!(b.allowed_hosts.contains(&"desk.tail.ts.net".to_string()) && b.allowed_hosts.contains(&"100.64.1.2:9871".to_string()));
        assert!(b.allowed_hosts.contains(&"localhost".to_string()));
    }

    #[test]
    fn tailnet_access_is_refused_without_an_interface_or_hosts() {
        let base = McpConfig { allow_tailnet: true, ..Default::default() };
        assert_eq!(base.binding().unwrap_err(), ConfigError::BadBindAddr(String::new()));
        let wildcard = McpConfig { bind_addr: "0.0.0.0".into(), allowed_hosts: vec!["h".into()], ..base.clone() };
        assert_eq!(wildcard.binding().unwrap_err(), ConfigError::Unspecified);
        let v6 = McpConfig { bind_addr: "::".into(), allowed_hosts: vec!["h".into()], ..base.clone() };
        assert_eq!(v6.binding().unwrap_err(), ConfigError::Unspecified);
        let none = McpConfig { bind_addr: "100.64.1.2".into(), allowed_hosts: vec!["  ".into()], ..base.clone() };
        assert_eq!(none.binding().unwrap_err(), ConfigError::NoAllowedHosts);
        let bad = McpConfig { bind_addr: "100.64.1.2".into(), allowed_hosts: vec!["http://x/y".into()], ..base };
        assert!(matches!(bad.binding().unwrap_err(), ConfigError::BadHost(_)));
    }

    #[test]
    fn origins_need_a_scheme_and_no_path() {
        let ok = McpConfig { allowed_origins: vec!["https://app.example.com:443".into()], ..Default::default() };
        assert_eq!(ok.binding().unwrap().allowed_origins.len(), 1);
        for bad in ["app.example.com", "https://a/b", "ftp://a"] {
            let cfg = McpConfig { allowed_origins: vec![bad.into()], ..Default::default() };
            assert_eq!(cfg.binding().unwrap_err(), ConfigError::BadOrigin(bad.into()), "{bad}");
        }
    }

    #[test]
    fn settings_round_trip_through_the_file_and_missing_fields_default() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(load(dir.path()), (McpConfig::default(), None));
        let cfg = McpConfig { enabled: true, port: 9999, watched_folders: vec!["/data/pics".into()], ..Default::default() };
        save(dir.path(), &cfg).unwrap();
        assert_eq!(load(dir.path()).0, cfg);
        std::fs::write(dir.path().join(CONFIG_FILE), r#"{"port": 1234}"#).unwrap();
        let (partial, warning) = load(dir.path());
        assert_eq!((partial.port, partial.enabled, partial.rate_burst), (1234, false, 60));
        assert!(warning.is_none());
    }

    #[test]
    fn a_broken_file_falls_back_to_defaults_with_a_warning() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(CONFIG_FILE), "{ nope").unwrap();
        let (cfg, warning) = load(dir.path());
        assert_eq!(cfg, McpConfig::default());
        assert!(warning.unwrap().contains("not valid"));
    }

    #[test]
    fn the_command_line_port_turns_the_server_on() {
        let cli = parse_cli(&args(&["--theme", "x.json", "--mcp-port", "9100", "--mcp-config-dir=/tmp/c"]), None).unwrap();
        assert_eq!(cli, CliOverrides { port: Some(9100), config_dir: Some(PathBuf::from("/tmp/c")) });
        let cfg = cli.apply(McpConfig::default());
        assert!(cfg.enabled);
        assert_eq!(cfg.port, 9100);
        assert_eq!(parse_cli(&[], Some("9200")).unwrap().port, Some(9200));
        assert_eq!(parse_cli(&args(&["--mcp-port=0"]), Some("9200")).unwrap().port, Some(0));
        assert_eq!(parse_cli(&[], None).unwrap(), CliOverrides::default());
        assert!(!CliOverrides::default().apply(McpConfig::default()).enabled);
    }

    #[test]
    fn bad_command_line_values_name_the_flag() {
        assert!(parse_cli(&args(&["--mcp-port", "abc"]), None).unwrap_err().contains("--mcp-port"));
        assert!(parse_cli(&args(&["--mcp-port", "70000"]), None).unwrap_err().contains("70000"));
        assert!(parse_cli(&args(&["--mcp-port"]), None).unwrap_err().contains("needs a value"));
        assert!(parse_cli(&args(&["--mcp-config-dir"]), None).is_err());
    }
}
