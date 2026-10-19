//! Bearer token storage and comparison, `Host` and `Origin` allow-lists and the request rate limiter.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use base64::Engine as _;
use subtle::ConstantTimeEq;

use super::clock::Clock;

pub const TOKEN_FILE: &str = "mcp_token";
const TOKEN_BYTES: usize = 32;

/// Creates `dir` (mode 0700 on Unix) when it does not exist.
pub fn ensure_private_dir(dir: &Path) -> std::io::Result<()> {
    if dir.is_dir() {
        return Ok(());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        std::fs::DirBuilder::new().recursive(true).mode(0o700).create(dir)
    }
    #[cfg(not(unix))]
    {
        std::fs::create_dir_all(dir)
    }
}

/// A new token: 32 random bytes as unpadded base64url.
pub fn generate_token() -> std::io::Result<String> {
    let mut bytes = [0u8; TOKEN_BYTES];
    getrandom::fill(&mut bytes).map_err(|e| std::io::Error::other(format!("no random bytes: {e}")))?;
    Ok(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes))
}

/// True for a string `generate_token` could have produced.
pub fn is_well_formed(token: &str) -> bool {
    token.len() == 43 && token.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

pub fn token_path(dir: &Path) -> PathBuf {
    dir.join(TOKEN_FILE)
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

/// The stored token, creating the file (mode 0600) on first use or when its content is not a token.
pub fn load_or_create_token(dir: &Path) -> std::io::Result<String> {
    let path = token_path(dir);
    if let Ok(text) = std::fs::read_to_string(&path) {
        let token = text.trim();
        if is_well_formed(token) {
            return Ok(token.to_string());
        }
    }
    replace_token(dir)
}

/// Writes a fresh token over the stored one and returns it.
pub fn replace_token(dir: &Path) -> std::io::Result<String> {
    ensure_private_dir(dir)?;
    let token = generate_token()?;
    write_private(&token_path(dir), &token)?;
    Ok(token)
}

/// Compares two secrets in time that does not depend on where they differ.
pub fn tokens_equal(a: &str, b: &str) -> bool {
    a.as_bytes().ct_eq(b.as_bytes()).into()
}

/// The token of an `Authorization: Bearer <token>` header value.
pub fn bearer_token(header: &str) -> Option<&str> {
    let (scheme, token) = header.trim().split_once(' ')?;
    (scheme.eq_ignore_ascii_case("bearer") && !token.trim().is_empty()).then(|| token.trim())
}

/// First characters of a token for display.
pub fn mask(token: &str) -> String {
    let shown: String = token.chars().take(4).collect();
    format!("{shown}{}", "*".repeat(token.chars().count().saturating_sub(4).min(12)))
}

/// Why a request was turned away.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reject {
    BadHost,
    BadOrigin,
    MissingToken,
    BadToken,
    RateLimited { retry_after: Duration },
    TooLarge,
}

impl Reject {
    pub fn status(&self) -> u16 {
        match self {
            Self::BadHost | Self::BadOrigin => 403,
            Self::MissingToken | Self::BadToken => 401,
            Self::RateLimited { .. } => 429,
            Self::TooLarge => 413,
        }
    }

    pub fn message(&self) -> &'static str {
        match self {
            Self::BadHost => "host not allowed",
            Self::BadOrigin => "origin not allowed",
            Self::MissingToken => "bearer token required",
            Self::BadToken => "bearer token rejected",
            Self::RateLimited { .. } => "too many requests",
            Self::TooLarge => "request too large",
        }
    }
}

/// Host (lower case, no brackets) and port of an authority like `host:80`, `[::1]:80` or `::1`.
fn split_authority(authority: &str) -> (String, Option<String>) {
    let a = authority.trim().to_ascii_lowercase();
    if let Some(rest) = a.strip_prefix('[') {
        let (host, after) = rest.split_once(']').unwrap_or((rest, ""));
        return (host.to_string(), after.strip_prefix(':').filter(|p| !p.is_empty()).map(str::to_string));
    }
    match a.matches(':').count() {
        0 => (a, None),
        1 => {
            let (host, port) = a.split_once(':').unwrap_or((&a, ""));
            (host.to_string(), Some(port.to_string()).filter(|p| !p.is_empty()))
        }
        _ => (a, None),
    }
}

/// Accepted `Host` and `Origin` values.
#[derive(Clone, Debug)]
pub struct HostPolicy {
    hosts: Vec<String>,
    origins: Vec<String>,
}

impl HostPolicy {
    pub fn new(hosts: Vec<String>, origins: Vec<String>) -> Self {
        Self { hosts: hosts.into_iter().map(|h| h.trim().to_ascii_lowercase()).collect(), origins: origins.into_iter().map(|o| o.trim().trim_end_matches('/').to_ascii_lowercase()).collect() }
    }

    /// A `Host` value passes when an entry names its host and either has no port or the same port.
    pub fn check_host(&self, host: Option<&str>) -> Result<(), Reject> {
        let host = host.map(str::trim).filter(|h| !h.is_empty()).ok_or(Reject::BadHost)?;
        let (request_host, request_port) = split_authority(host);
        let ok = self.hosts.iter().any(|allowed| {
            let (allowed_host, allowed_port) = split_authority(allowed);
            allowed_host == request_host && (allowed_port.is_none() || allowed_port == request_port)
        });
        if ok { Ok(()) } else { Err(Reject::BadHost) }
    }

    /// Requests without an `Origin` pass; one with an `Origin` must be listed.
    pub fn check_origin(&self, origin: Option<&str>) -> Result<(), Reject> {
        let Some(origin) = origin.map(str::trim) else { return Ok(()) };
        let o = origin.trim_end_matches('/').to_ascii_lowercase();
        if self.origins.iter().any(|allowed| *allowed == o) { Ok(()) } else { Err(Reject::BadOrigin) }
    }
}

/// A token bucket: `burst` requests at once, refilled at `per_minute`.
pub struct RateLimiter {
    burst: f64,
    per_sec: f64,
    clock: Arc<dyn Clock>,
    state: Mutex<(f64, Instant)>,
}

impl RateLimiter {
    pub fn new(per_minute: u32, burst: u32, clock: Arc<dyn Clock>) -> Self {
        let burst = f64::from(burst.max(1));
        let now = clock.now();
        Self { burst, per_sec: f64::from(per_minute.max(1)) / 60.0, clock, state: Mutex::new((burst, now)) }
    }

    /// Takes one request from the bucket, or says how long until one is free.
    pub fn take(&self) -> Result<(), Duration> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let now = self.clock.now();
        let elapsed = now.saturating_duration_since(state.1).as_secs_f64();
        state.0 = (state.0 + elapsed * self.per_sec).min(self.burst);
        state.1 = now;
        if state.0 >= 1.0 {
            state.0 -= 1.0;
            return Ok(());
        }
        Err(Duration::from_secs_f64(((1.0 - state.0) / self.per_sec).max(0.001)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::mcp::clock::FakeClock;

    #[test]
    fn tokens_are_43_url_safe_characters_and_differ() {
        let a = generate_token().unwrap();
        let b = generate_token().unwrap();
        assert_ne!(a, b);
        assert!(is_well_formed(&a), "{a}");
        assert_eq!(base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(&a).unwrap().len(), 32);
        assert!(!is_well_formed("short"));
        assert!(!is_well_formed(&"a".repeat(44)));
        assert!(!is_well_formed(&format!("{}+", "a".repeat(42))));
    }

    #[test]
    fn the_token_file_is_created_once_and_reused() {
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("smart-media");
        let first = load_or_create_token(&sub).unwrap();
        assert_eq!(load_or_create_token(&sub).unwrap(), first);
        assert_eq!(std::fs::read_to_string(token_path(&sub)).unwrap(), first);
        std::fs::write(token_path(&sub), format!("  {first}\n")).unwrap();
        assert_eq!(load_or_create_token(&sub).unwrap(), first, "surrounding whitespace is ignored");
    }

    #[cfg(unix)]
    #[test]
    fn the_token_file_and_folder_are_private() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("smart-media");
        load_or_create_token(&sub).unwrap();
        assert_eq!(std::fs::metadata(token_path(&sub)).unwrap().permissions().mode() & 0o777, 0o600);
        assert_eq!(std::fs::metadata(&sub).unwrap().permissions().mode() & 0o777, 0o700);
        std::fs::set_permissions(token_path(&sub), std::fs::Permissions::from_mode(0o644)).unwrap();
        let rotated = replace_token(&sub).unwrap();
        assert_eq!(std::fs::metadata(token_path(&sub)).unwrap().permissions().mode() & 0o777, 0o600);
        assert_eq!(std::fs::read_to_string(token_path(&sub)).unwrap(), rotated);
    }

    #[test]
    fn a_damaged_token_file_is_replaced() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(token_path(dir.path()), "not a token").unwrap();
        let token = load_or_create_token(dir.path()).unwrap();
        assert!(is_well_formed(&token));
    }

    #[test]
    fn the_comparison_accepts_only_equal_secrets() {
        let t = generate_token().unwrap();
        assert!(tokens_equal(&t, &t.clone()));
        assert!(!tokens_equal(&t, &format!("{}x", &t[..42])));
        assert!(!tokens_equal(&t, &t[..42]));
        assert!(!tokens_equal(&t, ""));
        assert!(tokens_equal("", ""));
    }

    #[test]
    fn bearer_headers_parse() {
        assert_eq!(bearer_token("Bearer abc"), Some("abc"));
        assert_eq!(bearer_token("bearer   abc "), Some("abc"));
        assert_eq!(bearer_token("BEARER abc"), Some("abc"));
        assert_eq!(bearer_token("Basic abc"), None);
        assert_eq!(bearer_token("Bearer"), None);
        assert_eq!(bearer_token("Bearer "), None);
        assert_eq!(bearer_token(""), None);
    }

    #[test]
    fn the_mask_hides_most_of_the_token() {
        let m = mask("abcdefghijklmnopqrstuvwxyz");
        assert!(m.starts_with("abcd") && !m.contains("efgh"));
        assert_eq!(mask("ab"), "ab");
    }

    fn policy(hosts: &[&str], origins: &[&str]) -> HostPolicy {
        HostPolicy::new(hosts.iter().map(|s| s.to_string()).collect(), origins.iter().map(|s| s.to_string()).collect())
    }

    #[test]
    fn loopback_hosts_pass_with_or_without_a_port() {
        let p = policy(&["localhost", "127.0.0.1", "::1"], &[]);
        for ok in ["localhost", "localhost:9871", "LOCALHOST:1", "127.0.0.1:9871", "127.0.0.1", "[::1]:9871", "[::1]", "::1"] {
            assert_eq!(p.check_host(Some(ok)), Ok(()), "{ok}");
        }
        for bad in ["evil.com", "evil.com:9871", "127.0.0.1.evil.com", "localhost.evil.com:80", "192.168.1.5:9871", "10.0.0.1", ""] {
            assert_eq!(p.check_host(Some(bad)), Err(Reject::BadHost), "{bad}");
        }
        assert_eq!(p.check_host(None), Err(Reject::BadHost));
    }

    #[test]
    fn an_entry_with_a_port_matches_only_that_port() {
        let p = policy(&["localhost", "desk.tail.ts.net:9871", "100.64.1.2"], &[]);
        assert_eq!(p.check_host(Some("desk.tail.ts.net:9871")), Ok(()));
        assert_eq!(p.check_host(Some("desk.tail.ts.net:9999")), Err(Reject::BadHost));
        assert_eq!(p.check_host(Some("desk.tail.ts.net")), Err(Reject::BadHost));
        assert_eq!(p.check_host(Some("100.64.1.2:55")), Ok(()));
        assert_eq!(p.check_host(Some("100.64.1.3")), Err(Reject::BadHost));
    }

    #[test]
    fn origins_are_rejected_unless_listed() {
        let none = policy(&[], &[]);
        assert_eq!(none.check_origin(None), Ok(()));
        assert_eq!(none.check_origin(Some("https://evil.com")), Err(Reject::BadOrigin));
        assert_eq!(none.check_origin(Some("null")), Err(Reject::BadOrigin));
        let listed = policy(&[], &["https://app.example.com:443/"]);
        assert_eq!(listed.check_origin(Some("https://APP.example.com:443")), Ok(()));
        assert_eq!(listed.check_origin(Some("https://app.example.com")), Err(Reject::BadOrigin));
    }

    #[test]
    fn reject_reasons_map_to_http_statuses() {
        let cases = [(Reject::BadHost, 403), (Reject::BadOrigin, 403), (Reject::MissingToken, 401), (Reject::BadToken, 401), (Reject::RateLimited { retry_after: Duration::from_secs(1) }, 429), (Reject::TooLarge, 413)];
        for (r, status) in cases {
            assert_eq!(r.status(), status, "{r:?}");
            assert!(!r.message().is_empty());
        }
    }

    #[test]
    fn the_bucket_allows_a_burst_then_refills_with_time() {
        let clock = FakeClock::new();
        let limiter = RateLimiter::new(60, 3, clock.clone());
        for _ in 0..3 {
            assert!(limiter.take().is_ok());
        }
        let wait = limiter.take().unwrap_err();
        assert!(wait > Duration::from_millis(900) && wait <= Duration::from_secs(1), "{wait:?}");
        clock.advance(Duration::from_millis(500));
        assert!(limiter.take().is_err(), "half a token is not enough");
        clock.advance(Duration::from_millis(600));
        assert!(limiter.take().is_ok());
        assert!(limiter.take().is_err());
        clock.advance(Duration::from_secs(3600));
        for _ in 0..3 {
            assert!(limiter.take().is_ok(), "the bucket caps at the burst");
        }
        assert!(limiter.take().is_err());
    }
}
