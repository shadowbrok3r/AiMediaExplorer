//! State dump: a JSON snapshot of the tabs, tasks, engines and settings summary for tests and smoke scripts. Keys and tokens never go into it.

use std::path::Path;
use std::sync::LazyLock;
use std::time::Instant;

use regex::Regex;
use serde::Serialize;

use crate::UiSettings;
use crate::app::SmartMediaApp;
use crate::settings::ui_prefs::UiPrefs;
use crate::status::StatusMeta;
use crate::tasks::Task;
use crate::ui::file_table::FileExplorer;
use crate::ui::file_table::table::columns::Col;
use crate::ui::file_table::table::view_mode::ViewMode;
use crate::ui::tabs::{BrowserId, TabKind};

/// Version of the dump layout.
pub const VERSION: u32 = 1;
/// Most tasks listed, newest first.
const MAX_TASKS: usize = 60;
/// Frames that must have run before an idle dump is written.
pub const MIN_FRAMES: u64 = 8;
/// Longest wait for the application to settle before an idle dump is written anyway.
pub const IDLE_TIMEOUT_SECS: u64 = 30;
/// Consecutive quiet frames that count as idle.
pub const QUIET_FRAMES: u32 = 3;
const MASK: &str = "[redacted]";

#[derive(Debug, Serialize)]
pub struct BuildInfo {
    pub name: &'static str,
    pub version: &'static str,
    pub profile: &'static str,
    pub os: &'static str,
    pub arch: &'static str,
    pub features: Vec<&'static str>,
}

impl BuildInfo {
    pub fn current() -> Self {
        let mut features = Vec::new();
        if cfg!(feature = "cpu") {
            features.push("cpu");
        }
        if cfg!(ame_ffmpeg) {
            features.push("native-video");
        }
        Self {
            name: env!("CARGO_PKG_NAME"),
            version: env!("CARGO_PKG_VERSION"),
            profile: if cfg!(debug_assertions) { "debug" } else { "release" },
            os: std::env::consts::OS,
            arch: std::env::consts::ARCH,
            features,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct SortDump {
    pub column: String,
    pub descending: bool,
}

#[derive(Debug, Serialize)]
pub struct QueryDump {
    pub mode: String,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct TabDump {
    pub id: u64,
    pub title: String,
    pub scope: String,
    /// The browser the commands and the inspector act on.
    pub active: bool,
    /// Rows held by the tab.
    pub rows: usize,
    /// Rows the filters let through, as of the last frame the tab was drawn.
    pub filtered: usize,
    pub selected: usize,
    pub view: &'static str,
    pub tile: f32,
    pub sort: Vec<SortDump>,
    pub query: QueryDump,
    /// The tab's filter spec as compact JSON.
    pub filter: String,
    pub filters_active: usize,
    pub loading: bool,
    /// Why the tab has not read its scope: `lazy`, `waiting`, `offline` or `search`.
    pub pending: Option<String>,
    pub error: Option<String>,
    /// The tab shows rows held in memory, which the workspace file cannot bring back.
    pub rows_only: bool,
}

#[derive(Debug, Serialize)]
pub struct TaskDump {
    pub id: u64,
    pub kind: &'static str,
    pub title: String,
    pub state: String,
    pub done: u64,
    pub total: u64,
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct EngineDump {
    pub name: &'static str,
    pub state: String,
    pub model: Option<&'static str>,
    pub detail: String,
    pub progress: [u64; 2],
    pub error: Option<String>,
}

/// The settings that are safe to show: switches, model names, counts and which providers have a key. No key, token or password is in it.
#[derive(Debug, Serialize)]
pub struct SettingsSummary {
    pub loaded: bool,
    pub auto_indexing: bool,
    pub auto_clip_embeddings: bool,
    pub auto_save_to_database: bool,
    pub overwrite_descriptions: bool,
    pub clip_model: Option<String>,
    pub reranker_model: Option<String>,
    pub vision_model_dir: Option<String>,
    pub ai_chat_provider: Option<String>,
    pub chat_model: Option<String>,
    /// Endpoint without credentials or query.
    pub chat_endpoint: Option<String>,
    /// Providers that have a key stored; the keys themselves are not shown.
    pub providers_with_key: Vec<&'static str>,
    pub min_size_bytes: Option<u64>,
    pub max_size_bytes: Option<u64>,
    pub excluded_dirs: usize,
    pub excluded_extensions: usize,
    pub recent_paths: usize,
    pub recent_models: usize,
    pub mcp_stdio_enabled: bool,
    pub zeroclaw_configured: bool,
    pub restore_tabs: bool,
    pub restore_last_scope: bool,
    pub watched_folders: usize,
}

/// `url` without user info, query and fragment.
pub fn strip_url(url: &str) -> String {
    let (scheme, rest) = url.split_once("://").map_or(("", url), |(s, r)| (s, r));
    let rest = rest.split(['?', '#']).next().unwrap_or_default();
    let rest = rest.rsplit_once('@').map_or(rest, |(_, host)| host);
    if scheme.is_empty() { rest.to_string() } else { format!("{scheme}://{rest}") }
}

impl SettingsSummary {
    pub fn from(settings: &UiSettings, loaded: bool, prefs: &UiPrefs) -> Self {
        let keys = [
            ("openai", &settings.openai_api_key),
            ("grok", &settings.grok_api_key),
            ("gemini", &settings.gemini_api_key),
            ("groq", &settings.groq_api_key),
            ("openrouter", &settings.openrouter_api_key),
        ];
        Self {
            loaded,
            auto_indexing: settings.auto_indexing,
            auto_clip_embeddings: settings.auto_clip_embeddings,
            auto_save_to_database: settings.auto_save_to_database,
            overwrite_descriptions: settings.overwrite_descriptions,
            clip_model: settings.clip_model.clone(),
            reranker_model: settings.reranker_model.clone(),
            vision_model_dir: settings.joycaption_model_dir.clone(),
            ai_chat_provider: settings.ai_chat_provider.clone(),
            chat_model: settings.openai_default_model.clone(),
            chat_endpoint: settings.openai_base_url.as_deref().map(strip_url),
            providers_with_key: keys.into_iter().filter(|(_, key)| key.as_deref().is_some_and(|k| !k.trim().is_empty())).map(|(name, _)| name).collect(),
            min_size_bytes: settings.db_min_size_bytes,
            max_size_bytes: settings.db_max_size_bytes,
            excluded_dirs: settings.excluded_dirs.as_ref().map_or(0, Vec::len),
            excluded_extensions: settings.db_excluded_exts.as_ref().map_or(0, Vec::len),
            recent_paths: settings.recent_paths.len(),
            recent_models: settings.recent_models.len(),
            mcp_stdio_enabled: settings.mcp_stdio_enabled.unwrap_or(false),
            zeroclaw_configured: settings.zeroclaw.as_ref().is_some_and(|z| !z.host.trim().is_empty()),
            restore_tabs: prefs.startup.restore_tabs,
            restore_last_scope: prefs.startup.restore_last_scope,
            watched_folders: prefs.library.watched.len(),
        }
    }
}

/// Every secret the settings hold, for masking wherever their text could turn up.
pub fn secrets_of(settings: &UiSettings) -> Vec<String> {
    [&settings.openai_api_key, &settings.grok_api_key, &settings.gemini_api_key, &settings.groq_api_key, &settings.openrouter_api_key, &settings.openai_organization]
        .into_iter()
        .filter_map(|v| v.as_deref())
        .map(str::trim)
        .filter(|v| v.len() >= 4)
        .map(str::to_string)
        .collect()
}

static PATTERNS: LazyLock<Vec<(Regex, &'static str)>> = LazyLock::new(|| {
    [
        (r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}", "Bearer [redacted]"),
        (r"\b(?:sk|gsk|xai|pk|rk|hf)[-_][A-Za-z0-9_-]{12,}", MASK),
        (r"\bAIza[A-Za-z0-9_-]{20,}", MASK),
        (r#"(?i)\b(api[_-]?key|token|secret|password|passwd|authorization)(["']?\s*[:=]\s*["']?)[^\s"',;&]+"#, "$1$2[redacted]"),
        (r"://[^/\s@]+@", "://[redacted]@"),
    ]
    .into_iter()
    .filter_map(|(pattern, with)| Regex::new(pattern).ok().map(|r| (r, with)))
    .collect()
});

/// `text` with every secret in `secrets` and everything that looks like a key, token or password masked.
pub fn redact(text: &str, secrets: &[String]) -> String {
    let mut out = text.to_string();
    for secret in secrets.iter().filter(|s| !s.is_empty()) {
        out = out.replace(secret.as_str(), MASK);
    }
    for (pattern, with) in PATTERNS.iter() {
        out = pattern.replace_all(&out, *with).into_owned();
    }
    out
}

/// The whole snapshot.
#[derive(Debug, Serialize)]
pub struct StateDump {
    pub version: u32,
    pub build: BuildInfo,
    pub frames: u64,
    pub database_ready: bool,
    pub active_tab: Option<String>,
    pub tabs: Vec<TabDump>,
    /// Titles of the dock tabs that are not browsers.
    pub other_tabs: Vec<String>,
    pub tasks: Vec<TaskDump>,
    pub engines: Vec<EngineDump>,
    pub settings: SettingsSummary,
}

fn pending_name(ex: &FileExplorer) -> Option<String> {
    use crate::ui::file_table::scope::PendingKind;
    ex.nav.pending.as_ref().map(|p| {
        match p.kind {
            PendingKind::Lazy => "lazy",
            PendingKind::Waiting => "waiting",
            PendingKind::Offline(_) => "offline",
            PendingKind::Search => "search",
        }
        .to_string()
    })
}

fn tab_dump(id: BrowserId, ex: &FileExplorer, active: bool, secrets: &[String]) -> TabDump {
    let spec = ex.filter_spec();
    let offline = match ex.nav.pending.as_ref().map(|p| &p.kind) {
        Some(crate::ui::file_table::scope::PendingKind::Offline(reason)) => Some(reason.clone()),
        _ => None,
    };
    TabDump {
        id: id.0,
        title: redact(&ex.tab_title(), secrets),
        scope: redact(&ex.scope().format(), secrets),
        active,
        rows: ex.table.len(),
        filtered: ex.view.len(),
        selected: ex.selection_count(),
        view: match ex.browse.mode {
            ViewMode::Grid => "grid",
            ViewMode::Details => "details",
        },
        tile: ex.browse.tile,
        sort: ex.view.sort().keys().iter().map(|k| SortDump { column: Col::from_id(k.column).map_or_else(|| k.column.to_string(), |c| c.menu_name().to_string()), descending: k.descending }).collect(),
        query: QueryDump { mode: format!("{:?}", ex.toolbar.query.mode).to_lowercase(), text: redact(&ex.toolbar.query.text, secrets) },
        filter: redact(&serde_json::to_string(&spec).unwrap_or_default(), secrets),
        filters_active: spec.chips().len(),
        loading: ex.is_loading(),
        pending: pending_name(ex),
        error: ex.nav.error.as_ref().or(offline.as_ref()).map(|e| redact(e, secrets)),
        rows_only: ex.nav.rows_only,
    }
}

fn task_dump(task: &Task, secrets: &[String]) -> TaskDump {
    TaskDump {
        id: task.id,
        kind: task.kind.label(),
        title: redact(&task.title, secrets),
        state: format!("{:?}", task.state).to_lowercase(),
        done: task.done,
        total: task.total,
        error: task.error.as_ref().map(|e| redact(e, secrets)),
    }
}

fn engine_dump(meta: &StatusMeta, secrets: &[String]) -> EngineDump {
    EngineDump {
        name: meta.name,
        state: format!("{:?}", meta.state).to_lowercase(),
        model: meta.model,
        detail: redact(&meta.detail, secrets),
        progress: [meta.progress_current, meta.progress_total],
        error: meta.error.as_ref().map(|e| redact(e, secrets)),
    }
}

/// The snapshot of `app` with `settings` and `prefs` summarised; `loaded` says whether `settings` came from the library.
pub fn build(app: &SmartMediaApp, frames: u64, settings: &UiSettings, loaded: bool, prefs: &UiPrefs) -> StateDump {
    let secrets = secrets_of(settings);
    let ctx = &app.context;
    let active = ctx.active_kind();
    let mut tabs = Vec::new();
    let mut other_tabs = Vec::new();
    for (_, kind) in app.tree.iter_all_tabs() {
        match kind {
            TabKind::Browser { id } => {
                let explorer = if *id == BrowserId::MAIN { Some(&ctx.file_explorer) } else { ctx.filtered_tabs.get(id) };
                if let Some(ex) = explorer {
                    tabs.push(tab_dump(*id, ex, active == *kind, &secrets));
                }
            }
            other => other_tabs.push(other.fixed_title().to_string()),
        }
    }
    let active_tab = tabs.iter().find(|t| t.active).map(|t| t.scope.clone());
    let tasks = crate::tasks::with(|r| r.tasks().iter().rev().take(MAX_TASKS).map(|t| task_dump(t, &secrets)).collect());
    StateDump {
        version: VERSION,
        build: BuildInfo::current(),
        frames,
        database_ready: crate::database::is_ready(),
        active_tab,
        tabs,
        other_tabs,
        tasks,
        engines: crate::status::engine_snapshots().iter().map(|m| engine_dump(m, &secrets)).collect(),
        settings: SettingsSummary::from(settings, loaded, prefs),
    }
}

/// Writes `dump` to `path` through a temporary file; the folder is created when missing.
pub fn write(path: &Path, dump: &StateDump) -> std::io::Result<usize> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let text = serde_json::to_string_pretty(dump).map_err(std::io::Error::other)?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, &text)?;
    std::fs::rename(&tmp, path)?;
    Ok(text.len())
}

/// A dump waiting for the application to settle.
#[derive(Debug)]
pub struct Request {
    pub path: String,
    pub since: Instant,
    pub quiet: u32,
}

/// What the idle check looks at.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Activity {
    pub frames: u64,
    /// A browser is reading its scope, or waiting for the library to read it.
    pub loading: bool,
}

impl Request {
    pub fn new(path: String, now: Instant) -> Self {
        Self { path, since: now, quiet: 0 }
    }

    /// True when the dump should be written this frame: enough frames ran and nothing has been loading for a few frames, or the wait ran out.
    pub fn ready(&mut self, activity: Activity, now: Instant) -> bool {
        if now.saturating_duration_since(self.since).as_secs() >= IDLE_TIMEOUT_SECS {
            return true;
        }
        if activity.frames < MIN_FRAMES || activity.loading {
            self.quiet = 0;
            return false;
        }
        self.quiet += 1;
        self.quiet >= QUIET_FRAMES
    }
}

/// True while a browser that is in front of its leaf is reading its scope or waiting for the library to read it; a tab behind another is not drawn and does not move on.
pub fn is_loading(app: &SmartMediaApp) -> bool {
    use crate::ui::file_table::scope::PendingKind;
    let busy = |ex: &FileExplorer| ex.is_loading() || matches!(ex.nav.pending.as_ref().map(|p| &p.kind), Some(PendingKind::Waiting));
    app.tree.iter_leaves().filter_map(|(_, leaf)| leaf.tabs.get(leaf.active.0).and_then(TabKind::browser_id)).any(|id| {
        let explorer = if id == BrowserId::MAIN { Some(&app.context.file_explorer) } else { app.context.filtered_tabs.get(&id) };
        explorer.is_some_and(busy)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn fake_settings() -> UiSettings {
        let mut s = UiSettings::default();
        s.openai_api_key = Some("sk-FAKEOPENAIKEY1234567890abcdef".into());
        s.grok_api_key = Some("xai-FAKEGROKKEY1234567890abcdef".into());
        s.gemini_api_key = Some("AIzaFAKEGEMINIKEY1234567890abcdef".into());
        s.groq_api_key = Some("gsk_FAKEGROQKEY1234567890abcdef".into());
        s.openrouter_api_key = Some("or-FAKEOPENROUTERKEY".into());
        s.openai_organization = Some("org-FAKEORG123456".into());
        s.openai_base_url = Some("https://user:hunter2pass@llm.example.com/v1?key=queryKEY999#frag".into());
        s.openai_default_model = Some("gpt-test".into());
        s.ai_chat_provider = Some("openai".into());
        s.excluded_dirs = Some(vec!["/a".into(), "/b".into()]);
        s
    }

    fn json_of(summary: &SettingsSummary) -> String {
        serde_json::to_string(summary).unwrap()
    }

    fn assert_no_secret(text: &str, settings: &UiSettings) {
        for secret in secrets_of(settings).iter().chain(["hunter2pass".to_string(), "queryKEY999".to_string()].iter()) {
            assert!(!text.contains(secret.as_str()), "{secret} leaked into {text}");
        }
    }

    #[test]
    fn the_summary_names_the_providers_with_a_key_and_never_the_keys() {
        let settings = fake_settings();
        let summary = SettingsSummary::from(&settings, true, &UiPrefs::default());
        let text = json_of(&summary);
        assert_no_secret(&text, &settings);
        assert_eq!(summary.providers_with_key, vec!["openai", "grok", "gemini", "groq", "openrouter"]);
        assert_eq!(summary.chat_endpoint.as_deref(), Some("https://llm.example.com/v1"));
        assert_eq!(summary.excluded_dirs, 2);
        assert!(summary.loaded);
    }

    #[test]
    fn the_summary_has_no_field_that_could_hold_a_secret() {
        let value = serde_json::to_value(SettingsSummary::from(&fake_settings(), true, &UiPrefs::default())).unwrap();
        for key in value.as_object().unwrap().keys() {
            let lower = key.to_lowercase();
            assert!(!["api_key", "apikey", "token", "secret", "password", "organization"].iter().any(|bad| lower.contains(bad)), "field {key}");
        }
    }

    #[test]
    fn every_secret_field_of_the_settings_is_covered_by_the_masking() {
        let settings = fake_settings();
        let record = serde_json::to_value(&settings).unwrap();
        let mut secret_values = Vec::new();
        for (name, value) in record.as_object().unwrap() {
            let lower = name.to_lowercase();
            if ["key", "token", "secret", "password", "organization"].iter().any(|bad| lower.contains(bad)) {
                if let Some(text) = value.as_str() {
                    secret_values.push((name.clone(), text.to_string()));
                }
            }
        }
        assert!(secret_values.len() >= 6, "{secret_values:?}");
        let masked_secrets = secrets_of(&settings);
        for (name, value) in secret_values {
            assert!(masked_secrets.contains(&value), "{name} is not in the list of secrets that are masked");
            assert!(!redact(&format!("failed with {value}"), &masked_secrets).contains(&value), "{name}");
        }
    }

    #[test]
    fn text_that_carries_a_key_is_masked_even_when_the_key_is_not_known() {
        let none: Vec<String> = Vec::new();
        for (text, secret) in [
            ("401 Incorrect API key provided: sk-ABCDEFGHIJKLMNOPQRSTUV", "sk-ABCDEFGHIJKLMNOPQRSTUV"),
            ("Authorization: Bearer abcdef0123456789abcdef", "abcdef0123456789abcdef"),
            ("request to https://bob:swordfish@host/x failed", "swordfish"),
            ("config api_key=\"zzzz1111\" rejected", "zzzz1111"),
            ("token: ghp_looks_like_a_token_value", "ghp_looks_like_a_token_value"),
            ("bad key AIzaSyA-1234567890abcdefghijkl", "AIzaSyA-1234567890abcdefghijkl"),
        ] {
            let out = redact(text, &none);
            assert!(!out.contains(secret), "{text} -> {out}");
            assert!(out.contains("redacted"), "{out}");
        }
        assert_eq!(redact("plain text with no secrets", &none), "plain text with no secrets");
        assert_eq!(redact("/home/u/Pictures/trip 2024", &none), "/home/u/Pictures/trip 2024");
    }

    #[test]
    fn urls_lose_their_credentials_query_and_fragment() {
        assert_eq!(strip_url("https://a:b@host:8080/v1?k=1#f"), "https://host:8080/v1");
        assert_eq!(strip_url("http://localhost:1234/v1"), "http://localhost:1234/v1");
        assert_eq!(strip_url("host:99/v1?x=y"), "host:99/v1");
        assert_eq!(strip_url(""), "");
    }

    #[test]
    fn a_dump_waits_for_enough_quiet_frames() {
        let t0 = Instant::now();
        let mut request = Request::new("/tmp/x.json".into(), t0);
        let busy = Activity { frames: 100, loading: true };
        let early = Activity { frames: 2, loading: false };
        let idle = Activity { frames: 100, loading: false };
        assert!(!request.ready(early, t0));
        assert!(!request.ready(busy, t0));
        assert!(!request.ready(idle, t0));
        assert!(!request.ready(idle, t0));
        assert!(request.ready(idle, t0), "the third quiet frame in a row");
        let mut flaky = Request::new("/tmp/x.json".into(), t0);
        assert!(!flaky.ready(idle, t0) && !flaky.ready(idle, t0));
        assert!(!flaky.ready(busy, t0), "loading starts the count over");
        assert!(!flaky.ready(idle, t0));
    }

    #[test]
    fn a_dump_is_written_anyway_when_the_application_never_settles() {
        let t0 = Instant::now();
        let mut request = Request::new("/tmp/x.json".into(), t0);
        let busy = Activity { frames: 1000, loading: true };
        assert!(!request.ready(busy, t0 + Duration::from_secs(IDLE_TIMEOUT_SECS - 1)));
        assert!(request.ready(busy, t0 + Duration::from_secs(IDLE_TIMEOUT_SECS)));
    }

    #[test]
    fn the_dump_is_written_through_a_temporary_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("state.json");
        let dump = StateDump {
            version: VERSION,
            build: BuildInfo::current(),
            frames: 5,
            database_ready: false,
            active_tab: None,
            tabs: Vec::new(),
            other_tabs: vec!["Logs".into()],
            tasks: Vec::new(),
            engines: Vec::new(),
            settings: SettingsSummary::from(&UiSettings::default(), false, &UiPrefs::default()),
        };
        let bytes = write(&path, &dump).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(bytes, text.len());
        assert!(!path.with_extension("json.tmp").exists());
        let value: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(value["version"], 1);
        assert_eq!(value["build"]["name"], "smart-media");
        assert_eq!(value["other_tabs"][0], "Logs");
    }
}
