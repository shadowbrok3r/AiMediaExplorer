//! Gateway usage per session and per day: each finished turn is added as it arrives and the totals are kept in a small JSON file.

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use serde::{Deserialize, Serialize};
use zeroclaw_client::Usage;

use super::events::now_ms;

pub const FILE: &str = "zc_usage.json";
/// Sessions kept, newest activity first.
const MAX_SESSIONS: usize = 500;
/// Days kept, newest first.
const MAX_DAYS: usize = 800;

/// Sums over a set of turns.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Totals {
    pub turns: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost_usd: f64,
}

impl Totals {
    fn add(&mut self, usage: &Usage) {
        self.turns += 1;
        self.input_tokens += usage.input_tokens.unwrap_or(0);
        self.output_tokens += usage.output_tokens.unwrap_or(0);
        self.cost_usd += usage.cost_usd.unwrap_or(0.0).max(0.0);
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SessionUsage {
    pub session: String,
    pub first_ms: u64,
    pub last_ms: u64,
    /// The model of the newest turn.
    pub model: Option<String>,
    #[serde(flatten)]
    pub totals: Totals,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DayUsage {
    /// Local date as `YYYY-MM-DD`.
    pub day: String,
    #[serde(flatten)]
    pub totals: Totals,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct UsageData {
    /// When counting started, as Unix milliseconds; zero before the first turn.
    pub since_ms: u64,
    pub total: Totals,
    /// Newest activity first.
    pub sessions: Vec<SessionUsage>,
    /// Newest day first.
    pub days: Vec<DayUsage>,
}

/// The local date of a Unix-millisecond stamp.
pub fn day_of(at_ms: u64) -> String {
    chrono::DateTime::<chrono::Utc>::from_timestamp_millis(at_ms as i64).map(|t| t.with_timezone(&chrono::Local).format("%Y-%m-%d").to_string()).unwrap_or_default()
}

pub struct UsageStore {
    path: Option<PathBuf>,
    data: Mutex<UsageData>,
}

impl UsageStore {
    /// A store that keeps nothing on disk.
    pub fn in_memory() -> Self {
        Self { path: None, data: Mutex::new(UsageData::default()) }
    }

    /// A store backed by `path`; unreadable files start empty.
    pub fn at(path: PathBuf) -> Self {
        let data = std::fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str::<UsageData>(&t).ok()).unwrap_or_default();
        Self { path: Some(path), data: Mutex::new(data) }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, UsageData> {
        self.data.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn path(&self) -> Option<&std::path::Path> {
        self.path.as_deref()
    }

    pub fn snapshot(&self) -> UsageData {
        self.lock().clone()
    }

    /// Adds one finished turn of `session`.
    pub fn record_at(&self, at_ms: u64, session: &str, usage: &Usage) {
        let mut data = self.lock();
        if data.since_ms == 0 {
            data.since_ms = at_ms;
        }
        data.total.add(usage);
        match data.sessions.iter_mut().find(|s| s.session == session) {
            Some(s) => {
                s.totals.add(usage);
                s.last_ms = s.last_ms.max(at_ms);
                s.model = usage.model.clone().or(s.model.take());
            }
            None => {
                let mut s = SessionUsage { session: session.to_string(), first_ms: at_ms, last_ms: at_ms, model: usage.model.clone(), totals: Totals::default() };
                s.totals.add(usage);
                data.sessions.push(s);
            }
        }
        data.sessions.sort_by(|a, b| b.last_ms.cmp(&a.last_ms));
        data.sessions.truncate(MAX_SESSIONS);
        let day = day_of(at_ms);
        match data.days.iter_mut().find(|d| d.day == day) {
            Some(d) => d.totals.add(usage),
            None => {
                let mut d = DayUsage { day, totals: Totals::default() };
                d.totals.add(usage);
                data.days.push(d);
            }
        }
        data.days.sort_by(|a, b| b.day.cmp(&a.day));
        data.days.truncate(MAX_DAYS);
        self.save(&data);
    }

    /// Forgets everything counted so far.
    pub fn reset(&self) {
        let mut data = self.lock();
        *data = UsageData::default();
        self.save(&data);
    }

    fn save(&self, data: &UsageData) {
        let Some(path) = &self.path else { return };
        let written = (|| -> std::io::Result<()> {
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir)?;
            }
            let part = path.with_extension("json.part");
            std::fs::write(&part, serde_json::to_vec_pretty(data).map_err(std::io::Error::other)?)?;
            std::fs::rename(&part, path)
        })();
        if let Err(e) = written {
            log::warn!("[zc] usage not saved to {}: {e}", path.display());
        }
    }
}

/// The file the usage is kept in: next to the other settings files, or a private one under test.
fn default_path() -> Option<PathBuf> {
    if cfg!(test) {
        return Some(std::env::temp_dir().join(format!("ame-zc-usage-{}.json", std::process::id())));
    }
    crate::ai::mcp::config::default_dir().map(|d| d.join(FILE))
}

static STORE: OnceLock<UsageStore> = OnceLock::new();

/// The usage of the application.
pub fn store() -> &'static UsageStore {
    STORE.get_or_init(|| default_path().map(UsageStore::at).unwrap_or_else(UsageStore::in_memory))
}

/// Adds a finished turn of `session` to the usage of the application.
pub fn record(session: &str, usage: &Usage) {
    store().record_at(now_ms(), session, usage);
}

/// Number with thousands separators.
pub fn grouped(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// A cost as dollars; small amounts keep four decimals.
pub fn dollars(cost: f64) -> String {
    if cost > 0.0 && cost < 0.01 { format!("${cost:.4}") } else { format!("${cost:.2}") }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(model: &str, input: u64, output: u64, cost: f64) -> Usage {
        Usage { input_tokens: Some(input), output_tokens: Some(output), cost_usd: Some(cost), model: Some(model.into()), ..Default::default() }
    }

    const DAY: u64 = 48 * 3600 * 1000;
    const T0: u64 = 1_760_000_000_000;

    #[test]
    fn turns_add_up_per_session_per_day_and_in_total() {
        let store = UsageStore::in_memory();
        store.record_at(T0, "a", &usage("m1", 100, 20, 0.5));
        store.record_at(T0 + 1000, "a", &usage("m2", 10, 2, 0.25));
        store.record_at(T0 + 2000, "b", &usage("m1", 1, 1, 0.0));
        store.record_at(T0 + DAY, "a", &usage("m2", 7, 3, 0.1));
        let d = store.snapshot();
        assert_eq!(d.since_ms, T0);
        assert_eq!((d.total.turns, d.total.input_tokens, d.total.output_tokens), (4, 118, 26));
        assert!((d.total.cost_usd - 0.85).abs() < 1e-9);
        assert_eq!(d.sessions.len(), 2);
        let a = d.sessions.iter().find(|s| s.session == "a").unwrap();
        assert_eq!((a.totals.turns, a.totals.input_tokens, a.first_ms, a.last_ms), (3, 117, T0, T0 + DAY));
        assert_eq!(a.model.as_deref(), Some("m2"), "the newest model");
        assert_eq!(d.sessions[0].session, "a", "newest activity first");
        assert_eq!(d.days.len(), 2);
        assert_eq!(d.days[0].day, day_of(T0 + DAY), "newest day first");
        assert_eq!(d.days[0].totals.turns, 1);
        assert_eq!((d.days[1].totals.turns, d.days[1].totals.input_tokens), (3, 111));
        assert_eq!(d.days.iter().map(|d| d.totals.turns).sum::<u64>(), d.total.turns);
    }

    #[test]
    fn missing_figures_count_as_zero_and_never_go_negative() {
        let store = UsageStore::in_memory();
        store.record_at(T0, "a", &Usage::default());
        store.record_at(T0, "a", &Usage { cost_usd: Some(-3.0), ..Default::default() });
        let d = store.snapshot();
        assert_eq!((d.total.turns, d.total.input_tokens, d.total.cost_usd), (2, 0, 0.0));
        assert_eq!(d.sessions[0].model, None);
    }

    #[test]
    fn the_lists_are_bounded_and_keep_the_newest() {
        let store = UsageStore::in_memory();
        for i in 0..MAX_SESSIONS as u64 + 20 {
            store.record_at(T0 + i * 1000, &format!("s{i}"), &usage("m", 1, 1, 0.0));
        }
        let d = store.snapshot();
        assert_eq!(d.sessions.len(), MAX_SESSIONS);
        assert_eq!(d.sessions[0].session, format!("s{}", MAX_SESSIONS + 19));
        assert!(!d.sessions.iter().any(|s| s.session == "s0"));
        assert_eq!(d.total.turns, MAX_SESSIONS as u64 + 20, "the total keeps every turn");
    }

    #[test]
    fn reset_forgets_everything() {
        let store = UsageStore::in_memory();
        store.record_at(T0, "a", &usage("m", 5, 5, 1.0));
        store.reset();
        assert_eq!(store.snapshot(), UsageData::default());
        store.record_at(T0 + 5, "a", &usage("m", 1, 1, 0.0));
        assert_eq!(store.snapshot().since_ms, T0 + 5);
    }

    #[test]
    fn totals_survive_a_restart_and_a_damaged_file_starts_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sub").join(FILE);
        let store = UsageStore::at(path.clone());
        store.record_at(T0, "a", &usage("m", 10, 4, 0.2));
        assert!(path.is_file() && !path.with_extension("json.part").exists());
        let back = UsageStore::at(path.clone());
        assert_eq!(back.snapshot(), store.snapshot());
        back.record_at(T0 + 1, "a", &usage("m", 1, 1, 0.0));
        assert_eq!(back.snapshot().total.turns, 2);
        std::fs::write(&path, "{ not json").unwrap();
        assert_eq!(UsageStore::at(path).snapshot(), UsageData::default());
    }

    #[test]
    fn a_file_with_missing_fields_still_reads() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(FILE);
        std::fs::write(&path, r#"{"total":{"turns":3},"sessions":[{"session":"x"}]}"#).unwrap();
        let d = UsageStore::at(path).snapshot();
        assert_eq!((d.total.turns, d.sessions[0].session.as_str(), d.sessions[0].totals.turns), (3, "x", 0));
    }

    #[test]
    fn numbers_and_costs_read_well() {
        assert_eq!([grouped(0), grouped(999), grouped(1000), grouped(1234567)], ["0", "999", "1,000", "1,234,567"]);
        assert_eq!((dollars(0.0), dollars(0.00123), dollars(1.5), dollars(12.345)), ("$0.00".to_string(), "$0.0012".to_string(), "$1.50".to_string(), "$12.35".to_string()));
    }

    #[test]
    fn the_global_store_records_turns() {
        let before = store().snapshot().total.turns;
        record("global-test", &usage("m", 1, 1, 0.0));
        assert!(store().snapshot().total.turns > before);
        assert!(store().snapshot().sessions.iter().any(|s| s.session == "global-test"));
    }
}
