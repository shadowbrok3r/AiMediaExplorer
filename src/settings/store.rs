//! Authoritative in-memory settings with patch updates, a generation counter and debounced merge saves.

use std::collections::{BTreeSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use async_trait::async_trait;
use once_cell::sync::Lazy;

use crate::database::settings::{
    SETTINGS_CACHE, UiSettings, changed_fields, copy_fields, persisted_json,
};

/// Quiet time after the last change before dirty fields are written.
pub const DEBOUNCE: Duration = Duration::from_millis(400);
const RETRY_DELAY: Duration = Duration::from_secs(3);
const MAX_RETRIES: u32 = 5;
const PATCH_LOG_CAP: usize = 64;
const VALUE_TEXT_CAP: usize = 96;

/// Settings read from storage; `existed` is false when no record was stored yet.
pub struct Loaded {
    pub settings: UiSettings,
    pub existed: bool,
}

/// Storage the store loads from and merges patches into.
#[async_trait]
pub trait Backend: Send + Sync + 'static {
    async fn load(&self) -> anyhow::Result<Loaded>;
    async fn merge(&self, fields: &[&'static str], settings: &UiSettings) -> anyhow::Result<()>;
}

/// SurrealDB-backed storage.
pub struct DbBackend;

#[async_trait]
impl Backend for DbBackend {
    async fn load(&self) -> anyhow::Result<Loaded> {
        let (settings, existed) = crate::database::settings::get_settings_checked().await?;
        Ok(Loaded { settings, existed })
    }

    async fn merge(&self, fields: &[&'static str], settings: &UiSettings) -> anyhow::Result<()> {
        crate::database::settings::merge_settings_in_db(fields, settings).await
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoadState {
    Pending,
    Loaded,
    Failed(String),
}

#[derive(Clone, Debug)]
pub struct FieldChange {
    pub field: String,
    pub old: String,
    pub new: String,
}

#[derive(Clone, Debug)]
pub struct PatchRecord {
    pub generation: u64,
    pub at: String,
    pub who: String,
    pub changes: Vec<FieldChange>,
}

#[derive(Clone, Debug, Default)]
pub struct SaveStatus {
    pub last_ok_at: Option<String>,
    pub last_error: Option<String>,
    pub batches_saved: u64,
}

struct State {
    settings: UiSettings,
    load: LoadState,
}

pub struct SettingsStore {
    state: RwLock<State>,
    generation: AtomicU64,
    dirty: Mutex<BTreeSet<&'static str>>,
    patches: Mutex<VecDeque<PatchRecord>>,
    save: Mutex<SaveStatus>,
    backend: Arc<dyn Backend>,
    flush_gate: tokio::sync::Mutex<()>,
    flush_scheduled: AtomicBool,
    retries: AtomicU32,
    load_event: AtomicBool,
    banner_dismissed: AtomicBool,
    debounce: Duration,
}

fn now_text() -> String {
    chrono::Local::now().format("%H:%M:%S%.3f").to_string()
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// True for fields holding credentials.
pub fn is_secret_field(field: &str) -> bool {
    field.ends_with("_api_key")
}

fn describe_value(field: &str, value: Option<&serde_json::Value>) -> String {
    let Some(value) = value else { return "none".into() };
    if value.is_null() {
        return "none".into();
    }
    if is_secret_field(field) {
        return "<redacted>".into();
    }
    let mut text = value.to_string();
    if text.chars().count() > VALUE_TEXT_CAP {
        text = text.chars().take(VALUE_TEXT_CAP).collect::<String>() + "...";
    }
    text
}

impl SettingsStore {
    pub fn new(backend: Arc<dyn Backend>) -> Arc<Self> {
        Self::with_debounce(backend, DEBOUNCE)
    }

    pub fn with_debounce(backend: Arc<dyn Backend>, debounce: Duration) -> Arc<Self> {
        Arc::new(Self {
            state: RwLock::new(State { settings: UiSettings::default(), load: LoadState::Pending }),
            generation: AtomicU64::new(0),
            dirty: Mutex::new(BTreeSet::new()),
            patches: Mutex::new(VecDeque::new()),
            save: Mutex::new(SaveStatus::default()),
            backend,
            flush_gate: tokio::sync::Mutex::new(()),
            flush_scheduled: AtomicBool::new(false),
            retries: AtomicU32::new(0),
            load_event: AtomicBool::new(false),
            banner_dismissed: AtomicBool::new(false),
            debounce,
        })
    }

    /// Counter bumped by every applied change, load and load failure.
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    pub fn load_state(&self) -> LoadState {
        self.state.read().unwrap_or_else(|e| e.into_inner()).load.clone()
    }

    pub fn is_loaded(&self) -> bool {
        self.load_state() == LoadState::Loaded
    }

    /// Failure text when the stored settings could not be read.
    pub fn load_failure(&self) -> Option<String> {
        match self.load_state() {
            LoadState::Failed(e) => Some(e),
            _ => None,
        }
    }

    pub fn read<R>(&self, f: impl FnOnce(&UiSettings) -> R) -> R {
        f(&self.state.read().unwrap_or_else(|e| e.into_inner()).settings)
    }

    pub fn snapshot(&self) -> UiSettings {
        self.read(UiSettings::clone)
    }

    /// Snapshot and the generation it belongs to.
    pub fn snapshot_with_generation(&self) -> (UiSettings, u64) {
        let st = self.state.read().unwrap_or_else(|e| e.into_inner());
        (st.settings.clone(), self.generation())
    }

    /// The settings once hydrated from storage, otherwise `None`.
    pub fn snapshot_loaded(&self) -> Option<UiSettings> {
        let st = self.state.read().unwrap_or_else(|e| e.into_inner());
        (st.load == LoadState::Loaded).then(|| st.settings.clone())
    }

    /// Applies `f` and records only the persisted fields it changed; returns whether anything changed.
    pub fn update(self: &Arc<Self>, who: &str, f: impl FnOnce(&mut UiSettings)) -> bool {
        let (fields, generation, loaded) = {
            let mut st = self.state.write().unwrap_or_else(|e| e.into_inner());
            let before = st.settings.clone();
            f(&mut st.settings);
            let fields = changed_fields(&before, &st.settings);
            if fields.is_empty() {
                return false;
            }
            let generation = self.generation.fetch_add(1, Ordering::AcqRel) + 1;
            self.log_patch(generation, who, &before, &st.settings, &fields);
            lock(&self.dirty).extend(fields.iter().copied());
            let loaded = st.load == LoadState::Loaded;
            if loaded {
                *lock(&SETTINGS_CACHE) = Some(st.settings.clone());
            }
            (fields, generation, loaded)
        };
        log::debug!("[settings] gen {generation} {who}: {}", fields.join(","));
        if loaded {
            self.schedule_flush(self.debounce);
        }
        true
    }

    /// Applies the fields of `src` that differ from the live settings.
    pub fn merge_from(self: &Arc<Self>, who: &str, src: &UiSettings) -> bool {
        self.update(who, |cur| {
            let fields = changed_fields(cur, src);
            copy_fields(cur, src, &fields);
        })
    }

    /// Replaces the live settings with freshly read ones, keeping fields edited before the load finished.
    pub fn apply_loaded(self: &Arc<Self>, who: &str, loaded: Loaded) {
        let pending: Vec<&'static str> = lock(&self.dirty).iter().copied().collect();
        {
            let mut st = self.state.write().unwrap_or_else(|e| e.into_inner());
            let before = st.settings.clone();
            let mut next = loaded.settings;
            if !pending.is_empty() {
                copy_fields(&mut next, &before, &pending);
            }
            let fields = changed_fields(&before, &next);
            let generation = self.generation.fetch_add(1, Ordering::AcqRel) + 1;
            let note = if loaded.existed { who.to_string() } else { format!("{who} (no stored record)") };
            self.log_patch(generation, &note, &before, &next, &fields);
            st.settings = next;
            st.load = LoadState::Loaded;
            *lock(&SETTINGS_CACHE) = Some(st.settings.clone());
        }
        self.load_event.store(true, Ordering::Release);
        self.banner_dismissed.store(false, Ordering::Release);
        if !pending.is_empty() {
            self.schedule_flush(self.debounce);
        }
    }

    /// Marks the stored settings as unreadable; saving stays disabled until a load succeeds.
    pub fn set_load_failed(&self, who: &str, error: String) {
        let mut st = self.state.write().unwrap_or_else(|e| e.into_inner());
        st.load = LoadState::Failed(error.clone());
        let generation = self.generation.fetch_add(1, Ordering::AcqRel) + 1;
        let mut log = lock(&self.patches);
        log.push_back(PatchRecord {
            generation,
            at: now_text(),
            who: who.to_string(),
            changes: vec![FieldChange { field: "<load>".into(), old: "-".into(), new: format!("failed: {error}") }],
        });
        while log.len() > PATCH_LOG_CAP {
            log.pop_front();
        }
        self.banner_dismissed.store(false, Ordering::Release);
    }

    /// Reads the stored settings; a read failure is kept as the load state and returned.
    pub async fn reload(self: &Arc<Self>) -> anyhow::Result<()> {
        match self.backend.load().await {
            Ok(loaded) => {
                self.apply_loaded("load", loaded);
                Ok(())
            }
            Err(e) => {
                let text = format!("{e:#}");
                log::error!("[settings] load failed: {text}");
                self.set_load_failed("load", text);
                Err(e)
            }
        }
    }

    /// Writes the dirty fields as a merge; returns how many fields were written.
    pub async fn flush(self: &Arc<Self>) -> anyhow::Result<usize> {
        let _gate = self.flush_gate.lock().await;
        let (fields, snapshot) = {
            let st = self.state.read().unwrap_or_else(|e| e.into_inner());
            match &st.load {
                LoadState::Loaded => {}
                LoadState::Pending => anyhow::bail!("settings are not loaded yet; changes stay in memory"),
                LoadState::Failed(e) => anyhow::bail!("settings failed to load ({e}); saving is disabled"),
            }
            let mut dirty = lock(&self.dirty);
            if dirty.is_empty() {
                return Ok(0);
            }
            let fields: Vec<&'static str> = std::mem::take(&mut *dirty).into_iter().collect();
            (fields, st.settings.clone())
        };
        match self.backend.merge(&fields, &snapshot).await {
            Ok(()) => {
                let mut status = lock(&self.save);
                status.last_ok_at = Some(now_text());
                status.last_error = None;
                status.batches_saved += 1;
                self.retries.store(0, Ordering::Release);
                Ok(fields.len())
            }
            Err(e) => {
                lock(&self.dirty).extend(fields.iter().copied());
                lock(&self.save).last_error = Some(format!("{e:#}"));
                Err(e)
            }
        }
    }

    /// Marks every persisted field dirty and flushes immediately.
    pub async fn force_save(self: &Arc<Self>) -> anyhow::Result<usize> {
        lock(&self.dirty).extend(crate::database::settings::PERSISTED_FIELDS.iter().copied());
        self.flush().await
    }

    fn schedule_flush(self: &Arc<Self>, delay: Duration) {
        if self.flush_scheduled.swap(true, Ordering::AcqRel) {
            return;
        }
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            self.flush_scheduled.store(false, Ordering::Release);
            return;
        };
        let this = Arc::clone(self);
        handle.spawn(async move {
            tokio::time::sleep(delay).await;
            this.flush_scheduled.store(false, Ordering::Release);
            if let Err(e) = this.flush().await {
                log::error!("[settings] save failed: {e:#}");
                let attempt = this.retries.fetch_add(1, Ordering::AcqRel);
                if attempt == 0 {
                    crate::ui::dialogs::toast(egui_toast::ToastKind::Error, format!("Settings could not be saved: {e:#}"));
                }
                if this.is_loaded() && attempt < MAX_RETRIES {
                    this.schedule_flush(RETRY_DELAY);
                }
            }
        });
    }

    fn log_patch(&self, generation: u64, who: &str, before: &UiSettings, after: &UiSettings, fields: &[&'static str]) {
        let (old, new) = (persisted_json(before), persisted_json(after));
        let changes = fields
            .iter()
            .map(|f| FieldChange {
                field: (*f).to_string(),
                old: describe_value(f, old.get(*f)),
                new: describe_value(f, new.get(*f)),
            })
            .collect();
        let mut log = lock(&self.patches);
        log.push_back(PatchRecord { generation, at: now_text(), who: who.to_string(), changes });
        while log.len() > PATCH_LOG_CAP {
            log.pop_front();
        }
    }

    pub fn dirty_fields(&self) -> Vec<&'static str> {
        lock(&self.dirty).iter().copied().collect()
    }

    /// Most recent patches, newest last.
    pub fn patches(&self) -> Vec<PatchRecord> {
        lock(&self.patches).iter().cloned().collect()
    }

    pub fn save_status(&self) -> SaveStatus {
        lock(&self.save).clone()
    }

    /// True once after each successful load.
    pub fn take_load_event(&self) -> bool {
        self.load_event.swap(false, Ordering::AcqRel)
    }

    pub fn dismiss_banner(&self) {
        self.banner_dismissed.store(true, Ordering::Release);
    }

    pub fn banner_dismissed(&self) -> bool {
        self.banner_dismissed.load(Ordering::Acquire)
    }

    /// Persisted fields as JSON with credentials replaced by a redaction marker.
    pub fn json_redacted(&self) -> serde_json::Value {
        let mut map = self.read(persisted_json);
        for (key, value) in map.iter_mut() {
            if is_secret_field(key) && !value.is_null() {
                let len = value.as_str().map(|s| s.chars().count()).unwrap_or(0);
                *value = serde_json::Value::String(format!("<redacted, {len} chars>"));
            }
        }
        serde_json::Value::Object(map)
    }
}

static STORE: Lazy<Arc<SettingsStore>> = Lazy::new(|| SettingsStore::new(Arc::new(DbBackend)));

/// The process-wide settings store.
pub fn store() -> &'static Arc<SettingsStore> {
    &STORE
}

/// Detects store changes by polling its generation.
pub struct Watcher {
    store: Arc<SettingsStore>,
    seen: u64,
}

impl Watcher {
    pub fn new() -> Self {
        Self::of(store())
    }

    pub fn of(store: &Arc<SettingsStore>) -> Self {
        Self { store: Arc::clone(store), seen: store.generation() }
    }

    /// True when the store changed since the last call.
    pub fn changed(&mut self) -> bool {
        let g = self.store.generation();
        if g == self.seen {
            return false;
        }
        self.seen = g;
        true
    }
}

impl Default for Watcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Fresh copy of the live settings and its generation, for read-only mirrors.
pub fn new_mirror() -> (UiSettings, u64) {
    store().snapshot_with_generation()
}

/// Refreshes `mirror` when the store moved past `seen`; true if it was replaced.
pub fn sync_mirror(mirror: &mut UiSettings, seen: &mut u64) -> bool {
    if store().generation() == *seen {
        return false;
    }
    let (fresh, generation) = store().snapshot_with_generation();
    *mirror = fresh;
    *seen = generation;
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[derive(Default)]
    struct Mock {
        merges: Mutex<Vec<(Vec<&'static str>, UiSettings)>>,
        load: Mutex<Option<anyhow::Result<UiSettings>>>,
        fail_merges: AtomicBool,
        merge_calls: AtomicUsize,
    }

    #[async_trait]
    impl Backend for Mock {
        async fn load(&self) -> anyhow::Result<Loaded> {
            match lock(&self.load).take() {
                Some(Ok(settings)) => Ok(Loaded { settings, existed: true }),
                Some(Err(e)) => Err(e),
                None => Ok(Loaded { settings: UiSettings::default(), existed: false }),
            }
        }

        async fn merge(&self, fields: &[&'static str], settings: &UiSettings) -> anyhow::Result<()> {
            self.merge_calls.fetch_add(1, Ordering::AcqRel);
            if self.fail_merges.load(Ordering::Acquire) {
                anyhow::bail!("disk full");
            }
            lock(&self.merges).push((fields.to_vec(), settings.clone()));
            Ok(())
        }
    }

    fn loaded_store() -> (Arc<SettingsStore>, Arc<Mock>) {
        let mock = Arc::new(Mock::default());
        let store = SettingsStore::with_debounce(mock.clone(), Duration::from_millis(20));
        store.apply_loaded("test", Loaded { settings: UiSettings::default(), existed: true });
        (store, mock)
    }

    #[tokio::test]
    async fn update_persists_only_the_changed_fields() {
        let (store, mock) = loaded_store();
        assert!(store.update("test", |s| s.filter_skip_icons = true));
        assert_eq!(store.dirty_fields(), vec!["filter_skip_icons"]);
        assert_eq!(store.flush().await.unwrap(), 1);
        let merges = lock(&mock.merges);
        assert_eq!(merges.len(), 1);
        assert_eq!(merges[0].0, vec!["filter_skip_icons"]);
        assert!(merges[0].1.filter_skip_icons);
    }

    #[tokio::test]
    async fn a_noop_update_changes_nothing() {
        let (store, _mock) = loaded_store();
        let before = store.generation();
        assert!(!store.update("test", |s| s.filter_skip_icons = s.filter_skip_icons));
        assert_eq!(store.generation(), before);
        assert!(store.dirty_fields().is_empty());
        assert_eq!(store.flush().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn updates_are_debounced_into_one_merge() {
        let (store, mock) = loaded_store();
        store.update("a", |s| s.auto_indexing = true);
        store.update("b", |s| s.overwrite_descriptions = true);
        store.update("c", |s| s.clip_model = Some("x".into()));
        tokio::time::sleep(Duration::from_millis(150)).await;
        let merges = lock(&mock.merges);
        assert_eq!(merges.len(), 1, "burst should coalesce");
        assert_eq!(merges[0].0, vec!["auto_indexing", "clip_model", "overwrite_descriptions"]);
    }

    #[tokio::test]
    async fn generation_and_watcher_track_changes() {
        let (store, _mock) = loaded_store();
        let g0 = store.generation();
        let mut watcher = Watcher::of(&store);
        assert!(!watcher.changed());
        store.update("test", |s| s.auto_indexing = true);
        assert_eq!(store.generation(), g0 + 1);
        assert!(watcher.changed(), "a change is reported once");
        assert!(!watcher.changed());
        let (_s, g) = store.snapshot_with_generation();
        assert_eq!(g, g0 + 1);
    }

    #[tokio::test]
    async fn a_mirror_follows_the_store_only_when_it_changed() {
        let (store, _mock) = loaded_store();
        let (mut mirror, mut seen) = store.snapshot_with_generation();
        let sync = |mirror: &mut UiSettings, seen: &mut u64| {
            if store.generation() == *seen {
                return false;
            }
            let (fresh, generation) = store.snapshot_with_generation();
            *mirror = fresh;
            *seen = generation;
            true
        };
        assert!(!sync(&mut mirror, &mut seen));
        store.update("test", |s| s.filter_skip_icons = true);
        assert!(sync(&mut mirror, &mut seen));
        assert!(mirror.filter_skip_icons);
    }

    #[tokio::test]
    async fn nothing_is_written_before_a_successful_load() {
        let mock = Arc::new(Mock::default());
        let store = SettingsStore::with_debounce(mock.clone(), Duration::from_millis(10));
        store.update("early", |s| s.auto_indexing = true);
        tokio::time::sleep(Duration::from_millis(60)).await;
        assert_eq!(mock.merge_calls.load(Ordering::Acquire), 0);
        assert!(store.flush().await.is_err());
        assert_eq!(store.dirty_fields(), vec!["auto_indexing"]);
    }

    #[tokio::test]
    async fn edits_made_before_the_load_survive_it() {
        let mock = Arc::new(Mock::default());
        let mut stored = UiSettings::default();
        stored.clip_model = Some("stored".into());
        stored.auto_indexing = false;
        *lock(&mock.load) = Some(Ok(stored));
        let store = SettingsStore::with_debounce(mock.clone(), Duration::from_millis(10));
        store.update("early", |s| s.auto_indexing = true);
        store.reload().await.unwrap();
        let live = store.snapshot();
        assert!(live.auto_indexing, "pre-load edit kept");
        assert_eq!(live.clip_model.as_deref(), Some("stored"), "stored value kept");
        assert!(store.take_load_event());
        assert!(!store.take_load_event());
    }

    #[tokio::test]
    async fn a_load_failure_is_surfaced_and_blocks_saving() {
        let mock = Arc::new(Mock::default());
        *lock(&mock.load) = Some(Err(anyhow::anyhow!("invalid type: null, expected a boolean")));
        let store = SettingsStore::with_debounce(mock.clone(), Duration::from_millis(10));
        assert!(store.reload().await.is_err());
        assert!(store.load_failure().unwrap().contains("expected a boolean"));
        assert!(!store.is_loaded());
        assert!(store.snapshot_loaded().is_none());
        store.update("edit", |s| s.auto_indexing = true);
        tokio::time::sleep(Duration::from_millis(60)).await;
        assert_eq!(mock.merge_calls.load(Ordering::Acquire), 0, "no write while the stored record is unreadable");
        assert!(store.flush().await.is_err());
        store.reload().await.unwrap();
        assert!(store.is_loaded());
        assert!(store.load_failure().is_none());
    }

    #[tokio::test]
    async fn a_failed_save_keeps_the_fields_dirty() {
        let (store, mock) = loaded_store();
        mock.fail_merges.store(true, Ordering::Release);
        store.update("test", |s| s.auto_indexing = true);
        assert!(store.flush().await.is_err());
        assert_eq!(store.dirty_fields(), vec!["auto_indexing"]);
        assert!(store.save_status().last_error.unwrap().contains("disk full"));
        mock.fail_merges.store(false, Ordering::Release);
        assert_eq!(store.flush().await.unwrap(), 1);
        assert!(store.save_status().last_error.is_none());
    }

    #[tokio::test]
    async fn merge_from_applies_only_differing_fields() {
        let (store, _mock) = loaded_store();
        store.update("test", |s| s.recent_models = vec!["m".into()]);
        let mut copy = store.snapshot();
        copy.filter_skip_icons = true;
        assert!(store.merge_from("copy", &copy));
        let live = store.snapshot();
        assert!(live.filter_skip_icons);
        assert_eq!(live.recent_models, vec!["m".to_string()]);
        let patches = store.patches();
        assert_eq!(patches.last().unwrap().changes.len(), 1);
        assert_eq!(patches.last().unwrap().who, "copy");
    }

    #[tokio::test]
    async fn credentials_are_redacted_in_json_and_the_patch_log() {
        let (store, _mock) = loaded_store();
        store.update("keys", |s| s.openai_api_key = Some("sk-secret-value".into()));
        let json = store.json_redacted();
        let text = json.to_string();
        assert!(!text.contains("sk-secret-value"));
        assert!(text.contains("<redacted, 15 chars>"));
        let patch = store.patches().pop().unwrap();
        assert_eq!(patch.changes[0].field, "openai_api_key");
        assert_eq!(patch.changes[0].new, "<redacted>");
    }

    #[tokio::test]
    async fn the_patch_log_is_bounded() {
        let (store, _mock) = loaded_store();
        for i in 0..(PATCH_LOG_CAP + 10) {
            store.update("loop", |s| s.scan_found_batch_max = Some(i + 1));
        }
        assert_eq!(store.patches().len(), PATCH_LOG_CAP);
    }

    #[tokio::test]
    async fn force_save_writes_every_persisted_field() {
        let (store, mock) = loaded_store();
        let n = store.force_save().await.unwrap();
        assert_eq!(n, crate::database::settings::PERSISTED_FIELDS.len());
        assert_eq!(lock(&mock.merges).len(), 1);
    }

    #[tokio::test]
    async fn the_database_merge_keeps_untouched_stored_fields_and_reads_old_records() {
        crate::settings::testing::mem_db();
        let db = &*crate::database::DB;

        // A record written before `mcp_stdio_enabled` existed: nulls dropped so Option fields are absent.
        let mut old = persisted_json(&UiSettings::default());
        old.remove("mcp_stdio_enabled");
        old.retain(|_, v| !v.is_null());
        old.insert("clip_model".into(), serde_json::json!("stored-model"));
        old.insert("openai_default_model".into(), serde_json::json!("stored-default"));
        let query = format!("UPSERT user_settings:ShadowbrokerPC CONTENT {}", serde_json::Value::Object(old));
        db.query(query).await.unwrap().check().unwrap();

        let (loaded, existed) = crate::database::settings::get_settings_checked().await.unwrap();
        assert!(existed);
        assert_eq!(loaded.clip_model.as_deref(), Some("stored-model"));
        assert_eq!(loaded.mcp_stdio_enabled, None);

        // Patch one field from a settings value that holds different values elsewhere.
        let mut stale = UiSettings::default();
        stale.clip_model = Some("stale-model".into());
        stale.auto_indexing = true;
        crate::database::settings::merge_settings_in_db(&["auto_indexing"], &stale).await.unwrap();
        let (after, _) = crate::database::settings::get_settings_checked().await.unwrap();
        assert!(after.auto_indexing, "patched field written");
        assert_eq!(after.clip_model.as_deref(), Some("stored-model"), "untouched field not overwritten");
        assert_eq!(after.openai_default_model.as_deref(), Some("stored-default"));

        // An unreadable record is an error, never silently defaults.
        db.query("UPSERT user_settings:ShadowbrokerPC CONTENT { auto_indexing: 'not a bool' }")
            .await
            .unwrap()
            .check()
            .unwrap();
        assert!(crate::database::settings::get_settings_checked().await.is_err());
        assert!(crate::database::settings::merge_settings_in_db(&["auto_indexing"], &stale).await.is_err());
    }
}
