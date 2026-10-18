//! SurrealDB backend of the egui-theme-kit theme store: saved user themes, the active theme and the startup pin.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use anyhow::Context as _;
use egui_theme_kit::saved::{SavedTheme, theme_from_json, theme_to_json};
use egui_theme_kit::store::{
    CachedStore, NamedTheme, StartupTheme, StoreOp, StoreSnapshot, ThemeBackend, ThemeStore,
};
use serde_json::{Value, json};
use surrealdb::engine::local::Db;
use surrealdb::{Connection, Surreal};
use tokio::runtime::Handle;

pub const TABLE: &str = "ui_themes";
/// Record key of the singleton holding the active theme and the startup pin.
pub const ACTIVE_ID: &str = "__active__";

/// Longest queue of writes held while the database is not connected.
const QUEUE_CAP: usize = 256;

const DDL: &str = r#"
    DEFINE TABLE IF NOT EXISTS ui_themes TYPE NORMAL SCHEMAFULL PERMISSIONS FULL;
    DEFINE FIELD IF NOT EXISTS name ON ui_themes TYPE string PERMISSIONS FULL;
    DEFINE FIELD IF NOT EXISTS kind ON ui_themes TYPE string ASSERT $value IN ['user', 'active'] PERMISSIONS FULL;
    DEFINE FIELD IF NOT EXISTS theme ON ui_themes TYPE option<object> FLEXIBLE PERMISSIONS FULL;
    DEFINE FIELD IF NOT EXISTS startup ON ui_themes TYPE option<object> FLEXIBLE PERMISSIONS FULL;
    DEFINE FIELD IF NOT EXISTS updated ON ui_themes TYPE datetime DEFAULT time::now() PERMISSIONS FULL;
"#;

static SCHEMA_READY: AtomicBool = AtomicBool::new(false);

/// Defines the `ui_themes` table on a handle already pointed at NS/DB; on success the app store is marked ready and reloads.
pub async fn ensure_schema<C: Connection>(db: &Surreal<C>) -> anyhow::Result<()> {
    let outcome = async {
        db.query(DDL).await?.check()?;
        anyhow::Ok(())
    }
    .await;
    match &outcome {
        Ok(()) => {
            SCHEMA_READY.store(true, Ordering::SeqCst);
            if let Some(store) = APP_STORE.get() {
                store.reload();
            }
        }
        Err(e) => {
            log::error!("[themes] schema setup failed: {e:#}");
            crate::ui::dialogs::toast(egui_toast::ToastKind::Warning, format!("Themes cannot be saved to the database: {e}"));
        }
    }
    outcome
}

/// True once [`ensure_schema`] has succeeded on the app database.
pub fn schema_ready() -> bool {
    SCHEMA_READY.load(Ordering::SeqCst)
}

fn fnv1a(text: &str) -> u32 {
    text.bytes().fold(0x811c_9dc5, |h, b| (h ^ u32::from(b)).wrapping_mul(0x0100_0193))
}

/// Record key of a theme name: lowercase words joined by `-`, with a hash suffix when that mapping loses characters.
pub fn slug(name: &str) -> String {
    let lower = name.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
    let mut words = String::new();
    let mut dash = false;
    for c in lower.chars() {
        if c.is_ascii_alphanumeric() {
            if dash && !words.is_empty() {
                words.push('-');
            }
            dash = false;
            words.push(c);
        } else {
            dash = true;
        }
    }
    let lossless = lower.split(' ').all(|w| !w.is_empty() && w.chars().all(|c| c.is_ascii_alphanumeric()));
    match (words.is_empty(), lossless) {
        (true, _) => format!("theme-{:08x}", fnv1a(&lower)),
        (false, true) => words,
        (false, false) => format!("{words}-{:08x}", fnv1a(&lower)),
    }
}

fn theme_value(theme: &SavedTheme) -> anyhow::Result<Value> {
    Ok(serde_json::from_slice(&theme_to_json(theme)?)?)
}

fn theme_from_value(value: &Value) -> anyhow::Result<SavedTheme> {
    anyhow::ensure!(value.is_object(), "the stored theme is not an object");
    theme_from_json(&serde_json::to_vec(value)?)
}

async fn run_query(db: &Surreal<Db>, sql: &str, vars: Value) -> anyhow::Result<()> {
    db.query(sql).bind(vars).await?.check()?;
    Ok(())
}

async fn execute(db: &Surreal<Db>, op: &StoreOp) -> anyhow::Result<()> {
    match op {
        StoreOp::SaveActive(theme) => {
            let theme = theme_value(theme)?;
            run_query(
                db,
                "UPSERT type::record('ui_themes', $id) MERGE { name: $id, kind: 'active', theme: $theme, updated: time::now() }",
                json!({ "id": ACTIVE_ID, "theme": theme }),
            )
            .await
        }
        StoreOp::ClearActive => {
            run_query(
                db,
                "UPSERT type::record('ui_themes', $id) MERGE { name: $id, kind: 'active', theme: NONE, updated: time::now() }",
                json!({ "id": ACTIVE_ID }),
            )
            .await
        }
        StoreOp::SaveUser(named) => {
            run_query(
                db,
                "UPSERT type::record('ui_themes', $id) CONTENT { name: $name, kind: 'user', theme: $theme, updated: time::from_secs($secs) }",
                json!({
                    "id": slug(&named.name),
                    "name": named.name,
                    "theme": theme_value(&named.theme)?,
                    "secs": named.updated_at,
                }),
            )
            .await
        }
        StoreOp::DeleteUser(name) => {
            run_query(db, "DELETE type::record('ui_themes', $id)", json!({ "id": slug(name) })).await
        }
        StoreOp::RenameUser { from, to } => rename(db, from, to).await,
        StoreOp::SetStartup(StartupTheme::LastUsed) => {
            run_query(
                db,
                "UPSERT type::record('ui_themes', $id) MERGE { name: $id, kind: 'active', startup: NONE, updated: time::now() }",
                json!({ "id": ACTIVE_ID }),
            )
            .await
        }
        StoreOp::SetStartup(startup) => {
            run_query(
                db,
                "UPSERT type::record('ui_themes', $id) MERGE { name: $id, kind: 'active', startup: $startup, updated: time::now() }",
                json!({ "id": ACTIVE_ID, "startup": serde_json::to_value(startup)? }),
            )
            .await
        }
    }
}

async fn rename(db: &Surreal<Db>, from: &str, to: &str) -> anyhow::Result<()> {
    let (from_id, to_id) = (slug(from), slug(to));
    if from_id == to_id {
        return run_query(
            db,
            "UPDATE type::record('ui_themes', $id) SET name = $to, updated = time::now()",
            json!({ "id": from_id, "to": to }),
        )
        .await;
    }
    let rows: Vec<Value> = db
        .query("SELECT theme FROM type::record('ui_themes', $id)")
        .bind(json!({ "id": from_id }))
        .await?
        .take(0)?;
    let theme = rows
        .first()
        .and_then(|row| row.get("theme"))
        .filter(|theme| theme.is_object())
        .with_context(|| format!("there is no stored theme named \"{from}\""))?
        .clone();
    run_query(
        db,
        "BEGIN; \
         UPSERT type::record('ui_themes', $to_id) CONTENT { name: $to, kind: 'user', theme: $theme, updated: time::now() }; \
         DELETE type::record('ui_themes', $from_id); \
         COMMIT;",
        json!({ "from_id": from_id, "to_id": to_id, "to": to, "theme": theme }),
    )
    .await
}

async fn read_snapshot(db: &Surreal<Db>) -> anyhow::Result<StoreSnapshot> {
    let rows: Vec<Value> = db
        .query("SELECT name, kind, theme, startup, time::unix(updated) AS secs FROM ui_themes")
        .await?
        .take(0)?;
    let mut snapshot = StoreSnapshot::default();
    for row in rows {
        let name = row.get("name").and_then(Value::as_str).unwrap_or_default().to_owned();
        match row.get("kind").and_then(Value::as_str) {
            Some("active") => read_active(&row, &mut snapshot),
            Some("user") => match read_user(&row, &name) {
                Ok(named) => snapshot.user_themes.push(named),
                Err(e) => snapshot.warnings.push(format!("skipped saved theme \"{name}\": {e:#}")),
            },
            other => snapshot.warnings.push(format!("skipped record \"{name}\" of kind {other:?}")),
        }
    }
    Ok(snapshot)
}

fn read_active(row: &Value, snapshot: &mut StoreSnapshot) {
    if let Some(theme) = row.get("theme").filter(|v| !v.is_null()) {
        match theme_from_value(theme) {
            Ok(theme) => snapshot.active = Some(theme),
            Err(e) => snapshot.warnings.push(format!("the active theme is unreadable: {e:#}")),
        }
    }
    if let Some(startup) = row.get("startup").filter(|v| !v.is_null()) {
        snapshot.startup = serde_json::from_value(startup.clone()).unwrap_or_default();
    }
}

fn read_user(row: &Value, name: &str) -> anyhow::Result<NamedTheme> {
    let name = egui_theme_kit::store::clean_name(name)?;
    let theme = theme_from_value(row.get("theme").context("no theme field")?)?;
    Ok(NamedTheme { name, theme, updated_at: row.get("secs").and_then(Value::as_u64).unwrap_or(0) })
}

/// The live database handle, or none while the database is not connected.
pub type Provider = Arc<dyn Fn() -> Option<Surreal<Db>> + Send + Sync>;

/// What a successful read of the table found.
#[derive(Clone, Debug)]
pub struct LoadEvent {
    pub snapshot: StoreSnapshot,
    /// The active theme was changed while the database was unavailable and has been written since.
    pub active_written_after_start: bool,
}

#[derive(Default)]
struct Link {
    repaint: Mutex<Option<egui::Context>>,
    event: Mutex<Option<LoadEvent>>,
    loaded: AtomicBool,
    active_queued: AtomicBool,
    queued: AtomicUsize,
}

struct SurrealBackend {
    provider: Provider,
    rt: Handle,
    queue: Vec<StoreOp>,
    skip_first_load: bool,
    link: Arc<Link>,
}

impl Link {
    fn wake(&self) {
        if let Some(ctx) = self.repaint.lock().ok().and_then(|slot| slot.clone()) {
            ctx.request_repaint();
        }
    }
}

impl SurrealBackend {
    fn defer(&mut self, op: StoreOp) {
        let is_active = matches!(op, StoreOp::SaveActive(_) | StoreOp::ClearActive);
        if is_active {
            self.queue.retain(|queued| !matches!(queued, StoreOp::SaveActive(_) | StoreOp::ClearActive));
            self.link.active_queued.store(true, Ordering::SeqCst);
        }
        if self.queue.len() >= QUEUE_CAP {
            self.queue.remove(0);
        }
        self.queue.push(op);
        self.link.queued.store(self.queue.len(), Ordering::SeqCst);
    }

    fn replay(&mut self, db: &Surreal<Db>) -> Result<(), String> {
        let mut failures = Vec::new();
        for op in std::mem::take(&mut self.queue) {
            if let Err(e) = self.rt.block_on(execute(db, &op)) {
                failures.push(format!("{e:#}"));
            }
        }
        self.link.queued.store(0, Ordering::SeqCst);
        if failures.is_empty() { Ok(()) } else { Err(failures.join("; ")) }
    }
}

impl ThemeBackend for SurrealBackend {
    fn load(&mut self) -> Result<StoreSnapshot, String> {
        if std::mem::take(&mut self.skip_first_load) {
            return Ok(StoreSnapshot::default());
        }
        let Some(db) = (self.provider)() else {
            return Err("The database is not connected, so themes are not being saved to it.".to_owned());
        };
        let replayed = self.replay(&db);
        let mut snapshot = self.rt.block_on(read_snapshot(&db)).map_err(|e| format!("{e:#}"))?;
        if let Err(e) = replayed {
            snapshot.warnings.push(format!("queued theme changes failed: {e}"));
        }
        let active_written_after_start = self.link.active_queued.swap(false, Ordering::SeqCst);
        if let Ok(mut slot) = self.link.event.lock() {
            *slot = Some(LoadEvent { snapshot: snapshot.clone(), active_written_after_start });
        }
        self.link.loaded.store(true, Ordering::SeqCst);
        self.link.wake();
        Ok(snapshot)
    }

    fn apply(&mut self, op: &StoreOp) -> Result<(), String> {
        let Some(db) = (self.provider)() else {
            self.defer(op.clone());
            return Ok(());
        };
        let replayed = self.replay(&db);
        let applied = self.rt.block_on(execute(&db, op)).map_err(|e| format!("{e:#}"));
        let outcome = match (replayed, applied) {
            (Ok(()), applied) => applied,
            (Err(e), Ok(())) => Err(e),
            (Err(e), Err(f)) => Err(format!("{e}; {f}")),
        };
        if outcome.is_err() {
            self.link.wake();
        }
        outcome
    }
}

/// A [`ThemeStore`] over the `ui_themes` table: reads come from a cache, writes queue until the database is connected.
pub struct SurrealThemeStore {
    inner: CachedStore,
    link: Arc<Link>,
}

impl SurrealThemeStore {
    /// A store over `provider`; the first read happens on [`Self::reload`], never on the calling thread.
    pub fn new(provider: Provider, rt: Handle) -> Self {
        let link = Arc::new(Link::default());
        let backend = SurrealBackend {
            provider,
            rt,
            queue: Vec::new(),
            skip_first_load: true,
            link: Arc::clone(&link),
        };
        Self { inner: CachedStore::open(backend), link }
    }

    /// Wakes `ctx` when a read finishes or a write fails.
    pub fn set_repaint(&self, ctx: egui::Context) {
        if let Ok(mut slot) = self.link.repaint.lock() {
            *slot = Some(ctx);
        }
    }

    /// Re-reads the table in the background and publishes a [`LoadEvent`] when it succeeds.
    pub fn reload(&self) {
        self.inner.reload();
    }

    /// Blocks until every queued write has been handled by the worker.
    pub fn flush(&self) {
        self.inner.flush();
    }

    /// [`Self::flush`] bounded by `timeout`; returns whether it finished.
    pub fn flush_within(self: &Arc<Self>, timeout: Duration) -> bool {
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let store = Arc::clone(self);
        std::thread::spawn(move || {
            store.flush();
            let _ = done_tx.send(());
        });
        done_rx.recv_timeout(timeout).is_ok()
    }

    /// The pending [`LoadEvent`], if a read finished since the last call.
    pub fn take_load_event(&self) -> Option<LoadEvent> {
        self.link.event.lock().ok()?.take()
    }

    /// True once a read from the database has succeeded.
    pub fn is_loaded(&self) -> bool {
        self.link.loaded.load(Ordering::SeqCst)
    }

    /// Writes held back because the database is not connected.
    pub fn queued_ops(&self) -> usize {
        self.link.queued.load(Ordering::SeqCst)
    }
}

impl ThemeStore for SurrealThemeStore {
    fn load_active(&self) -> Option<SavedTheme> {
        self.inner.load_active()
    }

    fn save_active(&self, theme: &SavedTheme) {
        self.inner.save_active(theme);
    }

    fn clear_active(&self) {
        self.inner.clear_active();
    }

    fn user_themes(&self) -> Vec<NamedTheme> {
        self.inner.user_themes()
    }

    fn save_user_theme(&self, name: &str, theme: &SavedTheme) {
        self.inner.save_user_theme(name, theme);
    }

    fn delete_user_theme(&self, name: &str) {
        self.inner.delete_user_theme(name);
    }

    fn rename_user_theme(&self, from: &str, to: &str) {
        self.inner.rename_user_theme(from, to);
    }

    fn startup(&self) -> StartupTheme {
        self.inner.startup()
    }

    fn set_startup(&self, startup: StartupTheme) {
        self.inner.set_startup(startup);
    }

    fn revision(&self) -> u64 {
        self.inner.revision()
    }

    fn last_error(&self) -> Option<String> {
        self.inner.last_error()
    }

    fn clear_error(&self) {
        self.inner.clear_error();
    }
}

impl egui_theme_kit::startup::ThemeStore for SurrealThemeStore {
    fn load_active(&self) -> Option<SavedTheme> {
        ThemeStore::load_active(self)
    }

    fn save_active(&self, theme: &SavedTheme) {
        ThemeStore::save_active(self, theme);
    }
}

static APP_STORE: OnceLock<Arc<SurrealThemeStore>> = OnceLock::new();

/// The process-wide store over the app database; it holds writes until the schema is ready.
pub fn app_store() -> &'static Arc<SurrealThemeStore> {
    APP_STORE.get_or_init(|| {
        let provider: Provider = Arc::new(|| schema_ready().then(|| super::DB.clone()));
        Arc::new(SurrealThemeStore::new(provider, Handle::current()))
    })
}

#[cfg(test)]
pub(crate) mod tests;
