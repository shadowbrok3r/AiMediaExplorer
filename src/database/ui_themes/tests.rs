use std::collections::HashSet;
use std::sync::LazyLock;

use egui_theme_kit::presets::{self, themes_equal};
use surrealdb::engine::local::Mem;

use super::*;

pub(crate) static RT: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("test runtime")
});

pub(crate) struct Fixture {
    pub(crate) db: Surreal<Db>,
    pub(crate) ready: Arc<AtomicBool>,
}

impl Fixture {
    pub(crate) fn new() -> Self {
        let db = RT.block_on(async {
            let db = Surreal::new::<Mem>(()).await.expect("memory database");
            db.use_ns("test").use_db("test").await.expect("namespace");
            ensure_schema(&db).await.expect("schema");
            db
        });
        Self { db, ready: Arc::new(AtomicBool::new(true)) }
    }

    pub(crate) fn unready() -> Self {
        let fixture = Self::new();
        fixture.ready.store(false, Ordering::SeqCst);
        fixture
    }

    pub(crate) fn store(&self) -> SurrealThemeStore {
        let (db, ready) = (self.db.clone(), Arc::clone(&self.ready));
        let provider: Provider = Arc::new(move || ready.load(Ordering::SeqCst).then(|| db.clone()));
        SurrealThemeStore::new(provider, RT.handle().clone())
    }

    pub(crate) fn loaded(&self) -> SurrealThemeStore {
        let store = self.store();
        store.reload();
        store.flush();
        store
    }

    pub(crate) fn raw(&self, sql: &str) -> Vec<Value> {
        RT.block_on(async { self.db.query(sql).await.expect("query").take(0).expect("rows") })
    }
}

pub(crate) fn preset_theme(id: &str) -> SavedTheme {
    presets::preset(id).unwrap_or_else(|| panic!("preset {id}")).theme()
}

fn names(store: &SurrealThemeStore) -> Vec<String> {
    store.user_themes().into_iter().map(|t| t.name).collect()
}

#[test]
fn slugs_are_lowercase_words_and_injective_across_lossy_names() {
    assert_eq!(slug("Late Night"), "late-night");
    assert_eq!(slug("LATE   night"), "late-night");
    assert_eq!(slug("Neon 2"), "neon-2");
    let lossy = ["late-night", "Late_Night", "late night!", "Late.Night"].map(slug);
    let distinct: HashSet<&String> = lossy.iter().collect();
    assert_eq!(distinct.len(), lossy.len(), "{lossy:?}");
    assert!(lossy.iter().all(|s| s.starts_with("late-night-")));
    assert_ne!(slug("late-night"), slug("late night"));
    assert!(slug("!!!").starts_with("theme-"));
    assert!(slug("Caf\u{e9}").starts_with("caf-"));
    assert_ne!(slug("a"), ACTIVE_ID);
}

#[test]
fn the_schema_is_idempotent_and_rejects_unknown_kinds() {
    let fixture = Fixture::new();
    RT.block_on(ensure_schema(&fixture.db)).expect("second run");
    RT.block_on(async {
        let bad = fixture
            .db
            .query("CREATE ui_themes:bad CONTENT { name: 'x', kind: 'bogus' }")
            .await
            .expect("query");
        assert!(bad.check().is_err(), "the kind assertion must reject 'bogus'");
    });
}

#[test]
fn a_user_theme_round_trips_through_the_database() {
    let fixture = Fixture::new();
    let store = fixture.store();
    let noir = preset_theme("noir");
    store.save_user_theme("  Late   Night ", &noir);
    store.flush();

    let reopened = fixture.loaded();
    let saved = reopened.user_themes();
    assert_eq!(names(&reopened), ["Late Night"]);
    assert!(themes_equal(&saved[0].theme, &noir));
    assert!(saved[0].updated_at > 0);
    assert!(reopened.last_error().is_none(), "{:?}", reopened.last_error());

    let rows = fixture.raw("SELECT name, kind FROM ui_themes");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["kind"], "user");
}

#[test]
fn the_theme_is_stored_as_an_object() {
    let fixture = Fixture::new();
    let store = fixture.store();
    store.save_user_theme("Obj", &preset_theme("graphite"));
    store.flush();
    let rows = fixture.raw("SELECT theme.density AS density, type::is_object(theme) AS is_object FROM ui_themes");
    assert_eq!(rows[0]["is_object"], true, "{rows:?}");
    assert!(rows[0]["density"].is_string(), "{rows:?}");
}

#[test]
fn every_preset_survives_the_round_trip_unchanged() {
    let fixture = Fixture::new();
    let store = fixture.store();
    let defs = presets::presets();
    assert!(defs.len() >= 19);
    for def in &defs {
        store.save_user_theme(def.name, &def.theme());
    }
    store.save_active(&preset_theme("noir-glass"));
    store.flush();

    let reopened = fixture.loaded();
    for def in &defs {
        let found = reopened
            .user_themes()
            .into_iter()
            .find(|t| t.name == def.name)
            .unwrap_or_else(|| panic!("{} missing", def.name));
        assert!(themes_equal(&found.theme, &def.theme()), "{} changed in the database", def.name);
    }
    let active = reopened.load_active().expect("active theme");
    assert!(themes_equal(&active, &preset_theme("noir-glass")));
    assert!(reopened.last_error().is_none(), "{:?}", reopened.last_error());
}

#[test]
fn same_name_in_another_case_replaces_the_record() {
    let fixture = Fixture::new();
    let store = fixture.store();
    store.save_user_theme("Late Night", &preset_theme("noir"));
    store.save_user_theme("LATE NIGHT", &preset_theme("frost"));
    store.flush();
    let reopened = fixture.loaded();
    assert_eq!(names(&reopened), ["LATE NIGHT"]);
    assert_eq!(fixture.raw("SELECT * FROM ui_themes").len(), 1);
    assert!(themes_equal(&reopened.user_themes()[0].theme, &preset_theme("frost")));
}

#[test]
fn rename_moves_the_record_and_keeps_the_theme() {
    let fixture = Fixture::new();
    let store = fixture.store();
    let noir = preset_theme("noir");
    store.save_user_theme("alpha", &noir);
    store.save_user_theme("beta", &preset_theme("frost"));
    store.rename_user_theme("alpha", "Zeta Prime");
    store.flush();

    let reopened = fixture.loaded();
    assert_eq!(names(&reopened), ["beta", "Zeta Prime"]);
    let zeta = reopened.user_themes().into_iter().find(|t| t.name == "Zeta Prime").unwrap();
    assert!(themes_equal(&zeta.theme, &noir));
    assert_eq!(fixture.raw("SELECT * FROM ui_themes").len(), 2);
}

#[test]
fn a_case_only_rename_updates_the_name_in_place() {
    let fixture = Fixture::new();
    let store = fixture.store();
    store.save_user_theme("late night", &preset_theme("noir"));
    store.rename_user_theme("late night", "Late Night");
    store.flush();
    let reopened = fixture.loaded();
    assert_eq!(names(&reopened), ["Late Night"]);
    assert_eq!(fixture.raw("SELECT * FROM ui_themes").len(), 1);
}

#[test]
fn renaming_onto_an_existing_name_is_refused_and_nothing_changes() {
    let fixture = Fixture::new();
    let store = fixture.store();
    store.save_user_theme("alpha", &preset_theme("noir"));
    store.save_user_theme("beta", &preset_theme("frost"));
    store.rename_user_theme("alpha", "BETA");
    assert!(store.last_error().is_some());
    store.flush();
    assert_eq!(names(&fixture.loaded()), ["alpha", "beta"]);
}

#[test]
fn delete_removes_the_record_and_resets_a_pinned_startup() {
    let fixture = Fixture::new();
    let store = fixture.store();
    store.save_user_theme("alpha", &preset_theme("noir"));
    store.save_user_theme("beta", &preset_theme("frost"));
    store.set_startup(StartupTheme::User("alpha".to_owned()));
    store.flush();
    assert_eq!(fixture.loaded().startup(), StartupTheme::User("alpha".to_owned()));

    store.delete_user_theme("alpha");
    store.flush();
    let reopened = fixture.loaded();
    assert_eq!(names(&reopened), ["beta"]);
    assert_eq!(reopened.startup(), StartupTheme::LastUsed);
    assert_eq!(fixture.raw("SELECT * FROM ui_themes WHERE kind = 'user'").len(), 1);
}

#[test]
fn the_active_theme_and_startup_pin_live_in_one_singleton() {
    let fixture = Fixture::new();
    let store = fixture.store();
    store.save_active(&preset_theme("frost"));
    store.set_startup(StartupTheme::Preset("noir".to_owned()));
    store.flush();

    let rows = fixture.raw("SELECT name, kind FROM ui_themes WHERE kind = 'active'");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["name"], ACTIVE_ID);
    let reopened = fixture.loaded();
    assert!(themes_equal(&reopened.load_active().unwrap(), &preset_theme("frost")));
    assert_eq!(reopened.startup(), StartupTheme::Preset("noir".to_owned()));

    store.save_active(&preset_theme("studio"));
    store.flush();
    assert!(themes_equal(&fixture.loaded().load_active().unwrap(), &preset_theme("studio")));
    assert_eq!(fixture.loaded().startup(), StartupTheme::Preset("noir".to_owned()), "saving the active theme keeps the pin");

    store.clear_active();
    store.set_startup(StartupTheme::LastUsed);
    store.flush();
    let cleared = fixture.loaded();
    assert!(cleared.load_active().is_none());
    assert_eq!(cleared.startup(), StartupTheme::LastUsed);
    assert_eq!(fixture.raw("SELECT * FROM ui_themes WHERE kind = 'active'").len(), 1);
}

#[test]
fn unknown_fields_and_broken_records_do_not_stop_the_load() {
    let fixture = Fixture::new();
    let good = serde_json::to_value(preset_theme("noir").style).unwrap();
    RT.block_on(async {
        fixture.db.query("DEFINE FIELD extra ON ui_themes TYPE any").await.unwrap().check().unwrap();
        let theme = json!({
            "style": good,
            "glass": null,
            "density": "Regular",
            "text_scale": "Normal",
            "from_a_future_release": { "nested": [1, 2, 3] },
        });
        fixture
            .db
            .query(
                "CREATE ui_themes:future CONTENT { name: 'Future', kind: 'user', theme: $theme, extra: { a: 1 }, updated: time::now() };
                 CREATE ui_themes:empty CONTENT { name: 'Empty', kind: 'user', updated: time::now() };
                 CREATE ui_themes:nameless CONTENT { name: '   ', kind: 'user', theme: $theme, updated: time::now() }",
            )
            .bind(json!({ "theme": theme }))
            .await
            .unwrap()
            .check()
            .unwrap();
    });
    let store = fixture.loaded();
    assert_eq!(names(&store), ["Future"]);
    let error = store.last_error().expect("skipped records are reported");
    assert!(error.contains("Empty") && error.contains("skipped"), "{error}");
}

#[test]
fn writes_queue_while_the_database_is_unavailable_and_replay_in_order() {
    let fixture = Fixture::unready();
    let store = fixture.store();
    store.save_user_theme("alpha", &preset_theme("noir"));
    store.save_active(&preset_theme("frost"));
    store.save_active(&preset_theme("studio"));
    store.rename_user_theme("alpha", "omega");
    store.set_startup(StartupTheme::User("omega".to_owned()));
    store.flush();

    assert!(store.last_error().is_none(), "{:?}", store.last_error());
    assert_eq!(names(&store), ["omega"], "the cache already reflects the writes");
    assert_eq!(store.queued_ops(), 4, "the two active saves coalesce");
    assert!(fixture.raw("SELECT * FROM ui_themes").is_empty());
    assert!(!store.is_loaded());

    fixture.ready.store(true, Ordering::SeqCst);
    store.reload();
    store.flush();

    assert_eq!(store.queued_ops(), 0);
    assert!(store.is_loaded());
    let event = store.take_load_event().expect("a load event");
    assert!(event.active_written_after_start);
    assert!(themes_equal(event.snapshot.active.as_ref().unwrap(), &preset_theme("studio")));
    assert_eq!(event.snapshot.startup, StartupTheme::User("omega".to_owned()));
    assert!(store.take_load_event().is_none());

    let reopened = fixture.loaded();
    assert_eq!(names(&reopened), ["omega"]);
    assert!(themes_equal(&reopened.load_active().unwrap(), &preset_theme("studio")));
    assert_eq!(reopened.startup(), StartupTheme::User("omega".to_owned()));
}

#[test]
fn a_write_after_the_database_connects_replays_the_queue_first() {
    let fixture = Fixture::unready();
    let store = fixture.store();
    store.save_user_theme("alpha", &preset_theme("noir"));
    store.flush();
    fixture.ready.store(true, Ordering::SeqCst);
    store.rename_user_theme("alpha", "beta");
    store.flush();
    assert!(store.last_error().is_none(), "{:?}", store.last_error());
    assert_eq!(names(&fixture.loaded()), ["beta"]);
}

#[test]
fn the_first_load_event_without_queued_writes_carries_the_stored_state() {
    let fixture = Fixture::new();
    let writer = fixture.store();
    writer.save_active(&preset_theme("frost"));
    writer.save_user_theme("alpha", &preset_theme("noir"));
    writer.flush();

    let store = fixture.store();
    assert!(store.take_load_event().is_none(), "construction never reads on the calling thread");
    assert!(store.load_active().is_none());
    store.reload();
    store.flush();
    let event = store.take_load_event().expect("a load event");
    assert!(!event.active_written_after_start);
    assert!(themes_equal(event.snapshot.active.as_ref().unwrap(), &preset_theme("frost")));
    assert_eq!(event.snapshot.user_themes.len(), 1);
    assert!(themes_equal(&store.load_active().unwrap(), &preset_theme("frost")));
}

#[test]
fn reloading_while_the_database_is_unavailable_reports_an_error_and_keeps_the_cache() {
    let fixture = Fixture::unready();
    let store = fixture.store();
    store.save_user_theme("alpha", &preset_theme("noir"));
    store.reload();
    store.flush();
    assert!(store.last_error().is_some_and(|e| e.contains("not connected")));
    assert_eq!(names(&store), ["alpha"]);
    assert!(store.take_load_event().is_none());
}

#[test]
fn switching_databases_loads_each_ones_own_themes() {
    let first = Fixture::new();
    let second = Fixture::new();
    first.store().save_user_theme("one", &preset_theme("noir"));
    second.store().save_user_theme("two", &preset_theme("frost"));
    let (a, b) = (first.store(), second.store());
    a.flush();
    b.flush();
    assert_eq!(names(&first.loaded()), ["one"]);
    assert_eq!(names(&second.loaded()), ["two"]);

    let swapped_db = Arc::new(Mutex::new(first.db.clone()));
    let handle = Arc::clone(&swapped_db);
    let provider: Provider = Arc::new(move || Some(handle.lock().unwrap().clone()));
    let store = SurrealThemeStore::new(provider, RT.handle().clone());
    store.reload();
    store.flush();
    assert_eq!(names(&store), ["one"]);
    *swapped_db.lock().unwrap() = second.db.clone();
    store.reload();
    store.flush();
    assert_eq!(names(&store), ["two"]);
}

#[test]
fn flush_within_returns_once_the_worker_is_idle() {
    let fixture = Fixture::new();
    let store = Arc::new(fixture.store());
    store.save_active(&preset_theme("frost"));
    assert!(store.flush_within(Duration::from_secs(10)));
}

#[test]
fn a_finished_read_wakes_the_ui() {
    let fixture = Fixture::new();
    let store = fixture.store();
    let ctx = egui::Context::default();
    let wakes = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&wakes);
    ctx.set_request_repaint_callback(move |_| {
        counter.fetch_add(1, Ordering::SeqCst);
    });
    store.set_repaint(ctx);
    store.reload();
    store.flush();
    assert_eq!(wakes.load(Ordering::SeqCst), 1);
}

#[test]
fn the_app_store_reloads_when_the_app_schema_becomes_ready() {
    let _enter = RT.enter();
    crate::settings::testing::mem_db();
    let store = app_store();
    RT.block_on(ensure_schema(&super::super::DB)).expect("schema");
    assert!(schema_ready());
    store.flush();
    let event = store.take_load_event().expect("ensure_schema reloads the app store");
    assert!(!event.active_written_after_start);
}
