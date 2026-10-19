//! Sidebar data loaded in the background, cached per source and refreshed on a timer or a signal.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui::Context;
use surrealdb::types::RecordId;

use super::links::{CollectionEntry, CountEntry, ScanEntry, sort_counts};
use crate::command::{AppCommand, ToastLevel};
use crate::utilities::explorer::{DriveInfo, PhysicalDrive};
use crate::utilities::types::QuickAccess;

/// Scans listed in the sidebar.
const SCAN_LIMIT: usize = 20;
/// Wait before a failed load is tried again.
const ERROR_RETRY: Duration = Duration::from_secs(15);
/// Youngest age at which a stale list loads again.
const STALE_MIN_AGE: Duration = Duration::from_secs(2);

static REFRESH_EPOCH: AtomicU64 = AtomicU64::new(0);

/// Asks every sidebar list to reload on its next visible frame.
pub fn request_refresh() {
    REFRESH_EPOCH.fetch_add(1, Ordering::Relaxed);
}

/// A list the sidebar loads on its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Source {
    Places,
    Library,
    Collections,
    Tags,
    Categories,
    Scans,
}

impl Source {
    pub const ALL: [Source; 6] = [Self::Places, Self::Library, Self::Collections, Self::Tags, Self::Categories, Self::Scans];

    pub fn name(self) -> &'static str {
        match self {
            Self::Places => "places",
            Self::Library => "library count",
            Self::Collections => "collections",
            Self::Tags => "tags",
            Self::Categories => "categories",
            Self::Scans => "scans",
        }
    }

    /// Age after which a visible list loads again.
    pub fn ttl(self) -> Duration {
        match self {
            Self::Collections => Duration::from_secs(8),
            Self::Places => Duration::from_secs(20),
            Self::Library | Self::Scans => Duration::from_secs(30),
            Self::Tags | Self::Categories => Duration::from_secs(120),
        }
    }

    fn index(self) -> usize {
        Self::ALL.iter().position(|s| *s == self).unwrap_or(0)
    }
}

/// Load state of one source.
#[derive(Clone, Debug, Default)]
pub struct Meta {
    pub loaded_at: Option<Instant>,
    pub loading: bool,
    pub error: Option<String>,
    pub loads: u32,
    stale: bool,
}

impl Meta {
    pub fn age(&self, now: Instant) -> Option<Duration> {
        self.loaded_at.map(|t| now.saturating_duration_since(t))
    }

    fn due(&self, source: Source, now: Instant) -> bool {
        if self.loading {
            return false;
        }
        match self.age(now) {
            None => true,
            Some(age) if self.stale => age >= STALE_MIN_AGE,
            Some(age) if self.error.is_some() => age >= ERROR_RETRY,
            Some(age) => age >= source.ttl(),
        }
    }
}

/// User folders, drives and WSL mounts read from the system.
#[derive(Clone, Default)]
pub struct PlacesRaw {
    pub quick: Vec<QuickAccess>,
    pub drives: Vec<DriveInfo>,
    pub wsl: Vec<QuickAccess>,
    pub physical: Vec<PhysicalDrive>,
}

impl PlacesRaw {
    /// Reads the places from the system; blocks on drive and WSL queries.
    pub fn read() -> Self {
        use crate::utilities::explorer as sys;
        let wsl = sys::list_wsl_distros().iter().flat_map(|distro| sys::wsl_dynamic_mounts(distro)).collect();
        Self { quick: sys::quick_access(), drives: sys::list_drive_infos(), wsl, physical: sys::list_physical_drives() }
    }
}

#[derive(Default)]
struct Inner {
    meta: [Meta; 6],
    epoch: u64,
    auto_load: bool,
    total: Option<u64>,
    collections: Arc<Vec<CollectionEntry>>,
    tags: Arc<Vec<CountEntry>>,
    categories: Arc<Vec<CountEntry>>,
    scans: Arc<Vec<ScanEntry>>,
    places: Arc<PlacesRaw>,
}

impl Inner {
    fn mark_library_stale(&mut self) {
        for source in Source::ALL.into_iter().filter(|s| *s != Source::Places) {
            self.meta[source.index()].stale = true;
        }
    }
}

enum Loaded {
    Places(PlacesRaw),
    Total(u64),
    Collections(Vec<CollectionEntry>),
    Tags(Vec<CountEntry>),
    Categories(Vec<CountEntry>),
    Scans(Vec<ScanEntry>),
}

/// Shared cache of the sidebar lists.
#[derive(Clone, Default)]
pub struct SidebarData {
    inner: Arc<Mutex<Inner>>,
}

impl SidebarData {
    /// A cache that loads from the system and the database.
    pub fn live() -> Self {
        let data = Self::default();
        data.with(|i| {
            i.auto_load = true;
            i.epoch = REFRESH_EPOCH.load(Ordering::Relaxed);
        });
        data
    }

    fn with<R>(&self, f: impl FnOnce(&mut Inner) -> R) -> R {
        f(&mut self.inner.lock().unwrap_or_else(|e| e.into_inner()))
    }

    pub fn auto_load(&self) -> bool {
        self.with(|i| i.auto_load)
    }

    pub fn meta(&self, source: Source) -> Meta {
        self.with(|i| i.meta[source.index()].clone())
    }

    pub fn total(&self) -> Option<u64> {
        self.with(|i| i.total)
    }

    pub fn collections(&self) -> Arc<Vec<CollectionEntry>> {
        self.with(|i| i.collections.clone())
    }

    pub fn tags(&self) -> Arc<Vec<CountEntry>> {
        self.with(|i| i.tags.clone())
    }

    pub fn categories(&self) -> Arc<Vec<CountEntry>> {
        self.with(|i| i.categories.clone())
    }

    pub fn scans(&self) -> Arc<Vec<ScanEntry>> {
        self.with(|i| i.scans.clone())
    }

    pub fn places(&self) -> Arc<PlacesRaw> {
        self.with(|i| i.places.clone())
    }

    /// Marks one source to load again on its next visible frame.
    pub fn invalidate(&self, source: Source) {
        self.with(|i| i.meta[source.index()].stale = true);
    }

    /// Marks every list that reads the library to load again.
    pub fn invalidate_all(&self) {
        self.with(Inner::mark_library_stale);
    }

    /// Starts a background load of `source` when `wanted` and the cache is empty, stale or expired.
    pub fn poll(&self, ctx: &Context, source: Source, wanted: bool, now: Instant) {
        let start = self.with(|i| {
            let epoch = REFRESH_EPOCH.load(Ordering::Relaxed);
            if epoch != i.epoch {
                i.epoch = epoch;
                i.mark_library_stale();
            }
            let m = &mut i.meta[source.index()];
            let go = wanted && i.auto_load && m.due(source, now);
            if go {
                m.loading = true;
            }
            go
        });
        if !start {
            return;
        }
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            self.finish(source, Err("no async runtime".to_string()), Instant::now());
            return;
        };
        let (data, ctx) = (self.clone(), ctx.clone());
        handle.spawn(async move {
            let result = load(source).await.map_err(|e| format!("{e:#}"));
            data.finish(source, result, Instant::now());
            ctx.request_repaint();
        });
    }

    fn finish(&self, source: Source, result: Result<Loaded, String>, now: Instant) {
        self.with(|i| {
            let m = &mut i.meta[source.index()];
            m.loading = false;
            m.loaded_at = Some(now);
            m.loads += 1;
            m.stale = false;
            match result {
                Ok(loaded) => {
                    m.error = None;
                    match loaded {
                        Loaded::Places(v) => i.places = Arc::new(v),
                        Loaded::Total(n) => i.total = Some(n),
                        Loaded::Collections(v) => i.collections = Arc::new(v),
                        Loaded::Tags(v) => i.tags = Arc::new(v),
                        Loaded::Categories(v) => i.categories = Arc::new(v),
                        Loaded::Scans(v) => i.scans = Arc::new(v),
                    }
                }
                Err(e) => m.error = Some(e),
            }
        });
    }

    /// Creates a collection in the background and reloads the list.
    pub fn create_collection(&self, ctx: &Context, name: String) {
        self.collection_op(ctx, async move {
            let made = crate::database::collections::create(&crate::DB, &name).await?;
            Ok(format!("Created \"{}\"", made.name))
        });
    }

    pub fn rename_collection(&self, ctx: &Context, id: RecordId, new_name: String) {
        self.collection_op(ctx, async move {
            let done = crate::database::collections::rename(&crate::DB, &id, &new_name).await?;
            Ok(format!("Renamed to \"{}\"; {} memberships moved", done.collection.name, done.moved))
        });
    }

    pub fn delete_collection(&self, ctx: &Context, id: RecordId, name: String) {
        self.collection_op(ctx, async move {
            let removed = crate::database::collections::delete(&crate::DB, &id).await?;
            Ok(format!("Deleted \"{name}\" and its {removed} memberships; files were not touched"))
        });
    }

    fn collection_op(&self, ctx: &Context, op: impl std::future::Future<Output = anyhow::Result<String>> + Send + 'static) {
        let Ok(handle) = tokio::runtime::Handle::try_current() else { return };
        let (data, ctx) = (self.clone(), ctx.clone());
        handle.spawn(async move {
            let (kind, text) = match op.await {
                Ok(text) => (ToastLevel::Success, text),
                Err(e) => (ToastLevel::Error, format!("{e:#}")),
            };
            crate::command::send(AppCommand::Toast { kind, text });
            data.invalidate(Source::Collections);
            ctx.request_repaint();
        });
    }

    #[cfg(test)]
    pub fn preset(&self, f: impl FnOnce(&mut PresetData)) {
        self.with(|i| {
            let mut preset = PresetData { total: i.total, collections: (*i.collections).clone(), tags: (*i.tags).clone(), categories: (*i.categories).clone(), scans: (*i.scans).clone(), places: (*i.places).clone() };
            f(&mut preset);
            i.total = preset.total;
            i.collections = Arc::new(preset.collections);
            i.tags = Arc::new(preset.tags);
            i.categories = Arc::new(preset.categories);
            i.scans = Arc::new(preset.scans);
            i.places = Arc::new(preset.places);
            let now = Instant::now();
            i.meta.iter_mut().for_each(|m| m.loaded_at = Some(now));
        });
    }
}

/// Values a test puts into the cache.
#[cfg(test)]
pub struct PresetData {
    pub total: Option<u64>,
    pub collections: Vec<CollectionEntry>,
    pub tags: Vec<CountEntry>,
    pub categories: Vec<CountEntry>,
    pub scans: Vec<ScanEntry>,
    pub places: PlacesRaw,
}

async fn load(source: Source) -> anyhow::Result<Loaded> {
    use crate::database::scope_queries::counter::{Site, note};
    match source {
        Source::Library => note(Site::SidebarLibrary),
        Source::Collections => note(Site::SidebarCollections),
        Source::Tags => note(Site::SidebarTags),
        Source::Categories => note(Site::SidebarCategories),
        Source::Scans => note(Site::SidebarScans),
        Source::Places => {}
    }
    match source {
        Source::Places => {
            Ok(Loaded::Places(tokio::task::spawn_blocking(PlacesRaw::read).await?))
        }
        Source::Library => Ok(Loaded::Total(crate::database::surql::count_rows(&crate::DB, crate::database::THUMBNAILS).await?)),
        Source::Collections => {
            let rows = crate::database::collections::list_with_counts(&crate::DB).await?;
            Ok(Loaded::Collections(rows.into_iter().map(|c| CollectionEntry { id: c.collection.id.clone(), name: c.collection.name.clone(), count: c.count }).collect()))
        }
        Source::Tags => Ok(Loaded::Tags(counts(crate::database::Thumbnail::list_tag_counts().await?))),
        Source::Categories => Ok(Loaded::Categories(counts(crate::database::Thumbnail::list_category_counts().await?))),
        Source::Scans => {
            let rows = crate::database::CachedScan::list_recent(SCAN_LIMIT).await?;
            Ok(Loaded::Scans(rows.into_iter().filter_map(scan_entry).collect()))
        }
    }
}

fn counts(rows: Vec<(String, i64)>) -> Vec<CountEntry> {
    let mut entries: Vec<CountEntry> = rows.into_iter().filter(|(name, _)| !name.trim().is_empty()).map(|(name, count)| CountEntry { name, count }).collect();
    sort_counts(&mut entries);
    entries
}

fn scan_entry(scan: crate::database::CachedScan) -> Option<ScanEntry> {
    let scan_id = scan.scan_id?;
    let started: chrono::DateTime<chrono::Utc> = scan.started.into();
    let started = started.with_timezone(&chrono::Local).format("%Y-%m-%d %H:%M").to_string();
    let title = scan.title.filter(|t| !t.trim().is_empty()).unwrap_or_else(|| crate::command::Scope::Folder(scan.root.clone()).default_title());
    Some(ScanEntry { scan_id, title, root: scan.root, started, total: scan.total })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loaded(now: Instant, age: Duration, stale: bool) -> Meta {
        Meta { loaded_at: Some(now - age), stale, ..Default::default() }
    }

    #[test]
    fn a_list_that_never_loaded_is_due_at_once() {
        assert!(Meta::default().due(Source::Tags, Instant::now()));
    }

    #[test]
    fn a_stale_list_waits_until_it_is_old_enough_to_be_worth_loading_again() {
        let now = Instant::now();
        assert!(!loaded(now, Duration::from_millis(300), true).due(Source::Collections, now), "a burst of changes loads once");
        assert!(loaded(now, STALE_MIN_AGE, true).due(Source::Collections, now));
        assert!(loaded(now, Duration::from_secs(5), true).due(Source::Library, now));
    }

    #[test]
    fn a_fresh_list_waits_for_its_timer_and_a_loading_one_never_starts_twice() {
        let now = Instant::now();
        assert!(!loaded(now, Duration::from_secs(5), false).due(Source::Collections, now));
        assert!(loaded(now, Source::Collections.ttl(), false).due(Source::Collections, now));
        let mut busy = Meta::default();
        busy.loading = true;
        assert!(!busy.due(Source::Library, now));
    }

    #[test]
    fn a_failed_list_is_tried_again_after_the_retry_wait_even_when_stale_is_off() {
        let now = Instant::now();
        let mut failed = loaded(now, Duration::from_secs(3), false);
        failed.error = Some("down".into());
        assert!(!failed.due(Source::Tags, now));
        failed.loaded_at = Some(now - ERROR_RETRY);
        assert!(failed.due(Source::Tags, now));
    }

    #[test]
    fn invalidating_everything_leaves_the_places_alone() {
        let data = SidebarData::default();
        data.invalidate_all();
        for source in Source::ALL {
            assert_eq!(data.meta(source).stale, source != Source::Places, "{source:?}");
        }
        data.invalidate(Source::Places);
        assert!(data.meta(Source::Places).stale, "an explicit request still marks them");
    }

    #[test]
    fn a_refresh_request_marks_the_library_lists_stale_on_the_next_poll() {
        let data = SidebarData::default();
        data.with(|i| i.auto_load = false);
        request_refresh();
        data.poll(&Context::default(), Source::Library, false, Instant::now());
        assert!(data.meta(Source::Tags).stale);
        assert!(!data.meta(Source::Places).stale);
    }
}
