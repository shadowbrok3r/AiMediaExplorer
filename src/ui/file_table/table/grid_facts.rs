//! Per-tile facts that are not on the row: offline state, folder item counts, collection counts and video durations, probed in the background for visible tiles.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crossbeam::channel::{Receiver, Sender, unbounded};
use egui::Context;

/// Folder item counts stop here and show as "N+".
pub const DIR_COUNT_CAP: u32 = 999;
/// Probes one background job runs.
const BATCH: usize = 64;
/// Background jobs running at once.
const MAX_INFLIGHT: usize = 2;
/// Time a tile must stay drawn before its probes start.
const LINGER: Duration = Duration::from_millis(60);
/// Frames a queued probe survives without its tile being drawn.
const PROBE_PATIENCE: u64 = 6;
/// Video probes one background job runs; small so a job ends soon after its tiles scroll away.
const DURATION_BATCH: usize = 4;
/// Pump calls a queued video probe survives without its tile being drawn.
const DURATION_PATIENCE: u64 = 2;
/// Known paths kept before the cache starts over.
const KNOWN_CAP: usize = 30_000;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Facts {
    pub offline: Option<bool>,
    pub dir_items: Option<u32>,
    pub collections: Option<u32>,
    pub duration_secs: Option<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ProbeKind {
    /// Whether the file is missing from the disk.
    Offline,
    /// Entries in a folder.
    DirItems,
    /// Collections the path is a member of.
    Collections,
    /// Playing time of a video.
    Duration,
}

impl ProbeKind {
    fn bit(self) -> u8 {
        match self {
            ProbeKind::Offline => 1,
            ProbeKind::DirItems => 2,
            ProbeKind::Collections => 4,
            ProbeKind::Duration => 8,
        }
    }
}

/// What a finished probe found for one path.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Found {
    pub offline: Option<bool>,
    pub dir_items: Option<u32>,
    pub collections: Option<u32>,
    pub duration_secs: Option<f32>,
}

/// Where the probes read from; a stand-in answers without touching disk or database in tests.
pub trait ProbeSource: Send + Sync + 'static {
    fn run(&self, batch: &[(String, ProbeKind)]) -> Vec<(String, Found)>;
}

fn is_virtual(path: &str) -> bool {
    path.contains("://")
}

/// Probes the real disk and the collections tables.
pub struct SystemProbes;

impl ProbeSource for SystemProbes {
    fn run(&self, batch: &[(String, ProbeKind)]) -> Vec<(String, Found)> {
        let mut out: HashMap<String, Found> = HashMap::new();
        let mut member_paths: Vec<String> = Vec::new();
        for (path, kind) in batch {
            let found = out.entry(path.clone()).or_default();
            match kind {
                ProbeKind::Offline => found.offline = Some(!is_virtual(path) && std::fs::metadata(path).is_err()),
                ProbeKind::DirItems => {
                    if !is_virtual(path) {
                        found.dir_items = std::fs::read_dir(path).ok().map(|d| d.take(DIR_COUNT_CAP as usize + 1).count() as u32);
                    }
                }
                ProbeKind::Collections => member_paths.push(path.clone()),
                ProbeKind::Duration => {
                    found.duration_secs = crate::video::probe(std::path::Path::new(path)).ok().and_then(|info| info.duration_secs).map(|d| d as f32);
                }
            }
        }
        if !member_paths.is_empty() && crate::database::is_connected() {
            let handle = crate::ui::file_table::thumbs::service().runtime().clone();
            crate::database::scope_queries::counter::note(crate::database::scope_queries::counter::Site::TileFacts);
            let counts = handle.block_on(async { crate::database::collections::counts_for(&crate::database::DB, &member_paths).await });
            match counts {
                Ok(counts) => {
                    for path in &member_paths {
                        out.entry(path.clone()).or_default().collections = Some(counts.get(path).copied().unwrap_or(0));
                    }
                }
                Err(e) => log::debug!("[grid] collection counts failed: {e:#}"),
            }
        }
        out.into_iter().collect()
    }
}

/// One probe waiting to start.
#[derive(Clone, Copy, Debug)]
struct Waiting {
    /// When the tile was first drawn.
    since: Instant,
    /// The pump frame the tile was last drawn in.
    seen: u64,
    /// When the tile was last drawn.
    seen_at: Instant,
    order: u64,
}

/// The waiting probes of one path, one slot per kind except videos.
#[derive(Default)]
struct Slots([Option<Waiting>; 3]);

const WAITING_KINDS: [ProbeKind; 3] = [ProbeKind::Offline, ProbeKind::DirItems, ProbeKind::Collections];

fn slot_of(kind: ProbeKind) -> Option<usize> {
    WAITING_KINDS.iter().position(|k| *k == kind)
}

/// Facts about the paths that were drawn, gathered without blocking the UI.
pub struct TileFacts {
    known: HashMap<String, Facts>,
    asked: HashMap<String, u8>,
    /// Probes waiting for their tile to stay on screen, by path.
    waiting: HashMap<String, Slots>,
    waiting_count: usize,
    next_order: u64,
    linger: Duration,
    patience: u64,
    /// When the current frame was pumped.
    now: Instant,
    dropped_probes: u64,
    /// Queued video probes with the pump frame their tile was last drawn in.
    durations: HashMap<String, u64>,
    inflight: usize,
    duration_inflight: usize,
    tx: Sender<Done>,
    rx: Receiver<Done>,
    source: std::sync::Arc<dyn ProbeSource>,
    frame: u64,
    started: u64,
    finished: u64,
    dropped: u64,
}

struct Done {
    duration: bool,
    found: Vec<(String, Found)>,
}

impl Default for TileFacts {
    fn default() -> Self {
        Self::new(std::sync::Arc::new(SystemProbes))
    }
}

impl TileFacts {
    pub fn new(source: std::sync::Arc<dyn ProbeSource>) -> Self {
        Self::with_linger(source, LINGER)
    }

    /// Facts whose probes start after their tile stayed drawn for `linger`.
    pub fn with_linger(source: std::sync::Arc<dyn ProbeSource>, linger: Duration) -> Self {
        Self::with_tuning(source, linger, PROBE_PATIENCE)
    }

    /// Facts whose probes start after their tile stayed drawn for `linger` and are dropped once it was not drawn for `patience` frames.
    pub fn with_tuning(source: std::sync::Arc<dyn ProbeSource>, linger: Duration, patience: u64) -> Self {
        let (tx, rx) = unbounded();
        Self { known: HashMap::new(), asked: HashMap::new(), waiting: HashMap::new(), waiting_count: 0, next_order: 0, linger, patience, now: Instant::now(), dropped_probes: 0, durations: HashMap::new(), inflight: 0, duration_inflight: 0, tx, rx, source, frame: 0, started: 0, finished: 0, dropped: 0 }
    }

    pub fn get(&self, path: &str) -> Facts {
        self.known.get(path).copied().unwrap_or_default()
    }

    /// Records a duration known from elsewhere, such as a video probe.
    pub fn set_duration(&mut self, path: &str, secs: f32) {
        self.known.entry(path.to_string()).or_default().duration_secs = Some(secs);
    }

    /// Queues a probe for `path` unless one was asked for before; a queued video probe is kept alive while its tile is drawn.
    pub fn want(&mut self, path: &str, kind: ProbeKind) {
        match self.asked.get_mut(path) {
            Some(bits) if *bits & kind.bit() != 0 => {
                if kind == ProbeKind::Duration {
                    if let Some(seen) = self.durations.get_mut(path) {
                        *seen = self.frame;
                    }
                } else if let Some(waiting) = self.waiting.get_mut(path).and_then(|slots| slot_of(kind).and_then(|i| slots.0[i].as_mut())) {
                    waiting.seen = self.frame;
                    waiting.seen_at = self.now;
                }
                return;
            }
            Some(bits) => *bits |= kind.bit(),
            None => {
                self.asked.insert(path.to_string(), kind.bit());
            }
        }
        match slot_of(kind) {
            None => {
                self.durations.insert(path.to_string(), self.frame);
            }
            Some(slot) => {
                let waiting = Waiting { since: self.now, seen: self.frame, seen_at: self.now, order: self.next_order };
                self.next_order += 1;
                self.waiting.entry(path.to_string()).or_default().0[slot] = Some(waiting);
                self.waiting_count += 1;
            }
        }
    }

    // Drops the waiting probes whose tiles went away and takes up to a batch of those whose tiles stayed drawn for the linger time, oldest first.
    fn take_ready(&mut self) -> Vec<(String, ProbeKind)> {
        let oldest_seen = self.frame.saturating_sub(self.patience);
        let mut ready: Vec<(u64, String, ProbeKind)> = Vec::new();
        let mut stale: Vec<(String, ProbeKind)> = Vec::new();
        for (path, slots) in &self.waiting {
            for (slot, waiting) in slots.0.iter().enumerate() {
                let Some(waiting) = waiting else { continue };
                if waiting.seen < oldest_seen {
                    stale.push((path.clone(), WAITING_KINDS[slot]));
                } else if waiting.seen_at.saturating_duration_since(waiting.since) >= self.linger {
                    ready.push((waiting.order, path.clone(), WAITING_KINDS[slot]));
                }
            }
        }
        for (path, kind) in stale {
            self.remove_waiting(&path, kind);
            if let Some(bits) = self.asked.get_mut(&path) {
                *bits &= !kind.bit();
            }
            self.dropped_probes += 1;
        }
        ready.sort_unstable_by_key(|(order, _, _)| *order);
        ready.truncate(BATCH);
        ready
            .into_iter()
            .map(|(_, path, kind)| {
                self.remove_waiting(&path, kind);
                (path, kind)
            })
            .collect()
    }

    fn remove_waiting(&mut self, path: &str, kind: ProbeKind) {
        let Some(slot) = slot_of(kind) else { return };
        let Some(slots) = self.waiting.get_mut(path) else { return };
        if slots.0[slot].take().is_some() {
            self.waiting_count -= 1;
        }
        if slots.0.iter().all(Option::is_none) {
            self.waiting.remove(path);
        }
    }

    // The shortest time until a waiting probe has been drawn long enough.
    fn next_ready_in(&self) -> Option<Duration> {
        self.waiting.values().flat_map(|slots| slots.0.iter().flatten()).map(|w| self.linger.saturating_sub(w.seen_at.saturating_duration_since(w.since))).min()
    }

    /// Applies finished probes and starts queued ones; returns whether any fact changed.
    pub fn pump(&mut self, ctx: &Context) -> bool {
        self.frame += 1;
        self.now = Instant::now();
        let mut changed = false;
        while let Ok(done) = self.rx.try_recv() {
            if done.duration {
                self.duration_inflight = self.duration_inflight.saturating_sub(1);
            } else {
                self.inflight = self.inflight.saturating_sub(1);
            }
            self.finished += 1;
            for (path, found) in done.found {
                let facts = self.known.entry(path).or_default();
                facts.offline = found.offline.or(facts.offline);
                facts.dir_items = found.dir_items.or(facts.dir_items);
                facts.collections = found.collections.or(facts.collections);
                facts.duration_secs = found.duration_secs.or(facts.duration_secs);
                changed = true;
            }
        }
        if self.known.len() > KNOWN_CAP {
            let durations: Vec<(String, f32)> = self.known.iter().filter_map(|(p, f)| f.duration_secs.map(|d| (p.clone(), d))).collect();
            self.known.clear();
            self.asked.clear();
            self.waiting.clear();
            self.waiting_count = 0;
            self.durations.clear();
            for (p, d) in durations {
                self.known.entry(p).or_default().duration_secs = Some(d);
            }
        }
        while self.inflight < MAX_INFLIGHT {
            let batch = self.take_ready();
            if batch.is_empty() {
                break;
            }
            self.spawn(batch, false, ctx.clone());
        }
        if let Some(wait) = self.next_ready_in() {
            ctx.request_repaint_after(wait.max(Duration::from_millis(1)));
        }
        self.drop_stale_durations();
        while self.duration_inflight < MAX_INFLIGHT && !self.durations.is_empty() {
            let batch = self.nearest_durations();
            self.spawn(batch, true, ctx.clone());
        }
        changed
    }

    // Forgets queued video probes whose tiles have not been drawn for a while, so they can be asked again.
    fn drop_stale_durations(&mut self) {
        let oldest = self.frame.saturating_sub(DURATION_PATIENCE);
        let stale: Vec<String> = self.durations.iter().filter(|(_, seen)| **seen < oldest).map(|(p, _)| p.clone()).collect();
        for path in stale {
            self.durations.remove(&path);
            if let Some(bits) = self.asked.get_mut(&path) {
                *bits &= !ProbeKind::Duration.bit();
            }
            self.dropped += 1;
        }
    }

    // Takes the most recently drawn queued video probes.
    fn nearest_durations(&mut self) -> Vec<(String, ProbeKind)> {
        let mut by_age: Vec<(u64, String)> = self.durations.iter().map(|(p, seen)| (*seen, p.clone())).collect();
        by_age.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        by_age.truncate(DURATION_BATCH);
        by_age
            .into_iter()
            .map(|(_, path)| {
                self.durations.remove(&path);
                (path, ProbeKind::Duration)
            })
            .collect()
    }

    fn spawn(&mut self, batch: Vec<(String, ProbeKind)>, duration: bool, ctx: Context) {
        if duration {
            self.duration_inflight += 1;
        } else {
            self.inflight += 1;
        }
        self.started += 1;
        let (tx, source) = (self.tx.clone(), self.source.clone());
        let job = move || {
            let found = source.run(&batch);
            let _ = tx.send(Done { duration, found });
            ctx.request_repaint_after(Duration::from_millis(16));
        };
        if std::thread::Builder::new().name("grid-facts".into()).spawn(job).is_err() {
            if duration {
                self.duration_inflight -= 1;
            } else {
                self.inflight -= 1;
            }
        }
    }

    /// Probes waiting to start.
    pub fn queued(&self) -> usize {
        self.waiting_count + self.durations.len()
    }

    pub fn inflight(&self) -> usize {
        self.inflight + self.duration_inflight
    }

    /// Video probes dropped because their tiles scrolled away before they started.
    pub fn dropped_durations(&self) -> u64 {
        self.dropped
    }

    /// Probes dropped because their tiles stopped being drawn.
    pub fn dropped_probes(&self) -> u64 {
        self.dropped_probes
    }

    /// Paths with at least one known fact.
    pub fn known_len(&self) -> usize {
        self.known.len()
    }

    /// Batches started and finished since the start.
    pub fn batches(&self) -> (u64, u64) {
        (self.started, self.finished)
    }

    /// Forgets everything so visible tiles are probed again.
    pub fn reset(&mut self) {
        self.known.clear();
        self.asked.clear();
        self.waiting.clear();
        self.waiting_count = 0;
        self.durations.clear();
    }

    /// Paths already asked about, for tests.
    #[cfg(test)]
    pub fn asked_len(&self) -> usize {
        self.asked.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Instant;

    struct Fake {
        batches: AtomicUsize,
        largest: AtomicUsize,
    }

    impl ProbeSource for Fake {
        fn run(&self, batch: &[(String, ProbeKind)]) -> Vec<(String, Found)> {
            self.batches.fetch_add(1, Ordering::SeqCst);
            self.largest.fetch_max(batch.len(), Ordering::SeqCst);
            batch
                .iter()
                .map(|(p, k)| {
                    let mut f = Found::default();
                    match k {
                        ProbeKind::Offline => f.offline = Some(p.contains("gone")),
                        ProbeKind::DirItems => f.dir_items = Some(7),
                        ProbeKind::Collections => f.collections = Some(2),
                        ProbeKind::Duration => f.duration_secs = Some(12.5),
                    }
                    (p.clone(), f)
                })
                .collect()
        }
    }

    fn fake() -> (Arc<Fake>, TileFacts) {
        let src = Arc::new(Fake { batches: AtomicUsize::new(0), largest: AtomicUsize::new(0) });
        (src.clone(), TileFacts::with_tuning(src, Duration::ZERO, u64::MAX))
    }

    fn settle(facts: &mut TileFacts) {
        let ctx = Context::default();
        let start = Instant::now();
        loop {
            facts.pump(&ctx);
            if facts.inflight() == 0 && facts.queued() == 0 {
                return;
            }
            assert!(start.elapsed() < Duration::from_secs(5), "probes never finished");
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn a_probe_is_asked_once_per_path_and_kind() {
        let (src, mut facts) = fake();
        for _ in 0..5 {
            facts.want("/a.jpg", ProbeKind::Offline);
            facts.want("/a.jpg", ProbeKind::Collections);
        }
        settle(&mut facts);
        assert_eq!(src.largest.load(Ordering::SeqCst), 2);
        assert_eq!(facts.get("/a.jpg"), Facts { offline: Some(false), collections: Some(2), ..Facts::default() });
        facts.want("/a.jpg", ProbeKind::Offline);
        assert_eq!(facts.queued(), 0, "already asked");
        facts.want("/a.jpg", ProbeKind::DirItems);
        assert_eq!(facts.queued(), 1, "a new kind is a new question");
    }

    #[test]
    fn probes_run_in_bounded_batches() {
        let (src, mut facts) = fake();
        for i in 0..200 {
            facts.want(&format!("/p/{i}.jpg"), ProbeKind::Offline);
        }
        settle(&mut facts);
        assert!(src.largest.load(Ordering::SeqCst) <= BATCH);
        assert_eq!(src.batches.load(Ordering::SeqCst), 4);
        assert_eq!(facts.known_len(), 200);
        assert_eq!(facts.batches(), (4, 4));
    }

    fn source() -> Arc<Fake> {
        Arc::new(Fake { batches: AtomicUsize::new(0), largest: AtomicUsize::new(0) })
    }

    #[test]
    fn a_probe_waits_until_its_tile_has_stayed_drawn_for_the_linger_time() {
        let src = source();
        let mut facts = TileFacts::with_linger(src.clone(), Duration::from_millis(40));
        let ctx = Context::default();
        facts.pump(&ctx);
        facts.want("/stays.jpg", ProbeKind::Offline);
        facts.pump(&ctx);
        assert_eq!((facts.inflight(), facts.queued()), (0, 1), "too early");
        std::thread::sleep(Duration::from_millis(50));
        facts.pump(&ctx);
        facts.want("/stays.jpg", ProbeKind::Offline);
        facts.pump(&ctx);
        assert_eq!((facts.inflight(), facts.queued()), (1, 0), "the tile stayed drawn");
        settle(&mut facts);
        assert_eq!(facts.get("/stays.jpg").offline, Some(false));
    }

    #[test]
    fn a_tile_drawn_once_is_never_probed_however_long_the_probe_waits() {
        let src = source();
        let mut facts = TileFacts::with_tuning(src.clone(), Duration::from_millis(10), u64::MAX);
        let ctx = Context::default();
        facts.pump(&ctx);
        facts.want("/once.jpg", ProbeKind::Offline);
        std::thread::sleep(Duration::from_millis(30));
        for _ in 0..5 {
            facts.pump(&ctx);
        }
        assert_eq!((facts.inflight(), facts.queued()), (0, 1), "it was drawn for no time at all");
        assert_eq!(src.batches.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn a_probe_whose_tile_scrolled_away_is_dropped_and_asked_again_when_it_returns() {
        let src = source();
        let mut facts = TileFacts::with_linger(src.clone(), Duration::from_secs(3600));
        let ctx = Context::default();
        facts.want("/gone.jpg", ProbeKind::Collections);
        for _ in 0..PROBE_PATIENCE + 2 {
            facts.pump(&ctx);
        }
        assert_eq!(facts.queued(), 0);
        assert_eq!(facts.dropped_probes(), 1);
        assert_eq!(src.batches.load(Ordering::SeqCst), 0, "it never started");
        facts.want("/gone.jpg", ProbeKind::Collections);
        assert_eq!(facts.queued(), 1, "the returning tile asks again");
    }

    #[test]
    fn a_probe_for_a_tile_that_stays_drawn_is_kept_alive() {
        let mut facts = TileFacts::with_linger(source(), Duration::from_secs(3600));
        let ctx = Context::default();
        for _ in 0..PROBE_PATIENCE * 3 {
            facts.pump(&ctx);
            facts.want("/here.jpg", ProbeKind::Offline);
        }
        assert_eq!((facts.queued(), facts.dropped_probes()), (1, 0));
    }

    #[test]
    fn scrolling_past_tiles_starts_no_probes_at_all() {
        let src = source();
        let mut facts = TileFacts::with_linger(src.clone(), Duration::from_millis(60));
        let ctx = Context::default();
        let frames = 80u64;
        for frame in 0..frames {
            facts.pump(&ctx);
            for col in 0..6 {
                facts.want(&format!("/scroll/{frame}_{col}.jpg"), ProbeKind::Collections);
                facts.want(&format!("/scroll/{frame}_{col}.jpg"), ProbeKind::Offline);
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        settle(&mut facts);
        assert_eq!(src.batches.load(Ordering::SeqCst), 0, "no tile stayed drawn long enough");
        assert_eq!(facts.known_len(), 0);
        assert_eq!(facts.dropped_probes(), frames * 12);
    }

    #[test]
    fn tiles_that_stop_scrolling_are_probed_in_batches() {
        let src = source();
        let mut facts = TileFacts::with_linger(src.clone(), Duration::from_millis(30));
        let ctx = Context::default();
        let started = Instant::now();
        while started.elapsed() < Duration::from_millis(150) {
            facts.pump(&ctx);
            for i in 0..100 {
                facts.want(&format!("/rest/{i}.jpg"), ProbeKind::Offline);
            }
            std::thread::sleep(Duration::from_millis(3));
        }
        settle(&mut facts);
        assert_eq!(facts.known_len(), 100);
        assert!(src.largest.load(Ordering::SeqCst) <= BATCH);
        assert!(src.batches.load(Ordering::SeqCst) <= 4, "{} batches", src.batches.load(Ordering::SeqCst));
    }

    #[test]
    fn the_oldest_waiting_probes_start_first() {
        let src = source();
        let mut facts = TileFacts::with_tuning(src.clone(), Duration::ZERO, u64::MAX);
        let ctx = Context::default();
        for i in 0..BATCH * 2 + 10 {
            facts.want(&format!("/order/{i:03}.jpg"), ProbeKind::Offline);
        }
        facts.pump(&ctx);
        assert_eq!((facts.inflight(), facts.queued()), (2, 10), "two full batches started");
        let mut left: Vec<&String> = facts.waiting.keys().collect();
        left.sort();
        assert_eq!(left.first().map(|p| p.as_str()), Some("/order/128.jpg"), "the newest ten wait");
    }

    #[test]
    fn a_reset_forgets_the_waiting_probes() {
        let (_, mut facts) = fake();
        facts.want("/a", ProbeKind::Offline);
        facts.want("/b", ProbeKind::Collections);
        assert_eq!(facts.queued(), 2);
        facts.reset();
        assert_eq!(facts.queued(), 0);
        assert!(facts.next_ready_in().is_none());
    }

    #[test]
    fn video_probes_run_in_small_batches() {
        let (src, mut facts) = fake();
        for i in 0..10 {
            facts.want(&format!("/v/{i}.mp4"), ProbeKind::Duration);
        }
        let ctx = Context::default();
        let start = Instant::now();
        loop {
            for i in 0..10 {
                facts.want(&format!("/v/{i}.mp4"), ProbeKind::Duration);
            }
            facts.pump(&ctx);
            if (0..10).all(|i| facts.get(&format!("/v/{i}.mp4")).duration_secs.is_some()) {
                break;
            }
            assert!(start.elapsed() < Duration::from_secs(5), "probes never finished");
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!(src.largest.load(Ordering::SeqCst) <= DURATION_BATCH);
        assert_eq!(facts.dropped_durations(), 0, "tiles that stay drawn keep their probes");
    }

    #[test]
    fn a_video_probe_is_dropped_when_its_tile_scrolls_away_and_asked_again_when_it_returns() {
        let (src, mut facts) = fake();
        facts.frame = 100;
        facts.want("/gone.mp4", ProbeKind::Duration);
        facts.drop_stale_durations();
        assert_eq!(facts.queued(), 1, "a probe for a tile drawn this frame stays");
        facts.frame += DURATION_PATIENCE + 1;
        facts.drop_stale_durations();
        assert_eq!(facts.queued(), 0);
        assert_eq!(facts.dropped_durations(), 1);
        assert_eq!(src.batches.load(Ordering::SeqCst), 0, "it never started");
        facts.want("/gone.mp4", ProbeKind::Duration);
        assert_eq!(facts.queued(), 1, "the returning tile asks again");
        settle(&mut facts);
        assert_eq!(facts.get("/gone.mp4").duration_secs, Some(12.5));
    }

    #[test]
    fn the_most_recently_drawn_video_probes_start_first() {
        let (_, mut facts) = fake();
        for (frame, path) in [(10, "/old.mp4"), (20, "/mid.mp4"), (30, "/new.mp4")] {
            facts.frame = frame;
            facts.want(path, ProbeKind::Duration);
        }
        let batch = facts.nearest_durations();
        let order: Vec<&str> = batch.iter().map(|(p, _)| p.as_str()).collect();
        assert_eq!(order, ["/new.mp4", "/mid.mp4", "/old.mp4"]);
        assert_eq!(facts.queued(), 0);
    }

    #[test]
    fn nothing_runs_without_a_pump_and_unknown_paths_are_empty() {
        let (src, mut facts) = fake();
        facts.want("/a", ProbeKind::Offline);
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(src.batches.load(Ordering::SeqCst), 0);
        assert_eq!(facts.get("/a"), Facts::default());
    }

    #[test]
    fn durations_set_from_outside_survive_a_cache_reset() {
        let (_, mut facts) = fake();
        facts.set_duration("/v.mp4", 83.0);
        assert_eq!(facts.get("/v.mp4").duration_secs, Some(83.0));
        let ctx = Context::default();
        for i in 0..KNOWN_CAP + 5 {
            facts.known.insert(format!("/x/{i}"), Facts::default());
        }
        facts.pump(&ctx);
        assert!(facts.known_len() < 10, "the cache started over");
        assert_eq!(facts.get("/v.mp4").duration_secs, Some(83.0));
        assert_eq!(facts.asked_len(), 0);
    }

    #[test]
    fn the_system_probe_reads_the_disk() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.jpg");
        std::fs::write(&file, b"x").unwrap();
        for i in 0..3 {
            std::fs::write(dir.path().join(format!("f{i}")), b"x").unwrap();
        }
        let gone = dir.path().join("gone.jpg").display().to_string();
        let file = file.display().to_string();
        let folder = dir.path().display().to_string();
        let got: HashMap<String, Found> = SystemProbes
            .run(&[(file.clone(), ProbeKind::Offline), (gone.clone(), ProbeKind::Offline), (folder.clone(), ProbeKind::DirItems), ("zip:///x.zip!/a.jpg".into(), ProbeKind::Offline)])
            .into_iter()
            .collect();
        assert_eq!(got[&file].offline, Some(false));
        assert_eq!(got[&gone].offline, Some(true));
        assert_eq!(got[&folder].dir_items, Some(4));
        assert_eq!(got["zip:///x.zip!/a.jpg"].offline, Some(false), "archive entries are never offline");
    }

    #[test]
    fn folder_counts_stop_at_the_cap() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..DIR_COUNT_CAP + 20 {
            std::fs::write(dir.path().join(format!("f{i}")), b"").unwrap();
        }
        let folder = dir.path().display().to_string();
        let got = SystemProbes.run(&[(folder.clone(), ProbeKind::DirItems)]);
        assert_eq!(got[0].1.dir_items, Some(DIR_COUNT_CAP + 1));
    }
}
