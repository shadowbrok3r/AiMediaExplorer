//! Per-tile facts that are not on the row: offline state, folder item counts, collection counts and video durations, probed in the background for visible tiles.

use std::collections::HashMap;
use std::time::Duration;

use crossbeam::channel::{Receiver, Sender, unbounded};
use egui::Context;

/// Folder item counts stop here and show as "N+".
pub const DIR_COUNT_CAP: u32 = 999;
/// Probes one background job runs.
const BATCH: usize = 64;
/// Background jobs running at once.
const MAX_INFLIGHT: usize = 2;
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

/// Facts about the paths that were drawn, gathered without blocking the UI.
pub struct TileFacts {
    known: HashMap<String, Facts>,
    asked: HashMap<String, u8>,
    pending: Vec<(String, ProbeKind)>,
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
        let (tx, rx) = unbounded();
        Self { known: HashMap::new(), asked: HashMap::new(), pending: Vec::new(), durations: HashMap::new(), inflight: 0, duration_inflight: 0, tx, rx, source, frame: 0, started: 0, finished: 0, dropped: 0 }
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
                }
                return;
            }
            Some(bits) => *bits |= kind.bit(),
            None => {
                self.asked.insert(path.to_string(), kind.bit());
            }
        }
        if kind == ProbeKind::Duration {
            self.durations.insert(path.to_string(), self.frame);
        } else {
            self.pending.push((path.to_string(), kind));
        }
    }

    /// Applies finished probes and starts queued ones; returns whether any fact changed.
    pub fn pump(&mut self, ctx: &Context) -> bool {
        self.frame += 1;
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
            self.durations.clear();
            for (p, d) in durations {
                self.known.entry(p).or_default().duration_secs = Some(d);
            }
        }
        while self.inflight < MAX_INFLIGHT && !self.pending.is_empty() {
            let take = self.pending.len().min(BATCH);
            let batch: Vec<(String, ProbeKind)> = self.pending.drain(..take).collect();
            self.spawn(batch, false, ctx.clone());
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
        self.pending.len() + self.durations.len()
    }

    pub fn inflight(&self) -> usize {
        self.inflight + self.duration_inflight
    }

    /// Video probes dropped because their tiles scrolled away before they started.
    pub fn dropped_durations(&self) -> u64 {
        self.dropped
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
        self.pending.clear();
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
        (src.clone(), TileFacts::new(src))
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
