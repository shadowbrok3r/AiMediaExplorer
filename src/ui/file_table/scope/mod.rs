//! What a browser tab shows and how it got there: the current scope, its history and the loader feeding the rows.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crossbeam::channel::{Receiver, Sender};

use crate::command::Scope;

pub mod badge;
pub mod duplicates;
pub mod failed;
pub mod history;
pub mod load;
pub mod nav;
pub mod pending;
pub mod runner;

pub use badge::{Badge, Badges};
pub use history::{Entry, History};
pub use load::{Outcome, Request, RowBatch};
pub use pending::{Pending, PendingKind};

#[cfg(test)]
mod nav_tests;

/// Loads kept in the log of a tab.
pub const LOG_CAPACITY: usize = 20;

/// A message from a loader to the tab that asked for the load.
#[derive(Clone, Debug)]
pub struct LoadMsg {
    /// Id of the load; 0 for messages that do not belong to one.
    pub load: u64,
    pub body: LoadBody,
}

#[derive(Clone, Debug)]
pub enum LoadBody {
    Rows(RowBatch),
    /// Files known to have a CLIP embedding.
    Embedded(Vec<String>),
    Done(Outcome),
    Failed(String),
}

/// The load a tab is waiting for.
#[derive(Clone, Debug)]
pub struct ActiveLoad {
    pub id: u64,
    pub scope: Scope,
    pub cancel: Arc<AtomicBool>,
    pub started: Instant,
    pub rows: usize,
    pub batches: usize,
    pub finished: bool,
    pub cancelled: bool,
    /// The load fetches a further page of the scope already shown.
    pub page: bool,
}

impl ActiveLoad {
    pub fn is_running(&self) -> bool {
        !self.finished && !self.cancelled
    }
}

/// How a past load went.
#[derive(Clone, Debug)]
pub struct LoadRecord {
    pub scope: String,
    pub kind: &'static str,
    pub rows: usize,
    pub batches: usize,
    pub first_batch: Option<Duration>,
    pub elapsed: Duration,
    pub error: Option<String>,
    pub cancelled: bool,
    pub total: Option<u64>,
}

/// Where a search continues once the user asks for more.
#[derive(Clone, Debug)]
pub struct More {
    pub start: usize,
    pub vector: Option<Arc<Vec<f32>>>,
}

/// The view to bring back after a load: the first row shown and the selected row.
#[derive(Clone, Debug, PartialEq)]
pub struct Restore {
    pub scroll_row: Option<usize>,
    pub anchor: Option<String>,
}

/// Navigation state of one tab.
pub struct Nav {
    /// What the tab shows.
    pub scope: Scope,
    pub history: History,
    pub active: Option<ActiveLoad>,
    /// Text of the address box.
    pub input: String,
    /// Title shown instead of the one the scope gives, until the tab navigates.
    pub title: Option<String>,
    pub more: Option<More>,
    pub restore: Option<Restore>,
    /// The tab shows rows held in memory, which no scope can load again.
    pub rows_only: bool,
    /// Why the last load failed.
    pub error: Option<String>,
    pub log: VecDeque<LoadRecord>,
    /// Loads started by this tab, for ids.
    pub loads_started: u64,
    pub tx: Sender<LoadMsg>,
    pub rx: Receiver<LoadMsg>,
    /// Paths of the rows loaded for the engine to learn about.
    pub hydrate: Vec<String>,
    /// When the filesystem listing of the current scope began.
    pub fs_started: Option<Instant>,
    /// A restored scope that is not loaded yet, or cannot be.
    pub pending: Option<Pending>,
    /// Further paths to select once the first of the restored selection is in.
    pub restore_select: Vec<String>,
}

impl Nav {
    pub fn new(scope: Scope) -> Self {
        let (tx, rx) = crossbeam::channel::unbounded();
        Self {
            input: scope.format(),
            history: History::new(scope.clone()),
            scope,
            active: None,
            title: None,
            more: None,
            restore: None,
            rows_only: false,
            error: None,
            log: VecDeque::new(),
            loads_started: 0,
            tx,
            rx,
            hydrate: Vec::new(),
            fs_started: None,
            pending: None,
            restore_select: Vec::new(),
        }
    }

    pub fn push_log(&mut self, record: LoadRecord) {
        if self.log.len() >= LOG_CAPACITY {
            self.log.pop_front();
        }
        self.log.push_back(record);
    }

    /// Stops the load in flight.
    pub fn cancel_active(&mut self) {
        if let Some(active) = self.active.as_mut().filter(|a| a.is_running()) {
            active.cancel.store(true, Ordering::Relaxed);
            active.cancelled = true;
        }
    }

    pub fn is_loading(&self) -> bool {
        self.active.as_ref().is_some_and(ActiveLoad::is_running)
    }
}

/// Longest error text kept from a failed load.
const ERROR_MAX: usize = 240;

/// First line of `text`, cut to a length the interface can show.
pub fn one_line(text: &str) -> String {
    let first = text.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or_default();
    if first.chars().count() <= ERROR_MAX { first.to_string() } else { first.chars().take(ERROR_MAX - 3).collect::<String>() + "..." }
}

/// Text an address box accepts as a scope: a scope string, or a plain path that means a folder.
pub fn parse_address(text: &str) -> Result<Scope, String> {
    let text = text.trim();
    if text.is_empty() {
        return Err("type a path or a scope".to_string());
    }
    if let Ok(scope) = Scope::parse(text) {
        return Ok(scope);
    }
    let looks_like_path = text.starts_with('/') || text.starts_with('~') || text.starts_with("\\\\") || text.chars().nth(1) == Some(':') || text.contains(['/', '\\']);
    if looks_like_path { Ok(Scope::Folder(text.to_string())) } else { Scope::parse(text).map_err(|e| e.to_string()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::SmartView;

    #[test]
    fn the_address_box_reads_scope_strings_and_plain_paths() {
        assert_eq!(parse_address("lib:all"), Ok(Scope::Library));
        assert_eq!(parse_address("  tag:sunset "), Ok(Scope::Tag("sunset".into())));
        assert_eq!(parse_address("smart:duplicates"), Ok(Scope::Smart(SmartView::Duplicates)));
        assert_eq!(parse_address("/home/u/Pictures"), Ok(Scope::Folder("/home/u/Pictures".into())));
        assert_eq!(parse_address("C:\\Users\\u"), Ok(Scope::Folder("C:\\Users\\u".into())));
        assert_eq!(parse_address("\\\\wsl.localhost\\Ubuntu"), Ok(Scope::Folder("\\\\wsl.localhost\\Ubuntu".into())));
        assert_eq!(parse_address("zip:///a.zip!/"), Ok(Scope::Archive("zip:///a.zip!/".into())));
        assert_eq!(parse_address("relative/dir"), Ok(Scope::Folder("relative/dir".into())));
    }

    #[test]
    fn the_address_box_rejects_empty_and_unknown_text_with_a_reason() {
        assert!(parse_address("   ").is_err());
        let err = parse_address("wat:1").unwrap_err();
        assert!(err.contains("unknown prefix"), "{err}");
        assert!(parse_address("smart:nope").unwrap_err().contains("unknown smart view"));
    }

    #[test]
    fn an_error_is_cut_to_its_first_line_and_a_readable_length() {
        assert_eq!(one_line("disk full\n  at frame 1\n  at frame 2"), "disk full");
        assert_eq!(one_line("\n\n  indented first line \nsecond"), "indented first line");
        assert_eq!(one_line(""), "");
        let long = one_line(&"x".repeat(5_000));
        assert_eq!(long.chars().count(), ERROR_MAX);
        assert!(long.ends_with("..."));
    }

    #[test]
    fn the_load_log_keeps_the_newest_records() {
        let mut nav = Nav::new(Scope::Library);
        for i in 0..LOG_CAPACITY + 5 {
            nav.push_log(LoadRecord { scope: format!("s{i}"), kind: "library", rows: i, batches: 1, first_batch: None, elapsed: Duration::ZERO, error: None, cancelled: false, total: None });
        }
        assert_eq!(nav.log.len(), LOG_CAPACITY);
        assert_eq!(nav.log.front().unwrap().scope, "s5");
        assert_eq!(nav.log.back().unwrap().rows, LOG_CAPACITY + 4);
    }

    #[test]
    fn cancelling_flags_the_running_load_once() {
        let mut nav = Nav::new(Scope::Library);
        let flag = Arc::new(AtomicBool::new(false));
        nav.active = Some(ActiveLoad { id: 1, scope: Scope::Library, cancel: flag.clone(), started: Instant::now(), rows: 0, batches: 0, finished: false, cancelled: false, page: false });
        assert!(nav.is_loading());
        nav.cancel_active();
        assert!(flag.load(Ordering::Relaxed));
        assert!(!nav.is_loading());
    }
}
