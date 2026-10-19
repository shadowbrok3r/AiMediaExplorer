//! Ring buffer of the commands the dispatcher ran, for the Command log window.

use std::collections::VecDeque;
use std::sync::Mutex;

use chrono::{DateTime, Local};
use once_cell::sync::Lazy;

use super::dispatch::CommandError;

/// Commands kept in the log.
pub const LOG_CAPACITY: usize = 500;

/// What running a command produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    Ok(String),
    Err(CommandError),
}

impl Outcome {
    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Ok(_))
    }

    /// One line describing the result.
    pub fn text(&self) -> String {
        match self {
            Self::Ok(text) => text.clone(),
            Self::Err(e) => e.to_string(),
        }
    }
}

/// One dispatched command.
#[derive(Clone, Debug)]
pub struct LogEntry {
    pub seq: u64,
    pub at: DateTime<Local>,
    pub json: String,
    pub outcome: Outcome,
}

/// The last [`LOG_CAPACITY`] entries, oldest dropped first.
#[derive(Debug)]
pub struct CommandLog {
    entries: VecDeque<LogEntry>,
    capacity: usize,
    next_seq: u64,
}

impl Default for CommandLog {
    fn default() -> Self {
        Self::with_capacity(LOG_CAPACITY)
    }
}

impl CommandLog {
    pub fn with_capacity(capacity: usize) -> Self {
        Self { entries: VecDeque::new(), capacity: capacity.max(1), next_seq: 1 }
    }

    pub fn push(&mut self, json: String, outcome: Outcome) {
        self.push_entry(Local::now(), json, outcome);
    }

    fn push_entry(&mut self, at: DateTime<Local>, json: String, outcome: Outcome) {
        if self.entries.len() == self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(LogEntry { seq: self.next_seq, at, json, outcome });
        self.next_seq += 1;
    }

    /// Moves the entries of `other` to the end of this log, renumbering them.
    pub fn append(&mut self, other: CommandLog) {
        for e in other.entries {
            self.push_entry(e.at, e.json, e.outcome);
        }
    }

    pub fn entries(&self) -> impl DoubleEndedIterator<Item = &LogEntry> + ExactSizeIterator {
        self.entries.iter()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn failures(&self) -> usize {
        self.entries.iter().filter(|e| !e.outcome.is_ok()).count()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

static LOG: Lazy<Mutex<CommandLog>> = Lazy::new(|| Mutex::new(CommandLog::default()));

/// Runs `f` on the application's command log.
pub fn with_log<R>(f: impl FnOnce(&mut CommandLog) -> R) -> R {
    f(&mut LOG.lock().unwrap_or_else(|e| e.into_inner()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_log_keeps_the_newest_entries_and_numbers_them_monotonically() {
        let mut log = CommandLog::with_capacity(3);
        for i in 0..5 {
            log.push(format!("{i}"), Outcome::Ok(String::new()));
        }
        let seqs: Vec<_> = log.entries().map(|e| (e.seq, e.json.clone())).collect();
        assert_eq!(seqs, vec![(3, "2".to_string()), (4, "3".to_string()), (5, "4".to_string())]);
        assert_eq!(log.len(), 3);
    }

    #[test]
    fn the_default_log_holds_five_hundred_entries() {
        let mut log = CommandLog::default();
        for i in 0..(LOG_CAPACITY + 25) {
            log.push(i.to_string(), Outcome::Ok(String::new()));
        }
        assert_eq!(log.len(), 500);
        assert_eq!(log.entries().next().unwrap().json, "25");
    }

    #[test]
    fn appending_keeps_order_renumbers_and_respects_the_capacity() {
        let mut main = CommandLog::with_capacity(3);
        main.push("a".into(), Outcome::Ok(String::new()));
        let mut other = CommandLog::with_capacity(10);
        for j in ["b", "c", "d"] {
            other.push(j.into(), Outcome::Ok(String::new()));
        }
        main.append(other);
        let got: Vec<_> = main.entries().map(|e| (e.seq, e.json.as_str())).collect();
        assert_eq!(got, vec![(2, "b"), (3, "c"), (4, "d")]);
    }

    #[test]
    fn clearing_empties_the_log_and_failures_are_counted() {
        let mut log = CommandLog::default();
        log.push("a".into(), Outcome::Ok("fine".into()));
        log.push("b".into(), Outcome::Err(CommandError::invalid("nope")));
        assert_eq!(log.failures(), 1);
        log.clear();
        assert!(log.is_empty());
    }
}
