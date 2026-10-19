//! What background work reports back to the Settings page.

use std::path::PathBuf;

use crossbeam::channel::{Receiver, Sender};

use super::metadata_ops::{Op, Preview};
use crate::database::snapshot::SnapshotInfo;

/// A result for the page to show.
#[derive(Debug)]
pub enum Event {
    /// A new library folder is ready, made by a copy or a restore: the folder, or why it failed.
    NewLibrary(Result<PathBuf, String>),
    /// The snapshots found next to the library.
    Snapshots(Vec<SnapshotInfo>),
    /// The count an edit would touch.
    Previewed(Op, Result<Preview, String>),
    /// A bulk edit finished: what it did, or why it failed.
    Edited(Result<String, String>),
    /// A provider answered the connection test: the number of models, or the error.
    Tested(String, Result<usize, String>),
}

pub type Tx = Sender<Event>;
pub type Rx = Receiver<Event>;

/// A channel for page events.
pub fn channel() -> (Tx, Rx) {
    crossbeam::channel::unbounded()
}
