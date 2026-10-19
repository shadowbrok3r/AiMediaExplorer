//! Command bus: requests to the application as serializable commands, run once per frame (see `docs/commands.md`).

pub mod actions;
pub mod browser;
pub mod bus;
pub mod cli;
pub mod context;
pub mod dispatch;
pub mod legacy;
pub mod log;
pub mod model;
pub mod scope;
pub mod sink;
#[cfg(test)]
mod sink_tests;

pub use bus::{CommandBus, CommandSender, global, install_repaint, request_repaint, send, sender};
pub use dispatch::{CommandError, CommandResult, CommandSink, ErrorKind, FRAME_BUDGET, Pumped, dispatch, pump};
pub use log::{CommandLog, LOG_CAPACITY, LogEntry, Outcome, with_log};
pub use browser::{DialogKind, QueryMode};
pub use model::*;
pub use scope::{Scope, ScopeParseError, SmartView};

/// Failed commands that become toasts in one frame at most.
const FAILURE_TOASTS: usize = 3;

/// Runs the queued commands of the global bus against `sink`, records them in the global log and toasts failures.
pub fn pump_global(sink: &mut dyn CommandSink) -> usize {
    let mut local = CommandLog::with_capacity(FRAME_BUDGET);
    let pumped = pump(global(), &mut local, sink, FRAME_BUDGET);
    if pumped.ran > 0 {
        with_log(|log| log.append(local));
    }
    for (name, error) in pumped.failures.iter().take(FAILURE_TOASTS) {
        let _ = sink.toast(ToastLevel::Warning, format!("{name}: {}", error.message));
    }
    pumped.ran
}

/// Parses `text` with [`parse_input`] and queues the commands; returns how many.
pub fn send_text(text: &str) -> Result<usize, String> {
    let commands = parse_input(text)?;
    let count = commands.len();
    commands.into_iter().for_each(send);
    Ok(count)
}
