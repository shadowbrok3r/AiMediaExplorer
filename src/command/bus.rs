//! The command channel: any thread sends, the frame loop drains.

use crossbeam::channel::{Receiver, Sender, unbounded};
use once_cell::sync::Lazy;
use std::sync::OnceLock;

use super::model::AppCommand;

/// Cloneable sending end of a [`CommandBus`].
#[derive(Clone)]
pub struct CommandSender {
    tx: Sender<AppCommand>,
}

impl CommandSender {
    /// Queues `cmd`; wakes the UI when a repaint hook is installed.
    pub fn send(&self, cmd: AppCommand) {
        if self.tx.send(cmd).is_ok() {
            wake();
        }
    }
}

/// One unbounded channel of commands in send order.
pub struct CommandBus {
    tx: Sender<AppCommand>,
    rx: Receiver<AppCommand>,
}

impl Default for CommandBus {
    fn default() -> Self {
        Self::new()
    }
}

impl CommandBus {
    pub fn new() -> Self {
        let (tx, rx) = unbounded();
        Self { tx, rx }
    }

    pub fn sender(&self) -> CommandSender {
        CommandSender { tx: self.tx.clone() }
    }

    /// Queues `cmd` without waking the UI.
    pub fn push(&self, cmd: AppCommand) {
        let _ = self.tx.send(cmd);
    }

    /// Up to `max` queued commands, oldest first.
    pub fn drain(&self, max: usize) -> Vec<AppCommand> {
        self.rx.try_iter().take(max).collect()
    }

    pub fn len(&self) -> usize {
        self.rx.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rx.is_empty()
    }
}

static BUS: Lazy<CommandBus> = Lazy::new(CommandBus::new);
static REPAINT: OnceLock<eframe::egui::Context> = OnceLock::new();

/// The application's bus.
pub fn global() -> &'static CommandBus {
    &BUS
}

/// A sending handle to the application's bus.
pub fn sender() -> CommandSender {
    BUS.sender()
}

/// Queues `cmd` on the application's bus from any thread.
pub fn send(cmd: AppCommand) {
    sender().send(cmd);
}

/// Makes every later [`send`] request a repaint of `ctx`.
pub fn install_repaint(ctx: &eframe::egui::Context) {
    let _ = REPAINT.set(ctx.clone());
}

fn wake() {
    request_repaint();
}

/// Wakes the UI when a repaint hook is installed.
pub fn request_repaint() {
    if let Some(ctx) = REPAINT.get() {
        ctx.request_repaint();
    }
}

/// Helpers for tests that share the process-wide bus.
#[cfg(test)]
pub(crate) mod testing {
    use super::*;
    use std::sync::{Mutex, MutexGuard};

    static LOCK: Mutex<()> = Mutex::new(());

    /// Held by every test that drains the global bus.
    pub fn serial() -> MutexGuard<'static, ()> {
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Removes and returns the queued commands `pred` accepts; the others stay queued in order.
    pub fn take_matching(pred: impl Fn(&AppCommand) -> bool) -> Vec<AppCommand> {
        let all = global().drain(usize::MAX);
        let (taken, kept): (Vec<_>, Vec<_>) = all.into_iter().partition(|c| pred(c));
        kept.into_iter().for_each(|c| global().push(c));
        taken
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::model::ToastLevel;

    fn toast(text: &str) -> AppCommand {
        AppCommand::Toast { kind: ToastLevel::Info, text: text.to_string() }
    }

    fn text_of(cmd: &AppCommand) -> &str {
        match cmd {
            AppCommand::Toast { text, .. } => text,
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn commands_come_out_in_the_order_they_went_in() {
        let bus = CommandBus::new();
        for i in 0..20 {
            bus.push(toast(&i.to_string()));
        }
        let drained = bus.drain(100);
        let texts: Vec<_> = drained.iter().map(text_of).collect();
        assert_eq!(texts, (0..20).map(|i| i.to_string()).collect::<Vec<_>>());
        assert!(bus.is_empty());
    }

    #[test]
    fn a_drain_budget_leaves_the_rest_queued_in_order() {
        let bus = CommandBus::new();
        for i in 0..5 {
            bus.push(toast(&i.to_string()));
        }
        assert_eq!(bus.drain(2).len(), 2);
        assert_eq!(bus.len(), 3);
        let rest = bus.drain(10);
        assert_eq!(text_of(&rest[0]), "2");
        assert_eq!(rest.len(), 3);
    }

    #[test]
    fn senders_on_many_threads_deliver_everything_and_keep_per_thread_order() {
        let bus = CommandBus::new();
        let handles: Vec<_> = (0..4)
            .map(|t| {
                let sender = bus.sender();
                std::thread::spawn(move || {
                    for i in 0..50 {
                        sender.send(toast(&format!("{t}:{i}")));
                    }
                })
            })
            .collect();
        handles.into_iter().for_each(|h| h.join().unwrap());
        let all = bus.drain(1000);
        assert_eq!(all.len(), 200);
        for t in 0..4 {
            let mine: Vec<usize> = all
                .iter()
                .map(text_of)
                .filter_map(|s| s.strip_prefix(&format!("{t}:")).map(|n| n.parse().unwrap()))
                .collect();
            assert_eq!(mine, (0..50).collect::<Vec<_>>(), "thread {t}");
        }
    }

    #[test]
    fn the_global_handle_reaches_the_global_bus() {
        let _serial = testing::serial();
        let marker = "bus-test-marker-7f3a";
        send(toast(marker));
        let seen = testing::take_matching(|c| matches!(c, AppCommand::Toast { text, .. } if text == marker));
        assert_eq!(seen.len(), 1);
    }
}
