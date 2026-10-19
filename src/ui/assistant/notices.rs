//! Notices other parts of the application add to the open chat, such as the end of an agent job.

use std::sync::Mutex;

use once_cell::sync::Lazy;

use super::entry::Level;

/// Notices kept while no panel is reading them.
const CAPACITY: usize = 20;

static NOTICES: Lazy<Mutex<Vec<(Level, String)>>> = Lazy::new(|| Mutex::new(Vec::new()));

fn lock() -> std::sync::MutexGuard<'static, Vec<(Level, String)>> {
    NOTICES.lock().unwrap_or_else(|e| e.into_inner())
}

/// Queues a notice for the open chat.
pub fn push(level: Level, text: impl Into<String>) {
    let mut queue = lock();
    queue.push((level, text.into()));
    if queue.len() > CAPACITY {
        queue.remove(0);
    }
    drop(queue);
    crate::command::request_repaint();
}

/// The notices queued since the last call, oldest first.
pub fn take() -> Vec<(Level, String)> {
    std::mem::take(&mut *lock())
}

/// Holds the queue still while a test uses it.
#[cfg(test)]
pub(crate) fn serial() -> std::sync::MutexGuard<'static, ()> {
    static SERIAL: Mutex<()> = Mutex::new(());
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notices_come_out_once_in_order_and_the_queue_is_bounded() {
        let _g = serial();
        take();
        push(Level::Info, "a");
        push(Level::Error, "b");
        assert_eq!(take(), vec![(Level::Info, "a".to_string()), (Level::Error, "b".to_string())]);
        assert!(take().is_empty());
        for i in 0..CAPACITY + 5 {
            push(Level::Warn, format!("n{i}"));
        }
        let got = take();
        assert_eq!(got.len(), CAPACITY);
        assert_eq!(got[0].1, "n5");
    }
}
