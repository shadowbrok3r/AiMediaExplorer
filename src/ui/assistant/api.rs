//! Mailboxes other parts of the application use to reach the assistant: attachments, messages to send and requests to show it.

use std::sync::Mutex;

use once_cell::sync::Lazy;

/// Paths the command dispatcher hands to the chat until it is shown.
pub static ATTACH_REQUESTS: Lazy<Mutex<Vec<Vec<String>>>> = Lazy::new(|| Mutex::new(Vec::new()));

static SUBMITS: Lazy<Mutex<Vec<String>>> = Lazy::new(|| Mutex::new(Vec::new()));
static OPEN: Lazy<Mutex<bool>> = Lazy::new(|| Mutex::new(false));

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Queues `text` to be sent as if the person had typed it.
pub fn submit(text: String) {
    lock(&SUBMITS).push(text);
    crate::command::request_repaint();
}

/// Messages queued with [`submit`].
pub fn take_submits() -> Vec<String> {
    std::mem::take(&mut *lock(&SUBMITS))
}

/// Asks for the assistant to be shown.
pub fn request_open() {
    *lock(&OPEN) = true;
    crate::command::request_repaint();
}

/// True once per [`request_open`].
pub fn take_open() -> bool {
    std::mem::take(&mut *lock(&OPEN))
}

/// The paths of every queued batch, without repeats.
pub fn drain_batches(queue: &Mutex<Vec<Vec<String>>>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for batch in lock(queue).drain(..) {
        for p in batch {
            if !out.contains(&p) {
                out.push(p);
            }
        }
    }
    out
}

/// Paths queued with `attach_to_assistant`, without repeats.
pub fn take_attachments() -> Vec<String> {
    drain_batches(&ATTACH_REQUESTS)
}

/// Holds the mailboxes still while a test uses them.
#[cfg(test)]
pub(crate) fn serial() -> std::sync::MutexGuard<'static, ()> {
    static SERIAL: Mutex<()> = Mutex::new(());
    lock(&SERIAL)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn submitted_messages_come_out_once_in_order() {
        let _g = serial();
        take_submits();
        submit("a".into());
        submit("b".into());
        assert_eq!(take_submits(), ["a", "b"]);
        assert!(take_submits().is_empty());
    }

    #[test]
    fn an_open_request_is_consumed_once() {
        let _g = serial();
        take_open();
        assert!(!take_open());
        request_open();
        assert!(take_open());
        assert!(!take_open());
    }

    #[test]
    fn attachments_are_merged_without_repeats() {
        let queue = Mutex::new(vec![vec!["/a".to_string(), "/b".to_string()], vec!["/b".to_string(), "/c".to_string()]]);
        assert_eq!(drain_batches(&queue), ["/a", "/b", "/c"]);
        assert!(drain_batches(&queue).is_empty());
    }
}
