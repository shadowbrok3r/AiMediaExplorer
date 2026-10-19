//! Telling the person that an agent job ended: a toast and a notice in the open chat.

use crate::command::{AppCommand, ToastLevel};

/// Where the end of an agent job is reported.
pub trait Notifier: Send + Sync {
    fn notify(&self, level: ToastLevel, text: &str);
}

/// A toast and a notice in the assistant.
pub struct AppNotifier;

impl Notifier for AppNotifier {
    fn notify(&self, level: ToastLevel, text: &str) {
        crate::command::send(AppCommand::Toast { kind: level, text: text.to_string() });
        let chat_level = match level {
            ToastLevel::Error => crate::ui::assistant::entry::Level::Error,
            ToastLevel::Warning => crate::ui::assistant::entry::Level::Warn,
            _ => crate::ui::assistant::entry::Level::Info,
        };
        crate::ui::assistant::notices::push(chat_level, text);
    }
}

/// Keeps what it was told, for tests.
#[cfg(test)]
#[derive(Default)]
pub struct RecordingNotifier {
    pub sent: std::sync::Mutex<Vec<(ToastLevel, String)>>,
}

#[cfg(test)]
impl RecordingNotifier {
    pub fn messages(&self) -> Vec<(ToastLevel, String)> {
        self.sent.lock().unwrap().clone()
    }
}

#[cfg(test)]
impl Notifier for RecordingNotifier {
    fn notify(&self, level: ToastLevel, text: &str) {
        self.sent.lock().unwrap().push((level, text.to_string()));
    }
}
