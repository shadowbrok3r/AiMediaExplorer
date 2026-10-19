//! Native dialogs that run on a tokio task and report through queues the UI drains each frame.

use std::path::PathBuf;
use std::sync::Mutex;

use egui_toast::ToastKind;
use once_cell::sync::Lazy;

/// Picked path to apply to UI state owned by a specific surface.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DialogResult {
    /// A folder or file chosen for the field named `slot`.
    Picked { slot: &'static str, path: PathBuf },
    AssistantAttach(PathBuf),
}

static RESULTS: Lazy<Mutex<Vec<DialogResult>>> = Lazy::new(|| Mutex::new(Vec::new()));
static TOASTS: Lazy<Mutex<Vec<(ToastKind, String)>>> = Lazy::new(|| Mutex::new(Vec::new()));

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Queues a toast from any thread.
pub fn toast(kind: ToastKind, text: impl Into<String>) {
    lock(&TOASTS).push((kind, text.into()));
}

/// Drains queued toasts.
pub fn take_toasts() -> Vec<(ToastKind, String)> {
    std::mem::take(&mut *lock(&TOASTS))
}

/// Queues a dialog result and wakes the UI.
pub fn deliver(ctx: &egui::Context, result: DialogResult) {
    lock(&RESULTS).push(result);
    ctx.request_repaint();
}

/// Removes and returns the first queued result that `pick` accepts.
pub fn take<T>(mut pick: impl FnMut(&DialogResult) -> Option<T>) -> Option<T> {
    let mut results = lock(&RESULTS);
    let index = results.iter().position(|r| pick(r).is_some())?;
    let result = results.remove(index);
    pick(&result)
}

/// Opens a folder picker for the field `slot`; the chosen folder is queued as [`DialogResult::Picked`].
pub fn pick_folder_into(ctx: &egui::Context, title: &str, slot: &'static str) {
    let deliver_ctx = ctx.clone();
    pick_folder(ctx, title, move |path| deliver(&deliver_ctx, DialogResult::Picked { slot, path }));
}

/// Opens a file picker for the field `slot`; the chosen file is queued as [`DialogResult::Picked`].
pub fn pick_file_into(ctx: &egui::Context, title: &str, slot: &'static str) {
    let deliver_ctx = ctx.clone();
    pick_file(ctx, title, move |path| deliver(&deliver_ctx, DialogResult::Picked { slot, path }));
}

/// Takes the path queued for the field `slot`.
pub fn take_picked(slot: &'static str) -> Option<PathBuf> {
    take(|r| match r {
        DialogResult::Picked { slot: s, path } if *s == slot => Some(path.clone()),
        _ => None,
    })
}

/// Opens a folder picker without blocking the UI; `on_done` runs with the chosen folder.
pub fn pick_folder(ctx: &egui::Context, title: &str, on_done: impl FnOnce(PathBuf) + Send + 'static) {
    let (ctx, title) = (ctx.clone(), title.to_string());
    tokio::spawn(async move {
        if let Some(handle) = rfd::AsyncFileDialog::new().set_title(title).pick_folder().await {
            on_done(handle.path().to_path_buf());
            ctx.request_repaint();
        }
    });
}

/// Opens a file picker without blocking the UI; `on_done` runs with the chosen file.
pub fn pick_file(ctx: &egui::Context, title: &str, on_done: impl FnOnce(PathBuf) + Send + 'static) {
    let (ctx, title) = (ctx.clone(), title.to_string());
    tokio::spawn(async move {
        if let Some(handle) = rfd::AsyncFileDialog::new().set_title(title).pick_file().await {
            on_done(handle.path().to_path_buf());
            ctx.request_repaint();
        }
    });
}

/// Opens a save dialog without blocking the UI; `on_done` runs with the chosen file.
pub fn save_file(
    ctx: &egui::Context,
    title: &str,
    file_name: &str,
    filter: Option<(&str, &[&str])>,
    on_done: impl FnOnce(PathBuf) + Send + 'static,
) {
    let (ctx, title, file_name) = (ctx.clone(), title.to_string(), file_name.to_string());
    let filter = filter.map(|(name, exts)| (name.to_string(), exts.iter().map(|e| e.to_string()).collect::<Vec<_>>()));
    tokio::spawn(async move {
        let mut dialog = rfd::AsyncFileDialog::new().set_title(title).set_file_name(file_name);
        if let Some((name, exts)) = &filter {
            dialog = dialog.add_filter(name, exts);
        }
        if let Some(handle) = dialog.save_file().await {
            on_done(handle.path().to_path_buf());
            ctx.request_repaint();
        }
    });
}

/// Asks a yes/no question without blocking the UI; `on_yes` runs only on confirmation.
pub fn confirm(
    ctx: &egui::Context,
    title: &str,
    description: &str,
    level: rfd::MessageLevel,
    on_yes: impl FnOnce() + Send + 'static,
) {
    let (ctx, title, description) = (ctx.clone(), title.to_string(), description.to_string());
    tokio::spawn(async move {
        let answer = rfd::AsyncMessageDialog::new()
            .set_title(title)
            .set_description(description)
            .set_level(level)
            .set_buttons(rfd::MessageButtons::YesNo)
            .show()
            .await;
        if answer == rfd::MessageDialogResult::Yes {
            on_yes();
            ctx.request_repaint();
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_returns_the_first_matching_result_and_leaves_the_rest() {
        lock(&RESULTS).clear();
        let ctx = egui::Context::default();
        deliver(&ctx, DialogResult::Picked { slot: "a", path: "/a".into() });
        deliver(&ctx, DialogResult::AssistantAttach("/b".into()));
        let picked = take(|r| match r {
            DialogResult::AssistantAttach(p) => Some(p.clone()),
            _ => None,
        });
        assert_eq!(picked, Some(PathBuf::from("/b")));
        assert!(take(|r| matches!(r, DialogResult::AssistantAttach(_)).then_some(())).is_none());
        assert!(matches!(take(|r| Some(r.clone())), Some(DialogResult::Picked { slot: "a", .. })));
    }

    #[test]
    fn a_picked_path_is_taken_by_its_slot_only() {
        lock(&RESULTS).clear();
        let ctx = egui::Context::default();
        deliver(&ctx, DialogResult::Picked { slot: "one", path: "/one".into() });
        deliver(&ctx, DialogResult::Picked { slot: "two", path: "/two".into() });
        assert_eq!(take_picked("two"), Some(PathBuf::from("/two")));
        assert_eq!(take_picked("two"), None);
        assert_eq!(take_picked("one"), Some(PathBuf::from("/one")));
    }

    #[test]
    fn toasts_drain_once() {
        lock(&TOASTS).clear();
        toast(ToastKind::Info, "hello");
        assert_eq!(take_toasts().len(), 1);
        assert!(take_toasts().is_empty());
    }
}
