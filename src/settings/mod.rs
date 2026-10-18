//! Settings: the authoritative store, start-directory fallbacks and the load-failure banner.

pub mod banner;
pub mod dirs;
pub mod mcp;
mod store;
#[cfg(test)]
pub(crate) mod testing;

pub use store::*;

/// Applies `f` to the live settings through the process-wide store.
pub fn update(who: &str, f: impl FnOnce(&mut crate::UiSettings)) -> bool {
    store().update(who, f)
}

/// Copy of the live settings.
pub fn snapshot() -> crate::UiSettings {
    store().snapshot()
}

/// Writes pending changes before the process exits, waiting at most `timeout`.
pub fn flush_blocking(timeout: std::time::Duration) {
    let store = std::sync::Arc::clone(store());
    if !store.is_loaded() || store.dirty_fields().is_empty() {
        return;
    }
    let Ok(handle) = tokio::runtime::Handle::try_current() else { return };
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    handle.spawn(async move {
        if let Err(e) = store.flush().await {
            log::error!("[settings] final save failed: {e:#}");
        }
        let _ = done_tx.send(());
    });
    let _ = done_rx.recv_timeout(timeout);
}
