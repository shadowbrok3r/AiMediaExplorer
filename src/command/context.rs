//! Ambient facts commands and tools read about the focused view.

use std::sync::Mutex;

static ACTIVE_COLLECTION: Mutex<Option<String>> = Mutex::new(None);

/// Name of the collection the focused tab is showing.
pub fn active_collection() -> Option<String> {
    ACTIVE_COLLECTION.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

pub fn set_active_collection(name: Option<String>) {
    *ACTIVE_COLLECTION.lock().unwrap_or_else(|e| e.into_inner()) = name;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_active_collection_is_shared_and_clearable() {
        set_active_collection(Some("Holidays".into()));
        assert_eq!(active_collection().as_deref(), Some("Holidays"));
        set_active_collection(None);
        assert_eq!(active_collection(), None);
    }
}
