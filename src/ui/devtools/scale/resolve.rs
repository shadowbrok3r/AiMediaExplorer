//! Looks up the biggest tag, category and collection of the library for the `@top` scope names of a run.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::bench::Resolved;

/// How long the lookups wait for the database to connect.
const CONNECT_WAIT: Duration = Duration::from_secs(60);

// None while the lookup runs, Some(None) when the library has nothing of the kind.
type Slot = Arc<Mutex<Option<Option<String>>>>;

fn slot() -> Slot {
    Arc::new(Mutex::new(None))
}

fn read(slot: &Slot) -> Resolved {
    match slot.lock().unwrap_or_else(|e| e.into_inner()).clone() {
        None => Resolved::Pending,
        Some(None) => Resolved::Missing,
        Some(Some(name)) => Resolved::Ready(name),
    }
}

fn fill(slot: &Slot, name: Option<String>) {
    *slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(name);
}

/// The name with the highest count; the first in name order among equals.
pub fn top_of(counts: impl IntoIterator<Item = (String, i64)>) -> Option<String> {
    counts.into_iter().filter(|(name, n)| *n > 0 && !name.trim().is_empty()).min_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0))).map(|(name, _)| name)
}

/// The biggest names of the library, looked up in the background.
pub struct TopNames {
    tag: Slot,
    category: Slot,
    collection: Slot,
}

impl TopNames {
    /// Starts the three lookups on the application runtime.
    pub fn start() -> Self {
        let this = Self { tag: slot(), category: slot(), collection: slot() };
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            fill(&this.tag, None);
            fill(&this.category, None);
            fill(&this.collection, None);
            return this;
        };
        let (tag, category, collection) = (this.tag.clone(), this.category.clone(), this.collection.clone());
        handle.spawn(async move {
            let cancel = AtomicBool::new(false);
            if !crate::ui::file_table::scope::load::wait_for_connection(&cancel, CONNECT_WAIT).await {
                fill(&tag, None);
                fill(&category, None);
                fill(&collection, None);
                return;
            }
            let started = std::time::Instant::now();
            fill(&tag, crate::database::Thumbnail::list_tag_counts().await.ok().and_then(top_of));
            super::note_timing("tag counts", started.elapsed());
            let started = std::time::Instant::now();
            fill(&category, crate::database::Thumbnail::list_category_counts().await.ok().and_then(top_of));
            super::note_timing("category counts", started.elapsed());
            let started = std::time::Instant::now();
            let members = crate::database::collections::list_with_counts(&crate::database::DB).await.ok();
            super::note_timing("collection counts", started.elapsed());
            fill(&collection, members.and_then(|all| top_of(all.into_iter().map(|c| (c.collection.name.clone(), c.count as i64 + 1)))));
        });
        this
    }

    /// Names already known, for tests.
    pub fn known(tag: Option<&str>, category: Option<&str>, collection: Option<&str>) -> Self {
        let this = Self { tag: slot(), category: slot(), collection: slot() };
        fill(&this.tag, tag.map(str::to_string));
        fill(&this.category, category.map(str::to_string));
        fill(&this.collection, collection.map(str::to_string));
        this
    }

    /// `scope` with `@top` replaced by the biggest name of its kind.
    pub fn resolve(&self, scope: &str) -> Resolved {
        let (prefix, slot) = match scope.split_once(':') {
            Some(("tag", _)) => ("tag", &self.tag),
            Some(("cat", _)) => ("cat", &self.category),
            Some(("col", _)) => ("col", &self.collection),
            _ => return Resolved::Missing,
        };
        match read(slot) {
            Resolved::Ready(name) => Resolved::Ready(format!("{prefix}:{name}")),
            other => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_top_name_has_the_most_rows_and_ties_go_to_the_first_name() {
        let counts = vec![("b".to_string(), 5), ("a".to_string(), 5), ("c".to_string(), 9), ("".to_string(), 50), ("d".to_string(), 0)];
        assert_eq!(top_of(counts), Some("c".to_string()));
        assert_eq!(top_of(vec![("b".to_string(), 5), ("a".to_string(), 5)]), Some("a".to_string()));
        assert_eq!(top_of(Vec::new()), None);
        assert_eq!(top_of(vec![("x".to_string(), 0)]), None);
    }

    #[test]
    fn a_known_name_fills_the_scope_prefix() {
        let top = TopNames::known(Some("sunset"), Some("landscape"), Some("Holidays"));
        assert_eq!(top.resolve("tag:@top"), Resolved::Ready("tag:sunset".into()));
        assert_eq!(top.resolve("cat:@top"), Resolved::Ready("cat:landscape".into()));
        assert_eq!(top.resolve("col:@top"), Resolved::Ready("col:Holidays".into()));
    }

    #[test]
    fn a_kind_without_names_is_missing_and_an_unknown_scope_too() {
        let top = TopNames::known(None, Some("x"), None);
        assert_eq!(top.resolve("tag:@top"), Resolved::Missing);
        assert_eq!(top.resolve("lib:@top"), Resolved::Missing);
        assert_eq!(top.resolve("nonsense"), Resolved::Missing);
    }

    #[test]
    fn a_running_lookup_is_pending() {
        let top = TopNames { tag: slot(), category: slot(), collection: slot() };
        assert_eq!(top.resolve("tag:@top"), Resolved::Pending);
        fill(&top.tag, Some("late".into()));
        assert_eq!(top.resolve("tag:@top"), Resolved::Ready("tag:late".into()));
    }
}
