//! Back and forward history of one browser tab: the scopes it showed with where the user was in each.

use crate::command::Scope;

/// Most entries one tab remembers; the oldest are dropped first.
pub const HISTORY_LIMIT: usize = 100;

/// A scope the tab showed and the view the user left it with.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub scope: Scope,
    /// View position of the first row shown.
    pub scroll_row: usize,
    /// Path of the selection anchor.
    pub anchor: Option<String>,
}

impl Entry {
    pub fn new(scope: Scope) -> Self {
        Self { scope, scroll_row: 0, anchor: None }
    }
}

/// A line of entries with a cursor on the one being shown.
#[derive(Clone, Debug, Default)]
pub struct History {
    entries: Vec<Entry>,
    cursor: usize,
}

impl History {
    pub fn new(scope: Scope) -> Self {
        Self { entries: vec![Entry::new(scope)], cursor: 0 }
    }

    pub fn current(&self) -> Option<&Entry> {
        self.entries.get(self.cursor)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Index of the entry being shown.
    pub fn position(&self) -> usize {
        self.cursor
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    pub fn can_back(&self) -> bool {
        self.cursor > 0
    }

    pub fn can_forward(&self) -> bool {
        self.cursor + 1 < self.entries.len()
    }

    /// Records where the user is in the current entry before leaving it.
    pub fn save_view(&mut self, scroll_row: usize, anchor: Option<String>) {
        if let Some(entry) = self.entries.get_mut(self.cursor) {
            entry.scroll_row = scroll_row;
            entry.anchor = anchor;
        }
    }

    /// Shows `scope` as a new entry after the current one and drops everything forward of it; false when it already is the current scope.
    pub fn push(&mut self, scope: Scope) -> bool {
        if self.current().is_some_and(|e| e.scope == scope) {
            return false;
        }
        self.entries.truncate(self.cursor + 1);
        self.entries.push(Entry::new(scope));
        if self.entries.len() > HISTORY_LIMIT {
            let excess = self.entries.len() - HISTORY_LIMIT;
            self.entries.drain(..excess);
        }
        self.cursor = self.entries.len() - 1;
        true
    }

    /// Replaces the scope of the current entry, keeping the entries around it.
    pub fn replace(&mut self, scope: Scope) {
        match self.entries.get_mut(self.cursor) {
            Some(entry) => *entry = Entry::new(scope),
            None => self.entries.push(Entry::new(scope)),
        }
        self.cursor = self.cursor.min(self.entries.len() - 1);
    }

    /// Moves to the previous entry and returns it.
    pub fn back(&mut self) -> Option<&Entry> {
        if !self.can_back() {
            return None;
        }
        self.cursor -= 1;
        self.current()
    }

    /// Moves to the next entry and returns it.
    pub fn forward(&mut self) -> Option<&Entry> {
        if !self.can_forward() {
            return None;
        }
        self.cursor += 1;
        self.current()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn folder(p: &str) -> Scope {
        Scope::Folder(p.to_string())
    }

    fn scopes(h: &History) -> Vec<String> {
        h.entries().iter().map(|e| e.scope.format()).collect()
    }

    #[test]
    fn a_new_history_shows_its_first_scope_and_goes_nowhere() {
        let mut h = History::new(folder("/a"));
        assert_eq!(h.current().unwrap().scope, folder("/a"));
        assert!(!h.can_back() && !h.can_forward());
        assert!(h.back().is_none() && h.forward().is_none());
        assert_eq!((h.len(), h.position()), (1, 0));
    }

    #[test]
    fn pushing_moves_forward_and_back_returns() {
        let mut h = History::new(folder("/a"));
        assert!(h.push(folder("/b")));
        assert!(h.push(Scope::Library));
        assert_eq!(h.position(), 2);
        assert_eq!(h.back().unwrap().scope, folder("/b"));
        assert_eq!(h.back().unwrap().scope, folder("/a"));
        assert!(h.back().is_none());
        assert_eq!(h.forward().unwrap().scope, folder("/b"));
        assert_eq!(h.forward().unwrap().scope, Scope::Library);
        assert!(h.forward().is_none());
    }

    #[test]
    fn pushing_the_scope_being_shown_changes_nothing() {
        let mut h = History::new(folder("/a"));
        assert!(!h.push(folder("/a")));
        assert_eq!(h.len(), 1);
        h.push(folder("/b"));
        assert!(!h.push(folder("/b")));
        assert_eq!(h.len(), 2);
    }

    #[test]
    fn a_push_after_going_back_drops_the_forward_entries() {
        let mut h = History::new(folder("/a"));
        h.push(folder("/b"));
        h.push(folder("/c"));
        h.back();
        h.back();
        assert!(h.can_forward());
        h.push(Scope::Tag("x".into()));
        assert_eq!(scopes(&h), vec!["fs:/a", "tag:x"]);
        assert!(!h.can_forward());
        assert_eq!(h.position(), 1);
    }

    #[test]
    fn the_view_is_saved_on_the_entry_being_left_and_comes_back_with_it() {
        let mut h = History::new(folder("/a"));
        h.save_view(42, Some("/a/x.jpg".into()));
        h.push(folder("/b"));
        h.save_view(7, None);
        let back = h.back().unwrap();
        assert_eq!((back.scroll_row, back.anchor.as_deref()), (42, Some("/a/x.jpg")));
        let forward = h.forward().unwrap();
        assert_eq!((forward.scroll_row, forward.anchor.clone()), (7, None));
    }

    #[test]
    fn a_fresh_entry_starts_at_the_top_with_no_anchor() {
        let mut h = History::new(folder("/a"));
        h.save_view(9, Some("p".into()));
        h.push(folder("/b"));
        let e = h.current().unwrap();
        assert_eq!((e.scroll_row, e.anchor.as_deref()), (0, None));
    }

    #[test]
    fn the_oldest_entries_are_dropped_past_the_limit() {
        let mut h = History::new(folder("/0"));
        for i in 1..=HISTORY_LIMIT + 20 {
            h.push(folder(&format!("/{i}")));
        }
        assert_eq!(h.len(), HISTORY_LIMIT);
        assert_eq!(h.position(), HISTORY_LIMIT - 1);
        assert_eq!(h.current().unwrap().scope, folder(&format!("/{}", HISTORY_LIMIT + 20)));
        assert_eq!(h.entries()[0].scope, folder("/21"));
    }

    #[test]
    fn replacing_swaps_the_current_entry_only() {
        let mut h = History::new(folder("/a"));
        h.push(folder("/b"));
        h.push(folder("/c"));
        h.back();
        h.replace(Scope::Library);
        assert_eq!(scopes(&h), vec!["fs:/a", "lib:all", "fs:/c"]);
        assert_eq!(h.position(), 1);
    }

    #[test]
    fn an_empty_default_history_accepts_its_first_push() {
        let mut h = History::default();
        assert!(h.is_empty() && h.current().is_none());
        assert!(h.push(folder("/a")));
        assert_eq!(h.len(), 1);
        h.save_view(3, None);
        assert_eq!(h.current().unwrap().scroll_row, 3);
    }
}
