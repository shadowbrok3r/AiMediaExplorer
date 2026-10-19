//! Counts of the database round trips of the browsing paths, by call site, read by the Scale window.

use std::sync::atomic::{AtomicU64, Ordering};

/// A place that queries the database on behalf of the browser.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Site {
    /// A page of rows of a scope.
    ScopePage,
    /// The CLIP presence check of a batch of rows.
    ScopeEmbedded,
    /// Rows looked up by path or collection membership.
    ScopeLookup,
    /// Cached scans, counts and vectors read for a scope.
    ScopeMeta,
    SidebarLibrary,
    SidebarCollections,
    SidebarTags,
    SidebarCategories,
    SidebarScans,
    /// Thumbnail blobs read for tiles.
    ThumbRead,
    /// Which paths have a stored thumbnail.
    ThumbPresence,
    ThumbWrite,
    /// Collection counts probed for drawn tiles.
    TileFacts,
}

impl Site {
    pub const ALL: [Site; 13] = [
        Self::ScopePage,
        Self::ScopeEmbedded,
        Self::ScopeLookup,
        Self::ScopeMeta,
        Self::SidebarLibrary,
        Self::SidebarCollections,
        Self::SidebarTags,
        Self::SidebarCategories,
        Self::SidebarScans,
        Self::ThumbRead,
        Self::ThumbPresence,
        Self::ThumbWrite,
        Self::TileFacts,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::ScopePage => "scope page",
            Self::ScopeEmbedded => "scope clip check",
            Self::ScopeLookup => "scope lookup",
            Self::ScopeMeta => "scope meta",
            Self::SidebarLibrary => "sidebar library",
            Self::SidebarCollections => "sidebar collections",
            Self::SidebarTags => "sidebar tags",
            Self::SidebarCategories => "sidebar categories",
            Self::SidebarScans => "sidebar scans",
            Self::ThumbRead => "thumb read",
            Self::ThumbPresence => "thumb presence",
            Self::ThumbWrite => "thumb write",
            Self::TileFacts => "tile facts",
        }
    }
}

static COUNTS: [AtomicU64; Site::ALL.len()] = [const { AtomicU64::new(0) }; Site::ALL.len()];

/// Records one query made from `site`.
pub fn note(site: Site) {
    COUNTS[site as usize].fetch_add(1, Ordering::Relaxed);
}

/// Queries made so far, per site in [`Site::ALL`] order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub by_site: [u64; Site::ALL.len()],
}

impl Counts {
    pub fn total(&self) -> u64 {
        self.by_site.iter().sum()
    }

    pub fn of(&self, site: Site) -> u64 {
        self.by_site[site as usize]
    }

    /// Queries made since `earlier`.
    pub fn since(&self, earlier: &Counts) -> Counts {
        let mut by_site = [0; Site::ALL.len()];
        for (i, slot) in by_site.iter_mut().enumerate() {
            *slot = self.by_site[i].saturating_sub(earlier.by_site[i]);
        }
        Counts { by_site }
    }
}

pub fn snapshot() -> Counts {
    let mut by_site = [0; Site::ALL.len()];
    for (slot, count) in by_site.iter_mut().zip(&COUNTS) {
        *slot = count.load(Ordering::Relaxed);
    }
    Counts { by_site }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notes_add_up_per_site_and_in_total() {
        let before = snapshot();
        note(Site::ScopePage);
        note(Site::ScopePage);
        note(Site::TileFacts);
        let delta = snapshot().since(&before);
        assert!(delta.of(Site::ScopePage) >= 2);
        assert!(delta.of(Site::TileFacts) >= 1);
        assert!(delta.total() >= 3);
    }

    #[test]
    fn a_site_is_indexed_by_its_position_in_the_list() {
        for (i, site) in Site::ALL.iter().enumerate() {
            assert_eq!(*site as usize, i);
            assert!(!site.label().is_empty());
        }
    }

    #[test]
    fn since_never_goes_negative() {
        let a = Counts { by_site: [5; Site::ALL.len()] };
        let b = Counts { by_site: [9; Site::ALL.len()] };
        assert_eq!(a.since(&b).total(), 0);
        assert_eq!(b.since(&a).total(), 4 * Site::ALL.len() as u64);
    }
}
