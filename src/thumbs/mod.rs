//! Thumbnail pipeline: encoded blobs in a store, served to the UI as textures through a byte-budgeted cache, a bounded decode pool and batched reads.

pub mod cache;
pub mod core;
pub mod decode;
pub mod key;
pub mod recent;
pub mod service;
pub mod stats;
pub mod store;
pub mod surreal_store;
#[cfg(test)]
pub(crate) mod perf;
#[cfg(test)]
mod service_tests;

use std::sync::Arc;

pub use key::{CacheKey, ThumbKey, Tier};
pub use service::{ScopeId, ThumbService, ThumbView, alloc_scope, default_sink};

/// The app-wide thumbnail service.
pub fn service() -> Arc<ThumbService> {
    service::global()
}

/// Whether the larger preview can be decoded straight from the image file behind `path`.
pub fn can_preview(path: &str) -> bool {
    core::default_generatable(path) && path.rsplit_once('.').is_some_and(|(_, ext)| crate::utilities::types::is_image(&ext.to_ascii_lowercase()))
}

/// Moves a row's legacy text thumbnail into the service, leaving the row without pixel data.
pub fn adopt_legacy_one(service: &ThumbService, row: &mut crate::Thumbnail) {
    if let Some(thumb) = row.thumbnail_b64.take() {
        if !row.path.is_empty() && row.file_type != "<DIR>" {
            service.ingest_legacy(&row.path, thumb);
        }
    }
}

/// Moves the legacy text thumbnails of `rows` into the service in one batch.
pub fn adopt_legacy(service: &ThumbService, rows: &mut [crate::Thumbnail]) -> usize {
    let mut taken = Vec::new();
    for row in rows.iter_mut() {
        if let Some(thumb) = row.thumbnail_b64.take() {
            if !row.path.is_empty() && row.file_type != "<DIR>" {
                taken.push((row.path.clone(), thumb));
            }
        }
    }
    let n = taken.len();
    service.ingest_legacy_batch(taken);
    n
}

/// Read-only view of which paths have a stored thumbnail, for status counts.
#[derive(Clone)]
pub struct ThumbPresence {
    service: Arc<ThumbService>,
}

impl ThumbPresence {
    pub fn new(service: Arc<ThumbService>) -> Self {
        Self { service }
    }

    /// Number of known stored thumbnails; changes as thumbnails arrive.
    pub fn len(&self) -> usize {
        self.service.presence_len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contains_key(&self, path: &str) -> bool {
        self.service.has_thumb(path)
    }
}
