//! What the write tools, the guide resource and the job reports need beyond the library: proposal stores, edit jobs, the notifier and the guide.

use std::sync::Arc;

use super::clock::Clock;
use super::ctx::DbSource;
use super::edits::{self, EditJobs};
use super::guide::{GuideSource, SettingsGuide};
use super::notify::{AppNotifier, Notifier};
use super::taxonomy::TaxonomyStore;

pub struct AgentCtx {
    pub taxonomy: Arc<TaxonomyStore>,
    pub edits: Arc<EditJobs>,
    pub notifier: Arc<dyn Notifier>,
    pub guide: Arc<dyn GuideSource>,
}

impl AgentCtx {
    /// The application's stores, Image Edit backend, settings guide and toasts.
    pub fn production(db: DbSource, clock: Arc<dyn Clock>) -> Arc<Self> {
        let notifier: Arc<dyn Notifier> = Arc::new(AppNotifier);
        Arc::new(Self {
            taxonomy: Arc::new(TaxonomyStore::new(db, clock.clone())),
            edits: EditJobs::new(crate::ui::image_edit::shared_backend(), edits::default_scratch(), clock, notifier.clone()),
            notifier,
            guide: Arc::new(SettingsGuide),
        })
    }
}
