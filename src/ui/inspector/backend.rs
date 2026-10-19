//! Database work of the inspector behind a trait, so the UI flow runs against the real store or an in-memory one.

use async_trait::async_trait;
use surrealdb::types::RecordId;
use surrealdb::{Connection, Surreal};

use crate::database::collections;
use crate::database::metadata_edit::{self, BatchReport, LibraryFacts, MetaPatch, RowSeed, SaveOutcome, TagChange};

/// What the library knows about one file.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Library {
    pub facts: LibraryFacts,
    /// Names of the collections the file is in.
    pub collections: Vec<String>,
    /// Why the library could not be read, such as a database that is not connected.
    pub error: Option<String>,
}

/// A collection to pick in the batch editor.
#[derive(Clone, Debug, PartialEq)]
pub struct CollectionInfo {
    pub id: RecordId,
    pub name: String,
    pub count: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CollectionOp {
    /// Adds the paths, creating the collection when it does not exist.
    Add { name: String, paths: Vec<String> },
    Remove { name: String, paths: Vec<String> },
}

#[async_trait]
pub trait Backend: Send + Sync + 'static {
    async fn save(&self, seed: RowSeed, patch: MetaPatch) -> Result<SaveOutcome, String>;
    async fn batch_tags(&self, seeds: Vec<RowSeed>, change: TagChange) -> Result<BatchReport, String>;
    async fn batch_category(&self, seeds: Vec<RowSeed>, category: Option<String>) -> Result<BatchReport, String>;
    async fn library(&self, path: String) -> Result<Library, String>;
    async fn tags(&self) -> Result<Vec<String>, String>;
    async fn categories(&self) -> Result<Vec<String>, String>;
    async fn collections(&self) -> Result<Vec<CollectionInfo>, String>;
    /// Runs a collection change and returns the sentence for the toast.
    async fn collection_edit(&self, op: CollectionOp) -> Result<String, String>;
}

/// Where the database handle comes from.
enum Handle<C: Connection> {
    Own(Surreal<C>),
    /// A handle that is connected after the backend exists; read afresh on every call.
    Shared(fn() -> &'static Surreal<C>),
}

/// The backend over a database handle.
pub struct DbBackend<C: Connection> {
    handle: Handle<C>,
    /// Refuses work while the global store is not connected.
    require_connected: bool,
}

impl<C: Connection> DbBackend<C> {
    pub fn new(db: Surreal<C>) -> Self {
        Self { handle: Handle::Own(db), require_connected: false }
    }

    /// A backend over the handle `get` returns, which may be connected later.
    pub fn shared(get: fn() -> &'static Surreal<C>, require_connected: bool) -> Self {
        Self { handle: Handle::Shared(get), require_connected }
    }
}

fn global_db() -> &'static Surreal<surrealdb::engine::local::Db> {
    &crate::database::DB
}

impl DbBackend<surrealdb::engine::local::Db> {
    /// The backend over the application's database.
    pub fn global() -> Self {
        Self::shared(global_db, true)
    }
}

fn text(e: anyhow::Error) -> String {
    format!("{e:#}")
}

impl<C: Connection> DbBackend<C> {
    fn ready(&self) -> Result<&Surreal<C>, String> {
        if self.require_connected {
            if !crate::database::is_connected() {
                return Err("The database is not connected".to_string());
            }
            if !crate::database::is_ready() {
                return Err("The database is still starting".to_string());
            }
        }
        Ok(match &self.handle {
            Handle::Own(db) => db,
            Handle::Shared(get) => get(),
        })
    }
}

#[async_trait]
impl<C: Connection> Backend for DbBackend<C> {
    async fn save(&self, seed: RowSeed, patch: MetaPatch) -> Result<SaveOutcome, String> {
        metadata_edit::save_patch(self.ready()?, &seed, &patch).await.map_err(text)
    }

    async fn batch_tags(&self, seeds: Vec<RowSeed>, change: TagChange) -> Result<BatchReport, String> {
        metadata_edit::batch_tags(self.ready()?, &seeds, &change).await.map_err(text)
    }

    async fn batch_category(&self, seeds: Vec<RowSeed>, category: Option<String>) -> Result<BatchReport, String> {
        metadata_edit::batch_category(self.ready()?, &seeds, category.as_deref()).await.map_err(text)
    }

    async fn library(&self, path: String) -> Result<Library, String> {
        let db = self.ready()?;
        let facts = metadata_edit::library_facts(db, &path).await.map_err(text)?;
        let collections = collections::collections_of(db, &path).await.map_err(text)?.into_iter().map(|c| c.name).collect();
        Ok(Library { facts, collections, error: None })
    }

    async fn tags(&self) -> Result<Vec<String>, String> {
        metadata_edit::all_tags(self.ready()?).await.map_err(text)
    }

    async fn categories(&self) -> Result<Vec<String>, String> {
        metadata_edit::all_categories(self.ready()?).await.map_err(text)
    }

    async fn collections(&self) -> Result<Vec<CollectionInfo>, String> {
        let list = collections::list_with_counts(self.ready()?).await.map_err(text)?;
        Ok(list.into_iter().map(|c| CollectionInfo { id: c.collection.id, name: c.collection.name, count: c.count }).collect())
    }

    async fn collection_edit(&self, op: CollectionOp) -> Result<String, String> {
        let db = self.ready()?;
        match op {
            CollectionOp::Add { name, paths } => {
                let (collection, created) = collections::get_or_create(db, &name).await.map_err(text)?;
                let report = collections::add_paths(db, &collection.id, &paths).await.map_err(text)?;
                let new = if created { " (new collection)" } else { "" };
                Ok(if report.already > 0 {
                    format!("Added {} to \"{}\"{new}; {} were already in it", report.added, collection.name, report.already)
                } else {
                    format!("Added {} to \"{}\"{new}", report.added, collection.name)
                })
            }
            CollectionOp::Remove { name, paths } => {
                let collection = collections::get_by_name(db, &name).await.map_err(text)?.ok_or_else(|| format!("There is no collection \"{name}\""))?;
                let removed = collections::remove_paths(db, &collection.id, &paths).await.map_err(text)?;
                Ok(format!("Removed {removed} from \"{}\"; {} were not in it", collection.name, paths.len().saturating_sub(removed)))
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod testing {
    use std::sync::Mutex;

    use super::*;

    /// Backend whose save can be held back or made to fail.
    pub struct ScriptedBackend<B: Backend> {
        pub inner: B,
        pub fail_saves: Mutex<Option<String>>,
        pub saves: Mutex<Vec<(RowSeed, MetaPatch)>>,
    }

    impl<B: Backend> ScriptedBackend<B> {
        pub fn new(inner: B) -> Self {
            Self { inner, fail_saves: Mutex::new(None), saves: Mutex::new(Vec::new()) }
        }
    }

    #[async_trait]
    impl<B: Backend> Backend for ScriptedBackend<B> {
        async fn save(&self, seed: RowSeed, patch: MetaPatch) -> Result<SaveOutcome, String> {
            self.saves.lock().unwrap().push((seed.clone(), patch.clone()));
            if let Some(e) = self.fail_saves.lock().unwrap().clone() {
                return Err(e);
            }
            self.inner.save(seed, patch).await
        }
        async fn batch_tags(&self, seeds: Vec<RowSeed>, change: TagChange) -> Result<BatchReport, String> {
            self.inner.batch_tags(seeds, change).await
        }
        async fn batch_category(&self, seeds: Vec<RowSeed>, category: Option<String>) -> Result<BatchReport, String> {
            self.inner.batch_category(seeds, category).await
        }
        async fn library(&self, path: String) -> Result<Library, String> {
            self.inner.library(path).await
        }
        async fn tags(&self) -> Result<Vec<String>, String> {
            self.inner.tags().await
        }
        async fn categories(&self) -> Result<Vec<String>, String> {
            self.inner.categories().await
        }
        async fn collections(&self) -> Result<Vec<CollectionInfo>, String> {
            self.inner.collections().await
        }
        async fn collection_edit(&self, op: CollectionOp) -> Result<String, String> {
            self.inner.collection_edit(op).await
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use surrealdb::engine::local::{Db, Mem};

    use super::*;
    use crate::Thumbnail;

    static LATE: LazyLock<Surreal<Db>> = LazyLock::new(Surreal::init);

    fn late() -> &'static Surreal<Db> {
        &LATE
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_shared_handle_connected_after_the_backend_exists_reads_the_connected_session() {
        let backend = DbBackend::shared(late, false);
        assert!(backend.library("/x/a.jpg".into()).await.is_err(), "nothing is connected yet");
        LATE.connect::<Mem>(()).await.unwrap();
        LATE.use_ns(crate::database::NS).use_db(crate::database::DB_NAME).await.unwrap();
        crate::database::define_schema(&LATE).await.unwrap();
        crate::database::collections::ensure_schema(&LATE).await.unwrap();
        let lib = backend.library("/x/a.jpg".into()).await.expect("the library is readable");
        assert!(!lib.facts.indexed && lib.collections.is_empty());
        let mut row = Thumbnail::new("a.jpg");
        row.path = "/x/a.jpg".into();
        row.file_type = "jpg".into();
        let outcome = backend.save(RowSeed::from_row(&row), MetaPatch { caption: Some(Some("late".into())), ..Default::default() }).await.expect("the save reaches the database");
        assert!(outcome.created);
        assert!(backend.library("/x/a.jpg".into()).await.unwrap().facts.indexed);
    }
}
