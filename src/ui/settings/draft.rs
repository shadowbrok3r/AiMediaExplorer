//! The copy of the settings the page edits, with dirty tracking and apply as one patch of the changed fields.

use std::sync::Arc;

use crate::database::settings::{UiSettings, changed_fields, copy_fields};
use crate::settings::SettingsStore;

/// Name recorded in the store's patch log for an Apply.
pub const WHO: &str = "settings:apply";

/// What Apply changed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Applied {
    pub fields: Vec<&'static str>,
}

/// Work the engines need after some settings changed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Effect {
    /// The vision model folder changed.
    VisionFolder,
    /// The CLIP model or how its vectors are made changed; the engine reloads on next use.
    ClipEngine,
    /// The reranker repository or its options changed.
    Reranker,
    /// The automatic description and CLIP switches changed.
    AutoFlags,
}

/// The effects of changing `fields`.
pub fn effects_for(fields: &[&str]) -> Vec<Effect> {
    let any = |names: &[&str]| fields.iter().any(|f| names.contains(f));
    let mut out = Vec::new();
    if any(&["joycaption_model_dir"]) {
        out.push(Effect::VisionFolder);
    }
    if any(&["clip_model", "clip_augment_with_text"]) {
        out.push(Effect::ClipEngine);
    }
    if any(&["reranker_model", "jina_max_length", "jina_doc_type", "jina_query_type", "jina_enable_multimodal"]) {
        out.push(Effect::Reranker);
    }
    if any(&["auto_indexing", "auto_clip_embeddings"]) {
        out.push(Effect::AutoFlags);
    }
    out
}

/// The settings being edited: what they were when the edit began and what they are now.
#[derive(Clone, Debug)]
pub struct Draft {
    baseline: UiSettings,
    pub value: UiSettings,
}

impl Draft {
    pub fn new(from: UiSettings) -> Self {
        Self { baseline: from.clone(), value: from }
    }

    /// A draft of the live settings.
    pub fn of(store: &SettingsStore) -> Self {
        Self::new(store.snapshot())
    }

    /// Fields the user changed and did not apply.
    pub fn dirty_fields(&self) -> Vec<&'static str> {
        changed_fields(&self.baseline, &self.value)
    }

    pub fn is_dirty(&self) -> bool {
        !self.dirty_fields().is_empty()
    }

    /// Takes in changes made elsewhere to fields the user has not edited.
    pub fn sync(&mut self, live: &UiSettings) {
        let dirty = self.dirty_fields();
        let moved: Vec<&'static str> = changed_fields(&self.baseline, live).into_iter().filter(|f| !dirty.contains(f)).collect();
        if moved.is_empty() {
            return;
        }
        copy_fields(&mut self.baseline, live, &moved);
        copy_fields(&mut self.value, live, &moved);
    }

    /// Writes the changed fields into `store` as one patch; the draft is clean afterwards.
    pub fn apply(&mut self, store: &Arc<SettingsStore>) -> Applied {
        let fields = self.dirty_fields();
        if fields.is_empty() {
            return Applied::default();
        }
        let value = self.value.clone();
        store.update(WHO, |cur| copy_fields(cur, &value, &fields));
        copy_fields(&mut self.baseline, &self.value, &fields);
        Applied { fields }
    }

    /// Drops the edits and shows the live settings again.
    pub fn revert(&mut self, live: UiSettings) {
        *self = Self::new(live);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::{Backend, Loaded};
    use async_trait::async_trait;

    struct Nop;

    #[async_trait]
    impl Backend for Nop {
        async fn load(&self) -> anyhow::Result<Loaded> {
            Ok(Loaded { settings: UiSettings::default(), existed: true })
        }
        async fn merge(&self, _: &[&'static str], _: &UiSettings) -> anyhow::Result<()> {
            Ok(())
        }
    }

    async fn loaded_store() -> Arc<SettingsStore> {
        let store = SettingsStore::with_debounce(Arc::new(Nop), std::time::Duration::from_secs(3600));
        store.apply_loaded("test", Loaded { settings: UiSettings::default(), existed: true });
        store
    }

    #[tokio::test]
    async fn a_fresh_draft_is_clean_and_applying_it_writes_nothing() {
        let store = loaded_store().await;
        let generation = store.generation();
        let mut draft = Draft::of(&store);
        assert!(!draft.is_dirty());
        assert_eq!(draft.apply(&store), Applied::default());
        assert_eq!(store.generation(), generation, "opening and applying nothing changes nothing");
        assert!(store.dirty_fields().is_empty());
    }

    #[tokio::test]
    async fn editing_marks_exactly_the_edited_fields_dirty_and_revert_clears_them() {
        let store = loaded_store().await;
        let mut draft = Draft::of(&store);
        draft.value.filter_skip_icons = true;
        draft.value.clip_model = Some("clip-vit-b32".into());
        assert_eq!(draft.dirty_fields(), vec!["filter_skip_icons", "clip_model"]);
        draft.value.filter_skip_icons = false;
        assert_eq!(draft.dirty_fields(), vec!["clip_model"], "putting a value back is not a change");
        draft.revert(store.snapshot());
        assert!(!draft.is_dirty());
        assert_eq!(draft.value.clip_model, UiSettings::default().clip_model);
    }

    #[tokio::test]
    async fn apply_is_one_patch_of_only_the_changed_fields() {
        let store = loaded_store().await;
        let mut draft = Draft::of(&store);
        draft.value.auto_indexing = true;
        draft.value.ai_prompt_template = "Describe".into();
        let before = store.patches().len();
        let applied = draft.apply(&store);
        assert_eq!(applied.fields, vec!["auto_indexing", "ai_prompt_template"]);
        let patches = store.patches();
        assert_eq!(patches.len(), before + 1, "one patch per Apply");
        let patch = patches.last().unwrap();
        assert_eq!(patch.who, WHO);
        assert_eq!(patch.changes.iter().map(|c| c.field.as_str()).collect::<Vec<_>>(), vec!["auto_indexing", "ai_prompt_template"]);
        assert!(!draft.is_dirty());
        assert!(store.read(|s| s.auto_indexing));
    }

    #[tokio::test]
    async fn apply_never_overwrites_fields_another_part_changed_meanwhile() {
        let store = loaded_store().await;
        let mut draft = Draft::of(&store);
        draft.value.overwrite_descriptions = true;
        store.update("assistant:model", |s| s.openai_default_model = Some("gpt-x".into()));
        store.update("explorer:recent-path", |s| s.push_recent_path("/a".into()));
        draft.apply(&store);
        let live = store.snapshot();
        assert!(live.overwrite_descriptions);
        assert_eq!(live.openai_default_model.as_deref(), Some("gpt-x"), "the stale copy of the draft did not overwrite it");
        assert_eq!(live.recent_paths, vec!["/a".to_string()]);
    }

    #[tokio::test]
    async fn sync_takes_in_outside_changes_to_untouched_fields_only() {
        let store = loaded_store().await;
        let mut draft = Draft::of(&store);
        draft.value.filter_skip_icons = true;
        store.update("a", |s| s.auto_save_to_database = true);
        store.update("b", |s| s.filter_skip_icons = false);
        draft.sync(&store.snapshot());
        assert!(draft.value.auto_save_to_database, "an untouched field follows the store");
        assert!(draft.value.filter_skip_icons, "an edited field keeps the edit");
        assert_eq!(draft.dirty_fields(), vec!["filter_skip_icons"]);
    }

    #[tokio::test]
    async fn applying_to_a_store_that_is_not_loaded_keeps_the_change_in_memory() {
        let store = SettingsStore::with_debounce(Arc::new(Nop), std::time::Duration::from_secs(3600));
        let mut draft = Draft::of(&store);
        draft.value.auto_clip_embeddings = true;
        let applied = draft.apply(&store);
        assert_eq!(applied.fields, vec!["auto_clip_embeddings"]);
        assert!(store.read(|s| s.auto_clip_embeddings));
        assert!(store.dirty_fields().contains(&"auto_clip_embeddings"));
        assert!(!store.is_loaded());
    }

    #[test]
    fn changed_fields_ask_for_the_work_their_engines_need() {
        assert!(effects_for(&["filter_skip_icons", "excluded_dirs"]).is_empty());
        assert_eq!(effects_for(&["joycaption_model_dir"]), vec![Effect::VisionFolder]);
        assert_eq!(effects_for(&["clip_model"]), vec![Effect::ClipEngine]);
        assert_eq!(effects_for(&["jina_max_length", "reranker_model"]), vec![Effect::Reranker]);
        assert_eq!(effects_for(&["auto_indexing"]), vec![Effect::AutoFlags]);
        assert_eq!(
            effects_for(&["auto_clip_embeddings", "joycaption_model_dir", "clip_augment_with_text"]),
            vec![Effect::VisionFolder, Effect::ClipEngine, Effect::AutoFlags]
        );
    }
}
