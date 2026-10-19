//! The library guide in the settings, and its push into the gateway memory, which only ever happens when the person asks.

use std::sync::Arc;

use zeroclaw_client::{MemoryQuery, MemoryStore};

use super::events::{Pending, now_ms};
use super::runtime::{Shared, ZcRuntime};
use crate::ai::mcp::guide::{LibraryGuide, MEMORY_KEY};
use crate::settings::SettingsStore;

/// Memory category of the guide entry on the gateway.
pub const CATEGORY: &str = "core";

/// What the gateway took.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pushed {
    pub at_ms: u64,
    pub key: String,
    pub bytes: usize,
}

/// What the gateway holds under the guide key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Remote {
    pub bytes: usize,
    pub same_as_local: bool,
}

/// The guide as stored in the settings; the shipped text until it is edited.
pub fn guide() -> LibraryGuide {
    guide_in(crate::settings::store())
}

pub fn guide_in(store: &SettingsStore) -> LibraryGuide {
    store.read(|s| s.library_guide.clone()).unwrap_or_default()
}

/// Stores the edited guide text; what was pushed stays on record.
pub fn save_text(text: &str) {
    save_text_in(crate::settings::store(), text);
}

pub fn save_text_in(store: &Arc<SettingsStore>, text: &str) {
    let text = text.to_string();
    store.update("zc:guide", move |s| {
        let mut g = s.library_guide.clone().unwrap_or_default();
        g.set_text(&text);
        s.library_guide = Some(g);
    });
}

/// Back to the shipped text.
pub fn reset_text() {
    reset_text_in(crate::settings::store());
}

pub fn reset_text_in(store: &Arc<SettingsStore>) {
    store.update("zc:guide", |s| {
        let mut g = s.library_guide.clone().unwrap_or_default();
        g.reset();
        s.library_guide = Some(g);
    });
}

/// Records that `text` reached the gateway at `at_ms`.
pub fn record_push_in(store: &Arc<SettingsStore>, at_ms: u64, text: &str) {
    let text = text.to_string();
    store.update("zc:guide_push", move |s| {
        let mut g = s.library_guide.clone().unwrap_or_default();
        g.mark_pushed(at_ms, &text);
        s.library_guide = Some(g);
    });
}

fn agent_of(rt: &ZcRuntime) -> Option<String> {
    Some(rt.profile().agent.trim().to_string()).filter(|a| !a.is_empty())
}

async fn store_entry(sh: &Shared, agent: Option<String>, text: &str) -> Result<Pushed, String> {
    let rest = sh.rest()?;
    let mut entry = MemoryStore::new(MEMORY_KEY, text);
    entry.category = Some(CATEGORY.to_string());
    entry.agent = agent;
    rest.memory_store(&entry).await.map_err(|e| sh.redact(&e.to_string()))?;
    Ok(Pushed { at_ms: now_ms(), key: MEMORY_KEY.to_string(), bytes: text.len() })
}

/// Stores `text` in the gateway memory of the profile's agent, without recording the push.
pub fn push(rt: &ZcRuntime, text: String) -> Pending<Result<Pushed, String>> {
    let agent = agent_of(rt);
    rt.job(move |sh| async move { store_entry(&sh, agent, &text).await })
}

/// Pushes the guide of `store` and records the push in it once the gateway took the text.
pub fn push_guide(rt: &ZcRuntime, store: Arc<SettingsStore>) -> Pending<Result<Pushed, String>> {
    let agent = agent_of(rt);
    let text = guide_in(&store).effective().to_string();
    rt.job(move |sh| async move {
        let pushed = store_entry(&sh, agent, &text).await?;
        record_push_in(&store, pushed.at_ms, &text);
        Ok(pushed)
    })
}

/// Asks the gateway what it holds under the guide key; `None` when nothing is there.
pub fn remote(rt: &ZcRuntime, local: String) -> Pending<Result<Option<Remote>, String>> {
    let agent = agent_of(rt);
    rt.job(move |sh| async move {
        let rest = sh.rest()?;
        let found = rest.memory(&MemoryQuery { agent, category: Some(CATEGORY.to_string()), ..Default::default() }).await.map_err(|e| sh.redact(&e.to_string()))?;
        Ok(found.into_iter().find(|e| e.key == MEMORY_KEY).map(|e| Remote { bytes: e.content.len(), same_as_local: e.content == local }))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::UiSettings;
    use crate::settings::{Backend, Loaded};
    use crate::zc::runtime_tests::runtime_for;
    use zeroclaw_client::mock::MockGateway;

    struct Nop;

    #[async_trait::async_trait]
    impl Backend for Nop {
        async fn load(&self) -> anyhow::Result<Loaded> {
            Ok(Loaded { settings: UiSettings::default(), existed: true })
        }
        async fn merge(&self, _: &[&'static str], _: &UiSettings) -> anyhow::Result<()> {
            Ok(())
        }
    }

    fn store() -> Arc<SettingsStore> {
        let store = SettingsStore::with_debounce(Arc::new(Nop), std::time::Duration::from_secs(3600));
        store.apply_loaded("test", Loaded { settings: UiSettings::default(), existed: true });
        store
    }

    async fn until<T>(job: &mut Pending<T>) -> T {
        let end = std::time::Instant::now() + std::time::Duration::from_secs(8);
        loop {
            if let Some(v) = job.take() {
                return v;
            }
            assert!(std::time::Instant::now() < end, "the job never finished");
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_push_stores_the_guide_under_its_key_for_the_agent_and_nothing_is_sent_on_its_own() {
        let mock = MockGateway::start().await;
        let rt = runtime_for(&mock, "default", "memory-push");
        assert!(mock.requests().iter().all(|r| r.path != "/api/memory"), "no memory traffic before a push");
        let text = "# My guide\nTags are lower case.".to_string();
        let pushed = until(&mut push(&rt, text.clone())).await.unwrap();
        assert_eq!((pushed.key.as_str(), pushed.bytes), (MEMORY_KEY, text.len()));
        let stored = mock.memory();
        assert_eq!(stored.len(), 1);
        assert_eq!((stored[0]["key"].clone(), stored[0]["content"].clone(), stored[0]["category"].clone(), stored[0]["agent"].clone()), (MEMORY_KEY.into(), text.clone().into(), CATEGORY.into(), "default".into()));
        let again = until(&mut push(&rt, "second".to_string())).await.unwrap();
        assert!(again.at_ms >= pushed.at_ms);
        assert_eq!(mock.memory().len(), 1, "a second push replaces the entry");
        assert_eq!(mock.memory()[0]["content"], "second");
        let posts: Vec<_> = mock.requests().into_iter().filter(|r| r.path == "/api/memory" && r.method == "POST").collect();
        assert_eq!(posts.len(), 2);
        assert!(posts.iter().all(|r| r.authorized));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn pushing_the_guide_records_the_time_and_the_text_that_went() {
        let mock = MockGateway::start().await;
        let rt = runtime_for(&mock, "default", "memory-record");
        let store = store();
        assert!(guide_in(&store).changed_since_push());
        let pushed = until(&mut push_guide(&rt, store.clone())).await.unwrap();
        let g = guide_in(&store);
        assert_eq!(g.pushed_ms, Some(pushed.at_ms));
        assert!(!g.changed_since_push());
        assert_eq!(mock.memory()[0]["content"].as_str().unwrap(), crate::ai::mcp::guide::DEFAULT_GUIDE, "the shipped text goes until it is edited");
        save_text_in(&store, "Edited notes");
        assert!(guide_in(&store).changed_since_push(), "an edit shows as unpushed");
        assert_eq!(guide_in(&store).pushed_ms, Some(pushed.at_ms), "the record of the push stays");
        until(&mut push_guide(&rt, store.clone())).await.unwrap();
        assert_eq!(mock.memory()[0]["content"], "Edited notes");
        assert!(!guide_in(&store).changed_since_push());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_failed_push_records_nothing() {
        let mock = MockGateway::start().await;
        let rt = runtime_for(&mock, "default", "memory-failed");
        mock.override_rest("POST", "/api/memory", 500, serde_json::json!({"error": "memory store is read only"}), std::time::Duration::ZERO);
        let store = store();
        let err = until(&mut push_guide(&rt, store.clone())).await.unwrap_err();
        assert!(err.contains("read only"), "{err}");
        assert_eq!(guide_in(&store).pushed_ms, None);
        assert!(mock.memory().is_empty());
        let offline = ZcRuntime::new(tokio::runtime::Handle::current(), None);
        let err = until(&mut push(&offline, "text".into())).await.unwrap_err();
        assert!(err.contains("not set") || err.contains("token") || err.contains("address"), "{err}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn the_remote_copy_is_found_and_compared_with_the_local_text() {
        let mock = MockGateway::start().await;
        let rt = runtime_for(&mock, "default", "memory-remote");
        assert_eq!(until(&mut remote(&rt, "x".into())).await.unwrap(), None);
        until(&mut push(&rt, "hello".into())).await.unwrap();
        assert_eq!(until(&mut remote(&rt, "hello".into())).await.unwrap(), Some(Remote { bytes: 5, same_as_local: true }));
        assert_eq!(until(&mut remote(&rt, "other".into())).await.unwrap(), Some(Remote { bytes: 5, same_as_local: false }));
    }

    #[test]
    fn the_settings_keep_the_edited_text_and_reset_returns_the_shipped_one() {
        let store = store();
        assert!(guide_in(&store).is_default());
        save_text_in(&store, "My own notes");
        assert_eq!(guide_in(&store).effective(), "My own notes");
        reset_text_in(&store);
        assert!(guide_in(&store).is_default());
        assert_eq!(store.read(|s| s.library_guide.clone()).map(|g| g.text), Some(None), "the field stays, the text goes");
    }
}
