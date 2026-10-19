//! The chats the gateway holds for the profile's agent: listing, renaming and deleting them over REST.

use std::sync::Arc;

use zeroclaw_client::{Error, SessionRow, group_sessions};

use super::events::Pending;
use super::runtime::{Shared, ZcRuntime};

/// Rows per page of the browser.
pub const PAGE_SIZE: usize = 12;

/// Why a gateway call failed, as far as the browser tells the person what to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    /// The gateway refused the token.
    Unauthorized,
    /// The gateway could not be reached.
    Offline,
    /// The gateway answered with an error, or the profile is incomplete.
    Failed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionsError {
    pub fault: Fault,
    pub message: String,
}

impl SessionsError {
    fn from_client(sh: &Shared, e: Error) -> Self {
        let fault = if e.is_auth() {
            Fault::Unauthorized
        } else if matches!(e, Error::Connect(_) | Error::Timeout(_)) {
            Fault::Offline
        } else {
            Fault::Failed
        };
        Self { fault, message: sh.redact(&e.to_string()) }
    }

    fn setup(message: String) -> Self {
        Self { fault: Fault::Failed, message }
    }
}

/// The agent alias the browser lists chats for.
fn agent_of(sh: &Shared) -> String {
    sh.profile().agent.trim().to_string()
}

/// The chats of the profile's agent, newest activity first, without the empty ones a bare connect leaves.
pub fn list(rt: &ZcRuntime) -> Pending<Result<Vec<SessionRow>, SessionsError>> {
    rt.job(|sh| async move {
        let rest = sh.rest().map_err(SessionsError::setup)?;
        let infos = rest.sessions().await.map_err(|e| SessionsError::from_client(&sh, e))?;
        Ok(group_sessions(Some(&agent_of(&sh)), &infos))
    })
}

/// Names every store of one chat; a store that is already gone is skipped, and it is an error when none was there.
pub fn rename(rt: &ZcRuntime, keys: Vec<String>, name: String) -> Pending<Result<(), SessionsError>> {
    rt.job(move |sh| async move {
        let name = name.trim().to_string();
        if name.is_empty() {
            return Err(SessionsError::setup("Enter a name".to_string()));
        }
        let rest = sh.rest().map_err(SessionsError::setup)?;
        let mut renamed = 0;
        for key in &keys {
            match rest.rename_session(key, &name).await {
                Ok(()) => renamed += 1,
                Err(Error::Http { status: 404, .. }) => {}
                Err(e) => return Err(SessionsError::from_client(&sh, e)),
            }
        }
        if renamed == 0 {
            return Err(SessionsError { fault: Fault::Failed, message: "The gateway has no such chat any more (Session not found)".to_string() });
        }
        Ok(())
    })
}

/// Deletes every store of one chat; true when at least one existed.
pub fn delete(rt: &ZcRuntime, keys: Vec<String>) -> Pending<Result<bool, SessionsError>> {
    rt.job(move |sh: Arc<Shared>| async move {
        let rest = sh.rest().map_err(SessionsError::setup)?;
        let mut any = false;
        for key in &keys {
            any |= rest.delete_session(key).await.map_err(|e| SessionsError::from_client(&sh, e))?;
        }
        Ok(any)
    })
}

/// The rows of page `page` (counted from zero) and the number of pages; an out-of-range page reads as the last one.
pub fn page<T>(rows: &[T], page: usize, size: usize) -> (&[T], usize, usize) {
    let size = size.max(1);
    let pages = rows.len().div_ceil(size).max(1);
    let page = page.min(pages - 1);
    let start = page * size;
    (&rows[start.min(rows.len())..(start + size).min(rows.len())], page, pages)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zc::runtime_tests::runtime_for;
    use serde_json::json;
    use zeroclaw_client::mock::{MockGateway, MockOptions};

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

    fn unique(prefix: &str) -> String {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        format!("{prefix}-{}", N.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }

    async fn gateway() -> (MockGateway, Arc<ZcRuntime>) {
        let mock = MockGateway::start_with(MockOptions { agents: vec!["media".into(), "voice".into()], ..Default::default() }).await;
        mock.add_session("old", "media", 4, "2026-09-09T12:00:00+00:00");
        mock.add_session("empty", "media", 0, "2026-09-11T11:59:00+00:00");
        mock.add_session("other", "voice", 9, "2026-09-11T11:00:00+00:00");
        mock.add_session("new", "media", 7, "2026-09-11T10:00:00+00:00");
        mock.add_session_row(json!({"session_id": "new", "session_key": "new", "agent_alias": "media", "name": "Tag sunsets", "message_count": 3, "last_activity": "2026-09-11T10:30:00+00:00"}));
        let rt = runtime_for(&mock, "media", &unique("sessions"));
        (mock, rt)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn the_list_has_the_agents_chats_with_messages_newest_first() {
        let (_mock, rt) = gateway().await;
        let rows = until(&mut list(&rt)).await.unwrap();
        assert_eq!(rows.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(), ["new", "old"], "no empty shells, no other agent");
        assert_eq!(rows[0].name.as_deref(), Some("Tag sunsets"));
        assert_eq!(rows[0].message_count, 10, "the twin stores of one chat are one row");
        assert_eq!(rows[0].keys, ["gw_new", "new"]);
        assert_eq!(rows[1].message_count, 4);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn renaming_changes_the_name_on_the_gateway() {
        let (mock, rt) = gateway().await;
        until(&mut rename(&rt, vec!["gw_old".into()], "  Holiday tags  ".into())).await.unwrap();
        let named = mock.session_rows().into_iter().find(|s| s["session_id"] == "old").unwrap();
        assert_eq!(named["name"], "Holiday tags");
        let rows = until(&mut list(&rt)).await.unwrap();
        assert_eq!(rows.iter().find(|r| r.id == "old").unwrap().name.as_deref(), Some("Holiday tags"));
        let err = until(&mut rename(&rt, vec!["gw_old".into()], "   ".into())).await.unwrap_err();
        assert_eq!(err.message, "Enter a name");
        let err = until(&mut rename(&rt, vec!["gw_ghost".into(), "ghost".into()], "x".into())).await.unwrap_err();
        assert_eq!(err.fault, Fault::Failed);
        assert!(err.message.contains("Session not found"), "{err:?}");
        until(&mut rename(&rt, vec!["gw_ghost".into(), "gw_new".into(), "new".into()], "Both stores".into())).await.unwrap();
        let both: Vec<String> = mock.session_rows().iter().filter(|s| s["session_id"] == "new").filter_map(|s| s["name"].as_str().map(str::to_string)).collect();
        assert_eq!(both, ["Both stores", "Both stores"], "every store of the chat is named, a missing one is skipped");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn deleting_removes_every_store_of_the_chat_and_only_that_chat() {
        let (mock, rt) = gateway().await;
        assert!(until(&mut delete(&rt, vec!["gw_new".into(), "new".into()])).await.unwrap(), "at least one store existed");
        let left: Vec<String> = mock.session_rows().iter().map(|s| s["session_id"].as_str().unwrap().to_string()).collect();
        assert!(!left.contains(&"new".to_string()), "{left:?}");
        assert!(left.contains(&"old".to_string()) && left.contains(&"other".to_string()));
        assert!(!until(&mut delete(&rt, vec!["new".into()])).await.unwrap(), "a chat that is already gone is not an error");
        assert!(!mock.requests().iter().any(|r| r.method == "DELETE" && r.path == "/api/sessions/old"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_refused_token_and_an_unreachable_gateway_are_told_apart() {
        let (mock, rt) = gateway().await;
        rt.save_token("wrong-token").unwrap();
        let err = until(&mut list(&rt)).await.unwrap_err();
        assert_eq!(err.fault, Fault::Unauthorized, "{err:?}");
        assert!(!err.message.contains("wrong-token"));
        drop(mock);
        rt.save_token("test-token").unwrap();
        let err = until(&mut list(&rt)).await.unwrap_err();
        assert_eq!(err.fault, Fault::Offline, "{err:?}");
        let blank = ZcRuntime::new(tokio::runtime::Handle::current(), None);
        let err = until(&mut list(&blank)).await.unwrap_err();
        assert_eq!(err.fault, Fault::Failed);
        assert!(!err.message.is_empty());
    }

    #[test]
    fn pages_cover_the_rows_and_clamp_out_of_range_requests() {
        let rows: Vec<u32> = (0..30).collect();
        let (first, p, pages) = page(&rows, 0, 12);
        assert_eq!((first.len(), p, pages), (12, 0, 3));
        let (last, p, _) = page(&rows, 2, 12);
        assert_eq!((last.len(), p, last[0]), (6, 2, 24));
        let (clamped, p, _) = page(&rows, 99, 12);
        assert_eq!((clamped, p), (last, 2));
        let (none, p, pages) = page::<u32>(&[], 3, 12);
        assert_eq!((none.len(), p, pages), (0, 0, 1));
        assert_eq!(page(&rows, 0, 0).2, 30, "a zero page size reads as one");
    }
}
