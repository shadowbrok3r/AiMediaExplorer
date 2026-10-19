//! Chat sessions with a backend and typed entries, and the bounded local transcript cache the assistant opens from.

use surrealdb::types::{Datetime, RecordId};
use surrealdb::{Connection, Surreal};

use super::assistant_chat::{AssistantMessage, AssistantSession};
use crate::{ASSISTANT_MESSAGES, ASSISTANT_SESSIONS, DB, database::{db_activity, db_set_error}};

/// Entry kinds a message row can carry.
pub const KINDS: [&str; 8] = ["user", "assistant", "reasoning", "tool_call", "tool_result", "approval", "plan", "notice"];

/// Rows kept per session in the local transcript cache.
pub const CACHE_ROWS: usize = 300;

const SCHEMA_DDL: &str = r#"
    DEFINE FIELD IF NOT EXISTS backend ON assistant_sessions TYPE option<string> PERMISSIONS FULL;
    DEFINE FIELD IF NOT EXISTS agent ON assistant_sessions TYPE option<string> PERMISSIONS FULL;
    DEFINE FIELD IF NOT EXISTS gateway_id ON assistant_sessions TYPE option<string> PERMISSIONS FULL;
    DEFINE FIELD IF NOT EXISTS kind ON assistant_messages TYPE option<string> PERMISSIONS FULL;
    DEFINE FIELD IF NOT EXISTS turn ON assistant_messages TYPE option<int> PERMISSIONS FULL;
    DEFINE FIELD IF NOT EXISTS meta ON assistant_messages TYPE option<string> PERMISSIONS FULL;
"#;

static SCHEMA_READY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Adds the backend, agent, gateway, kind, turn and meta fields; safe to run again.
pub async fn ensure_schema_on<C: Connection>(db: &Surreal<C>) -> anyhow::Result<()> {
    db.query(SCHEMA_DDL).await?.check()?;
    Ok(())
}

/// [`ensure_schema_on`] for the application database, once per process.
pub async fn ensure_schema() -> anyhow::Result<()> {
    if SCHEMA_READY.load(std::sync::atomic::Ordering::SeqCst) {
        return Ok(());
    }
    ensure_schema_on(&DB).await?;
    SCHEMA_READY.store(true, std::sync::atomic::Ordering::SeqCst);
    Ok(())
}

/// A message row to add.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NewRow {
    pub role: String,
    pub kind: String,
    pub content: String,
    pub turn: Option<i64>,
    pub meta: Option<String>,
    pub attachments: Option<Vec<String>>,
}

/// Creates a session under the key `key` (a new id when `None`); the key is also the gateway's session id.
pub async fn create_session_on<C: Connection>(db: &Surreal<C>, key: Option<&str>, title: &str, backend: &str, agent: Option<&str>, gateway_id: Option<&str>) -> anyhow::Result<RecordId> {
    let key = key.map(str::to_string).unwrap_or_else(|| surrealdb_types::Uuid::new_v4().to_string());
    let row = AssistantSession {
        id: RecordId::new(ASSISTANT_SESSIONS, key),
        title: title.to_string(),
        created: Some(chrono::Utc::now().into()),
        updated: Some(chrono::Utc::now().into()),
        backend: Some(backend.to_string()),
        agent: agent.map(str::to_string),
        gateway_id: gateway_id.map(str::to_string),
    };
    let created: Option<AssistantSession> = db.create(ASSISTANT_SESSIONS).content(row).await?.take();
    created.map(|r| r.id).ok_or_else(|| anyhow::anyhow!("the session was not created"))
}

pub async fn create_session_with(key: Option<&str>, title: &str, backend: &str, agent: Option<&str>, gateway_id: Option<&str>) -> anyhow::Result<RecordId> {
    let _ga = db_activity("Create chat session");
    ensure_schema().await?;
    create_session_on(&DB, key, title, backend, agent, gateway_id).await.map_err(|e| {
        db_set_error(format!("create session failed: {e}"));
        e
    })
}

/// Every session, newest activity first.
pub async fn list_session_rows_on<C: Connection>(db: &Surreal<C>) -> anyhow::Result<Vec<AssistantSession>> {
    let rows: Vec<AssistantSession> = db.query("SELECT * FROM assistant_sessions ORDER BY updated DESC, created DESC").await?.take(0)?;
    Ok(rows)
}

pub async fn list_session_rows() -> anyhow::Result<Vec<AssistantSession>> {
    let _ga = db_activity("List chat sessions");
    ensure_schema().await?;
    list_session_rows_on(&DB).await
}

pub async fn rename_session_on<C: Connection>(db: &Surreal<C>, id: &RecordId, title: &str) -> anyhow::Result<()> {
    db.query("UPDATE $id SET title = $title, updated = time::now()").bind(("id", id.clone())).bind(("title", title.to_string())).await?.check()?;
    Ok(())
}

pub async fn rename_session(id: &RecordId, title: &str) -> anyhow::Result<()> {
    rename_session_on(&DB, id, title).await
}

/// Adds rows in order; each gets a timestamp a microsecond after the one before so ordering by time keeps their order.
pub async fn append_rows_on<C: Connection>(db: &Surreal<C>, session: &RecordId, rows: Vec<NewRow>) -> anyhow::Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    let base = chrono::Utc::now();
    let records: Vec<AssistantMessage> = rows
        .into_iter()
        .enumerate()
        .map(|(i, r)| AssistantMessage {
            id: RecordId::new(ASSISTANT_MESSAGES, surrealdb_types::Uuid::new_v4().to_string()),
            session_ref: session.clone(),
            role: r.role,
            content: r.content,
            attachments: r.attachments,
            attachments_refs: None,
            created: Some((base + chrono::Duration::microseconds(i as i64)).into()),
            kind: Some(r.kind),
            turn: r.turn,
            meta: r.meta,
        })
        .collect();
    db.query("INSERT INTO assistant_messages $rows").bind(("rows", records)).await?.check()?;
    db.query("UPDATE $id SET updated = time::now()").bind(("id", session.clone())).await?.check()?;
    Ok(())
}

/// Adds rows to the cache of `session` and trims it to [`CACHE_ROWS`].
pub async fn append_rows(session: &RecordId, rows: Vec<NewRow>) -> anyhow::Result<()> {
    let _ga = db_activity("Append chat rows");
    ensure_schema().await?;
    append_rows_on(&DB, session, rows).await.map_err(|e| {
        db_set_error(format!("insert messages failed: {e}"));
        e
    })?;
    trim_on(&DB, session, CACHE_ROWS).await.map(|_| ())
}

/// The newest `limit` rows of a session, oldest first.
pub async fn load_recent_on<C: Connection>(db: &Surreal<C>, session: &RecordId, limit: usize) -> anyhow::Result<Vec<AssistantMessage>> {
    let mut rows: Vec<AssistantMessage> = db
        .query("SELECT * FROM assistant_messages WITH INDEX idx_assistant_session WHERE session_ref = $sid ORDER BY created DESC LIMIT $limit")
        .bind(("sid", session.clone()))
        .bind(("limit", limit as i64))
        .await?
        .take(0)?;
    rows.reverse();
    Ok(rows)
}

pub async fn load_recent(session: &RecordId, limit: usize) -> anyhow::Result<Vec<AssistantMessage>> {
    let _ga = db_activity("Load chat messages");
    ensure_schema().await?;
    load_recent_on(&DB, session, limit).await
}

/// Deletes everything but the newest `keep` rows; returns how many went.
pub async fn trim_on<C: Connection>(db: &Surreal<C>, session: &RecordId, keep: usize) -> anyhow::Result<usize> {
    let cutoff: Option<Datetime> = db
        .query("SELECT VALUE created FROM assistant_messages WITH INDEX idx_assistant_session WHERE session_ref = $sid ORDER BY created DESC LIMIT 1 START $keep")
        .bind(("sid", session.clone()))
        .bind(("keep", keep as i64))
        .await?
        .take(0)?;
    let Some(cutoff) = cutoff else { return Ok(0) };
    let gone: Vec<AssistantMessage> = db
        .query("DELETE assistant_messages WHERE session_ref = $sid AND created <= $cutoff RETURN BEFORE")
        .bind(("sid", session.clone()))
        .bind(("cutoff", cutoff))
        .await?
        .take(0)?;
    Ok(gone.len())
}

/// Removes every row of a session but keeps the session.
pub async fn clear_messages_on<C: Connection>(db: &Surreal<C>, session: &RecordId) -> anyhow::Result<()> {
    db.query("DELETE assistant_messages WITH INDEX idx_assistant_session WHERE session_ref = $sid").bind(("sid", session.clone())).await?.check()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use surrealdb::engine::local::{Db, Mem};

    async fn fresh() -> Surreal<Db> {
        let db = Surreal::new::<Mem>(()).await.expect("memory database");
        db.use_ns("test").use_db("test").await.expect("namespace");
        crate::database::define_schema(&db).await.expect("base schema");
        ensure_schema_on(&db).await.expect("assistant schema");
        db
    }

    fn row(kind: &str, text: &str, turn: i64) -> NewRow {
        NewRow { role: if kind == "user" { "user".into() } else { "assistant".into() }, kind: kind.into(), content: text.into(), turn: Some(turn), ..Default::default() }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn the_schema_is_additive_and_safe_to_run_twice() {
        let db = fresh().await;
        ensure_schema_on(&db).await.unwrap();
        db.query("CREATE assistant_sessions:legacy SET title = 'Old chat'").await.unwrap().check().unwrap();
        let rows = list_session_rows_on(&db).await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!((rows[0].title.as_str(), rows[0].backend.as_deref()), ("Old chat", None), "a legacy row reads back with no backend");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_session_keeps_its_backend_agent_and_gateway() {
        let db = fresh().await;
        let id = create_session_on(&db, Some("11111111-2222-3333-4444-555555555555"), "Chat 1", "zeroclaw", Some("media"), Some("claw.example")).await.unwrap();
        assert!(format!("{id:?}").contains("11111111-2222-3333-4444-555555555555"));
        let rows = list_session_rows_on(&db).await.unwrap();
        assert_eq!((rows[0].backend.as_deref(), rows[0].agent.as_deref(), rows[0].gateway_id.as_deref()), (Some("zeroclaw"), Some("media"), Some("claw.example")));
        rename_session_on(&db, &id, "Sunsets").await.unwrap();
        assert_eq!(list_session_rows_on(&db).await.unwrap()[0].title, "Sunsets");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn rows_come_back_in_order_with_their_kind_turn_and_meta() {
        let db = fresh().await;
        let id = create_session_on(&db, None, "t", "zeroclaw", None, None).await.unwrap();
        let mut call = row("tool_call", "library_stats", 0);
        call.meta = Some(r#"{"id":"c1"}"#.into());
        append_rows_on(&db, &id, vec![row("user", "stats?", 0), call, row("assistant", "12 items", 0)]).await.unwrap();
        append_rows_on(&db, &id, vec![row("user", "again", 1)]).await.unwrap();
        let rows = load_recent_on(&db, &id, 50).await.unwrap();
        assert_eq!(rows.iter().map(|r| r.content.as_str()).collect::<Vec<_>>(), ["stats?", "library_stats", "12 items", "again"]);
        assert_eq!(rows[1].kind.as_deref(), Some("tool_call"));
        assert_eq!(rows[1].meta.as_deref(), Some(r#"{"id":"c1"}"#));
        assert_eq!((rows[3].turn, rows[3].role.as_str()), (Some(1), "user"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn the_cache_keeps_only_the_newest_rows() {
        let db = fresh().await;
        let id = create_session_on(&db, None, "t", "zeroclaw", None, None).await.unwrap();
        let rows: Vec<NewRow> = (0..40).map(|i| row("assistant", &format!("m{i}"), i)).collect();
        append_rows_on(&db, &id, rows).await.unwrap();
        let removed = trim_on(&db, &id, 25).await.unwrap();
        assert_eq!(removed, 15);
        let left = load_recent_on(&db, &id, 100).await.unwrap();
        assert_eq!(left.len(), 25);
        assert_eq!((left[0].content.as_str(), left[24].content.as_str()), ("m15", "m39"));
        assert_eq!(trim_on(&db, &id, 25).await.unwrap(), 0);
        let recent = load_recent_on(&db, &id, 3).await.unwrap();
        assert_eq!(recent.iter().map(|r| r.content.as_str()).collect::<Vec<_>>(), ["m37", "m38", "m39"]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn clearing_a_session_leaves_the_session_and_other_sessions_alone() {
        let db = fresh().await;
        let a = create_session_on(&db, None, "a", "local", None, None).await.unwrap();
        let b = create_session_on(&db, None, "b", "local", None, None).await.unwrap();
        append_rows_on(&db, &a, vec![row("user", "x", 0)]).await.unwrap();
        append_rows_on(&db, &b, vec![row("user", "y", 0)]).await.unwrap();
        clear_messages_on(&db, &a).await.unwrap();
        assert!(load_recent_on(&db, &a, 10).await.unwrap().is_empty());
        assert_eq!(load_recent_on(&db, &b, 10).await.unwrap().len(), 1);
        assert_eq!(list_session_rows_on(&db).await.unwrap().len(), 2);
    }

    #[test]
    fn every_entry_kind_the_ui_stores_is_listed() {
        for k in ["user", "assistant", "reasoning", "tool_call", "tool_result", "approval", "plan", "notice"] {
            assert!(KINDS.contains(&k), "{k}");
        }
    }
}
