use crate::{ASSISTANT_MESSAGES, DB, database::{db_activity, db_set_detail, db_set_error}};
use serde::{Deserialize, Serialize};
use surrealdb::types::{RecordId, Datetime, SurrealValue};

#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct AssistantSession {
    pub id: RecordId,
    pub title: String,
    pub created: Option<Datetime>,
    pub updated: Option<Datetime>,
    /// `zeroclaw`, `direct` or `local`; missing on sessions from before backends existed.
    #[serde(default)]
    pub backend: Option<String>,
    #[serde(default)]
    pub agent: Option<String>,
    #[serde(default)]
    pub gateway_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct AssistantMessage {
    pub id: RecordId,
    pub session_ref: RecordId,
    pub role: String, // "user" | "assistant"
    pub content: String,
    pub attachments: Option<Vec<String>>,
    pub attachments_refs: Option<Vec<RecordId>>,
    pub created: Option<Datetime>,
    /// Entry kind; missing on rows from before entries had kinds.
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub turn: Option<i64>,
    /// JSON text with the entry's extra data (tool name, arguments, ids).
    #[serde(default)]
    pub meta: Option<String>,
}

pub async fn load_messages(session_id: &RecordId) -> anyhow::Result<Vec<AssistantMessage>, anyhow::Error> {
    let _ga = db_activity("Load chat messages");
    db_set_detail("Loading messages");
    let rows: Vec<AssistantMessage> = DB
        .query("SELECT * FROM assistant_messages WITH INDEX idx_assistant_session WHERE session_ref = $sid ORDER BY created ASC")
        .bind(("sid", session_id.clone()))
        .await?
        .take(0)?;
    Ok(rows)
}

pub async fn append_message(session_id: &RecordId, role: &str, content: &str, attachments: Option<Vec<String>>) -> anyhow::Result<(), anyhow::Error> {
    let _ga = db_activity("Append chat message");
    db_set_detail(format!("{role}"));
    // Best-effort convert file paths to thumbnail record ids for reloadable previews
    let attachments_refs: Option<Vec<RecordId>> = if let Some(paths) = &attachments {
        let mut out: Vec<RecordId> = Vec::new();
        for p in paths.iter() {
            if let Ok(Some(thumb)) = crate::Thumbnail::get_thumbnail_by_path(p).await {
                out.push(thumb.id);
            }
        }
        if out.is_empty() { None } else { Some(out) }
    } else { None };

    let new_msg = AssistantMessage {
        session_ref: session_id.clone(),
        role: role.to_string(),
        content: content.to_string(),
        attachments,
        attachments_refs,
        id: RecordId::new(ASSISTANT_MESSAGES, surrealdb_types::Uuid::new_v4().to_string()),
        created: Some(chrono::Utc::now().into()),
        kind: None,
        turn: None,
        meta: None,
    };
    let _: Option<AssistantMessage> = DB
        .create(ASSISTANT_MESSAGES)
        .content(new_msg)
        .await
        .map_err(|e| { db_set_error(format!("insert message failed: {e}")); e })?
        .take();
    Ok(())
}

pub async fn delete_session(session_id: &RecordId) -> anyhow::Result<(), anyhow::Error> {
    let _ga = db_activity("Delete chat session");
    db_set_detail(format!("{session_id:?}"));
    // Delete messages for this session first (by index for efficiency)
    // Then delete the session row itself
    let _ = DB
        .query("DELETE assistant_messages WITH INDEX idx_assistant_session WHERE session_ref = $sid")
        .bind(("sid", session_id.clone()))
        .await?;
    let _ = DB.delete::<Option<AssistantSession>>(session_id.clone()).await?;
    Ok(())
}
