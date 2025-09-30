#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSessionNew {
    pub title: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessageNew {
    pub session_ref: RecordId,
    pub role: String,
    pub content: String,
    pub attachments: Option<Vec<String>>,
    pub attachments_refs: Option<Vec<RecordId>>,
}
use serde::{Deserialize, Serialize};
use surrealdb::RecordId;

use crate::{DB, database::{db_activity, db_set_detail, db_set_error}};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSessionRow {
    pub id: RecordId,
    pub title: String,
    pub created: Option<surrealdb::sql::Datetime>,
    pub updated: Option<surrealdb::sql::Datetime>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessageRow {
    pub id: RecordId,
    pub session_ref: RecordId,
    pub role: String, // "user" | "assistant"
    pub content: String,
    pub attachments: Option<Vec<String>>,
    pub attachments_refs: Option<Vec<RecordId>>,
    pub created: Option<surrealdb::sql::Datetime>,
}

pub async fn create_session(title: &str) -> anyhow::Result<RecordId, anyhow::Error> {
    let _ga = db_activity("Create chat session");
    db_set_detail(format!("{title}"));
    #[derive(Deserialize)]
    struct IdOnly { id: RecordId }
    let new_row = ChatSessionNew { title: title.to_string() };
    let row: Option<IdOnly> = DB
        .create("assistant_sessions")
        .content(new_row)
        .await
        .map_err(|e| { db_set_error(format!("create session failed: {e}")); e })?
        .take();
    Ok(row.map(|r| r.id).unwrap())
}

pub async fn list_sessions() -> anyhow::Result<Vec<(RecordId, String)>, anyhow::Error> {
    let _ga = db_activity("List chat sessions");
    db_set_detail("Loading sessions");
    let rows: Vec<ChatSessionRow> = DB
        .query("SELECT * FROM assistant_sessions ORDER BY updated DESC, created DESC")
        .await?
        .take(0)?;
    Ok(rows.into_iter().map(|r| (r.id, r.title)).collect())
}

pub async fn load_messages(session_id: &RecordId) -> anyhow::Result<Vec<ChatMessageRow>, anyhow::Error> {
    let _ga = db_activity("Load chat messages");
    db_set_detail("Loading messages");
    let rows: Vec<ChatMessageRow> = DB
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
    let new_msg = ChatMessageNew {
        session_ref: session_id.clone(),
        role: role.to_string(),
        content: content.to_string(),
        attachments,
        attachments_refs,
    };
    let _: Option<ChatMessageRow> = DB
        .create("assistant_messages")
        .content(new_msg)
        .await
        .map_err(|e| { db_set_error(format!("insert message failed: {e}")); e })?
        .take();
    // Touch session updated timestamp
    let _: Option<ChatSessionRow> = DB
        .query("UPDATE $sid SET updated = time::now()")
        .bind(("sid", session_id.clone()))
        .await?
        .take(0)?;
    Ok(())
}

pub async fn delete_session(session_id: &RecordId) -> anyhow::Result<(), anyhow::Error> {
    let _ga = db_activity("Delete chat session");
    db_set_detail(format!("{session_id}"));
    // Delete messages for this session first (by index for efficiency)
    // Then delete the session row itself
    let _ = DB
        .query("DELETE assistant_messages WITH INDEX idx_assistant_session WHERE session_ref = $sid")
        .bind(("sid", session_id.clone()))
        .await?;
    let _ = DB.delete::<Option<ChatSessionRow>>(session_id.clone()).await?;
    Ok(())
}
