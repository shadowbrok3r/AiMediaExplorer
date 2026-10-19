//! The chat sessions: the list, switching, the transcript cache and titles. Database work runs in the background and comes back through a channel.

use crossbeam::channel::{Receiver, Sender};
use surrealdb::types::RecordId;

use crate::database::assistant_chat::{AssistantMessage, AssistantSession};
use crate::database::assistant_store::{self, CACHE_ROWS};
use crate::database::collections::key_of;

/// Title of a session before it has been named.
pub const NEW_TITLE: &str = "New chat";

/// One session as the list shows it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionInfo {
    pub id: RecordId,
    /// The record key; also the session id on the gateway.
    pub key: String,
    pub title: String,
    /// `zeroclaw`, `direct` or `local`; `None` for sessions from before backends existed.
    pub backend: Option<String>,
    pub agent: Option<String>,
}

impl SessionInfo {
    pub fn from_row(row: &AssistantSession) -> Self {
        Self { key: key_of(&row.id), id: row.id.clone(), title: row.title.clone(), backend: row.backend.clone(), agent: row.agent.clone() }
    }

    /// A session from before backends: it opens read-only.
    pub fn is_legacy(&self) -> bool {
        self.backend.is_none()
    }

    pub fn has_default_title(&self) -> bool {
        is_default_title(&self.title)
    }
}

/// True for the names a session gets before the person or the title job names it.
pub fn is_default_title(title: &str) -> bool {
    let t = title.trim();
    t == NEW_TITLE || t.is_empty() || t.strip_prefix("Chat ").is_some_and(|n| n.chars().all(|c| c.is_ascii_digit()) && !n.is_empty())
}

/// The first words of a message as a title.
pub fn title_from_text(text: &str) -> String {
    let flat: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let short: String = flat.chars().take(40).collect();
    if short.is_empty() { NEW_TITLE.to_string() } else if flat.chars().count() > 40 { format!("{short}...") } else { short }
}

/// Cleans a title the model wrote: one line, no quotes, at most 60 characters.
pub fn clean_title(raw: &str) -> Option<String> {
    let line = raw.lines().map(str::trim).find(|l| !l.is_empty())?;
    let t = line.trim_matches(|c: char| c == '"' || c == '\'' || c == '*' || c == '`' || c == '#').trim().trim_end_matches('.').trim();
    if t.is_empty() {
        return None;
    }
    Some(if t.chars().count() > 60 { t.chars().take(57).collect::<String>() + "..." } else { t.to_string() })
}

/// Answers the background jobs send back.
pub enum Loaded {
    List(Vec<SessionInfo>),
    Created(SessionInfo),
    Messages { key: String, rows: Vec<AssistantMessage> },
    Failed(String),
}

pub struct Sessions {
    pub list: Vec<SessionInfo>,
    pub active: Option<usize>,
    pub listed: bool,
    /// The list answer has arrived.
    pub loaded: bool,
    pub creating: bool,
    /// Index of the session the trash button asked about.
    pub confirm_delete: Option<usize>,
    tx: Sender<Loaded>,
    rx: Receiver<Loaded>,
}

impl Default for Sessions {
    fn default() -> Self {
        let (tx, rx) = crossbeam::channel::unbounded();
        Self { list: Vec::new(), active: None, listed: false, loaded: false, creating: false, confirm_delete: None, tx, rx }
    }
}

impl Sessions {
    pub fn active_info(&self) -> Option<&SessionInfo> {
        self.active.and_then(|i| self.list.get(i))
    }

    pub fn find(&self, key: &str) -> Option<usize> {
        self.list.iter().position(|s| s.key == key)
    }

    /// Reads the session list once.
    pub fn request_list(&mut self) {
        if std::mem::replace(&mut self.listed, true) {
            return;
        }
        let tx = self.tx.clone();
        tokio::spawn(async move {
            let msg = match assistant_store::list_session_rows().await {
                Ok(rows) => Loaded::List(rows.iter().map(SessionInfo::from_row).collect()),
                Err(e) => Loaded::Failed(format!("Chats could not be listed: {e}")),
            };
            let _ = tx.try_send(msg);
        });
    }

    /// Creates a session with a new id.
    pub fn create(&mut self, backend: &str, agent: Option<String>, gateway: Option<String>) {
        if std::mem::replace(&mut self.creating, true) {
            return;
        }
        let (tx, backend) = (self.tx.clone(), backend.to_string());
        tokio::spawn(async move {
            let msg = match assistant_store::create_session_with(None, NEW_TITLE, &backend, agent.as_deref(), gateway.as_deref()).await {
                Ok(id) => Loaded::Created(SessionInfo { key: key_of(&id), id, title: NEW_TITLE.to_string(), backend: Some(backend), agent }),
                Err(e) => Loaded::Failed(format!("A chat could not be created: {e}")),
            };
            let _ = tx.try_send(msg);
        });
    }

    /// Saves a session that was opened before the database was ready, under the same key.
    pub fn save_new(&self, info: &SessionInfo, gateway: &str, rows: Vec<assistant_store::NewRow>) {
        let (key, title, backend, agent, gateway) = (info.key.clone(), info.title.clone(), info.backend.clone().unwrap_or_else(|| "local".into()), info.agent.clone(), gateway.to_string());
        tokio::spawn(async move {
            match assistant_store::create_session_with(Some(&key), &title, &backend, agent.as_deref(), Some(&gateway)).await {
                Ok(id) => {
                    if let Err(e) = assistant_store::append_rows(&id, rows).await {
                        log::warn!("[assistant] the chat could not be saved: {e:#}");
                    }
                }
                Err(e) => log::warn!("[assistant] the chat could not be saved: {e:#}"),
            }
        });
    }

    /// Reads the cached rows of session `index`.
    pub fn load_messages(&self, index: usize) {
        let Some(info) = self.list.get(index) else { return };
        let (tx, id, key) = (self.tx.clone(), info.id.clone(), info.key.clone());
        tokio::spawn(async move {
            let msg = match assistant_store::load_recent(&id, CACHE_ROWS).await {
                Ok(rows) => Loaded::Messages { key, rows },
                Err(e) => Loaded::Failed(format!("The chat could not be loaded: {e}")),
            };
            let _ = tx.try_send(msg);
        });
    }

    /// Removes session `index` from the list and the database; the gateway keeps its own copy.
    pub fn delete(&mut self, index: usize) -> Option<SessionInfo> {
        if index >= self.list.len() {
            return None;
        }
        let gone = self.list.remove(index);
        self.confirm_delete = None;
        self.active = match self.active {
            Some(a) if a == index => None,
            Some(a) if a > index => Some(a - 1),
            other => other,
        };
        let id = gone.id.clone();
        tokio::spawn(async move {
            let _ = crate::database::assistant_chat::delete_session(&id).await;
        });
        Some(gone)
    }

    /// Renames session `index` in the list and the database.
    pub fn rename(&mut self, index: usize, title: &str) {
        let Some(info) = self.list.get_mut(index) else { return };
        info.title = title.to_string();
        let (id, title) = (info.id.clone(), title.to_string());
        tokio::spawn(async move {
            let _ = assistant_store::rename_session(&id, &title).await;
        });
    }

    /// Answers that arrived since the last call.
    pub fn poll(&mut self) -> Vec<Loaded> {
        let got: Vec<Loaded> = self.rx.try_iter().collect();
        for m in &got {
            match m {
                Loaded::Created(_) => self.creating = false,
                Loaded::Failed(_) => self.creating = false,
                Loaded::List(_) => self.loaded = true,
                _ => {}
            }
        }
        got
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_titles_are_recognised() {
        for t in ["New chat", "", "  ", "Chat 1", "Chat 12"] {
            assert!(is_default_title(t), "{t:?}");
        }
        for t in ["Chat", "Chat x", "Sunset tags", "Chatting"] {
            assert!(!is_default_title(t), "{t:?}");
        }
    }

    #[test]
    fn a_title_comes_from_the_first_words() {
        assert_eq!(title_from_text("  find  sunsets\nin Spain "), "find sunsets in Spain");
        assert_eq!(title_from_text(""), "New chat");
        let long = title_from_text(&"word ".repeat(30));
        assert!(long.ends_with("...") && long.chars().count() <= 43);
    }

    #[test]
    fn a_model_title_is_cleaned_to_one_short_line() {
        assert_eq!(clean_title("\"Sunset photos in Spain\"\nextra explanation"), Some("Sunset photos in Spain".to_string()));
        assert_eq!(clean_title("**Tagging plan.**"), Some("Tagging plan".to_string()));
        assert_eq!(clean_title("  \n  "), None);
        assert_eq!(clean_title("\"\""), None);
        assert!(clean_title(&"x".repeat(200)).unwrap().chars().count() <= 60);
    }

    #[test]
    fn a_session_without_a_backend_is_legacy() {
        let info = |backend: Option<&str>| SessionInfo { id: RecordId::new("assistant_sessions", "k"), key: "k".into(), title: "Chat 1".into(), backend: backend.map(str::to_string), agent: None };
        assert!(info(None).is_legacy());
        assert!(!info(Some("zeroclaw")).is_legacy());
        assert!(info(Some("local")).has_default_title());
    }

    #[tokio::test]
    async fn deleting_keeps_the_active_index_pointing_at_the_same_session() {
        let mut s = Sessions::default();
        let mk = |k: &str| SessionInfo { id: RecordId::new("assistant_sessions", k), key: k.into(), title: k.into(), backend: Some("local".into()), agent: None };
        s.list = vec![mk("a"), mk("b"), mk("c")];
        s.active = Some(2);
        s.delete(0);
        assert_eq!((s.active, s.list[s.active.unwrap()].key.as_str()), (Some(1), "c"));
        s.delete(1);
        assert_eq!(s.active, None, "deleting the open session leaves none open");
        assert!(s.delete(9).is_none());
        assert_eq!(s.find("b"), Some(0));
    }

    #[test]
    fn polling_marks_the_list_loaded_and_ends_a_creation() {
        let mut s = Sessions::default();
        s.creating = true;
        assert!(!s.loaded);
        s.tx.send(Loaded::List(vec![])).unwrap();
        s.tx.send(Loaded::Failed("x".into())).unwrap();
        assert_eq!(s.poll().len(), 2);
        assert!(s.loaded && !s.creating);
        assert!(s.poll().is_empty());
    }

    #[tokio::test]
    async fn renaming_updates_the_list() {
        let mut s = Sessions::default();
        s.list = vec![SessionInfo { id: RecordId::new("assistant_sessions", "a"), key: "a".into(), title: "x".into(), backend: None, agent: None }];
        s.rename(0, "Better");
        assert_eq!(s.list[0].title, "Better");
        s.rename(5, "ignored");
    }
}
