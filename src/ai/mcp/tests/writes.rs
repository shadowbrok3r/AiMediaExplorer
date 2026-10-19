use surrealdb::types::SurrealValue;
use serde_json::json;

use crate::ai::mcp::proposals::Status;
use crate::ai::mcp::snapshot::{self, AppSnapshot};
use crate::ai::mcp::test_support::*;
use crate::command::{ActionId, AppCommand, Scope, Selector, Target};

async fn row(f: &Fixture, rel: &str) -> (Option<String>, Vec<String>, Option<String>) {
    #[derive(Debug, SurrealValue)]
    struct R {
        caption: Option<String>,
        tags: Option<Vec<String>>,
        category: Option<String>,
    }
    let rows: Vec<R> = f.db.query("SELECT caption, tags, category FROM thumbnails WHERE path = $p").bind(("p", f.path(rel))).await.unwrap().take(0).unwrap();
    let r = rows.into_iter().next().unwrap();
    (r.caption, r.tags.unwrap_or_default(), r.category)
}

#[tokio::test]
async fn a_proposal_is_stored_and_changes_nothing() {
    let f = fixture().await;
    let before = row(&f, "photos/sub/beach3.png").await;
    let out = f.call("metadata_propose", json!({"path": f.path("photos/sub/beach3.png"), "caption": "Dunes at noon", "tags_add": ["dune", "Sand"], "category": "landscape", "reason": "looks like it"})).await.unwrap();
    assert_eq!(out.data["status"], "pending");
    assert_eq!(out.data["created"], true);
    assert_eq!(out.data["written"], false);
    assert!(out.data["id"].as_str().unwrap().starts_with('p'));
    assert_eq!(row(&f, "photos/sub/beach3.png").await, before, "the library row is untouched");
    let listed = f.call("metadata_list_proposals", json!({})).await.unwrap();
    assert_eq!(listed.data["total"], 1);
    assert_eq!(listed.data["counts"]["pending"], 1);
    assert_eq!(listed.data["items"][0]["changes"]["tags_add"], json!(["dune", "Sand"]));
    assert_eq!(listed.data["items"][0]["source"], "tester");
}

#[tokio::test]
async fn an_identical_pending_proposal_is_not_filed_twice() {
    let f = fixture().await;
    let args = json!({"path": f.path("photos/forest1.png"), "tags_add": ["pine"]});
    let first = f.call("metadata_propose", args.clone()).await.unwrap();
    let second = f.call("metadata_propose", args.clone()).await.unwrap();
    assert_eq!(first.data["id"], second.data["id"]);
    assert_eq!(second.data["created"], false);
    let other = f.call("metadata_propose", json!({"path": f.path("photos/forest1.png"), "tags_add": ["fir"]})).await.unwrap();
    assert_ne!(other.data["id"], first.data["id"]);
    assert_eq!(f.call("metadata_list_proposals", json!({"path": f.path("photos/forest1.png")})).await.unwrap().data["total"], 2);
}

#[tokio::test]
async fn proposals_are_checked_before_they_are_stored() {
    let f = fixture().await;
    let outside = f.call("metadata_propose", json!({"path": f.path("outside_library/secret.png"), "caption": "x"})).await.unwrap_err();
    assert_eq!(outside.kind(), "denied");
    let not_indexed = f.call("metadata_propose", json!({"path": f.path("photos/notes.txt"), "caption": "x"})).await.unwrap_err();
    assert_eq!(not_indexed.kind(), "not_found");
    let empty = f.call("metadata_propose", json!({"path": f.path("photos/beach1.png"), "caption": "   "})).await.unwrap_err();
    assert_eq!(empty.kind(), "invalid_arguments");
    let bad_status = f.call("metadata_list_proposals", json!({"status": "maybe"})).await.unwrap_err();
    assert_eq!(bad_status.kind(), "invalid_arguments");
    assert_eq!(f.ctx.proposals.counts().await.unwrap().pending, 0);
}

#[tokio::test]
async fn accepting_writes_the_changes_and_closes_the_proposal() {
    let f = fixture().await;
    let out = f.call("metadata_propose", json!({"path": f.path("photos/beach1.png"), "caption": "Golden beach", "tags_add": ["gold", "SEA"], "tags_remove": ["summer"], "category": "coast"})).await.unwrap();
    let id = out.data["id"].as_str().unwrap().to_string();
    let accepted = f.ctx.proposals.accept(&id).await.unwrap();
    assert_eq!(accepted.status, Status::Accepted);
    assert!(accepted.decided_ms.is_some());
    let (caption, tags, category) = row(&f, "photos/beach1.png").await;
    assert_eq!(caption.as_deref(), Some("Golden beach"));
    assert_eq!(category.as_deref(), Some("coast"));
    assert_eq!(tags, vec!["beach", "sea", "gold"]);
    let again = f.ctx.proposals.accept(&id).await.unwrap_err();
    assert_eq!(again.kind(), "invalid_arguments");
    assert!(again.message().contains("already accepted"));
    let counts = f.ctx.proposals.counts().await.unwrap();
    assert_eq!((counts.pending, counts.accepted), (0, 1));
    let history = f.call("metadata_list_proposals", json!({"status": "accepted"})).await.unwrap();
    assert_eq!(history.data["total"], 1);
}

#[tokio::test]
async fn rejecting_keeps_the_library_as_it_was() {
    let f = fixture().await;
    let before = row(&f, "photos/city1.png").await;
    let out = f.call("metadata_propose", json!({"path": f.path("photos/city1.png"), "caption": "Gotham"})).await.unwrap();
    let id = out.data["id"].as_str().unwrap().to_string();
    assert_eq!(f.ctx.proposals.reject(&id).await.unwrap().status, Status::Rejected);
    assert_eq!(row(&f, "photos/city1.png").await, before);
    assert_eq!(f.ctx.proposals.reject(&id).await.unwrap_err().kind(), "invalid_arguments");
    assert_eq!(f.ctx.proposals.accept("pdeadbeef").await.unwrap_err().kind(), "not_found");
}

#[tokio::test]
async fn accepting_after_the_file_left_the_library_fails_the_proposal() {
    let f = fixture().await;
    let out = f.call("metadata_propose", json!({"path": f.path("photos/city1.png"), "caption": "Gotham"})).await.unwrap();
    let id = out.data["id"].as_str().unwrap().to_string();
    f.db.query("DELETE thumbnails WHERE filename = 'city1.png'").await.unwrap().check().unwrap();
    assert_eq!(f.ctx.proposals.accept(&id).await.unwrap_err().kind(), "not_found");
    let stored = f.ctx.proposals.get(&id).await.unwrap().unwrap();
    assert_eq!(stored.status, Status::Failed);
    assert!(stored.note.unwrap().contains("no longer"));
}

#[tokio::test]
async fn accepting_values_the_row_already_has_is_a_harmless_no_op() {
    let f = fixture().await;
    let out = f.call("metadata_propose", json!({"path": f.path("photos/beach1.png"), "caption": "Sunny beach", "tags_add": ["sea"]})).await.unwrap();
    let id = out.data["id"].as_str().unwrap().to_string();
    let accepted = f.ctx.proposals.accept(&id).await.unwrap();
    assert_eq!(accepted.status, Status::Accepted);
    assert_eq!(accepted.note.as_deref(), Some("the row already had these values"));
}

#[tokio::test]
async fn the_catalog_has_no_tool_that_writes_metadata_directly() {
    let f = fixture().await;
    let names: Vec<String> = f.registry.catalog().into_iter().map(|r| r.name).collect();
    for name in names {
        let direct = ["set_", "update_", "write_", "delete_", "remove_", "rename_", "move_", "apply_", "save_"];
        assert!(!direct.iter().any(|p| name.starts_with(p)), "{name} looks like a direct write");
    }
}

#[tokio::test]
async fn opening_a_scope_sends_the_command_to_the_ui() {
    let f = fixture().await;
    let out = f.call("ui_open_scope", json!({"scope": "tag:sunset", "target": "background", "title": "Sunsets"})).await.unwrap();
    assert_eq!(out.data["queued"], true);
    assert_eq!(out.data["scope"], "tag:sunset");
    f.call("ui_open_scope", json!({"scope": "lib:all"})).await.unwrap();
    let folder = f.call("ui_open_scope", json!({"scope": format!("fs:{}", f.path("photos/sub"))})).await.unwrap();
    assert!(folder.text.contains("photos/sub"));
    let sent = f.ui.commands();
    assert_eq!(sent.len(), 3);
    assert!(matches!(&sent[0], AppCommand::OpenScope { scope: Scope::Tag(t), target: Target::Background, title: Some(title), .. } if t == "sunset" && title == "Sunsets"));
    assert!(matches!(&sent[1], AppCommand::OpenScope { scope: Scope::Library, target: Target::NewTab, title: None, .. }));
    assert!(matches!(&sent[2], AppCommand::OpenScope { scope: Scope::Folder(p), .. } if p.ends_with("photos/sub")));
}

#[tokio::test]
async fn scopes_outside_the_allowed_folders_or_malformed_are_refused() {
    let f = fixture().await;
    let outside = f.call("ui_open_scope", json!({"scope": format!("fs:{}", f.path("outside_library"))})).await.unwrap_err();
    assert_eq!(outside.kind(), "denied");
    let etc = f.call("ui_open_scope", json!({"scope": "fs:/etc"})).await.unwrap_err();
    assert_eq!(etc.kind(), "denied");
    let similar = f.call("ui_open_scope", json!({"scope": format!("sim:{}", f.path("outside_library/secret.png"))})).await.unwrap_err();
    assert_eq!(similar.kind(), "denied");
    let archive = f.call("ui_open_scope", json!({"scope": "zip:///etc/a.zip!/inner"})).await.unwrap_err();
    assert_eq!(archive.kind(), "denied");
    assert_eq!(f.call("ui_open_scope", json!({"scope": "nonsense"})).await.unwrap_err().kind(), "invalid_arguments");
    assert_eq!(f.call("ui_open_scope", json!({"scope": "lib:all", "target": "sideways"})).await.unwrap_err().kind(), "invalid_arguments");
    assert!(f.ui.commands().is_empty());
}

#[tokio::test]
async fn running_an_action_sends_a_checked_command() {
    let f = fixture().await;
    f.call("ui_run", json!({"action": "reveal", "paths": [f.path("photos/beach1.png")]})).await.unwrap();
    f.call("ui_run", json!({"action": "generate_clip"})).await.unwrap();
    f.call("ui_run", json!({"action": "refine", "selector": "focus"})).await.unwrap();
    let sent = f.ui.commands();
    assert!(matches!(&sent[0], AppCommand::Run { action: ActionId::Reveal, on: Selector::Paths(p) } if p == &[f.path("photos/beach1.png")]));
    assert!(matches!(&sent[1], AppCommand::Run { action: ActionId::GenerateClip, on: Selector::Selection }));
    assert!(matches!(&sent[2], AppCommand::Run { action: ActionId::Refine, on: Selector::Focus }));
}

#[tokio::test]
async fn actions_refuse_foreign_paths_unknown_actions_and_crowds() {
    let f = fixture().await;
    let outside = f.call("ui_run", json!({"action": "reveal", "paths": [f.path("outside_library/secret.png")]})).await.unwrap_err();
    assert_eq!(outside.kind(), "denied");
    let mixed = f.call("ui_run", json!({"action": "reveal", "paths": [f.path("photos/beach1.png"), "/etc/passwd"]})).await.unwrap_err();
    assert_eq!(mixed.kind(), "denied");
    for action in ["open", "delete", "move", "overwrite", "rename"] {
        let err = f.call("ui_run", json!({"action": action})).await.unwrap_err();
        assert_eq!(err.kind(), "invalid_arguments", "{action}");
    }
    let many: Vec<String> = (0..51).map(|i| f.path(&format!("photos/{i}.png"))).collect();
    assert_eq!(f.call("ui_run", json!({"action": "reveal", "paths": many})).await.unwrap_err().kind(), "invalid_arguments");
    assert!(f.ui.commands().is_empty(), "nothing reaches the UI when a call is refused");
}

#[tokio::test]
async fn the_context_reports_what_the_ui_published_and_when_it_is_stale() {
    let f = fixture().await;
    let before = f.call("app_get_context", json!({})).await.unwrap();
    assert_eq!(before.data["stale"], true);
    assert!(before.text.contains("not reported"));
    let snap = AppSnapshot { tab: Some("Explorer".into()), scope: Some("fs:/pics".into()), row_count: Some(40), selection_count: 3, filters: vec!["Search: sun".into()], collection: Some("Holidays".into()), db_connected: true, ..Default::default() };
    assert!(snapshot::store(&f.ctx.snapshot, snap, f.ctx.clock.unix_ms()));
    let fresh = f.call("app_get_context", json!({})).await.unwrap();
    assert_eq!(fresh.data["stale"], false);
    assert_eq!(fresh.data["snapshot"]["selection_count"], 3);
    assert_eq!(fresh.data["snapshot"]["filters"], json!(["Search: sun"]));
    assert_eq!(fresh.data["snapshot"]["collection"], "Holidays");
    assert!(fresh.text.contains("Explorer") && fresh.text.contains("3 selected"));
    assert!(fresh.data["allowed_folders"].as_array().unwrap().iter().any(|p| p.as_str().unwrap().ends_with("photos")));
    f.clock.advance(std::time::Duration::from_secs(31));
    let old = f.call("app_get_context", json!({})).await.unwrap();
    assert_eq!(old.data["stale"], true);
    assert_eq!(old.data["age_ms"], 31_000);
}
