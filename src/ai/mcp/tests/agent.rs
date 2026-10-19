use std::time::{Duration, Instant};

use serde_json::{Value, json};
use surrealdb::types::SurrealValue;

use crate::ai::mcp::edits::{EditState, MAX_ACTIVE, Review};
use crate::ai::mcp::proposals::Status;
use crate::ai::mcp::taxonomy::{MAX_PENDING, MAX_SOURCES};
use crate::ai::mcp::test_support::*;
use crate::command::ToastLevel;

async fn tags_of(f: &Fixture, rel: &str) -> Vec<String> {
    let rows: Vec<Option<Vec<String>>> = f.db.query("SELECT VALUE tags FROM thumbnails WHERE path = $p").bind(("p", f.path(rel))).await.unwrap().take(0).unwrap();
    rows.into_iter().next().flatten().unwrap_or_default()
}

async fn category_of(f: &Fixture, rel: &str) -> Option<String> {
    let rows: Vec<Option<String>> = f.db.query("SELECT VALUE category FROM thumbnails WHERE path = $p").bind(("p", f.path(rel))).await.unwrap().take(0).unwrap();
    rows.into_iter().next().flatten()
}

fn id_of(out: &crate::ai::mcp::error::ToolOutput) -> String {
    out.data["id"].as_str().unwrap().to_string()
}

// ------------------------------------------------------------------ taxonomy proposals

#[tokio::test]
async fn a_tag_rename_is_filed_with_counts_and_changes_nothing() {
    let f = fixture().await;
    let before = (tags_of(&f, "photos/beach1.png").await, tags_of(&f, "photos/beach2.png").await);
    let out = f.call("taxonomy_propose_rename", json!({"kind": "tag", "from": ["beach"], "to": "seaside", "reason": "one word for the coast"})).await.unwrap();
    assert_eq!(out.data["status"], "pending");
    assert_eq!((out.data["created"].clone(), out.data["written"].clone()), (json!(true), json!(false)));
    assert_eq!(out.data["kind"], "tag");
    assert_eq!(out.data["from"], json!(["beach"]));
    assert_eq!(out.data["preview"]["items"], 2);
    assert_eq!(out.data["preview"]["merged"], 0);
    assert_eq!(out.data["preview"]["per_name"], json!([{"name": "beach", "items": 2}]));
    assert!(out.data["preview"]["summary"].as_str().unwrap().contains("2 items"));
    assert!(out.data["id"].as_str().unwrap().starts_with('x'));
    assert_eq!(out.data["source"], "tester");
    assert_eq!((tags_of(&f, "photos/beach1.png").await, tags_of(&f, "photos/beach2.png").await), before, "the library is untouched");
    let listed = f.call("taxonomy_list_proposals", json!({})).await.unwrap();
    assert_eq!(listed.data["total"], 1);
    assert_eq!(listed.data["counts"]["pending"], 1);
    assert_eq!(listed.data["items"][0]["to"], "seaside");
    assert_eq!(listed.data["items"][0]["items_at_filing"], 2);
}

#[tokio::test]
async fn a_merge_counts_the_items_that_already_carry_the_new_name() {
    let f = fixture().await;
    let out = f.call("taxonomy_propose_rename", json!({"kind": "tag", "from": ["beach", "sea"], "to": "dusk"})).await.unwrap();
    assert_eq!(out.data["preview"]["items"], 2, "beach1 has both, beach2 has beach");
    assert_eq!(out.data["preview"]["merged"], 1, "beach2 already carries dusk");
    assert_eq!(out.data["preview"]["target_items"], 1);
    assert_eq!(out.data["preview"]["per_name"], json!([{"name": "beach", "items": 2}, {"name": "sea", "items": 1}]));
    assert!(out.text.contains("2 items change") && out.text.contains("merge"), "{}", out.text);
}

#[tokio::test]
async fn a_category_rename_counts_items_by_category() {
    let f = fixture().await;
    let out = f.call("taxonomy_propose_rename", json!({"kind": "category", "from": ["landscape"], "to": "nature"})).await.unwrap();
    assert_eq!(out.data["preview"]["items"], 2);
    assert_eq!(out.data["preview"]["target_items"], 1, "forest1 is already nature");
}

#[tokio::test]
async fn an_identical_pending_taxonomy_proposal_is_not_filed_twice() {
    let f = fixture().await;
    let args = json!({"kind": "tag", "from": ["beach", "sea"], "to": "coast"});
    let first = f.call("taxonomy_propose_rename", args.clone()).await.unwrap();
    let again = f.call("taxonomy_propose_rename", json!({"kind": "tag", "from": ["sea", "beach"], "to": "coast"})).await.unwrap();
    assert_eq!(first.data["id"], again.data["id"], "the order of the names does not matter");
    assert_eq!(again.data["created"], false);
    let other = f.call("taxonomy_propose_rename", json!({"kind": "tag", "from": ["beach"], "to": "coast"})).await.unwrap();
    assert_ne!(other.data["id"], first.data["id"]);
    assert_eq!(f.call("taxonomy_list_proposals", json!({})).await.unwrap().data["total"], 2);
}

#[tokio::test]
async fn taxonomy_proposals_are_checked_before_they_are_stored() {
    let f = fixture().await;
    let cases = [
        (json!({"kind": "tag", "from": [], "to": "x"}), "invalid_arguments"),
        (json!({"kind": "tag", "from": ["  "], "to": "x"}), "invalid_arguments"),
        (json!({"kind": "tag", "from": ["beach"], "to": " "}), "invalid_arguments"),
        (json!({"kind": "tag", "from": ["beach"], "to": "beach"}), "invalid_arguments"),
        (json!({"kind": "tag", "from": ["beach", "sea"], "to": "sea"}), "invalid_arguments"),
        (json!({"kind": "tag", "from": ["nothing-like-this"], "to": "x"}), "not_found"),
        (json!({"kind": "category", "from": ["no such category"], "to": "x"}), "not_found"),
    ];
    for (args, kind) in cases {
        let err = f.call("taxonomy_propose_rename", args.clone()).await.unwrap_err();
        assert_eq!(err.kind(), kind, "{args}: {err}");
    }
    let many: Vec<String> = (0..MAX_SOURCES + 1).map(|i| format!("t{i}")).collect();
    let err = f.call("taxonomy_propose_rename", json!({"kind": "tag", "from": many, "to": "x"})).await.unwrap_err();
    assert!(err.message().contains("at most"), "{err}");
    let bad_kind = f.call("taxonomy_propose_rename", json!({"kind": "folder", "from": ["a"], "to": "b"})).await.unwrap_err();
    assert_eq!(bad_kind.kind(), "invalid_arguments", "an unknown kind is a bad argument");
    let missing = f.call("taxonomy_propose_rename", json!({"kind": "tag", "to": "b"})).await.unwrap_err();
    assert_eq!(missing.kind(), "invalid_arguments");
    assert_eq!(f.call("taxonomy_list_proposals", json!({"status": "maybe"})).await.unwrap_err().kind(), "invalid_arguments");
    assert_eq!(f.ctx.agent.taxonomy.counts().await.unwrap().pending, 0, "nothing was stored");
}

#[tokio::test]
async fn names_are_cleaned_and_cut_before_they_are_stored() {
    let f = fixture().await;
    let out = f.call("taxonomy_propose_rename", json!({"kind": "tag", "from": ["  beach\u{0} "], "to": format!("{}\n{}", "x".repeat(30), "y".repeat(30)), "reason": "r".repeat(900)})).await.unwrap();
    assert_eq!(out.data["from"], json!(["beach"]));
    let to = out.data["to"].as_str().unwrap();
    assert_eq!(to.chars().count(), 40);
    assert!(to.contains(' ') && !to.contains('\n'));
    assert_eq!(out.data["reason"].as_str().unwrap().chars().count(), crate::ai::mcp::proposals::MAX_REASON);
}

#[tokio::test]
async fn too_many_pending_taxonomy_proposals_are_refused() {
    let f = fixture().await;
    for i in 0..MAX_PENDING {
        f.ctx.agent.taxonomy.create(crate::ai::mcp::taxonomy::NewTaxonomy { rename: crate::ai::mcp::taxonomy::Rename::new(crate::ai::mcp::taxonomy::Kind::Tag, &["beach".to_string()], &format!("n{i}")).unwrap(), reason: None, source: "bulk".into() }, 2).await.unwrap();
    }
    let err = f.call("taxonomy_propose_rename", json!({"kind": "tag", "from": ["beach"], "to": "one-too-many"})).await.unwrap_err();
    assert_eq!(err.kind(), "busy");
    assert!(err.message().contains("review"), "{err}");
}

#[tokio::test]
async fn accepting_a_tag_merge_applies_it_through_the_journal_and_undo_restores_it() {
    let f = fixture().await;
    let before: Vec<Vec<String>> = vec![tags_of(&f, "photos/beach1.png").await, tags_of(&f, "photos/beach2.png").await];
    let out = f.call("taxonomy_propose_rename", json!({"kind": "tag", "from": ["beach", "sea"], "to": "coast"})).await.unwrap();
    let accepted = f.ctx.agent.taxonomy.accept(&id_of(&out)).await.unwrap();
    assert_eq!(accepted.status, Status::Accepted);
    assert!(accepted.applied.unwrap() >= 2, "{:?}", accepted.applied);
    for rel in ["photos/beach1.png", "photos/beach2.png"] {
        let tags = tags_of(&f, rel).await;
        assert!(tags.contains(&"coast".to_string()) && !tags.contains(&"beach".to_string()) && !tags.contains(&"sea".to_string()), "{rel}: {tags:?}");
    }
    assert!(tags_of(&f, "photos/beach1.png").await.contains(&"summer".to_string()), "other tags stay");
    let entries = crate::database::journal::recent_with(&f.db, 20).await.unwrap();
    assert!(entries.iter().any(|e| e.source.contains("Rename tag \"beach\"")), "{entries:?}");
    assert!(entries.iter().any(|e| e.source.contains("Rename tag \"sea\"")), "one batch per old name");
    while crate::database::journal::undo_last_with(&f.db).await.map(|r| !r.is_empty()).unwrap_or(false) {}
    assert_eq!(vec![tags_of(&f, "photos/beach1.png").await, tags_of(&f, "photos/beach2.png").await], before);
    let again = f.ctx.agent.taxonomy.accept(&id_of(&out)).await.unwrap_err();
    assert_eq!(again.kind(), "invalid_arguments");
    assert!(again.message().contains("already accepted"));
}

#[tokio::test]
async fn accepting_a_category_rename_updates_every_item() {
    let f = fixture().await;
    let out = f.call("taxonomy_propose_rename", json!({"kind": "category", "from": ["landscape"], "to": "scenery"})).await.unwrap();
    let accepted = f.ctx.agent.taxonomy.accept(&id_of(&out)).await.unwrap();
    assert_eq!((accepted.status, accepted.applied), (Status::Accepted, Some(2)));
    assert_eq!(category_of(&f, "photos/beach1.png").await.as_deref(), Some("scenery"));
    assert_eq!(category_of(&f, "photos/beach2.png").await.as_deref(), Some("scenery"));
    assert_eq!(category_of(&f, "photos/forest1.png").await.as_deref(), Some("nature"));
    let undone = crate::database::journal::undo_last_with(&f.db).await.unwrap();
    assert_eq!(undone.applied.len(), 2);
    assert_eq!(category_of(&f, "photos/beach1.png").await.as_deref(), Some("landscape"));
}

#[tokio::test]
async fn rejecting_a_taxonomy_proposal_keeps_the_library_as_it_was() {
    let f = fixture().await;
    let before = tags_of(&f, "photos/forest1.png").await;
    let out = f.call("taxonomy_propose_rename", json!({"kind": "tag", "from": ["forest"], "to": "woods"})).await.unwrap();
    let id = id_of(&out);
    assert_eq!(f.ctx.agent.taxonomy.reject(&id).await.unwrap().status, Status::Rejected);
    assert_eq!(tags_of(&f, "photos/forest1.png").await, before);
    assert_eq!(f.ctx.agent.taxonomy.reject(&id).await.unwrap_err().kind(), "invalid_arguments");
    assert_eq!(f.ctx.agent.taxonomy.accept("xdeadbeef").await.unwrap_err().kind(), "not_found");
    let rejected = f.call("taxonomy_list_proposals", json!({"status": "rejected"})).await.unwrap();
    assert_eq!(rejected.data["total"], 1);
    assert_eq!(f.call("taxonomy_list_proposals", json!({"status": "pending"})).await.unwrap().data["total"], 0);
}

#[tokio::test]
async fn accepting_after_the_library_changed_is_a_harmless_no_op() {
    let f = fixture().await;
    let out = f.call("taxonomy_propose_rename", json!({"kind": "tag", "from": ["city"], "to": "town"})).await.unwrap();
    f.db.query("UPDATE thumbnails SET tags = ['night'] WHERE filename = 'city1.png'").await.unwrap().check().unwrap();
    let accepted = f.ctx.agent.taxonomy.accept(&id_of(&out)).await.unwrap();
    assert_eq!((accepted.status, accepted.applied), (Status::Accepted, Some(0)));
    assert!(accepted.note.unwrap().contains("no item carried"));
}

#[tokio::test]
async fn taxonomy_pages_are_bounded() {
    let f = fixture().await;
    for to in ["a1", "a2", "a3"] {
        f.call("taxonomy_propose_rename", json!({"kind": "tag", "from": ["beach"], "to": to})).await.unwrap();
    }
    let page = f.call("taxonomy_list_proposals", json!({"limit": 2})).await.unwrap();
    assert_eq!((page.data["returned"].clone(), page.data["total"].clone(), page.data["next_offset"].clone()), (json!(2), json!(3), json!(2)));
    let clamped = f.call("taxonomy_list_proposals", json!({"limit": 5000})).await.unwrap();
    assert_eq!(clamped.data["limit"], 100);
}

// ------------------------------------------------------------------ image edit

async fn status_of(f: &Fixture, id: &str) -> Value {
    f.call("image_edit_status", json!({"job_id": id})).await.unwrap().data
}

async fn wait_for(f: &Fixture, id: &str, state: &str) -> Value {
    let end = Instant::now() + Duration::from_secs(20);
    loop {
        let data = status_of(f, id).await;
        if data["state"] == state {
            return data;
        }
        assert!(Instant::now() < end, "job {id} did not reach {state}: {data}");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

async fn start(f: &Fixture, rel: &str, extra: Value) -> Value {
    let mut args = json!({"path": f.path(rel), "prompt": "make it warm", "steps": 2});
    args.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    f.call("image_edit_start", args).await.unwrap().data
}

fn scratch_files(f: &Fixture) -> Vec<String> {
    std::fs::read_dir(f.ctx.agent.edits.scratch_dir()).map(|d| d.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().to_string()).collect()).unwrap_or_default()
}

#[tokio::test]
async fn an_edit_starts_at_once_runs_in_the_background_and_leaves_the_original_alone() {
    let f = fixture().await;
    let original = std::fs::read(f.path("photos/beach1.png")).unwrap();
    let started = Instant::now();
    let job = start(&f, "photos/beach1.png", json!({})).await;
    assert!(started.elapsed() < Duration::from_millis(250), "the call does not wait for the model");
    let id = job["job_id"].as_str().unwrap().to_string();
    assert!(id.starts_with('e'));
    assert!(["queued", "loading_model", "running"].contains(&job["state"].as_str().unwrap()), "{job}");
    assert_eq!(job["path"], f.path("photos/beach1.png"));
    let done = wait_for(&f, &id, "done").await;
    assert_eq!(done["review"], "pending");
    assert_eq!(done["progress"]["done"], 2);
    assert_eq!((done["result"]["width"].clone(), done["result"]["height"].clone()), (json!(8), json!(8)));
    assert!(done["result"]["bytes"].as_u64().unwrap() > 20);
    assert!(done["note"].as_str().unwrap().contains("waits for the user"));
    assert!(done.get("saved_to").is_some_and(|v| v.is_null()));
    assert!(!done.to_string().contains(&*f.ctx.agent.edits.scratch_dir().to_string_lossy()), "the scratch path stays inside the application: {done}");
    assert_eq!(std::fs::read(f.path("photos/beach1.png")).unwrap(), original, "the original is never written");
    let written = scratch_files(&f);
    assert_eq!(written, vec![format!("{id}.png")], "the result is in the scratch folder, nothing else");
    assert!(f.ctx.agent.edits.scratch_dir().starts_with(f.dir()), "the test scratch folder is inside the fixture");
    let png = std::fs::read(f.ctx.agent.edits.scratch_dir().join(format!("{id}.png"))).unwrap();
    assert_eq!(&png[..4], b"\x89PNG");
    let messages = f.notifier.messages();
    assert_eq!(messages.len(), 1, "{messages:?}");
    assert_eq!(messages[0].0, ToastLevel::Success);
    assert!(messages[0].1.contains("beach1.png") && messages[0].1.contains("Review"), "{}", messages[0].1);
}

#[tokio::test]
async fn the_job_shows_up_in_the_gpu_queue_and_in_the_status_list() {
    let f = fixture().await;
    let job = start(&f, "photos/forest1.png", json!({})).await;
    let id = job["job_id"].as_str().unwrap().to_string();
    let end = Instant::now() + Duration::from_secs(10);
    loop {
        let q = f.call("gpu_queue_status", json!({})).await.unwrap().data;
        if q["running"]["kind"] == "image_edit" || q["recent"].as_array().is_some_and(|r| r.iter().any(|j| j["kind"] == "image_edit")) {
            break;
        }
        assert!(Instant::now() < end, "the edit never reached the GPU queue: {q}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    wait_for(&f, &id, "done").await;
    let q = f.call("gpu_queue_status", json!({})).await.unwrap().data;
    let recent = q["recent"].as_array().unwrap();
    let job = recent.iter().find(|j| j["kind"] == "image_edit").unwrap();
    assert_eq!(job["state"], "done");
    assert!(job["label"].as_str().unwrap().contains(&id));
    let list = f.call("image_edit_status", json!({})).await.unwrap();
    assert_eq!((list.data["total"].clone(), list.data["active"].clone(), list.data["waiting_for_review"].clone()), (json!(1), json!(0), json!(1)));
    assert_eq!(list.data["items"][0]["job_id"], id.as_str());
}

#[tokio::test]
async fn edit_arguments_are_checked_and_the_path_policy_applies() {
    let f = fixture().await;
    let ok = json!({"path": f.path("photos/beach1.png"), "prompt": "x"});
    let with = |k: &str, v: Value| {
        let mut a = ok.clone();
        a[k] = v;
        a
    };
    for (args, kind) in [
        (with("prompt", json!("   ")), "invalid_arguments"),
        (with("steps", json!(0)), "invalid_arguments"),
        (with("steps", json!(51)), "invalid_arguments"),
        (with("guidance_scale", json!(0.5)), "invalid_arguments"),
        (with("guidance_scale", json!(25.0)), "invalid_arguments"),
        (with("strength", json!(0.0)), "invalid_arguments"),
        (with("strength", json!(1.5)), "invalid_arguments"),
        (with("path", json!(f.path("outside_library/secret.png"))), "denied"),
        (with("path", json!(format!("{}/../outside_library/secret.png", f.path("photos")))), "invalid_arguments"),
        (with("path", json!(f.path("photos/missing.png"))), "not_found"),
        (with("path", json!(f.path("photos/notes.txt"))), "invalid_arguments"),
        (with("path", json!("relative/a.png")), "invalid_arguments"),
        (with("path", json!("/etc/passwd")), "denied"),
    ] {
        let err = f.call("image_edit_start", args.clone()).await.unwrap_err();
        assert_eq!(err.kind(), kind, "{args}: {err}");
    }
    assert!(f.call("image_edit_start", json!({"prompt": "x"})).await.is_err(), "the path is required");
    assert_eq!(f.call("image_edit_status", json!({})).await.unwrap().data["total"], 0, "nothing was started");
    assert!(scratch_files(&f).is_empty());
}

#[tokio::test]
async fn long_prompts_are_cut_and_set_values_reach_the_backend() {
    let f = fixture().await;
    let job = start(&f, "photos/city1.png", json!({"prompt": format!("neon {}", "x ".repeat(800)), "negative_prompt": "blur", "guidance_scale": 5.5, "strength": 0.4})).await;
    assert!(job["prompt"].as_str().unwrap().chars().count() <= 300);
    let jobs = f.ctx.agent.edits.list();
    assert_eq!(jobs.len(), 1);
    assert!(jobs[0].prompt.chars().count() <= 1000);
    assert_eq!((jobs[0].negative_prompt.as_deref(), jobs[0].guidance_scale, jobs[0].strength, jobs[0].steps), (Some("blur"), 5.5, 0.4, 2));
    wait_for(&f, job["job_id"].as_str().unwrap(), "done").await;
}

#[tokio::test]
async fn at_most_three_jobs_are_active_and_a_queued_job_reports_its_place() {
    let f = fixture().await;
    let mut ids = Vec::new();
    for rel in ["photos/beach1.png", "photos/beach2.png", "photos/forest1.png"] {
        ids.push(start(&f, rel, json!({"steps": 12})).await["job_id"].as_str().unwrap().to_string());
    }
    assert_eq!(MAX_ACTIVE, 3);
    let err = f.call("image_edit_start", json!({"path": f.path("photos/city1.png"), "prompt": "x"})).await.unwrap_err();
    assert_eq!(err.kind(), "busy");
    assert!(err.message().contains("already queued or running"), "{err}");
    let end = Instant::now() + Duration::from_secs(10);
    let third = loop {
        let s = status_of(&f, &ids[2]).await;
        if s["queue_position"].is_number() {
            break s;
        }
        assert!(Instant::now() < end, "no queue position: {s}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    };
    assert_eq!(third["state"], "queued");
    assert!(third["queue_position"].as_u64().unwrap() >= 1, "{third}");
    for id in &ids {
        f.call("image_edit_cancel", json!({"job_id": id})).await.unwrap();
    }
    let list = f.call("image_edit_status", json!({})).await.unwrap();
    assert_eq!(list.data["active"], 0);
}

#[tokio::test]
async fn a_running_job_can_be_cancelled_and_leaves_no_result() {
    let f = fixture().await;
    let job = start(&f, "photos/beach1.png", json!({"steps": 12})).await;
    let id = job["job_id"].as_str().unwrap().to_string();
    wait_for(&f, &id, "running").await;
    let cancelled = f.call("image_edit_cancel", json!({"job_id": id})).await.unwrap();
    assert_eq!(cancelled.data["state"], "cancelled");
    assert_eq!(status_of(&f, &id).await["state"], "cancelled");
    let again = f.call("image_edit_cancel", json!({"job_id": id})).await.unwrap_err();
    assert_eq!(again.kind(), "invalid_arguments");
    assert!(again.message().contains("already cancelled"));
    tokio::time::sleep(Duration::from_millis(400)).await;
    assert_eq!(status_of(&f, &id).await["state"], "cancelled", "a cancelled job stays cancelled");
    assert!(scratch_files(&f).is_empty(), "no result is kept: {:?}", scratch_files(&f));
    assert!(f.notifier.messages().is_empty(), "a cancel the agent asked for is not reported");
    assert_eq!(f.call("image_edit_cancel", json!({"job_id": "enope"})).await.unwrap_err().kind(), "not_found");
    assert_eq!(f.call("image_edit_status", json!({"job_id": "enope"})).await.unwrap_err().kind(), "not_found");
}

#[tokio::test]
async fn a_finished_job_cannot_be_cancelled() {
    let f = fixture().await;
    let job = start(&f, "photos/beach1.png", json!({})).await;
    let id = job["job_id"].as_str().unwrap().to_string();
    wait_for(&f, &id, "done").await;
    let err = f.call("image_edit_cancel", json!({"job_id": id})).await.unwrap_err();
    assert_eq!(err.kind(), "invalid_arguments");
}

#[tokio::test]
async fn a_failed_edit_reports_the_error_and_tells_the_user() {
    let f = fixture().await;
    let job = start(&f, "photos/beach1.png", json!({"prompt": "simulate oom now"})).await;
    let id = job["job_id"].as_str().unwrap().to_string();
    let failed = wait_for(&f, &id, "failed").await;
    assert!(failed["error"].as_str().unwrap().contains("OUT_OF_MEMORY"), "{failed}");
    assert!(failed["result"].is_null());
    assert!(scratch_files(&f).is_empty());
    let messages = f.notifier.messages();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].0, ToastLevel::Error);
    assert!(messages[0].1.contains("failed"), "{}", messages[0].1);
    let started = Instant::now();
    let next = start(&f, "photos/beach2.png", json!({})).await;
    wait_for(&f, next["job_id"].as_str().unwrap(), "done").await;
    assert!(started.elapsed() < Duration::from_secs(10), "a failed job does not block the queue");
}

#[tokio::test]
async fn the_result_can_be_saved_as_a_new_file_next_to_the_original_and_never_over_it() {
    let f = fixture().await;
    let original = std::fs::read(f.path("photos/beach2.png")).unwrap();
    let job = start(&f, "photos/beach2.png", json!({})).await;
    let id = job["job_id"].as_str().unwrap().to_string();
    wait_for(&f, &id, "done").await;
    let (saved, target) = f.ctx.agent.edits.save_new(&id).unwrap();
    assert_eq!(target, f.dir().join("photos").join("beach2-edited.png"));
    assert!(target.is_file());
    assert_eq!((saved.review, saved.saved_to.as_deref()), (Review::Saved, Some(target.display().to_string().as_str())));
    assert_eq!(std::fs::read(f.path("photos/beach2.png")).unwrap(), original);
    assert!(scratch_files(&f).is_empty(), "the scratch copy goes once the file is saved");
    assert_eq!(f.ctx.agent.edits.save_new(&id).unwrap_err().kind(), "invalid_arguments");
    assert_eq!(f.ctx.agent.edits.reject(&id).unwrap_err().kind(), "invalid_arguments");
    let status = status_of(&f, &id).await;
    assert_eq!(status["review"], "saved");
    assert!(status["saved_to"].as_str().unwrap().ends_with("beach2-edited.png"));
    let second = start(&f, "photos/beach2.png", json!({})).await;
    let id2 = second["job_id"].as_str().unwrap().to_string();
    wait_for(&f, &id2, "done").await;
    let (_, again) = f.ctx.agent.edits.save_new(&id2).unwrap();
    assert!(again.ends_with("beach2-edited-2.png"), "a taken name is never replaced: {again:?}");
}

#[tokio::test]
async fn the_result_can_be_added_to_the_library() {
    let f = fixture().await;
    let job = start(&f, "photos/forest1.png", json!({})).await;
    let id = job["job_id"].as_str().unwrap().to_string();
    wait_for(&f, &id, "done").await;
    let (saved, target) = f.ctx.agent.edits.add_to_library(&id, &f.ctx.db).await.unwrap();
    assert_eq!(saved.review, Review::Added);
    #[derive(Debug, SurrealValue)]
    struct P {
        path: String,
    }
    let rows: Vec<P> = f.db.query("SELECT path FROM thumbnails WHERE path = $p").bind(("p", target.display().to_string())).await.unwrap().take(0).unwrap();
    assert_eq!(rows.len(), 1, "the new file is a library row");
    assert_eq!(row_count(&f.db, "thumbnails").await, 6);
}

#[tokio::test]
async fn a_rejected_result_is_deleted() {
    let f = fixture().await;
    let job = start(&f, "photos/city1.png", json!({})).await;
    let id = job["job_id"].as_str().unwrap().to_string();
    wait_for(&f, &id, "done").await;
    assert_eq!(scratch_files(&f).len(), 1);
    let rejected = f.ctx.agent.edits.reject(&id).unwrap();
    assert_eq!(rejected.review, Review::Rejected);
    assert!(scratch_files(&f).is_empty());
    assert!(!f.dir().join("photos").join("city1-edited.png").exists());
    assert_eq!(status_of(&f, &id).await["review"], "rejected");
    assert_eq!(f.ctx.agent.edits.counts(), (0, 0));
}

#[tokio::test]
async fn results_nobody_reviewed_stop_new_edits() {
    let f = fixture().await;
    for i in 0..crate::ai::mcp::edits::MAX_UNREVIEWED {
        let job = start(&f, "photos/beach1.png", json!({"steps": 1})).await;
        let id = job["job_id"].as_str().unwrap().to_string();
        wait_for(&f, &id, "done").await;
        assert_eq!(f.ctx.agent.edits.counts().1, i + 1);
    }
    let err = f.call("image_edit_start", json!({"path": f.path("photos/beach1.png"), "prompt": "x"})).await.unwrap_err();
    assert_eq!(err.kind(), "busy");
    assert!(err.message().contains("review"), "{err}");
    let oldest = f.ctx.agent.edits.list().pop().unwrap();
    f.ctx.agent.edits.reject(&oldest.id).unwrap();
    assert!(f.call("image_edit_start", json!({"path": f.path("photos/beach1.png"), "prompt": "x", "steps": 1})).await.is_ok());
}

#[tokio::test]
async fn the_status_can_attach_the_finished_result_as_a_jpeg() {
    let f = fixture().await;
    let job = start(&f, "photos/beach1.png", json!({})).await;
    let id = job["job_id"].as_str().unwrap().to_string();
    let early = f.call("image_edit_status", json!({"job_id": id, "include_preview": true})).await.unwrap();
    assert!(early.images.is_empty(), "nothing to attach before the job is done");
    wait_for(&f, &id, "done").await;
    let out = f.call("image_edit_status", json!({"job_id": id, "include_preview": true})).await.unwrap();
    assert_eq!(out.images.len(), 1);
    assert_eq!(out.images[0].mime, "image/jpeg");
    assert!(out.text.contains("attached"));
    let plain = f.call("image_edit_status", json!({"job_id": id})).await.unwrap();
    assert!(plain.images.is_empty());
    f.ctx.agent.edits.reject(&id).unwrap();
    let gone = f.call("image_edit_status", json!({"job_id": id, "include_preview": true})).await.unwrap();
    assert!(gone.images.is_empty());
}

#[tokio::test]
async fn a_source_that_vanished_fails_the_job_and_does_not_panic() {
    let f = fixture().await;
    let job = start(&f, "photos/beach1.png", json!({"steps": 12})).await;
    let id = job["job_id"].as_str().unwrap().to_string();
    std::fs::remove_file(f.path("photos/beach1.png")).unwrap();
    let failed = wait_for(&f, &id, "failed").await;
    assert!(failed["error"].is_string());
    assert_eq!(f.ctx.agent.edits.get(&id).unwrap().state, EditState::Failed);
}

// ------------------------------------------------------------------ catalog

#[tokio::test]
async fn the_new_tools_have_documented_object_schemas_and_the_right_classes() {
    let f = fixture().await;
    let catalog = f.registry.catalog();
    let find = |n: &str| catalog.iter().find(|r| r.name == n).unwrap_or_else(|| panic!("{n} missing"));
    for (name, class, required) in [
        ("taxonomy_propose_rename", "propose", vec!["kind", "from", "to"]),
        ("taxonomy_list_proposals", "read", vec![]),
        ("image_edit_start", "gpu", vec!["path", "prompt"]),
        ("image_edit_status", "read", vec![]),
        ("image_edit_cancel", "gpu", vec!["job_id"]),
    ] {
        let row = find(name);
        assert_eq!(row.class.label(), class, "{name}");
        assert_eq!(row.input_schema["type"], "object");
        let required_in_schema: Vec<String> = row.input_schema.get("required").and_then(|r| r.as_array()).map(|r| r.iter().map(|v| v.as_str().unwrap().to_string()).collect()).unwrap_or_default();
        for field in required {
            assert!(required_in_schema.contains(&field.to_string()), "{name} should require {field}: {}", row.input_schema);
        }
        for (prop, schema) in row.input_schema["properties"].as_object().unwrap() {
            assert!(schema.get("description").is_some() || schema.get("$ref").is_some() || schema.get("allOf").is_some(), "{name}.{prop} is undocumented: {schema}");
        }
        assert!(row.description.len() > 60, "{name}");
    }
    assert!(find("taxonomy_propose_rename").input_schema.to_string().contains("category"), "the kind choices are in the schema");
}
