//! Headless tests of the agent proposals panel in Settings > Metadata tools, over an in-memory library and the edit job store of the MCP tests.

use std::time::Duration;

use eframe::egui::{CentralPanel, Context, Pos2, RawInput, Rect, Shape, epaint::ClippedShape, vec2};
use serde_json::json;

use super::agent_proposals::Panel;
use crate::ai::mcp::test_support::{Fixture, fixture};
use crate::command::{AppCommand, ToastLevel};
use crate::ui::test_support::{press_release, test_ctx};

fn text_rects(shapes: &[ClippedShape]) -> Vec<(String, Rect)> {
    fn walk(shape: &Shape, out: &mut Vec<(String, Rect)>) {
        match shape {
            Shape::Text(t) => out.push((t.galley.text().to_string(), t.galley.rect.translate(t.pos.to_vec2()))),
            Shape::Vec(v) => v.iter().for_each(|s| walk(s, out)),
            _ => {}
        }
    }
    let mut out = Vec::new();
    shapes.iter().for_each(|s| walk(&s.shape, &mut out));
    out
}

struct Rig {
    ctx: Context,
    panel: Panel,
    shapes: Vec<ClippedShape>,
    width: f32,
    db_ready: bool,
}

impl Rig {
    fn new(f: &Fixture, width: f32) -> Self {
        Self { ctx: test_ctx(), panel: Panel::with_agent(f.ctx.agent.clone(), f.ctx.db.clone()), shapes: Vec::new(), width, db_ready: true }
    }

    fn frame_with(&mut self, events: Vec<eframe::egui::Event>) {
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(self.width, 1400.0))), events, ..Default::default() };
        let Self { ctx, panel, db_ready, .. } = self;
        let full = ctx.run_ui(input, |ui| {
            CentralPanel::default().show(ui, |ui| panel.ui(ui, *db_ready));
        });
        self.shapes = full.shapes.clone();
        full.drop_without_applying_deltas();
    }

    fn frame(&mut self) {
        self.frame_with(Vec::new());
    }

    fn names(&self) -> Vec<String> {
        text_rects(&self.shapes).into_iter().map(|t| t.0).collect()
    }

    fn has(&self, text: &str) -> bool {
        self.names().iter().any(|t| t == text)
    }

    fn has_containing(&self, part: &str) -> bool {
        self.names().iter().any(|t| t.contains(part))
    }

    fn click(&mut self, text: &str) {
        let rect = text_rects(&self.shapes).into_iter().find(|(t, _)| t == text).map(|(_, r)| r).unwrap_or_else(|| panic!("no {text:?}: {:?}", self.names()));
        self.frame_with(press_release(rect.center()));
        self.frame();
    }

    async fn until(&mut self, what: &str, mut cond: impl FnMut(&mut Rig) -> bool) {
        let end = std::time::Instant::now() + Duration::from_secs(10);
        while std::time::Instant::now() < end {
            self.frame();
            if cond(self) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(15)).await;
        }
        panic!("timed out waiting for {what}: {:?}", self.names());
    }
}

async fn propose(f: &Fixture, args: serde_json::Value) -> String {
    f.call("taxonomy_propose_rename", args).await.unwrap().data["id"].as_str().unwrap().to_string()
}

async fn finished_edit(f: &Fixture, rel: &str) -> String {
    let started = f.call("image_edit_start", json!({"path": f.path(rel), "prompt": "warm", "steps": 1})).await.unwrap();
    let id = started.data["job_id"].as_str().unwrap().to_string();
    let end = std::time::Instant::now() + Duration::from_secs(10);
    while f.ctx.agent.edits.get(&id).unwrap().result.is_none() {
        assert!(std::time::Instant::now() < end, "the edit never finished");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    id
}

fn toasts(marker: &str) -> Vec<(ToastLevel, String)> {
    crate::command::bus::testing::take_matching(|c| matches!(c, AppCommand::Toast { text, .. } if text.contains(marker))).into_iter().filter_map(|c| if let AppCommand::Toast { kind, text } = c { Some((kind, text)) } else { None }).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_pending_rename_shows_its_counts_its_source_and_its_reason() {
    let f = fixture().await;
    propose(&f, json!({"kind": "tag", "from": ["beach", "sea"], "to": "dusk", "reason": "same coast"})).await;
    let mut r = Rig::new(&f, 700.0);
    r.until("the card", |r| r.has("Merge tag 'beach', 'sea' into 'dusk'")).await;
    assert!(r.has_containing("2 items change") && r.has_containing("merge"), "{:?}", r.names());
    assert!(r.has_containing("Reason: same coast") && r.has_containing("From tester"), "{:?}", r.names());
    assert!(r.has("Accept") && r.has("Reject") && r.has("Refresh"));
    assert!(r.has_containing("1 rename, 0 edited pictures"), "{:?}", r.names());
}

#[tokio::test(flavor = "multi_thread")]
async fn accepting_asks_for_confirmation_then_applies_through_the_journal() {
    let _serial = crate::command::bus::testing::serial();
    let f = fixture().await;
    let id = propose(&f, json!({"kind": "category", "from": ["landscape"], "to": "scenery"})).await;
    let mut r = Rig::new(&f, 700.0);
    r.until("the card", |r| r.has("Accept")).await;
    r.click("Accept");
    assert!(r.has("Apply to 2 items") && r.has("Cancel"), "{:?}", r.names());
    let before: Vec<Option<String>> = f.db.query("SELECT VALUE category FROM thumbnails WHERE filename = 'beach1.png'").await.unwrap().take(0).unwrap();
    assert_eq!(before[0].as_deref(), Some("landscape"), "nothing is written before the confirmation");
    r.click("Cancel");
    assert!(r.has("Accept") && !r.has("Apply to 2 items"));
    r.click("Accept");
    r.click("Apply to 2 items");
    r.until("the rename to be applied", |r| r.has_containing("2 changes written")).await;
    let after: Vec<Option<String>> = f.db.query("SELECT VALUE category FROM thumbnails WHERE filename = 'beach1.png'").await.unwrap().take(0).unwrap();
    assert_eq!(after[0].as_deref(), Some("scenery"));
    assert_eq!(f.ctx.agent.taxonomy.get(&id).await.unwrap().unwrap().status.as_str(), "accepted");
    assert!(!crate::database::journal::recent_with(&f.db, 10).await.unwrap().is_empty(), "the change is in the journal");
    r.until("the card to go", |r| !r.has("Accept")).await;
    assert!(r.has("Nothing waits for review."), "{:?}", r.names());
    assert!(toasts("changes written").iter().any(|(k, _)| *k == ToastLevel::Success), "a toast reports the change");
}

#[tokio::test(flavor = "multi_thread")]
async fn rejecting_keeps_the_library_as_it_was() {
    let _serial = crate::command::bus::testing::serial();
    let f = fixture().await;
    let id = propose(&f, json!({"kind": "tag", "from": ["forest"], "to": "woods"})).await;
    let mut r = Rig::new(&f, 700.0);
    r.until("the card", |r| r.has("Reject")).await;
    r.click("Reject");
    r.until("the rejection", |r| r.has_containing("Rejected: Rename tag 'forest'")).await;
    assert_eq!(f.ctx.agent.taxonomy.get(&id).await.unwrap().unwrap().status.as_str(), "rejected");
    let tags: Vec<Option<Vec<String>>> = f.db.query("SELECT VALUE tags FROM thumbnails WHERE filename = 'forest1.png'").await.unwrap().take(0).unwrap();
    assert!(tags[0].clone().unwrap().contains(&"forest".to_string()));
    assert!(crate::database::journal::recent_with(&f.db, 10).await.unwrap().is_empty());
    toasts("Rejected");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_rename_that_no_longer_matches_cannot_be_accepted() {
    let f = fixture().await;
    propose(&f, json!({"kind": "tag", "from": ["city"], "to": "town"})).await;
    f.db.query("UPDATE thumbnails SET tags = ['night'] WHERE filename = 'city1.png'").await.unwrap().check().unwrap();
    let mut r = Rig::new(&f, 700.0);
    r.until("the card with the new count", |r| r.has_containing("Nothing to change")).await;
    let accept = text_rects(&r.shapes).into_iter().any(|(t, _)| t == "Accept");
    assert!(accept, "the button is drawn but disabled");
    r.click("Accept");
    assert!(!r.has_containing("Apply to"), "a rename that changes nothing asks for nothing: {:?}", r.names());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_finished_edit_can_be_saved_added_or_discarded() {
    let _serial = crate::command::bus::testing::serial();
    let f = fixture().await;
    let original = std::fs::read(f.path("photos/beach1.png")).unwrap();
    let id = finished_edit(&f, "photos/beach1.png").await;
    let mut r = Rig::new(&f, 700.0);
    r.until("the result card", |r| r.has("Edited copy of beach1.png")).await;
    assert!(r.has_containing("8x8 pixels") && r.has_containing("Prompt: warm") && r.has_containing("The original is untouched"), "{:?}", r.names());
    assert!(r.has("Save as new file") && r.has("Add to library") && r.has("Discard"));
    r.click("Save as new file");
    r.until("the file", |r| r.has_containing("Saved ")).await;
    assert!(f.dir().join("photos/beach1-edited.png").is_file());
    assert_eq!(std::fs::read(f.path("photos/beach1.png")).unwrap(), original, "never over the original");
    r.until("the card to go", |r| !r.has("Edited copy of beach1.png")).await;
    assert_eq!(f.ctx.agent.edits.get(&id).unwrap().review.as_str(), "saved");

    let library_id = finished_edit(&f, "photos/forest1.png").await;
    r.until("the second card", |r| r.has("Edited copy of forest1.png")).await;
    r.click("Add to library");
    r.until("the library row", |r| r.has_containing("added it to the library")).await;
    assert_eq!(f.ctx.agent.edits.get(&library_id).unwrap().review.as_str(), "added");
    assert_eq!(crate::ai::mcp::test_support::row_count(&f.db, "thumbnails").await, 6);

    let third = finished_edit(&f, "photos/city1.png").await;
    r.until("the third card", |r| r.has("Edited copy of city1.png")).await;
    r.click("Discard");
    r.until("the discard", |r| r.has_containing("Rejected the edit of city1.png")).await;
    assert_eq!(f.ctx.agent.edits.get(&third).unwrap().review.as_str(), "rejected");
    assert!(!f.dir().join("photos/city1-edited.png").exists());
    toasts("Saved");
}

#[tokio::test(flavor = "multi_thread")]
async fn without_a_library_accepting_is_disabled_and_the_panel_says_why() {
    let f = fixture().await;
    propose(&f, json!({"kind": "tag", "from": ["beach"], "to": "seaside"})).await;
    let mut r = Rig::new(&f, 700.0);
    r.db_ready = false;
    r.until("the card", |r| r.has("Accept")).await;
    assert!(r.has_containing("The library is not open"));
    r.click("Accept");
    assert!(!r.has_containing("Apply to"), "a disabled button does nothing");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_panel_fits_every_width() {
    let f = fixture().await;
    propose(&f, json!({"kind": "tag", "from": ["beach", "sea", "summer"], "to": "a-rather-long-new-tag-name", "reason": "a long reason that has to wrap on narrow screens without spilling out"})).await;
    finished_edit(&f, "photos/beach2.png").await;
    for width in [280.0, 360.0, 700.0, 1000.0, 1600.0] {
        let mut r = Rig::new(&f, width);
        r.until("both cards", |r| r.has_containing("Merge tag") && r.has_containing("Edited copy")).await;
        r.frame();
        for (text, rect) in text_rects(&r.shapes) {
            assert!(rect.left() >= -0.5 && rect.right() <= width + 0.5, "{text:?} spills out at {width}: {rect:?}");
        }
    }
}

#[test]
fn a_panel_without_a_runtime_draws_and_stays_idle() {
    let ctx = test_ctx();
    let mut panel = Panel::default();
    for _ in 0..2 {
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(700.0, 800.0))), ..Default::default() };
        let full = ctx.run_ui(input, |ui| {
            CentralPanel::default().show(ui, |ui| panel.ui(ui, true));
        });
        let drawn: Vec<String> = text_rects(&full.shapes).into_iter().map(|t| t.0).collect();
        assert!(drawn.iter().any(|t| t == "Agent proposals"), "{drawn:?}");
        full.drop_without_applying_deltas();
    }
    assert!(!panel.working);
}
