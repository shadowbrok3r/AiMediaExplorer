//! The library guide and the gateway configuration snippet sections of the connection card.

use std::sync::{Arc, Mutex};

use egui::{CollapsingHeader, Context, Id, RichText, ScrollArea, TextEdit, Ui};

use super::events::{Pending, now_ms};
use super::memory::{self, Pushed, Remote};
use super::profile::ZcProfile;
use super::runtime::ZcRuntime;
use super::snippet::{self, Ping};
use super::widgets::ago;
use crate::icons::{self, svg};
use crate::settings::SettingsStore;

/// Tallest the guide editor and the snippet grow before they scroll.
const EDITOR_HEIGHT: f32 = 220.0;

/// Edits and running jobs of the two sections.
#[derive(Default)]
pub struct State {
    /// The guide text in the editor; `None` until the section first draws.
    draft: Option<String>,
    /// The saved text the draft was loaded from.
    base: String,
    push: Option<Pending<Result<Pushed, String>>>,
    verify: Option<Pending<Result<Option<Remote>, String>>>,
    test: Option<Pending<Ping>>,
    guide_note: Option<(bool, String)>,
    ping_note: Option<(bool, String)>,
}

fn state(ctx: &Context) -> Arc<Mutex<State>> {
    let id = Id::new("ame_zc_card_memory_state");
    ctx.data_mut(|d| d.get_temp_mut_or_insert_with::<Arc<Mutex<State>>>(id, Default::default).clone())
}

/// Local date and time of a Unix-millisecond stamp.
fn when(at_ms: u64) -> String {
    chrono::DateTime::<chrono::Utc>::from_timestamp_millis(at_ms as i64).map(|t| t.with_timezone(&chrono::Local).format("%Y-%m-%d %H:%M").to_string()).unwrap_or_default()
}

impl State {
    fn poll(&mut self) {
        if let Some(result) = self.push.as_mut().and_then(Pending::take) {
            self.push = None;
            self.guide_note = Some(match result {
                Ok(p) => (true, format!("Pushed {} characters to the gateway memory as {}.", p.bytes, p.key)),
                Err(e) => (false, format!("The gateway did not take the guide: {e}")),
            });
        }
        if let Some(result) = self.verify.as_mut().and_then(Pending::take) {
            self.verify = None;
            self.guide_note = Some(match result {
                Ok(Some(r)) if r.same_as_local => (true, format!("The gateway holds the same guide ({} characters).", r.bytes)),
                Ok(Some(r)) => (false, format!("The gateway holds a different guide ({} characters). Push to replace it.", r.bytes)),
                Ok(None) => (false, "The gateway holds no guide yet.".to_string()),
                Err(e) => (false, format!("The gateway could not be asked: {e}")),
            });
        }
        if let Some(result) = self.test.as_mut().and_then(Pending::take) {
            self.test = None;
            self.ping_note = Some((result.ok, format!("{} ({} ms)", result.detail, result.millis)));
        }
    }

    fn busy(&self) -> bool {
        self.push.is_some() || self.verify.is_some() || self.test.is_some()
    }
}

fn note_row(ui: &mut Ui, note: &Option<(bool, String)>) {
    if let Some((ok, text)) = note {
        let color = if *ok { ui.visuals().text_color() } else { ui.visuals().error_fg_color };
        ui.label(RichText::new(text).color(color));
    }
}

/// Draws the guide and the snippet sections for `profile`.
pub fn section(ui: &mut Ui, rt: &Arc<ZcRuntime>, profile: &ZcProfile) {
    section_in(ui, rt, profile, crate::settings::store());
}

/// [`section`] over the guide kept in `store`.
pub fn section_in(ui: &mut Ui, rt: &Arc<ZcRuntime>, profile: &ZcProfile, store: &Arc<SettingsStore>) {
    let shared = state(ui.ctx());
    let mut st = shared.lock().unwrap_or_else(|e| e.into_inner());
    st.poll();
    if st.busy() {
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
    }
    guide_section(ui, rt, store, &mut st);
    ui.add_space(4.0);
    snippet_section(ui, rt, profile, &mut st);
}

fn guide_section(ui: &mut Ui, rt: &Arc<ZcRuntime>, store: &Arc<SettingsStore>, st: &mut State) {
    CollapsingHeader::new(RichText::new("Library guide").strong()).default_open(false).show(ui, |ui| {
        ui.label(RichText::new("Notes for the agent: naming, preferred tags and categories, how to treat sensitive pictures. It reads them as the resource library://guide. Nothing is sent to the gateway memory until you press Push.").weak());
        let guide = memory::guide_in(store);
        let saved = guide.effective().to_string();
        if st.draft.is_none() || (st.draft.as_deref() == Some(st.base.as_str()) && st.base != saved) {
            st.draft = Some(saved.clone());
            st.base = saved.clone();
        }
        let mut draft = st.draft.clone().unwrap_or_default();
        ScrollArea::vertical().id_salt("zc-guide-scroll").max_height(EDITOR_HEIGHT).min_scrolled_height(EDITOR_HEIGHT).auto_shrink([false, false]).show(ui, |ui| {
            ui.add(TextEdit::multiline(&mut draft).desired_rows(8).desired_width(ui.available_width()).id_salt("zc-guide-edit"));
        });
        let unsaved = draft != saved;
        let online = rt.health().is_online();
        ui.horizontal_wrapped(|ui| {
            if ui.add_enabled(unsaved, icons::button_widget(ui, svg::FLOPPY_DISK, "Save guide")).clicked() {
                memory::save_text_in(store, &draft);
                st.base = memory::guide_in(store).effective().to_string();
                draft = st.base.clone();
                st.guide_note = Some((true, "Saved. The agent reads the new text the next time it opens the resource.".to_string()));
            }
            if ui.add_enabled(!guide.is_default() || unsaved, icons::button_widget(ui, svg::ARROW_CLOCKWISE, "Reset to default")).clicked() {
                memory::reset_text_in(store);
                st.base = memory::guide_in(store).effective().to_string();
                draft = st.base.clone();
                st.guide_note = Some((true, "The shipped guide is back.".to_string()));
            }
        });
        ui.horizontal_wrapped(|ui| {
            let can_push = online && !unsaved && st.push.is_none();
            let hint = if unsaved { "Save the guide first" } else if !online { "The gateway is not online" } else { "Store the guide in the gateway memory of this agent" };
            if ui.add_enabled(can_push, icons::button_widget(ui, svg::UPLOAD_SIMPLE, "Push to gateway")).on_hover_text(hint).clicked() {
                st.push = Some(memory::push_guide(rt, store.clone()));
                st.guide_note = None;
            }
            if ui.add_enabled(online && st.verify.is_none(), icons::button_widget(ui, svg::PULSE, "Check gateway copy")).clicked() {
                st.verify = Some(memory::remote(rt, saved.clone()));
                st.guide_note = None;
            }
        });
        let fresh = memory::guide_in(store);
        match fresh.pushed_ms {
            None => {
                ui.label(RichText::new("Never pushed.").weak());
            }
            Some(at) => {
                let text = format!("Last pushed {} ({}).", when(at), ago(now_ms(), at));
                ui.label(RichText::new(text).weak());
                if fresh.changed_since_push() {
                    ui.label(RichText::new("The guide changed since then.").color(ui.visuals().warn_fg_color));
                }
            }
        }
        if st.busy() && (st.push.is_some() || st.verify.is_some()) {
            ui.label(RichText::new("Working...").weak());
        }
        note_row(ui, &st.guide_note);
        st.draft = Some(draft);
    });
}

fn snippet_section(ui: &mut Ui, rt: &Arc<ZcRuntime>, profile: &ZcProfile, st: &mut State) {
    CollapsingHeader::new(RichText::new("Gateway configuration").strong()).default_open(false).show(ui, |ui| {
        ui.label(RichText::new("Add this to the gateway so its agent can use the tools server. The token is a placeholder: paste it from the token file. This application never changes the gateway configuration.").weak());
        let text = snippet::build(profile);
        let mut shown = text.as_str();
        ScrollArea::vertical().id_salt("zc-snippet-scroll").max_height(EDITOR_HEIGHT).min_scrolled_height(EDITOR_HEIGHT).auto_shrink([false, false]).show(ui, |ui| {
            ui.add(TextEdit::multiline(&mut shown).code_editor().desired_rows(8).desired_width(ui.available_width()).id_salt("zc-snippet"));
        });
        let running = super::tools::status().running;
        ui.horizontal_wrapped(|ui| {
            if icons::button(ui, svg::COPY, "Copy snippet").clicked() {
                ui.ctx().copy_text(text.clone());
                st.ping_note = Some((true, "Copied. Paste it into the gateway configuration and fill in the token.".to_string()));
            }
            let hint = if running { "Calls util_ping through the tools server address" } else { "Start the tools server first" };
            if ui.add_enabled(running && st.test.is_none(), icons::button_widget(ui, svg::PULSE, "Test tools server")).on_hover_text(hint).clicked() {
                st.test = Some(snippet::test(rt));
                st.ping_note = None;
            }
        });
        if st.test.is_some() {
            ui.label(RichText::new("Calling util_ping...").weak());
        }
        note_row(ui, &st.ping_note);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::test_support::{press_release, test_ctx};
    use crate::zc::runtime_tests::runtime_for;
    use egui::{CentralPanel, Event, Pos2, RawInput, Rect, vec2};
    use std::time::Duration;
    use zeroclaw_client::mock::MockGateway;

    struct Nop;

    #[async_trait::async_trait]
    impl crate::settings::Backend for Nop {
        async fn load(&self) -> anyhow::Result<crate::settings::Loaded> {
            Ok(crate::settings::Loaded { settings: crate::UiSettings::default(), existed: true })
        }
        async fn merge(&self, _: &[&'static str], _: &crate::UiSettings) -> anyhow::Result<()> {
            Ok(())
        }
    }

    fn private_store() -> Arc<SettingsStore> {
        let store = SettingsStore::with_debounce(Arc::new(Nop), Duration::from_secs(3600));
        store.apply_loaded("test", crate::settings::Loaded { settings: crate::UiSettings::default(), existed: true });
        store
    }

    struct Rig {
        ctx: Context,
        rt: Arc<ZcRuntime>,
        profile: ZcProfile,
        store: Arc<SettingsStore>,
        shapes: Vec<egui::epaint::ClippedShape>,
        width: f32,
    }

    impl Rig {
        fn new(rt: Arc<ZcRuntime>, width: f32) -> Self {
            let profile = rt.profile();
            Self { ctx: test_ctx(), rt, profile, store: private_store(), shapes: Vec::new(), width }
        }

        fn frame_with(&mut self, events: Vec<Event>) {
            let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(self.width, 1600.0))), events, ..Default::default() };
            let (rt, profile, store) = (self.rt.clone(), self.profile.clone(), self.store.clone());
            let full = self.ctx.run_ui(input, |ui| {
                CentralPanel::default().show(ui, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| section_in(ui, &rt, &profile, &store));
                });
            });
            self.shapes = full.shapes.clone();
            full.drop_without_applying_deltas();
        }

        fn frame(&mut self) {
            self.frame_with(Vec::new());
        }

        fn texts(&self) -> Vec<(String, egui::Rect)> {
            fn walk(shape: &egui::Shape, out: &mut Vec<(String, egui::Rect)>) {
                match shape {
                    egui::Shape::Text(t) => out.push((t.galley.text().to_string(), t.galley.rect.translate(t.pos.to_vec2()))),
                    egui::Shape::Vec(v) => v.iter().for_each(|s| walk(s, out)),
                    _ => {}
                }
            }
            let mut out = Vec::new();
            self.shapes.iter().for_each(|s| walk(&s.shape, &mut out));
            out
        }

        fn has_containing(&self, part: &str) -> bool {
            self.texts().iter().any(|(t, _)| t.contains(part))
        }

        fn click(&mut self, text: &str) {
            let rect = self.texts().into_iter().find(|(t, _)| t == text).map(|(_, r)| r).unwrap_or_else(|| panic!("no {text:?}: {:?}", self.texts().into_iter().map(|t| t.0).collect::<Vec<_>>()));
            self.frame_with(press_release(rect.center()));
            self.frame();
        }

        fn open_headers(&mut self) {
            self.frame();
            for header in ["Library guide", "Gateway configuration"] {
                self.click(header);
            }
            for _ in 0..12 {
                self.frame();
            }
        }
    }

    #[test]
    fn the_sections_are_closed_until_opened_and_then_show_their_controls() {
        let rt = ZcRuntime::new(tokio::runtime::Builder::new_current_thread().build().unwrap().handle().clone(), None);
        let mut r = Rig::new(rt, 520.0);
        r.frame();
        assert!(r.has_containing("Library guide") && r.has_containing("Gateway configuration"));
        assert!(!r.has_containing("Push to gateway"), "closed sections draw only their headers");
        r.open_headers();
        for control in ["Save guide", "Reset to default", "Push to gateway", "Check gateway copy", "Copy snippet", "Test tools server"] {
            assert!(r.has_containing(control), "{control}: {:?}", r.texts().into_iter().map(|t| t.0).collect::<Vec<_>>());
        }
        assert!(r.has_containing("Never pushed."));
        assert!(r.has_containing("[[mcp.servers]]"), "the snippet is drawn");
        assert!(r.has_containing("Bearer <paste the token"), "with a placeholder for the token");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn the_snippet_names_the_agent_and_never_the_token() {
        let mock = MockGateway::start().await;
        let rt = runtime_for(&mock, "photos", "card-memory-snippet");
        let mut r = Rig::new(rt, 520.0);
        r.open_headers();
        assert!(r.has_containing("[agents.photos]"), "{:?}", r.texts().into_iter().map(|t| t.0).collect::<Vec<_>>());
        assert!(!r.texts().iter().any(|(t, _)| t.contains(mock.token().unwrap())), "the gateway token is not drawn");
        if let Ok(real) = crate::ai::mcp::hub().token() {
            assert!(!r.texts().iter().any(|(t, _)| t.contains(&real)), "the tools token is not drawn");
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn pushing_needs_an_online_gateway_and_a_saved_guide_and_shows_the_time() {
        let mock = MockGateway::start().await;
        let rt = runtime_for(&mock, "default", "card-memory-push");
        let mut r = Rig::new(rt.clone(), 520.0);
        r.open_headers();
        let end = std::time::Instant::now() + Duration::from_secs(8);
        while !rt.health().is_online() && std::time::Instant::now() < end {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(rt.health().is_online());
        r.frame();
        r.click("Push to gateway");
        let end = std::time::Instant::now() + Duration::from_secs(8);
        while !r.has_containing("Pushed ") && std::time::Instant::now() < end {
            tokio::time::sleep(Duration::from_millis(20)).await;
            r.frame();
        }
        assert!(r.has_containing("Pushed "), "{:?}", r.texts().into_iter().map(|t| t.0).collect::<Vec<_>>());
        assert_eq!(mock.memory().len(), 1);
        assert_eq!(mock.memory()[0]["key"], crate::ai::mcp::guide::MEMORY_KEY);
        r.frame();
        assert!(r.has_containing("Last pushed "), "the time of the push shows");
        assert!(memory::guide_in(&r.store).pushed_ms.is_some());
    }

    #[test]
    fn the_sections_fit_the_narrowest_widths() {
        for width in [280.0, 360.0, 520.0, 700.0, 1000.0, 1600.0] {
            let rt = ZcRuntime::new(tokio::runtime::Builder::new_current_thread().build().unwrap().handle().clone(), None);
            let mut r = Rig::new(rt, width);
            r.open_headers();
            r.frame();
            for (text, rect) in r.texts() {
                assert!(rect.left() >= -0.5 && rect.right() <= width + 0.5, "{text:?} spills out at {width}: {rect:?}");
            }
        }
    }
}
