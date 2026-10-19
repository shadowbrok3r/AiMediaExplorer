//! The connection card: gateway address, pairing, agent, health, a test message, tools and privacy. Settings > Assistant and the first-run wizard both draw it.

use std::sync::{Arc, Mutex};

use egui::{ComboBox, Context, Id, RichText, TextEdit, Ui};
use zeroclaw_client::{CompleteRequest, CompleteResponse};

use super::events::Pending;
use super::import::{self, Imported};
use super::profile::{BackendPref, ZcProfile};
use super::runtime::ZcRuntime;
use super::{card_tools, save_profile_with, widgets};
use crate::icons::{self, svg};

/// How much of the card is drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Everything, for the settings page.
    Settings,
    /// Setup steps only, for the first-run wizard; tools and privacy stay in Settings.
    Wizard,
}

/// Text of the one-word test message.
const TEST_PROMPT: &str = "Reply with the single word: pong";

#[derive(Default)]
pub(super) struct Card {
    /// Profile being edited; `None` until the card first draws.
    pub draft: Option<ZcProfile>,
    /// The runtime's profile when the draft was loaded, to tell edits from outside changes.
    base: Option<ZcProfile>,
    pub code: String,
    pub agents: Vec<String>,
    detect: Option<Pending<Result<Vec<String>, String>>>,
    pair: Option<Pending<Result<(), String>>>,
    test: Option<Pending<Result<CompleteResponse, String>>>,
    pub outcome: Option<(bool, String)>,
    /// What the ZC Codex config offers; `None` before the first look.
    candidate: Option<Option<Imported>>,
    pub images_confirmed: bool,
    pub skipped: bool,
    pub tools: card_tools::ToolsDraft,
}

fn state(ctx: &Context) -> Arc<Mutex<Card>> {
    let id = Id::unique("ame_zc_card_state");
    ctx.data_mut(|d| d.get_temp_mut_or_insert_with::<Arc<Mutex<Card>>>(id, Default::default).clone())
}

impl Card {
    fn poll(&mut self, rt: &ZcRuntime) {
        if let Some(result) = self.detect.as_mut().and_then(Pending::take) {
            self.detect = None;
            match result {
                Ok(agents) => {
                    self.outcome = Some((true, format!("The gateway offers {} agents", agents.len())));
                    if let (Some(d), Some(first)) = (self.draft.as_mut(), agents.first()) {
                        if d.agent.trim().is_empty() {
                            d.agent = first.clone();
                        }
                    }
                    self.agents = agents;
                }
                Err(e) => self.outcome = Some((false, e)),
            }
        }
        if let Some(result) = self.pair.as_mut().and_then(Pending::take) {
            self.pair = None;
            match result {
                Ok(()) => {
                    if let Some(d) = self.draft.clone() {
                        self.apply(rt, d);
                    }
                    self.code.clear();
                    self.outcome = Some((true, "Paired. The token is stored.".to_string()));
                    self.detect_agents(rt);
                }
                Err(e) => self.outcome = Some((false, e)),
            }
        }
        if let Some(result) = self.test.as_mut().and_then(Pending::take) {
            self.test = None;
            self.outcome = Some(match result {
                Ok(reply) => (true, format!("The gateway answered ({}): {}", reply.model, reply.response.trim())),
                Err(e) => (false, e),
            });
        }
    }

    fn apply(&mut self, rt: &ZcRuntime, profile: ZcProfile) {
        save_profile_with(rt, profile.clone());
        self.base = Some(rt.profile());
        self.draft = Some(profile);
    }

    fn detect_agents(&mut self, rt: &ZcRuntime) {
        if let Some(d) = &self.draft {
            self.detect = Some(rt.detect_agents(&d.host, None));
        }
    }

    fn busy(&self) -> bool {
        self.detect.is_some() || self.pair.is_some() || self.test.is_some()
    }
}

fn labelled(ui: &mut Ui, label: &str, add: impl FnOnce(&mut Ui)) {
    ui.label(RichText::new(label).weak());
    add(ui);
    ui.add_space(4.0);
}

fn outcome_row(ui: &mut Ui, outcome: &Option<(bool, String)>) {
    if let Some((ok, text)) = outcome {
        let color = if *ok { ui.visuals().text_color() } else { ui.visuals().error_fg_color };
        ui.label(RichText::new(text).color(color));
    }
}

/// Draws the card; true once a wizard step is complete (the gateway is online) or was skipped.
pub fn show(ui: &mut Ui, rt: &Arc<ZcRuntime>, mode: Mode) -> bool {
    rt.attach_ctx(ui.ctx());
    let shared = state(ui.ctx());
    let mut card = shared.lock().unwrap_or_else(|e| e.into_inner());
    let live = rt.profile();
    if card.draft.is_none() || (card.draft == card.base && card.base.as_ref() != Some(&live)) {
        card.draft = Some(live.clone());
        card.base = Some(live.clone());
    }
    if card.candidate.is_none() {
        card.candidate = Some(import::default_path().and_then(|p| import::import_from(&p).ok()));
    }
    card.poll(rt);
    if card.busy() {
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
    }

    let health = rt.health();
    widgets::health_row(ui, &health);
    ui.add_space(6.0);

    let mut draft = card.draft.clone().unwrap_or_default();
    let before = draft.clone();
    if mode == Mode::Settings {
        labelled(ui, "Assistant backend", |ui| {
            let mut pref = draft.backend_pref();
            ComboBox::from_id_salt("zc_backend").selected_text(pref.label()).width(ui.available_width().min(260.0)).show_ui(ui, |ui| {
                for b in BackendPref::ALL {
                    ui.selectable_value(&mut pref, b, b.label());
                }
            });
            draft.backend = pref.as_str().to_string();
        });
        labelled(ui, "Name", |ui| {
            ui.add(TextEdit::singleline(&mut draft.name).desired_width(ui.available_width()));
        });
    }

    if let Some(Some(found)) = card.candidate.clone() {
        if !rt.has_token() || !draft.is_configured() {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new(format!("Found {} at {} in the ZC Codex config.", found.name, found.host)).weak());
                if icons::button(ui, svg::DOWNLOAD_SIMPLE, "Use it").clicked() {
                    draft.name = found.name.clone();
                    draft.host = found.host.clone();
                    draft.agent = found.agent.clone();
                    match rt.save_token(found.token()) {
                        Ok(()) => card.outcome = Some((true, format!("Imported {}. The token is stored and not shown.", found.name))),
                        Err(e) => card.outcome = Some((false, e)),
                    }
                    card.agents.clear();
                    card.apply(rt, draft.clone());
                    card.detect_agents(rt);
                }
            });
            ui.add_space(4.0);
        }
    }

    labelled(ui, "Gateway address", |ui| {
        ui.add(TextEdit::singleline(&mut draft.host).hint_text("100.96.121.37 or wss://claw.example.app").desired_width(ui.available_width()));
    });
    labelled(ui, "Agent alias", |ui| {
        ui.horizontal_wrapped(|ui| {
            let width = (ui.available_width() - 110.0).clamp(100.0, 260.0);
            ui.add(TextEdit::singleline(&mut draft.agent).hint_text("media").desired_width(width));
            if !card.agents.is_empty() {
                ComboBox::from_id_salt("zc_agents").selected_text("Agents").width(90.0).show_ui(ui, |ui| {
                    for agent in card.agents.clone() {
                        ui.selectable_value(&mut draft.agent, agent.clone(), agent);
                    }
                });
            }
        });
    });

    labelled(ui, "Pairing code", |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.add(TextEdit::singleline(&mut card.code).hint_text("one-time code").desired_width(150.0));
            let can_pair = !card.code.trim().is_empty() && !draft.host.trim().is_empty() && card.pair.is_none();
            if ui.add_enabled(can_pair, icons::button_widget(ui, svg::LINK, "Pair")).clicked() {
                card.pair = Some(rt.pair(&draft.host, &card.code));
                card.outcome = None;
            }
        });
    });
    ui.label(RichText::new(if rt.has_token() { "A token is stored (never shown)." } else { "No token yet." }).weak());
    ui.add_space(4.0);

    ui.horizontal_wrapped(|ui| {
        let dirty = card.draft.as_ref() != card.base.as_ref() || draft != before;
        if ui.add_enabled(dirty, icons::button_widget(ui, svg::FLOPPY_DISK, "Apply")).clicked() {
            card.apply(rt, draft.clone());
        }
        if icons::button(ui, svg::PULSE, "Check").clicked() {
            if draft != live {
                card.apply(rt, draft.clone());
            }
            rt.check();
        }
        if ui.add_enabled(card.detect.is_none() && rt.has_token(), icons::button_widget(ui, svg::LIST_MAGNIFYING_GLASS, "Detect agents")).clicked() {
            card.detect_agents(rt);
        }
        let can_test = health.is_online() && card.test.is_none();
        if ui.add_enabled(can_test, icons::button_widget(ui, svg::PAPER_PLANE_RIGHT, "Send test message")).clicked() {
            card.test = Some(rt.complete(CompleteRequest::new(TEST_PROMPT).with_hint(draft.hint())));
            card.outcome = None;
        }
        if rt.has_token() && ui.button("Forget token").on_hover_text("Delete the stored token").clicked() {
            match rt.clear_token() {
                Ok(()) => card.outcome = Some((true, "The token was deleted.".to_string())),
                Err(e) => card.outcome = Some((false, e)),
            }
        }
    });
    if card.busy() {
        ui.label(RichText::new("Working...").weak());
    }
    outcome_row(ui, &card.outcome);
    card.draft = Some(draft.clone());

    if mode == Mode::Settings {
        ui.add_space(6.0);
        ui.separator();
        card_tools::tools_section(ui, &mut card, &mut draft);
        ui.add_space(4.0);
        card_tools::privacy_section(ui, &mut card, &mut draft);
        if card.draft.as_ref() != Some(&draft) {
            card.draft = Some(draft.clone());
        }
        return health.is_online();
    }

    ui.add_space(6.0);
    let done = health.is_online() || card.skipped;
    ui.horizontal(|ui| {
        if !health.is_online() && ui.button("Skip for now").clicked() {
            card.skipped = true;
        }
        if health.is_online() {
            ui.label(RichText::new("The assistant is ready.").strong());
        }
    });
    done
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::test_support::{press_release, test_ctx};
    use crate::zc::runtime_tests::{runtime_for, temp_dir};
    use crate::zc::secrets::TokenStore;
    use egui::{CentralPanel, Event, Pos2, RawInput, Rect, vec2};
    use std::time::Duration;
    use zeroclaw_client::mock::{MockGateway, MockOptions};

    struct Rig {
        ctx: Context,
        rt: Arc<ZcRuntime>,
        shapes: Vec<egui::epaint::ClippedShape>,
        mode: Mode,
        width: f32,
        done: bool,
    }

    impl Rig {
        fn new(rt: Arc<ZcRuntime>, mode: Mode, width: f32) -> Self {
            Self { ctx: test_ctx(), rt, shapes: Vec::new(), mode, width, done: false }
        }

        fn frame_with(&mut self, events: Vec<Event>) {
            let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(self.width, 900.0))), events, ..Default::default() };
            let (rt, mode) = (self.rt.clone(), self.mode);
            let mut done = false;
            let full = self.ctx.run_ui(input, |ui| {
                CentralPanel::default().show(ui, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| done = show(ui, &rt, mode));
                });
            });
            self.shapes = full.shapes.clone();
            self.done = done;
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

        fn names(&self) -> Vec<String> {
            self.texts().into_iter().map(|t| t.0).collect()
        }

        fn has(&self, text: &str) -> bool {
            self.texts().iter().any(|(t, _)| t == text)
        }

        fn has_containing(&self, part: &str) -> bool {
            self.texts().iter().any(|(t, _)| t.contains(part))
        }

        fn click(&mut self, text: &str) {
            let rect = self.texts().into_iter().find(|(t, _)| t == text).map(|(_, r)| r).unwrap_or_else(|| panic!("no {text:?}: {:?}", self.names()));
            self.frame_with(press_release(rect.center()));
            self.frame();
        }

        async fn until(&mut self, what: &str, mut cond: impl FnMut(&mut Rig) -> bool) {
            let end = std::time::Instant::now() + Duration::from_secs(8);
            while std::time::Instant::now() < end {
                self.frame();
                if cond(self) {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            panic!("timed out waiting for {what}: {:?}", self.names());
        }

        fn with_card<R>(&self, f: impl FnOnce(&mut Card) -> R) -> R {
            let shared = state(&self.ctx);
            let mut card = shared.lock().unwrap_or_else(|e| e.into_inner());
            f(&mut card)
        }
    }

    fn fresh_runtime(name: &str) -> Arc<ZcRuntime> {
        ZcRuntime::new(tokio::runtime::Handle::current(), Some(TokenStore::at(temp_dir(name))))
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn the_settings_card_draws_every_section() {
        let mut r = Rig::new(fresh_runtime("card-sections"), Mode::Settings, 520.0);
        r.frame();
        for expected in ["Not set up", "Assistant backend", "Gateway address", "Agent alias", "Pairing code", "Tools server", "Privacy", "Apply", "Check", "Detect agents", "Send test message", "Pair"] {
            assert!(r.has(expected), "{expected} in {:?}", r.names());
        }
        assert!(r.has_containing("No token yet"));
        assert!(r.has_containing("/mcp"), "the tools server shows its address: {:?}", r.names());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn pairing_through_the_card_stores_the_token_and_the_gateway_comes_online() {
        let mock = MockGateway::start_with(MockOptions { agents: vec!["media".into()], ..Default::default() }).await;
        let rt = fresh_runtime("card-pair");
        let mut r = Rig::new(rt.clone(), Mode::Settings, 520.0);
        r.frame();
        r.with_card(|c| {
            c.draft.as_mut().unwrap().host = mock.url();
            c.code = "123456".into();
        });
        r.frame();
        r.click("Pair");
        r.until("the pairing to finish", |r| rt.has_token() && r.has_containing("offers 1 agents")).await;
        assert!(r.has_containing("A token is stored (never shown)."));
        let agent = r.with_card(|c| c.draft.as_ref().unwrap().agent.clone());
        assert_eq!(agent, "media", "the first agent fills an empty alias");
        r.click("Apply");
        r.until("online", |r| r.rt.health().is_online() && r.has("Online")).await;
        assert!(rt.profile().is_configured());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn the_test_message_goes_through_complete_and_shows_the_answer() {
        let mock = MockGateway::start().await;
        mock.set_complete_reply("pong");
        let rt = runtime_for(&mock, "default", "card-test");
        let mut r = Rig::new(rt, Mode::Settings, 520.0);
        r.until("online", |r| r.has("Online")).await;
        r.click("Send test message");
        r.until("the answer", |r| r.has_containing("The gateway answered") && r.has_containing("pong")).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_codex_config_is_offered_and_importing_stores_the_token_without_showing_it() {
        let dir = temp_dir("card-import");
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("config.json");
        std::fs::write(&file, include_str!("fixtures/zc-codex-config.json")).unwrap();
        import::test_path::set(Some(file));
        let rt = fresh_runtime("card-import-rt");
        let mut r = Rig::new(rt.clone(), Mode::Settings, 520.0);
        r.frame();
        assert!(r.has_containing("Found claw (public) at wss://claw.example.test"), "{:?}", r.names());
        r.click("Use it");
        r.frame();
        import::test_path::set(None);
        assert!(rt.has_token());
        assert_eq!(rt.profile().host, "wss://claw.example.test");
        assert_eq!(rt.profile().agent, "media");
        assert!(!r.names().iter().any(|t| t.contains("fixture-token")), "the token is never drawn");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn pictures_for_a_gateway_outside_the_network_need_a_second_confirmation() {
        let rt = fresh_runtime("card-privacy");
        rt.set_profile(ZcProfile { host: "wss://claw.example.app".into(), agent: "media".into(), ..Default::default() });
        let mut r = Rig::new(rt, Mode::Settings, 520.0);
        r.frame();
        assert!(r.has_containing("outside your network"), "{:?}", r.names());
        r.click("Allow images in chat messages");
        assert!(!r.with_card(|c| c.draft.as_ref().unwrap().allow_images), "not without the confirmation");
        r.click("I understand, allow pictures anyway");
        r.click("Allow images in chat messages");
        assert!(r.with_card(|c| c.draft.as_ref().unwrap().allow_images));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn pictures_for_a_local_gateway_need_only_the_checkbox() {
        let rt = fresh_runtime("card-privacy-local");
        rt.set_profile(ZcProfile { host: "192.168.1.20".into(), agent: "media".into(), ..Default::default() });
        let mut r = Rig::new(rt, Mode::Settings, 520.0);
        r.frame();
        assert!(!r.has_containing("outside your network"));
        r.click("Allow images in chat messages");
        assert!(r.with_card(|c| c.draft.as_ref().unwrap().allow_images));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn the_wizard_step_is_compact_and_completes_when_online_or_skipped() {
        let mock = MockGateway::start().await;
        let mut r = Rig::new(fresh_runtime("card-wizard"), Mode::Wizard, 360.0);
        r.frame();
        assert!(!r.has("Tools server") && !r.has("Privacy"), "tools and privacy stay in Settings");
        assert!(r.has("Skip for now") && !r.done);
        r.click("Skip for now");
        assert!(r.done, "skipping completes the step");
        let ok = runtime_for(&mock, "default", "card-wizard-ok");
        let mut r2 = Rig::new(ok, Mode::Wizard, 360.0);
        r2.until("the step to complete", |r| r.done).await;
        assert!(r2.has("The assistant is ready."));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn the_card_fits_the_narrowest_widths() {
        for width in [280.0, 360.0, 520.0] {
            let mut r = Rig::new(fresh_runtime(&format!("card-width-{}", width as u32)), Mode::Settings, width);
            r.frame();
            r.frame();
            for (text, rect) in r.texts() {
                assert!(rect.left() >= -0.5 && rect.right() <= width + 0.5, "{text:?} spills out at {width}: {rect:?}");
            }
        }
    }
}
