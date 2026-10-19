//! The assistant: chat over the gateway socket, a direct provider or local models, in the right rail or its own window.

pub mod api;
pub mod approvals;
pub mod backend;
pub mod composer;
pub mod context;
pub mod direct;
pub mod emit;
pub mod entry;
mod flow;
pub mod header;
mod images;
mod layout;
pub mod local;
pub mod providers;
pub mod sessions;
mod shortcuts;
pub mod tools_debug;
pub mod transcript;

#[cfg(test)]
mod tests;

use std::sync::Arc;
use std::time::Instant;

use crossbeam::channel::{Receiver, Sender};
use egui::Context;

use self::backend::{BackendKind, DirectState, Resolved, TurnRequest, ZeroClawBackend};
use self::composer::Composer;
use self::context::ContextInput;
use self::direct::DirectBackend;
use self::entry::Transcript;
use self::local::LocalBackend;
use self::providers::ModelBrowser;
use self::sessions::Sessions;
use self::tools_debug::ToolsDebug;
use crate::UiSettings;
use crate::ui::file_table::AIUpdate;
use crate::zc::events::{Pending, Tagged, now_ms};
use crate::zc::health::Health;
use crate::zc::profile::ZcProfile;
use crate::zc::runtime::ZcRuntime;
use zeroclaw_client::CompleteResponse;

pub use api::ATTACH_REQUESTS;
pub use layout::PanelOut;
pub use shortcuts::{handle_shortcuts, show_assistant};

/// Request to attach file paths to the chat. Safe to call from any UI code.
pub fn request_attach_to_chat(paths: Vec<String>) {
    if paths.is_empty() {
        return;
    }
    crate::command::send(crate::command::AppCommand::AttachToAssistant { paths });
}

/// How long a stopped turn waits for the gateway to confirm before the panel ends it itself.
const STOP_GRACE: std::time::Duration = std::time::Duration::from_secs(4);

pub struct AssistantPanel {
    pub(crate) rt: Arc<ZcRuntime>,
    pub sessions: Sessions,
    pub transcript: Transcript,
    pub composer: Composer,
    pub(crate) view: transcript::View,
    /// Connection number of the gateway session this panel listens to.
    pub(crate) epoch: u64,
    /// Number of the newest direct or local turn; events of older ones are stale.
    pub(crate) local_epoch: u64,
    pub(crate) local_turn: u64,
    pub(crate) local_tx: Sender<Tagged>,
    pub(crate) local_rx: Receiver<Tagged>,
    pub(crate) zc: ZeroClawBackend,
    pub(crate) direct_backend: DirectBackend,
    pub(crate) local_backend: LocalBackend,
    /// The backend of the turn that is running.
    pub(crate) running_on: Option<BackendKind>,
    pub(crate) settings: UiSettings,
    pub(crate) settings_gen: u64,
    pub(crate) profile: ZcProfile,
    pub(crate) health: Health,
    pub(crate) direct: DirectState,
    pub resolved: Resolved,
    pub(crate) input: ContextInput,
    pub temp: f32,
    pub model_override: Option<String>,
    pub(crate) models: ModelBrowser,
    pub(crate) tools_dbg: ToolsDebug,
    /// A turn built but waiting for the chat socket to open.
    pub(crate) queued: Option<TurnRequest>,
    /// The send waits for the session to exist.
    pub(crate) send_when_created: bool,
    /// The session now exists: send what is in the composer on the next draw.
    pub(crate) pending_send: bool,
    pub(crate) title_job: Option<(String, Pending<Result<CompleteResponse, String>>)>,
    /// Session and health state a connection was last tried for, so a failure is not retried every frame.
    pub(crate) connect_tried: Option<(String, crate::zc::health::HealthKind)>,
    pub(crate) stopping_since: Option<Instant>,
    /// The person dismissed the setup card.
    pub(crate) setup_skipped: bool,
    pub(crate) ai_update_tx: Option<Sender<AIUpdate>>,
    /// Events dropped because they belonged to another session or connection.
    pub stale_dropped: u64,
    /// Read the application's mailboxes (attachments, queued messages); tests turn this off.
    pub(crate) mailboxes: bool,
    /// When the panel was last drawn; the chat socket opens only while it is on screen.
    pub(crate) last_drawn: Option<Instant>,
}

impl AssistantPanel {
    pub fn new() -> Self {
        Self::with_runtime(crate::zc::runtime().clone())
    }

    /// A panel over `rt`; tests give each panel its own runtime.
    pub fn with_runtime(rt: Arc<ZcRuntime>) -> Self {
        let (local_tx, local_rx) = crossbeam::channel::bounded(1024);
        let (settings, settings_gen) = crate::settings::new_mirror();
        Self {
            zc: ZeroClawBackend::new(rt.clone()),
            rt,
            sessions: Sessions::default(),
            transcript: Transcript::default(),
            composer: Composer::default(),
            view: Default::default(),
            epoch: 0,
            local_epoch: 0,
            local_turn: 0,
            direct_backend: DirectBackend::new(local_tx.clone()),
            local_backend: LocalBackend::new(local_tx.clone()),
            local_tx,
            local_rx,
            running_on: None,
            settings,
            settings_gen,
            profile: ZcProfile::default(),
            health: Health::Unconfigured("Not set up yet".to_string()),
            direct: DirectState::default(),
            resolved: Resolved::NeedsSetup,
            input: ContextInput::default(),
            temp: 0.2,
            model_override: None,
            models: ModelBrowser::default(),
            tools_dbg: ToolsDebug::default(),
            queued: None,
            send_when_created: false,
            pending_send: false,
            title_job: None,
            connect_tried: None,
            stopping_since: None,
            setup_skipped: false,
            ai_update_tx: None,
            stale_dropped: 0,
            mailboxes: true,
            last_drawn: None,
        }
    }

    pub fn runtime(&self) -> &Arc<ZcRuntime> {
        &self.rt
    }

    /// Re-reads the settings, the profile and the gateway's health, and picks the backend.
    pub fn refresh(&mut self) {
        if crate::settings::sync_mirror(&mut self.settings, &mut self.settings_gen) {
            self.model_override = None;
        }
        self.profile = self.rt.profile();
        self.health = self.rt.health();
        self.direct = DirectState::of(&self.settings);
        self.resolved = backend::resolve(&backend::Inputs { pref: self.profile.backend_pref(), profile: &self.profile, has_token: self.rt.has_token(), health: &self.health, direct: &self.direct });
    }

    /// Per-frame work that must run whether or not the panel is on screen: mailboxes, session loads, events from the backends, timers.
    pub fn receive(&mut self, ctx: &Context) {
        self.rt.attach_ctx(ctx);
        self.refresh();
        let now = now_ms();
        self.take_mailboxes();
        self.sessions_step();
        self.drain_events(now);
        self.timers(ctx, now);
        self.ensure_connected();
    }
}

impl Default for AssistantPanel {
    fn default() -> Self {
        Self::new()
    }
}
