//! The ZeroClaw runtime: gateway profile, token, health, the chat connection and one-shot REST jobs, shared with the interface through channels.

use std::collections::VecDeque;
use std::future::Future;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crossbeam::channel::{Receiver, Sender, TrySendError};
use tokio::runtime::Handle;
use zeroclaw_client::{ApprovalDecision, CompleteRequest, CompleteResponse, Config, Error, InlineImage, RawFrame, RestClient, Usage};

use super::debug::FrameLog;
use super::events::{ChatEv, Link, LinkState, Pending, Tagged, UsageTotals, now_ms};
use super::health::{self, Health, SWEEP};
use super::profile::ZcProfile;
use super::secrets::TokenStore;
use super::session::{self, Mailbox, SessionCmd, Tag};

/// Events waiting for the interface before senders slow down.
pub const EVENT_CAPACITY: usize = 1024;
const NOTE_CAP: usize = 100;
/// Longest an event waits for room in a full channel before it is dropped.
const EMIT_PATIENCE: Duration = Duration::from_secs(2);

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

pub(super) struct Live {
    epoch: u64,
    session: String,
    mailbox: Mailbox,
}

#[derive(Default)]
struct ProbeState {
    in_flight: bool,
    last: Option<Instant>,
    /// Bumped when the connection choices change; a probe that started before is ignored.
    generation: u64,
}

/// State shared by the runtime handle, its connection tasks and background jobs.
pub struct Shared {
    handle: Handle,
    tokens: Option<TokenStore>,
    token: Mutex<Option<String>>,
    profile: Mutex<ZcProfile>,
    health: Mutex<Health>,
    probe: Mutex<ProbeState>,
    link: Mutex<Link>,
    live: Mutex<Option<Live>>,
    epoch: AtomicU64,
    evt_tx: Sender<Tagged>,
    frames: Mutex<FrameLog>,
    usage: Mutex<UsageTotals>,
    notes: Mutex<VecDeque<(u64, String)>>,
    ctx: Mutex<Option<egui::Context>>,
    dropped: AtomicU64,
    /// Show health on the ZEROCLAW engine dot and under Needs attention; only the application's runtime does.
    publish: AtomicBool,
}

impl Shared {
    pub fn repaint(&self) {
        match &*lock(&self.ctx) {
            Some(ctx) => ctx.request_repaint(),
            None => crate::command::request_repaint(),
        }
    }

    pub fn profile(&self) -> ZcProfile {
        lock(&self.profile).clone()
    }

    pub(super) fn token(&self) -> Option<String> {
        lock(&self.token).clone()
    }

    /// Removes the token from `text`.
    pub fn redact(&self, text: &str) -> String {
        match self.token() {
            Some(token) if !token.is_empty() => text.replace(&token, "***"),
            _ => text.to_string(),
        }
    }

    pub fn note(&self, text: impl Into<String>) {
        let mut notes = lock(&self.notes);
        notes.push_back((now_ms(), self.redact(&text.into())));
        while notes.len() > NOTE_CAP {
            notes.pop_front();
        }
        drop(notes);
        self.repaint();
    }

    /// The configuration of the chat socket: the profile and the stored token.
    pub(super) fn chat_config(&self) -> Result<Config, String> {
        let (profile, token) = (self.profile(), self.token());
        if !profile.is_configured() {
            return Err("The gateway address or the agent alias is not set".to_string());
        }
        let token = token.ok_or("There is no token: pair with the gateway or import one")?;
        profile.endpoint()?;
        let mut cfg = Config::new(profile.host.trim(), token).with_agent(profile.agent.trim()).with_device_name("aimediaexplorer");
        cfg.request_timeout = Duration::from_secs(20);
        Ok(cfg)
    }

    pub fn rest(&self) -> Result<RestClient, String> {
        let cfg = self.chat_config()?;
        RestClient::new(&cfg).map_err(|e| self.redact(&e.to_string()))
    }

    pub(super) fn is_current(&self, tag: &Tag) -> bool {
        self.epoch.load(Ordering::SeqCst) == tag.epoch
    }

    pub(super) fn link_state(&self) -> LinkState {
        lock(&self.link).state
    }

    /// Records a state change of the connection of `tag` unless a newer connection replaced it, and tells the interface.
    pub(super) fn set_link(&self, tag: &Tag, state: LinkState, detail: impl Into<String>) {
        if !self.is_current(tag) {
            return;
        }
        let detail = detail.into();
        lock(&self.link).set(state, detail.clone());
        let _ = self.evt_tx.try_send(Tagged { session: tag.session.clone(), epoch: tag.epoch, ev: ChatEv::Link { state, detail } });
        self.repaint();
    }

    pub(super) fn release(&self, tag: &Tag) {
        let mut live = lock(&self.live);
        if live.as_ref().is_some_and(|l| l.epoch == tag.epoch) {
            *live = None;
        }
    }

    pub(super) async fn emit(&self, tag: &Tag, ev: ChatEv) {
        let mut item = Tagged { session: tag.session.clone(), epoch: tag.epoch, ev };
        let started = Instant::now();
        loop {
            match self.evt_tx.try_send(item) {
                Ok(()) => {
                    self.repaint();
                    return;
                }
                Err(TrySendError::Disconnected(_)) => return,
                Err(TrySendError::Full(back)) => {
                    if started.elapsed() > EMIT_PATIENCE {
                        self.dropped.fetch_add(1, Ordering::Relaxed);
                        return;
                    }
                    item = back;
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            }
        }
    }

    pub(super) fn record_frame(&self, frame: RawFrame) {
        lock(&self.frames).push(frame, |text| self.redact(text));
        self.repaint();
    }

    pub(super) fn add_usage(&self, usage: &Usage) {
        lock(&self.usage).add(usage);
    }

    /// The gateway refused the token: say so everywhere and stop probing until the person acts.
    pub(super) fn refused(&self) {
        self.apply_health(Health::Unauthorized);
    }

    pub(super) async fn fail_link(&self, tag: &Tag, why: String) {
        self.set_link(tag, LinkState::Failed, why);
        self.release(tag);
    }

    /// Reports why the first connection failed, and lets health follow when the gateway cannot be reached or refused the token.
    pub(super) async fn connect_failed(&self, tag: &Tag, error: &Error) {
        let text = self.redact(&error.to_string());
        if error.is_auth() {
            self.refused();
            self.emit(tag, ChatEv::AuthRefused).await;
        } else if matches!(error, Error::Connect(_) | Error::Timeout(_)) {
            self.apply_health(health::classify_error(error));
        }
        self.set_link(tag, LinkState::Failed, text);
        self.release(tag);
    }

    pub(super) fn apply_health(&self, health: Health) {
        let profile = self.profile();
        let changed = {
            let mut current = lock(&self.health);
            let changed = *current != health;
            *current = health.clone();
            changed
        };
        if self.publish.load(Ordering::Relaxed) {
            health::publish(&health, &profile);
        }
        if changed {
            self.note(format!("health: {} ({})", health.label(), health.detail()));
        }
        self.repaint();
    }

    /// Starts a probe unless one runs; a refused token is only re-checked when `force` is set.
    fn check(self: &Arc<Self>, force: bool) -> bool {
        let generation = {
            let mut probe = lock(&self.probe);
            if probe.in_flight || (!force && lock(&self.health).is_refused()) {
                return false;
            }
            probe.in_flight = true;
            probe.generation
        };
        if !lock(&self.health).is_online() {
            self.apply_health(Health::Checking);
        }
        let sh = self.clone();
        self.handle.spawn(async move {
            let (profile, token) = (sh.profile(), sh.token());
            let found = health::probe(&profile, token.as_deref()).await;
            let mut probe = lock(&sh.probe);
            probe.in_flight = false;
            probe.last = Some(Instant::now());
            let current = probe.generation == generation;
            drop(probe);
            if current {
                sh.apply_health(found);
            } else {
                sh.check(true);
            }
        });
        true
    }
}

/// The interface's handle on the ZeroClaw runtime.
pub struct ZcRuntime {
    sh: Arc<Shared>,
    events: Receiver<Tagged>,
}

impl ZcRuntime {
    /// A runtime on `handle` that keeps its token in `tokens`.
    pub fn new(handle: Handle, tokens: Option<TokenStore>) -> Arc<Self> {
        let (evt_tx, events) = crossbeam::channel::bounded(EVENT_CAPACITY);
        let token = tokens.as_ref().and_then(TokenStore::load);
        let sh = Arc::new(Shared {
            handle,
            tokens,
            token: Mutex::new(token),
            profile: Mutex::new(ZcProfile::default()),
            health: Mutex::new(Health::Unconfigured("Not set up yet".to_string())),
            probe: Mutex::new(ProbeState::default()),
            link: Mutex::new(Link::default()),
            live: Mutex::new(None),
            epoch: AtomicU64::new(0),
            evt_tx,
            frames: Mutex::new(FrameLog::default()),
            usage: Mutex::new(UsageTotals::default()),
            notes: Mutex::new(VecDeque::new()),
            ctx: Mutex::new(None),
            dropped: AtomicU64::new(0),
            publish: AtomicBool::new(false),
        });
        Arc::new(Self { sh, events })
    }

    pub fn shared(&self) -> &Arc<Shared> {
        &self.sh
    }

    /// Makes this runtime's health the one the status bar and Needs attention show.
    pub fn set_publish(&self, on: bool) {
        self.sh.publish.store(on, Ordering::Relaxed);
    }

    /// Where repaint requests go; the frame loop sets this every frame.
    pub fn attach_ctx(&self, ctx: &egui::Context) {
        let mut slot = lock(&self.sh.ctx);
        if slot.is_none() {
            *slot = Some(ctx.clone());
        }
    }

    pub fn profile(&self) -> ZcProfile {
        self.sh.profile()
    }

    /// Replaces the profile. A new address or agent closes the chat socket and checks the gateway again. True when something changed.
    pub fn set_profile(&self, profile: ZcProfile) -> bool {
        let moved = {
            let mut current = lock(&self.sh.profile);
            if *current == profile {
                return false;
            }
            let moved = current.host != profile.host || current.agent != profile.agent;
            *current = profile;
            moved
        };
        if moved {
            lock(&self.sh.probe).generation += 1;
            self.disconnect();
            self.sh.check(true);
        }
        self.sh.repaint();
        true
    }

    pub fn has_token(&self) -> bool {
        self.sh.token().is_some()
    }

    /// Where the token file is, for display.
    pub fn token_path(&self) -> Option<PathBuf> {
        self.sh.tokens.as_ref().map(TokenStore::path)
    }

    /// Stores a token and checks the gateway with it.
    pub fn save_token(&self, token: &str) -> Result<(), String> {
        let token = token.trim();
        if let Some(store) = &self.sh.tokens {
            store.save(token).map_err(|e| format!("The token was not saved: {e}"))?;
        }
        *lock(&self.sh.token) = Some(token.to_string()).filter(|t| !t.is_empty());
        lock(&self.sh.probe).generation += 1;
        self.sh.check(true);
        Ok(())
    }

    /// Deletes the stored token and closes the chat socket.
    pub fn clear_token(&self) -> Result<(), String> {
        if let Some(store) = &self.sh.tokens {
            store.clear().map_err(|e| format!("The token was not removed: {e}"))?;
        }
        *lock(&self.sh.token) = None;
        lock(&self.sh.probe).generation += 1;
        self.disconnect();
        self.sh.check(true);
        Ok(())
    }

    pub fn health(&self) -> Health {
        lock(&self.sh.health).clone()
    }

    /// Probes the gateway now, even after a refused token. False when a probe is already running.
    pub fn check(&self) -> bool {
        self.sh.check(true)
    }

    /// Runs the periodic sweep: a probe every [`SWEEP`], never after a refused token.
    pub fn tick(&self) {
        self.tick_at(Instant::now());
    }

    /// [`tick`](Self::tick) with the time given, for tests.
    pub fn tick_at(&self, now: Instant) {
        let due = {
            let probe = lock(&self.sh.probe);
            !probe.in_flight && probe.last.is_none_or(|at| now.saturating_duration_since(at) >= SWEEP)
        };
        let profile = self.sh.profile();
        if due && profile.is_configured() && self.has_token() {
            self.sh.check(false);
        }
    }

    pub fn link(&self) -> Link {
        lock(&self.sh.link).clone()
    }

    /// The number of the newest connection; events of older ones are stale.
    pub fn epoch(&self) -> u64 {
        self.sh.epoch.load(Ordering::SeqCst)
    }

    /// Opens (or resumes) the chat session `session`; the one open before is closed. Returns the connection number its events carry.
    pub fn connect(&self, session: &str, name: Option<&str>) -> u64 {
        let epoch = self.sh.epoch.fetch_add(1, Ordering::SeqCst) + 1;
        let mailbox = Mailbox::new();
        let old = lock(&self.sh.live).replace(Live { epoch, session: session.to_string(), mailbox: mailbox.clone() });
        if let Some(old) = old {
            old.mailbox.send(SessionCmd::Close);
        }
        {
            let mut link = lock(&self.sh.link);
            link.session = Some(session.to_string());
            link.epoch = epoch;
            link.set(LinkState::Connecting, "starting");
        }
        let tag = Tag { session: session.to_string(), epoch };
        let name = name.map(str::to_string);
        self.sh.handle.spawn(session::run(self.sh.clone(), tag, name, mailbox));
        self.sh.repaint();
        epoch
    }

    /// Closes the chat socket; a running turn goes on at the gateway.
    pub fn disconnect(&self) {
        let old = lock(&self.sh.live).take();
        self.sh.epoch.fetch_add(1, Ordering::SeqCst);
        if let Some(old) = old {
            old.mailbox.send(SessionCmd::Close);
        }
        let mut link = lock(&self.sh.link);
        link.session = None;
        link.epoch = self.sh.epoch.load(Ordering::SeqCst);
        link.set(LinkState::Idle, "no session");
        drop(link);
        self.sh.repaint();
    }

    /// The session id of the open connection.
    pub fn live_session(&self) -> Option<String> {
        lock(&self.sh.live).as_ref().map(|l| l.session.clone())
    }

    fn to_live(&self, cmd: SessionCmd) -> bool {
        match &*lock(&self.sh.live) {
            Some(live) => {
                live.mailbox.send(cmd);
                true
            }
            None => false,
        }
    }

    /// Sends a message; without an open connection the text comes back as a failed send.
    pub fn send(&self, text: String, images: Vec<InlineImage>) {
        let draft = text.clone();
        if !self.to_live(SessionCmd::Send { text, images }) {
            let _ = self.sh.evt_tx.try_send(Tagged {
                session: lock(&self.sh.link).session.clone().unwrap_or_default(),
                epoch: self.epoch(),
                ev: ChatEv::SendFailed { text: draft, error: "not connected".to_string() },
            });
        }
    }

    pub fn stop(&self) -> bool {
        self.to_live(SessionCmd::Stop)
    }

    pub fn approve(&self, request_id: &str, decision: ApprovalDecision) -> bool {
        self.to_live(SessionCmd::Approval { request_id: request_id.to_string(), decision })
    }

    pub fn reconnect(&self) -> bool {
        self.to_live(SessionCmd::Reconnect)
    }

    /// Sends a hand-written frame (Devtools); the result arrives as a notice.
    pub fn raw_send(&self, json: &str) -> bool {
        self.to_live(SessionCmd::Raw(json.to_string()))
    }

    pub fn load_history(&self, page_size: usize, more: bool) -> bool {
        self.to_live(SessionCmd::History { page_size, more })
    }

    /// Up to `max` pending events, oldest first.
    pub fn drain(&self, max: usize) -> Vec<Tagged> {
        self.events.try_iter().take(max).collect()
    }

    /// Events waiting for the interface.
    pub fn backlog(&self) -> usize {
        self.events.len()
    }

    /// Events dropped because the interface did not keep up.
    pub fn dropped(&self) -> u64 {
        self.sh.dropped.load(Ordering::Relaxed)
    }

    pub fn with_frames<R>(&self, f: impl FnOnce(&mut FrameLog) -> R) -> R {
        f(&mut lock(&self.sh.frames))
    }

    pub fn usage(&self) -> UsageTotals {
        lock(&self.sh.usage).clone()
    }

    pub fn note(&self, text: impl Into<String>) {
        self.sh.note(text);
    }

    /// Notes, newest first.
    pub fn notes(&self) -> Vec<(u64, String)> {
        lock(&self.sh.notes).iter().rev().cloned().collect()
    }

    /// Runs `work` in the background; the result is polled from the frame loop.
    pub fn job<T, Fut>(&self, work: impl FnOnce(Arc<Shared>) -> Fut) -> Pending<T>
    where
        T: Send + 'static,
        Fut: Future<Output = T> + Send + 'static,
    {
        let (tx, rx) = crossbeam::channel::bounded(1);
        let sh = self.sh.clone();
        let fut = work(sh.clone());
        self.sh.handle.spawn(async move {
            let _ = tx.send(fut.await);
            sh.repaint();
        });
        Pending::new(rx)
    }

    /// One model call through `/api/complete`.
    pub fn complete(&self, request: CompleteRequest) -> Pending<Result<CompleteResponse, String>> {
        self.job(|sh| async move {
            let rest = sh.rest()?;
            rest.complete(&request).await.map_err(|e| sh.redact(&e.to_string()))
        })
    }

    /// Exchanges a one-time pairing code for a token and stores it. The token never leaves the runtime.
    pub fn pair(&self, host: &str, code: &str) -> Pending<Result<(), String>> {
        let (host, code) = (host.trim().to_string(), code.trim().to_string());
        let store = self.sh.tokens.clone();
        let this = self.sh.clone();
        self.job(move |_| async move {
            if host.is_empty() || code.is_empty() {
                return Err("Enter the gateway address and the pairing code".to_string());
            }
            let paired = RestClient::pair(&Config::new(&host, ""), &code).await.map_err(|e| e.to_string())?;
            if !paired.paired || paired.token.trim().is_empty() {
                return Err("The gateway did not pair".to_string());
            }
            if let Some(store) = store {
                store.save(&paired.token).map_err(|e| format!("The token was not saved: {e}"))?;
            }
            *lock(&this.token) = Some(paired.token.trim().to_string());
            lock(&this.probe).generation += 1;
            Ok(())
        })
    }

    /// Asks the gateway at `host` which agents it offers, with `token` or the stored one.
    pub fn detect_agents(&self, host: &str, token: Option<String>) -> Pending<Result<Vec<String>, String>> {
        let host = host.trim().to_string();
        self.job(move |sh| async move {
            let token = token.or_else(|| sh.token()).ok_or("There is no token to ask with")?;
            let rest = RestClient::new(&Config::new(&host, token)).map_err(|e| sh.redact(&e.to_string()))?;
            rest.agents().await.map_err(|e| sh.redact(&e.to_string()))
        })
    }
}
