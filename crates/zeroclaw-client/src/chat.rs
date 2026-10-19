use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpStream;
use tokio::sync::{mpsc, oneshot, watch};
use tokio::time::{Instant, timeout, timeout_at};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::protocol::{CloseFrame, WebSocketConfig};
use tokio_tungstenite::tungstenite::{self, Message};
use tokio_tungstenite::{Connector, MaybeTlsStream, WebSocketStream, connect_async_tls_with_config};

use crate::attachments::{InlineImage, compose_message};
use crate::backoff::Backoff;
use crate::config::{Config, WsAuth};
use crate::error::{Error, Result};
use crate::event::{Direction, DisconnectReason, Event, RawFrame, Translator};
use crate::frames::{ApprovalDecision, ClientFrame, ServerFrame, error_code_is_fatal};
use crate::rest::RestClient;
use crate::tls::{client_config, install_crypto_provider};
use crate::types::AbortOutcome;

type Ws = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// Queued events at which socket reads pause.
const OUTBOX_HIGH_WATER: usize = 512;
/// Resume watchdog re-checks of a turn the gateway still reports as running.
const MAX_RESUME_CHECKS: u32 = 3;
const CLOSE_GRACE: Duration = Duration::from_secs(2);

/// Options for opening a chat session.
#[derive(Debug, Clone, Default)]
pub struct ConnectOptions {
    /// Session id to resume.
    pub session_id: Option<String>,
    /// Name given to a new session.
    pub name: Option<String>,
    /// Working directory sent in the `connect` frame.
    pub cwd: Option<String>,
    /// Also emit [`Event::Raw`] for every frame in both directions.
    pub raw_tap: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnState {
    Connecting,
    Open,
    Reconnecting,
    Closed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AuthMode {
    Header,
    Query,
    NoAuth,
}

/// Bounded receiver of a session's events.
pub struct EventStream {
    rx: mpsc::Receiver<Event>,
}

impl EventStream {
    pub async fn next(&mut self) -> Option<Event> {
        self.rx.recv().await
    }

    /// Next event if one is ready.
    pub fn try_next(&mut self) -> Option<Event> {
        self.rx.try_recv().ok()
    }
}

impl futures_core::Stream for EventStream {
    type Item = Event;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Event>> {
        self.rx.poll_recv(cx)
    }
}

enum Command {
    Send(ClientFrame, oneshot::Sender<Result<()>>),
    Reconnect,
    Close,
}

struct Shared {
    cmd: mpsc::Sender<Command>,
    session_id: String,
    config: Arc<Config>,
    rest: RestClient,
    state: watch::Receiver<ConnState>,
}

/// A live `/ws/chat` session; clones share one socket, closed when the last clone drops.
#[derive(Clone)]
pub struct ChatSession {
    shared: Arc<Shared>,
}

impl ChatSession {
    /// Opens the first connection and starts the supervisor that reconnects on drops.
    pub async fn connect(config: Config, opts: ConnectOptions) -> Result<(ChatSession, EventStream)> {
        install_crypto_provider();
        if config.agent_name().is_empty() {
            return Err(Error::Config("an agent alias is required: the gateway has no default agent".into()));
        }
        let config = Arc::new(config);
        let rest = RestClient::new(&config)?;
        let mut auth = initial_auth(&config);
        let opened = open(&config, &opts, opts.session_id.as_deref(), &mut auth).await?;

        let (cmd_tx, cmd_rx) = mpsc::channel(64);
        let (ev_tx, ev_rx) = mpsc::channel(config.event_buffer.max(16));
        let (state_tx, state_rx) = watch::channel(ConnState::Open);
        let session_id = opened.session_id.clone();
        let supervisor = Supervisor {
            backoff: Backoff::from_clock(config.reconnect.clone()),
            cfg: config.clone(),
            opts,
            rest: rest.clone(),
            session_id: session_id.clone(),
            auth,
            translator: Translator::new(),
            outbox: VecDeque::new(),
            t0: Instant::now(),
            state_tx,
        };
        tokio::spawn(supervisor.run(opened, cmd_rx, ev_tx));
        let shared = Shared { cmd: cmd_tx, session_id, config, rest, state: state_rx };
        Ok((ChatSession { shared: Arc::new(shared) }, EventStream { rx: ev_rx }))
    }

    pub fn session_id(&self) -> &str {
        &self.shared.session_id
    }

    pub fn config(&self) -> &Config {
        &self.shared.config
    }

    /// REST client with the same credentials.
    pub fn rest(&self) -> &RestClient {
        &self.shared.rest
    }

    /// The stored transcript of this session, newest `page_size` messages first.
    pub async fn history(&self, page_size: usize, tool_context: bool) -> Result<crate::history::HistoryPager> {
        self.shared.rest.history(&self.shared.session_id, page_size, tool_context).await
    }

    pub fn state(&self) -> ConnState {
        *self.shared.state.borrow()
    }

    /// Resolves when the state changes; `None` once the session is gone.
    pub async fn state_changed(&self) -> Option<ConnState> {
        let mut rx = self.shared.state.clone();
        rx.changed().await.ok()?;
        let state = *rx.borrow();
        Some(state)
    }

    /// Text-only [`send_user_message`](Self::send_user_message).
    pub async fn send_text(&self, text: &str) -> Result<()> {
        self.send_user_message(text, &[]).await
    }

    /// Sends a user message with `images` appended as inline markers; never queued across a reconnect.
    pub async fn send_user_message(&self, text: &str, images: &[InlineImage]) -> Result<()> {
        let body = compose_message(text, images);
        if body.trim().is_empty() {
            return Err(Error::EmptyMessage);
        }
        let limit = self.shared.config.max_outbound_bytes;
        if body.len() > limit {
            return Err(Error::TooLarge { size: body.len(), limit });
        }
        self.send_frame(ClientFrame::Message { content: body }).await
    }

    pub async fn respond_approval(&self, request_id: &str, decision: ApprovalDecision) -> Result<()> {
        self.send_frame(ClientFrame::ApprovalResponse { request_id: request_id.to_string(), decision }).await
    }

    pub async fn approve(&self, request_id: &str) -> Result<()> {
        self.respond_approval(request_id, ApprovalDecision::Approve).await
    }

    /// Approves and stops asking for this tool.
    pub async fn approve_always(&self, request_id: &str) -> Result<()> {
        self.respond_approval(request_id, ApprovalDecision::Always).await
    }

    pub async fn deny(&self, request_id: &str) -> Result<()> {
        self.respond_approval(request_id, ApprovalDecision::Deny).await
    }

    /// Stops the running turn through `POST /api/sessions/{id}/abort`.
    pub async fn abort(&self) -> Result<AbortOutcome> {
        self.shared.rest.abort(&self.shared.session_id).await
    }

    /// Drops the socket and reconnects immediately, skipping any pending backoff.
    pub async fn reconnect(&self) -> Result<()> {
        self.shared.cmd.send(Command::Reconnect).await.map_err(|_| Error::Closed)
    }

    /// Closes the socket and ends the session; the stream ends after a final `Disconnected`.
    pub async fn close(&self) {
        let _ = self.shared.cmd.send(Command::Close).await;
    }

    async fn send_frame(&self, frame: ClientFrame) -> Result<()> {
        let (ack_tx, ack_rx) = oneshot::channel();
        let budget = self.shared.config.write_timeout + Duration::from_secs(1);
        timeout(budget, async {
            self.shared.cmd.send(Command::Send(frame, ack_tx)).await.map_err(|_| Error::Closed)?;
            ack_rx.await.map_err(|_| Error::Closed)?
        })
        .await
        .map_err(|_| Error::Timeout("send"))?
    }
}

fn initial_auth(cfg: &Config) -> AuthMode {
    if !cfg.has_token() {
        return AuthMode::NoAuth;
    }
    match cfg.ws_auth {
        WsAuth::Query => AuthMode::Query,
        WsAuth::Auto | WsAuth::Header => AuthMode::Header,
    }
}

struct Opened {
    ws: Ws,
    session_id: String,
    resumed: bool,
    message_count: u64,
    name: Option<String>,
    handshake_raw: Vec<(Direction, serde_json::Value)>,
}

/// Connects, sends `connect` and waits for `session_start`; `WsAuth::Auto` falls back from header to query auth on 401.
async fn open(cfg: &Config, opts: &ConnectOptions, resume: Option<&str>, auth: &mut AuthMode) -> Result<Opened> {
    let mut mode = *auth;
    let mut fell_back = false;
    loop {
        match open_once(cfg, opts, resume, mode).await {
            Ok(opened) => {
                *auth = mode;
                return Ok(opened);
            }
            Err(Error::Unauthorized) if cfg.ws_auth == WsAuth::Auto && mode == AuthMode::Header && !fell_back => {
                mode = AuthMode::Query;
                fell_back = true;
            }
            Err(e) => return Err(e),
        }
    }
}

/// [`open`] that returns the auth mode it ended with instead of borrowing it.
async fn open_tracked(cfg: &Config, opts: &ConnectOptions, resume: &str, mut auth: AuthMode) -> (Result<Opened>, AuthMode) {
    let result = open(cfg, opts, Some(resume), &mut auth).await;
    (result, auth)
}

async fn open_once(cfg: &Config, opts: &ConnectOptions, resume: Option<&str>, mode: AuthMode) -> Result<Opened> {
    let endpoint = cfg.endpoint()?;
    let mut query: Vec<(&str, &str)> = vec![("agent", cfg.agent_name())];
    if let Some(id) = resume {
        query.push(("session_id", id));
    } else if let Some(name) = opts.name.as_deref() {
        query.push(("name", name));
    }
    let token = cfg.token().map(|t| t.expose().to_string());
    if mode == AuthMode::Query
        && let Some(t) = token.as_deref()
    {
        query.push(("token", t));
    }
    let url = endpoint.ws_url("/ws/chat", &query);
    let mut request = url.as_str().into_client_request().map_err(|e| Error::Config(e.to_string()))?;
    if mode == AuthMode::Header
        && let Some(t) = token.as_deref()
    {
        let value = tungstenite::http::HeaderValue::from_str(&format!("Bearer {t}"))
            .map_err(|_| Error::Config("token contains characters that cannot be sent in a header".into()))?;
        request.headers_mut().insert(tungstenite::http::header::AUTHORIZATION, value);
    }
    let ws_config = WebSocketConfig::default()
        .max_message_size(Some(cfg.max_frame_bytes))
        .max_frame_size(Some(cfg.max_frame_bytes));
    let connector = if endpoint.tls() {
        Some(Connector::Rustls(client_config(cfg.extra_root_pem.as_deref())?))
    } else {
        None
    };
    let (mut ws, _) = timeout(cfg.connect_timeout, connect_async_tls_with_config(request, Some(ws_config), true, connector))
        .await
        .map_err(|_| Error::Timeout("connect"))?
        .map_err(|e| map_ws_error(cfg, e))?;

    let mut handshake_raw = Vec::new();
    let hello = ClientFrame::Connect {
        device_name: Some(cfg.device_name.clone()),
        capabilities: vec!["text".into()],
        cwd: opts.cwd.clone(),
    };
    handshake_raw.push((Direction::Out, serde_json::to_value(&hello).unwrap_or_default()));
    timeout(cfg.write_timeout, ws.send(Message::Text(hello.to_json().into())))
        .await
        .map_err(|_| Error::Timeout("sending connect"))?
        .map_err(|e| map_ws_error(cfg, e))?;

    let deadline = Instant::now() + cfg.handshake_timeout;
    loop {
        let next = timeout_at(deadline, ws.next()).await.map_err(|_| Error::Timeout("session_start"))?;
        let text = match next {
            None => return Err(Error::Protocol("gateway closed before session_start".into())),
            Some(Err(e)) => return Err(map_ws_error(cfg, e)),
            Some(Ok(Message::Text(t))) => t,
            Some(Ok(Message::Close(_))) => return Err(Error::Protocol("gateway closed before session_start".into())),
            Some(Ok(_)) => continue,
        };
        let frame = ServerFrame::parse(&text);
        handshake_raw.push((Direction::In, serde_json::from_str(&text).unwrap_or_default()));
        match frame {
            ServerFrame::SessionStart { session_id, resumed, message_count, name } => {
                if let Some(want) = resume
                    && want != session_id
                {
                    let _ = ws.close(None).await;
                    return Err(Error::SessionMismatch { requested: want.to_string(), received: session_id });
                }
                return Ok(Opened { ws, session_id, resumed, message_count, name, handshake_raw });
            }
            ServerFrame::Error { code: Some(code), message } if error_code_is_fatal(&code) => {
                return Err(Error::Gateway { code, message: cfg.redact(&message) });
            }
            _ => {}
        }
    }
}

fn map_ws_error(cfg: &Config, e: tungstenite::Error) -> Error {
    use tungstenite::Error as E;
    match e {
        E::Http(response) => match response.status().as_u16() {
            401 => Error::Unauthorized,
            403 => Error::Forbidden,
            status => {
                let body = response.body().as_deref().map(String::from_utf8_lossy).unwrap_or_default();
                let message: String = cfg.redact(body.trim()).chars().take(300).collect();
                Error::Rejected { status, message }
            }
        },
        E::Capacity(_) => Error::Protocol("frame exceeds the size limit".into()),
        E::Url(e) => Error::Config(e.to_string()),
        other => Error::Connect(cfg.redact(&other.to_string())),
    }
}

enum Exit {
    /// Do not reconnect.
    Stop(DisconnectReason),
    /// The socket is gone; `retry` allows a reconnect.
    Lost { reason: DisconnectReason, retry: bool },
}

struct Supervisor {
    cfg: Arc<Config>,
    opts: ConnectOptions,
    rest: RestClient,
    session_id: String,
    auth: AuthMode,
    translator: Translator,
    outbox: VecDeque<Event>,
    t0: Instant,
    backoff: Backoff,
    state_tx: watch::Sender<ConnState>,
}

impl Supervisor {
    fn emit(&mut self, event: Event) {
        self.outbox.push_back(event);
    }

    fn raw(&mut self, dir: Direction, frame: serde_json::Value) {
        if self.opts.raw_tap {
            let t_ms = self.t0.elapsed().as_millis() as u64;
            self.outbox.push_back(Event::Raw(RawFrame { t_ms, dir, frame }));
        }
    }

    fn set_state(&self, state: ConnState) {
        let _ = self.state_tx.send(state);
    }

    async fn run(mut self, first: Opened, mut cmd_rx: mpsc::Receiver<Command>, ev_tx: mpsc::Sender<Event>) {
        let mut next = Some(first);
        let mut reconnect = false;
        let mut immediate = false;
        loop {
            let opened = match next.take() {
                Some(o) => o,
                None => match self.reconnect_loop(immediate, &mut cmd_rx, &ev_tx).await {
                    Some(o) => o,
                    None => break,
                },
            };
            for (dir, frame) in opened.handshake_raw.clone() {
                self.raw(dir, frame);
            }
            self.emit(Event::Connected {
                session_id: opened.session_id.clone(),
                resumed: opened.resumed,
                message_count: opened.message_count,
                name: opened.name.clone(),
                reconnect,
            });
            self.set_state(ConnState::Open);
            reconnect = true;
            match self.drive(opened.ws, &mut cmd_rx, &ev_tx).await {
                Exit::Stop(reason) => {
                    self.emit(Event::Disconnected { reason, will_retry: false });
                    break;
                }
                Exit::Lost { reason, retry } => {
                    let will_retry = retry && self.cfg.reconnect.enabled;
                    immediate = reason == DisconnectReason::Requested;
                    self.emit(Event::Disconnected { reason, will_retry });
                    if !will_retry {
                        break;
                    }
                }
            }
        }
        self.set_state(ConnState::Closed);
        let deadline = Instant::now() + CLOSE_GRACE;
        while let Some(event) = self.outbox.pop_front() {
            if !matches!(timeout_at(deadline, ev_tx.send(event)).await, Ok(Ok(()))) {
                break;
            }
        }
    }

    /// Retries with backoff until a connection opens, a failure is permanent, or the session is closed.
    async fn reconnect_loop(
        &mut self,
        mut immediate: bool,
        cmd_rx: &mut mpsc::Receiver<Command>,
        ev_tx: &mpsc::Sender<Event>,
    ) -> Option<Opened> {
        self.set_state(ConnState::Reconnecting);
        let mut last_error: Option<String> = None;
        loop {
            let Some(mut delay) = self.backoff.next_delay() else {
                self.emit(Event::Disconnected {
                    reason: DisconnectReason::Fatal("reconnect attempts exhausted".into()),
                    will_retry: false,
                });
                return None;
            };
            if std::mem::take(&mut immediate) {
                delay = Duration::ZERO;
            }
            self.emit(Event::Reconnecting { attempt: self.backoff.attempt(), delay, last_error: last_error.clone() });
            if !self.wait(delay, cmd_rx, ev_tx).await {
                self.emit(Event::Disconnected { reason: DisconnectReason::Requested, will_retry: false });
                return None;
            }
            let (cfg, opts, session_id) = (self.cfg.clone(), self.opts.clone(), self.session_id.clone());
            let attempt = open_tracked(&cfg, &opts, &session_id, self.auth);
            tokio::pin!(attempt);
            // Answers commands while the attempt runs.
            let (result, auth) = loop {
                tokio::select! {
                    result = &mut attempt => break result,
                    cmd = cmd_rx.recv() => match cmd {
                        None | Some(Command::Close) => {
                            self.emit(Event::Disconnected { reason: DisconnectReason::Requested, will_retry: false });
                            return None;
                        }
                        Some(Command::Reconnect) => {}
                        Some(Command::Send(_, ack)) => {
                            let _ = ack.send(Err(Error::NotConnected));
                        }
                    },
                    _ = ev_tx.closed() => return None,
                }
            };
            self.auth = auth;
            match result {
                Ok(opened) => {
                    self.backoff.reset();
                    return Some(opened);
                }
                Err(e) if e.is_retryable() => last_error = Some(self.cfg.redact(&e.to_string())),
                Err(e) => {
                    self.emit(Event::Disconnected {
                        reason: DisconnectReason::Fatal(self.cfg.redact(&e.to_string())),
                        will_retry: false,
                    });
                    return None;
                }
            }
        }
    }

    /// Sleeps for `delay` while answering commands and draining events. False when the session was closed.
    async fn wait(&mut self, delay: Duration, cmd_rx: &mut mpsc::Receiver<Command>, ev_tx: &mpsc::Sender<Event>) -> bool {
        let sleep = tokio::time::sleep(delay);
        tokio::pin!(sleep);
        loop {
            tokio::select! {
                biased;
                permit = ev_tx.reserve(), if !self.outbox.is_empty() => match permit {
                    Ok(p) => {
                        if let Some(e) = self.outbox.pop_front() {
                            p.send(e);
                        }
                    }
                    Err(_) => return false,
                },
                _ = ev_tx.closed() => return false,
                cmd = cmd_rx.recv() => match cmd {
                    None | Some(Command::Close) => return false,
                    Some(Command::Reconnect) => return true,
                    Some(Command::Send(_, ack)) => {
                        let _ = ack.send(Err(Error::NotConnected));
                    }
                },
                _ = &mut sleep => return true,
            }
        }
    }

    async fn drive(&mut self, ws: Ws, cmd_rx: &mut mpsc::Receiver<Command>, ev_tx: &mpsc::Sender<Event>) -> Exit {
        let (mut sink, mut stream) = ws.split();
        let cfg = self.cfg.clone();
        let mut ping = tokio::time::interval_at(Instant::now() + cfg.ping_interval, cfg.ping_interval);
        ping.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut awaiting_pong: Option<Instant> = None;
        let mut resume_deadline = self.translator.turn_open().then(|| Instant::now() + cfg.resume_grace);
        let mut resume_checks = 0u32;
        let mut fatal_code: Option<String> = None;

        loop {
            let watchdog = async {
                match resume_deadline {
                    Some(at) => tokio::time::sleep_until(at).await,
                    None => std::future::pending::<()>().await,
                }
            };
            tokio::select! {
                biased;
                permit = ev_tx.reserve(), if !self.outbox.is_empty() => match permit {
                    Ok(p) => {
                        if let Some(e) = self.outbox.pop_front() {
                            p.send(e);
                        }
                    }
                    Err(_) => return Exit::Stop(DisconnectReason::Requested),
                },
                _ = ev_tx.closed() => return Exit::Stop(DisconnectReason::Requested),
                cmd = cmd_rx.recv() => match cmd {
                    None | Some(Command::Close) => {
                        close_socket(&mut sink).await;
                        return Exit::Stop(DisconnectReason::Requested);
                    }
                    Some(Command::Reconnect) => {
                        close_socket(&mut sink).await;
                        return Exit::Lost { reason: DisconnectReason::Requested, retry: true };
                    }
                    Some(Command::Send(frame, ack)) => {
                        self.raw(Direction::Out, serde_json::to_value(&frame).unwrap_or_default());
                        let sent = timeout(cfg.write_timeout, sink.send(Message::Text(frame.to_json().into()))).await;
                        match sent {
                            Ok(Ok(())) => {
                                let _ = ack.send(Ok(()));
                            }
                            Ok(Err(e)) => {
                                let _ = ack.send(Err(Error::NotConnected));
                                return Exit::Lost { reason: DisconnectReason::Transport(cfg.redact(&e.to_string())), retry: true };
                            }
                            Err(_) => {
                                let _ = ack.send(Err(Error::Timeout("socket write")));
                                return Exit::Lost { reason: DisconnectReason::Transport("write timed out".into()), retry: true };
                            }
                        }
                    }
                },
                msg = stream.next(), if self.outbox.len() < OUTBOX_HIGH_WATER => match msg {
                    None => return lost(DisconnectReason::Transport("socket ended".into()), &fatal_code),
                    Some(Err(tungstenite::Error::Capacity(_))) => {
                        self.emit(Event::Error {
                            turn: self.translator.turn(),
                            code: Some("FRAME_TOO_LARGE".into()),
                            message: format!("inbound frame exceeded {} bytes", cfg.max_frame_bytes),
                            ends_turn: false,
                        });
                        return Exit::Lost { reason: DisconnectReason::FrameTooLarge, retry: false };
                    }
                    Some(Err(e)) => return lost(DisconnectReason::Transport(cfg.redact(&e.to_string())), &fatal_code),
                    Some(Ok(message)) => {
                        awaiting_pong = None;
                        match message {
                            Message::Text(text) => self.on_text(&text, &mut fatal_code, &mut resume_deadline),
                            Message::Ping(_) => {
                                let _ = timeout(cfg.write_timeout, sink.flush()).await;
                            }
                            Message::Close(frame) => {
                                let (code, reason) = close_parts(frame);
                                return lost(DisconnectReason::Remote { code, reason }, &fatal_code);
                            }
                            _ => {}
                        }
                    }
                },
                _ = ping.tick() => {
                    if self.outbox.len() >= OUTBOX_HIGH_WATER {
                        // No pong is expected while reads are paused.
                        awaiting_pong = None;
                    }
                    if awaiting_pong.is_some_and(|since| since.elapsed() >= cfg.pong_timeout) {
                        return Exit::Lost { reason: DisconnectReason::PingTimeout, retry: true };
                    }
                    match timeout(cfg.write_timeout, sink.send(Message::Ping(Vec::new().into()))).await {
                        Ok(Ok(())) => {
                            awaiting_pong.get_or_insert_with(Instant::now);
                        }
                        Ok(Err(e)) => {
                            return Exit::Lost { reason: DisconnectReason::Transport(cfg.redact(&e.to_string())), retry: true };
                        }
                        Err(_) => {
                            return Exit::Lost { reason: DisconnectReason::Transport("ping write timed out".into()), retry: true };
                        }
                    }
                },
                _ = watchdog => {
                    resume_deadline = None;
                    resume_checks += 1;
                    match self.turn_still_running().await {
                        Some(true) if resume_checks < MAX_RESUME_CHECKS => {
                            resume_deadline = Some(Instant::now() + cfg.resume_grace);
                        }
                        Some(false) => {
                            let mut lost = Vec::new();
                            self.translator.lose_turn(&mut lost);
                            self.outbox.extend(lost);
                        }
                        _ if resume_checks < MAX_RESUME_CHECKS => {
                            resume_deadline = Some(Instant::now() + cfg.resume_grace);
                        }
                        _ => {}
                    }
                },
            }
        }
    }

    /// Asks the gateway whether the session's turn is still running; `None` when it cannot say.
    async fn turn_still_running(&self) -> Option<bool> {
        let budget = self.cfg.request_timeout.min(Duration::from_secs(5));
        match timeout(budget, self.rest.session_state(&self.session_id)).await {
            Ok(Ok(state)) => Some(state.running),
            Ok(Err(Error::Http { status: 404, .. })) => Some(false),
            _ => None,
        }
    }

    fn on_text(&mut self, text: &str, fatal_code: &mut Option<String>, resume_deadline: &mut Option<Instant>) {
        let frame = ServerFrame::parse(text);
        if self.opts.raw_tap {
            let value = serde_json::from_str(text).unwrap_or_else(|_| serde_json::Value::String(text.to_string()));
            self.raw(Direction::In, value);
        }
        match &frame {
            ServerFrame::Error { code: Some(code), .. } if error_code_is_fatal(code) => *fatal_code = Some(code.clone()),
            ServerFrame::TurnResume { .. } => *resume_deadline = None,
            _ => {}
        }
        let mut events = Vec::new();
        self.translator.feed(frame, &mut events);
        self.outbox.extend(events);
    }
}

fn lost(reason: DisconnectReason, fatal_code: &Option<String>) -> Exit {
    match fatal_code {
        Some(code) => Exit::Lost { reason: DisconnectReason::ServerError(code.clone()), retry: false },
        None => Exit::Lost { reason, retry: true },
    }
}

fn close_parts(frame: Option<CloseFrame>) -> (Option<u16>, String) {
    match frame {
        Some(f) => (Some(u16::from(f.code)), f.reason.to_string()),
        None => (None, String::new()),
    }
}

async fn close_socket<S>(sink: &mut S)
where
    S: futures_util::Sink<Message> + Unpin,
{
    let _ = timeout(CLOSE_GRACE, sink.send(Message::Close(None))).await;
    let _ = timeout(CLOSE_GRACE, sink.close()).await;
}
