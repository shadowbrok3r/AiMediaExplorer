//! Loopback mock gateway: `/ws/chat` frames and REST routes on one port, scripted with `Turn` and `ConnScript`.

use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use axum::Router;
use axum::body::Bytes;
use axum::extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Query, State};
use axum::http::{HeaderMap, Method, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use futures_util::stream::SplitSink;
use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio::sync::{Notify, mpsc, watch};

use crate::config::{Config, ReconnectPolicy};
use crate::replay::Recording;
use crate::text::{decode_component, rfc3339_utc};
use crate::types::COMPLETE_INPUT_CAP;

/// Where the mock accepts the WebSocket token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MockAuth {
    Any,
    /// Accepts only the `Authorization` header.
    HeaderOnly,
    /// Accepts only `?token=`, as behind a proxy that strips the header.
    QueryOnly,
}

#[derive(Debug, Clone)]
pub struct MockOptions {
    /// `None` disables authentication.
    pub token: Option<String>,
    pub agents: Vec<String>,
    pub ws_auth: MockAuth,
    pub pair_code: Option<String>,
}

impl Default for MockOptions {
    fn default() -> Self {
        Self {
            token: Some("test-token".into()),
            agents: vec!["default".into()],
            ws_auth: MockAuth::Any,
            pair_code: Some("123456".into()),
        }
    }
}

/// One scripted action of the mock gateway.
#[derive(Debug, Clone)]
pub enum Step {
    Send(Value),
    Text(String),
    Delay(Duration),
    /// Wait (up to 5 s) for the next client frame whose `type` matches.
    AwaitClient(String),
    Ping,
    /// A `chunk` frame padded to at least this many bytes.
    Oversize(usize),
    /// Close handshake, then end the connection.
    Close,
    /// End the connection with no close frame.
    Drop,
    /// Stop reading and writing but keep the connection open.
    Stall,
}

/// The frames the mock plays in answer to one client `message`.
#[derive(Debug, Clone, Default)]
pub struct Turn(pub Vec<Step>);

impl Turn {
    pub fn new() -> Self {
        Self::default()
    }

    /// agent_start, a few chunks and a done frame carrying `reply` and usage numbers.
    pub fn reply(reply: &str) -> Self {
        let mut t = Turn::new().frame(json!({"type":"agent_start","model_provider":"mock","model":"mock-model"}));
        for word in reply.split_inclusive(' ') {
            t = t.chunk(word);
        }
        t.done(reply)
    }

    pub fn step(mut self, step: Step) -> Self {
        self.0.push(step);
        self
    }

    pub fn frame(self, v: Value) -> Self {
        self.step(Step::Send(v))
    }

    pub fn chunk(self, text: &str) -> Self {
        self.frame(json!({"type":"chunk","content":text}))
    }

    pub fn thinking(self, text: &str) -> Self {
        self.frame(json!({"type":"thinking","content":text}))
    }

    pub fn tool_call(self, id: &str, name: &str, args: Value) -> Self {
        self.frame(json!({"type":"tool_call","id":id,"name":name,"args":args}))
    }

    pub fn tool_result(self, id: &str, name: &str, output: &str) -> Self {
        self.frame(json!({"type":"tool_result","id":id,"name":name,"output":output}))
    }

    /// An approval request followed by a wait for the client's `approval_response`.
    pub fn approval(self, request_id: &str, tool: &str, summary: &str) -> Self {
        self.frame(json!({
            "type":"approval_request","request_id":request_id,"tool":tool,
            "arguments_summary":summary,"arguments":{},"timeout_secs":300
        }))
        .step(Step::AwaitClient("approval_response".into()))
    }

    pub fn done(self, full_response: &str) -> Self {
        self.frame(json!({
            "type":"done","full_response":full_response,"input_tokens":100,"output_tokens":20,
            "tokens_used":120,"cost_usd":0.001,"model":"mock-model","provider":"mock",
            "max_context_tokens":1000,"last_input_tokens":100
        }))
    }

    pub fn error(self, code: &str, message: &str) -> Self {
        self.frame(json!({"type":"error","code":code,"message":message}))
    }

    pub fn aborted(self) -> Self {
        self.frame(json!({"type":"aborted"}))
    }

    pub fn delay_ms(self, ms: u64) -> Self {
        self.step(Step::Delay(Duration::from_millis(ms)))
    }

    pub fn drop_connection(self) -> Self {
        self.step(Step::Drop)
    }

    pub fn stall(self) -> Self {
        self.step(Step::Stall)
    }
}

/// How the mock treats one connection attempt.
#[derive(Debug, Clone, Default)]
pub struct ConnScript {
    /// Refuse the upgrade with this status and body.
    pub reject: Option<(u16, String)>,
    /// Answer `session_start` with this id regardless of the request.
    pub session_id: Option<String>,
    /// Played right after the `connected` ack.
    pub after_connect: Vec<Step>,
    /// Sleep before the upgrade completes.
    pub upgrade_delay: Option<Duration>,
    /// Play the steps from the very first byte and send no automatic handshake; used by replays.
    pub raw: bool,
}

impl ConnScript {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reject(mut self, status: u16, body: &str) -> Self {
        self.reject = Some((status, body.to_string()));
        self
    }

    pub fn wrong_session(mut self, id: &str) -> Self {
        self.session_id = Some(id.to_string());
        self
    }

    pub fn upgrade_delay(mut self, d: Duration) -> Self {
        self.upgrade_delay = Some(d);
        self
    }

    /// Sends `turn_resume` carrying `frames`, then the rest of the turn live.
    pub fn resume(mut self, frames: Vec<Value>, then: Turn) -> Self {
        self.after_connect.push(Step::Send(json!({"type":"turn_resume","frames":frames})));
        self.after_connect.extend(then.0);
        self
    }

    pub fn after_connect(mut self, turn: Turn) -> Self {
        self.after_connect.extend(turn.0);
        self
    }

    /// Replays a recording at its recorded gaps scaled by `speed`, waiting for each client frame it contains.
    pub fn replay(recording: &Recording, speed: f64) -> Self {
        let mut steps = Vec::new();
        for (entry, delay) in recording.entries.iter().zip(recording.delays(speed)) {
            if !delay.is_zero() {
                steps.push(Step::Delay(delay));
            }
            match entry.dir {
                crate::event::Direction::In => steps.push(Step::Send(entry.frame.clone())),
                crate::event::Direction::Out => {
                    let ty = entry.frame.get("type").and_then(Value::as_str).unwrap_or("").to_string();
                    steps.push(Step::AwaitClient(ty));
                }
            }
        }
        Self { after_connect: steps, raw: true, ..Self::default() }
    }
}

/// A frame the mock received from a client.
#[derive(Debug, Clone, PartialEq)]
pub struct Received {
    pub conn: usize,
    pub frame: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthUsed {
    Header,
    Query,
    Anonymous,
}

/// One WebSocket upgrade attempt.
#[derive(Debug, Clone)]
pub struct ConnInfo {
    pub id: usize,
    pub auth: AuthUsed,
    pub accepted: bool,
    pub status: u16,
    pub agent: Option<String>,
    pub requested_session: Option<String>,
    pub session_id: Option<String>,
    pub name: Option<String>,
}

/// One REST request the mock served.
#[derive(Debug, Clone)]
pub struct RestRequest {
    pub method: String,
    pub path: String,
    pub query: String,
    pub authorized: bool,
    pub body: Value,
}

struct RestOverride {
    status: u16,
    body: Value,
    delay: Duration,
}

struct Inner {
    conns: Vec<ConnInfo>,
    received: Vec<Received>,
    turns: VecDeque<Turn>,
    scripts: VecDeque<ConnScript>,
    sessions: Vec<Value>,
    messages: HashMap<String, Vec<Value>>,
    states: HashMap<String, (String, Option<String>)>,
    media: HashMap<String, Vec<u8>>,
    memory: Vec<Value>,
    requests: Vec<RestRequest>,
    overrides: HashMap<(String, String), RestOverride>,
    complete_reply: Option<String>,
    complete_script: VecDeque<(u16, Value, Duration)>,
    hints: Vec<String>,
    next_session: usize,
    live: usize,
}

struct Shared {
    opts: MockOptions,
    inner: Mutex<Inner>,
    notify: Notify,
    kick: watch::Sender<u64>,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn authorized(&self, headers: &HeaderMap) -> bool {
        match &self.opts.token {
            None => true,
            Some(t) => headers
                .get(header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer "))
                .is_some_and(|got| got == t),
        }
    }
}

fn now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    rfc3339_utc(secs)
}

/// A running mock gateway. Dropping it stops the server and closes every socket.
pub struct MockGateway {
    addr: SocketAddr,
    shared: Arc<Shared>,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<()>,
    scheme: &'static str,
}

impl MockGateway {
    pub async fn start() -> MockGateway {
        Self::start_with(MockOptions::default()).await
    }

    pub async fn start_with(opts: MockOptions) -> MockGateway {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind loopback");
        Self::start_on(opts, listener, false).await
    }

    /// Serves on a caller-supplied listener; `tls` selects the `https` scheme in `url()`.
    pub async fn start_on<L>(opts: MockOptions, listener: L, tls: bool) -> MockGateway
    where
        L: axum::serve::Listener<Addr = SocketAddr>,
    {
        let addr = listener.local_addr().expect("local addr");
        let (kick, _) = watch::channel(0u64);
        let shared = Arc::new(Shared {
            opts,
            inner: Mutex::new(Inner {
                conns: Vec::new(),
                received: Vec::new(),
                turns: VecDeque::new(),
                scripts: VecDeque::new(),
                sessions: Vec::new(),
                messages: HashMap::new(),
                states: HashMap::new(),
                media: HashMap::new(),
                memory: Vec::new(),
                requests: Vec::new(),
                overrides: HashMap::new(),
                complete_reply: None,
                complete_script: VecDeque::new(),
                hints: vec!["quick".into()],
                next_session: 1,
                live: 0,
            }),
            notify: Notify::new(),
            kick,
        });
        let app = Router::new().route("/ws/chat", get(ws_handler)).fallback(rest_handler).with_state(shared.clone());
        let (shutdown, rx) = tokio::sync::oneshot::channel::<()>();
        let task = tokio::spawn(async move {
            let _ = axum::serve(listener, app).with_graceful_shutdown(async move { let _ = rx.await; }).await;
        });
        MockGateway { addr, shared, shutdown: Some(shutdown), task, scheme: if tls { "https" } else { "http" } }
    }

    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// `http://127.0.0.1:port` (`https` for a TLS listener).
    pub fn url(&self) -> String {
        format!("{}://{}", self.scheme, self.addr)
    }

    pub fn token(&self) -> Option<&str> {
        self.shared.opts.token.as_deref()
    }

    /// A client config for this mock with millisecond timeouts and reconnect delays.
    pub fn config(&self) -> Config {
        let mut c = Config::new(self.url(), self.token().unwrap_or("")).with_agent("default");
        c.connect_timeout = Duration::from_secs(2);
        c.handshake_timeout = Duration::from_secs(2);
        c.request_timeout = Duration::from_secs(3);
        c.complete_timeout = Duration::from_secs(3);
        c.ping_interval = Duration::from_secs(5);
        c.pong_timeout = Duration::from_secs(15);
        c.resume_grace = Duration::from_millis(300);
        c.reconnect = ReconnectPolicy {
            enabled: true,
            initial: Duration::from_millis(20),
            max: Duration::from_millis(100),
            multiplier: 2.0,
            jitter: 0.0,
            max_attempts: Some(30),
        };
        c
    }


    /// Queues the answer to the next `message` frame; unscripted messages are echoed.
    pub fn script_turn(&self, turn: Turn) {
        self.shared.lock().turns.push_back(turn);
    }

    /// Queues the behavior of the next connection attempt.
    pub fn script_connection(&self, script: ConnScript) {
        self.shared.lock().scripts.push_back(script);
    }

    /// Abruptly ends every live socket.
    pub fn kick(&self) {
        self.shared.kick.send_modify(|n| *n += 1);
    }

    /// Frames received from clients, in arrival order.
    pub fn received(&self) -> Vec<Received> {
        self.shared.lock().received.clone()
    }

    pub fn received_of_type(&self, ty: &str) -> Vec<Value> {
        self.received()
            .into_iter()
            .map(|r| r.frame)
            .filter(|f| f.get("type").and_then(Value::as_str) == Some(ty))
            .collect()
    }

    /// Every upgrade attempt, accepted or refused.
    pub fn connections(&self) -> Vec<ConnInfo> {
        self.shared.lock().conns.clone()
    }

    /// Sockets currently open.
    pub fn live_connections(&self) -> usize {
        self.shared.lock().live
    }

    /// Waits until `pred` holds for the received frames or the timeout passes.
    pub async fn wait_received(&self, timeout: Duration, pred: impl Fn(&[Received]) -> bool) -> bool {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let notified = self.shared.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if pred(&self.shared.lock().received) {
                return true;
            }
            if tokio::time::timeout_at(deadline, notified).await.is_err() {
                return pred(&self.shared.lock().received);
            }
        }
    }


    pub fn add_session(&self, id: &str, agent: &str, message_count: u64, last_activity: &str) {
        self.shared.lock().sessions.push(json!({
            "session_id": id, "session_key": format!("gw_{id}"), "agent_alias": agent,
            "message_count": message_count, "created_at": last_activity, "last_activity": last_activity,
        }));
    }

    pub fn add_session_row(&self, row: Value) {
        self.shared.lock().sessions.push(row);
    }

    pub fn set_messages(&self, key: &str, messages: Vec<Value>) {
        self.shared.lock().messages.insert(key.to_string(), messages);
    }

    /// `state` is `idle`, `running` or `error`.
    pub fn set_state(&self, id: &str, state: &str, started: Option<&str>) {
        self.shared.lock().states.insert(id.to_string(), (state.to_string(), started.map(str::to_string)));
    }

    pub fn set_media(&self, session: &str, path: &str, bytes: Vec<u8>) {
        self.shared.lock().media.insert(format!("{session}\0{path}"), bytes);
    }

    pub fn set_complete_reply(&self, reply: &str) {
        self.shared.lock().complete_reply = Some(reply.to_string());
    }

    /// Queues the answer to the next `/api/complete` call; unscripted calls use the reply or echo.
    pub fn script_complete(&self, status: u16, body: Value, delay: Duration) {
        self.shared.lock().complete_script.push_back((status, body, delay));
    }

    /// Queues a successful `/api/complete` answer carrying `reply`.
    pub fn script_complete_reply(&self, reply: &str) {
        self.script_complete(200, json!({"response": reply, "model": "mock-model", "hint": "quick"}), Duration::ZERO);
    }

    /// Pins the response for `method path`, replacing the built-in handler.
    pub fn override_rest(&self, method: &str, path: &str, status: u16, body: Value, delay: Duration) {
        self.shared
            .lock()
            .overrides
            .insert((method.to_ascii_uppercase(), path.to_string()), RestOverride { status, body, delay });
    }

    pub fn requests(&self) -> Vec<RestRequest> {
        self.shared.lock().requests.clone()
    }

    pub fn memory(&self) -> Vec<Value> {
        self.shared.lock().memory.clone()
    }

    pub fn session_rows(&self) -> Vec<Value> {
        self.shared.lock().sessions.clone()
    }
}

impl Drop for MockGateway {
    fn drop(&mut self) {
        self.shared.kick.send_modify(|n| *n += 1);
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        self.task.abort();
    }
}


type Tx = SplitSink<WebSocket, Message>;

enum Played {
    Done,
    Dropped,
}

async fn ws_handler(
    State(sh): State<Arc<Shared>>,
    Query(q): Query<HashMap<String, String>>,
    headers: HeaderMap,
    ws: WebSocketUpgrade,
) -> Response {
    let bearer = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::to_string);
    let query_token = q.get("token").cloned();
    let auth_used = match (&bearer, &query_token) {
        (Some(_), _) if sh.opts.ws_auth != MockAuth::QueryOnly => AuthUsed::Header,
        (_, Some(_)) if sh.opts.ws_auth != MockAuth::HeaderOnly => AuthUsed::Query,
        _ => AuthUsed::Anonymous,
    };
    let provided = match auth_used {
        AuthUsed::Header => bearer,
        AuthUsed::Query => query_token,
        AuthUsed::Anonymous => None,
    };
    let authed = match &sh.opts.token {
        None => true,
        Some(t) => provided.as_deref() == Some(t.as_str()),
    };

    let (conn_id, script) = {
        let mut inner = sh.lock();
        let id = inner.conns.len() + 1;
        inner.conns.push(ConnInfo {
            id,
            auth: auth_used,
            accepted: false,
            status: 101,
            agent: q.get("agent").cloned(),
            requested_session: q.get("session_id").cloned(),
            session_id: None,
            name: q.get("name").cloned(),
        });
        (id, if authed { inner.scripts.pop_front().unwrap_or_default() } else { ConnScript::default() })
    };
    sh.notify.notify_waiters();
    let refuse = |status: u16, body: String| {
        let mut inner = sh.lock();
        if let Some(c) = inner.conns.iter_mut().find(|c| c.id == conn_id) {
            c.status = status;
        }
        drop(inner);
        sh.notify.notify_waiters();
        (StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_REQUEST), body).into_response()
    };

    if !authed {
        return refuse(401, "Unauthorized: provide Authorization header, Sec-WebSocket-Protocol bearer, or ?token= query param".into());
    }
    let Some(agent) = q.get("agent").filter(|a| !a.trim().is_empty()).cloned() else {
        return refuse(400, "Missing required `agent` query parameter".into());
    };
    if !sh.opts.agents.contains(&agent) {
        return refuse(400, format!("Unknown agent `{agent}` - no [agents.{agent}] entry configured."));
    }
    if let Some((status, body)) = script.reject.clone() {
        return refuse(status, body);
    }
    if let Some(delay) = script.upgrade_delay {
        tokio::time::sleep(delay).await;
    }

    let requested = q.get("session_id").cloned();
    let (session_id, resumed) = {
        let mut inner = sh.lock();
        let (sid, resumed) = match (&script.session_id, &requested) {
            (Some(forced), _) => (forced.clone(), false),
            (None, Some(r)) => {
                let known = inner.sessions.iter().any(|s| s["session_id"] == json!(r));
                (r.clone(), known)
            }
            (None, None) => {
                let n = inner.next_session;
                inner.next_session += 1;
                (format!("mock-session-{n}"), false)
            }
        };
        if !inner.sessions.iter().any(|s| s["session_id"] == json!(sid)) {
            let mut row = json!({
                "session_id": sid, "session_key": format!("gw_{sid}"), "agent_alias": agent,
                "message_count": 0, "created_at": now(), "last_activity": now(),
            });
            if let Some(name) = q.get("name") {
                row["name"] = json!(name);
            }
            inner.sessions.push(row);
        }
        if let Some(c) = inner.conns.iter_mut().find(|c| c.id == conn_id) {
            c.accepted = true;
            c.session_id = Some(sid.clone());
        }
        (sid, resumed)
    };

    let name = q.get("name").cloned();
    ws.on_upgrade(move |socket| async move {
        let mut kick = sh.kick.subscribe();
        sh.lock().live += 1;
        sh.notify.notify_waiters();
        tokio::select! {
            _ = kick.changed() => {}
            _ = serve_socket(socket, sh.clone(), conn_id, session_id, resumed, name, script) => {}
        }
        sh.lock().live -= 1;
        sh.notify.notify_waiters();
    })
}

struct AbortOnDrop(tokio::task::JoinHandle<()>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

async fn serve_socket(
    socket: WebSocket,
    sh: Arc<Shared>,
    conn: usize,
    session_id: String,
    resumed: bool,
    name: Option<String>,
    script: ConnScript,
) {
    let (mut tx, mut rx) = socket.split();
    let (frame_tx, mut frames) = mpsc::unbounded_channel::<Value>();
    let stall = Arc::new(Notify::new());
    let reader_stall = stall.clone();
    let reader_sh = sh.clone();
    let _reader = AbortOnDrop(tokio::spawn(async move {
        loop {
            tokio::select! {
                msg = rx.next() => match msg {
                    Some(Ok(Message::Text(text))) => {
                        let frame = serde_json::from_str::<Value>(&text).unwrap_or(Value::String(text.to_string()));
                        reader_sh.lock().received.push(Received { conn, frame: frame.clone() });
                        reader_sh.notify.notify_waiters();
                        let _ = frame_tx.send(frame);
                    }
                    Some(Ok(Message::Close(_))) | Some(Err(_)) | None => break,
                    Some(Ok(_)) => {}
                },
                _ = reader_stall.notified() => std::future::pending::<()>().await,
            }
        }
    }));

    if script.raw {
        if let Played::Done = play(script.after_connect, &mut tx, &mut frames, &stall).await {
            while frames.recv().await.is_some() {}
        }
        return;
    }

    let mut start = json!({
        "type":"session_start","session_id":session_id,"resumed":resumed,
        "message_count": message_count(&sh, &session_id),
    });
    if let Some(n) = &name {
        start["name"] = json!(n);
    }
    if send_json(&mut tx, &start).await.is_err() {
        return;
    }
    let mut after_connect = Some(script.after_connect);
    while let Some(frame) = frames.recv().await {
        match frame.get("type").and_then(Value::as_str) {
            Some("connect") => {
                if send_json(&mut tx, &json!({"type":"connected","message":"Connection established"})).await.is_err() {
                    return;
                }
                if let Some(steps) = after_connect.take()
                    && let Played::Dropped = play(steps, &mut tx, &mut frames, &stall).await
                {
                    return;
                }
            }
            Some("message") => {
                let content = frame.get("content").and_then(Value::as_str).unwrap_or("").to_string();
                let turn = sh.lock().turns.pop_front().unwrap_or_else(|| Turn::reply(&format!("echo: {content}")));
                let full = turn.0.iter().rev().find_map(|s| match s {
                    Step::Send(v) if v["type"] == "done" => v["full_response"].as_str().map(str::to_string),
                    _ => None,
                });
                record_user(&sh, &session_id, &content);
                if let Played::Dropped = play(turn.0, &mut tx, &mut frames, &stall).await {
                    return;
                }
                finish_turn(&sh, &session_id, full);
            }
            _ => {}
        }
    }
}

fn message_count(sh: &Shared, sid: &str) -> u64 {
    sh.lock().sessions.iter().find(|s| s["session_id"] == json!(sid)).and_then(|s| s["message_count"].as_u64()).unwrap_or(0)
}

fn record_user(sh: &Shared, sid: &str, content: &str) {
    let mut inner = sh.lock();
    let key = format!("gw_{sid}");
    inner.messages.entry(key).or_default().push(json!({"role":"user","content":content,"created_at":now()}));
    inner.states.insert(sid.to_string(), ("running".into(), Some(now())));
    if let Some(s) = inner.sessions.iter_mut().find(|s| s["session_id"] == json!(sid)) {
        s["message_count"] = json!(s["message_count"].as_u64().unwrap_or(0) + 1);
        s["last_activity"] = json!(now());
    }
}

fn finish_turn(sh: &Shared, sid: &str, full: Option<String>) {
    let mut inner = sh.lock();
    inner.states.insert(sid.to_string(), ("idle".into(), None));
    if let Some(full) = full {
        inner.messages.entry(format!("gw_{sid}")).or_default().push(json!({"role":"assistant","content":full,"created_at":now()}));
        if let Some(s) = inner.sessions.iter_mut().find(|s| s["session_id"] == json!(sid)) {
            s["message_count"] = json!(s["message_count"].as_u64().unwrap_or(0) + 1);
        }
    }
}

async fn send_json(tx: &mut Tx, v: &Value) -> Result<(), ()> {
    tx.send(Message::Text(v.to_string().into())).await.map_err(|_| ())
}

async fn play(
    steps: Vec<Step>,
    tx: &mut Tx,
    frames: &mut mpsc::UnboundedReceiver<Value>,
    stall: &Notify,
) -> Played {
    for step in steps {
        match step {
            Step::Send(v) => {
                if send_json(tx, &v).await.is_err() {
                    return Played::Dropped;
                }
            }
            Step::Text(s) => {
                if tx.send(Message::Text(s.into())).await.is_err() {
                    return Played::Dropped;
                }
            }
            Step::Delay(d) => tokio::time::sleep(d).await,
            Step::AwaitClient(ty) => {
                let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
                loop {
                    match tokio::time::timeout_at(deadline, frames.recv()).await {
                        Ok(Some(f)) if f.get("type").and_then(Value::as_str) == Some(ty.as_str()) => break,
                        Ok(Some(_)) => {}
                        Ok(None) => return Played::Dropped,
                        Err(_) => break,
                    }
                }
            }
            Step::Ping => {
                if tx.send(Message::Ping(Vec::new().into())).await.is_err() {
                    return Played::Dropped;
                }
            }
            Step::Oversize(n) => {
                let pad = "x".repeat(n);
                if send_json(tx, &json!({"type":"chunk","content":pad})).await.is_err() {
                    return Played::Dropped;
                }
            }
            Step::Close => {
                let _ = tx.send(Message::Close(Some(CloseFrame { code: 1000, reason: "mock close".into() }))).await;
                let _ = tx.flush().await;
                return Played::Dropped;
            }
            Step::Drop => return Played::Dropped,
            Step::Stall => {
                stall.notify_one();
                std::future::pending::<()>().await;
            }
        }
    }
    Played::Done
}


fn json_response(status: u16, body: Value) -> Response {
    (StatusCode::from_u16(status).unwrap_or(StatusCode::OK), [(header::CONTENT_TYPE, "application/json")], body.to_string())
        .into_response()
}

fn err(status: u16, message: &str) -> Response {
    json_response(status, json!({"error": message}))
}

async fn rest_handler(State(sh): State<Arc<Shared>>, method: Method, uri: Uri, headers: HeaderMap, body: Bytes) -> Response {
    let path = uri.path().to_string();
    let query_str = uri.query().unwrap_or("").to_string();
    let body_json: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    let authorized = sh.authorized(&headers);
    sh.lock().requests.push(RestRequest {
        method: method.to_string(),
        path: path.clone(),
        query: query_str.clone(),
        authorized,
        body: body_json.clone(),
    });
    sh.notify.notify_waiters();

    let pinned = sh.lock().overrides.get(&(method.to_string(), path.clone())).map(|o| (o.status, o.body.clone(), o.delay));
    if let Some((status, body, delay)) = pinned {
        tokio::time::sleep(delay).await;
        return json_response(status, body);
    }

    let query: HashMap<String, String> = query_str
        .split('&')
        .filter(|p| !p.is_empty())
        .map(|p| {
            let (k, v) = p.split_once('=').unwrap_or((p, ""));
            (decode_component(k), decode_component(&v.replace('+', " ")))
        })
        .collect();

    if path == "/health" && method == Method::GET {
        return json_response(200, json!({"status":"ok","paired":true,"require_pairing":sh.opts.token.is_some(),"runtime":{"mock":true}}));
    }
    if path == "/pair" && method == Method::POST {
        let code = headers.get("x-pairing-code").and_then(|v| v.to_str().ok()).unwrap_or("");
        return match (&sh.opts.pair_code, &sh.opts.token) {
            (Some(expected), Some(token)) if expected == code => {
                json_response(200, json!({"paired":true,"persisted":true,"token":token,"message":"Save this token"}))
            }
            _ => err(403, "Invalid pairing code"),
        };
    }
    if !authorized {
        return err(401, "Unauthorized");
    }

    let segments: Vec<String> = path.trim_matches('/').split('/').map(decode_component).collect();
    let segs: Vec<&str> = segments.iter().map(String::as_str).collect();
    match (method.as_str(), segs.as_slice()) {
        ("GET", ["api", "config", "agent-options"]) => json_response(
            200,
            json!({"agents": sh.opts.agents, "model_providers": [], "risk_profiles": [], "mcp_bundles": [], "channels": []}),
        ),
        ("GET", ["api", "config"]) => json_response(200, json!({"agents": sh.opts.agents, "gateway": {"paired_tokens": "***"}})),
        ("GET", ["api", "sessions"]) => json_response(200, json!({"sessions": sh.lock().sessions.clone()})),
        ("GET", ["api", "sessions", "running"]) => {
            let inner = sh.lock();
            let running: Vec<Value> = inner
                .states
                .iter()
                .filter(|(_, (s, _))| s == "running")
                .map(|(id, _)| json!({"session_id": id, "message_count": 0}))
                .collect();
            json_response(200, json!({"sessions": running}))
        }
        ("GET", ["api", "sessions", id, "messages"]) => {
            let inner = sh.lock();
            let messages = inner.messages.get(*id).or_else(|| inner.messages.get(&format!("gw_{id}"))).cloned().unwrap_or_default();
            json_response(200, json!({"session_id": id, "messages": messages, "session_persistence": true}))
        }
        ("GET", ["api", "sessions", id, "state"]) => {
            let inner = sh.lock();
            let known = inner.sessions.iter().any(|s| s["session_id"] == json!(id));
            match inner.states.get(*id) {
                Some((state, started)) => {
                    let mut v = json!({"session_id": id, "state": state});
                    if let Some(at) = started {
                        v["turn_started_at"] = json!(at);
                    }
                    json_response(200, v)
                }
                None if known => json_response(200, json!({"session_id": id, "state": "idle"})),
                None => err(404, "Session not found"),
            }
        }
        ("GET", ["api", "sessions", id, "media"]) => {
            let inner = sh.lock();
            match inner.media.get(&format!("{id}\0{}", query.get("path").cloned().unwrap_or_default())) {
                Some(bytes) => (StatusCode::OK, [(header::CONTENT_TYPE, "application/octet-stream")], bytes.clone()).into_response(),
                None => err(404, "image_file_unavailable"),
            }
        }
        ("POST", ["api", "sessions", id, "abort"]) => {
            let mut inner = sh.lock();
            let running = inner.states.get(*id).is_some_and(|(s, _)| s == "running");
            if running {
                inner.states.insert(id.to_string(), ("idle".into(), None));
                json_response(200, json!({"status":"aborted"}))
            } else {
                json_response(200, json!({"status":"no_active_response"}))
            }
        }
        ("PUT", ["api", "sessions", id]) => {
            let name = body_json["name"].as_str().unwrap_or("").trim().to_string();
            if name.is_empty() {
                return err(400, "name is required");
            }
            let mut inner = sh.lock();
            let by_key = inner.sessions.iter().position(|s| s["session_key"] == json!(id));
            let at = by_key.or_else(|| inner.sessions.iter().position(|s| s["session_id"] == json!(id)));
            match at.and_then(|i| inner.sessions.get_mut(i)) {
                Some(s) => {
                    s["name"] = json!(name);
                    json_response(200, json!({"session_id": id, "name": name}))
                }
                None => err(404, "Session not found"),
            }
        }
        ("DELETE", ["api", "sessions", id]) => {
            let mut inner = sh.lock();
            let before = inner.sessions.len();
            let by_key = inner.sessions.iter().any(|s| s["session_key"] == json!(id));
            inner.sessions.retain(|s| if by_key { s["session_key"] != json!(id) } else { s["session_id"] != json!(id) });
            inner.messages.remove(*id);
            inner.messages.remove(&format!("gw_{id}"));
            if inner.sessions.len() == before { err(404, "Session not found") } else { json_response(200, json!({"deleted": true})) }
        }
        ("POST", ["api", "complete"]) => {
            let scripted = sh.lock().complete_script.pop_front();
            match scripted {
                Some((status, body, delay)) => {
                    tokio::time::sleep(delay).await;
                    json_response(status, body)
                }
                None => complete(&sh, &body_json),
            }
        }
        ("GET", ["api", "memory"]) => {
            let inner = sh.lock();
            let recall = ["query", "since", "until"].iter().any(|k| query.contains_key(*k));
            let needle = query.get("query").map(|q| q.to_lowercase()).unwrap_or_default();
            let entries: Vec<Value> = inner
                .memory
                .iter()
                .filter(|e| query.get("agent").is_none_or(|a| e["agent"].as_str() == Some(a.as_str())))
                .filter(|e| query.get("category").is_none_or(|c| e["category"].as_str() == Some(c.as_str())))
                .filter(|e| {
                    !recall
                        || format!("{} {}", e["key"].as_str().unwrap_or(""), e["content"].as_str().unwrap_or(""))
                            .to_lowercase()
                            .contains(&needle)
                })
                .cloned()
                .collect();
            json_response(200, json!({"entries": entries}))
        }
        ("POST", ["api", "memory"]) => {
            let (Some(key), Some(content)) = (body_json["key"].as_str(), body_json["content"].as_str()) else {
                return err(422, "key and content are required");
            };
            let mut inner = sh.lock();
            inner.memory.retain(|e| e["key"] != json!(key) || e["agent"] != body_json["agent"]);
            inner.memory.push(json!({
                "key": key, "content": content,
                "category": body_json["category"].as_str().unwrap_or("core"),
                "agent": body_json["agent"], "timestamp": now(),
            }));
            json_response(200, json!({"status":"ok"}))
        }
        ("DELETE", ["api", "memory", key]) => {
            let mut inner = sh.lock();
            let agent = query.get("agent").cloned();
            let before = inner.memory.len();
            inner.memory.retain(|e| {
                !(e["key"] == json!(key) && agent.as_deref().is_none_or(|a| e["agent"].as_str() == Some(a)))
            });
            json_response(200, json!({"status":"ok","deleted": inner.memory.len() != before}))
        }
        _ => err(404, "not found"),
    }
}

fn complete(sh: &Shared, body: &Value) -> Response {
    let Some(prompt) = body["prompt"].as_str() else { return err(400, "invalid body: missing field `prompt`") };
    if prompt.trim().is_empty() {
        return err(400, "prompt is empty");
    }
    let size = prompt.len() + body["system"].as_str().map_or(0, str::len);
    if size > COMPLETE_INPUT_CAP {
        return err(400, &format!("input is {size} bytes; the cap is {COMPLETE_INPUT_CAP}"));
    }
    if body["temperature"].as_f64().is_some_and(|t| !(0.0..=2.0).contains(&t)) {
        return err(400, "temperature must be between 0.0 and 2.0");
    }
    let hint = body["hint"].as_str().map(str::trim).filter(|h| !h.is_empty()).unwrap_or("quick");
    let inner = sh.lock();
    if !inner.hints.iter().any(|h| h == hint) {
        return err(400, &format!("no [[model_routes]] entry has hint `{hint}`"));
    }
    let reply = inner.complete_reply.clone().unwrap_or_else(|| format!("echo: {prompt}"));
    json_response(200, json!({"response": reply, "model": "mock-model", "hint": hint}))
}
