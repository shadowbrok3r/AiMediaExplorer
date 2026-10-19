use std::time::Duration;

use reqwest::Method;
use serde_json::Value;

use crate::config::{Config, Endpoint, Token, redact_token};
use crate::error::{Error, Result};
use crate::history::{HistoryPager, StoredMessage, merge_by_time};
use crate::text::component;
use crate::tls::{install_crypto_provider, parse_pem_certs};
use crate::types::{
    AbortOutcome, AgentOptions, CompleteRequest, CompleteResponse, Health, MemoryEntry, MemoryQuery, MemoryStore,
    PairResult, SessionInfo, TurnState,
};

/// Largest error body read from a failed response.
const ERROR_BODY_LIMIT: usize = 8 * 1024;

/// Outcome of [`RestClient::probe`].
#[derive(Debug, Clone, PartialEq)]
pub struct Probe {
    pub health: Health,
    pub auth: AuthState,
    /// Agent aliases when the token was accepted.
    pub agents: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthState {
    /// No token configured; authentication was not checked.
    NotChecked,
    Accepted,
    Rejected(u16),
}

/// Authenticated REST access to a gateway. Cheap to clone.
#[derive(Clone)]
pub struct RestClient {
    http: reqwest::Client,
    endpoint: Endpoint,
    token: Option<Token>,
    request_timeout: Duration,
    complete_timeout: Duration,
    max_body: usize,
}

impl std::fmt::Debug for RestClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RestClient").field("endpoint", &self.endpoint.host_port()).finish_non_exhaustive()
    }
}

impl RestClient {
    pub fn new(config: &Config) -> Result<Self> {
        install_crypto_provider();
        let endpoint = config.endpoint()?;
        let mut builder = reqwest::Client::builder()
            .user_agent(concat!("zeroclaw-client/", env!("CARGO_PKG_VERSION")))
            .connect_timeout(config.connect_timeout)
            .timeout(config.request_timeout)
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy();
        if let Some(pem) = &config.extra_root_pem {
            parse_pem_certs(pem)?;
            for cert in reqwest::Certificate::from_pem_bundle(pem.as_bytes())
                .map_err(|e| Error::Config(format!("invalid root certificate: {e}")))?
            {
                builder = builder.add_root_certificate(cert);
            }
        }
        let http = builder.build().map_err(|e| Error::Config(format!("cannot build HTTP client: {e}")))?;
        Ok(Self {
            http,
            endpoint,
            token: config.token().cloned(),
            request_timeout: config.request_timeout,
            complete_timeout: config.complete_timeout,
            max_body: config.max_rest_body_bytes,
        })
    }

    pub fn endpoint(&self) -> &Endpoint {
        &self.endpoint
    }

    async fn call(&self, req: Req<'_>) -> Result<Vec<u8>> {
        let url = self.endpoint.http_url(req.path, req.query);
        let mut builder = self.http.request(req.method.clone(), url);
        if req.auth
            && let Some(t) = &self.token
        {
            builder = builder.bearer_auth(t.expose());
        }
        for (k, v) in req.headers {
            builder = builder.header(*k, *v);
        }
        if let Some(body) = req.body {
            builder = builder.json(body);
        }
        if let Some(t) = req.timeout {
            builder = builder.timeout(t);
        }
        let limit = req.limit.unwrap_or(self.max_body);
        let mut response = builder.send().await.map_err(|e| self.map_err(e))?;
        let status = response.status();
        let read_limit = if status.is_success() { limit } else { ERROR_BODY_LIMIT };
        let too_large = || Error::ResponseTooLarge { path: req.path.to_string(), limit: read_limit };
        if status.is_success() && response.content_length().is_some_and(|n| n > read_limit as u64) {
            return Err(too_large());
        }
        let mut bytes = Vec::new();
        loop {
            match response.chunk().await {
                Ok(Some(chunk)) => {
                    if bytes.len() + chunk.len() > read_limit {
                        if status.is_success() {
                            return Err(too_large());
                        }
                        bytes.extend_from_slice(&chunk[..read_limit - bytes.len()]);
                        break;
                    }
                    bytes.extend_from_slice(&chunk);
                }
                Ok(None) => break,
                Err(e) => return Err(self.map_err(e)),
            }
        }
        if status.is_success() {
            return Ok(bytes);
        }
        let message = error_message(&bytes);
        let message = redact_token(&message, self.token.as_ref());
        Err(match status.as_u16() {
            401 => Error::Unauthorized,
            403 => Error::Forbidden,
            code => Error::Http { status: code, path: req.path.to_string(), message },
        })
    }

    fn map_err(&self, e: reqwest::Error) -> Error {
        let text = redact_token(&e.to_string(), self.token.as_ref());
        if e.is_timeout() { Error::Timeout("request") } else { Error::Connect(text) }
    }

    async fn json(&self, req: Req<'_>) -> Result<Value> {
        let path = req.path.to_string();
        let bytes = self.call(req).await?;
        serde_json::from_slice(&bytes).map_err(|e| Error::Decode { path, message: e.to_string() })
    }

    fn get<'a>(&self, path: &'a str) -> Req<'a> {
        Req::new(Method::GET, path)
    }

    /// `GET /health`, which needs no token.
    pub async fn health(&self) -> Result<Health> {
        let v = self.json(Req { auth: false, ..self.get("/health") }).await?;
        serde_json::from_value(v).map_err(|e| Error::Decode { path: "/health".into(), message: e.to_string() })
    }

    /// `GET /api/config/agent-options`, used as the authenticated token check.
    pub async fn agent_options(&self) -> Result<AgentOptions> {
        let v = self.json(self.get("/api/config/agent-options")).await?;
        serde_json::from_value(v)
            .map_err(|e| Error::Decode { path: "/api/config/agent-options".into(), message: e.to_string() })
    }

    pub async fn agents(&self) -> Result<Vec<String>> {
        Ok(self.agent_options().await?.agents)
    }

    /// Health plus token check; a refused token is reported in the result, an unreachable gateway is an error.
    pub async fn probe(&self) -> Result<Probe> {
        let health = self.health().await?;
        if self.token.is_none() {
            return Ok(Probe { health, auth: AuthState::NotChecked, agents: Vec::new() });
        }
        match self.agent_options().await {
            Ok(options) => Ok(Probe { health, auth: AuthState::Accepted, agents: options.agents }),
            Err(Error::Unauthorized) => Ok(Probe { health, auth: AuthState::Rejected(401), agents: Vec::new() }),
            Err(Error::Forbidden) => Ok(Probe { health, auth: AuthState::Rejected(403), agents: Vec::new() }),
            Err(e) => Err(e),
        }
    }

    /// `GET /api/config`, secrets masked by the gateway.
    pub async fn config_get(&self) -> Result<Value> {
        self.json(self.get("/api/config")).await
    }

    pub async fn sessions(&self) -> Result<Vec<SessionInfo>> {
        let v = self.json(self.get("/api/sessions")).await?;
        Ok(v.get("sessions")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|s| serde_json::from_value(s.clone()).ok())
            .collect())
    }

    /// One session's list row, matched on its id or store key; `None` when the gateway does not list it.
    pub async fn session(&self, id: &str) -> Result<Option<SessionInfo>> {
        Ok(self.sessions().await?.into_iter().find(|s| s.session_id == id || s.key() == id))
    }

    /// Ids of the sessions with a turn running now.
    pub async fn running(&self) -> Result<Vec<String>> {
        let v = self.json(self.get("/api/sessions/running")).await?;
        Ok(v.get("sessions")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|s| s.get("session_id").and_then(Value::as_str).map(str::to_string))
            .collect())
    }

    pub async fn session_state(&self, id: &str) -> Result<TurnState> {
        let path = format!("/api/sessions/{}/state", component(id));
        TurnState::parse(&self.json(self.get(&path)).await?)
    }

    /// Stops the running turn (`POST /api/sessions/{id}/abort`).
    pub async fn abort(&self, id: &str) -> Result<AbortOutcome> {
        let path = format!("/api/sessions/{}/abort", component(id));
        let v = self.json(Req::new(Method::POST, &path)).await?;
        match v.get("status").and_then(Value::as_str) {
            Some("aborted") => Ok(AbortOutcome::Aborted),
            Some("no_active_response") => Ok(AbortOutcome::NothingRunning),
            _ => Err(Error::Decode { path, message: "gateway did not acknowledge the abort".into() }),
        }
    }

    pub async fn rename_session(&self, id: &str, name: &str) -> Result<()> {
        let path = format!("/api/sessions/{}", component(id));
        let body = serde_json::json!({ "name": name });
        self.call(Req { body: Some(&body), ..Req::new(Method::PUT, &path) }).await.map(|_| ())
    }

    /// Deletes a session; `false` when it was already gone.
    pub async fn delete_session(&self, id: &str) -> Result<bool> {
        let path = format!("/api/sessions/{}", component(id));
        match self.call(Req::new(Method::DELETE, &path)).await {
            Ok(_) => Ok(true),
            Err(Error::Http { status: 404, .. }) => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// Stored messages of one store, oldest first. `tool_context` interleaves tool calls, results and reasoning.
    pub async fn messages(&self, key: &str, tool_context: bool) -> Result<Vec<StoredMessage>> {
        let path = format!("/api/sessions/{}/messages", component(key));
        let query: &[(&str, &str)] = if tool_context { &[("tool_context", "true")] } else { &[] };
        let v = self.json(Req { query, ..self.get(&path) }).await?;
        Ok(parse_messages(&v))
    }

    /// Messages of every store under one conversation, interleaved by time. Fails only when no store could be read.
    pub async fn messages_merged(&self, keys: &[String], tool_context: bool) -> Result<Vec<StoredMessage>> {
        let mut all = Vec::new();
        let mut last_error = None;
        let mut read_any = false;
        for key in keys {
            match self.messages(key, tool_context).await {
                Ok(m) => {
                    read_any = true;
                    all.extend(m);
                }
                Err(e) => last_error = Some(e),
            }
        }
        match (read_any, last_error) {
            (false, Some(e)) => Err(e),
            _ => Ok(merge_by_time(all)),
        }
    }

    /// A lazily paged transcript: newest `page_size` messages first, older pages on demand.
    pub async fn history(&self, id: &str, page_size: usize, tool_context: bool) -> Result<HistoryPager> {
        Ok(HistoryPager::new(self.messages(id, tool_context).await?, page_size))
    }

    /// Bytes of an image the session's output referenced by path.
    pub async fn media(&self, id: &str, path: &str) -> Result<Vec<u8>> {
        let route = format!("/api/sessions/{}/media", component(id));
        self.call(Req { query: &[("path", path)], ..self.get(&route) }).await
    }

    /// `POST /api/complete`: one model call, validated against the gateway's limits first.
    pub async fn complete(&self, request: &CompleteRequest) -> Result<CompleteResponse> {
        request.validate()?;
        let body = serde_json::to_value(request).map_err(|e| Error::Config(e.to_string()))?;
        let v = self
            .json(Req { body: Some(&body), timeout: Some(self.complete_timeout), ..Req::new(Method::POST, "/api/complete") })
            .await?;
        serde_json::from_value(v).map_err(|e| Error::Decode { path: "/api/complete".into(), message: e.to_string() })
    }

    /// Lists memory entries, or recalls them when a query or time range is set.
    pub async fn memory(&self, q: &MemoryQuery) -> Result<Vec<MemoryEntry>> {
        let mut query: Vec<(&str, &str)> = Vec::new();
        for (key, value) in [
            ("agent", &q.agent),
            ("query", &q.query),
            ("category", &q.category),
            ("since", &q.since),
            ("until", &q.until),
        ] {
            if let Some(v) = value {
                query.push((key, v.as_str()));
            }
        }
        let v = self.json(Req { query: &query, ..self.get("/api/memory") }).await?;
        Ok(v.get("entries")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|e| serde_json::from_value(e.clone()).ok())
            .collect())
    }

    pub async fn memory_store(&self, entry: &MemoryStore) -> Result<()> {
        let body = serde_json::to_value(entry).map_err(|e| Error::Config(e.to_string()))?;
        self.call(Req { body: Some(&body), ..Req::new(Method::POST, "/api/memory") }).await.map(|_| ())
    }

    /// Deletes one entry; `true` when it existed.
    pub async fn memory_delete(&self, key: &str, agent: Option<&str>) -> Result<bool> {
        let path = format!("/api/memory/{}", component(key));
        let query: Vec<(&str, &str)> = agent.map(|a| ("agent", a)).into_iter().collect();
        let v = self.json(Req { query: &query, ..Req::new(Method::DELETE, &path) }).await?;
        Ok(v.get("deleted").and_then(Value::as_bool).unwrap_or(false))
    }

    /// Exchanges a one-time pairing code for a bearer token. Sends no credentials.
    pub async fn pair(config: &Config, code: &str) -> Result<PairResult> {
        let mut anonymous = config.clone();
        anonymous.set_token("");
        let client = RestClient::new(&anonymous)?;
        let v = client
            .json(Req { headers: &[("X-Pairing-Code", code)], auth: false, ..Req::new(Method::POST, "/pair") })
            .await?;
        serde_json::from_value(v).map_err(|e| Error::Decode { path: "/pair".into(), message: e.to_string() })
    }

    pub fn request_timeout(&self) -> Duration {
        self.request_timeout
    }
}

struct Req<'a> {
    method: Method,
    path: &'a str,
    query: &'a [(&'a str, &'a str)],
    headers: &'a [(&'a str, &'a str)],
    body: Option<&'a Value>,
    auth: bool,
    timeout: Option<Duration>,
    limit: Option<usize>,
}

impl<'a> Req<'a> {
    fn new(method: Method, path: &'a str) -> Self {
        Self { method, path, query: &[], headers: &[], body: None, auth: true, timeout: None, limit: None }
    }
}

/// `{"error": "..."}` bodies give their message; anything else is cut to a readable length.
fn error_message(body: &[u8]) -> String {
    let text = String::from_utf8_lossy(body);
    let message = serde_json::from_str::<Value>(&text)
        .ok()
        .and_then(|v| v.get("error").and_then(Value::as_str).map(str::to_string))
        .unwrap_or_else(|| text.trim().to_string());
    message.chars().take(300).collect()
}

/// Rows of a `{"messages":[...]}` body; entries that are not objects are skipped and null fields read as empty.
pub fn parse_messages(v: &Value) -> Vec<StoredMessage> {
    v.get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|m| m.is_object())
        .map(|m| {
            let s = |k: &str| m.get(k).and_then(Value::as_str);
            StoredMessage {
                role: s("role").unwrap_or("").to_string(),
                content: s("content").unwrap_or("").to_string(),
                created_at: s("created_at").map(str::to_string),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn error_message_prefers_the_error_field() {
        assert_eq!(error_message(br#"{"error":"Session not found"}"#), "Session not found");
        assert_eq!(error_message(b"plain failure \n"), "plain failure");
        assert_eq!(error_message(&vec![b'x'; 1000]).len(), 300);
        assert_eq!(error_message(b""), "");
    }

    #[test]
    fn messages_parse_tolerates_nulls_and_junk() {
        let v = json!({"messages":[
            {"role":"user","content":"hi","created_at":"2026-09-11T01:00:00+00:00"},
            {"role":"assistant","content":null},
            "junk",
            {"content":"no role"}
        ]});
        let m = parse_messages(&v);
        assert_eq!(m.len(), 3);
        assert_eq!(m[0].created_at.as_deref(), Some("2026-09-11T01:00:00+00:00"));
        assert_eq!(m[1].content, "");
        assert_eq!(m[2].role, "");
        assert!(parse_messages(&json!({})).is_empty());
    }

    #[test]
    fn client_builds_from_config_and_rejects_bad_input() {
        assert!(RestClient::new(&Config::new("127.0.0.1:1", "t")).is_ok());
        assert!(RestClient::new(&Config::new("", "t")).is_err());
        let mut bad_pem = Config::new("127.0.0.1:1", "t");
        bad_pem.extra_root_pem = Some("nope".into());
        assert!(RestClient::new(&bad_pem).is_err());
    }
}
