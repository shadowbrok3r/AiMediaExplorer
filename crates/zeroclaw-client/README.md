# zeroclaw-client

Client for the ZeroClaw gateway (`/ws/chat` plus the REST API). Pure Rust: no UI, no GPU, no dependency on the app crate. Pinned against the fork at `f076d7b55` (workspace v0.8.5); the wire details come from `docs/research/zeroclaw-integration.md` and were checked against `crates/zeroclaw-gateway/src/{ws,ws_hub,api,api_complete}.rs`.

| Module | Content |
| --- | --- |
| `config` | `Config` (token never printed by `Debug`), `Endpoint` parsing, `ReconnectPolicy`, `WsAuth` |
| `chat` | `ChatSession`: connect, send, approvals, abort, reconnect, keepalive; `EventStream` |
| `frames`, `event` | wire frames (`ServerFrame` keeps unknown frames), typed `Event`s, the frame-to-event `Translator` |
| `rest` | `RestClient`: health, sessions, history pager, `/api/complete`, memory, pairing, config read |
| `history`, `attachments`, `text`, `types` | `unwrap_stored`, `strip_date_stamp`, percent-encoding, `[IMAGE:..]` markers, REST structs |
| `tls` | `install_crypto_provider()` |
| `replay`, `mock` (feature `mock`) | JSONL recordings; loopback mock gateway |
| `zcprobe` (binary) | command line probe |

## Chat

```rust,no_run
use zeroclaw_client::{ChatSession, Config, ConnectOptions, Event};

async fn chat() -> Result<(), zeroclaw_client::Error> {
    let token = std::env::var("ZC_TOKEN").unwrap_or_default();
    let cfg = Config::new("100.96.121.37:42617", token).with_agent("media");
    let (session, mut events) = ChatSession::connect(cfg, ConnectOptions::default()).await?;
    session.send_text("find sunsets").await?;
    while let Some(event) = events.next().await {
        match event {
            Event::Token { text, .. } => print!("{text}"),
            Event::ApprovalRequest(req) => session.deny(&req.request_id).await?,
            Event::TurnResumed { .. } => { /* drop what was rendered for this turn; it is replayed */ }
            Event::Done { usage, .. } => {
                println!("\ncontext used: {:?}", usage.context_fraction());
                break;
            }
            Event::Disconnected { will_retry: false, .. } => break,
            _ => {}
        }
    }
    session.close().await;
    Ok(())
}
```

`ChatSession::connect` opens the first connection itself, so a refused token, unknown agent or session mismatch is returned from that call. After that a supervisor task reconnects on its own and reports through the stream. `EventStream` also implements `futures_core::Stream` and has `try_next()` for polling once per UI frame.

* `Config::agent` is required: the gateway has no default agent.
* Token: `Authorization: Bearer` first; with `WsAuth::Auto` a 401 retries once with `?token=` and the working mode is remembered for reconnects. The token is redacted from every error string.
* Resume: `ConnectOptions::session_id` becomes `?session_id=`; the `session_start` id is compared with the requested one (`Error::SessionMismatch`). Reconnects resume the same id.
* Messages: `send_user_message(text, &[InlineImage])` puts images inside the text as `[IMAGE:data:<mime>;base64,..]` (PNG, JPEG, GIF, WebP by magic bytes, at most 8 MiB each); `send_text` is the text-only form. While a turn runs a message is steering. A send is never queued or replayed across a reconnect (`Error::NotConnected`).
* Approvals: `approve`, `approve_always`, `deny` send `approval_response`. The gateway denies on its own after `timeout_secs` (default 300).
* Stop: `abort()` is `POST /api/sessions/{id}/abort`; the socket has no stop frame. An `Aborted` event follows.
* Buffers: events go through a bounded channel (`event_buffer`, default 1024); a slow reader stops socket reads (never unbounded memory). Inbound frames over `max_frame_bytes` (16 MiB) end the connection with `Event::Error{code:"FRAME_TOO_LARGE"}` and no reconnect. Outbound messages over `max_outbound_bytes` (24 MiB) fail with `Error::TooLarge`.
* Keepalive: WebSocket ping every `ping_interval` (20 s); no inbound traffic for `pong_timeout` (60 s) after a ping is a lost connection. Reconnect backoff is 0.5 s doubling to 30 s with 20% downward jitter, unlimited attempts (`ReconnectPolicy`).
* Mid-turn drops: the turn survives on the gateway. After reconnecting the gateway sends `turn_resume` with the whole turn so far. `Event::TurnResumed` means "forget what you rendered for this turn" and also marks a turn joined mid-way. If no `turn_resume` arrives within `resume_grace` (10 s) the client asks `GET /api/sessions/{id}/state`; an idle session yields `Event::TurnLost` (reload history).
* Debug tap: `ConnectOptions::raw_tap` adds `Event::Raw` for every frame in both directions; `replay::Recording::from_raw(..).to_jsonl()` writes the recording format below.

### Frames

| Dir | `type` | Fields | Becomes |
| --- | --- | --- | --- |
| out | `connect` | `device_name`, `capabilities`, `cwd` | sent on open |
| out | `message` | `content` | `send_user_message` / `send_text` |
| out | `approval_response` | `request_id`, `decision` (`approve`, `always`, `deny`) | `respond_approval` |
| in | `session_start` | `session_id`, `resumed`, `message_count`, `name?` | `Event::Connected` |
| in | `connected` | `message` | ignored |
| in | `agent_start` / `agent_end` | `model_provider`, `model` | `TurnStarted` / ignored |
| in | `thinking`, `chunk` | `content` | `Thinking`, `Token` (`segment` counts tool calls so far) |
| in | `tool_call` | `id`, `name`, `args` | `ToolCall` (`index` within the turn) |
| in | `tool_result` | `id`, `name`, `output` | `ToolResult` (`index` of the call with that id) |
| in | `approval_request` | `request_id`, `tool`, `arguments_summary`, `arguments`, `timeout_secs` | `ApprovalRequest` |
| in | `plan` | `entries[{content,status}]` | `Plan` |
| in | `history_trimmed` | `dropped_messages`, `kept_turns`, `reason` | `HistoryTrimmed` |
| in | `turn_resume` | `frames[]` | `TurnResumed`, then the nested frames |
| in | `done` | `full_response`, `input_tokens`, `output_tokens`, `tokens_used`, `cost_usd`, `model`, `provider`, `max_context_tokens`, `last_input_tokens` | `Done{full_response, usage}` |
| in | `aborted` | | `Aborted` |
| in | `error` | `message`, `code` | `Error{ends_turn}`; `PROVIDER_ERROR`, `AUTH_ERROR`, `AGENT_ERROR` end the turn; `AGENT_INIT_FAILED`, `INVALID_CWD`, `NEEDS_ONBOARDING` end the connection (no reconnect) |
| in | anything else | | `Unknown{kind, raw}` |

`Done.full_response` is authoritative over the summed tokens. Turn numbers start at 0 and advance on `Done`, `Aborted` and turn-ending errors.

## REST

```rust,no_run
use zeroclaw_client::{CompleteRequest, Config, RestClient, group_sessions};

async fn rest() -> Result<(), zeroclaw_client::Error> {
    let cfg = Config::new("https://claw.example.app", "token").with_agent("media");
    let rest = RestClient::new(&cfg)?;
    let probe = rest.probe().await?;
    println!("{:?} {:?}", probe.auth, probe.agents);

    let rows = group_sessions(Some("media"), &rest.sessions().await?);
    if let Some(row) = rows.first() {
        let mut pager = rest.history(&row.id, 50, false).await?;
        for item in pager.next_page() {
            println!("{}: {}", item.id, item.text);
        }
    }

    let reply = rest.complete(&CompleteRequest::new("Suggest three tags for a sunset photo")).await?;
    println!("{}", reply.response);
    Ok(())
}
```

* `complete` checks the gateway's limits first: `system` plus `prompt` at most 64 KiB (`COMPLETE_INPUT_CAP`), temperature 0 to 2, a non-empty prompt. `CompleteRequest::with_image` appends an `[IMAGE:..]` marker and re-checks the cap.
* History: the gateway has no paging parameters, so `history()` fetches the transcript once and `HistoryPager` serves the newest page first, parsing only the pages asked for. `messages_merged` joins the `gw_` twin store and the channel store by timestamp.
* Memory (`memory`, `memory_store`, `memory_delete`), `pair`, `config_get` (secrets masked by the gateway) exist. Config writes are types only: `ConfigPatch` and `PatchOp` build an RFC 6902 body; nothing in this crate sends `PATCH /api/config`.
* HTTP: redirects are never followed, system proxies are ignored, response bodies are capped (`max_rest_body_bytes`, 24 MiB), 401 and 403 map to `Error::Unauthorized` and `Error::Forbidden`.

## TLS

`install_crypto_provider()` installs the ring provider once, idempotently, and leaves a provider installed by someone else alone. `ChatSession::connect` and `RestClient::new` call it. The app lock holds both `ring` and `aws-lc-rs`, and rustls then panics in `ClientConfig::builder()` at first use; `tests/tls_provider.rs` reproduces that and `tests/tls_autoinstall.rs` proves the client avoids it. WebSocket TLS uses an explicit `ClientConfig` (webpki roots plus `Config::extra_root_pem`), HTTPS uses reqwest with the platform verifier plus the same extra root.

## zcprobe

```text
zcprobe [--url URL] [--token TOKEN | --token-file PATH] [--agent ALIAS] [--timeout S] [--json] <command>
  health                  reachability, token check, agent list
  sessions [--all]        sessions of --agent (empty shells hidden unless --all)
  history <id> [--limit N] [--tool-context]
  complete <prompt> [--hint H] [--system S] [--temperature T]
  chat <prompt> [--session ID] [--image PATH].. [--auto-approve] [--thinking]
  frames [<prompt>] [--session ID] [--seconds N] [--auto-approve]
```

`ZC_URL`, `ZC_TOKEN` and `ZC_AGENT` fill in missing flags. Exit codes: 0 ok, 1 failure, 2 usage, 3 credentials refused, 4 gateway unreachable. The token is never printed: all output is scrubbed, and `--token-file` or `ZC_TOKEN` keep it out of the process list. `chat` writes the answer to stdout and tool, approval and usage lines to stderr (`--json`: one event per line on stdout); approvals are denied unless `--auto-approve`. `frames` writes the recording format:

```text
{"t_ms":0,"dir":"in","frame":{"type":"session_start","session_id":"..","resumed":false,"message_count":0}}
{"t_ms":3,"dir":"out","frame":{"type":"connect","device_name":"zcprobe","capabilities":["text"]}}
```

## Mock gateway and fixtures

```rust,no_run
use zeroclaw_client::mock::{MockGateway, Turn};
use zeroclaw_client::{ChatSession, ConnectOptions};

async fn scripted() {
    let mock = MockGateway::start().await;
    mock.script_turn(Turn::new().thinking("hm").chunk("Hello").done("Hello"));
    let (session, _events) = ChatSession::connect(mock.config(), ConnectOptions::default()).await.unwrap();
    session.send_text("hi").await.unwrap();
}
```

Feature `mock` serves `/ws/chat` and the REST routes on one loopback port. Scripts: `Turn` (frames played for the next `message`: chunks, tool events, `approval` that waits for the response, `error`, `delay_ms`, `drop_connection`, `stall`, `Step::Oversize`) and `ConnScript` (per connection: `reject(status, body)`, `wrong_session`, `upgrade_delay`, `resume(frames, then)`, `replay(recording, speed)`). `MockOptions::ws_auth` simulates a proxy that strips the header or the query. REST state: `add_session`, `set_messages`, `set_state`, `set_media`, `override_rest`; `requests()`, `received()` and `connections()` record what clients sent. `start_on` serves any listener (the TLS tests use it).

`tests/fixtures/*.jsonl` are recordings hand-written from the documented frames (plain turn, tool calls with approval, `turn_resume`, error codes with unknown frames, aborted turn). `tests/replay.rs` replays each through the mock into a real `ChatSession` and compares the events; record new ones with `zcprobe frames`.

## Tests

```text
cargo test -p zeroclaw-client        # offline, no GPU, a few seconds
ZC_URL=.. ZC_TOKEN_FILE=.. ZC_AGENT=media cargo test -p zeroclaw-client --test live -- --ignored --nocapture
```

`tests/live.rs` holds one smoke body (probe, one turn, resume, history, abort, `/api/complete`) that runs against the mock on every `cargo test` and against a real gateway with `--ignored`.

## Drift

The gateway builds `/ws/chat` frames with `json!`; there are no shared serde types, and `turn_resume` is a fork feature.

* Unknown frame types, and known types whose fields changed shape, are kept whole as `ServerFrame::Unknown` / `Event::Unknown`. Extra fields are ignored. Scalar fields accept null, numbers and booleans.
* Field aliases are accepted where consumers saw both spellings: `tool_call.name|tool`, `args|arguments`, `tool_result.output|result|content`.
* Resume works only through `?session_id=`; `?session=` and a session id inside `connect` are ignored by the gateway and mint a new session, which `Error::SessionMismatch` reports.
* `POST /api/sessions/{id}/messages` appends an assistant notification and does not run a turn; it is not wrapped.
* Stored assistant and tool messages are JSON envelopes (`unwrap_stored`); user messages carry an injected date preamble (`strip_date_stamp`).
* When a gateway upgrade changes behavior, add a recording under `tests/fixtures` with `zcprobe frames` and a case in `tests/replay.rs`.
* Not verified against a live gateway: everything here runs against the mock and hand-written recordings. Live checks to do once a token is available: header auth through `wss://claw.shadowbroker.app`, `turn_resume` after a real drop, vision on the `media` agent.
