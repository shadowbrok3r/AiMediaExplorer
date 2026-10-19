# ZeroClaw as the default AI backend of AiMediaExplorer

Study 5 of 5. Read-only research. Everything below was read in code; items not verified are marked **[UNVERIFIED]** and listed again in section 7 as spikes.

Path aliases used for citations:

| Alias | Path |
| --- | --- |
| `ZC` | `/home/shadowbroker/Desktop/zeroclaw` (ZeroClaw fork, branch `main`, `f076d7b55`, workspace v0.8.5) |
| `ZCX` | `/home/shadowbroker/Desktop/zc-codex` (Codex bridge + ZeroClaw phone client) |
| `CUI` | `/home/shadowbroker/Documents/Rust/Mobile/comfyui-android` |
| `MT` | `/home/shadowbroker/Documents/Rust/WorkProjects/MastertechProject` |
| `AME` | `/home/shadowbroker/Documents/Rust/AiMediaExplorer` (HEAD `1d1a1db`) |

---

## 0. TL;DR

- ZeroClaw is a single-binary agent runtime. Clients reach it through an **HTTP+WebSocket gateway on `:42617`** (pairing-token bearer auth). Chat = `GET /ws/chat?agent=<alias>` (JSON text frames). Everything else (sessions, one-shot completion, memory, health) is REST.
- Tools reach a ZeroClaw agent only through **ZeroClaw's own tool set + MCP servers that ZeroClaw connects to as a client** (`<server>__<tool>` names, per-agent `mcp_bundles`, per-tool `auto_approve`/`always_ask`). `/ws/chat` has **no client-side tool-call frames**; the only server-to-client questions are `approval_request` frames.
- The user's gateway is **remote** (`ubuntu-ai-amd`, tailnet `100.96.121.37:42617`, public `wss://claw.shadowbroker.app`). The desktop is on the tailnet. So AiMediaExplorer must host its own **MCP server (Streamable HTTP) reachable from the gateway**, not the current stdio one.
- Three existing consumers prove the wire: `ZCX/android/src/zeroclaw.rs` (full chat client, 2098 lines), `CUI/src/zeroclaw.rs` (one-turn client), `MT` (REST viewer + `/api/complete` + memory + hosts its own MCP at `:9004`). The same helper code (`unwrap_stored`, `strip_date_stamp`, percent-encoding) is already **copied three times**.
- AME's current AI surface has defects that must be fixed before the swap (section 3.2): MCP stdio server started twice on the GUI process's own stdio, dotted tool names that ZeroClaw's OpenRouter provider silently drops, assistant sends **no conversation history**, API key written to the in-app log at ERROR level, tokens not tagged per turn.
- Recommended shape: new workspace crate `crates/zeroclaw-client` (no egui, no CUDA, fast tests) + app glue module `src/zc/`; default backend = `Auto` resolving to ZeroClaw; local GPU models stay local and are exposed to the agent as ~12 MCP tools; writes go through the existing Refine proposal UI; 8 independently shippable phases (section 6).

---

## 1. What ZeroClaw is

### 1.1 Process and deployment model

- One Rust binary (`~/.cargo/bin/zeroclaw`, 0.8.4 installed on the desktop; the fork builds 0.8.5). Subcommands include `daemon`, `agent -a <alias>`, `acp`, `config`, `service install` (`ZC/README.md:38-52`).
- A long-lived **daemon** hosts: the agent runtime, the HTTP/WS **gateway** (`crates/zeroclaw-gateway`), 30+ channels (Discord etc.), cron, SOP engine, memory, and a local JSON-RPC socket (`ZC/docs/book/src/architecture/rpc-socket.md`).
- Config is one TOML file (`~/.zeroclaw/config.toml`, schema v3). Sections: `[providers.models.<type>.<alias>]`, `[agents.<alias>]`, `[risk_profiles.<alias>]`, `[[mcp.servers]]`, `[mcp_bundles.<alias>]`, `[[model_routes]]`, `[gateway]`, `[nodes]`. Editable by CLI, dashboard, or REST (`PATCH /api/config`, `PUT /api/config/prop`; secrets write-only) (`ZC/docs/book/src/gateway/api.md`).
- **Every WS session is bound to a configured agent alias**; there is no default agent (`ZC/crates/zeroclaw-gateway/src/ws.rs:167-186`: HTTP 400 for a missing or unknown `?agent=`).
- On the user's desktop `~/.zeroclaw/config.toml` holds only `[gateway] paired_tokens`; no agents/providers. The real gateway is the ubuntu box. Deployed agents (ZCX `README.md:363`): `coder comfy companion default promptsmith research reverse voice`.
- Model path on the box: gateway -> LiteLLM `:4000` -> llama.cpp workers (`qwen3.8-27b`, `-np 4 --kv-unified`). Consequences for AME (`ZCX/HANDOFF.md:238-239`, `zeroclaw-homelab/CACHE-AND-ROUTING.md`):
  - prompt cache is per worker process; a conversation re-prefilled on the other worker costs minutes -> keep **one stable ZeroClaw session id per chat**, never replay history from the client;
  - about two full-context sessions saturate one box, a third is rejected -> AME must serialize background AI jobs and never fan out parallel agent sessions.

### 1.2 Surfaces

| Surface | Transport / format | Auth | Who uses it |
| --- | --- | --- | --- |
| Gateway `:42617` (default port `CUI/src/zeroclaw.rs:26`) | HTTP + WS (axum); `wss://` behind the public origin | pairing bearer token | all consumers |
| `GET /ws/chat?agent=&session_id=&name=&cwd=&adopt=&token=` | WS text frames, one JSON object each; optional subprotocol `zeroclaw.v1` | `Authorization: Bearer`, `Sec-WebSocket-Protocol: bearer.<t>`, or `?token=` in that precedence (`ws.rs:93-125`) | `ZCX/android`, `CUI` |
| REST `/api/*` | JSON | bearer | all |
| `POST /api/complete` | one model call through a `[[model_routes]]` hint; no agent, no tools, no session; 64 KiB input cap; temp 0..2 (`ZC/crates/zeroclaw-gateway/src/api_complete.rs`) | bearer | `MT`, `ZCX/android` titles |
| `POST /webhook?agent=` | sync `{message}` -> `{response, model}` agent turn **with tools** via `run_gateway_chat_with_tools`; optional `X-Session-Id` header, <= 128 chars (`lib.rs:3076-3220`, `webhook_session_id` 222) | bearer or `X-Webhook-Secret` | nobody yet |
| `GET /api/events` | SSE of runtime events | bearer | dashboard |
| RPC socket `<data_dir>/daemon.sock` or `\\.\pipe\zeroclaw-<hash>` | NDJSON JSON-RPC 2.0: `initialize`, `session/new`, `session/prompt`, `session/cancel`, notifications `session/update`, server->client `elicitation/create` | OS permissions (0600/ACL) | `zerocode` TUI |
| ACP (`zeroclaw acp` stdio, and `/acp` WS) | JSON-RPC 2.0 NDJSON; `session/request_permission` for approvals | stdio: none; WS: bearer | IDEs |
| MCP **client** | stdio / http (Streamable HTTP, `Mcp-Session-Id`, SSE responses handled) / sse | per-server `headers` | agent tools |
| `/ws/nodes` | WS JSON: `register{node_id,capabilities[]}` <- gateway `invoke{call_id,capability,args}` -> `result{call_id,success,output,error}` | `nodes.auth_token` or pairing | device nodes; see risk R7 |
| `POST /pair` + `POST /admin/paircode/new` | one-time code exchange; admin route loopback-only | code / none | pairing (`ZCX/README.md:181-183`) |

### 1.3 Auth and pairing

1. On the box (loopback): `POST /admin/paircode/new` -> one-time code.
2. Client: `POST /pair` with header `X-Pairing-Code: <code>` -> bearer token. Consumed on first exchange (`ZC/crates/zeroclaw-gateway/src/lib.rs:2384-2470`).
3. `GET /health` is unauthenticated and returns `{status, paired, require_pairing, runtime}` (`lib.rs:2326`).
4. Tokens are long (67 chars in the zc-codex config). `/pair` is rate-limited and brute-force limited.
5. WS auth failure returns HTTP 401 at upgrade (`ws.rs:137-150`); missing/unknown agent returns HTTP 400.

### 1.4 `/ws/chat` wire protocol (verified in `ws.rs`)

Client -> server:

| Frame | Notes |
| --- | --- |
| `{"type":"connect","device_name?","capabilities?":["text"],"cwd?"}` | optional first frame; ack is `connected`. A `session_id` inside it is **ignored for resume** (`ws.rs:530-560`, `ZCX/README.md:170-176`). |
| `{"type":"message","content":"..."}` | starts a turn; **while a turn runs it is steering** (queued, `STEERING_QUEUE_FULL`/`STEERING_CLOSED` refusals). Empty -> `EMPTY_CONTENT`. |
| `{"type":"approval_response","request_id":"...","decision":"approve"\|"always"\|"deny"}` | answers `approval_request` (`ws.rs:853-890`). |
| WS ping/pong | gateway pings per `gateway.websocket_ping_interval_secs`; clients should ping every 20 s (proxy idle drops, `ZCX/README.md:315-318`). |

Server -> client:

| Frame | Fields |
| --- | --- |
| `session_start` | `session_id, resumed, message_count, name?` (`ws.rs:497`) |
| `connected` | `message` |
| `turn_resume` | `frames[]` (fork feature: replays the running turn to a reconnecting socket; `ZC/LOCAL-CHANGES.md`) |
| `agent_start`, `agent_end` | `agent_end` carries `model_provider, model` |
| `thinking` | `content` (reasoning delta) |
| `chunk` | `content` (answer delta) |
| `tool_call` | `id, name, args` (note `args`, not `arguments`) |
| `tool_result` | `id, name, output` |
| `approval_request` | `request_id, tool, arguments_summary, arguments, timeout_secs` (WS default budget 300 s, `ws.rs:32`) |
| `plan` | `entries[{content,status}]` |
| `history_trimmed` | `dropped_messages, kept_turns, reason` |
| `done` | `full_response, input_tokens, output_tokens, tokens_used, cost_usd, model, provider, max_context_tokens, last_input_tokens` (`ws.rs:2028`); `full_response` is authoritative over summed chunks |
| `aborted` | turn stopped |
| `error` | `message, code`; codes seen: `AGENT_INIT_FAILED, INVALID_CWD, INVALID_JSON, UNKNOWN_MESSAGE_TYPE, EMPTY_CONTENT, NEEDS_ONBOARDING, INVALID_APPROVAL_RESPONSE, STEERING_*, SESSION_QUEUE_FULL/TIMEOUT, PROVIDER_ERROR, AUTH_ERROR, AGENT_ERROR` (`ws.rs:2157-2174`). `PROVIDER_ERROR/AUTH_ERROR/AGENT_ERROR` end the turn. |
| `cron_result` and other event-bus frames | broadcast to sockets of the matching session |

Behaviors that bite (all verified by consumers or code):

- **Session id is client-chosen.** `session_id = ?session_id or fresh UUID`, stored under key `gw_<id>` (`ws.rs:358-361`). Resume = `?session_id=<id>` only; `?session=` and the connect-frame field are accepted and silently ignored (a new session is minted). Always compare the returned `session_start.session_id` with the requested one (`ZCX/android/src/zeroclaw.rs:146-148`).
- Socket replays nothing on resume; fetch history over REST. Stored assistant/tool messages are JSON **envelopes** (`{"content","reasoning_content","tool_calls"}`), so a parser is needed (`unwrap_stored`).
- User messages carry an injected `[CURRENT DATE & TIME: ...]` or `[YYYY-MM-DD HH:MM:SS +00:00]` preamble to strip when rendering.
- A turn **outlives its socket** in the fork (`LOCAL-CHANGES.md` "a turn outlives its client's socket"); explicit stop = `POST /api/sessions/{id}/abort`. The socket accepts only `message` and `approval_response`.
- Images ride **inside** `content` as `[IMAGE:data:<mime>;base64,...]` (no attachment field). Provider strips the marker into a real image part. `ZCX` caps at 8 MB and accepts only PNG/JPEG/GIF/WebP by magic bytes.
- Vision requires the agent's provider to be vision-capable or a configured `[multimodal] vision_model_provider` (`ZC/docs/book/src/providers/configuration.md:23`).

### 1.5 REST endpoints consumers rely on

| Endpoint | Use | First seen |
| --- | --- | --- |
| `GET /api/config/agent-options` | list agent aliases (also the health probe: authed, cheap) | `ZCX/android/src/zeroclaw.rs:537`, `engine.rs:1646` |
| `GET /api/sessions` | rows `{session_id, session_key, agent_alias, name, message_count, last_activity}`; twin stores `gw_<id>`/`<id>` share an id | `zeroclaw.rs:560` |
| `GET /api/sessions/{id}/messages[?tool_context=true]` | history; `id` resolved verbatim, then `gw_`, `rpc_`, `cc_` | `zeroclaw.rs:701`, `MT/.../api.rs:163` |
| `GET /api/sessions/{id}/state` | `idle`/`running`/`error` + `turn_started_at` | `zeroclaw.rs:904` |
| `GET /api/sessions/running` | sessions mid-turn | `zeroclaw.rs:570` |
| `POST /api/sessions/{id}/abort` | stop (`aborted` / `no_active_response`) | `zeroclaw.rs:929` |
| `PUT/DELETE /api/sessions/{id}` | rename/delete | `zeroclaw.rs:424-441` |
| `GET /api/sessions/{id}/media?path=` | images referenced by `[IMAGE:..]` in assistant/tool output; 24 MiB cap | `zeroclaw.rs:455` |
| `POST /api/sessions/{id}/messages` | **appends an assistant-role notification and wakes connected sockets; it does NOT run a turn** (`api.rs:1890-1990`; the book table says "run a turn", which is wrong) | unused |
| `POST /api/complete` | `{hint:"quick", system?, prompt, temperature?}` -> `{response, model, hint}` | `MT openai_bridge.rs:310`, `ZCX titles.rs` |
| `GET/POST /api/memory` | `?query&agent` recall; `{key,content,category,agent}` store | `MT headless/codex/zeroclaw.rs:73-100` |
| `GET /api/cron`, `/api/cron/{id}/runs`, `POST /api/cron/{id}/run` | automations | `MT`, `ZCX` |

### 1.6 Sessions, memory ("palace"), personality

- Sessions are server-side in SQLite; client holds only an id. Names via `?name=` or `PUT`.
- Memory has two layers: per-agent memory (`/api/memory`, consolidated at turn end) and the **Palace** (curated cross-project brain, MCP at `ZC_PALACE_MCP_URL`, tools `palace_{brief,search,remember,threads,projects}`; project argument mandatory) (`ZCX/deploy/PALACE.md`, `ZCX/src/palace.rs`).
- Agent personality = markdown files in the agent workspace (`IDENTITY.md, SOUL.md, USER.md, AGENTS.md, TOOLS.md, MEMORY.md`), editable via `/api/personality` (`ZCX/README.md:131-161`). User's repo for these: `zeroclaw-homelab/agents/<alias>/workspace`.

### 1.7 Tools, MCP, approvals, sandbox

- ZeroClaw is an MCP **client**; tool name = `<server>__<tool>` with the raw MCP tool name appended unchanged (`ZC/crates/zeroclaw-tools/src/mcp_client.rs:912`).
- An agent connects to **no MCP server unless a bundle grants it** (`agents.<a>.mcp_bundles` -> `[mcp_bundles.<b>].servers`); changes take effect at **session (socket) start** (`ZC/docs/book/src/tools/mcp.md`). The agent (and its granted MCP servers) is built per socket after the optional `connect` frame (`Agent::from_live_config_with_session_cwd_and_mcp_backchannel`, `ws.rs:591`; bundle doc says grants resolve at session construction), so a tool server that comes up after the socket opened is invisible until reconnect (inferred, not tested).
- Server fields (`schema.rs:5131`): `name, transport(stdio|http|sse), url, command, args, env, headers, tool_timeout_secs (cap 600), max_response_bytes (default 16 MB), pinned_resources, tls_ca_cert_path`.
- Approvals: risk profile `level = readonly|supervised|full`; per-tool `auto_approve`, `always_ask`, `allowed_tools`, `excluded_tools` matched on the prefixed name; unanswered approvals deny after timeout (`ZC/docs/book/src/security/autonomy.md`).
- Filesystem boundary: `workspace_only` + `allowed_roots` + `forbidden_paths`, OS sandboxes. These govern **ZeroClaw's own file/shell tools on the gateway host**, which cannot see the desktop's files. The desktop's files are reachable only through tools AME serves.
- Embedded blobs in MCP tool results (`resource` + base64 `blob`) are materialized under the session workspace `uploads/` (<= 10 MiB, <= 64 blobs) and replaced by an `[IMAGE:..]`/`[Document:..]` marker (`mcp.md` "Embedded resource blobs"). Plain MCP `image` content handling **[UNVERIFIED]**.
- Provider filter: `OpenRouterModelProvider` drops any tool whose name fails `^[a-zA-Z0-9_-]{1,64}$` (`ZC/crates/zeroclaw-providers/src/openrouter.rs:281,1060`, test `convert_tools_skips_invalid_names`). MCP names are not sanitized upstream of it, so dotted names are lost on that provider.

---

## 2. How the three consumers embed or talk to it

### 2.1 `ZCX` (zc-codex) - two backends behind one `Event` stream

- Root crate `zc-codex` (lib + `zc-codex` CLI + `zc-codexd` daemon) is the **Codex app-server** JSON-RPC bridge, not a ZeroClaw client. Reusable ideas only: `Event` enum (`src/lib.rs:42`) and timer `Coalescer` (`src/lib.rs:111`). Its deps (socket2, process supervision, reqwest 0.12) make it a poor dependency for AME.
- The ZeroClaw gateway client is in the **Android app crate** `ZCX/android/src/zeroclaw.rs`, reached directly (no daemon in front: `README.md:128-130`):

| Function | Line | Role |
| --- | --- | --- |
| `Client::connect` | 59 | connects `/ws/chat`, sends `connect`, waits for `session_start`, spawns writer (20 s ping) + reader; errors on session-id mismatch (146-148) |
| `send_message` / `respond` | 215 / 226 | `message` and `approval_response` frames |
| `Inline::read`, `image_mime` | 260, 298 | `[IMAGE:data:..]` marker, 8 MB cap, magic-byte MIME |
| `http_body` | 374 | bounded-body authed HTTP, redirects refused |
| `complete`, `agents`, `sessions`, `running`, `abort`, `turn_state`, `media` | 357, 537, 560, 570, 929, 904, 455 | REST |
| `parse_history`, `unwrap_stored` | 734, 824 | envelope unpacking, last 200 messages |
| `Turns`, `translate` | 954, 989 | frame -> `Event`; **row ids carry connection nonce + turn number + tool index** so a reply never overwrites the previous turn (README:309-313) |

- Health: `probe_health` (`android/src/engine.rs:1646`, 6 s budget) calls `agent-options`; states `Unknown/Checking/Online/Unauthorized/Offline/Unconfigured` (`android/src/types.rs:7-53`); swept every 45 s.
- UI plumbing: UI thread synchronous; worker owns tokio runtime; `Cmd`/`Msg` over `std::sync::mpsc`; worker calls `request_repaint` per message (`android/src/engine.rs:1-8`).
- Transcript cache: newest 300 rows to disk 2 s after last change; reconnect **merges** server history into the local transcript instead of replacing it (`README.md:234-243`).
- ZeroClaw used as a **service** by the Codex path: `web_tools.rs` (`/api/agents/{agent}/web-tools`), `palace.rs`, `rea.rs`, `critic.rs`; these show the pattern "external agent runtime borrows ZeroClaw memory/search".
- Tests: loopback WS servers in `tests/cancellation.rs`, `tests/approval_resubscribe.rs`; `a_gateway_cannot_silently_substitute_a_different_session` (`zeroclaw.rs:1216`); ignored live tests via `ZC_HOST/ZC_TOKEN_FILE`.
- Machine config on this desktop (`~/.config/zc-codex/config.json`): `machines[]` with `name, host, token, kind (Codex|ZeroClaw), agent`; entries exist for `zeroclaw` (`100.96.121.37:42617`, agent `companion`) and `claw (public)` (`wss://claw.shadowbroker.app`). AME can import from here.

### 2.2 `CUI` (comfyui-android) - minimal one-turn client

- `src/zeroclaw.rs` (406 lines): `Gateway{host,token,agent}` with `ready()`, `tls()`, `host_port()` (adds `:42617` unless `wss://`/`https://`), `url()` (`/ws/chat?agent=&token=` percent-encoded), `url_for_log()` (token blanked). `turn()` (153) connects, sends `connect` + `message`, reads `chunk` until `done`, 120 s budget, uses `done.full_response` as authority. `from_zc_codex_config()` (279) imports a machine from the ZC Codex config.
- Engine choice pattern worth copying: `RewriteEngine::{Auto, ZeroClaw, Server, Device}` (`src/types.rs:1757`); `Auto` prefers ZeroClaw; **an explicit pick that is unavailable reports unavailable instead of silently running another engine** (`resolve`, 1788).
- Async bridge: engine owns a tokio runtime; each request returns a `std::sync::mpsc::Receiver`; worker calls `ctx.request_repaint()` per delta (`src/engine.rs:878-912`).
- Settings UI: host + token (password field) + agent + "Import from ZC Codex" (`src/app.rs:23079-23170`).

### 2.3 `MT` (Mastertech4.0) - "default = ZeroClaw" without a chat socket

Mastertech does not drive `/ws/chat`. It uses ZeroClaw four ways:

1. **Default completion backend.** `custom_api_base()` returns `Some` only when the user set an OpenAI-compatible endpoint; `None` means "ZeroClaw serves the call" (`MT/displays/src/ai/mod.rs:34-37`). `OpenAiMcpSession::stream_command_completions` branches on it (`displays/src/mcp/openai_bridge.rs:93-95`) -> `complete_via_zeroclaw` (247) -> `POST /api/complete {hint:"quick", prompt, temperature:0.2}` (310-328), 30 s timeout, quiet no-op if no gateway access.
2. **Credentials.** URL compiled in from env `ZEROCLAW_GATEWAY_URL` (`database/src/lib.rs:295`); token never compiled; read from env `ZEROCLAW_GATEWAY_TOKEN`/`MTECH_ZC_TOKEN` or `<APPDATA|HOME>/MasterTech/zeroclaw.json` (`.env.example:37-41`, `displays/src/ai/mcp_chat.rs:6`); for Root users also a DB row `zeroclaw_gateway:shop` with SurrealDB permissions limiting reads to active Root users (`database/schema/zeroclaw_gateway.surql`, `database/src/schema/zeroclaw_gateway.rs:11-36`; `Debug` redacts the token).
3. **Viewer.** `displays/src/tabs/zeroclaw/{api.rs,mod.rs}`: sessions, transcripts (`?tool_context=true`), automations, run-now, unread counts, 30 s list poll, 5 s/60 s transcript poll.
4. **Memory + skills + audit.** `displays/src/headless/codex/zeroclaw.rs`: `GET/POST /api/memory`, shop skill bundle `/api/skills/bundles/{b}/skills`; ZeroClaw's `webhook-audit` hook posts tool-call trails to `axum_server/src/routes/api/audit.rs` (`/api/v1/audit/zeroclaw`).

And the **tool side**: Mastertech hosts an rmcp **Streamable HTTP MCP server inside the GUI process** (`displays/src/plugins/mcp_bridge.rs:13773-13820`: `StreamableHttpService` nested at `/mcp` on `MTECH_MCP_HTTP_ADDR` default `127.0.0.1:9004`, 8 h session keep-alive) plus TCP :9003 and stdio. ZeroClaw/Claude/Codex agents call tools named `mastertech__*`. `Mastertech4.0/mcp-proxy` (Node) is a stdio<->HTTP reconnecting shim so clients survive app restarts.

### 2.4 Comparison

| Aspect | ZCX android | CUI | MT |
| --- | --- | --- | --- |
| Chat socket | yes, full | one turn | none |
| Sessions/resume | yes | no | read-only viewer |
| Approvals | modal | no | no |
| `/api/complete` | titles | no | default backend |
| Memory | agent files editor | no | recall/remember |
| Own MCP server for the agent | no | no | yes (:9004) |
| Credential storage | config.json | settings | env / json / Root-only DB row |
| Health | probe + dot | none | heartbeat in audit tab |

Duplication to remove: `unwrap_stored` (`ZCX zeroclaw.rs:824`, `MT api.rs:218`), `strip_date_stamp` (`ZCX:872`, `MT api.rs:275`), percent-encoding (`ZCX:445`, `MT api.rs:67`, `CUI:102`), session-row parsing. Three copies justify a shared crate (section 5.6).

---

## 3. AME's current AI surface (verified)

### 3.1 Map

| Area | Files (lines) | What it does |
| --- | --- | --- |
| Remote chat client | `src/ai/openai_compat.rs` (253) | `ProviderConfig{provider,api_key,base_url,model,organization,temperature,zdr}` (22-31); `stream_multimodal_reply` -> `stream_openai_compatible` (148) builds `[system, user(prompt+images)]` (208); OpenRouter via BYOT stream types (`openrouter_types.rs`); model listing (`fetch_openrouter_models_json` 121) |
| Local-model glue | `src/ai/mod.rs` (75) | `GLOBAL_AI_ENGINE` (31), `init_global_ai_engine_async` (33), `unload_heavy_models_except` (52) |
| MCP server | `src/ai/mcp/mod.rs` (175) | rmcp 3.5 stdio server, 3 tools |
| Search/index/describe | `src/ai/ai_search/{mod,generate,generate_description,index,search,cache}.rs` | `AISearchEngine`: in-memory `files`, CLIP text/image search, JoyCaption `VisionDescription{description,caption,tags,category}`, streaming JSON field parser (`mod.rs:596-793`) |
| Refinement | `src/ai/refine.rs` (624), `src/ui/refine.rs` (546) | kNN-neighbour voting + `Reranker` trait (Heuristic, Jina "stand-in" on a fastembed CLIP text encoder) and a sequential per-row "cloud" path (ui/refine.rs:129-263) |
| Local models | `joycap/*`, `candle_llava/*`, `clip/*`, `siglip/*`, `reranker/mod.rs`, `jina_m0/*`, `qwen2_5_vl/*`, `qwen_image_edit/*`, `qwen_image_transformer`, `vae`, `flux_transformer` | CUDA candle/fastembed/ort; JoyCaption runs on a dedicated OS thread with an `mpsc` work queue (`joycap/mod.rs:286-362`) |
| Assistant UI | `src/ui/assistant.rs` (1839) | see 3.3 |
| Image edit UI | `src/ui/image_edit.rs` (526) | Qwen Image Edit load/run/preview/cancel panel |
| Chat persistence | `src/database/assistant_chat.rs` (106), schema `src/database/mod.rs:57-58,84-93,156` | `assistant_sessions`, `assistant_messages` (role, content, attachments, attachments_refs), SCHEMAFULL |
| Settings | `UiSettings` / `UiSettingsDb` `src/database/settings.rs:11-119` (note: **not** `utilities/types.rs`, which has no settings); modal `src/receive.rs:142-820` | provider enum string, per-provider keys, one shared `openai_default_model` |
| Status chips | `src/ui/status.rs` | `RegisteredStatus` registry: DB, VISION, CLIP, RERANK, QWEN-EDIT, ASSIST |
| App->UI command buses | `src/app.rs:12-14`, `ui/assistant.rs:16`, `ui/refine.rs:41` | `OPEN_SETTINGS_REQUEST`, `OPEN_TAB_REQUESTS` (`FilterRequest::{NewTab,OpenDatabaseAll,OpenPath}`), `ATTACH_REQUESTS`, `REFINE_SELECTION_REQUESTS` |
| UI entry points | `ui/file_table/table/mod.rs:864-1112`, `preview_pane.rs` | context menu: attach to chat, open selection in tab, generate/regenerate description, find similar, refine selection |

### 3.2 Defects and constraints that matter for the swap

| # | Finding | Evidence | Why it matters |
| --- | --- | --- | --- |
| F1 | MCP stdio server is started **twice on the GUI process's own stdin/stdout** | `src/main.rs:144-149`, `src/receive.rs:45-49` | Useless unless some client spawned the GUI as its child; two readers fight; cannot be reached by a remote ZeroClaw. A second process cannot take over: SurrealKV holds an exclusive `LOCK` file (`surrealkv-0.21.4/src/lockfile.rs:54,79`) and `DB.connect::<SurrealKv>` is embedded (`database/mod.rs:44`); GPU models would also load twice. |
| F2 | Tool names contain dots: `util.ping`, `media.search`, `media.describe` | `ai/mcp/mod.rs:58,64,86` | become `aimedia__media.search` in ZeroClaw; dropped by the OpenRouter provider filter (1.7). |
| F3 | MCP tool quality | `ai/mcp/mod.rs:63-111`, `database/clip_embeddings.rs:39-69` | `score` is a cosine **distance** (lower is better) named like a similarity; `find_similar_by_embedding` ignores `k`/`ef` (query hard-codes `<\| 64, 64 \|>`), so `top_k > 64` never works; results are `{path, score}` only (no caption/tags/category/thumbnail, no pagination); no metadata search, no tag/category listing, no item detail, no library context; `media.describe` mutates the DB and blocks the GPU with no dry-run and accepts any readable path; `impl ServerHandler for MCPServer {}` is empty, so server info is the rmcp default with no instructions. |
| F4 | Assistant sends **no conversation history**: request is `[system, latest user]` | `openai_compat.rs:208`, `ui/assistant.rs:1376` | multi-turn chat is not multi-turn; ZeroClaw's server-side history fixes this for free. |
| F5 | Streamed tokens are untagged and append to "the last assistant message" | `ui/assistant.rs:839-847`; Stop only aborts the JoinHandle (482-487) | stale tokens after Stop/session switch land in the wrong bubble; remote turn keeps running. |
| F6 | API key and full base64 image payloads logged at ERROR | `openai_compat.rs:137,170,196,204,214`; logger level Info in `main.rs` | visible in the in-app Logs tab; must be removed before adding a gateway token. |
| F7 | Provider API keys are plaintext in SurrealDB `user_settings`, defaulted from env | `database/settings.rs:42-46,243-247` | do not add the ZeroClaw token there. |
| F8 | Provider -> (key, base, org) resolution is copy-pasted three times; one `openai_default_model` is shared by all providers | `assistant.rs:1322-1353`, `ui/refine.rs:218-224`, `receive.rs:335-439` | switching providers clobbers the model name. |
| F9 | `simple_text_completion` discards stream errors (`let _ = ...`) | `openai_compat.rs:144` | cloud refine reports success with empty text. |
| F10 | Refinement is not LLM-based locally; "cloud" mode is sequential, capped at 64 rows, text-only, no cancel | `ai/refine.rs:16-22`, `ui/refine.rs:230`; Jina engine load is a placeholder (`reranker/mod.rs:57-64`) | clean place to plug a batch LLM refiner. |
| F11 | Assistant is a separate OS viewport, forced open when settings load | `receive.rs:964-973`, `receive.rs:67` | should be a dock tab in the UI reorganization. |
| F12 | All sessions and all messages are loaded at startup (N+1 `Thumbnail::get_by_id`) | `ui/assistant.rs:737-773` | does not scale; ZeroClaw history should load lazily. |
| F13 | Windows **release** build requests Administrator | `build.rs:18` | any tool surface the app exposes inherits elevation; path allow-list is mandatory. |
| F14 | Both `aws-lc-rs` and `ring` are in `Cargo.lock`; nothing calls `CryptoProvider::install_default` | `Cargo.lock:395,7380`; `grep install_default src` empty | adding a rustls WebSocket client can panic at first TLS use **[UNVERIFIED at runtime]**; `MT/admin-agent/src/main.rs:14` installs one explicitly. |
| F15 | JoyCaption default dir is a Windows path constant | `src/app.rs:15` | on Linux the local vision worker needs the setting or silently falls back. |
| F16 | Persisted default provider is `local-joycaption` | `settings.rs:242` | changing `Default` does not migrate existing rows; need a new field. |
| F17 | Binary-only crate (`main.rs`, no `lib.rs`), CUDA candle + ort | `Cargo.toml` | integration tests cannot import modules; builds need CUDA. A separate pure-Rust crate gives fast protocol tests. |
| F18 | Existing helpers are directly reusable | `Thumbnail::update_or_create_thumbnail` updates only non-empty fields (`thumbnails.rs:432-470`); `rename_category/rename_tag/delete_tag/prune_tags/list_*_counts` exist (`thumbnails.rs:262-410`) | tool implementations mostly exist. |

### 3.3 `ui/assistant.rs` anatomy (1839 lines)

| Lines | Concern |
| --- | --- |
| 32-92, 116-176 | `AssistantPanel` state: 30+ fields, ~14 channels |
| 192-717 | `ui()`: top bar, bottom composer (attach menu, MCP Tools menu 240-313, inference menu 319-340, provider combo 346-396, model box 398-416, attachment chips 418-473, send/stop/regen 475-500), left panel (recent models 527-568, sessions 574-683), center transcript (685-716, hard-coded `max_msg_width = 376.0`) |
| 719-876 | `receive()`: attach queue, initial session load, new-session, session-loaded, thumbnails, token drain, OpenRouter models channel |
| 879-1171 | OpenRouter model browser window (~290 lines) |
| 1174-1581 | `send_current_prompt`: slash commands `/ping /search /describe` (1223-1313), cloud path (1315-1450), local path (1451-1546), persistence (1547-1580) |
| 1583-1800 | bubbles, fenced-code rendering with syntect (`render_message_inner` 1703) |
| 1802-1839 | thumbnail cache + DB fetch |

---

## 4. Mapping onto ZeroClaw

### 4.1 What moves, what stays

| Capability | Today | Decision | Why |
| --- | --- | --- | --- |
| Assistant chat (cloud providers) | `openai_compat` stream | **ZeroClaw agent session over `/ws/chat`** (default); Direct OpenAI-compatible stays as selectable backend | server-side history, tools, memory, approvals, cache affinity |
| Tool use over the library | slash commands calling MCP fns in-process | **agent calls AME MCP tools** | models decide; slash commands remain as debug |
| Tag/category/caption refinement | heuristic + sequential cloud | **ZeroClaw `/api/complete` batch JSON** (hint `quick`) seeded with local kNN neighbours; proposals into the existing Refine UI | one request per N rows, no tools needed, cheap pool route |
| Session titles, tag suggestions, query->filter translation | none | `/api/complete` small calls (pattern: `ZCX titles.rs`, `MT openai_bridge.rs`) | text-only, fast |
| Bulk image captioning (JoyCaption) | local GPU | **stays local** | thousands of images, offline, privacy, 8 MB inline cap and 5-15 s remote latency per image, 2-session pool ceiling |
| Interactive "describe this image" | local or cloud | local JoyCaption by default; optional ZeroClaw vision turn if the agent has vision | no hard dependency on a vision model on the box |
| CLIP / SigLIP embeddings, HNSW KNN | local | **stays local, always** | vectors must come from one model/version per library; DB-resident |
| Reranker (Jina), fastembed | local | stays local | latency, offline |
| Qwen image edit | local candle/GGUF | stays local; **exposed as an async job tool** | 20B model on the user's GPU |
| Prompt templates | `ai_prompt_template` (settings.rs:61) for context-menu describe; hard-coded strings elsewhere (`generate_description.rs:33,52,74`) | keep for local bulk; agent-facing guidance moves to agent workspace files | tune without app release |

### 4.2 Local models exposed as tools

Reuse the existing in-process engine (`GLOBAL_AI_ENGINE`, `DB`) so the tool server must live **inside the GUI process**. Proposed catalog (all names `[a-z0-9_]`, <= 40 chars; with `aimedia__` prefix <= 64):

| Tool | Class | Backed by | Notes |
| --- | --- | --- | --- |
| `util_ping` | L0 | existing | connectivity test |
| `library_stats` | L0 | `Thumbnail::count_all_thumbnails`, `list_category_counts`, `list_tag_counts`, `clip_missing_count` | counts, coverage |
| `library_search_text` | L0 | new DB query over filename/caption/description/tags/category | paged, returns id/path/filename/caption/category/tags |
| `library_search_semantic` | L0 | CLIP/SigLIP text embed + `find_similar_by_embedding` (fix k/ef) | returns `similarity` (higher is better) |
| `library_find_similar` | L0 | same, by id/path | |
| `library_get_item` | L0 | `Thumbnail::get_thumbnail_by_path/get_by_id` | full metadata |
| `library_list_tags` / `library_list_categories` | L0 | `list_tag_counts` / `list_category_counts` | paged |
| `library_get_preview` | L1 | `generate_image_preview_png(path, <=768)` -> JPEG | returns image content; ZeroClaw handling of MCP `image` content **[UNVERIFIED]** |
| `app_get_context` | L0 | explorer state: current dir, selection, tab, filters | lets the agent see what the user is looking at |
| `media_describe` | L2 | `generate_vision_description` | `apply=false` default: returns JSON only; GPU queue |
| `media_embed` | L2 | `clip_generate_for_paths` | |
| `image_edit_run` / `job_status` / `job_cancel` | L2 | `QwenImageEditPipeline` | returns job id at once; output to `generated/`, never overwrites; completion pushed into chat via `POST /api/sessions/{id}/messages` |
| `metadata_propose` | L3 | pushes `RefinementProposal`s (`generator="zeroclaw:<agent>"`) | **no direct DB write**; user accepts in Refine tab |
| `taxonomy_propose_rename` | L3 | `rename_tag/rename_category` preview counts | proposal with affected-row count |
| `ui_open_results` | L1 | `OPEN_TAB_REQUESTS` `FilterRequest::NewTab` | result sets as tabs |

Not exposed in v1: delete/move/rename files, arbitrary SQL, arbitrary path read, shell.

Cross-cutting rules:
- **Path policy**: tools take library ids or paths that must canonicalize under indexed roots (`parent_dir` of stored rows / active logical-group roots), honor `excluded_dirs`, reject symlink escapes. Today `media.describe` accepts any path.
- **GPU scheduler**: agent tool sequences can thrash model loads because `unload_heavy_models_except` (`ai/mod.rs:52`) is called ad hoc (e.g. `ui/image_edit.rs:128,293`). Add a single-flight `GpuQueue` (concurrency 1, residency window, queue position in results).
- **Pagination + size caps**: <= 50 rows/page, text fields truncated, previews <= 768 px.
- **Server info**: set name/version/instructions; add `pinned_resources` URI `library://guide` (taxonomy conventions, tool-use guidance) in the gateway server entry.
- **Auditing**: in-app "Agent activity" log of every tool call (name, args summary, duration, result size) in addition to ZeroClaw receipts.

### 4.3 How the remote agent reaches the tools

| Option | Mechanism | Fit | Problems |
| --- | --- | --- | --- |
| **A. App-hosted Streamable HTTP MCP (recommended)** | rmcp `StreamableHttpService` in the GUI process (as `MT mcp_bridge.rs:13773`), bearer auth; gateway entry `[[mcp.servers]] name="aimedia" transport="http" url="http://<desktop-tailnet-ip>:<port>/mcp" headers={Authorization="Bearer ..."}` + `mcp_bundles` on a dedicated `media` agent | works with the remote gateway on the tailnet (desktop is `100.116.134.128`, gateway `100.96.121.37`); ZeroClaw's HTTP transport handles `Mcp-Session-Id` + SSE (`mcp_transport.rs:44-50,1251`) | rmcp 3.5.1 defaults `allowed_hosts = [localhost,127.0.0.1,::1]` (DNS-rebinding guard, `tower.rs:203`) so the tailnet IP/host must be added; MCP exists only while the app is open; socket must reconnect after the server starts (1.7); needs feature `transport-streamable-http-server` + `axum` (already in tree 0.8.9). |
| B. Local ZeroClaw + stdio shim | `zeroclaw daemon` on the desktop; `smart-media mcp-proxy` bridges stdio <-> app's loopback HTTP (like `MT mcp-proxy`) | zero network exposure | needs a configured local daemon + provider; **not** the user's current setup. |
| C. `/ws/nodes` reverse registration | app connects out, registers capabilities, gateway sends `invoke` | solves reachability | tool wiring into the agent loop **not found** (`NodeTool::new`/`all_capabilities` have no call site outside their modules); text-only `output`; 30 s timeout; names `node:<id>:<cap>` fail the OpenAI-name filter; treat as spike only. |
| D. No tools | chat + `/api/complete` only | simplest | loses the point of the agent. |

Provisioning the gateway side: Settings shows a ready-to-paste TOML snippet (Appendix A) and an optional **Apply to gateway** button using `PATCH /api/config` (JSON Patch) with explicit confirmation; exact keys to be confirmed against `/api/docs` first. Also ship `agents/media/workspace/{IDENTITY,SOUL,AGENTS,TOOLS}.md` templates for `zeroclaw-homelab`.

### 4.4 Assistant UI changes

Structure (split `ui/assistant.rs` into `ui/assistant/`):

| Module | Content | From lines |
| --- | --- | --- |
| `mod.rs` | `AssistantPanel` shell and layout only | 192-717 |
| `transcript.rs` | `Entry` enum + renderer (reuse syntect code blocks) | 1583-1800 |
| `composer.rs` | input, attachment chips, send/stop/regenerate, context chips | 210-519 |
| `sessions.rs` | session list, lazy load, remote-session browser | 574-683, 737-821 |
| `backend.rs` | `ChatBackend` {ZeroClaw, Direct, Local} and turn dispatch | 1174-1546 |
| `approvals.rs` | inline approval cards with countdown | new |
| `tools_debug.rs` | tool tester (generated from tool schemas), replaces the 3-tool menu | 240-313 |
| move out | OpenRouter model browser -> Settings > AI > Direct providers | 879-1171 |

Changes:
- `ChatRole {User, Assistant}` -> `Entry {User, Assistant, Reasoning, ToolCall, ToolResult, Approval, Plan, Notice}`; tool/thinking rows collapsed by default; usage ring from `done` (`input_tokens/max_context_tokens`).
- Tokens carry `(session_id, turn, segment)`; late frames for a stopped/switched session are dropped (fixes F5).
- Stop = `POST /api/sessions/{id}/abort` + ignore until `aborted`/`done`.
- Context chips (`@selection`, `@folder`, `@filters`) replace the implicit "auto-attach current selection" (assistant.rs:423-443); snapshot sent as text, images only on request.
- Slash commands stay as a debug/quick-action layer mapped to tool calls.
- Header: backend chip (gateway - agent - health dot), "tools offline" banner when the MCP server is not reachable from the gateway.
- Becomes a dock tab (add to `TABS`, `ui/tabs.rs:4-8`) with optional pop-out viewport.

### 4.5 Chat history persistence

- **ZeroClaw is authoritative** for ZeroClaw-backed chats. Set `session_id` = the local `assistant_sessions` record key (already a v4 UUID, `assistant_chat.rs:29`), so no mapping table is needed; the gateway stores it as `gw_<uuid>`.
- Local DB keeps: session row (title, backend, gateway profile id, agent) and a **bounded transcript cache** (newest 300 rows) for instant open and offline viewing; reconnect merges server history in rather than replacing (pattern `ZCX README.md:234-243`).
- Schema is additive (SCHEMAFULL tables, `database/mod.rs:57-58`): `DEFINE FIELD IF NOT EXISTS` for `backend`, `agent`, `gateway_id` on `assistant_sessions`; `kind`, `turn`, `meta` (object) on `assistant_messages` (kind: user/assistant/reasoning/tool_call/tool_result/notice).
- Legacy sessions keep `backend="legacy"` (read-only reopen, or "continue with ZeroClaw" by seeding the first message).
- `?name=<title>` on connect names the remote session; `PUT /api/sessions/{id}` renames; title generation by `/api/complete`.
- Session list: local list plus optional "browse gateway sessions" filtered by `agent_alias`, skipping `message_count == 0` (bare connects mint empty sessions).

### 4.6 Refinement via ZeroClaw

- New `ProposalSource::ZeroClaw` beside Heuristic/Jina. For each batch of 20-40 rows build one `/api/complete` request: system = schema + library conventions; prompt = rows (path id, current caption/tags/category) **plus the local kNN neighbour vocabulary** (existing `tag_cands/cat_cands`, `ai/refine.rs:175-202`) so suggestions stay consistent with the library; response = JSON array of `{id, new_category?, new_tags?, new_caption?, new_description?}`; parse with the existing `parse_jsonish` (`ui/refine.rs:523`).
- Output feeds `RefinementProposal` unchanged (`generator = "zeroclaw:<hint>"`); Accept path is the existing `update_or_create_thumbnail`.
- Concurrency 1 with progress + cancel; text-only; 64 KiB cap per call (`api_complete.rs` `INPUT_CAP`) bounds batch size.
- Optional: send a 256-px JPEG thumbnail marker inside the prompt for vision refinement - the 64 KiB cap could allow ~20-40 KB JPEGs; whether `chat_with_system` expands `[IMAGE:..]` markers **[UNVERIFIED]**.

### 4.7 Offline and fallback behavior

| State | Behavior |
| --- | --- |
| Gateway unreachable at start | chip red with reason (`health_from_error` taxonomy); composer disabled with banner; local actions (CLIP search, describe, edit) unaffected; retry with backoff + manual Check |
| Token refused (401) | "Re-pair" card, never auto-retry |
| Socket drops mid-turn | keep the turn (server outlives socket), reconnect with `session_id`, receive `turn_resume`, then history sync |
| Tools offline (MCP unreachable) | chat works without tools; banner offers "Restart tools" and reconnect |
| `Auto` backend | ZeroClaw if configured and last probe ok; otherwise show onboarding. **Never silently route to a cloud provider**; an explicit pick that is unavailable reports unavailable (rule from `CUI types.rs:1779`) |
| Local-only mode | the existing no-chat behavior: CLIP text search and JoyCaption describe (assistant.rs:1451-1546) |
| Privacy | per-profile "allow image upload" (default off for the public `wss://` origin); images downscaled to <= 1024 px JPEG, 8 MB cap |

### 4.8 Configuration UX

- Backend selector: **`Auto` (default) | ZeroClaw | Direct provider | Local only**. New field, not the legacy `ai_chat_provider` (F16). Single nested struct stored as one `UiSettings` field to avoid editing the 4 mirrored places (struct, DB struct, two `From` impls).
- ZeroClaw profile: `name, host, agent, complete_hint ("quick"), allow_images, tools_enabled, tools_bind, tools_port`. Token in OS keyring (`keyring` crate) with a 0600 file fallback under `directories::ProjectDirs`, **not** in SurrealDB (F7). Pattern: `zc-codexd` keeps `~/.config/zc-codexd/token` 0600.
- Onboarding card on first run: (1) auto-detect `~/.config/zc-codex/config.json` first `kind=ZeroClaw` machine (selected index first; token never displayed) - same logic as `CUI zeroclaw.rs:279`; (2) or enter host + one-time pairing code -> `POST /pair`; (3) Detect agents via `agent-options`; (4) show health + "Send test message".
- Host parsing identical to `Gateway::host_port` (bare host -> `:42617`; `wss://` stays TLS, scheme never rewritten).
- Prefer `Authorization: Bearer` on the WS upgrade; consumers use `?token=` because it survived their handshake path - test through `wss://claw.shadowbroker.app` before choosing **[UNVERIFIED]**.
- Direct providers keep their settings, but behind one `ProviderResolver` (fixes F8) with per-provider model fields; the OpenRouter model browser moves here.
- Settings home: dedicated Settings > AI page (Connection, Tools server, Direct providers, Local models, Privacy) instead of the 700-line modal in `receive.rs`.

### 4.9 Security model

| Concern | Control |
| --- | --- |
| Gateway token | keyring/0600 file, redacted `Debug`, `url_for_log()`, never in logs/URLs shown, never in SurrealDB |
| Existing leaks | remove `log::error!` of `cfg` and message bodies (F6) first |
| Tool server auth | random 256-bit bearer per install, constant-time compare, stored 0600; bind `127.0.0.1` by default; tailnet bind is an explicit toggle; reject requests with an `Origin` header; keep rmcp `allowed_hosts` explicit (add only the chosen tailnet IP/host) |
| File access | library-id/path allow-list under indexed roots; no raw bytes except bounded previews; no delete/move tools; symlinks refused; Windows release runs elevated (F13) so this is mandatory |
| Write safety | metadata writes only as proposals (Refine tab); bulk taxonomy ops show affected-row counts; undo journal before any future direct-apply |
| Confirmation prompts | two layers: ZeroClaw `approval_request` for gated tools (`always_ask` on `aimedia__image_edit_run`, `aimedia__taxonomy_propose_rename`, others by default `auto_approve` for L0), rendered in-app with `arguments_summary`, JSON args, countdown from `timeout_secs`, buttons Approve / Always / Deny; and in-app proposal review |
| Remote prompt injection | tool results and filenames are untrusted data; do not let tool output escalate privileges; cap lengths; strip control chars |
| Image egress | explicit per-profile switch; thumbnails not originals |
| Rate/cost | one GPU job at a time; per-minute tool-call cap; refine batches serialized |

---

## 5. Integration design

### 5.1 Crate and module boundaries

```
AiMediaExplorer/
  Cargo.toml                       [workspace] members = ["crates/zeroclaw-client"]
  crates/zeroclaw-client/          pure Rust, no egui/candle/surreal
    src/lib.rs
    src/gateway.rs                 Gateway{host,tls,agent}, host_port, urls, redaction, import_zc_codex()
    src/frames.rs                  ServerFrame (serde tag="type", Unknown(Value)), ClientFrame
    src/chat.rs                    ChatClient: connect/resume, send, approve, stop, events
    src/rest.rs                    complete, agents, health, sessions, messages, state, abort, memory, pair
    src/history.rs                 unwrap_stored, parse_history, strip_date_stamp, inline-image markers
    src/event.rs                   ChatEvent, TurnId, Usage
    src/mock.rs   (feature "mock") loopback gateway + JSONL replay
    src/bin/zcprobe.rs             CLI: probe | complete | chat | sessions | pair
  src/zc/                          app glue
    mod.rs                         ZcRuntime (handle, cmd/evt channels, egui::Context)
    settings.rs  profile.rs  secrets.rs
    health.rs                      ZEROCLAW status chip + 45 s sweep
    chat_bridge.rs  complete_jobs.rs
    tools/{mod,server,catalog,policy,gpu_queue}.rs   MCP server + tool impls
  src/ui/assistant/                split from assistant.rs (4.4)
  src/ui/settings/ai.rs            new Settings > AI page
```

Cargo changes are small but touch `Cargo.toml`; other worktrees are editing it, so land the workspace + crate in one isolated commit early.

### 5.2 Core types (sketch)

```rust
pub struct Gateway { pub host: String, pub agent: String }

pub enum ChatEvent {
    Connected { session_id: String, resumed: bool, message_count: u64 },
    TurnStarted { turn: u64 },
    Text { turn: u64, segment: u32, delta: String },
    Reasoning { turn: u64, segment: u32, delta: String },
    ToolCall { turn: u64, id: String, name: String, args: serde_json::Value },
    ToolResult { turn: u64, id: String, name: String, output: String },
    Approval { request_id: String, tool: String, summary: String, args: serde_json::Value, timeout_secs: u64 },
    Plan { entries: Vec<PlanEntry> },
    Usage { input: Option<u64>, output: Option<u64>, window: Option<u64>, cost_usd: Option<f64>, model: Option<String> },
    TurnDone { turn: u64, full: String },
    TurnAborted { turn: u64 },
    Error { code: Option<String>, message: String, ends_turn: bool },
    Closed { reason: String },
}

pub enum Health { Unknown, Checking, Online(String), Unauthorized, Offline(String), Unconfigured(String) }
```

Unknown frame types are kept as `Unknown(Value)` and surfaced to the debug tap, never dropped (the gateway adds frame types over time: `CUI zeroclaw.rs:125`).

### 5.3 Async/streaming bridge into egui

- Reuse the app's existing tokio runtime (`#[tokio::main]`, `tokio::spawn` is used everywhere). `ZcRuntime` holds `egui::Context` + `crossbeam` `Sender<Cmd>` / `Receiver<Msg>` (crossbeam is already the app's idiom).
- Worker calls `ctx.request_repaint()` after each flush instead of the UI polling with `request_repaint_after(50ms)` (assistant.rs:850). Coalesce text/thinking deltas to <= ~30 Hz (100 ms timer, immediate flush on `done`/tool boundaries; same idea as `Coalescer`).
- Bounded event channel (1024, as in `ZCX`), UI drains up to N events per frame.
- Every event is tagged `(session, turn)`; transcript entries are keyed `session:nonce:turn:segment:toolIndex` (the fix for "reply overwrites previous turn", `ZCX README.md:309-313`).
- Cancellation: `Cmd::Stop` -> REST abort + local suppression; `Cmd::Disconnect` drops the last client clone, which closes the socket (leaked sockets otherwise, `ZCX README.md:320-326`).
- Writes: 15 s timeout with acks; reconnect never auto-resends a prompt (failed sends restore the draft).
- Status integration: `ZEROCLAW_STATUS` in the existing registry (`ui/status.rs:203` pattern); `done.model/provider` into the hover card.

### 5.4 Connection/health UI, debug panel, CLI

- Health probe = `GET /health` (anon: reachable, `require_pairing`) then `GET /api/config/agent-options` (authed: token valid and alias exists) with a 6 s budget; distinguishes "refused/offline", "401", "unknown agent".
- Debug panel (user preference: every feature gets one): connection state machine, last 500 frames as a filterable list with `egui_json_tree` (already a dependency), record-to-JSONL toggle, manual raw-frame send, Force reconnect / Drop socket, session inspector (`state`, `running`, `messages`), `/api/complete` playground (hint/system/prompt/temp), tool tester auto-generated from tool JSON schemas (in-process and via HTTP with the real bearer), "Ask gateway to ping tools" (prompt that calls `aimedia__util_ping`), usage/cost totals.
- CLI in the client crate (`zcprobe`) mirrors `ZCX examples/web_tools_probe.rs`: `probe`, `complete --hint quick`, `chat --agent media --prompt ...`, `sessions`, `pair --code`. Runs without CUDA.

### 5.5 Test strategy

| Layer | What | How |
| --- | --- | --- |
| Pure unit | frame -> event mapping, `unwrap_stored`, history parsing, date-stamp strip, URL building, token redaction | port the cases from `ZCX android/src/zeroclaw.rs:1192+` tests verbatim |
| Loopback mock gateway | scripted scenarios: plain turn, thinking + tool_call + approval, `error` codes ending turn, wrong session on resume, mid-turn drop + `turn_resume`, 401, slow thinking, oversize frames, ping loss | `tokio_tungstenite::accept_async` + axum REST (pattern: `ZCX tests/cancellation.rs`, `zeroclaw.rs:1216`) |
| Record/replay | debug panel records `{t_ms, dir, frame}` JSONL with token redaction; `MockGateway::replay(path, speed)` re-serves it; fixtures committed under `crates/zeroclaw-client/tests/fixtures/` | captured from the real gateway once per behavior |
| Live (ignored) | `ZC_HOST`, `ZC_TOKEN_FILE`, `ZC_AGENT` env; connect, 1 turn, resume, abort, `complete` | `cargo test -p zeroclaw-client -- --ignored live_*` (pattern: `ZCX README.md:378-383`) |
| MCP tools | in-process router tests; HTTP e2e with rmcp client; contract test that every tool name matches `^[a-z0-9_]{1,40}$`, schemas are objects, `aimedia__`+name <= 64; policy tests for path allow-list | `cargo test` in the app crate (needs CUDA build) or split pure policy into the client crate |
| UI | headless layout test for transcript wrapping (pattern: `jsontree.rs` guard in `ZCX`); manual debug panel | |

### 5.6 Dependency approach

| Option | Pros | Cons | Verdict |
| --- | --- | --- | --- |
| **1. In-repo workspace crate `crates/zeroclaw-client`, later promoted to its own git repo** | no new repo to manage; builds/tests without CUDA; one place for protocol knowledge; consistent with user's `path =` and `git = ... branch = "egui-037"` habits; `tokio-tungstenite 0.28.0` is already in `Cargo.lock` (via surrealdb `protocol-ws`), so no extra compile cost | touches `Cargo.toml` (merge conflicts with parallel worktrees) | **recommended** |
| 2. Separate git repo now (`shadowbrok3r/zeroclaw-client`) used by AME, ZCX android, CUI, MT | kills the 3-way duplication | extraction work up front; `ZCX android` pins old `reqwest 0.12`, `MT` has its own vendoring; slows phase 1 | do after phase 4 |
| 3. Depend on `zc-codex` lib | has `Event`, `Coalescer` | Codex-specific, pulls daemon/socket2/process code, reqwest 0.12 duplicate, the ZeroClaw client is NOT in it | no |
| 4. Depend on `zeroclaw-api`/`zeroclaw-config` | typed `TurnEvent` | `/ws/chat` frames are hand-built `json!` (no shared serde types); edition 2024 / rust 1.96 workspace; heavy graph; internal churn | no |
| 5. Copy `ZCX/android/src/zeroclaw.rs` into AME | fastest | 4th copy, android-specific types (`Machine`, `Event`, `session::with_images`) | no |

Versions to align: `reqwest 0.13` (app) with `rustls` and `json`; `tokio-tungstenite 0.28` with a rustls roots feature; `futures-util`; install the rustls default provider at startup (F14). rmcp: add feature `transport-streamable-http-server` (and `server-side-http`) to the existing `rmcp = "3.5.1"`; `axum = "0.8"` (already resolved 0.8.9).

---

## 6. Phased implementation plan (each step shippable)

| Phase | Deliverable | Ships as | Depends | Effort | Main risk |
| --- | --- | --- | --- | --- | --- |
| 0. Hygiene | remove duplicate stdio MCP spawn (F1); stop logging cfg/images (F6); make `simple_text_completion` return errors (F9); install rustls provider (F14); fix `find_similar_by_embedding` k/ef and expose similarity (F3); rename MCP tools to underscores + add server info; unit tests for the MCP router | safer app, no new feature | - | 1-2 d | low |
| 1. `zeroclaw-client` + `zcprobe` | frames, chat, rest, history, mock + replay, CLI; unit + loopback tests | CLI usable against the real gateway; zero UI change | 0 | 3-4 d | wire drift: pin against fork `f076d7b55` |
| 2. Connection settings, health, debug panel | Settings > AI > Connection, onboarding (import from zc-codex config / pairing), keyring/0600 token, `ZEROCLAW` status chip, frame inspector, `/api/complete` playground | "Connect ZeroClaw" works, nothing depends on it | 1 | 2-3 d | keyring on Linux (Secret Service) fallback |
| 3. One-shot AI jobs on `/api/complete` | ZeroClaw refine source (batch JSON), session title generation, tag suggestions in preview pane, optional NL->filter | AI Refinements gains a ZeroClaw option; no chat change | 2 | 2-3 d | JSON adherence of the `quick` model; batch size vs 64 KiB |
| 4. Assistant chat on `/ws/chat` | `ChatBackend`, transcript `Entry` model, approvals UI, Stop/abort, reconnect + `turn_resume`, lazy history, local cache; split `assistant.rs`; Direct backend gets real history | **default backend = ZeroClaw (Auto)**; old providers remain | 2 | 5-7 d | session-id resume subtleties; vision availability; long thinking latency |
| 5. Tools server + gateway provisioning | Streamable HTTP MCP in the GUI process, bearer + host allow-list, catalog v1 (read-only + previews), `GpuQueue`, path policy, `mcp-proxy` subcommand (optional), Appendix-A snippet + Apply button, `media` agent workspace files | agent can search/describe the library | 4 | 4-6 d | gateway reachability; names/filters per provider; MCP image content |
| 6. Write path + UI actions | `metadata_propose`, `taxonomy_propose_rename`, `ui_open_results`, `app_get_context`, image-edit job tool + completion push | agent-assisted curation with review | 5 | 4-5 d | proposal volume; GPU contention |
| 7. Memory, polish | library style guide in `/api/memory` and pinned `library://guide`; gateway session browser; usage/cost view; optional vision turns; migrate other apps to the shared crate (option 2) | | 6 | open | |

Order note: phases 2-3 give user-visible AI value before the large UI reorganization lands; phase 4 should wait for the dock layout decision (Assistant as a tab) but its core (`ChatBackend`, bridge) is UI-independent.

---

## 7. Risks, unknowns, spikes

| # | Item | Spike |
| --- | --- | --- |
| R1 | Gateway -> desktop reachability and Windows firewall for the tools port | bind to tailnet IP, add `allowed_hosts`, call from the box with curl using the bearer |
| R2 | MCP tool name filtering on non-OpenRouter providers (the user's pool is llama.cpp via LiteLLM) | list tools in a test session; confirm underscore names survive |
| R3 | MCP `image` content handling in ZeroClaw (previews) | return an image block and see what the model receives |
| R4 | `/api/complete` with `[IMAGE:..]` markers within the 64 KiB cap | POST a 30 KB JPEG marker to a vision-capable hint |
| R5 | Vision availability for the `media` agent (`[multimodal] vision_model_provider`) | attach an image in a `connect`ed session |
| R6 | WS auth via `Authorization` header through `wss://claw.shadowbroker.app` | try header vs `?token=` |
| R7 | `/ws/nodes` tool wiring and name validity | only if option A proves unreachable |
| R8 | rustls provider ambiguity at runtime (F14) | open a `wss://` socket in a test binary of the app's feature set |
| R9 | `PATCH /api/config` exact keys for `[[mcp.servers]]` / bundles | read `/api/docs` on the box before building "Apply" |
| R10 | Pool saturation (2 sessions per box) when chat + refine + other apps run together | enforce app-side single job; surface `SESSION_QUEUE_*` errors |
| R11 | `/webhook?agent=` (+ `X-Session-Id`) as a non-streaming agent-with-tools channel for unattended batch jobs | probe once; auth mode depends on `gateway.require_pairing` / `webhook_secret` |
| R12 | Fork drift: frame types are hand-built JSON; fork features (`turn_resume`) are not upstream | keep `Unknown` passthrough + replay fixtures per behavior |

---

## 8. Open questions for the user

1. Run the `media` agent only on the remote gateway, or also support a local desktop `zeroclaw daemon` (option B)?
2. OK to expose the desktop MCP port on the tailnet with a bearer token (needed for the remote gateway)?
3. Which model/pool for `media` (text-only qwen vs a vision model)? Determines whether image chat works at all.
4. May the app change gateway config via `PATCH /api/config` ("Apply"), or only show a snippet?
5. Keep the Windows release manifest `requireAdministrator` (`build.rs:18`)?
6. Existing assistant sessions: keep read-only, migrate, or drop?

---

## Appendix A - illustrative ZeroClaw-side config (verify keys against `/api/docs`)

```toml
# gateway host, edited via `zeroclaw config` or PATCH /api/config
[[mcp.servers]]
name = "aimedia"
transport = "http"
url = "http://100.116.134.128:9014/mcp"
headers = { Authorization = "Bearer <app-generated-token>" }
tool_timeout_secs = 300
pinned_resources = ["library://guide"]

[mcp_bundles.aimedia]
servers = ["aimedia"]

[risk_profiles.media]
level = "supervised"
auto_approve = ["aimedia__library_stats", "aimedia__library_search_text", "aimedia__library_search_semantic", "aimedia__library_get_item"]
always_ask = ["aimedia__image_edit_run", "aimedia__taxonomy_propose_rename"]

[agents.media]
risk_profile = "media"
mcp_bundles = ["aimedia"]
# model_provider = "<alias>"   # choose a vision-capable one if image chat is wanted
```

## Appendix B - frame cheat sheet

```text
-> {"type":"connect","device_name":"aimediaexplorer","capabilities":["text"]}
<- {"type":"session_start","session_id":"<uuid>","resumed":false,"message_count":0}
<- {"type":"connected","message":"Connection established"}
-> {"type":"message","content":"find sunsets"}
<- {"type":"thinking","content":"..."}
<- {"type":"tool_call","id":"c1","name":"aimedia__library_search_semantic","args":{"query":"sunset"}}
<- {"type":"approval_request","request_id":"r1","tool":"aimedia__image_edit_run","arguments_summary":"...","arguments":{},"timeout_secs":300}
-> {"type":"approval_response","request_id":"r1","decision":"approve"}
<- {"type":"tool_result","id":"c1","name":"aimedia__library_search_semantic","output":"..."}
<- {"type":"chunk","content":"Found 12 ..."}
<- {"type":"done","full_response":"...","input_tokens":1234,"max_context_tokens":131072,"model":"...","provider":"..."}
```

Resume URL: `/ws/chat?agent=media&session_id=<uuid>` (only `session_id` works). Stop: `POST /api/sessions/<uuid>/abort`. History: `GET /api/sessions/<uuid>/messages`.

## Appendix C - key code pointers

- ZeroClaw WS handler: `ZC/crates/zeroclaw-gateway/src/ws.rs` (auth 93-150, session 345-500, frames 1685-1730, approval 853, done 2028, errors 2157)
- ZeroClaw routes: `ZC/crates/zeroclaw-gateway/src/lib.rs:1705-2052`
- ZeroClaw MCP client: `ZC/crates/zeroclaw-tools/src/mcp_client.rs:890-1280`, `mcp_transport.rs:1136+`
- Reference chat client: `ZCX/android/src/zeroclaw.rs`
- Reference one-turn client: `CUI/src/zeroclaw.rs`
- Reference default-backend switch: `MT/displays/src/ai/mod.rs:34`, `MT/displays/src/mcp/openai_bridge.rs:93,247,310`
- Reference in-GUI MCP host: `MT/displays/src/plugins/mcp_bridge.rs:13773`
- AME assistant: `AME/src/ui/assistant.rs`; settings modal `AME/src/receive.rs:142-820`; settings struct `AME/src/database/settings.rs:53`; MCP `AME/src/ai/mcp/mod.rs`
