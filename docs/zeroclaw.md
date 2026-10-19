# ZeroClaw assistant

The assistant answers through a ZeroClaw gateway by default. It lives in the Ask tab of the right rail (Ctrl+J) and can be popped out into its own window. Code: `src/zc/` (profile, token, health, runtime, connection card, `zc` command) and `src/ui/assistant/` (transcript, composer, backends). Wire details are in `docs/research/zeroclaw-integration.md` and `crates/zeroclaw-client/README.md`; the tools the agent can call are in `docs/mcp.md`.

## Backends

The backend pick is part of the gateway profile (`auto`, `zeroclaw`, `direct`, `local`) and is changed from the chip at the top of the assistant or from Settings.

| Pick | Answers with | Needs |
|---|---|---|
| Auto (default) | ZeroClaw when a gateway address, an agent alias and a token exist and the last probe was Online; the setup card before that | nothing else |
| ZeroClaw | the gateway agent over `/ws/chat` | same as above; reports what is missing when it is not there |
| Direct provider | an OpenAI-compatible endpoint or OpenRouter, with the conversation so far as history | a provider and key in Settings |
| Local only | CLIP text search and the local vision model | nothing |

Auto never falls back to a cloud provider. An explicit pick that cannot answer says why and offers the fix (Pair, Settings, Retry); it is not replaced by another backend. Quick actions (`/help`, `/ping`, `/search`, `/describe`) run locally in every backend.

## Profile and token

The profile is one nested settings field (`UiSettings::zeroclaw`, database settings record): `name`, `host`, `agent`, `complete_hint`, `allow_images`, `tools_enabled`, `tools_bind`, `tools_port`, `backend`. `host` is a bare host (port 42617 is added), `host:port` or an `http(s)`/`ws(s)` URL; `wss://` stays TLS.

The token is never in the database or a log. It lives in `<config>/smart-media/zeroclaw_token` (mode 0600 in a 0700 folder; the OS keyring is not used because it needs system services this build does not depend on). Debug output of every type that could hold it is redacted, and error text, notes and the frame recording have the token removed.

## Health

A probe is `GET /health`, then `GET /api/config/agent-options` with the token, within 6 s. Results:

| State | Meaning | Engine dot | Needs attention |
|---|---|---|---|
| Not set up | no address, agent or token | idle | |
| Checking | probe running | initializing | |
| Online | token accepted and the agent exists | idle, hover card shows agent, host and model | |
| Offline | refused, timeout or an HTTP error | waiting | |
| Token refused | HTTP 401 or 403 | error | "ZeroClaw token refused" with Re-pair |
| Unknown agent | the gateway has no agent of that name | waiting | |

A sweep probes every 45 s. A refused token is never probed again or reconnected automatically; Check, Pair, Import or a new token starts over. The status bar shows the `ZEROCLAW` engine dot like the other engines.

## Connection card

Settings > Assistant calls `zc::connection_ui`, the first-run wizard calls `zc::onboarding_ui`. Fields: gateway address, agent alias (Detect agents fills a picker), pairing code (Pair exchanges it for a token), Import from a ZC Codex config (`~/.config/zc-codex/config.json`: the selected ZeroClaw machine, else the first; the token is stored and never shown), Check, Send test message (`/api/complete`), Forget token. Sections for the tools server (status, masked token, Start, Stop, Rotate, tailnet exposure, copy of the gateway configuration) and privacy (allow images; for a gateway outside your network a second confirmation is needed).

## Chat

- Entries: You, Assistant, Reasoning, Tool call, Tool result, Approval, Plan, Notice. Reasoning and tool rows start collapsed; long messages collapse with Expand.
- Approvals show the tool, a summary, the arguments, a countdown and Approve, Always and Deny. The gateway denies on its own when the time runs out.
- Context chips `@selection`, `@folder`, `@filters` share a text snapshot of the focused browser with the next message. Nothing is attached implicitly.
- Pictures go to the gateway only when attached, only when the profile allows images, downscaled to 1024 px JPEG. Otherwise the file names go in the message.
- Stop calls the gateway's abort endpoint and ignores frames of that turn until it confirms (or 4 s pass). Frames of another session or an older connection are dropped.
- A dropped socket reconnects with the same session id; the gateway replays the running turn (`turn_resume`). A send is never repeated automatically; a failed send gives the draft back.
- The gateway history is read lazily (newest 50 messages, Load earlier for more). A cache of the newest 300 rows per session is kept in `assistant_messages` (fields `kind`, `turn`, `meta`; sessions have `backend`, `agent`, `gateway_id`); the gateway wins where it has history. Sessions from before backends open read-only.
- Titles: the first exchange of a ZeroClaw chat is named through `/api/complete`; other chats use the first words of the question.

## One-shot jobs

Refine, the Ask mode of the query bar and the Suggest button of the Inspector send text-only requests to the gateway's `/api/complete` (no agent, no tools, no session). They share one job service, `src/zc/jobs/`:

- **Single flight.** One job runs at a time; up to 8 wait in a queue (a ninth is refused). The gateway's pool serves about two sessions per box, so jobs never run in parallel.
- **Tracked.** Every job is a task (Refine jobs are Refine tasks, the others Assistant tasks) with progress per request and Cancel in the Task Center, the Refine tab, the query bar or Devtools. Cancelling stops at once, abandons the request in flight and keeps what earlier requests produced.
- **Input cap.** The gateway refuses more than 64 KiB of system plus prompt text. Refine splits its rows into requests of 20 to 40 rows that fit (`zc::jobs::batch`); a request over the cap is refused before it is sent.
- **Hint.** The route hint is the profile's `complete_hint` (default `quick`, `zc` `configure` with `hint`); a job can name another, and the Refine tab has a Hint field.
- **Retries.** Connection errors, timeouts, HTTP 408, 425, 429 and 5xx are retried up to three times with a doubling wait (0.8 s, 1.6 s, capped at 10 s; a timeout only once). Anything else, 400 included, fails the job. HTTP 401 and 403 are never retried: the job fails, health turns to Token refused and the Re-pair entry of Needs attention appears. A connection that still fails after its retries turns health to Offline.
- **Availability.** Jobs run only while the gateway is Online; the Ask segment, the Suggest button and the Refine source switch are disabled otherwise and say why. A gateway whose first probe is still running is waited for (up to 8 s).
- **Usage.** `/api/complete` returns no token counts, so the totals count requests, retries, bytes and time, and estimate tokens at four bytes each.
- **Privacy.** Only text is sent: file names, folder names (never full paths), tags, categories, captions and descriptions. Prompts and replies are kept in memory for Devtools > ZeroClaw jobs (50 jobs, 128 KiB per text), never written to the log (the log holds sizes, counts and times) and never contain the token.

JSON in replies is read tolerantly (`zc::jobs::extract`): code fences, prose around the value, trailing commas, a single object where an array was asked for, and cut-off arrays, of which the complete leading elements are kept.

### Refine source

Generator > Use ZeroClaw in the Refine tab (`docs/tools.md`) asks the gateway for new categories, tags, captions and descriptions. The system prompt carries the reply schema, the library conventions (lowercase tags and categories) and the library's most used tags and categories; each row carries its file name, folder and metadata and, when known, what its nearest neighbours by CLIP carry (the stored embedding of the file; the text of the row embedded by CLIP only when that model is already loaded). Proposals use the generator name `zeroclaw:<hint>`, the source label `ZeroClaw <hint>`, and a confidence that is the neighbours' agreement with the proposed tags and category (none without neighbours). Accepting goes through the metadata journal like every other proposal.

### Ask mode

The fifth segment of the query bar. A sentence becomes a filter through one call whose system prompt holds the exact filter fields, their meaning, today's date, the library's tags and categories and four examples (each checked against the parser by a test). The reply must be a JSON object that the filter parser accepts: unknown fields, malformed dates and wrong types are refused, as is a reply that sets nothing. A valid reply is shown as chips under the field with Apply; applying is `set_filter` with `replace`. Nothing else is ever run from a reply.

### Suggest

Inspector > AI > Suggest (single file) asks for up to eight new tags, a category and a caption from the file's metadata and its neighbours. Each suggestion is a button; a click writes it through the metadata editor as one journal batch ("Accepted suggestion", undone with Ctrl+Z), Add all writes everything as one batch.

## Devtools > ZeroClaw jobs

`--devtools zeroclaw_jobs` or the Devtools menu. Status (availability, hint, queue depth, usage totals), the running job with Cancel, the queue, a submit box for a Manual job, the history (requests, bytes, estimated tokens, time, attempts, outcome) and a viewer of the system text, prompt, reply or parsed reply of every request, with the file paths masked as `<path>` unless unchecked. Driven without the mouse by the `zc` ops `jobs`, `job_submit`, `job_cancel`, `refine` and `suggest`, and by `--query ask:<sentence>` (`docs/commands.md`).

## Devtools > ZeroClaw

`--devtools zeroclaw` or the Devtools menu. Connection state machine and history, Check, Connect, Force reconnect, Drop socket, Ask gateway to ping tools; the last 500 frames in both directions with direction and text filter and a JSON viewer; record to JSONL (token removed; the format `zcprobe` and the mock gateway replay); send a hand-written frame (only `message` and `approval_response` exist on this socket); session inspector (state, running sessions, stored messages); `/api/complete` playground; usage totals per model; notes. Driven without the mouse by the `zc` command (`docs/commands.md`).

## Testing without a real gateway

`zeroclaw-client` has a loopback mock gateway (feature `mock`). `cargo test zc::` and `cargo test ui::assistant::` run the runtime and the panel against it: plain turn, tool call with approval, error ending a turn, mid-turn drop and resume, refused token without retry, Stop, session switch, chips and pictures, titles, layouts from 280 to 760 px. `smart-media --zc-status --zc-url <mock-url> --zc-agent default --zc-token-file <file>` prints the probe result as JSON and exits.

Not verified here: a real gateway (header authentication through `wss://claw.shadowbroker.app`, `turn_resume` after a real drop, vision on the `media` agent), tools reachability from the gateway to this desktop.
