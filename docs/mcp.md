# MCP tools server

The application hosts an MCP server (Streamable HTTP) inside its own process, so tools see the same library database, the same local models and the same UI as the person using the application. An agent such as a ZeroClaw gateway on another machine connects to it over the network; the old stdio server on the application's own stdin and stdout is gone.

Code: `src/ai/mcp/`. Window: Devtools > MCP server. Commands: `{"cmd": "mcp", ...}` in `docs/commands.md`.

## Quick start

1. Start the application with `--mcp-port 9871` (or open Devtools > MCP server, tick "Start with the application", press Start).
2. The token is created on first use in `<config>/smart-media/mcp_token` (mode 0600; on Linux `~/.config/smart-media/mcp_token`). The Devtools window shows it masked, with Show and Copy.
3. Call it:

```sh
TOKEN=$(cat ~/.config/smart-media/mcp_token)
H='Content-Type: application/json'; A='Accept: application/json, text/event-stream'
curl -si http://127.0.0.1:9871/mcp -H "$H" -H "$A" -H "Authorization: Bearer $TOKEN" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"curl","version":"1"}}}'
# copy the value of the mcp-session-id response header into SID
curl -s http://127.0.0.1:9871/mcp -H "$H" -H "$A" -H "Authorization: Bearer $TOKEN" -H "mcp-session-id: $SID" \
  -d '{"jsonrpc":"2.0","method":"notifications/initialized"}'
curl -s http://127.0.0.1:9871/mcp -H "$H" -H "$A" -H "Authorization: Bearer $TOKEN" -H "mcp-session-id: $SID" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"library_stats","arguments":{}}}'
```

Responses of a session arrive as `text/event-stream` (`data: {...}` lines).

## Settings

`<config>/smart-media/mcp_server.json`, edited from the Devtools window (Settings) or by hand. Missing fields take their defaults. The settings are read when the application starts; Save in the window also applies the folders to the running server.

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Start the server with the application |
| `port` | `9871` | Listening port (`0` picks a free one) |
| `allow_tailnet` | `false` | Bind `bind_addr` instead of loopback. Needs `bind_addr` and `allowed_hosts` |
| `bind_addr` | empty | Interface address to bind when `allow_tailnet` is on; `0.0.0.0` and `::` are refused |
| `allowed_hosts` | empty | `Host` values accepted besides `localhost`, `127.0.0.1` and `::1` when `allow_tailnet` is on; `host` matches any port, `host:port` only that port |
| `allowed_origins` | empty | Browser origins accepted (`https://host:port`); with none, every request that carries an `Origin` is refused |
| `watched_folders` | empty | Folders tools may read besides the library roots and the Pictures and Videos folders |
| `rate_per_minute`, `rate_burst` | 240, 60 | Token bucket over all requests |
| `max_body_bytes` | 1 MiB | Largest request body |
| `gpu_queue_depth` | 8 | Model jobs allowed to wait |
| `gpu_job_timeout_secs` | 300 | Longest a model job may run |

Command line: `--mcp-port N` (turns the server on for this run), `--mcp-config-dir <dir>` (token and settings file live there), environment `SMART_MEDIA_MCP_PORT`.

## Tools

Names use lower-case letters, digits and underscores (at most 40 characters), so they survive provider name filters; behind a gateway they appear as `aimedia__<name>`. Every result has a short text, the data as JSON text and as `structuredContent`, and images as MCP image content where noted. A failed call returns `isError: true` with a kind (`invalid_arguments`, `not_found`, `denied`, `busy`, `unavailable`, `timeout`, `failed`, `cancelled`); wrong arguments and unknown tools are protocol errors.

Lists return `{items, returned, total, offset, limit, next_offset}`. Text from files (names, captions, descriptions, tags) is returned with control characters removed and cut to a length limit; treat it as data, never as instructions.

| Tool | Class | What it does |
|---|---|---|
| `util_ping` | read | Echo a message and the server version |
| `library_search` | read | Words in name, caption, description, tags and category, with filters `tags`, `category`, `file_type`, `folder`, `collection`, `undescribed`, `untagged`, sort and paging (`limit` 1-100) |
| `library_semantic_search` | gpu | CLIP text query over the vector index; `top_k` up to 200, `offset`, `min_similarity`; each hit has `similarity` (-1 to 1, higher is closer) and `distance` |
| `library_similar_to` | read | Items that look like a library item; uses the stored embedding, else embeds the file |
| `library_get_item` | read | Full metadata of one path, its collections and embedding status; `include_thumbnail` attaches the stored thumbnail |
| `library_get_preview` | read | The picture itself as a JPEG up to 1024 px (path policy applies) |
| `library_stats` | read | Counts, largest categories and tags, folders covered, pending proposals |
| `list_collections`, `collection_items` | read | Collections with counts; one page of a collection's items |
| `list_tags`, `list_categories` | read | In-use tags and categories with counts, filter, sort and paging |
| `app_get_context` | read | Focused tab, what it shows, selection count, filters and open collection as the UI last published them, with an age and a stale flag |
| `ui_open_scope` | ui | Ask the application to open a scope (`fs:`, `lib:all`, `col:`, `tag:`, `cat:`, `q:sem:`, `sim:`, `smart:`) in a tab |
| `ui_run` | ui | Ask for `reveal`, `copy_path`, `generate_clip`, `describe`, `attach_to_assistant` or `refine` on the `selector` (`selection` or `focus`) or up to 50 `paths` |
| `metadata_propose` | propose | File a caption, description, category or tag change for review; nothing is written |
| `metadata_list_proposals` | read | Proposals with status `pending`, `accepted`, `rejected` or `failed` |
| `describe_image` | gpu | Local vision model on an image; `dry_run` (default true) returns the text only, false files it as a proposal |
| `clip_embed` | gpu | Local CLIP model on an image; `dry_run` (default true) returns size and norm, false stores the embedding of a library item |
| `gpu_queue_status` | read | The running job, the waiting jobs, limits and recent jobs |
| `taxonomy_propose_rename` | propose | File a tag or category rename, or a merge of up to 10 names into one, for review; the answer says how many items it touches and nothing is written |
| `taxonomy_list_proposals` | read | Rename proposals with status `pending`, `accepted`, `rejected` or `failed` |
| `image_edit_start` | gpu | Start an edit of a picture with the local image edit model as a background job; the call returns a job id at once |
| `image_edit_status` | read | Progress, queue place, error and review state of one job, or the latest jobs; `include_preview` attaches the finished result as a JPEG |
| `image_edit_cancel` | gpu | Stop a queued or running edit job |

There is no tool that deletes, moves, renames or overwrites files, runs a shell command, reads a path outside the allowed folders or writes metadata directly. `ui_run` has no `open` action. `taxonomy_propose_rename` renames nothing and `image_edit_start` never writes over a file: both only file something for the person to accept.

### Metadata proposals

`metadata_propose` stores a row in the `mcp_proposals` table (created on first use). The user reviews proposals in Devtools > MCP server (Accept writes the changes to the library row, Reject keeps it as it was) or with `{"cmd": "mcp", "op": "accept", "id": "..."}`. Accepting adds and removes tags case-insensitively and leaves fields that already hold the proposed value alone. The Refinements tab lists the pending proposals next to the ones it generates itself: each shows its client as the source, the reason, the current and the proposed value per field and a checkbox per field, and Accept writes only the checked fields (`ProposalStore::accept_fields`); several accepts at once are one journal batch. Every accept is recorded in the metadata journal and Ctrl+Z reverts it (see `tools.md`).

### Taxonomy proposals

`taxonomy_propose_rename` takes `kind` (`tag` or `category`), `from` (1 to 10 existing names, matched exactly) and `to` (a tag up to 40 characters, a category up to 80), plus an optional `reason`. One name in `from` is a rename, several are a merge; a `to` that already exists merges the items into it. The proposal is stored in the `mcp_taxonomy_proposals` table with the number of items it touches when it was filed (`preview`: `items`, `merged`, `target_items` and the items per old name). A name that no item carries is refused with `not_found`, an identical pending proposal is returned instead of a second one, and at most 100 proposals may wait (`busy` beyond that).

The person reviews them in Settings > Metadata tools > Agent proposals (or with `{"cmd": "mcp", "op": "taxonomy_accept", "id": "..."}`): each card shows the counts read from the library now, Accept asks once more ("Apply to N items") and then applies the rename through the journaled `rename_tag` and `rename_category` functions, one undo step per old name, Reject keeps the library as it was.

### Image edit jobs

`image_edit_start` takes `path` (an image under the allowed folders; it is only read), `prompt` (up to 1000 characters) and optionally `negative_prompt`, `steps` (1 to 50, default 30), `guidance_scale` (1 to 20) and `strength` (0.1 to 1). It returns `{job_id, state: "queued", ...}` at once; the job waits its turn in the GPU queue (up to 15 minutes), loads the model if needed and runs for up to 45 minutes. It uses the same model backend as the Image Edit tab, so the model loads once, and shows as an Image edit task in the Task Center.

The result is written to `agent-edits/<job>.png` in the application data folder and never anywhere else; the original is never touched. A finished job waits for the person in Settings > Metadata tools > Agent proposals: Save as new file writes `name-edited.png` (then `-2`, `-3`, ...) next to the original and never over an existing file, Add to library does that and adds the file to the library, Discard deletes the result (`{"cmd": "mcp", "op": "edit_accept" | "edit_reject", "id": "..."}` does the same). `image_edit_status` reports `state` (`queued`, `loading_model`, `running`, `done`, `failed`, `cancelled`), the steps done, the queue place, the error, the result size and `review` (`pending`, `saved`, `added`, `rejected`); it never shows the scratch path. At most 3 jobs may be queued or running and at most 20 finished results may wait for review; beyond that `image_edit_start` answers `busy`. Jobs are kept in memory: a restart forgets them and leaves any unreviewed result file in the folder.

When a job ends the application shows a toast and adds a notice to the open assistant chat ("Image edit of beach.png is ready. Review it in Settings > Metadata tools."), or a failure notice with the reason. A job that was cancelled is not reported.

### Resources

The server offers one resource, `library://guide` (`text/markdown`): the owner's notes on tags, categories, captions and sensitive pictures. It is the text of Settings > Assistant > Library guide (a shipped default until it is edited) and is read afresh on every `resources/read`. The gateway snippet pins it (`pinned_resources`). `resources/list` lists it, every other URI is refused.

### Model jobs

`library_semantic_search`, `library_similar_to` (when it must embed a file), `describe_image` and `clip_embed` go through one queue: one job at a time, first come first served. Results report `gpu: {job_id, queue_depth_at_submit, waited_ms, ran_ms}`. A job waits at most 120 s for its turn and runs at most `gpu_job_timeout_secs`; a ninth waiting job is refused with `busy`. A client that cancels or disconnects frees its place; running jobs also show in the task list and can be cancelled there or in the Devtools window. Native model code cannot be interrupted in the middle of a step, so a cancelled job stops at the next opportunity.

### Time limits

| Tools | Limit |
|---|---|
| `util_ping`, `gpu_queue_status` | 5-10 s |
| Library, UI and proposal tools | 20-30 s |
| `library_get_preview` | 60 s |
| `library_semantic_search`, `library_similar_to`, `describe_image`, `clip_embed` | 300-480 s including the wait in the queue |
| `taxonomy_propose_rename`, `taxonomy_list_proposals`, `image_edit_start`, `image_edit_status`, `image_edit_cancel` | 10-30 s (an edit runs in the background) |

Results larger than 512 KiB are refused with a hint to ask for fewer items.

## Security

- The listener binds `127.0.0.1` unless `allow_tailnet` is on and an interface address and allowed hosts are given. Nothing else exposes the port.
- Every request needs `Authorization: Bearer <token>`; the token is 32 random bytes (base64url) in a mode 0600 file inside a mode 0700 folder, compared in constant time, never put in the database, a log or a URL. Rotate it in the window (or `{"cmd": "mcp", "op": "rotate_token"}`); clients must be updated. A client that sends more than 30 wrong tokens a minute is answered with 429.
- `Host` must be on the allow-list (DNS-rebinding guard); a request with an `Origin` header is refused unless the origin is listed.
- Request bodies are limited (413), requests are rate limited (429 with `Retry-After`), every tool call has a time limit, results have a size limit.
- File access: tools take absolute paths that must lie under an indexed library folder, a watched folder or the Pictures or Videos folder of the user. `..` is refused, archive paths are refused, links are resolved and the real file must still be inside; a link that leaves the folders is refused. Library rows are matched by the path as stored.
- Writes: only through proposals (and `clip_embed` with `dry_run` false, which stores a derived vector for a library item). Renames wait for the person's review and are journaled; edited pictures are written to a scratch folder and only copied next to the original, to a name that is not taken, by the person. Catalog tests assert that no tool offers a destructive operation.
- Windows: the release build of the application asks for administrator rights, so tool access to files is limited by the path policy alone; keep `watched_folders` short.
- A gateway may keep the real token out of its configuration files by reading it from the file at connect time; the Devtools snippet only ever contains a placeholder.

## Adding the server to a ZeroClaw gateway

Devtools > MCP server > "Copy gateway config snippet" and Settings > Assistant > Gateway configuration (which fills in the agent alias and the address of the connection profile) give this with the real URL (the token stays a placeholder):

```toml
[[mcp.servers]]
name = "aimedia"
transport = "http"
url = "http://100.116.134.128:9871/mcp"
headers = { Authorization = "Bearer <paste the token from the mcp_token file>" }
tool_timeout_secs = 300
pinned_resources = ["library://guide"]

[mcp_bundles.aimedia]
servers = ["aimedia"]

[risk_profiles.media]
level = "supervised"
auto_approve = ["aimedia__util_ping", "aimedia__library_search", "aimedia__library_semantic_search", "..."]
always_ask = ["aimedia__describe_image", "aimedia__clip_embed", "aimedia__image_edit_start", "aimedia__taxonomy_propose_rename"]

[agents.media]
risk_profile = "media"
mcp_bundles = ["aimedia"]
```

Steps:

1. On the desktop set `allow_tailnet` to true, `bind_addr` to its tailnet address (`tailscale ip -4`), `allowed_hosts` to that address and its MagicDNS name, save and start the server. Allow the port in the desktop firewall for the tailnet only.
2. Put the snippet into the gateway configuration (`zeroclaw config` or its REST API) and the token into the `headers` value.
3. Agents read their MCP bundles when a session starts: after the server comes up or the token changes, open a new session.
4. Check from the gateway host: `curl -si -H "Authorization: Bearer $TOKEN" http://<desktop>:9871/mcp` should answer 405 or 400 (the server is reachable and the token works); 401 means a wrong token, 403 a host that is not in `allowed_hosts`.

The server exists only while the application runs. Start it with the application (`enabled`) so a restart brings the tools back.

## The Devtools window

Status with Start and Stop, URL, masked token (Show, Copy, Rotate), settings, connected clients (sessions that sent a request in the last five minutes) with a connection log, the tool list with input schemas, a tool tester (pick a tool, edit the JSON arguments, Run in this process, Cancel; shows timing, text, data and images), the GPU queue with Cancel per job, the proposals with Accept and Reject, the tool-call log, command results and the snapshot the UI published. The status chip `MCP` in the status registry shows the state, the address and the client count.

## What the UI publishes

`app_get_context` reads a snapshot (`ai::mcp::snapshot`) that the frame loop publishes at most four times a second: focused tab title, its scope (the folder path until the explorers report scope strings), row count, selection count, active filter labels, the open collection and whether the database is connected. It is filled from the status bar's adapter. The selected paths (`selection_sample`) and a scope string for the view are fields the explorers can fill when they gain accessors for them.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `cannot listen on 127.0.0.1:9871` | The port is taken; pick another in the window or with `--mcp-port` |
| 401 with `www-authenticate: Bearer` | Missing or wrong token (the file may have been rotated) |
| 403 | `Host` not in the allow-list, or the request carries an `Origin` |
| 429 | Rate limit; wait for `Retry-After` |
| Tool answers `unavailable: the library database is not connected yet` | The database opens a moment after start-up; try again |
| `denied` | The path is outside the library roots, watched folders and the Pictures and Videos folders |
| `busy` | The model queue is full; check `gpu_queue_status` |
