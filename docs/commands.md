# Commands

Every request to the application is an `AppCommand` (`src/command/`). Menus, buttons, the command runner in Devtools, start-up flags and the assistant or MCP integration all send the same commands. A command is a JSON object with a `cmd` field; the other fields depend on the command.

The JSON form is the stable surface. Names are `snake_case`. Fields marked optional can be left out. Unknown command names, unknown enum values and malformed scopes are rejected with an error, never ignored.

## Sending commands

| From | How |
|---|---|
| Rust | `crate::command::send(AppCommand::...)` from any thread; the next frame runs it |
| Text | `crate::command::send_text(text)`: a JSON object, a JSON array of objects, or a scope string (opens the scope in a new tab) |
| Devtools | View > Devtools > Command log: the runner box takes the same text; Run again re-sends a logged command |
| Command line | `--run '<json or scope>'`, `--scope '<scope>'` (both repeatable), `--devtools tasks,command-log,perf` and `--wizard <step>` queue commands for the first frame; `--dump-state <path>` writes the state snapshot once the application is idle |

The dispatcher drains the bus once per frame (at most 256 commands), runs each against the application and records it in the command log (500 entries; Devtools > Command log). A failed command shows a warning toast naming it, at most three per frame.

## Scopes

A scope says what a browser tab shows. Its string form is lossless: `Scope::parse(scope.format()) == scope` for every valid scope. In JSON a scope is that string.

| Scope | String | Example |
|---|---|---|
| Folder | `fs:<path>` | `fs:/home/u/Pictures` |
| Archive | the virtual path (`zip://`, `tar://`, `7z://`) | `zip:///data/a.zip!/inner` |
| Library | `lib:all` | `lib:all` |
| Recent | `recent:all` | `recent:all` |
| Collection | `col:<name>` | `col:Holidays` |
| Tag | `tag:<tag>` | `tag:sunset` |
| Category | `cat:<category>` | `cat:landscape` |
| Semantic search | `q:sem:"<query>"` (JSON-escaped; quotes optional when typing) | `q:sem:"a red car"` |
| Similar to a file | `sim:<path>` | `sim:/a/b.jpg` |
| Scan result | `scan:<id>` | `scan:42` |
| Smart view | `smart:<name>` | `smart:no-clip` |

Smart views: `no-clip`, `undescribed`, `untagged`, `failed-thumbs`, `duplicates`.

Every scope can be opened. A scope shown in a tab replaces what the tab showed and is remembered in the history of that tab (see [Navigation](#navigation)).

| Scope | What the tab lists |
|---|---|
| `fs:<path>` | The folder, listed by the file scanner and merged with the library rows of the folder. `recursive` lists everything below it and records the scan |
| archive | The inside of the archive, with the password prompt for encrypted ones |
| `lib:all` | Every row of the library in path order, read in batches of 500 while the view fills |
| `recent:all` | The 1000 newest additions to the library, newest first |
| `col:<name>` | The members of the collection in path order, 500 at a time. Members that are not in the library are listed from the file system and carry the Not indexed badge |
| `tag:<tag>` | Rows with exactly that tag |
| `cat:<category>` | Rows of exactly that category |
| `q:sem:"<text>"` | The files closest to the text by CLIP, 100 per page, nearest first, with a Similarity column |
| `sim:<path>` | The files closest to the CLIP embedding of that file, the file itself left out, 100 per page. A file without an embedding fails with a message saying so |
| `scan:<id>` | The files a recorded recursive scan found. The id is the number in the scan's title (`Scan #<id>`); ids are unique across launches |
| `smart:no-clip` | Images with no CLIP embedding |
| `smart:undescribed` | Images with no description or an empty one |
| `smart:untagged` | Rows with no tags |
| `smart:failed-thumbs` | Files whose thumbnail failed in any tab since the application started; the same list the Needs attention entry counts |
| `smart:duplicates` | Files with identical contents, see below |

Search scopes show the first page and a footer under the rows with **Load more**; scrolling to the last row loads the next page by itself. The similarity is the cosine similarity of the CLIP vectors, from 1.0 (the same) downwards.

### Duplicates

`smart:duplicates` reads the library once in path order and groups the rows by size, which needs no file access. Only groups of two or more files are checked further. A row that has a stored content hash (blake3 of the whole file, written by indexing) uses it; the other rows of the group are read and hashed with blake3 on background threads, four at a time. Rows that share a size and a hash form a group of exact duplicates. The view lists the members of all groups, one group after the other, and the Hash column shows the group's hash. Near-duplicates (resized or re-encoded copies) are not found by this view; use `sim:<path>` for those.

### Badges

Each file row shows a badge in the Name column: **Indexed** (in the library), **Not indexed** (on disk only) and **Offline** (in the library but missing on disk). The disk check runs in the background for rows that are drawn, once per file a minute, never over the whole view.

## Commands

### open_scope

Shows a scope in a tab.

```json
{"cmd": "open_scope", "scope": "fs:/home/u/Pictures", "target": "new_tab"}
```

| Field | Meaning |
|---|---|
| `scope` | A scope string |
| `target` | `current` (default; replace what the focused tab shows and push its history), `new_tab` (a new tab that takes focus), `background` (a new tab behind the focused one) |
| `title` | Optional tab title; the tab goes back to the title its scope gives once it navigates |
| `recursive` | Optional; scan the folder recursively |

A folder or archive that does not exist is a `not_found` error and changes nothing. If a tool tab is in front, `current` brings the browser forward first. Searches, similar-to, tags and categories never open tabs on their own: a click replaces the tab, and Ctrl+click opens a new one.

### navigate

Moves the focused tab. `op`: `back`, `forward`, `up` (the parent folder, or the library for any scope that is not a place on disk), `home`, `refresh` (read the scope again, keeping the history), `load_more` (the next page of a search) and `cancel` (stop the load in flight). An operation that has nowhere to go is an `invalid` error. Alt+Left, Alt+Right and Alt+Up (and the mouse back and forward buttons) send the same commands.

```json
{"cmd": "navigate", "op": "back"}
```

### open_rows

Shows rows that already exist in memory. Internal: the rows are not part of the JSON form, so a JSON `open_rows` opens an empty tab. The command log shows a `row_count`.

```json
{"cmd": "open_rows", "title": "Category: landscape", "background": false}
```

### open_tool

Opens or focuses a tool tab: `image_edit`, `refinements`, `logs`.

```json
{"cmd": "open_tool", "tool": "image_edit", "path": "/home/u/Pictures/a.png", "target": "new_tab"}
```

### run

Runs an action on files. `on` says which files: `"selection"` (selected rows of the focused tab), `"focus"` (the previewed file), `"view"` (every file row of the focused tab) or `{"paths": [...]}`. An empty set is an `invalid` error: an action never falls back to "all rows".

```json
{"cmd": "run", "action": "generate_clip", "on": "selection"}
{"cmd": "run", "action": "copy_path", "on": {"paths": ["/a.jpg", "/b.jpg"]}}
{"cmd": "run", "action": "tag", "on": "selection"}
{"cmd": "run", "action": "remove_from_library", "on": "view"}
```

The actions are the entries of the action registry (`src/command/actions.rs`), which also builds the selection action bar and the context menu of the grid and the table; Devtools > Browser actions lists each with whether it can run on the current selection, the reason when it cannot, and a Run button. An action that cannot run on the files it is given is an `invalid` error that names the reason. Actions that need more than a file list (tags, a collection, a folder, a confirmation) do not guess: they open the same popover or confirmation the action bar opens, over the focused tab, and change nothing until the user answers.

| Action | Does |
|---|---|
| `open` | Opens each file with the system default (at most 20); a single folder or archive opens in the tab |
| `reveal` | Opens the folder of each file (at most 20) |
| `copy_path` | Copies the paths to the clipboard, one per line |
| `add_to_library` | Saves the files to the library and the active collection; a toast reports the counts |
| `remove_from_library` | Asks first with the exact count (always, also from a command), then deletes the rows and CLIP embeddings; files on disk and collection memberships stay |
| `describe` | Describes the images with the vision model; files that already have a description are skipped unless the overwrite setting is on |
| `regenerate_description` | Describes the images again and replaces the description |
| `generate_clip` | Generates CLIP embeddings; a toast reports the result |
| `regenerate_clip` | Generates CLIP embeddings again |
| `find_similar` | Opens `sim:<path>` for the first file |
| `tag`, `category` | Open the batch popover for those files; applying it adds and removes tags, or sets or clears the category, and toasts the counts |
| `add_to_collection`, `remove_from_collection` | Open the collection picker (with Create new when adding) |
| `open_in_new_tab` | Shows the rows in a new tab |
| `copy_to`, `move_to` | Open a folder picker; `move_to` asks first. Both run as Backup tasks and toast the counts |
| `exclude_folder` | Adds the selected folders to the excluded folders of the settings and the tab's filter |
| `attach_to_assistant` | Same as `attach_to_assistant` |
| `refine` | Same as `refine` |

### show_settings

Opens the Settings tab, or focuses it when it is open. `section` is optional: `general`, `library`, `ai_models`, `automation`, `assistant`, `metadata_tools`, `shortcuts`, `appearance`, `advanced`; without it the tab keeps the section it shows. Every older way in (File > Preferences, View > Appearance, the database items of the File menu, the sidebar Connect button, the palette) sends this command. See `settings.md`.

```json
{"cmd": "show_settings", "section": "appearance"}
```

### wizard

Opens the first-run wizard at `step` (`location`, `folders`, `ai`, `scan`), or at the first step that is still to do. The wizard opens by itself when no library exists. The command line form is `--wizard <step>`.

```json
{"cmd": "wizard", "step": "folders"}
{"cmd": "wizard"}
```

### backup

Backs up or restores the library without a dialog, as Database tasks (Task Center). `op` is `export` (a `.surql` file at `path`; the API keys are left out of the settings record unless `redact_keys` is `false`, thumbnail blobs are included with `include_thumbs`), `restore` (builds a new library in the folder `dir` from the `.surql` file at `path`; `dir` must be missing or empty and the open library is not touched), `import` (runs the `.surql` file at `path` against the open library, no confirmation; records that exist already make it fail) or `snapshot` (copies the library folder into the snapshots folder). Export and import wait for the library to open (at most two minutes), so they can be sent with `--run` at start-up.

```json
{"cmd": "backup", "op": "export", "path": "/home/u/backups/library.surql"}
{"cmd": "backup", "op": "export", "path": "/home/u/backups/full.surql", "redact_keys": false}
{"cmd": "backup", "op": "restore", "path": "/home/u/backups/library.surql", "dir": "/home/u/restored-library"}
{"cmd": "backup", "op": "import", "path": "/home/u/backups/other.surql"}
{"cmd": "backup", "op": "snapshot"}
```

### attach_to_assistant

Attaches files to the assistant chat: opens the right rail on the Ask tab (or the pop-out window when the assistant is popped out) and shows the files as attachment chips in the composer.

```json
{"cmd": "attach_to_assistant", "paths": ["/a.jpg"]}
```

### zc

Sets up and drives the ZeroClaw gateway connection of the assistant (`docs/zeroclaw.md`). `op` picks the operation; the other fields are the ones the op reads. Work that needs the network finishes in the background and reports through a toast and the notes of Devtools > ZeroClaw. A command never carries a token: `configure` reads it from `token_file`, and `pair` and `import` store it in the token file of the application.

| Op | Does | Fields |
|---|---|---|
| `status` | Returns the profile, health, the chat socket, usage totals, frame counts and the tools server as one JSON line | |
| `check` | Probes the gateway now (`/health`, then the authenticated agent check; 6 s) | |
| `connect` | Opens the chat socket for `session` (a new id when left out) | `session`, `name` |
| `disconnect` | Closes the chat socket; a running turn goes on at the gateway | |
| `reconnect` | Drops the socket and connects again at once | |
| `complete` | One model call through `/api/complete` | `prompt`, `hint`, `system`, `temperature` |
| `configure` | Changes the profile; a new host or agent closes the socket and checks again | `host`, `agent`, `name`, `hint`, `backend` (`auto`, `zeroclaw`, `direct`, `local`), `allow_images`, `tools`, `token_file` |
| `pair` | Exchanges a one-time pairing code for a token and stores it | `host`, `code`, `agent` |
| `import` | Takes the selected ZeroClaw machine of a ZC Codex config (default `~/.config/zc-codex/config.json`) | `path` |
| `send` | Sends `text` from the assistant as if typed | `text` |
| `stop` | Stops the running turn | |
| `open` | Shows the assistant in the right rail | |
| `ping_tools` | Asks the agent to call `aimedia__util_ping` and report the result | |
| `forget_token` | Deletes the stored token | |

```json
{"cmd": "zc", "op": "status"}
{"cmd": "zc", "op": "configure", "host": "100.96.121.37", "agent": "media", "token_file": "/home/u/.config/zc/token", "allow_images": false}
{"cmd": "zc", "op": "pair", "host": "wss://claw.example.app", "code": "123456", "agent": "media"}
{"cmd": "zc", "op": "import", "path": "/home/u/.config/zc-codex/config.json"}
{"cmd": "zc", "op": "check"}
{"cmd": "zc", "op": "connect", "session": "demo"}
{"cmd": "zc", "op": "complete", "prompt": "Suggest three tags for a sunset", "hint": "quick", "temperature": 0.2}
{"cmd": "zc", "op": "send", "text": "How many pictures are in the library?"}
{"cmd": "zc", "op": "ping_tools"}
```

Start-up flag: `--zc-status [--zc-url URL] [--zc-agent ALIAS] [--zc-token-file PATH]` probes a gateway, prints one JSON line and exits without opening the window or the database (exit 0 online, 2 not set up, 3 token refused, 4 unreachable, 1 anything else). `ZC_URL`, `ZC_AGENT` and `ZC_TOKEN_FILE` fill in missing flags.

### refine

Opens the AI Refinements tab limited to the files.

```json
{"cmd": "refine", "paths": ["/a.jpg", "/b.jpg"]}
```

### undo and redo

`undo` reverts the newest batch of the metadata journal (`src/database/journal.rs`); `redo` re-applies the batch that was undone last. Every metadata write the application makes is one batch: an Inspector edit, a batch tag or category change, a tag or category rename or delete, adding files to a collection or removing them, accepting a Refine or MCP proposal. A row that was edited again since is left alone and counted as skipped; the toast says how many changes came back, in how many files, and how many were skipped. Ctrl+Z and Ctrl+Shift+Z send the same commands while no text field has focus. A new change clears what could be redone, and the journal keeps the newest 5000 entries (whole batches). Devtools > Journal lists the entries and batches and has Undo, Redo and Clear buttons.

```json
{"cmd": "undo"}
{"cmd": "redo"}
```

### workspace

Saves, reloads or deletes the workspace file (`<config>/smart-media/workspace.json`, see `settings.md`). `save` writes the open tabs, the dock layout and the window now. `restore` reads the file and replaces the open tabs, floating windows included, with what it holds: every tab reads its scope when it is first shown, and the window and the sidebar stay as they are. `forget` deletes the file. A file written by a newer version is neither overwritten nor deleted. Devtools > Workspace has the same three buttons.

```json
{"cmd": "workspace", "op": "save"}
{"cmd": "workspace", "op": "restore"}
{"cmd": "workspace", "op": "forget"}
```

### dump_state

Writes a JSON snapshot of the application to `path` (the folder is created), through a temporary file and a rename. With `when_idle` the command only queues the dump: it is written once at least eight frames have run and no tab that is in front of its leaf has been reading its scope or waiting for the library for three frames in a row (a tab behind another is not drawn, so it cannot move on), or after 30 seconds whatever is going on; `--dump-state <path>` is this form, and the application keeps running afterwards. Scripts and smoke checks read the file instead of the screen.

```json
{"cmd": "dump_state", "path": "/tmp/ame-state.json"}
{"cmd": "dump_state", "path": "/tmp/ame-state.json", "when_idle": true}
```

| Part | Holds |
|---|---|
| `version`, `build`, `frames`, `database_ready` | layout version, program name, version, profile, OS, architecture and features, frames run, whether the library is ready |
| `tabs` | one entry per browser tab: `id`, `title`, `scope`, `active` (the focused browser), `rows` (rows held), `filtered` (rows the filters let through, as of the last frame the tab was drawn), `selected`, `view` and `tile`, `sort`, `query` (mode and text), `filter` (the filter spec as compact JSON) and `filters_active`, `loading`, `pending` (`lazy`, `waiting`, `offline` or `search` for a restored tab that has not read its scope), `error`, `rows_only` |
| `active_tab`, `other_tabs` | scope of the focused browser; titles of the dock tabs that are not browsers |
| `tasks` | the newest 60 tasks: kind, title, state, progress, error |
| `engines` | every engine chip: name, state, model, detail, progress, error |
| `settings` | a summary: the switches, model names, counts, the endpoint without credentials or query, and which providers have a key |

No key, token or password is ever written. The settings part is built from a fixed list of safe fields, and every free text in the dump (titles, errors, details, queries, paths) is masked for the stored keys and for anything that looks like a key, a bearer token, a `key=` or `password=` value or user info in a URL.

### toast

Shows a toast. `kind`: `info`, `success`, `warning`, `error`.

```json
{"cmd": "toast", "kind": "success", "text": "Done"}
```

### task_update

Reports on a task in the task registry. `event` is one of the following.

| Event | JSON |
|---|---|
| Start a task (the `id` is ignored; the result text names the new id) | `{"begin": {"kind": "debug", "title": "Fake", "total": 100}}` |
| Progress | `{"progress": {"done": 5, "total": 100}}` |
| Detail text | `{"detail": {"text": "halfway"}}` |
| Finish | `"complete"` |
| Fail | `{"fail": {"error": "disk full"}}` |
| Ask the task to stop | `"cancel"` |

Task kinds: `scan`, `thumbnails`, `describe`, `clip`, `refine`, `image_edit`, `database`, `legacy_import`, `thumb_migration`, `backup`, `model_load`, `assistant`, `debug`.

```json
{"cmd": "task_update", "id": 7, "event": {"progress": {"done": 5, "total": 100}}}
```

Tasks started by `begin` count as user-initiated (they toast when they complete or fail) and have no worker, so `cancel` ends them at once.

### clear_selection

Clears the selection of the focused tab.

```json
{"cmd": "clear_selection"}
```

### query

Sets the query bar of the focused tab. `mode` is `name` or `text` (live filters: the text goes to the name condition or the search-text condition of the tab's filter, so rows are only hidden, never removed), `semantic` (search the library by meaning, as Enter does) or `similar` (files like the one `text` names, else the focused file). `text` can be left out. Without `run` the bar is set and its live filter applied: `name` and `text` filter at once, `semantic` and `similar` clear the live filter and wait for Enter. With `"run": true` Enter is pressed: `semantic` opens `q:sem:"<text>"` and `similar` opens `sim:<path>` in the focused tab. An empty semantic text, or `similar` with no text and no focused file, is an `invalid` error. The Query window of Devtools shows the text, what typing runs, what Enter runs and the history of runs.

```json
{"cmd": "query", "mode": "name", "text": "beach"}
{"cmd": "query", "mode": "semantic", "text": "a red car", "run": true}
```

Start-up flag: `--query <mode>:<text>` (repeatable) is the same command with `run`, for example `--query semantic:'a red car'`.

### dialog

Opens the Scan dialog or the Process dialog of the focused tab (Tools > Scan folder, the Scan and Process buttons of the toolbar and the Scan / Process menu of the status bar open the same dialogs). `folder` is optional: it fills the folder field and, for Process, chooses the Folder scope. The dialogs are described in `docs/browser.md`.

```json
{"cmd": "dialog", "dialog": "scan", "folder": "/home/u/Pictures"}
{"cmd": "dialog", "dialog": "process"}
```

### focus_path

Selects the row of a path in the focused tab, which the inspector rail previews and opens when it was hidden; `not_found` when the tab has no such row or the filters hide it.

```json
{"cmd": "focus_path", "path": "/home/u/Pictures/a.jpg"}
```

### open_devtools

Opens a debug window: `theme_panel`, `settings_store`, `wizard`, `db_maintenance`, `legacy_import`, `thumb_storage`, `thumb_cache`, `tasks`, `command_log`, `perf`, `zeroclaw`, `row_store`, `collections`, `query`, `browser_actions`, `video`, `scope`, `filters`, `grid`, `inspector`, `journal`, `workspace`, `refine`, `sidebar`, `actions`, `mcp_server`.

The `scope` window takes a scope string and shows how it parses (kind, title, parent, whether a folder exists), opens it in the focused tab or a new one, steps back and forward, lists the history stack of the focused tab with the scroll row and selection saved in each entry, and shows the timing of its recent loads: rows, batches, time to the first batch and total time.

```json
{"cmd": "open_devtools", "window": "tasks"}
```

### connect_database

Opens the database at the configured folder. The application does this by itself at start-up when the folder exists, as a Database task; when it fails, Needs attention lists it with a Retry button that sends this command. It does nothing when the database is already open.

```json
{"cmd": "connect_database"}
```

### show_task_center

Opens the task list popover of the status bar.

```json
{"cmd": "show_task_center"}
```

### set_filter

Changes the filters of the focused tab. Filters never remove rows: a row that is not listed stays in the tab and comes back when the filter is removed. `filter` is a JSON object that is merged into the tab's current filters (nested objects merge, lists and plain values replace, `null` turns that part off). With `"replace": true` it is merged into the defaults instead (everything listed except files under 10 KiB). Unknown fields and malformed dates or lists are an `invalid` error.

```json
{"cmd": "set_filter", "filter": {"size": {"min": null, "max": 5242880}, "types": {"videos": false}}}
{"cmd": "set_filter", "filter": {"tags": ["sunset"], "clip": "yes"}, "replace": true}
```

| Field | Meaning |
|---|---|
| `types` | `images`, `videos`, `dirs`, `archives`, `other`: `false` hides that kind |
| `size` | `min` and `max` in bytes |
| `modified`, `created` | `after` and `before` as `YYYY-MM-DD` (local days, inclusive); `created` is the date the row was added to the library |
| `excluded_terms` | Lowercase text; a file whose path contains one is not listed |
| `excluded_dirs` | Folders; everything below one is not listed |
| `query` | Text that must appear in the name, type, path, tags, category or description |
| `name` | Text that must appear in the file name |
| `skip_icons` | Hides `.ico` files and small icon-sized images |
| `ext_allow`, `ext_deny` | Extensions without the dot; with `ext_allow` set only those are listed |
| `indexed`, `description`, `clip`, `offline` | `any` (default), `yes` or `no` |
| `categories` | Any of these categories |
| `tags`, `tags_all` | Any of these tags, or all of them when `tags_all` is true |

Folders are listed unless `types.dirs` is false or `query`, `name` or `excluded_dirs` exclude them; the other conditions apply to files only.

### clear_filters

Removes every filter of the focused tab, the default size limit included.

```json
{"cmd": "clear_filters"}
```

### apply_saved_view

Applies a saved filter view (Views in the filter strip) to the focused tab by name, ignoring case. `not_found` when no loaded view has that name.

```json
{"cmd": "apply_saved_view", "name": "Holidays"}
```

### set_view_mode

Switches the focused tab between the thumbnail grid and the details table, and sets how the grid looks. The choice is remembered for new tabs and the next session (`<config>/smart-media/view_prefs.json`). A tab opens in the grid by default.

```json
{"cmd": "set_view_mode", "mode": "grid", "size": 160, "fit": "crop", "captions": true}
{"cmd": "set_view_mode", "mode": "details"}
{"cmd": "set_view_mode", "mode": "grid", "stress": 40}
```

| Field | Meaning |
|---|---|
| `mode` | `grid` or `details` |
| `size` | Optional tile size in points, 64 to 320 (values outside are clamped); the presets are S 96, M 160 and L 256 |
| `fit` | Optional: `crop` fills each square tile, `fit` shows the whole image |
| `captions` | Optional: file names under the tiles |
| `stress` | Optional: rows per second the grid scrolls up and down by itself, for performance checks (0 stops it; Devtools > Grid has the same control) |

The command acts on the focused tab; in a `--run` list the tab opened by an earlier entry takes focus only after its first frame, so the main explorer is the one that changes. Devtools > Grid drives the same settings for the tab it shows.

Keys in the grid: arrows move by tile and row, Page Up and Page Down by a screen, Home and End to the ends, Shift extends, Ctrl+A selects all, Enter opens, Space toggles the preview, Esc clears, plus and minus resize the tiles, Ctrl+wheel zooms, F9 switches Grid and Details, Ctrl+1 and Ctrl+2 pick one. Dragging over empty space selects with a band (hold Ctrl to add).

### inspector_toggle

Shows or hides the inspector rail. With `open` the rail is set to that state, without it the state flips. The rail follows the selection of the focused tab; `focus_path` selects a file and shows the rail, and Devtools > Inspector shows its state and can drive it without the mouse. The rail width, tabs and fit mode are kept in `<config dir>/smart-media/inspector_prefs.json`.

```json
{"cmd": "inspector_toggle", "open": true}
{"cmd": "inspector_toggle"}
```

## Navigation

Every browser tab keeps a history: the scopes it showed, with the first row on screen and the selected row when the user left each. Back and Forward bring the scope back with that view; the rows are read again, and the position returns once they have arrived. A scope that cannot be opened any more (a folder that was deleted) leaves the tab where it is and the reason shows in the footer.

A tab is named after its scope (a folder's name, `Tag: sunset`, `Search: red car`); a title given with a command lasts until the tab navigates. The address box of the toolbar takes a scope string or a plain path.

Loads run as Database tasks in the status bar and can be cancelled there; a failed load keeps the rows already read, shows the reason in the footer with a Retry, and the task list offers the same retry.

### open_palette

Opens the command palette (Ctrl+K), optionally with `query` typed in. The palette matches commands, places, collections, tags, categories, scans and settings sections; a query also offers `Search library for "<query>"`, which sends `open_scope` with a `q:sem:` scope. Devtools > Actions lists every palette command with its shortcut, whether it can run now and why not.

```json
{"cmd": "open_palette", "query": "no clip"}
```
### mcp

Controls the MCP tools server (`docs/mcp.md`) and the metadata proposals it collects. `op` is one of the following; work that needs the database or the network finishes in the background and reports through a toast and the notes of Devtools > MCP server.

| Op | Does |
|---|---|
| `status` | Returns whether the server runs, where, and the counters |
| `start` / `stop` | Starts or stops the listener |
| `rotate_token` | Writes a new bearer token; clients need the new one |
| `proposals` | Lists the pending proposals |
| `accept` | Accepts the proposal `id` and writes its changes to the library |
| `reject` | Rejects the proposal `id` |
| `call` | Runs the tool `tool` with `args` in this process (as the Devtools tester does) |

```json
{"cmd": "mcp", "op": "status"}
{"cmd": "mcp", "op": "start"}
{"cmd": "mcp", "op": "proposals"}
{"cmd": "mcp", "op": "accept", "id": "p1a2b3c4d"}
{"cmd": "mcp", "op": "call", "tool": "library_stats", "args": {}}
```

Start-up flags: `--mcp-port N` serves on port N for this run (0 picks a free port), `--mcp-config-dir <dir>` keeps the token and the settings file in `<dir>`, and `SMART_MEDIA_MCP_PORT` does the same as the port flag.

## Results

Each command ends with a result recorded in the log: a short text, or an error with one of these kinds.

| Kind | Meaning |
|---|---|
| `invalid` | The request is malformed or acts on nothing |
| `unsupported` | The application cannot do this yet |
| `not_found` | The thing named does not exist |
| `failed` | The attempt was made and failed |

## The old request queues

The request queues that predate the bus are gone; every caller sends a command.

| Old | Now |
|---|---|
| `OPEN_TAB_REQUESTS` with `FilterRequest` | `open_scope`, `open_tool` or `open_rows` |
| `OPEN_SETTINGS_REQUEST` | `show_settings` |
| `assistant::request_attach_to_chat(paths)` | queues `attach_to_assistant` |
| `refine::request_refine_for_paths(paths)` | queues `refine` |
| `file_table::ACTIVE_GROUP_NAME` | `command::context::{active_collection, set_active_collection}` hold the collection scanned and selected files are added to |
| `FilterRequest::OpenDatabaseAll` | `open_scope` with `lib:all` |

`assistant::ATTACH_REQUESTS` and `refine::REFINE_SELECTION_REQUESTS` remain as the mailboxes the dispatcher fills for the panels, which read them while they are shown.
