# Commands

Every request to the application is an `AppCommand` (`src/command/`). Menus, buttons, the command runner in Devtools, start-up flags and the assistant or MCP integration all send the same commands. A command is a JSON object with a `cmd` field; the other fields depend on the command.

The JSON form is the stable surface. Names are `snake_case`. Fields marked optional can be left out. Unknown command names, unknown enum values and malformed scopes are rejected with an error, never ignored.

## Sending commands

| From | How |
|---|---|
| Rust | `crate::command::send(AppCommand::...)` from any thread; the next frame runs it |
| Text | `crate::command::send_text(text)`: a JSON object, a JSON array of objects, or a scope string (opens the scope in a new tab) |
| Devtools | View > Devtools > Command log: the runner box takes the same text; Run again re-sends a logged command |
| Command line | `--run '<json or scope>'` (repeatable) and `--devtools tasks,command-log,perf` queue commands for the first frame |

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

Scopes the application can open today: `fs:`, archives, `lib:all`, `tag:`, `cat:` and `smart:failed-thumbs`. The others return an `unsupported` error until their loaders exist.

## Commands

### open_scope

Shows a scope in a tab.

```json
{"cmd": "open_scope", "scope": "fs:/home/u/Pictures", "target": "new_tab"}
```

| Field | Meaning |
|---|---|
| `scope` | A scope string |
| `target` | `current` (replace the focused tab; folders, archives and the library only), `new_tab` (default), `background` (new tab behind the focused one) |
| `title` | Optional tab title |
| `recursive` | Optional; scan the folder recursively |

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
```

| Action | Does |
|---|---|
| `open` | Opens each file with the system default (at most 20) |
| `reveal` | Opens the folder of each file (at most 20) |
| `copy_path` | Copies the paths to the clipboard, one per line |
| `generate_clip` | Generates CLIP embeddings; a toast reports the result |
| `describe` | Reserved: returns `unsupported` |
| `attach_to_assistant` | Same as `attach_to_assistant` |
| `refine` | Same as `refine` |

### show_settings

Opens settings. `section` is optional: `general`, `library`, `ai_models`, `automation`, `assistant`, `metadata_tools`, `shortcuts`, `appearance`, `advanced`. Today `appearance` opens the theme window, `advanced` the egui settings window and everything else the preferences window.

```json
{"cmd": "show_settings", "section": "appearance"}
```

### attach_to_assistant

Attaches files to the assistant chat and shows the assistant window.

```json
{"cmd": "attach_to_assistant", "paths": ["/a.jpg"]}
```

### refine

Opens the AI Refinements tab limited to the files.

```json
{"cmd": "refine", "paths": ["/a.jpg", "/b.jpg"]}
```

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

### focus_path

Selects and previews the row of a path in the focused tab; `not_found` when the tab has no such row.

```json
{"cmd": "focus_path", "path": "/home/u/Pictures/a.jpg"}
```

### open_devtools

Opens a debug window: `theme_panel`, `settings_store`, `db_maintenance`, `legacy_import`, `thumb_storage`, `thumb_cache`, `tasks`, `command_log`, `perf`, `row_store`.

```json
{"cmd": "open_devtools", "window": "tasks"}
```

### show_task_center

Opens the task list popover of the status bar.

```json
{"cmd": "show_task_center"}
```

## Results

Each command ends with a result recorded in the log: a short text, or an error with one of these kinds.

| Kind | Meaning |
|---|---|
| `invalid` | The request is malformed or acts on nothing |
| `unsupported` | The application cannot do this yet |
| `not_found` | The thing named does not exist |
| `failed` | The attempt was made and failed |

## The old request queues

The statics that predate the bus still exist as thin wrappers, so existing call sites keep working.

| Old | Now |
|---|---|
| `app::OPEN_TAB_REQUESTS.lock().unwrap().push(FilterRequest)` | queues `open_scope`, `open_tool` or `open_rows` |
| `app::OPEN_SETTINGS_REQUEST.store(true, _)` | queues `show_settings` |
| `assistant::request_attach_to_chat(paths)` | queues `attach_to_assistant` |
| `refine::request_refine_for_paths(paths)` | queues `refine` |
| `file_table::ACTIVE_GROUP_NAME` | `command::context::active_collection()` is the shared accessor; the explorer still writes its own static until it is switched over |

`assistant::ATTACH_REQUESTS` and `refine::REFINE_SELECTION_REQUESTS` remain as the mailboxes the dispatcher fills for the panels, which read them while they are shown.
