# Settings, first-run wizard and backups

Code: `src/ui/settings/` (the page), `src/ui/wizard/` (the first-run wizard and its checklist card), `src/settings/` (the store, `ui_prefs.rs`, `location.rs`, the banners), `src/ui/devtools/wizard.rs`.

## The Settings tab

A dock tab (`File > Preferences`, Ctrl+, , the palette, `{"cmd":"show_settings","section":"..."}`). Left: the section list; at the top: a search box, the dirty marker, **Revert** and **Apply**. Below 640 px the section list becomes a row of buttons above the content. The search box matches the names of the settings and lists the hits; **Show** jumps to the section and outlines the setting.

Opening the page changes nothing. Edits stay in a draft until Apply: the library settings are written as **one patch of the changed fields** (`SettingsStore::update`, patch name `settings:apply`), so a stale copy never overwrites what another part changed meanwhile. The tab title reads `Settings (unsaved)` while edits are pending, and the draft survives closing the tab. Actions that act at once (watched folders, log level, backups, resets, model buttons) are marked as such.

| Section | Contents |
|---|---|
| General | restore tabs and the last scope on launch, default view, thumbnail size and fit, captions and badges, confirmations |
| Library | library folder (check, use, copy to a new folder), backup and restore, watched folders, scan defaults |
| AI models | vision folder with a check, device, CLIP/SigLIP model, reranker and Jina options, image-edit pointer, load and reload buttons, free memory |
| Automation | auto indexing, auto CLIP, overwrite descriptions, prompt template with reset |
| Assistant and providers | the ZeroClaw card (`zc::connection_ui`), the direct providers with masked keys, endpoint, model, organization and a connection test |
| Metadata tools | rename or merge a category or tag, delete a tag, limit tags per item; each shows the counts first and asks |
| Shortcuts | every registered action and command with its binding, from the palette registry |
| Appearance | the theme picker and the egui options (Save keeps them across restarts) |
| Advanced | log level, performance HUD, MCP server window, raw settings JSON, resets |

### Where each setting lives

General values that are not part of the settings record live in `<config>/smart-media/ui_prefs.json`; the default view and thumbnail size are the app-wide view preferences in `view_prefs.json` that new tabs start from. Everything else is the settings record in the library.

| Old place | New place |
|---|---|
| Preferences > Database: folder, Setup | Library > Library location |
| Preferences > Database: rename category, rename tag, delete tag, prune | Metadata tools |
| Preferences > AI: vision folder, Use This | AI models > Vision |
| Preferences > AI: provider, keys, base URL, model, organization | Assistant and providers |
| Preferences > AI: reranker, Jina options | AI models > Reranker |
| Preferences > AI: auto indexing, auto CLIP, overwrite descriptions, prompt template | Automation |
| Preferences > AI: blend image and text, overwrite embeddings, CLIP model | AI models > CLIP and SigLIP |
| Preferences > AI: skip icons, auto save, excluded directories | Library > Scan defaults |
| Preferences > AI: MCP tools server | Advanced > MCP tools server |
| Preferences > Filters: date range | Library > Scan defaults |
| Reload CLIP / reranker, generate missing embeddings | AI models > Load models now |
| Filter menu (global): `db_min_size_bytes`, `db_max_size_bytes` | Library > Scan defaults |
| Quick Access > Performance tweaks: `scan_found_batch_max` | Library > Scan defaults (and Tools > Scan settings) |
| `db_excluded_exts` (no UI before) | Library > Scan defaults |
| `ext_enabled`, `group_by_category` (no UI, unused) | Advanced > raw settings JSON |
| `recent_paths`, `recent_models`, `last_used_model` | written by the explorer and assistant; Assistant > Recent models can clear the models |
| Egui Settings window | Appearance > Egui options |
| View > Appearance window | Appearance > Theme |
| File > Database: export, import, choose folder, backup and restore | Library > Backup and restore, Library location |

## ui_prefs.json

`<config>/smart-media/ui_prefs.json`, versioned (`"version": 1`):

```json
{
  "version": 1,
  "startup": { "restore_tabs": false, "restore_last_scope": false, "last_scope": "lib:all", "tabs": ["lib:all"] },
  "confirmations": { "delete_from_library": true, "move_files": true, "import_backup": true, "restore_snapshot": true },
  "library": { "watched": ["/home/u/Pictures"] },
  "checklist": { "dismissed": false, "location": "done", "folders": "done", "ai": "skipped", "scan": "pending", "ai_mode": "off" }
}
```

A missing, unreadable or newer-version file gives the defaults without a panic; a newer file is never overwritten. Missing fields take defaults and over-long lists are cut. Writes go through a temporary file and a rename, and only when something changed.

The two switches of `startup` decide what a launch brings back from the workspace file (next section): **Restore tabs** the whole workspace, **Restore the last scope** only the focused scope, in the main tab. `startup.tabs` and `startup.last_scope` are what older versions recorded; they are read only while no workspace file exists.

## workspace.json

`<config>/smart-media/workspace.json`, versioned (`"version": 1`). It holds what is open and where: the window, the dock layout and the state of every browser tab. The sidebar and right rail are not in it, they stay in `ui_layout.json` (the file names it as `layout_file`).

```json
{
  "version": 1,
  "layout_file": "ui_layout.json",
  "window": { "size": [1200.0, 800.0], "position": [40.0, 30.0], "maximized": false },
  "dock": {
    "node": "split", "dir": "horizontal", "fraction": 0.4,
    "first": { "node": "leaf", "tabs": [ { "kind": "browser", "id": 0 }, { "kind": "browser", "id": 1 } ], "active": 1 },
    "second": { "node": "leaf", "tabs": [ { "kind": "logs" } ], "active": 0 }
  },
  "focused": { "kind": "browser", "id": 1 },
  "browsers": [
    {
      "id": 0, "scope": "fs:/home/u/Pictures",
      "view": { "mode": "details", "tile": 200.0, "fit": "fit", "captions": false, "badges": true },
      "sort": { "keys": [ { "column": 2, "descending": true } ] },
      "query": { "mode": "text", "text": "beach" },
      "filter": { "tags": [ "sun" ] },
      "selection": [ "/home/u/Pictures/a.jpg" ]
    },
    { "id": 1, "scope": "lib:all", "view": { "mode": "grid", "tile": 160.0, "fit": "crop", "captions": true, "badges": true }, "query": { "mode": "name", "text": "" } }
  ]
}
```

| Part | Kept |
|---|---|
| `window` | inner size, outer position (where the system tells it) and whether it was maximized; a maximized window keeps the size and place it had before |
| `dock` | the main surface: splits with their direction and share, the tabs of each leaf in order, the tab in front; tab kinds are `browser`, `refine`, `image_edit`, `settings`, `devtools` and `logs` |
| `floating` | the windows tabs were dragged out into (at most 8): their tabs as a dock of their own, their position inside the application window and their size |
| `focused` | the tab that had the focus |
| `browsers` | per browser tab: scope, view mode, tile size, fit, captions and badges, sort, query mode and text, the filter (only when it differs from the defaults of the settings, in the form of `set_filter`) and the selected paths (only when 50 or fewer). Tabs that show rows held in memory (a selection opened in a new tab) have no scope and are left out. The `id` only names a tab inside the file; the main browser is 0 |

**When it is written.** About a second after the last change (a change that is undone before that is not written), when the application closes, and by `{"cmd": "workspace", "op": "save"}`. A write goes through a temporary file and a rename, and only when something changed.

**Safe reading.** A missing, unreadable or newer-version file gives the defaults without a panic, and a newer file is never overwritten or deleted; an unreadable one is kept next to the file as `workspace.json.bad` before the next write replaces it. Whatever cannot be used inside a readable file is dropped (a tab whose scope does not read, an unknown tab kind, a filter with unknown fields, a tab that no leaf shows, a leaf without tabs) and the rest is kept. Lists are cut (64 browser tabs, 50 selected paths, 512 characters of query), shares of a split are brought into 5% to 95%, splits nested deeper than 16 levels become one leaf, and the window is held to at least 700 by 400, to the monitor once it is known and to a place where 80 px of its top edge can be reached (one monitor width or height of slack lets a window on a neighbouring monitor stay there).

**At launch** (the switches above; `--run` or `--scope` on the command line wins and nothing is restored). The window opens with the saved size and place (eframe's own memory of the window is switched off while Restore tabs is on, so the file is the only source; with it off eframe keeps remembering the window as before). The dock is rebuilt, the main browser is the one saved as 0 and the others get new ids. Nothing is read at that point: a tab reads its scope when it is first drawn, so tabs behind the front tab of their leaf cost nothing until they are shown. A scope that needs the library waits for it (a placeholder says so, at most 20 s); a folder that is gone, a library that does not connect, or a search (searches are never run by themselves) shows a placeholder with the reason and a **Retry** (**Run search**) button instead of rows, and never holds the start-up back. The selection is made once its rows are in.

**Devtools > Workspace** shows the file, the last write, the writes of this run, whether the live state equals the file, what the launch restored, the live state and the file as JSON, and has Save now, Reload (restores the file into the running application) and Forget (deletes it). `{"cmd": "workspace", "op": "save|restore|forget"}` does the same without the mouse, and `{"cmd": "dump_state", ...}` and `--dump-state <path>` write a snapshot of the whole application for smoke checks (see `commands.md`).

## The first-run wizard

A blocking dialog that opens when the configured library folder does not exist (`{"cmd":"wizard"}` and `--wizard <step>` open it at will). Every step can be skipped; the result is kept as a checklist in `ui_prefs.json`.

1. **Library location.** Offers a library found next to the program (`db/path.txt`, `db/ai_search` in the working directory or the program folder), a new library in `<data dir>/smart-media/db`, or another folder. A missing or empty folder gets a new library, a folder that holds a library is opened as it is, and **a folder with other files is refused**: nothing is ever overwritten. The choice is written to `<config>/smart-media/db_location.txt` (and to `db/path.txt` when that file exists), so it no longer depends on the working directory; `db/path.txt` still wins when present.
2. **Add folders.** Pictures and Videos are ticked when the system has them (no panic when it has not); more can be added. They become the watched folders.
3. **AI features.** Off, local models (a vision folder with a check and a Detect button) or ZeroClaw (`zc::onboarding_ui`).
4. **First scan.** Scans every watched folder (one tab each) and can switch on saving scanned files to the library.

The checklist card (`wizard::checklist_card`) shows on an empty Library listing until the steps are finished or the card is hidden; Settings > Advanced brings it back.

## Backup and restore

Settings > Library > Backup and restore. All actions run as Database tasks (Task Center, progress, errors).

| Action | What it does |
|---|---|
| Export library | a `.surql` file of the open library through the atomic exporter; **Redact API keys** (on by default) writes every table but the settings and then the settings records without their credential fields, so no key is ever on disk |
| Restore from a .surql file | builds a **new library folder** from the file (the folder must be missing or empty, nothing is overwritten and the open library is not touched; works with no library open) and offers to use it |
| Import into this library | runs a file against the open library after a confirmation (switch off under General > Confirmations); records that exist already make it fail, so it suits merging, not restoring |
| Snapshot now | copies the library folder into the snapshots folder; a snapshot contains the keys |
| Snapshot Restore | staged while the library is open (applied on the next start, the old folder is moved aside), done at once when it is not |
| Copy the library to a new folder | copies the library files into a new folder inside the chosen one (never into an existing folder) and offers to use the copy |

Choosing a library folder while one is open takes effect on the next start. Verify, delete and tuning stay in Devtools > Database maintenance. Every action has a `backup` command for scripts (`docs/commands.md`).

A new library creates the `Default` group record when it opens, so a `.surql` export cannot be imported into a library that has been open already; restore into a new folder instead.

## Database error banner

A library that does not open shows a banner under the title bar with **Reconnect**, **Change location** (Settings > Library) and **Open logs**; the Needs attention entry with its Retry stays.

## Devtools: Wizard

`View > Devtools > Wizard` (`{"cmd":"open_devtools","window":"wizard"}`, `--devtools wizard`): the wizard state, a button that forces each step, close, **Simulate first run** (resets the checklist and opens step 1), the checklist with a switch per step, and the live and on-disk `ui_prefs.json`. Devtools > Settings store links to every Settings section.
