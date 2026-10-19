# AiMediaExplorer: UI audit and information architecture (Study 1)

Repo `/home/shadowbroker/Documents/Rust/AiMediaExplorer`, master @ `1d1a1db`, egui/eframe 0.37. Read-only audit; nothing modified, nothing built.
Paths are repo-relative unless absolute. Phase 2 (theme/icons) and Phase 4 (zeroclaw) are treated as given; no styling/icon proposals here.

Evidence tags: **[V]** read directly in this repo (file:line). **[F]** verified in vendored fork source under `~/.cargo/git/checkouts`. **[E]** estimate/arithmetic. **[U]** likely from reading, not observed at runtime.
Severity: **S1** data loss / crash. **S2** wrong or misleading behavior. **S3** usability / performance. **S4** polish / dead code.

---

## 0. Executive summary

- UI-related code is 12,218 of 31,431 lines (39%): `src/main.rs`, `src/app.rs`, `src/receive.rs`, `src/ui/**` [V].
- **Architecture, in one sentence:** every dock tab is a full `FileExplorer` (102 fields, 10 channel pairs, own toolbar + left pane + right pane), cross-module communication is five global statics, and settings live in at least six independent copies [V].
- **The UI is organised by implementation (menus per widget, a "Database Mode" toggle, a tab per query), not by user task.** Same action lives in 2-4 places; filters/selection/groups/AI actions are scattered across 6 popups [V].
- **Data-loss class defects exist today** (all [V]):
  - "Delete N From DB" deletes every non-dir row in the table, not the selection (`selection_menu.rs:101-131`).
  - Backup "Copy/Move N items" acts on all rows (`selection_menu.rs:39-61`, `backup.rs:3-9`).
  - Tabs opened after startup hold `UiSettings::default()` and call `save_settings(whole record)`, overwriting the persisted record including API keys (`table/mod.rs:93`, `filter_menu.rs:81..337`, `settings.rs:307`).
  - `FileExplorer::new` panics on machines without a Pictures dir (`file_table/mod.rs:328-333`).
- **Performance structure:** per-frame O(rows) loops in the bottom bar, 1 message/frame on the shared thumbnail channel (O(N^2) row streaming), thumbnails stored as base64 inside `SELECT *` rows, sync image decode inside `tokio::spawn`, decode spawned every frame on cache miss [V].
- **Linux is a second-class target today** (user's daily OS): no video thumbnails, CPU/RAM/VRAM meters are stubs, `./db` is CWD-relative, Windows default model path [V].
- **Recommendation:** Browser-centric shell (Sidebar | Browser(Grid/List) | Inspector/Assistant rail | Task bar), `Scope` instead of modes/tabs-per-query, one selection model, one command bus, one settings store, one task registry. Migration is 12 PR-sized steps (section 6); steps M0-M2 are logic-only and low-conflict with Phase 2.

---

## 1. Code map

### 1.1 Size and ownership

| File | Lines | Role |
|---|---:|---|
| `src/main.rs` | 155 | `eframe::App::ui`, drains `OPEN_TAB_REQUESTS`, builds `DockArea`, `main()`; spawns MCP stdio |
| `src/app.rs` | 244 | `SmartMediaApp{tree: DockState<String>, context}`, `SmartMediaContext` (38 pub fields), 3 global statics, fonts |
| `src/receive.rs` | 975 | `SmartMediaContext::receive`: first-run init, toasts, settings/db-ready channels, Preferences window (640 lines), Egui Settings window, DB Setup modal, Assistant viewport |
| `src/ui/tabs.rs` | 73 | `TabViewer for SmartMediaContext` (tab = `String`) |
| `src/ui/navbar.rs` | 446 | Top bar (File/View menus, breadcrumbs, status buttons) + bottom stats bar |
| `src/ui/status.rs` | 415 | Global status registry (`StatusMeta`), hover cards, `GearSpinner` |
| `src/ui/assistant.rs` | 1839 | Chat panel, sessions, OpenRouter model browser, send pipeline (cloud/local/slash/CLIP) |
| `src/ui/refine.rs` | 546 | AI Refinements panel (proposal generation + accept) |
| `src/ui/image_edit.rs` | 526 | Qwen image-edit panel |
| `src/ui/file_table/mod.rs` | 1479 | `FileExplorer` struct (102 fields), `ui()`, pagination, tab-request handling, similarity paging, thumb scheduling |
| `.../table/mod.rs` | 1130 | `FileTableViewer` (`RowViewer` for egui-data-table), columns, cell click logic, context menu + actions |
| `.../receive/*.rs` | 1300 | 5 per-explorer channel drains (scan 826, ai_update 196, thumbnail 98, preload 89, clip 23) |
| `.../menus/*.rs` | 1424 | Toolbar (190), filter (454), scan (271), selection (145), logical groups (331), mode toggle (33) |
| `.../preview_pane.rs` | 539 | Right pane: preview, metadata editor, AI buttons, ffmpeg player (feature-gated) |
| `.../quick_access_pane.rs` | 405 | Left pane: groups, user dirs, recent, drives, WSL, perf tweaks, recent scans |
| `.../explorer/{mod,navigation,backup}.rs`, `load.rs` | 650 | FS populate, history, backup copy/move, DB paging loaders |

### 1.2 Surface inventory (every screen, panel, modal, menu)

| # | Surface | Kind | Entry | State owner | Fed by |
|--:|---|---|---|---|---|
| 1 | App frame | `eframe::App` | `main.rs:13` | `SmartMediaApp` | - |
| 2 | Top bar: File / View menus, breadcrumbs, status buttons | `Panel::top` (24 px) | `navbar.rs:8` | reads **main** `context.file_explorer` (`navbar.rs:211`) | `status::STATUSES` |
| 3 | Bottom bar: scan %, VRAM/CPU/RAM, thumbs, counts, Select + Logical Groups menus | `Panel::bottom` | `navbar.rs:265` | `context.active_explorer()` | `explorer.table`, `viewer.selected` |
| 4 | Dock area (splits, undock, "+", close-all) | `egui_dock` | `main.rs:77-93` | `tree`, `open_tabs`, `filtered_tabs` | `OPEN_TAB_REQUESTS` |
| 5 | "File Explorer" tab | tab | `tabs.rs:24` -> `FileExplorer::ui` (`file_table/mod.rs:473`) | `context.file_explorer` | 10 channels |
| 6 | Dynamic query tabs: `Scan: <path>`, `Recent: <p>`, `Drive: X`, `Category: X`, `Tag: X`, `Similar to X`, `Selected (n) - f`, `Entire Database`, `Failed thumbnails (n)` | tab | `main.rs:20-75` | `context.filtered_tabs[title]` (full `FileExplorer` each) | `OPEN_TAB_REQUESTS` |
| 7 | Explorer toolbar (gear, up/back/fwd/refresh/home, path box, AI checkbox, Apply Filter, filter, eye, search box, Scan, mode button) | `Panel::top` inside tab | `menus/mod.rs:11` | the tab's `FileExplorer` | - |
| 8 | Filter popup (terms, date, size, ext toggles, skip-icons, saved filter groups, restore scan) | `MenuButton` | `menus/filter_menu.rs:6` | explorer + **global settings** | `filter_groups_rx` |
| 9 | Scan popup (5 scans, cancel, save view to DB, auto-save, bulk describe, CLIP, cancel bulk, follow-vision) | `MenuButton` | `menus/scan_menu.rs:5` | explorer | - |
| 10 | Select popup (backup copy/move, exclude dirs, attach to chat, delete from DB, refine) | `MenuButton` (bottom bar) | `menus/selection_menu.rs:7` | explorer | - |
| 11 | Logical Groups popup (add selection/view/unassigned, copy, create, select/open/rename/delete) | `MenuButton` (bottom bar) | `menus/logical_groups.rs:5` | explorer | `logical_groups_rx` |
| 12 | "Database Mode"/"FileSystem Mode" button | `Button` | `menus/table_menu.rs:4` | `viewer.mode` | - |
| 13 | Quick Access pane (hidden by default) | `Panel::left` per tab | `quick_access_pane.rs:8` | explorer | `crate::quick_access()`, `list_drive_infos()` each frame |
| 14 | Preview pane (forced open on every click) | `Panel::right` per tab | `preview_pane.rs:105` | `current_thumb`, `streaming_interim` | `thumbnail_rx`, `ai_update_rx` |
| 15 | Table (11 columns, sort, resize, context menu, hotkeys) | `egui_data_table::Renderer` | `file_table/mod.rs:744` | `table: DataTable<Thumbnail>` | channels |
| 16 | Pagination bars (recursive-scan pages, DB pages) | inline | `file_table/mod.rs:493-668` | explorer | - |
| 17 | Archive password modal | `Window` | `file_table/mod.rs:753-811` | explorer queue | `<PASSWORD_REQUIRED>` sentinel row |
| 18 | Logs tab | `egui_logger` | `tabs.rs:27-38` | global logger | - |
| 19 | AI Refinements tab | panel | `refine.rs:64` | `context.refinements` | `refine_rx`, `REFINE_SELECTION_REQUESTS` |
| 20 | Image Edit tab | panel | `image_edit.rs:91` | `context.image_edit` | `OPEN_TAB_REQUESTS` (title "Image Edit") |
| 21 | User Preferences (Database / AI / Filters) | `Window` (non-modal, scroll) | `receive.rs:142-782` | `settings_draft` | `ui_settings_rx` |
| 22 | Egui Settings | `Window` | `receive.rs:790-827` | `ctx.options` | - |
| 23 | Database Setup (Local / WebSocket "coming soon") | `Window` | `receive.rs:830-961` | `show_db_select_modal` | startup path check |
| 24 | AI Assistant (sessions, model opts, MCP tools menu, Inference menu, input bar) | separate **OS window** (`show_viewport_immediate`) | `receive.rs:964-974` -> `assistant.rs:192` | `context.assistant` (+ borrows main explorer) | `ATTACH_REQUESTS`, 6 channels |
| 25 | OpenRouter models browser | `Window` | `assistant.rs:879-1171` | assistant | `models_rx` |
| 26 | Status indicators + hover cards (DB, VISION, CLIP, RERANK, QWEN-EDIT, ASSIST) | buttons | `status.rs:213-332` | `status::STATUSES` | backend `set_state` calls |
| 27 | Toasts | `egui_toast` | `receive.rs:8-12,53-59` | `context.toasts` | `toast_rx` |
| 28 | Native blocking dialogs (`rfd`) | OS | 22 call sites (`navbar.rs` 11, `selection_menu.rs` 6, `receive.rs` 3, others 2) | - | - |

### 1.3 App state flow

```
main.rs: SmartMediaApp { tree: DockState<String>, context: SmartMediaContext }
  per frame, eframe::App::ui:
   1 context.receive(ctx)                         receive.rs:7
       toasts overlay; first_run init (DB path check -> db::new | DB modal; STYLE json; MCP stdio spawn #2)
       poll: toast_rx, ui_settings_rx (-> copies into main explorer + assistant; sets assistant_window_open=true), db_ready_rx,
             OPEN_SETTINGS_REQUEST, refine_rx
       draw: Preferences window, Egui Settings window, DB Setup modal, Assistant viewport (OS window)
   2 self.navbar(ui)                              navbar.rs:7   (top + bottom bars; reads main/active explorer)
   3 drain OPEN_TAB_REQUESTS                      main.rs:20    (NewTab | OpenDatabaseAll | OpenPath) -> new FileExplorer in filtered_tabs
   4 CentralPanel -> egui_dock::DockArea.show_inside(&mut context)
        TabViewer::ui(tab: String): sets active_tab_title (last-rendered, not focused); match title:
          "File Explorer" -> context.file_explorer.ui    "Logs" -> egui_logger    "AI Refinements" -> refinements.ui
          "Image Edit" -> image_edit.ui                  other -> filtered_tabs[title].ui
        FileExplorer::ui:
          receive x5 (ai_update, clip, preload, thumbnail[1 msg/frame], scan) -> preview_pane -> quick_access_pane
          -> toolbar -> pagination -> status chips -> data-table -> password modal -> requested_tabs -> OPEN_TAB_REQUESTS
```

Only each dock leaf's active tab gets `TabViewer::ui` (standard egui_dock behavior), and `FileExplorer::receive` is called only from `FileExplorer::ui` (`file_table/mod.rs:474`), so background tabs never drain their channels (scan/AI/thumbnail messages queue in unbounded channels until the tab is shown) [V for the call chain; dock behavior not re-checked in the fork]. The `background: bool` on `FilterRequest` is ignored: every new tab takes focus (`main.rs:56-58,69-71`).

### 1.4 State ownership: the settings problem

| Copy of `UiSettings` | Where | Refreshed | Written back |
|---|---|---|---|
| `SETTINGS_CACHE` (global) | `database/settings.rs:280` | after each `save_settings`, at db-ready | `save_settings` |
| `context.ui_settings` | `app.rs:27` | settings load, Preferences Save | Preferences |
| `context.settings_draft` | `app.rs:45` | first open of window; cleared only on Save | Preferences widgets |
| main `file_explorer.viewer.ui_settings` | `table/mod.rs:49` | settings load, Preferences Save | filter/scan/select menus, quick access |
| **every other tab's** `viewer.ui_settings` | same field, constructed via `UiSettings::default()` (`table/mod.rs:93`) | **never** | same menus -> whole-record overwrite |
| `assistant.settings` | `assistant.rs:63` | settings load only; **not** on Preferences Save (`receive.rs:675-677`) | 10 `save_settings` calls in `assistant.rs` |
| AI engine / refine / reranker | `load_settings()` (cache) | - | - |

Consequences [V]: tabs from Quick Access/Scan/Category/Entire Database use default filters (min size, excluded dirs, skip-icons, prompt template, `overwrite_descriptions`) instead of the user's; any filter click in such a tab persists defaults over the saved record (`settings.rs:307-317` upserts the full struct, API keys included; defaults re-read env vars at `settings.rs:243-247`). The assistant keeps stale provider/keys after Preferences Save and overwrites Preferences changes on next `push_recent_model` (`assistant.rs:1413`).

### 1.5 Global statics and channels

| Item | Producers | Consumer |
|---|---|---|
| `app::OPEN_TAB_REQUESTS` (20 refs, 11 files) | menus, table, preview, scan receive, quick access | `main.rs:20` only |
| `app::OPEN_SETTINGS_REQUEST` | navbar | `receive.rs:113` |
| `assistant::ATTACH_REQUESTS` | selection menu, table context menu | `assistant.rs:719` (only while the assistant OS window is shown) |
| `refine::REFINE_SELECTION_REQUESTS` | selection menu, table context menu | `refine.rs:64`, only while the tab is rendered; **nothing opens the tab** |
| `file_table::ACTIVE_GROUP_NAME` | `receive/mod.rs:16` | table context menu |
| `status::STATUSES` (HashMap) | backend (`ai/`, `database/`) | status buttons |
| per `FileExplorer` (10 pairs) | thumbnail, scan, ai_update, clip_embedding, db_preload, logical_groups, filter_groups, video_frame | its own `receive_*` |
| per `SmartMediaContext` | ui_settings, db_ready, toast, refine | `receive.rs` |
| per `AssistantPanel` (6 pairs) | chat, models, new_session, initial_sessions, session_loaded, db_thumb | `assistant.rs:719` |

Backend coupling to UI is small and fixable: `ai/` and `database/` reference only `crate::ui::status::*` (~40 refs) and `app::DEFAULT_JOYCAPTION_PATH` [V].

### 1.6 Per-frame cost of one visible explorer (structure, not measured)

- Bottom bar iterates `ex.table` **twice** per frame, calling `row_passes_filter` (extension parse, allocations, linear `is_image` scan) per row (`navbar.rs:331,375`); `request_repaint_after(300ms)` forces >=3 fps even when idle (`navbar.rs:289`) [V]. At 5,000 rows/page (`db_limit`, `recursive_page_size`) that is ~10k row evaluations per frame [E].
- Quick Access with "Recent Scans" expanded spawns a Tokio task and blocks the UI thread up to 50 ms per frame (`quick_access_pane.rs:305-314`); "Drives" calls `list_drive_infos()` (reads `/proc/mounts` + statvfs, or Win32) every frame (`:127`) [V].
- Preview pane spawns a 1600 px decode on every frame while the cache key is missing (`preview_pane.rs:173-199`); for non-decodable files (video on Linux, archive entries) it never resolves, so it spawns forever while the pane is open [V]. Same pattern in `image_edit.rs:455-479`.
- All thumbnail/preview decode runs synchronously inside `tokio::spawn` (not `spawn_blocking`) at `preview_pane.rs:178`, `receive_scan.rs:533-568`, `file_table/mod.rs:1327,1407`, starving runtime workers shared with SurrealDB and AI [V].
- `thumbnail_rx.try_recv()` is `if let`, i.e. **one message per frame** (`receive_thumbnail.rs:8`); the same channel carries selection events and streamed DB rows (groups, virtual tags/categories: `load.rs:5-86`) with an O(N) `any()` duplicate check per row (`:64`) => O(N^2) and frame-rate-bound [V].
- Thumbnails are base64 PNG strings in `Thumbnail.thumbnail_b64`; every list query is `SELECT *` (`thumbnails.rs:139-148,194-206`); rows are cloned freely (`receive_ai_update.rs:36,55`, page cache `receive_preload.rs:28`). 256 px photo PNGs are ~100-150 KB, so a 5,000-row page is on the order of 0.6-1 GB of strings before decode caches [E].

---

## 2. Workflow-by-workflow usability audit

Legend for "Fix in": section 6 step ids (M0..M11).

### 2.1 First run and onboarding

| Observation | Evidence | Sev | Fix in |
|---|---|---|---|
| Startup **panics** if the OS has no Pictures dir (`UserDirs::picture_dir().unwrap()`) | `file_table/mod.rs:328-333` | S1 | M0 |
| DB location is CWD-relative (`./db/path.txt`, `./db/ai_search`); launching from another directory shows "no database" modal and creates a second DB | `database/mod.rs:29-32,176-184`, `receive.rs:18-36` | S2 | M8 |
| "Database Setup" is non-blocking; app is fully interactive with no DB; WebSocket option is a dead branch with a "coming soon" error | `receive.rs:830-931` | S3 | M8 |
| No model setup guidance: default vision path is a Windows path; AI features fail late with log lines only | `app.rs:15`, `receive.rs:290-294` | S3 | M8 |
| Default `db_min_size_bytes = 10 KiB` silently hides small files; nothing in the UI shows it | `settings.rs:224`, `receive_scan.rs:107` | S3 | M6 |
| Main explorer scans Pictures immediately at construction, before DB is ready (retry path exists) | `file_table/mod.rs:467-469`, `explorer/mod.rs:146-163` | S4 | M4 |
| Each new tab runs `wsl -l -q` / `wmic` synchronously on Windows | `file_table/mod.rs:420-421`, `explorer.rs:362-535` | S3 | M4 |

### 2.2 Browse folders

| Observation | Evidence | Sev |
|---|---|---|
| **Modes**: FileSystem vs Database is a per-tab toggle labelled with the *target* mode ("Database Mode" while in FS). Toggling FS->DB flips the current tab's mode **and** opens an "Entire Database" tab, leaving the old tab in DB mode showing FS rows | `table_menu.rs:5-21` | S2 |
| DB mode "folder" view lists only rows whose `parent_dir` equals the path; no child-folder rows, so you cannot drill down, only retype the path | `thumbnails.rs:194-206`, `load.rs:88-130` | S3 |
| Single click on a folder navigates immediately (no select/rename/context); folders can never be in `viewer.selected` | `receive_thumbnail.rs:47-52`, `table/mod.rs:757-766` | S2 |
| Back/forward/up/home are 5 glyph buttons; no keyboard (no Alt+Left, F5, Ctrl+L, Backspace) | `menus/mod.rs:41-45` | S3 |
| Top breadcrumbs always show/drive the **main** explorer, not the focused tab | `navbar.rs:211,242` | S2 |
| Path box is a plain 400 px text field; in DB mode it doubles as a path-prefix filter and its Enter triggers an AI search using the *other* text box | `menus/mod.rs:50-72` | S2 |
| Side navigation (Quick Access) is hidden behind a gear glyph whose tooltip says "Options" and opens a panel titled "Quick Access" | `menus/mod.rs:28-35`, `quick_access_pane.rs:17` | S3 |
| Default window 1000x750; summed fixed widths of toolbar (path 400 + search 200 + 8 controls) and bottom bar (4 meters x 100 + 7 labels + 2 menus) exceed 1000 px; items clip | `main.rs:139`, `menus/mod.rs`, `navbar.rs:265-443` | S3 [E] |
| Dock "+" buttons do nothing (`on_add` not implemented); all tabs closable incl. the main explorer; layout not persisted (`save()` is empty) | `main.rs:83,98-100`, `tabs.rs`, `tab_viewer.rs:67` [F] | S3 |
| Archive navigation works (zip/tar/7z, password queue) but clicking `.rar` falls into the zip branch (no rar handler) | `file_table/mod.rs:897-933`, `archive.rs:22-160`, `types.rs` ARCHIVE_EXTS | S4 |

### 2.3 Scan

| Observation | Evidence | Sev |
|---|---|---|
| "Scan" is a popup mixing scanning, saving to DB, bulk AI, auto-save toggle and follow-vision. "Recursive Scan" opens a **new tab**; Scan Images/Videos/Archives **replace the current tab's table** | `scan_menu.rs:17-116` | S3 |
| Four near-identical scan blocks (copy-paste) | `scan_menu.rs:25-116` | S4 |
| Recursive tabs ignore saved filters (default settings copy; section 1.4) | `navigation.rs:48-67`, `main.rs:60-72` | S1/S2 |
| Progress is only a 100 px bar in the bottom bar of the *active* explorer, and it resets itself when drawn after completion | `navbar.rs:272-285` | S3 |
| Cancel exists (`Cancel Scan`) but only for the tab's current scan id | `scan_menu.rs:118-120` | S4 |
| Failed thumbnails spawn an extra "Failed thumbnails (n)" tab after the scan; on Linux **every video** fails (video thumbs are Windows-only), so each video scan ends with a tab full of errors | `receive_scan.rs:733-768`, `thumbs.rs:99-111` | S2 |
| RAW: thumb/preview code handles ~13 RAW extensions but `IMAGE_EXTS` contains only `arw`, so other RAW files are never listed | `thumbs.rs:18-36`, `types.rs:8` | S3 |
| UpdateThumb does O(N) `iter_mut().find` over `last_scan_rows` and `recursive_filtered_rows` per thumbnail | `receive_scan.rs:630-645` | S3 |

### 2.4 Filter and search

| Observation | Evidence | Sev |
|---|---|---|
| Filtering is **destructive** in 6 places (`table.retain`); "Clear All" cannot restore rows (code comment admits it); only the text/type toggles are non-destructive | `filter_menu.rs:34,124,184,219`, `selection_menu.rs:128`, `filter_menu.rs:46-50` | S2 |
| Type toggles (Images/Videos/Folders) may not refresh the table: the data-table cache invalidates on the viewer's filter *hash*, which is the text only | `table/mod.rs:195`, `validation.rs:15` [F] | S2 [U] |
| Filter state is split: text/type toggles on the explorer, size/date/skip-icons in **global** settings (affect every tab, persisted on each click) | `filter_menu.rs:81-214` | S2 |
| No active-filter indicator unless a saved preset was applied; "Filtered X of Y" excludes settings-based filters | `file_table/mod.rs:669-722`, `navbar.rs:436` | S3 |
| Search: "Search for files" box = substring over name/type/path/tags/category; semantic search = tick "AI" next to the *path* box, type in the *search* box, then press Enter in the path box or "Apply Filter" | `table/mod.rs:111-147`, `menus/mod.rs:57-72` | S2 |
| Semantic results appear in the same table with an extra "Similarity" column; "Load 50 more" button appears in the toolbar | `menus/mod.rs:176-184` | S4 |
| Date picker defaults to *today* when unset, so a click yields a same-day range | `filter_menu.rs:61-72` | S3 |
| Saved filter groups: apply/delete only, no rename/update; stored in `filter_groups` | `filter_menu.rs:257-365` | S4 |
| Assistant free-text with no attachment runs a CLIP text search and opens a results tab | `assistant.rs:1491-1545` | S4 |

### 2.5 Group, tag, categorize (Logical Groups, Category/Tag views)

| Observation | Evidence | Sev |
|---|---|---|
| "Logical Group" is **single-membership** (`thumbnails.logical_group` = one record, required) but the UI speaks in "Add selection to", "Copy From -> To", "Remove selection from group" | `database/mod.rs:73`, `database/logical_groups.rs:85-116`, UI wording at `menus/logical_groups.rs:44-216` | S2 |
| "Remove from group" sets the field to `NONE` on a non-optional typed field and ignores the result; "Delete group" does the same; likely fails silently | `database/logical_groups.rs:102-149`, `table/mod.rs:1039-1041` | S2 [U] |
| "Active group" is hidden state (colored text on a bottom-bar button); gates auto-save association; default "Default" selected silently | `receive/mod.rs:36-45`, `logical_groups.rs:8-24` | S3 |
| Group UI exists in two places (bottom-bar popup 331 lines, Quick Access list) with different capabilities | `quick_access_pane.rs:28-74` | S3 |
| Category/Tag cells open a **new tab** per click; "Group by Categories/Tags" fills the current table with fake `<DIR>` rows `cat://X`/`tag://X` that open a new tab | `table/mod.rs:339-365,543-557`, `load.rs:5-65` | S3 |
| Counts per tag/category exist in DB (`list_tag_counts`, `list_category_counts`) but nothing calls them | `thumbnails.rs:314-346` | S4 |
| Tag/category maintenance (rename/merge/delete/prune) is buried in Preferences > Database, with free-text inputs and no preview or confirm | `receive.rs:178-258` | S3 |
| Metadata edit (caption/category/tags/description) in the preview pane commits only on **Enter**; blur discards silently; persists only if an AI-engine metadata record already exists; no dirty indicator | `preview_pane.rs:311-384` | S2 |
| Description editor appears only if a description already exists | `preview_pane.rs:311,328` | S3 |

### 2.6 Preview (image, video, RAW, archive)

| Observation | Evidence | Sev |
|---|---|---|
| Preview pane **re-opens on every row click** (and on AI follow); cannot browse with it closed | `receive_thumbnail.rs:55-58,90-93` | S2 |
| Multi-select sends one event per row through the 1-per-frame channel, so preview lags by N frames and ends on an arbitrary row | `table/mod.rs:757-766` | S3 |
| `follow_active_vision` defaults to **true**: during bulk description the preview jumps to whatever is being described, hijacking `current_thumb` (and clones the whole table per hop) | `file_table/mod.rs:371`, `receive_ai_update.rs:33-41` | S2 |
| Image preview is `available/1.3` bounded, no zoom/pan/fit/actual-size, no next/prev, no full-screen | `file_table/mod.rs:1444-1460` | S3 |
| Video playback exists only with cargo feature `ffmpeg` (off by default); the duration slider never renders (`video_duration_secs` is initialised to `None` and never set, `file_table/mod.rs:297,367`); no seek | `preview_pane.rs:200-257`, `Cargo.toml` features | S3 |
| Archive entries preview via per-click extract; password modal is per-explorer | `table/mod.rs:484-523` | S4 |
| Open file / open folder is a heading-sized button with 3 gestures documented in a tooltip | `preview_pane.rs:116-129` | S3 |

### 2.7 Image edit

| Observation | Evidence | Sev |
|---|---|---|
| Second "Edit Image" while the tab exists creates a stub tab "Image Edit (2)" that renders only its title (`uniquify_title` + string-matched tab body); the real editor tab is not focused | `main.rs:24-35,103-116`, `tabs.rs:24-52` | S2 |
| **"Save Result As..." can never appear**: `has_result` is a per-frame `&mut false` read before it is set in the later column | `image_edit.rs:95,416,508` | S2 |
| Result lives only in a tab-local cache; no "save next to original", "add to library", or compare slider | `image_edit.rs:404-408` | S3 |
| Model repo, GGUF repo/file, precision, preview-every-N all inline at the top, same weight as Prompt | `image_edit.rs:97-250` | S3 |
| Load duplicates logic for "Load Model" and "Run Edit" (two 100-line pipelines) | `image_edit.rs:115-177,253-410` | S4 |

### 2.8 AI Assistant

| Observation | Evidence | Sev |
|---|---|---|
| Opens as a **separate OS window at every launch** (`assistant_window_open = true` on settings load); close-request is not handled, so only the View menu toggle closes it | `receive.rs:67,964-974` | S2 [U] |
| Bound to the **main** explorer (`current_thumb`, update channel), not the focused tab | `receive.rs:972`, `assistant.rs:220,299,1206` | S2 |
| "Inference" menu edits (endpoint, org) are discarded each frame (settings cloned per frame; typing lost) | `assistant.rs:323-340` | S2 |
| Provider/model selection exists in 3 places (Preferences, assistant bottom bar, left rail); all providers share one `openai_default_model` field | `receive.rs:352-433`, `assistant.rs:346-416` | S2 |
| Attachments: auto "selected" chip only appears if some other attachment exists; unclear what will be sent | `assistant.rs:418-473,1180-1186` | S3 |
| Slash commands `/ping /search /describe` are undiscoverable (hint text mentions only Enter/Shift+Enter) | `assistant.rs:510,1223-1311` | S3 |
| Results of `/search` go to an explorer tab via `AIUpdate::SimilarResults{origin_path:"mcp:media.search:<arg>"}` | `assistant.rs:1259-1264` | S4 |
| Messages are plain labels (no markdown), 376 px max bubble width hard-coded | `assistant.rs:692,1754-1757` | S4 |
| Sessions persist in DB (`assistant_sessions/messages`) and reload; delete has no confirm; "clear all sessions" button only clears the active chat | `assistant.rs:578-584,661-682` | S3 |

### 2.9 Refine tags/metadata

| Observation | Evidence | Sev |
|---|---|---|
| "Refine selection" (menu and context menu) queues paths but **does not open the Refinements tab**; the queue is drained only while the tab renders | `selection_menu.rs:132-141`, `table/mod.rs:1107-1111`, `refine.rs:64-73`; only opener is `navbar.rs:20-30` | S2 |
| "Accept All" applies every proposal regardless of checkboxes and filter; no confirm; "Reject" is a disabled button | `refine.rs:381-394,514` | S2 |
| Proposals are shown as Rust `Debug` text (`Some("x")`, `{:#?}` vec); no thumbnail, no per-field accept, no diff | `refine.rs:453-481` | S3 |
| Starting generation unloads other heavy models (hidden side effect) | `refine.rs:131` | S3 |
| Accept goes through `update_or_create_thumbnail`, which only overwrites non-empty fields; it can add but never clear | `thumbnails.rs:440-457` | S4 |
| Reachable only via File > Database > Refine or View > tabs | `navbar.rs:20-30`, `tabs.rs:5-9` | S3 |

### 2.10 Backup, restore, settings

| Observation | Evidence | Sev |
|---|---|---|
| Backup/Import are File > Database items using blocking OS dialogs (frozen UI while open); import "may overwrite existing data" with no progress or result besides a toast | `navbar.rs:58-107` | S3 |
| DB export includes `user_settings` => **plaintext API keys** in `.surql` | `settings.rs:11-50`, `database/mod.rs:259` | S2 |
| "Copy/Move N items..." files operation uses a folder dialog; label reflects selection but action takes all non-dir rows; Move has no confirmation | `selection_menu.rs:39-61`, `backup.rs:3-90` | S1 |
| "Delete N From DB" removes **all** non-dir rows (confirm shows that count, label shows selection count) and then `retain`s the table without fixing `table_index` | `selection_menu.rs:101-131` | S1 |
| Preferences window: unsafe aliasing of `&mut` draft (`*mut` deref twice), Save without Cancel/Revert, draft persists unsaved across close, "Reload CLIP Now" silently saves, section order Database/AI/Filters with filters settings placed under AI | `receive.rs:147-152,623-781` | S3 |
| "Egui Settings > Save" reports success but nothing persists (`egui_preferences` not in `UiSettingsDb`; also `STYLE` re-applied) | `settings.rs:11-50,161`, `receive.rs:63,800-818` | S2 |
| Several settings have no UI: `db_excluded_exts`, `ext_enabled`, `group_by_category` | grep | S4 |
| "Enable Auto Indexing/CLIP" in AI menu flips an atomic only (not persisted, reverts on restart) | `navbar.rs:119-133` | S3 |

### 2.11 Selection, destructive ops, keyboard, drag and drop, undo, feedback

| Observation | Evidence | Sev |
|---|---|---|
| Selection model = data-table cell cursor + `viewer.selected: HashSet<String>` synced by `on_highlight_change`; dirs excluded; no shift-range/ctrl-toggle semantics documented; selection survives neither tab nor scope change | `table/mod.rs:757-766` | S3 |
| Menus fall back to **all rows** when selection is empty (`Copy N items` uses `all_paths`) | `selection_menu.rs:39-43` | S1 |
| Table is a spreadsheet editor in disguise: row insertion/deletion allowed by default, hotkeys Ctrl+X/C/V, Ctrl+Shift+V, Ctrl+D, Ctrl+Shift+D, Ctrl+Delete, Delete, Backspace, Ctrl+Z/Y act on in-memory rows. `table_index` stores positions and is not updated by these ops, and lookups never check `row.path` | `viewer.rs:248-255,558-582` [F], `receive_scan.rs:574-577,359-364` | S2 [U] |
| Ctrl+C copies mismatched columns: codec indices (3=category,4=tags,5=modified,6=size,7=type) predate the current column order (3=size,4=type,5=CLIP,...) | `codec.rs:11-52` vs `table/mod.rs:178-184` | S3 |
| App-level shortcuts: **none**. Escape/Space/Enter/Ctrl+F/Ctrl+L/F5/Delete have no app meaning | grep `key_pressed` | S3 |
| Drag and drop: window flag set (`with_drag_and_drop(true)`), no handler for dropped files/folders; no row drag to groups/folders | `main.rs:140` | S3 |
| Undo: none beyond table row ops | grep | S3 |
| Feedback: toasts for settings/DB export/import and refine accept; everything else is `log::info!` into the Logs tab; status "X" button is a no-op; status order comes from a `HashMap` | `status.rs:181-186,329` | S3 |
| 24 `request_repaint` sites + 22 blocking `rfd` dialogs + 141 `tokio::spawn` in UI code | grep | S3 |

### 2.12 Cross-cutting defect register (inherit-risk list for Step M0)

| ID | Sev | Defect | Evidence |
|---|---|---|---|
| D01 | S1 | Delete-from-DB ignores selection | `selection_menu.rs:101-131` |
| D02 | S1 | Backup copy/move ignore selection; move unconfirmed | `selection_menu.rs:39-61`, `backup.rs:3-9` |
| D03 | S1 | Settings clobbered by stale copies (tabs, assistant) | section 1.4 |
| D04 | S1 | Startup unwrap panic (no Pictures dir) | `file_table/mod.rs:328-333` |
| D05 | S2 | Image Edit duplicate stub tab | `main.rs:24-35`, `tabs.rs` |
| D06 | S2 | Image Edit result can't be saved | `image_edit.rs:95,416,508` |
| D07 | S2 | Assistant Inference menu discards edits | `assistant.rs:323-340` |
| D08 | S2 | Refine selection doesn't open tab | `selection_menu.rs:139`, `table/mod.rs:1110` |
| D09 | S2 | Refine Accept All ignores flags; Reject disabled; Debug text | `refine.rs:381-394,514,453-481` |
| D10 | S3 | Preview/ImageEdit decode spawned every frame on miss; sync decode in async | `preview_pane.rs:173-199`, `image_edit.rs:455-479` |
| D11 | S3 | Bottom bar O(rows) x2 per frame + 300 ms repaint | `navbar.rs:289-441` |
| D12 | S3 | Recent Scans blocks UI 50 ms/frame; drives polled per frame | `quick_access_pane.rs:127,305-314` |
| D13 | S3 | 1 thumbnail message per frame; O(N^2) streamed rows | `receive_thumbnail.rs:8,64` |
| D14 | S2 | `follow_active_vision` default on | `file_table/mod.rs:371` |
| D15 | S2 | Preview pane forced open on click | `receive_thumbnail.rs:55-58` |
| D16 | S2 | Folders never selectable; "Exclude selected directories" always disabled | `table/mod.rs:757-766`, `selection_menu.rs:19-24,72` |
| D17 | S2 | Assistant OS window auto-opens; close not handled | `receive.rs:67,964-974` [U] |
| D18 | S2 | `db_loading` cleared next frame regardless of completion | `receive/mod.rs:49-66`, `load.rs:96` |
| D19 | S2 | Type toggles don't invalidate table cache | `table/mod.rs:195`, `validation.rs:15` [F,U] |
| D20 | S2 | Table row-mutation hotkeys active; `table_index` unsound | `viewer.rs:248-255` [F], `receive_scan.rs:574` |
| D21 | S3 | Ctrl+C column mapping stale | `codec.rs` |
| D22 | S3 | Dock "+" dead; background open ignored; layout not persisted | `main.rs:83,56-58,98-100` |
| D23 | S2 | Breadcrumbs, "Generate CLIP (Selected)", assistant bound to main explorer; `active_tab_title` = last rendered tab | `navbar.rs:146,211`, `tabs.rs:22` |
| D24 | S3 | Preferences: unsafe aliasing, no revert, shared model field | `receive.rs:148-152,352-433` |
| D25 | S2 | Egui Settings Save does not persist | `settings.rs:161`, `receive.rs:800` |
| D26 | S3 | Linux gaps: video thumbs, meters, relative `./db`, Windows model path | `thumbs.rs:107`, `windows.rs:228,231,265`, `database/mod.rs:29-32`, `app.rs:15` |
| D27 | S3 | RAW extension list mismatch | `types.rs:8` vs `thumbs.rs:18-36` |
| D28 | S3 | Blocking `rfd` dialogs on UI thread (22) | grep |
| D29 | S4 | MCP stdio server spawned twice | `main.rs:145-149`, `receive.rs:45-49` |
| D30 | S2 | Group semantics (single-membership) vs UI wording; remove-from-group likely fails | `database/mod.rs:73`, `database/logical_groups.rs:102-149` [U] |
| D31 | S3 | Hidden default min-size filter | `settings.rs:224` |
| D32 | S4 | Status "X" no-op; indicator order unspecified | `status.rs:329,181` |
| D33 | S2 | Destructive filtering via `retain` | `filter_menu.rs:34,124,184,219` |
| D34 | S2 | Path box / search box coupling for AI search | `menus/mod.rs:57-72` |
| D35 | S3 | Base64 thumbs in `SELECT *` rows | `thumbnails.rs:139-148` [E] |
| D36 | S4 | Dead: `Page` enum, `ViewMode`, `open_image_edit_with_path`, `ext_enabled`, `group_by_category`, `only_with_*`, `db_excluded_exts` UI | `app.rs:75`, `types.rs:44`, `app.rs:236` |

---

## 3. Proposed information architecture

### 3.1 Principles

1. **Scope, not mode.** Where you are (folder, library, collection, tag, search, similar-to, scan result, archive) is a value; "FS vs DB" becomes a per-row badge plus an "Indexed" filter.
2. **One selection, one action surface.** Every action takes an explicit target (selection | focused item | this view) and shows the count; no silent "all rows" fallback.
3. **Non-destructive views.** Filters, search, sort never remove rows from memory; they build an index over a row store.
4. **Global chrome, local content.** Sidebar, inspector, assistant rail, task bar exist once; only the Browser content is per tab.
5. **Everything is a command.** Menu, button, palette, shortcut, context menu, MCP/assistant all dispatch the same `AppCommand`.
6. **State is visible.** Active filters, active collection, running tasks, offline files are chips/badges, not colored button text.
7. **Debuggable by design** (user preference): each new surface registers a Devtools panel and a CLI/test hook (section 3.11).

### 3.2 Window layout

```
+--------------------------------------------------------------------------------------------+
| File  View  Tools  Help  |  [ Search library or run a command (Ctrl+K) ]  |  DB Vision CLIP  |
+----------+---------------------------------------------------------+-----------------------+
| SIDEBAR  | tabs: [Pictures x] [Tag: cat x] [+]       (dock, optional)| RIGHT RAIL  [Inspect|Ask]
|          | [<][>][^][R]  Pictures > 2024 > Trip  (click to edit)    |                       |
| PLACES   | query: [ text........ (Name|Tags|Semantic|Similar) ] chips: |  preview (fit/zoom)   |
|  Quick   |   [Images x][>=10KB x][Indexed x] [+Filter] [Views v]    |  name / open / reveal |
|  Drives  | view: [Grid|List] size [--o--] sort [Date v]             |  tabs: Info Meta AI   |
|  Recent  +---------------------------------------------------------+       Edit            |
| LIBRARY  |                                                         |  metadata (autosave)  |
|  All     |      virtualized grid / list                            |  AI: describe, CLIP,  |
|  Recent  |                                                         |   similar, refine     |
|  No CLIP |                                                         |  multi-select: batch  |
| COLLECT. |                                                         |   tag/category editor |
| TAGS     | selection bar: 12 selected (48 MB) [Tag][Collect][Describe]|                       |
| SCANS    |   [CLIP][Refine][Ask AI][Copy to][Move to][More v]       |  (Ask = assistant)    |
+----------+---------------------------------------------------------+-----------------------+
| 1,234 items . 12 selected . 3 filters (clear) | Scan 45% . CLIP 120/800 . Describe 3/50 [^] | Logs |
+--------------------------------------------------------------------------------------------+
```

- Sidebar, right rail, task bar are `Panel`s outside the dock, bound to the **focused** browser (`DockState::find_active_focused`, [F] `dock_state/mod.rs:175`).
- Center keeps `egui_dock` for browser tabs plus tool tabs (Refine, Image Edit, Settings, Devtools). Splits and undock stay available; default is one browser tab.
- Narrow widths (< ~900 px): sidebar collapses to an icon rail, right rail becomes an overlay drawer; toolbar controls collapse into a "..." menu. Minimum supported width should be decided up front (open question).
- Assistant lives in the right rail next to Inspector (one rail, not three columns); optional pop-out window preserved.
- Chrome visibility toggles in View menu + shortcuts; sizes persisted (section 3.9).

### 3.3 Navigation model

| Concept | Definition |
|---|---|
| `Scope` | `Folder(path)`, `Archive(zip/tar/7z virtual path)`, `Library`, `Recent`, `Collection(name)`, `Tag(t)`, `Category(c)`, `Semantic(text)`, `Similar(path)`, `Scan(id)`, `Smart(NoClip \| Undescribed \| Untagged \| FailedThumbs \| Duplicates)` |
| String form | `fs:/path`, `lib:all`, `col:Name`, `tag:x`, `cat:x`, `q:sem:"..."`, `sim:/path`, `scan:42`, `zip://...!/`, `smart:no-clip` (used for history, persistence, sidebar links, MCP/assistant deep links) |
| History | per browser tab: back/forward stack of scopes + scroll + selection anchor; Alt+Left/Right |
| New tab policy | default = reuse current tab; Ctrl/Middle-click or "Open in new tab" creates one; searches/similar/tag/category never auto-spawn tabs; Failed thumbnails become a Smart scope + Task Center entry |
| Tab title | derived from scope (replaces `uniquify_title` and the string-matched tab body) |
| Tab type | `enum TabKind { Browser(BrowserId), Refine, ImageEdit, Settings, Devtools, Logs }` instead of `String` (not persisted today, so safe to change) |
| Folder rows | single click selects; double click/Enter opens; folders are selectable (fixes D16) |
| Source merging | Folder scope shows FS listing merged with DB metadata (already done via `db_lookup` in `receive_scan.rs:126,333`); offline library rows show an "offline" badge; "Indexed only" is a filter chip |

### 3.4 Query bar, global search, command palette

- **Query bar** (per browser): one text field + mode segmented control `Name | Tags/Text | Semantic | Similar`; Enter runs; Semantic uses the existing text-embedding KNN (`ClipEmbeddingRow::find_similar_by_embedding`, `menus/mod.rs:86-144`) and appends pages on scroll (replaces "Load N more").
- **Filter chips** under it: Type, Size, Date, Excluded terms, Indexed, Has description, Has CLIP, Category, Tag. Every active filter shows a removable chip; defaults (e.g. >=10 KB) are chips too. `Views` dropdown = saved filter groups (`filter_groups` table) with save/rename/update.
- **Command palette (Ctrl+K)**: fuzzy over (a) commands (every `AppCommand`), (b) places/collections/tags/saved views, (c) recent folders, (d) semantic search ("Search images for ..."), (e) settings sections. Top bar search box is a collapsed palette entry point.
- **Path bar**: breadcrumb with click-to-edit (Ctrl+L), autocompletion from recent paths.

### 3.5 Selection model and contextual actions

`Selection { ids: IndexSet<PathKey>, anchor, focus }` owned by the browser view; click = single, Ctrl = toggle, Shift = range, Ctrl+A = all in view; **focus** drives Inspector (fixes multi-select preview lag, D15). Empty selection never means "everything".

| Action | Targets | Where | Confirm | Undo | Existing code reused |
|---|---|---|---|---|---|
| Open / Reveal / Copy path | focus | double-click, context, Inspector | no | - | `open::that` |
| Add to library (index) | selection or "this view" (explicit) | action bar, context | count shown | - | `upsert_rows_and_get_ids` |
| Remove from library | selection | action bar, context, Delete | yes, shows count | metadata journal (M10) | `delete_thumbnails_and_embeddings_by_paths` |
| Describe / Regenerate | selection | action bar, Inspector AI | bulk > N | - | `stream_vision_description` |
| Generate CLIP / Regenerate | selection | same | bulk > N | - | `clip_generate_for_paths` |
| Find similar | focus | context, Inspector | no | - | `find_similar_by_embedding` |
| Tag / Category (batch) | selection | action bar, Inspector | no | journal | `update_or_create_thumbnail` |
| Add to / remove from collection | selection | action bar, drag onto sidebar | no | journal | `LogicalGroup::*` |
| Refine... | selection | action bar | no | - | Refine tool |
| Ask AI | selection | action bar, drag to rail | no | - | `request_attach_to_chat` |
| Edit image | focus | Inspector | no | - | Image Edit tool |
| Copy to / Move to | selection | action bar | Move: yes | op log | `backup.rs` |
| Exclude folder from scans | selected folders | context | no | - | settings `excluded_dirs` |
| Generate for "this view" / "folder" | explicit scope picker in dialog with counts | Tools menu, palette | yes | - | scan_menu bulk |

Context menu = same action list (keep the existing item ids: `generate_description`, `regenerate_description`, `generate_clip_embeddings`, `regenerate_clip_embedding`, `add_selection_to_group`, `remove_selection_from_group`, `open_selection_in_new_tab`, `exclude_dirs_from_scan`, `find_similar`, `refine_selection`, `attach_to_chat_window`; `table/mod.rs:797-860`).

### 3.6 Inspector (right rail)

- Header: name, Open / Reveal / Copy path, offline badge.
- Preview: fit/fill/100%, zoom/pan, next/prev through the view; "Open viewer" (lightbox overlay, arrows, Esc); video player when `ffmpeg` feature available, else poster frame plus "Open externally".
- Tabs: **Info** (size, dates, type, hash, dimensions, collection, CLIP/description status) | **Metadata** (caption, category, tags as chips, description; autosave on blur with a dirty/saved indicator; create DB row on first edit instead of silently doing nothing) | **AI** (Describe/Regenerate with streaming text, Generate CLIP, Find similar, status) | **Edit** (Image Edit launcher).
- Multi-select: counts, common tags, batch add/remove tags, set category, add to collection, Describe/CLIP all.
- Streaming interim text shows for the focused item only; AI "follow" becomes a Task Center toggle that is off by default (fixes D14).

### 3.7 Task Center and status bar

| Element | Content |
|---|---|
| Left | `N items . M selected (size) . K filters (clear)`; computed once per view revision, not per frame (fixes D11) |
| Middle | aggregated activity chip(s): Scan %, Thumbnails, Describe queue (`index_active/queue_len/completed`), CLIP, Refine, Image edit, Export/Import, Copy/Move, model loads; click opens popover list |
| Popover rows | title, progress, elapsed, Cancel (existing: `cancel_scan`, `cancel_bulk_descriptions`, edit cancel flag), Retry, "Show failures" (opens Smart scope), error text |
| Needs attention | encrypted-archive password prompts, failed thumbnails, model load errors (replaces modal-only flow and the extra "Failed thumbnails" tab) |
| Right | engine dots (DB, Vision, CLIP, Rerank, Edit, Assistant) with existing hover cards from `status.rs`; Logs button |
| Performance | CPU/RAM/VRAM move under Devtools > Perf and show only where the platform provides data (stubs on Linux, `windows.rs:228,231,265`) |
| Toast policy | toast on completion/failure of user-initiated tasks only; background progress never toasts |

`TaskRegistry` wraps today's `StatusMeta` (state, detail, progress, error, device, model) and adds id, kind, cancel handle, started/ended.

### 3.8 Settings structure (full page tab, search box, Apply/Revert, dirty marker)

| Section | Contents | Source fields (`UiSettings`) |
|---|---|---|
| General | startup behavior (restore tabs/last scope), default view, thumbnail size, confirmations | new (UI layout, not in DB record; see 3.9) |
| Library | DB location (+ "move/copy DB"), backup/restore (progress, redact keys option), watched locations, scan defaults (min/max size, skip icons, excluded dirs/exts/terms, date range), batch size, auto-save-to-library, active collection default | `excluded_dirs`, `db_excluded_exts`, `filter_skip_icons`, `db_min_size_bytes`, `db_max_size_bytes`, `filter_modified_after/before`, `scan_found_batch_max`, `auto_save_to_database`, `recent_paths` |
| AI Models | Vision (model dir picker, device), CLIP/SigLIP model, reranker + Jina options, image-edit model (repo/GGUF/precision), memory policy ("unload other models"), Load now / Reload buttons | `joycaption_model_dir`, `clip_model`, `reranker_model`, `jina_*`, `clip_augment_with_text`, `clip_overwrite_embeddings` |
| Automation | auto-indexing, auto CLIP, overwrite descriptions, prompt template (with reset) | `auto_indexing`, `auto_clip_embeddings`, `overwrite_descriptions`, `ai_prompt_template` |
| Assistant and providers | backend (Phase 4: zeroclaw default), per-provider key (masked) / base URL / model **per provider**, test connection | `ai_chat_provider`, `*_api_key`, `openai_base_url`, `openai_default_model`, `openai_organization`, `recent_models`, `last_used_model` |
| Metadata tools | rename/merge category and tag, delete tag, prune tags per item: preview counts then confirm | (DB ops in `thumbnails.rs:348-413`) |
| Shortcuts | table of all commands and bindings, rebinding later | new |
| Appearance | Phase 2 owns this (theme, Egui settings window retires here) | `egui_preferences` (not persisted today) |
| Advanced / Devtools | logs level, perf HUD, MCP tool tester, raw settings JSON viewer, "reset settings" | - |

Settings state: one `SettingsStore` (`Arc<RwLock<UiSettings>>` + revision + `update(|s| ..)` patch + change events), replacing all copies in section 1.4; saves are debounced patches (never whole-record writes from stale copies).

### 3.9 Persistence of UI layout

Use eframe storage (`save()` is already a stub at `main.rs:98-100`; `persistence` feature already enabled in `Cargo.toml`) for a versioned `ui_layout_v1` JSON: dock state, sidebar/rail widths and open flags, last scope per tab, view mode, thumbnail size, sort, column visibility. **Do not** add layout fields to `UiSettingsDb` (see risk R1).

### 3.10 Empty, first-run, error states

| State | Treatment |
|---|---|
| No database / path unreadable | Blocking first-run wizard: (1) location (default `ProjectDirs` data dir; detect legacy `./db/path.txt`), (2) add folders (prefill Pictures/Videos if present; no panic if absent), (3) AI mode: Off / Local models (path pickers, detect) / Cloud-zeroclaw, (4) start first scan. Each step skippable; progress persisted as a checklist card on the empty Library |
| Library empty | Card with the checklist + "Scan a folder" |
| Folder empty / filtered to zero | "N items hidden by 3 filters" with Clear; if query active offer "Search semantically instead" |
| AI not configured | Inline CTA in Inspector AI tab and Describe dialogs ("Choose model folder") |
| Offline files | Row badge + "Locate..." later; included in Smart scopes |
| Scan running | Placeholder skeletons + Task Center chip; no empty-state flash |
| DB error | Banner with Reconnect / Change location / Open logs |

### 3.11 Keyboard map (collision-checked against `default_hotkeys`, `viewer.rs:558-582` [F])

| Scope | Keys |
|---|---|
| Global | Ctrl+K palette; Ctrl+F query; Ctrl+L path; Ctrl+, settings; Ctrl+B sidebar; Ctrl+I inspector; Ctrl+J assistant; Ctrl+Shift+T tasks; F5/Ctrl+R refresh; Alt+Left/Right/Up nav; Ctrl+T/W/Tab tabs; F1 shortcut sheet; F12 devtools; Esc = clear selection -> clear query -> close overlay |
| Browser | Arrows move focus; Enter open; Space quick look; Ctrl+A all; Shift/Ctrl+click range/toggle; Ctrl+1/2 grid/list; +/- thumbnail size; T tag; C collection; D describe; Delete remove from library (confirm); Ctrl+Shift+C copy path |
| Viewer | Left/Right prev/next; +/-/0 zoom; I info; Esc close |
| Table (list view) | override `hotkeys()` to drop Cut/Paste/Duplicate/Delete*; set `allow_row_insertions/deletions = false`; keep copy only after fixing the codec |

### 3.12 Devtools (per user preference: debug panel for every feature)

One `Devtools` tab (F12) with sections: Command log (every `AppCommand` with args and result), Task registry (force progress/fail buttons), Perf (frame time, rows, texture cache size, scan batch timings from `perf_batches`, decode queue depth, CPU/RAM/VRAM where available), Settings (raw JSON + patch log), Scope inspector (current scope string, history, filter spec), DB console (read-only SurQL, row counts per table), MCP tool tester (move the existing assistant menu items `assistant.rs:240-313`), Logs (embed `egui_logger`). CLI hooks: `--scope <string>`, `--run <command>`, `--dump-state` for headless smoke checks.

### 3.13 Feature mapping: current location -> new home (nothing dropped)

| Current | New home |
|---|---|
| File > Database > View Entire Database | Sidebar Library > All media |
| File > Database > Refine (AI, DB-only) | Tools > Refine; Selection bar "Refine..." |
| Group by Categories / Tags (virtual folders) | Sidebar Tags and Categories sections (with counts) |
| Choose DB Folder / Export Backup / Import | Settings > Library > Database and Backup |
| AI > Enable AI Search | Settings > AI Models "Load now"; auto-load on first use; engine dot menu |
| AI > Enable Auto Indexing / Auto CLIP | Settings > Automation (persisted) + Task Center toggles |
| AI > Generate CLIP (Missing / Selected / Count) | Sidebar Smart "No CLIP (n)" + selection/palette "Generate CLIP" |
| Preferences | Settings tab |
| View > Egui Settings | Settings > Appearance (Phase 2) / Advanced |
| View > tabs (File Explorer / Logs / AI Refinements) | View menu panel toggles; Tools menu; Devtools |
| View > AI Assistant Window | Right rail "Ask" (+ pop-out) |
| Top breadcrumbs | Per-browser breadcrumb/path bar |
| Top status buttons | Task bar engine dots |
| Bottom: scan bar, thumbs bar | Task Center |
| Bottom: VRAM/CPU/RAM | Devtools > Perf |
| Bottom: Total Size / Videos / Images / Dirs / Filtered | Status bar left (cached) |
| Bottom: Select menu | Selection action bar + context menu |
| Bottom: Logical Groups menu | Sidebar Collections + "Add to collection" dialog |
| Toolbar: gear | Sidebar toggle (Ctrl+B) |
| Toolbar: up / back / forward / refresh / home | Nav buttons + Alt shortcuts + F5 |
| Toolbar: clear DB table (double-click) | removed (close/Reset view) |
| Toolbar: path box, AI checkbox, Apply Filter | Path bar + Query bar (Semantic mode) |
| Toolbar: Filter popup | Filter chips + "+ Filter" popover + Views dropdown |
| Toolbar: Search for files | Query bar |
| Toolbar: eye (preview) | Inspector toggle (Ctrl+I) |
| Toolbar: Scan popup | "Scan..." dialog (scope, types, save to library, group, auto-describe/CLIP) + Task Center |
| Scan: Save Current View to DB | "Add to library" action (explicit scope) |
| Scan: Bulk Generate Descriptions / Generate CLIP / Cancel Bulk | "Process..." dialog from selection/view/folder + Task Center cancel |
| Scan: Follow active vision | Task Center toggle (default off) |
| Toolbar: Database Mode button | removed (Scope + Indexed chip) |
| Toolbar: Load N more (similarity) | infinite scroll in Semantic/Similar scopes |
| Table columns / sort / resize / persistence | List view (kept) + Grid view |
| Category / Tag cell click | filter chip (click) / new tab (Ctrl+click) |
| Table context menu (11 custom items) | Unified context menu + action bar (same ids) |
| Preview pane (open/reveal, Edit Image, metadata, Generate, Find Similar, Find Similar (Engine), Generate CLIP, video) | Inspector (Find Similar (Engine) moves to Devtools as a comparison probe) |
| Quick Access: Logical Groups, User Directories, Recent Paths, Drives, WSL | Sidebar Collections, Places (Quick, Recent, Drives, WSL) |
| Quick Access: Performance Tweaks (FoundBatch) | Settings > Library > Scanning |
| Quick Access: Recent Scans | Sidebar Scans |
| Quick Access: Scan performance | Devtools > Perf |
| Archive password modal | Kept as modal, queued also under Task Center "Needs attention" |
| Pagination bars | Virtualized infinite list; page size becomes an internal loader detail |
| Dynamic tabs (Scan:, Recent:, Drive:, Category:, Tag:, Similar to, Selected, Entire Database, Failed thumbnails) | Scopes (history) with optional new tab |
| AI Refinements tab | Refine tool tab (thumbnails, per-field accept/reject, undo) |
| Image Edit tab | Image Edit tool tab launched from Inspector (single instance, focus-if-open) |
| Logs tab | Devtools > Logs and status bar link |
| Preferences > Database (rename/merge tag/category, delete tag, prune) | Settings > Metadata tools |
| Assistant window: sessions, Model Options, MCP Tools menu, Inference menu, OpenRouter models | Right rail: sessions drawer, model picker, tools menu moves to Devtools |
| DB Setup modal | First-run wizard (and Settings > Library > Database) |

---

## 4. Architecture seams the new IA needs

Kept deliberately small; these are the minimum to make steps shippable.

```rust
enum Scope { Folder(PathBuf), Archive(String), Library, Recent, Collection(String), Tag(String),
             Category(String), Semantic(String), Similar(String), Scan(u64), Smart(SmartView) }
enum Target { Current, NewTab, Background }
enum AppCommand { OpenScope{scope: Scope, target: Target}, Run{action: ActionId, on: Selector},
                  ShowSettings(Section), AttachToAssistant(Vec<PathKey>), Refine(Vec<PathKey>),
                  Toast(ToastKind, String), TaskUpdate(TaskId, TaskEvent), .. }
struct BrowserView { scope, history, query, filters: FilterSpec, selection, view_mode,
                     store: RowStore, index: ViewIndex, loader: LoaderHandle }
```

- `RowStore`: `Vec<Row>` + `HashMap<PathKey, usize>` maintained together (fixes unsound `table_index`, D20); `PathKey` normalizes separators/case on Windows (replaces `paths_equivalent_windows`, `receive_ai_update.rs:7-23`).
- `ViewIndex`: visible/sorted indices + `revision`; all counts/stats cached per revision.
- `Row` is a light summary (path, name, type, size, modified, flags, category, tags); thumbnail bytes live in a `ThumbService` (LRU texture cache, worker-pool decode via `spawn_blocking`/rayon, optional disk cache); list queries stop selecting `thumbnail_b64`.
- `CommandBus`: one `crossbeam` channel drained once per frame in `App::ui`; replaces `OPEN_TAB_REQUESTS`, `OPEN_SETTINGS_REQUEST`, `ATTACH_REQUESTS`, `REFINE_SELECTION_REQUESTS`, `ACTIVE_GROUP_NAME` (kept as shims during migration).
- `SettingsStore`, `TaskRegistry` as in 3.7/3.8. Move `ui::status` to a UI-neutral module so `ai/` and `database/` stop importing `crate::ui`.
- Blocking `rfd` -> `AsyncFileDialog` + command result.
- Assistant boundary: `trait AssistantBackend` (local joycaption, OpenAI-compatible, zeroclaw in Phase 4) fed with `AssistantContext{ selection, scope, filters }`; same `AppCommand`s are what a tool-calling backend would invoke.

---

## 5. Risks

| ID | Risk | Mitigation |
|---|---|---|
| R1 | Adding a **non-Option** field to `UiSettingsDb` likely makes existing `user_settings` records fail to deserialize; the failure is swallowed (`receive.rs:98-109`) and the app silently runs on defaults [U] | New settings as `Option<T>` only; UI layout in eframe storage; add a settings-load failure banner |
| R2 | `egui-data-table` / `egui_dock` / `egui-toast` / `egui_json_tree` are forks pinned to `egui-037` branches (`Cargo.toml:12-13,46,59`); each egui bump needs 4 fork ports | New shell reduces dock use; list view can move to `egui_extras::TableBuilder` and drop data-table; decide on dock later |
| R3 | `Thumbnail` is used as DB row, UI row and AI-engine DTO; slimming the UI row must not change the stored shape | `Row` projection + explicit `SELECT` field lists; keep `Thumbnail` for writes |
| R4 | Sentinels in data: `file_type` = `<DIR>`, `<ARCHIVE>`, `<PASSWORD_REQUIRED>`; `category` = `<FAILED_THUMB>`; virtual paths `cat://`, `tag://`, `zip://`, `tar://`, `7z://` | Wrap in typed `RowKind` at the edge; keep reading sentinels; never persist new ones |
| R5 | Phase 2 worktrees likely touch `navbar.rs`, `status.rs`, `table/mod.rs`, `quick_access_pane.rs` (glyphs/theme) [U] | Land M0-M2 (logic-only) first or rebase on Phase 2; start M3 from merged Phase 2; route all new icons through one indirection |
| R6 | Grid view memory (textures/VRAM) and decode throughput | Texture LRU with byte budget, devtools perf panel, visible-range-only decode |
| R7 | Windows-only paths (WSL, `wmic`, shell thumbs) not exercised on Linux and vice versa | keep `cfg` gates, add per-OS smoke list to each PR, replace `wmic` |
| R8 | Backups contain plaintext API keys | Option to exclude `user_settings` on export; move keys to OS keyring later |
| R9 | Group semantics change (many-to-many) needs data migration | Phase separately; until then label as "Collection (a file is in one)" and fix remove path |
| R10 | MCP stdio server starts twice and shares stdio with a GUI | single start; make opt-in via flag/setting |
| R11 | Big-bang rewrite risk of `FileExplorer` (102 fields) | extract types one at a time (`RowStore` -> `Selection` -> `Scope`) behind the existing struct; every step keeps the old UI running |

### Behaviors and data that must be preserved

| Area | Items |
|---|---|
| DB (SurrealKV via `DB`, ns `file_explorer`, db `ai_search`) | tables `thumbnails`, `user_settings` (record `user_settings:ShadowbrokerPC`, `settings.rs:213`), `filter_groups`, `logical_groups`, `cached_scans`, `cached_scan_items`, `clip_embeddings` (unique `path`, HNSW dim 1024 cosine), `assistant_sessions`, `assistant_messages`; indexes `lg_name_idx`, `path_idx`, `idx_parent_dir`, etc. (`database/mod.rs:41-172`); `./db/path.txt` and `./db/ai_search` legacy location |
| Settings keys | all `UiSettingsDb` fields (`settings.rs:11-50`): `ext_enabled, excluded_dirs, group_by_category, auto_indexing, ai_prompt_template, overwrite_descriptions, filter_modified_after/before, filter_skip_icons, db_min/max_size_bytes, db_excluded_exts, auto_clip_embeddings, clip_augment_with_text, clip_overwrite_embeddings, clip_model, recent_paths(10), recent_models(8), last_used_model, auto_save_to_database, joycaption_model_dir, reranker_model, jina_*, scan_found_batch_max, ai_chat_provider, *_api_key, openai_base_url, openai_default_model, openai_organization` |
| Data semantics | `Thumbnail` fields; unique `path`; `logical_group` required record (default `logical_groups:Default`); `update_or_create_thumbnail` merge-not-clear semantics; scan filter semantics (`Filters`, `FiltersExt`); `FilterGroup` slug naming |
| Channel protocols | `AIUpdate::{Interim, Final, SimilarResults{origin_path}}` with origin conventions `"query:<text>"`, `"mcp:media.search[:arg]"`, plain path = image similarity; `ScanEnvelope{scan_id, ScanMsg::*}` and `cancel_scan(id)`; `FilterRequest::{NewTab, OpenDatabaseAll, OpenPath}` semantics |
| Assistant/MCP hooks | `request_attach_to_chat`, `AssistantPanel::set_ui_settings`, slash commands `/ping /search /describe`, MCP tools `util.ping`, `media.search`, `media.describe` (`ai/mcp/mod.rs:58-111`) incl. UI helpers `ping_tool/media_search_tool/media_describe_tool`, `ASSIST_STATUS`, session persistence, OpenRouter model browser, `GLOBAL_AI_ENGINE` atomics (`auto_descriptions_enabled`, `auto_clip_enabled`, `cancel_bulk`, `index_*` counters) |
| Status registry | `DB_STATUS, VISION_STATUS, CLIP_STATUS, RERANK_STATUS, QWEN_EDIT_STATUS, ASSIST_STATUS`, `GlobalStatusIndicator` API used from `ai/` and `database/` |
| Features | archive browsing (zip/tar/7z, password queue), RAW decode path, WSL/drive quick access, CLIP/SigLIP model choices, reranker (Heuristic, Jina M0), cloud refine, Qwen image edit (GGUF/precision fallback), recent paths/models/scans, cached scan reopen |

---

## 6. Phased migration plan (each step = independently shippable PR chunk; app runs after each)

Effort: S <= 1 day, M 2-4 days, L 1-2 weeks, XL > 2 weeks [E].

| Step | PR scope | User-visible result | Kept | Rewritten | Deleted | Debug/test hook added | Effort |
|---|---|---|---|---|---|---|---|
| **M0.1** Safety fixes | D01, D02 (act on selection, confirm with counts, no empty-selection fallback), D04 (no unwrap; fallback dir), D05, D06, D07, D08, D09 (Accept All honors flags; Reject works), D14 (follow off), D15, D16 (dirs selectable) | Destructive buttons do what they say; no crash on minimal Linux | all UI | handlers in `selection_menu.rs`, `backup.rs`, `refine.rs`, `image_edit.rs`, `tabs.rs`, `main.rs` | dead `Reject` stub | unit tests for selection resolution | M |
| **M0.2** `SettingsStore` | single store + patch updates; remove `viewer.ui_settings`, `assistant.settings` copies; fix D03, D24 shared model field (per-provider keys stored as today, added fields `Option`) | Tabs honor saved filters; Prefs changes reach assistant immediately | DB schema unchanged | `receive.rs` Preferences code reads draft from store; `settings.rs` helpers | `settings_draft` unsafe aliasing | Devtools-lite: log every patch | M |
| **M0.3** Perf quick wins | D10 (in-flight set + `spawn_blocking`), D11 (cache stats per revision), D12, D13 (`while` drain, HashMap dup check), D18, D35 (heavy decode off async workers) | Smoother idle CPU, faster group open | all | `navbar.rs` bottom stats, `receive_thumbnail.rs`, `quick_access_pane.rs` | per-frame loops | frame-time overlay (egui `ctx.input(.stable_dt)`) behind `--perf-hud` | M |
| **M1** Core types and `CommandBus` | move `ui::status` to neutral module; `AppCommand` + bus; shims for the 5 statics; central dispatcher in `App::ui`; `PathKey` | none (internals) | all UI | `main.rs` drain loop, statics become wrappers | the 5 statics (after migration) | command log panel (temporary tab) | M |
| **M2** Task Center + status bar | `TaskRegistry` over `StatusMeta`; new bottom bar and engine dots; hide unsupported meters; cancel buttons; "needs attention" for passwords/failed thumbs | Single place for progress, cancel, errors | `status.rs` data model, cancel fns | bottom half of `navbar.rs`, `status_bar_inline` | bottom meters/labels, "Failed thumbnails" auto-tab | Tasks devtools list with fake-task buttons | M |
| **M3** Shell + Sidebar | new top bar (slim menus + palette stub); global left `Panel` with Places/Library/Collections/Tags/Scans (content moved from `quick_access_pane.rs`; tags/categories use existing count queries); links emit `OpenScope` (still opens tabs) | Always-visible navigation; Quick Access gear gone | `quick_access()`, `list_drive_infos()`, `CachedScan` | `navbar.rs` top, `quick_access_pane.rs` -> `sidebar.rs` | per-tab left panel, gear button | sidebar section toggles in Devtools | L |
| **M4a** `RowStore`/`ViewIndex`/`Selection` | extract from `FileExplorer`; `table_index` replaced; selection owned by view; list view still data-table (insert/delete/hotkeys disabled, codec fixed) | Correct thumbnails on rows; multi-select preview fixed | data-table renderer | `file_table/mod.rs` internals, `table/mod.rs` hotkeys | `table_index`, `selected` set | row-store invariant checks in Devtools | L |
| **M4b** `Scope` + loaders + history | unify FS/DB/group/category/tag/similar loaders; per-tab history; remove Database Mode button; tab reuse policy; offline badge | No modes; Back/Forward everywhere | `utilities/scan.rs`, `database/*` | `load.rs`, `navigation.rs`, `explorer/mod.rs`, `receive_*` | `table_menu.rs`, `OpenDatabaseAll` | Scope inspector (string in/out), `--scope` CLI | XL |
| **M4c** `FilterSpec` | non-destructive filters over `ViewIndex`; active chips (read-only first); remove all `table.retain` | Filters reversible; hidden-filter indicator | `FiltersExt` | `filter_menu.rs` logic | 6 `retain` sites | filter spec dump | L |
| **M5** Inspector | global right rail from focused selection: preview, metadata (autosave, dirty), AI tab, batch editor; basic viewer overlay | Preview never hijacks; edits don't vanish | `thumbs.rs` generators | `preview_pane.rs` -> `inspector.rs` | per-tab right panel, forced-open logic | Inspector state dump | L |
| **M6** Query bar, chips, actions | new toolbar (nav, breadcrumb, query bar, view switch); filter chips + saved views; Scan dialog; Process dialog; selection action bar; collections picker; context menu unified | Search/filter/act in one place | `ClipEmbeddingRow` KNN, saved filters, bulk fns | `menus/*` | `filter_menu.rs`, `scan_menu.rs`, `selection_menu.rs`, `logical_groups.rs` menus | action registry list with enable/disable reason | XL |
| **M7** Grid view + ThumbService | virtualized grid, size slider, texture LRU, worker decode, optional separate thumbnail storage; list view via data-table or `TableBuilder` | Photo-library feel; large folders smooth | DB thumbs (read path) | `table/mod.rs` cell drawing | per-row base64 clone paths | texture-cache and decode-queue panels | XL |
| **M8** Settings + onboarding | Settings tab per 3.8; first-run wizard; backup/restore UI with progress and key redaction; metadata tools; async file dialogs | Guided setup; safer backups | `database::{export_to, import_from}` | `receive.rs` windows | Preferences/Egui Settings/DB modal windows, `Page` enum | settings JSON viewer | L |
| **M9** Assistant rail | embed `AssistantPanel` in right rail; context chips; slash-command help; `AssistantBackend` trait; remove auto-open OS window (keep pop-out) | One window; Phase 4 plug-in point | sessions, MCP, OpenRouter browser | `assistant.rs` UI shell | viewport auto-open | MCP tester moved to Devtools | L |
| **M10** Tools + undo | Refine and Image Edit as polished tool tabs (thumbnails, per-field accept, save/compare/add-to-library); metadata edit journal with Undo | Reviewable AI changes; Ctrl+Z for metadata | `ai/refine.rs`, `qwen_image_edit` | `refine.rs`, `image_edit.rs` UI | duplicate load pipelines | journal viewer | L |
| **M11** Cleanup and persistence | persist layout (eframe storage); delete shims/dead code; split `receive.rs`; per-OS smoke checklist | Restores workspace on launch | - | - | old `navbar.rs`, `menus/`, `quick_access_pane.rs`, `preview_pane.rs`, global statics, `codec.rs`, dead settings UI | `--dump-state` CLI | M |

Ordering notes: M0-M2 touch mostly logic and bottom-bar code (low conflict with Phase 2 theme/icons). M3+ should start from merged Phase 2. M4b is the main risk and should land only after M4a; M7 can start in parallel with M5 once `RowStore` exists. Phase 4 (zeroclaw) needs only M1 (command bus) + M9 (backend trait).

---

## 7. Uncertainties and open questions

| # | Item |
|---|---|
| U1 | Behaviors tagged [U] (D17, D19, D20 outcomes, D30, R1) are from reading only; nothing was run |
| U2 | Window-width overflow and thumbnail memory figures are arithmetic estimates |
| U3 | Whether `.claude/worktrees` Phase 2 branches restructure `navbar.rs`/`status.rs` was not inspected (out of scope) |
| Q1 | Should "Logical Groups" become many-to-many Collections (needs migration) or stay single-membership (rename "Library folders")? |
| Q2 | Is it acceptable to move thumbnail bytes out of `thumbnails` rows (blob/disk cache) with a one-time migration? |
| Q3 | Keep egui_dock tabs for multiple browsers (power-user compare) or single browser with history and optional split? |
| Q4 | Minimum window size to support (current default 1000x750)? |
| Q5 | Will the zeroclaw backend need to drive the explorer (select, filter, tag)? If yes, `AppCommand` should be designed as its tool surface now |
| Q6 | Should Linux get video thumbnails/playback (ffmpeg feature on by default) as part of this phase? |
| Q7 | Drop `egui-data-table` in favor of `egui_extras::TableBuilder` for list view? |
| Q8 | OK to default the DB location to the OS data dir (with legacy detection) instead of CWD-relative? |

---

## Appendix A. `UiSettings` fields: where they are edited today

| Field | Edited in |
|---|---|
| `excluded_dirs` | Preferences/AI; Select menu (always disabled, D16); table context menu |
| `auto_indexing` | Preferences/AI; AI menu (atomic only) |
| `auto_clip_embeddings` | Preferences/AI; AI menu (atomic only) |
| `ai_prompt_template`, `overwrite_descriptions` | Preferences/AI |
| `filter_modified_after/before` | Preferences/Filters; Filter menu (global) |
| `filter_skip_icons` | Preferences/AI; Filter menu |
| `db_min/max_size_bytes` | Filter menu only (default min 10 KiB) |
| `db_excluded_exts`, `ext_enabled`, `group_by_category` | no UI |
| `clip_model` | Preferences/AI (UI fallback "unicom-vit-b32" vs Default "siglip2-large-patch16-512", `receive.rs:580`, `settings.rs:230`) |
| `clip_augment_with_text`, `clip_overwrite_embeddings` | Preferences/AI |
| `recent_paths` | written by `push_history` on main-explorer nav |
| `recent_models`, `last_used_model` | assistant |
| `auto_save_to_database` | Preferences/AI; Scan menu checkbox |
| `joycaption_model_dir`, `reranker_model`, `jina_*` | Preferences/AI |
| `scan_found_batch_max` | Quick Access > Performance Tweaks |
| `ai_chat_provider`, keys, `openai_*` | Preferences/AI; assistant bottom bar; assistant Inference menu (broken) |
| `egui_preferences` | Egui Settings window (not persisted) |

## Appendix B. Key code pointers

| Topic | Location |
|---|---|
| Tab-open protocol | `main.rs:20-75`, `file_table/mod.rs:34-59`, `app.rs:14` |
| God struct | `file_table/mod.rs:99-312` |
| Selection / row click behavior | `table/mod.rs:474-699,757-766`, `receive_thumbnail.rs` |
| Destructive actions | `selection_menu.rs:101-131`, `backup.rs`, `filter_menu.rs` |
| Settings persistence | `database/settings.rs`, `receive.rs:61-110,672-781` |
| Scan pipeline UI side | `receive_scan.rs`, `utilities/scan.rs` |
| Status registry | `ui/status.rs` |
| Platform stubs | `thumbs.rs:99-111`, `windows.rs:227-231,265-271`, `explorer.rs` |
| Fork behavior checked | `egui-data-table` `viewer.rs:248-255,530-584`, `draw/state/validation.rs:4-55`; `egui_dock` `tab_viewer.rs:67`, `dock_state/mod.rs:175,370` |
