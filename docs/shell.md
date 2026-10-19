# Window shell

Top strip, global sidebar, command palette and the saved window layout. Code: `src/ui/shell.rs`, `src/ui/sidebar/`, `src/ui/palette/`, `src/ui/navbar.rs`, `src/ui/titlebar.rs`.

## Top strip

`File`, `View`, `Tools` and `Help` menus, then the palette entry (Ctrl+K), the window title and the engine chips. Below 1150 px the chips shrink to icons and the palette entry loses its hint text; at 700 px (the minimum window width) the palette entry is an icon.

| Menu | Entries |
|---|---|
| File | Open folder, Scan folder, Database (view entire database, backup and restore, export, import, choose folder), Preferences, Quit |
| View | Sidebar (Ctrl+B), Bottom bar, Right rail, Appearance, Egui Settings, Custom title bar, Tabs, AI Assistant Window, Devtools |
| Tools | Scan folder, AI processing, Refine, Image edit, Collections, Scan settings (batch size) |
| Help | Logs, Command palette, About |

## Sidebar

Always bound to the focused browser tab. Every row sends `open_scope` (see `commands.md`): a click opens the scope in the current tab, Ctrl or middle click in a new tab, Shift click on a folder scans it recursively. Right click offers the same choices plus Copy scope; collections also have Rename and Delete.

| Section | Rows | Loaded |
|---|---|---|
| Places | user folders, drives, WSL mounts, recent folders | system, every 20 s |
| Library | All media (count), Recent, No CLIP, Undescribed, Untagged, Failed thumbnails (count), Duplicates | count every 30 s |
| Collections | collections with live counts, New collection | every 8 s |
| Tags, Categories | top 12 by count, filter box, show more | when opened, every 120 s |
| Scans | last 20 saved scans | every 30 s |

Lists load in the background and only while their section is open. A finished task reloads them. Other code can call `ui::sidebar::request_refresh()` after changing tags, categories or collections. Without a connected database only Places show, with a Connect button.

Below 900 px the sidebar is an icon rail: one icon per section opens its rows in a flyout.

## Saved layout

`<config dir>/smart-media/ui_layout.json`, written when the layout changes and the pointer is up.

```json
{
  "version": 1,
  "sidebar_open": true,
  "sidebar_width": 240.0,
  "section_open": { "places": true, "tags": true },
  "bottom_bar_open": true,
  "right_rail_open": true
}
```

Missing fields take defaults, the width is clamped to 170..420, and a missing, unreadable or newer-version file gives the defaults without panicking (a newer file is not overwritten).

## Command palette

Ctrl+K, the strip entry, `Help > Command palette` or `{"cmd": "open_palette", "query": "..."}`. Type to match commands, places, collections, tags, categories, scans and settings sections. Up and Down move, Enter runs, Ctrl+Enter opens scopes in a new tab, Esc closes. Any text also offers `Search library for "<text>"`. Entries that cannot run show why.

## Devtools

| Window | Shows |
|---|---|
| Sidebar | mode and width, section toggles, list cache age and refresh, the layout JSON and a disconnected-library preview |
| Actions | every palette command with its id, shortcut, enabled state and reason, and a Run button |
| Perf | also lists the scan batch timings of the focused tab (moved from the old Quick Access pane) |

Open them with `View > Devtools` or `--devtools sidebar,actions`.

## Where each old entry went

| Old entry | Now |
|---|---|
| File > Database > View Entire Database | File > Database > View Entire Database; sidebar Library > All media |
| File > Database > Refine (AI, DB-only) | Tools > Refine (AI, DB-only) |
| File > Database > Group by Categories, Group by Tags | sidebar Categories and Tags sections (counts, filter box) |
| File > Database > Choose DB Folder, Database maintenance, Export Backup, Import | File > Database (same entries) |
| File > AI > Enable AI Search, Enable Auto Indexing, Enable Auto CLIP | Tools > AI processing (same entries) |
| File > AI > Generate CLIP (Missing, Selected), CLIP Missing Count | Tools > AI processing (same entries) |
| File > Preferences | File > Preferences |
| File > Appearance, View > Appearance | View > Appearance |
| File > Exit | File > Quit |
| View > Egui Settings | View > Egui Settings |
| View > File Explorer, Logs, AI Refinements | View > Tabs; Help > Logs; Tools > Refine |
| View > AI Assistant Window | View > AI Assistant Window |
| View > Icon gallery | View > Devtools > Icon gallery |
| View > Devtools | View > Devtools (plus Sidebar and Actions) |
| View > Custom title bar | View > Custom title bar |
| Explorer gear button, Quick Access pane | sidebar (Ctrl+B) |
| Quick Access: Collections | sidebar Collections |
| Quick Access: User Directories, Recent Paths, Drives, WSL, physical drives | sidebar Places |
| Quick Access: Performance Tweaks | Tools > Scan settings |
| Quick Access: Recent Scans | sidebar Scans |
| Quick Access: Scan performance | Devtools > Perf |
