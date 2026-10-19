# Browser: toolbar, query bar, action bar, dialogs

The row above the rows of every browser tab, the bar that appears over the bottom of the rows while something is selected, the context menu, and the Scan and Process dialogs. Code: `src/ui/file_table/{toolbar,query,action_bar,batch,shortcuts}.rs`, `src/ui/file_table/dialogs/`, `src/ui/file_table/menus/breadcrumbs.rs`, the registry in `src/command/actions.rs`.

## Toolbar

One row, `Back Forward Up Refresh Home | path area | query bar | Scan Process ...`. Back, Forward and Up use the history of the tab (Alt+Left, Alt+Right, Alt+Up; the mouse buttons do the same), Refresh reads the scope again (F5 or Ctrl+R). The row folds into the More menu as it narrows; nothing overlaps or clips at any width.

| Row width | Navigation | Mode control | Query field | Scan, Process |
|---|---|---|---|---|
| wide (about 1150 px and up) | five buttons | four named segments | up to 280 px | buttons with text |
| medium | five buttons | four icons | 170 px | icon buttons |
| narrow | four (Home in More) | one icon opening a menu | 150 px | Scan icon, Process in More |
| tight (about 520 px and less) | three (Refresh, Home in More) | one icon | 90 to 110 px | both in More |

The More menu always has Copy address, Open in the file manager, Scan this folder and Search (Ctrl+F), plus whatever the row folded away.

### Path area

Breadcrumbs for a place on disk (a click opens that folder, Ctrl or middle click a new tab, hidden leading folders sit under `...`), the name and scope of anything else. A click on the empty part of the area, or Ctrl+L, turns it into a text field with the current place selected. Enter opens what is typed: a path, or any scope string (`lib:all`, `tag:sunset`, `smart:no-clip`, `q:sem:"red car"`, see `commands.md`). Esc cancels.

Under the field, up to eight suggestions: folders inside the folder typed so far, the recent places (the tab's history and the saved recent folders), and scope strings while the text is not a path. Up and Down choose, Tab completes (a folder gets a separator so the next level completes in turn), Enter opens the chosen one, a click opens a suggestion. The disk is read only when the text changes.

## Restored tabs

With Settings > General > Restore tabs on, a tab that comes back from `workspace.json` (see `settings.md`) shows its view settings at once but reads its scope when it is first drawn: a tab behind the front tab of its leaf stays empty until you bring it forward. A tab that cannot read its scope shows a placeholder instead of rows: **Waiting for the library** while the library connects (at most 20 s), **Not available right now** with the reason (a folder that is gone, a library that did not connect) and a **Retry** button, or **Search not run** with **Run search** for a saved semantic or similar-file search. Any navigation in the tab replaces the placeholder. Devtools > Workspace shows what was restored; `state.json` from `--dump-state` lists `pending` per tab.

## Query bar

A text field with a mode: **Name | Text | Semantic | Similar**.

| Mode | While typing | Enter |
|---|---|---|
| Name | file name contains the text (live, through the tab's filter) | keeps the filter |
| Text | text found in name, type, path, tags, category or description (live) | keeps the filter |
| Semantic | no filter | opens `q:sem:"<text>"`, which pages results by CLIP |
| Similar | no filter | opens `sim:<path>` for the path typed, or for the focused file when the field is empty |

Name and Text write the `name` and `query` conditions of the tab's filter, so they show as chips (`Name: ...`, `Search: ...`), hide rows without removing them and go away with the chip. The bar follows changes made elsewhere: removing the chip empties it, a search scope puts its text in it, a filter set by command or saved view shows in it. Ctrl+F puts the cursor in the field; Esc clears the text and keeps the cursor, a second Esc leaves the field. Devtools > Query shows the text, what typing runs, what Enter runs and a history of runs; `{"cmd": "query", ...}` and `--query <mode>:<text>` drive it without the mouse.

## Selection action bar

While rows are selected a bar floats over the bottom of the rows with the count and size (`12 selected (48 MB)`), the actions below and a button that clears the selection. Actions that do not fit the width go into a More menu, the ones that matter least first. A button that cannot run on this selection is dimmed and its tooltip says why. An empty selection never means everything: with nothing selected there is no bar and every action refuses.

| Action | Key | Needs | Asks first |
|---|---|---|---|
| Open | Enter | a selected item (at most 20) | no |
| Reveal | | a selected item (at most 20) | no |
| Copy path | Ctrl+Shift+C | a selected item | no |
| Add to library | | files, library connected | above 25 files |
| Remove from library | Delete | files, library connected | always, with the count |
| Tag | T | files, library connected | the popover shows the count |
| Category | | files, library connected | the popover shows the count |
| Add to collection, Remove from collection | C | files, library connected | the picker shows the count |
| Describe, Regenerate | D | images, vision model set up | above 25 images |
| CLIP, Regenerate CLIP | | images | above 25 images |
| Find similar | | a file, library connected | no |
| Refine... | | files | no |
| Ask AI | | files | no |
| Copy to | | files on disk | folder picker |
| Move to | | files on disk | always, then a folder picker |
| Exclude folder | | a selected folder | no |
| New tab (menu only) | | a selected item | no |

**Tag and Category** open a popover above the bar. Tag shows the tags on the selection (click one to mark it for removal), a field that takes comma separated tags and suggests known ones, and an `Add to N files` button. Category sets or clears one category for the whole selection, with suggestions. **Add to collection** lists the collections with their counts, filters by name, and offers `Create "<name>" and add N files` for a name that does not exist; Remove from collection lists them too. A popover belongs to the selection it opened on and closes when the selection changes, on Esc or on a click outside.

**Confirmations** name the exact number of files in the text and on the button. Removing from the library deletes the rows and their CLIP embeddings; files on disk and collection memberships are not touched. In a folder the rows stay (they are still files) and lose their Indexed badge; in the library, a collection, a tag or a search they leave the view. Move to asks before it opens the folder picker.

**Results** toast counts: `Added 2 tags to 3 files; 1 file had no change`, `Removed 3 files from the library (2 CLIP embeddings deleted)`, `Describing 4 files; 2 already described`. Copy and move run as Backup tasks with progress and a Cancel in the task list.

## Context menu

The context menu of the grid and the table lists the same registry, in groups (Open, Library, Organize, AI, Files), with the shortcut of each and the reason an entry is off in its tooltip. It keeps the item ids of the earlier table (`generate_description`, `regenerate_description`, `generate_clip_embeddings`, `regenerate_clip_embedding`, `add_selection_to_group`, `remove_selection_from_group`, `open_selection_in_new_tab`, `exclude_dirs_from_scan`, `find_similar`, `refine_selection`, `attach_to_chat_window`) and adds `open`, `reveal`, `copy_path`, `add_to_library`, `remove_from_library`, `tag_selection`, `category_selection`, `copy_to` and `move_to`. A click runs the entry through the same executor as the bar. Devtools > Browser actions lists every registry action with its JSON name, menu id, key, whether it can run on the current selection, the reason when not, and a Run button; `{"cmd": "run", "action": "...", "on": ...}` reaches the same code (`commands.md`).

## Scan dialog

Tools > Scan folder, the Scan button, Process / Scan in the status bar, the palette and `{"cmd": "dialog", "dialog": "scan"}` open it. Fields: the folder (with Browse and a shortcut to the folder shown), Include subfolders, Add the files found to the library (saved with their collection when the scan is done, as a task with progress), and where the results go (a new tab or this tab). Below, the filters that will decide what is listed, as chips: for a new tab the defaults of the settings (including the 10 KiB minimum size), for this tab its own.

**A scan loads everything.** Size, date, icon, extension and excluded-text conditions no longer drop files while the scan runs; every image, video and archive it finds becomes a row and the tab's filter decides which are listed. Removing a chip brings the rows back without a rescan (a scan that fits one page of 5000 rows keeps every row; pages of a larger scan hold the rows the filter lists). Excluded folders are filtered the same way, so the scan still walks them.

## Process dialog

Tools > Process files, the Process button, the status bar menu, the palette and `{"cmd": "dialog", "dialog": "process"}` open it. **Files**: the selection (with its image count), this view (what the filters list) or a folder (counted in the background, with or without subfolders). **What to do**: describe with the vision model, generate CLIP embeddings, or both, each with a replace flag and the count it would work on (`Describe: 1 of 2 images (1 already described)`). More than 25 images, or a folder, asks once more with the exact count. Each run is a task (Describe or Clip in the task list) with progress and Cancel; descriptions stream into the tab as they finish, and a checkbox keeps the preview on the image being described (off by default). Replacing embeddings deletes the old ones of those images just before the new ones are made.

## Keys

| Keys | Does | Works |
|---|---|---|
| Ctrl+F | cursor in the query field | anywhere while a browser is in front |
| Ctrl+L | edit the address | anywhere while a browser is in front |
| F5, Ctrl+R | read the scope again | anywhere while a browser is in front |
| T, C, D, Delete | Tag, Add to collection, Describe, Remove from library on the selection | while the pointer is over the tab or the last click was in its rows |
| Ctrl+Shift+C | Copy path | same, handled by the table |
| Esc | clear the selection, then the query; closes a popover or dialog first | same |

The plain keys never fire while a text field has the keyboard, while a menu, popover, dialog or the address field is open, or when nothing is selected. They do not collide with the table keys (arrows, Page keys, Home, End, Enter, Space, Ctrl+A, plus and minus, F9, Ctrl+1 and Ctrl+2), Ctrl+K (palette), Ctrl+B (sidebar), Ctrl+I (inspector) or Alt+arrows (history).

## Devtools

| Window | Shows |
|---|---|
| Query | the bar's text and mode, what typing and Enter run, the filter conditions it wrote, a history of runs, controls to set and run it |
| Browser actions | every registry action, whether it can run now and why not, what it would act on, how it asks, a Run button, and a history of runs with their origin (bar, menu, key, command, devtools) |

Open them with View > Devtools or `--devtools query,browser_actions`.
