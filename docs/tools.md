# Refine, Image Edit and the metadata journal

Code: `src/ui/refine.rs` and `src/ui/refine/`, `src/ui/image_edit.rs` and `src/ui/image_edit/`, `src/ai/refine.rs` and `src/ai/refine/`, `src/database/journal.rs`, `src/ui/journal_ui.rs`, `src/ui/devtools/{journal,refine}.rs`. Commands: `undo`, `redo`, `refine`, `open_tool` (`commands.md`).

## Metadata journal

Every metadata write the application makes is recorded in the `metadata_journal` table (created when first used, SCHEMAFULL): one entry per path per change with the batch, the time, a source label, the field, the value before and after (JSON text). Fields are `caption`, `description`, `category`, `tags` and `collection` (membership, with the collection key as target). The newest 5000 entries are kept; whole batches are dropped, oldest first, and the newest batch always stays.

| Writer | Label of the batch |
|---|---|
| Inspector autosave (`metadata_edit::save_patch`) | Edit metadata |
| Inspector and action batch tags (`batch_tags`) | Add tags, Remove tags, Edit tags |
| Batch category (`batch_category`) | Set category, Clear category |
| Tag and category tools (`Thumbnail::rename_tag`, `delete_tag`, `rename_category`, `prune_tags`) | Rename tag "a" to "b", Delete tag "a", Rename category..., Limit tags to N per file |
| Collections menu and Inspector (`journal::add_to_collection_with`, `remove_from_collection_with`, `copy_collection_with`) | Add to "name", Remove from "name", Copy "a" to "b" |
| Refine accept | Accepted refinements (one batch per bulk accept) |
| Accepting MCP proposals | Accepted proposal from <client>, Accepted MCP proposals (several at once) |

`undo` reverts the newest batch that is still applied; `redo` re-applies the batch undone last; a new change clears what could be redone. Each entry is applied only when the row still holds the value the entry expects. A row that was edited since is skipped, the entry is marked skipped (so it never blocks the stack) and the toast counts it: "Undid 12 changes in 12 files (Add tags); 2 skipped because the file changed since". Rows that an edit created in the library stay there when the edit is undone. Ctrl+Z and Ctrl+Shift+Z do the same while no text field has focus.

### Recording from a new writer

```rust
let batch = journal::begin_batch("Add tags");                                          // 1
journal::record(batch, &path, Field::Tags, Value::tags(&before), Value::tags(&after)); // 2, per path
journal::commit(batch).await;                                                          // 3
```

`journal::Recorder` wraps the three steps for any connection (`Recorder::new(label)`, `.text(...)`, `.tags(...)`, `.member(...)`, `.commit_with(&db)`); a journal failure is logged and never fails the edit. `record` ignores changes whose before and after are equal and keeps the first before and the last after of repeats. `commit` writes the whole batch in one transaction. Collection changes use `record_member(batch, path, collection_key, before, after)`. Writers that call the functions above (`metadata_edit::*`, `collections` through `journal::*_collection_with`, `Thumbnail::*_tag`) need nothing more; a writer that updates `thumbnails` or `collection_items` directly must record by hand.

Devtools > Journal lists the newest entries and batches, the conflicts of the last undo or redo, has Undo, Redo and Clear buttons and a form that writes a field of a file through the same journaled writer.

## Refine tab

Opens from Tools > Refine, from "Refine" on a selection and with `{"cmd": "refine", "paths": [...]}`; the files of a selection limit the next run. Proposals come from three places and show in one list: the Heuristic or Jina M0 reranker (neighbours by CLIP vote on tags, category, caption and description), a cloud model, and the MCP store (`docs/mcp.md`), whose pending proposals are read when the tab is shown and every 8 s.

- **Cards**: a thumbnail, the file name and folder, the source (Heuristic, Jina M0, Cloud <provider>, MCP <client>), the confidence (the mean share of the 50 nearest neighbours that carry the proposed tags and category; none for cloud and MCP proposals), the reason of an MCP proposal, and per field a checkbox with the current and the proposed value side by side. Tags show added tags with `+` and removed ones struck through with `-`.
- **Accept** on a card writes the checked fields of that proposal. **Accept ... fields in ... files** in the bar writes the checked fields of every proposal the filters show, after a confirmation with the exact counts; proposals with nothing checked and proposals the filters hide are left alone. Nothing checked means nothing is written. **Reject shown** drops the shown proposals (stored ones are rejected in the store) after a confirmation.
- **Filters**: text (path, source, reason, current and proposed values), source, minimum confidence with a switch for proposals that have none, and a sort (path, confidence, source, newest). The list shows 40 cards and a "Show more" button.
- **ZeroClaw source**: Generator > Use ZeroClaw asks the ZeroClaw gateway instead of the local reranker or a cloud provider (`docs/zeroclaw.md`). The switch works only while the gateway is Online, otherwise it is disabled and the reason is shown. Rows go in batches of 20 to 40 under the gateway's 64 KiB limit; each batch is one request, proposals arrive as batches come back, Cancel stops after the request in flight and keeps what was made, and the job is the Refine task. Text only: no pictures are sent. A Hint field names another route hint than the profile's. `{"cmd": "zc", "op": "refine", "paths": [...]}` opens the tab for files and starts such a run (`docs/commands.md`).
- **Run**: Generate proposals runs as a Refine task in the Task Center with progress and Cancel (`tasks::start`); proposals arrive in batches of 12 and survive tab switches. Generating again replaces the proposals of the session and keeps the stored ones. Running the heuristic stops the vision worker, running Jina M0 also unloads CLIP, a cloud run leaves the local models alone.

## Image Edit tab

Opens from the Inspector's Edit tab, Tools > Image edit and `{"cmd": "open_tool", "tool": "image_edit", "path": "..."}` (the file is loaded; a second open focuses the same tab).

- **Prompt** with ten presets (Shift+click adds a preset to the prompt), a negative prompt, guidance, strength, steps, full precision and a live preview every N steps. Model settings (repo, GGUF repo and file, Fetch, Load, Unload) are collapsed.
- **Run edit** loads the model when it is not loaded, then edits. Progress comes from the pipeline status and shows in the tab and as an Image edit task in the Task Center; Cancel in either place stops at the next step and keeps no result.
- **Compare**: side by side, swipe (drag the divider) or flip (click) between the original and the result. The original is the preview of the file from the thumbnail service.
- **Save as new file...** asks for a name; **Save next to original** writes `name-edited.png` (then `-2`, `-3`, ...) in the folder of the original; **Replace original...** asks first and names the file; the original is never written otherwise. **Add to library** saves the result next to the original when it has no file yet, adds it to the library and to the chosen or new collection (journaled).
- **History**: the last 12 results of the session with the prompt and whether they were saved; picking one brings back its source image.
- **Errors**: out of memory, model load and unreadable source each say what to do; out of memory offers "Retry at lower precision" (full precision off, at most 20 steps). The pipeline also retries once at reduced precision by itself.

### Without the model

Devtools > Refine and Image Edit shows the state of both tabs, adds sample proposals with made-up values for library files, drives the bulk buttons and switches Image Edit to a test backend that grays, inverts or warms the picture in a moment (a prompt containing "simulate oom" fails like a GPU out of memory). `SMART_MEDIA_IMAGE_EDIT_BACKEND=test` starts the tab on that backend. Open the window at start-up with `--devtools refine`.
