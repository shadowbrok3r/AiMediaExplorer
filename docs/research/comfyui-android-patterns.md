# comfyui-android patterns for AME phase 3 (UI reorganization)

Study 4 of 5. Read-only. No builds run. Scope: phase 3 only (no theme/icon/style proposals; phase 2 owns those).

Path aliases used below:

| Alias | Path |
| --- | --- |
| CA | `/home/shadowbroker/Documents/Rust/Mobile/comfyui-android` (working tree, v0.23.12, uncommitted edits on top of HEAD `13721a2`) |
| AME | `/home/shadowbroker/Documents/Rust/AiMediaExplorer` (master, `1d1a1db`) |
| EM | `/home/shadowbroker/Documents/Rust/Mobile/EguiMobile` (framework CA depends on) |
| ZC | `/home/shadowbroker/Desktop/zc-codex` (outside the brief; read only to compare ZeroClaw clients) |

Evidence convention: a `file:line` is something I read. "Design" sections are my proposals. `(unverified)` marks anything I did not confirm in code or by running it. CA line numbers are from the working tree on 2026-10-18 and will drift because CA has uncommitted changes.

---

## 0. Corrections to the brief (read first)

| Brief said | Reality | Consequence |
| --- | --- | --- |
| `apps_builtin/` at top level | It is `CA src/apps_builtin/*.json` (8 files) | none |
| `appearance.rs` = theme/appearance system | It is a **prompt-tag look-axis classifier** (hair/eyes/outfit/pose/camera/environment), `appearance.rs:365`. UI theme = `theme.rs` + `vendor/egui-mobile-theme` | The "appearance" file matters only for tag grouping |
| `cooc.rs` powers search | It is **prompt-tag co-occurrence** learned from prompts the user queues and from auto-tag results, `cooc.rs:27-146`, `app.rs:6581,27529` | Useful for "related tags", not for ranking images |
| `history.rs` / `backup.rs` = media history/backup | `history.rs` = **graph-editor undo/redo snapshots**; `backup.rs` = **AES-GCM encrypted Settings export**. Prompt history is `PromptHist` in `types.rs:1407` | Different scope than expected |
| `zeroclaw.rs` covers protocol, sessions, tools | It is a **one-shot, text-only** client: fresh connection per request, no sessions, tool frames ignored (`zeroclaw.rs:1-17,121-140`). The full client is `ZC android/src/zeroclaw.rs` | See §3.10: port ZC's client for the assistant, CA's for utility calls |
| `clip_index` / `tag_index` imply a gallery search UI of their own | The only text-search path is a View-menu toggle + `Enter` in the gallery search box (`app.rs:22302,22212`). Results are limited to the **loaded listing** | AME has a full DB, so the design in §6 differs |
| `uiwf`, `workflow`, `graphview`, `schema`, `h3`, `apps` | ComfyUI-specific (graph editor, prompt grammar, workflow conversion) | Principles only (§3.11, §4) |

---

## 1. TL;DR: what to take, in order

| # | Take | From | Why it matters in AME (verified problem) | Effort |
| --- | --- | --- | --- | --- |
| 1 | Byte-budgeted `ThumbCache` + bounded off-thread decode + per-frame claim budget + size tiers | `CA gallery.rs:1746-1954`, `engine.rs:2335-2390`, `app.rs:24744` | AME renders thumbs as base64 PNG in DB rows -> `HashMap<String,Arc<[u8]>>` -> egui `bytes://` loader. That loader **spawns one OS thread per decode and never evicts** (`egui_extras-0.37.0 loaders/background_decode.rs:62-96,103-109`). Cell code base64-decodes on the UI thread (`AME table/mod.rs:261-275`) | M |
| 2 | Ranked-result overlay (`Semantic`/`Similar`) + memoized visible set keyed by a signature hash | `CA app.rs:568,23473-23544,23657-23738` | AME's search is a 200 px "Search for files" field + an "AI" checkbox (DB mode only) fired by Enter on the *path* box or an "Apply Filter" button (`AME menus/mod.rs:51-130,168-170`; AI checkbox 62, Apply Filter 66), plus preview-pane buttons (`preview_pane.rs:440-532`) and a nav-menu tangle (`navbar.rs:108-170`) | M |
| 3 | Idle-gated, single-flight indexing pumps with coverage latch, failed-set, batched saves, Resume/Rebuild | `CA app.rs:27547-27820,6685-6803` | AME fires `tokio::spawn` per file and has per-file DB round trips in search (`AME ai_search/generate.rs:50-58,85-93`) | L |
| 4 | Faceted tag window + folder cards with fanned cover stacks | `CA app.rs:23409-23468,22476-22626` | AME has `tags_idx`/`category_idx` and no browse UI | M |
| 5 | Field-level diff sheet (`RemixDiffRow`) and streaming review modal (`ExpandReview`) | `CA gallery.rs:148-254`, `engine.rs:878-913` | AME Refine applies proposals with a per-row checkbox only (`AME refine.rs:49-61`) | M |
| 6 | Perf toolkit: `FrameProf` + `BgLap` + `ui_detail` + procfs `sysmon` HUD | `CA app.rs:995-1100,28361,29145`, `sysmon.rs` | AME's CPU/RAM/VRAM bars are Windows-only; `gpu_mem_mb()` returns `None` elsewhere (`AME windows.rs:231-232`) | S-M |
| 7 | Pure preflight validators + one-tap `Fix` + error modal with repeat folding | `CA preflight.rs`, `lint.rs:9-28`, `app.rs:28417-28550` | AME launches heavy jobs (Qwen edit, JoyCaption, rerank) with ad-hoc checks | S-M |
| 8 | `Player` facade (ctrl atomics + `status()` poll + pause-when-hidden) | `CA player.rs`, `EM egui-android/src/video/surface.rs:401-504` | AME video preview is a CPU RGBA stream paced by `thread::sleep`, no audio, no seek, duration never set (`AME preview_pane.rs:49-79,252`; `file_table/mod.rs:367`) | L (backend decision) |
| 9 | ZeroClaw: use ZC's client for assistant (sessions, tools, approvals, images); CA's for single-turn utility calls | §3.10 | Phase 4 | M-L |
| 10 | Pure `MaskCanvas` + replayable strokes + `ViewXform` | `CA mask.rs` | AME Image Edit has prompt/steps/strength only (`AME image_edit.rs:9-13`) | M |
| 11 | Centroid "smart collections" with review deck, denied memory, online suggestions | `CA app.rs:17820,19002-19065,25356-25500`, `clip_index.rs:127-174` | Best fit for "categories" the user defines by example | L |
| 12 | Settings autosave diff + corrupt-file guard; encrypted export | `CA app.rs:5835-5913`, `backup.rs` | AME stores provider API keys as plain `Option<String>` fields of `UiSettings` (`AME database/settings.rs:42-46,104-108`; used at `assistant.rs:1322-1347`) and `eframe::App::save` is a stub (`main.rs:97`) | S |

Not worth taking: anything touch-specific, the Android storage/cache plumbing, the ComfyUI graph machinery, the anime-tag vocabularies (unless WD14 tagging is in scope for AME; §9 Q3).

---

## 2. comfyui-android in one page

- Crate split: portable core (`engine, schema, uiwf, types, workflow, gallery, clip_index, tag_index, cooc, tags, lint, preflight, mask, sysmon, media_meta, backup, zeroclaw`) compiles and tests on the host. `mod app` is `#[cfg(target_os="android")]` (`lib.rs:41-42`), so host `cargo check` never sees the UI (`CLAUDE.md:113-124`). Pure logic carries unit tests (ThumbCache, ClipIndex, TagIndex, Cooc, History, Preflight, Mask, Sysmon).
- UI: one `ComfyApp` god object (`app.rs` = 30,926 lines, 630 fns) implementing `EguiApp::update`. Tabs: Create, Graph, Gallery, Settings(+Logs).
- Async: `Engine` owns a 2-worker tokio runtime, one `std::sync::mpsc` of `Msg`, `drain()` each frame, `ctx.request_repaint()` after each send (`engine.rs:444-515,2595-2597`). UI never awaits.
- Frame skeleton (`app.rs:27849-28322`): `thumbs.begin_frame()` -> drain `Msg` -> timed "bg" phase (polls, pumps, autosave) -> modal Back-key ladder -> panels -> windows -> `end_frame()` profiling.
- Evidence of measure-then-fix: commit `27b1cbc` ("kill the bg stalls and gallery jank") lists on-device-confirmed fixes: index saves off UI thread, budgeted FUSE stats, pumps latch off when done, gallery visible/groups/facets memoized behind a signature, off-screen tiles cost zero item access.

---

## 3. Subsystem findings

### 3.1 Gallery rendering (images)

**Thumbnail pipeline** (all verified):

1. Grid asks `thumbs.get(thumb_key)`; on miss `thumbs.claim(key)` (single-shot) then `engine.fetch_thumb` (`app.rs:24782-24807`).
2. `fetch_thumb` (`engine.rs:2335-2390`): key = `"{subfolder}/{filename}#{size}"`. Tries local full-image cache first inside `spawn_blocking(read_full_cache -> decode_thumb)`, else HTTP then `spawn_blocking(decode_thumb)`. Comment at `2352-2354`: the cache-hit decode used to run on the UI thread at ~13 ms/tile and was "the whole scroll-time frame cost".
3. `decode_thumb` (`engine.rs:4454-4461`): `image::load_from_memory` -> `DynamicImage::thumbnail(size,size)` (fast integer downscale, per `image-0.25.10 dynimage.rs:913-925`) -> `ColorImage`; size clamped 64..1024.
4. `Msg::Thumb` handler (`app.rs:3796-3825`): records aspect ratio per item (`thumb_aspects`), warns if `w*h*4 > 4 MB` ("a decode escaped its clamp"), `ctx.load_texture(.., TextureOptions::LINEAR)`, `thumbs.insert(key, tex, bytes)`.
5. Tier by tile size: `GalleryView::thumb_size()` = 1024 / 512 / 320 for 1 / 2 / 3 columns (`types.rs:2807-2813`). Size is part of the key so a column change never collides.

**`ThumbCache`** (`gallery.rs:1741-1954`), the single most valuable piece:

| Member | Behaviour | Line |
| --- | --- | --- |
| `textures`, `order: VecDeque<(key,cost)>`, `bytes` | LRU-ish insertion order; budget **64 MB** in bytes, not count ("a 320px thumb is ~0.4 MB, a one-column 1024px read is ~4 MB") | 1746-1761 |
| `seen: RefCell<HashMap<key,frame>>` + `frame` | Marks each key read this frame; `begin_frame()` ages them; "on screen" = read this or previous frame (previous counts because the message drain runs *before* the draw) | 1841-1866 |
| `insert` eviction loop | Second chance: an on-screen entry goes to the back once per pass (`spared` set), so a burst of N inserts cannot evict visible tiles; terminates because each entry is requeued at most once | 1911-1948 |
| `claim` / `reset_pending` | Single-shot fetch claim so a visible tile does not queue a request per frame | 1870-1877 |
| `forget(prefix "{key}#")`, `forget_exact` | Drop all tiers of one item / one exact key | 1880-1909 |
| `downscale_rgba` | Box-downscale guard for producers that ignore the size hint | 1771-1807 |
| tests | `claim_is_single_shot` 2782, `a_visible_tile_outlives_one_that_scrolled_away` 2829, `a_drawn_tile_survives_a_whole_drain_of_inserts` 2856, `eviction_terminates_when_everything_is_on_screen` 2944, `eviction_is_by_bytes_not_count` 2965, `reinsert_does_not_leak_budget` 3005 | 2782-3014 |

Excerpt (the eviction core, `gallery.rs:1937-1947`):

```rust
let mut spared = HashSet::new();
while self.bytes > BUDGET_BYTES && self.order.len() > 1 {
    let Some((old, cost)) = self.order.pop_front() else { break };
    if self.on_screen(&old) && spared.insert(old.clone()) { self.order.push_back((old, cost)); continue; }
    self.seen.borrow_mut().remove(&old); self.textures.remove(&old); self.bytes = self.bytes.saturating_sub(cost);
}
```

**Grid** (`gallery_grid`, `app.rs:24711-24859`):

- Row virtualization by hand: for `cols > 1` rows are fixed height, so it computes `first/last` from `ui.clip_rect()` and `ui.cursor().top()` and emits two `allocate_space` spacers (`24724-24740, 24855-24857`). The 1-column feed has variable row heights (uses the image aspect from `thumb_aspects`, 24751-24761) and keeps the full loop, which only costs allocations.
- `claim_budget = 12` fetch dispatches per frame; if exhausted, `request_repaint_after(120ms)` (24744, 24790-24793). Comment: a fast scrollbar drag otherwise claims whole screens the user already passed.
- Tile = `ui.put(rect, Image)` for pixels plus a **separate** `ui.interact(rect, id, Sense::click())` so "thumbnail completion and refreshed row order must not change touch ownership" (24811-24812).
- Records `gallery_on_screen (lo,hi)` and `gallery_drawn` order (24771-24774); `update_prefetch_window` turns that into the prefetch range (23303-23317).
- Badges: video play badge (`video_badge` 29439), "tag-indexed" dot (24819), "unseen" dot + nav-icon count (24826, 29262), selection overlay (29570).
- Debug: `ui_detail = "grid 12ms claims 5"` is fed to the frame profiler (24133-24137).

**Memoized view** (`app.rs:23657-23738, 23935-23949`): `gallery_view_sig()` hashes every input (list gen, len, media, rating, sort, group, tag text, facets, `tag_index.len()`, `clip_index.len()`, `gallery_dep_epoch`, ranked keys, character names). `compute_gallery_visible()` is O(N) and runs only when the sig changes. Comment at `1117-1119`: before memoization this cost 40-230 ms/frame at ~1250 items.

**Groups as folder cards** (`group_folders` 22629, `folder_card` 22572, `paint_stack` 22476, `album_card` 22492): square cards, newest 3 pictures fanned (center-cropped by UV, 22501-22506), name + count under it. Draw closures hold `&self`, so they push fetches into a `CoverWants` struct applied afterwards (`app.rs:1105`, `fetch_cover_wants` 22516). Covers are asked once per `(album id, count)` (22452).

**Selection**: tap toggles in select mode; browse mode needs a deliberate double-tap (`tile_double_tapped` 24871); long-press 0.6 s then drag paints selection with edge auto-scroll (24632-24704); selection bar actions operate on `compute_gallery_visible()` not the raw list (24262-24276: "All -> Trash could delete images the filter was hiding").

**Pull-to-refresh** (`GalleryPullGesture` `gallery.rs:14-82`, tests 1961-2066): touch-only state machine.

**Disk caches** (`gallery.rs:1184-1653`): full images in a flat dir; `.key` sidecars hold the original key so the listing can be rebuilt offline (`offline_items` 1489); `write_atomic` = `.part-<pid>-<seq>` + `sync_all` + rename (1275-1293); budget 32 GB and 1 GB free-space reserve, **nothing is ever deleted to make room** (1193-1198, 1599); size total scanned at most every 5 min (1201, 1609); stale `.part` files reaped by age (1640-1644); on-device renders exempt from wipes (1313, 1565). Android-only: `.nomedia` (1191), `statvfs` over FUSE (1578-1595).

**Viewer** (`Viewer` 535, `open_viewer` 25082, `gallery_viewer` 25865):

- Opens by firing parallel fetches (full image, albums, workflow) and shows the best cached thumb (1024 -> 512 -> 320) while the full image loads (26349-26359): progressive display.
- Zoom/pan (`viewer_media_rect` 25138-25194): MAX_ZOOM 8; anchor-preserving zoom `pan += (at - slot.center() - pan) * (1 - next/zoom)`; pan clamped to `slack = (shown-avail)/2`; double-click toggles fit/2x at the cursor; `owns_gesture` is false while any sheet/menu is open (25159). Math ports as-is; swap `multi_touch()` for `zoom_delta()` + wheel on desktop.
- All actions collected into `enum Act` during immediate-mode drawing, matched afterwards (25866-25894, 26383+). Same idea as `CoverWants`: no borrow fights, one place where state changes.
- Filmstrip (`filmstrip` 26853): horizontal `ScrollArea`, `scroll_to_rect` to center current. **Not virtualized** (see §8).
- Metadata overlay (`viewer_meta_overlay` 26618): floating panel, per-section copy buttons, tag chips from the local tag index (26631, 26812-26825).

### 3.2 Video (`player.rs`, `EM egui-android/src/video/surface.rs`)

- `Player` (`player.rs:17-32`) wraps `SurfacePlayer` (Android MediaPlayer -> SurfaceTexture -> external-OES texture painted by a glow callback; `surface.rs:1-3,247-466`; no pixels cross JNI).
- Control surface is atomics (`PlayerCtrl{paused, seek:i32, looping, muted}` 10-15) written by the UI; `pump()` (121-174) applies them to the native player **only when the tuple changes** (`applied`, 125-135), converts a frame-index seek to ms (138), and suppresses `cur` updates while a seek is pending until position is within `500/fps` ms or 2 s elapses (154-162).
- `status()` returns `{position_ms, duration_ms, width, height, ready, playing, fps, frame_count, has_audio, error}` (`surface.rs:401-412`). `pump` repaints every 16 ms only while playing/seeking/not ready (171-173).
- Lifecycle in `update()`: pause when backgrounded or viewer closed, remember `auto_paused` to resume only if the user had not paused (`app.rs:27936-27947`, `on_pause` 28327).
- Controls (`app.rs:26243-26334`): play/pause (restarts at end), loop, mute (only if `has_audio`), clock label with reserved width, frame-index scrub slider writing `ctrl.seek`.
- `last_frame_png` (`player.rs:41-83`): decode to last frame with a 60 s deadline and miss thresholds (Android MediaCodec quirk).
- Android-only: JNI, SurfaceTexture, MediaCodec. **Transferable: the facade and its state machine.** AME's current preview video: ffmpeg-next thread, `thread::sleep(1/fps)` pacing, unbounded channel (`file_table/mod.rs:322`), duration never assigned so the scrub slider never renders (`preview_pane.rs:252`, `file_table/mod.rs:297,367`).

### 3.3 Metadata

- `media_meta::embedded_workflow` (`media_meta.rs:30`): scans raw bytes for ComfyUI markers (`class_type|last_node_id|widgets_values`) because PNG `tEXt`, MP4 `ilst` and Matroska `SimpleTag` all store contiguous JSON text. Bounds: 8 runs, 64 `{` starts per run, 4 MiB per run, depth 6 (16-25). Prefers UI-format graph over API-format. Tests with synthetic PNG/MP4/WebM (149-187). **For AME use real chunk parsing instead** (PNG text chunks, MP4 boxes); the byte-scan exists because Android has no demuxer in the pure-Rust path.
- Graph interpretation `ImageMeta` (`gallery.rs:96-117`) via `parse_workflow_meta_for(raw, filename)` (277): handles API and UI formats, multi-SaveImage workflows by walking back from the SaveImage whose prefix matches the file (339-410), picks KSampler over FaceDetailer (417), follows reroutes/passthroughs for prompt text (958-1140). This is ComfyUI-specific; the **shape** (a typed summary struct + tolerant parsers + tests) is the pattern.
- **Two-stage metadata**: server's quick summary arrives first (`item_meta_from_gate` `engine.rs:2617`) and is flagged `meta_partial`; the full scrape replaces it; actions that need exactness (Remix) wait (`app.rs:558-559, 26220-26227`).
- `remix_diff_rows(meta, params)` (`gallery.rs:211-254`) lists only fields whose incoming value differs from current (`RemixField` toggles 150-160); applying is per-field (`RemixApply` `app.rs:637`).
- A1111-style `parameters` PNG text is **not** handled (plain text, not JSON).

### 3.4 CLIP / tag indexes, co-occurrence, search

**Stored data**

| Index | Struct | Persistence | Key facts |
| --- | --- | --- | --- |
| CLIP | `ClipIndex{entries, by_key}` `clip_index.rs:17-21` | Binary `CIDX` v1: `u32 klen, key, u32 dim, f32*dim, f32 score(NaN=None)` (1-3, 73-87). Truncated tail yields parsed prefix (90-121, test 281) | Vectors L2-normalized so dot = cosine; optional LAION aesthetic score = `dot(w,emb)+bias` (`EM local-clip embed.rs:44`); ~2.1 KB/entry at 512-d (`clip_index.rs:74`) |
| Tags | `TagIndex{HashMap<key,TagEntry{version,general,character,rating}>}` `tag_index.rs:49` | JSON, whole-file rewrite (`app.rs:6655-6681`) | `contains()` is version-gated so stale entries re-tag (66); `top_tags(keys,n)` counts a tag once per image (85-101); `is_nsfw` from WD14 rating (104) |
| Co-occurrence | `CoocModel{tags: HashMap<tag,{total, neighbors}>}` `cooc.rs:27` | JSON, small (`app.rs:6570`) | Caps: 4000 tags, 200 neighbors (10-12); score `raw/(total+1)` damps ubiquitous tags (62-70); `suggest` 82, `rerank_boost` 110, `blend_rank` 149 |
| Dictionary | `TagDict` `tags.rs:24` | Bundled `assets/tags/danbooru.csv.gz` (740,878 bytes), parse ~50 ms (131) | Prefix suggest via `partition_point` on folded names (142), alias -> canonical, category 0/1/3/4/5, bounded-Levenshtein `close_match` (227) |
| Axis classifier | `classify_axis` `appearance.rs:365` | Static tables | Anime-booru vocabulary (hair/eyes/nails/outfit/pose/camera/env). Machinery is generic, vocabulary is not |

**Background pumps** (`pump_clipemb` `app.rs:27617-27752`, `pump_autotag` 27547-27612; both polled every frame from `update()` 27969-27978):

1. Gate: pack present, index loaded, not generating, not the other pump (they share one accelerator) (27621-27630).
2. Coverage latch: `(gallery_gen, listing len, index len, failed len)` stored when a scan finds no work; while unchanged the O(N) scan is skipped (27633-27644).
3. Pick next un-indexed key from the loaded listing; else fall back to a **worker-thread walk** of cache sidecars filling a queue, with `clipemb_rescan_after` back-off (27656-27708). If still nothing and the server has more pages, page the gallery to cover the index (27713-27725).
4. Single flight (`pending`/`rx`), cached bytes read on the worker, not the UI (27774-27794). Failures go to `clipemb_failed`.
5. Drain: `poll_clipemb` (27798) inserts, bumps `gallery_dep_epoch` (invalidates the memo), increments `dirty`; saves every 8 (27453, 27808) via `save_clip_index` (6750): clone index, write `.tmp`, rename, `AtomicBool` in-flight guard.
6. User controls: "Resume CLIP index" (clears pending/failed/walk) and "Rebuild" (deletes file) (23245-23270); progress "N / M embedded" with a "looks stuck" hint (23279-23286).

**Query path** (all verified):

- Toggle `gallery_semantic` (default `true`, `app.rs:2264`; persisted 5634) in the View menu (22302). `gallery_semantic_active()` = toggle && index non-empty && pack has text tower + tokenizer (22786-22793). When active, the server text query is suppressed (22799-22805).
- Enter -> `start_clip_search` (23473): worker thread embeds text; `poll_clip_search` (23497) does `rank_candidates(&emb, &index, &{}, 0.15)`, keeps 200, drops stale results if the box changed (23509), sets `RankedGallery::Semantic(keys)` and a status line "N matches for q (x in view, y indexed)" (23524-23529).
- "More like this" from viewer = `top_similar(key, 60)` (26517); from multi-select = `character_centroid(selection)` -> `rank_candidates(.., 0.1)` -> 120 (24301-24313). Banner "Similar images [Show all]" (23951-23958).
- Ranked view overrides filters and grouping (23690-23694, 23939-23942); Back clears it (22395, 22385-22397).
- Sort by aesthetic score (`GallerySort::Score` 23727-23736).

**Tag browsing UI** (`tags_window` 23409-23468): modal window; two text boxes (narrow the grid by tag text / filter the list), "N tags selected" + Clear, list of `(tag, count)` for the top 400 tags over the **loaded listing**, tap toggles a facet, facets are AND (23715). Button disabled until something is tagged (22246-22261). Weakness: key list + `top_tags` rebuilt every frame while open (23448-23449).

**Characters as smart collections** (`character.rs`, `app.rs:17820,19002-19065`): seeds (LoRA-name matches, portrait) -> `character_centroid` (cached 19002) -> `find_character_images` ranks by cos >= 0.55 (`CHARACTER_MATCH_COS` 590) -> swipe review deck (`Triage` 599; accept = album-add + remember approved; deny = remember denied; `commit_triage` 25409) -> centroid sharpens -> `rescan_suggestions` (19051). New embeddings are scored online against every collection; >= 0.62 become *pending suggestions*, never silent moves (19015-19046, cap 200).

### 3.5 History, backup, settings

- `history.rs`: debounced snapshot undo for the graph editor. Hash the whole graph each frame (`fingerprint` 171, cheap, no alloc), commit a clone only after the hash is stable for 0.4 s and no drag is active (88-106). Caps: 24 entries, 24 MB (22-25). Undo during the settle window counts as its own step (136-148). 11 tests (264-445). Pattern is generic; the content is Snarl-specific.
- `backup.rs`: `CMFYBK01 || salt16 || nonce12 || AES-256-GCM(settings JSON)`, key = Argon2id (19,456 KiB, t=2, p=1) (`backup.rs:9-50`); passphrase >= 8; `list_backups` over several dirs (86).
- Settings I/O (`app.rs:5835-5913`): load tries candidate paths; an unreadable or unparsable file sets `settings_write_blocked` and **autosave refuses to overwrite it** (5873-5879); autosave runs at most once/second, serializes, compares to `last_saved`, writes only on change, dual-writes an internal file and an external mirror (5884-5909). Comment at 24445: serialization order must be stable or the file is rewritten every second.
- Prompt history: `PromptHist{positive,negative}` capped at 60, skip exact repeats (`types.rs:1407-1427`).

### 3.6 Monitoring and logging

- `sysmon::Sampler::tick()` (`sysmon.rs:45`): 1 Hz limit; reads `/proc/self/stat`, `/proc/self/task/*/stat`, `/proc/self/status` (VmRSS), `/proc/meminfo`; folds same-name threads (tokio workers) into one row; guards: ignore deltas < 0.25 s (USER_HZ quantization), reseed if > 10 s stale (62-65). Pure parsers with tests (137-254). GPU = Adreno `kgsl` sysfs only (108). `thread_cpu_ms()` via `CLOCK_THREAD_CPUTIME_ID` (128) separates "our code was slow" from "scheduler preempted us".
- `FrameProf` (`app.rs:995-1078`): per-frame wall ms + thread CPU ms; warns at >= 32 ms with attribution `msgs + bg[section name, ms] + ui` and the `ui_detail` string; throttled 0.5 s; 5 s baseline "frames: N in Ns (x/s), avg, worst". `BgLap` (1082-1100) names the slowest background section. `end_frame` (28361).
- HUD (`perf_overlay_window` 29145-29221): cpu%, gpu%, ui avg/worst, rss, free mem, top 4 thread groups, `active_tasks()` (29095). Tap to minimise.
- `Logger` (`logger.rs`): 2000-line ring buffer with a sequence cursor `take_new(&mut cursor)`, mirrors to logcat in 3800-byte chunks (96-109). AME already has `egui_logger`; not worth porting.

### 3.7 Frost / glass (cosmetic; defer to phase 2)

`frost.rs`: real backdrop blur via `backdrop_blur_egui::GrabPassRenderer` created from the live `glow` context (72-88). Once per frame it collects visible layer rects of order Foreground/Tooltip minus an opaque-ids list and full-screen scrims (>90% of screen) (98-126), and paints frost into an `Order::Middle` area so it lands *behind* the panes (140-162). Repaint policy `Live` while pointer down/scrolling else `Bounded(400 ms)` (128-136). Requires a translucent `window_fill` (100). Graph nodes use `glass_rects`, capped at 12 (177-216). AME's eframe uses the glow renderer (Cargo features), so it is technically portable. Cost is a framebuffer grab + blur per pane per frame. Decision belongs to the theme phase.

### 3.8 Theme (structure only)

`theme.rs` wraps `egui_mobile_theme`: `ThemeConfig{colors(15 roles), control_height, text_scale 0.8-1.3, spacing_scale 0.75-1.5, rounding<=16, glass}` (`vendor/.../config.rs:66-88`), `Density::{Compact 22pt, Touch 36pt}` (`vendor/.../lib.rs:12-13`), semantic `Role` colors read with `egui_mobile_theme::color(ctx, Role)` and `styled(ui, Role, ..)`. Desktop density already exists, so phase 2 can reuse it. `bar_height`/`bar_button` force equal heights in a bar (`theme.rs:61-75`). Icon set: `icon_set!` macro -> `include_bytes!` SVG -> `bytes://icons/x.svg`, tinted at paint time (`glyphs.rs:7-85`), guarded by `every_icon_renders` test (222). Tag-category chip fills: `tag_category_fill` (`theme.rs:238`).

### 3.9 Preflight, lint, errors

- `preflight::validate(wf, schemas) -> Vec<Problem{node,class,input,kind}>` (198). Policy encoded in comments: judge only model-file enums because "a false block is worse than a rare server-side rejection" (214-217); unknown classes skipped. Auto-repairs return human notes (`snap_installed_enums` 68, `retype_combo_values` 97, `settle_dynamic_combos` 144).
- `lint::lint(params, ckpt, loras) -> Vec<LintIssue{severity, msg, fix: Option<Fix>}>` where `Fix` is a **whole-field replacement** so applying is one assignment (`lint.rs:15-28`).
- UI: blocking `preflight_window` with per-row "Fix" that jumps to the node (`app.rs:28917-28988`).
- `report_error` folds identical repeats into a counter, computes `error_fix_for(detail)` from known error text, and the modal offers Copy / Go to Logs / Dismiss / the fix (28417-28550). Pattern: map a recognized failure to a one-tap remedy that also pins the corrected setting.

### 3.10 ZeroClaw

**CA `zeroclaw.rs` (verified, 406 lines):**

| Aspect | Behaviour | Line |
| --- | --- | --- |
| Endpoint | `ws(s)://host[:42617]/ws/chat?agent=<a>&token=<t>`; token percent-encoded in the query string "the only one of the gateway's three accepted places that survives a plain websocket handshake"; scheme never auto-switched; `url_for_log` redacts token | 84-98, 25-26 |
| Handshake | send `{"type":"connect","device_name":..,"capabilities":["text"]}` then `{"type":"message","content":..}` | 169-181 |
| Server frames used | `chunk` -> delta, `done.full_response` -> authoritative final, `error{code,message}`, `aborted`; `thinking`, `session_start`, `connected`, tool frames -> ignored; unknown types ignored | 127-140 |
| Sessions | none: "Sessions are per request" (module doc) | 5-6 |
| Tool use | none (frames ignored); no approval handling | 121-123 |
| Streaming | deltas -> `TurnEvent::Delta`; on `Done` if `full != streamed` the UI gets `ExpandMsg::Replace` | 211-222, `engine.rs:894-905` |
| Timeouts / cancel | connect 15 s; whole turn 120 s; cancel `AtomicBool` polled by racing a 250 ms sleep against `read.next()`; mid-turn close counts as success if text arrived | 30-33, 190-210 |
| Errors | strings prefixed `ZeroClaw:`; no typed errors, retry, or heartbeat | 165-209 |
| Prompting | "Reply with ONLY ..." instructions, JSON variation shape, `json_object()` tolerant of fences/preambles | 232-274 |
| Config import | `from_zc_codex_config` reads host/token of the selected ZeroClaw machine, deliberately not its agent | 279-297 |

**ZC `android/src/zeroclaw.rs` (the fuller client; outside the brief, read lines 1-240):**

| Aspect | Behaviour |
| --- | --- |
| Resume | `&session_id=` query param is the only spelling the gateway resumes on; `connect.session_id` and `?session=` are accepted then ignored (mint a new session) (17-18, 70-75). Waits up to 20 s for `session_start`, errors if the resumed id differs (130-148) |
| Frames out | `connect`, `message`, `approval_response{request_id, decision: approve\|always\|deny}` (15-20, 225-233) |
| Frames in | `session_start connected chunk thinking tool_call tool_result plan agent_start agent_end approval_request done aborted error turn_resume{frames}` (21-23) |
| Liveness | 20 s WebSocket ping from the writer task, 15 s bounded writes with ack, no inbound-silence timeout (long tool calls are silent) (99-127, 157) |
| Images | no attachment field: `[IMAGE:data:<mime>;base64,...]` marker inside `content` (208-222) |
| REST | `/api/sessions`, `/api/sessions/{id}/media`, `/jobs`, `/experiments`, `/running` (423-610) |

**Recommendation:** AME's assistant should be built on ZC's client semantics (events, approvals, resume, ping, image markers). CA's client is the right size for single-turn calls (Refine rewrite, Image Edit prompt expand). Define one `AssistantBackend` trait now (phase 4 swaps default to ZeroClaw). Open: should the gateway call AME's own MCP server (`AME ai/mcp/mod.rs`) as tools? (§9 Q6).

### 3.11 Other reusable pieces

- `mask.rs` (393 lines, pure, tested): `MaskCanvas` (u8 alpha, feathered stamped discs at radius/2 spacing, paint=max, erase=min) 6-78; replayable `StrokeRec` list, undo = rasterize minus last gesture (97-114); `ViewXform` zoom/pan with `pinch(...)` keeping the focus point fixed (143-200); `bake_alpha` writes alpha = 255 - mask into a PNG (225). Android-only: `PointerKind`/`accept_paint` stylus gating (203-222).
- Inpaint UI keeps telemetry fields behind `show_debug` (`app.rs:798-832`): matches the user's "debug panel for every feature" preference.
- `apps.rs` + `apps_builtin/*.json`: composable post-process steps as data (`AppDef{id,version,requires,knobs,overrides,nodes,output}` `apps.rs:32-54`): missing requirement shown as chip with pack name; `overrides` applied as a layer over params "so removing the step reverts by itself" (47-50). ComfyUI graph fragments do not transfer; the data-driven *recipe + knobs* idea might for Refine/Edit presets (low priority).
- `local_engine.rs`: one `run_lock()` serializes accelerator jobs, resident-model slots with explicit `drop_*_cache()` before loading another family, shared utility stack kept warm (`186,241-289`). AME already has `unload_heavy_models_except` (`AME ai/mod.rs:52`); the indexer must respect it.
- `tests/gpu-player`: separate smoke APK reusing `src/player.rs`, prints `COMFY_GPU_SMOKE PASS` to logcat after checking prepare, metadata, paused seek, clock advance, resume, audio, loop (`README.md`). AME equivalent: a `--selftest` CLI that exercises thumbnail decode, CLIP embed, KNN, preview decode on fixtures and prints PASS/FAIL.
- `apps_builtin`, `uiwf`, `workflow`, `schema`: schema layer parses each node type independently so one nonconforming custom node cannot fail the catalog (`README.md:14-20`). Principle for any third-party list AME parses (OpenRouter model list, HF repo listings).

---

## 4. Pattern catalogue

Verdict: **Adopt** (port nearly as-is), **Adapt** (keep idea, change mechanism), **Skip**.
Size: S <= 0.5 d / <=150 LOC; M 1-3 d / 150-600 LOC; L 3-8 d / 600-1500 LOC; XL > 8 d. Rough, one developer who knows AME, excluding Windows polish.

| ID | Pattern | Source | Why | AME target | Verdict | Size |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | Byte-budget `ThumbCache` + on-screen second chance + `claim` | `gallery.rs:1746-1954` (+tests 2782-3014) | Bounds VRAM/RAM, no refetch loops | Replace `viewer.thumb_cache` + `bytes://` path (`AME table/mod.rs:230-290,396-430`, `preview_pane.rs`, `assistant.rs:49`, `image_edit.rs:15`) | Adopt | S |
| P2 | Decode off UI thread, bounded workers, disk-hit fast path | `engine.rs:2335-2390,4454` | Removes thread-per-image + base64 on UI | New `ThumbService` (rayon/tokio blocking pool) | Adapt | M |
| P3 | Size tiers in the key (`{id}#{px}`) | `types.rs:2807,3547` | Zoom slider re-requests cleanly | Grid tile-size slider | Adopt | S |
| P4 | Manual fixed-row virtualization with two spacers | `app.rs:24724-24740,24855` | Off-screen rows cost zero | New Grid view | Adopt | S |
| P5 | Per-frame claim budget + repaint fallback | `app.rs:24744,24793` | Fast scroll does not flood decode | Grid | Adopt | S |
| P6 | Memoized visible set behind a signature hash | `app.rs:23657-23738,23935` | Kills O(N) per frame | Grid + ranked tabs | Adopt | S |
| P7 | `enum Act` / `CoverWants` deferred mutation | `app.rs:25866,1105` | Avoids borrow tangles in immediate mode; replaces global `Mutex<Vec>` queues (`AME app.rs:12-14`, `assistant.rs:16`, `refine.rs:41`) | Panel convention | Adopt | S |
| P8 | Idle-gated single-flight pump + coverage latch + failed set + dirty batch | `app.rs:27547-27820` | Quiet when done; recoverable | Background `Indexer` | Adapt | L |
| P9 | Index save: clone-free, off-thread, tmp+rename, in-flight guard | `app.rs:6750-6775` | No torn files, no UI stall | N/A with DB; keep for any sidecar file (cooc, settings) | Adopt idea | S |
| P10 | Pure embedding math: normalized dot, centroid, `rank_candidates(min_cos)`, aesthetic head | `clip_index.rs:51-174`, `EM local-clip embed.rs:34-46` | Testable ranking core | `ai/embed_math.rs` | Adopt | S |
| P11 | `RankedGallery` overlay + Show-all banner + stale-result guard | `app.rs:568,23473-23544,23951` | One mental model for search/similar/category | Result tabs (`FilterRequest::NewTab` already carries `showing_similarity`, `similar_scores`, `origin_path`: `AME preview_pane.rs:139-146`, `file_table/mod.rs:41`) | Adapt | M |
| P12 | Centroid smart collections + review deck + denied memory + online suggest | `app.rs:17820,19002-19065,25356-25500` | Categories by example | "Collections" | Adapt | L |
| P13 | Tag facets with counts + Tags window | `tag_index.rs:85-101`, `app.rs:23409-23468` | Faceted browse | DB-backed facet panel | Adapt | M |
| P14 | Co-occurrence suggestions + `blend_rank` | `cooc.rs` | Related tags row; tag suggestions in Refine | In-memory model, persisted blob | Adapt | S-M |
| P15 | `TagDict` (aliases, categories, counts, `close_match`) | `tags.rs` | Normalise/merge tags | Only if booru tags exist (§9 Q3) | Adapt | S |
| P16 | Folder cards with fanned cover stack | `app.rs:22476-22626` | Browse categories/tags/collections visually | Browse panel | Adopt | M |
| P17 | Two-stage metadata with `partial` flag | `app.rs:558,26220`; `engine.rs:2617` | Show something instantly, gate exact actions | Preview pane (DB fields first, EXIF/gen-info later) | Adopt idea | S |
| P18 | Field-level diff rows with toggles | `gallery.rs:148-254` | Accept per field | Refine panel | Adopt | M |
| P19 | Streaming review modal (`Delta/Replace/Done/Error`) | `engine.rs:878-913`, `app.rs:7121-7164` | Rewrites stream into a diff | Refine, Image Edit prompt expand | Adopt | M |
| P20 | ZeroClaw one-shot turn | `zeroclaw.rs:153-225` | Utility LLM calls | Refine/Edit | Adopt | S |
| P21 | `Problem`/`Fix` preflight + whole-field `Fix` | `preflight.rs`, `lint.rs:15-28` | Fail early with a remedy | Pre-run checks for AI jobs | Adapt | M |
| P22 | Error modal: fold repeats, known-error -> one-tap fix | `app.rs:28417-28550` | Less log-diving | Global error surface (today `egui_toast` strings) | Adopt | S |
| P23 | `FrameProf` + `BgLap` + `ui_detail` | `app.rs:995-1100,28361` | Attributes jank to a phase | `App::update` wrapper | Adopt | S |
| P24 | `sysmon` (procfs) + HUD | `sysmon.rs`, `app.rs:29145` | Linux parity for CPU/RSS/threads/mem | Merge with `windows.rs`; add NVML for GPU (not in CA) | Adapt | S-M |
| P25 | Settings autosave diff + corrupt guard + mirror | `app.rs:5835-5913` | Prevents silent data loss | `UiSettings` | Adopt | S |
| P26 | Encrypted export (AES-GCM + Argon2id) | `backup.rs` | Safe key export | Settings export | Adopt | S |
| P27 | Viewer anchor-zoom/pan math | `app.rs:25138-25194` | Correct focal zoom | Preview/viewer | Adopt | S |
| P28 | `MaskCanvas` + strokes + `ViewXform` + `bake_alpha` | `mask.rs` | Inpaint masks | Image Edit | Adopt | M |
| P29 | `Player` facade + pause-when-hidden | `player.rs`, `app.rs:27936` | Stable playback state machine | `VideoBackend` trait | Adapt | L (backend) |
| P30 | Cache hygiene: atomic write, budget+reserve, part reaping | `gallery.rs:1275-1293,1599,1640` | Safe disk thumb cache | Thumbnail disk cache | Adapt | M |
| P31 | `overlay_blocking()` guard for raw-pointer gestures | `app.rs:29381` | Gestures ignore content under a modal | Marquee select, viewer pan | Adopt | S |
| P32 | Debounced snapshot undo | `history.rs` | Undo for multi-field edits | Image Edit params; Refine bulk apply uses an op journal instead | Adapt | M |
| P33 | "Unseen" set + badge | `app.rs:24826,29262` | "New since last visit" | Explorer / Quick Access | Adapt | S |
| P34 | Self-test harness | `tests/gpu-player/README.md` | Verifiable features | `--selftest` / devtools page | Adapt | S-M |
| P35 | Tolerant JSON extraction | `zeroclaw.rs:270`, `schema.rs` | LLM answers with fences/preambles | AME already has `parse_jsonish` (`refine.rs:524`); reuse the CA tests | Adopt | S |
| P36 | Edit-distance "did you mean" | `tags.rs:197-265`, `lint.rs:233-276` | Typos in tags/search | Search box suggestions | Adapt | S |
| P37 | Resident-model slot + run lock | `local_engine.rs:186,241-289` | Single accelerator discipline | Keep AME's `unload_heavy_models_except` | Skip (exists) | - |
| P38 | Data-driven recipes with knobs | `apps.rs` | Presets as files | Maybe for Refine presets | Skip for now | - |

---

## 5. What does not transfer

| Area | Where | Why not |
| --- | --- | --- |
| Pull-to-refresh, swipe-to-change, double-tap-to-open, long-press paint-select, haptics | `gallery.rs:14-82`, `app.rs:24632,24871,25217` | Touch. Desktop equivalents: F5, arrow keys, double-click/Enter, marquee + Shift/Ctrl ranges |
| 1-3 column toggle | `types.rs:2807` | Desktop wants a continuous tile-size slider (64-512 px) |
| Bottom bars that collapse for the IME, safe-area insets, `TEXT_ACTIONS_BAR_H`, Back-key ladder | `app.rs:92,28008-28163` | Android. The ladder is also the god-object smell (§8): use a modal stack |
| `up_menu` open-upward popups | `theme.rs:329` | Exists to dodge the Android nav bar |
| `.nomedia`, MediaScan, `statvfs` over FUSE, durable `/sdcard` root | `gallery.rs:1184-1198,1578` | Android storage |
| `SurfacePlayer`, MediaCodec, JNI, GL external-OES texture | `EM surface.rs` | Android. Desktop backend is a new decision (§7 Preview) |
| QNN/HTP NPU packs, `local-*` crates | `local_engine.rs`, `Cargo.toml` | Android NPU. AME uses candle CUDA / fastembed |
| Self-update, PackageInstaller, notification progress | `app.rs:28384` | Android |
| comfy-gate APIs: albums, facets, trash, tombstones, LoRA library, Civitai download | `gallery.rs:1355-1381`, `app.rs:24351-24473` | Server-specific. AME deletes local files; its trash/undo is a separate design |
| ComfyUI graph editor, workflow conversion, H3 grammar | `graphview.rs`, `uiwf.rs`, `workflow.rs`, `h3.rs` | Domain |
| Anime-booru look-axis vocabularies, character cards, NSFW rating filter | `appearance.rs`, `character.rs`, `types.rs:2722` | Domain; reuse only if WD14 tagging is adopted |
| Frost/glass | `frost.rs` | Cosmetic; phase 2 |

---

## 6. Design: semantic search + tag/category browsing for AME

### 6.1 What AME has today (verified) and the gaps

| Capability | Where | Gap |
| --- | --- | --- |
| Metadata fields `tags: Vec<String>`, `category`, `caption`, `description`, `hash`, `thumbnail_b64` | `AME database/mod.rs:60-73` | No taxonomy/normalisation; thumbnails stored as base64 strings |
| Indexes `tags_idx`, `category_idx` | `database/mod.rs:146-147` | No facet counts or browse UI |
| CLIP embeddings table + HNSW cosine | `database/mod.rs:75-82,153` | **`DIMENSION 1024` fixed**; settings allow 512-d models (`clip-vit-b32`, `unicom-vit-b32`) and 1024-d SigLIP (`ai/clip/mod.rs:13-35,62`). Mixed dims cannot share one index (unverified at runtime) |
| KNN | `database/clip_embeddings.rs:39-69` | Query hard-codes `<| 64, 64 |>`; `k`/`ef` args are not used by the operator, so results cap at 64 and `START` paging past 64 returns nothing (`file_table/mod.rs:1006-1023` pages with `start`). Unverified at runtime; SurrealDB 3 semantics not tested |
| Search entry | "Search for files" field (`menus/mod.rs:168-170`) + "AI" checkbox in DB mode (`menus/mod.rs:62`); the AI path embeds the text, calls `find_similar_by_embedding(q, batch*2, 256, offset)`, then **filters by type/size after the KNN** and does one `load_clip_embeddings_for_path` DB read per hit (`menus/mod.rs:66-125`; type/size filters at 113-124); results replace the main table (`showing_similarity`, origin `query:<text>`) with a "Load N more" button (`menus/mod.rs:181`) | Triggered by Enter on the *path* box or "Apply Filter", not on the field itself; post-filtering can leave a page nearly empty; per-hit DB reads |
| Text -> image search (other paths) | `file_table/mod.rs:1004-1067` (`append_more_similar`), `ai_search/generate.rs:28-61` | `clip_search_text/image` loop every in-memory file and do a **DB round trip per file** (`generate.rs:50-58,85-93`); `AISearchEngine::search` is substring counting and fetches an embedding per hit then discards it (`search.rs:25-27`) |
| Results UX | Either the current table (`showing_similarity`) or `FilterRequest::NewTab{title, rows, showing_similarity, similar_scores, origin_path}` (`preview_pane.rs:139-146`) | Ranked results with a score column already exist; they are not a first-class view and clobber the table in the search-field path |
| Indexing | `ensure_index_worker` mpsc + atomics (`ai_search/index.rs:161-196`), flags `auto_clip_enabled`/`auto_descriptions_enabled` | Controls scattered in a nav menu (`navbar.rs:108-170`); status chip exists (`status.rs`) |
| Embedding engines | fastembed CLIP, SigLIP via candle (`siglip_logits_image_vs_texts` `ai/clip/mod.rs:39`) | Zero-shot labels possible already |

### 6.2 What the user sees

1. **Omnibox** at the top of the Explorer tab (focus with Ctrl+F / Ctrl+K) replacing the trio "Search for files" + "AI" checkbox + "Apply Filter" and firing on its own Enter. One field, three behaviours, shown as removable chips:
   - plain words -> *Meaning* search (embedding) merged with *Text* match on filename/caption/tags/category;
   - `tag:sunset`, `cat:landscape`, `type:video`, `in:<folder>`, `rating>=…` tokens -> structured filters;
   - drop an image or "More like this" -> *Similar*.
   Suggestions while typing: tags by prefix with counts (P15), "did you mean" (P36), related tags from co-occurrence (P14).
2. **Result = a tab** (reuse the `NewTab` model; today the search field instead overwrites the current table) titled `Search: red car`, with a banner `Meaning search · 183 results · cutoff 0.21 [Strictness ▾] [Clear]`, optional score badge on tiles, Grid/Details toggle, selection actions identical to the main tab. Facet bar above the grid shows tags/categories **with counts for the current result set**.
3. **Browse** section in Quick Access (or a "Library" dock tab) with three sub-tabs:
   - *Categories*: cards with fanned cover stack (P16) + count; includes `Uncategorized` and `Needs review (low confidence)`.
   - *Tags*: filterable list/cloud with counts, multi-select AND/OR, "related tags" row, right-click rename/merge/delete across the library (AME already has `ref_old_tag/ref_new_tag/ref_delete_tag` state, `AME app.rs:67-71`).
   - *Collections*: smart collections defined by examples (P12) with a "N suggestions to review" badge.
4. **Index health** in the status bar: `Embeddings 12,430 / 18,902 · 14/s · ETA 7 min · 3 failed` with hover card actions Pause / Resume / Retry failed / Rebuild / Change model. Auto-pauses with the reason shown ("paused: Qwen edit running").
5. **Devtools page** (user preference for debug UIs): search debugger (query vector norm, top-50 with raw cosine/distance, timing split embed/search/hydrate, active model/dim/count, HNSW params, failed list with errors, coverage signature) plus `smart-media --search "text" --json` and `--index-status` CLI.

Wireframe (Explorer tab, results state):

```
[<][>][^][home] D:\Photos\2025                  [ red car at night      x ] [Meaning|Text] [Strictness v] [Grid|Details] [Preview]
 facets:  tag: night (212)  car (190)  street (88)  | cat: Vehicles (160)  Cityscape (41)   related: neon, rain, taxi
 banner:  Meaning search · 183 results · cutoff 0.21 · embed 9 ms · search 14 ms        [Clear]  [Save as collection]
+---------+---------+---------+---------+---------+---------+          (virtualized grid, ThumbCache, tile-size slider)
| thumb   | thumb   | thumb   | thumb   | thumb   | thumb   |          badges: video, indexed, score on hover
+---------+---------+---------+---------+---------+---------+
status:  DB ok | CLIP 12,430/18,902 14/s ETA 7m [pause] | VISION idle | RERANK idle | cpu 23% ram 9.1G vram 6.2/12G
```

### 6.3 Data model (sketch; SurrealQL not run)

```
-- one table per embedding model/dim (HNSW needs DIMENSION at DEFINE time)
emb_siglip1024 { thumb: record<thumbnails>, vec: array<float>, aesthetic: option<float>,
                 src_hash: string, ver: int, created, updated }   -- vec L2-normalized
  UNIQUE (thumb); HNSW(vec, DIMENSION 1024, COSINE)
emb_clip512    { ... same, DIMENSION 512 }

thumbnails (existing) + clip_ver: option<int>, tags_ver: option<int>, caption_ver: option<int>   -- per-stage watermarks, indexed
                     + category_auto: option<{id: string, score: float}>   -- never overwrite AI/user `category`
tag        { id: folded name, display, count, axis: option<string> }       -- maintained incrementally
taxonomy   { id, label, prompts: array<string>, threshold, sort }          -- zero-shot categories
collection { id, name, model, seeds: array<record>, denied: array<record>, centroid: array<float>, min_cos, pending: array<record> }
```

- `src_hash` (blake3, AME already has `blake3` + `hash`) lets the indexer skip unchanged files like `index_file_internal` already does (`ai_search/index.rs:33-76`).
- `ver` is the preprocessing/schema version, mirroring `TagIndex::contains` version gating (`tag_index.rs:66`) and `ClipIndex::SCHEMA_VERSION` (`clip_index.rs:8`).
- Cooc stays an in-memory model persisted as one small blob (caps keep it ~4000 x 200 entries, `cooc.rs:10-12`).

### 6.4 Background indexing

- Stages with watermarks: `thumb -> hash -> embed -> tags/caption -> category_auto`. Each stage idempotent on `(src_hash, model, ver)`.
- One scheduler owns: a priority queue (visible window first, mirroring `update_prefetch_window` `app.rs:23303`; then the open folder; then the rest); a GPU lane (semaphore) honouring `unload_heavy_models_except`; **pause rules** equal to the CA gate `!running && !wd14_running` (`app.rs:27625`): do not embed while Qwen edit / local JoyCaption / local assistant owns VRAM; resume automatically.
- **Batch embeds** (16-64 images per forward) on CUDA. CA is strictly one image at a time because it has one NPU session (`app.rs:27544`; `local_engine.rs:186,863-864` run lock, "one HTP session at a time"); do not copy that throughput limit. (Throughput gain is expected, not measured.)
- Persistence: batch writes of 32-128 rows per transaction; never clone the whole index on the UI thread (CA does, `app.rs:6664,6759`).
- Failure handling: per-item failure record with reason; "Retry failed" button (CA: `clipemb_failed`, 23245-23258).
- Progress: `CLIP_STATUS.set_progress(done,total)` + rate + ETA; counters feed the HUD.
- Startup: stream existing vectors into the in-memory matrix on a worker; the UI never waits (CA: `ensure_clip_index_warm` 6695).

### 6.5 Query path

1. UI thread: omnibox changes -> debounce 250 ms for Meaning (Enter commits immediately). Build `SearchRequest{id, text, filters, mode}`.
2. `SearchService` task (tokio) owns the embedding engines and a read-only view of vectors:
   a. parse tokens; structured filters -> DB `WHERE` (tags/category/type/path).
   b. Meaning: `embed_text` (GPU, ms) -> vector.
   c. Rank: either SurrealDB HNSW with a **parametrized K** and oversampling when filters are present (HNSW filters are post-filters; unverified), or an in-memory exact scan over a contiguous `Vec<f32>`/f16 matrix with `rayon` (100k x 1024 f32 ~ 400 MB; estimate: tens of ms on 8 cores, not measured; GPU matmul via candle is the next step if it matters). Recommended: HNSW for default, exact path behind a devtools toggle so the two can be compared on real data.
   d. Cutoff: **relative** (`score >= max(min_abs, top1 - delta)`) instead of CA's fixed `0.15` (`app.rs:23504`), because cosine ranges differ per model; expose a Strictness slider and show the score histogram in devtools.
   e. Optional fuse with text hits by reciprocal rank.
3. Reply `{request_id, hits: Vec<(ThumbId, score)>, timings}` on a channel; UI drops stale ids (CA: `app.rs:23509`).
4. UI opens/updates the result tab; rows hydrated lazily by page; counts for the facet bar computed in the service and memoized by signature (CA P6).
5. Similar: centroid of the selection (`character_centroid`) -> same rank step; exclude the selection.
6. Collections: `rank_candidates(centroid, min_cos)`; online scoring of new vectors against each collection (`suggest_for_new_key`), pending-only, never silent.

Ranked results **should compose with filters** (CA's ranked view ignores them, `app.rs:23690`); AME has the DB to intersect.

### 6.6 Categories vs tags: how they are produced

- *Tags*: JoyCaption / Qwen-VL output already lands in `thumbnails.tags`. Normalise with `fold` (`tags.rs:29`), alias map, optional dictionary (P15). Facet counts from DB aggregate or an inverted index (`HashMap<tag, bitmap>`); `GROUP BY` on array fields in SurrealDB was not tested (unverified), so keep an in-memory facet index as the plan B.
- *Categories*: two sources kept separate: `category` (AI/user text, existing) and `category_auto` (zero-shot). Zero-shot = embed each taxonomy prompt once with the text tower, then one matmul `image_vecs x label_vecs` over stored vectors (no re-reading images). SigLIP/CLIP score scales differ; calibrate per model in devtools.
- *Collections*: user-defined by examples, centroid + review (P12).

### 6.7 Phasing

| Phase | Deliverable | Depends on | Size |
| --- | --- | --- | --- |
| A | Fix KNN K cap, parametrize; per-dim tables/indexes; remove per-file DB loops in `clip_search_*` | none | M |
| B | `ThumbService` + `ThumbCache` + disk thumb cache | none | M |
| C | `Indexer` service + status card + Pause/Resume/Retry/Rebuild + devtools | A | L |
| D | `SearchService` + omnibox + ranked tabs + Strictness | A, C | L |
| E | Facet bar + Browse panel (Categories/Tags) + cover cards | B, D | M |
| F | Collections (centroid, review deck, suggestions) | D | L |
| G | Cooc related tags + did-you-mean | E | S |

### 6.8 Risks and unknowns

- SurrealDB 3 HNSW: literal-only `K`, post-filter behavior, aggregate over array fields, concurrent write cost (unverified; check before committing to A).
- Mixed-dimension migration for existing user data (existing 1024-d rows keep working only if they are 1024-d; check what is stored).
- Zero-shot thresholds are model-specific; needs calibration data.
- Memory of an in-memory matrix at 200k+ images.
- Whether WD14 booru tags are in scope (changes P15/appearance value).

---

## 7. Mapping to AME panels

Sizes use §4 scale. "Port" = adapted from CA; "New" = no CA equivalent.

### Grid (File Explorer table + new Grid view)

AME today: `egui-data-table` with a thumbnail column; cell renders `Image` from `bytes://`, base64-decoding on the UI thread on miss (`AME table/mod.rs:230-290`); per-page thumb scheduling with `tokio::spawn` per file (`file_table/mod.rs:1366-1441`); `prefetch_next_page_thumbs` has a placeholder for in-flight accounting (`1359-1361`).

| Work | Detail | Size |
| --- | --- | --- |
| ThumbCache + service | P1-P3, P30; replace base64-in-DB reads with disk cache keyed by `blake3(path+mtime+size)`; DB keeps only a small pointer | M |
| Grid view | P4-P6, P31, tile-size slider, badges (video, indexed, tag count, unseen), selection (click, Ctrl, Shift, marquee), keyboard nav, Space quick-look; table stays as "Details" | L (new, ~700-900 LOC) |
| Ranked tabs | P11: `RankedSource` enum on `NewTab` + score badge | S-M |
| Facet bar | P13/P16 | M |
| Perf attribution | P23 `ui_detail` for grid | S |

### Preview pane

AME: 400 px right panel, 1600 px PNG generated via base64 round trip, editable description/caption/category/tags, buttons Generate / Find Similar / Find Similar (Engine) / Generate CLIP (`preview_pane.rs:105-535`).

| Work | Detail | Size |
| --- | --- | --- |
| Progressive image | show cached 320/1024 thumb then 1600 px via ThumbCache; P27 zoom/pan in a "Viewer" mode (Enter/double-click) | M |
| Metadata | P17 two-stage; clickable tag chips -> add facet; copy buttons; generation-info section for AI-art files (real PNG chunk parsing, not CA's byte scan) | M |
| Actions | collapse the four AI buttons into an "AI" menu with preflight (P21) and statuses; drop the duplicate engine path | S |
| Video | `VideoBackend` trait mirroring `SurfaceStatus` fields + `Player` state machine (P29). Backend choice: keep `ffmpeg-next` (software, add audio clock + seek) or libmpv render API into an FBO via an egui glow `PaintCallback` (feasibility under eframe unverified) | L (ffmpeg) / XL (libmpv) |
| Filmstrip | virtualized from day one (§8) | S |

### Assistant

AME: sessions/messages in SurrealDB, providers by string match (`assistant.rs:1208-1353`), slash commands bound to MCP tools (`1223-1311`), own `thumb_cache: HashMap<String,TextureHandle>` (49).

| Work | Detail | Size |
| --- | --- | --- |
| `AssistantBackend` trait | `OpenAiCompat`, `LocalCandle`, `ZeroClaw`; common `Event{Delta, Thinking, ToolCall, ToolResult, Approval, Done, Error}` | M |
| ZeroClaw backend | port ZC client semantics (`session_id` resume, ping, approvals, `[IMAGE:..]`); persist gateway `session_id` in `assistant_sessions` | M-L |
| Attachments | ThumbCache for chips; image markers | S |
| Tool results | `/search` and Search tool open a ranked tab (reuse SearchService) | S |
| Error surface | P22 | S |

### Refine

AME: proposals list with `selected` bool, generators Heuristic / Jina M0 / Qwen placeholder / cloud model (`refine.rs:7-61,106-130`); scope via `limit_paths`.

| Work | Detail | Size |
| --- | --- | --- |
| Field-level diff | P18: per-field toggles `category/tags/caption/description`, "Apply selected", op journal for undo | M |
| Streaming per item | P19 deltas + cancel + partial results note | S-M |
| ZeroClaw engine | P20 as another generator beside Jina/cloud; reuse `parse_jsonish` + CA instruction patterns | S |
| Tag intelligence | P14 co-occurrence suggestions, P36 typo fixes, rename/merge UI wired to existing `ref_*` state | M |
| Preflight | reranker model present, key set, VRAM headroom (`windows.rs:60`) | S |

### Quick Access

AME: left panel with Logical Groups, User Directories, Recent paths (`quick_access_pane.rs:8-120`).

| Work | Detail | Size |
| --- | --- | --- |
| Browse | Categories / Tags / Collections cards (P16) | M |
| Index health card | status + actions | S |
| "New since last visit" | P33 | S |

### Status

AME: global `STATUSES` registry with `RegisteredStatus` handles, chips with hover card and optional progress (`status.rs`), CPU/RAM/VRAM bars (`navbar.rs:290-312`, Windows only).

| Work | Detail | Size |
| --- | --- | --- |
| Extend `StatusMeta` | rate, ETA, queue length, paused reason, actions | S |
| Linux monitor | P24: procfs CPU/RSS/threads/mem; GPU via NVML (outside CA) | S-M |
| HUD | P23/P24 toggle in Settings | S |
| Fix | `set_model` `Box::leak`s a new `&'static str` per call (`status.rs:147-153`) | S |
| Self-test | P34 | S-M |

### Image Edit

AME: prompt/negative/strength/steps/guidance, live preview every N steps, cancel flag (`image_edit.rs:9-50,247-275`); no mask.

| Work | Detail | Size |
| --- | --- | --- |
| Mask painter | P28: `mask.rs` ports almost verbatim (drop `PointerKind`); brush UI + `ViewXform` wheel zoom. Whether Qwen-Image-Edit in AME consumes a mask is unknown; fallback = composite edited result over source with the feathered mask (unverified feasibility) | M |
| Prompt expand | P19/P20 | S |
| Remix | open an AI-art file with its embedded prompt filled in (needs PNG text chunk parser) | S-M |

### Roll-up

| Block | Size |
| --- | --- |
| Thumbs + grid + ranked tabs | L + M |
| Indexer + search + browse | L + L + M |
| Collections | L |
| Preview/viewer (excluding video backend) | M |
| Video backend | L-XL |
| Assistant (ZeroClaw) | M-L |
| Refine | M |
| Perf/status/preflight/error | M total |
| Mask editor | M |

---

## 8. Defects and anti-patterns in CA (do not copy) and AME-side findings

| # | Finding | Evidence | Instead |
| --- | --- | --- | --- |
| 1 | 30,926-line `ComfyApp`, Back-key priority ladder spelled out in `update` | `app.rs:28008-28163` | Per-panel structs, modal stack, `Act` enums |
| 2 | `GalleryItem::key()` allocates a `String` on every call; used per tile per frame and inside O(N) loops | `types.rs:3541`, `app.rs:24777,19019` | Interned ids (`u64`/`Arc<str>`) computed once |
| 3 | Filmstrip builds `thumb_key(320)` for **every** item every frame before the visibility check | `app.rs:26876-26891` | Virtualize like the grid |
| 4 | Tags window rebuilds the key list and `top_tags` each frame | `app.rs:23448-23449` | Memoize by signature |
| 5 | `top_similar` clones every key, sorts all, then truncates | `clip_index.rs:54-61` | Score indices, clone top-k only; partial selection |
| 6 | Whole-index `clone()` on the UI thread before each save | `app.rs:6664,6759` | Append-only/DB batches |
| 7 | `suggest_for_new_key` does an O(N) `find` with allocation, and `rescan_suggestions` calls it per key (O(N^2)) | `app.rs:19019,19051-19065` | Id lookup map |
| 8 | Semantic results only intersect the **loaded** listing ("x in view") | `app.rs:23522-23528` | Query the full DB |
| 9 | Ranked view ignores active filters | `app.rs:23690-23694` | Intersect |
| 10 | Secrets (`api_key`, session, `zc_token`) live in plain settings JSON; only the optional export is encrypted | `backup.rs:106-118`, `app.rs:5665-5667` | OS keyring or encrypted store (AME has the same exposure: plain `Option<String>` API keys, `database/settings.rs:42-46`) |
| 11 | ZeroClaw token travels in the WebSocket query string (protocol constraint) | `CA zeroclaw.rs:82-92`, `ZC android/src/zeroclaw.rs:67-69` | Use `wss` for non-tailnet hosts; never log URLs (CA redacts, `zeroclaw.rs:94-98`) |
| 12 | Per-frame polling pumps rely on a latch to avoid O(N) scans | `app.rs:27633` | Event-driven tasks in AME (tokio) |

AME-side findings from reading the target code (not CA issues): A1 egui_extras image loader thread-per-decode and no eviction (`egui_extras-0.37.0 background_decode.rs:62-96`); A2 HNSW dim fixed 1024 and KNN literal 64 (`database/mod.rs:153`, `clip_embeddings.rs:55`); A3 per-file DB round trips in `clip_search_text/image` and `search` (`generate.rs:50-58,85-93`, `search.rs:25-27`); A4 no non-Windows system monitor (`windows.rs:231-232`); A5 `set_model` leak (`status.rs:147-153`); A6 video preview limits (`preview_pane.rs:49-79,252`); A7 `eframe::App::save` stub (`main.rs:97`); A8 `recursive_prefetch_inflight` is set to `candidates.len()` (`file_table/mod.rs:1316`) and never decremented (the raw-pointer decrement is stubbed, 1326, 1359-1361), so the guard at 1274 blocks every prefetch wave after the first (only other writes: init 0 at 438; verified by grep).

---

## 9. Open questions for the user

1. Default embedding model(s): SigLIP2-large (1024-d) only, or also CLIP B/32 (512-d)? Decides table/index layout (§6.3).
2. Library scale target (10k / 100k / 1M)? Decides HNSW vs in-memory exact scan.
3. Is WD14/booru tagging planned for AME, or are tags only from JoyCaption/Qwen-VL captions? (Decides whether `tags.rs`/`appearance.rs` are worth porting.)
4. Move thumbnails out of the DB into a disk cache (and keep DB exports small), or keep base64 in rows?
5. Video: stay on `ffmpeg-next` or accept a libmpv system dependency on Linux and Windows?
6. Should the ZeroClaw agent call AME's MCP server as tools (and which agent name / tool set)? Does the assistant need approval prompts?
7. Keep a rating (SFW/NSFW) filter in the explorer, or drop it?
8. New "Library" dock tab versus extending Quick Access for Browse?

---

## Appendix: CA file index (lines, role)

| File | Lines | Role | Port value |
| --- | --- | --- | --- |
| `gallery.rs` | 3112 | meta parsers, remix diff, caches, ThumbCache, grouping | High (ThumbCache, diff rows, cache hygiene) |
| `app.rs` | 30926 | all UI | Reference only |
| `engine.rs` | 5332 | tokio runtime, HTTP/WS, Msg | Pattern (Engine + Msg) |
| `clip_index.rs` | 292 | embedding store + math | High |
| `tag_index.rs` | 210 | tag store + facets | Medium |
| `cooc.rs` | 286 | co-occurrence | Medium |
| `tags.rs` | 832 | dictionary, chips, typo | Conditional |
| `appearance.rs` | 725 | look-axis classifier | Conditional |
| `character.rs` | 604 | cards from pictures | Low |
| `player.rs` | 184 | video facade | High (API shape) |
| `frost.rs` | 217 | glass | Phase 2 |
| `media_meta.rs` | 325 | embedded JSON scan | Low (use real parsers) |
| `sysmon.rs` | 254 | procfs sampler | High |
| `history.rs` | 460 | debounced snapshot undo | Low-Medium |
| `backup.rs` | 144 | encrypted export | Medium |
| `preflight.rs` | 570 | validators + repairs | Pattern |
| `lint.rs` | 433 | prompt lint with fixes | Pattern |
| `zeroclaw.rs` | 406 | one-shot WS client | Medium |
| `mask.rs` | 393 | inpaint mask | High |
| `theme.rs` / `vendor/egui-mobile-theme` | 1082 / - | theme | Phase 2 |
| `glyphs.rs` / `icons.rs` | 232 / 154 | icons | Phase 2 |
| `logger.rs` | 110 | log ring buffer | Low |
| `apps.rs` + `src/apps_builtin` | 2891 + 8 JSON | data-driven recipes | Low |
| `local_engine.rs` | 1341 | NPU runs, run lock | Reference |
| `graphview.rs, uiwf.rs, workflow.rs, schema.rs, h3.rs, types.rs` | 4048, 1419, 1670, 943, 1784, 4972 | ComfyUI domain | Skip |
| `tests/gpu-player` | - | device smoke test | Pattern |

---

## Appendix B: key snippets (trimmed and lightly paraphrased from CA)

Fixed-row virtualization (`app.rs:24727-24740, 24855-24857`):

```rust
let row_h = tile + spacing_y;
let n_rows = indices.len().div_ceil(cols.max(1));
let top = ui.cursor().top();
let first = ((((clip.top() - top) / row_h).floor()).max(0.0) as usize).min(n_rows);
let last = ((((clip.bottom() - top) / row_h).floor()).max(0.0) as usize + 1).min(n_rows);
if first > 0 { ui.allocate_space(egui::vec2(avail, first as f32 * row_h - spacing_y)); }
// draw rows first..last
if last < n_rows { ui.allocate_space(egui::vec2(avail, (n_rows - last) as f32 * row_h - spacing_y)); }
```

Anchor zoom (`app.rs:25150-25156`):

```rust
let anchor_zoom = |at: egui::Pos2, next: f32, zoom: &mut f32, pan: &mut egui::Vec2| {
    let next = next.clamp(1.0, MAX_ZOOM);
    if (next - *zoom).abs() > f32::EPSILON {
        *pan += (at - slot.center() - *pan) * (1.0 - next / *zoom);
        *zoom = next;
    }
};
```

Coverage latch (`app.rs:27633-27644`):

```rust
let coverage = (self.gallery_gen, self.gallery.len(), self.clip_index.len(), self.clipemb_failed.len());
if self.clipemb_covered == Some(coverage) { /* save if dirty */ return; }
// scan for work; when none is left: self.clipemb_covered = Some(coverage);
```

Whole-field fixes (`lint.rs:15-28`):

```rust
pub enum Fix { SetPositive(String), SetNegative(String), SetLoraTriggers(String) }
pub struct LintIssue { pub severity: Severity, pub msg: String, pub fix: Option<Fix> }
```

Atomic write (`gallery.rs:1275-1293`, shape): create `.part-<pid>-<seq>` beside the target, fill, `sync_all`, `rename` over the target, remove the part on any failure.

Playback facade status (`EM surface.rs:401-412`): `position_ms, duration_ms, width, height, ready, playing, fps, frame_count, has_audio, error`.
