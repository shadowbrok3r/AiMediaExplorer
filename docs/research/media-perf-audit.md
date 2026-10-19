# Media pipeline audit: thumbnails, previews, RAW, archives, video

Repo: `/home/shadowbroker/Documents/Rust/AiMediaExplorer` @ `1d1a1db` (egui/eframe 0.37, glow renderer, image 0.25.10, rawkit 0.1.0, ffmpeg-next 9.0.0, SurrealDB 3.3.4).
Nothing in the repo was modified. All experiments ran in the scratchpad (`.../scratchpad/{ffcheck,ffposter,rawfix,thumbbench,uibench,dbbench,repo-copy}`); `repo-copy` is a source copy used to compile-verify the proposed patches.

Evidence labels used below: **[M]** measured by me, **[R]** reproduced, **[C]** code-evident (read, not run), **[E]** estimate.

---

## 0. Executive summary

The pipeline works for a few thousand files and breaks well before 100k. Root causes, in order of impact:

| # | Root cause | Evidence | Effect |
|---|---|---|---|
| 1 | Thumbnails are stored as base64 PNG strings inside every `Thumbnail` row and that row is cloned 3-6 times (table, `last_scan_rows`, `recursive_filtered_rows`, `db_page_cache`, `db_lookup`) | [C] `thumbs.rs:84-96`, `receive_scan.rs:630-645`, `file_table/mod.rs:1235-1240`, `receive_preload.rs:21-47` | ~0.8 MB RAM per thumbnail [E] (real photo: 77 KB PNG = 103 KB base64 [M]); 100k files = ~80 GB |
| 2 | UI thread does O(N) work per message and per frame: linear `find` per thumbnail, whole-table clone per frame, two full table passes in the status bar | [M] 10.2 s total lookup cost at 100k rows; 0.12-0.87 s per clone at 5k-50k rows; 4-45 ms/frame status bar (release), 10-200 ms/frame (dev) | frame stalls that grow with library size |
| 3 | No bounded caches or eviction anywhere (app `thumb_cache`, egui bytes/image/texture loaders, DB page cache, preview keys) | [C] `thumb_cache` is only pruned in `rebuild_recursive_page` (`mod.rs:1179-1187`); egui loaders have no LRU (`texture_loader.rs:119-135`) | memory grows monotonically with everything ever shown |
| 4 | Dev profile compiles every dependency at opt-level 0; only `target/debug` exists in the repo | [M] 32 MP JPEG decode 10.5 s (dev) vs 0.14 s (release or dev with deps at O3) | thumbnail generation 17-75x slower in daily use |
| 5 | A panic inside rawkit reaches Rayon's abort path and kills the process | [R] exit 134, "Rayon: detected unexpected panic; aborting" on your real `_DSC7596.ARW` | any unsupported Sony ARW crashes the app |
| 6 | Several redundant or missing decode paths: each image is thumbnailed by two independent pipelines, nothing consults a persistent thumbnail cache, nothing is prioritized by viewport, nothing is cancelled on navigation, video thumbnails do not exist on Linux | [C] `scan.rs:435`, `receive_scan.rs:382-387`, `explorer/mod.rs:140-163`, `thumbs.rs:104-109` | 2x CPU, re-decode on every visit, thumbnails for rows nobody sees |

Known defects, both resolved with compile- and run-verified patches (section 4):
- `cargo check --features ffmpeg`: reproduced the 6 errors in the real crate. Fixing them exposes a 7th, previously masked error (`TextureHandle::set` needs `&mut`). Patched copy passes `cargo check` and `cargo check --features ffmpeg`.
- rawkit panic: reproduced. Fix = embedded-JPEG-preview-first decode (13 ms per ARW thumbnail [M]) + `catch_unwind` around rawkit + pool `panic_handler`. Verified on the real ARW, including the forced "no preview" path.

Highest-value quick wins (all independent of the UI reorganization): dev-profile dependency opt-level, panic boundary + RAW preview, O(1) lookups, filter-then-clone, cached status-bar counts, preview-spawn guard, JPEG thumbnails instead of PNG, cancel-on-navigate. Full list in section 6.

---

## 1. Pipeline as built

### 1.1 End-to-end flow

```
FILE (jpg/png/arw/mp4/zip!entry)
  |
  | A. scan thumbs   scan.rs:407-443 -> enqueue_thumb_job -> crossbeam bounded(256) -> 3 rayon workers (THUMB_POOL, 2-4 threads)
  | B. UI-driven     receive_scan.rs:382-387,525-570 -> tokio::spawn per file + Semaphore(6)        (duplicate of A)
  | C. page-level    file_table/mod.rs:1367-1441 schedule_missing_thumbs_for_current_page; :1266-1364 prefetch (dead after 1st wave)
  | D. click         table/mod.rs:560-665 spawn_blocking (FS) / tokio::spawn with sync code (archives)
  | E. archive list explorer/mod.rs:29-91 tokio::spawn per media entry -> extract to temp file -> decode
  | F. preview       preview_pane.rs:173-199 tokio::spawn per frame until cached (1600 px, Lanczos3, PNG)
  | G. image edit    image_edit.rs:455-478 same per-frame spawn; Arc<Mutex<HashMap>> cache
  | H. video         preview_pane.rs:11-103 std::thread, ffmpeg-next (feature "ffmpeg", currently does not compile)
  v
generate_image_thumb_data (thumbs.rs:10-97):  image::open (full decode) -> thumbnail(256,256) -> PNG -> base64 -> "data:image/png;base64,..."
  RAW: raw_decode.rs:3-42  rawkit decode+process_8bit (full demosaic) -> Lanczos3 -> PNG
  video: Windows Shell only (thumbs.rs:99-102); Linux returns Err (thumbs.rs:104-109)
  |
  v  ScanMsg::UpdateThumb{ thumb: String }  /  Thumbnail{ thumbnail_b64 }   (unbounded crossbeam channels: scan_tx, thumbnail_tx)
UI thread, FileExplorer::ui -> receive()  (receive/mod.rs:8-13): ai_update, clip (1 msg), preload, thumbnail (1 msg), scan (drain all)
  - table.get_mut(idx).thumbnail_b64 = String                       receive_scan.rs:574-578
  - last_scan_rows[..].thumbnail_b64 and recursive_filtered_rows[..]  receive_scan.rs:630-645  (linear find)
  - base64 decode -> viewer.thumb_cache: HashMap<String, Arc<[u8]>>   receive_scan.rs:647-658
  - pending_thumb_rows (32) -> upsert_rows_and_get_ids -> SurrealDB   receive_scan.rs:659-679
  |
  v  per visible cell (table/mod.rs:231-290):
Image::new(ImageSource::Bytes{ uri: "bytes://<path>", Bytes::Shared(Arc<[u8]>) }).texture_options(mipmap Linear)
  -> egui_extras ImageCrateLoader: one OS thread per URI, image::load_from_memory -> Arc<ColorImage> cached forever  (background_decode.rs:62)
  -> egui DefaultTextureLoader: ctx.load_texture -> glow texImage2D + generate_mipmap (egui_glow painter.rs:631); handle cached forever
```

### 1.2 Caches, lifetimes, eviction

| Tier | Structure | Key | Holds | Bound | Cleared when |
|---|---|---|---|---|---|
| Row | `Thumbnail.thumbnail_b64: Option<String>` | row | base64 PNG with `data:` prefix | none | row dropped |
| Scan snapshot | `last_scan_rows: Vec<Thumbnail>` + `last_scan_paths` | path | full row clones incl. base64 | none | new scan |
| Filtered snapshot | `recursive_filtered_rows: Vec<Thumbnail>` | - | another full clone of every row | none | filter signature change |
| DB page cache | `db_page_cache: BTreeMap<usize, Vec<Thumbnail>>` | page | `table.iter().cloned()` per visited page (`receive_preload.rs:28,35`; `mod.rs:608,637`) | none | `load_database_rows` at offset 0 |
| DB lookup | `db_lookup: HashMap<String, Thumbnail>` | path | one more full copy of the last preload | last message | next preload |
| App bytes cache | `viewer.thumb_cache: HashMap<String, Arc<[u8]>>` | path or `preview::<path>,1600` | decoded PNG bytes | none | only `rebuild_recursive_page` prunes non-preview keys (`mod.rs:1179-1187`); preview keys are never pruned |
| egui bytes loader | `bytes://<path>` | uri | shared `Arc` (no copy) | none | never (`forget_image` is not called anywhere) |
| egui image loader | `DecodeCache<Arc<ColorImage>>` | uri | RGBA CPU copy (256x171 = 175 KB) | none | never |
| egui texture loader | `DefaultTextureLoader` | (uri, TextureOptions) | `TextureHandle` (GPU, +33% mipmaps) | none (`end_pass` only trims multi-size SVG buckets, `texture_loader.rs:119-135`) | never |
| Video | `video_texture: Option<TextureHandle>` | - | last frame | 1 | never reset between files |
| Disk | none | - | - | - | derived images are never persisted except as base64 in SurrealDB |

Memory per displayed thumbnail (real 256x171 photo; PNG size from your ARW preview, 77 KB written by the `image` crate [M]):

| Copy | Bytes |
|---|---|
| `row.thumbnail_b64` (table) | 103 KB |
| `last_scan_rows` copy (recursive scans) | 103 KB |
| `recursive_filtered_rows` copy | 103 KB |
| `thumb_cache` PNG bytes | 77 KB |
| egui `ColorImage` (only once drawn) | 175 KB |
| GPU texture incl. mipmaps (only once drawn) | 233 KB |
| Total, recursive scan, drawn | ~800 KB [E] |

At 5,000 rows without scrolling: 0.9-1.4 GB resident [E]. Scrolling through the list adds ~408 KB per row drawn. Alternatives for the same pixels [M, PIL]: JPEG q82 12.3 KB, WebP lossy q80 8.5 KB, WebP lossless 46.6 KB, raw RGBA 175 KB.

### 1.3 Threads and channels

| Work | Executor | Size | Notes |
|---|---|---|---|
| Directory walk | `tokio::task::spawn_blocking` -> jwalk `RayonNewPool(max(8, min(logical,16)))` | 16 threads here | `scan.rs:233,304`; per-file `metadata()` runs serially on the consumer loop (`scan.rs:425` -> `:578`) |
| Scan thumbnails | rayon `THUMB_POOL` | `clamp(2,4)` threads, 3 worker loops | `scan.rs:43-51,105-144`; no `panic_handler` |
| UI-driven thumbnails | tokio tasks, blocking decode inside `async` | `Semaphore(6)` per explorer | `mod.rs:370,1407-1421`; tokio multi-thread default = 32 workers here |
| Preview, image edit, archive | tokio tasks / spawn_blocking | unbounded | per-frame spawn, see P-13 |
| egui image decode | `std::thread::Builder::spawn` per URI | unbounded | `background_decode.rs:62` |
| Video | `std::thread::spawn` | 1 | unbounded frame channel (`mod.rs:322`) |
| Channels | crossbeam `unbounded` everywhere except thumbnail queue `bounded(256)` and settings | - | hidden dock tabs never drain theirs (`tabs.rs:24-40`: `ui()` only runs for the visible tab) |

### 1.4 RAW, archives, video, previews

- **RAW**: extension list at `thumbs.rs:18-36` routes 13 formats to `raw_decode.rs`, but rawkit 0.1.0 "currently only works with the `.arw` files from Sony's cameras" (its README). `IMAGE_EXTS` only lists `arw` among RAW (`types.rs:7-8`), so scans never list the other RAW formats and their routing in `thumbs.rs` is unreachable from scans. A failure falls through to `image::open`, whose error replaces the RAW error (`thumbs.rs:45-80`). Decode is a full demosaic (`process_8bit`) to produce a 256 px thumbnail (`raw_decode.rs:13-35`).
- **Archives**: listing (`archive.rs:30-190`) is fine. Thumbnails reopen the archive per entry: zip = `ZipArchive::new` + extract to a temp file (`archive.rs:263-315`), 7z = `ArchiveReader::open` + `read_file` whole entry in memory (`:317-373`), tar = sequential scan from the start per entry (`:375-441`, O(N^2) for N entries, `.tar.gz` re-decompresses the prefix each time). One tokio task per media entry is spawned with synchronous work inside (`explorer/mod.rs:57-90`). Results are not cached.
- **Preview pane**: `generate_image_preview_png(path, 1600)` = full decode + `resize_exact(Lanczos3)` + PNG (`thumbs.rs:389-429`). In Database mode the key is pre-seeded with the 256 px thumbnail, so the full preview is never generated (`preview_pane.rs:162-173`).
- **Video**: Windows = Shell thumbnail; Linux = `Err` (`thumbs.rs:104-109`, `scan.rs:168-179`), so every video becomes a "failed thumbnail". Playback thread exists behind `--features ffmpeg` (section 4a).
- **Other decode sites**: CLIP (`ai/clip/embed.rs:7`), SigLIP (`ai/siglip/mod.rs:50`), JoyCaption (`ai/joycap/generate.rs:72`), Qwen edit (`qwen_image_edit/model.rs:1376`) each decode the original again with `ImageReader::open(..).decode()`; none handle RAW.

---

## 2. Findings

Severity: S1 breaks at scale (freeze/OOM/crash), S2 large waste, S3 latent or minor.

### 2.1 UI thread

| ID | Sev | Finding | Evidence | Impact |
|---|---|---|---|---|
| P-04 | S1 | Linear search per `UpdateThumb`: `last_scan_rows.iter_mut().find` and `recursive_filtered_rows.iter_mut().find`. `table_index` exists but is not used. Same pattern in `receive_thumbnail` (`table.iter_mut().find`, `table.iter().any`) and `receive_ai_update` | `receive_scan.rs:630-645`, `receive_thumbnail.rs:32,64`, `receive_ai_update.rs:72` | [M] total lookup time for N thumbnails: 23 ms @5k, 343 ms @20k, **10.2 s @100k** (release; x2 for the filtered snapshot; dev build slower). `table.iter_mut()` also marks the data-table dirty every call (P-30) |
| P-03 | S1 | `apply_recursive_filter_and_sort` runs every frame in recursive mode, builds a `format!` signature, and when it recomputes it does `iter().cloned().filter(..)`: it deep-clones every row (including base64) before filtering. The signature contains `last_scan_rows.len()`, so it recomputes whenever new rows arrive, and recomputes every frame if the result is empty | `file_table/mod.rs:484-487,1200-1252` (clone at `:1235-1240`) | [M] clone cost with 100 KB thumbnails: 118 ms @5k rows, 372 ms @20k, 872 ms @50k per recompute. Index-only filter: 0.4 ms @50k |
| P-05 | S1 | Status bar walks the whole table twice per frame (`row_passes_filter` with `to_lowercase` allocations, extension parse, `HashSet<String>` lookups). Shallow scans are not paginated, so tables reach 50k+ rows | `navbar.rs:336-348,375-402` | [M] release: 3.8 ms (5k, no filter) to 45 ms (50k, filter). Dev opt-level 0: 10-26 ms (5k) and 112-203 ms (50k) per frame. At opt-level 1: 0.7-1.6 ms and 12-22 ms |
| P-14 | S2 | `receive_thumbnail` handles one message per frame (`if let Ok`). `on_highlight_change` sends one cloned row per highlighted row; `load_logical_group_by_name` and virtual category views send one message per row | `receive_thumbnail.rs:8`, `table/mod.rs:757-766`, `load.rs:20-27,76` | selecting or loading 5,000 rows = 5,000 frames (~83 s at 60 Hz) [E]; each message also logs 3 info lines (`receive_thumbnail.rs:18,43,46`) |
| P-30 | S2 | Any `DerefMut` on `DataTable` (`push`, `get_mut`, `iter_mut`, `clear`, `retain`) marks its UI cache dirty; the next frame `validate_cc` re-runs `filter_row` on every row and re-sorts. During scans nearly every frame is dirty. With a text filter `row_passes_filter` allocates up to 5 lowercase strings per row; `compare_cell` for Tags allocates `join(",")` per comparison | `egui-data-table src/lib.rs:148-151` (`deref_mut` -> `mark_dirty`), `draw/state/validation.rs:85-120`, `table/mod.rs:111-147,747` | about one status-bar pass per dirty frame = 1.9-2.7 ms @5k rows (release) [E from M], 10x more in dev |
| P-11 | S2 | Blocking IO on the UI thread: `file_to_thumbnail` calls `std::fs::metadata` again although `FoundFile.modified` is already known (called from `receive_scan.rs:121,328`); the "Skip likely icons" toggle runs `table.retain` with `metadata` + `ImageReader::open().into_dimensions()` per row; AI follow clones the full table | `thumbs.rs:245`, `filter_menu.rs:226-246`, `receive_ai_update.rs:36-37,55-56` | 100k stats on cold or network storage = seconds to minutes [E]; the retain is also destructive (rows are removed, not hidden) |
| P-13 | S2 | Preview and image-edit panes spawn a full decode **every frame** until the result lands, with no in-flight set. If decode fails nothing is cached, so it retries forever | `preview_pane.rs:173-199`, `image_edit.rs:455-478` | thundering herd of 1600 px Lanczos decodes; unsupported files loop forever (an ARW that panics inside `tokio::spawn` logs a panic every frame) |
| P-26 | S3 | `receive_scan` drains the whole channel each frame with no time budget; hidden tabs never drain, so their channels hold every `UpdateThumb` string (100k x ~100 KB) until the tab is shown | `receive_scan.rs:10`, `tabs.rs:24-40` | multi-second first frame after switching tabs [E] |

### 2.2 Memory and caches

| ID | Sev | Finding | Evidence | Impact |
|---|---|---|---|---|
| P-01 | S1 | Thumbnail pixels live in the metadata row as base64 PNG, with the `data:image/png;base64,` prefix, and are duplicated across snapshots | see section 1.2 | ~0.8 MB per drawn thumbnail [E]; 100k = ~80 GB |
| P-02 | S1 | Nothing is evicted: `thumb_cache`, egui loaders (never `forget_image`), `db_page_cache` (500 MB per 5,000-row page, plus another clone on every page flip at `mod.rs:608,637`), preview keys (kept by `retain` at `mod.rs:1186`), per-tab duplicates (`FileExplorer::new(true)` per filtered tab owns its own `thumb_cache`, `thumb_scheduled`, `Semaphore(6)`) | `mod.rs:1179-1187`, `receive_preload.rs:28,35`, `texture_loader.rs:119-135` | RSS grows with everything ever displayed |
| P-19 | S3 | Cache key is the path only (`entry().or_insert_with`, URI `bytes://<path>`). A modified or re-thumbnailed file never refreshes. `show_cell_editor` loads the same URI with different `TextureOptions` = second texture | `receive_scan.rs:148,378,655`, `table/mod.rs:276-286,421-431` | stale thumbnails until restart |
| P-22 | S3 | `egui_logger` keeps an unbounded `Vec<Record>` and only trims inside its UI draw; with the Logs tab hidden it grows forever. Hot paths log at info per item | `egui_logger-0.13.0/src/ui.rs:291-292`, `receive_thumbnail.rs:18,43,46`, `thumbnails.rs:161,177,417` | slow leak plus a global mutex taken from worker threads |
| P-27 | S3 | Typed-message problem: `thumbnail_tx: Sender<Thumbnail>` carries new rows, selection events, directory navigation (a `<DIR>` row navigates), password prompts (`file_type == "<PASSWORD_REQUIRED>"`), and `preview::` cache fills | `receive_thumbnail.rs:12-95`, `table/mod.rs:523,682,764` | every selection clones a row with its base64; hard to reason about ordering |

### 2.3 Redundant and missing work

| ID | Sev | Finding | Evidence | Impact |
|---|---|---|---|---|
| P-08 | S2 | Every image in a scan is thumbnailed twice: the scanner enqueues it into `THUMB_POOL` (`scan.rs:435`) and `receive_scan` also spawns a tokio task per file because `item.thumb_data` is always `None` (`scan.rs:595`, `receive_scan.rs:382-387`). `thumb_scheduled` does not know about the scanner's jobs. The second result just overwrites the first. Worker counts (3 + 6) ignore the 32 cores | `scan.rs:44-47,106`, `mod.rs:370` | 2x CPU for every file [C]; the duplicate path also masks lost jobs (below) |
| P-09 | S2 | No persistent thumbnail lookup before decoding. DB rows are fetched only after the whole scan finishes (`spawn_scan(..).await` then `get_all_thumbnails_from_directory`) and only fill rows that lack a thumbnail | `explorer/mod.rs:146-163`, `receive_preload.rs:49-68` | every visit to a folder re-decodes every image |
| P-10 | S2 | No viewport priority, no cancellation. Job order is queue order plus `swap_remove` order. Navigating does not cancel the previous scan (`current_scan_id` is overwritten; `cancel_scan` is only called by the Cancel button). Old jobs sit in the shared `bounded(256)` queue ahead of new ones. At scan end `finish()` drops unflushed deferred jobs and the `for job in deferred.drain(..)` loop drops the rest on the first full queue | `explorer/mod.rs:139-145`, `scan_menu.rs:119`, `scan.rs:59-92,518-526,554-560` | thumbnails for rows nobody sees delay the ones on screen; stale scans burn CPU |
| P-16 | S2 | Archive thumbnails: open per entry; tar O(N^2); 7z whole entry in RAM; temp-file round trip; sync work inside `tokio::spawn` | `archive.rs:241-441`, `explorer/mod.rs:57-90`, `table/mod.rs:492-519,572-625` | 5,000-image tar.gz = 5,000 sequential decompressions of the prefix [E] |
| P-12 | S2 | Blocking decode inside `async` tasks (not `spawn_blocking`) | `mod.rs:1327-1341,1407-1421`, `receive_scan.rs:533-550` | holds a tokio worker per decode; 6 per explorer tab |
| P-17 | S2 | Linux has no video thumbnails; every video lands in the "Failed thumbnails" tab | `thumbs.rs:104-109`, `scan.rs:168-179` | all video rows show the broken-link icon |
| P-18 | S2 | RAW routing and error masking (section 1.4); HEIC/HEIF/AVIF are in `IMAGE_EXTS` but `image` 0.25.10 default features only enable the AVIF **encoder** (`avif = [ravif, rgb]`, decode needs `avif-native`), HEIC has no decoder | `types.rs:7-8`, `thumbs.rs:45-80`, `image-0.25.10/Cargo.toml:61-76` | those files fail on Linux (Windows has the Shell fallback) |
| P-20 | S3 | `recursive_prefetch_inflight` is set to `candidates.len()` and never decremented, so `prefetch_next_page_thumbs` runs exactly once per session | `mod.rs:1274,1316,1326-1361` | dead feature |
| P-21 | S3 | `table_index` can go stale: the viewer does not override `allow_row_deletions` / `allow_row_insertions` (default `true` in `egui-data-table viewer.rs:248-255`) and `table.retain` is used in `filter_menu.rs:226` | `receive_scan.rs:574,164,359` | wrong row updated after a UI delete/paste (read, not reproduced) |
| P-25 | S3 | Database mode seeds `preview::<path>,1600` with the 256 px thumbnail, so the high-resolution preview is never produced | `preview_pane.rs:162-173` | blurry preview in DB mode |
| P-28 | S3 | Placeholder shows the broken-link icon whenever `thumbnail_b64.is_none()`, i.e. also while the thumbnail is merely pending | `table/mod.rs:252-260` | misleading UI; no pending/failed state |
| P-24 | S3 | Assistant attachment loader base64-decodes the stored string without stripping the `data:` prefix that `thumbs.rs:96` produces | `assistant.rs:1819-1831` | attachment thumbnails likely never show (read, not reproduced) |
| P-29 | S3 | Adjacent: `ThumbCodec` columns (3=Category, 4=Tags, 5=Modified, 6=Size, 7=Type) no longer match the viewer's column order (3=Size, 4=Type, 5=CLIP, 6=Modified ...) | `table/codec.rs:12-24` vs `table/mod.rs:172-176` | copy/paste writes into the wrong fields |

### 2.4 Database

Benchmarks: embedded SurrealKV on tmpfs, release build, 3,000 rows, 75 KB random payload = 100 KB base64 per row (matches the measured real-photo size) [M].

| Access | Time |
|---|---|
| `SELECT *` directory page, thumbnails inline (`thumbnails.rs:202-207`) | 279-457 ms (300 MB payload) |
| Field list without `thumbnail_b64` on the same table (`find_thumbs_from_paths` style, `thumbnails.rs:253`) | 174-263 ms (SurrealKV still reads the whole record) |
| Same rows in a metadata-only table | **15-18 ms** |
| `SELECT * WHERE path` single row | 0.12-0.24 ms |
| Per-file upsert = `SELECT *` + `UPDATE ... WHERE path` + `SELECT VALUE id` (`thumbnails.rs:565-598,601-645`) | 1.15 ms/file (865/s) |
| Insert 250-row batches | 0.33-0.36 ms/row |

Conclusions: projecting the column out is not enough (-40%); the thumbnail must live in a separate record or store (15-20x on page loads, and no 500 MB pages). Per-file round trips are cheap in-process; they are a throughput issue (100k files = ~115 s sequential), not a UI issue, but they run three queries per file and drag the base64 through `SELECT *`. `receive_thumbnail.rs:74-86` also issues a `get_embedding` query per arriving row (`SELECT *` returns the 1024-float vector just to test presence).

---

## 3. Measurements

Method: Rust micro-benchmarks that replicate the app's exact code path with the same crate versions (image 0.25.10 default features), 12 real images (8000x4000 panoramas, 7680x4320 wallpapers, PNG art, your ARW). Sources are in the scratchpad (`thumbbench`, `uibench`, `dbbench`). CPU: 32 logical cores. No GUI session was run, so end-to-end frame times are not measured.

### 3.1 Producer cost per image (release, ms)

| Image | Pixels | decode (`image::open`) | `thumbnail(256)` | PNG enc | b64 | consumer: b64 dec + PNG dec |
|---|---|---|---|---|---|---|
| panorama JPEG | 8000x4000 | 141-416 | 24-26 | 0.2 | 0.02 | 0.22 |
| wallpaper JPEG | 7680x4320 | 200 | 22 | 0.2 | 0.02 | 0.29 |
| UI screenshot JPEG | 2560x1600 | 69 | 5 | 0.2 | 0.02 | 0.24 |
| PNG art | 5119x3413 | 133 | 16 | 0.5 | 0.02 | 0.33 |
| PNG art | 2912x1632 | 40 | 6 | 0.2 | 0.01 | 0.24 |

Average over the 12: 133 ms producer per image, 0.26 ms consumer. **Decode dominates; the PNG/base64 round trip costs almost no time but 5-8x the bytes of a JPEG** (avg PNG 55.6 KB vs JPEG q82 11.5 KB; real photo 77 KB vs 12.3 KB).

DCT-scaled JPEG decode (libjpeg-turbo `draft` via PIL, `jpeg-decoder` scale + `fast_image_resize` in Rust) gave only 1.1-1.9x on these high-bitrate files [M]: Huffman decode dominates, so this is not a top lever. Not decoding at all (cache, embedded previews) is.

### 3.2 Dev profile

`[profile.dev]` (`Cargo.toml:107-109`) sets only `strip` and `debug`; every dependency compiles at opt-level 0 (only `target/debug` exists in the repo).

| Build | 32 MP JPEG decode | 2560x1600 decode | PNG 2912x1632 decode | status-bar frame, 50k rows |
|---|---|---|---|---|
| dev (current) | 10,512 ms | 1,193 ms | 410 ms | 112-203 ms |
| dev + `[profile.dev.package."*"] opt-level = 3` | 144 ms | 140 ms | 243 ms (generic code instantiated in the app crate stays at O0) | - |
| dev + app `opt-level = 1` + deps O3 | **134 ms** | **38 ms** | **34 ms** | **12-22 ms** |
| release | 141 ms | 69 ms | 40 ms | 23-45 ms |

Single runs per cell; file-cache state and scheduling noise apply (the 140 ms in the second column is higher than its release counterpart for that reason or because generic `image` code is instantiated in the benchmark crate at O0).

### 3.3 UI-thread loops (release unless noted)

| Loop | N | Result |
|---|---|---|
| linear `find` per thumbnail (`receive_scan.rs:630-645`) | 5k / 20k / 100k | 23 ms / 343 ms / 10.2 s total |
| clone all rows then filter (`mod.rs:1235-1240`), 100 KB thumbnails | 5k / 20k / 50k | 118 ms / 372 ms / 872 ms per recompute |
| status bar, two passes, no filter | 5k / 50k | 3.8 ms / 22.8 ms (dev: 10.4 / 111.9 ms) |
| status bar, filter "sunset" | 5k / 50k | 5.4 ms / 45.0 ms (dev: 26.0 / 203.3 ms) |

### 3.4 Preview path (release)

| Image | decode | `resize_exact(Lanczos3)` to 1600 | `thumbnail()` to 1600 | PNG enc | PNG size | JPEG q85 size |
|---|---|---|---|---|---|---|
| 8000x4000 JPEG | 117 ms | 193 ms | 67 ms | 6 ms | 1.84 MB | 258 KB |
| 5119x3413 PNG | 66 ms | 112 ms | 28 ms | 9 ms | 3.0 MB | 409 KB |

Lanczos3 costs 2-3x `thumbnail()`; the 1600 px preview is stored as PNG (1.4-3.0 MB) and decoded again by egui.

### 3.5 RAW and video prototypes

| Case | Result |
|---|---|
| ARW `_DSC7596.ARW` (Sony ZV-E10, 6048x4024, 14-bit compressed, 24.8 MB) | rawkit: panic `assertion failed: bits_per_sample == 12` at `rawkit-0.1.0/src/decoder/arw2.rs:35`, process abort (exit 134) [R] |
| Embedded preview extraction (IFD0 tags 0x0201/0x0202, 1616x1080 JPEG, 377,579 B) | 0.2-3.3 ms |
| ARW -> 256 px thumbnail via preview (full patched `raw_decode.rs`) | 13.4 ms; 1600 px preview in 38.5 ms |
| 4K H.264 (generated) -> 256x144 poster frame via ffmpeg-next, seek to 10% | 20-34 ms |

---

## 4. Known defects

### 4a. `cargo check --features ffmpeg`

Reproduced on the real crate (copy at `.../scratchpad/repo-copy`, ffmpeg-sys-next 9.0.1 builds against system FFmpeg libavcodec 63.1.101):

| # | Location | Error | Cause in ffmpeg-next 9.0.0 |
|---|---|---|---|
| 1,2 | `preview_pane.rs:12,77` | `StdArc` not found | alias is imported in `file_table/mod.rs:8`, not in this file |
| 3 | `:26` | `and_then(\|c\| c.decoder())` expected `Result`, found `Decoder` | `Context::decoder()` returns `Decoder` (`codec/decoder/decoder.rs`); only `.video()` returns `Result<Video>` |
| 4 | `:50` | no method `read_packet` on `Input` | removed; use `Input::packets()` (`format/context/input.rs:144`, yields `(Stream, Packet)`, ends on EOF or terminal error) |
| 5,6 | `:75` | no method `num` / `den` on `Rational` | `Rational(pub i32, pub i32)` exposes `numerator()` / `denominator()` (`util/rational.rs:10-20`) |
| 7 (masked) | `:210` | `tex.set(..)` cannot borrow `*tex` as mutable | surfaces after 1-6 are fixed; `TextureHandle::set` takes `&mut self` |

Corrected code (unified diff against the repo file; **not applied**). Verified: `cargo check` and `cargo check --features ffmpeg` both finish clean on the patched copy.

```diff
--- a/src/ui/file_table/preview_pane.rs
+++ b/src/ui/file_table/preview_pane.rs
@@ -4,6 +4,7 @@
 #[cfg(feature = "ffmpeg")]
 use {
     std::sync::atomic::Ordering,
+    std::sync::Arc as StdArc,
     crossbeam::channel::TryRecvError,
 };
 
@@ -23,7 +24,7 @@
                     if let Some(stream) = ictx.streams().best(ffmpeg_next::media::Type::Video) {
                         let stream_index = stream.index();
                         // Setup decoder
-                        let context_decoder = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters()).and_then(|c| c.decoder()).and_then(|d| d.video()).map_err(|e| e.to_string());
+                        let context_decoder = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters()).and_then(|c| c.decoder().video()).map_err(|e| e.to_string());
                         if let Ok(mut decoder) = context_decoder {
                             // Prepare scaler to RGBA
                             let mut scaler = ffmpeg_next::software::scaling::Context::get(
@@ -46,9 +47,10 @@
                                 avg_frame_rate.numerator() as f64 / avg_frame_rate.denominator() as f64
                             } else { 30.0 };
 
+                            let mut packets = ictx.packets();
                             while !stop_flag.load(Ordering::SeqCst) {
-                                match ictx.read_packet() {
-                                    Ok(pkt) => {
+                                match packets.next() {
+                                    Some((_, pkt)) => {
                                         if pkt.stream() == stream_index {
                                             if let Err(e) = decoder.send_packet(&pkt) {
                                                 log::warn!("decoder send_packet error: {:?}", e);
@@ -72,7 +74,7 @@
                                                                 let end = start + w * 4;
                                                                 rgba.extend_from_slice(&data[start..end.min(data.len())]);
                                                             }
-                                                            let pts = decoded.pts().map(|p| p as f64 * f64::from(time_base.num()) / f64::from(time_base.den())).unwrap_or_else(|| (frame_index as f64)/fps);
+                                                            let pts = decoded.pts().map(|p| p as f64 * f64::from(time_base.numerator()) / f64::from(time_base.denominator())).unwrap_or_else(|| (frame_index as f64)/fps);
                                                             frame_index += 1;
                                                             let _ = tx.send(super::VideoFrame { rgba: StdArc::from(rgba.into_boxed_slice()), width: w, height: h, pts });
                                                             // Pace according to fps
@@ -83,9 +85,9 @@
                                             }
                                         }
                                     }
-                                    Err(e) => {
-                                        // End of file or error
-                                        log::debug!("ffmpeg read_packet error/end: {:?}", e);
+                                    None => {
+                                        // End of file or unrecoverable demux error
+                                        log::debug!("ffmpeg demux finished");
                                         break;
                                     }
                                 }
@@ -206,7 +208,7 @@
                             match self.video_frame_rx.try_recv() {
                                 Ok(frame) => {
                                     let color = egui::ColorImage::from_rgba_unmultiplied([frame.width, frame.height], &frame.rgba);
-                                    if let Some(tex) = &self.video_texture {
+                                    if let Some(tex) = &mut self.video_texture {
                                         tex.set(color, egui::TextureOptions::LINEAR);
                                     } else {
                                         // Create a new texture handle and store it
```

Runtime defects that remain in that thread after it compiles (not fixed in the diff):

- Pacing is `sleep(1/fps)` after each frame regardless of decode time or `pts` (`preview_pane.rs:79`).
- `video_frame_tx/rx` is unbounded (`mod.rs:322`); while paused or hidden the decoder keeps queueing 4K RGBA frames (33 MB each) and the UI uploads every queued frame in one pass (`preview_pane.rs:205-224`). Use `bounded(2)` and keep only the newest frame.
- Frames are RGBA at source resolution (`:29-37`); scale to the display size (<= 1280 px) in swscale.
- `send_eof()` is called after the loop but frames are not drained (`:94`).
- `video_duration_secs` is never set (only `None` at `mod.rs:367`), so the slider never renders; play restarts at 0 every time; there is no seek.
- `video_texture` is not reset when `current_thumb` changes, so the previous video's last frame shows first.

### 4b. rawkit panic kills the process

Call path [R, verified by reproduction]:

```
scan.rs:109   THUMB_POOL.spawn(move || { while let Ok(job) = rx.recv() { ...
scan.rs:120     generate_thumb(&job.path, &job.kind)            -> thumbs.rs:168-179
thumbs.rs:40      raw_decode::decode_raw_to_png_bytes(path, 256)
raw_decode.rs:13    rawkit::RawImage::decode(&mut reader)
rawkit lib.rs        arw_ifd.compression == 32767 && strip_byte_counts[0] == w*h  -> decoder::arw2::decode
arw2.rs:35             assert!(bits_per_sample == 12)            // file has 14 bits -> panic
rayon-core registry.rs:375-384 catch_unwind(): no panic_handler registered -> AbortIfPanic -> abort()
```

Why the process dies: `ThreadPoolBuilder` in `scan.rs:43-51` has no `panic_handler`; for `spawn`ed jobs rayon-core aborts when none is set ("If no panic handler is set, the default is to abort the process", `rayon-core-1.13.0/src/lib.rs:564`; `registry.rs:375-384`). Even with a handler, the panic would end the worker's `while` loop and silently lose one of the 3 workers, so the boundary has to be per job.

Why rawkit panics so easily: `RawImage::decode` is `unwrap()` / `assert!` / `todo!()` throughout (`lib.rs`: `identify_camera_model(..).unwrap()`, `todo!()` for non-ARW2 compression; `arw2.rs`: asserts on strip count, compression, `bits_per_sample == 12`, CFA dims; `uncompressed.rs`: same). It supports Sony ARW only (README). `process_8bit` demosaics the whole sensor (~11-17 bytes per pixel peak [E], 260-400 MB for 24 MP) to make a 256 px thumbnail. Other makers' RAW in the extension list either return `Err` or hit `todo!()`.

Fix (four parts, all compile- and run-verified; unified diffs against the repo plus the full replacement `raw_decode.rs`):

1. **Preview first.** ARW/NEF/DNG/ORF/PEF/RW2/CR2 are TIFF containers with embedded JPEGs. Walk IFD0, its chain and SubIFDs, collect `JPEGInterchangeFormat/Length` (0x0201/0x0202) and single-strip JPEG-compressed IFDs, choose the largest that decodes, apply IFD0 orientation. Fuji RAF has the JPEG offset/length at bytes 84/88; anything else falls back to scanning the first 8 MiB for JPEG SOI markers. Decoding the 1616x1080 JPEG and shrinking took 13 ms for a thumbnail.
2. **Panic boundary at the source.** Only `rawkit_decode_guarded` calls rawkit, inside `catch_unwind(AssertUnwindSafe(..))`; it is used only for `.arw` when no preview exists. Requires `panic = "unwind"` (already set at `Cargo.toml:103`).
3. **Backstop.** `panic_handler` on `THUMB_POOL` and a per-job `catch_unwind` in the worker loop so any other decoder panic becomes a `ThumbFailed` message instead of an abort.
4. **Error surfacing.** RAW errors are returned as-is (`raw: ...`) instead of falling through to `image::open`, which replaced them with "format could not be determined". The existing `ScanMsg::ThumbFailed` path then lists them in the Failed tab with the real reason.

Verification on the real ARW: pool without `panic_handler` returns `Ok(75074 bytes)` in 13 ms; with the preview pointers zeroed (forces the rawkit path) it returns `Err("rawkit panicked ...: assertion failed: bits_per_sample == 12 (unsupported ARW variant, e.g. 14-bit compressed)")` and the process survives; a non-RAW file returns `Err`. CR3/RAF/RW2/ORF layouts follow the container specs but were **not** tested (no samples).

`thumbs.rs` and `scan.rs` hunks:

```diff
--- a/src/utilities/thumbs.rs
+++ b/src/utilities/thumbs.rs
@@ -9,43 +9,17 @@
 
 pub fn generate_image_thumb_data(path: &Path) -> Result<String, String> {
     log::debug!("[thumb] generating image thumb: {}", path.display());
-    // Detect RAW formats we should decode via raw_decode helper first
     let ext = path
         .extension()
         .and_then(|e| e.to_str())
         .map(|s| s.to_ascii_lowercase())
         .unwrap_or_default();
-    let is_raw = matches!(ext.as_str(),
-        // Sony RAW
-        "arw" | "srf" | "sr2" |
-        // Canon RAW
-        "cr2" | "cr3" |
-        // Nikon RAW
-        "nef" | "nrw" |
-        // Adobe DNG
-        "dng" |
-        // Olympus
-        "orf" |
-        // Fuji
-        "raf" |
-        // Pentax
-        "pef" |
-        // Panasonic
-        "rw2" |
-        // Generic
-        "raw");
-
-    if is_raw {
-        // Try raw-to-png path first
-        match crate::utilities::raw_decode::decode_raw_to_png_bytes(path, 256) {
-            Ok(png) => {
-                let b64 = BASE64.encode(&png);
-                return Ok(format!("data:image/png;base64,{}", b64));
-            }
-            Err(e) => {
-                log::warn!("[thumb] raw decode failed {}: {} — falling back", path.display(), e);
-            }
-        }
+
+    if crate::utilities::raw_decode::is_raw_ext(&ext) {
+        // RAW errors are returned directly.
+        let png = crate::utilities::raw_decode::decode_raw_to_png_bytes(path, 256)
+            .map_err(|e| format!("raw: {e:#}"))?;
+        return Ok(format!("data:image/png;base64,{}", BASE64.encode(&png)));
     }
 
     let img = match image::open(path) {
@@ -392,13 +366,10 @@
         .and_then(|e| e.to_str())
         .map(|s| s.to_ascii_lowercase())
         .unwrap_or_default();
-    let is_raw = matches!(ext.as_str(),
-        "arw"|"srf"|"sr2"|"cr2"|"cr3"|"nef"|"nrw"|"dng"|"orf"|"raf"|"pef"|"rw2"|"raw");
 
-    if is_raw {
-        if let Ok(png) = crate::utilities::raw_decode::decode_raw_to_png_bytes(path, max_dim) {
-            return Ok(png);
-        }
+    if crate::utilities::raw_decode::is_raw_ext(&ext) {
+        return crate::utilities::raw_decode::decode_raw_to_png_bytes(path, max_dim)
+            .map_err(|e| format!("raw: {e:#}"));
     }
 
     match image::open(path) {
--- a/src/utilities/scan.rs
+++ b/src/utilities/scan.rs
@@ -46,6 +46,7 @@
     rayon::ThreadPoolBuilder::new()
         .num_threads(workers)
         .thread_name(|i| format!("thumb-{}", i))
+        .panic_handler(|p| log::error!("[thumb] pool task panicked: {}", panic_text(p.as_ref())))
         .build()
         .expect("thumb pool")
 });
@@ -56,6 +57,12 @@
 // Instrumentation for thumbnail queue back-pressure
 static THUMB_DEFERRED: Lazy<std::sync::Mutex<Vec<ThumbJob>>> = Lazy::new(|| std::sync::Mutex::new(Vec::new()));
 
+fn panic_text(p: &(dyn std::any::Any + Send)) -> String {
+    p.downcast_ref::<&str>().map(|s| (*s).to_string())
+        .or_else(|| p.downcast_ref::<String>().cloned())
+        .unwrap_or_else(|| "unknown panic".to_string())
+}
+
 fn flush_deferred_some(max_items: usize) {
     if max_items == 0 { return; }
     if let Some(tx) = THUMB_TX.get() {
@@ -117,7 +124,9 @@
                     }
                     continue; 
                 }
-                match generate_thumb(&job.path, &job.kind) {
+                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| generate_thumb(&job.path, &job.kind)))
+                    .unwrap_or_else(|p| Err(format!("thumbnail worker panicked: {}", panic_text(p.as_ref()))));
+                match result {
                     Ok(thumb_b64) => {
                         if let Ok(map) = SCAN_ROUTERS.lock() {
                             if let Some(tx) = map.get(&job.scan_id) {
```

Replacement `src/utilities/raw_decode.rs` (full file; public API `decode_raw_to_png_bytes(path, max_dim)` is unchanged, `is_raw_ext` and `decode_raw_image` are new):

```rust
use image::{imageops::FilterType, DynamicImage, GenericImageView, ImageFormat, ImageReader, RgbImage};
use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek, SeekFrom};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;

pub fn is_raw_ext(ext: &str) -> bool {
    matches!(ext, "arw" | "srf" | "sr2" | "cr2" | "cr3" | "nef" | "nrw" | "dng" | "orf" | "raf" | "pef" | "rw2" | "raw")
}

/// Decodes RAW to PNG bytes (long side <= `max_dim`) via the embedded JPEG preview, then guarded rawkit.
pub fn decode_raw_to_png_bytes(path: &Path, max_dim: u32) -> anyhow::Result<Vec<u8>> {
    let img = decode_raw_image(path, max_dim)?;
    let mut out = Vec::new();
    img.write_to(&mut Cursor::new(&mut out), ImageFormat::Png)?;
    Ok(out)
}

pub fn decode_raw_image(path: &Path, max_dim: u32) -> anyhow::Result<DynamicImage> {
    let preview = embedded_jpeg_preview(path);
    if let Some((bytes, orientation)) = preview {
        match image::load_from_memory_with_format(&bytes, ImageFormat::Jpeg) {
            Ok(img) => return Ok(fit(apply_exif_orientation(img, orientation), max_dim)),
            Err(e) => log::warn!("[raw] embedded preview undecodable {}: {e}", path.display()),
        }
    }
    let is_arw = path.extension().is_some_and(|e| e.eq_ignore_ascii_case("arw"));
    if !is_arw {
        anyhow::bail!("RAW format has no embedded JPEG preview and no full decoder: {}", path.display());
    }
    Ok(fit(rawkit_decode_guarded(path)?, max_dim))
}

fn fit(img: DynamicImage, max_dim: u32) -> DynamicImage {
    let (w, h) = img.dimensions();
    if w.max(h) > max_dim { img.resize(max_dim, max_dim, FilterType::Triangle) } else { img }
}

fn panic_text(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>().map(|s| (*s).to_string())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_string())
}

/// Runs rawkit with panics converted to errors.
fn rawkit_decode_guarded(path: &Path) -> anyhow::Result<DynamicImage> {
    let res = catch_unwind(AssertUnwindSafe(|| -> anyhow::Result<DynamicImage> {
        let mut reader = BufReader::new(File::open(path)?);
        let raw = rawkit::RawImage::decode(&mut reader)?;
        let rgb = raw.process_8bit();
        let (w, h) = (rgb.width as u32, rgb.height as u32);
        let img: RgbImage = RgbImage::from_raw(w, h, rgb.data)
            .ok_or_else(|| anyhow::anyhow!("rawkit produced unexpected buffer size"))?;
        Ok(DynamicImage::ImageRgb8(img))
    }));
    match res {
        Ok(r) => r,
        Err(p) => anyhow::bail!("rawkit panicked on {}: {} (unsupported ARW variant, e.g. 14-bit compressed)", path.display(), panic_text(p.as_ref())),
    }
}

fn apply_exif_orientation(img: DynamicImage, o: u16) -> DynamicImage {
    match o {
        2 => img.fliph(),
        3 => img.rotate180(),
        4 => img.flipv(),
        5 => img.rotate90().fliph(),
        6 => img.rotate90(),
        7 => img.rotate270().fliph(),
        8 => img.rotate270(),
        _ => img,
    }
}

// Embedded JPEG preview extraction for TIFF-family RAW, Fuji RAF and SOI-scan fallback.

#[derive(Clone, Copy)]
enum Endian { Le, Be }

struct Entry { tag: u16, ty: u16, count: u32, field: [u8; 4] }

struct Tiff<R: Read + Seek> { r: R, e: Endian, len: u64 }

struct Candidate { offset: u64, len: u64 }

impl<R: Read + Seek> Tiff<R> {
    fn open(mut r: R) -> Option<Self> {
        let len = r.seek(SeekFrom::End(0)).ok()?;
        r.seek(SeekFrom::Start(0)).ok()?;
        let mut h = [0u8; 8];
        r.read_exact(&mut h).ok()?;
        let e = match &h[0..2] { b"II" => Endian::Le, b"MM" => Endian::Be, _ => return None };
        let me = Tiff { r, e, len };
        // 42 = TIFF, 0x4F52 / 0x5352 = ORF, 0x55 = RW2
        if !matches!(me.u16(&h[2..4]), 42 | 0x4F52 | 0x5352 | 0x55) { return None; }
        Some(me)
    }
    fn u16(&self, b: &[u8]) -> u16 {
        let a = [b[0], b[1]];
        match self.e { Endian::Le => u16::from_le_bytes(a), Endian::Be => u16::from_be_bytes(a) }
    }
    fn u32(&self, b: &[u8]) -> u32 {
        let a = [b[0], b[1], b[2], b[3]];
        match self.e { Endian::Le => u32::from_le_bytes(a), Endian::Be => u32::from_be_bytes(a) }
    }
    fn first_ifd(&mut self) -> Option<u32> {
        self.r.seek(SeekFrom::Start(4)).ok()?;
        let mut b = [0u8; 4];
        self.r.read_exact(&mut b).ok()?;
        Some(self.u32(&b))
    }
    fn read_ifd(&mut self, off: u32) -> Option<(Vec<Entry>, u32)> {
        if off == 0 || u64::from(off) + 2 > self.len { return None; }
        self.r.seek(SeekFrom::Start(off.into())).ok()?;
        let mut n = [0u8; 2];
        self.r.read_exact(&mut n).ok()?;
        let n = self.u16(&n) as usize;
        if n == 0 || n > 512 { return None; }
        let mut buf = vec![0u8; n * 12 + 4];
        self.r.read_exact(&mut buf).ok()?;
        let entries = (0..n)
            .map(|i| {
                let c = &buf[i * 12..i * 12 + 12];
                Entry { tag: self.u16(&c[0..2]), ty: self.u16(&c[2..4]), count: self.u32(&c[4..8]), field: [c[8], c[9], c[10], c[11]] }
            })
            .collect();
        Some((entries, self.u32(&buf[n * 12..n * 12 + 4])))
    }
    fn scalar(&self, en: &Entry) -> Option<u32> {
        match en.ty {
            3 => Some(u32::from(self.u16(&en.field[0..2]))),
            4 => Some(self.u32(&en.field)),
            _ => None,
        }
    }
    fn longs(&mut self, en: &Entry) -> Option<Vec<u32>> {
        if en.ty != 4 || en.count == 0 || en.count > 16 { return None; }
        if en.count == 1 { return Some(vec![self.u32(&en.field)]); }
        self.r.seek(SeekFrom::Start(self.u32(&en.field).into())).ok()?;
        let mut b = vec![0u8; en.count as usize * 4];
        self.r.read_exact(&mut b).ok()?;
        Some(b.chunks_exact(4).map(|c| self.u32(c)).collect())
    }
}

fn tiff_candidates(path: &Path) -> Option<(Vec<Candidate>, u16)> {
    let mut t = Tiff::open(BufReader::new(File::open(path).ok()?))?;
    let mut queue: Vec<u32> = vec![t.first_ifd()?];
    let mut seen: Vec<u32> = Vec::new();
    let mut out: Vec<Candidate> = Vec::new();
    let (mut orientation, mut first) = (1u16, true);
    while let Some(off) = queue.pop() {
        if seen.contains(&off) || seen.len() > 32 { continue; }
        seen.push(off);
        let Some((entries, next)) = t.read_ifd(off) else { continue };
        let (mut jif, mut jif_len) = (None, None);
        let (mut compression, mut photometric) = (0u32, 0u32);
        let (mut strip_off, mut strip_len, mut strip_cnt) = (None, None, 0u32);
        for en in &entries {
            match en.tag {
                0x0103 => compression = t.scalar(en).unwrap_or(0),
                0x0106 => photometric = t.scalar(en).unwrap_or(0),
                0x0111 => { strip_off = t.scalar(en); strip_cnt = en.count; }
                0x0117 => strip_len = t.scalar(en),
                0x0112 if first => orientation = t.scalar(en).unwrap_or(1) as u16,
                0x0201 => jif = t.scalar(en),
                0x0202 => jif_len = t.scalar(en),
                0x014A => { if let Some(v) = t.longs(en) { queue.extend(v); } }
                _ => {}
            }
        }
        first = false;
        if let (Some(o), Some(l)) = (jif, jif_len) { out.push(Candidate { offset: o.into(), len: l.into() }); }
        // Single-strip JPEG IFD that is not CFA or linear-raw data.
        if matches!(compression, 6 | 7) && !matches!(photometric, 32803 | 34892) && strip_cnt == 1 {
            if let (Some(o), Some(l)) = (strip_off, strip_len) { out.push(Candidate { offset: o.into(), len: l.into() }); }
        }
        if next != 0 { queue.push(next); }
    }
    Some((out, orientation))
}

fn raf_candidate(path: &Path) -> Option<Candidate> {
    let mut f = File::open(path).ok()?;
    let mut h = [0u8; 92];
    f.read_exact(&mut h).ok()?;
    if &h[0..15] != b"FUJIFILMCCD-RAW" { return None; }
    Some(Candidate {
        offset: u32::from_be_bytes([h[84], h[85], h[86], h[87]]).into(),
        len: u32::from_be_bytes([h[88], h[89], h[90], h[91]]).into(),
    })
}

fn scan_candidates(path: &Path) -> Vec<Candidate> {
    let mut buf = Vec::new();
    let Ok(f) = File::open(path) else { return Vec::new() };
    if f.take(8 << 20).read_to_end(&mut buf).is_err() { return Vec::new(); }
    buf.windows(3)
        .enumerate()
        .filter(|(_, w)| w == &[0xFF, 0xD8, 0xFF])
        .map(|(i, _)| Candidate { offset: i as u64, len: (buf.len() - i) as u64 })
        .collect()
}

fn read_range(path: &Path, c: &Candidate) -> std::io::Result<Vec<u8>> {
    let mut f = File::open(path)?;
    f.seek(SeekFrom::Start(c.offset))?;
    let mut b = Vec::new();
    f.take(c.len.min(64 << 20)).read_to_end(&mut b)?;
    Ok(b)
}

/// Largest decodable embedded JPEG and the IFD0 orientation.
fn embedded_jpeg_preview(path: &Path) -> Option<(Vec<u8>, u16)> {
    let (mut cands, orientation) = tiff_candidates(path).unwrap_or((Vec::new(), 1));
    if let Some(c) = raf_candidate(path) { cands.push(c); }
    if cands.is_empty() { cands = scan_candidates(path); }
    let mut best: Option<(u64, Vec<u8>)> = None;
    for c in &cands {
        let Ok(bytes) = read_range(path, c) else { continue };
        if bytes.len() < 4 || bytes[0..3] != [0xFF, 0xD8, 0xFF] { continue; }
        let Ok((w, h)) = ImageReader::with_format(Cursor::new(&bytes), ImageFormat::Jpeg).into_dimensions() else { continue };
        let px = u64::from(w) * u64::from(h);
        if best.as_ref().is_none_or(|(p, _)| px > *p) { best = Some((px, bytes)); }
    }
    best.map(|(_, b)| (b, orientation))
}
```

Residual: a full RAW decode for makers other than Sony ARW still does not exist. Options in section 5.6.

---

## 5. Target architecture for 100k+ files

### 5.1 Principles

1. Pixels never live in rows. The catalog holds metadata; thumbnails live in a persistent encoded store and a bounded GPU texture cache.
2. The UI thread only draws. Decode, filter, sort and layout run on workers and publish immutable snapshots; per-frame work is O(visible), never O(N).
3. Work is demand-driven: requests carry viewport priority and a cancel token; navigation cancels everything not in the new viewport.
4. Every stage is timed and counted, and the counters have a debug panel (your preference for devtools pages and debug UI).

### 5.2 Data flow

```
scan (jwalk, existing)  ->  Catalog (Vec<MediaItem>, ItemId = u32)  ->  View (Vec<u32> after filter/sort, built off-thread, swapped atomically)
                                                |
UI each frame: visible range + margin  ->  Interest set  ->  Scheduler (priority heap, generation, cancel tokens)
                                                |
                     L1 TextureCache (bytes budget, LRU + on-screen protection)  <--  upload budget per frame
                                                |  miss
                     L2 ThumbStore (key = blake3(path, size, mtime, params_ver) -> JPEG/WebP bytes)  <-- batch get
                                                |  miss
                     decode pool (rayon, cores-2) / io pool (2-4) / video pool (2)
                       JPEG/PNG/WebP: image + zune-jpeg      RAW: embedded preview -> LibRaw/rawler fallback
                       video: ffmpeg poster                  archive: stream entry into memory (capped)
                       each job in catch_unwind; results -> bounded channel -> UI drains with count/time budget
```

### 5.3 Components

**Catalog.** `MediaItem { id: u32, path: Arc<str>, dir_id: u32, kind: u8, size: u64, mtime_ns: i64, w: u32, h: u32, flags }` (~100 B; 100k = ~10 MB). Selection becomes `HashSet<u32>`. The existing `scan.rs` walker, `FiltersExt`, batching and `ScanMsg` stay. Remove the second `metadata()` call on the UI thread: use `FoundFile.modified`. Filter/sort produce `Vec<u32>` (0.4 ms for 50k in the measurement above, off-thread), cached by `(catalog_version, filter_hash)`.

**ThumbStore (L2).** Key = `blake3(path || size || mtime_ns || thumb_params_version)`; value = JPEG q80-85 (opaque) or PNG (alpha) at 256 px long side: 8-12 KB [M] -> 100k = ~1.2 GB instead of ~10 GB base64. Storage choice: one SQLite (WAL) or `redb` file with a `blobs(key BLOB PRIMARY KEY, w, h, fmt, bytes)` table (both crates exist in the local registry) beats 100k small files on NTFS; file-per-key with 2-level sharding (what `comfyui-android/src/gallery.rs` does for full images) is the simplest alternative. Load the set of present keys into memory at startup (100k x 8 B) so misses never touch disk. `get_many(visible)` per frame window, `put` batched. Optional read-through of `~/.cache/thumbnails/{normal,large}` (freedesktop spec) on Linux. The `thumbnails` DB table keeps AI metadata only; migrate existing `thumbnail_b64` once, in the background.

**Scheduler and pools.**
- Requests: `{ id, path, kind, stamp, priority, generation, cancel: Arc<AtomicBool> }` in a `BinaryHeap` behind `Mutex + Condvar`; priority = distance from viewport center. The UI submits the interest window each frame (diffed, O(changes)); items leaving the window+margin are cancelled; workers check `cancel` before open, after read, after decode.
- Pools: `decode` = `available_parallelism() - 2` (cap 16), `io` = 2-4 (network drives, archives, ffmpeg spawn), `video` = 2. Replace the current 3 + 6 fixed workers and the three duplicate producers (P-08).
- Every job runs in `catch_unwind`; pools get a `panic_handler`; failures become `ThumbState::Failed(reason)` shown on the tile and listed in a debug panel (replaces the "Failed thumbnails" tab).
- Results: `crossbeam::channel::bounded(256)`; the UI drains with a budget (<= 8 uploads or <= 2 ms per frame).
- Keep tokio for DB/network/AI; never decode in an `async` block without `spawn_blocking`.

**TextureCache (L1).** Port `ThumbCache` from `/home/shadowbroker/Documents/Rust/Mobile/comfyui-android/src/gallery.rs:1741-1960` (byte budget, insertion-order eviction with a "second chance" for tiles drawn in the last frame, `claim()`/`pending` set that stops per-frame request spam, `forget`/`forget_exact`). Budget 256-512 MB (about 1,100-2,200 thumbnails at 233 KB with mipmaps). Create textures yourself with `ctx.load_texture(name, ColorImage, opts)` from decoded RGBA and draw with `Image::from_texture(..).uv(cover_uv).paint_at(ui, rect)` as Luma does (`/home/shadowbroker/Documents/Rust/Mobile/Luma/src/app.rs:467-483`). This bypasses `egui_extras`' per-URI OS thread, `ColorImage` retention and byte-loader copy. Any remaining loader-based images (icons): set `ctx.options_mut(|o| o.reduce_texture_memory = true)` *after* `receive.rs:68` replaces the options (egui 0.37 `texture_loader.rs:90-100` then drops the byte and `ColorImage` copies after upload). GPU-friendly sizes: 256 px tier for grid tiles >= 160 px, optional 128 px tier for the 75 px list rows, 1024 px proxy for preview; mipmaps (`mipmap_mode = Some(Linear)`, supported by egui_glow `painter.rs:631`) only where minification exceeds 2x.

**Grid.** Custom virtualized grid instead of one table row per file: `ScrollArea::show_viewport`, visible row range = `viewport.min.y / row_h`. Two layouts: fixed square tiles with cover crop (works with unknown dimensions), and justified rows (needs `w,h`; one O(N) prefix-sum pass on a worker, then binary search from scroll offset to index). Placeholder states: pending (shimmer/average color), failed (reason on hover), ready. Keep `egui_data_table` for the details/list view, fed by the `View` indices; its selection, context-menu, hotkey and custom-action plumbing is worth keeping, but its row-level `filter_row`/`compare_cell` run on all rows each time the table is marked dirty (`validate_cc`, `egui-data-table draw/state/validation.rs:85-120`), so set `viewer.filter` and sort only through the cached `View`.

**Viewer.**
- Progressive: 256 px thumbnail immediately, 1024 px proxy (cached in L2 as a second tier, also reused by CLIP/SigLIP/JoyCaption instead of re-decoding originals, see 1.4), then full resolution.
- Large images: decode once on a worker into `Arc<RgbaImage>`; build a lazy 512-1024 px tile pyramid (box-filter mips); draw only intersecting tiles at the LOD matching zoom; LRU tile budget 128-256 MB; pointer-centered zoom and pan (Luma `view.pinch` pattern in `app/editor.rs`). Images up to ~8k can be one mipmapped texture (8000x4000 = 128 MB + 33%); above the GL max texture side (`ctx.input(|i| i.max_texture_side)`) tiles are mandatory.
- RAW: embedded preview first (instant), full decode afterwards if a decoder is available.

**Video.**

| Option | Use | Pros | Cons |
|---|---|---|---|
| `ffmpeg-next` in-process (feature `ffmpeg`, fixed per 4a) | poster frames, scrub strips, muted preview | verified: 20-34 ms 4K -> 256 px; no process spawn | needs FFmpeg dev libs; Windows needs vcpkg or the static `build` feature; GPL/LGPL packaging |
| `ffmpeg` CLI sidecar (Luma pattern: `native/raw_decode.cpp` and FFmpeg run out of process) | posters, strips, fallback | crash isolation, no link-time dependency; ~30-60 ms spawn per call [E] | one process per call; bundle or detect the binary |
| libmpv render API into a glow FBO via `PaintCallback` | full playback | hardware decode, A/V sync, seeking | integration effort; LGPL/GPL build |
| GStreamer | - | - | heavy; not recommended here |

Recommendation: poster frame at 10% via ffmpeg-next (code below) feeds the grid; scrubbing = a filmstrip of 8-12 frames at 160 px extracted once and stored in L2 under `(key, frame_idx)`, selected by hover x-position; playback pane = bounded(2) frame channel, pts-based pacing, seek via `Input::seek` + `decoder.flush()`, scaling to display size; hardware decode only for the full-size player later.

Verified poster-frame function (4K H.264 -> 256x144 in 20-34 ms; add `image` to the ffmpeg feature path; does not apply display-matrix rotation yet):

```rust
/// Poster frame for the grid: seek to `at_fraction` of the duration, decode one frame, scale to fit `max_dim`.
pub fn video_poster_rgba(path: &Path, max_dim: u32, at_fraction: f64) -> Result<image::RgbaImage, String> {
    ff::init().map_err(|e| e.to_string())?;
    let mut ictx = ff::format::input(path).map_err(|e| e.to_string())?;
    let (stream_index, params) = {
        let s = ictx.streams().best(ff::media::Type::Video).ok_or("no video stream")?;
        (s.index(), s.parameters())
    };
    let mut decoder = ff::codec::context::Context::from_parameters(params)
        .map_err(|e| e.to_string())?
        .decoder()
        .video()
        .map_err(|e| e.to_string())?;

    // Container duration is in AV_TIME_BASE (microseconds); seek lands on the previous keyframe.
    let dur = ictx.duration();
    if dur > 0 {
        let target = (dur as f64 * at_fraction.clamp(0.0, 0.9)) as i64;
        if ictx.seek(target, ..target).is_ok() {
            decoder.flush();
        }
    }

    let (sw, sh) = (decoder.width(), decoder.height());
    if sw == 0 || sh == 0 { return Err("zero-sized video".into()); }
    let scale = (f64::from(max_dim) / f64::from(sw.max(sh))).min(1.0);
    let (dw, dh) = (((f64::from(sw) * scale) as u32).max(2) & !1, ((f64::from(sh) * scale) as u32).max(2) & !1);
    let mut scaler = ff::software::scaling::Context::get(
        decoder.format(), sw, sh, ff::format::Pixel::RGBA, dw, dh, ff::software::scaling::flag::Flags::AREA,
    ).map_err(|e| e.to_string())?;

    let mut decoded = ff::frame::Video::empty();
    let mut out = ff::frame::Video::empty();
    for (_, pkt) in ictx.packets().filter(|(s, _)| s.index() == stream_index) {
        if decoder.send_packet(&pkt).is_err() { continue; }
        if decoder.receive_frame(&mut decoded).is_ok() {
            scaler.run(&decoded, &mut out).map_err(|e| e.to_string())?;
            let stride = out.stride(0);
            let data = out.data(0);
            let mut rgba = Vec::with_capacity((dw * dh * 4) as usize);
            for row in 0..dh as usize {
                rgba.extend_from_slice(&data[row * stride..row * stride + dw as usize * 4]);
            }
            return image::RgbaImage::from_raw(dw, dh, rgba).ok_or_else(|| "rgba size mismatch".to_string());
        }
    }
    Err("no frame decoded".into())
}
```

**RAW beyond ARW (5.6).** Options for full decode when no preview exists: LibRaw out of process as in `/home/shadowbroker/Documents/Rust/Mobile/Luma/native/raw_decode.cpp` (`preview` mode = `unpack_thumb`, `full` mode = demosaic; isolates crashes and memory), the pure-Rust `rawler` crate (not built or tested here; check API and licence), or `libraw-rs`. All keep the preview-first path above as the fast route.

**Instrumentation (debug UI).**
- `PerfHud` (toggle F3): frame-time ring buffer (last 240 frames), p50/p95/p99/max, repaint count, bar graph via `ui.painter()` polylines. Sketch (not compiled):

```rust
struct FrameHud { ring: [f32; 240], i: usize }
impl FrameHud {
    fn push(&mut self, dt_ms: f32) { self.ring[self.i % 240] = dt_ms; self.i += 1; }
    fn pct(&self, p: f32) -> f32 { let mut v = self.ring; v.sort_by(f32::total_cmp); v[((239.0 * p) as usize).min(239)] }
}
```

- Per-stage timers (log-bucket histograms): scan walk, stat, queue wait, file read, decode, resize, encode, store put, texture upload; top-20 slowest files with per-stage split.
- Cache panel: L1 entries/bytes/budget, hit rate, evictions per second, "thrash" counter (evicted then re-requested within 2 s), L2 hit rate, in-flight and queued by priority, cancelled count, pool busy workers, failure list with reasons.
- Memory: RSS (existing `system_mem_mb`), `ctx.loaders_ui(ui)` (egui built-in loader byte sizes), estimated GPU bytes.
- Optional `puffin`/`tracing` feature; `egui_kittest` 0.37 (present in the local registry) for headless frame-time regression tests; a `--bench-grid <dir|N>` CLI that replays a scripted scroll and prints percentiles.
- Already present and worth merging into the HUD: scan batch timings (`quick_access_pane.rs:278-396`, `perf_batches`), `SMART_MEDIA_SCAN_TIMING=1` (`scan.rs:240`), status indicators (`ui/status.rs`).

### 5.4 Sizing at 100k files [E unless noted]

| Item | Current | Target |
|---|---|---|
| Thumbnail bytes stored | ~10 GB base64 PNG (77 KB PNG [M] x 1.33 x 100k) | ~1.2 GB JPEG q82 (12.3 KB [M]) |
| RAM for thumbnails | 20-80 GB depending on view | L1 budget 256-512 MB + catalog ~20 MB |
| Cold index time, 24 MP photos at ~100 ms decode | 2 pipelines x 100k x 100 ms / 9 threads = ~37 min (release); hours in dev | 100k x 100 ms / 28 threads = ~6 min; warm visits = L2 hits (~1 ms each) |
| UI work per frame | O(N) several times | O(visible + margin) |
| Directory page from DB | 280-457 ms per 3,000 rows [M] | 15-18 ms per 3,000 rows [M] |

### 5.5 Keep / change / delete

| Keep | Change | Delete |
|---|---|---|
| `scan.rs` walker, batches, `ScanMsg`, filters; `filtering.rs` | `thumbs.rs`: return RGBA/encoded bytes, drop base64 and the `data:` prefix; keep Windows Shell fallback and `generate_image_preview_png` signature | `recursive_prefetch_*`, `prefetch_next_page_thumbs`, `schedule_missing_thumbs_for_current_page`, FoundBatch tokio spawns, `thumb_scheduled`, `thumb_semaphore` |
| `archive.rs` handler trait/registry and listing | archive thumbnails: stream entry into memory, one open per archive, tar index, cap entry size | `db_page_cache` (or cap at 2 pages without pixels), `db_lookup` pixel copies, `pending_thumb_rows` base64 persistence |
| DB schema for AI metadata, HNSW embeddings, groups, settings | `thumbnails` row loses `thumbnail_b64`; dir queries select fields; upserts batched into one statement | `last_scan_rows` / `recursive_filtered_rows` as `Vec<Thumbnail>` |
| `egui_data_table` fork for the details view (selection, hotkeys, context menu, custom actions) | row model becomes indices into the catalog; viewer reads from catalog | overloaded `thumbnail_tx: Sender<Thumbnail>`; replace with a typed `UiEvent` enum (Select, Navigate, PreviewReady, PasswordRequired, ThumbReady) |
| Navigation/back-forward, quick access, menus | `receive_*` loops get count/time budgets | per-tab caches (share one global L1/L2) |
| New `raw_decode.rs` (section 4b) | FFmpeg thread -> bounded channel, pts pacing, seek | |

### 5.6 Phasing

1. Quick wins (section 6, S items): no layout dependency.
2. Behind the current table: ThumbStore + unified scheduler + TextureCache; thumbnails leave the rows; JPEG encoding.
3. New grid/viewer on the reorganized UI; HUD lands with step 2 so every later step is measurable.
4. Tiled viewer, filmstrip scrubbing, video playback, RAW full decode.

---

## 6. Quick wins before the UI reorganization

Ordered by value over effort. Effort: S <= 1 day, M 1-3 days, L > 3 days.

| Rank | ID | Change | Effort | Impact | Risk | Files |
|---|---|---|---|---|---|---|
| 1 | QW-1 | `[profile.dev] opt-level = 1` and `[profile.dev.package."*"] opt-level = 3` | S | decode 10.5 s -> 0.13 s, status bar 112-203 ms -> 12-22 ms per frame in dev [M]; slower first clean build only | low | `Cargo.toml:107-109` |
| 2 | QW-2 | RAW fix: panic boundary, pool `panic_handler`, preview-first decode, surface errors (section 4b) | S | removes the process abort; ARW thumbnail 13 ms | low; CR3/RAF/RW2 untested | `raw_decode.rs`, `thumbs.rs`, `scan.rs` |
| 3 | QW-3 | ffmpeg compile fix (section 4a diff) | S | `--features ffmpeg` builds | none | `preview_pane.rs` |
| 4 | QW-4 | O(1) lookups: path -> index maps for `last_scan_rows` and `recursive_filtered_rows`; use `table_index` in `receive_thumbnail`/`receive_ai_update`, validating `rows[idx].path == key` and rebuilding on mismatch (P-21) | S | 10.2 s -> ~0 at 100k [M] | low | `receive_scan.rs:630-645`, `receive_thumbnail.rs:32,64`, `receive_ai_update.rs:72` |
| 5 | QW-5 | `apply_recursive_filter_and_sort`: filter indices first, clone only survivors (or store indices); treat empty result as valid cache; drop the per-frame `format!` signature | S | 118-872 ms per recompute -> <1 ms [M] | low | `mod.rs:484-487,1200-1252` |
| 6 | QW-6 | Status bar: compute counts once per (table version, filter hash, selection version), not per frame | S | 4-45 ms/frame [M] -> ~0 | low | `navbar.rs:336-402` |
| 7 | QW-7 | In-flight set + negative cache for hi-res preview and image-edit preview | S | stops per-frame decode storms and infinite retry | low | `preview_pane.rs:173-199`, `image_edit.rs:455-478` |
| 8 | QW-8 | `receive_thumbnail`: bounded `while` drain (e.g. 64 msgs or 2 ms); `on_highlight_change` sends only the last highlighted row; remove 3 info logs per message; batch `load_logical_group_by_name` | S | 5,000-row selection/group load 83 s -> instant [E] | low | `receive_thumbnail.rs:8`, `table/mod.rs:757-766`, `load.rs:76` |
| 9 | QW-9 | UI-thread IO: use `FoundFile.modified` in `file_to_thumbnail`; move the skip-icons retain to a worker or use size only, and hide instead of delete | S | no per-row `stat`/open on the UI thread | low | `thumbs.rs:245`, `filter_menu.rs:226-246` |
| 10 | QW-10 | Cancel the previous scan whenever a new scan/navigation starts (`cancel_scan(old_id)`), make cancel per-scan `AtomicBool` | S | stops stale decode work and queue head-of-line blocking | low | `explorer/mod.rs:139-145`, `scan_menu.rs`, `scan.rs:14-33` |
| 11 | QW-11 | Encode thumbnails as JPEG q82-85 when the image has no alpha, PNG otherwise (decoders sniff the format, so old rows still load); strip the `data:` prefix everywhere (also fixes P-24) | S | 5-8x fewer bytes in RAM/DB/channels [M]; encode 0.4-1.4 ms | low | `thumbs.rs:82-96`, `raw_decode.rs`, `assistant.rs:1819` |
| 12 | QW-12 | Cap `thumb_cache` by bytes (reuse `ThumbCache` logic) and call `ctx.forget_image(uri)` on eviction; set `reduce_texture_memory = true` after `receive.rs:68`; cap `db_page_cache` to 2 pages | S/M | bounds RSS | medium (re-decode on scroll back) | `table/mod.rs:44`, `receive.rs:68`, `receive_preload.rs:28` |
| 13 | QW-13 | Linux video thumbnails via ffmpeg-next poster function or `ffmpeg` CLI (replaces the `Err`) | S/M | videos get thumbnails; Failed tab stops filling | low | `thumbs.rs:104-109`, `scan.rs:168-179` |
| 14 | QW-14 | Remove the duplicate pipeline: keep `THUMB_POOL` as the single producer, fix `finish()` / `drain` so unflushed jobs are not dropped, size the pool from cores | M | halves CPU for scans | medium (must not lose jobs) | `scan.rs`, `receive_scan.rs:382-387,525-570` |
| 15 | QW-15 | Dir listing query selects fields (not `*`) and per-row `get_embedding` becomes a single `SELECT VALUE thumb_ref` for the page | M | removes 100 KB per row from page loads; N queries -> 1 | low | `thumbnails.rs:202-207`, `receive_thumbnail.rs:74-86` |
| 16 | QW-16 | Persistent L2 thumbnail store consulted before decode; DB rows stop carrying pixels (separate table: 15-18 ms per 3,000 rows [M]) | M/L | kills re-decode per visit and 500 MB pages | medium (migration) | new `thumb_store.rs`, `thumbnails.rs`, `scan.rs` |
| 17 | QW-17 | Perf HUD + cache panel (section 5.3) | M | makes every other change measurable | low | new `ui/perf.rs` |
| 18 | QW-18 | Archive thumbnails: single open per archive, stream into memory with a size cap, tar index | M | removes O(N^2) tar; no temp files | medium | `archive.rs:241-441`, `explorer/mod.rs:29-91` |
| 19 | QW-19 | Typed `UiEvent` channel replacing overloaded `thumbnail_tx` | M | clearer ordering, no row clones per selection | medium (touches many call sites; coordinate with the UI reorganization) | `mod.rs`, `table/mod.rs`, `receive_thumbnail.rs` |

Items 1-11 are small, local diffs that do not depend on the layout. Items 12-15 change data flow but not visuals. 16-19 are the bridge to the new architecture.

---

## 7. Limits of this audit

- No GUI session was run. Frame-time figures come from replaying the exact loops (`uibench`), not from the app; real impact also depends on egui layout cost. The dev-profile figures assume you run `cargo run` in the default dev profile (only `target/debug` exists).
- Memory per thumbnail uses one real photo (77 KB PNG) plus synthetic/CG images (16-89 KB); typical camera photos are probably in the same range or larger.
- DB numbers are embedded SurrealKV on tmpfs in release; a disk-backed, debug-built app will be slower.
- Not verified: CR3/RAF/RW2/ORF preview extraction (no samples), HEIC/AVIF decode failures (read from feature flags), `table_index` staleness (P-21), assistant prefix bug (P-24), 7z solid-archive cost, Windows build of ffmpeg-next, hardware-accelerated decode, `rawler` API.
- A full `cargo test` or GUI run of the patched copy was not performed; `cargo check` (default and `--features ffmpeg`) passes on it and the RAW module was run standalone against the real file.

## 8. Artifacts (scratchpad: `/tmp/claude-1000/-home-shadowbroker-Documents-Rust-AiMediaExplorer/6e7b9214-e630-4117-abc2-33cd4637f368/scratchpad`)

| Path | Content |
|---|---|
| `repo-copy/` | source copy with all proposed patches applied (not the repo) |
| `research/parts/ffmpeg.diff`, `thumbs_scan.diff`, `raw_decode.rs`, `poster.rs` | exact patch text used above |
| `rawfix/`, `ffcheck/`, `ffposter/` | RAW reproduction and fix tests, ffmpeg error reproduction, poster-frame prototype |
| `thumbbench/`, `uibench/`, `dbbench/` | benchmark sources |
