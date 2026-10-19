# Study 3: Luma patterns for AiMediaExplorer

Scope: what Luma (`/home/shadowbroker/Documents/Rust/Mobile/Luma`, v0.1.17, egui 0.37) does for photo/video rendering, and what AiMediaExplorer (AME, `/home/shadowbroker/Documents/Rust/AiMediaExplorer`, HEAD 1d1a1db, egui/eframe 0.37 on the glow backend) should port. Theme, styling and icon-pack work is out of scope; where an icon/loader API appears it is only because the brief asked how Luma uses `egui_extras` loaders.

Citation forms: `Luma:file:line`, `AME:file:line`, `egui-0.37:file:line` (cargo registry source at `~/.cargo/registry/src/*/egui-0.37.0`), `EguiMobile:file:line` (`~/.cargo/git/checkouts/eguimobile-02e6d2d1c2b802d0/df95504`, the rev Luma pins).

Evidence tags: **[V]** read in code, **[E]** checked in egui/egui_extras 0.37 source, **[I]** my inference from code (not measured), **[U]** unverified.

Nothing was built or run. Luma has no measured frame-time data (`Luma:VERIFICATION.md:638` says no performance benchmark exists), so every "keeps frame times low" claim below is a design bound visible in code, not a measurement.

---

## 1. Summary

1. **Luma's photo viewer and video player are not egui-rendered.** Full-resolution photo viewing is a native Android `View` (`PhotoView.java`, BitmapRegionDecoder tile cache) inside a `PopupWindow` positioned over an egui "stage" rect (`Luma:GalleryController.java:524-526, 556-563`; `Luma:app.rs:1450-1468`). Video is LibVLC -> MediaCodec -> SurfaceTexture -> an `egui_glow` PaintCallback (`EguiMobile:crates/egui-android/src/video/surface.rs`). The README says so: "no full-frame readback or egui pixel uploads during playback" (`Luma:README.md:23`). AME cannot copy these; it needs the algorithms re-implemented over egui textures.
2. **What egui does in Luma:** thumbnail grid and rail, the photo-editor canvas (texture + crop/mask overlays), the video timeline/filmstrip/overlay editor, dialogs, progress. These are the portable parts.
3. **Best transferable assets, in order of value to AME:**
   - bounded LRU `TextureHandle` cache + in-flight request gating (`Luma:app.rs:450-466, 423-435`);
   - the disk thumbnail cache design (identity-hashed filenames, atomic write, mtime-LRU prune, version suffix) (`Luma:MediaFiles.java:105-114, 292-302, 510-522`);
   - `ViewXform` zoom/pan math (`Luma:mask.rs:151-221`) and shared crop overlay (`Luma:editor.rs:770-872`);
   - proxy-edit / full-res-export "recipe" architecture (`Luma:edit.rs`, `editor.rs:212-246`);
   - the headless egui test harness and AccessKit rig (`Luma:app.rs:2672-2808`, `app/studio.rs:3371-3486`);
   - `Selection` range-drag model (`Luma:model.rs:42-112`).
4. **AME has no grid/gallery view today** (table only, 75 pt rows: `AME:ui/file_table/mod.rs:748`). Luma's `library()` is the template for a new one.
5. **AME's thumbnail/preview plumbing has verified defects that Luma's approach avoids** (section 8): `bytes://` URIs reused for changing content (stale image), unbounded decode/texture caches with no eviction, a decode OS thread per image, one-thumbnail-per-frame receive, spawn-per-frame preview tasks, no video thumbnails on non-Windows, full RAW demosaic for 256 px thumbnails.
6. **Luma has no histogram, no scopes, no before/after slider, no mouse-wheel zoom, no keyboard shortcuts beyond Escape/Back, no context menus.** Those are net-new for AME desktop. Luma's colour tooling is a pure-Rust 11-control grade compiled to a 33^3 LUT used only by the video path (`Luma:color.rs`), while the photo editor uses a 3-control CPU recipe.
7. **Do not copy Luma's per-frame O(N) list handling** (`visible_items()` clones the filtered list every call, `Luma:app.rs:616-636`): fine for a phone gallery, wrong for a database-backed explorer.

---

## 2. Where Luma renders what

| Concern | Who draws it | Evidence |
|---|---|---|
| Library grid, album covers, thumbnail rail | egui (textures + painter) | `Luma:app.rs:651-1059, 1311-1394` |
| Thumbnail generation | Java worker (1 thread) -> JPEG on disk -> Rust loads into egui texture | `GalleryController.java:35, 439-451`; `app.rs:261-282` |
| Full-res photo viewer (zoom/pan/tiles/swipe) | Native `PhotoView` in a `PopupWindow` over the egui stage | `PhotoView.java`; `GalleryController.java:524-563` |
| Video frames | GPU SurfaceTexture painted by `GpuVideoSurface` PaintCallback | `app.rs:1416-1434`; `surface.rs:247-320` |
| Photo editor canvas, crop handles, mask overlay | egui (one image texture + painter shapes + one overlay texture) | `editor.rs:517-697` |
| Photo editor preview pixels | Rust CPU (`Recipe::preview`) on a <=1600 px proxy | `edit.rs:87-109`; `editor.rs:212-246` |
| Photo export | Java `PhotoEdits.render` on full-res decode (<=32 MP) | `PhotoEdits.java:96-120` |
| Video timeline/filmstrip/overlays | egui painter | `app/studio.rs:1629-1944` |
| Video colour grade preview | GPU LUT in the video shader; FFmpeg `lut3d` for export | `app.rs:196-206`; `surface.rs:17-31` |
| Event/state transport | JSON `bridge::poll()` each frame + `bridge::send()` | `bridge.rs`; `app.rs:208-436` |

Screenshots checked (`Luma:evidence/manage-0.1.4/drag-selection.png`, `editor-0.1.4/photo-editor-landscape.png`, `navigation-0.1.4/landscape-photo-sidebar.png`): portrait grid with selection badges + bottom action bar; landscape photo viewer = stage left + 260 pt sidebar with controls and thumbnail rail; photo editor landscape = canvas left + tab column (Crop/Adjust/Qwen edit) right.

---

## 3. How Luma loads, caches and draws (answers to brief item 1)

### 3.1 Thumbnail textures: sizes, lifetimes, eviction [V]

Flow: grid cell calls `thumb(item, at)` -> miss -> send `{"op":"thumb","key","item","time","width":384}` to Java -> Java writes a 384 px JPEG (q88) in its cache dir -> emits `thumb` event with a file path -> Rust reads the file, `image::load_from_memory`, builds `ColorImage`, `ctx.load_texture(key, .., LINEAR)` and stores `(TextureHandle, last_used_frame)` (`app.rs:261-282`).

| Item | Value | Where |
|---|---|---|
| Cache | `HashMap<String,(TextureHandle,u64)>`, frame stamp updated on every hit | `app.rs:80, 450-455` |
| Key | `uri:modified:size:at_ms` (changes when file changes) | `model.rs:33-35`; test `model.rs:470-481` |
| Eviction | when `len > 180`, sort by stamp, drop oldest until 150 (runs once per `pump`) | `app.rs:423-433` |
| In-flight map | `requested: HashMap<key,Instant>`; skip if requested <45 s ago; hard cap 80 entries; entries pruned after 45 s | `app.rs:456-463, 434-435` |
| Java-side cap | 64 pending keys, 1-thread pool | `GalleryController.java:35, 441` |
| Decode location | UI thread, inside `pump()`, tiny JPEG (384 px) | `app.rs:263-280` |
| Placeholder | PANEL rect + kind icon while loading | `app.rs:468, 479-481` |
| Cover crop | UV rectangle, no CPU crop | `app.rs:467-483` |

Cover-fit by UV (essential excerpt):

```rust
let factor = (rect.width() / source.x).max(rect.height() / source.y);
let visible = rect.size() / (source * factor);
let uv = Rect::from_center_size(pos2(0.5, 0.5), visible);
egui::Image::from_texture(&texture).uv(uv).tint(tint).corner_radius(6).paint_at(ui, rect);
```

Notes:
- Requests only fire for drawn cells (virtualized rows, plus `ui.is_rect_visible(rect)` guard in non-virtualized pickers, `app/studio.rs:2706, 2762`), so scrolling does not queue off-screen work.
- Video frame thumbnails use the same cache with `at` quantized to 100 ms (`app/studio.rs:1766`) to bound distinct keys.
- No priority/cancel for requests that scroll off-screen; the 1-thread executor and caps bound the damage. **[I]**
- Per-cell `thumb_key()` allocates a `String` every frame **[V]**; AME should key by a precomputed `u64` hash.

### 3.2 Grid layout and virtualization [V]

- Columns: `cols = clamp(floor(avail_width / 118), 3, 8)`, `side = (avail_width - 5*(cols-1)) / cols`, square cells, 5 pt gap (`app.rs:973-974`). Albums: 180 pt min, 2..5 cols, 12 pt gap (`app.rs:877-878`).
- Virtualization: `ScrollArea::vertical().show_rows(ui, side, rows, |ui, range| ..)`; each row is `ui.horizontal` of `allocate_exact_size` cells (`app.rs:975-1054`).
- **Gotcha:** `show_rows` adds `item_spacing.y` itself, so the passed row height must equal the drawn row height; a mismatch made album rows jump 15.84 pt at offset 213 (`app.rs:886` comment; `VERIFICATION.md:515-517`). A headless regression test covers it (`app.rs:2580-2670`).
- Cell chrome is painter-only: strip overlay (`rect_filled` alpha 190), label via `text_at` (single-row `LayoutJob`, `break_anywhere`, painted with `painter.galley`), kind icon, favourite star, `pick_badge` circle, selected stroke (`app.rs:998-1047, 2132-2155`). No per-cell widgets except one `Sense::click` rect, which keeps per-cell cost low. **[I]**
- Selection inset animates with `ctx.animate_bool_with_time(Id, picked, 0.12)` (`app.rs:990-996`).
- Horizontal thumbnail rail is virtualized by hand with `show_viewport`: compute `first/last` index from `viewport.left()/right()`, draw only that range, `scroll_to_rect(..., Some(Align::Center))` only on selection change or width change so manual browsing never snaps back (`app.rs:1311-1394`).
- Non-virtualized `egui::Grid` is used only for small pickers (<=240 items, `app/studio.rs:2696, 2703`).

### 3.3 Photo viewer (native, but the algorithm is the reusable part) [V]

`PhotoView.java` (219 lines):
- Two layers: a low-res `preview` bitmap (decoded with power-of-two `inSampleSize` until max side <=1600, `:81-84`) always drawn; sharp tiles drawn over it.
- Tiles: `sample` = largest power of two with `sample*2*scale <= 1`; tile edge `512*sample` source px; key `generation:sample:x:y`; only tiles intersecting the inverse-mapped viewport are requested (`:184-210`).
- Caches/limits: `LruCache` 24 MB by `getAllocationByteCount` (`:24-26`), max 32 pending tile jobs (`:195`), single decode thread (`:22`).
- Cancellation: monotonically increasing `generation` token checked on the worker and again when posting results (`:73, 91, 200-206`); old decoder recycled off-thread.
- Gestures: pinch scale clamped `[fit, 8]` about the focus point (`:41-48`), pan clamp (`:143-148`), double-tap fit/1:1 toggle (`:52-56`), swipe-to-navigate only at fit zoom, threshold `max(48dp, min(18% width, 120dp))`, horizontal dominance 1.5x, never after multi-touch (`:149-168`).
- Fallback for formats BitmapRegionDecoder rejects (GIF/BMP): `ImageDecoder` to <=3200 px software bitmap (`:107-125`).
- Open sequence: RAW shows a 1600 px embedded-preview first, swaps to the developed full-res when ready (`GalleryController.java:530-540`).

Porting implication **[E]**: `Context::load_texture` has a `debug_assert!` that both sides <= `max_texture_side`, default 2048 until the backend reports otherwise (`egui-0.37:src/context.rs:2822-2830`; `src/input_state/mod.rs:350`). Anything bigger than the GPU limit needs tiling or downscaling. Android's `BitmapRegionDecoder` has no desktop equivalent in the `image` crate; a desktop tile viewer must decode once to RAM (or a pyramid) and upload 512 px tile textures. **[U]** on cost for 100 MP files.

### 3.4 Photo editor canvas [V]

Architecture (`editor.rs`, `edit.rs`, `PhotoEdits.java`):
- **Proxy edit, full-res export.** Java decodes an EXIF-oriented, sRGB-converted PNG capped at 1600 px (`PhotoEdits.java:76-90`); Rust holds it as `RgbaImage` (`editor.rs:123-137`). Edits are a small `Copy + PartialEq` `Recipe {crop, rotation, flip, exposure, contrast, saturation}` (`edit.rs:62-83`). Export re-renders the recipe on a full decode (<=32 MP) in Java (`PhotoEdits.java:96-120`); the Rust preview comment says it "matches PhotoEdits.colorMatrix" (`edit.rs:86`) and tests pin rotation/flip pixel order and crop dimensions (`edit.rs:151-179`).
- **Equality-gated rebuild.** `prepare_texture` only re-runs `Recipe::preview` (clone + per-pixel loop) and re-uploads when `(recipe, crop_mode)` changed (`editor.rs:212-246`). CPU cost per slider change is a full-proxy clone plus a float loop on the UI thread **[I]**; not measured by Luma.
- **Crop mode shows the uncropped transformed image** with an overlay; other tabs show the cropped image (`editor.rs:226-230`).
- **Canvas draw:** `painter.with_clip_rect(area).image(tex.id(), view, uv(0,0)-(1,1), WHITE)` (`editor.rs:559-564`); fit rect = aspect fit in the allocated area (`editor.rs:534-535`).
- **Zoom/pan:** `ViewXform {zoom, pan}` in screen points relative to the fit rect, `MAX_ZOOM` 8, `pinch(fit, area, factor, focus, translate)` keeps the content under the focus point fixed then clamps pan (`mask.rs:149-221`). Fed from `ui.input(|i| i.multi_touch())` with `zoom_delta` and `translation_delta` (`editor.rs:539-558`). No unit tests exist for `ViewXform` in Luma **[V]**.
- **Crop:** shared photo/video `crop_overlay`: four dim rects (alpha 165), 1.5 pt stroke, 10x10 corner squares, rule-of-thirds lines; drag grabs the nearest corner within 30 pt, else the interior moves; resizing keeps min size 0.02 (`editor.rs:770-872`). Only corners + interior; no edge handles, and aspect presets set the starting rect only (`Crop::aspect`, `edit.rs:33-49`), so dragging a corner breaks the ratio **[V]**.
- **Mask overlay:** `MaskCanvas` is a u8 grid capped at 512 px on its long side (`editor.rs:237-243`); strokes are recorded as `StrokeRec` lists and the mask is the deterministic rasterization, so undo = rasterize all but the last group (`mask.rs:104-121`; `editor.rs:431-433`). The overlay is a separate RGBA texture (pink, alpha `m/2`) updated in place via `TextureHandle::set` and drawn with the same `view`/UV as the photo (`editor.rs:699-712, 664-671`). `bake_alpha` writes `255 - mask` into the source PNG's alpha for ComfyUI inpainting (`mask.rs:244-262`).
- **Before/after:** a toggle (`selectable_value Before / Qwen result`) swapping which texture is drawn (`editor.rs:327-337, 529-533`); video has hold-to-compare (`app/studio.rs:2565-2566`). There is no split slider.
- **Guard rails:** `busy` disables the controls (`editor.rs:258, 339`); leaving with unsaved edits opens a confirm window (`editor.rs:184-199, 301-316`); every async event carries a `session` string and mismatching events are ignored (`editor.rs:119-122`); results are held as `Option<(path, TextureHandle)>` and shown after review, then "Save copy" writes a new file, never the original (`editor.rs:150-167, 264-269`).
- **Responsive split:** two columns only when `available.x >= 600 && available.x > available.y*1.5`; otherwise stack with stage height `available.y - 290` (`editor.rs:274-299`). Comment explains a tall keyboard can make a portrait viewport look landscape-shaped.

### 3.5 Histogram and colour tools [V]

- **No histogram, vectorscope, waveform scope or clipping overlay exists anywhere in Luma** (grep over `src/`, `java/`, docs).
- `color::Adjust` (11 controls: exposure, brightness, contrast, highlights, shadows, whites, blacks, temperature, tint, vibrance, saturation) is pure Rust in linear light with `develop/apply`, `cube()` (.cube text for FFmpeg `lut3d`) and `strip()` (RGBA8 1089x33 = 33^3 LUT unrolled) (`color.rs:1-151`). Table-driven slider list `CONTROLS` with fn-pointer field accessors (`color.rs:29-41`) is rendered by a 10-line loop with custom formatter/parser (`app/studio.rs:2570-2582`).
- Tests are strong and portable: neutral = identity, exposure doubles linear light, every control keeps grey ramps monotonic, controls move the tones they name, `cube` and `strip` sample the same grid (`color.rs:153-247`).
- **Use:** only the video path (GPU LUT in the shader via `set_lut`, `app.rs:196-206`; FFmpeg `lut3d`). The photo editor's Adjust tab is the 3-slider `Recipe` (`editor.rs:363-376`).
- GPU grade shader samples the strip with manual trilinear along blue (`surface.rs:17-31`).

### 3.6 Video and frame handling [V]

- Playback is native (see 3.1 summary). Rust owns only a `Timeline` model and the transport UI.
- `Timeline` (`model.rs:235-295`): fractional-fps-safe frame stepping (`step_time` rounds the frame index, test uses 30000/1001, `model.rs:299-318`), zoom window `(start,end)` clamped for short/empty clips (`:274-294`, test `:340-354`), in/out marks always ordered and inside the clip (`:285-294`, test `:321-337`), `timecode()`.
- Scrub (`app.rs:1844-1916`): press origin picks Seek/In/Out handle (22 pt), dragging up 60/120 pt switches to 5x/20x precision (`:1866-1872`), preview loupe requests throttled to 100 ms (`:1885`), the pointer is paused during drag and resumed on release, and `pending_seek` suppresses reported-position updates until the player reaches the target or 2 s pass (`:401-419`). Studio throttles seeks to 90 ms (`app/studio.rs:2044`).
- Filmstrip: tiles drawn only for the visible x-range, each tile a `thumb()` texture with UV cover (`app/studio.rs:1742-1773`).
- Waveform columns use the **peak** of each bin, not decimation, so transients survive zoom-out (`edit.rs:121-145`, test `:181-192`).
- Drag release gotcha: egui-winit releases the button at the last delivered move and only `Touch{End}` carries the lift point, so fast swipes lose 60-160 ms; Luma reads the lift position from the event list (`app/studio.rs:807-822`; `VERIFICATION.md:165-167`).

### 3.7 DNG / RAW [V]

- LibRaw runs as a separate process (OOM isolation on mobile): `preview` = `unpack_thumb` + `dcraw_thumb_writer` (embedded JPEG), `full` = `unpack` + `dcraw_process` (camera WB, 8-bit sRGB, AHD `user_qual=3`, `no_auto_bright`) -> PPM (`native/raw_decode.cpp:1-42`).
- Thumbnails and the 1600 px first view use the **embedded preview**, scaled with FFmpeg; full development happens afterwards and is cached as a high-quality JPEG (q:v 1, yuvj444p) (`MediaFiles.java:333-349, 366-379`; `GalleryController.java:530-540`).
- Galaxy DNG 1.7 (JPEG XL linear RAW) needs the Adobe DNG SDK helper; LibRaw alone failed (`VERIFICATION.md:496-514`; `native/dng/`).
- Cache keys carry a version suffix (`:frame-v2`, `:full-v1`, `:dng-full-v1`) so a decoder fix invalidates old files (`MediaFiles.java:294, 367`).
- Edits operate on the rendered 8-bit sRGB image, not sensor data (`README.md:89-90`).

### 3.8 Background work and progress [V]

- One JSON state blob (`self.native`) polled each frame; keys like `exporting`, `progress`, `export_status`, `managing`, `manage_progress`, `photo_saving`, `qwen_busy`, `scanning` (`app.rs:208-210`, `2309-2377`).
- Presentation patterns: top `ProgressBar` with text and a cancel icon (`app.rs:2309-2322`); floating `Area` (Foreground, anchored top-centre) with progress + Stop (`app.rs:2350-2377`); toast `Area` bottom-centre for 9 s, interactive only when it carries Undo (`app.rs:2378-2419`); `ui.spinner()` next to counts while scanning (`app.rs:748-750`).
- Export runs on a `std::thread` and returns through `mpsc::channel`, polled with `try_recv` each frame (`app/studio.rs:3045-3072, 1157-1165`).
- Repaint cadence: `request_repaint_after` 16 ms while playing, 50 ms with an item open or busy, 180 ms idle (`app.rs:2420-2427`). This exists because Luma must poll JNI; AME's channel-driven `request_repaint()` model is already event-driven.

### 3.9 Gestures vs mouse/keyboard: portable or not

| Luma input | Portable? | Desktop replacement |
|---|---|---|
| `Response::long_touched()` to start selection, `max_click_duration=0.5` hack (`app.rs:652-654, 1070`) | No. In egui 0.37 `max_click_duration` doubles as the long-touch->secondary-click threshold (`egui-0.37:src/input_state/mod.rs:71-74`) | Ctrl/Shift-click, marquee, checkbox column |
| `scroll_source(ScrollSource::ALL)` drag-to-scroll (`app.rs:882, 977`) | No. 0.37 default `DragScroll::OnTouch` is right for desktop (`egui-0.37:scroll_area.rs:217-228`) | default |
| `multi_touch()` pinch/rotate/translate (`editor.rs:539`, `app/studio.rs:1486`) | Math yes, source no | `ui.input(|i| i.zoom_delta())` (merges ctrl+wheel and trackpad pinch, `egui-0.37:input_state/mod.rs:557-565`), `smooth_scroll_delta`, middle/space drag |
| Stylus pressure/eraser/palm rejection (`editor.rs:579-659`, `mask.rs:223-242`) | Optional (tablet users) | `egui::Event::Touch{force}` exists on desktop with pen input **[U]** |
| Haptics, safe-area, IME toolbar, `Hand` layout | No | none |
| Drag-range selection (`model.rs:42-112`) | Yes (pure) | extend with anchor via Shift, add marquee |
| Escape/Back consumes one "layer" (`app.rs:562-594, 2291-2296`) | Yes | Esc closes innermost panel/modal |
| Mouse wheel zoom, hotkeys, right-click, hover highlight | Not present in Luma | new |

### 3.10 `egui_extras` loaders and custom painters [V]

- Cargo: `egui_extras 0.37` with `default-features=false, features=["image","svg"]`; `image 0.25` with only `jpeg`, `png` (`Luma:Cargo.toml:13-15`). `egui_extras::install_image_loaders(ctx)` is called once in `theme()` (`app.rs:2266`).
- Loaders are used **only for static SVG icons**: `Image::from_bytes(format!("bytes://luma/{name}.svg"), include_bytes!(..)).fit_to_exact_size(..).tint(color)` (`icons.rs:130-134`). Photos/thumbnails/results never go through `bytes://`; they are `TextureHandle`s created with `ctx.load_texture`. This is the key contrast with AME (section 8).
- Custom painting: `Painter::{rect_filled,rect_stroke,hline,vline,circle_filled,galley,image}`, `egui::Mesh` for gradients (`app/studio.rs:849-857, 1800-1806`), `Shape::Noop` placeholder + `painter.set(idx, shape)` to draw a glow behind a widget after the widget is laid out (`app.rs:724-727`; `icons.rs:137, 147`), `Shadow::as_shape` (`icons.rs:162-170`).
- Custom GPU painting exists only inside EguiMobile's `egui_glow::CallbackFn` (see P30).

### 3.11 What actually keeps Luma's frame times low (evidence-based) [V]/[I]

1. Virtualized rows and manual viewport culling everywhere (grid, rail, timeline, filmstrip).
2. Small, bounded texture working set: 384 px thumbs, <=180 textures, 512 px mask canvas, 1600 px editor proxy.
3. Expensive work is off-thread or out-of-process: thumbnails, RAW develop, FFmpeg, whisper, export; the UI thread only decodes tiny JPEGs (<=64 outstanding).
4. Re-render keyed on equality/hash: editor texture on `(Recipe, crop_mode)`; overlay text/picture textures on a hash of the serialized clip + frame size (`app/studio.rs:507-555`); LUT rebuilt only when `Adjust` changes (`app.rs:197-200`).
5. In-place texture updates (`TextureHandle::set`) for overlays and previews instead of allocating new handles (`editor.rs:708`, `app/studio.rs:532`).
6. Hard caps: 80 in-flight thumb requests, 32 tile jobs, 24 MB tile cache, 120 undo steps, 12k-char JSON dump.
7. No GPU readback for video.
8. Not low-cost: `visible_items()` O(N) clone per call per frame (`app.rs:616-636`, called at `:665`, `:813`, `:1312`), per-cell `String` key allocation, per-slider-tick full-proxy CPU recolour. These are acceptable only because Luma's libraries are phone-sized.

---

## 4. Pattern catalogue (reusable on desktop)

Size: **S** < ~200 LOC / ~1 day, **M** ~200-600 LOC / 1-3 days, **L** larger. Sizes are my estimates **[I]**.

### Textures, grids, selection

| ID | Pattern | Source | Notes for AME | Size |
|---|---|---|---|---|
| P1 | LRU `TextureHandle` cache with frame stamps, evict 180->150 | `Luma:app.rs:80, 261-282, 423-435, 450-466` | Replace `bytes://` + `thumb_cache`. Prefer a `u64` key (blake3 of path+mtime+size+variant); consider `lru` crate (0.18 is in the local registry **[U]** whether AME can use it) | M |
| P2 | In-flight request map with timeout + cap; request only drawn cells | `app.rs:456-463, 434-435`; `app/studio.rs:2706` | Replaces `thumb_scheduled` whole-page scheduling (`AME:file_table/mod.rs:1366-1441`) | S |
| P3 | UV cover crop + `corner_radius` | `app.rs:467-483` | Direct | S |
| P4 | `show_rows` with fixed square cells + row-height/spacing rule | `app.rs:973-979, 886` | New grid view | M |
| P5 | Manual viewport virtualization for horizontal strips | `app.rs:1311-1394` | Filmstrip/similar-results strip | S |
| P6 | Painter-drawn cell chrome (`text_at`, badge, strip) | `app.rs:2132-2155, 998-1047` | Also use inside `show_cell_view` col 0 | S |
| P7 | `Selection` (anchored range drag, base-set restore, toggle_all) + tests | `model.rs:42-112, 364-409` | Add Shift/Ctrl and marquee | S |
| P8 | Edge auto-scroll during drag | `app.rs:1088-1111` | Direct (`scroll_with_delta_animation(.., ScrollAnimation::none())`) | S |
| P9 | `animate_bool_with_time` for selection inset | `app.rs:990-996` | Direct | S |

### Disk cache and viewer

| ID | Pattern | Source | Notes for AME | Size |
|---|---|---|---|---|
| P10 | Content-identity filename (SHA-256 of `uri:modified:size:variant`, 16 bytes hex), `setLastModified` on hit, `.pending` file + atomic rename, prune oldest >1 GB, version suffix to invalidate | `MediaFiles.java:105-114, 292-302, 510-522` | AME already depends on `blake3`, `directories`, `tempfile` | M |
| P11 | Decode at reduced size (loadThumbnail / `inSampleSize` / `ImageDecoder.setTargetSize`) then JPEG q88 | `MediaFiles.java:316-329`; `PhotoEdits.java:62-73` | Android APIs. The desktop `image` 0.25.10 JPEG path is zune-jpeg and exposes no scaled decode **[E]**; a scaled-decode crate is **[U]** (not in local registry) | M |
| P12 | Two-layer tile viewer (low-res preview + 512*sample tiles, LRU 24 MB, <=32 pending) | `PhotoView.java:24-26, 81-84, 169-212` | Needs a desktop decode/pyramid strategy | L |
| P13 | Generation/session token to drop stale async results | `PhotoView.java:73, 91, 200-206`; `GalleryController.java:456, 474-480`; `editor.rs:119-122` | Use for preview, edit results, thumbs | S |
| P14 | Progressive open: cached thumb/embedded preview first, full later | `GalleryController.java:530-540` | Preview pane + RAW | S-M |
| P15 | Pinch-about-focus + pan clamp | `PhotoView.java:41-48, 143-148`; `mask.rs:189-220` | Pure Rust version in `ViewXform` | S |

### Editor

| ID | Pattern | Source | Notes | Size |
|---|---|---|---|---|
| P16 | Proxy edit + full-res export from the same recipe, parity tests | `editor.rs:123-137, 212-246`; `edit.rs:62-119, 151-179` | Rust-only on desktop: `Recipe::preview` already works on any `RgbaImage` | S-M |
| P17 | Equality-gated texture rebuild | `editor.rs:212-246` | Direct | S |
| P18 | Shared crop overlay with press-origin retention + 4-corner & square tests | `editor.rs:770-872, 926-1031` | Shrink hit radius (30 pt is touch-sized), add edge handles and aspect lock | S-M |
| P19 | `ViewXform` | `mask.rs:149-221` | Add tests; feed `zoom_delta()` + pointer | S |
| P20 | `MaskCanvas` + replayable `StrokeRec` undo + `bake_alpha` | `mask.rs:1-121, 244-262`; `editor.rs:639-654` | Only useful with an inpaint-capable backend (AME `EditOptions` has no mask: `AME:ai/qwen_image_edit/mod.rs:22-37`) | M |
| P21 | In-place overlay texture update | `editor.rs:699-712` | Also for diffusion step previews | S |
| P22 | Compare toggle / hold-to-compare | `editor.rs:327-337, 529-533`; `app/studio.rs:2565-2566` | Add split slider via two clipped `image` calls (same `with_clip_rect` pattern as `editor.rs:559`) | S |
| P23 | Stage/controls responsive split rule | `editor.rs:274-299` | Fits dock tabs of varying size | S |
| P24 | Explicit `TextEdit` id + caret-follow on reflow | `editor.rs:454-481`; `VERIFICATION.md:244-268` | Prevents focus loss when columns<->stacked | S |
| P25 | Snapshot `History` (cap 120) committed after 0.45 s idle with no pointer down | `src/studio.rs:1205-1260`; `app/studio.rs:576-589` | `Recipe` is `Copy`, trivial | S |
| P26 | Rotated hit boxes + corner handles (delete/edit/copy/rotate), snap to centre (7 pt) and 90 deg (4 deg) | `app/studio.rs:91-124, 1473-1627` | Only for overlays/annotations | M |
| P27 | Pointer lift-point helper | `app/studio.rs:807-822` | Touch-specific | skip on desktop |
| P28 | Texture cache keyed by hash of serialized state (`DefaultHasher` over `serde_json` + frame size) | `app/studio.rs:507-574` | Pattern for generated previews | S |

### Colour, video

| ID | Pattern | Source | Notes | Size |
|---|---|---|---|---|
| P29 | Pure `color::Adjust` + table-driven slider list + `strip()` LUT + tests | `color.rs`; `app/studio.rs:2556-2590` | Reusable verbatim | S |
| P30 | GPU grading via `egui_glow::CallbackFn`: own program/VAO, LUT on unit 2, viewport from `info.viewport_in_pixels()`, restore GL state | `EguiMobile:egui-android/src/video/surface.rs:1-40, 247-320` | Same backend family as AME (glow). Shader is GLSL ES 1.00 with `samplerExternalOES`; desktop needs `sampler2D` and a `#version 140/330` variant **[U]** on which GL version eframe gets on Linux/Windows | L |
| P31 | `Timeline` model + tests | `model.rs:235-355` | Direct if AME adds trimming/scrub | S |
| P32 | Scrub precision drag, 90-100 ms throttle, pending-seek reconciliation | `app.rs:401-419, 1844-1916` | Preview-pane video | M |
| P33 | Filmstrip from cached frame thumbs, 100 ms quantization, x-range culling | `app/studio.rs:1742-1773` | Optional | M |
| P34 | Peak-pooled waveform | `edit.rs:121-145` | Optional | S |

### Progress, structure, tests, debug

| ID | Pattern | Source | Notes | Size |
|---|---|---|---|---|
| P35 | Progress bar+cancel row, floating progress `Area`, toast+Undo | `app.rs:2309-2322, 2350-2419` | AME has `egui-toast` and `ui/status.rs`; reuse shapes only | S |
| P36 | Thread + `mpsc` result polled with `try_recv` | `app/studio.rs:3045-3072, 1157-1165` | AME uses crossbeam already | S |
| P37 | Mode/dialog stack where Back/Esc closes the innermost | `app.rs:562-594` | Direct | S |
| P38 | Headless frames: `Context::default()`, `ctx.run_ui(RawInput{screen_rect,time,events})`, synthesized `Touch`/`PointerButton` events, scan `output.shapes` for text/rect | `app.rs:2672-2737, 2740-2808, 2905-2977` | Mouse variants trivially simpler | S-M |
| P39 | AccessKit rig: `ctx.enable_accesskit()`, read `platform_output.accesskit_update` labels+bounds, `tap_label`, `drag`, `idle` | `app/studio.rs:3371-3486` | `egui_kittest 0.37` exists in the local registry as an official equivalent; AME does not depend on it **[E]** | S |
| P40 | Virtual-list geometry regression (row y-position stability across tiny scroll steps both directions at two window sizes) | `app.rs:2580-2670` | Run against AME's new grid | S |
| P41 | Control-bounds log under a debug flag (`icons::note/begin/report`, JSON on change) | `icons.rs:5-56` | Desktop: env var + stderr, or the AccessKit tree | S |
| P42 | Debug panel: hit-box overlay toggle, mapping/timing readout, render ms, copy-JSON/copy-command, sample data, reset | `app/studio.rs:3073-3162`; pointer telemetry overlay `editor.rs:683-696` | Matches your "debug UI per feature" preference | S per feature |
| P43 | Overflow-guard `egui::Plugin` (`on_end_pass` + `output_hook`, debug-only, red outline + log, `Policy::{Report,Panic,Off}`, `allow(ui)`) | `EguiMobile:egui-mobile-core/src/overflow.rs` | Pure egui, no Android; useful for resizable dock panes. Only horizontal edges checked | S-M |
| P44 | `handed()`: place a group by previous-pass width stored in `ui.data`, `request_discard` when it changes | `app.rs:41-56` | Optional; `request_discard` costs a pass (`egui-0.37:src/context.rs:2313-2315`) | S |
| P45 | Layout settings persisted as small serde JSON | `app.rs:34-39, 185-194` | AME already persists in SurrealDB/eframe | skip |

---

## 5. APIs, versions, workarounds

**Luma stack:** `egui 0.37` (feature `monochrome_emoji_fonts`), `egui_extras 0.37` (`image`,`svg`), `image 0.25` (`jpeg`,`png`), `skrifa 0.44`, `harfrust 0.12`, `tiny-skia 0.11`, `epaint_default_fonts 0.37`, edition 2024, EguiMobile rev `df955042` as both runtime and `android-activity` patch (`Luma:Cargo.toml`). AME: `eframe/egui 0.37` with `glow` only, `egui_extras 0.37` with `all_loaders`, `image 0.25` default features (`AME:Cargo.toml`).

**egui 0.37 APIs Luma relies on (all present in 0.37 source [E] unless noted):**

| API | Used for | Cite |
|---|---|---|
| `Context::run_ui(RawInput, FnMut(&mut Ui)) -> FullOutput` | headless tests | `egui-0.37:context.rs:950` |
| `ScrollArea::show_rows`, `show_viewport`, `scroll_source(ScrollSource::ALL)`, `Ui::scroll_with_delta_animation`, `scroll_to_rect` | grid/rail | `Luma:app.rs:880-884, 1320-1339, 1106` |
| `scroll_area::State::load/store`, `unclamped_offset()` | tests (0.37 API) | `app.rs:2627-2640` |
| `Style.scroll.kinetic: KineticScrollStyle {decay_time 0.5, stop_distance, rubber_band true}` | touch/trackpad momentum + rubber-band (Luma 0.1.17 notes it) | `egui-0.37:style.rs:586, 851-885` |
| `Image::from_texture(&tex)/from_texture((id,size)).uv().tint().corner_radius().rotate().paint_at(ui, rect)` | covers, rotated overlays | `app.rs:474-478`; `app/studio.rs:1422-1425` |
| `Painter::with_clip_rect(rect).image(tex_id, rect, uv, tint)` | canvas | `editor.rs:559` |
| `TextureHandle::set(ColorImage, TextureOptions)` | in-place update | `editor.rs:708` |
| `Context::animate_bool_with_time`, `Response::long_touched`, `is_pointer_button_down_on`, `drag_started/drag_stopped/interact_pointer_pos`, `InputState::pointer.press_origin()/press_start_time()`, `multi_touch()` | gestures | various |
| `Popup::menu(&response).show`, `Modal::new(id)`, `Window::anchor().fixed_size()`, `Area::order(Foreground).anchor()` | menus/dialogs | `app.rs:1141, 1933, 1953, 2351` |
| `Id::unique(..)`, `ui.scope_id()`, `IdSalt::new`, `egui::Role` (replace `Id::new`, `ui.id()`, `WidgetType`) | 0.37 migration (commit `fda250c`); `Id::new` is `#[deprecated]` | `egui-0.37:id.rs:91-107`; `ui.rs:363` |
| `egui::Plugin` (`on_end_pass`, `output_hook`), `Context::add_plugin`, `debug_painter` | overflow guard | `egui-0.37:plugin.rs:38-50`; `context.rs:1884, 2376` |
| `Context::enable_accesskit`, `PlatformOutput::accesskit_update` | test rig | `egui-0.37:context.rs:4234`; `data/output.rs:185` |

**Documented gotchas (Luma code/docs) and egui-source facts that matter for AME:**

1. `show_rows` row height must equal the drawn row height including spacing (`app.rs:886`; `VERIFICATION.md:515-517`).
2. egui clears `press_origin()` on release before `drag_stopped()` fires; store the gesture origin yourself (`editor.rs:841-845`; `VERIFICATION.md:375-397`, mouse and touch).
3. Touch: only `Touch{End}` carries the lift point (`app/studio.rs:807-822`).
4. Widget ids must be explicit for `TextEdit` that can be reparented by layout changes (`editor.rs:454-457`; `VERIFICATION.md:244-259`).
5. Pasting into an empty `TextEdit` can change galley height a frame later; follow the caret only when geometry changed (`editor.rs:462-481`).
6. `ctx.options_mut(|o| o.input_options.max_click_duration = ..)` changes long-touch timing (`app.rs:652-654`; default 0.8, `egui-0.37:input_state/mod.rs:115`).
7. Headless tests call `output.textures_delta.clear()` after each frame (`app.rs:2714`; `editor.rs:893`); reason not documented **[U]**.
8. `Context::load_texture` debug-asserts side <= `max_texture_side` (default 2048) **[E]** (`context.rs:2822-2830`).
9. `DefaultBytesLoader::insert` uses `entry().or_insert_with_key`, so a second `include_bytes` for the same URI **does not replace** the first **[E]** (`egui-0.37:load/bytes_loader.rs:14-26`). `DefaultTextureLoader` caches by `(uri, TextureOptions)` and only evicts multi-size SVG buckets (`load/texture_loader.rs:55-118`). `egui_extras::DecodeCache` is keyed by URI and spawns one OS thread per decode on native (`egui_extras-0.37:loaders/background_decode.rs`). Only `ctx.forget_image(uri)`/`forget_all_images()` or `options.reduce_texture_memory` (drops byte/decoded copies after upload, `texture_loader.rs:84-97`) release memory. Luma's `bytes://` use is safe only because its bytes are immutable embedded SVGs.
10. `request_discard` costs an extra pass; egui documents "be very conservative" (`context.rs:2313-2315`).

---

## 6. What does NOT transfer, and what not to copy

**Does not transfer (Android-specific or off-scope):**
- Native `PhotoView`/`PopupWindow`/`SurfaceView`, LibVLC, MediaCodec/SurfaceTexture, `samplerExternalOES`, JNI `bridge` and the JSON poll loop, MediaStore/SAF/trash/consent/share intents, `egui_mobile::Host` (haptics, safe-area, IME/text toolbar, stylus probe, documents dir).
- Touch ergonomics: long-press selection, bottom `action_bar`, one-handed `Hand` layout, `ScrollSource::ALL`, swipe-to-navigate.
- Mobile build/test tooling: Python+adb+OCR smoke scripts (`scripts/smoke-*.py`), logcat `LumaWidgets`.
- Video editor stack: `compose.rs` (FFmpeg graph), `speech.rs`/whisper, `studio.rs` project model, `caption.rs` (skrifa/harfrust/tiny-skia text rasterizer, Android font scan paths `/system/fonts`). Reusable only if AME adds text overlays or video editing.
- LibRaw/Adobe DNG SDK out-of-process helpers (C++ toolchain, mobile OOM rationale). AME uses `rawkit 0.1.0`, whose public API is `decode`, `process_8bit/16bit`, `apply`, `demosaic_and_apply` with **no embedded-preview extractor** [E] (`rawkit-0.1.0/src/lib.rs:119-235`); an embedded-JPEG route needs another crate or FFI **[U]**.
- Qwen/ComfyUI server client flow (credentials, resume, comfy-gate); AME runs a local candle pipeline.

**Do not copy (works for Luma, wrong for AME):**
- `visible_items()` cloning and re-filtering the whole list per frame, `eligible()` lowercasing the query per item (`app.rs:602-636`).
- Per-cell `format!` thumbnail keys (`model.rs:33-35`).
- UI-thread decode of every arriving thumbnail without a per-frame time budget (`app.rs:263-280`); fine at <=64 outstanding tiny JPEGs, not at DB scale.
- CPU `Recipe::preview` that clones the whole proxy on each slider tick (`edit.rs:87-109`) if the proxy grows beyond ~2 MP or sliders become more than 3; use GPU (P30) or a smaller proxy **[I]**.
- Touch-sized hit radii (30 pt crop corners, 22 pt scrub handles) on a mouse UI.
- Single `native: Value` JSON blob for state; AME already has typed channels.

---

## 7. Mapping to AME panels

Size/risk are estimates **[I]**. "Current" cites AME code I read.

### 7.1 File grid / table (`AME:ui/file_table/`)

| # | Current (AME) | Port (Luma) | How | Size |
|---|---|---|---|---|
| T1 | Thumbnails are base64 PNG data URLs on `Thumbnail.thumbnail_b64`, decoded into `thumb_cache: HashMap<String, Arc<[u8]>>`, shown as `Image::new(ImageSource::Bytes{uri:"bytes://{path}"})` per cell per frame (`table/mod.rs:44, 251-289`; `file_table/mod.rs:1444-1479`). Memory: bytes + decoded `ColorImage` + GPU texture per URI, never evicted except page-prune of the bytes map (`file_table/mod.rs:1178-1187`) | P1 + P3 | `ThumbCache` owning `HashMap<u64,(TextureHandle,u64)>`; `show_cell_view` col 0 calls `cache.get_or_request(row)` and paints with UV cover; keep `egui_extras` loaders only for `include_image!` icons | M |
| T2 | Page-wide scheduling: every row lacking a thumb is queued (`file_table/mod.rs:1366-1441`), 6-permit tokio semaphore running CPU decode on async workers (`:370`) | P2 | Request on draw, cap in-flight (~80), time out, dedupe by key; do CPU work on `spawn_blocking`/rayon, not async workers | S-M |
| T3 | `generate_image_thumb_data`: `image::open` (full decode) -> `thumbnail(256,256)` -> PNG -> base64 (`utilities/thumbs.rs:10-97`); RAW goes through full `rawkit` demosaic even for 256 px (`:38-49`; `raw_decode.rs:5-43`) | P10, P11 | Disk cache of JPEG thumbs under a cache dir keyed by blake3(path,mtime,size,variant); atomic write; mtime-LRU prune. DB can keep a small b64 for offline/DB mode. Scaled JPEG decode and embedded RAW previews need new crates **[U]** | M |
| T4 | Receive path takes **one** message per frame: `if let Ok(thumbnail) = self.thumbnail_rx.try_recv()` (`receive/receive_thumbnail.rs:8`) | Luma drains all events per pump (`app.rs:214`) | Loop `try_recv` with a ~2-4 ms or N-upload budget, then `request_repaint()` if more remain | S |
| T5 | No grid mode; table rows 75 pt (`file_table/mod.rs:748`) | P4, P6, P7, P8, P9, `pick_badge` | New `ViewMode::Grid` in `FileExplorer::ui`, same `Thumbnail` rows and filters; cell size slider; desktop input: Ctrl/Shift click, marquee, arrow-key nav, hover, context menu | M-L |
| T6 | No tests of scroll/selection geometry | P38, P40 | Headless `run_ui` tests for the new grid; optional `egui_kittest` | S-M |
| T7 | Video thumbs: `generate_video_thumb_data` returns `Err` on non-Windows (`utilities/thumbs.rs:104-109`) so Linux rows show `broken_link.png` (`table/mod.rs:252-259`) | `MediaFiles.java:351-352` (FFmpeg one-frame at `-ss t`, scale) | With the optional `ffmpeg` feature, seek+decode one frame via `ffmpeg-next` (already used in `preview_pane.rs`), cache through T3 | M |
| T8 | Dock tab sizes vary | P23 pattern for the table/grid toolbar rows | Optional | S |

### 7.2 Preview pane (`AME:ui/file_table/preview_pane.rs`)

| # | Current (AME) | Port (Luma) | How | Size |
|---|---|---|---|---|
| V1 | Shows "No Image" until the 1600 px preview arrives (`file_table/mod.rs:1444-1460`) | P14 | Draw the already-cached grid texture instantly, swap to the preview texture when ready | S |
| V2 | Preview = `generate_image_preview_png(path,1600)`: full decode, Lanczos3 resize, PNG encode, base64 round trip through `thumbnail_tx` (`utilities/thumbs.rs:389-429`; `preview_pane.rs:152, 178-198`) | P10, P11, P13 | JPEG preview in disk cache; deliver as RGBA/path; generation token per selected path | S-M |
| V3 | While the preview is absent, **a new tokio task is spawned on every UI pass** (`preview_pane.rs:173-199`, no in-flight set); same in `image_edit.rs:455-479` | P2/P13 | In-flight key set + generation token | S |
| V4 | Image is shown with `max_size(avail/1.3)` only; no zoom/pan/fit/1:1 | P15/P19 (`ViewXform`), P12 for true 1:1 | Fit/zoom/pan on the <=2048 px texture with `zoom_delta()` + drag + double-click fit; true 1:1 on large images needs tiles (3.3). Dock pane resize works with the P23 rule | S (zoom/pan) / L (tiles) |
| V5 | Video: ffmpeg thread -> RGBA `Vec` -> `ColorImage` -> `TextureHandle::set` each frame; **unbounded** channel; pacing by `sleep(1/fps)` in the decoder thread; slider not wired to seek (`preview_pane.rs:12-103, 204-254`; channels `file_table/mod.rs:322`) | P31, P32; P30 later | Bound the channel / keep latest frame; wire seek with `pending_seek`; frame-step math from `Timeline`; GPU YUV path is the long-term analogue of Luma's SurfaceTexture | M (transport) / L (GPU) |
| V6 | RAW preview = full demosaic | P14 | Two-stage: embedded preview first (extractor **[U]**), developed full image later | M |
| V7 | No histogram/EXIF overlay | none in Luma | New: 256-bin RGB/luma on a <=512 px proxy computed off-thread, drawn with painter; toggle in the pane | S |
| V8 | Metadata editors/buttons/AI controls inline | not Luma's concern | out of scope | - |

### 7.3 Image-edit tab (`AME:ui/image_edit.rs`)

Current: one ~500-line `ui()` mixing model loading, GGUF fetching, run orchestration and display; source preview and result both via `ImageSource::Bytes{uri:"bytes://imageedit::..."}`; two-column fixed layout (`image_edit.rs:91-526`).

| # | Gap | Port (Luma) | How | Size |
|---|---|---|---|---|
| E1 | **Stale result bug:** result URI `bytes://imageedit::result::{path}` and step-preview URI `bytes://imageedit::preview_step::{path}` are constant for a path while bytes change (`image_edit.rs:486-517`). Per 5.9, the second `include_bytes` is ignored and decode/texture caches are URI-keyed, so a re-run shows the first result and the live preview sticks on the first step. No `forget_image` call exists anywhere in AME (grep). **[E]+[V]** for the mechanism; not reproduced at runtime | P21/`texture()` helper (`editor.rs:742-751`) | Hold `TextureHandle`s: `result`, `step_preview` (update with `.set`), `source`; drop old handles on new run | S |
| E2 | No canvas | P19 + `with_clip_rect().image()` (`editor.rs:559`) + P22 | Stage with fit/zoom/pan, Before/Result toggle, then a split-slider compare (two clipped draws of the same fit rect) | S-M |
| E3 | Fixed `columns(2)` regardless of tab size (`image_edit.rs:438`) | P23 | Split only when wide; otherwise stack | S |
| E4 | Async results unscoped | P13 | `session` id on every job/event; ignore mismatches | S |
| E5 | `Cancel Run` exists (`image_edit.rs:412-415`) but no unsaved-edit guard or busy gating of the whole panel | `editor.rs:184-199, 258, 301-316, 339` | `busy` -> `add_enabled_ui`; discard confirm; "Save copy" next to result, never overwrite source | S |
| E6 | No non-AI edits | P16, P17, P18, P19, P25 | `Recipe`/`Crop` (copy `edit.rs`), proxy from the T3/V2 preview, `crop_overlay` with smaller hit radius, rotate/flip, exposure/contrast/saturation; export applies recipe to the full decode with `Recipe::preview` + `crop_imm` (pure Rust) | M |
| E7 | Mask painting for local edits | P20 | **Blocked/conditional:** `EditOptions` has no mask (`ai/qwen_image_edit/mod.rs:22-37`; no "mask" matches in `qwen_image_edit/*.rs`). Only worthwhile with an inpaint-capable local/remote backend | M (+ backend) |
| E8 | Colour tools / histogram | P29 (pure), P30 (GPU) | CPU histogram on proxy (S); `Adjust` sliders via `CONTROLS` loop (S); live GPU LUT preview through a glow `PaintCallback` (L, GL version **[U]**) | S / L |
| E9 | Debug UI | P42 | Toggle: show fit/view rects, pointer + zoom telemetry, edit options JSON copy, step timings | S |
| E10 | Structure | `PhotoEditor` keeps UI state + `event()` handler separate from workers (`editor.rs:119-182`) | Split `ImageEditPanel` into `EditSession` (state, events) and thin UI; makes headless testing possible | S-M |

### 7.4 Suggested order (dependencies, not a plan commitment)

1. T1+T2+T4 (texture cache, gating, drain budget) and E1 (stale-URI bug): unblocks everything and removes memory growth.
2. T3/V2 (disk cache, JPEG previews, progressive load) + V3.
3. E2/E3/E4/E5 (edit stage, compare, session tokens) + V4 (zoom/pan).
4. T5 grid view + T6 tests.
5. E6 recipe/crop; V7/E8 histogram; T7 video thumbs.
6. GPU paths (P30) and tile viewer only if profiling shows the CPU paths are the limit.

---

## 8. AME defects found while mapping (verified in code)

| # | Finding | Evidence |
|---|---|---|
| D1 | Constant `bytes://` URIs for changing content (edit result, step preview) will show stale images | `image_edit.rs:486-517`; `egui-0.37:load/bytes_loader.rs:14-26`; `egui_extras-0.37:loaders/image_loader.rs:108-125` |
| D2 | Image/texture caches never evict; `forget_image` unused. Thumbnails keep PNG bytes (`Arc` shared), a decoded `ColorImage`, and a GPU texture per row ever shown | `grep forget_image` empty; `egui-0.37:load/texture_loader.rs`; `table/mod.rs:276-285` |
| D3 | `egui_extras` spawns one OS thread per image decode (no pool) | `egui_extras-0.37:loaders/background_decode.rs` (native path); impact under fast scroll **[I]** |
| D4 | Thumbnail receiver handles one message per frame | `receive_thumbnail.rs:8` |
| D5 | Preview/edit-preview tasks spawned every pass until the result arrives | `preview_pane.rs:173-199`; `image_edit.rs:455-479` |
| D6 | No video thumbnails on Linux | `utilities/thumbs.rs:104-109` |
| D7 | RAW thumbnails use full demosaic; photo thumbs/previews decode the whole file; PNG+Lanczos3+base64 for previews | `raw_decode.rs`; `thumbs.rs:10-97, 389-429` |
| D8 | Video frame channel unbounded | `file_table/mod.rs:322`; `preview_pane.rs:77` |

Checked and not a defect: AME has no `egui::Id::new(..)` calls (only `RecordId::new`), so the 0.37 `Id::new` deprecation does not affect it.

---

## 9. Open questions and uncertainty

- **Grid view scope:** do you want a real gallery grid (T5), or only a better table? It is the largest single port.
- **Inpaint/mask:** will the local Qwen pipeline gain mask support, or will masked edits go to a remote backend (ComfyUI/zeroclaw)? Decides E7.
- **Desktop GL version:** custom shaders (P30) depend on the context eframe/glow creates on Linux and Windows; not checked.
- **Scaled JPEG decode and RAW embedded previews:** no crate verified; `image` 0.25.10 (zune-jpeg) and `rawkit 0.1.0` do not expose them.
- **Tile viewer cost on 50-100 MP files:** not measured; Luma caps exports at 32 MP and the developed RAW cache is a single JPEG.
- **No frame-time baseline:** neither Luma nor AME has numbers. Profile AME's table scroll and slider-drag before choosing GPU work.
- **egui AccessKit in release builds on desktop:** the rig only enables it in tests; whether to ship it for external automation is open.
- Luma evidence I did not inspect: `Transcriber.java`, `EguiVpnService.java`, `MediaManager.java`, `QwenEdits.java` (server flow), and `caption.rs` rasterizer internals beyond its header and API.
