# Dependencies
## ONNX Runtime ( https://github.com/microsoft/onnxruntime/releases )
Windows with GPU: onnxruntime-win-x64-gpu-1.22.1.zip
Windows with CPU: onnxruntime-win-x64-1.22.1.zip

- $env:ORT_LIB_LOCATION="C:\path\to\onnx\runtime"
or
- setx ORT_LIB_LOCATION "C:\path\to\onnx\runtime"

## FFMPEG
- Linux: the in-process video decoder (`ffmpeg-next` 9) is always built, so FFmpeg 7 or newer development files, `clang` and `pkg-config` are required to compile.
    - Arch/Manjaro: `sudo pacman -S ffmpeg clang pkgconf`
    - Debian/Ubuntu: `sudo apt install libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev libavdevice-dev clang pkg-config`
- Windows: https://www.gyan.dev/ffmpeg/builds/
    - setx FFMPEG_DIR "C:\path\to\ffmpeg"
    - the decoder is optional there: `cargo build --features ffmpeg`. Without it the Windows shell thumbnail is used, and the `ffmpeg` and `ffprobe` commands when they are on `PATH`.

# Video
`src/video` decodes video for thumbnails, the probe API and the preview player. The build script sets `cfg(ame_ffmpeg)` on Linux and with `--features ffmpeg`; that enables the in-process backend (`AME_NO_NATIVE_VIDEO=1` at build time turns it off). Everywhere else the `ffmpeg`/`ffprobe` commands are used when they are on `PATH` (override with `AME_FFMPEG` and `AME_FFPROBE`), and files without a decodable frame get a generated placeholder tile instead of an error.

- Thumbnails: a frame at about 10% of the duration (up to four positions are tried when a frame is blank or near black), upright after the rotation metadata and the pixel aspect ratio, scaled to 256 px, stored like image thumbnails. Placeholders are stored like any other thumbnail and stay until the file changes or its stored thumbnail is deleted.
- Limits: at most `AME_VIDEO_JOBS` decodes at once (default a quarter of the cores, 2 to 4), 20 s per file, cancellable. Damaged, empty, audio-only and 8K files return typed errors or placeholders without panics.
- Probe: `video::probe(path)` returns duration, size, frame rate, codecs, audio tracks, rotation, bitrate and creation time.
- Preview: the right pane plays the selected video muted (play, pause, seek). Audio is not played.
- View > Devtools > Video probes a file, generates its thumbnail with timings and plays it.
- Tests build their clips with the `ffmpeg` command and skip when it is missing.

# Theme and icons
The styling system is the `egui-theme-kit` and `egui-icon-pack` crates (https://github.com/shadowbrok3r/egui-theme-kit), pinned by `rev` in `Cargo.toml`.
`src/theme` re-exports the kit and adds the app font stack (`assets/fonts`), the CLI aliases and the Appearance window; `src/icons` generates the app icon table from `assets/icons` with `egui-icon-pack-build`.

- File > Appearance... (also under View and in Preferences) opens the theme picker: presets, saved themes, density, text size, glass and colors.
- The theme is stored per database in the `ui_themes` table (`ui_themes:__active__` holds the active theme and the startup pin, one record per saved theme). eframe storage keeps a copy that is read before the database opens.
- Switches: `--theme <file.json>`, `--theme-dump`, `--theme-panel`, `--no-frost`; variables `EGUI_THEME_KIT_THEME` and `EGUI_THEME_KIT_NO_FROST`. The old `SMART_MEDIA_THEME` and `SMART_MEDIA_NO_FROST` still work.
- View > Devtools > Theme panel (Ctrl+Alt+T) is the kit debug panel.

# Importing a legacy (SurrealDB 2.x) library
`smart-media --db-import-legacy <old store dir>` imports a library written by an older build of the app on SurrealDB 2.x (for example a Windows-side `db/ai_search`). The old store is never opened in place: it is copied to a work folder, exported with the official SurrealDB 2.3.10 CLI (located via `--surreal-bin` or `AME_SURREAL_BIN`, else the cached copy, else downloaded from the GitHub release into the per-user data folder and checked against a pinned SHA-256), converted in a streaming pass and imported. View > Devtools > Legacy import does the same from the UI.

- Path map: Windows paths are remapped with `legacy-pathmap.toml` next to the database folder (longest prefix wins, case-insensitive, `unmapped`/`missing` = keep|drop|tag). When the file is missing a suggestion is built from the mounted volumes (`/run/media`, `/media`, `/mnt`), the paths recorded in the old store and size-matched sample files, saved for review, and the run stops until it is reviewed or `--accept-suggested-map` is given. Automount names under `/run/media/<user>/<label>` can change; mount the volume at a fixed place or use a symlink in the map.
- `--dry-run` prints the mapping table (rows, share of files that exist, case-repaired paths, missing prefixes) and writes no store.
- `--merge` (default when the database exists) keeps existing rows (`INSERT IGNORE`, groups by name, chat appended) and takes a snapshot first; `--into-new-store <dir>` builds a fresh store. User settings are skipped unless `--settings fields:a,b` names fields to merge; API keys are only merged when named.
- The export holds the old settings including API keys: it is written with mode 0600 and deleted afterwards unless `--keep-export` is given.
- Afterwards run `--db-thumbs migrate` then `--db-thumbs finish` (or pass `--then-migrate-thumbs`). Blobs are keyed by the stored path, so finish the path map first.
- Exit codes: 0 ok, 1 failed, 2 usage, 3 database in use, 4 engine unavailable, 5 verification problems, 6 path map needs review, 7 cancelled.

# Window title bar
The main window draws its own title bar (logo, File and View menus, engine status chips, minimize, maximize and close) and resize edges; secondary windows such as the AI Assistant keep the system title bar.

- View > Custom title bar switches to the system title bar and back; the choice is saved to `<config dir>/smart-media/window_settings.json`.
- `--system-titlebar` or `EGUI_THEME_KIT_SYSTEM_TITLEBAR=1` keeps the system title bar for one launch without changing the saved choice.
