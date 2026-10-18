# Dependencies
## ONNX Runtime ( https://github.com/microsoft/onnxruntime/releases )
Windows with GPU: onnxruntime-win-x64-gpu-1.22.1.zip
Windows with CPU: onnxruntime-win-x64-1.22.1.zip

- $env:ORT_LIB_LOCATION="C:\path\to\onnx\runtime"
or
- setx ORT_LIB_LOCATION "C:\path\to\onnx\runtime"

## FFMPEG
- Windows: https://www.gyan.dev/ffmpeg/builds/
    - setx FFMPEG_DIR "C:\path\to\ffmpeg"

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
