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
