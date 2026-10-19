# Smoke checks

What to look at after a change, per operating system, and how to run the application against a scratch library without touching real data. Linux is the target and is checked headless; Windows and macOS are manual and best effort.

## Run the application safely against a scratch library

The application keeps its library in the folder named by `./db/path.txt`, else by `<config>/smart-media/db_location.txt`, else in `./db/ai_search`; settings, `ui_prefs.json`, `ui_layout.json`, `view_prefs.json` and `workspace.json` live in `<config>/smart-media/`. Unknown command-line flags start the window, so a careless run opens the real library. Always isolate all of it:

1. Work in a scratch folder outside the repository (`SCRATCH=$HOME/.cache/ame-scratch/<name>`). Never start the program from the repository root, the worktree root or any folder with a `db` folder in it.
2. Give it its own `HOME` and `XDG_CONFIG_HOME`, `XDG_DATA_HOME` and `XDG_CACHE_HOME` under `$SCRATCH`, a `TMPDIR` and a working directory under `$SCRATCH` too. Without that the config directory of the real user is read and written.
3. Point the library at a scratch folder: create an empty `$SCRATCH/library` and write its path into `$XDG_CONFIG_HOME/smart-media/db_location.txt` (one line). With no pointer the first-run wizard opens instead, which is also a valid check.
4. Give the pictures folder a generated media folder: a `user-dirs.dirs` file in `$XDG_CONFIG_HOME` with `XDG_PICTURES_DIR="$SCRATCH/media"`. The start folder of the first tab is the Pictures folder.
5. Keep the model downloads away: `HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1`; delete `$SCRATCH/hf` afterwards, the application tries to fetch several GB when AI features start.
6. `LIBGL_ALWAYS_SOFTWARE=1` for a headless run.
7. Use only flags that are documented: `--scope`, `--run`, `--query`, `--devtools`, `--wizard`, `--dump-state` (`docs/commands.md`), `--zc-status` and the `--db-*` tools. Use the mock gateway of `crates/zeroclaw-client` for anything that needs ZeroClaw, never a real gateway or token.

A generated media folder for the checks below (22 KiB files so the default minimum size does not hide them):

```sh
mkdir -p "$SCRATCH/media/trip" "$SCRATCH/media/family"
for i in 1 2 3 4 5 6; do head -c 22528 /dev/urandom > "$SCRATCH/media/trip/img_$i.jpg"; done
for i in 1 2 3; do head -c 22528 /dev/urandom > "$SCRATCH/media/family/pic_$i.png"; done
```

The files are not decodable pictures; the rows, scopes, tabs and layout can be checked with them, thumbnails show the failed state.

## Linux: private Xvfb recipe

A window-manager-less X server of your own, on a display number nobody else uses, with a renamed copy of the binaries so that stopping them by exact name cannot hit another session.

```sh
SCRATCH=$HOME/.cache/ame-scratch/smoke-$USER
BIN=$SCRATCH/bin
mkdir -p "$BIN" "$SCRATCH"/{home/.config,run,logs,shots,tmp}
cp /usr/bin/Xvfb "$BIN/ame-xvfb"
cp target/debug/smart-media "$BIN/ame-smoke"          # built with cargo build; copy, never run the one in the repository root
D=:193                                                 # pick a number in :170-:199 that has no /tmp/.X11-unix/X<number>
"$BIN/ame-xvfb" $D -screen 0 1280x800x24 >/dev/null 2>&1 &
XPID=$!
cd "$SCRATCH/run"
env -i PATH=$PATH HOME=$SCRATCH/home XDG_CONFIG_HOME=$SCRATCH/home/.config XDG_DATA_HOME=$SCRATCH/home/.local/share \
  XDG_CACHE_HOME=$SCRATCH/home/.cache TMPDIR=$SCRATCH/tmp HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 \
  DISPLAY=$D LIBGL_ALWAYS_SOFTWARE=1 RUST_BACKTRACE=1 \
  "$BIN/ame-smoke" --dump-state "$SCRATCH/state.json" >"$SCRATCH/logs/run.log" 2>&1 &
APID=$!
```

Driving it:

| Step | Command |
|---|---|
| Keyboard focus (needed once, there is no window manager) | `DISPLAY=$D xdotool search --name "Smart Media" windowfocus` |
| Click | `DISPLAY=$D xdotool mousemove X Y click 1` |
| Keys | `DISPLAY=$D xdotool key ctrl+k`, then `type "quit"` and `key Return` for the palette |
| Resize or move | `DISPLAY=$D xdotool search --name "Smart Media" windowsize W H`, `windowmove X Y` |
| Screenshot | `DISPLAY=$D import -window root "$SCRATCH/shots/name.png"`, then look at the image |
| Without the mouse | `--scope`, `--run '<json>'`, `--devtools tasks,workspace` on the command line, or `{"cmd": ...}` from Devtools > Command log |
| State | `--dump-state <path>` writes `state.json` once the application is idle; read it instead of the screen |
| Quit politely (writes the workspace) | Ctrl+K, `quit`, Enter, then wait until the process is gone (a busy machine takes several seconds) before starting the next run; killing the process leaves the last file written one second after the last change |
| Clean up | `kill $APID`, `kill $XPID` (by PID), or `pkill -x ame-smoke` and `pkill -x ame-xvfb` (exact name, never `pkill -f`); `rm -rf "$SCRATCH/hf"` |

Never use a display you did not start yourself, and never point `DISPLAY` at the real session.

### What to check

Each item has a cheaper check in `state.json` where one exists.

| Area | Look at | Pass when |
|---|---|---|
| Start-up | the log, the first frame | no panic, no `ERROR` about the config files; the window has the title strip, sidebar, a browser tab and the bottom bar; with no library folder the first-run wizard is open |
| Browser | a folder scope with the generated media | grid with tiles and names, Back and Forward move through history, the breadcrumb shows the folder, `state.json` lists the tab with `rows` equal to the files |
| Query bar | type `img_2` in Name mode | rows are hidden, not removed (`rows` stays, `filtered` drops); Esc empties it |
| Views | Ctrl+1 and Ctrl+2, the size slider | grid and details switch, the tile size changes, `view` and `tile` follow in `state.json` |
| Selection | click, Shift and Ctrl click | the action bar shows the count, the right rail follows the focused file |
| Engine chips | the right of the title strip | one icon per engine (database, vision, CLIP, reranker, image edit, assistant, gateway), a hover card per chip; shrink the window to 700 px wide: the chips fold to one gear |
| Tabs | open a second folder (`--run '{"cmd":"open_scope","scope":"fs:...","target":"new_tab"}'`) | a tab per scope, closing one leaves the others |
| Settings | Ctrl+, | the nine sections draw, Apply and Revert are disabled until something changes |
| Workspace | see the next section | the tabs, split and window come back |
| Devtools | `--devtools workspace,tasks,command-log` | each window draws and its buttons work |
| Narrow window | `windowsize 700 400` | nothing overlaps, the sidebar is a rail, no panic in the log |

### Workspace round trip

1. In a fresh scratch config switch on **Restore tabs** (Settings > General, or write `{"version":1,"startup":{"restore_tabs":true}}` to `$XDG_CONFIG_HOME/smart-media/ui_prefs.json` before the first start).
2. Start, open two tabs with different scopes (`--run` or the sidebar), split one tab into its own leaf by dragging it onto the right square of the drop overlay that appears in the middle of the dock (dropping it anywhere else turns it into a floating window, which is saved too), select two files, resize the window with `windowsize`, wait two seconds.
3. Quit politely. `workspace.json` now lists both scopes, the split, the selection and the window size (`cat` it, or Devtools > Workspace).
4. Start again with `--dump-state`. Screenshot: the same tabs, the split, the floating windows, the selection and the window size and place are back (`xdotool search --name "Smart Media" getwindowgeometry`); the tab behind the front tab of its leaf shows `"pending": "lazy"` in `state.json` until it is brought to the front. The dump itself appears after about ten frames.
5. Move one of the folders away, quit, start again: that tab shows "Not available right now" with the reason and a Retry button, the others are normal (a floating window may cover the placeholder; close it); move the folder back and press Retry.
6. Start with `--scope lib:all`: nothing is restored (the command line wins) and the file is replaced by what this run shows.

## Linux: checks without a display

- `cargo check`, `cargo check --tests` and `cargo check --features cpu` pass.
- `cargo test` has no failures. Tests never touch the real config: the prefs store is memory only under test and the workspace saver of a test application has no path.
- `cargo test workspace` covers the file, the restore and the dump; `cargo test command::` the command vocabulary and `docs/commands.md`.

## Windows (manual, best effort)

The Windows build is compile-checked only. On a machine that has it:

1. `cargo build`, then run from an empty working folder with a scratch `%APPDATA%` (`set APPDATA=...`, `set LOCALAPPDATA=...`) and a library folder written to `%APPDATA%\smart-media\db_location.txt`.
2. The title bar (custom strip and caption buttons, or the system bar with `--system-titlebar`), the minimum size of 700 by 400, drag and drop of a folder onto the window.
3. Quick Access places: drives, WSL mounts (`\\wsl.localhost\...`), the Pictures folder; a folder scope on a drive letter and on a UNC path.
4. Thumbnails of images, RAW files and, with the `ffmpeg` feature, videos; the CPU, RAM and VRAM meters of the bottom bar.
5. The workspace round trip above, and that `%APPDATA%\smart-media\workspace.json` is written.
6. The program asks for administrator rights in release builds (the manifest); run a debug build when that is in the way.

## macOS (manual, best effort)

Not built by the project today. When a build exists: the same list as Windows without the drive and WSL items, `~/Library/Application Support` and `~/Library/Preferences` instead of `%APPDATA%` for the files named above (`directories` decides), the window buttons, Cmd instead of Ctrl for the shortcuts, and no video thumbnails unless the `ffmpeg` feature is built.
