# Scale and performance

How the browser behaves on a real library (18,758 rows, a 2.4 GB SurrealKV store with thumbnails, descriptions, tags, collections and CLIP embeddings), how to measure it again, and what was changed because of what the measurements showed.

## Tools

| Tool | What it gives |
|---|---|
| Devtools > Scale (`{"cmd": "open_devtools", "window": "scale"}`, `--devtools scale`) | Live counters: resident memory (heap against mapped files), CPU, the start-up milestones, the rows of the focused tab (store, view index, selection, memory per row), frame times of the whole application, the thumbnail caches, database queries by call site, who asked for the frames, and the buttons and report table of scripted runs |
| `scale_bench` command ([commands.md](commands.md#scale_bench)) | Scripted suites against the focused tab: `scroll`, `ui`, `scopes`, `record`, `all`; every step is measured and the report is written as JSON |
| `src/database/scope_queries/counter.rs` | One counter per call site that queries the database for the browser: scope pages, CLIP checks, lookups, sidebar lists, thumbnail reads, presence checks and writes, tile facts |
| `--features alloc-stats` | A counting allocator; the Scale window and the reports then show the allocations of the interface thread per frame |
| `scale.sh` (below) | Starts a private X server and the application on a copy of the library, runs the suite and a wheel test, takes screenshots, prints the tables |

A step report holds: frames and frames per second; minimum, mean, median, 95th percentile and maximum of the frame time, of the time the interface took to build and of the CPU time of the frame; the three slowest frames and when they happened; the database queries by call site; CPU use of the process in percent of one core; resident memory split into heap and mapped files; the texture cache and how many textures were evicted and decoded; the frames each place that asks for a repaint was behind; and for a scope step the load record of the tab: rows, batches, time to the first batch, total time, time spent reading pages and waiting for CLIP checks.

## Method

- Test data: a copy of the real library (`ai_search`, 18,758 rows, 17,646 CLIP embeddings, 1.4 GB of thumbnail blobs). The media files of the rows lived on another drive, so every row is database only and shows the offline badge; library scopes work as usual, folder scopes are not part of the test. The pristine copy is never opened; every run starts from a working copy.
- The application runs under a private Xvfb with Mesa software rendering (`LIBGL_ALWAYS_SOFTWARE=1`), an isolated `HOME`, and in a network namespace with only loopback (`unshare -rn`): the Hugging Face client of the application downloads models in spite of `HF_HUB_OFFLINE=1`, so that variable is not enough.
- Machine: AMD Ryzen 9 9950X (32 threads), 91 GB of RAM, shared with other builds while measuring. Frame times are those of software rendering at 1000x750 and are far above what the GPU of the same machine gives; the time the interface takes to build and the CPU use of the application say what the code costs. Compare runs of the same binary back to back and treat differences of a few milliseconds as noise.
- "Before" is the binary built from the same tree with the measurement tools and none of the fixes; "after" is this tree. Pairs ran back to back with a warm database cache unless a row says cold (page cache of the store dropped first).

## Numbers on this machine

All numbers are from a library of 18,758 rows with the default filters (14,478 rows shown; the filters hide 4,280 small files).

### Loading scopes

Time from the open command to the last row, and to the first batch, in milliseconds; two pairs of runs.

| Scope | Rows | Before total (first batch) | After total (first batch) |
|---|---|---|---|
| Library, all | 18,758 | 3,255 to 4,727 (200 to 910) | 518 to 647 (61 to 72) |
| Library again with the Task Center open | 18,758 | 3,413 | 467 |
| Recent | 1,000 | 183 to 262 (113 to 159) | 182 to 226 (116 to 145) |
| Undescribed | 17,498 | 3,418 to 4,155 (72 to 111) | 1,498 to 1,763 (111 to 147) |
| Untagged | 18,388 | 3,801 to 3,854 (122 to 90) | 1,328 to 1,729 (128 to 160) |
| Biggest tag | 216 | 209 to 263 | 223 to 276 |
| Biggest category | 157 | 62 to 74 | 77 to 78 |
| Biggest collection (empty here) | 0 | | 1 to 2 |
| Duplicates | 2 | 507 to 523 | 465 to 579 |

After the change the library load is bound by the page reads (38 pages of 500 rows at 11 ms each); before it, 84% of the time went to the CLIP check of each page.

### Start-up

Seconds since the process started, warm and cold.

| Milestone | Before | After |
|---|---|---|
| First frame | 0.12 to 0.24 | 0.11 to 0.24 |
| Database connected | 0.8 to 4.2 | 0.7 to 1.8 |
| Database ready (schema and defaults) | 0.8 to 4.2 | 0.8 to 2.7 |
| First library rows | 1.7 to 5.1 | 1.0 to 2.9 |
| Library loaded | 4.4 to 8.9 | 1.5 to 3.3 |
| Library loaded, seconds after the database was ready, warm | 3.3 to 4.7 | 0.6 to 0.7 |
| Library loaded, seconds after the database was ready, cold store | 3.6 | 1.7 |

Opening the store takes what the engine takes (0.7 to 4.2 s over all runs, whatever the binary) and is not part of this work; under heavy load from other builds the first frame took up to 3.8 s. The last two rows are the part this work changes.

### Frames

Final binary, software rendering, the grid at 160 px tiles over 14,478 rows; times in milliseconds (the details and the grid tables of the same suite are within noise of the binary before the changes).

| Step | Frames per s | Frame mean / p95 / max | Interface build (mean) |
|---|---|---|---|
| Scroll the whole library, grid (10 s) | 98 | 10.3 / 15.3 / 43.6 | 1.5 |
| Scroll the whole library, details | 94 | 10.6 / 15.3 / 23.6 | 2.5 |
| Tile size 96, 256, 160 | 102 to 121 | 8.3 to 9.9 / 11.4 to 14.3 / 13.9 to 17.0 | 1.1 to 1.5 |
| Sort by name, size, modified | 94 to 103 | 9.7 to 10.7 / 12.9 to 17.1 / 15.7 to 20.3 | 1.4 to 1.8 |
| Grid to details and back | 56 to 197 | 5.1 to 17.7 / 8.6 to 37.2 / 13.3 to 62.4 | 0.9 to 1.6 |
| Filter chips on and off | 105 to 261 | 3.9 to 9.8 / 6.2 to 20.4 / 10.4 to 25.5 | 0.7 to 1.4 |
| Select all (18,758 rows, action bar showing) | 87 | 11.2 / 17.0 / 28.9 | 1.7 |
| Task Center open | 74 | 13.7 / 18.6 / 26.5 | 1.4 |

Scrolling the library from the first row to the last in ten seconds, about 2,000 tile rows, builds the interface in 1.5 ms per frame on average; the rest of the frame time is software rendering. Loading the library with the Task Center open took 467 ms against 472 ms without it.

### Idle

Thirty seconds with the library open, software rendering, no window of Devtools open.

| State | Before | After |
|---|---|---|
| Nothing going on | 1.4 frames/s, 7.0% of a core | 1.4 frames/s, 7 to 8% |
| A toast on screen | 108 frames/s, 664% | 3.5 frames/s, 23% |
| An engine status running (a database snapshot) | 102 frames/s, 507% | 23 frames/s, 118% |
| Empty library, first-run wizard waiting | 332% | 113% |

The 1.4 frames per second at rest come from the one-second poll of the task registry (`src/main.rs`) and the sidebar lists that reload after 8, 20 and 30 seconds; with a GPU such a frame costs a few milliseconds. In steady state the browser makes no database queries per frame: 4 to 5 queries in 30 seconds, all of the sidebar.

### Memory

| Number | Value |
|---|---|
| Heap when settled with the library open | 1.83 GB before, 1.84 to 1.95 GB after |
| Mapped files when settled | 0.99 to 1.94 GB: the 3 GB debug executable and the database store; depends on what the page cache holds, not on the code |
| Rows in memory | 23.6 MiB for 18,758 rows (1.29 KiB per row, the view index adds 4 bytes) |
| Texture cache | 256 MiB budget, full during the first pass and flat after it (about 920 textures); 5,500 evictions per pass of the whole library |
| Encoded thumbnail cache | 32 MiB, flat |
| Heap growth over five passes of the whole library (grid, details, grid, details, grid) | +437, +297, +158, +137 and +104 MB, then +13 MB over the `ui` suite: the caches of the application are flat, so the growth is the allocator and the engine; it slows with every pass and was not followed further |

### Database queries

| Where | Before | After |
|---|---|---|
| Whole library load | 125 (38 page reads, 38 CLIP checks, 38 presence checks) | 119 (the same, the CLIP checks cheaper) |
| Collection counts for tiles while scrolling the library for 10 s | 629 to 757 | 0 |
| Thumbnail reads while scrolling for 10 s | about one per frame (630 to 1,500) | the same |
| Sidebar lists during the first library load | 7 | 0 to 2 |
| Idle, per 30 s | 4 to 5 | 4 to 5 |

The queries behind the CLIP check, measured directly on the working copy (milliseconds for one page of 500 rows):

| Query | Time |
|---|---|
| Page of rows, `ORDER BY path` over the path index | 15 to 22 |
| CLIP check, `path IN [500 paths]` | 113 to 132 (943 cold) |
| CLIP check, `path >= first AND path <= last` | 32 to 45 |
| All paths of the CLIP table in one query | 1,000 to 1,190 |
| Tag counts of the whole library | 94 |
| Category counts | 61 to 95 |

## What was found and changed

1. **The CLIP check dominated every big load.** A page of 500 rows takes 15 to 22 ms, its CLIP check (`path IN` with 500 paths is a union of 500 index lookups) 113 to 132 ms, and the loader waited for one after the other. Library pages are consecutive in path order, so one scan over the unique path index between the first and the last path gives the same answer in 32 to 45 ms (`embedded_in_range`). The loader now reads pages ahead while up to four earlier pages are checked (`stream_pages`), keeps the batches in page order, and the No CLIP view no longer checks at all (it filters on the very thing the check finds). Library: 3.3 to 4.7 s down to 0.5 to 0.6 s; the first batch arrives after 61 to 72 ms. Tag, category, undescribed and untagged scopes keep the lookup form (their rows are not consecutive) and gain the read-ahead: undescribed and untagged load in 1.3 to 1.8 s instead of 3.4 to 4.2 s.
2. **Every toast kept the application drawing at full speed.** `Toast::default()` of the toast crate never expires and, with its progress bar on, asks for a repaint every frame; the application added its toasts that way (`src/receive.rs`). One "Scan finished" toast was enough for 100 frames per second until the user clicked it away (664% of CPU with software rendering). Toasts now carry no progress bar and ask for no frames while they wait: 3.5 frames per second, 23%.
3. **Spinning gears redrew at the full frame rate.** An engine status that was running or starting made the gear spin with an unconditional `request_repaint()` (`src/ui/status.rs`), and a status that never leaves Initializing, such as the database of an unfinished first run, kept the whole interface at the full frame rate. The gear now asks for a frame every 50 ms: 102 to 23 frames per second while a status runs.
4. **Tile facts were probed for tiles that scrolled past.** Every drawn tile started a collection-count query (one background thread and one query per frame while scrolling, 629 to 757 per ten seconds). A probe now starts only for a tile that stayed drawn for 60 ms and is dropped when the tile is not drawn for six frames: no queries during a scroll, the badges of tiles that stay appear after about 120 ms.
5. **The view preferences were written on the interface thread.** Switching the mode or the tile size wrote `view_prefs.json` (create, write, rename) between two frames. On a quiet disk that takes 0.15 ms, with the disk busy 14.5 ms on average and up to 1,125 ms (a synchronous write and a `sync` loop ran next to it); the runs before the change that shared the machine with other builds showed single frames of 89 to 1,316 ms in exactly the steps that change the mode or the tile size, the seven runs after it showed one frame above 63 ms (264 ms, the first switch to the details view of a run). A writer thread now takes the write (0.007 ms to hand over); each write has its own temporary file.
6. **A scope opened right at the start could fail with "Specify a namespace to use".** The loader waited for the database handle to be connected, which is true before the namespace is selected; it now waits for the database to be ready (`wait_for_connection`).
7. **Every finished task reloaded every sidebar list, the places included.** The lists reload at most every two seconds after a change now, and the places, which come from the system, are left to their own timer: 7 sidebar queries during the first library load become 0 to 2.

## Not changed

- The task registry polls once a second (`IDLE_POLL`, `src/main.rs`), so an idle application draws a frame per second; making engine statuses wake the interface by themselves would remove that.
- Opening the store takes 0.7 to 4.2 s; with this library the process holds 0.4 GB before the store opens and 2.8 GB after the first library load, 1.8 GB of it heap. The settings of the engine are not exposed.
- The default filter "Skip icons" probes the dimensions of every image of the library on a background thread each time the library is opened; with the media on a slow drive that is one file open per row.
- The thumbnail service issues one batched read per frame while new tiles scroll in (630 to 1,500 per ten seconds); batching over a few frames would cut that at the price of a later first pixel.
- The CLIP check is still 30 to 40 ms of database work per page; showing the CLIP column lazily for visible rows would remove it from the load but changes what the column shows while the load runs.
- The heap still grew by 104 MB on the fifth pass of the whole library and by 13 MB over the `ui` suite after it (see Memory); its source is not one of the application caches and was not tracked down.
- The sidebar layout file is still written on the interface thread (rarely: when a section is opened or closed).
- Models: `HF_HUB_OFFLINE=1` does not stop the downloads of the application (about 1 GB in 15 seconds on the first library open); the test setup uses a network namespace instead.
- Windows: the process meters (memory, CPU, start-up time) report nothing there; the counters and the suites work.

## The recipe

`scale.sh` builds nothing: it needs a built binary, a pristine copy of the library and the tools named in its header. Run it from a scratch directory outside the repository; it leaves the reports, samplers and screenshots in `out/scale-<time>/`.

```bash
#!/usr/bin/env bash
# scale.sh: scale and performance smoke run of the AiMediaExplorer binary on a copy of a real library.
#
# usage: scale.sh [--fresh] [--suite all|scopes|scroll|ui|record] [--bin NAME] [--seconds N] [--out DIR] [--no-wheel]
#
# Builds nothing. Uses the binary that is already built (hard-linked under a private name), a private Xvfb, a private
# network namespace (loopback only, no network), an isolated HOME and the working library copy. Needs: Xvfb, xdotool,
# ImageMagick import, unshare, ip, python3.
#
# Environment (all optional):
#   AME_SCALE_ROOT      scratch root                      (default: the directory of this script)
#   AME_SCALE_PRISTINE  pristine copy of the library      (default: $AME_SCALE_ROOT/../realcopy/db)
#   AME_SCALE_SOURCE    built binary to run               (default: $AME_SCALE_ROOT/target/debug/smart-media)
#   AME_SCALE_DISPLAY   X display for the private server  (default :187, must be free)
#   AME_SCALE_SCREEN    screen size                       (default 1400x1000x24)
#
# Steps: (1) the scale_bench command runs the chosen suite inside the application and writes a JSON report;
# (2) a second launch records 24 s of the library grid while xdotool turns the mouse wheel over it (real input events);
# (3) a third launch lets the library settle and shoots the Perf and Scale windows at rest; (4) screenshots of the Scale
# window after the suite, and of the grid and the details view at the full window size; (5) a process sampler writes RSS and CPU
# once a second; everything is stopped by PID and the isolated home is deleted.
set -u

ROOT=${AME_SCALE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
PRISTINE=${AME_SCALE_PRISTINE:-$ROOT/../realcopy/db}
SOURCE=${AME_SCALE_SOURCE:-$ROOT/target/debug/smart-media}
DISP=${AME_SCALE_DISPLAY:-:187}
SCREEN=${AME_SCALE_SCREEN:-1400x1000x24}
SUITE=all
BIN=ame-scale
FRESH=0
WHEEL=1
OUT=
SECONDS_ARG=

while [ $# -gt 0 ]; do
  case $1 in
    --fresh) FRESH=1 ;;
    --suite) SUITE=$2; shift ;;
    --bin) BIN=$2; shift ;;
    --seconds) SECONDS_ARG=$2; shift ;;
    --out) OUT=$2; shift ;;
    --no-wheel) WHEEL=0 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
  shift
done

STAMP=$(date +%Y%m%d-%H%M%S)
OUT=${OUT:-$ROOT/out/scale-$STAMP}
WORK=$ROOT/work
HOME_DIR=$WORK/home
DBDIR=$WORK/db
mkdir -p "$OUT" "$WORK/bin" "$WORK/tmp" "$WORK/cwd"

[ -x "$SOURCE" ] || { echo "no built binary at $SOURCE" >&2; exit 1; }
ln -f "$SOURCE" "$WORK/bin/$BIN" 2>/dev/null || cp "$SOURCE" "$WORK/bin/$BIN"
[ -x "$WORK/bin/xvfb-scale" ] || cp "$(command -v Xvfb)" "$WORK/bin/xvfb-scale"

if [ "$FRESH" = 1 ] || [ ! -d "$DBDIR/ai_search" ]; then
  rm -rf "$DBDIR"
  cp -r "$PRISTINE" "$DBDIR"
fi

# Pictures folder with a few generated files; the library rows themselves are database-only.
mkdir -p "$WORK/media"
[ -e "$WORK/media/gen1.png" ] || python3 -I - "$WORK/media" <<'PY'
import struct, sys, zlib
def png(path, w, h, rgb):
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
for i, rgb in enumerate([(200, 60, 60), (60, 200, 60), (60, 60, 200)], 1):
    png(f"{sys.argv[1]}/gen{i}.png", 320, 240, rgb)
PY

summarize() {
  python3 -I - "$1" <<'PY'
import json, sys

def f(v, nd=1):
    return "-" if v is None else f"{v:.{nd}f}"

r = json.load(open(sys.argv[1]))
print(f"== {sys.argv[1]}: suite {r['suite']}, {r['seconds']:.1f} s, rss {f(r.get('rss_start_mb'),0)} -> {f(r.get('rss_end_mb'),0)} MB, peak {f(r.get('peak_rss_mb'),0)} MB")
if r.get("marks"):
    print("   marks: " + ", ".join(f"{n}={t:.2f}s" for n, t in r["marks"]))
if r.get("timings"):
    print("   lookups: " + ", ".join(f"{n}={t:.0f}ms" for n, t in r["timings"]))
print(f"{'step':28} {'frames':>6} {'fps':>6} {'rows':>6} {'first ms':>8} {'total ms':>8} {'dt avg':>7} {'p95':>6} {'max':>7} {'ui avg':>6} {'queries':>7} {'cpu%':>5} {'rssMB':>6}")
for s in r["steps"]:
    l = s.get("load") or {}
    sp = s.get("spread") or {}
    dt, ui = sp.get("dt", {}), sp.get("ui", {})
    print(f"{s['label'][:28]:28} {s['frames']:6d} {s['fps']:6.1f} {(l.get('rows') if l else s['view_rows']):6} {f(l.get('first_batch_ms'),0):>8} {f(l.get('elapsed_ms'),0):>8} "
          f"{f(dt.get('avg')):>7} {f(dt.get('p95')):>6} {f(dt.get('max')):>7} {f(ui.get('avg')):>6} {s['queries']:7d} {f(s.get('cpu_percent'),0):>5} {f(s.get('rss_mb'),0):>6}"
          + (f"  [{s['note']}]" if s.get("note") else ""))
    if s["queries_by_site"]:
        print("      queries: " + ", ".join(f"{k}={v}" for k, v in s["queries_by_site"].items()))
    if s.get("repaint_causes") and s["label"] in ("idle", "record"):
        print("      repaint causes: " + "; ".join(f"{n}x {c}" for c, n in s["repaint_causes"]))
PY
}

setup_home() {
  rm -rf "$HOME_DIR"
  mkdir -p "$HOME_DIR/.config/smart-media" "$HOME_DIR/.local/share" "$HOME_DIR/.cache" "$HOME_DIR/hf"
  printf 'XDG_PICTURES_DIR="%s"\nXDG_DESKTOP_DIR="%s"\n' "$WORK/media" "$WORK/media" > "$HOME_DIR/.config/user-dirs.dirs"
  printf '%s' "$DBDIR/ai_search" > "$HOME_DIR/.config/smart-media/db_location.txt"
}

start_xvfb() {
  if [ -e "/tmp/.X11-unix/X${DISP#:}" ]; then echo "display $DISP is already in use" >&2; return 1; fi
  "$WORK/bin/xvfb-scale" "$DISP" -screen 0 "$SCREEN" >/dev/null 2>&1 &
  XPID=$!
  sleep 1.5
}

# start_app LOG args...: the app in a private network namespace with the isolated environment; sets APID.
start_app() {
  local log=$1
  shift
  cd "$WORK/cwd" || return 1
  unshare -rn sh -c 'ip link set lo up; exec "$@"' sh env -i PATH="$PATH" HOME="$HOME_DIR" XDG_CONFIG_HOME="$HOME_DIR/.config" \
    XDG_DATA_HOME="$HOME_DIR/.local/share" XDG_CACHE_HOME="$HOME_DIR/.cache" TMPDIR="$WORK/tmp" DISPLAY="$DISP" \
    HF_HOME="$HOME_DIR/hf" HF_HUB_OFFLINE=1 LIBGL_ALWAYS_SOFTWARE=1 RUST_BACKTRACE=1 "$WORK/bin/$BIN" "$@" >"$log" 2>&1 &
  APID=$!
}

stop_all() {
  if [ -n "${APID:-}" ]; then kill "$APID" 2>/dev/null; sleep 1; kill -9 "$APID" 2>/dev/null; fi
  if [ -n "${SPID:-}" ]; then kill "$SPID" 2>/dev/null; fi
  if [ -n "${XPID:-}" ]; then kill "$XPID" 2>/dev/null; fi
  APID=; SPID=; XPID=
}
trap 'stop_all; rm -rf "$HOME_DIR"' EXIT

# sample CSV: once a second the elapsed time, RSS (KiB), CPU ticks and thread count of the app.
sample() {
  local csv=$1 i=0
  echo "t,rss_kib,cpu_ticks,threads" > "$csv"
  while kill -0 "$APID" 2>/dev/null; do
    printf '%s,%s,%s,%s\n' "$i" \
      "$(awk '/VmRSS/{print $2}' /proc/$APID/status 2>/dev/null)" \
      "$(awk '{s=$0; sub(/.*\) /,"",s); n=split(s,a," "); print a[12]+a[13]}' /proc/$APID/stat 2>/dev/null)" \
      "$(awk '/Threads/{print $2}' /proc/$APID/status 2>/dev/null)" >> "$csv"
    sleep 1
    i=$((i+1))
  done
}

shot() { DISPLAY=$DISP import -window root "$OUT/$1.png" 2>/dev/null; }
xd() { DISPLAY=$DISP xdotool "$@"; }

# fullsize: stretches the application window over the screen.
fullsize() {
  local w=${SCREEN%%x*} rest=${SCREEN#*x}
  xd search --name "Smart Media" windowmove 0 0 windowsize "$w" "${rest%%x*}" 2>/dev/null
  sleep 2
}

# ---------------------------------------------------------------- run 1: the suite inside the application
SEC=
[ -n "$SECONDS_ARG" ] && SEC=",\"seconds\":$SECONDS_ARG"
setup_home
start_xvfb || exit 1
start_app "$OUT/suite.log" --devtools scale --run "{\"cmd\":\"scale_bench\",\"suite\":\"$SUITE\"$SEC,\"report\":\"$OUT/$SUITE.json\"}"
sample "$OUT/suite-samples.csv" &
SPID=$!
echo "suite $SUITE running; report: $OUT/$SUITE.json"
LIMIT=900
while [ ! -s "$OUT/$SUITE.json" ] && [ "$LIMIT" -gt 0 ] && kill -0 "$APID" 2>/dev/null; do
  sleep 2
  LIMIT=$((LIMIT-2))
done
sleep 1
fullsize
xd search --name "Smart Media" windowfocus 2>/dev/null
shot scale-window
# The report table is at the bottom of the window.
xd mousemove 300 400
xd click --repeat 60 --delay 10 5
sleep 1
shot scale-window-report
stop_all

# ---------------------------------------------------------------- run 2: real wheel input over the library grid
if [ "$WHEEL" = 1 ]; then
  setup_home
  start_xvfb || exit 1
  start_app "$OUT/wheel.log" --run '{"cmd":"scale_bench","suite":"record","scopes":["lib:all"],"seconds":24,"report":"'"$OUT"'/wheel.json"}'
  sample "$OUT/wheel-samples.csv" &
  SPID=$!
  sleep 16
  xd search --name "Smart Media" windowfocus 2>/dev/null
  xd mousemove 640 520
  for burst in 1 2 3 4 5 6 7 8; do xd click --repeat 25 --delay 8 5; sleep 0.4; done
  LIMIT=60
  while [ ! -s "$OUT/wheel.json" ] && [ "$LIMIT" -gt 0 ] && kill -0 "$APID" 2>/dev/null; do sleep 2; LIMIT=$((LIMIT-2)); done
  # The library at the full window size in both views.
  fullsize
  xd search --name "Smart Media" windowfocus 2>/dev/null
  shot grid-view
  xd key F9; sleep 2; shot details-view
  stop_all
fi

# ---------------------------------------------------------------- run 3: the Perf and Scale windows at rest
setup_home
start_xvfb || exit 1
start_app "$OUT/rest.log" --devtools perf,scale --run lib:all
sleep 45
fullsize
xd search --name "Smart Media" windowfocus 2>/dev/null
sleep 25
shot rest-perf-scale
stop_all

echo "reports and screenshots are in $OUT"
for f in "$OUT"/*.json; do
  [ -e "$f" ] && summarize "$f"
done
```

How to read the result:

- `all.json` (and the table the script prints): look at the **frame mean and p95** of the scroll and view steps first, at the **queries** column (anything above a handful in a step that touches nothing is a loop), at the **idle** step (frames per second and repaint causes: more than two frames per second at rest means something asks for frames), at the load steps (`first ms` is what the user waits for, `total ms` the whole library) and at the **heap** between the scroll passes (it should stop growing).
- `wheel.json`: the same library grid driven by real wheel events.
- `grid-view.png`, `details-view.png`: the library at the full window size; `scale-window.png`, `scale-window-report.png`: the Scale window after the suite; `rest-perf-scale.png`: the Perf and Scale windows at rest.
- Expected on the machine above with software rendering: library load 0.5 to 0.7 s, first batch under 100 ms, scroll steps 90 to 100 frames per second, idle 1.4 frames per second with no repaint cause but the task poll, heap flat after the third scroll pass.
