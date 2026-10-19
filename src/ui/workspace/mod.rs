//! The workspace of the application: what is open and where, written when it changes and restored at launch (see `docs/settings.md`).

pub mod capture;
pub mod dump;
pub mod restore;

use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use eframe::egui::{Context, ViewportBuilder, ViewportCommand, pos2, vec2};
use serde::{Deserialize, Serialize};

use crate::app::SmartMediaApp;
use crate::command::{CommandError, CommandResult};
use crate::settings::ui_prefs::{self, Startup};
use crate::settings::workspace::{LoadNote, Saver, WindowState, Workspace, default_path, load_from};
use capture::WindowInfo;

/// What a `workspace` command does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Op {
    /// Write the file now.
    Save,
    /// Read the file and reopen what it holds, replacing the tabs.
    Restore,
    /// Delete the file.
    Forget,
}

impl Op {
    pub const ALL: [Op; 3] = [Self::Save, Self::Restore, Self::Forget];

    pub fn label(self) -> &'static str {
        match self {
            Self::Save => "Save now",
            Self::Restore => "Reload",
            Self::Forget => "Forget",
        }
    }
}

/// How much of the saved workspace a launch brings back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Off,
    /// The scope of the focused browser, in the main tab.
    Scope,
    /// The dock layout, every browser tab and the window.
    Full,
}

/// True when the command line already says what to open.
pub fn explicit_request(args: &[String]) -> bool {
    args.iter().any(|a| {
        let flag = a.split_once('=').map_or(a.as_str(), |(f, _)| f);
        matches!(flag, "--run" | "--scope")
    })
}

/// What the preferences ask for, unless the command line decides what to open.
pub fn mode_for(startup: &Startup, args: &[String]) -> Mode {
    if explicit_request(args) {
        Mode::Off
    } else if startup.restore_tabs {
        Mode::Full
    } else if startup.restore_last_scope {
        Mode::Scope
    } else {
        Mode::Off
    }
}

/// The saved workspace and what the launch does with it.
#[derive(Debug)]
pub struct Launch {
    pub mode: Mode,
    pub path: Option<PathBuf>,
    pub note: LoadNote,
    /// What the file holds.
    pub file: Option<Workspace>,
    /// What to restore: the file, or for a launch that finds none the tabs older versions kept in the preferences.
    pub workspace: Option<Workspace>,
}

impl Launch {
    /// Reads the workspace file at `path` and decides what the launch restores.
    pub fn read(args: &[String], startup: &Startup, path: Option<PathBuf>) -> Self {
        let mode = mode_for(startup, args);
        let (file, note) = match &path {
            Some(path) => load_from(path),
            None => (None, LoadNote::Memory),
        };
        let legacy = (file.is_none() && mode != Mode::Off && !matches!(note, LoadNote::Corrupt(_) | LoadNote::TooNew(_))).then(|| Workspace::from_legacy(startup, mode == Mode::Full)).flatten();
        let workspace = file.clone().or(legacy);
        Self { mode, path, note, file, workspace }
    }

    /// True when the main explorer should wait for its saved scope instead of scanning the start folder.
    pub fn restoring(&self) -> bool {
        match (self.mode, &self.workspace) {
            (Mode::Full, Some(ws)) => ws.has_tabs(),
            (Mode::Scope, Some(ws)) => ws.focused_browser().is_some(),
            _ => false,
        }
    }

    /// The window to open with.
    pub fn window(&self) -> Option<WindowState> {
        self.workspace.as_ref().filter(|_| self.mode == Mode::Full).and_then(|ws| ws.window)
    }

    /// The saver of this run, which knows what the file holds.
    pub fn saver(&self) -> Saver {
        Saver::from_load(self.path.clone(), self.file.clone(), self.note.clone())
    }
}

static LAUNCH: OnceLock<Launch> = OnceLock::new();

/// The launch of this process: the command line, the preferences and the workspace file read once.
pub fn launch() -> &'static Launch {
    LAUNCH.get_or_init(|| {
        let args: Vec<String> = std::env::args().skip(1).collect();
        Launch::read(&args, &ui_prefs::get().startup, default_path())
    })
}

/// `builder` with the saved window size, place and maximized state.
pub fn with_saved_window(builder: ViewportBuilder, window: Option<WindowState>) -> ViewportBuilder {
    let Some(window) = window.and_then(|w| w.clamped(None)) else { return builder };
    let builder = builder.with_inner_size(vec2(window.size[0], window.size[1])).with_maximized(window.maximized);
    match window.position {
        Some([x, y]) => builder.with_position(pos2(x, y)),
        None => builder,
    }
}

/// True when eframe keeps the window size and place itself, as it always did: not while the workspace file restores them.
pub fn eframe_keeps_window() -> bool {
    launch().mode != Mode::Full
}

/// The window of this launch applied to `builder`.
pub fn launch_viewport(builder: ViewportBuilder) -> ViewportBuilder {
    with_saved_window(builder, launch().window())
}

/// Restores what `launch` holds into `app`; true when something was brought back.
pub fn restore_launch(app: &mut SmartMediaApp, launch: &Launch) -> Option<restore::Applied> {
    let workspace = launch.workspace.as_ref()?;
    let result = match launch.mode {
        Mode::Off => return None,
        Mode::Scope => restore::apply_scope(app, workspace),
        Mode::Full => restore::apply_full(app, workspace),
    };
    match result {
        Ok(applied) => {
            log::info!("[workspace] restored {}", applied.text());
            Some(applied)
        }
        Err(e) => {
            log::warn!("[workspace] not restored: {e}");
            None
        }
    }
}

/// Where the workspace of this run stands.
pub struct Session {
    pub saver: Saver,
    pub mode: Mode,
    /// What the launch restored, for the Devtools window.
    pub restored: Option<String>,
    pub frames: u64,
    /// The workspace as of the last frame.
    pub current: Option<Workspace>,
    /// The geometry the window had last while it was a normal window.
    last_window: Option<WindowState>,
    /// The restored window is still to be checked against the monitor.
    window_check: bool,
    pub dump: Option<dump::Request>,
    pub last_dump: Option<Result<String, String>>,
}

impl Session {
    /// A session that never touches the disk.
    pub fn memory() -> Self {
        Self { saver: Saver::memory(), mode: Mode::Off, restored: None, frames: 0, current: None, last_window: None, window_check: false, dump: None, last_dump: None }
    }

    /// The session of this run after `launch` restored `applied`.
    pub fn live(launch: &Launch, applied: Option<&restore::Applied>) -> Self {
        let restored = match (launch.mode, applied) {
            (Mode::Off, _) => None,
            (_, Some(a)) => Some(a.text()),
            (_, None) => Some("nothing to restore".to_string()),
        };
        Self { saver: launch.saver(), mode: launch.mode, restored, last_window: launch.window(), window_check: launch.window().is_some(), ..Self::memory() }
    }
}

/// The monitor-aware correction of a restored window: the size and place to apply when the window does not fit the monitor.
pub fn window_correction(info: &WindowInfo) -> Option<WindowState> {
    let (inner, outer, monitor) = (info.inner?, info.outer, info.monitor?);
    if info.maximized || info.fullscreen {
        return None;
    }
    let current = WindowState { size: [inner.width(), inner.height()], position: outer.map(|o| [o.min.x, o.min.y]), maximized: false };
    let fixed = current.clamped(Some(monitor))?;
    let moved = fixed.position != current.position && fixed.position.zip(current.position).is_some_and(|(a, b)| (a[0] - b[0]).abs() > 1.0 || (a[1] - b[1]).abs() > 1.0);
    let resized = (fixed.size[0] - current.size[0]).abs() > 1.0 || (fixed.size[1] - current.size[1]).abs() > 1.0;
    (moved || resized).then_some(fixed)
}

impl SmartMediaApp {
    /// Once per frame: takes note of the workspace, writes it when it has been quiet, and runs the dump the command line asked for.
    pub fn workspace_tick(&mut self, ctx: &Context) {
        let now = Instant::now();
        let info = WindowInfo::read(ctx);
        let window = capture::window_state(&info, &mut self.context.workspace.last_window);
        self.context.workspace.frames += 1;
        if self.context.workspace.window_check {
            if let Some(fixed) = window_correction(&info) {
                ctx.send_viewport_cmd(ViewportCommand::InnerSize(vec2(fixed.size[0], fixed.size[1])));
                if let Some([x, y]) = fixed.position {
                    ctx.send_viewport_cmd(ViewportCommand::OuterPosition(pos2(x, y)));
                }
            }
            if info.monitor.is_some() && info.inner.is_some() {
                self.context.workspace.window_check = false;
            }
        }
        let state = capture::capture(self, window);
        let session = &mut self.context.workspace;
        session.saver.observe(now, &state);
        session.saver.poll(now);
        if let Some(wait) = session.saver.wake_in(now) {
            ctx.request_repaint_after(wait + Duration::from_millis(30));
        }
        session.current = Some(state);
        crate::ui::devtools::workspace::publish(ctx, &self.context.workspace);
        self.run_dump_request(ctx);
    }

    /// Writes the workspace when the application closes.
    pub fn workspace_exit(&mut self, ctx: &Context) {
        let info = WindowInfo::read(ctx);
        let window = capture::window_state(&info, &mut self.context.workspace.last_window);
        let state = capture::capture(self, window);
        self.context.workspace.saver.flush(&state);
    }

    /// Runs a `workspace` command.
    pub fn run_workspace_op(&mut self, op: Op) -> CommandResult {
        let window = self.context.workspace.last_window;
        let state = capture::capture(self, window);
        match op {
            Op::Save => {
                let session = &mut self.context.workspace;
                session.saver.save_now(&state).map_err(CommandError::failed)?;
                let path = session.saver.path().map(|p| p.display().to_string()).unwrap_or_default();
                session.current = Some(state);
                Ok(format!("saved the workspace to {path}"))
            }
            Op::Restore => {
                let path = self.context.workspace.saver.path().map(PathBuf::from).ok_or_else(|| CommandError::invalid("memory only, there is no file to read"))?;
                let (loaded, note) = load_from(&path);
                let workspace = loaded.ok_or_else(|| CommandError::not_found(format!("{}: {}", path.display(), note.text())))?;
                let applied = restore::apply_full(self, &workspace).map_err(CommandError::failed)?;
                self.context.workspace.restored = Some(applied.text());
                Ok(format!("restored {}", applied.text()))
            }
            Op::Forget => {
                self.context.workspace.saver.forget().map_err(CommandError::failed)?;
                Ok("deleted the workspace file".to_string())
            }
        }
    }

    /// Queues a state dump: written at once, or once the application is idle.
    pub fn request_dump(&mut self, path: String, when_idle: bool) -> CommandResult {
        if when_idle {
            self.context.workspace.dump = Some(dump::Request::new(path.clone(), Instant::now()));
            return Ok(format!("the state is written to {path} once the application is idle"));
        }
        self.write_dump(&path)
    }

    fn write_dump(&mut self, path: &str) -> CommandResult {
        let settings = crate::settings::store();
        let (snapshot, loaded) = (settings.snapshot(), settings.is_loaded());
        let report = dump::build(self, self.context.workspace.frames, &snapshot, loaded, &ui_prefs::get());
        let result = dump::write(std::path::Path::new(path), &report).map(|bytes| format!("wrote {bytes} bytes to {path}")).map_err(|e| format!("{path}: {e}"));
        self.context.workspace.last_dump = Some(result.clone());
        result.map_err(CommandError::failed)
    }

    fn run_dump_request(&mut self, ctx: &Context) {
        let Some(mut request) = self.context.workspace.dump.take() else { return };
        let activity = dump::Activity { frames: self.context.workspace.frames, loading: dump::is_loading(self) };
        if request.ready(activity, Instant::now()) {
            if let Err(e) = self.write_dump(&request.path) {
                log::warn!("[workspace] {e}");
            }
        } else {
            self.context.workspace.dump = Some(request);
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }
}

#[cfg(test)]
mod tests;
