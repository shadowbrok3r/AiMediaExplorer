//! Startup behaviour: records the open browser scopes and reopens them on the next launch when General asks for it.

use std::time::{Duration, Instant};

use crate::app::SmartMediaContext;
use crate::command::{AppCommand, Scope, Target};
use crate::settings::ui_prefs::{self, Startup};
use crate::ui::file_table::scope::nav::check_openable;

/// Longest wait for the library before a restore opens the scopes that do not need it.
pub const LIBRARY_WAIT: Duration = Duration::from_secs(20);

/// Scopes that are opened again on launch; searches by text or file need an engine and are left out.
pub fn is_restorable(scope: &Scope) -> bool {
    !matches!(scope, Scope::Semantic(_) | Scope::Similar(_))
}

/// True when the command line already says what to open.
pub fn explicit_request(args: &[String]) -> bool {
    args.iter().any(|a| {
        let flag = a.split_once('=').map_or(a.as_str(), |(f, _)| f);
        matches!(flag, "--run" | "--scope")
    })
}

/// The scopes to open on launch, the first for the focused tab and the others as background tabs.
pub fn plan(startup: &Startup) -> Vec<Scope> {
    let wanted: Vec<&String> = if startup.restore_tabs {
        startup.tabs.iter().collect()
    } else if startup.restore_last_scope {
        startup.last_scope.iter().collect()
    } else {
        Vec::new()
    };
    let mut scopes: Vec<Scope> = Vec::new();
    for text in wanted {
        match Scope::parse(text) {
            Ok(scope) if is_restorable(&scope) && !scopes.contains(&scope) => scopes.push(scope),
            Ok(_) => {}
            Err(e) => log::warn!("[startup] skipping saved scope: {e}"),
        }
    }
    scopes
}

/// The commands that open `scopes`; scopes that cannot be shown are skipped.
pub fn commands(scopes: &[Scope]) -> Vec<AppCommand> {
    let mut out = Vec::new();
    for scope in scopes {
        if let Err(why) = check_openable(scope) {
            log::info!("[startup] not restoring {scope}: {why}");
            continue;
        }
        let target = if out.is_empty() { Target::Current } else { Target::Background };
        out.push(AppCommand::OpenScope { scope: scope.clone(), target, title: None, recursive: false });
    }
    out
}

/// Stores the scopes now open; true when the saved ones changed.
pub fn record(startup: &mut Startup, focused: Option<String>, tabs: Vec<String>) -> bool {
    let changed = startup.last_scope != focused || startup.tabs != tabs;
    startup.last_scope = focused;
    startup.tabs = tabs;
    changed
}

/// Where the launch is in reopening what was open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Waiting for the first frames, and for the library when the saved scopes need it.
    Waiting,
    /// Reopened or skipped; the open scopes are recorded from now on.
    Recording,
}

/// Frames after a restore during which nothing is recorded, so the saved scopes survive until the tabs exist.
const SETTLE_FRAMES: u8 = 3;

/// The launch restore of one run.
#[derive(Debug)]
pub struct Tracker {
    pub stage: Stage,
    started: Instant,
    explicit: bool,
    quiet: u8,
}

impl Default for Tracker {
    fn default() -> Self {
        let args: Vec<String> = std::env::args().skip(1).collect();
        Self::new(explicit_request(&args))
    }
}

impl Tracker {
    pub fn new(explicit: bool) -> Self {
        Self { stage: Stage::Waiting, started: Instant::now(), explicit, quiet: 0 }
    }

    /// What to do this frame: the commands to send once, when the saved scopes should open now.
    pub fn step(&mut self, startup: &Startup, db_ready: bool, now: Instant) -> Vec<AppCommand> {
        if self.stage == Stage::Recording {
            return Vec::new();
        }
        let scopes = if self.explicit { Vec::new() } else { plan(startup) };
        if scopes.is_empty() {
            self.stage = Stage::Recording;
            return Vec::new();
        }
        let needs_library = scopes.iter().any(|s| !matches!(s, Scope::Folder(_) | Scope::Archive(_)));
        if needs_library && !db_ready && now.saturating_duration_since(self.started) < LIBRARY_WAIT {
            return Vec::new();
        }
        self.stage = Stage::Recording;
        self.quiet = SETTLE_FRAMES;
        commands(&scopes)
    }

    /// True when this frame should record the open scopes.
    pub fn should_record(&mut self) -> bool {
        if self.stage != Stage::Recording {
            return false;
        }
        if self.quiet > 0 {
            self.quiet -= 1;
            return false;
        }
        true
    }
}

/// Scope strings of the focused browser and of every open browser, the focused one first.
pub fn open_scopes(context: &SmartMediaContext) -> (Option<String>, Vec<String>) {
    use crate::ui::tabs::TabKind;
    let focused = context.active_kind();
    let mut entries: Vec<(TabKind, Scope)> = Vec::new();
    if context.open_tabs.contains(&TabKind::MAIN) {
        entries.push((TabKind::MAIN, context.file_explorer.scope().clone()));
    }
    let mut others: Vec<_> = context.filtered_tabs.iter().collect();
    others.sort_by_key(|(id, _)| **id);
    entries.extend(others.into_iter().map(|(id, ex)| (TabKind::browser(*id), ex.scope().clone())));
    entries.retain(|(_, scope)| is_restorable(scope));
    entries.sort_by_key(|(kind, _)| *kind != focused);
    let focused_scope = entries.iter().find(|(kind, _)| *kind == focused).map(|(_, scope)| scope.format());
    (focused_scope, entries.into_iter().map(|(_, scope)| scope.format()).collect())
}

/// Runs the launch restore and records the open scopes; call once per frame.
pub fn tick(context: &mut SmartMediaContext) {
    if context.startup.stage == Stage::Waiting {
        let startup = ui_prefs::get().startup;
        for command in context.startup.step(&startup, context.db_ready, Instant::now()) {
            crate::command::send(command);
        }
    }
    if context.startup.should_record() {
        let (focused, tabs) = open_scopes(context);
        ui_prefs::update(|p| {
            record(&mut p.startup, focused, tabs);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn startup(tabs: &[&str], last: Option<&str>, restore_tabs: bool, restore_scope: bool) -> Startup {
        Startup { restore_tabs, restore_last_scope: restore_scope, last_scope: last.map(str::to_string), tabs: tabs.iter().map(|t| t.to_string()).collect() }
    }

    #[test]
    fn nothing_is_restored_unless_asked() {
        assert!(plan(&startup(&["lib:all"], Some("lib:all"), false, false)).is_empty());
    }

    #[test]
    fn the_last_scope_alone_opens_one_scope_and_tabs_open_them_all() {
        let s = startup(&["tag:sun", "lib:all"], Some("tag:sun"), false, true);
        assert_eq!(plan(&s), vec![Scope::Tag("sun".into())]);
        let s = startup(&["tag:sun", "lib:all", "tag:sun"], Some("tag:sun"), true, false);
        assert_eq!(plan(&s), vec![Scope::Tag("sun".into()), Scope::Library], "a scope is opened once");
    }

    #[test]
    fn searches_and_unreadable_scopes_are_left_out() {
        let s = startup(&["q:sem:\"a cat\"", "sim:/a.jpg", "nonsense", "recent:all"], None, true, false);
        assert_eq!(plan(&s), vec![Scope::Recent]);
    }

    #[test]
    fn the_first_scope_replaces_the_focused_tab_and_the_rest_open_behind_it() {
        let dir = tempfile::tempdir().unwrap();
        let folder = Scope::Folder(dir.path().display().to_string());
        let cmds = commands(&[folder.clone(), Scope::Library, Scope::Folder("/definitely/not/here".into())]);
        assert_eq!(cmds.len(), 2, "a folder that is gone is skipped");
        assert!(matches!(&cmds[0], AppCommand::OpenScope { scope, target: Target::Current, .. } if *scope == folder));
        assert!(matches!(&cmds[1], AppCommand::OpenScope { scope: Scope::Library, target: Target::Background, .. }));
    }

    #[test]
    fn recording_reports_only_real_changes() {
        let mut s = Startup::default();
        assert!(record(&mut s, Some("lib:all".into()), vec!["lib:all".into()]));
        assert!(!record(&mut s, Some("lib:all".into()), vec!["lib:all".into()]));
        assert!(record(&mut s, Some("lib:all".into()), vec!["lib:all".into(), "tag:a".into()]));
        assert!(record(&mut s, None, Vec::new()));
    }

    #[test]
    fn a_command_line_scope_or_run_wins_over_the_restore() {
        let args = |list: &[&str]| list.iter().map(|a| a.to_string()).collect::<Vec<_>>();
        assert!(explicit_request(&args(&["--scope", "lib:all"])));
        assert!(explicit_request(&args(&["--run={\"cmd\":\"x\"}"])));
        assert!(!explicit_request(&args(&["--devtools", "tasks", "--wizard", "folders"])));
        let mut t = Tracker::new(true);
        assert!(t.step(&startup(&["fs:/"], Some("fs:/"), true, true), true, Instant::now()).is_empty());
        assert_eq!(t.stage, Stage::Recording);
    }

    #[test]
    fn saved_library_scopes_wait_for_the_library_but_not_forever() {
        let s = startup(&["lib:all"], Some("lib:all"), true, false);
        let mut t = Tracker::new(false);
        let start = Instant::now();
        assert!(t.step(&s, false, start).is_empty());
        assert_eq!(t.stage, Stage::Waiting);
        assert_eq!(t.step(&s, true, start).len(), 1, "once the library is ready the scopes open");
        assert_eq!(t.stage, Stage::Recording);
        assert!(t.step(&s, true, start).is_empty(), "only once");

        let mut late = Tracker::new(false);
        assert_eq!(late.step(&s, false, start + LIBRARY_WAIT + Duration::from_secs(1)).len(), 1, "a library that never opens does not hold the restore forever");
    }

    #[test]
    fn folders_open_without_waiting_for_the_library() {
        let dir = tempfile::tempdir().unwrap();
        let text = format!("fs:{}", dir.path().display());
        let s = startup(&[text.as_str()], Some(text.as_str()), true, false);
        let mut t = Tracker::new(false);
        assert_eq!(t.step(&s, false, Instant::now()).len(), 1);
    }

    #[test]
    fn nothing_is_recorded_until_the_restore_has_settled() {
        let s = startup(&["lib:all"], Some("lib:all"), true, false);
        let mut t = Tracker::new(false);
        assert!(!t.should_record(), "still waiting");
        t.step(&s, true, Instant::now());
        let frames: Vec<bool> = (0..5).map(|_| t.should_record()).collect();
        assert_eq!(frames, vec![false, false, false, true, true]);
        let mut plain = Tracker::new(false);
        plain.step(&Startup::default(), false, Instant::now());
        assert!(plain.should_record(), "without a restore recording starts at once");
    }
}
