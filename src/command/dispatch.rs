//! Runs commands against a [`CommandSink`] and records each in the command log.

use std::fmt;

use super::bus::CommandBus;
use super::log::{CommandLog, Outcome};
use super::model::{ActionId, AppCommand, DevtoolsWindow, NavOp, Section, Selector, Target, Tool, ToastLevel};
use super::scope::Scope;
use super::model::RowsTab;
use crate::tasks::{TaskEvent, TaskId};

/// Commands run per frame; the rest wait for the next one.
pub const FRAME_BUDGET: usize = 256;

/// Why a command did nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    /// The request is malformed or empty.
    Invalid,
    /// The application cannot do this yet.
    Unsupported,
    /// The thing named does not exist.
    NotFound,
    /// The attempt was made and failed.
    Failed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommandError {
    pub kind: ErrorKind,
    pub message: String,
}

impl CommandError {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self { kind: ErrorKind::Invalid, message: message.into() }
    }

    pub fn unsupported(message: impl Into<String>) -> Self {
        Self { kind: ErrorKind::Unsupported, message: message.into() }
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self { kind: ErrorKind::NotFound, message: message.into() }
    }

    pub fn failed(message: impl Into<String>) -> Self {
        Self { kind: ErrorKind::Failed, message: message.into() }
    }
}

impl fmt::Display for CommandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self.kind {
            ErrorKind::Invalid => "invalid",
            ErrorKind::Unsupported => "unsupported",
            ErrorKind::NotFound => "not found",
            ErrorKind::Failed => "failed",
        };
        write!(f, "{kind}: {}", self.message)
    }
}

impl std::error::Error for CommandError {}

/// The text describing what a command did.
pub type CommandResult = Result<String, CommandError>;

/// What the application offers to commands. Each method does the work of one command variant.
pub trait CommandSink {
    fn open_scope(&mut self, scope: &Scope, target: Target, title: Option<&str>, recursive: bool) -> CommandResult;
    fn open_rows(&mut self, tab: RowsTab) -> CommandResult;
    fn open_tool(&mut self, tool: Tool, path: Option<&str>, target: Target) -> CommandResult;
    /// The paths `on` stands for in the focused tab.
    fn resolve(&mut self, on: &Selector) -> Vec<String>;
    fn run_action(&mut self, action: ActionId, paths: Vec<String>) -> CommandResult;
    fn show_settings(&mut self, section: Option<Section>) -> CommandResult;
    fn attach_to_assistant(&mut self, paths: Vec<String>) -> CommandResult;
    fn refine(&mut self, paths: Vec<String>) -> CommandResult;
    fn toast(&mut self, level: ToastLevel, text: String) -> CommandResult;
    fn task_update(&mut self, id: TaskId, event: TaskEvent) -> CommandResult;
    fn clear_selection(&mut self) -> CommandResult;
    fn focus_path(&mut self, path: &str) -> CommandResult;
    fn open_devtools(&mut self, window: DevtoolsWindow) -> CommandResult;
    fn show_task_center(&mut self) -> CommandResult;
    /// Moves the focused tab through its history or its load.
    fn navigate(&mut self, op: NavOp) -> CommandResult;
    /// Opens the database at the configured folder.
    fn connect_database(&mut self) -> CommandResult;
    fn set_filter(&mut self, filter: &serde_json::Value, replace: bool) -> CommandResult;
    fn clear_filters(&mut self) -> CommandResult;
    fn apply_saved_view(&mut self, name: &str) -> CommandResult;
}

/// Runs one command; an action on an empty set of files is an error, never "all files".
pub fn dispatch(sink: &mut dyn CommandSink, cmd: AppCommand) -> CommandResult {
    match cmd {
        AppCommand::OpenScope { scope, target, title, recursive } => sink.open_scope(&scope, target, title.as_deref(), recursive),
        AppCommand::OpenRows(tab) => sink.open_rows(tab),
        AppCommand::OpenTool { tool, path, target } => sink.open_tool(tool, path.as_deref(), target),
        AppCommand::Run { action, on } => {
            let paths = sink.resolve(&on);
            if paths.is_empty() {
                return Err(CommandError::invalid(format!("{} has no files to act on", action.label())));
            }
            sink.run_action(action, paths)
        }
        AppCommand::ShowSettings { section } => sink.show_settings(section),
        AppCommand::AttachToAssistant { paths } if paths.is_empty() => Err(CommandError::invalid("no paths to attach")),
        AppCommand::AttachToAssistant { paths } => sink.attach_to_assistant(paths),
        AppCommand::Refine { paths } if paths.is_empty() => Err(CommandError::invalid("no paths to refine")),
        AppCommand::Refine { paths } => sink.refine(paths),
        AppCommand::Toast { kind, text } => sink.toast(kind, text),
        AppCommand::TaskUpdate { id, event } => sink.task_update(id, event),
        AppCommand::ClearSelection => sink.clear_selection(),
        AppCommand::FocusPath { path } => sink.focus_path(&path),
        AppCommand::OpenDevtools { window } => sink.open_devtools(window),
        AppCommand::ShowTaskCenter => sink.show_task_center(),
        AppCommand::Navigate { op } => sink.navigate(op),
        AppCommand::ConnectDatabase => sink.connect_database(),
        AppCommand::SetFilter { filter, replace } => sink.set_filter(&filter, replace),
        AppCommand::ClearFilters => sink.clear_filters(),
        AppCommand::ApplySavedView { name } if name.trim().is_empty() => Err(CommandError::invalid("no view name")),
        AppCommand::ApplySavedView { name } => sink.apply_saved_view(&name),
    }
}

/// What a [`pump`] did.
#[derive(Debug, Default)]
pub struct Pumped {
    pub ran: usize,
    /// Name and error of each command that failed, in run order.
    pub failures: Vec<(&'static str, CommandError)>,
}

/// Drains up to `budget` commands from `bus`, runs each and records it in `log`.
pub fn pump(bus: &CommandBus, log: &mut CommandLog, sink: &mut dyn CommandSink, budget: usize) -> Pumped {
    let batch = bus.drain(budget);
    let mut pumped = Pumped { ran: batch.len(), failures: Vec::new() };
    for cmd in batch {
        let (name, json) = (cmd.name(), cmd.log_json());
        let outcome = match dispatch(sink, cmd) {
            Ok(text) => Outcome::Ok(text),
            Err(e) => {
                log::warn!("[command] {json}: {e}");
                pumped.failures.push((name, e.clone()));
                Outcome::Err(e)
            }
        };
        log.push(json, outcome);
    }
    pumped
}

#[cfg(test)]
pub(crate) mod fake {
    use super::*;

    /// Sink that records which operation each command reached.
    #[derive(Default)]
    pub struct FakeSink {
        pub calls: Vec<String>,
        pub selection: Vec<String>,
        pub fail_with: Option<CommandError>,
    }

    impl FakeSink {
        fn done(&mut self, call: String) -> CommandResult {
            self.calls.push(call);
            match &self.fail_with {
                Some(e) => Err(e.clone()),
                None => Ok("done".to_string()),
            }
        }
    }

    impl CommandSink for FakeSink {
        fn open_scope(&mut self, scope: &Scope, target: Target, title: Option<&str>, recursive: bool) -> CommandResult {
            self.done(format!("open_scope {scope} {target:?} {title:?} {recursive}"))
        }
        fn open_rows(&mut self, tab: RowsTab) -> CommandResult {
            self.done(format!("open_rows {} {}", tab.title, tab.rows.len()))
        }
        fn open_tool(&mut self, tool: Tool, path: Option<&str>, target: Target) -> CommandResult {
            self.done(format!("open_tool {tool:?} {path:?} {target:?}"))
        }
        fn resolve(&mut self, on: &Selector) -> Vec<String> {
            match on {
                Selector::Paths(p) => p.clone(),
                _ => self.selection.clone(),
            }
        }
        fn run_action(&mut self, action: ActionId, paths: Vec<String>) -> CommandResult {
            self.done(format!("run {action:?} {}", paths.join(",")))
        }
        fn show_settings(&mut self, section: Option<Section>) -> CommandResult {
            self.done(format!("show_settings {section:?}"))
        }
        fn attach_to_assistant(&mut self, paths: Vec<String>) -> CommandResult {
            self.done(format!("attach {}", paths.join(",")))
        }
        fn refine(&mut self, paths: Vec<String>) -> CommandResult {
            self.done(format!("refine {}", paths.join(",")))
        }
        fn toast(&mut self, level: ToastLevel, text: String) -> CommandResult {
            self.done(format!("toast {level:?} {text}"))
        }
        fn task_update(&mut self, id: TaskId, event: TaskEvent) -> CommandResult {
            self.done(format!("task {id} {event:?}"))
        }
        fn clear_selection(&mut self) -> CommandResult {
            self.done("clear_selection".to_string())
        }
        fn focus_path(&mut self, path: &str) -> CommandResult {
            self.done(format!("focus {path}"))
        }
        fn open_devtools(&mut self, window: DevtoolsWindow) -> CommandResult {
            self.done(format!("devtools {window:?}"))
        }
        fn show_task_center(&mut self) -> CommandResult {
            self.done("task_center".to_string())
        }
        fn navigate(&mut self, op: NavOp) -> CommandResult {
            self.done(format!("navigate {op:?}"))
        }
        fn connect_database(&mut self) -> CommandResult {
            self.done("connect_database".to_string())
        }
        fn set_filter(&mut self, filter: &serde_json::Value, replace: bool) -> CommandResult {
            self.done(format!("set_filter {filter} {replace}"))
        }
        fn clear_filters(&mut self) -> CommandResult {
            self.done("clear_filters".to_string())
        }
        fn apply_saved_view(&mut self, name: &str) -> CommandResult {
            self.done(format!("apply_saved_view {name}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fake::FakeSink;
    use super::*;

    #[test]
    fn every_command_variant_reaches_its_sink_method() {
        let mut sink = FakeSink { selection: vec!["/sel".into()], ..Default::default() };
        let cmds = vec![
            AppCommand::open(Scope::Library),
            AppCommand::OpenTool { tool: Tool::Logs, path: None, target: Target::Current },
            AppCommand::Run { action: ActionId::Open, on: Selector::Selection },
            AppCommand::ShowSettings { section: Some(Section::Library) },
            AppCommand::AttachToAssistant { paths: vec!["/a".into()] },
            AppCommand::Refine { paths: vec!["/b".into()] },
            AppCommand::Toast { kind: ToastLevel::Info, text: "hi".into() },
            AppCommand::TaskUpdate { id: 4, event: TaskEvent::Complete },
            AppCommand::ClearSelection,
            AppCommand::FocusPath { path: "/f".into() },
            AppCommand::OpenDevtools { window: DevtoolsWindow::Tasks },
            AppCommand::ShowTaskCenter,
            AppCommand::Navigate { op: NavOp::Forward },
            AppCommand::ConnectDatabase,
            AppCommand::SetFilter { filter: serde_json::json!({"skip_icons": true}), replace: true },
            AppCommand::ClearFilters,
            AppCommand::ApplySavedView { name: "Holidays".into() },
        ];
        for cmd in cmds {
            assert!(dispatch(&mut sink, cmd).is_ok());
        }
        assert_eq!(
            sink.calls,
            vec![
                "open_scope lib:all Current None false",
                "open_tool Logs None Current",
                "run Open /sel",
                "show_settings Some(Library)",
                "attach /a",
                "refine /b",
                "toast Info hi",
                "task 4 Complete",
                "clear_selection",
                "focus /f",
                "devtools Tasks",
                "task_center",
                "navigate Forward",
                "connect_database",
                "set_filter {\"skip_icons\":true} true",
                "clear_filters",
                "apply_saved_view Holidays",
            ]
        );
    }

    #[test]
    fn an_action_on_nothing_is_an_error_and_never_runs() {
        let mut sink = FakeSink::default();
        let err = dispatch(&mut sink, AppCommand::Run { action: ActionId::GenerateClip, on: Selector::Selection }).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Invalid);
        assert!(sink.calls.is_empty());
        let err = dispatch(&mut sink, AppCommand::Run { action: ActionId::CopyPath, on: Selector::Paths(vec![]) }).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Invalid);
        assert!(dispatch(&mut sink, AppCommand::AttachToAssistant { paths: vec![] }).is_err());
        assert!(dispatch(&mut sink, AppCommand::Refine { paths: vec![] }).is_err());
        assert!(dispatch(&mut sink, AppCommand::ApplySavedView { name: "  ".into() }).is_err());
        assert!(sink.calls.is_empty());
    }

    #[test]
    fn explicit_paths_bypass_the_selection() {
        let mut sink = FakeSink { selection: vec!["/sel".into()], ..Default::default() };
        dispatch(&mut sink, AppCommand::Run { action: ActionId::Reveal, on: Selector::Paths(vec!["/x".into()]) }).unwrap();
        assert_eq!(sink.calls, vec!["run Reveal /x"]);
    }

    #[test]
    fn pump_runs_in_order_logs_each_and_respects_the_budget() {
        let bus = CommandBus::new();
        let mut log = CommandLog::default();
        let mut sink = FakeSink::default();
        for i in 0..5 {
            bus.push(AppCommand::FocusPath { path: format!("/{i}") });
        }
        assert_eq!(pump(&bus, &mut log, &mut sink, 3).ran, 3);
        assert_eq!(sink.calls, vec!["focus /0", "focus /1", "focus /2"]);
        assert_eq!(bus.len(), 2);
        assert_eq!(pump(&bus, &mut log, &mut sink, 10).ran, 2);
        assert_eq!(log.len(), 5);
        assert!(log.entries().all(|e| e.outcome.is_ok()));
        assert!(log.entries().next().unwrap().json.contains("\"cmd\":\"focus_path\""));
    }

    #[test]
    fn failures_are_logged_with_their_error() {
        let bus = CommandBus::new();
        let mut log = CommandLog::default();
        let mut sink = FakeSink { fail_with: Some(CommandError::unsupported("no loader")), ..Default::default() };
        bus.push(AppCommand::open(Scope::Scan(9)));
        let pumped = pump(&bus, &mut log, &mut sink, 10);
        assert_eq!(pumped.failures.len(), 1);
        assert_eq!(pumped.failures[0].0, "open_scope");
        let entry = log.entries().next().unwrap();
        assert_eq!(entry.outcome, Outcome::Err(CommandError::unsupported("no loader")));
        assert_eq!(entry.outcome.text(), "unsupported: no loader");
        assert_eq!(log.failures(), 1);
    }

    #[test]
    fn an_empty_bus_runs_nothing() {
        let bus = CommandBus::new();
        let mut log = CommandLog::default();
        let mut sink = FakeSink::default();
        assert_eq!(pump(&bus, &mut log, &mut sink, FRAME_BUDGET).ran, 0);
        assert!(log.is_empty());
    }
}
