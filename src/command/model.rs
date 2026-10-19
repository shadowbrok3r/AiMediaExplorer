//! The command vocabulary: serializable requests any part of the app, a script or the assistant can make.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::scope::Scope;
use crate::database::Thumbnail;
use crate::tasks::{TaskEvent, TaskId};
use crate::ui::file_table::FilterRequest;

/// Where a newly opened scope goes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Target {
    /// Replace what the focused tab shows.
    Current,
    /// A new tab that takes focus.
    #[default]
    NewTab,
    /// A new tab behind the focused one.
    Background,
}

/// Which files an action acts on; never an implicit "everything".
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Selector {
    /// The selected rows of the focused tab.
    Selection,
    /// The row under the focus (the previewed file).
    Focus,
    /// Every row of the focused tab.
    View,
    Paths(Vec<String>),
}

/// An operation on files, run through [`AppCommand::Run`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionId {
    Open,
    Reveal,
    CopyPath,
    GenerateClip,
    Describe,
    AttachToAssistant,
    Refine,
}

impl ActionId {
    pub const ALL: [ActionId; 7] = [
        Self::Open,
        Self::Reveal,
        Self::CopyPath,
        Self::GenerateClip,
        Self::Describe,
        Self::AttachToAssistant,
        Self::Refine,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Open => "Open",
            Self::Reveal => "Show in folder",
            Self::CopyPath => "Copy path",
            Self::GenerateClip => "Generate CLIP",
            Self::Describe => "Describe",
            Self::AttachToAssistant => "Attach to assistant",
            Self::Refine => "Refine",
        }
    }
}

/// A part of the settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Section {
    General,
    Library,
    AiModels,
    Automation,
    Assistant,
    MetadataTools,
    Shortcuts,
    Appearance,
    Advanced,
}

/// Severity of a toast.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToastLevel {
    Info,
    Success,
    Warning,
    Error,
}

impl ToastLevel {
    pub fn kind(self) -> egui_toast::ToastKind {
        match self {
            Self::Info => egui_toast::ToastKind::Info,
            Self::Success => egui_toast::ToastKind::Success,
            Self::Warning => egui_toast::ToastKind::Warning,
            Self::Error => egui_toast::ToastKind::Error,
        }
    }
}

/// A tool tab that exists once.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tool {
    ImageEdit,
    Refinements,
    Logs,
}

impl Tool {
    /// Dock tab title of the tool.
    pub fn tab_title(self) -> &'static str {
        match self {
            Self::ImageEdit => "Image Edit",
            Self::Refinements => "AI Refinements",
            Self::Logs => "Logs",
        }
    }
}

/// A debug window of the Devtools menu.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DevtoolsWindow {
    ThemePanel,
    SettingsStore,
    DbMaintenance,
    LegacyImport,
    ThumbStorage,
    ThumbCache,
    Tasks,
    CommandLog,
    Perf,
    RowStore,
    Collections,
    Video,
}

/// Rows that already exist in memory, shown in a new tab.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RowsTab {
    pub title: String,
    /// Not part of the JSON form.
    #[serde(skip)]
    pub rows: Vec<Thumbnail>,
    #[serde(default)]
    pub showing_similarity: bool,
    /// Not part of the JSON form.
    #[serde(skip)]
    pub similar_scores: Option<HashMap<String, f32>>,
    #[serde(default)]
    pub origin_path: Option<String>,
    #[serde(default)]
    pub background: bool,
}

fn is_false(v: &bool) -> bool {
    !*v
}

/// One request to the application. The JSON form is `{"cmd": "<snake_case name>", ...fields}`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum AppCommand {
    /// Show `scope` in a tab.
    OpenScope {
        scope: Scope,
        #[serde(default)]
        target: Target,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        #[serde(default, skip_serializing_if = "is_false")]
        recursive: bool,
    },
    /// Show rows held in memory in a new tab.
    OpenRows(RowsTab),
    /// Open or focus a tool tab.
    OpenTool {
        tool: Tool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        path: Option<String>,
        #[serde(default)]
        target: Target,
    },
    /// Run `action` on the files `on` selects.
    Run { action: ActionId, on: Selector },
    ShowSettings {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        section: Option<Section>,
    },
    AttachToAssistant { paths: Vec<String> },
    Refine { paths: Vec<String> },
    Toast { kind: ToastLevel, text: String },
    /// Report progress of a task, or start one with `begin`.
    TaskUpdate { id: TaskId, event: TaskEvent },
    ClearSelection,
    /// Select and preview the row of `path` in the focused tab.
    FocusPath { path: String },
    OpenDevtools { window: DevtoolsWindow },
    /// Open the task list popover of the status bar.
    ShowTaskCenter,
}

impl AppCommand {
    /// The `cmd` name of the JSON form.
    pub fn name(&self) -> &'static str {
        match self {
            Self::OpenScope { .. } => "open_scope",
            Self::OpenRows(_) => "open_rows",
            Self::OpenTool { .. } => "open_tool",
            Self::Run { .. } => "run",
            Self::ShowSettings { .. } => "show_settings",
            Self::AttachToAssistant { .. } => "attach_to_assistant",
            Self::Refine { .. } => "refine",
            Self::Toast { .. } => "toast",
            Self::TaskUpdate { .. } => "task_update",
            Self::ClearSelection => "clear_selection",
            Self::FocusPath { .. } => "focus_path",
            Self::OpenDevtools { .. } => "open_devtools",
            Self::ShowTaskCenter => "show_task_center",
        }
    }

    /// Compact JSON for the command log; in-memory rows appear as a count.
    pub fn log_json(&self) -> String {
        let Self::OpenRows(tab) = self else {
            return serde_json::to_string(self).unwrap_or_default();
        };
        let mut value = serde_json::to_value(self).unwrap_or(serde_json::Value::Null);
        if let Some(map) = value.as_object_mut() {
            map.insert("row_count".to_string(), tab.rows.len().into());
        }
        value.to_string()
    }

    /// A scope opened in a new focused tab.
    pub fn open(scope: Scope) -> Self {
        Self::OpenScope { scope, target: Target::NewTab, title: None, recursive: false }
    }
}

impl PartialEq for AppCommand {
    fn eq(&self, other: &Self) -> bool {
        self.log_json() == other.log_json()
    }
}

impl From<FilterRequest> for AppCommand {
    fn from(req: FilterRequest) -> Self {
        let target = |background| if background { Target::Background } else { Target::NewTab };
        match req {
            FilterRequest::OpenPath { title, path, recursive, background } => {
                Self::OpenScope { scope: Scope::Folder(path), target: target(background), title: Some(title), recursive }
            }
            FilterRequest::OpenDatabaseAll { title, background } => {
                Self::OpenScope { scope: Scope::Library, target: target(background), title: Some(title), recursive: false }
            }
            FilterRequest::NewTab { title, origin_path, background, .. } if title.starts_with("Image Edit") => {
                Self::OpenTool { tool: Tool::ImageEdit, path: origin_path, target: target(background) }
            }
            FilterRequest::NewTab { title, rows, showing_similarity, similar_scores, origin_path, background } => {
                Self::OpenRows(RowsTab { title, rows, showing_similarity, similar_scores, origin_path, background })
            }
        }
    }
}

/// Reads the text of the command runner: a JSON command, a JSON array of commands or a scope string.
pub fn parse_input(text: &str) -> Result<Vec<AppCommand>, String> {
    let text = text.trim();
    match text.chars().next() {
        None => Err("nothing to run".to_string()),
        Some('{') => serde_json::from_str::<AppCommand>(text).map(|c| vec![c]).map_err(|e| format!("bad command: {e}")),
        Some('[') => serde_json::from_str::<Vec<AppCommand>>(text).map_err(|e| format!("bad command list: {e}")),
        Some(_) => Scope::parse(text).map(|scope| vec![AppCommand::open(scope)]).map_err(|e| e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::TaskKind;
    use crate::command::scope::SmartView;

    fn round_trip(cmd: &AppCommand) -> AppCommand {
        let json = serde_json::to_string(cmd).unwrap();
        serde_json::from_str(&json).unwrap_or_else(|e| panic!("{json}: {e}"))
    }

    fn samples() -> Vec<AppCommand> {
        vec![
            AppCommand::open(Scope::Folder("/p".into())),
            AppCommand::OpenScope { scope: Scope::Library, target: Target::Background, title: Some("All".into()), recursive: true },
            AppCommand::OpenScope { scope: Scope::Semantic("a \"b\"".into()), target: Target::Current, title: None, recursive: false },
            AppCommand::OpenScope { scope: Scope::Smart(SmartView::Duplicates), target: Target::NewTab, title: None, recursive: false },
            AppCommand::OpenRows(RowsTab { title: "Rows".into(), origin_path: Some("/a".into()), background: true, ..Default::default() }),
            AppCommand::OpenTool { tool: Tool::ImageEdit, path: Some("/a.png".into()), target: Target::NewTab },
            AppCommand::Run { action: ActionId::GenerateClip, on: Selector::Selection },
            AppCommand::Run { action: ActionId::CopyPath, on: Selector::Paths(vec!["/a".into(), "/b".into()]) },
            AppCommand::Run { action: ActionId::Reveal, on: Selector::Focus },
            AppCommand::Run { action: ActionId::Describe, on: Selector::View },
            AppCommand::ShowSettings { section: Some(Section::AiModels) },
            AppCommand::ShowSettings { section: None },
            AppCommand::AttachToAssistant { paths: vec!["/a".into()] },
            AppCommand::Refine { paths: vec!["/a".into()] },
            AppCommand::Toast { kind: ToastLevel::Warning, text: "careful".into() },
            AppCommand::TaskUpdate { id: 3, event: TaskEvent::Progress { done: 1, total: 4 } },
            AppCommand::TaskUpdate { id: 0, event: TaskEvent::Begin { kind: TaskKind::Debug, title: "x".into(), total: 10 } },
            AppCommand::TaskUpdate { id: 3, event: TaskEvent::Fail { error: "boom".into() } },
            AppCommand::TaskUpdate { id: 3, event: TaskEvent::Complete },
            AppCommand::ClearSelection,
            AppCommand::FocusPath { path: "/a".into() },
            AppCommand::OpenDevtools { window: DevtoolsWindow::CommandLog },
            AppCommand::ShowTaskCenter,
        ]
    }

    #[test]
    fn every_command_survives_the_json_round_trip() {
        for cmd in samples() {
            assert_eq!(round_trip(&cmd).log_json(), cmd.log_json(), "{cmd:?}");
        }
    }

    #[test]
    fn the_json_form_is_tagged_with_the_snake_case_name() {
        for cmd in samples() {
            let value = serde_json::to_value(&cmd).unwrap();
            assert_eq!(value["cmd"], cmd.name(), "{cmd:?}");
        }
    }

    #[test]
    fn the_documented_examples_parse_to_the_expected_commands() {
        let open: AppCommand = serde_json::from_str(r#"{"cmd":"open_scope","scope":"fs:/home/u/Pictures","target":"new_tab"}"#).unwrap();
        assert!(matches!(open, AppCommand::OpenScope { scope: Scope::Folder(ref p), target: Target::NewTab, .. } if p == "/home/u/Pictures"));
        let run: AppCommand = serde_json::from_str(r#"{"cmd":"run","action":"generate_clip","on":"selection"}"#).unwrap();
        assert!(matches!(run, AppCommand::Run { action: ActionId::GenerateClip, on: Selector::Selection }));
        let paths: AppCommand = serde_json::from_str(r#"{"cmd":"run","action":"copy_path","on":{"paths":["/a"]}}"#).unwrap();
        assert!(matches!(paths, AppCommand::Run { on: Selector::Paths(ref p), .. } if p == &["/a".to_string()]));
        let task: AppCommand = serde_json::from_str(r#"{"cmd":"task_update","id":7,"event":{"progress":{"done":2,"total":9}}}"#).unwrap();
        assert!(matches!(task, AppCommand::TaskUpdate { id: 7, event: TaskEvent::Progress { done: 2, total: 9 } }));
    }

    #[test]
    fn every_json_example_in_the_command_docs_parses() {
        let docs = include_str!("../../docs/commands.md");
        let mut in_block = false;
        let mut parsed = std::collections::BTreeSet::new();
        for line in docs.lines() {
            if line.starts_with("```") {
                in_block = line.trim() == "```json";
                continue;
            }
            if in_block && !line.trim().is_empty() {
                let cmd: AppCommand = serde_json::from_str(line).unwrap_or_else(|e| panic!("docs example {line}: {e}"));
                parsed.insert(cmd.name());
            }
        }
        for name in ["open_scope", "open_rows", "open_tool", "run", "show_settings", "attach_to_assistant", "refine", "toast", "task_update", "clear_selection", "focus_path", "open_devtools", "show_task_center"] {
            assert!(parsed.contains(name), "the docs have no example for {name}");
        }
    }

    #[test]
    fn optional_fields_default_when_missing() {
        let cmd: AppCommand = serde_json::from_str(r#"{"cmd":"open_scope","scope":"lib:all"}"#).unwrap();
        assert!(matches!(cmd, AppCommand::OpenScope { target: Target::NewTab, title: None, recursive: false, .. }));
        let settings: AppCommand = serde_json::from_str(r#"{"cmd":"show_settings"}"#).unwrap();
        assert!(matches!(settings, AppCommand::ShowSettings { section: None }));
    }

    #[test]
    fn unknown_commands_and_fields_are_errors_not_silent_noops() {
        assert!(serde_json::from_str::<AppCommand>(r#"{"cmd":"format_disk"}"#).is_err());
        assert!(serde_json::from_str::<AppCommand>(r#"{"cmd":"run","action":"open"}"#).is_err());
        assert!(serde_json::from_str::<AppCommand>(r#"{"cmd":"open_scope","scope":"nope"}"#).is_err());
    }

    #[test]
    fn the_log_json_starts_with_the_command_name() {
        for cmd in samples() {
            let line = cmd.log_json();
            assert!(line.starts_with(&format!("{{\"cmd\":\"{}\"", cmd.name())) || matches!(cmd, AppCommand::OpenRows(_)), "{line}");
        }
    }

    #[test]
    fn rows_stay_out_of_the_json_but_the_log_counts_them() {
        let cmd = AppCommand::OpenRows(RowsTab { title: "T".into(), rows: vec![Thumbnail::default(), Thumbnail::default()], ..Default::default() });
        let json = serde_json::to_value(&cmd).unwrap();
        assert!(json.get("rows").is_none());
        let logged: serde_json::Value = serde_json::from_str(&cmd.log_json()).unwrap();
        assert_eq!(logged["row_count"], 2);
    }

    #[test]
    fn legacy_requests_map_onto_commands() {
        let path = AppCommand::from(FilterRequest::OpenPath { title: "Scan: /a".into(), path: "/a".into(), recursive: true, background: true });
        assert!(matches!(path, AppCommand::OpenScope { scope: Scope::Folder(ref p), target: Target::Background, recursive: true, title: Some(ref t) } if p == "/a" && t == "Scan: /a"));
        let all = AppCommand::from(FilterRequest::OpenDatabaseAll { title: "Entire Database".into(), background: false });
        assert!(matches!(all, AppCommand::OpenScope { scope: Scope::Library, target: Target::NewTab, .. }));
        let edit = AppCommand::from(FilterRequest::NewTab {
            title: "Image Edit".into(),
            rows: vec![],
            showing_similarity: false,
            similar_scores: None,
            origin_path: Some("/a.png".into()),
            background: false,
        });
        assert!(matches!(edit, AppCommand::OpenTool { tool: Tool::ImageEdit, path: Some(ref p), .. } if p == "/a.png"));
        let rows = AppCommand::from(FilterRequest::NewTab {
            title: "Category: x".into(),
            rows: vec![Thumbnail::default()],
            showing_similarity: true,
            similar_scores: Some(HashMap::from([("/a".to_string(), 0.5)])),
            origin_path: Some("query:cat".into()),
            background: false,
        });
        match rows {
            AppCommand::OpenRows(tab) => {
                assert_eq!((tab.title.as_str(), tab.rows.len(), tab.showing_similarity), ("Category: x", 1, true));
                assert_eq!(tab.similar_scores.unwrap()["/a"], 0.5);
            }
            other => panic!("expected rows, got {other:?}"),
        }
    }

    #[test]
    fn an_image_edit_tab_without_a_file_still_opens_the_tool() {
        let cmd = AppCommand::from(FilterRequest::NewTab {
            title: "Image Edit".into(),
            rows: vec![],
            showing_similarity: false,
            similar_scores: None,
            origin_path: None,
            background: true,
        });
        assert!(matches!(cmd, AppCommand::OpenTool { tool: Tool::ImageEdit, path: None, target: Target::Background }));
    }

    #[test]
    fn the_runner_reads_json_lists_and_scope_strings() {
        assert!(matches!(parse_input(r#" {"cmd":"clear_selection"} "#).unwrap()[..], [AppCommand::ClearSelection]));
        assert_eq!(parse_input(r#"[{"cmd":"clear_selection"},{"cmd":"show_task_center"}]"#).unwrap().len(), 2);
        let scope = parse_input("tag:sunset").unwrap();
        assert!(matches!(&scope[..], [AppCommand::OpenScope { scope: Scope::Tag(t), target: Target::NewTab, .. }] if t == "sunset"));
        assert!(parse_input("   ").is_err());
        assert!(parse_input("{").unwrap_err().starts_with("bad command"));
        assert!(parse_input("gibberish").is_err());
    }
}
