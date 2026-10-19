//! Start-up commands from `--run <json|scope>`, `--scope <scope>`, `--query <mode>:<text>`, `--devtools <window>[,<window>...]`, `--wizard <step>` and `--dump-state <path>`.

use super::model::{AppCommand, DevtoolsWindow, parse_input};
use super::scope::Scope;

fn devtools_command(name: &str) -> Result<AppCommand, String> {
    let snake = name.trim().replace('-', "_");
    serde_json::from_value::<DevtoolsWindow>(serde_json::Value::String(snake.clone()))
        .map(|window| AppCommand::OpenDevtools { window })
        .map_err(|_| format!("unknown devtools window '{name}'"))
}

/// The query command `--query <mode>:<text>` stands for.
fn query_command(value: &str) -> Result<AppCommand, String> {
    let (mode, text) = value.split_once(':').unwrap_or((value, ""));
    let mode = serde_json::from_value::<super::browser::QueryMode>(serde_json::Value::String(mode.trim().to_lowercase())).map_err(|_| format!("unknown query mode '{}' (use name, text, semantic or similar)", mode.trim()))?;
    Ok(AppCommand::Query { mode, text: text.to_string(), run: true })
}

/// The wizard command `--wizard <step>` stands for.
fn wizard_command(name: &str) -> Result<AppCommand, String> {
    crate::ui::wizard::Step::parse(name).map(|step| AppCommand::Wizard { step: Some(step) }).ok_or_else(|| format!("--wizard: unknown step '{name}' (location, folders, ai or scan)"))
}

/// The command `--dump-state <path>` stands for: the snapshot is written once the application is idle.
fn dump_command(path: &str) -> Result<AppCommand, String> {
    if path.trim().is_empty() {
        return Err("--dump-state: the path is empty".to_string());
    }
    Ok(AppCommand::DumpState { path: path.to_string(), when_idle: true })
}

/// The commands the arguments ask for, in the order given.
pub fn startup_commands(args: &[String]) -> Result<Vec<AppCommand>, String> {
    let mut commands = Vec::new();
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        let (flag, inline) = match arg.split_once('=') {
            Some((flag, value)) => (flag, Some(value.to_string())),
            None => (arg.as_str(), None),
        };
        if flag != "--run" && flag != "--devtools" && flag != "--scope" && flag != "--query" && flag != "--wizard" && flag != "--dump-state" {
            continue;
        }
        let value = match inline {
            Some(v) => v,
            None => args.next().cloned().ok_or_else(|| format!("{flag} needs a value"))?,
        };
        if flag == "--run" {
            commands.extend(parse_input(&value).map_err(|e| format!("--run: {e}"))?);
        } else if flag == "--scope" {
            commands.push(Scope::parse(&value).map(AppCommand::open).map_err(|e| format!("--scope: {e}"))?);
        } else if flag == "--query" {
            commands.push(query_command(&value).map_err(|e| format!("--query: {e}"))?);
        } else if flag == "--wizard" {
            commands.push(wizard_command(&value)?);
        } else if flag == "--dump-state" {
            commands.push(dump_command(&value)?);
        } else {
            for name in value.split(',').filter(|n| !n.trim().is_empty()) {
                commands.push(devtools_command(name)?);
            }
        }
    }
    Ok(commands)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::model::Target;
    use crate::command::scope::Scope;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|a| a.to_string()).collect()
    }

    #[test]
    fn other_arguments_are_ignored() {
        assert!(startup_commands(&args(&["--db-info", "--theme", "x.json", "--no-frost"])).unwrap().is_empty());
        assert!(startup_commands(&[]).unwrap().is_empty());
    }

    #[test]
    fn run_takes_json_lists_and_scopes_in_order() {
        let cmds = startup_commands(&args(&["--run", "tag:sunset", "--run={\"cmd\":\"show_task_center\"}", "--run", "[{\"cmd\":\"clear_selection\"}]"])).unwrap();
        assert_eq!(cmds.len(), 3);
        assert!(matches!(&cmds[0], AppCommand::OpenScope { scope: Scope::Tag(t), target: Target::Current, .. } if t == "sunset"));
        assert!(matches!(cmds[1], AppCommand::ShowTaskCenter));
        assert!(matches!(cmds[2], AppCommand::ClearSelection));
    }

    #[test]
    fn scope_takes_a_scope_string_each_time_it_is_given() {
        let cmds = startup_commands(&args(&["--scope", "lib:all", "--scope=smart:no-clip", "--scope", "q:sem:a red car"])).unwrap();
        let scopes: Vec<String> = cmds.iter().map(|c| match c { AppCommand::OpenScope { scope, target: Target::Current, .. } => scope.format(), other => panic!("{other:?}") }).collect();
        assert_eq!(scopes, vec!["lib:all", "smart:no-clip", "q:sem:\"a red car\""]);
    }

    #[test]
    fn scope_rejects_json_and_unknown_scopes_naming_the_flag() {
        assert!(startup_commands(&args(&["--scope", "nonsense"])).unwrap_err().starts_with("--scope:"));
        assert!(startup_commands(&args(&["--scope", "{\"cmd\":\"clear_selection\"}"])).unwrap_err().starts_with("--scope:"));
        assert!(startup_commands(&args(&["--scope"])).unwrap_err().contains("--scope needs a value"));
    }

    #[test]
    fn run_takes_filter_commands() {
        let cmds = startup_commands(&args(&["--run", r#"[{"cmd":"set_filter","filter":{"skip_icons":true},"replace":true},{"cmd":"apply_saved_view","name":"Holidays"},{"cmd":"clear_filters"}]"#, "--devtools", "filters"])).unwrap();
        assert!(matches!(&cmds[0], AppCommand::SetFilter { filter, replace: true } if filter["skip_icons"] == true));
        assert!(matches!(&cmds[1], AppCommand::ApplySavedView { name } if name == "Holidays"));
        assert!(matches!(cmds[2], AppCommand::ClearFilters));
        assert!(matches!(cmds[3], AppCommand::OpenDevtools { window: DevtoolsWindow::Filters }));
    }

    #[test]
    fn query_takes_a_mode_and_text_and_runs_it() {
        let cmds = startup_commands(&args(&["--query", "semantic:a red car", "--query=name:beach", "--query", "similar:"])).unwrap();
        assert!(matches!(&cmds[0], AppCommand::Query { mode: crate::command::QueryMode::Semantic, text, run: true } if text == "a red car"));
        assert!(matches!(&cmds[1], AppCommand::Query { mode: crate::command::QueryMode::Name, text, run: true } if text == "beach"));
        assert!(matches!(&cmds[2], AppCommand::Query { mode: crate::command::QueryMode::Similar, text, run: true } if text.is_empty()));
        assert!(startup_commands(&args(&["--query", "fuzzy:x"])).unwrap_err().starts_with("--query: unknown query mode 'fuzzy'"));
        assert!(startup_commands(&args(&["--query"])).unwrap_err().contains("--query needs a value"));
        let colon = startup_commands(&args(&["--query", "text:key: value"])).unwrap();
        assert!(matches!(&colon[0], AppCommand::Query { text, .. } if text == "key: value"), "only the first colon splits");
    }

    #[test]
    fn devtools_accepts_lists_with_dashes_or_underscores() {
        let cmds = startup_commands(&args(&["--devtools", "tasks,command-log", "--devtools=perf,scope"])).unwrap();
        let windows: Vec<_> = cmds.iter().map(|c| match c { AppCommand::OpenDevtools { window } => *window, other => panic!("{other:?}") }).collect();
        assert_eq!(windows, vec![DevtoolsWindow::Tasks, DevtoolsWindow::CommandLog, DevtoolsWindow::Perf, DevtoolsWindow::Scope]);
    }

    #[test]
    fn wizard_takes_a_step_name() {
        let cmds = startup_commands(&args(&["--wizard", "folders", "--wizard=AI", "--devtools", "wizard"])).unwrap();
        assert!(matches!(cmds[0], AppCommand::Wizard { step: Some(crate::ui::wizard::Step::Folders) }));
        assert!(matches!(cmds[1], AppCommand::Wizard { step: Some(crate::ui::wizard::Step::Ai) }));
        assert!(matches!(cmds[2], AppCommand::OpenDevtools { window: DevtoolsWindow::Wizard }));
        assert!(startup_commands(&args(&["--wizard", "tour"])).unwrap_err().starts_with("--wizard:"));
        assert!(startup_commands(&args(&["--wizard"])).unwrap_err().contains("--wizard needs a value"));
    }

    #[test]
    fn dump_state_takes_a_path_and_waits_for_the_application_to_settle() {
        let cmds = startup_commands(&args(&["--dump-state", "/tmp/s.json", "--dump-state=/tmp/t.json"])).unwrap();
        assert!(matches!(&cmds[0], AppCommand::DumpState { path, when_idle: true } if path == "/tmp/s.json"));
        assert!(matches!(&cmds[1], AppCommand::DumpState { path, when_idle: true } if path == "/tmp/t.json"));
        assert!(startup_commands(&args(&["--dump-state"])).unwrap_err().contains("--dump-state needs a value"));
        assert!(startup_commands(&args(&["--dump-state="])).unwrap_err().starts_with("--dump-state:"));
    }

    #[test]
    fn bad_input_names_the_flag() {
        assert!(startup_commands(&args(&["--run"])).unwrap_err().contains("--run needs a value"));
        assert!(startup_commands(&args(&["--run", "nonsense"])).unwrap_err().starts_with("--run:"));
        assert!(startup_commands(&args(&["--devtools", "nope"])).unwrap_err().contains("unknown devtools window 'nope'"));
    }
}
