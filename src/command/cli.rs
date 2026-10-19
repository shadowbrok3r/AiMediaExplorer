//! Start-up commands from `--run <json|scope>` and `--devtools <window>[,<window>...]`.

use super::model::{AppCommand, DevtoolsWindow, parse_input};

fn devtools_command(name: &str) -> Result<AppCommand, String> {
    let snake = name.trim().replace('-', "_");
    serde_json::from_value::<DevtoolsWindow>(serde_json::Value::String(snake.clone()))
        .map(|window| AppCommand::OpenDevtools { window })
        .map_err(|_| format!("unknown devtools window '{name}'"))
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
        if flag != "--run" && flag != "--devtools" {
            continue;
        }
        let value = match inline {
            Some(v) => v,
            None => args.next().cloned().ok_or_else(|| format!("{flag} needs a value"))?,
        };
        if flag == "--run" {
            commands.extend(parse_input(&value).map_err(|e| format!("--run: {e}"))?);
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
        assert!(matches!(&cmds[0], AppCommand::OpenScope { scope: Scope::Tag(t), target: Target::NewTab, .. } if t == "sunset"));
        assert!(matches!(cmds[1], AppCommand::ShowTaskCenter));
        assert!(matches!(cmds[2], AppCommand::ClearSelection));
    }

    #[test]
    fn devtools_accepts_lists_with_dashes_or_underscores() {
        let cmds = startup_commands(&args(&["--devtools", "tasks,command-log", "--devtools=perf"])).unwrap();
        let windows: Vec<_> = cmds.iter().map(|c| match c { AppCommand::OpenDevtools { window } => *window, other => panic!("{other:?}") }).collect();
        assert_eq!(windows, vec![DevtoolsWindow::Tasks, DevtoolsWindow::CommandLog, DevtoolsWindow::Perf]);
    }

    #[test]
    fn bad_input_names_the_flag() {
        assert!(startup_commands(&args(&["--run"])).unwrap_err().contains("--run needs a value"));
        assert!(startup_commands(&args(&["--run", "nonsense"])).unwrap_err().starts_with("--run:"));
        assert!(startup_commands(&args(&["--devtools", "nope"])).unwrap_err().contains("unknown devtools window 'nope'"));
    }
}
