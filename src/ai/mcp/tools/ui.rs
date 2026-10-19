//! Tools that show something in the application: the context, scopes and non-destructive actions.

use std::sync::Arc;
use std::time::Duration;

use rmcp::schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use super::common::safe_text;
use crate::ai::mcp::ctx::{CallInfo, ToolCtx};
use crate::ai::mcp::error::{ToolError, ToolOutput};
use crate::ai::mcp::registry::{ToolClass, ToolEntry, entry};
use crate::command::{ActionId, AppCommand, Scope, Selector, Target};

const UI_TIMEOUT: Duration = Duration::from_secs(20);
/// Paths one action call may name.
pub const MAX_ACTION_PATHS: usize = 50;
/// A snapshot older than this is reported as stale.
const STALE_AFTER_MS: u64 = 30_000;

pub fn entries() -> Vec<ToolEntry> {
    vec![
        entry("app_get_context", ToolClass::Read, UI_TIMEOUT, "What the user is looking at", "The focused tab, what it shows, how many items are selected, the active filters and the open collection, as the UI last reported them.", app_get_context),
        entry("ui_open_scope", ToolClass::Ui, UI_TIMEOUT, "Show a scope in the application", "Open a tab in the application that shows a scope: a folder, the whole library, a tag, a category, a collection, a semantic search or a smart view. The application runs the request on its next frame.", ui_open_scope),
        entry("ui_run", ToolClass::Ui, UI_TIMEOUT, "Run a safe action on files", "Ask the application to reveal files, copy their paths, generate CLIP embeddings, attach them to the assistant or open the refinement view. Actions that delete, move or overwrite files do not exist here.", ui_run),
    ]
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
#[serde(default)]
pub struct ContextArgs {}

async fn app_get_context(ctx: Arc<ToolCtx>, _call: CallInfo, _args: ContextArgs) -> Result<ToolOutput, ToolError> {
    ctx.policy.refresh(&ctx.db).await;
    let snapshot = crate::ai::mcp::snapshot::current(&ctx.snapshot);
    let now = ctx.clock.unix_ms();
    let stale = snapshot.updated_ms == 0 || now.saturating_sub(snapshot.updated_ms) > STALE_AFTER_MS;
    let text = match (&snapshot.tab, stale) {
        (_, _) if snapshot.updated_ms == 0 => "the UI has not reported its state yet".to_string(),
        (Some(tab), _) => format!("tab '{}' with {} selected", safe_text(tab, 60), snapshot.selection_count),
        (None, _) => match &snapshot.scope {
            Some(scope) => format!("main view '{}' with {} selected", safe_text(scope, 120), snapshot.selection_count),
            None => "no view is open".to_string(),
        },
    };
    let data = json!({
        "snapshot": snapshot,
        "stale": stale,
        "age_ms": (snapshot.updated_ms != 0).then(|| now.saturating_sub(snapshot.updated_ms)),
        "allowed_folders": ctx.policy.roots().iter().map(|r| r.display().to_string()).collect::<Vec<_>>(),
        "gpu_jobs": ctx.gpu.depth(),
    });
    Ok(ToolOutput::new(text, data))
}

#[derive(Clone, Copy, Debug, Default, Deserialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
#[serde(rename_all = "snake_case")]
pub enum TargetArg {
    /// Replace what the focused tab shows (folders and the library only).
    Current,
    /// A new tab that takes focus.
    #[default]
    NewTab,
    /// A new tab behind the focused one.
    Background,
}

impl From<TargetArg> for Target {
    fn from(t: TargetArg) -> Self {
        match t {
            TargetArg::Current => Self::Current,
            TargetArg::NewTab => Self::NewTab,
            TargetArg::Background => Self::Background,
        }
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
pub struct OpenScopeArgs {
    /// Scope string: fs:<folder>, lib:all, recent:all, col:<collection>, tag:<tag>, cat:<category>, q:sem:"<query>", sim:<path>, smart:<name> (no-clip, undescribed, untagged, failed-thumbs, duplicates).
    pub scope: String,
    /// Where to show it; default new_tab.
    pub target: Option<TargetArg>,
    /// Title of the new tab.
    pub title: Option<String>,
}

/// The file system path a scope reads, if it reads one.
pub fn scope_path(scope: &Scope) -> Option<String> {
    match scope {
        Scope::Folder(p) | Scope::Similar(p) => Some(p.clone()),
        Scope::Archive(p) => {
            let rest = ["zip://", "tar://", "7z://"].iter().find_map(|s| p.strip_prefix(s)).unwrap_or(p);
            Some(rest.split('!').next().unwrap_or(rest).to_string())
        }
        _ => None,
    }
}

async fn ui_open_scope(ctx: Arc<ToolCtx>, _call: CallInfo, args: OpenScopeArgs) -> Result<ToolOutput, ToolError> {
    let scope = Scope::parse(args.scope.trim()).map_err(|e| ToolError::Invalid(e.to_string()))?;
    if let Some(path) = scope_path(&scope) {
        ctx.policy.refresh(&ctx.db).await;
        ctx.policy.check_lexical(&path)?;
    }
    let title = args.title.as_deref().map(|t| safe_text(t, 80)).filter(|t| !t.is_empty());
    let command = AppCommand::OpenScope { scope: scope.clone(), target: args.target.unwrap_or_default().into(), title, recursive: false };
    let logged = command.log_json();
    ctx.ui.send(command);
    Ok(ToolOutput::new(format!("asked the application to open {}", safe_text(&scope.format(), 100)), json!({"queued": true, "scope": scope.format(), "command": serde_json::from_str::<serde_json::Value>(&logged).unwrap_or_default()})))
}

#[derive(Clone, Copy, Debug, Deserialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
#[serde(rename_all = "snake_case")]
pub enum ActionArg {
    /// Open the folder of each file.
    Reveal,
    /// Copy the paths to the clipboard.
    CopyPath,
    /// Generate CLIP embeddings.
    GenerateClip,
    /// Describe the files (not available yet).
    Describe,
    /// Attach the files to the assistant chat.
    AttachToAssistant,
    /// Open the refinement view for the files.
    Refine,
}

impl From<ActionArg> for ActionId {
    fn from(a: ActionArg) -> Self {
        match a {
            ActionArg::Reveal => Self::Reveal,
            ActionArg::CopyPath => Self::CopyPath,
            ActionArg::GenerateClip => Self::GenerateClip,
            ActionArg::Describe => Self::Describe,
            ActionArg::AttachToAssistant => Self::AttachToAssistant,
            ActionArg::Refine => Self::Refine,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
#[serde(rename_all = "snake_case")]
pub enum SelectorArg {
    /// The items selected in the focused tab.
    #[default]
    Selection,
    /// The item shown in the preview.
    Focus,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
pub struct RunArgs {
    /// What to do.
    pub action: ActionArg,
    /// Which files when paths is empty: the selection (default) or the previewed file.
    pub selector: Option<SelectorArg>,
    /// Absolute paths inside the allowed folders; when given they replace the selector. At most 50.
    #[serde(default)]
    pub paths: Vec<String>,
}

async fn ui_run(ctx: Arc<ToolCtx>, _call: CallInfo, args: RunArgs) -> Result<ToolOutput, ToolError> {
    let selector = if args.paths.is_empty() {
        match args.selector.unwrap_or_default() {
            SelectorArg::Selection => Selector::Selection,
            SelectorArg::Focus => Selector::Focus,
        }
    } else {
        if args.paths.len() > MAX_ACTION_PATHS {
            return Err(ToolError::invalid(format!("at most {MAX_ACTION_PATHS} paths per call")));
        }
        ctx.policy.refresh(&ctx.db).await;
        let mut checked = Vec::with_capacity(args.paths.len());
        for p in &args.paths {
            checked.push(ctx.policy.check_lexical(p.trim())?.to_string_lossy().to_string());
        }
        Selector::Paths(checked)
    };
    let action: ActionId = args.action.into();
    let command = AppCommand::Run { action, on: selector };
    let logged = command.log_json();
    ctx.ui.send(command);
    Ok(ToolOutput::new(format!("asked the application to run '{}'", action.label()), json!({"queued": true, "command": serde_json::from_str::<serde_json::Value>(&logged).unwrap_or_default()})))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scopes_name_the_file_system_path_they_read() {
        assert_eq!(scope_path(&Scope::parse("fs:/home/u/Pictures").unwrap()).as_deref(), Some("/home/u/Pictures"));
        assert_eq!(scope_path(&Scope::parse("sim:/a/b.jpg").unwrap()).as_deref(), Some("/a/b.jpg"));
        assert_eq!(scope_path(&Scope::parse("zip:///data/a.zip!/inner").unwrap()).as_deref(), Some("/data/a.zip"));
        assert_eq!(scope_path(&Scope::parse("lib:all").unwrap()), None);
        assert_eq!(scope_path(&Scope::parse("tag:sunset").unwrap()), None);
    }

    #[test]
    fn argument_enums_map_to_the_command_vocabulary() {
        assert_eq!(Target::from(TargetArg::Background), Target::Background);
        assert_eq!(ActionId::from(ActionArg::CopyPath), ActionId::CopyPath);
        assert_eq!(ActionId::from(ActionArg::GenerateClip), ActionId::GenerateClip);
    }

    #[test]
    fn the_action_enum_has_no_open_and_nothing_destructive() {
        let schema = rmcp::schemars::schema_for!(RunArgs);
        let text = serde_json::to_string(&schema).unwrap();
        for forbidden in ["\"open\"", "delete", "move", "overwrite", "rename"] {
            assert!(!text.contains(forbidden), "{forbidden} in {text}");
        }
        for allowed in ["reveal", "copy_path", "generate_clip", "attach_to_assistant", "refine"] {
            assert!(text.contains(allowed), "{allowed} missing");
        }
    }
}
