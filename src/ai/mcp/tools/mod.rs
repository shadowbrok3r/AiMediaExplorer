//! The tool catalog: names, arguments and what each tool does.

pub mod common;
pub mod gpu;
pub mod image_edit;
pub mod library;
pub mod preview;
pub mod propose;
pub mod taxonomy;
pub mod ui;

use std::sync::Arc;
use std::time::Duration;

use rmcp::schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use super::activity::{ActivityLog, ServerStats};
use super::ctx::{CallInfo, ToolCtx};
use super::error::{ToolError, ToolOutput};
use super::registry::{ToolClass, ToolEntry, ToolRegistry, entry};

/// Longest tool name; with a gateway prefix of the form `server__` the result stays under 64 characters.
pub const MAX_NAME_LEN: usize = 40;

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
pub struct PingArgs {
    /// Text to echo back.
    pub message: Option<String>,
}

async fn util_ping(ctx: Arc<ToolCtx>, _call: CallInfo, args: PingArgs) -> Result<ToolOutput, ToolError> {
    let message = args.message.unwrap_or_default();
    let echoed = common::safe_text(&message, 200);
    Ok(ToolOutput::new(format!("pong: {echoed}"), json!({"message": echoed, "server": env!("CARGO_PKG_NAME"), "version": env!("CARGO_PKG_VERSION"), "time_ms": ctx.clock.unix_ms()})))
}

/// Every tool of the server in catalog order.
pub fn all_entries() -> Vec<ToolEntry> {
    let mut entries = vec![entry("util_ping", ToolClass::Read, Duration::from_secs(5), "Ping", "Echo a message and report the server version; use it to check that the connection and the token work.", util_ping)];
    entries.extend(library::entries());
    entries.extend(preview::entries());
    entries.extend(ui::entries());
    entries.extend(propose::entries());
    entries.extend(gpu::entries());
    entries.extend(taxonomy::entries());
    entries.extend(image_edit::entries());
    entries
}

pub fn build_registry(activity: Arc<ActivityLog>, stats: Arc<ServerStats>) -> ToolRegistry {
    ToolRegistry::new(all_entries(), activity, stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> ToolRegistry {
        build_registry(Arc::new(ActivityLog::default()), Arc::new(ServerStats::default()))
    }

    #[test]
    fn names_are_lower_snake_case_and_fit_the_gateway_limit() {
        let re = regex::Regex::new(r"^[a-z0-9_]{1,40}$").unwrap();
        let catalog = registry().catalog();
        assert!(catalog.len() >= 24, "{} tools", catalog.len());
        let mut seen = std::collections::HashSet::new();
        for row in &catalog {
            assert!(re.is_match(&row.name), "bad name {}", row.name);
            assert!(row.name.len() <= MAX_NAME_LEN);
            assert!(format!("aimedia__{}", row.name).len() <= 64);
            assert!(!row.name.contains('.'), "dotted name {}", row.name);
            assert!(seen.insert(row.name.clone()), "duplicate {}", row.name);
        }
    }

    #[test]
    fn the_original_dotted_tools_are_gone_and_the_catalog_has_the_required_tools() {
        let names: Vec<String> = registry().catalog().into_iter().map(|r| r.name).collect();
        for gone in ["util.ping", "media.search", "media.describe"] {
            assert!(!names.contains(&gone.to_string()));
        }
        for required in [
            "util_ping", "library_search", "library_semantic_search", "library_similar_to", "library_get_item", "library_stats", "list_collections", "collection_items", "list_tags", "list_categories",
            "app_get_context", "ui_open_scope", "ui_run", "metadata_propose", "describe_image", "clip_embed", "taxonomy_propose_rename", "taxonomy_list_proposals", "image_edit_start", "image_edit_status", "image_edit_cancel",
        ] {
            assert!(names.contains(&required.to_string()), "{required} missing");
        }
    }

    #[test]
    fn every_schema_is_an_object_with_typed_properties_and_every_tool_is_described() {
        for row in registry().catalog() {
            assert_eq!(row.input_schema["type"], "object", "{}", row.name);
            assert!(row.input_schema.get("properties").is_some_and(|p| p.is_object()), "{} has no properties", row.name);
            assert!(row.description.len() > 30, "{} needs a description", row.name);
            assert!(!row.title.is_empty(), "{} needs a title", row.name);
            assert!(row.timeout_secs >= 5);
        }
    }

    #[test]
    fn no_tool_name_or_description_offers_a_destructive_operation() {
        for row in registry().catalog() {
            for word in ["delete", "remove_file", "overwrite", "rename_file", "move_file", "shell", "execute"] {
                assert!(!row.name.contains(word), "{} contains {word}", row.name);
            }
        }
    }

    #[test]
    fn read_tools_are_annotated_read_only_and_nothing_is_destructive() {
        for e in all_entries() {
            let a = e.tool.annotations.as_ref().expect("annotations");
            assert_eq!(a.destructive_hint, Some(false), "{}", e.name);
            assert_eq!(a.read_only_hint, Some(e.class == ToolClass::Read), "{}", e.name);
        }
    }
}
