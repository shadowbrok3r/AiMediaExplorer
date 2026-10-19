//! MCP tools server hosted by the application: Streamable HTTP with a bearer token, tools over the library, the UI and the local models.

pub mod activity;
pub mod agent;
pub mod auth;
pub mod clock;
pub mod command;
pub mod config;
pub mod ctx;
pub mod edits;
pub mod error;
pub mod gpu;
pub mod guide;
pub mod legacy;
pub mod notify;
pub mod policy;
pub mod proposals;
pub mod registry;
pub mod resources;
pub mod server;
pub mod snapshot;
pub mod taxonomy;
pub mod tools;
#[cfg(test)]
pub(crate) mod test_support;
#[cfg(test)]
mod tests;

pub use legacy::{MediaDescribeResponse, MediaSearchHit, media_describe_tool, media_search_tool, ping_tool};
pub use server::{Hub, MCP_STATUS, ServerState, hub};

/// Reads `--mcp-port`, `--mcp-config-dir` and `SMART_MEDIA_MCP_PORT`, then starts the server when it is switched on.
pub fn init(args: &[String]) -> Result<(), String> {
    let overrides = config::parse_cli(args, std::env::var("SMART_MEDIA_MCP_PORT").ok().as_deref())?;
    server::set_overrides(overrides);
    let hub = hub().clone();
    crate::status::GlobalStatusIndicator::set_state(&*MCP_STATUS, crate::status::StatusState::Idle, "Stopped");
    if let Some(warning) = hub.warning() {
        log::warn!("[MCP] {warning}");
    }
    if hub.config().enabled {
        tokio::spawn(async move {
            if let Err(e) = hub.start().await {
                crate::command::send(crate::command::AppCommand::Toast { kind: crate::command::ToastLevel::Error, text: format!("MCP server not started: {e}") });
            }
        });
    }
    Ok(())
}
