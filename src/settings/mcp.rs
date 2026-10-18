//! Opt-in start of the MCP stdio server, which shares the GUI process's stdin and stdout.

use std::sync::atomic::{AtomicBool, Ordering};

static STARTED: AtomicBool = AtomicBool::new(false);

/// True when launched with `--mcp-stdio` or `SMART_MEDIA_MCP_STDIO=1`.
pub fn requested_by_flag() -> bool {
    requested(std::env::args().skip(1), std::env::var("SMART_MEDIA_MCP_STDIO").ok().as_deref())
}

fn requested(args: impl IntoIterator<Item = String>, env: Option<&str>) -> bool {
    env == Some("1") || args.into_iter().any(|a| a == "--mcp-stdio")
}

/// Starts the stdio server unless it already started; returns whether this call started it.
pub fn start_once() -> bool {
    if STARTED.swap(true, Ordering::AcqRel) {
        return false;
    }
    tokio::spawn(async move {
        if let Err(e) = crate::ai::mcp::serve_stdio_background().await {
            log::warn!("[MCP] stdio server not started: {e}");
        }
    });
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_server_stays_off_without_a_flag_or_env() {
        assert!(!requested(["--theme".to_string(), "x.json".to_string()], None));
        assert!(!requested(Vec::<String>::new(), Some("0")));
    }

    #[test]
    fn the_flag_or_env_turns_it_on() {
        assert!(requested(["--mcp-stdio".to_string()], None));
        assert!(requested(Vec::<String>::new(), Some("1")));
    }
}
