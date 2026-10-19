//! Shims with the API of the request queues that existed before the command bus; each call queues a command.

use std::convert::Infallible;
use std::sync::atomic::Ordering;

use super::model::{AppCommand, Section};
use super::send;
use crate::ui::file_table::FilterRequest;

/// Stands in for the former `Mutex<Vec<FilterRequest>>` of tab requests.
pub struct TabRequests;

/// Guard of [`TabRequests::lock`]; pushing queues the matching command.
pub struct TabRequestsGuard;

impl TabRequests {
    /// Mirrors `Mutex::lock`; never fails.
    pub fn lock(&self) -> Result<TabRequestsGuard, Infallible> {
        Ok(TabRequestsGuard)
    }
}

impl TabRequestsGuard {
    pub fn push(&mut self, request: FilterRequest) {
        send(AppCommand::from(request));
    }
}

/// Stands in for the former `AtomicBool` that asked for the settings window.
pub struct SettingsRequest;

impl SettingsRequest {
    /// `store(true, _)` queues a show-settings command; `false` is ignored.
    pub fn store(&self, requested: bool, _order: Ordering) {
        if requested {
            send(AppCommand::ShowSettings { section: None::<Section> });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::model::{Target, Tool};
    use crate::command::scope::Scope;

    use crate::command::bus::testing::{serial, take_matching as queued_matching};

    #[test]
    fn pushing_a_tab_request_queues_the_equivalent_command() {
        let _serial = serial();
        let request = FilterRequest::OpenPath { title: "legacy-shim-title".into(), path: "/legacy/shim".into(), recursive: true, background: true };
        crate::app::OPEN_TAB_REQUESTS.lock().unwrap().push(request);
        let found = queued_matching(|c| matches!(c, AppCommand::OpenScope { title: Some(t), .. } if t == "legacy-shim-title"));
        assert_eq!(found.len(), 1);
        assert!(matches!(&found[0], AppCommand::OpenScope { scope: Scope::Folder(p), target: Target::Background, recursive: true, .. } if p == "/legacy/shim"));
    }

    #[test]
    fn an_image_edit_request_becomes_the_tool_command() {
        let _serial = serial();
        let request = FilterRequest::NewTab {
            title: "Image Edit".into(),
            rows: vec![],
            showing_similarity: false,
            similar_scores: None,
            origin_path: Some("/legacy/shim.png".into()),
            background: false,
        };
        crate::app::OPEN_TAB_REQUESTS.lock().unwrap().push(request);
        let found = queued_matching(|c| matches!(c, AppCommand::OpenTool { path: Some(p), .. } if p == "/legacy/shim.png"));
        assert!(matches!(&found[..], [AppCommand::OpenTool { tool: Tool::ImageEdit, .. }]));
    }

    #[test]
    fn the_settings_flag_queues_a_command_only_when_set() {
        let _serial = serial();
        queued_matching(|c| matches!(c, AppCommand::ShowSettings { .. }));
        crate::app::OPEN_SETTINGS_REQUEST.store(false, Ordering::Relaxed);
        assert!(queued_matching(|c| matches!(c, AppCommand::ShowSettings { .. })).is_empty());
        crate::app::OPEN_SETTINGS_REQUEST.store(true, Ordering::Relaxed);
        let found = queued_matching(|c| matches!(c, AppCommand::ShowSettings { section: None }));
        assert_eq!(found.len(), 1);
    }
}
