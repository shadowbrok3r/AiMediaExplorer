//! Debug windows for subsystems, opened from the View menu.

pub mod actions;
pub mod browser_actions;
pub mod collections;
pub mod command_log;
pub mod db_maintenance;
pub mod filters;
pub mod grid;
pub mod inspector;
pub mod journal;
pub mod legacy_import;
pub mod mcp;
pub mod perf;
pub mod query;
pub mod refine;
pub mod row_store;
pub mod scope;
pub mod scan_perf;
pub mod settings_store;
pub mod sidebar;
pub mod tasks;
pub mod thumb_cache;
pub mod thumb_storage;
pub mod video;

use eframe::egui::Context;

use crate::command::DevtoolsWindow;

pub fn open_settings_store(ctx: &Context) {
    settings_store::set_open(ctx, true);
}

pub fn toggle_settings_store(ctx: &Context) {
    settings_store::set_open(ctx, !settings_store::is_open(ctx));
}

pub fn open_db_maintenance(ctx: &Context) {
    db_maintenance::set_open(ctx, true);
}

pub fn toggle_db_maintenance(ctx: &Context) {
    db_maintenance::set_open(ctx, !db_maintenance::is_open(ctx));
}

pub fn toggle_legacy_import(ctx: &Context) {
    legacy_import::set_open(ctx, !legacy_import::is_open(ctx));
}

pub fn toggle_row_store(ctx: &Context) {
    row_store::toggle(ctx);
}

pub fn toggle_collections(ctx: &Context) {
    collections::set_open(ctx, !collections::is_open(ctx));
}

pub fn toggle_query(ctx: &Context) {
    query::toggle(ctx);
}

pub fn toggle_browser_actions(ctx: &Context) {
    browser_actions::toggle(ctx);
}

pub fn toggle_thumb_storage(ctx: &Context) {
    thumb_storage::set_open(ctx, !thumb_storage::is_open(ctx));
}

pub fn toggle_video(ctx: &Context) {
    video::toggle(ctx);
}

pub fn toggle_scope(ctx: &Context) {
    scope::set_open(ctx, !scope::is_open(ctx));
}

pub fn toggle_filters(ctx: &Context) {
    filters::toggle(ctx);
}

pub fn toggle_inspector(ctx: &Context) {
    inspector::toggle(ctx);
}

pub fn toggle_journal(ctx: &Context) {
    journal::set_open(ctx, !journal::is_open(ctx));
}

pub fn toggle_refine(ctx: &Context) {
    refine::set_open(ctx, !refine::is_open(ctx));
}

pub fn toggle_mcp(ctx: &Context) {
    mcp::toggle(ctx);
}

pub fn toggle_tasks(ctx: &Context) {
    tasks::set_open(ctx, !tasks::is_open(ctx));
}

pub fn toggle_command_log(ctx: &Context) {
    command_log::set_open(ctx, !command_log::is_open(ctx));
}

pub fn toggle_perf(ctx: &Context) {
    perf::set_open(ctx, !perf::is_open(ctx));
}

pub fn toggle_grid(ctx: &Context) {
    grid::toggle(ctx);
}

pub fn toggle_sidebar(ctx: &Context) {
    sidebar::set_open(ctx, !sidebar::is_open(ctx));
}

pub fn toggle_actions(ctx: &Context) {
    actions::set_open(ctx, !actions::is_open(ctx));
}

/// Opens `window`, whichever debug window it names.
pub fn open(ctx: &Context, window: DevtoolsWindow) {
    match window {
        DevtoolsWindow::ThemePanel => crate::theme::debug_panel::set_open(ctx, true),
        DevtoolsWindow::SettingsStore => settings_store::set_open(ctx, true),
        DevtoolsWindow::DbMaintenance => db_maintenance::set_open(ctx, true),
        DevtoolsWindow::LegacyImport => legacy_import::set_open(ctx, true),
        DevtoolsWindow::ThumbStorage => thumb_storage::set_open(ctx, true),
        DevtoolsWindow::ThumbCache => thumb_cache::set_open(ctx, true),
        DevtoolsWindow::Tasks => tasks::set_open(ctx, true),
        DevtoolsWindow::CommandLog => command_log::set_open(ctx, true),
        DevtoolsWindow::Perf => perf::set_open(ctx, true),
        DevtoolsWindow::RowStore => row_store::set_open(ctx, true),
        DevtoolsWindow::Collections => collections::set_open(ctx, true),
        DevtoolsWindow::Query => query::set_open(ctx, true),
        DevtoolsWindow::BrowserActions => browser_actions::set_open(ctx, true),
        DevtoolsWindow::Video => video::set_open(ctx, true),
        DevtoolsWindow::Scope => scope::set_open(ctx, true),
        DevtoolsWindow::Filters => filters::set_open(ctx, true),
        DevtoolsWindow::Grid => grid::set_open(ctx, true),
        DevtoolsWindow::Inspector => inspector::set_open(ctx, true),
        DevtoolsWindow::Journal => journal::set_open(ctx, true),
        DevtoolsWindow::Refine => refine::set_open(ctx, true),
        DevtoolsWindow::Sidebar => sidebar::set_open(ctx, true),
        DevtoolsWindow::Actions => actions::set_open(ctx, true),
        DevtoolsWindow::McpServer => mcp::set_open(ctx, true),
    }
    ctx.request_repaint();
}

/// Draws every open devtools window.
pub fn show(ctx: &Context) {
    settings_store::show(ctx);
    db_maintenance::show(ctx);
    legacy_import::show(ctx);
    thumb_cache::show(ctx);
    row_store::show(ctx);
    thumb_storage::show(ctx);
    tasks::show(ctx);
    command_log::show(ctx);
    perf::show(ctx);
    collections::show(ctx);
    query::show(ctx);
    browser_actions::show(ctx);
    video::show(ctx);
    scope::show(ctx);
    filters::show(ctx);
    grid::show(ctx);
    inspector::show(ctx);
    journal::show(ctx);
    refine::show(ctx);
    sidebar::show(ctx);
    actions::show(ctx);
    mcp::show(ctx);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_open(ctx: &Context, w: DevtoolsWindow) -> bool {
        match w {
            DevtoolsWindow::SettingsStore => settings_store::is_open(ctx),
            DevtoolsWindow::DbMaintenance => db_maintenance::is_open(ctx),
            DevtoolsWindow::LegacyImport => legacy_import::is_open(ctx),
            DevtoolsWindow::ThumbStorage => thumb_storage::is_open(ctx),
            DevtoolsWindow::ThumbCache => thumb_cache::is_open(ctx),
            DevtoolsWindow::Tasks => tasks::is_open(ctx),
            DevtoolsWindow::CommandLog => command_log::is_open(ctx),
            DevtoolsWindow::Perf => perf::is_open(ctx),
            DevtoolsWindow::RowStore => row_store::is_open(ctx),
            DevtoolsWindow::Collections => collections::is_open(ctx),
            DevtoolsWindow::Query => query::is_open(ctx),
            DevtoolsWindow::BrowserActions => browser_actions::is_open(ctx),
            DevtoolsWindow::Video => video::is_open(ctx),
            DevtoolsWindow::Scope => scope::is_open(ctx),
            DevtoolsWindow::Filters => filters::is_open(ctx),
            DevtoolsWindow::Grid => grid::is_open(ctx),
            DevtoolsWindow::Inspector => inspector::is_open(ctx),
            DevtoolsWindow::Journal => journal::is_open(ctx),
            DevtoolsWindow::Refine => refine::is_open(ctx),
            DevtoolsWindow::Sidebar => sidebar::is_open(ctx),
            DevtoolsWindow::Actions => actions::is_open(ctx),
            DevtoolsWindow::McpServer => mcp::is_open(ctx),
            DevtoolsWindow::ThemePanel => crate::theme::debug_panel::is_open(ctx),
        }
    }

    #[test]
    fn every_window_name_opens_its_window() {
        let ctx = crate::ui::test_support::test_ctx();
        let windows = [
            DevtoolsWindow::ThemePanel,
            DevtoolsWindow::SettingsStore,
            DevtoolsWindow::DbMaintenance,
            DevtoolsWindow::LegacyImport,
            DevtoolsWindow::ThumbStorage,
            DevtoolsWindow::ThumbCache,
            DevtoolsWindow::Tasks,
            DevtoolsWindow::CommandLog,
            DevtoolsWindow::Perf,
            DevtoolsWindow::RowStore,
            DevtoolsWindow::Collections,
            DevtoolsWindow::Query,
            DevtoolsWindow::BrowserActions,
            DevtoolsWindow::Video,
            DevtoolsWindow::Scope,
            DevtoolsWindow::Filters,
            DevtoolsWindow::Grid,
            DevtoolsWindow::Inspector,
            DevtoolsWindow::Journal,
            DevtoolsWindow::Refine,
            DevtoolsWindow::Sidebar,
            DevtoolsWindow::Actions,
            DevtoolsWindow::McpServer,
        ];
        for w in windows {
            assert!(!is_open(&ctx, w), "{w:?} starts closed");
            open(&ctx, w);
            assert!(is_open(&ctx, w), "{w:?} opens");
        }
    }

    #[test]
    fn the_toggles_flip_the_new_windows() {
        let ctx = crate::ui::test_support::test_ctx();
        toggle_tasks(&ctx);
        toggle_command_log(&ctx);
        toggle_perf(&ctx);
        assert!(tasks::is_open(&ctx) && command_log::is_open(&ctx) && perf::is_open(&ctx));
        toggle_tasks(&ctx);
        assert!(!tasks::is_open(&ctx));
    }
}
