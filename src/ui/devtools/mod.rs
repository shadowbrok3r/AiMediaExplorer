//! Debug windows for subsystems, opened from the View menu.

pub mod db_maintenance;
pub mod settings_store;
pub mod thumb_cache;
pub mod thumb_storage;

use eframe::egui::Context;

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

pub fn toggle_thumb_storage(ctx: &Context) {
    thumb_storage::set_open(ctx, !thumb_storage::is_open(ctx));
}

/// Draws every open devtools window.
pub fn show(ctx: &Context) {
    settings_store::show(ctx);
    db_maintenance::show(ctx);
    thumb_cache::show(ctx);
    thumb_storage::show(ctx);
}
