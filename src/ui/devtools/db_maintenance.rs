//! Devtools window hosting the database maintenance panel: snapshots, .surql export/verify/restore and engine info.

use eframe::egui::{Context, Id, ScrollArea, Window};

fn open_id() -> Id {
    Id::unique("ame_devtools_db_maintenance_open")
}

pub fn is_open(ctx: &Context) -> bool {
    ctx.data(|d| d.get_temp::<bool>(open_id())).unwrap_or(false)
}

pub fn set_open(ctx: &Context, open: bool) {
    ctx.data_mut(|d| d.insert_temp(open_id(), open));
}

/// Draws the window when open.
pub fn show(ctx: &Context) {
    let mut open = is_open(ctx);
    if !open {
        return;
    }
    Window::new("Database maintenance")
        .open(&mut open)
        .default_width(760.0)
        .default_height(560.0)
        .show(ctx, |ui| {
            ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                crate::database::maintenance::show(ui);
            });
        });
    set_open(ctx, open);
}
