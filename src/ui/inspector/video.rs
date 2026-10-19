//! Video in the preview: the in-process player behind a poster picture, stopped whenever the rail stops showing that video.

use egui::{Context, Ui, vec2};

use super::facts::FileFacts;
use super::preview::{area_height, picture};
use super::state::Inspector;
use crate::Thumbnail;
use crate::ui::file_table::FileExplorer;

/// Draws the player of `row`; the poster is the picture until playback starts.
#[cfg(ame_ffmpeg)]
pub fn show(ui: &mut Ui, insp: &mut Inspector, ex: &FileExplorer, row: &Thumbnail, facts: Option<&FileFacts>) {
    let slot = crate::video::view::slot(ui.ctx(), ("inspector", ex.viewer.scope));
    let mut view = slot.lock().unwrap_or_else(|e| e.into_inner());
    let width = ui.available_width();
    let max = vec2(width, area_height(width));
    view.ui(ui, std::path::Path::new(&row.path), max, |ui| {
        picture(ui, insp, ex, row, facts);
    });
}

/// Shows the poster only, with a hint that playback needs the native video support.
#[cfg(not(ame_ffmpeg))]
pub fn show(ui: &mut Ui, insp: &mut Inspector, ex: &FileExplorer, row: &Thumbnail, facts: Option<&FileFacts>) {
    picture(ui, insp, ex, row, facts);
    ui.weak("Playback is not available in this build. Use Open to play the file.");
}

/// Stops and forgets the player of the explorer with `scope`.
#[cfg(ame_ffmpeg)]
pub fn stop(ctx: &Context, scope: u32) {
    let slot = crate::video::view::slot(ctx, ("inspector", scope));
    slot.lock().unwrap_or_else(|e| e.into_inner()).reset();
}

#[cfg(not(ame_ffmpeg))]
pub fn stop(_ctx: &Context, _scope: u32) {}

/// Whether the player of `scope` currently decodes a file.
#[cfg(ame_ffmpeg)]
pub fn is_playing(ctx: &Context, scope: u32) -> bool {
    let slot = crate::video::view::slot(ctx, ("inspector", scope));
    slot.lock().unwrap_or_else(|e| e.into_inner()).is_active()
}

#[cfg(not(ame_ffmpeg))]
pub fn is_playing(_ctx: &Context, _scope: u32) -> bool {
    false
}
