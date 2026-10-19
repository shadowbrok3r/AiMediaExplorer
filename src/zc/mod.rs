//! ZeroClaw glue: connection settings, health, onboarding and the chat bridge.

use eframe::egui::Ui;

/// Draws the ZeroClaw connection card of Settings > Assistant.
pub fn connection_ui(ui: &mut Ui) {
    ui.weak("The ZeroClaw connection is not set up in this build.");
}

/// Draws the ZeroClaw step of the first-run wizard; true once the step is complete or skipped.
pub fn onboarding_ui(ui: &mut Ui) -> bool {
    ui.weak("The ZeroClaw connection is not set up in this build.");
    true
}
