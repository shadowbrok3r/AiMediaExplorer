use egui_dock::{NodeIndex, SurfaceIndex, tab_viewer::OnCloseResponse};
use eframe::egui::*;

pub const TABS: [&str; 4] = [
    "File Explorer",
    "Logs",
    "AI Assistant",
    "AI Refinements",
];

impl egui_dock::TabViewer for crate::app::SmartMediaContext {
    type Tab = String;

    fn title(&mut self, tab: &mut Self::Tab) -> WidgetText {
        tab.as_str().into()
    }

    fn ui(&mut self, ui: &mut Ui, tab: &mut Self::Tab) {
        // Record active tab for context-aware metrics (e.g., thumbnails progress)
        self.active_tab_title = Some(tab.clone());
        match tab.as_str() {
            "File Explorer" => self.file_explorer.ui(ui),
            "AI Assistant" => self.assistant.ui(ui, &mut self.file_explorer),
            "AI Refinements" => self.refinements.ui(ui),
            "Image Edit" => self.image_edit.ui(ui),
            "Logs" => egui_logger::logger_ui()
                .warn_color(Color32::from_rgb(94, 215, 221)) 
                .error_color(Color32::from_rgb(255, 55, 102)) 
                .log_levels([true, true, true, false, false])
                .enable_category("eframe".to_string(), false)
                .enable_category("eframe::native::glow_integration".to_string(), false)
                .enable_category("egui_glow::shader_version".to_string(), false)
                .enable_category("egui_glow::painter".to_string(), false)
                .show(ui),
            _ => {
                // Dynamic tabs: delegate to filtered explorers if present
                if let Some(ex) = self.filtered_tabs.get_mut(tab) {
                    ex.ui(ui);
                } else {
                    ui.label(tab.as_str());
                }
            }
        }
    }

    fn context_menu(
        &mut self,
        _ui: &mut Ui,
        tab: &mut Self::Tab,
        _surface: SurfaceIndex,
        _node: NodeIndex,
    ) {
        match tab.as_str() {

            _ => {

            }
        }
    }

    fn is_closeable(&self, _: &Self::Tab) -> bool { true }

    fn on_close(&mut self, tab: &mut Self::Tab) -> OnCloseResponse {
    self.open_tabs.remove(tab);
    // Also drop any dynamic explorer for this tab
    self.filtered_tabs.remove(tab);
        OnCloseResponse::Close
    }
}