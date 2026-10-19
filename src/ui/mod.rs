pub mod navbar;
pub mod shell;
pub mod sidebar;
pub mod palette;
pub mod file_table;
pub mod status;
pub mod tabs;
pub mod assistant;
pub mod refine;
pub mod image_edit;
pub mod journal_ui;
pub mod devtools;
pub mod inspector;
pub mod dialogs;
pub mod taskbar;
pub mod titlebar;
pub mod settings;
pub mod wizard;
#[cfg(test)]
pub(crate) mod test_support;

/// Icon-then-text button tinted as a destructive action.
pub fn danger_icon_button(
    ui: &mut eframe::egui::Ui,
    icon: crate::icons::Icon,
    text: impl Into<eframe::egui::WidgetText>,
) -> eframe::egui::Response {
    use eframe::egui::class::HasClasses as _;
    let button = crate::icons::button_widget(ui, icon, text);
    match crate::theme::ButtonKind::Danger.class() {
        Some(class) => ui.add(button.with_class(class)),
        None => ui.add(button),
    }
}
