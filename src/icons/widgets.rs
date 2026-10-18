//! Icon widgets: labels, buttons and menu entries whose icons follow the active text colors.

use egui::{
    Button, Color32, Image, InnerResponse, Response, RichText, Sense, TextStyle, Ui, Vec2,
    WidgetText,
    containers::menu::{self, MenuButton, SubMenuButton},
};

use super::pack::Icon;

/// Icon edge as a multiple of the Body font size, so icons follow the theme text scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IconSize {
    Sm,
    Md,
    Lg,
    Tool,
}

impl IconSize {
    /// Edge in points for the current Body font size.
    pub fn px(self, ui: &Ui) -> f32 {
        let body = ui
            .style()
            .text_styles
            .get(&TextStyle::Body)
            .map_or(13.0, |font| font.size);
        let factor = match self {
            Self::Sm => 0.92,
            Self::Md => 1.08,
            Self::Lg => 1.38,
            Self::Tool => 1.6,
        };
        (body * factor).round().max(8.0)
    }
}

/// Icon image tinted with the current text color, for atoms that do not follow it.
pub fn tinted_image(ui: &Ui, icon: Icon, size: IconSize) -> Image<'static> {
    icon.image(size.px(ui)).tint(ui.visuals().text_color())
}

/// Icon-then-text atoms for widgets that take `IntoAtoms`.
pub fn atoms(ui: &Ui, icon: Icon, text: impl Into<WidgetText>) -> (Image<'static>, WidgetText) {
    (tinted_image(ui, icon, IconSize::Md), text.into())
}

/// Icon-then-text button builder; the icon takes the button's hover-aware text color.
pub fn button_widget(ui: &Ui, icon: Icon, text: impl Into<WidgetText>) -> Button<'static> {
    let size = IconSize::Md.px(ui);
    Button::new((icon.image(size), text.into()))
        .gap(5.0)
        .image_tint_follows_text_color(true)
}

/// Icon-then-text button.
pub fn button(ui: &mut Ui, icon: Icon, text: impl Into<WidgetText>) -> Response {
    let widget = button_widget(ui, icon, text);
    ui.add(widget)
}

/// Compact icon-then-text button.
pub fn small_button(ui: &mut Ui, icon: Icon, text: impl Into<WidgetText>) -> Response {
    let widget = button_widget(ui, icon, text).small();
    ui.add(widget)
}

/// Selectable icon-then-text button.
pub fn selectable_button(
    ui: &mut Ui,
    icon: Icon,
    selected: bool,
    text: impl Into<WidgetText>,
) -> Response {
    let size = IconSize::Md.px(ui);
    let widget = Button::selectable(selected, (icon.image(size), text.into()))
        .image_tint_follows_text_color(true);
    ui.add(widget)
}

/// Square icon-only button builder.
pub fn icon_button_widget(ui: &Ui, icon: Icon, size: IconSize) -> Button<'static> {
    let edge = size.px(ui);
    let side = ui.spacing().interact_size.y.max(edge + 4.0);
    Button::new(icon.image(edge))
        .image_tint_follows_text_color(true)
        .min_size(Vec2::splat(side))
}

/// Square icon-only button with a hover tooltip.
pub fn icon_button(ui: &mut Ui, icon: Icon, tooltip: &str) -> Response {
    let widget = icon_button_widget(ui, icon, IconSize::Md);
    ui.add(widget).on_hover_text(tooltip)
}

/// Square icon-only button drawn in a fixed color, with a hover tooltip.
pub fn icon_button_tinted(ui: &mut Ui, icon: Icon, tooltip: &str, color: Color32) -> Response {
    let edge = IconSize::Md.px(ui);
    let side = ui.spacing().interact_size.y.max(edge + 4.0);
    let widget = Button::new(icon.image(edge).tint(color)).min_size(Vec2::splat(side));
    ui.add(widget).on_hover_text(tooltip)
}

/// Square icon-only toggle button with a hover tooltip.
pub fn toggle_button(ui: &mut Ui, icon: Icon, selected: bool, tooltip: &str) -> Response {
    let edge = IconSize::Md.px(ui);
    let side = ui.spacing().interact_size.y.max(edge + 4.0);
    let widget = Button::selectable(selected, icon.image(edge))
        .image_tint_follows_text_color(true)
        .min_size(Vec2::splat(side));
    ui.add(widget).on_hover_text(tooltip)
}

/// Non-interactive icon-then-text label; lays out the same in left-to-right and right-to-left parents.
pub fn label(ui: &mut Ui, icon: Icon, text: impl Into<WidgetText>) -> Response {
    let size = IconSize::Md.px(ui);
    let widget = Button::new((icon.image(size), text.into()))
        .frame(false)
        .sense(Sense::empty())
        .image_tint_follows_text_color(true);
    ui.add(widget)
}

/// Non-interactive icon-then-heading label.
pub fn heading(ui: &mut Ui, icon: Icon, text: impl Into<String>) -> Response {
    let size = ui
        .style()
        .text_styles
        .get(&TextStyle::Heading)
        .map_or(18.0, |font| font.size);
    let widget = Button::new((icon.image(size), RichText::new(text.into()).heading()))
        .frame(false)
        .sense(Sense::empty())
        .image_tint_follows_text_color(true);
    ui.add(widget)
}

/// Checkbox with a leading icon.
pub fn checkbox(ui: &mut Ui, checked: &mut bool, icon: Icon, text: impl Into<WidgetText>) -> Response {
    let image = tinted_image(ui, icon, IconSize::Md);
    ui.checkbox(checked, (image, text.into()))
}

/// Menu (or sub-menu inside a menu) button with a leading icon.
pub fn menu_button<R>(
    ui: &mut Ui,
    icon: Icon,
    text: impl Into<WidgetText>,
    add_contents: impl FnOnce(&mut Ui) -> R,
) -> InnerResponse<Option<R>> {
    show_menu(ui, button_widget(ui, icon, text), add_contents)
}

/// Icon-only menu button with a hover tooltip.
pub fn icon_menu_button<R>(
    ui: &mut Ui,
    icon: Icon,
    tooltip: &str,
    add_contents: impl FnOnce(&mut Ui) -> R,
) -> InnerResponse<Option<R>> {
    let button = icon_button_widget(ui, icon, IconSize::Md);
    let mut result = show_menu(ui, button, add_contents);
    result.response = result.response.on_hover_text(tooltip);
    result
}

/// Menu button from a prebuilt [`Button`]; becomes a sub-menu entry inside a menu.
pub fn show_menu<R>(
    ui: &mut Ui,
    button: Button<'static>,
    add_contents: impl FnOnce(&mut Ui) -> R,
) -> InnerResponse<Option<R>> {
    let (response, inner) = if menu::is_in_menu(ui) {
        SubMenuButton::from_button(button.right_text(SubMenuButton::arrow_atom(None)))
            .ui(ui, add_contents)
    } else {
        MenuButton::from_button(button).ui(ui, add_contents)
    };
    InnerResponse::new(inner.map(|i| i.inner), response)
}
