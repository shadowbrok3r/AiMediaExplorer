//! Control colour rules, weights and button kinds the widget theme paints stock egui widgets with.

use eframe::egui::{
    Button, Color32, Context, FontFamily, FontId, Id, Stroke, Style, TextStyle, class::ClassName,
};

use super::{fonts, tokens::Tokens, tokens::mix, widget_theme};

/// Gap between a checkbox and its label.
pub(crate) const LABEL_GAP: f32 = 8.0;
/// Inset of the check mark inside its box.
pub(crate) const CHECK_INSET: f32 = 3.0;
/// Largest corner radius of a checkbox's box.
const BOX_RADIUS: u8 = 4;
/// Share of the text colour mixed into a hovered checkbox or field edge.
const HOVER_LIFT: f32 = 0.12;
/// Share of the text colour mixed into a hovered field border.
const HOVER_EDGE: f32 = 0.18;
/// Share of white mixed into a primary fill on hover.
const PRIMARY_HOVER_LIFT: f32 = 0.10;
/// Share of black mixed into a primary fill while pressed.
const PRIMARY_PRESS_SHADE: f32 = 0.18;
/// Share of the error hue in a danger button's fill, at rest, hovered and pressed.
const DANGER_WASH: [f32; 3] = [0.14, 0.22, 0.30];
/// Share of the error hue in a danger button's border.
const DANGER_EDGE: f32 = 0.55;

/// Interaction state a control paints.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum State {
    Rest,
    Hovered,
    Pressed,
}

/// What a button is for, which decides its colours.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ButtonKind {
    /// Field fill with a neutral border.
    #[default]
    Neutral,
    /// Filled with the primary role.
    Primary,
    /// Error-tinted wash for destructive actions.
    Danger,
    /// Transparent until hovered.
    Ghost,
}

impl ButtonKind {
    pub const ALL: [Self; 4] = [Self::Neutral, Self::Primary, Self::Danger, Self::Ghost];

    /// The widget-theme class giving a stock egui button this kind's colours; none for neutral.
    pub fn class(self) -> Option<ClassName> {
        match self {
            Self::Neutral => None,
            Self::Primary => Some(widget_theme::PRIMARY),
            Self::Danger => Some(widget_theme::DANGER),
            Self::Ghost => Some(widget_theme::GHOST),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Neutral => "Neutral",
            Self::Primary => "Primary",
            Self::Danger => "Danger",
            Self::Ghost => "Ghost",
        }
    }
}

/// A stock egui button of `kind`.
pub fn button(kind: ButtonKind, text: impl Into<eframe::egui::WidgetText>) -> Button<'static> {
    use eframe::egui::class::HasClasses as _;
    let button = Button::new(text);
    match kind.class() {
        Some(class) => button.with_class(class),
        None => button,
    }
}

/// A primary-filled stock button.
pub fn primary_button(text: impl Into<eframe::egui::WidgetText>) -> Button<'static> {
    button(ButtonKind::Primary, text)
}

/// An error-tinted stock button.
pub fn danger_button(text: impl Into<eframe::egui::WidgetText>) -> Button<'static> {
    button(ButtonKind::Danger, text)
}

/// A transparent-until-hovered stock button.
pub fn ghost_button(text: impl Into<eframe::egui::WidgetText>) -> Button<'static> {
    button(ButtonKind::Ghost, text)
}

/// Fill, border and ink for a button of `kind` in `state`.
pub(crate) fn button_colors(
    t: &Tokens,
    kind: ButtonKind,
    state: State,
) -> (Color32, Color32, Color32) {
    match kind {
        ButtonKind::Neutral => match state {
            State::Rest => (t.field, t.field_border, t.text),
            State::Hovered => (t.hover, t.field_border, t.text),
            State::Pressed => (t.pressed, t.primary_border, t.text),
        },
        ButtonKind::Primary => match state {
            State::Rest => (t.primary, Color32::TRANSPARENT, t.on_primary),
            State::Hovered => (
                mix(t.primary, Color32::WHITE, PRIMARY_HOVER_LIFT),
                Color32::TRANSPARENT,
                t.on_primary,
            ),
            State::Pressed => (
                mix(t.primary, Color32::BLACK, PRIMARY_PRESS_SHADE),
                Color32::TRANSPARENT,
                t.on_primary,
            ),
        },
        ButtonKind::Danger => {
            let wash = match state {
                State::Rest => DANGER_WASH[0],
                State::Hovered => DANGER_WASH[1],
                State::Pressed => DANGER_WASH[2],
            };
            (
                mix(t.field, t.error, wash),
                mix(t.field, t.error, DANGER_EDGE),
                t.error,
            )
        }
        ButtonKind::Ghost => match state {
            State::Rest => (Color32::TRANSPARENT, Color32::TRANSPARENT, t.text_dim),
            State::Hovered => (t.hover, Color32::TRANSPARENT, t.text),
            State::Pressed => (t.pressed, Color32::TRANSPARENT, t.text),
        },
    }
}

/// Fill, border and ink of a selected control.
pub(crate) fn selected_colors(t: &Tokens) -> (Color32, Color32, Color32) {
    (t.primary_soft, t.primary_border, t.text)
}

/// Border of a field box in the given interaction state.
pub(crate) fn field_border(t: &Tokens, state: State, focused: bool) -> Color32 {
    if focused {
        t.primary_border
    } else if state == State::Rest {
        t.field_border
    } else {
        mix(t.field_border, t.text, HOVER_EDGE)
    }
}

/// Fill and border of a dropdown box in `state`, `open` while its list shows.
pub(crate) fn dropdown_colors(t: &Tokens, state: State, open: bool) -> (Color32, Color32) {
    let fill = match state {
        State::Rest => t.field,
        State::Hovered => t.hover,
        State::Pressed => t.pressed,
    };
    let border = if open {
        t.primary_border
    } else {
        t.field_border
    };
    (fill, border)
}

/// Corner radius of a checkbox's box.
pub(crate) fn box_radius(t: &Tokens) -> u8 {
    t.radius_sm.min(BOX_RADIUS)
}

/// Fill and border of a checkbox's box, checked while `on`, in `state`.
pub(crate) fn checkbox_colors(t: &Tokens, on: bool, state: State) -> (Color32, Color32) {
    match (on, state) {
        (true, State::Rest) => (t.primary, Color32::TRANSPARENT),
        (true, _) => (
            mix(t.primary, Color32::WHITE, HOVER_LIFT),
            Color32::TRANSPARENT,
        ),
        (false, State::Rest) => (t.field, t.field_border),
        (false, State::Hovered) => (t.hover, t.field_border),
        (false, State::Pressed) => (t.pressed, t.primary_border),
    }
}

/// A 1 px stroke in `color`, or none when `color` is invisible.
pub(crate) fn box_stroke(color: Color32) -> Stroke {
    if color.a() == 0 {
        Stroke::NONE
    } else {
        Stroke::new(1.0, color)
    }
}

fn weights_id() -> Id {
    Id::unique("smart_media.kit.weights")
}

/// Whether the medium family is installed, probed once per pass.
fn medium_installed(ctx: &Context) -> bool {
    let pass = ctx.cumulative_pass_nr();
    if let Some((at, medium)) = ctx.data(|d| d.get_temp::<(u64, bool)>(weights_id()))
        && at == pass
    {
        return medium;
    }
    let medium = ctx.fonts(|f| f.definitions().families.contains_key(&fonts::MEDIUM_FAMILY));
    ctx.data_mut(|d| d.insert_temp(weights_id(), (pass, medium)));
    medium
}

/// `text_style` resolved against `style` at the medium weight when its family is proportional.
pub(crate) fn medium_font_in(ctx: &Context, style: &Style, text_style: TextStyle) -> FontId {
    let font = text_style.resolve(style);
    if font.family == FontFamily::Proportional && medium_installed(ctx) {
        FontId::new(font.size, fonts::MEDIUM_FAMILY.clone())
    } else {
        font
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_coloured_kind_has_a_class_and_neutral_has_none() {
        assert_eq!(ButtonKind::Neutral.class(), None);
        for kind in ButtonKind::ALL.into_iter().skip(1) {
            assert!(kind.class().is_some(), "{kind:?}");
        }
    }

    #[test]
    fn the_box_stroke_vanishes_with_an_invisible_colour() {
        assert_eq!(box_stroke(Color32::TRANSPARENT), Stroke::NONE);
        assert_eq!(box_stroke(Color32::RED), Stroke::new(1.0, Color32::RED));
    }

    #[test]
    fn selected_and_primary_states_use_the_primary_roles() {
        let t = Tokens::derive(&Style::default(), Color32::GREEN, Color32::RED);
        let (fill, edge, _) = selected_colors(&t);
        assert_eq!((fill, edge), (t.primary_soft, t.primary_border));
        let (fill, _, ink) = button_colors(&t, ButtonKind::Primary, State::Rest);
        assert_eq!((fill, ink), (t.primary, t.on_primary));
        assert_eq!(field_border(&t, State::Rest, true), t.primary_border);
    }
}
