//! Helpers for headless frames: a themed context, pointer clicks and the text a frame drew.

use eframe::egui::epaint::ClippedShape;
use eframe::egui::{Context, Event, Modifiers, PointerButton, Pos2, Shape};

/// A context with the app fonts and startup theme installed.
pub(crate) fn test_ctx() -> Context {
    let ctx = Context::default();
    ctx.set_fonts(crate::theme::fonts::definitions(crate::theme::fonts::TextScale::Normal));
    crate::theme::bootstrap_startup_theme(&ctx);
    ctx
}

/// Pointer move, press and release at `pos`, as one frame's events.
pub(crate) fn press_release(pos: Pos2) -> Vec<Event> {
    let button = |pressed| Event::PointerButton { pos, button: PointerButton::Primary, pressed, modifiers: Modifiers::NONE };
    vec![Event::PointerMoved(pos), button(true), button(false)]
}

/// Every text shape of a frame with its position.
pub(crate) fn texts(shapes: &[ClippedShape]) -> Vec<(String, Pos2)> {
    fn walk(shape: &Shape, out: &mut Vec<(String, Pos2)>) {
        match shape {
            Shape::Text(t) => out.push((t.galley.text().to_string(), t.pos)),
            Shape::Vec(v) => v.iter().for_each(|s| walk(s, out)),
            _ => {}
        }
    }
    let mut out = Vec::new();
    shapes.iter().for_each(|s| walk(&s.shape, &mut out));
    out
}

/// Where `text` was drawn; panics with everything drawn when it was not.
pub(crate) fn text_pos(shapes: &[ClippedShape], text: &str) -> Pos2 {
    let all = texts(shapes);
    match all.iter().find(|(t, _)| t == text) {
        Some((_, at)) => *at,
        None => panic!("no {text:?} drawn: {all:?}"),
    }
}
