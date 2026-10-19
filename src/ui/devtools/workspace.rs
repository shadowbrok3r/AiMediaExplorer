//! Devtools window for the workspace: the file, its last write, the live and saved state, and Save now, Reload, Forget and the state dump.

use std::time::{Instant, SystemTime};

use eframe::egui::{CollapsingHeader, Context, Grid, Id, RichText, ScrollArea, TextEdit, Ui, Window};

use crate::command::{self, AppCommand};
use crate::ui::workspace::{Mode, Op, Session};

fn open_id() -> Id {
    Id::unique("ame_devtools_workspace_open")
}

fn info_id() -> Id {
    Id::unique("ame_devtools_workspace_info")
}

fn dump_path_id() -> Id {
    Id::unique("ame_devtools_workspace_dump_path")
}

pub fn is_open(ctx: &Context) -> bool {
    ctx.data(|d| d.get_temp::<bool>(open_id())).unwrap_or(false)
}

pub fn set_open(ctx: &Context, open: bool) {
    ctx.data_mut(|d| d.insert_temp(open_id(), open));
}

pub fn toggle(ctx: &Context) {
    set_open(ctx, !is_open(ctx));
}

/// What the window shows, recorded by [`publish`].
#[derive(Clone, Debug)]
pub struct Info {
    pub path: Option<String>,
    pub note: String,
    pub writable: bool,
    pub last_write: Option<SystemTime>,
    pub saves: u32,
    pub pending: bool,
    pub last_error: Option<String>,
    pub mode: &'static str,
    pub restored: Option<String>,
    pub frames: u64,
    pub current: String,
    pub file: Result<String, String>,
    pub file_read: Instant,
    pub in_sync: bool,
    pub layout_file: Option<String>,
    pub last_dump: Option<Result<String, String>>,
    pub dump_pending: Option<String>,
}

fn mode_name(mode: Mode) -> &'static str {
    match mode {
        Mode::Off => "off (nothing is restored at launch)",
        Mode::Scope => "scope (the focused scope is restored)",
        Mode::Full => "full (tabs, layout and window are restored)",
    }
}

/// Records the state of `session` for the window while it is open.
pub fn publish(ctx: &Context, session: &Session) {
    if !is_open(ctx) {
        return;
    }
    let previous = ctx.data(|d| d.get_temp::<Info>(info_id()));
    let reuse = previous.as_ref().filter(|p| p.saves == session.saver.saves && p.file_read.elapsed().as_millis() < 1000);
    let (file, file_read) = match reuse {
        Some(p) => (p.file.clone(), p.file_read),
        None => (session.saver.file_text(), Instant::now()),
    };
    let current = session.current.as_ref();
    let in_sync = current.is_some() && current == session.saver.written();
    let info = Info {
        path: session.saver.path().map(|p| p.display().to_string()),
        note: session.saver.note.text(),
        writable: session.saver.is_writable(),
        last_write: session.saver.last_write,
        saves: session.saver.saves,
        pending: session.saver.is_pending(),
        last_error: session.saver.last_error.clone(),
        mode: mode_name(session.mode),
        restored: session.restored.clone(),
        frames: session.frames,
        current: current.map(|w| w.to_json()).unwrap_or_default(),
        file,
        file_read,
        in_sync,
        layout_file: crate::ui::sidebar::layout::default_path().map(|p| p.display().to_string()),
        last_dump: session.last_dump.clone(),
        dump_pending: session.dump.as_ref().map(|d| d.path.clone()),
    };
    ctx.data_mut(|d| d.insert_temp(info_id(), info));
}

/// How long ago `time` was, in words.
pub fn age(time: SystemTime) -> String {
    match SystemTime::now().duration_since(time) {
        Ok(d) if d.as_secs() < 2 => "just now".to_string(),
        Ok(d) if d.as_secs() < 120 => format!("{} s ago", d.as_secs()),
        Ok(d) if d.as_secs() < 7200 => format!("{} min ago", d.as_secs() / 60),
        Ok(d) => format!("{} h ago", d.as_secs() / 3600),
        Err(_) => "in the future".to_string(),
    }
}

/// Draws the window when open.
pub fn show(ctx: &Context) {
    let mut open = is_open(ctx);
    if !open {
        return;
    }
    Window::new("Devtools: Workspace").open(&mut open).default_width(560.0).default_height(620.0).show(ctx, |ui| body(ui));
    set_open(ctx, open);
}

/// Window contents.
pub fn body(ui: &mut Ui) {
    let Some(info) = ui.ctx().data(|d| d.get_temp::<Info>(info_id())) else {
        ui.weak("The state appears with the next frame.");
        return;
    };
    Grid::new("workspace_dev_info").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
        ui.label("File");
        ui.monospace(info.path.clone().unwrap_or_else(|| "memory only".to_string()));
        ui.end_row();
        ui.label("Load");
        ui.monospace(&info.note);
        ui.end_row();
        ui.label("Last write");
        ui.monospace(info.last_write.map(age).unwrap_or_else(|| "never in this run".to_string()));
        ui.end_row();
        ui.label("Writes this run");
        ui.monospace(info.saves.to_string());
        ui.end_row();
        ui.label("Change waiting");
        ui.monospace(info.pending.to_string());
        ui.end_row();
        ui.label("Live state equals file");
        ui.monospace(info.in_sync.to_string());
        ui.end_row();
        ui.label("Restore at launch");
        ui.monospace(info.mode);
        ui.end_row();
        ui.label("Restored");
        ui.monospace(info.restored.clone().unwrap_or_else(|| "nothing".to_string()));
        ui.end_row();
        ui.label("Frames");
        ui.monospace(info.frames.to_string());
        ui.end_row();
        ui.label("Sidebar and rail");
        ui.monospace(info.layout_file.clone().unwrap_or_else(|| "memory only".to_string()));
        ui.end_row();
    });
    if let Some(error) = &info.last_error {
        ui.colored_label(ui.visuals().error_fg_color, format!("Last write failed: {error}"));
    }
    if !info.writable {
        ui.colored_label(ui.visuals().warn_fg_color, "The file is not written: it is from a newer version, or there is no file.");
    }

    ui.add_space(6.0);
    ui.horizontal(|ui| {
        for op in Op::ALL {
            let hint = serde_json::to_string(&AppCommand::Workspace { op }).unwrap_or_default();
            if ui.button(op.label()).on_hover_text(hint).clicked() {
                command::send(AppCommand::Workspace { op });
            }
        }
    });

    ui.separator();
    ui.label(RichText::new("State dump").strong());
    let mut path = ui.ctx().data(|d| d.get_temp::<String>(dump_path_id())).unwrap_or_else(|| std::env::temp_dir().join("ame-state.json").display().to_string());
    ui.horizontal(|ui| {
        ui.add(TextEdit::singleline(&mut path).desired_width(320.0));
        if ui.button("Dump state").on_hover_text("{\"cmd\":\"dump_state\",\"path\":\"...\"}").clicked() {
            command::send(AppCommand::DumpState { path: path.clone(), when_idle: false });
        }
    });
    ui.ctx().data_mut(|d| d.insert_temp(dump_path_id(), path));
    if let Some(waiting) = &info.dump_pending {
        ui.weak(format!("Waiting for the application to settle, then writing {waiting}"));
    }
    match &info.last_dump {
        Some(Ok(text)) => ui.weak(text),
        Some(Err(e)) => ui.colored_label(ui.visuals().error_fg_color, e),
        None => ui.weak("No dump written in this run."),
    };

    ui.separator();
    CollapsingHeader::new("Live state").default_open(true).show(ui, |ui| {
        let mut view = info.current.as_str();
        ScrollArea::vertical().id_salt("workspace_dev_current").max_height(240.0).show(ui, |ui| {
            ui.add(TextEdit::multiline(&mut view).code_editor().desired_width(f32::INFINITY));
        });
    });
    CollapsingHeader::new("File on disk").show(ui, |ui| match &info.file {
        Ok(text) => {
            let mut view = text.as_str();
            ScrollArea::vertical().id_salt("workspace_dev_file").max_height(240.0).show(ui, |ui| {
                ui.add(TextEdit::multiline(&mut view).code_editor().desired_width(f32::INFINITY));
            });
        }
        Err(why) => {
            ui.weak(why);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::bus::testing::{serial, take_matching};
    use crate::settings::workspace::{BrowserState, SavedNode, SavedTab, Saver, Workspace};
    use eframe::egui::{CentralPanel, Pos2, RawInput, Rect, vec2};

    fn sample() -> Workspace {
        let browser = BrowserState::new(0, &crate::command::Scope::Library);
        Workspace { dock: Some(SavedNode::leaf(vec![SavedTab::Browser { id: 0 }])), browsers: vec![browser], ..Workspace::default() }
    }

    fn session(dir: &std::path::Path) -> Session {
        let (saver, _) = Saver::at(dir.join("workspace.json"));
        let mut s = Session::memory();
        s.saver = saver;
        s.current = Some(sample());
        s.frames = 12;
        s
    }

    fn frame(ctx: &Context, events: Vec<eframe::egui::Event>) -> Vec<eframe::egui::epaint::ClippedShape> {
        let input = RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(700.0, 900.0))), events, ..Default::default() };
        let mut out = ctx.run_ui(input, |ui| {
            CentralPanel::default().show(ui, |ui| body(ui));
        });
        out.textures_delta.clear();
        out.shapes
    }

    #[test]
    fn nothing_is_published_while_the_window_is_closed() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = Context::default();
        publish(&ctx, &session(dir.path()));
        assert!(ctx.data(|d| d.get_temp::<Info>(info_id())).is_none());
        set_open(&ctx, true);
        publish(&ctx, &session(dir.path()));
        assert!(ctx.data(|d| d.get_temp::<Info>(info_id())).is_some());
    }

    #[test]
    fn the_window_shows_the_file_the_write_times_and_the_state() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = crate::ui::test_support::test_ctx();
        set_open(&ctx, true);
        let mut s = session(dir.path());
        s.saver.save_now(&sample()).unwrap();
        publish(&ctx, &s);
        frame(&ctx, Vec::new());
        let shapes = frame(&ctx, Vec::new());
        let text = crate::ui::test_support::texts(&shapes).into_iter().map(|(t, _)| t).collect::<Vec<_>>().join("\n");
        for wanted in ["workspace.json", "Last write", "just now", "Save now", "Reload", "Forget", "Live state equals file", "true", "Dump state"] {
            assert!(text.contains(wanted), "{wanted} missing:\n{text}");
        }
        let info = ctx.data(|d| d.get_temp::<Info>(info_id())).unwrap();
        assert!(info.in_sync && info.file.as_ref().is_ok_and(|t| t.contains("\"scope\": \"lib:all\"")));
        assert!(info.current.contains("\"browsers\""));
    }

    #[test]
    fn a_state_that_differs_from_the_file_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = Context::default();
        set_open(&ctx, true);
        let mut s = session(dir.path());
        s.saver.save_now(&sample()).unwrap();
        let mut changed = sample();
        changed.focused = None;
        changed.browsers[0].selection = vec!["/a.jpg".into()];
        s.current = Some(changed);
        publish(&ctx, &s);
        assert!(!ctx.data(|d| d.get_temp::<Info>(info_id())).unwrap().in_sync);
    }

    #[test]
    fn the_buttons_send_the_workspace_commands() {
        let _serial = serial();
        take_matching(|_| true);
        let dir = tempfile::tempdir().unwrap();
        let ctx = crate::ui::test_support::test_ctx();
        set_open(&ctx, true);
        publish(&ctx, &session(dir.path()));
        frame(&ctx, Vec::new());
        let shapes = frame(&ctx, Vec::new());
        for op in Op::ALL {
            let at = crate::ui::test_support::text_pos(&shapes, op.label());
            let events = crate::ui::test_support::press_release(at + vec2(6.0, 6.0));
            frame(&ctx, events);
        }
        let sent = take_matching(|c| matches!(c, AppCommand::Workspace { .. }));
        let ops: Vec<Op> = sent.iter().map(|c| match c { AppCommand::Workspace { op } => *op, other => panic!("{other:?}") }).collect();
        assert_eq!(ops, vec![Op::Save, Op::Restore, Op::Forget]);
    }

    #[test]
    fn the_dump_button_sends_the_path_of_the_field() {
        let _serial = serial();
        take_matching(|_| true);
        let dir = tempfile::tempdir().unwrap();
        let ctx = crate::ui::test_support::test_ctx();
        set_open(&ctx, true);
        publish(&ctx, &session(dir.path()));
        ctx.data_mut(|d| d.insert_temp(dump_path_id(), "/tmp/chosen-state.json".to_string()));
        frame(&ctx, Vec::new());
        let shapes = frame(&ctx, Vec::new());
        let at = crate::ui::test_support::text_pos(&shapes, "Dump state");
        frame(&ctx, crate::ui::test_support::press_release(at + vec2(6.0, 6.0)));
        let sent = take_matching(|c| matches!(c, AppCommand::DumpState { .. }));
        assert!(matches!(&sent[..], [AppCommand::DumpState { path, when_idle: false }] if path == "/tmp/chosen-state.json"));
    }

    #[test]
    fn ages_are_written_in_words() {
        let now = SystemTime::now();
        assert_eq!(age(now), "just now");
        assert_eq!(age(now - std::time::Duration::from_secs(30)), "30 s ago");
        assert_eq!(age(now - std::time::Duration::from_secs(600)), "10 min ago");
        assert_eq!(age(now - std::time::Duration::from_secs(10_000)), "2 h ago");
    }
}
