//! Metadata edit state machine: the draft of the focused file, which fields differ from the stored row, when a save is due and how a save ends.

use std::time::{Duration, Instant};

use crate::Thumbnail;
use crate::database::metadata_edit::{MetaPatch, SaveOutcome, normalize_tags};

/// Idle time after the last edit before a save starts without a blur or Enter.
pub const DEBOUNCE: Duration = Duration::from_millis(1200);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Field {
    Caption,
    Category,
    Description,
    Tags,
}

impl Field {
    pub const ALL: [Field; 4] = [Field::Caption, Field::Category, Field::Description, Field::Tags];

    fn bit(self) -> u8 {
        1 << (self as u8)
    }

    pub fn label(self) -> &'static str {
        match self {
            Field::Caption => "caption",
            Field::Category => "category",
            Field::Description => "description",
            Field::Tags => "tags",
        }
    }
}

/// A set of fields.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fields(u8);

impl Fields {
    pub fn contains(self, f: Field) -> bool {
        self.0 & f.bit() != 0
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    fn set(&mut self, f: Field, on: bool) {
        if on {
            self.0 |= f.bit();
        } else {
            self.0 &= !f.bit();
        }
    }

    pub fn iter(self) -> impl Iterator<Item = Field> {
        Field::ALL.into_iter().filter(move |f| self.contains(*f))
    }

    pub fn labels(self) -> String {
        self.iter().map(Field::label).collect::<Vec<_>>().join(", ")
    }
}

/// The editable fields of one file.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MetaDraft {
    pub caption: String,
    pub category: String,
    pub description: String,
    pub tags: Vec<String>,
}

impl MetaDraft {
    pub fn from_row(row: &Thumbnail) -> Self {
        Self {
            caption: row.caption.clone().unwrap_or_default(),
            category: row.category.clone().unwrap_or_default(),
            description: row.description.clone().unwrap_or_default(),
            tags: row.tags.clone(),
        }
    }

    fn same(&self, other: &Self, f: Field) -> bool {
        match f {
            Field::Caption => self.caption.trim() == other.caption.trim(),
            Field::Category => self.category.trim() == other.category.trim(),
            Field::Description => self.description.trim() == other.description.trim(),
            Field::Tags => normalize_tags(&self.tags) == normalize_tags(&other.tags),
        }
    }

    fn copy_field(&mut self, from: &Self, f: Field) {
        match f {
            Field::Caption => self.caption = from.caption.clone(),
            Field::Category => self.category = from.category.clone(),
            Field::Description => self.description = from.description.clone(),
            Field::Tags => self.tags = from.tags.clone(),
        }
    }

    /// The patch writing `fields` of this draft; blank text clears the field.
    pub fn patch_for(&self, fields: Fields) -> MetaPatch {
        let text = |s: &str| Some(Some(s.trim().to_string()).filter(|s| !s.is_empty()));
        let mut patch = MetaPatch::default();
        for f in fields.iter() {
            match f {
                Field::Caption => patch.caption = text(&self.caption),
                Field::Category => patch.category = text(&self.category),
                Field::Description => patch.description = text(&self.description),
                Field::Tags => patch.tags = Some(normalize_tags(&self.tags)),
            }
        }
        patch
    }
}

/// What the save indicator shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SaveState {
    /// Nothing differs from the stored row.
    Clean,
    Dirty,
    Saving,
    /// The last save succeeded and nothing changed since.
    Saved,
    Error,
}

impl SaveState {
    pub fn label(self) -> &'static str {
        match self {
            SaveState::Clean => "No changes",
            SaveState::Dirty => "Unsaved changes",
            SaveState::Saving => "Saving",
            SaveState::Saved => "Saved",
            SaveState::Error => "Not saved",
        }
    }
}

/// A save to run: the fields of `path` that changed.
#[derive(Clone, Debug, PartialEq)]
pub struct SaveRequest {
    pub path: String,
    pub rev: u64,
    pub fields: Fields,
    pub patch: MetaPatch,
}

#[derive(Clone, Debug)]
struct InFlight {
    rev: u64,
    fields: Fields,
    sent: MetaDraft,
}

/// How [`MetaEditor::finished`] treated a result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Applied {
    /// The result belonged to the current save.
    Current,
    /// The file or the save is no longer current; the result only matters for the row and the toast.
    Stale,
}

/// Draft and save bookkeeping of the focused file.
#[derive(Debug)]
pub struct MetaEditor {
    path: Option<String>,
    pub draft: MetaDraft,
    baseline: MetaDraft,
    dirty: Fields,
    in_flight: Option<InFlight>,
    next_rev: u64,
    last_edit: Option<Instant>,
    flush: bool,
    saved: bool,
    error: Option<String>,
    last_error: Option<String>,
    forced: Option<SaveState>,
    pub debounce: Duration,
}

impl Default for MetaEditor {
    fn default() -> Self {
        Self {
            path: None,
            draft: MetaDraft::default(),
            baseline: MetaDraft::default(),
            dirty: Fields::default(),
            in_flight: None,
            next_rev: 1,
            last_edit: None,
            flush: false,
            saved: false,
            error: None,
            last_error: None,
            forced: None,
            debounce: DEBOUNCE,
        }
    }
}

impl MetaEditor {
    pub fn path(&self) -> Option<&str> {
        self.path.as_deref()
    }

    pub fn dirty_fields(&self) -> Fields {
        self.dirty
    }

    pub fn is_dirty(&self) -> bool {
        !self.dirty.is_empty()
    }

    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// The last save error, kept after the indicator moved on.
    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    pub fn is_saving(&self) -> bool {
        self.in_flight.is_some()
    }

    pub fn state(&self) -> SaveState {
        if let Some(forced) = self.forced {
            return forced;
        }
        if self.in_flight.is_some() {
            SaveState::Saving
        } else if self.error.is_some() {
            SaveState::Error
        } else if !self.dirty.is_empty() {
            SaveState::Dirty
        } else if self.saved {
            SaveState::Saved
        } else {
            SaveState::Clean
        }
    }

    /// Points the editor at `path` with the stored values `row`; returns the save of unsaved edits of the previous file.
    pub fn bind(&mut self, path: &str, row: &MetaDraft) -> Option<SaveRequest> {
        if self.path.as_deref() == Some(path) {
            return None;
        }
        let pending = self.take_pending_all();
        self.path = Some(path.to_string());
        self.draft = row.clone();
        self.baseline = row.clone();
        self.dirty = Fields::default();
        self.in_flight = None;
        self.last_edit = None;
        self.flush = false;
        self.saved = false;
        self.error = None;
        self.forced = None;
        pending
    }

    /// Forgets the file (empty selection); returns the save of unsaved edits.
    pub fn unbind(&mut self) -> Option<SaveRequest> {
        let pending = self.take_pending_all();
        *self = Self { next_rev: self.next_rev, debounce: self.debounce, last_error: self.last_error.take(), ..Self::default() };
        pending
    }

    fn take_pending_all(&mut self) -> Option<SaveRequest> {
        let path = self.path.clone()?;
        let mut fields = Fields::default();
        for f in self.dirty.iter() {
            let sent_as_is = self.in_flight.as_ref().is_some_and(|i| i.fields.contains(f) && i.sent.same(&self.draft, f));
            fields.set(f, !sent_as_is);
        }
        if fields.is_empty() {
            return None;
        }
        let rev = self.next_rev;
        self.next_rev += 1;
        Some(SaveRequest { path, rev, fields, patch: self.draft.patch_for(fields) })
    }

    /// Takes over stored values for the fields that are not being edited, such as a description that arrived from the model.
    pub fn sync_row(&mut self, row: &MetaDraft) {
        if self.path.is_none() {
            return;
        }
        for f in Field::ALL {
            let sending = self.in_flight.as_ref().is_some_and(|i| i.fields.contains(f));
            if !self.dirty.contains(f) && !sending {
                self.baseline.copy_field(row, f);
                self.draft.copy_field(row, f);
            }
        }
    }

    fn refresh_dirty(&mut self, f: Field) {
        let dirty = !self.draft.same(&self.baseline, f);
        self.dirty.set(f, dirty);
    }

    fn touched(&mut self, now: Instant) {
        self.last_edit = Some(now);
        self.saved = false;
        self.forced = None;
        self.error = None;
    }

    pub fn set_text(&mut self, f: Field, value: String, now: Instant) {
        let slot = match f {
            Field::Caption => &mut self.draft.caption,
            Field::Category => &mut self.draft.category,
            Field::Description => &mut self.draft.description,
            Field::Tags => return,
        };
        if *slot == value {
            return;
        }
        *slot = value;
        self.refresh_dirty(f);
        self.touched(now);
    }

    pub fn set_tags(&mut self, tags: Vec<String>, now: Instant) {
        if self.draft.tags == tags {
            return;
        }
        self.draft.tags = tags;
        self.refresh_dirty(Field::Tags);
        self.touched(now);
    }

    /// Asks for the pending edits to be saved at the next [`poll`](Self::poll), as a blur or Enter does.
    pub fn request_flush(&mut self) {
        self.flush = true;
    }

    /// The save that is due now, if any.
    pub fn poll(&mut self, now: Instant) -> Option<SaveRequest> {
        let path = self.path.clone()?;
        if self.dirty.is_empty() || self.in_flight.is_some() {
            return None;
        }
        let waited = self.last_edit.map_or(self.debounce, |t| now.saturating_duration_since(t));
        let due = self.flush || (self.error.is_none() && waited >= self.debounce);
        if !due {
            return None;
        }
        let rev = self.next_rev;
        self.next_rev += 1;
        let fields = self.dirty;
        self.in_flight = Some(InFlight { rev, fields, sent: self.draft.clone() });
        self.flush = false;
        self.forced = None;
        Some(SaveRequest { path, rev, fields, patch: self.draft.patch_for(fields) })
    }

    /// When the next debounced save becomes due, for scheduling a repaint.
    pub fn due_in(&self, now: Instant) -> Option<Duration> {
        if self.dirty.is_empty() || self.in_flight.is_some() || self.error.is_some() {
            return None;
        }
        let since = self.last_edit.map_or(Duration::ZERO, |t| now.saturating_duration_since(t));
        Some(self.debounce.saturating_sub(since))
    }

    /// Records how the save `rev` of `path` ended.
    pub fn finished(&mut self, path: &str, rev: u64, result: Result<SaveOutcome, String>) -> Applied {
        let current = self.path.as_deref() == Some(path) && self.in_flight.as_ref().is_some_and(|i| i.rev == rev);
        if !current {
            return Applied::Stale;
        }
        let Some(sent) = self.in_flight.take() else { return Applied::Stale };
        match result {
            Ok(_) => {
                for f in sent.fields.iter() {
                    self.baseline.copy_field(&sent.sent, f);
                    self.refresh_dirty(f);
                }
                self.error = None;
                self.saved = self.dirty.is_empty();
            }
            Err(message) => {
                self.last_error = Some(message.clone());
                self.error = Some(message);
            }
        }
        Applied::Current
    }

    /// Shows `state` in the indicator until the next real change, for the devtools window.
    pub fn simulate(&mut self, state: Option<SaveState>) {
        self.forced = state;
        if state == Some(SaveState::Error) {
            self.last_error = Some("simulated save error".to_string());
        }
    }

    pub fn is_simulated(&self) -> bool {
        self.forced.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draft(caption: &str, tags: &[&str]) -> MetaDraft {
        MetaDraft { caption: caption.into(), tags: tags.iter().map(|t| t.to_string()).collect(), ..Default::default() }
    }

    fn bound(now: Instant) -> MetaEditor {
        let mut e = MetaEditor::default();
        assert!(e.bind("/a.jpg", &draft("cap", &["x"])).is_none());
        let _ = now;
        e
    }

    fn ok() -> Result<SaveOutcome, String> {
        Ok(SaveOutcome::default())
    }

    #[test]
    fn a_new_binding_is_clean_and_an_edit_makes_the_field_dirty() {
        let t0 = Instant::now();
        let mut e = bound(t0);
        assert_eq!(e.state(), SaveState::Clean);
        e.set_text(Field::Caption, "new".into(), t0);
        assert_eq!(e.state(), SaveState::Dirty);
        assert!(e.dirty_fields().contains(Field::Caption) && !e.dirty_fields().contains(Field::Tags));
        e.set_text(Field::Caption, "cap".into(), t0);
        assert_eq!(e.state(), SaveState::Clean, "typing the stored value back leaves nothing to save");
    }

    #[test]
    fn the_debounce_holds_a_save_back_until_the_edit_is_idle() {
        let t0 = Instant::now();
        let mut e = bound(t0);
        e.set_text(Field::Caption, "n".into(), t0);
        assert!(e.poll(t0 + Duration::from_millis(100)).is_none());
        assert!(e.due_in(t0 + Duration::from_millis(200)).unwrap() > Duration::from_millis(900));
        e.set_text(Field::Caption, "ne".into(), t0 + Duration::from_millis(1000));
        assert!(e.poll(t0 + Duration::from_millis(1300)).is_none(), "each edit restarts the wait");
        let req = e.poll(t0 + Duration::from_millis(2300)).expect("due after the idle time");
        assert_eq!(req.path, "/a.jpg");
        assert_eq!(req.patch.caption, Some(Some("ne".to_string())));
        assert!(req.patch.tags.is_none(), "only the changed fields are sent");
        assert_eq!(e.state(), SaveState::Saving);
    }

    #[test]
    fn a_flush_saves_at_once_and_dirty_to_saving_to_saved() {
        let t0 = Instant::now();
        let mut e = bound(t0);
        e.set_text(Field::Category, "pets".into(), t0);
        e.request_flush();
        let req = e.poll(t0).expect("flush saves without waiting");
        assert_eq!(e.state(), SaveState::Saving);
        assert!(e.poll(t0).is_none(), "one save at a time");
        assert_eq!(e.finished("/a.jpg", req.rev, ok()), Applied::Current);
        assert_eq!(e.state(), SaveState::Saved);
        assert!(!e.is_dirty());
        e.set_text(Field::Category, "dogs".into(), t0);
        assert_eq!(e.state(), SaveState::Dirty, "an edit after a save is unsaved again");
    }

    #[test]
    fn an_error_keeps_the_edits_and_waits_for_a_retry() {
        let t0 = Instant::now();
        let mut e = bound(t0);
        e.set_text(Field::Description, "d".into(), t0);
        e.request_flush();
        let req = e.poll(t0).unwrap();
        e.finished("/a.jpg", req.rev, Err("db offline".into()));
        assert_eq!(e.state(), SaveState::Error);
        assert_eq!(e.error(), Some("db offline"));
        assert!(e.is_dirty(), "failed edits stay unsaved");
        assert!(e.poll(t0 + Duration::from_secs(60)).is_none(), "no automatic retry loop");
        e.request_flush();
        let retry = e.poll(t0 + Duration::from_secs(60)).expect("the retry button saves again");
        assert_ne!(retry.rev, req.rev);
        e.finished("/a.jpg", retry.rev, ok());
        assert_eq!(e.state(), SaveState::Saved);
        assert_eq!(e.error(), None);
        assert_eq!(e.last_error(), Some("db offline"), "the devtools keep the last error");
    }

    #[test]
    fn a_new_edit_after_an_error_is_saved_again_by_the_debounce() {
        let t0 = Instant::now();
        let mut e = bound(t0);
        e.set_text(Field::Caption, "a".into(), t0);
        e.request_flush();
        let req = e.poll(t0).unwrap();
        e.finished("/a.jpg", req.rev, Err("x".into()));
        e.set_text(Field::Caption, "ab".into(), t0 + Duration::from_secs(1));
        assert_eq!(e.state(), SaveState::Dirty);
        assert!(e.poll(t0 + Duration::from_secs(3)).is_some());
    }

    #[test]
    fn edits_made_while_saving_stay_dirty_and_go_out_in_the_next_save() {
        let t0 = Instant::now();
        let mut e = bound(t0);
        e.set_text(Field::Caption, "one".into(), t0);
        e.request_flush();
        let first = e.poll(t0).unwrap();
        e.set_text(Field::Caption, "one two".into(), t0 + Duration::from_millis(50));
        e.finished("/a.jpg", first.rev, ok());
        assert_eq!(e.state(), SaveState::Dirty, "the text typed during the save is not saved yet");
        let second = e.poll(t0 + Duration::from_secs(5)).unwrap();
        assert_eq!(second.patch.caption, Some(Some("one two".to_string())));
        e.finished("/a.jpg", second.rev, ok());
        assert_eq!(e.state(), SaveState::Saved);
    }

    #[test]
    fn changing_the_file_flushes_unsaved_edits_of_the_old_one() {
        let t0 = Instant::now();
        let mut e = bound(t0);
        e.set_tags(vec!["x".into(), "y".into()], t0);
        let pending = e.bind("/b.jpg", &draft("other", &[])).expect("the old file's edits are not dropped");
        assert_eq!(pending.path, "/a.jpg");
        assert_eq!(pending.patch.tags, Some(vec!["x".to_string(), "y".to_string()]));
        assert_eq!(e.path(), Some("/b.jpg"));
        assert_eq!(e.state(), SaveState::Clean);
        assert_eq!(e.draft.caption, "other");
        assert!(e.bind("/b.jpg", &draft("ignored", &[])).is_none(), "rebinding the same file keeps the draft");
        assert_eq!(e.draft.caption, "other");
    }

    #[test]
    fn a_clean_rebind_has_nothing_to_flush_and_stale_results_are_recognised() {
        let t0 = Instant::now();
        let mut e = bound(t0);
        e.set_text(Field::Caption, "z".into(), t0);
        e.request_flush();
        let req = e.poll(t0).unwrap();
        assert!(e.bind("/b.jpg", &MetaDraft::default()).is_none(), "the edit is already on its way");
        assert_eq!(e.finished("/a.jpg", req.rev, ok()), Applied::Stale, "the result of the old file does not touch the new draft");
        assert_eq!(e.state(), SaveState::Clean);
        assert_eq!(e.finished("/b.jpg", 999, ok()), Applied::Stale);
    }

    #[test]
    fn the_row_refreshes_fields_that_are_not_being_edited() {
        let t0 = Instant::now();
        let mut e = bound(t0);
        e.set_text(Field::Caption, "mine".into(), t0);
        let incoming = MetaDraft { caption: "from the model".into(), description: "a long description".into(), category: "cat".into(), tags: vec!["t".into()] };
        e.sync_row(&incoming);
        assert_eq!(e.draft.caption, "mine", "a field being edited is kept");
        assert_eq!(e.draft.description, "a long description");
        assert_eq!(e.draft.category, "cat");
        assert_eq!(e.draft.tags, vec!["t".to_string()]);
        assert_eq!(e.dirty_fields().labels(), "caption");
    }

    #[test]
    fn blank_text_is_saved_as_a_cleared_field_and_tags_are_normalized() {
        let t0 = Instant::now();
        let mut e = bound(t0);
        e.set_text(Field::Caption, "   ".into(), t0);
        e.set_tags(vec!["a".into(), "A".into(), " b ".into()], t0);
        e.request_flush();
        let req = e.poll(t0).unwrap();
        assert_eq!(req.patch.caption, Some(None));
        assert_eq!(req.patch.tags, Some(vec!["a".to_string(), "b".to_string()]));
        assert_eq!(req.fields.labels(), "caption, tags");
    }

    #[test]
    fn unbinding_flushes_and_forgets_the_file() {
        let t0 = Instant::now();
        let mut e = bound(t0);
        e.set_text(Field::Caption, "q".into(), t0);
        let pending = e.unbind().expect("unsaved edits are saved when the selection empties");
        assert_eq!(pending.path, "/a.jpg");
        assert_eq!(e.path(), None);
        assert_eq!(e.state(), SaveState::Clean);
        assert!(e.poll(t0 + Duration::from_secs(9)).is_none());
    }

    #[test]
    fn simulated_states_show_until_a_real_change() {
        let t0 = Instant::now();
        let mut e = bound(t0);
        for s in [SaveState::Dirty, SaveState::Saving, SaveState::Saved, SaveState::Error] {
            e.simulate(Some(s));
            assert_eq!(e.state(), s);
            assert!(e.is_simulated());
        }
        assert_eq!(e.last_error(), Some("simulated save error"));
        e.set_text(Field::Caption, "real".into(), t0);
        assert_eq!(e.state(), SaveState::Dirty);
        assert!(!e.is_simulated());
        e.simulate(None);
        assert_eq!(e.state(), SaveState::Dirty);
    }
}
