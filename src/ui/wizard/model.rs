//! The first-run wizard without any drawing: steps, the checklist rules and what each step collects.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::settings::location::{self, Candidate, Check};
use crate::settings::ui_prefs::{Checklist, StepState};
use crate::ui::settings::models::{FolderCheck, check_model_folder};

/// One step of the wizard.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Step {
    #[default]
    Location,
    Folders,
    Ai,
    Scan,
}

impl Step {
    pub const ALL: [Step; 4] = [Self::Location, Self::Folders, Self::Ai, Self::Scan];

    pub fn index(self) -> usize {
        Self::ALL.iter().position(|s| *s == self).unwrap_or(0)
    }

    pub fn next(self) -> Option<Step> {
        Self::ALL.get(self.index() + 1).copied()
    }

    pub fn prev(self) -> Option<Step> {
        self.index().checked_sub(1).map(|i| Self::ALL[i])
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Location => "Library location",
            Self::Folders => "Add folders",
            Self::Ai => "AI features",
            Self::Scan => "First scan",
        }
    }

    pub fn slug(self) -> &'static str {
        match self {
            Self::Location => "location",
            Self::Folders => "folders",
            Self::Ai => "ai",
            Self::Scan => "scan",
        }
    }

    /// The step a `--wizard` value names.
    pub fn parse(text: &str) -> Option<Step> {
        let text = text.trim().to_lowercase().replace('-', "_");
        Self::ALL.into_iter().find(|s| s.slug() == text)
    }
}

impl Checklist {
    pub fn state(&self, step: Step) -> StepState {
        match step {
            Step::Location => self.location,
            Step::Folders => self.folders,
            Step::Ai => self.ai,
            Step::Scan => self.scan,
        }
    }

    pub fn set(&mut self, step: Step, state: StepState) {
        match step {
            Step::Location => self.location = state,
            Step::Folders => self.folders = state,
            Step::Ai => self.ai = state,
            Step::Scan => self.scan = state,
        }
    }

    /// The first step that is still to do.
    pub fn first_pending(&self) -> Option<Step> {
        Step::ALL.into_iter().find(|s| !self.state(*s).is_finished())
    }

    /// Steps done or skipped, and the number of steps.
    pub fn progress(&self) -> (usize, usize) {
        (Step::ALL.iter().filter(|s| self.state(**s).is_finished()).count(), Step::ALL.len())
    }

    pub fn is_complete(&self) -> bool {
        self.first_pending().is_none()
    }

    /// True when the card should show on an empty library.
    pub fn card_visible(&self, library_is_empty: bool) -> bool {
        library_is_empty && !self.dismissed && !self.is_complete()
    }
}

/// Which library location the first step takes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Choice {
    /// The older library at this index of the detected list.
    Legacy(usize),
    /// The default folder in the data directory.
    Default,
    /// A folder typed or picked.
    Custom,
}

/// The first step: where the library goes.
#[derive(Clone, Debug)]
pub struct LocationForm {
    pub choice: Choice,
    pub custom: String,
    pub detected: Vec<Candidate>,
    pub default_path: PathBuf,
}

impl LocationForm {
    pub fn new(detected: Vec<Candidate>, default_path: PathBuf) -> Self {
        let choice = if detected.is_empty() { Choice::Default } else { Choice::Legacy(0) };
        Self { choice, custom: String::new(), detected, default_path }
    }

    /// The form for the running program: older libraries around it and the default folder.
    pub fn here() -> Self {
        Self::new(location::detect_here(), location::default_location())
    }

    /// The folder the current choice names.
    pub fn path(&self) -> PathBuf {
        match &self.choice {
            Choice::Legacy(i) => self.detected.get(*i).map(|c| c.path.clone()).unwrap_or_else(|| self.default_path.clone()),
            Choice::Default => self.default_path.clone(),
            Choice::Custom => PathBuf::from(self.custom.trim()),
        }
    }

    /// What taking the current choice does to the folder.
    pub fn check(&self) -> Check {
        location::check_location(&self.path())
    }

    /// The folder to use, or why the choice cannot be taken.
    pub fn resolve(&self) -> Result<(PathBuf, Check), String> {
        match self.check() {
            Check::Rejected(why) => Err(why),
            check => Ok((self.path(), check)),
        }
    }
}

/// The second step: the folders the library is built from.
#[derive(Clone, Debug, Default)]
pub struct FoldersForm {
    pub entries: Vec<(String, bool)>,
    pub input: String,
}

/// Pictures and Videos of the user, when the system has them.
pub fn default_folders() -> Vec<PathBuf> {
    let Some(dirs) = directories::UserDirs::new() else { return Vec::new() };
    [dirs.picture_dir(), dirs.video_dir()].into_iter().flatten().filter(|p| p.is_dir()).map(Path::to_path_buf).collect()
}

impl FoldersForm {
    /// The form with `folders` ticked and `already` kept.
    pub fn prefilled(folders: &[PathBuf], already: &[String]) -> Self {
        let mut form = Self::default();
        for path in already {
            form.add(path, true);
        }
        for path in folders {
            form.add(&path.display().to_string(), true);
        }
        form
    }

    /// Adds `path` unless it is empty or listed; false when nothing was added.
    pub fn add(&mut self, path: &str, on: bool) -> bool {
        let path = path.trim();
        if path.is_empty() || self.entries.iter().any(|(p, _)| p == path) {
            return false;
        }
        self.entries.push((path.to_string(), on));
        true
    }

    /// The folders that are ticked.
    pub fn chosen(&self) -> Vec<String> {
        self.entries.iter().filter(|(_, on)| *on).map(|(p, _)| p.clone()).collect()
    }
}

/// The third step: how the AI features run.
#[derive(Clone, Debug, Default)]
pub struct AiForm {
    pub mode: crate::settings::ui_prefs::AiMode,
    pub vision_dir: String,
    pub found: Vec<PathBuf>,
}

impl AiForm {
    /// What the chosen vision folder looks like.
    pub fn vision_check(&self) -> Option<FolderCheck> {
        let text = self.vision_dir.trim();
        (!text.is_empty()).then(|| check_model_folder(Path::new(text)))
    }
}

/// Model folders found one level below `roots`, at most `limit`.
pub fn detect_model_folders(roots: &[PathBuf], limit: usize) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for root in roots {
        let candidates: Vec<PathBuf> = std::iter::once(root.clone()).chain(std::fs::read_dir(root).into_iter().flatten().flatten().map(|e| e.path())).collect();
        for dir in candidates {
            if found.len() >= limit {
                return found;
            }
            if check_model_folder(&dir).is_ok() && !found.contains(&dir) {
                found.push(dir);
            }
        }
    }
    found
}

/// Places a model folder is likely to be in.
pub fn model_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(hf) = std::env::var("HF_HOME") {
        roots.push(PathBuf::from(hf).join("hub"));
    }
    if let Some(dirs) = directories::BaseDirs::new() {
        let home = dirs.home_dir();
        roots.extend([home.join("models"), home.join("Models"), home.join("Documents").join("models"), home.join(".cache").join("huggingface").join("hub")]);
        roots.push(dirs.data_dir().join("models"));
    }
    if let Some(dir) = directories::UserDirs::new().and_then(|d| d.download_dir().map(Path::to_path_buf)) {
        roots.push(dir);
    }
    roots.retain(|r| r.is_dir());
    roots
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::location::Via;

    fn store_in(dir: &Path) {
        std::fs::create_dir_all(dir.join("manifest")).unwrap();
    }

    #[test]
    fn the_steps_run_in_order_and_parse_from_their_names() {
        assert_eq!(Step::Location.next(), Some(Step::Folders));
        assert_eq!(Step::Scan.next(), None);
        assert_eq!(Step::Location.prev(), None);
        assert_eq!(Step::Scan.prev(), Some(Step::Ai));
        for step in Step::ALL {
            assert_eq!(Step::parse(step.slug()), Some(step));
            assert!(!step.label().is_empty());
        }
        assert_eq!(Step::parse(" AI "), Some(Step::Ai));
        assert_eq!(Step::parse("nope"), None);
    }

    #[test]
    fn steps_survive_json_with_their_slugs() {
        for step in Step::ALL {
            assert_eq!(serde_json::to_value(step).unwrap(), serde_json::json!(step.slug()));
            assert_eq!(serde_json::from_value::<Step>(serde_json::json!(step.slug())).unwrap(), step);
        }
    }

    #[test]
    fn the_checklist_tracks_progress_and_the_next_step() {
        let mut c = Checklist::default();
        assert_eq!(c.progress(), (0, 4));
        assert_eq!(c.first_pending(), Some(Step::Location));
        c.set(Step::Location, StepState::Done);
        c.set(Step::Folders, StepState::Skipped);
        assert_eq!(c.progress(), (2, 4));
        assert_eq!(c.first_pending(), Some(Step::Ai));
        assert!(!c.is_complete());
        c.set(Step::Ai, StepState::Done);
        c.set(Step::Scan, StepState::Skipped);
        assert!(c.is_complete());
        assert_eq!(c.first_pending(), None);
    }

    #[test]
    fn the_card_shows_only_on_an_empty_library_with_work_left_and_not_dismissed() {
        let mut c = Checklist::default();
        assert!(c.card_visible(true));
        assert!(!c.card_visible(false), "a library with files needs no card");
        c.dismissed = true;
        assert!(!c.card_visible(true));
        c.dismissed = false;
        for s in Step::ALL {
            c.set(s, StepState::Done);
        }
        assert!(!c.card_visible(true), "a finished checklist leaves the card away");
    }

    #[test]
    fn the_location_defaults_to_a_new_library_and_prefers_an_older_one_when_found() {
        let none = LocationForm::new(Vec::new(), PathBuf::from("/data/db"));
        assert_eq!(none.choice, Choice::Default);
        assert_eq!(none.path(), PathBuf::from("/data/db"));
        let cand = Candidate { path: PathBuf::from("/old/ai_search"), via: Via::DefaultFolder, origin: PathBuf::from("/old/db") };
        let mut some = LocationForm::new(vec![cand], PathBuf::from("/data/db"));
        assert_eq!(some.choice, Choice::Legacy(0));
        assert_eq!(some.path(), PathBuf::from("/old/ai_search"));
        some.choice = Choice::Custom;
        some.custom = "  /mine  ".into();
        assert_eq!(some.path(), PathBuf::from("/mine"));
    }

    #[test]
    fn an_older_library_is_opened_and_a_busy_folder_is_refused() {
        let root = tempfile::tempdir().unwrap();
        let old = root.path().join("db").join("ai_search");
        store_in(&old);
        let found = location::detect_legacy(Some(root.path()), None);
        let form = LocationForm::new(found, root.path().join("new"));
        let (path, check) = form.resolve().unwrap();
        assert_eq!((path, check), (old, Check::Existing));

        let busy = root.path().join("busy");
        std::fs::create_dir_all(&busy).unwrap();
        std::fs::write(busy.join("photo.jpg"), "x").unwrap();
        let mut form = LocationForm::new(Vec::new(), root.path().join("new"));
        form.choice = Choice::Custom;
        form.custom = busy.display().to_string();
        assert!(form.resolve().is_err(), "an existing folder with other files is never taken over");
        assert!(busy.join("photo.jpg").exists());
    }

    #[test]
    fn a_new_library_goes_into_a_new_or_empty_folder_only() {
        let root = tempfile::tempdir().unwrap();
        let mut form = LocationForm::new(Vec::new(), root.path().join("fresh").join("db"));
        assert_eq!(form.resolve().unwrap().1, Check::New);
        form.default_path = root.path().to_path_buf();
        assert_eq!(form.resolve().unwrap().1, Check::New, "an empty folder is fine");
        store_in(root.path());
        assert_eq!(form.resolve().unwrap().1, Check::Existing, "a folder with a library is opened, not replaced");
    }

    #[test]
    fn the_folders_form_lists_each_path_once_and_returns_the_ticked_ones() {
        let mut f = FoldersForm::prefilled(&[PathBuf::from("/pics"), PathBuf::from("/vids")], &["/pics".to_string(), "/extra".to_string()]);
        assert_eq!(f.entries.len(), 3);
        assert!(!f.add(" /pics ", true));
        assert!(!f.add("  ", true));
        assert!(f.add("/more", false));
        f.entries[1].1 = false;
        assert_eq!(f.chosen(), vec!["/pics".to_string(), "/vids".to_string()]);
    }

    #[test]
    fn missing_user_folders_give_an_empty_default_list_instead_of_a_panic() {
        let dirs = default_folders();
        assert!(dirs.iter().all(|d| d.is_dir()));
    }

    #[test]
    fn model_folders_are_found_below_the_roots() {
        let root = tempfile::tempdir().unwrap();
        let good = root.path().join("llama-joycaption");
        std::fs::create_dir_all(&good).unwrap();
        std::fs::write(good.join("config.json"), "{}").unwrap();
        std::fs::write(good.join("model.safetensors"), "x").unwrap();
        let bad = root.path().join("notes");
        std::fs::create_dir_all(&bad).unwrap();
        std::fs::write(bad.join("readme.txt"), "x").unwrap();
        assert_eq!(detect_model_folders(&[root.path().to_path_buf()], 5), vec![good.clone()]);
        assert!(detect_model_folders(&[root.path().to_path_buf()], 0).is_empty());
        assert!(detect_model_folders(&[PathBuf::from("/definitely/not/here")], 5).is_empty());
        let form = AiForm { vision_dir: good.display().to_string(), ..Default::default() };
        assert_eq!(form.vision_check(), Some(FolderCheck::Looks(1)));
        assert_eq!(AiForm::default().vision_check(), None);
    }
}
