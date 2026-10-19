//! Command palette: fuzzy search over commands, places, collections, tags, settings and library search.

pub mod entries;
pub mod matcher;
mod view;

#[cfg(test)]
mod tests;

pub use entries::{ActionContext, Sources, registry};
pub use view::{Hit, hits, show};

use crate::command::{AppCommand, Target};
use crate::icons::Icon;
use crate::ui::shell::ShellAction;

/// What running an entry does.
#[derive(Clone, Debug)]
pub enum Effect {
    Command(AppCommand),
    Shell(ShellAction),
}

impl Effect {
    /// The same effect with scopes and tools opened in a new tab.
    pub fn in_new_tab(&self) -> Effect {
        match self {
            Effect::Command(AppCommand::OpenScope { scope, title, recursive, .. }) => {
                Effect::Command(AppCommand::OpenScope { scope: scope.clone(), target: Target::NewTab, title: title.clone(), recursive: *recursive })
            }
            Effect::Command(AppCommand::OpenTool { tool, path, .. }) => Effect::Command(AppCommand::OpenTool { tool: *tool, path: path.clone(), target: Target::NewTab }),
            other => other.clone(),
        }
    }
}

/// The group an entry belongs to; shown beside its title.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Action,
    Place,
    Recent,
    Collection,
    Tag,
    Category,
    Scan,
    Search,
}

impl Kind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Action => "Command",
            Self::Place => "Place",
            Self::Recent => "Recent folder",
            Self::Collection => "Collection",
            Self::Tag => "Tag",
            Self::Category => "Category",
            Self::Scan => "Scan",
            Self::Search => "Search",
        }
    }
}

/// One thing the palette can run.
#[derive(Clone, Debug)]
pub struct Entry {
    /// Stable name such as `shell.toggle_sidebar`.
    pub id: String,
    pub kind: Kind,
    pub title: String,
    /// Grey text after the title, such as a scope string.
    pub detail: String,
    /// Extra words the query can match.
    pub keywords: String,
    pub shortcut: Option<&'static str>,
    pub icon: Icon,
    pub effect: Effect,
    /// Why the entry cannot run right now.
    pub enabled: Result<(), String>,
}

/// State of the open palette.
#[derive(Clone, Debug, Default)]
pub struct PaletteState {
    pub open: bool,
    pub query: String,
    pub selected: usize,
    /// Why the last Enter on a disabled entry did nothing.
    pub notice: Option<String>,
}

impl PaletteState {
    /// Opens the palette with `query` typed in.
    pub fn open_with(&mut self, query: Option<String>) {
        self.open = true;
        self.query = query.unwrap_or_default();
        self.selected = 0;
        self.notice = None;
    }

    pub fn close(&mut self) {
        self.open = false;
        self.notice = None;
    }

    pub fn toggle(&mut self) {
        if self.open {
            self.close();
        } else {
            self.open_with(None);
        }
    }
}
