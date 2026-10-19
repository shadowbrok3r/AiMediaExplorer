//! Vocabulary of the browser toolbar commands: query modes and dialogs.

use serde::{Deserialize, Serialize};

/// How the text of the query bar is used.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryMode {
    /// Live filter on the file name.
    #[default]
    Name,
    /// Live filter on name, type, path, tags, category and description.
    Text,
    /// Search by meaning (CLIP); Enter opens a search scope.
    Semantic,
    /// Files like a given file; Enter opens a similarity scope.
    Similar,
    /// A sentence the ZeroClaw gateway turns into filters; Enter asks, then applies the previewed filters.
    Ask,
}

impl QueryMode {
    pub const ALL: [QueryMode; 5] = [Self::Name, Self::Text, Self::Semantic, Self::Similar, Self::Ask];

    pub fn label(self) -> &'static str {
        match self {
            Self::Name => "Name",
            Self::Text => "Text",
            Self::Semantic => "Semantic",
            Self::Similar => "Similar",
            Self::Ask => "Ask",
        }
    }

    /// Hint shown in the empty query field.
    pub fn hint(self) -> &'static str {
        match self {
            Self::Name => "Filter by file name",
            Self::Text => "Filter by name, tags, category, description",
            Self::Semantic => "Describe what to find, then Enter",
            Self::Similar => "Path of a file (empty: the focused file), then Enter",
            Self::Ask => "Describe the files in a sentence, then Enter",
        }
    }

    /// True when typing filters the rows as it goes.
    pub fn is_live(self) -> bool {
        matches!(self, Self::Name | Self::Text)
    }
}

/// A dialog of the browser that a command can open.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DialogKind {
    Scan,
    Process,
}

impl DialogKind {
    pub const ALL: [DialogKind; 2] = [Self::Scan, Self::Process];

    pub fn label(self) -> &'static str {
        match self {
            Self::Scan => "Scan folder",
            Self::Process => "Process files",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes_and_dialogs_use_snake_case_json() {
        for mode in QueryMode::ALL {
            let json = serde_json::to_value(mode).unwrap();
            assert_eq!(json, mode.label().to_lowercase());
            assert_eq!(serde_json::from_value::<QueryMode>(json).unwrap(), mode);
        }
        assert_eq!(serde_json::to_value(DialogKind::Scan).unwrap(), "scan");
        assert_eq!(serde_json::from_value::<DialogKind>("process".into()).unwrap(), DialogKind::Process);
        assert!(serde_json::from_value::<QueryMode>("fuzzy".into()).is_err());
    }

    #[test]
    fn only_name_and_text_filter_while_typing() {
        let live: Vec<_> = QueryMode::ALL.into_iter().filter(|m| m.is_live()).collect();
        assert_eq!(live, [QueryMode::Name, QueryMode::Text]);
        assert!(QueryMode::ALL.iter().all(|m| !m.hint().is_empty()));
    }
}
