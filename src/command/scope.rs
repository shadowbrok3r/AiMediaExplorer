//! Where the browser looks: a value with a lossless one-line string form.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Archive schemes the virtual paths of the scanner use.
const ARCHIVE_SCHEMES: [&str; 3] = ["zip://", "tar://", "7z://"];

/// A saved view over the library.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SmartView {
    NoClip,
    Undescribed,
    Untagged,
    FailedThumbs,
    Duplicates,
}

impl SmartView {
    pub const ALL: [SmartView; 5] = [Self::NoClip, Self::Undescribed, Self::Untagged, Self::FailedThumbs, Self::Duplicates];

    /// Name used in the string form.
    pub fn slug(self) -> &'static str {
        match self {
            Self::NoClip => "no-clip",
            Self::Undescribed => "undescribed",
            Self::Untagged => "untagged",
            Self::FailedThumbs => "failed-thumbs",
            Self::Duplicates => "duplicates",
        }
    }

    /// Words shown to the user.
    pub fn label(self) -> &'static str {
        match self {
            Self::NoClip => "Files without CLIP",
            Self::Undescribed => "Files without a description",
            Self::Untagged => "Files without tags",
            Self::FailedThumbs => "Failed thumbnails",
            Self::Duplicates => "Duplicates",
        }
    }

    fn from_slug(slug: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|v| v.slug() == slug)
    }
}

/// What a browser tab shows.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Scope {
    /// A folder on disk.
    Folder(String),
    /// A virtual path inside an archive, `zip://<archive>!/<inner>`.
    Archive(String),
    /// Everything stored in the database.
    Library,
    /// Recently scanned or added files.
    Recent,
    Collection(String),
    Tag(String),
    Category(String),
    /// Natural-language search over the embeddings.
    Semantic(String),
    /// Files similar to the file at this path.
    Similar(String),
    /// The result of one scan.
    Scan(u64),
    Smart(SmartView),
}

/// The text could not be read as a scope.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScopeParseError {
    pub input: String,
    pub reason: &'static str,
}

impl fmt::Display for ScopeParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "not a scope ({}): {:?}", self.reason, self.input)
    }
}

impl std::error::Error for ScopeParseError {}

impl Scope {
    /// The string form, which `parse` reads back unchanged.
    pub fn format(&self) -> String {
        match self {
            Self::Folder(p) => format!("fs:{p}"),
            Self::Archive(p) if ARCHIVE_SCHEMES.iter().any(|s| p.starts_with(s)) => p.clone(),
            Self::Archive(p) => format!("arc:{p}"),
            Self::Library => "lib:all".to_string(),
            Self::Recent => "recent:all".to_string(),
            Self::Collection(n) => format!("col:{n}"),
            Self::Tag(t) => format!("tag:{t}"),
            Self::Category(c) => format!("cat:{c}"),
            Self::Semantic(q) => format!("q:sem:{}", serde_json::to_string(q).unwrap_or_else(|_| "\"\"".to_string())),
            Self::Similar(p) => format!("sim:{p}"),
            Self::Scan(id) => format!("scan:{id}"),
            Self::Smart(v) => format!("smart:{}", v.slug()),
        }
    }

    /// Reads the string form; a semantic query may also be given without quotes.
    pub fn parse(input: &str) -> Result<Self, ScopeParseError> {
        let err = |reason| ScopeParseError { input: input.to_string(), reason };
        if ARCHIVE_SCHEMES.iter().any(|s| input.starts_with(s)) {
            return Ok(Self::Archive(input.to_string()));
        }
        let (head, rest) = input.split_once(':').ok_or_else(|| err("missing prefix"))?;
        let non_empty = |v: &str, reason| if v.is_empty() { Err(err(reason)) } else { Ok(v.to_string()) };
        match head {
            "fs" => non_empty(rest, "empty path").map(Self::Folder),
            "arc" => non_empty(rest, "empty archive path").map(Self::Archive),
            "lib" if rest == "all" => Ok(Self::Library),
            "recent" if rest == "all" => Ok(Self::Recent),
            "col" => non_empty(rest, "empty collection name").map(Self::Collection),
            "tag" => non_empty(rest, "empty tag").map(Self::Tag),
            "cat" => non_empty(rest, "empty category").map(Self::Category),
            "sim" => non_empty(rest, "empty path").map(Self::Similar),
            "scan" => rest.parse::<u64>().map(Self::Scan).map_err(|_| err("scan id is not a number")),
            "smart" => SmartView::from_slug(rest).map(Self::Smart).ok_or_else(|| err("unknown smart view")),
            "q" => {
                let query = rest.strip_prefix("sem:").ok_or_else(|| err("only q:sem: queries exist"))?;
                let text = if query.starts_with('"') { serde_json::from_str::<String>(query).map_err(|_| err("bad quoted query"))? } else { query.to_string() };
                non_empty(&text, "empty query").map(Self::Semantic)
            }
            _ => Err(err("unknown prefix")),
        }
    }

    /// Tab title for a scope opened without an explicit one.
    pub fn default_title(&self) -> String {
        let leaf = |p: &str| p.trim_end_matches(['/', '\\']).rsplit(['/', '\\', '!']).next().filter(|s| !s.is_empty()).unwrap_or(p).to_string();
        match self {
            Self::Folder(p) => leaf(p),
            Self::Archive(p) => leaf(p),
            Self::Library => "Entire Database".to_string(),
            Self::Recent => "Recent".to_string(),
            Self::Collection(n) => format!("Collection: {n}"),
            Self::Tag(t) => format!("Tag: {t}"),
            Self::Category(c) => format!("Category: {c}"),
            Self::Semantic(q) => format!("Search: {q}"),
            Self::Similar(p) => format!("Similar to {}", leaf(p)),
            Self::Scan(id) => format!("Scan #{id}"),
            Self::Smart(v) => v.label().to_string(),
        }
    }
}

impl fmt::Display for Scope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.format())
    }
}

impl FromStr for Scope {
    type Err = ScopeParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl Serialize for Scope {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.format())
    }
}

impl<'de> Deserialize<'de> for Scope {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        Self::parse(&text).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn samples() -> Vec<Scope> {
        let mut all = vec![
            Scope::Folder("/home/u/Pictures".into()),
            Scope::Folder("C:\\Users\\u\\My Pictures".into()),
            Scope::Folder("/with space/and:colon".into()),
            Scope::Archive("zip:///data/a.zip!/".into()),
            Scope::Archive("7z:///data/b.7z!/inner/dir".into()),
            Scope::Archive("not-a-scheme".into()),
            Scope::Library,
            Scope::Recent,
            Scope::Collection("Best Of: 2024".into()),
            Scope::Tag("red car".into()),
            Scope::Category("landscape".into()),
            Scope::Semantic("a \"quoted\" word and a \\ slash".into()),
            Scope::Semantic("plain".into()),
            Scope::Similar("/a/b.jpg".into()),
            Scope::Scan(42),
            Scope::Scan(u64::MAX),
        ];
        all.extend(SmartView::ALL.into_iter().map(Scope::Smart));
        all
    }

    #[test]
    fn every_scope_survives_the_string_round_trip() {
        for scope in samples() {
            let text = scope.format();
            assert_eq!(Scope::parse(&text).as_ref(), Ok(&scope), "{text}");
            assert_eq!(text.parse::<Scope>().as_ref(), Ok(&scope));
            assert_eq!(scope.to_string(), text);
        }
    }

    #[test]
    fn every_scope_survives_the_json_round_trip_as_a_plain_string() {
        for scope in samples() {
            let json = serde_json::to_string(&scope).unwrap();
            assert_eq!(json, serde_json::to_string(&scope.format()).unwrap());
            assert_eq!(serde_json::from_str::<Scope>(&json).unwrap(), scope);
        }
    }

    #[test]
    fn the_documented_forms_parse() {
        assert_eq!(Scope::parse("fs:/path"), Ok(Scope::Folder("/path".into())));
        assert_eq!(Scope::parse("lib:all"), Ok(Scope::Library));
        assert_eq!(Scope::parse("col:Name"), Ok(Scope::Collection("Name".into())));
        assert_eq!(Scope::parse("tag:x"), Ok(Scope::Tag("x".into())));
        assert_eq!(Scope::parse("cat:x"), Ok(Scope::Category("x".into())));
        assert_eq!(Scope::parse("q:sem:\"a cat\""), Ok(Scope::Semantic("a cat".into())));
        assert_eq!(Scope::parse("sim:/p.png"), Ok(Scope::Similar("/p.png".into())));
        assert_eq!(Scope::parse("scan:42"), Ok(Scope::Scan(42)));
        assert_eq!(Scope::parse("zip:///a.zip!/"), Ok(Scope::Archive("zip:///a.zip!/".into())));
        assert_eq!(Scope::parse("smart:no-clip"), Ok(Scope::Smart(SmartView::NoClip)));
    }

    #[test]
    fn an_unquoted_semantic_query_is_accepted() {
        assert_eq!(Scope::parse("q:sem:red car"), Ok(Scope::Semantic("red car".into())));
    }

    #[test]
    fn malformed_text_is_rejected_with_a_reason() {
        for bad in ["", "plain", "fs:", "lib:some", "scan:x", "scan:-1", "smart:nope", "q:other:x", "q:sem:\"open", "q:sem:", "wat:1", "tag:"] {
            assert!(Scope::parse(bad).is_err(), "{bad:?} should not parse");
        }
        let err = Scope::parse("wat:1").unwrap_err();
        assert!(err.to_string().contains("unknown prefix"));
        assert!(serde_json::from_str::<Scope>("\"wat:1\"").is_err());
    }

    #[test]
    fn titles_name_the_leaf_of_a_path() {
        assert_eq!(Scope::Folder("/a/b/".into()).default_title(), "b");
        assert_eq!(Scope::Archive("zip:///a/x.zip!/in/er".into()).default_title(), "er");
        assert_eq!(Scope::Similar("/a/b.jpg".into()).default_title(), "Similar to b.jpg");
        assert_eq!(Scope::Library.default_title(), "Entire Database");
        assert_eq!(Scope::Scan(7).default_title(), "Scan #7");
    }
}
