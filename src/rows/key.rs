//! Row identity: the normalized path a row is stored, looked up and selected under.

use std::borrow::Borrow;
use std::fmt;
use std::sync::Arc;

/// How raw paths fold into keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KeyPolicy {
    /// Keys equal the raw path.
    Exact,
    /// Separators fold to backslash and letters to lowercase.
    Windows,
}

impl KeyPolicy {
    /// The policy of the running platform.
    pub const NATIVE: KeyPolicy = if cfg!(windows) { KeyPolicy::Windows } else { KeyPolicy::Exact };
}

impl Default for KeyPolicy {
    fn default() -> Self {
        Self::NATIVE
    }
}

/// Virtual path schemes whose text is never folded as a file system path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scheme {
    Category,
    Tag,
    Zip,
    Tar,
    SevenZ,
}

impl Scheme {
    pub const ALL: [Scheme; 5] = [Scheme::Category, Scheme::Tag, Scheme::Zip, Scheme::Tar, Scheme::SevenZ];

    pub fn prefix(self) -> &'static str {
        match self {
            Scheme::Category => "cat://",
            Scheme::Tag => "tag://",
            Scheme::Zip => "zip://",
            Scheme::Tar => "tar://",
            Scheme::SevenZ => "7z://",
        }
    }

    /// Whether the scheme names a location inside an archive file.
    pub fn is_archive(self) -> bool {
        matches!(self, Scheme::Zip | Scheme::Tar | Scheme::SevenZ)
    }
}

/// Scheme of a virtual path and the text after `scheme://`.
pub fn split_virtual(path: &str) -> Option<(Scheme, &str)> {
    Scheme::ALL.into_iter().find_map(|s| path.strip_prefix(s.prefix()).map(|rest| (s, rest)))
}

fn fold_windows(text: &str, out: &mut String) {
    for c in text.chars() {
        if c == '/' {
            out.push('\\');
        } else {
            out.extend(c.to_lowercase());
        }
    }
}

/// The key text for `path` under `policy`; virtual paths keep their scheme and in-archive part verbatim.
pub fn normalize(path: &str, policy: KeyPolicy) -> String {
    if policy == KeyPolicy::Exact {
        return path.to_string();
    }
    let mut out = String::with_capacity(path.len());
    match split_virtual(path) {
        Some((Scheme::Category | Scheme::Tag, _)) => out.push_str(path),
        Some((scheme, rest)) => {
            out.push_str(scheme.prefix());
            match rest.split_once("!/") {
                Some((archive, internal)) => {
                    fold_windows(archive, &mut out);
                    out.push_str("!/");
                    out.push_str(internal);
                }
                None => fold_windows(rest, &mut out),
            }
        }
        None => fold_windows(path, &mut out),
    }
    out
}

/// Whether two raw paths name the same row under `policy`.
pub fn paths_equivalent(a: &str, b: &str, policy: KeyPolicy) -> bool {
    match policy {
        KeyPolicy::Exact => a == b,
        KeyPolicy::Windows => normalize(a, policy) == normalize(b, policy),
    }
}

/// Normalized path used as a row's identity; clones share the text.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PathKey(Arc<str>);

impl PathKey {
    /// Key for `path` under the platform policy.
    pub fn new(path: &str) -> Self {
        Self::with_policy(path, KeyPolicy::NATIVE)
    }

    pub fn with_policy(path: &str, policy: KeyPolicy) -> Self {
        match policy {
            KeyPolicy::Exact => PathKey(Arc::from(path)),
            KeyPolicy::Windows => PathKey(Arc::from(normalize(path, policy))),
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn scheme(&self) -> Option<Scheme> {
        split_virtual(&self.0).map(|(s, _)| s)
    }

    pub fn is_virtual(&self) -> bool {
        self.scheme().is_some()
    }
}

/// Lookups by normalized text; hashing and equality match `str`.
impl Borrow<str> for PathKey {
    fn borrow(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for PathKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PathKey({:?})", &*self.0)
    }
}

impl fmt::Display for PathKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: KeyPolicy = KeyPolicy::Windows;
    const X: KeyPolicy = KeyPolicy::Exact;

    #[test]
    fn exact_policy_keeps_every_byte() {
        for p in ["/home/a/B.jpg", "C:\\Users\\A\\b.JPG", "zip:///x/A.zip!/Inner/B.png", "cat://Landscape", "", "/with space/ü.jpg"] {
            assert_eq!(PathKey::with_policy(p, X).as_str(), p);
        }
        assert_ne!(PathKey::with_policy("/a/B.jpg", X), PathKey::with_policy("/a/b.jpg", X));
        assert_ne!(PathKey::with_policy("/a/b.jpg", X), PathKey::with_policy("\\a\\b.jpg", X));
    }

    #[test]
    fn windows_policy_folds_separators_and_case() {
        let a = PathKey::with_policy("C:/Users/Me/Pictures/IMG_1.JPG", W);
        let b = PathKey::with_policy("c:\\users\\me\\pictures\\img_1.jpg", W);
        assert_eq!(a, b);
        assert_eq!(a.as_str(), "c:\\users\\me\\pictures\\img_1.jpg");
        assert_ne!(a, PathKey::with_policy("c:\\users\\me\\pictures\\img_2.jpg", W));
    }

    #[test]
    fn windows_policy_folds_non_ascii_letters() {
        assert_eq!(PathKey::with_policy("C:\\Fotos\\ÜBER.jpg", W), PathKey::with_policy("c:/fotos/über.jpg", W));
    }

    #[test]
    fn category_and_tag_paths_are_never_folded() {
        for p in ["cat://Landscape", "cat://sub/Dir", "tag://Red Fox", "tag://a/B"] {
            assert_eq!(PathKey::with_policy(p, W).as_str(), p, "{p}");
        }
        assert_ne!(PathKey::with_policy("tag://Fox", W), PathKey::with_policy("tag://fox", W));
        assert_ne!(PathKey::with_policy("cat://Cats", W), PathKey::with_policy("cat://cats", W));
    }

    #[test]
    fn archive_paths_fold_the_archive_but_not_the_inside() {
        let a = PathKey::with_policy("zip://C:/Data/Pics.ZIP!/Inner/Dir/Photo.JPG", W);
        let b = PathKey::with_policy("zip://c:\\data\\pics.zip!/Inner/Dir/Photo.JPG", W);
        assert_eq!(a, b);
        assert_eq!(a.as_str(), "zip://c:\\data\\pics.zip!/Inner/Dir/Photo.JPG");
        assert_ne!(a, PathKey::with_policy("zip://c:\\data\\pics.zip!/inner/dir/photo.jpg", W), "entries inside an archive are case sensitive");
        for scheme in ["tar", "7z"] {
            let p = format!("{scheme}://D:/x/A.{scheme}!/E/F.png");
            assert_eq!(PathKey::with_policy(&p, W).as_str(), format!("{scheme}://d:\\x\\a.{scheme}!/E/F.png"));
        }
    }

    #[test]
    fn archive_paths_without_an_inner_marker_fold_as_a_whole() {
        assert_eq!(PathKey::with_policy("zip://C:/A.ZIP", W).as_str(), "zip://c:\\a.zip");
    }

    #[test]
    fn schemes_are_recognised_and_exposed() {
        assert_eq!(PathKey::with_policy("zip://a!/b", X).scheme(), Some(Scheme::Zip));
        assert_eq!(PathKey::with_policy("tar://a!/b", X).scheme(), Some(Scheme::Tar));
        assert_eq!(PathKey::with_policy("7z://a!/b", X).scheme(), Some(Scheme::SevenZ));
        assert_eq!(PathKey::with_policy("cat://a", X).scheme(), Some(Scheme::Category));
        assert_eq!(PathKey::with_policy("tag://a", X).scheme(), Some(Scheme::Tag));
        assert_eq!(PathKey::with_policy("/plain/a", X).scheme(), None);
        assert!(PathKey::with_policy("zip://a!/b", X).is_virtual());
        assert!(Scheme::Zip.is_archive() && !Scheme::Tag.is_archive());
        let (scheme, rest) = split_virtual("7z://x/y.7z!/z").unwrap();
        assert_eq!((scheme, rest), (Scheme::SevenZ, "x/y.7z!/z"));
        assert!(split_virtual("http://x").is_none());
    }

    #[test]
    fn equivalence_follows_the_policy() {
        assert!(paths_equivalent("C:/A/b.JPG", "c:\\a\\B.jpg", W));
        assert!(!paths_equivalent("C:/A/b.JPG", "c:\\a\\B.jpg", X));
        assert!(paths_equivalent("/a/b", "/a/b", X));
        assert!(!paths_equivalent("tag://A", "tag://a", W));
    }

    #[test]
    fn keys_hash_and_compare_like_their_text() {
        use std::collections::HashMap;
        let mut m: HashMap<PathKey, u32> = HashMap::new();
        m.insert(PathKey::with_policy("/a/b", X), 7);
        assert_eq!(m.get("/a/b"), Some(&7));
        assert_eq!(m.get(&PathKey::with_policy("/a/b", X)), Some(&7));
        assert!(m.get("/a/c").is_none());
    }

    #[test]
    fn clones_share_the_text() {
        let a = PathKey::with_policy("/a/b", X);
        let b = a.clone();
        assert!(std::ptr::eq(a.as_str().as_ptr(), b.as_str().as_ptr()));
        assert_eq!(format!("{a}"), "/a/b");
        assert_eq!(format!("{a:?}"), "PathKey(\"/a/b\")");
    }

    #[test]
    fn the_native_policy_matches_the_platform() {
        assert_eq!(KeyPolicy::default(), KeyPolicy::NATIVE);
        if cfg!(windows) {
            assert_eq!(KeyPolicy::NATIVE, W);
        } else {
            assert_eq!(KeyPolicy::NATIVE, X);
            assert_eq!(PathKey::new("/Mixed/Case.JPG").as_str(), "/Mixed/Case.JPG");
        }
    }
}
