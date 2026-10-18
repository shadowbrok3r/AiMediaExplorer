//! Thumbnail identity: blob key derived from the stored path, and the size tier of a cached texture.

use std::fmt;

/// First 16 bytes of the blake3 hash of the stored path string, as used for `thumb_blobs:<key>`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ThumbKey(pub [u8; 16]);

impl ThumbKey {
    pub fn from_path(path: &str) -> Self {
        let hash = blake3::hash(path.as_bytes());
        let mut key = [0u8; 16];
        key.copy_from_slice(&hash.as_bytes()[..16]);
        Self(key)
    }

    /// Lowercase hex, the record id suffix.
    pub fn hex(&self) -> String {
        let mut out = String::with_capacity(32);
        for b in self.0 {
            out.push(char::from_digit(u32::from(b >> 4), 16).unwrap_or('0'));
            out.push(char::from_digit(u32::from(b & 0xF), 16).unwrap_or('0'));
        }
        out
    }

    pub fn from_hex(hex: &str) -> Option<Self> {
        if hex.len() != 32 || !hex.is_ascii() {
            return None;
        }
        let mut key = [0u8; 16];
        for (i, slot) in key.iter_mut().enumerate() {
            *slot = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).ok()?;
        }
        Some(Self(key))
    }
}

impl fmt::Debug for ThumbKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ThumbKey({})", self.hex())
    }
}

impl fmt::Display for ThumbKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.hex())
    }
}

/// Resolution tier of a texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tier {
    /// Stored thumbnail, about 256 px on the long side.
    Thumb,
    /// Larger preview decoded straight from the source file.
    Preview,
}

impl Tier {
    pub const PREVIEW_MAX_DIM: u32 = 1600;

    pub fn label(self) -> &'static str {
        match self {
            Tier::Thumb => "thumb",
            Tier::Preview => "preview",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub key: ThumbKey,
    pub tier: Tier,
}

impl CacheKey {
    pub fn new(path: &str, tier: Tier) -> Self {
        Self { key: ThumbKey::from_path(path), tier }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_is_first_sixteen_bytes_of_blake3_in_lowercase_hex() {
        let path = "/photos/a b/img_1.jpg";
        let full = blake3::hash(path.as_bytes()).to_hex().to_string();
        let key = ThumbKey::from_path(path);
        assert_eq!(key.hex(), full[..32]);
        assert_eq!(key.hex(), key.hex().to_lowercase());
        assert_eq!(key.to_string().len(), 32);
    }

    #[test]
    fn hex_round_trips_and_rejects_bad_input() {
        let key = ThumbKey::from_path("/x.jpg");
        assert_eq!(ThumbKey::from_hex(&key.hex()), Some(key));
        assert_eq!(ThumbKey::from_hex("abc"), None);
        assert_eq!(ThumbKey::from_hex(&"zz".repeat(16)), None);
        assert_eq!(ThumbKey::from_hex(&"\u{e9}".repeat(16)), None);
    }

    #[test]
    fn different_paths_and_exact_strings_give_different_keys() {
        assert_ne!(ThumbKey::from_path("/a.jpg"), ThumbKey::from_path("/A.jpg"));
        assert_ne!(ThumbKey::from_path("/a.jpg"), ThumbKey::from_path("/a.jpg "));
        assert_eq!(ThumbKey::from_path("/a.jpg"), ThumbKey::from_path("/a.jpg"));
    }
}
