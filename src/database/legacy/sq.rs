//! Streaming reader and writer for the SurrealQL subset found in `surreal export` files.

use anyhow::{Result, anyhow, bail};
use std::io::Read;

#[derive(Clone, Debug, PartialEq)]
pub struct RecId {
    pub table: String,
    pub key_raw: String,
    pub kind: KeyKind,
    pub key: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum KeyKind {
    Bare,
    Int,
    Angle,
    Backtick,
    Complex,
}

impl RecId {
    pub fn named(table: &str, key: &str) -> Self {
        Self { table: table.to_string(), key_raw: key.to_string(), kind: KeyKind::Backtick, key: key.to_string() }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum V {
    None,
    Null,
    Bool(bool),
    Num(String),
    Str(String),
    Dt(String),
    Uuid(String),
    Rec(RecId),
    Arr(Vec<V>),
    Obj(Vec<(String, V)>),
    Raw(String),
}

pub const DEFAULT_CHUNK: usize = 8 << 20;

pub struct Src<R: Read> {
    r: R,
    buf: Vec<u8>,
    pos: usize,
    len: usize,
    base: u64,
    eof: bool,
}

impl<R: Read> Src<R> {
    pub fn new(r: R) -> Self {
        Self::with_capacity(r, DEFAULT_CHUNK)
    }

    pub fn with_capacity(r: R, cap: usize) -> Self {
        Self { r, buf: vec![0; cap.max(1)], pos: 0, len: 0, base: 0, eof: false }
    }

    pub fn offset(&self) -> u64 {
        self.base + self.pos as u64
    }

    fn fill(&mut self) -> Result<bool> {
        if self.eof {
            return Ok(false);
        }
        self.base += self.len as u64;
        self.pos = 0;
        self.len = 0;
        loop {
            match self.r.read(&mut self.buf) {
                Ok(0) => {
                    self.eof = true;
                    return Ok(false);
                }
                Ok(n) => {
                    self.len = n;
                    return Ok(true);
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.into()),
            }
        }
    }

    #[inline]
    pub fn peek(&mut self) -> Result<Option<u8>> {
        if self.pos < self.len {
            return Ok(Some(self.buf[self.pos]));
        }
        if self.fill()? { Ok(Some(self.buf[self.pos])) } else { Ok(None) }
    }

    #[inline]
    pub fn bump(&mut self) -> Result<Option<u8>> {
        let b = self.peek()?;
        if b.is_some() {
            self.pos += 1;
        }
        Ok(b)
    }

    fn expect(&mut self, c: u8) -> Result<()> {
        match self.bump()? {
            Some(b) if b == c => Ok(()),
            other => bail!("expected {:?} at byte {}, found {:?}", c as char, self.offset(), other.map(|b| b as char)),
        }
    }

    pub fn skip_ws(&mut self) -> Result<()> {
        while let Some(b) = self.peek()? {
            if b.is_ascii_whitespace() {
                self.pos += 1;
            } else {
                break;
            }
        }
        Ok(())
    }

    /// Skips whitespace plus `-- ...` and `# ...` comment lines.
    pub fn skip_ws_comments(&mut self) -> Result<()> {
        loop {
            self.skip_ws()?;
            match self.peek()? {
                Some(b'-') => {
                    self.pos += 1;
                    if self.peek()? == Some(b'-') {
                        self.skip_line()?;
                    } else {
                        bail!("stray '-' at statement level, byte {}", self.offset());
                    }
                }
                Some(b'#') => self.skip_line()?,
                _ => return Ok(()),
            }
        }
    }

    fn skip_line(&mut self) -> Result<()> {
        while let Some(b) = self.bump()? {
            if b == b'\n' {
                break;
            }
        }
        Ok(())
    }

    /// Reads a quoted string body (opening quote consumed) and decodes escapes.
    pub fn read_quoted(&mut self, q: u8) -> Result<String> {
        let mut out: Vec<u8> = Vec::new();
        loop {
            if self.pos >= self.len && !self.fill()? {
                bail!("unterminated string at byte {}", self.offset());
            }
            let window = &self.buf[self.pos..self.len];
            match memchr::memchr2(q, b'\\', window) {
                None => {
                    out.extend_from_slice(window);
                    self.pos = self.len;
                }
                Some(i) => {
                    out.extend_from_slice(&window[..i]);
                    let hit = window[i];
                    self.pos += i + 1;
                    if hit == q {
                        break;
                    }
                    let e = self.bump()?.ok_or_else(|| anyhow!("dangling escape"))?;
                    match e {
                        b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'),
                        b't' => out.push(b'\t'),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        b'0' => out.push(0),
                        b'\\' => out.push(b'\\'),
                        b'\'' => out.push(b'\''),
                        b'"' => out.push(b'"'),
                        b'`' => out.push(b'`'),
                        b'/' => out.push(b'/'),
                        b'u' => {
                            let cp = self.read_unicode_escape()?;
                            let mut tmp = [0u8; 4];
                            out.extend_from_slice(cp.encode_utf8(&mut tmp).as_bytes());
                        }
                        other => {
                            out.push(b'\\');
                            out.push(other);
                        }
                    }
                }
            }
        }
        Ok(String::from_utf8(out).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned()))
    }

    fn read_hex(&mut self, n: usize) -> Result<u32> {
        let mut v = 0u32;
        for _ in 0..n {
            let b = self.bump()?.ok_or_else(|| anyhow!("short unicode escape"))?;
            v = v * 16 + (b as char).to_digit(16).ok_or_else(|| anyhow!("bad hex digit"))?;
        }
        Ok(v)
    }

    fn read_unicode_escape(&mut self) -> Result<char> {
        let first = if self.peek()? == Some(b'{') {
            self.pos += 1;
            let mut v = 0u32;
            loop {
                let b = self.bump()?.ok_or_else(|| anyhow!("short unicode escape"))?;
                if b == b'}' {
                    break;
                }
                v = v * 16 + (b as char).to_digit(16).ok_or_else(|| anyhow!("bad hex digit"))?;
            }
            v
        } else {
            self.read_hex(4)?
        };
        if (0xD800..0xDC00).contains(&first) {
            if self.peek()? == Some(b'\\') {
                self.pos += 1;
                self.expect(b'u')?;
                let lo = self.read_hex(4)?;
                let cp = 0x10000 + ((first - 0xD800) << 10) + (lo.wrapping_sub(0xDC00) & 0x3FF);
                return Ok(char::from_u32(cp).unwrap_or('\u{FFFD}'));
            }
            return Ok('\u{FFFD}');
        }
        Ok(char::from_u32(first).unwrap_or('\u{FFFD}'))
    }

    fn read_ident(&mut self) -> Result<String> {
        let mut s = String::new();
        while let Some(b) = self.peek()? {
            if b.is_ascii_alphanumeric() || b == b'_' {
                s.push(b as char);
                self.pos += 1;
            } else {
                break;
            }
        }
        Ok(s)
    }

    fn read_bare_token(&mut self) -> Result<String> {
        let mut s = String::new();
        while let Some(b) = self.peek()? {
            if b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-' | b'+') {
                s.push(b as char);
                self.pos += 1;
            } else {
                break;
            }
        }
        Ok(s)
    }

    /// Captures a balanced expression verbatim until a top-level delimiter.
    fn read_raw_until_delim(&mut self, acc: String) -> Result<String> {
        let mut out: Vec<u8> = acc.into_bytes();
        let mut depth = 0i32;
        while let Some(b) = self.peek()? {
            match b {
                b',' | b'}' | b']' | b';' if depth == 0 => break,
                b'(' | b'[' | b'{' => {
                    depth += 1;
                    out.push(b);
                    self.pos += 1;
                }
                b')' | b']' | b'}' => {
                    depth -= 1;
                    out.push(b);
                    self.pos += 1;
                }
                b'\'' | b'"' => {
                    self.pos += 1;
                    let inner = self.read_quoted(b)?;
                    push_requoted(&mut out, b, &inner);
                }
                _ => {
                    out.push(b);
                    self.pos += 1;
                }
            }
        }
        let s = String::from_utf8_lossy(&out).into_owned();
        Ok(s.trim_end().to_string())
    }

    fn read_record_key(&mut self, table: String) -> Result<V> {
        match self.peek()? {
            Some(0xE2) => {
                let mut bytes = [0u8; 3];
                for slot in bytes.iter_mut() {
                    *slot = self.bump()?.ok_or_else(|| anyhow!("unterminated key"))?;
                }
                if bytes != [0xE2, 0x9F, 0xA8] {
                    bail!("unexpected key start at byte {}", self.offset());
                }
                let mut inner = Vec::new();
                loop {
                    let b = self.bump()?.ok_or_else(|| anyhow!("unterminated angle-bracket key"))?;
                    if b == 0xE2 && self.peek()? == Some(0x9F) {
                        self.pos += 1;
                        let b3 = self.bump()?.ok_or_else(|| anyhow!("unterminated angle-bracket key"))?;
                        if b3 == 0xA9 {
                            break;
                        }
                        inner.extend_from_slice(&[0xE2, 0x9F, b3]);
                        continue;
                    }
                    if b == b'\\' {
                        let e = self.bump()?.ok_or_else(|| anyhow!("dangling escape in key"))?;
                        inner.push(e);
                        continue;
                    }
                    inner.push(b);
                }
                let key = String::from_utf8_lossy(&inner).into_owned();
                let raw = format!("\u{27E8}{key}\u{27E9}");
                Ok(V::Rec(RecId { table, key_raw: raw, kind: KeyKind::Angle, key }))
            }
            Some(b'`') => {
                self.pos += 1;
                let key = self.read_quoted(b'`')?;
                Ok(V::Rec(RecId { table, key_raw: format!("`{}`", key.replace('\\', "\\\\").replace('`', "\\`")), kind: KeyKind::Backtick, key }))
            }
            Some(b'[') | Some(b'{') => {
                let raw = self.read_raw_until_delim(String::new())?;
                Ok(V::Rec(RecId { table, key_raw: raw.clone(), kind: KeyKind::Complex, key: raw }))
            }
            Some(b) if b.is_ascii_alphanumeric() || b == b'_' || b == b'-' => {
                let tok = self.read_bare_token()?;
                let kind = if tok.bytes().all(|c| c.is_ascii_digit()) { KeyKind::Int } else { KeyKind::Bare };
                Ok(V::Rec(RecId { table, key_raw: tok.clone(), kind, key: tok }))
            }
            other => bail!("bad record key start {:?} at byte {}", other.map(|b| b as char), self.offset()),
        }
    }

    pub fn parse_value(&mut self) -> Result<V> {
        self.skip_ws()?;
        let b = self.peek()?.ok_or_else(|| anyhow!("unexpected end of input"))?;
        match b {
            b'{' => {
                self.pos += 1;
                let mut fields = Vec::new();
                loop {
                    self.skip_ws()?;
                    match self.peek()? {
                        Some(b'}') => {
                            self.pos += 1;
                            break;
                        }
                        Some(b',') => {
                            self.pos += 1;
                            continue;
                        }
                        None => bail!("unterminated object"),
                        _ => {}
                    }
                    let key = match self.peek()? {
                        Some(q @ (b'\'' | b'"' | b'`')) => {
                            self.pos += 1;
                            self.read_quoted(q)?
                        }
                        _ => self.read_ident()?,
                    };
                    if key.is_empty() {
                        bail!("empty object key at byte {}", self.offset());
                    }
                    self.skip_ws()?;
                    self.expect(b':')?;
                    let v = self.parse_value()?;
                    fields.push((key, v));
                }
                Ok(V::Obj(fields))
            }
            b'[' => {
                self.pos += 1;
                let mut items = Vec::new();
                loop {
                    self.skip_ws()?;
                    match self.peek()? {
                        Some(b']') => {
                            self.pos += 1;
                            break;
                        }
                        Some(b',') => {
                            self.pos += 1;
                            continue;
                        }
                        None => bail!("unterminated array"),
                        _ => {}
                    }
                    items.push(self.parse_value()?);
                }
                Ok(V::Arr(items))
            }
            b'\'' | b'"' => {
                self.pos += 1;
                Ok(V::Str(self.read_quoted(b)?))
            }
            b'`' => {
                self.pos += 1;
                let s = self.read_quoted(b'`')?;
                Ok(V::Raw(format!("`{s}`")))
            }
            b if b.is_ascii_alphabetic() || b == b'_' => {
                let ident = self.read_ident()?;
                match self.peek()? {
                    Some(q @ (b'\'' | b'"')) if matches!(ident.as_str(), "d" | "u" | "r" | "b" | "s" | "f") => {
                        self.pos += 1;
                        let inner = self.read_quoted(q)?;
                        return Ok(match ident.as_str() {
                            "d" => V::Dt(inner),
                            "u" => V::Uuid(inner),
                            "r" => match inner.split_once(':') {
                                Some((t, k)) => V::Rec(RecId::named(t, k)),
                                None => V::Raw(format!("r'{inner}'")),
                            },
                            other => V::Raw(format!("{other}'{}'", inner.replace('\'', "\\'"))),
                        });
                    }
                    Some(b':') => {
                        self.pos += 1;
                        if self.peek()? == Some(b':') {
                            self.pos += 1;
                            let mut path = format!("{ident}::");
                            loop {
                                path.push_str(&self.read_ident()?);
                                if self.peek()? == Some(b':') {
                                    self.pos += 1;
                                    self.expect(b':')?;
                                    path.push_str("::");
                                } else {
                                    break;
                                }
                            }
                            return Ok(V::Raw(self.read_raw_until_delim(path)?));
                        }
                        return self.read_record_key(ident);
                    }
                    Some(b'(') => return Ok(V::Raw(self.read_raw_until_delim(ident)?)),
                    _ => {}
                }
                Ok(match ident.as_str() {
                    "NONE" | "none" => V::None,
                    "NULL" | "null" => V::Null,
                    "true" => V::Bool(true),
                    "false" => V::Bool(false),
                    _ => V::Raw(self.read_raw_until_delim(ident)?),
                })
            }
            b if b.is_ascii_digit() || matches!(b, b'-' | b'+' | b'.') => {
                let tok = self.read_bare_token()?;
                if let Some(b':') = self.peek()? {
                    return Ok(V::Raw(self.read_raw_until_delim(tok)?));
                }
                if tok.is_empty() {
                    bail!("empty numeric token at byte {}", self.offset());
                }
                Ok(V::Num(tok))
            }
            other => bail!("unexpected byte {:?} at {}", other as char, self.offset()),
        }
    }

    /// Reads a plain statement up to the terminating `;` with quotes honoured.
    pub fn read_statement_text(&mut self, acc: String) -> Result<String> {
        let mut out: Vec<u8> = acc.into_bytes();
        while let Some(b) = self.bump()? {
            match b {
                b';' => {
                    out.push(b';');
                    break;
                }
                b'\'' | b'"' | b'`' => {
                    let inner = self.read_quoted(b)?;
                    push_requoted(&mut out, b, &inner);
                }
                _ => out.push(b),
            }
        }
        Ok(String::from_utf8_lossy(&out).into_owned())
    }
}

fn push_requoted(out: &mut Vec<u8>, q: u8, inner: &str) {
    out.push(q);
    for &c in inner.as_bytes() {
        if c == q || c == b'\\' {
            out.push(b'\\');
        }
        out.push(c);
    }
    out.push(q);
}

#[derive(Debug)]
pub enum Item {
    Statement(String),
    Row(V),
    EndBatch,
}

/// Pull parser over a whole `surreal export` file.
pub struct ExportReader<R: Read> {
    pub src: Src<R>,
    in_insert: bool,
    pub statements: u64,
}

impl<R: Read> ExportReader<R> {
    pub fn new(r: R) -> Self {
        Self { src: Src::new(r), in_insert: false, statements: 0 }
    }

    pub fn with_capacity(r: R, cap: usize) -> Self {
        Self { src: Src::with_capacity(r, cap), in_insert: false, statements: 0 }
    }

    pub fn next_item(&mut self) -> Result<Option<Item>> {
        loop {
            if self.in_insert {
                self.src.skip_ws()?;
                match self.src.peek()? {
                    Some(b',') => {
                        self.src.bump()?;
                        continue;
                    }
                    Some(b']') => {
                        self.src.bump()?;
                        self.src.skip_ws()?;
                        if self.src.peek()? == Some(b';') {
                            self.src.bump()?;
                        }
                        self.in_insert = false;
                        return Ok(Some(Item::EndBatch));
                    }
                    None => bail!("unterminated INSERT"),
                    _ => return Ok(Some(Item::Row(self.src.parse_value()?))),
                }
            }
            self.src.skip_ws_comments()?;
            let Some(first) = self.src.peek()? else { return Ok(None) };
            let mut word = String::new();
            if first.is_ascii_alphabetic() {
                while let Some(b) = self.src.peek()? {
                    if b.is_ascii_alphabetic() {
                        word.push(b as char);
                        self.src.bump()?;
                    } else {
                        break;
                    }
                }
            }
            self.statements += 1;
            if word.eq_ignore_ascii_case("INSERT") {
                self.src.skip_ws()?;
                let mut prefix = format!("{word} ");
                if matches!(self.src.peek()?, Some(b'I' | b'i')) {
                    let mut second = String::new();
                    while let Some(b) = self.src.peek()? {
                        if !b.is_ascii_alphabetic() {
                            break;
                        }
                        second.push(b as char);
                        self.src.bump()?;
                    }
                    if !second.eq_ignore_ascii_case("IGNORE") {
                        prefix.push_str(&second);
                        prefix.push(' ');
                    }
                    self.src.skip_ws()?;
                }
                if self.src.peek()? == Some(b'[') {
                    self.src.bump()?;
                    self.in_insert = true;
                    continue;
                }
                let t = self.src.read_statement_text(prefix)?;
                return Ok(Some(Item::Statement(t)));
            }
            let t = self.src.read_statement_text(word)?;
            return Ok(Some(Item::Statement(t)));
        }
    }
}

pub fn write_str(out: &mut Vec<u8>, s: &str) {
    out.push(b'\'');
    for &b in s.as_bytes() {
        match b {
            b'\\' => out.extend_from_slice(b"\\\\"),
            b'\'' => out.extend_from_slice(b"\\'"),
            b'\n' => out.extend_from_slice(b"\\n"),
            b'\r' => out.extend_from_slice(b"\\r"),
            b'\t' => out.extend_from_slice(b"\\t"),
            b if b < 0x20 => out.extend_from_slice(format!("\\u{:04x}", b).as_bytes()),
            b => out.push(b),
        }
    }
    out.push(b'\'');
}

fn is_simple_ident(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') && !s.as_bytes()[0].is_ascii_digit()
}

pub fn write_key(out: &mut Vec<u8>, k: &str) {
    if is_simple_ident(k) {
        out.extend_from_slice(k.as_bytes());
    } else {
        write_str(out, k);
    }
}

/// Writes a record id with a backtick-quoted key unless it is a plain identifier or integer.
pub fn write_recid(out: &mut Vec<u8>, r: &RecId) {
    out.extend_from_slice(r.table.as_bytes());
    out.push(b':');
    match r.kind {
        KeyKind::Complex => out.extend_from_slice(r.key_raw.as_bytes()),
        KeyKind::Int if r.key.parse::<i64>().is_ok_and(|n| n.to_string() == r.key) => out.extend_from_slice(r.key_raw.as_bytes()),
        _ => {
            if is_simple_ident(&r.key) {
                out.extend_from_slice(r.key.as_bytes());
            } else {
                out.push(b'`');
                for ch in r.key.chars() {
                    if ch == '`' || ch == '\\' {
                        out.push(b'\\');
                    }
                    let mut tmp = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut tmp).as_bytes());
                }
                out.push(b'`');
            }
        }
    }
}

pub fn write_value(out: &mut Vec<u8>, v: &V) {
    match v {
        V::None => out.extend_from_slice(b"NONE"),
        V::Null => out.extend_from_slice(b"NULL"),
        V::Bool(true) => out.extend_from_slice(b"true"),
        V::Bool(false) => out.extend_from_slice(b"false"),
        V::Num(n) => out.extend_from_slice(n.as_bytes()),
        V::Str(s) => write_str(out, s),
        V::Dt(s) => {
            out.extend_from_slice(b"d'");
            out.extend_from_slice(s.as_bytes());
            out.push(b'\'');
        }
        V::Uuid(s) => {
            out.extend_from_slice(b"u'");
            out.extend_from_slice(s.as_bytes());
            out.push(b'\'');
        }
        V::Rec(r) => write_recid(out, r),
        V::Arr(items) => {
            out.push(b'[');
            for (i, it) in items.iter().enumerate() {
                if i > 0 {
                    out.extend_from_slice(b", ");
                }
                write_value(out, it);
            }
            out.push(b']');
        }
        V::Obj(fields) => {
            out.extend_from_slice(b"{ ");
            for (i, (k, val)) in fields.iter().enumerate() {
                if i > 0 {
                    out.extend_from_slice(b", ");
                }
                write_key(out, k);
                out.extend_from_slice(b": ");
                write_value(out, val);
            }
            out.extend_from_slice(b" }");
        }
        V::Raw(s) => out.extend_from_slice(s.as_bytes()),
    }
}

pub fn obj_get<'a>(fields: &'a [(String, V)], key: &str) -> Option<&'a V> {
    fields.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

pub fn obj_str<'a>(fields: &'a [(String, V)], key: &str) -> Option<&'a str> {
    match obj_get(fields, key) {
        Some(V::Str(s)) => Some(s.as_str()),
        _ => None,
    }
}

pub fn obj_take(fields: &mut Vec<(String, V)>, key: &str) -> Option<V> {
    let i = fields.iter().position(|(k, _)| k == key)?;
    Some(fields.remove(i).1)
}
