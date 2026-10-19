//! Tolerant JSON extraction from model replies: fenced code, prose around the value, trailing commas and truncated arrays.

use serde_json::Value;

/// The kind of value the caller wants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    Any,
    Array,
    Object,
    /// An array whose elements are all objects.
    Rows,
}

/// How much the extractor had to change to read the value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quality {
    /// The text, a fence or a slice of it parsed as it was.
    Clean,
    /// Trailing commas were removed or the value was unwrapped or wrapped to fit the shape.
    Repaired,
    /// The reply was cut off; only the complete leading elements are kept.
    Salvaged,
}

impl Quality {
    pub fn label(self) -> &'static str {
        match self {
            Self::Clean => "clean",
            Self::Repaired => "repaired",
            Self::Salvaged => "salvaged",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Extracted {
    pub value: Value,
    pub quality: Quality,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExtractError {
    Empty,
    NoJson,
}

impl std::fmt::Display for ExtractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => f.write_str("the reply was empty"),
            Self::NoJson => f.write_str("the reply held no JSON of the expected kind"),
        }
    }
}

/// Reads the JSON value `shape` asks for out of `text`.
pub fn extract(text: &str, shape: Shape) -> Result<Extracted, ExtractError> {
    let cleaned = strip_reasoning(text);
    let body = cleaned.trim().trim_start_matches('\u{feff}').trim();
    if body.is_empty() {
        return Err(ExtractError::Empty);
    }
    for candidate in candidates(body, shape) {
        if let Some(found) = read_candidate(candidate, shape) {
            return Ok(found);
        }
    }
    Err(ExtractError::NoJson)
}

/// The text without `<think>` blocks; an unterminated block drops the rest.
fn strip_reasoning(text: &str) -> String {
    let lower = text.to_ascii_lowercase();
    let mut out = String::with_capacity(text.len());
    let mut at = 0;
    while let Some(open) = lower[at..].find("<think>") {
        let open = at + open;
        out.push_str(&text[at..open]);
        match lower[open..].find("</think>") {
            Some(close) => at = open + close + "</think>".len(),
            None => return out,
        }
    }
    out.push_str(&text[at..]);
    out
}

/// Slices of `body` worth parsing, best first: the whole text, fenced blocks, then values starting at a bracket.
fn candidates(body: &str, shape: Shape) -> Vec<&str> {
    let mut out = vec![body];
    out.extend(fenced_blocks(body));
    let wanted: &[u8] = match shape {
        Shape::Object => b"{[",
        Shape::Any => b"[{",
        Shape::Array | Shape::Rows => b"[{",
    };
    for first in wanted {
        let mut tried = 0;
        for (i, b) in body.bytes().enumerate() {
            if b != *first {
                continue;
            }
            out.push(balanced_or_tail(&body[i..]));
            tried += 1;
            if tried >= 8 {
                break;
            }
        }
    }
    out
}

/// The contents of every fenced code block; an unterminated last fence runs to the end of the text.
fn fenced_blocks(body: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = body;
    while let Some(start) = rest.find("```") {
        let after = &rest[start + 3..];
        let content_start = match after.find('\n') {
            Some(nl) if after[..nl].trim().chars().all(|c| c.is_ascii_alphanumeric()) => nl + 1,
            _ => 0,
        };
        let content = &after[content_start..];
        match content.find("```") {
            Some(end) => {
                out.push(content[..end].trim());
                rest = &content[end + 3..];
            }
            None => {
                out.push(content.trim());
                break;
            }
        }
    }
    out
}

/// The balanced value at the start of `s`, or all of `s` when it is cut off.
fn balanced_or_tail(s: &str) -> &str {
    let (mut depth, mut in_str, mut esc) = (0i32, false, false);
    for (i, b) in s.bytes().enumerate() {
        if in_str {
            match (esc, b) {
                (true, _) => esc = false,
                (false, b'\\') => esc = true,
                (false, b'"') => in_str = false,
                _ => {}
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'[' | b'{' => depth += 1,
            b']' | b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &s[..=i];
                }
            }
            _ => {}
        }
    }
    s
}

fn read_candidate(candidate: &str, shape: Shape) -> Option<Extracted> {
    let candidate = candidate.trim();
    if !(candidate.starts_with('[') || candidate.starts_with('{')) {
        return None;
    }
    let mut attempts = vec![(candidate.to_string(), Quality::Clean), (strip_trailing_commas(candidate), Quality::Repaired)];
    attempts.extend(salvage(candidate).map(|text| (text, Quality::Salvaged)));
    for (text, quality) in attempts {
        let Ok(value) = serde_json::from_str::<Value>(&text) else { continue };
        if let Some((value, adjusted)) = fit(value, shape) {
            let quality = if quality == Quality::Clean && adjusted { Quality::Repaired } else { quality };
            return Some(Extracted { value, quality });
        }
    }
    None
}

/// `value` in the shape asked for, and whether it had to be unwrapped or wrapped.
fn fit(value: Value, shape: Shape) -> Option<(Value, bool)> {
    match (shape, value) {
        (Shape::Any, v) => Some((v, false)),
        (Shape::Object, v @ Value::Object(_)) => Some((v, false)),
        (Shape::Object, Value::Array(mut a)) if a.len() == 1 && a[0].is_object() => Some((a.remove(0), true)),
        (Shape::Array, v @ Value::Array(_)) => Some((v, false)),
        (Shape::Rows, Value::Array(a)) if a.iter().all(Value::is_object) => Some((Value::Array(a), false)),
        (Shape::Array | Shape::Rows, Value::Object(map)) => {
            let holds_rows = |v: &Value| v.as_array().is_some_and(|a| a.iter().all(Value::is_object) && (!a.is_empty() || map.len() == 1));
            let arrays: Vec<String> = map.iter().filter(|(_, v)| holds_rows(v)).map(|(k, _)| k.clone()).collect();
            if let [key] = arrays.as_slice() {
                let mut map = map;
                let inner = map.remove(key)?;
                return fit(inner, shape).map(|(v, _)| (v, true));
            }
            Some((Value::Array(vec![Value::Object(map)]), true))
        }
        _ => None,
    }
}

/// `text` without the commas that sit right before a closing bracket.
fn strip_trailing_commas(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let (mut in_str, mut esc) = (false, false);
    let mut last_copied = 0;
    for (i, &b) in bytes.iter().enumerate() {
        if in_str {
            match (esc, b) {
                (true, _) => esc = false,
                (false, b'\\') => esc = true,
                (false, b'"') => in_str = false,
                _ => {}
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b',' => {
                let next = bytes[i + 1..].iter().find(|c| !c.is_ascii_whitespace());
                if matches!(next, Some(b']') | Some(b'}')) {
                    out.push_str(&text[last_copied..i]);
                    last_copied = i + 1;
                }
            }
            _ => {}
        }
    }
    out.push_str(&text[last_copied..]);
    out
}

/// A cut-off container closed after its last complete top-level element or member.
fn salvage(text: &str) -> Option<String> {
    let open = *text.as_bytes().first()?;
    let close = match open {
        b'[' => ']',
        b'{' => '}',
        _ => return None,
    };
    let is_array = open == b'[';
    let (mut depth, mut in_str, mut esc) = (0i32, false, false);
    let mut last_end = None;
    for (i, b) in text.bytes().enumerate() {
        if in_str {
            match (esc, b) {
                (true, _) => esc = false,
                (false, b'\\') => esc = true,
                (false, b'"') => {
                    in_str = false;
                    if is_array && depth == 1 {
                        last_end = Some(i + 1);
                    }
                }
                _ => {}
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'[' | b'{' => depth += 1,
            b']' | b'}' => {
                depth -= 1;
                if depth == 1 {
                    last_end = Some(i + 1);
                }
                if depth == 0 {
                    return None;
                }
            }
            b',' if depth == 1 => last_end = Some(i),
            _ => {}
        }
    }
    let end = last_end.unwrap_or(1);
    let kept = text[..end].trim_end().trim_end_matches(',');
    Some(format!("{kept}{close}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct Case {
        name: &'static str,
        text: &'static str,
        shape: Shape,
        want: Result<(Value, Quality), ExtractError>,
    }

    fn ok(value: Value, quality: Quality) -> Result<(Value, Quality), ExtractError> {
        Ok((value, quality))
    }

    #[test]
    fn replies_of_every_shape_are_read_or_refused_as_listed() {
        let cases = vec![
            Case { name: "clean rows", text: r#"[{"id":1},{"id":2}]"#, shape: Shape::Rows, want: ok(json!([{"id":1},{"id":2}]), Quality::Clean) },
            Case { name: "clean object", text: r#"{"a":1}"#, shape: Shape::Object, want: ok(json!({"a":1}), Quality::Clean) },
            Case { name: "json fence", text: "```json\n[{\"id\":1}]\n```", shape: Shape::Rows, want: ok(json!([{"id":1}]), Quality::Clean) },
            Case { name: "bare fence", text: "```\n{\"a\":1}\n```", shape: Shape::Object, want: ok(json!({"a":1}), Quality::Clean) },
            Case { name: "fence with prose", text: "Sure!\n```json\n[{\"id\":1}]\n```\nAnything else?", shape: Shape::Rows, want: ok(json!([{"id":1}]), Quality::Clean) },
            Case { name: "prose around", text: "Here you go: [{\"id\":1}] hope it helps", shape: Shape::Rows, want: ok(json!([{"id":1}]), Quality::Clean) },
            Case { name: "trailing commas", text: r#"[{"id":1,"t":["a","b",],},{"id":2},]"#, shape: Shape::Rows, want: ok(json!([{"id":1,"t":["a","b"]},{"id":2}]), Quality::Repaired) },
            Case { name: "comma inside a string stays", text: r#"[{"id":1,"c":"a, ]"},]"#, shape: Shape::Rows, want: ok(json!([{"id":1,"c":"a, ]"}]), Quality::Repaired) },
            Case { name: "cut inside an object", text: r#"[{"id":1,"c":"x"},{"id":2,"c":"y"},{"id":3,"c":"tr"#, shape: Shape::Rows, want: ok(json!([{"id":1,"c":"x"},{"id":2,"c":"y"}]), Quality::Salvaged) },
            Case { name: "cut after a comma", text: r#"[{"id":1},{"id":2},"#, shape: Shape::Rows, want: ok(json!([{"id":1},{"id":2}]), Quality::Salvaged) },
            Case { name: "cut inside a nested array", text: r#"[{"id":1,"t":["a"]},{"id":2,"t":["b","#, shape: Shape::Rows, want: ok(json!([{"id":1,"t":["a"]}]), Quality::Salvaged) },
            Case { name: "cut inside an unterminated fence", text: "```json\n[{\"id\":1},{\"id\":2,\"c\":\"abc", shape: Shape::Rows, want: ok(json!([{"id":1}]), Quality::Salvaged) },
            Case { name: "cut before the first element ends", text: r#"[{"id":1,"c"#, shape: Shape::Rows, want: ok(json!([]), Quality::Salvaged) },
            Case { name: "cut scalar array keeps closed strings", text: r#"["a","b","c"#, shape: Shape::Array, want: ok(json!(["a", "b"]), Quality::Salvaged) },
            Case { name: "cut object keeps complete members", text: r#"{"tags":["a"],"category":"x","caption":"tru"#, shape: Shape::Object, want: ok(json!({"tags":["a"],"category":"x"}), Quality::Salvaged) },
            Case { name: "wrapped rows are unwrapped", text: r#"{"proposals":[{"id":1}]}"#, shape: Shape::Rows, want: ok(json!([{"id":1}]), Quality::Repaired) },
            Case { name: "a lone row is wrapped", text: r#"{"id":1,"new_category":"x"}"#, shape: Shape::Rows, want: ok(json!([{"id":1,"new_category":"x"}]), Quality::Repaired) },
            Case { name: "a one-row array is an object when one is wanted", text: r#"[{"a":1}]"#, shape: Shape::Object, want: ok(json!({"a":1}), Quality::Repaired) },
            Case { name: "brackets inside strings", text: r#"[{"t":"a]b}c"}]"#, shape: Shape::Rows, want: ok(json!([{"t":"a]b}c"}]), Quality::Clean) },
            Case { name: "braces in prose before the value", text: r#"Use {curly} words. Answer: {"a":"}"} done"#, shape: Shape::Object, want: ok(json!({"a":"}"}), Quality::Clean) },
            Case { name: "a note in brackets before the rows", text: r#"See [1] below: [{"id":1}]"#, shape: Shape::Rows, want: ok(json!([{"id":1}]), Quality::Clean) },
            Case { name: "reasoning blocks are dropped", text: "<think>maybe [{\"id\":9}]</think>[{\"id\":1}]", shape: Shape::Rows, want: ok(json!([{"id":1}]), Quality::Clean) },
            Case { name: "an unfinished reasoning block leaves nothing", text: "<think>hmm [{\"id\":9}]", shape: Shape::Rows, want: Err(ExtractError::Empty) },
            Case { name: "any shape takes a number array", text: "[1,2]", shape: Shape::Any, want: ok(json!([1, 2]), Quality::Clean) },
            Case { name: "rows refuse a number array", text: "[1,2]", shape: Shape::Rows, want: Err(ExtractError::NoJson) },
            Case { name: "no json at all", text: "I cannot help with that.", shape: Shape::Rows, want: Err(ExtractError::NoJson) },
            Case { name: "empty", text: "   \n", shape: Shape::Any, want: Err(ExtractError::Empty) },
            Case { name: "an object where rows have several arrays", text: r#"{"a":[1],"b":[2]}"#, shape: Shape::Rows, want: ok(json!([{"a":[1],"b":[2]}]), Quality::Repaired) },
            Case { name: "unbalanced garbage", text: "[[[", shape: Shape::Rows, want: ok(json!([]), Quality::Salvaged) },
        ];
        for case in cases {
            let got = extract(case.text, case.shape).map(|e| (e.value, e.quality));
            assert_eq!(got, case.want, "{}", case.name);
        }
    }

    #[test]
    fn a_long_reply_is_read_in_one_pass() {
        let rows: Vec<String> = (0..2000).map(|i| format!(r#"{{"id":{i},"new_tags":["a","b"],"new_caption":"text {i}"}}"#)).collect();
        let joined = rows.join(",");
        let text = format!("Here:\n```json\n[{}", &joined[..joined.len() - 10]);
        let got = extract(&text, Shape::Rows).unwrap();
        assert_eq!(got.quality, Quality::Salvaged);
        assert_eq!(got.value.as_array().map(Vec::len), Some(1999), "the last row is cut off");
    }

    #[test]
    fn quality_names_are_distinct() {
        let names: Vec<&str> = [Quality::Clean, Quality::Repaired, Quality::Salvaged].iter().map(|q| q.label()).collect();
        assert_eq!(names, ["clean", "repaired", "salvaged"]);
    }
}
