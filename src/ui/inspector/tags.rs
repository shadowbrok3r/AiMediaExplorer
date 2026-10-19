//! Tag arithmetic for the metadata editor and the batch editor: tallies over a selection, common tags and autocomplete.

use std::collections::HashMap;

use crate::database::metadata_edit::{normalize_tag, normalize_tags};

/// How many of the counted rows carry a tag.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TagCount {
    pub tag: String,
    pub count: usize,
}

/// Tags over a set of rows.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TagTally {
    pub rows: usize,
    /// Most used first, then alphabetical ignoring case.
    pub tags: Vec<TagCount>,
}

impl TagTally {
    /// Counts the tags of each row once per row, merging spellings that differ in case (the first spelling wins).
    pub fn of<'a>(rows: impl IntoIterator<Item = &'a [String]>) -> Self {
        let mut counts: HashMap<String, (String, usize)> = HashMap::new();
        let mut order: Vec<String> = Vec::new();
        let mut n = 0;
        for tags in rows {
            n += 1;
            for tag in normalize_tags(tags) {
                let key = tag.to_lowercase();
                match counts.get_mut(&key) {
                    Some(entry) => entry.1 += 1,
                    None => {
                        order.push(key.clone());
                        counts.insert(key, (tag, 1));
                    }
                }
            }
        }
        let mut tags: Vec<TagCount> = order.into_iter().filter_map(|k| counts.remove(&k)).map(|(tag, count)| TagCount { tag, count }).collect();
        tags.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.tag.to_lowercase().cmp(&b.tag.to_lowercase())).then_with(|| a.tag.cmp(&b.tag)));
        Self { rows: n, tags }
    }

    /// Tags every row has; none for an empty set.
    pub fn common(&self) -> Vec<&TagCount> {
        if self.rows == 0 {
            return Vec::new();
        }
        self.tags.iter().filter(|t| t.count == self.rows).collect()
    }

    /// Tags only some rows have.
    pub fn partial(&self) -> Vec<&TagCount> {
        self.tags.iter().filter(|t| t.count < self.rows).collect()
    }
}

/// Tags typed into the input: split at commas and normalized.
pub fn split_input(text: &str) -> Vec<String> {
    normalize_tags(text.split(','))
}

/// Splits typed text at the last comma: the tags completed so far and the text still being typed.
pub fn take_complete(text: &str) -> (Vec<String>, String) {
    match text.rsplit_once(',') {
        Some((done, rest)) => (split_input(done), rest.trim_start().to_string()),
        None => (Vec::new(), text.to_string()),
    }
}

/// Up to `limit` known tags matching `typed`, prefix matches before other matches, ignoring case, leaving out tags in `present`.
pub fn suggest(known: &[String], typed: &str, present: &[String], limit: usize) -> Vec<String> {
    let Some(needle) = normalize_tag(typed).map(|t| t.to_lowercase()) else { return Vec::new() };
    let have: std::collections::HashSet<String> = present.iter().map(|t| t.trim().to_lowercase()).collect();
    let mut prefix = Vec::new();
    let mut inner = Vec::new();
    for tag in known {
        let lower = tag.to_lowercase();
        if have.contains(&lower) || lower == needle {
            continue;
        }
        if lower.starts_with(&needle) {
            prefix.push(tag.clone());
        } else if lower.contains(&needle) {
            inner.push(tag.clone());
        }
    }
    prefix.extend(inner);
    prefix.truncate(limit);
    prefix
}

/// `tags` with `tag` added unless already there ignoring case.
pub fn with_tag(tags: &[String], tag: &str) -> Vec<String> {
    normalize_tags(tags.iter().map(String::as_str).chain(std::iter::once(tag)))
}

/// `tags` without `tag` ignoring case.
pub fn without_tag(tags: &[String], tag: &str) -> Vec<String> {
    let lower = tag.trim().to_lowercase();
    tags.iter().filter(|t| t.trim().to_lowercase() != lower).cloned().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(list: &[&str]) -> Vec<String> {
        list.iter().map(|t| t.to_string()).collect()
    }

    #[test]
    fn the_tally_counts_each_tag_once_per_row_ignoring_case() {
        let rows = [v(&["Sunset", "beach"]), v(&["sunset", "dog", "dog"]), v(&["SUNSET"])];
        let tally = TagTally::of(rows.iter().map(Vec::as_slice));
        assert_eq!(tally.rows, 3);
        assert_eq!(tally.tags[0], TagCount { tag: "Sunset".into(), count: 3 });
        assert_eq!(tally.tags.iter().map(|t| (t.tag.as_str(), t.count)).collect::<Vec<_>>(), vec![("Sunset", 3), ("beach", 1), ("dog", 1)]);
    }

    #[test]
    fn common_tags_are_those_of_every_row_and_the_rest_are_partial() {
        let rows = [v(&["a", "b"]), v(&["a", "c"]), v(&["a", "b", "c"])];
        let tally = TagTally::of(rows.iter().map(Vec::as_slice));
        assert_eq!(tally.common().iter().map(|t| t.tag.as_str()).collect::<Vec<_>>(), vec!["a"]);
        assert_eq!(tally.partial().iter().map(|t| (t.tag.as_str(), t.count)).collect::<Vec<_>>(), vec![("b", 2), ("c", 2)]);
    }

    #[test]
    fn an_empty_set_has_no_common_tags_and_untagged_rows_empty_the_intersection() {
        assert!(TagTally::of(std::iter::empty()).common().is_empty());
        let rows = [v(&["a"]), v(&[])];
        let tally = TagTally::of(rows.iter().map(Vec::as_slice));
        assert!(tally.common().is_empty());
        assert_eq!(tally.partial().len(), 1);
    }

    #[test]
    fn typed_tags_split_at_commas_and_drop_blanks_and_repeats() {
        assert_eq!(split_input(" a, b ,, A ,c d"), v(&["a", "b", "c d"]));
        assert!(split_input(" , ").is_empty());
    }

    #[test]
    fn typing_a_comma_completes_the_tags_before_it() {
        assert_eq!(take_complete("cat, dog, bi"), (v(&["cat", "dog"]), "bi".to_string()));
        assert_eq!(take_complete("cat,"), (v(&["cat"]), String::new()));
        assert_eq!(take_complete("cat"), (Vec::new(), "cat".to_string()));
        assert_eq!(take_complete(", ,"), (Vec::new(), String::new()));
    }

    #[test]
    fn suggestions_put_prefix_matches_first_and_skip_present_and_exact_tags() {
        let known = v(&["sunset", "Sun", "unsung", "moon", "sunrise", "night"]);
        assert_eq!(suggest(&known, "sun", &[], 8), v(&["sunset", "sunrise", "unsung"]), "an exact match is not suggested again");
        assert_eq!(suggest(&known, "SUN", &v(&["sunset"]), 8), v(&["sunrise", "unsung"]));
        assert_eq!(suggest(&known, "su", &[], 2), v(&["sunset", "Sun"]));
        assert!(suggest(&known, "  ", &[], 8).is_empty());
        assert!(suggest(&known, "zzz", &[], 8).is_empty());
    }

    #[test]
    fn adding_and_removing_a_tag_ignores_case() {
        assert_eq!(with_tag(&v(&["a"]), " A "), v(&["a"]));
        assert_eq!(with_tag(&v(&["a"]), "b c"), v(&["a", "b c"]));
        assert_eq!(without_tag(&v(&["Cat", "dog"]), "cat"), v(&["dog"]));
        assert_eq!(without_tag(&v(&["Cat"]), "bird"), v(&["Cat"]));
    }
}
