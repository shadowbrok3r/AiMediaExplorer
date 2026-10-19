//! Fuzzy matching of the command palette: ordered subsequence scoring with word-start and run bonuses.

const SCORE_MATCH: i32 = 16;
const BONUS_WORD_START: i32 = 12;
const BONUS_CAMEL: i32 = 8;
const BONUS_FIRST: i32 = 6;
const BONUS_CONSECUTIVE: i32 = 10;
const BONUS_EXACT_CASE: i32 = 1;
const PENALTY_GAP: i32 = 3;
const PENALTY_LEADING: i32 = 2;
const MAX_LEADING: i32 = 8;
/// Score cut from a token that only matches the keywords.
const PENALTY_KEYWORD: i32 = 24;
const NEG: i32 = i32::MIN / 2;

/// A successful match.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Match {
    pub score: i32,
    /// Character indices of the title that matched, ascending.
    pub positions: Vec<usize>,
}

/// An item with its match, as ranked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ranked {
    pub index: usize,
    pub found: Match,
}

fn lower(c: char) -> char {
    c.to_lowercase().next().unwrap_or(c)
}

fn boundary_bonus(prev: Option<char>, cur: char) -> i32 {
    match prev {
        None => BONUS_WORD_START + BONUS_FIRST,
        Some(p) if !p.is_alphanumeric() => BONUS_WORD_START,
        Some(p) if p.is_lowercase() && cur.is_uppercase() => BONUS_CAMEL,
        _ => 0,
    }
}

/// Matches `token` as an ordered subsequence of `text`, case-insensitively; `None` when it is not one.
pub fn fuzzy(token: &str, text: &str) -> Option<Match> {
    let query: Vec<char> = token.chars().filter(|c| !c.is_whitespace()).map(lower).collect();
    let chars: Vec<char> = text.chars().collect();
    if query.is_empty() {
        return Some(Match { score: 0, positions: Vec::new() });
    }
    let (m, n) = (query.len(), chars.len());
    if m > n {
        return None;
    }
    let folded: Vec<char> = chars.iter().copied().map(lower).collect();
    let bonus: Vec<i32> = (0..n).map(|j| boundary_bonus(j.checked_sub(1).map(|p| chars[p]), chars[j])).collect();
    let originals: Vec<char> = token.chars().filter(|c| !c.is_whitespace()).collect();

    let mut score = vec![vec![NEG; n]; m];
    let mut parent = vec![vec![usize::MAX; n]; m];
    for j in 0..n {
        if folded[j] == query[0] {
            let leading = (j as i32).min(MAX_LEADING) * PENALTY_LEADING;
            let case = if originals[0] == chars[j] { BONUS_EXACT_CASE } else { 0 };
            score[0][j] = SCORE_MATCH + bonus[j] + case - leading;
        }
    }
    for i in 1..m {
        let mut gap_best = (NEG, usize::MAX);
        for j in i..n {
            if j >= 2 {
                if gap_best.0 > NEG {
                    gap_best.0 -= PENALTY_GAP;
                }
                let from = score[i - 1][j - 2];
                if from > NEG && from - PENALTY_GAP > gap_best.0 {
                    gap_best = (from - PENALTY_GAP, j - 2);
                }
            }
            if folded[j] != query[i] {
                continue;
            }
            let (mut best, mut from) = (NEG, usize::MAX);
            if score[i - 1][j - 1] > NEG {
                (best, from) = (score[i - 1][j - 1] + BONUS_CONSECUTIVE, j - 1);
            }
            if gap_best.0 > best {
                (best, from) = gap_best;
            }
            if best > NEG {
                let case = if originals[i] == chars[j] { BONUS_EXACT_CASE } else { 0 };
                score[i][j] = best + SCORE_MATCH + bonus[j] + case;
                parent[i][j] = from;
            }
        }
    }
    let (end, top) = (0..n).map(|j| (j, score[m - 1][j])).max_by_key(|(j, s)| (*s, std::cmp::Reverse(*j)))?;
    if top <= NEG {
        return None;
    }
    let mut positions = vec![0; m];
    let mut at = end;
    for i in (0..m).rev() {
        positions[i] = at;
        if i > 0 {
            at = parent[i][at];
        }
    }
    let length_cost = (n as i32 - m as i32) / 8;
    Some(Match { score: top - length_cost, positions })
}

/// Matches every whitespace-separated token of `query` against `title`, or else against `keywords`.
pub fn score(query: &str, title: &str, keywords: &str) -> Option<Match> {
    let mut total = Match { score: 0, positions: Vec::new() };
    for token in query.split_whitespace() {
        if let Some(found) = fuzzy(token, title) {
            total.score += found.score;
            total.positions.extend(found.positions);
        } else if let Some(found) = fuzzy(token, keywords) {
            total.score += found.score - PENALTY_KEYWORD;
        } else {
            return None;
        }
    }
    total.positions.sort_unstable();
    total.positions.dedup();
    Some(total)
}

/// The items matching `query`, best first; ties and an empty query keep the given order.
pub fn rank<T>(query: &str, items: &[T], fields: impl Fn(&T) -> (&str, &str)) -> Vec<Ranked> {
    let mut ranked: Vec<Ranked> = items
        .iter()
        .enumerate()
        .filter_map(|(index, item)| {
            let (title, keywords) = fields(item);
            score(query, title, keywords).map(|found| Ranked { index, found })
        })
        .collect();
    if !query.trim().is_empty() {
        ranked.sort_by(|a, b| b.found.score.cmp(&a.found.score).then(a.index.cmp(&b.index)));
    }
    ranked
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pos(token: &str, text: &str) -> Vec<usize> {
        fuzzy(token, text).unwrap_or_else(|| panic!("{token:?} should match {text:?}")).positions
    }

    #[test]
    fn a_match_needs_the_characters_in_order() {
        assert_eq!(pos("lib", "Library"), vec![0, 1, 2]);
        assert!(fuzzy("bil", "Library").is_none());
        assert!(fuzzy("z", "Library").is_none());
        assert!(fuzzy("libraryy", "Library").is_none());
    }

    #[test]
    fn matching_ignores_case_and_whitespace_in_the_token() {
        assert!(fuzzy("LIB", "library").is_some());
        assert_eq!(fuzzy("l i b", "Library").map(|m| m.positions), Some(vec![0, 1, 2]));
    }

    #[test]
    fn an_empty_token_matches_everything_with_no_positions() {
        assert_eq!(fuzzy("", "anything"), Some(Match { score: 0, positions: vec![] }));
        assert_eq!(fuzzy("  ", ""), Some(Match { score: 0, positions: vec![] }));
    }

    #[test]
    fn a_prefix_beats_a_match_in_the_middle() {
        let prefix = fuzzy("sett", "Settings: Library").unwrap().score;
        let middle = fuzzy("sett", "Reset settings").unwrap().score;
        assert!(prefix > middle, "{prefix} vs {middle}");
    }

    #[test]
    fn word_starts_beat_scattered_letters() {
        let words = fuzzy("ec", "Export CLIP").unwrap();
        let scattered = fuzzy("ec", "Recent files").unwrap();
        assert!(words.score > scattered.score, "{words:?} vs {scattered:?}");
        assert_eq!(words.positions, vec![0, 7]);
    }

    #[test]
    fn a_run_beats_the_same_letters_apart() {
        let run = fuzzy("clip", "Generate CLIP now").unwrap().score;
        let apart = fuzzy("clip", "Cleanup lists in parts").unwrap().score;
        assert!(run > apart, "{run} vs {apart}");
    }

    #[test]
    fn positions_follow_the_best_alignment_not_the_first_letters() {
        assert_eq!(pos("fb", "foo bar"), vec![0, 4]);
        assert_eq!(pos("bar", "foobar bar"), vec![7, 8, 9]);
    }

    #[test]
    fn camel_case_humps_count_as_word_starts() {
        assert_eq!(pos("tm", "ThumbMenu"), vec![0, 5]);
    }

    #[test]
    fn shorter_titles_win_a_tie() {
        let short = fuzzy("tag", "Tag").unwrap().score;
        let long = fuzzy("tag", "Tag every single selected file now").unwrap().score;
        assert!(short > long);
    }

    #[test]
    fn unicode_text_is_matched_by_character() {
        assert_eq!(pos("ü", "Größe über"), vec![6]);
        assert!(fuzzy("e", "Größe").is_some());
    }

    #[test]
    fn every_token_must_match_the_title_or_the_keywords() {
        assert!(score("open lib", "Open library", "").is_some());
        assert!(score("open xyz", "Open library", "").is_none());
        let by_keyword = score("db", "Open library", "database db").unwrap();
        assert!(by_keyword.positions.is_empty());
        let by_title = score("lib", "Open library", "").unwrap();
        assert!(by_title.score > by_keyword.score);
    }

    #[test]
    fn token_order_does_not_matter() {
        assert!(score("lib open", "Open library", "").is_some());
    }

    #[test]
    fn ranking_sorts_by_score_and_keeps_ties_in_order() {
        let items = ["Reset settings", "Settings: Library", "Settings: Appearance", "Quit"];
        let ranked = rank("sett", &items, |s| (s, ""));
        let order: Vec<usize> = ranked.iter().map(|r| r.index).collect();
        assert_eq!(order, vec![1, 2, 0]);
    }

    #[test]
    fn an_empty_query_keeps_the_given_order() {
        let items = ["b", "a", "c"];
        let order: Vec<usize> = rank("", &items, |s| (s, "")).iter().map(|r| r.index).collect();
        assert_eq!(order, vec![0, 1, 2]);
        let order: Vec<usize> = rank("   ", &items, |s| (s, "")).iter().map(|r| r.index).collect();
        assert_eq!(order, vec![0, 1, 2]);
    }

    #[test]
    fn long_inputs_stay_fast() {
        let text = "word ".repeat(200);
        let started = std::time::Instant::now();
        for _ in 0..200 {
            assert!(fuzzy("wwww", &text).is_some());
        }
        assert!(started.elapsed() < std::time::Duration::from_secs(2));
    }
}
