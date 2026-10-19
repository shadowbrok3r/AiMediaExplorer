//! The Jina M0 text reranker over the engine that is loaded in this process.

use std::collections::BTreeSet;

use super::{HeuristicReranker, Reranker};

pub struct JinaReranker;

fn unique_lower(candidates: &[String]) -> Vec<String> {
    let set: BTreeSet<String> = candidates.iter().map(|c| c.trim().to_ascii_lowercase()).filter(|c| !c.is_empty()).collect();
    set.into_iter().collect()
}

fn trimmed(candidates: &[String]) -> Vec<String> {
    candidates.iter().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()
}

/// Scores of `candidates` against `query`, or none when the engine answered with another number of scores.
fn score(query: &str, candidates: &[String]) -> Option<Vec<f32>> {
    let pairs: Vec<(String, String)> = candidates.iter().map(|c| (query.to_string(), c.clone())).collect();
    let scores = crate::ai::reranker::jina_score_text_pairs_blocking(&pairs);
    if scores.is_empty() || scores.len() != pairs.len() {
        log::warn!("[Refine] Jina returned {} scores for {} pairs; using the heuristic", scores.len(), pairs.len());
        return None;
    }
    Some(scores)
}

/// Candidates best first by score.
pub(super) fn rank(candidates: Vec<String>, scores: Vec<f32>) -> Vec<String> {
    let mut zipped: Vec<(String, f32)> = candidates.into_iter().zip(scores).collect();
    zipped.sort_by(|a, b| b.1.total_cmp(&a.1));
    zipped.into_iter().map(|(s, _)| s).collect()
}

fn best(query: &str, candidates: Vec<String>) -> Option<String> {
    if candidates.is_empty() {
        return None;
    }
    match score(query, &candidates) {
        Some(scores) => rank(candidates, scores).into_iter().next(),
        None => candidates.into_iter().next(),
    }
}

impl Reranker for JinaReranker {
    fn name(&self) -> &'static str {
        "Jina M0"
    }

    fn rerank_tags(&mut self, query: &str, candidates: &[String]) -> Vec<String> {
        let cands = unique_lower(candidates);
        if cands.is_empty() {
            return Vec::new();
        }
        match score(query, &cands) {
            Some(scores) => rank(cands, scores).into_iter().take(12).collect(),
            None => HeuristicReranker.rerank_tags(query, candidates),
        }
    }

    fn rerank_category(&mut self, query: &str, candidates: &[String]) -> Option<String> {
        best(query, unique_lower(candidates))
    }

    fn rerank_caption(&mut self, query: &str, candidates: &[String]) -> Option<String> {
        best(query, trimmed(candidates))
    }

    fn rerank_description(&mut self, query: &str, candidates: &[String]) -> Option<String> {
        best(query, trimmed(candidates))
    }
}
