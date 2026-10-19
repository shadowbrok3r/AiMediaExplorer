//! Metadata refinement proposals: the neighbours of a file by CLIP embedding vote on its tags, category, caption and description.

use std::collections::{BTreeMap, BTreeSet};

use super::ai_search::AISearchEngine;
use crate::Thumbnail;

mod cloud;
mod jina;
#[cfg(test)]
mod tests;

pub use cloud::{CloudReply, cloud_prompt, parse_cloud_reply, provider_config, refine_with_cloud, reply_to_proposal};
pub use jina::JinaReranker;

/// Rows read for one run without a selection.
pub const RICH_ROW_LIMIT: usize = 200;
/// Neighbours that vote on each file.
const NEIGHBOURS: usize = 50;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RerankerType {
    #[default]
    Heuristic,
    JinaM0,
}

pub trait Reranker: Send + Sync {
    fn name(&self) -> &'static str;
    fn rerank_tags(&mut self, query: &str, candidates: &[String]) -> Vec<String>;
    fn rerank_category(&mut self, query: &str, candidates: &[String]) -> Option<String>;
    fn rerank_caption(&mut self, query: &str, candidates: &[String]) -> Option<String>;
    fn rerank_description(&mut self, query: &str, candidates: &[String]) -> Option<String>;
}

pub struct HeuristicReranker;

/// The most frequent non-blank text, else the longest.
fn most_common_text(candidates: &[String]) -> Option<String> {
    let mut counts: BTreeMap<String, i64> = BTreeMap::new();
    for c in candidates {
        let n = c.trim();
        if !n.is_empty() {
            *counts.entry(n.to_string()).or_default() += 1;
        }
    }
    if let Some((s, _)) = counts.iter().max_by_key(|(_, n)| **n) {
        return Some(s.clone());
    }
    candidates.iter().filter(|s| !s.trim().is_empty()).max_by_key(|s| s.len()).cloned()
}

impl Reranker for HeuristicReranker {
    fn name(&self) -> &'static str {
        "Heuristic"
    }
    fn rerank_tags(&mut self, _query: &str, candidates: &[String]) -> Vec<String> {
        let mut counts: BTreeMap<String, i64> = BTreeMap::new();
        for c in candidates {
            let n = normalize_tag(c);
            if !n.is_empty() {
                *counts.entry(n).or_default() += 1;
            }
        }
        let mut v: Vec<(String, i64)> = counts.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        v.into_iter().map(|(s, _)| s).take(12).collect()
    }
    fn rerank_category(&mut self, _query: &str, candidates: &[String]) -> Option<String> {
        let mut counts: BTreeMap<String, i64> = BTreeMap::new();
        for c in candidates {
            let n = normalize_category(c);
            if !n.is_empty() {
                *counts.entry(n).or_default() += 1;
            }
        }
        counts.into_iter().max_by_key(|(_, n)| *n).map(|(s, _)| s)
    }
    fn rerank_caption(&mut self, _query: &str, candidates: &[String]) -> Option<String> {
        most_common_text(candidates)
    }
    fn rerank_description(&mut self, _query: &str, candidates: &[String]) -> Option<String> {
        most_common_text(candidates)
    }
}

pub(crate) fn normalize_tag(s: &str) -> String {
    s.trim().to_ascii_lowercase()
}

pub(crate) fn normalize_category(s: &str) -> String {
    s.trim().to_ascii_lowercase()
}

#[derive(Clone, Debug)]
pub struct Proposal {
    pub path: String,
    pub current_category: Option<String>,
    pub current_tags: Vec<String>,
    pub current_caption: Option<String>,
    pub current_description: Option<String>,
    pub new_category: Option<String>,
    pub new_tags: Vec<String>,
    pub new_caption: Option<String>,
    pub new_description: Option<String>,
    pub generator: String,
    /// Share of the neighbours that agree with the proposal, 0 to 1; none for generators that cannot say.
    pub confidence: Option<f32>,
}

impl Proposal {
    /// A proposal of nothing for `row`, to be filled with new values.
    pub fn for_row(row: &Thumbnail, generator: impl Into<String>) -> Self {
        Self {
            path: row.path.clone(),
            current_category: row.category.clone(),
            current_tags: row.tags.clone(),
            current_caption: row.caption.clone(),
            current_description: row.description.clone(),
            new_category: None,
            new_tags: Vec::new(),
            new_caption: None,
            new_description: None,
            generator: generator.into(),
            confidence: None,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.new_category.is_none() && self.new_tags.is_empty() && self.new_caption.is_none() && self.new_description.is_none()
    }
}

/// What the neighbours of a file carry.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Candidates {
    pub neighbours: usize,
    pub tags: Vec<String>,
    pub categories: Vec<String>,
    pub captions: Vec<String>,
    pub descriptions: Vec<String>,
}

impl Candidates {
    /// Collects the metadata of `rows`.
    pub fn from_rows<'a>(rows: impl IntoIterator<Item = &'a Thumbnail>) -> Self {
        let mut out = Self::default();
        for t in rows {
            out.neighbours += 1;
            out.tags.extend(t.tags.iter().cloned());
            out.categories.extend(t.category.iter().cloned());
            out.captions.extend(t.caption.iter().filter(|c| !c.trim().is_empty()).cloned());
            out.descriptions.extend(t.description.iter().filter(|d| !d.trim().is_empty()).cloned());
        }
        out
    }
}

/// Values a reranker proposes, after the ones the file already has are removed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Fields {
    pub category: Option<String>,
    pub tags: Vec<String>,
    pub caption: Option<String>,
    pub description: Option<String>,
}

impl Fields {
    pub fn is_empty(&self) -> bool {
        self.category.is_none() && self.tags.is_empty() && self.caption.is_none() && self.description.is_none()
    }
}

fn same_text(current: &Option<String>, new: &Option<String>) -> bool {
    matches!((current, new), (Some(c), Some(n)) if c.trim().eq_ignore_ascii_case(n.trim()))
}

fn lower_set(tags: &[String]) -> BTreeSet<String> {
    tags.iter().map(|s| s.to_ascii_lowercase()).collect()
}

/// The text a file's neighbours are searched with.
pub fn query_text(row: &Thumbnail) -> String {
    format!("{} {} {}", row.caption.clone().unwrap_or_default(), row.description.clone().unwrap_or_default(), row.tags.join(" "))
}

fn rerank_fields(reranker: &mut dyn Reranker, query: &str, cands: &Candidates, row: &Thumbnail) -> Fields {
    let mut fields = Fields {
        tags: reranker.rerank_tags(query, &cands.tags),
        category: reranker.rerank_category(query, &cands.categories),
        caption: reranker.rerank_caption(query, &cands.captions),
        description: reranker.rerank_description(query, &cands.descriptions),
    };
    if same_text(&row.category, &fields.category) {
        fields.category = None;
    }
    if same_text(&row.caption, &fields.caption) {
        fields.caption = None;
    }
    if same_text(&row.description, &fields.description) {
        fields.description = None;
    }
    if !fields.tags.is_empty() && lower_set(&row.tags) == lower_set(&fields.tags) {
        fields.tags.clear();
    }
    fields
}

fn share(values: &[String], chosen: &str, denominator: usize) -> f32 {
    if denominator == 0 {
        return 0.0;
    }
    let hits = values.iter().filter(|v| v.trim().eq_ignore_ascii_case(chosen.trim())).count();
    (hits as f32 / denominator as f32).clamp(0.0, 1.0)
}

/// Mean support among the neighbours for the proposed tags and category; caption and description count only when they are all that is proposed.
pub fn confidence(cands: &Candidates, fields: &Fields) -> Option<f32> {
    let mut votes: Vec<f32> = Vec::new();
    if let Some(c) = &fields.category {
        votes.push(share(&cands.categories, c, cands.categories.len()));
    }
    for t in &fields.tags {
        votes.push(share(&cands.tags, t, cands.neighbours));
    }
    if votes.is_empty() {
        if let Some(c) = &fields.caption {
            votes.push(share(&cands.captions, c, cands.captions.len()));
        }
        if let Some(d) = &fields.description {
            votes.push(share(&cands.descriptions, d, cands.descriptions.len()));
        }
    }
    (!votes.is_empty()).then(|| votes.iter().sum::<f32>() / votes.len() as f32)
}

/// The proposal for `row` from its neighbours, or none when the reranker and the fallback have nothing new to say.
pub fn build_proposal(row: &Thumbnail, cands: &Candidates, reranker: &mut dyn Reranker) -> Option<Proposal> {
    let query = query_text(row);
    let mut generator = reranker.name().to_string();
    let mut fields = rerank_fields(reranker, &query, cands, row);
    if fields.is_empty() {
        fields = rerank_fields(&mut HeuristicReranker, &query, cands, row);
        generator = "Heuristic (fallback)".to_string();
    }
    if fields.is_empty() {
        return None;
    }
    let mut proposal = Proposal::for_row(row, generator);
    proposal.confidence = confidence(cands, &fields);
    proposal.new_category = fields.category;
    proposal.new_tags = fields.tags;
    proposal.new_caption = fields.caption;
    proposal.new_description = fields.description;
    Some(proposal)
}

/// Which files a run reads.
#[derive(Clone, Copy, Debug)]
pub enum RowScope<'a> {
    /// These files only.
    Paths(&'a [String]),
    /// Up to this many files that have every metadata field.
    Rich(usize),
}

/// Stop and progress hooks of a run.
pub struct RunCtl<'a> {
    pub should_stop: &'a (dyn Fn() -> bool + Sync),
    pub progress: &'a (dyn Fn(u64, u64) + Sync),
}

pub struct ProposalGenerator {
    pub engine: &'static AISearchEngine,
}

fn has_metadata(row: &Thumbnail) -> bool {
    row.caption.as_deref().is_some_and(|c| !c.trim().is_empty()) || row.description.as_deref().is_some_and(|d| !d.trim().is_empty()) || !row.tags.is_empty() || row.category.is_some()
}

impl ProposalGenerator {
    pub fn new(engine: &'static AISearchEngine) -> Self {
        Self { engine }
    }

    /// The rows a run works on; with `Paths` the selected files that carry any metadata.
    pub async fn rows(&self, scope: RowScope<'_>) -> Vec<Thumbnail> {
        match scope {
            RowScope::Paths(paths) => match crate::database::collections::rows_for_paths(&crate::DB, paths).await {
                Ok(rows) => rows.into_iter().filter(has_metadata).collect(),
                Err(e) => {
                    log::warn!("[Refine] reading the selected rows failed: {e:#}");
                    Vec::new()
                }
            },
            RowScope::Rich(limit) => {
                let rows = self.engine.list_rich_thumbnail_rows(limit).await;
                if !rows.is_empty() {
                    return rows;
                }
                let files = self.engine.files.lock().await;
                files
                    .iter()
                    .filter(|f| {
                        let filled = |v: &Option<String>| v.as_ref().is_some_and(|s| !s.trim().is_empty());
                        filled(&f.description) && filled(&f.caption) && filled(&f.category) && !f.tags.is_empty()
                    })
                    .take(limit)
                    .cloned()
                    .collect()
            }
        }
    }

    /// Proposals for `rows`, one `on_emit` call per file that has something new; stops when `ctl` says so.
    pub async fn stream_rows(&self, rows: Vec<Thumbnail>, reranker: &mut dyn Reranker, ctl: &RunCtl<'_>, mut on_emit: impl FnMut(Proposal) + Send) {
        let total = rows.len() as u64;
        log::info!("[Refine] processing {total} rows");
        let mut skipped = 0usize;
        for (i, row) in rows.iter().enumerate() {
            if (ctl.should_stop)() {
                log::info!("[Refine] stopped after {i} of {total} rows");
                return;
            }
            (ctl.progress)(i as u64, total);
            let cands = self.neighbours(&query_text(row)).await;
            match build_proposal(row, &cands, reranker) {
                Some(p) => on_emit(p),
                None => skipped += 1,
            }
        }
        (ctl.progress)(total, total);
        log::info!("[Refine] done, {skipped} files had nothing new");
    }

    /// The metadata of the files closest to `text` by CLIP.
    async fn neighbours(&self, text: &str) -> Candidates {
        let Some(qv) = self.embed_text(text).await else { return Candidates::default() };
        match crate::database::ClipEmbeddingRow::find_similar_by_embedding(&qv, 64, 128, 0).await {
            Ok(hits) => Candidates::from_rows(hits.into_iter().take(NEIGHBOURS).filter_map(|h| h.thumb_ref).collect::<Vec<_>>().iter()),
            Err(e) => {
                log::debug!("[Refine] the neighbour search failed: {e:#}");
                Candidates::default()
            }
        }
    }

    async fn embed_text(&self, text: &str) -> Option<Vec<f32>> {
        let _ = self.engine.ensure_clip_engine().await.ok()?;
        let mut guard = self.engine.clip_engine.lock().await;
        guard.as_mut()?.embed_text(text).ok()
    }
}
