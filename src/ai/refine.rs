use super::ai_search::AISearchEngine;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RerankerType {
    Heuristic,
    JinaM0,
    QwenReranker,
}

impl Default for RerankerType {
    fn default() -> Self {
        RerankerType::Heuristic
    }
}

pub trait Reranker: Send + Sync {
    fn name(&self) -> &'static str;
    fn rerank_tags(&mut self, query: &str, candidates: &[String]) -> Vec<String>;
    fn rerank_category(&mut self, query: &str, candidates: &[String]) -> Option<String>;
    fn rerank_caption(&mut self, query: &str, candidates: &[String]) -> Option<String>;
    fn rerank_description(&mut self, query: &str, candidates: &[String]) -> Option<String>;
}

pub struct HeuristicReranker;

impl Reranker for HeuristicReranker {
    fn name(&self) -> &'static str {
        "Heuristic"
    }
    fn rerank_tags(&mut self, _query: &str, candidates: &[String]) -> Vec<String> {
        // Simple deterministic normalization and frequency-based sort
        let mut map: std::collections::BTreeMap<String, i64> = std::collections::BTreeMap::new();
        for c in candidates.iter() {
            let n = normalize_tag(c);
            if n.is_empty() {
                continue;
            }
            *map.entry(n).or_default() += 1;
        }
        let mut v: Vec<(String, i64)> = map.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        v.into_iter().map(|(s, _)| s).take(12).collect()
    }
    fn rerank_category(&mut self, _query: &str, candidates: &[String]) -> Option<String> {
        let mut map: std::collections::BTreeMap<String, i64> = std::collections::BTreeMap::new();
        for c in candidates.iter() {
            let n = normalize_category(c);
            if n.is_empty() {
                continue;
            }
            *map.entry(n).or_default() += 1;
        }
        map.into_iter().max_by_key(|(_, n)| *n).map(|(s, _)| s)
    }
    fn rerank_caption(&mut self, _query: &str, candidates: &[String]) -> Option<String> {
        // Pick the most frequent non-empty, else longest
        if candidates.is_empty() {
            return None;
        }
        let mut map: std::collections::BTreeMap<String, i64> = std::collections::BTreeMap::new();
        for c in candidates.iter() {
            let n = c.trim();
            if n.is_empty() {
                continue;
            }
            *map.entry(n.to_string()).or_default() += 1;
        }
        if let Some((s, _)) = map.iter().max_by_key(|(_, n)| **n) {
            return Some(s.clone());
        }
        candidates
            .iter()
            .filter(|s| !s.trim().is_empty())
            .max_by_key(|s| s.len())
            .cloned()
    }
    fn rerank_description(&mut self, _query: &str, candidates: &[String]) -> Option<String> {
        // Similar to caption
        if candidates.is_empty() {
            return None;
        }
        let mut map: std::collections::BTreeMap<String, i64> = std::collections::BTreeMap::new();
        for c in candidates.iter() {
            let n = c.trim();
            if n.is_empty() {
                continue;
            }
            *map.entry(n.to_string()).or_default() += 1;
        }
        if let Some((s, _)) = map.iter().max_by_key(|(_, n)| **n) {
            return Some(s.clone());
        }
        candidates
            .iter()
            .filter(|s| !s.trim().is_empty())
            .max_by_key(|s| s.len())
            .cloned()
    }
}

fn normalize_tag(s: &str) -> String {
    s.trim().to_ascii_lowercase()
}
fn normalize_category(s: &str) -> String {
    s.trim().to_ascii_lowercase()
}

pub struct ProposalGenerator {
    pub engine: &'static AISearchEngine,
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
}

impl ProposalGenerator {
    pub fn new(engine: &'static AISearchEngine) -> Self {
        Self { engine }
    }

    /// Streaming variant: emits proposals as they are produced.
    /// on_emit is called with each new Proposal. Caller can accumulate or forward to UI.
    pub async fn stream_proposals<R, F>(&self, limit: usize, reranker: &mut R, mut on_emit: F)
    where
        R: Reranker + ?Sized,
        F: FnMut(Proposal) + Send,
    {
        // Reuse the same source selection as batch version
        let mut rows = self.engine.list_rich_thumbnail_rows(limit).await;
        if rows.is_empty() {
            let files = self.engine.files.lock().await;
            rows = files
                .iter()
                .filter(|f| {
                    f.description
                        .as_ref()
                        .map(|s| !s.trim().is_empty())
                        .unwrap_or(false)
                        && f.caption
                            .as_ref()
                            .map(|s| !s.trim().is_empty())
                            .unwrap_or(false)
                        && f.category
                            .as_ref()
                            .map(|s| !s.trim().is_empty())
                            .unwrap_or(false)
                        && !f.tags.is_empty()
                })
                .take(limit)
                .cloned()
                .collect();
            drop(files);
        }
        let total = rows.len();
        log::info!("[Refine/Stream] Processing {} rows", total);
        let mut skipped_empty = 0usize;
        for row in rows.into_iter() {
            let query_text = format!(
                "{} {} {}",
                row.caption.clone().unwrap_or_default(),
                row.description.clone().unwrap_or_default(),
                row.tags.join(" ")
            );
            // Neighbor candidates via CLIP knn
            let mut tag_cands: Vec<String> = Vec::new();
            let mut cat_cands: Vec<String> = Vec::new();
            let mut cap_cands: Vec<String> = Vec::new();
            let mut desc_cands: Vec<String> = Vec::new();
            if let Some(qv) = self.embed_text(&query_text).await {
                if let Ok(neigh) =
                    crate::database::ClipEmbeddingRow::find_similar_by_embedding(&qv, 64, 128, 0).await
                {
                    for hit in neigh.into_iter().take(50) {
                        if let Some(t) = hit.thumb_ref {
                            tag_cands.extend(t.tags.clone());
                            if let Some(c) = t.category.clone() {
                                cat_cands.push(c);
                            }
                            if let Some(ca) = t.caption.clone() {
                                if !ca.trim().is_empty() {
                                    cap_cands.push(ca);
                                }
                            }
                            if let Some(ds) = t.description.clone() {
                                if !ds.trim().is_empty() {
                                    desc_cands.push(ds);
                                }
                            }
                        }
                    }
                }
            }
            // Rerank
            let mut new_tags = reranker.rerank_tags(&query_text, &tag_cands);
            let mut new_category = reranker.rerank_category(&query_text, &cat_cands);
            let mut new_caption = reranker.rerank_caption(&query_text, &cap_cands);
            let mut new_description = reranker.rerank_description(&query_text, &desc_cands);
            // Filter unchanged
            if let (Some(cur), Some(n)) = (&row.category, &new_category) {
                if cur.eq_ignore_ascii_case(n) {
                    new_category = None;
                }
            }
            if let (Some(cur), Some(n)) = (&row.caption, &new_caption) {
                if cur.trim().eq_ignore_ascii_case(n.trim()) {
                    new_caption = None;
                }
            }
            if let (Some(cur), Some(n)) = (&row.description, &new_description) {
                if cur.trim().eq_ignore_ascii_case(n.trim()) {
                    new_description = None;
                }
            }
            if !new_tags.is_empty() {
                let cur_set: std::collections::BTreeSet<String> =
                    row.tags.iter().map(|s| s.to_ascii_lowercase()).collect();
                let new_set: std::collections::BTreeSet<String> =
                    new_tags.iter().map(|s| s.to_ascii_lowercase()).collect();
                if cur_set == new_set {
                    new_tags.clear();
                }
            }
            // Fallback if empty
            let mut used_primary = false;
            used_primary |= new_category.is_some();
            used_primary |= !new_tags.is_empty();
            used_primary |= new_caption.is_some();
            used_primary |= new_description.is_some();
            if !used_primary {
                let mut h = HeuristicReranker;
                new_tags = h.rerank_tags(&query_text, &tag_cands);
                new_category = h.rerank_category(&query_text, &cat_cands);
                new_caption = h.rerank_caption(&query_text, &cap_cands);
                new_description = h.rerank_description(&query_text, &desc_cands);
                if let (Some(cur), Some(n)) = (&row.category, &new_category) {
                    if cur.eq_ignore_ascii_case(n) {
                        new_category = None;
                    }
                }
                if let (Some(cur), Some(n)) = (&row.caption, &new_caption) {
                    if cur.trim().eq_ignore_ascii_case(n.trim()) {
                        new_caption = None;
                    }
                }
                if let (Some(cur), Some(n)) = (&row.description, &new_description) {
                    if cur.trim().eq_ignore_ascii_case(n.trim()) {
                        new_description = None;
                    }
                }
                if !new_tags.is_empty() {
                    let cur_set: std::collections::BTreeSet<String> =
                        row.tags.iter().map(|s| s.to_ascii_lowercase()).collect();
                    let new_set: std::collections::BTreeSet<String> =
                        new_tags.iter().map(|s| s.to_ascii_lowercase()).collect();
                    if cur_set == new_set {
                        new_tags.clear();
                    }
                }
            }
            let has_any = new_category.is_some()
                || !new_tags.is_empty()
                || new_caption.is_some()
                || new_description.is_some();
            if has_any {
                let p = Proposal {
                    path: row.path.clone(),
                    current_category: row.category.clone(),
                    current_tags: row.tags.clone(),
                    current_caption: row.caption.clone(),
                    current_description: row.description.clone(),
                    new_category,
                    new_tags,
                    new_caption,
                    new_description,
                    generator: if used_primary {
                        reranker.name().to_string()
                    } else {
                        "Heuristic (fallback)".to_string()
                    },
                };
                on_emit(p);
            } else {
                skipped_empty += 1;
            }
        }
        log::info!("[Refine/Stream] done (skipped {} empty)", skipped_empty);
    }

    pub async fn generate_proposals<R: Reranker + ?Sized>(
        &self,
        limit: usize,
        reranker: &mut R,
    ) -> Vec<Proposal> {
        // Snapshot from DB: load only rows with rich metadata (desc, caption, category, tags)
        let mut rows = self.engine.list_rich_thumbnail_rows(limit).await;
        if rows.is_empty() {
            // Fallback: use in-memory files snapshot if DB returned nothing (e.g., early startup)
            let files = self.engine.files.lock().await;
            rows = files
                .iter()
                .filter(|f| {
                    f.description
                        .as_ref()
                        .map(|s| !s.trim().is_empty())
                        .unwrap_or(false)
                        && f.caption
                            .as_ref()
                            .map(|s| !s.trim().is_empty())
                            .unwrap_or(false)
                        && f.category
                            .as_ref()
                            .map(|s| !s.trim().is_empty())
                            .unwrap_or(false)
                        && !f.tags.is_empty()
                })
                .take(limit)
                .cloned()
                .collect();
            drop(files);
            if rows.is_empty() {
                log::warn!(
                    "[Refine] No rows available from DB or in-memory cache; proposals will be empty"
                );
            } else {
                log::info!(
                    "[Refine] Using in-memory cache with {} rows (limit={})",
                    rows.len(),
                    limit
                );
            }
        }
        log::info!(
            "[Refine] Fetched {} rows for proposal generation (limit={})",
            rows.len(),
            limit
        );
        let mut out: Vec<Proposal> = Vec::new();
        let mut skipped_empty: usize = 0;
        for row in rows.into_iter() {
            // Build a text query from current metadata
            let query_text = format!(
                "{} {} {}",
                row.caption.clone().unwrap_or_default(),
                row.description.clone().unwrap_or_default(),
                row.tags.join(" ")
            );
            log::warn!(
                "[Refine] Row: {} (tags={}, has_cat={})",
                row.path,
                row.tags.len(),
                row.category.is_some()
            );
            // Collect neighbor tags/categories for candidates (best-effort)
            let mut tag_cands: Vec<String> = Vec::new();
            let mut cat_cands: Vec<String> = Vec::new();
            let mut cap_cands: Vec<String> = Vec::new();
            let mut desc_cands: Vec<String> = Vec::new();
            if let Some(qv) = self.embed_text(&query_text).await {
                if let Ok(neigh) =
                    crate::database::ClipEmbeddingRow::find_similar_by_embedding(&qv, 64, 128, 0).await
                {
                    for hit in neigh.into_iter().take(50) {
                        if let Some(t) = hit.thumb_ref {
                            tag_cands.extend(t.tags.clone());
                            if let Some(c) = t.category.clone() {
                                cat_cands.push(c);
                            }
                            if let Some(ca) = t.caption.clone() {
                                if !ca.trim().is_empty() {
                                    cap_cands.push(ca);
                                }
                            }
                            if let Some(ds) = t.description.clone() {
                                if !ds.trim().is_empty() {
                                    desc_cands.push(ds);
                                }
                            }
                        }
                    }
                }
            }
            log::warn!(
                "[Refine] Candidates: tags={}, cats={}",
                tag_cands.len(),
                cat_cands.len()
            );

            // Primary reranker
            let mut new_tags = reranker.rerank_tags(&query_text, &tag_cands);
            let mut new_category = reranker.rerank_category(&query_text, &cat_cands);
            let mut new_caption = reranker.rerank_caption(&query_text, &cap_cands);
            let mut new_description = reranker.rerank_description(&query_text, &desc_cands);

            // Remove unchanged values (case-insensitive compare for strings; set compare for tags)
            if let (Some(cur), Some(n)) = (&row.category, &new_category) {
                if cur.eq_ignore_ascii_case(n) {
                    new_category = None;
                }
            }
            if let (Some(cur), Some(n)) = (&row.caption, &new_caption) {
                if cur.trim().eq_ignore_ascii_case(n.trim()) {
                    new_caption = None;
                }
            }
            if let (Some(cur), Some(n)) = (&row.description, &new_description) {
                if cur.trim().eq_ignore_ascii_case(n.trim()) {
                    new_description = None;
                }
            }
            if !new_tags.is_empty() {
                let cur_set: std::collections::BTreeSet<String> =
                    row.tags.iter().map(|s| s.to_ascii_lowercase()).collect();
                let new_set: std::collections::BTreeSet<String> =
                    new_tags.iter().map(|s| s.to_ascii_lowercase()).collect();
                if cur_set == new_set {
                    new_tags.clear();
                }
            }

            // If nothing proposed by primary reranker, fill with heuristic per-field
            let mut used_primary = false;
            used_primary |= new_category.is_some();
            used_primary |= !new_tags.is_empty();
            used_primary |= new_caption.is_some();
            used_primary |= new_description.is_some();

            if !used_primary {
                let mut h = HeuristicReranker;
                new_tags = h.rerank_tags(&query_text, &tag_cands);
                new_category = h.rerank_category(&query_text, &cat_cands);
                new_caption = h.rerank_caption(&query_text, &cap_cands);
                new_description = h.rerank_description(&query_text, &desc_cands);
                // Repeat unchanged filtering
                if let (Some(cur), Some(n)) = (&row.category, &new_category) {
                    if cur.eq_ignore_ascii_case(n) {
                        new_category = None;
                    }
                }
                if let (Some(cur), Some(n)) = (&row.caption, &new_caption) {
                    if cur.trim().eq_ignore_ascii_case(n.trim()) {
                        new_caption = None;
                    }
                }
                if let (Some(cur), Some(n)) = (&row.description, &new_description) {
                    if cur.trim().eq_ignore_ascii_case(n.trim()) {
                        new_description = None;
                    }
                }
                if !new_tags.is_empty() {
                    let cur_set: std::collections::BTreeSet<String> =
                        row.tags.iter().map(|s| s.to_ascii_lowercase()).collect();
                    let new_set: std::collections::BTreeSet<String> =
                        new_tags.iter().map(|s| s.to_ascii_lowercase()).collect();
                    if cur_set == new_set {
                        new_tags.clear();
                    }
                }
            }

            let has_any = new_category.is_some()
                || !new_tags.is_empty()
                || new_caption.is_some()
                || new_description.is_some();
            log::debug!(
                "[Refine] Reranked: new_tags={}, has_new_cat={}",
                new_tags.len(),
                new_category.is_some()
            );
            if has_any {
                out.push(Proposal {
                    path: row.path.clone(),
                    current_category: row.category.clone(),
                    current_tags: row.tags.clone(),
                    current_caption: row.caption.clone(),
                    current_description: row.description.clone(),
                    new_category,
                    new_tags,
                    new_caption,
                    new_description,
                    generator: if used_primary {
                        reranker.name().to_string()
                    } else {
                        "Heuristic (fallback)".to_string()
                    },
                });
            } else {
                skipped_empty += 1;
            }
        }
        if skipped_empty > 0 {
            log::info!(
                "[Refine] Generated {} proposals (skipped {} empty)",
                out.len(),
                skipped_empty
            );
        } else {
            log::info!("[Refine] Generated {} proposals", out.len());
        }
        out
    }

    async fn embed_text(&self, text: &str) -> Option<Vec<f32>> {
        let _ = self.engine.ensure_clip_engine().await.ok()?;
        let mut guard = self.engine.clip_engine.lock().await;
        guard.as_mut()?.embed_text(text).ok()
    }
}

// --- Jina M0 (text-only) reranker using fastembed as a stand-in forward pass ---
pub struct JinaM0TextReranker {
    // Reuse fastembed text encoder to score query vs. candidate tokens
    embedder: fastembed::TextEmbedding,
}

impl JinaM0TextReranker {
    pub fn new() -> anyhow::Result<Self> {
        let txt = fastembed::TextEmbedding::try_new(fastembed::TextInitOptions::new(
            fastembed::EmbeddingModel::ClipVitB32,
        ))?;
        Ok(Self { embedder: txt })
    }
    fn embed(&mut self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        let embs = self.embedder.embed(texts.to_vec(), None)?;
        Ok(embs)
    }
}

impl Reranker for JinaM0TextReranker {
    fn name(&self) -> &'static str {
        "JinaM0 (stand-in)"
    }
    fn rerank_tags(&mut self, query: &str, candidates: &[String]) -> Vec<String> {
        if candidates.is_empty() {
            return Vec::new();
        }
        // Dedup normalized candidates
        let mut norm: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for c in candidates {
            let n = normalize_tag(c);
            if !n.is_empty() {
                norm.insert(n);
            }
        }
        let cands: Vec<String> = norm.into_iter().collect();
        let mut all = vec![query.to_string()];
        all.extend(cands.iter().cloned());
        let Ok(embs) = self.embed(&all) else {
            return cands.into_iter().take(12).collect();
        };
        if embs.is_empty() {
            return cands.into_iter().take(12).collect();
        }
        let (q, rest) = embs.split_first().unwrap();
        let mut scored: Vec<(String, f32)> = rest
            .iter()
            .zip(cands.iter())
            .map(|(e, s)| (s.clone(), crate::ai::clip::cosine_similarity(q, e)))
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.into_iter().map(|(s, _)| s).take(12).collect()
    }
    fn rerank_category(&mut self, query: &str, candidates: &[String]) -> Option<String> {
        if candidates.is_empty() {
            return None;
        }
        let mut norm: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for c in candidates {
            let n = normalize_category(c);
            if !n.is_empty() {
                norm.insert(n);
            }
        }
        let cands: Vec<String> = norm.into_iter().collect();
        let mut all = vec![query.to_string()];
        all.extend(cands.iter().cloned());
        let Ok(embs) = self.embed(&all) else {
            return cands.into_iter().next();
        };
        let (q, rest) = embs.split_first().unwrap();
        rest.iter()
            .zip(cands.iter())
            .map(|(e, s)| (s.clone(), crate::ai::clip::cosine_similarity(q, e)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(s, _)| s)
    }
    fn rerank_caption(&mut self, query: &str, candidates: &[String]) -> Option<String> {
        if candidates.is_empty() {
            return None;
        }
        let cands: Vec<String> = candidates
            .iter()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if cands.is_empty() {
            return None;
        }
        let mut all = vec![query.to_string()];
        all.extend(cands.iter().cloned());
        let Ok(embs) = self.embed(&all) else {
            return cands.into_iter().next();
        };
        let (q, rest) = embs.split_first().unwrap();
        rest.iter()
            .zip(cands.iter())
            .map(|(e, s)| (s.clone(), crate::ai::clip::cosine_similarity(q, e)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(s, _)| s)
    }
    fn rerank_description(&mut self, query: &str, candidates: &[String]) -> Option<String> {
        // Same approach as caption
        self.rerank_caption(query, candidates)
    }
}
