//! The data of the Refine tab without any drawing: proposals with per-field choices, the filtered and sorted list, and the plan of a bulk accept.

use crate::ai::mcp::proposals::{Applied, FieldMask, Proposal as StoredProposal};
use crate::ai::refine::Proposal as Generated;
use crate::database::metadata_edit::MetaPatch;

/// A field a proposal can change, in display order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FieldKind {
    Category,
    Tags,
    Caption,
    Description,
}

impl FieldKind {
    pub const ALL: [FieldKind; 4] = [Self::Category, Self::Tags, Self::Caption, Self::Description];

    pub fn label(self) -> &'static str {
        match self {
            Self::Category => "Category",
            Self::Tags => "Tags",
            Self::Caption => "Caption",
            Self::Description => "Description",
        }
    }

    fn index(self) -> usize {
        Self::ALL.iter().position(|k| *k == self).unwrap_or(0)
    }
}

/// The value of a field on the file or in the proposal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FieldValue {
    Text(Option<String>),
    Tags(Vec<String>),
}

impl FieldValue {
    /// Text for lists and filters; empty values read as none.
    pub fn plain(&self) -> String {
        match self {
            Self::Text(Some(t)) if !t.trim().is_empty() => t.clone(),
            Self::Tags(t) if !t.is_empty() => t.join(", "),
            _ => "(none)".to_string(),
        }
    }
}

/// One field of a proposal with the user's choice.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldChange {
    pub kind: FieldKind,
    pub current: FieldValue,
    pub proposed: FieldValue,
    pub accepted: bool,
}

/// Where a proposal came from. Other generators, such as an agent added later, use `Other`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProposalSource {
    Heuristic,
    Reranker(String),
    Cloud(String),
    /// Stored by a tool client of the MCP server.
    Mcp(String),
    Other(String),
}

impl ProposalSource {
    /// Name shown on the card.
    pub fn label(&self) -> String {
        match self {
            Self::Heuristic => "Heuristic".to_string(),
            Self::Reranker(name) | Self::Other(name) => name.clone(),
            Self::Cloud(provider) => format!("Cloud {provider}"),
            Self::Mcp(client) if client.trim().is_empty() => "MCP".to_string(),
            Self::Mcp(client) => format!("MCP {client}"),
        }
    }

    pub fn is_mcp(&self) -> bool {
        matches!(self, Self::Mcp(_))
    }

    /// The source of a generator name such as "Heuristic (fallback)", "Jina M0" or "cloud:openrouter".
    pub fn from_generator(generator: &str) -> Self {
        if let Some(provider) = generator.strip_prefix("cloud:") {
            Self::Cloud(provider.to_string())
        } else if generator.starts_with("Heuristic") {
            Self::Heuristic
        } else {
            Self::Reranker(generator.to_string())
        }
    }
}

/// Identity of a proposal: a number in this session or the id in the proposal store.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ProposalKey {
    Local(u64),
    Mcp(String),
}

/// A proposed change of one file with its fields.
#[derive(Clone, Debug, PartialEq)]
pub struct Proposal {
    pub key: ProposalKey,
    pub path: String,
    pub source: ProposalSource,
    /// Share of the neighbours that agree, 0 to 1.
    pub confidence: Option<f32>,
    pub reason: Option<String>,
    pub created_ms: u64,
    pub changes: Vec<FieldChange>,
    /// Why the last attempt to apply it failed.
    pub error: Option<String>,
}

fn text_change(kind: FieldKind, current: &Option<String>, proposed: &Option<String>) -> Option<FieldChange> {
    let proposed = proposed.as_ref().map(|p| p.trim()).filter(|p| !p.is_empty())?;
    if current.as_deref().map(str::trim) == Some(proposed) {
        return None;
    }
    Some(FieldChange { kind, current: FieldValue::Text(current.clone()), proposed: FieldValue::Text(Some(proposed.to_string())), accepted: true })
}

fn tags_change(current: &[String], proposed: &[String]) -> Option<FieldChange> {
    if proposed.is_empty() || proposed == current {
        return None;
    }
    Some(FieldChange { kind: FieldKind::Tags, current: FieldValue::Tags(current.to_vec()), proposed: FieldValue::Tags(proposed.to_vec()), accepted: true })
}

/// The values a file has now, for the proposals of the store.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Current {
    pub category: Option<String>,
    pub tags: Vec<String>,
    pub caption: Option<String>,
    pub description: Option<String>,
}

impl Proposal {
    /// The proposal a generator made; none when it changes nothing.
    pub fn from_generated(key: ProposalKey, p: &Generated, created_ms: u64) -> Option<Self> {
        let changes: Vec<FieldChange> = [
            text_change(FieldKind::Category, &p.current_category, &p.new_category),
            tags_change(&p.current_tags, &p.new_tags),
            text_change(FieldKind::Caption, &p.current_caption, &p.new_caption),
            text_change(FieldKind::Description, &p.current_description, &p.new_description),
        ]
        .into_iter()
        .flatten()
        .collect();
        (!changes.is_empty()).then(|| Self {
            key,
            path: p.path.clone(),
            source: ProposalSource::from_generator(&p.generator),
            confidence: p.confidence,
            reason: None,
            created_ms,
            changes,
            error: None,
        })
    }

    /// The proposal of the store as it would apply to `current`; fields that already match are left out.
    pub fn from_stored(stored: &StoredProposal, current: &Current) -> Self {
        let existing = crate::ai::mcp::proposals::Existing { caption: current.caption.clone(), description: current.description.clone(), category: current.category.clone(), tags: current.tags.clone() };
        let applied: Applied = crate::ai::mcp::proposals::apply_changes(&existing, &stored.changes);
        let changes: Vec<FieldChange> = [
            text_change(FieldKind::Category, &current.category, &applied.category),
            applied.tags.as_ref().and_then(|t| tags_change(&current.tags, t)),
            text_change(FieldKind::Caption, &current.caption, &applied.caption),
            text_change(FieldKind::Description, &current.description, &applied.description),
        ]
        .into_iter()
        .flatten()
        .collect();
        Self {
            key: ProposalKey::Mcp(stored.id.clone()),
            path: stored.path.clone(),
            source: ProposalSource::Mcp(stored.source.clone()),
            confidence: None,
            reason: stored.reason.clone(),
            created_ms: stored.created_ms,
            changes,
            error: None,
        }
    }

    pub fn change(&self, kind: FieldKind) -> Option<&FieldChange> {
        self.changes.iter().find(|c| c.kind == kind)
    }

    pub fn accepted_count(&self) -> usize {
        self.changes.iter().filter(|c| c.accepted).count()
    }

    pub fn file_name(&self) -> &str {
        self.path.rsplit(['/', '\\']).next().unwrap_or(&self.path)
    }

    /// The fields that are checked.
    pub fn mask(&self) -> FieldMask {
        let on = |kind| self.change(kind).is_some_and(|c| c.accepted);
        FieldMask { caption: on(FieldKind::Caption), description: on(FieldKind::Description), category: on(FieldKind::Category), tags: on(FieldKind::Tags) }
    }

    /// The patch that writes the checked fields to the file, for proposals that carry their own values.
    pub fn patch(&self) -> MetaPatch {
        let mut patch = MetaPatch::default();
        for c in self.changes.iter().filter(|c| c.accepted) {
            match (c.kind, &c.proposed) {
                (FieldKind::Category, FieldValue::Text(v)) => patch.category = Some(v.clone()),
                (FieldKind::Caption, FieldValue::Text(v)) => patch.caption = Some(v.clone()),
                (FieldKind::Description, FieldValue::Text(v)) => patch.description = Some(v.clone()),
                (FieldKind::Tags, FieldValue::Tags(v)) => patch.tags = Some(v.clone()),
                _ => {}
            }
        }
        patch
    }

    /// Everything a text filter can match: name, path, source, reason and the current and proposed values.
    fn haystack(&self) -> String {
        let mut text = format!("{} {} {}", self.path, self.source.label(), self.reason.as_deref().unwrap_or(""));
        for c in &self.changes {
            text.push(' ');
            text.push_str(&c.current.plain());
            text.push(' ');
            text.push_str(&c.proposed.plain());
        }
        text.to_lowercase()
    }
}

/// Order of the list.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SortKey {
    #[default]
    Path,
    /// Most confident first; proposals without a confidence last.
    Confidence,
    Source,
    /// Most recently made first.
    Newest,
}

impl SortKey {
    pub const ALL: [SortKey; 4] = [Self::Path, Self::Confidence, Self::Source, Self::Newest];

    pub fn label(self) -> &'static str {
        match self {
            Self::Path => "Path",
            Self::Confidence => "Confidence",
            Self::Source => "Source",
            Self::Newest => "Newest",
        }
    }
}

/// What the list shows.
#[derive(Clone, Debug, PartialEq)]
pub struct ListFilter {
    pub text: String,
    /// The label of one source, or all.
    pub source: Option<String>,
    /// Proposals less confident than this are hidden, 0 to 1.
    pub min_confidence: f32,
    /// Whether proposals without a confidence pass when a minimum is set.
    pub include_unrated: bool,
    pub sort: SortKey,
}

impl Default for ListFilter {
    fn default() -> Self {
        Self { text: String::new(), source: None, min_confidence: 0.0, include_unrated: true, sort: SortKey::Path }
    }
}

impl ListFilter {
    /// True when anything narrows the list.
    pub fn is_narrowing(&self) -> bool {
        !self.text.trim().is_empty() || self.source.is_some() || self.min_confidence > 0.0
    }

    pub fn matches(&self, p: &Proposal) -> bool {
        if let Some(source) = &self.source {
            if p.source.label() != *source {
                return false;
            }
        }
        if self.min_confidence > 0.0 {
            match p.confidence {
                Some(c) if c + 1e-6 >= self.min_confidence => {}
                Some(_) => return false,
                None if self.include_unrated => {}
                None => return false,
            }
        }
        let text = self.text.trim().to_lowercase();
        text.is_empty() || p.haystack().contains(&text)
    }
}

/// Indices of the proposals the filter shows, in the order of its sort.
pub fn visible(proposals: &[Proposal], filter: &ListFilter) -> Vec<usize> {
    let mut shown: Vec<usize> = proposals.iter().enumerate().filter(|(_, p)| filter.matches(p)).map(|(i, _)| i).collect();
    match filter.sort {
        SortKey::Path => shown.sort_by(|a, b| proposals[*a].path.cmp(&proposals[*b].path)),
        SortKey::Confidence => shown.sort_by(|a, b| {
            let (ca, cb) = (proposals[*a].confidence, proposals[*b].confidence);
            cb.partial_cmp(&ca).unwrap_or(std::cmp::Ordering::Equal).then_with(|| proposals[*a].path.cmp(&proposals[*b].path))
        }),
        SortKey::Source => shown.sort_by(|a, b| proposals[*a].source.label().cmp(&proposals[*b].source.label()).then_with(|| proposals[*a].path.cmp(&proposals[*b].path))),
        SortKey::Newest => shown.sort_by(|a, b| proposals[*b].created_ms.cmp(&proposals[*a].created_ms).then_with(|| proposals[*a].path.cmp(&proposals[*b].path))),
    }
    shown
}

/// The distinct source labels of the proposals, in order of first appearance.
pub fn source_labels(proposals: &[Proposal]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for p in proposals {
        let label = p.source.label();
        if !out.contains(&label) {
            out.push(label);
        }
    }
    out
}

/// Checks or unchecks every field of the proposals at `indices`.
pub fn set_all(proposals: &mut [Proposal], indices: &[usize], accepted: bool) {
    for &i in indices {
        if let Some(p) = proposals.get_mut(i) {
            p.changes.iter_mut().for_each(|c| c.accepted = accepted);
        }
    }
}

/// One proposal in a plan with the fields that are checked.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanItem {
    pub key: ProposalKey,
    pub path: String,
    pub mask: FieldMask,
    /// The values to write for proposals of this session; the store applies its own.
    pub patch: Option<MetaPatch>,
}

/// What a bulk accept would do: only checked fields of the proposals given.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BulkPlan {
    pub items: Vec<PlanItem>,
    pub fields: usize,
    /// Fields per kind, in the order of [`FieldKind::ALL`].
    pub by_kind: [usize; 4],
    /// Proposals with no checked field, which a bulk accept leaves alone.
    pub unchecked: usize,
}

impl BulkPlan {
    pub fn files(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// "12 fields in 7 files (category 2, tags 7, caption 3)".
    pub fn summary(&self) -> String {
        let parts: Vec<String> = FieldKind::ALL.iter().filter(|k| self.by_kind[k.index()] > 0).map(|k| format!("{} {}", k.label().to_lowercase(), self.by_kind[k.index()])).collect();
        let files = self.files();
        format!("{} field{} in {} file{} ({})", self.fields, if self.fields == 1 { "" } else { "s" }, files, if files == 1 { "" } else { "s" }, parts.join(", "))
    }
}

/// The plan for the proposals at `indices`; a proposal with nothing checked is not part of it.
pub fn plan_accept(proposals: &[Proposal], indices: &[usize]) -> BulkPlan {
    let mut plan = BulkPlan::default();
    for &i in indices {
        let Some(p) = proposals.get(i) else { continue };
        let checked = p.accepted_count();
        if checked == 0 {
            plan.unchecked += 1;
            continue;
        }
        for c in p.changes.iter().filter(|c| c.accepted) {
            plan.by_kind[c.kind.index()] += 1;
        }
        plan.fields += checked;
        plan.items.push(PlanItem { key: p.key.clone(), path: p.path.clone(), mask: p.mask(), patch: matches!(p.key, ProposalKey::Local(_)).then(|| p.patch()) });
    }
    plan
}

/// How a tag differs between the current and the proposed list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TagMark {
    Kept,
    Added,
    Removed,
}

/// The current tags marked kept or removed, and the proposed tags marked kept or added; tags match ignoring case.
pub fn tag_diff(current: &[String], proposed: &[String]) -> (Vec<(String, TagMark)>, Vec<(String, TagMark)>) {
    let has = |list: &[String], tag: &str| list.iter().any(|t| t.trim().eq_ignore_ascii_case(tag.trim()));
    let now = current.iter().map(|t| (t.clone(), if has(proposed, t) { TagMark::Kept } else { TagMark::Removed })).collect();
    let next = proposed.iter().map(|t| (t.clone(), if has(current, t) { TagMark::Kept } else { TagMark::Added })).collect();
    (now, next)
}

/// Text cut to `max` characters with an ellipsis.
pub fn clipped(text: &str, max: usize) -> String {
    if text.chars().count() <= max { text.to_string() } else { format!("{}...", text.chars().take(max).collect::<String>().trim_end()) }
}
