//! The engine status registry: model loads, CLIP runs, refinements, image edits, assistant replies and database work.

use std::time::Duration;

use super::{Report, SourceCtx, TaskSource};
use crate::status::{StatusMeta, StatusState};
use crate::tasks::model::{AttentionItem, AttentionKind, CancelHandle, Observation, TaskKind, TaskSpec};

pub const SOURCE: &str = "engines";
/// Key of the observation made for the database engine.
pub const DB_KEY: &str = "engine:DB";
/// Age below which database work shows no chip.
const DB_VISIBLE_AFTER: Duration = Duration::from_millis(800);

type Snapshots = Box<dyn FnMut() -> Vec<StatusMeta> + Send>;

pub struct EngineSource {
    read: Snapshots,
}

impl EngineSource {
    pub fn new(read: Snapshots) -> Self {
        Self { read }
    }

    /// Reads the global status registry.
    pub fn live() -> Self {
        Self::new(Box::new(crate::status::engine_snapshots))
    }
}

fn is_loading(meta: &StatusMeta) -> bool {
    meta.state == StatusState::Initializing
}

fn is_running(meta: &StatusMeta) -> bool {
    meta.state == StatusState::Running
}

fn model_name(meta: &StatusMeta) -> &'static str {
    match meta.name {
        "VISION" => "vision model",
        "CLIP" => "CLIP model",
        "RERANK" => "reranker",
        "QWEN-EDIT" => "image edit model",
        other => other,
    }
}

/// Task kind and title for the active state of `meta`, if the engine is doing something worth listing.
fn active_task(meta: &StatusMeta) -> Option<(TaskKind, String)> {
    let loading = is_loading(meta);
    let running = is_running(meta);
    if !loading && !running {
        return None;
    }
    match meta.name {
        "DB" => Some((TaskKind::Database, "Database".to_string())),
        "VISION" if loading => Some((TaskKind::ModelLoad, format!("Loading {}", model_name(meta)))),
        "CLIP" | "RERANK" | "QWEN-EDIT" if loading => Some((TaskKind::ModelLoad, format!("Loading {}", model_name(meta)))),
        "CLIP" => Some((TaskKind::Clip, "CLIP embeddings".to_string())),
        "RERANK" => Some((TaskKind::Refine, "Refine proposals".to_string())),
        "QWEN-EDIT" => Some((TaskKind::ImageEdit, "Image edit".to_string())),
        "ASSIST" => Some((TaskKind::Assistant, "Assistant reply".to_string())),
        _ => None,
    }
}

fn is_model_engine(name: &str) -> bool {
    matches!(name, "VISION" | "CLIP" | "RERANK" | "QWEN-EDIT")
}

fn failure_kind(meta: &StatusMeta) -> TaskKind {
    match meta.name {
        "DB" => TaskKind::Database,
        "ASSIST" => TaskKind::Assistant,
        _ => TaskKind::ModelLoad,
    }
}

/// Observations and attention items for the engines.
pub fn observe(metas: &[StatusMeta], image_edit_cancel: Option<&CancelHandle>) -> (Vec<Observation>, Vec<AttentionItem>) {
    let mut observations = Vec::new();
    let mut attention = Vec::new();
    for meta in metas {
        let key = format!("engine:{}", meta.name);
        if meta.state == StatusState::Error {
            let error = meta.error.clone().unwrap_or_else(|| meta.detail.clone());
            let error = if error.is_empty() { "unknown error".to_string() } else { error };
            let title = if is_model_engine(meta.name) { format!("Loading {}", model_name(meta)) } else { meta.name.to_string() };
            observations.push(Observation::new(key.clone(), TaskSpec::new(failure_kind(meta), title)).failed(error.clone()));
            if is_model_engine(meta.name) {
                attention.push(AttentionItem {
                    key,
                    kind: AttentionKind::ModelError,
                    title: format!("{} failed", capitalised(model_name(meta))),
                    detail: error,
                    count: None,
                    action: None,
                });
            }
            continue;
        }
        let Some((kind, title)) = active_task(meta) else { continue };
        let mut spec = TaskSpec::new(kind, title).total(meta.progress_total).detail(meta.detail.clone());
        if kind == TaskKind::Database {
            spec = spec.visible_after(DB_VISIBLE_AFTER);
        }
        if kind == TaskKind::ImageEdit {
            if let Some(cancel) = image_edit_cancel {
                spec = spec.cancel(cancel.clone());
            }
        }
        observations.push(Observation::new(key, spec).done(meta.progress_current));
    }
    (observations, attention)
}

fn capitalised(s: &str) -> String {
    let mut chars = s.chars();
    chars.next().map(|c| c.to_uppercase().collect::<String>() + chars.as_str()).unwrap_or_default()
}

impl TaskSource for EngineSource {
    fn name(&self) -> &'static str {
        SOURCE
    }

    fn poll(&mut self, cx: &SourceCtx<'_>, out: &mut Report) {
        let metas = (self.read)();
        let (observations, attention) = observe(&metas, cx.app.image_edit_cancel.as_ref());
        out.observations = observations;
        out.attention = attention;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn meta(name: &'static str, state: StatusState) -> StatusMeta {
        StatusMeta { name, state, ..Default::default() }
    }

    fn keys(obs: &[Observation]) -> Vec<(&str, TaskKind)> {
        obs.iter().map(|o| (o.key.as_str(), o.spec.kind)).collect()
    }

    #[test]
    fn idle_and_waiting_engines_make_no_task() {
        let metas = vec![meta("DB", StatusState::Idle), meta("CLIP", StatusState::Waiting), meta("VISION", StatusState::Idle)];
        let (obs, attention) = observe(&metas, None);
        assert!(obs.is_empty() && attention.is_empty());
    }

    #[test]
    fn a_loading_model_is_a_model_load_and_a_running_one_its_own_kind() {
        let metas = vec![
            meta("VISION", StatusState::Initializing),
            meta("CLIP", StatusState::Running),
            meta("RERANK", StatusState::Running),
            meta("QWEN-EDIT", StatusState::Initializing),
            meta("ASSIST", StatusState::Running),
        ];
        let (obs, _) = observe(&metas, None);
        assert_eq!(
            keys(&obs),
            vec![
                ("engine:VISION", TaskKind::ModelLoad),
                ("engine:CLIP", TaskKind::Clip),
                ("engine:RERANK", TaskKind::Refine),
                ("engine:QWEN-EDIT", TaskKind::ModelLoad),
                ("engine:ASSIST", TaskKind::Assistant),
            ]
        );
        assert_eq!(obs[0].spec.title, "Loading vision model");
        assert_eq!(obs[3].spec.title, "Loading image edit model");
    }

    #[test]
    fn a_running_vision_engine_is_left_to_the_describe_queue() {
        let (obs, _) = observe(&[meta("VISION", StatusState::Running)], None);
        assert!(obs.is_empty());
    }

    #[test]
    fn progress_and_detail_come_from_the_engine() {
        let mut clip = meta("CLIP", StatusState::Running);
        clip.progress_current = 5;
        clip.progress_total = 20;
        clip.detail = "Generating missing".into();
        let (obs, _) = observe(&[clip], None);
        assert_eq!((obs[0].done, obs[0].spec.total, obs[0].spec.detail.as_str()), (5, 20, "Generating missing"));
    }

    #[test]
    fn database_work_is_hidden_until_it_lasts() {
        let (obs, _) = observe(&[meta("DB", StatusState::Running)], None);
        assert_eq!(obs[0].spec.kind, TaskKind::Database);
        assert_eq!(obs[0].spec.visible_after, DB_VISIBLE_AFTER);
        assert_eq!(obs[0].key, DB_KEY);
    }

    #[test]
    fn only_the_image_edit_gets_the_edit_cancel_handle() {
        let flag = Arc::new(AtomicBool::new(false));
        let cancel = CancelHandle::from_flag(flag.clone());
        let (obs, _) = observe(&[meta("QWEN-EDIT", StatusState::Running), meta("CLIP", StatusState::Running)], Some(&cancel));
        obs[0].spec.cancel.as_ref().expect("edit can be cancelled").call();
        assert!(flag.load(Ordering::Relaxed));
        assert!(obs[1].spec.cancel.is_none());
        let (obs, _) = observe(&[meta("QWEN-EDIT", StatusState::Running)], None);
        assert!(obs[0].spec.cancel.is_none());
    }

    #[test]
    fn a_failed_model_is_a_failed_observation_and_needs_attention() {
        let mut clip = meta("CLIP", StatusState::Error);
        clip.error = Some("out of memory".into());
        let (obs, attention) = observe(&[clip], None);
        assert_eq!(obs[0].failed.as_deref(), Some("out of memory"));
        assert_eq!(obs[0].spec.title, "Loading CLIP model");
        assert_eq!(attention.len(), 1);
        assert_eq!((attention[0].kind, attention[0].title.as_str(), attention[0].detail.as_str()), (AttentionKind::ModelError, "CLIP model failed", "out of memory"));
    }

    #[test]
    fn a_failed_database_is_recorded_but_is_not_a_model_error() {
        let mut db = meta("DB", StatusState::Error);
        db.error = Some("export failed".into());
        let (obs, attention) = observe(&[db], None);
        assert_eq!(obs[0].failed.as_deref(), Some("export failed"));
        assert!(attention.is_empty());
    }

    #[test]
    fn an_error_without_text_still_has_a_message() {
        let (obs, _) = observe(&[meta("RERANK", StatusState::Error)], None);
        assert_eq!(obs[0].failed.as_deref(), Some("unknown error"));
    }

    #[test]
    fn unknown_engines_are_ignored() {
        let (obs, _) = observe(&[meta("MYSTERY", StatusState::Running)], None);
        assert!(obs.is_empty());
    }
}
