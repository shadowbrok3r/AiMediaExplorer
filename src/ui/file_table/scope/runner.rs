//! Runs a scope loader as a tracked background task and sends its batches to the tab.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use crossbeam::channel::Sender;

use super::load::{self, Env, Request};
use super::{LoadBody, LoadMsg};
use crate::command::{AppCommand, Target};
use crate::tasks::{TaskKind, TaskSpec};

/// Age below which a load shows no chip in the status bar.
const VISIBLE_AFTER: Duration = Duration::from_millis(400);
/// How long a load waits for the database to connect.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(60);

/// Title of the task of a load.
pub fn task_title(req: &Request) -> String {
    let what = req.scope.default_title();
    if req.start > 0 { format!("Load more of {what}") } else { format!("Load {what}") }
}

/// The task a load shows in the status bar.
pub fn task_spec(req: &Request, cancel: &Arc<AtomicBool>) -> TaskSpec {
    let retry = AppCommand::OpenScope { scope: req.scope.clone(), target: Target::Current, title: None, recursive: false };
    TaskSpec::new(TaskKind::Database, task_title(req))
        .cancel_flag(cancel.clone())
        .retry(retry)
        .user_initiated()
        .visible_after(VISIBLE_AFTER)
        .toast_after(Duration::from_secs(24 * 3600))
}

/// Starts the load `id` of `req`; batches and the end arrive on `tx` as [`LoadMsg`]s with that id.
pub fn spawn(id: u64, req: Request, tx: Sender<LoadMsg>, cancel: Arc<AtomicBool>) {
    let spec = task_spec(&req, &cancel);
    let rows = tx.clone();
    tokio::spawn(async move {
        let result = crate::tasks::track(spec, move |reporter| async move {
            if !load::wait_for_connection(&cancel, CONNECT_TIMEOUT).await {
                if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                    return Ok(load::Outcome { cancelled: true, ..Default::default() });
                }
                anyhow::bail!("the database is not connected");
            }
            let env = Env::live(cancel.clone(), Arc::new(move |done, total| reporter.progress(done, total)));
            let mut emit = move |batch| {
                let sent = rows.send(LoadMsg { load: id, body: LoadBody::Rows(batch) }).is_ok();
                crate::command::request_repaint();
                sent
            };
            load::run(&crate::database::DB, &req, &env, &mut emit).await.map_err(|e| anyhow::anyhow!(super::one_line(&format!("{e:#}"))))
        })
        .await;
        let body = match result {
            Ok(outcome) => LoadBody::Done(outcome),
            Err(e) => LoadBody::Failed(format!("{e:#}")),
        };
        let _ = tx.send(LoadMsg { load: id, body });
        crate::command::request_repaint();
    });
}

/// Checks which of `paths` have a CLIP embedding and tells the tab.
pub fn spawn_embedded_check(paths: Vec<String>, tx: Sender<LoadMsg>) {
    if paths.is_empty() {
        return;
    }
    tokio::spawn(async move {
        if !crate::database::is_connected() {
            return;
        }
        if let Ok(found) = crate::database::scope_queries::embedded_paths(&crate::database::DB, &paths).await {
            if !found.is_empty() {
                let _ = tx.send(LoadMsg { load: 0, body: LoadBody::Embedded(found.into_iter().collect()) });
                crate::command::request_repaint();
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::Scope;

    #[test]
    fn the_task_names_the_scope_and_retries_by_reopening_it() {
        let req = Request::first(Scope::Tag("sunset".into()));
        let cancel = Arc::new(AtomicBool::new(false));
        let spec = task_spec(&req, &cancel);
        assert_eq!(spec.title, "Load Tag: sunset");
        assert_eq!(spec.kind, TaskKind::Database);
        assert!(spec.user_initiated && spec.cancel.is_some());
        assert_eq!(spec.visible_after, VISIBLE_AFTER);
        let retry = spec.retry.expect("a failed load can be retried");
        assert_eq!(retry.log_json(), r#"{"cmd":"open_scope","scope":"tag:sunset","target":"current"}"#);
    }

    #[test]
    fn a_further_page_says_so_in_its_title() {
        let req = Request { scope: Scope::Semantic("cat".into()), start: 100, vector: None };
        assert_eq!(task_title(&req), "Load more of Search: cat");
    }

    #[test]
    fn the_cancel_handle_of_the_task_sets_the_loaders_flag() {
        let cancel = Arc::new(AtomicBool::new(false));
        let spec = task_spec(&Request::first(Scope::Library), &cancel);
        spec.cancel.unwrap().call();
        assert!(cancel.load(std::sync::atomic::Ordering::Relaxed));
    }
}
