//! The model behind the Image Edit tab as a trait, so the job runs against the Qwen pipeline or a scripted stand-in.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use async_trait::async_trait;
use crossbeam::channel::Sender;

use super::model::{EditError, ModelSettings, Params};

/// One edit to run.
#[derive(Clone, Debug)]
pub struct EditRequest {
    pub source: PathBuf,
    pub params: Params,
    pub settings: ModelSettings,
}

/// What a running edit can use: the cancel flag, a progress report and a channel for step previews.
#[derive(Clone)]
pub struct RunCtl {
    pub cancel: Arc<AtomicBool>,
    /// Steps done, steps in all and a short text.
    pub progress: Arc<dyn Fn(u64, u64, &str) + Send + Sync>,
    pub preview: Option<Sender<Vec<u8>>>,
}

#[async_trait]
pub trait EditBackend: Send + Sync + 'static {
    /// True once a model is loaded and an edit can start at once.
    fn is_loaded(&self) -> bool;
    /// Loads the model named by `settings`.
    async fn load(&self, settings: &ModelSettings, params: &Params) -> Result<(), EditError>;
    /// Runs one edit and returns the PNG of the result; a cancelled run may return what it had.
    async fn edit(&self, req: &EditRequest, ctl: &RunCtl) -> Result<Vec<u8>, EditError>;
    /// Releases the model.
    async fn unload(&self);
}

#[cfg(test)]
pub(crate) mod fake {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;

    use super::*;

    /// A backend whose answers a test scripts.
    pub struct FakeBackend {
        pub loaded: AtomicBool,
        pub load_error: Mutex<Option<String>>,
        pub edit_errors: Mutex<Vec<String>>,
        pub steps: u64,
        pub step_delay: Duration,
        pub png: Vec<u8>,
        pub loads: AtomicUsize,
        pub edits: AtomicUsize,
        pub seen: Mutex<Vec<EditRequest>>,
    }

    impl FakeBackend {
        pub fn new(png: Vec<u8>) -> Arc<Self> {
            Arc::new(Self {
                loaded: AtomicBool::new(false),
                load_error: Mutex::new(None),
                edit_errors: Mutex::new(Vec::new()),
                steps: 3,
                step_delay: Duration::from_millis(2),
                png,
                loads: AtomicUsize::new(0),
                edits: AtomicUsize::new(0),
                seen: Mutex::new(Vec::new()),
            })
        }
    }

    #[async_trait]
    impl EditBackend for FakeBackend {
        fn is_loaded(&self) -> bool {
            self.loaded.load(Ordering::SeqCst)
        }

        async fn load(&self, _settings: &ModelSettings, _params: &Params) -> Result<(), EditError> {
            self.loads.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(self.step_delay).await;
            if let Some(e) = self.load_error.lock().unwrap().clone() {
                return Err(EditError::new(super::super::model::ErrorKind::ModelLoad, e));
            }
            self.loaded.store(true, Ordering::SeqCst);
            Ok(())
        }

        async fn edit(&self, req: &EditRequest, ctl: &RunCtl) -> Result<Vec<u8>, EditError> {
            self.edits.fetch_add(1, Ordering::SeqCst);
            self.seen.lock().unwrap().push(req.clone());
            for step in 1..=self.steps {
                if ctl.cancel.load(Ordering::Relaxed) {
                    return Ok(self.png.clone());
                }
                tokio::time::sleep(self.step_delay).await;
                (ctl.progress)(step, self.steps, "Denoising");
                if let Some(tx) = &ctl.preview {
                    let _ = tx.send(self.png.clone());
                }
            }
            let next_error = {
                let mut errors = self.edit_errors.lock().unwrap();
                if errors.is_empty() { None } else { Some(errors.remove(0)) }
            };
            match next_error {
                Some(e) => Err(EditError::classified(e)),
                None => Ok(self.png.clone()),
            }
        }

        async fn unload(&self) {
            self.loaded.store(false, Ordering::SeqCst);
        }
    }
}
