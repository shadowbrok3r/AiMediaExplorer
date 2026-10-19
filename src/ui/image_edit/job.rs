//! The edit job: loads the model when it is not loaded, runs the edit and reports each phase to the tab.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crossbeam::channel::{Receiver, Sender, unbounded};

use super::backend::{EditBackend, EditRequest, RunCtl};
use super::model::{EditError, ErrorKind, Phase, fit_dims};

/// Longest side of the picture kept for display; the saved file keeps every pixel.
const DISPLAY_MAX: u32 = 2048;

/// A result decoded for display.
#[derive(Clone, Debug)]
pub struct Decoded {
    pub size: [usize; 2],
    pub rgba: Vec<u8>,
    /// Dimensions of the saved image.
    pub full: [u32; 2],
}

/// What a job reports.
#[derive(Debug)]
pub enum Msg {
    Phase(Phase),
    /// A picture of the edit in progress.
    Preview(Decoded),
    /// A finished edit: the PNG and its decoded form.
    Finished { png: Arc<[u8]>, decoded: Decoded, source: String, prompt: String },
    Failed(EditError),
    Cancelled,
    /// The model was loaded or unloaded by the buttons of the tab.
    ModelLoaded(Result<(), EditError>),
    /// The GGUF file was fetched.
    Gguf(Result<String, String>),
    /// A file was written for the result with this id.
    Saved { id: u64, result: Result<String, String> },
    /// The result with this id was added to the library.
    Library { id: u64, result: Result<String, String> },
    /// Names of the collections for the choice.
    Collections(Vec<String>),
}

/// Decodes PNG (or any picture) bytes for display.
pub fn decode_for_display(bytes: &[u8]) -> Result<Decoded, EditError> {
    let img = image::load_from_memory(bytes).map_err(|e| EditError::new(ErrorKind::Other, format!("the result could not be decoded: {e}")))?;
    let full = [img.width(), img.height()];
    let (w, h) = fit_dims(img.width(), img.height(), DISPLAY_MAX);
    let rgba = if (w, h) == (img.width(), img.height()) { img.to_rgba8() } else { image::imageops::resize(&img.to_rgba8(), w, h, image::imageops::FilterType::Triangle) };
    Ok(Decoded { size: [rgba.width() as usize, rgba.height() as usize], rgba: rgba.into_raw(), full })
}

/// Runs one job to its end; every phase and the outcome go to `tx`.
pub async fn run_job(backend: Arc<dyn EditBackend>, req: EditRequest, cancel: Arc<AtomicBool>, tx: Sender<Msg>) {
    let send = |msg: Msg| {
        let _ = tx.send(msg);
        crate::command::request_repaint();
    };
    if !req.source.is_file() {
        send(Msg::Failed(EditError::new(ErrorKind::Source, format!("{} does not exist", req.source.display()))));
        return;
    }
    if !backend.is_loaded() {
        send(Msg::Phase(Phase::LoadingModel));
        if let Err(e) = backend.load(&req.settings, &req.params).await {
            send(Msg::Failed(e));
            return;
        }
        if cancel.load(Ordering::Relaxed) {
            send(Msg::Cancelled);
            return;
        }
    }
    let total = req.params.steps as u64;
    send(Msg::Phase(Phase::Running { done: 0, total, detail: "Starting".to_string() }));
    let progress_tx = tx.clone();
    let progress: Arc<dyn Fn(u64, u64, &str) + Send + Sync> = Arc::new(move |done, total, detail| {
        let _ = progress_tx.send(Msg::Phase(Phase::Running { done, total, detail: detail.to_string() }));
        crate::command::request_repaint();
    });
    let (preview_tx, preview_rx) = unbounded::<Vec<u8>>();
    let forward = {
        let tx = tx.clone();
        let rx: Receiver<Vec<u8>> = preview_rx;
        tokio::task::spawn_blocking(move || {
            while let Ok(png) = rx.recv() {
                if let Ok(decoded) = decode_for_display(&png) {
                    let _ = tx.send(Msg::Preview(decoded));
                    crate::command::request_repaint();
                }
            }
        })
    };
    let ctl = RunCtl { cancel: cancel.clone(), progress, preview: (req.params.preview_every_n > 0).then_some(preview_tx.clone()) };
    let outcome = backend.edit(&req, &ctl).await;
    drop(ctl);
    drop(preview_tx);
    let _ = forward.await;
    match outcome {
        _ if cancel.load(Ordering::Relaxed) => send(Msg::Cancelled),
        Err(e) => send(Msg::Failed(e)),
        Ok(png) => match tokio::task::spawn_blocking(move || decode_for_display(&png).map(|d| (png, d))).await {
            Ok(Ok((png, decoded))) => send(Msg::Finished { png: Arc::from(png.into_boxed_slice()), decoded, source: req.source.display().to_string(), prompt: req.params.prompt.clone() }),
            Ok(Err(e)) => send(Msg::Failed(e)),
            Err(e) => send(Msg::Failed(EditError::new(ErrorKind::Other, e.to_string()))),
        },
    }
}
