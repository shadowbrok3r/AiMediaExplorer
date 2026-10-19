use crate::Thumbnail;
use crate::command::Scope;
use std::path::Path;

impl crate::ui::file_table::FileExplorer {
    pub fn directory_to_thumbnail(p: &Path) -> Option<Thumbnail> {
        let name = p.file_name()?.to_string_lossy().to_string();
        let mut t = Thumbnail::default();
        t.path = p.to_string_lossy().to_string();
        t.filename = name;
        t.file_type = "<DIR>".into();
        Some(t)
    }

    /// Scans `current_path` and everything below it; the scan is recorded so it can be opened again as `Scope::Scan`.
    pub(crate) fn start_recursive_scan(&mut self) {
        self.recursive_scan = true;
        self.scan_done = false;
        self.file_scan_progress = 0.0;
        self.last_scan_rows.clear();
        self.last_scan_paths.clear();
        self.last_scan_root = Some(self.current_path.clone());
        self.viewer.thumbs.new_scope(self.viewer.scope);
        self.failed_thumbs.clear();
        let mut filters = crate::Filters::default();
        filters.root = std::path::PathBuf::from(self.current_path.clone());
        filters.include_archives = true;
        self.launch_recorded_scan(filters, true);
    }

    /// Starts a scan on this tab's channel; a recursive one is first written to the scan cache.
    pub(crate) fn launch_recorded_scan(&mut self, filters: crate::Filters, recursive: bool) {
        let scan_id = crate::next_scan_id();
        self.owning_scan_id = Some(scan_id);
        self.current_scan_id = Some(scan_id);
        self.nav.fs_started.get_or_insert_with(std::time::Instant::now);
        let tx = self.scan_tx.clone();
        let root = filters.root.display().to_string();
        tokio::spawn(async move {
            if recursive && crate::database::is_connected() {
                if let Err(e) = crate::database::CachedScan::create(&root, Some(format!("Scan: {root}")), scan_id).await {
                    log::warn!("[scan] could not record scan {scan_id}: {e}");
                }
            }
            crate::spawn_scan(filters, tx, recursive, scan_id).await;
        });
    }

    /// Shows rows held in memory, the way a results tab does.
    pub fn set_rows(&mut self, rows: Vec<crate::database::Thumbnail>) {
        self.show_rows(None, rows);
    }

    pub fn nav_back(&mut self) {
        self.go_back();
    }

    pub fn nav_forward(&mut self) {
        self.go_forward();
    }

    pub fn nav_up(&mut self) {
        self.go_up();
    }

    pub fn nav_home(&mut self) {
        self.go_home();
    }

    pub fn refresh(&mut self) {
        self.reload_scope();
    }

    /// Opens a folder by path.
    pub fn set_path(&mut self, path: impl Into<String>) {
        let path = crate::utilities::windows::normalize_wsl_unc(&path.into());
        if let Err(reason) = self.open_scope(Scope::Folder(path)) {
            crate::ui::dialogs::toast(egui_toast::ToastKind::Error, reason);
        }
    }
}
