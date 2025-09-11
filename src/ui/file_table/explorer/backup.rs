impl crate::ui::file_table::FileExplorer {
    // --- Backup helpers ---
    fn collect_visible_files(&self) -> Vec<std::path::PathBuf> {
        self.table
            .iter()
            .filter(|r| r.file_type != "<DIR>")
            .map(|r| std::path::PathBuf::from(r.path.clone()))
            .collect()
    }

    fn unique_dest_path(dest: std::path::PathBuf) -> std::path::PathBuf {
        if !dest.exists() { return dest; }
        let stem = dest.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let ext_opt = dest.extension().and_then(|s| s.to_str());
        let parent = dest.parent().map(|p| p.to_path_buf()).unwrap_or_default();
        let mut idx = 1u32;
        loop {
            let name = match ext_opt {
                Some(ext) => format!("{} ({}).{}", stem, idx, ext),
                None => format!("{} ({})", stem, idx),
            };
            let mut cand = parent.clone();
            cand.push(name);
            if !cand.exists() { return cand; }
            idx += 1;
            if idx > 10_000 { return dest; }
        }
    }

    fn do_copy_files(paths: Vec<std::path::PathBuf>, dest_dir: std::path::PathBuf) {
        tokio::spawn(async move {
            for src in paths {
                if !src.is_file() { continue; }
                let mut dest = dest_dir.clone();
                if let Some(name) = src.file_name() { dest.push(name); } else { continue; }
                if dest.exists() { dest = crate::ui::file_table::FileExplorer::unique_dest_path(dest); }
                let s2 = src.clone();
                let d2 = dest.clone();
                let res = tokio::task::spawn_blocking(move || std::fs::copy(&s2, &d2)).await;
                match res {
                    Ok(Ok(_)) => {}
                    Ok(Err(e)) => log::error!("Backup copy failed {} -> {}: {e:?}", src.display(), dest.display()),
                    Err(e) => log::error!("Backup copy join error {} -> {}: {e:?}", src.display(), dest.display()),
                }
            }
            log::info!("Backup copy complete to {}", dest_dir.display());
        });
    }

    fn do_move_files(paths: Vec<std::path::PathBuf>, dest_dir: std::path::PathBuf) {
        tokio::spawn(async move {
            for src in paths {
                if !src.is_file() { continue; }
                let mut dest = dest_dir.clone();
                if let Some(name) = src.file_name() { dest.push(name); } else { continue; }
                if dest.exists() { dest = crate::ui::file_table::FileExplorer::unique_dest_path(dest); }
                // Try rename first (fast within same volume), else fallback to copy+remove
                let s2 = src.clone();
                let d2 = dest.clone();
                let res = tokio::task::spawn_blocking(move || std::fs::rename(&s2, &d2)).await;
                let renamed_ok = matches!(res, Ok(Ok(())));
                if !renamed_ok {
                    let s3 = src.clone();
                    let d3 = dest.clone();
                    let copy_res = tokio::task::spawn_blocking(move || std::fs::copy(&s3, &d3)).await;
                    match copy_res {
                        Ok(Ok(_)) => {
                            let s4 = src.clone();
                            let _ = tokio::task::spawn_blocking(move || std::fs::remove_file(&s4)).await;
                        }
                        Ok(Err(e)) => log::error!("Backup move (copy phase) failed {} -> {}: {e:?}", src.display(), dest.display()),
                        Err(e) => log::error!("Backup move (copy phase) join error {} -> {}: {e:?}", src.display(), dest.display()),
                    }
                }
            }
            log::info!("Backup move complete to {}", dest_dir.display());
        });
    }

    pub fn backup_copy_visible_to_dir(&mut self, dir: std::path::PathBuf) {
        let files = self.collect_visible_files();
        if files.is_empty() { return; }
        Self::do_copy_files(files, dir);
    }

    pub fn backup_move_visible_to_dir(&mut self, dir: std::path::PathBuf) {
        let files = self.collect_visible_files();
        if files.is_empty() { return; }
        Self::do_move_files(files, dir);
    }

}