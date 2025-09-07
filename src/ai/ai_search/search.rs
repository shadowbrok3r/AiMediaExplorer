impl super::AISearchEngine {
    pub async fn search(&self, query: &str) -> anyhow::Result<Vec<crate::database::Thumbnail>, anyhow::Error> {
        let q = query.trim().to_ascii_lowercase();
        if q.is_empty() { return Ok(Vec::new()); }
        let files = self.files.lock().await;
        // Naive scoring: count occurrences of query in aggregated text fields
        let mut scored: Vec<(usize, crate::database::Thumbnail)> = files.iter().filter_map(|f| {
            let mut hay = String::new();
            hay.push_str(&f.filename);
            if let Some(d) = &f.description { hay.push('\n'); hay.push_str(d); }
            if let Some(c) = &f.caption { hay.push('\n'); hay.push_str(c); }
            if let Some(cat) = &f.category { hay.push('\n'); hay.push_str(cat); }
            if !f.tags.is_empty() { hay.push('\n'); hay.push_str(&f.tags.join(" ")); }
            let h_low = hay.to_ascii_lowercase();
            if h_low.contains(&q) {
                // crude occurrence count
                let mut count = 0usize;
                let mut idx = 0usize;
                while let Some(pos) = h_low[idx..].find(&q) { count += 1; idx += pos + q.len(); }
                Some((count, f.clone()))
            } else { None }
        }).collect();
        scored.sort_by(|a,b| b.0.cmp(&a.0));
        let mut out: Vec<crate::database::Thumbnail> = Vec::new();
        for (i,(score, f)) in scored.into_iter().enumerate() { 
            let mut clip_embedding = f.clone().get_embedding().await?;
            clip_embedding.similarity_score = Some(score as f32); if i < 100 { 
                out.push(f); 
            } else { 
                break; 
            } 
        }
        Ok(out)
    }

    pub async fn get_all_files(&self) -> anyhow::Result<Vec<crate::database::Thumbnail>, anyhow::Error> { 
        Ok(self.files.lock().await.clone()) 
    }

    pub async fn get_file_metadata(&self, path: &str) -> Option<crate::database::Thumbnail> { 
        self.files.lock().await.iter().find(|f| f.path == path).cloned() 
    }
}