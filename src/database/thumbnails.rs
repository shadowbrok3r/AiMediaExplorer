use std::time::Instant;

use crate::DB;
use surrealdb::RecordId;


impl crate::Thumbnail {
    // Fetch a single thumbnail row by exact path (leverages UNIQUE index on path)
    pub async fn get_thumbnail_by_path(
        path: &str,
    ) -> anyhow::Result<Option<Self>, anyhow::Error> {
        let resp: Option<Self> = DB
            .query("SELECT * FROM thumbnails WHERE path = $path")
            .bind(("path", path.to_string()))
            .await?
            .take(0)?;
        log::warn!("SELECT * FROM thumbnails WHERE path = $path: {:?}", resp.is_some());
        Ok(resp)
    }

    // Fetch a single thumbnail row by exact path (leverages UNIQUE index on path)
    pub async fn get_all_thumbnails_from_directory(
        path: &str,
    ) -> anyhow::Result<Vec<Self>, anyhow::Error> {
        let now = Instant::now();
        let resp: Vec<Self> = DB
            .query("SELECT * FROM thumbnails WITH INDEX idx_parent_dir WHERE parent_dir = $parent_directory")
            .bind(("parent_directory", path.to_string()))
            .await?
            .take(0)?;
        log::warn!("get_all_thumbnails_from_directory is empty: {:?}\nTime for query: {:?}", resp.is_empty(), now.elapsed().as_secs_f32());
        Ok(resp)
    }

    pub async fn get_thumbnail_id_by_path(
        path: &str,
    ) -> anyhow::Result<Option<RecordId>, anyhow::Error> {
        log::info!("Getting thumbnail for path: {path:?}");
        let resp: Option<RecordId> = DB
            .query("SELECT VALUE id FROM thumbnails WHERE path = $path")
            .bind(("path", path.to_string()))
            .await?
            .take(0)?;
        log::info!("Thumbnail for path is Some(): {resp:?}");
        Ok(resp)
    }

    pub async fn get_embedding(self) -> anyhow::Result<super::ClipEmbeddingRow, anyhow::Error> {
        log::info!("Checking embedding with thumb_ref == {:?}", self.id.clone());
        let embedding: Option<super::ClipEmbeddingRow> = DB
            .query("SELECT * FROM clip_embeddings WITH INDEX clip_thumb_ref_idx WHERE thumb_ref = $id")
            .bind(("id", self.id.clone()))
            .await?
            .take(0)?;

        log::info!("Embedding for image is Some: {:?}", embedding.is_some());
        Ok(embedding.unwrap_or_default())
    }

    pub async fn find_thumbs_from_paths(
        chunk_vec: Vec<String>,
    ) -> anyhow::Result<Vec<Self>, anyhow::Error> {
        let thumbs: Vec<Self> = DB
        .query("SELECT id, db_created, path, filename, file_type, size, modified, hash, description, caption, tags, category FROM thumbnails WHERE array::find($paths, path) != NONE")
        .bind(("paths", chunk_vec))
        .await?
        .take(0)?;

        Ok(thumbs)
    }

    // Save (insert) a single thumbnail row (best-effort). Does not deduplicate existing rows.
    pub async fn save_thumbnail_row(self) -> anyhow::Result<(), anyhow::Error> {
        log::info!("SAVING: {:?}", self.path);
        let _: Option<Self> = DB
            .create("thumbnails")
            .content::<Self>(self)
            .await?
            .take();
        Ok(())
    }

    pub async fn update_or_create_thumbnail(
        mut self,
        metadata: &super::Thumbnail,
        thumb_b64: Option<String>,
    ) -> anyhow::Result<(), anyhow::Error> {
        // Only update fields if new content present
        if let Some(desc) = &metadata.description {
            if desc.trim().len() > 0 {
                self.description = Some(desc.clone());
            }
        }
        if let Some(caption) = &metadata.caption {
            if caption.trim().len() > 0 {
                self.caption = Some(caption.clone());
            }
        }
        if !metadata.tags.is_empty() {
            self.tags = metadata.tags.clone();
        }
        if let Some(cat) = &metadata.category {
            if cat.trim().len() > 0 {
                self.category = Some(cat.clone());
            }
        }
    // embedding field on thumbnails removed; embeddings live in clip_embeddings table
        if thumb_b64.is_some() {
            self.thumbnail_b64 = thumb_b64;
        }
        if metadata.modified.is_some() {
            self.modified = metadata.modified.clone();
        }
        if let Some(h) = &metadata.hash {
            if !h.is_empty() {
                self.hash = Some(h.clone());
            }
        }

        // Upsert strategy: use an UPDATE with WHERE path match; if none updated, INSERT new.
        let updated: Option<super::Thumbnail> = DB
            .query(
                r#"
                    UPDATE thumbnails 
                        SET filename = $filename, 
                        file_type = $file_type, 
                        size = $size, 
                        description = $description, 
                        caption = $caption, 
                        tags = $tags, 
                        category = $category, 
                        thumbnail_b64 = $thumbnail_b64, 
                        modified = $modified, 
                        hash = $hash 
                    WHERE path = $path
                    "#,
            )
            .bind(("table", crate::database::THUMBNAILS))
            .bind(("filename", self.filename.clone()))
            .bind(("file_type", self.file_type.clone()))
            .bind(("size", self.size))
            .bind(("description", self.description.clone()))
            .bind(("caption", self.caption.clone()))
            .bind(("tags", self.tags.clone()))
            .bind(("category", self.category.clone()))
            .bind(("thumbnail_b64", self.thumbnail_b64.clone()))
            .bind(("modified", self.modified.clone()))
            .bind(("hash", self.hash.clone()))
            .bind(("path", self.path.clone()))
            .await?
            .take(0)?;

        log::info!("Cached data Is Some: {:?}", updated.is_some());
        Ok(())
    }

}

pub async fn save_thumbnail_batch(
    thumbs: Vec<super::Thumbnail>,
) -> anyhow::Result<(), anyhow::Error> {
    log::info!("save_thumbnail_batch");
    let _: Vec<super::Thumbnail> = DB
        .insert("thumbnails")
        .content::<Vec<super::Thumbnail>>(thumbs)
        .await?;
    Ok(())
}

pub async fn get_thumbnail_paths() -> anyhow::Result<Vec<String>, anyhow::Error> {
    let paths: Vec<String> = DB.query("SELECT VALUE path FROM thumbnails").await?.take(0)?;
    Ok(paths)
}


