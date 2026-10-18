//! Removes selected rows and their CLIP embeddings from the database.

use surrealdb::types::{RecordId, Value};

/// Deletes thumbnails by record id and embeddings by path or thumbnail reference; returns (embeddings, thumbnails) removed.
pub async fn delete_rows(ids: Vec<RecordId>, paths: Vec<String>) -> anyhow::Result<(usize, usize)> {
    if ids.is_empty() && paths.is_empty() {
        return Ok((0, 0));
    }
    let mut response = crate::DB
        .query(
            r#"
            DELETE clip_embeddings WHERE path IN $paths OR thumb_ref IN $ids RETURN BEFORE;
            DELETE thumbnails WHERE id IN $ids RETURN BEFORE;
            "#,
        )
        .bind(("paths", paths))
        .bind(("ids", ids))
        .await?;
    let embeddings: Vec<Value> = response.take(0)?;
    let thumbnails: Vec<Value> = response.take(1)?;
    Ok((embeddings.len(), thumbnails.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn deletes_only_the_given_rows_and_their_embeddings() {
        crate::settings::testing::mem_db();
        let db = &*crate::database::DB;
        db.query(
            r#"
            CREATE thumbnails:t1 SET path = '/x/1.jpg', filename = '1.jpg';
            CREATE thumbnails:t2 SET path = '/x/2.jpg', filename = '2.jpg';
            CREATE thumbnails:t3 SET path = '/x/3.jpg', filename = '3.jpg';
            CREATE clip_embeddings:e1 SET path = '/x/1.jpg', thumb_ref = thumbnails:t1, embedding = [0.1];
            CREATE clip_embeddings:e2 SET path = '/x/2.jpg', thumb_ref = thumbnails:t2, embedding = [0.2];
            CREATE clip_embeddings:e3 SET path = '/x/3.jpg', thumb_ref = thumbnails:t3, embedding = [0.3];
            "#,
        )
        .await
        .unwrap()
        .check()
        .unwrap();

        let ids = vec![RecordId::new("thumbnails", "t1"), RecordId::new("thumbnails", "t3")];
        let counts = delete_rows(ids, vec!["/x/1.jpg".into(), "/x/3.jpg".into()]).await.unwrap();
        assert_eq!(counts, (2, 2));

        let left: Vec<String> = db.query("SELECT VALUE path FROM thumbnails").await.unwrap().take(0).unwrap();
        assert_eq!(left, vec!["/x/2.jpg".to_string()]);
        let embeddings_left: Vec<String> = db.query("SELECT VALUE path FROM clip_embeddings").await.unwrap().take(0).unwrap();
        assert_eq!(embeddings_left, vec!["/x/2.jpg".to_string()]);
    }

    #[tokio::test]
    async fn an_empty_selection_touches_nothing() {
        assert_eq!(delete_rows(Vec::new(), Vec::new()).await.unwrap(), (0, 0));
    }
}
