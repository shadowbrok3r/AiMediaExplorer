use serde::{Deserialize, Serialize};
use surrealdb::types::{RecordId, Datetime, SurrealValue};
use crate::database::{db_activity, db_set_detail, db_set_error};

#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct FilterGroup {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<RecordId>,
    pub name: String,
    pub include_images: bool,
    pub include_videos: bool,
    pub include_dirs: bool,
    pub skip_icons: bool,
    pub min_size_bytes: Option<u64>,
    pub max_size_bytes: Option<u64>,
    pub excluded_terms: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created: Option<Datetime>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated: Option<Datetime>,
}

fn slugify_name(name: &str) -> String {
    let mut s = String::with_capacity(name.len());
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() { s.push(ch.to_ascii_lowercase()); }
        else if ch.is_whitespace() || ch == '-' { s.push('_'); }
        else { s.push('_'); }
    }
    // trim duplicate underscores
    let mut out = String::with_capacity(s.len());
    let mut prev = '\0';
    for ch in s.chars() {
        if ch == '_' && prev == '_' { continue; }
        out.push(ch);
        prev = ch;
    }
    out.trim_matches('_').to_string()
}

pub async fn save_filter_group(mut g: FilterGroup) -> anyhow::Result<FilterGroup, anyhow::Error> {
    let _ga = db_activity("Upsert filter_group");
    db_set_detail(format!("Saving filter group '{}'", g.name));
    let key = slugify_name(&g.name);
    let rid = RecordId::new(super::FILTER_GROUPS, key);
    g.id = Some(rid.clone());
    super::DB
        .upsert::<Option<FilterGroup>>(rid)
        .content::<FilterGroup>(g.clone())
        .await
        .map_err(|e| { db_set_error(format!("Save filter group failed: {e}")); e })?;
    Ok(g)
}

pub async fn list_filter_groups() -> anyhow::Result<Vec<FilterGroup>, anyhow::Error> {
    let _ga = db_activity("Select all filter_groups");
    db_set_detail("Loading filter groups".to_string());
    let res: Vec<FilterGroup> = super::DB.select(super::FILTER_GROUPS).await?;
    Ok(res)
}

pub async fn delete_filter_group_by_name(name: &str) -> anyhow::Result<(), anyhow::Error> {
    let _ga = db_activity("Delete filter_group by name");
    db_set_detail(format!("Deleting filter group '{}'", name));
    let key = slugify_name(name);
    let rid = RecordId::new(super::FILTER_GROUPS, key);
    let _deleted: Option<FilterGroup> = super::DB.delete(rid).await
        .map_err(|e| { db_set_error(format!("Delete filter group failed: {e}")); e })?;
    Ok(())
}
