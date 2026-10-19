use serde::{Deserialize, Serialize};
use surrealdb::{Connection, Surreal};
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
    /// JSON of the full filter spec; groups saved before it existed have only the fields above.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spec: Option<String>,
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
    let slug = out.trim_matches('_').to_string();
    if slug.is_empty() { format!("view_{}", &blake3::hash(name.to_lowercase().as_bytes()).to_hex()[..16]) } else { slug }
}

pub async fn save_filter_group(g: FilterGroup) -> anyhow::Result<FilterGroup, anyhow::Error> {
    let _ga = db_activity("Upsert filter_group");
    db_set_detail(format!("Saving filter group '{}'", g.name));
    save_filter_group_on(&*super::DB, g).await.map_err(|e| { db_set_error(format!("Save filter group failed: {e}")); e })
}

pub async fn list_filter_groups() -> anyhow::Result<Vec<FilterGroup>, anyhow::Error> {
    let _ga = db_activity("Select all filter_groups");
    db_set_detail("Loading filter groups".to_string());
    list_filter_groups_on(&*super::DB).await
}

pub async fn delete_filter_group_by_name(name: &str) -> anyhow::Result<(), anyhow::Error> {
    let _ga = db_activity("Delete filter_group by name");
    db_set_detail(format!("Deleting filter group '{}'", name));
    delete_filter_group_on(&*super::DB, name).await.map_err(|e| { db_set_error(format!("Delete filter group failed: {e}")); e })
}

pub async fn rename_filter_group(old: &str, new: &str) -> anyhow::Result<(), anyhow::Error> {
    let _ga = db_activity("Rename filter_group");
    db_set_detail(format!("Renaming filter group '{old}' to '{new}'"));
    rename_filter_group_on(&*super::DB, old, new).await.map_err(|e| { db_set_error(format!("Rename filter group failed: {e}")); e })
}

pub async fn save_filter_group_on<C: Connection>(db: &Surreal<C>, mut g: FilterGroup) -> anyhow::Result<FilterGroup, anyhow::Error> {
    let rid = RecordId::new(super::FILTER_GROUPS, slugify_name(&g.name));
    let now = Datetime::from(chrono::Utc::now());
    let existing = db.select::<Option<FilterGroup>>(rid.clone()).await?;
    g.id = Some(rid.clone());
    g.created = existing.and_then(|e| e.created).or(g.created).or_else(|| Some(now.clone()));
    g.updated = Some(now);
    db.upsert::<Option<FilterGroup>>(rid).content::<FilterGroup>(g.clone()).await?;
    Ok(g)
}

pub async fn list_filter_groups_on<C: Connection>(db: &Surreal<C>) -> anyhow::Result<Vec<FilterGroup>, anyhow::Error> {
    let mut groups: Vec<FilterGroup> = db.select(super::FILTER_GROUPS).await?;
    groups.sort_by_key(|g| g.name.to_lowercase());
    Ok(groups)
}

pub async fn delete_filter_group_on<C: Connection>(db: &Surreal<C>, name: &str) -> anyhow::Result<(), anyhow::Error> {
    let rid = RecordId::new(super::FILTER_GROUPS, slugify_name(name));
    let _deleted: Option<FilterGroup> = db.delete(rid).await?;
    Ok(())
}

/// Gives a saved group a new name, keeping its contents; fails when another group already has that name.
pub async fn rename_filter_group_on<C: Connection>(db: &Surreal<C>, old: &str, new: &str) -> anyhow::Result<(), anyhow::Error> {
    let new = new.trim();
    if new.is_empty() {
        anyhow::bail!("the new name is empty");
    }
    let old_rid = RecordId::new(super::FILTER_GROUPS, slugify_name(old));
    let new_rid = RecordId::new(super::FILTER_GROUPS, slugify_name(new));
    let Some(mut group) = db.select::<Option<FilterGroup>>(old_rid.clone()).await? else {
        anyhow::bail!("no saved view named '{old}'");
    };
    if old_rid != new_rid && db.select::<Option<FilterGroup>>(new_rid.clone()).await?.is_some() {
        anyhow::bail!("a saved view named '{new}' already exists");
    }
    group.name = new.to_string();
    group.id = Some(new_rid.clone());
    group.updated = Some(Datetime::from(chrono::Utc::now()));
    db.upsert::<Option<FilterGroup>>(new_rid.clone()).content::<FilterGroup>(group).await?;
    if old_rid != new_rid {
        let _deleted: Option<FilterGroup> = db.delete(old_rid).await?;
    }
    Ok(())
}
