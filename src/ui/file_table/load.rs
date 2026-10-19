use crate::command::Scope;

impl super::FileExplorer {
    /// Shows the members of a collection, read page by page.
    pub fn load_logical_group_by_name(&mut self, name: String) {
        if let Err(reason) = self.open_scope(Scope::Collection(name)) {
            log::warn!("[scope] {reason}");
        }
    }

    /// Shows a collection in this tab.
    pub fn open_collection(&mut self, name: &str) {
        self.load_logical_group_by_name(name.to_string());
    }
}
