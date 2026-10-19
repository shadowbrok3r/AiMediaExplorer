//! General section: startup behaviour, the default view, thumbnail size and confirmations.

use eframe::egui::{ComboBox, Slider, Ui};

use super::sections::name;
use super::widgets::{Focus, caption, group, toggle};
use crate::settings::ui_prefs::{self, Confirm, Confirmations};
use crate::ui::file_table::table::view_mode::{self, SizePreset, TILE_MAX, TILE_MIN, TileFit, ViewMode, ViewPrefs};

/// The values of this section.
#[derive(Clone, Debug, PartialEq)]
pub struct Values {
    pub restore_tabs: bool,
    pub restore_last_scope: bool,
    pub confirm: Confirmations,
    pub view: ViewPrefs,
}

impl Values {
    /// What is saved now.
    pub fn live() -> Self {
        let prefs = ui_prefs::get();
        Self { restore_tabs: prefs.startup.restore_tabs, restore_last_scope: prefs.startup.restore_last_scope, confirm: prefs.confirmations, view: view_mode::app_prefs() }
    }
}

/// The edited copy of the General values and the ones it started from.
#[derive(Clone, Debug)]
pub struct General {
    saved: Values,
    pub value: Values,
}

impl Default for General {
    fn default() -> Self {
        Self::new()
    }
}

impl General {
    pub fn new() -> Self {
        let live = Values::live();
        Self { saved: live.clone(), value: live }
    }

    /// Names of the parts that differ from what is saved.
    pub fn dirty_parts(&self) -> Vec<&'static str> {
        let (s, v) = (&self.saved, &self.value);
        let mut out = Vec::new();
        if s.restore_tabs != v.restore_tabs {
            out.push("restore_tabs");
        }
        if s.restore_last_scope != v.restore_last_scope {
            out.push("restore_last_scope");
        }
        if s.confirm != v.confirm {
            out.push("confirmations");
        }
        if s.view.mode != v.view.mode {
            out.push("default_view");
        }
        if s.view.tile != v.view.tile {
            out.push("tile_size");
        }
        if s.view.fit != v.view.fit {
            out.push("tile_fit");
        }
        if s.view.captions != v.view.captions {
            out.push("captions");
        }
        if s.view.badges != v.view.badges {
            out.push("badges");
        }
        out
    }

    pub fn is_dirty(&self) -> bool {
        !self.dirty_parts().is_empty()
    }

    /// Takes in changes made elsewhere to the parts the user has not edited.
    pub fn sync(&mut self) {
        let live = Values::live();
        macro_rules! follow {
            ($($path:tt)+) => {
                if self.value.$($path)+ == self.saved.$($path)+ {
                    self.value.$($path)+ = live.$($path)+.clone();
                }
                self.saved.$($path)+ = live.$($path)+.clone();
            };
        }
        follow!(restore_tabs);
        follow!(restore_last_scope);
        follow!(confirm);
        follow!(view.mode);
        follow!(view.tile);
        follow!(view.fit);
        follow!(view.captions);
        follow!(view.badges);
    }

    /// Saves the changed parts and returns how many there were.
    pub fn apply(&mut self) -> usize {
        let parts = self.dirty_parts();
        if parts.is_empty() {
            return 0;
        }
        let (s, v) = (self.saved.clone(), self.value.clone());
        ui_prefs::update(|p| {
            if s.restore_tabs != v.restore_tabs {
                p.startup.restore_tabs = v.restore_tabs;
            }
            if s.restore_last_scope != v.restore_last_scope {
                p.startup.restore_last_scope = v.restore_last_scope;
            }
            if s.confirm != v.confirm {
                p.confirmations = v.confirm;
            }
        });
        view_mode::update_app_prefs(|p| {
            if s.view.mode != v.view.mode {
                p.mode = v.view.mode;
            }
            if s.view.tile != v.view.tile {
                p.tile = v.view.tile;
            }
            if s.view.fit != v.view.fit {
                p.fit = v.view.fit;
            }
            if s.view.captions != v.view.captions {
                p.captions = v.view.captions;
            }
            if s.view.badges != v.view.badges {
                p.badges = v.view.badges;
            }
        });
        view_mode::flush_app_prefs();
        self.saved = self.value.clone();
        parts.len()
    }

    /// Drops the edits.
    pub fn revert(&mut self) {
        *self = Self::new();
    }

    pub fn ui(&mut self, ui: &mut Ui, focus: &Focus) {
        let v = &mut self.value;
        group(ui, "Startup");
        toggle(ui, focus, name::RESTORE_TABS, "Reopen the browser tabs that were open when the application closed", &mut v.restore_tabs);
        toggle(ui, focus, name::RESTORE_SCOPE, "Open the scope the focused tab showed when the application closed", &mut v.restore_last_scope);

        group(ui, "Browsing");
        ui.horizontal(|ui| {
            caption(ui, focus, name::DEFAULT_VIEW);
            ComboBox::from_id_salt("general-default-view").selected_text(v.view.mode.label()).show_ui(ui, |ui| {
                for mode in ViewMode::ALL {
                    ui.selectable_value(&mut v.view.mode, mode, mode.label());
                }
            });
        });
        ui.horizontal(|ui| {
            caption(ui, focus, name::THUMB_SIZE);
            ui.add(Slider::new(&mut v.view.tile, TILE_MIN..=TILE_MAX).suffix(" px").step_by(8.0));
            for preset in SizePreset::ALL {
                if ui.selectable_label(SizePreset::of(v.view.tile) == Some(preset), preset.label()).clicked() {
                    v.view.tile = preset.px();
                }
            }
        });
        ui.horizontal(|ui| {
            caption(ui, focus, name::THUMB_FIT);
            ComboBox::from_id_salt("general-tile-fit").selected_text(v.view.fit.label()).show_ui(ui, |ui| {
                for fit in [TileFit::Crop, TileFit::Fit] {
                    ui.selectable_value(&mut v.view.fit, fit, fit.label());
                }
            });
        });
        toggle(ui, focus, name::CAPTIONS, "", &mut v.view.captions);
        toggle(ui, focus, name::BADGES, "", &mut v.view.badges);

        group(ui, name::CONFIRMATIONS);
        for which in Confirm::ALL {
            let mut on = v.confirm.get(which);
            if ui.checkbox(&mut on, which.label()).changed() {
                v.confirm.set(which, on);
            }
        }
        ui.weak("Metadata tools always show what they will change and ask first.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reset_world() {
        ui_prefs::update(|p| *p = ui_prefs::UiPrefs::default());
        view_mode::update_app_prefs(|p| *p = ViewPrefs::default());
    }

    #[test]
    fn a_fresh_section_is_clean_and_matches_what_is_saved() {
        reset_world();
        let g = General::new();
        assert!(!g.is_dirty());
        assert_eq!(g.value, Values::live());
    }

    #[test]
    fn editing_marks_the_parts_dirty_and_revert_clears_them() {
        reset_world();
        let mut g = General::new();
        g.value.restore_tabs = true;
        g.value.view.mode = ViewMode::Details;
        g.value.confirm.set(Confirm::MoveFiles, false);
        assert_eq!(g.dirty_parts(), vec!["restore_tabs", "confirmations", "default_view"]);
        g.revert();
        assert!(!g.is_dirty());
        assert!(!g.value.restore_tabs);
    }

    #[test]
    fn apply_writes_the_edits_and_nothing_else() {
        reset_world();
        ui_prefs::update(|p| {
            p.startup.last_scope = Some("fs:/last".into());
            p.watch("/pics");
        });
        let mut g = General::new();
        g.value.restore_last_scope = true;
        g.value.view.tile = 200.0;
        assert_eq!(g.apply(), 2);
        let prefs = ui_prefs::get();
        assert!(prefs.startup.restore_last_scope && !prefs.startup.restore_tabs);
        assert_eq!(prefs.startup.last_scope.as_deref(), Some("fs:/last"), "other fields are untouched");
        assert_eq!(prefs.library.watched, vec!["/pics".to_string()]);
        assert_eq!(view_mode::app_prefs().tile, 200.0);
        assert!(!g.is_dirty());
        assert_eq!(g.apply(), 0);
        reset_world();
    }

    #[test]
    fn apply_does_not_overwrite_view_fields_changed_elsewhere_meanwhile() {
        reset_world();
        let mut g = General::new();
        g.value.view.captions = false;
        view_mode::update_app_prefs(|p| p.tile = 120.0);
        g.apply();
        let live = view_mode::app_prefs();
        assert!(!live.captions);
        assert_eq!(live.tile, 120.0, "the stale tile size in the draft did not overwrite the newer one");
        reset_world();
    }

    #[test]
    fn sync_follows_untouched_parts_only() {
        reset_world();
        let mut g = General::new();
        g.value.view.fit = TileFit::Fit;
        view_mode::update_app_prefs(|p| {
            p.tile = 100.0;
            p.fit = TileFit::Crop;
        });
        g.sync();
        assert_eq!(g.value.view.tile, 100.0);
        assert_eq!(g.value.view.fit, TileFit::Fit, "the edit stays");
        assert_eq!(g.dirty_parts(), vec!["tile_fit"]);
        reset_world();
    }

    #[test]
    fn view_preferences_stay_in_range() {
        reset_world();
        let mut g = General::new();
        g.value.view.tile = 5000.0;
        g.apply();
        assert!(view_mode::app_prefs().tile <= TILE_MAX);
        reset_world();
    }
}
