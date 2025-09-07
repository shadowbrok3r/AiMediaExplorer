use eframe::egui::*;
use crate::ui::status::{GlobalStatusIndicator, CLIP_STATUS, StatusState};
const STYLE: &str = r#"{"override_text_style":null,"override_font_id":null,"override_text_valign":"Center","text_styles":{"Small":{"size":10.0,"family":"Proportional"},"Body":{"size":14.0,"family":"Proportional"},"Monospace":{"size":12.0,"family":"Monospace"},"Button":{"size":14.0,"family":"Proportional"},"Heading":{"size":18.0,"family":"Proportional"}},"drag_value_text_style":"Button","wrap":null,"wrap_mode":null,"spacing":{"item_spacing":{"x":3.0,"y":3.0},"window_margin":{"left":12,"right":12,"top":12,"bottom":12},"button_padding":{"x":5.0,"y":3.0},"menu_margin":{"left":12,"right":12,"top":12,"bottom":12},"indent":18.0,"interact_size":{"x":40.0,"y":20.0},"slider_width":100.0,"slider_rail_height":8.0,"combo_width":100.0,"text_edit_width":280.0,"icon_width":14.0,"icon_width_inner":8.0,"icon_spacing":6.0,"default_area_size":{"x":600.0,"y":400.0},"tooltip_width":600.0,"menu_width":400.0,"menu_spacing":2.0,"indent_ends_with_horizontal_line":false,"combo_height":200.0,"scroll":{"floating":true,"bar_width":6.0,"handle_min_length":12.0,"bar_inner_margin":4.0,"bar_outer_margin":0.0,"floating_width":2.0,"floating_allocated_width":0.0,"foreground_color":true,"dormant_background_opacity":0.0,"active_background_opacity":0.4,"interact_background_opacity":0.7,"dormant_handle_opacity":0.0,"active_handle_opacity":0.6,"interact_handle_opacity":1.0}},"interaction":{"interact_radius":5.0,"resize_grab_radius_side":5.0,"resize_grab_radius_corner":10.0,"show_tooltips_only_when_still":true,"tooltip_delay":0.5,"tooltip_grace_time":0.2,"selectable_labels":true,"multi_widget_text_select":true},"visuals":{"dark_mode":true,"text_alpha_from_coverage":"TwoCoverageMinusCoverageSq","override_text_color":[207,216,220,255],"weak_text_alpha":0.6,"weak_text_color":null,"widgets":{"noninteractive":{"bg_fill":[0,0,0,0],"weak_bg_fill":[61,61,61,232],"bg_stroke":{"width":1.0,"color":[71,71,71,247]},"corner_radius":{"nw":6,"ne":6,"sw":6,"se":6},"fg_stroke":{"width":1.0,"color":[207,216,220,255]},"expansion":0.0},"inactive":{"bg_fill":[58,51,106,0],"weak_bg_fill":[8,8,8,231],"bg_stroke":{"width":1.5,"color":[48,51,73,255]},"corner_radius":{"nw":6,"ne":6,"sw":6,"se":6},"fg_stroke":{"width":1.0,"color":[207,216,220,255]},"expansion":0.0},"hovered":{"bg_fill":[37,29,61,97],"weak_bg_fill":[95,62,97,69],"bg_stroke":{"width":1.7,"color":[106,101,155,255]},"corner_radius":{"nw":6,"ne":6,"sw":6,"se":6},"fg_stroke":{"width":1.5,"color":[83,87,88,35]},"expansion":2.0},"active":{"bg_fill":[12,12,15,255],"weak_bg_fill":[39,37,54,214],"bg_stroke":{"width":1.0,"color":[12,12,16,255]},"corner_radius":{"nw":6,"ne":6,"sw":6,"se":6},"fg_stroke":{"width":2.0,"color":[207,216,220,255]},"expansion":1.0},"open":{"bg_fill":[20,22,28,255],"weak_bg_fill":[17,18,22,255],"bg_stroke":{"width":1.8,"color":[42,44,93,165]},"corner_radius":{"nw":6,"ne":6,"sw":6,"se":6},"fg_stroke":{"width":1.0,"color":[109,109,109,255]},"expansion":0.0}},"selection":{"bg_fill":[23,64,53,27],"stroke":{"width":1.0,"color":[12,12,15,255]}},"hyperlink_color":[135,85,129,255],"faint_bg_color":[17,18,22,255],"extreme_bg_color":[9,12,15,83],"text_edit_bg_color":null,"code_bg_color":[30,31,35,255],"warn_fg_color":[61,185,157,255],"error_fg_color":[255,55,102,255],"window_corner_radius":{"nw":6,"ne":6,"sw":6,"se":6},"window_shadow":{"offset":[0,0],"blur":7,"spread":5,"color":[17,17,41,118]},"window_fill":[11,11,15,255],"window_stroke":{"width":1.0,"color":[77,94,120,138]},"window_highlight_topmost":true,"menu_corner_radius":{"nw":6,"ne":6,"sw":6,"se":6},"panel_fill":[12,12,15,255],"popup_shadow":{"offset":[0,0],"blur":8,"spread":3,"color":[19,18,18,96]},"resize_corner_size":18.0,"text_cursor":{"stroke":{"width":2.0,"color":[197,192,255,255]},"preview":true,"blink":true,"on_duration":0.5,"off_duration":0.5},"clip_rect_margin":3.0,"button_frame":true,"collapsing_header_frame":true,"indent_has_left_vline":true,"striped":true,"slider_trailing_fill":true,"handle_shape":{"Rect":{"aspect_ratio":0.5}},"interact_cursor":"Crosshair","image_loading_spinners":true,"numeric_color_space":"GammaByte","disabled_alpha":0.5},"animation_time":0.083333336,"debug":{"debug_on_hover":false,"debug_on_hover_with_all_modifiers":false,"hover_shows_next":false,"show_expand_width":false,"show_expand_height":false,"show_resize":false,"show_interactive_widgets":false,"show_widget_hits":false,"show_unaligned":true},"explanation_tooltips":false,"url_in_tooltip":false,"always_scroll_the_only_direction":true,"scroll_animation":{"points_per_second":1000.0,"duration":{"min":0.1,"max":0.3}},"compact_menu_style":true}"#;
impl crate::app::SmartMediaApp {
    pub fn receive(&mut self, ctx: &eframe::egui::Context) {
        if self.first_run {
            egui_extras::install_image_loaders(ctx);
            let ui_settings_tx = self.ui_settings_tx.clone();
            let db_ready_tx = self.db_ready_tx.clone();
            tokio::spawn(async move {
                crate::ui::status::DB_STATUS.set_state(crate::ui::status::StatusState::Initializing, "Opening DB");
                let db = crate::database::new(db_ready_tx.clone()).await;
                crate::ui::status::DB_STATUS.set_state(crate::ui::status::StatusState::Running, "Loading settings");
                log::warn!("DB: {db:?}");
                let ui_settings = crate::database::get_settings().await?;
                log::info!("Got settings: {ui_settings:?}");
                let _ = ui_settings_tx.try_send(ui_settings);
                Ok::<(), anyhow::Error>(())
            });

            self.first_run = false;
            match serde_json::from_str::<eframe::egui::Style>(STYLE) {
                Ok(mut theme) => {
                    theme.visuals.widgets.active.fg_stroke = Stroke::new(1., Color32::WHITE);
                    let style = std::sync::Arc::new(theme);
                    ctx.set_style(style);
                }
                Err(e) => log::info!("Error setting theme: {e:?}")
            };
        }

        if let Ok(ui_settings) = self.ui_settings_rx.try_recv() {
            self.ui_settings = ui_settings;
            // Keep FileExplorer's cached settings in sync with the latest from DB
            self.file_explorer.viewer.ui_settings = self.ui_settings.clone();
            // Sync atomics to current settings
            let enable_clip = self.ui_settings.auto_clip_embeddings.clone();
            crate::ai::GLOBAL_AI_ENGINE.auto_descriptions_enabled.store(self.ui_settings.auto_indexing, std::sync::atomic::Ordering::Relaxed);
            crate::ai::GLOBAL_AI_ENGINE.auto_clip_enabled.store(enable_clip, std::sync::atomic::Ordering::Relaxed);
            tokio::spawn(async move {
                let _ = crate::ai::GLOBAL_AI_ENGINE.ensure_clip_engine().await;
                if enable_clip {
                    let added = crate::ai::GLOBAL_AI_ENGINE.clip_generate_recursive().await?;
                    log::info!("[CLIP] Auto backfill scheduled on settings load: added {added}");
                }
                Ok::<(), anyhow::Error>(())
            });
            if self.ui_settings.auto_indexing && !self.ai_ready {
                self.ai_initializing = true;
                tokio::spawn(async move { crate::ai::init_global_ai_engine_async().await; });
            }
        }

        if let Ok(_) = self.db_ready_rx.try_recv() {
            self.db_ready = true;
            crate::ui::status::DB_STATUS.set_state(crate::ui::status::StatusState::Idle, "Ready");
        }

        // Global request to open settings (e.g., from navbar)
        if crate::app::OPEN_SETTINGS_REQUEST.swap(false, std::sync::atomic::Ordering::Relaxed) {
            self.open_settings_modal = true;
        }

        // Poll global AI engine readiness (cheap lock attempt)
        if !self.ai_ready && self.ai_initializing {
            use std::sync::atomic::Ordering;
            let completed = crate::ai::GLOBAL_AI_ENGINE.index_completed.load(Ordering::Relaxed);
            let active = crate::ai::GLOBAL_AI_ENGINE.index_active.load(Ordering::Relaxed);
            if completed > 0 || active > 0 {
                self.ai_ready = true;
                self.ai_initializing = false;
                crate::ui::status::JOY_STATUS.set_state(crate::ui::status::StatusState::Idle, "Ready");
            }
        }
   
        if self.open_settings_modal {
            if self.settings_draft.is_none() { self.settings_draft = Some(self.ui_settings.clone()); }
            ctx
            .show_viewport_immediate(
            ViewportId::from_hash_of("Preferences Viewport"), 
            ViewportBuilder::default(), 
            |ctx, _| {
                CentralPanel::default()
                .show(ctx, |ui| {
                    let draft_ptr: *mut crate::UiSettings = self.settings_draft.as_mut().unwrap();
                    // SAFETY: We ensure single mutable access pattern by restricting to sequential blocks.
                    let draft = unsafe { &mut *draft_ptr };
                    ui.heading("Preferences");
                    ui.separator();
                    {
                        let d = unsafe { &mut *draft_ptr };
                        ui.collapsing("AI", |ui| {
                            ui.horizontal(|ui| {
                                ui.label("Auto Indexing");
                                ui.with_layout(Layout::right_to_left(egui::Align::Center), |ui| {  
                                    ui.checkbox(&mut d.auto_indexing, "Enable");
                                });
                            });
                            ui.horizontal(|ui| {
                                ui.label("Auto CLIP Embeddings");
                                ui.with_layout(Layout::right_to_left(egui::Align::Center), |ui| {  
                                    ui.checkbox(&mut d.auto_clip_embeddings, "Enable");
                                });
                            });
                            ui.horizontal(|ui| {
                                ui.label("Blend Image + Text Embeddings");
                                ui.with_layout(Layout::right_to_left(egui::Align::Center), |ui| {  
                                    ui.checkbox(&mut d.clip_augment_with_text, "Enable");
                                });
                            });
                            ui.horizontal(|ui| {
                                ui.label("Overwrite existing CLIP embeddings (auto)");
                                ui.with_layout(Layout::right_to_left(egui::Align::Center), |ui| {  
                                    ui.checkbox(&mut d.clip_overwrite_embeddings, "Enable");
                                });
                            });
                            ui.checkbox(&mut d.overwrite_descriptions, "Overwrite existing descriptions");
                            ui.separator();
                            ui.label("CLIP Model:");
                            let mut selected = d.clip_model.clone().unwrap_or_else(|| "siglip2-large-patch16-512".to_string());
                            let before = selected.clone();
                            egui::ComboBox::new("clip-model-select", "Model")
                                .selected_text(match selected.as_str() {
                                    "unicom-vit-b32" => "Unicom ViT-B/32 (fastembed)",
                                    "clip-vit-b32" => "OpenAI CLIP ViT-B/32 (fastembed)",
                                    "siglip-base-patch16-224" => "SigLIP v1 base 224 (experimental)",
                                    "siglip2-base-patch16-224" => "SigLIP v2 base 224 (experimental)",
                                    "siglip2-base-patch16-256" => "SigLIP v2 base 256 (experimental)",
                                    "siglip2-base-patch16-384" => "SigLIP v2 base 384 (experimental)",
                                    "siglip2-base-patch16-512" => "SigLIP v2 base 512 (experimental)",
                                    "siglip2-large-patch16-256" => "SigLIP v2 large 256 (experimental)",
                                    "siglip2-large-patch16-384" => "SigLIP v2 large 384 (experimental)",
                                    "siglip2-large-patch16-512" => "SigLIP v2 large 512 (experimental)",
                                    _ => "Unicom ViT-B/32 (fastembed)",
                                })
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut selected, "unicom-vit-b32".to_string(), "Unicom ViT-B/32 (fastembed)");
                                    ui.selectable_value(&mut selected, "clip-vit-b32".to_string(), "OpenAI CLIP ViT-B/32 (fastembed)");
                                    ui.separator();
                                    ui.selectable_value(&mut selected, "siglip-base-patch16-224".to_string(), "SigLIP v1 base 224 (experimental)");
                                    ui.selectable_value(&mut selected, "siglip2-base-patch16-224".to_string(), "SigLIP v2 base 224 (experimental)");
                                    ui.selectable_value(&mut selected, "siglip2-base-patch16-256".to_string(), "SigLIP v2 base 256 (experimental)");
                                    ui.selectable_value(&mut selected, "siglip2-base-patch16-384".to_string(), "SigLIP v2 base 384 (experimental)");
                                    ui.selectable_value(&mut selected, "siglip2-base-patch16-512".to_string(), "SigLIP v2 base 512 (experimental)");
                                    ui.selectable_value(&mut selected, "siglip2-large-patch16-256".to_string(), "SigLIP v2 large 256 (experimental)");
                                    ui.selectable_value(&mut selected, "siglip2-large-patch16-384".to_string(), "SigLIP v2 large 384 (experimental)");
                                    ui.selectable_value(&mut selected, "siglip2-large-patch16-512".to_string(), "SigLIP v2 large 512 (experimental)");
                                });
                            // Write back if user changed selection
                            if selected != before {
                                log::info!("[Prefs] Changed CLIP model to: {selected}");
                                d.clip_model = Some(selected);
                            }
                            ui.vertical_centered_justified(|ui| {
                                ui.label("Prompt Template:");
                                ui.text_edit_multiline(&mut d.ai_prompt_template);
                            });
                        });
                    }
                    ui.separator();
                    {
                        let d = unsafe { &mut *draft_ptr };
                        ui.collapsing("Filters", |ui| {
                            ui.checkbox(&mut d.filter_only_with_thumb, "Only with thumbnail");
                            ui.checkbox(&mut d.filter_only_with_description, "Only with description");
                            ui.horizontal(|ui| { 
                                ui.label("Modified After (YYYY-MM-DD)"); 
                                ui.with_layout(Layout::right_to_left(egui::Align::Center), |ui| {  
                                    ui.text_edit_singleline(d.filter_modified_after.get_or_insert(String::new())); 
                                });
                            });
                            ui.horizontal(|ui| { 
                                ui.label("Modified Before"); 
                                ui.with_layout(Layout::right_to_left(egui::Align::Center), |ui| {  
                                    ui.text_edit_singleline(d.filter_modified_before.get_or_insert(String::new())); 
                                });
                            });
                        });
                    }
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button(eframe::egui::RichText::new("Save").strong()).clicked() {
                            // Persist draft and propagate to explorer cache immediately
                            self.ui_settings = draft.clone();
                            self.file_explorer.viewer.ui_settings = self.ui_settings.clone();
                            let to_save = self.ui_settings.clone();
                            tokio::spawn(async move { 
                                crate::database::save_settings(&to_save);
                                log::info!("Database save settings ");
                            });
                            // Reflect CLIP model change immediately in status hover
                            let model_key = self
                                .ui_settings
                                .clip_model
                                .clone()
                                .unwrap_or_else(|| "unicom-vit-b32".to_string());
                            CLIP_STATUS.set_model(&model_key);
                            CLIP_STATUS.set_state(StatusState::Idle, format!("Model set: {} (reload on next use)", model_key));
                            let enable_clip = self.ui_settings.auto_clip_embeddings;
                            crate::ai::GLOBAL_AI_ENGINE.auto_descriptions_enabled.store(self.ui_settings.auto_indexing, std::sync::atomic::Ordering::Relaxed);
                            crate::ai::GLOBAL_AI_ENGINE.auto_clip_enabled.store(enable_clip, std::sync::atomic::Ordering::Relaxed);
                            // Reinitialize CLIP engine with selected model next time it's needed
                            {
                                let clip = crate::ai::GLOBAL_AI_ENGINE.clip_engine.clone();
                                tokio::spawn(async move { *clip.lock().await = None; });
                            }
                            if enable_clip {
                                tokio::spawn(async move {
                                    let added = crate::ai::GLOBAL_AI_ENGINE.clip_generate_recursive().await?;
                                    log::info!("[CLIP] Auto backfill after settings save: added {added}");
                                    Ok::<(), anyhow::Error>(())
                                });
                            }
                            if self.ui_settings.auto_indexing && !self.ai_ready {
                                self.ai_initializing = true;
                                tokio::spawn(async move { crate::ai::init_global_ai_engine_async().await; });
                            }
                            self.settings_draft = None;
                            self.open_settings_modal = false;
                        }
                        // Allow forcing a CLIP engine re-init immediately without waiting for next use
                        if ui.button("Reload CLIP Now").clicked() {
                            // Clear existing engine and reinitialize with current model
                            {
                                let clip = crate::ai::GLOBAL_AI_ENGINE.clip_engine.clone();
                                tokio::spawn(async move { *clip.lock().await = None; });
                            }
                            tokio::spawn(async move {
                                match crate::ai::GLOBAL_AI_ENGINE.ensure_clip_engine().await {
                                    Ok(_) => log::info!("[CLIP] Engine reloaded successfully"),
                                    Err(e) => log::error!("[CLIP] Engine reload failed: {e}"),
                                }
                            });
                        }
                        ui.with_layout(Layout::right_to_left(egui::Align::Center), |ui| {  
                            if ui.button("Close").clicked() { 
                                self.settings_draft = None;
                                self.open_settings_modal = false; 
                            }
                            if ui.button("Generate Missing CLIP Embeddings Now").clicked() {
                                tokio::spawn(async move {
                                    let count = crate::ai::GLOBAL_AI_ENGINE.clip_generate_recursive().await?;
                                    log::info!("[CLIP] Manual generation completed for {count} images");
                                    Ok::<(), anyhow::Error>(())
                                });
                            }
                        });
                    });
                });
            });

            if !self.open_settings_modal { self.open_settings_modal = false; }
        }
    
        ctx
        .show_viewport_immediate(
        ViewportId::from_hash_of("Egui Logger"), 
        ViewportBuilder::default(), 
        |ctx, _| {
            CentralPanel::default()
            .show(ctx, |ui| {
                egui_logger::logger_ui()
                .warn_color(Color32::from_rgb(94, 215, 221)) 
                .error_color(Color32::from_rgb(255, 55, 102)) 
                .log_levels([true, true, true, false, false])
                .enable_category("eframe".to_string(), false)
                .enable_category("eframe::native::glow_integration".to_string(), false)
                .enable_category("egui_glow::shader_version".to_string(), false)
                .enable_category("egui_glow::painter".to_string(), false)
                .show(ui);
            });
        });
    }
}