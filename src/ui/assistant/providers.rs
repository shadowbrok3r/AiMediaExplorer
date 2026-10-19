//! Direct providers: the one place that turns settings into a provider configuration, and the OpenRouter model browser the settings page can open.

use std::collections::HashSet;

use crossbeam::channel::{Receiver, Sender};
use egui::{Atoms, Button, Color32, ComboBox, Context, Layout, RichText, ScrollArea, TextEdit, Widget, Window, vec2};
use egui_json_tree::{DefaultExpand, JsonTree, JsonTreeStyle, JsonTreeVisuals, ToggleButtonsState};

use crate::UiSettings;
use crate::ai::openai_compat::{OpenRouterModel, ProviderConfig};
use crate::icons::{self, svg};

fn env_or(opt: &Option<String>, key: &str) -> Option<String> {
    opt.clone().filter(|k| !k.trim().is_empty()).or_else(|| std::env::var(key).ok())
}

/// The key, base URL and organization of `provider`.
pub fn credentials(settings: &UiSettings, provider: &str) -> (Option<String>, Option<String>, Option<String>) {
    match provider {
        "openai" => (env_or(&settings.openai_api_key, "OPENAI_API_KEY"), settings.openai_base_url.clone(), settings.openai_organization.clone()),
        "grok" => (env_or(&settings.grok_api_key, "GROK_API_KEY"), settings.openai_base_url.clone(), None),
        "gemini" => (env_or(&settings.gemini_api_key, "GEMINI_API_KEY"), settings.openai_base_url.clone(), None),
        "groq" => (env_or(&settings.groq_api_key, "GROQ_API_KEY"), settings.openai_base_url.clone(), None),
        "openrouter" => (env_or(&settings.openrouter_api_key, "OPENROUTER_API_KEY"), Some("https://openrouter.ai/api/v1".into()), None),
        _ => (env_or(&settings.openai_api_key, "OPENAI_API_KEY"), settings.openai_base_url.clone(), None),
    }
}

/// The model to ask: the override, else the shared default.
pub fn model_name(settings: &UiSettings, over: Option<&str>) -> String {
    over.map(str::to_string).or_else(|| settings.openai_default_model.clone()).unwrap_or_else(|| "gpt-5-mini".into())
}

/// The configuration for the provider chosen in `settings`, or why there is none.
pub fn provider_config(settings: &UiSettings, over: Option<&str>, temperature: f32) -> Result<ProviderConfig, String> {
    let provider = settings.ai_chat_provider.clone().unwrap_or_else(|| "local-joycaption".into());
    if provider == "local-joycaption" {
        return Err("No direct provider is chosen: pick one in Settings".to_string());
    }
    let (api_key, base_url, organization) = credentials(settings, &provider);
    if api_key.is_none() && provider != "custom" {
        return Err(format!("No API key for {provider}"));
    }
    Ok(ProviderConfig { provider, api_key, base_url, model: model_name(settings, over), organization, temperature: Some(temperature), zdr: true })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModelSort {
    NameAsc,
    NameDesc,
    PriceAsc,
    PriceDesc,
}

impl ModelSort {
    fn label(self) -> &'static str {
        match self {
            Self::NameAsc => "Name A-Z",
            Self::NameDesc => "Name Z-A",
            Self::PriceAsc => "Price low-high",
            Self::PriceDesc => "Price high-low",
        }
    }
}

/// State of the OpenRouter model browser window.
pub struct ModelBrowser {
    pub open: bool,
    filter: String,
    models: Vec<OpenRouterModel>,
    expanded: HashSet<String>,
    loading: bool,
    error: Option<String>,
    sort: ModelSort,
    in_text: bool,
    in_image: bool,
    in_audio: bool,
    out_text: bool,
    out_image: bool,
    out_audio: bool,
    free: bool,
    rx: Receiver<Result<Vec<OpenRouterModel>, String>>,
    tx: Sender<Result<Vec<OpenRouterModel>, String>>,
}

impl Default for ModelBrowser {
    fn default() -> Self {
        let (tx, rx) = crossbeam::channel::unbounded();
        Self {
            open: false,
            filter: String::new(),
            models: Vec::new(),
            expanded: HashSet::new(),
            loading: false,
            error: None,
            sort: ModelSort::NameAsc,
            in_text: false,
            in_image: false,
            in_audio: false,
            out_text: false,
            out_image: false,
            out_audio: false,
            free: false,
            rx,
            tx,
        }
    }
}

impl ModelBrowser {
    /// Opens the window and starts loading the model list.
    pub fn show_and_load(&mut self, settings: &UiSettings) {
        self.open = true;
        self.models.clear();
        self.expanded.clear();
        self.load(settings);
    }

    fn load(&mut self, settings: &UiSettings) {
        self.loading = true;
        self.error = None;
        let api_key = env_or(&settings.openrouter_api_key, "OPENROUTER_API_KEY");
        let tx = self.tx.clone();
        tokio::spawn(async move {
            let res = crate::ai::openai_compat::fetch_openrouter_models_json(api_key, Some("https://openrouter.ai/api/v1".into())).await.map_err(|e| e.to_string());
            let _ = tx.try_send(res);
        });
    }

    fn poll(&mut self, ctx: &Context) {
        if let Ok(res) = self.rx.try_recv() {
            match res {
                Ok(list) => {
                    self.models = list;
                    self.error = None;
                }
                Err(e) => {
                    log::warn!("OpenRouter models fetch failed: {e}");
                    self.error = Some(e);
                    self.models.clear();
                }
            }
            self.loading = false;
            ctx.request_repaint();
        } else if self.loading {
            ctx.request_repaint_after(std::time::Duration::from_millis(200));
        }
    }

    fn any_modality_filter(&self) -> bool {
        self.in_text || self.in_image || self.in_audio || self.out_text || self.out_image || self.out_audio
    }

    /// Models that pass the text, modality and price filters, in the chosen order.
    fn shown(&self) -> Vec<&OpenRouterModel> {
        let needle = self.filter.to_lowercase();
        let mut shown: Vec<&OpenRouterModel> = self.models.iter().filter(|m| needle.is_empty() || m.id.to_lowercase().contains(&needle)).collect();
        if self.any_modality_filter() {
            shown.retain(|m| {
                let arch = m.architecture.as_ref();
                let ins = arch.map(|a| a.input_modalities.clone()).unwrap_or_default();
                let outs = arch.map(|a| a.output_modalities.clone()).unwrap_or_default();
                let has = |list: &[String], what: &str| list.iter().any(|v| v.contains(what));
                (!self.in_text || has(&ins, "text")) && (!self.in_image || has(&ins, "image")) && (!self.in_audio || has(&ins, "audio")) && (!self.out_text || has(&outs, "text")) && (!self.out_image || has(&outs, "image")) && (!self.out_audio || has(&outs, "audio"))
            });
        }
        if self.free {
            shown.retain(|m| match &m.pricing {
                Some(p) => {
                    let zero = |s: &str| s.trim().is_empty() || parse_price(s) == Some(0.0);
                    zero(p.prompt.as_deref().unwrap_or("")) && zero(p.completion.as_deref().unwrap_or(""))
                }
                None => true,
            });
        }
        let price = |m: &OpenRouterModel, missing: f64| m.pricing.as_ref().and_then(|p| p.prompt.as_deref()).and_then(parse_price).unwrap_or(missing);
        match self.sort {
            ModelSort::NameAsc => shown.sort_by(|a, b| a.id.cmp(&b.id)),
            ModelSort::NameDesc => shown.sort_by(|a, b| b.id.cmp(&a.id)),
            ModelSort::PriceAsc => shown.sort_by(|a, b| price(a, f64::INFINITY).partial_cmp(&price(b, f64::INFINITY)).unwrap_or(std::cmp::Ordering::Equal)),
            ModelSort::PriceDesc => shown.sort_by(|a, b| price(b, -1.0).partial_cmp(&price(a, -1.0)).unwrap_or(std::cmp::Ordering::Equal)),
        }
        shown
    }
}

/// The leading number of a price text such as `$0.0000015`.
pub fn parse_price(text: &str) -> Option<f64> {
    let trimmed = text.trim().trim_start_matches('$');
    let number: String = trimmed.chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
    if number.is_empty() { None } else { number.parse().ok() }
}

/// Draws the OpenRouter model browser. Choosing a model makes it the default and a recent one in the settings and returns its id.
pub fn model_browser(ctx: &Context, state: &mut ModelBrowser, settings: &UiSettings) -> Option<String> {
    state.poll(ctx);
    if !state.open {
        return None;
    }
    let mut chosen = None;
    let mut open = state.open;
    let mut refresh = false;
    Window::new("OpenRouter Models").collapsible(true).resizable(true).open(&mut open).default_size(vec2(520.0, 420.0)).show(ctx, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label("Filter");
            ui.add(TextEdit::singleline(&mut state.filter).hint_text("name contains..."));
            ui.separator();
            ui.label("Input");
            ui.toggle_value(&mut state.in_text, "text");
            ui.toggle_value(&mut state.in_image, "image");
            ui.toggle_value(&mut state.in_audio, "audio");
            ui.separator();
            ui.label("Output");
            ui.toggle_value(&mut state.out_text, "text");
            ui.toggle_value(&mut state.out_image, "image");
            ui.toggle_value(&mut state.out_audio, "audio");
            ui.separator();
            ui.toggle_value(&mut state.free, "free");
            ui.separator();
            ui.label("Sort");
            ComboBox::from_id_salt("model_sort_combo").selected_text(state.sort.label()).show_ui(ui, |ui| {
                for (sort, icon) in [(ModelSort::NameAsc, svg::SORT_ASCENDING), (ModelSort::NameDesc, svg::SORT_DESCENDING), (ModelSort::PriceAsc, svg::SORT_ASCENDING), (ModelSort::PriceDesc, svg::SORT_DESCENDING)] {
                    ui.selectable_value(&mut state.sort, sort, icons::atoms(ui, icon, sort.label()));
                }
            });
            if icons::button(ui, svg::ARROW_CLOCKWISE, "Refresh").clicked() {
                refresh = true;
            }
        });
        ui.separator();
        let shown = state.shown();
        let (total, count) = (state.models.len(), shown.len());
        if state.loading {
            ui.label(RichText::new("Loading models...").weak());
        } else {
            ui.label(RichText::new(format!("{total} models ({count} shown)")).weak());
        }
        if let Some(err) = &state.error {
            ui.colored_label(ui.visuals().warn_fg_color, format!("Error: {err}"));
        }
        let mut toggled: Option<String> = None;
        ScrollArea::vertical().show(ui, |ui| {
            for model in shown {
                let id = &model.id;
                let is_open = state.expanded.contains(id);
                let toggle = if is_open { svg::CARET_DOWN } else { svg::CARET_RIGHT };
                let mut modalities: Vec<icons::Icon> = Vec::new();
                if let Some(a) = model.architecture.as_ref() {
                    for (name, icon) in [("text", svg::FILE_TEXT), ("image", svg::IMAGE), ("audio", svg::MUSIC_NOTES), ("video", svg::VIDEO_CAMERA)] {
                        if a.input_modalities.iter().any(|m| m.contains(name)) {
                            modalities.push(icon);
                        }
                    }
                }
                let price_suffix = match &model.pricing {
                    Some(pr) if pr.prompt.is_some() || pr.completion.is_some() => format!(" (${}/{})", pr.prompt.as_deref().filter(|p| !p.is_empty()).unwrap_or("?"), pr.completion.as_deref().filter(|p| !p.is_empty()).unwrap_or("?")),
                    _ => String::new(),
                };
                ui.horizontal(|ui| {
                    let tint = ui.visuals().text_color();
                    let edge = icons::IconSize::Sm.px(ui);
                    let accent = crate::theme::primary(ui);
                    let mut right = Atoms::new(RichText::new(price_suffix.clone()).color(accent));
                    for modality in &modalities {
                        right.push_right(modality.image(edge).tint(accent));
                    }
                    let btn = Button::selectable(is_open, (toggle.image(edge).tint(tint), id.clone())).min_size([ui.available_width() / 1.1, 15.].into()).right_text(right).ui(ui);
                    if btn.clicked() {
                        toggled = Some(id.clone());
                    }
                    ui.with_layout(Layout::right_to_left(egui::Align::Center), |ui| {
                        if icons::button(ui, svg::CHECK, "Use").on_hover_text("Set as current model").clicked() {
                            crate::settings::update("assistant:use-model", |s| {
                                s.openai_default_model = Some(id.clone());
                                s.push_recent_model(id);
                            });
                            chosen = Some(id.clone());
                        }
                        if icons::button(ui, svg::COPY, "Copy").on_hover_text("Copy model id").clicked() {
                            ui.ctx().copy_text(id.clone());
                        }
                    });
                });
                if is_open {
                    ui.add_space(4.0);
                    match serde_json::to_value(model) {
                        Ok(val) => {
                            let style = JsonTreeStyle::new().abbreviate_root(true).toggle_buttons_state(ToggleButtonsState::VisibleDisabled).visuals(JsonTreeVisuals { bool_color: Color32::LIGHT_BLUE, ..Default::default() });
                            JsonTree::new(format!("model-tree-{id}"), &val).style(style).default_expand(DefaultExpand::All).show(ui);
                        }
                        Err(_) => {
                            ui.colored_label(ui.visuals().error_fg_color, "(failed to serialize model)");
                        }
                    }
                    ui.separator();
                }
            }
        });
        if let Some(id) = toggled {
            if !state.expanded.remove(&id) {
                state.expanded.insert(id);
            }
        }
    });
    state.open = open;
    if refresh {
        state.load(settings);
    }
    chosen
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::openai_compat::{OpenRouterArchitecture, OpenRouterPricing};

    fn model(id: &str, input: &[&str], prompt: &str) -> OpenRouterModel {
        OpenRouterModel {
            id: id.into(),
            architecture: Some(OpenRouterArchitecture { input_modalities: input.iter().map(|s| s.to_string()).collect(), output_modalities: vec!["text".into()], ..Default::default() }),
            pricing: Some(OpenRouterPricing { prompt: Some(prompt.into()), completion: Some(prompt.into()), ..Default::default() }),
            ..Default::default()
        }
    }

    fn browser() -> ModelBrowser {
        let mut b = ModelBrowser::default();
        b.models = vec![model("b/vision", &["text", "image"], "$0.000002"), model("a/plain", &["text"], "$0"), model("c/pricey", &["text"], "$0.00005")];
        b
    }

    #[test]
    fn credentials_pick_the_key_base_and_organization_of_each_provider() {
        let mut s = UiSettings::default();
        s.openai_api_key = Some("k-openai".into());
        s.openrouter_api_key = Some("k-or".into());
        s.openai_organization = Some("org".into());
        s.openai_base_url = Some("http://x/v1".into());
        assert_eq!(credentials(&s, "openai"), (Some("k-openai".into()), Some("http://x/v1".into()), Some("org".into())));
        assert_eq!(credentials(&s, "openrouter"), (Some("k-or".into()), Some("https://openrouter.ai/api/v1".into()), None));
    }

    #[test]
    fn a_provider_config_needs_a_chosen_provider_and_a_key() {
        let mut s = UiSettings::default();
        s.ai_chat_provider = Some("local-joycaption".into());
        assert!(provider_config(&s, None, 0.2).unwrap_err().contains("provider"));
        s.ai_chat_provider = Some("openrouter".into());
        s.openrouter_api_key = Some("k".into());
        s.openai_default_model = Some("m/default".into());
        let cfg = provider_config(&s, None, 0.2).unwrap();
        assert_eq!((cfg.provider.as_str(), cfg.model.as_str(), cfg.temperature), ("openrouter", "m/default", Some(0.2)));
        assert_eq!(provider_config(&s, Some("m/override"), 0.5).unwrap().model, "m/override");
        s.openrouter_api_key = None;
        if std::env::var("OPENROUTER_API_KEY").is_err() {
            assert!(provider_config(&s, None, 0.2).unwrap_err().contains("openrouter"));
        }
        s.ai_chat_provider = Some("custom".into());
        s.openai_base_url = Some("http://localhost:11434/v1".into());
        assert!(provider_config(&s, None, 0.2).is_ok(), "a custom endpoint may need no key");
    }

    #[test]
    fn prices_read_their_leading_number() {
        assert_eq!(parse_price("$0.0000015"), Some(0.0000015));
        assert_eq!(parse_price(""), None);
        assert_eq!(parse_price("free"), None);
        assert_eq!(parse_price(" 2 per m"), Some(2.0));
    }

    #[test]
    fn the_browser_filters_and_sorts() {
        let mut b = browser();
        assert_eq!(b.shown().iter().map(|m| m.id.as_str()).collect::<Vec<_>>(), ["a/plain", "b/vision", "c/pricey"]);
        b.filter = "VIS".into();
        assert_eq!(b.shown().len(), 1);
        b.filter.clear();
        b.in_image = true;
        assert_eq!(b.shown().iter().map(|m| m.id.as_str()).collect::<Vec<_>>(), ["b/vision"]);
        b.in_image = false;
        b.free = true;
        assert_eq!(b.shown().iter().map(|m| m.id.as_str()).collect::<Vec<_>>(), ["a/plain"]);
        b.free = false;
        b.sort = ModelSort::PriceDesc;
        assert_eq!(b.shown()[0].id, "c/pricey");
        b.sort = ModelSort::PriceAsc;
        assert_eq!(b.shown()[0].id, "a/plain");
        b.sort = ModelSort::NameDesc;
        assert_eq!(b.shown()[0].id, "c/pricey");
    }

    #[test]
    fn the_browser_window_draws_and_can_be_closed() {
        let ctx = crate::ui::test_support::test_ctx();
        let mut b = browser();
        b.open = true;
        let settings = UiSettings::default();
        let mut drawn = Vec::new();
        for _ in 0..3 {
            let out = ctx.run_ui(egui::RawInput::default(), |ui| {
                model_browser(&ui.ctx().clone(), &mut b, &settings);
            });
            drawn = crate::ui::test_support::texts(&out.shapes);
            out.drop_without_applying_deltas();
        }
        assert!(drawn.iter().any(|(t, _)| t.contains("3 models")), "{drawn:?}");
    }
}
