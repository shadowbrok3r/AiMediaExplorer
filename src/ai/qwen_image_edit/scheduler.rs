use log::{debug, warn};

use super::config::SchedulerConfig;

#[derive(Debug, Clone)]
pub struct FlowMatchEulerDiscreteScheduler {
    pub num_train_timesteps: u32,
    pub prediction_type: String,
    pub timestep_spacing: String,
    // Derived schedules (training domain)
    pub betas: Vec<f32>,
    pub alphas_cumprod: Vec<f32>,
    pub sigmas_train: Vec<f32>, // sigma(t) = sqrt((1 - acp) / acp)
}

impl FlowMatchEulerDiscreteScheduler {
    pub fn from_config(cfg: &SchedulerConfig) -> Self {
        let num_train_timesteps = cfg.num_train_timesteps.unwrap_or(1000).max(1);
        // Build betas
        let betas: Vec<f32> = if let Some(tb) = &cfg.trained_betas {
            tb.clone()
        } else {
            let beta_start = cfg.beta_start.unwrap_or(0.0001f32);
            let beta_end = cfg.beta_end.unwrap_or(0.02f32);
            // We implement a simple linear schedule as a robust default
            let n = num_train_timesteps as usize;
            if n == 1 {
                vec![beta_start]
            } else {
                (0..n)
                    .map(|i| {
                        let t = i as f32 / (n as f32 - 1.0);
                        beta_start + t * (beta_end - beta_start)
                    })
                    .collect()
            }
        };
        // Alphas and cumulative product
        let mut alphas_cumprod = Vec::with_capacity(betas.len());
        let mut cum = 1.0f32;
        for &b in &betas {
            let a = (1.0f32 - b).clamp(1e-6, 1.0);
            cum *= a;
            alphas_cumprod.push(cum);
        }
        // Compute training sigmas in DDPM parameterization
        let sigmas_train: Vec<f32> = alphas_cumprod
            .iter()
            .map(|&acp| {
                let acp = acp.clamp(1e-6, 1.0);
                ((1.0 - acp) / acp).max(0.0).sqrt()
            })
            .collect();

        let prediction_type = cfg
            .prediction_type
            .clone()
            .unwrap_or_else(|| "epsilon".to_string());
        let timestep_spacing = cfg
            .timestep_spacing
            .clone()
            .unwrap_or_else(|| "linspace".to_string());

        Self {
            num_train_timesteps,
            prediction_type,
            timestep_spacing,
            betas,
            alphas_cumprod,
            sigmas_train,
        }
    }

    // Produce a sequence of discrete timesteps (indices) for inference.
    // Evenly spaced from (T-1) down to 0.
    pub fn timesteps(&self, num_inference_steps: usize) -> Vec<u32> {
        let train_t = self.num_train_timesteps as i64;
        let steps = num_inference_steps.max(1) as i64;
        let start = train_t - 1;
        let end = 0i64;
        let mut out = Vec::with_capacity(steps as usize);
        for i in 0..steps {
            let frac = if steps == 1 {
                0.0
            } else {
                i as f64 / (steps - 1) as f64
            };
            let t = ((1.0 - frac) * (start as f64) + frac * (end as f64)).round() as i64;
            out.push(t.max(0) as u32);
        }
        debug!("[flowmatch-euler] timesteps={:?}", out);
        out
    }

    // Build an inference sigma schedule by sampling the training sigmas at
    // evenly spaced (fractional) indices from high->low. Returns (steps+1) values
    // as commonly used by Euler solvers.
    pub fn inference_sigmas(&self, num_inference_steps: usize) -> Vec<f32> {
        let steps = num_inference_steps.max(1);
        let n_train = self.sigmas_train.len() as f32;
        let start = n_train - 1.0;
        let end = 0.0f32;
        let mut sigmas = Vec::with_capacity(steps + 1);
        for i in 0..steps {
            let frac = if steps == 1 {
                0.0
            } else {
                i as f32 / (steps as f32 - 1.0)
            };
            let idx_f = (1.0 - frac) * start + frac * end; // descending
            sigmas.push(self.sample_sigma_at(idx_f));
        }
        // Ensure we end at (near) zero to fully denoise
        sigmas.push(0.0);
        debug!(
            "[flowmatch-euler] inference sigmas(first,last)=({:.6},{:.6}) len={}",
            sigmas.first().copied().unwrap_or(0.0),
            sigmas.last().copied().unwrap_or(0.0),
            sigmas.len()
        );
        sigmas
    }

    fn sample_sigma_at(&self, idx_f: f32) -> f32 {
        let n = self.sigmas_train.len();
        if n == 0 {
            return 0.0;
        }
        if idx_f <= 0.0 {
            return self.sigmas_train[0];
        }
        let max_idx = (n - 1) as f32;
        if idx_f >= max_idx {
            return self.sigmas_train[n - 1];
        }
        let i0 = idx_f.floor() as usize;
        let i1 = i0 + 1;
        let t = idx_f - i0 as f32;
        let s0 = self.sigmas_train[i0];
        let s1 = self.sigmas_train[i1];
        s0 + t * (s1 - s0)
    }

    // Helper to scale inputs for networks operating in sigma-space
    // x_in = x / sqrt(sigma^2 + 1)
    pub fn scale_model_input(&self, latents: &mut [f32], sigma: f32) {
        let scale = 1.0f32 / (sigma * sigma + 1.0).sqrt();
        for v in latents.iter_mut() {
            *v *= scale;
        }
    }

    // Euler (ancestral-free) discrete step following Karras et al. style solvers:
    // x_{t-1} = x_t + (sigma_{t-1} - sigma_t) * d
    // where d approximates dx/dsigma predicted by the model.
    // For prediction_type:
    //  - "epsilon": d = model_output
    //  - "sample": model_output is x0, then epsilon = (x - x0)/sigma, d = epsilon
    //  - others: default to epsilon with a warning (once per call).
    pub fn step_euler(
        &self,
        latents: &mut [f32],
        model_output: &[f32],
        sigma_from: f32,
        sigma_to: f32,
    ) {
        assert_eq!(
            latents.len(),
            model_output.len(),
            "latents and model_output must have same length"
        );
        let ds = sigma_to - sigma_from; // note: negative (descending)
        match self.prediction_type.as_str() {
            "epsilon" => {
                for (x, &d) in latents.iter_mut().zip(model_output.iter()) {
                    *x += ds * d;
                }
            }
            "sample" => {
                // model_output ~ x0, derive epsilon and use as derivative
                let s = sigma_from.max(1e-6);
                for (x, &x0) in latents.iter_mut().zip(model_output.iter()) {
                    let eps = (*x - x0) / s;
                    *x += ds * eps;
                }
            }
            other => {
                warn!(
                    "[flowmatch-euler] unsupported prediction_type='{}', treating as 'epsilon'",
                    other
                );
                for (x, &d) in latents.iter_mut().zip(model_output.iter()) {
                    *x += ds * d;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timesteps_basic() {
        let s = FlowMatchEulerDiscreteScheduler::from_config(&SchedulerConfig {
            num_train_timesteps: Some(1000),
            ..Default::default()
        });
        let ts = s.timesteps(5);
        assert_eq!(ts.len(), 5);
        // Should start near 999 and end near 0
        assert!(ts[0] >= 900);
        assert!(ts.last().copied().unwrap() <= 10);
    }

    #[test]
    fn test_inference_sigmas_monotonic() {
        let s = FlowMatchEulerDiscreteScheduler::from_config(&SchedulerConfig {
            num_train_timesteps: Some(1000),
            ..Default::default()
        });
        let sigmas = s.inference_sigmas(10);
        assert_eq!(sigmas.len(), 11);
        // strictly non-increasing
        for i in 1..sigmas.len() {
            assert!(
                sigmas[i] <= sigmas[i - 1] + 1e-6,
                "not non-increasing at {}",
                i
            );
        }
        // last is zero
        assert!(sigmas.last().unwrap().abs() <= 1e-6);
    }

    #[test]
    fn test_step_euler_shapes() {
        let s = FlowMatchEulerDiscreteScheduler::from_config(&SchedulerConfig::default());
        let mut x = vec![0.5f32; 8];
        let d = vec![1.0f32; 8];
        let sigma_from = 1.0f32;
        let sigma_to = 0.8f32;
        // ds = -0.2, epsilon pred -> x += ds * d
        s.step_euler(&mut x, &d, sigma_from, sigma_to);
        for &v in &x {
            assert!((v - 0.3).abs() < 1e-6);
        }
    }
}
