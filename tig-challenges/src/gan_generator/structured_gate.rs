use super::{blob::Container, check_chain, dense, dense_cpu, expect_scalars, no_bias, normalize_cpu, Layer};
use anyhow::{anyhow, Result};

pub struct StructuredGate {
    pub trunk: Vec<Layer>,
    pub magnitude_head: Layer,
    pub gate_head: Layer,
    pub sparsity_head: Layer,
    /// The Conv3d gate coupling, baked to a dense map by the exporter.
    pub coupling: Layer,
    /// The fixed noise-smoothing kernel times the per-position scale, baked.
    pub smoothing: Layer,
    pub logit_clamp: f32,
    pub magnitude_floor: f32,
    pub eps: f32,
}

/// PyTorch's `F.softplus` with its defaults: beta 1, threshold 20.
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

impl StructuredGate {
    pub fn from_container(c: Container) -> Result<Self> {
        expect_scalars(&c.scalars, 3, "structured_gate")?;
        let n = c.tensors.len();
        if n < 10 || n % 2 != 0 {
            return Err(anyhow!("structured_gate blob has {} tensors; expected an even count of at least 10", n));
        }
        let trunk_layers = (n - 8) / 2;
        let mut it = c.tensors.into_iter();
        let mut trunk = Vec::new();
        for i in 0..trunk_layers {
            trunk.push(dense(&mut it, &format!("structured_gate trunk layer {}", i))?);
        }
        check_chain(&trunk, "structured_gate trunk")?;
        let magnitude_head = dense(&mut it, "magnitude_head")?;
        let gate_head = dense(&mut it, "gate_head")?;
        let sparsity_head = dense(&mut it, "sparsity_head")?;
        let coupling = no_bias(&mut it, "coupling")?;
        let smoothing = no_bias(&mut it, "smoothing")?;

        let hidden = trunk[trunk_layers - 1].out_dim;
        let out = c.output_dim;
        let checks = [
            (trunk[0].in_dim == c.latent_dim, "the trunk must consume latent_dim"),
            (magnitude_head.in_dim == hidden && magnitude_head.out_dim == out, "magnitude_head must map the trunk output to output_dim"),
            (gate_head.in_dim == hidden && gate_head.out_dim == out, "gate_head must map the trunk output to output_dim"),
            (sparsity_head.in_dim == hidden && sparsity_head.out_dim == 1, "sparsity_head must map the trunk output to one value"),
            (coupling.in_dim == out && coupling.out_dim == out, "coupling must be output_dim x output_dim"),
            (smoothing.in_dim == out && smoothing.out_dim == out, "smoothing must be output_dim x output_dim"),
            (c.scalars[0] > 0.0, "logit_clamp must be positive"),
        ];
        for (ok, message) in checks {
            if !ok {
                return Err(anyhow!("structured_gate blob: {}", message));
            }
        }
        Ok(StructuredGate { trunk, magnitude_head, gate_head, sparsity_head, coupling, smoothing,
                            logit_clamp: c.scalars[0], magnitude_floor: c.scalars[1], eps: c.scalars[2] })
    }

    fn trunk_cpu(&self, latent: &[f32]) -> Vec<f32> {
        let mut h = latent.to_vec();
        for layer in &self.trunk {
            h = dense_cpu(layer, &h, true); // activation after every trunk layer
        }
        h
    }

    fn logits_cpu(&self, h: &[f32]) -> Vec<f32> {
        let coupled = dense_cpu(&self.coupling, &dense_cpu(&self.gate_head, h, false), false);
        let s = dense_cpu(&self.sparsity_head, h, false)[0];
        coupled.iter().map(|c| self.logit_clamp * ((c + s) / self.logit_clamp).tanh()).collect()
    }

    /// `logit_j + smoothed_noise_j`: the quantity whose sign opens gate j.
    pub fn gate_margin_cpu(&self, latent: &[f32], gate_noise: &[f32]) -> Vec<f32> {
        let logits = self.logits_cpu(&self.trunk_cpu(latent));
        let smoothed = dense_cpu(&self.smoothing, gate_noise, false);
        logits.iter().zip(&smoothed).map(|(l, n)| l + n).collect()
    }

    pub fn forward_cpu(&self, latent: &[f32], gate_noise: &[f32]) -> Vec<f32> {
        let h = self.trunk_cpu(latent);
        let logits = self.logits_cpu(&h);
        let smoothed = dense_cpu(&self.smoothing, gate_noise, false);
        let magnitude: Vec<f32> = dense_cpu(&self.magnitude_head, &h, false)
            .iter()
            .map(|m| softplus(*m).max(self.magnitude_floor))
            .collect();

        let mut out = vec![0.0f32; logits.len()];
        let mut any_open = false;
        // Strict `>` keeps the FIRST maximum, as torch.argmax does.
        let mut best = 0usize;
        for j in 0..logits.len() {
            if logits[j] > logits[best] {
                best = j;
            }
            if logits[j] + smoothed[j] > 0.0 {
                out[j] = magnitude[j];
                any_open = true;
            }
        }
        if !any_open {
            out[best] = magnitude[best];
        }
        normalize_cpu(&mut out, self.eps);
        out
    }
}
