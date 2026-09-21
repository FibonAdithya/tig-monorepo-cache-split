use super::{blob::Container, check_chain, dense, dense_cpu, expect_scalars, normalize_cpu, Layer};
use anyhow::{anyhow, Result};

pub struct Mlp {
    pub layers: Vec<Layer>,
    /// `Some(eps)` for TIGGAN02 blobs, whose rows are unit-normalised as the
    /// WGAN sampler does. `None` only for the TIGGAN01 fixture.
    pub normalize_eps: Option<f32>,
}

impl Mlp {
    pub fn from_container(c: Container) -> Result<Self> {
        expect_scalars(&c.scalars, 1, "mlp")?;
        if c.tensors.is_empty() || c.tensors.len() % 2 != 0 {
            return Err(anyhow!("mlp blob has {} tensors; expected a positive even count", c.tensors.len()));
        }
        let count = c.tensors.len() / 2;
        let mut it = c.tensors.into_iter();
        let mut layers = Vec::new();
        for i in 0..count {
            layers.push(dense(&mut it, &format!("mlp layer {}", i))?);
        }
        check_chain(&layers, "mlp")?;
        if layers[0].in_dim != c.latent_dim {
            return Err(anyhow!("mlp header says latent {} but layer 0 consumes {}", c.latent_dim, layers[0].in_dim));
        }
        if layers[count - 1].out_dim != c.output_dim {
            return Err(anyhow!("mlp header says output {} but the last layer produces {}", c.output_dim, layers[count - 1].out_dim));
        }
        Ok(Mlp { layers, normalize_eps: Some(c.scalars[0]) })
    }

    pub fn forward_cpu(&self, latent: &[f32]) -> Vec<f32> {
        let mut current = latent.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            current = dense_cpu(layer, &current, i + 1 < self.layers.len());
        }
        if let Some(eps) = self.normalize_eps {
            normalize_cpu(&mut current, eps);
        }
        current
    }
}
