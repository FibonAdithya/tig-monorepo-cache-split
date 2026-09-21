use super::{blob::Container, check_chain, dense, dense_cpu, expect_scalars, no_bias, normalize_cpu, Layer};
use anyhow::{anyhow, Result};

pub struct Spherical {
    pub trunk: Vec<Layer>,
    pub direction: Layer,
    pub tangent_in: Layer,
    pub gamma: Layer,
    pub beta: Layer,
    pub tangent_out: Layer,
    pub cos_r: f32,
    pub sin_r: f32,
    pub eps: f32,
}

impl Spherical {
    pub fn from_container(c: Container) -> Result<Self> {
        expect_scalars(&c.scalars, 3, "spherical")?;
        let n = c.tensors.len();
        if n < 11 || n % 2 == 0 {
            return Err(anyhow!("spherical blob has {} tensors; expected an odd count of at least 11", n));
        }
        let trunk_layers = (n - 9) / 2;
        let mut it = c.tensors.into_iter();
        let mut trunk = Vec::new();
        for i in 0..trunk_layers {
            trunk.push(dense(&mut it, &format!("spherical trunk layer {}", i))?);
        }
        check_chain(&trunk, "spherical trunk")?;
        let direction = no_bias(&mut it, "direction")?;
        let tangent_in = dense(&mut it, "tangent_in")?;
        let gamma = dense(&mut it, "gamma")?;
        let beta = dense(&mut it, "beta")?;
        let tangent_out = dense(&mut it, "tangent_out")?;

        let hidden = trunk[trunk_layers - 1].out_dim;
        let checks = [
            (trunk[0].in_dim + tangent_in.in_dim == c.latent_dim, "latent_dim must equal trunk input + tangent_in input"),
            (direction.in_dim == hidden, "direction must consume the trunk output"),
            (direction.out_dim == c.output_dim, "direction must produce output_dim"),
            (gamma.in_dim == hidden && gamma.out_dim == tangent_in.out_dim, "gamma must map the trunk output to tangent_in's width"),
            (beta.in_dim == hidden && beta.out_dim == tangent_in.out_dim, "beta must map the trunk output to tangent_in's width"),
            (tangent_out.in_dim == tangent_in.out_dim, "tangent_out must consume tangent_in's width"),
            (tangent_out.out_dim == c.output_dim, "tangent_out must produce output_dim"),
        ];
        for (ok, message) in checks {
            if !ok {
                return Err(anyhow!("spherical blob: {}", message));
            }
        }
        Ok(Spherical { trunk, direction, tangent_in, gamma, beta, tangent_out,
                       cos_r: c.scalars[0], sin_r: c.scalars[1], eps: c.scalars[2] })
    }

    fn trunk_cpu(&self, latent: &[f32]) -> Vec<f32> {
        let mut h = latent[..self.trunk[0].in_dim].to_vec();
        for layer in &self.trunk {
            // Activation after EVERY trunk layer, the last included: the
            // PyTorch trunk is [Linear, LeakyReLU] x T, unlike the mlp's.
            h = dense_cpu(layer, &h, true);
        }
        h
    }

    /// The unit direction `u`. Public so a test can check `out . u == cos_r`.
    pub fn direction_cpu(&self, latent: &[f32]) -> Vec<f32> {
        let mut u = dense_cpu(&self.direction, &self.trunk_cpu(latent), false);
        normalize_cpu(&mut u, self.eps);
        u
    }

    /// The unit direction `u` and the projected tangent `v = t - (t.u) u`,
    /// BEFORE `v` is normalised.
    ///
    /// Every step of `forward_cpu` up to that point lives here and nowhere
    /// else, so `forward_cpu` and `projected_tangent_norm_cpu` cannot drift
    /// apart in the arithmetic or in the `mul_add` order. Public because the
    /// ungated test reconstructs `forward_cpu`'s output from what this returns;
    /// `generate_instance` never calls it.
    pub fn direction_and_projected_tangent_cpu(&self, latent: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let h = self.trunk_cpu(latent);
        let mut u = dense_cpu(&self.direction, &h, false);
        normalize_cpu(&mut u, self.eps);

        let z_skip = &latent[self.trunk[0].in_dim..];
        let a = dense_cpu(&self.tangent_in, z_skip, false);
        let g = dense_cpu(&self.gamma, &h, false);
        let b = dense_cpu(&self.beta, &h, false);
        let modulated: Vec<f32> = (0..a.len())
            .map(|j| {
                let v = a[j].mul_add(1.0 + g[j], b[j]);
                if v < 0.0 { v * 0.2 } else { v }
            })
            .collect();
        let mut t = dense_cpu(&self.tangent_out, &modulated, false);

        let mut dot = 0.0f32;
        for j in 0..t.len() {
            dot = t[j].mul_add(u[j], dot);
        }
        for j in 0..t.len() {
            t[j] = (-dot).mul_add(u[j], t[j]);
        }
        (u, t)
    }

    /// `||v||` for the projected tangent, i.e. the quantity `forward_cpu`
    /// divides by. Reported WITHOUT the `max(_, eps)` floor, which is what
    /// makes it a measurement rather than a copy of what the normalise step
    /// uses: when `v` is nearly parallel to `u` the projection leaves almost
    /// nothing behind, and the division then amplifies a difference in the last
    /// bits of `v` by up to `1 / ||v||`. On a GPU built with `--use_fast_math`
    /// sqrt and division are approximate, so such a row is where the CPU
    /// reference and the kernel could disagree by a visible amount. Nothing
    /// asserts a bound on it; `measure_nytimes_projected_tangent_norms` in
    /// `vector_search` prints the distribution.
    ///
    /// The sum of squares is accumulated in index order with `mul_add`, exactly
    /// as `normalize_cpu` accumulates it, so this returns bit for bit the value
    /// `forward_cpu` divides by whenever that value is at or above `eps`.
    pub fn projected_tangent_norm_cpu(&self, latent: &[f32]) -> f32 {
        let (_, v) = self.direction_and_projected_tangent_cpu(latent);
        let mut ss = 0.0f32;
        for x in v.iter() {
            ss = x.mul_add(*x, ss);
        }
        ss.sqrt()
    }

    pub fn forward_cpu(&self, latent: &[f32]) -> Vec<f32> {
        let (u, mut t) = self.direction_and_projected_tangent_cpu(latent);
        normalize_cpu(&mut t, self.eps);

        (0..u.len()).map(|j| self.cos_r.mul_add(u[j], self.sin_r * t[j])).collect()
    }
}
