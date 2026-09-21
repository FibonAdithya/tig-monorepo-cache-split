//! CPU-only generator code: blob parsing and one reference forward pass per
//! architecture. Deliberately outside the `c004` feature gate, like
//! `audit_sampling`, so it builds and tests on a machine with no CUDA toolkit.
//! The GPU drivers live in `vector_search::generator`.

pub mod blob;
pub mod mlp;
pub mod spherical;
pub mod structured_gate;
pub mod v1;

use anyhow::{anyhow, Result};
use blob::Tensor;
pub use mlp::Mlp;
pub use spherical::Spherical;
pub use structured_gate::StructuredGate;
pub use v1::Layer;

pub const ARCH_MLP: u32 = 0;
pub const ARCH_STRUCTURED_GATE: u32 = 1;
pub const ARCH_SPHERICAL: u32 = 2;

pub enum Generator {
    Mlp(Mlp),
    Spherical(Spherical),
    StructuredGate(StructuredGate),
}

impl Generator {
    pub fn from_blob(blob: &[u8]) -> Result<Self> {
        if blob.len() >= 8 && &blob[..8] == b"TIGGAN01" {
            // The v1 format has no normalisation step and no header dims.
            let weights = v1::parse_weights(blob)?;
            return Ok(Generator::Mlp(Mlp { layers: weights.layers, normalize_eps: None }));
        }
        let container = blob::parse_container(blob)?;
        match container.arch {
            ARCH_MLP => Ok(Generator::Mlp(Mlp::from_container(container)?)),
            ARCH_STRUCTURED_GATE => Ok(Generator::StructuredGate(StructuredGate::from_container(container)?)),
            ARCH_SPHERICAL => Ok(Generator::Spherical(Spherical::from_container(container)?)),
            other => Err(anyhow!("weight blob declares unknown arch {}", other)),
        }
    }

    pub fn latent_dim(&self) -> usize {
        match self {
            Generator::Mlp(m) => m.layers[0].in_dim,
            Generator::Spherical(s) => s.trunk[0].in_dim + s.tangent_in.in_dim,
            Generator::StructuredGate(s) => s.trunk[0].in_dim,
        }
    }

    pub fn output_dim(&self) -> usize {
        match self {
            Generator::Mlp(m) => m.layers.last().unwrap().out_dim,
            Generator::Spherical(s) => s.direction.out_dim,
            Generator::StructuredGate(s) => s.magnitude_head.out_dim,
        }
    }

    /// Reference forward pass. Exists to pin each GPU driver against PyTorch;
    /// `generate_instance` never calls it.
    ///
    /// `gate_noise` is the pre-smoothing logistic noise, one value per output
    /// coordinate: required for `StructuredGate`, refused for the others.
    pub fn forward_cpu(&self, latent: &[f32], gate_noise: Option<&[f32]>) -> Result<Vec<f32>> {
        if latent.len() != self.latent_dim() {
            return Err(anyhow!("latent has {} values; generator needs {}", latent.len(), self.latent_dim()));
        }
        match self {
            Generator::Mlp(m) => {
                if gate_noise.is_some() {
                    return Err(anyhow!("gate noise supplied to an mlp generator"));
                }
                Ok(m.forward_cpu(latent))
            }
            Generator::Spherical(s) => {
                if gate_noise.is_some() {
                    return Err(anyhow!("gate noise supplied to a spherical generator"));
                }
                Ok(s.forward_cpu(latent))
            }
            Generator::StructuredGate(s) => {
                let noise = gate_noise.ok_or_else(|| anyhow!("structured_gate needs gate noise"))?;
                if noise.len() != s.magnitude_head.out_dim {
                    return Err(anyhow!("gate noise has {} values; generator needs {}", noise.len(), s.magnitude_head.out_dim));
                }
                Ok(s.forward_cpu(latent, noise))
            }
        }
    }
}

impl std::fmt::Debug for Generator {
    /// Hand-written rather than derived: `Layer` holds megabytes of weight
    /// floats, so a derived `Debug` would dump them all on a failed
    /// assertion. Print the variant and each layer's (in_dim, out_dim) only.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Generator::Mlp(m) => f
                .debug_struct("Generator::Mlp")
                .field("layers", &m.layers.iter().map(|l| (l.in_dim, l.out_dim)).collect::<Vec<_>>())
                .field("normalize_eps", &m.normalize_eps)
                .finish(),
            Generator::Spherical(s) => f
                .debug_struct("Generator::Spherical")
                .field("trunk", &s.trunk.iter().map(|l| (l.in_dim, l.out_dim)).collect::<Vec<_>>())
                .field("direction", &(s.direction.in_dim, s.direction.out_dim))
                .field("tangent_in", &(s.tangent_in.in_dim, s.tangent_in.out_dim))
                .field("gamma", &(s.gamma.in_dim, s.gamma.out_dim))
                .field("beta", &(s.beta.in_dim, s.beta.out_dim))
                .field("tangent_out", &(s.tangent_out.in_dim, s.tangent_out.out_dim))
                .field("cos_r", &s.cos_r)
                .field("sin_r", &s.sin_r)
                .field("eps", &s.eps)
                .finish(),
            Generator::StructuredGate(s) => f
                .debug_struct("Generator::StructuredGate")
                .field("trunk", &s.trunk.iter().map(|l| (l.in_dim, l.out_dim)).collect::<Vec<_>>())
                .field("magnitude_head", &(s.magnitude_head.in_dim, s.magnitude_head.out_dim))
                .field("gate_head", &(s.gate_head.in_dim, s.gate_head.out_dim))
                .field("sparsity_head", &(s.sparsity_head.in_dim, s.sparsity_head.out_dim))
                .field("coupling", &(s.coupling.in_dim, s.coupling.out_dim))
                .field("smoothing", &(s.smoothing.in_dim, s.smoothing.out_dim))
                .field("logit_clamp", &s.logit_clamp)
                .field("magnitude_floor", &s.magnitude_floor)
                .field("eps", &s.eps)
                .finish(),
        }
    }
}

/// One dense layer, in exactly `gan_linear`'s order: start from the bias, then
/// fused multiply-adds over k = 0..in_dim. `mul_add` because the kernel uses
/// `fmaf`; a plain `a * b + c` rounds differently.
pub fn dense_cpu(layer: &Layer, input: &[f32], activate: bool) -> Vec<f32> {
    let mut out = Vec::with_capacity(layer.out_dim);
    for col in 0..layer.out_dim {
        let w = &layer.weights[col * layer.in_dim..(col + 1) * layer.in_dim];
        let mut acc = layer.bias[col];
        for k in 0..layer.in_dim {
            acc = input[k].mul_add(w[k], acc);
        }
        out.push(if activate && acc < 0.0 { acc * 0.2 } else { acc });
    }
    out
}

/// `x / max(||x||, eps)`, sum of squares accumulated in index order.
pub fn normalize_cpu(x: &mut [f32], eps: f32) {
    let mut ss = 0.0f32;
    for v in x.iter() {
        ss = v.mul_add(*v, ss);
    }
    let norm = ss.sqrt().max(eps);
    for v in x.iter_mut() {
        *v /= norm;
    }
}

// ---- helpers shared by every architecture's `from_container` ----

pub(crate) fn expect_scalars(scalars: &[f32], want: usize, arch: &str) -> Result<()> {
    if scalars.len() != want {
        return Err(anyhow!("{} blob has {} scalars; expected {}", arch, scalars.len(), want));
    }
    if scalars.iter().any(|s| !s.is_finite()) {
        return Err(anyhow!("{} blob has a non-finite scalar", arch));
    }
    Ok(())
}

pub(crate) fn next_tensor(it: &mut std::vec::IntoIter<Tensor>, what: &str) -> Result<Tensor> {
    it.next().ok_or_else(|| anyhow!("weight blob ran out of tensors at {}", what))
}

/// A weight tensor plus its `rows x 1` bias tensor.
pub(crate) fn dense(it: &mut std::vec::IntoIter<Tensor>, what: &str) -> Result<Layer> {
    let w = next_tensor(it, what)?;
    let b = next_tensor(it, what)?;
    if b.cols != 1 || b.rows != w.rows {
        return Err(anyhow!("{}: bias is {}x{} but the weight has {} rows", what, b.rows, b.cols, w.rows));
    }
    Ok(Layer { in_dim: w.cols, out_dim: w.rows, weights: w.data, bias: b.data })
}

/// A bias-free map, as a `Layer` with a zero bias so it runs through the same
/// dense code on both CPU and GPU.
pub(crate) fn no_bias(it: &mut std::vec::IntoIter<Tensor>, what: &str) -> Result<Layer> {
    let w = next_tensor(it, what)?;
    Ok(Layer { in_dim: w.cols, out_dim: w.rows, bias: vec![0.0; w.rows], weights: w.data })
}

/// Each layer must consume what the previous one produces.
pub(crate) fn check_chain(layers: &[Layer], what: &str) -> Result<()> {
    for i in 1..layers.len() {
        if layers[i].in_dim != layers[i - 1].out_dim {
            return Err(anyhow!("{} layer {} consumes {} but layer {} produces {}",
                what, i, layers[i].in_dim, i - 1, layers[i - 1].out_dim));
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::gan_generator::blob::tests::container_bytes;

    #[derive(serde::Deserialize)]
    struct Golden {
        latents: Vec<Vec<f32>>,
        gate_noise: Option<Vec<Vec<f32>>>,
        outputs: Vec<Vec<f32>>,
    }

    /// 1e-5 absolute on unit-norm outputs. PyTorch sums dot products in a
    /// different order than `dense_cpu`, so exact equality is not expected; the
    /// v1 fixture's largest observed deviation at this bound was 4.47e-7.
    /// `exact_support`: zeros must be zeros and non-zeros non-zeros (SIFT gate).
    pub(crate) fn assert_matches_golden(blob: &[u8], golden_json: &str, exact_support: bool) {
        let golden: Golden = serde_json::from_str(golden_json).unwrap();
        let generator = Generator::from_blob(blob).unwrap();
        assert_eq!(golden.latents.len(), 8, "golden file should carry 8 rows");
        for (row, (latent, expected)) in golden.latents.iter().zip(&golden.outputs).enumerate() {
            let noise = golden.gate_noise.as_ref().map(|n| n[row].as_slice());
            let got = generator.forward_cpu(latent, noise).unwrap();
            assert_eq!(got.len(), expected.len());
            for (i, (g, e)) in got.iter().zip(expected).enumerate() {
                if exact_support {
                    assert_eq!(*g == 0.0, *e == 0.0, "row {row} coord {i}: support differs (got {g}, expected {e})");
                }
                assert!((g - e).abs() < 1e-5, "row {row} coord {i}: got {g}, expected {e}");
            }
        }
    }

    #[test]
    fn glove_blob_matches_pytorch() {
        assert_matches_golden(
            include_bytes!("../vector_search/weights/glove_100_v1.bin"),
            include_str!("../vector_search/weights/glove_100_v1.golden.json"),
            false,
        );
    }

    #[test]
    fn glove_blob_declares_its_shape() {
        let g = Generator::from_blob(include_bytes!("../vector_search/weights/glove_100_v1.bin")).unwrap();
        assert_eq!((g.latent_dim(), g.output_dim()), (128, 100));
        assert!(matches!(g, Generator::Mlp(Mlp { normalize_eps: Some(e), .. }) if e == 1.0e-8));
    }

    #[test]
    fn tiggan01_still_loads_as_an_unnormalised_mlp() {
        let g = Generator::from_blob(include_bytes!("../vector_search/weights/v1_sift.bin")).unwrap();
        assert_eq!((g.latent_dim(), g.output_dim()), (128, 128));
        assert!(matches!(g, Generator::Mlp(Mlp { normalize_eps: None, .. })));
    }

    #[test]
    fn mlp_output_is_unit_norm() {
        let g = Generator::from_blob(include_bytes!("../vector_search/weights/glove_100_v1.bin")).unwrap();
        let latent: Vec<f32> = (0..128).map(|i| (i as f32) / 64.0 - 1.0).collect();
        let out = g.forward_cpu(&latent, None).unwrap();
        let norm = out.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm was {norm}");
    }

    #[test]
    fn rejects_unknown_arch() {
        let b = container_bytes(9, 4, 2, &[1e-8], &[(2, 4), (2, 1)]);
        assert!(Generator::from_blob(&b).unwrap_err().to_string().contains("arch 9"));
    }

    #[test]
    fn mlp_rejects_an_odd_tensor_count() {
        let b = container_bytes(0, 4, 2, &[1e-8], &[(2, 4)]);
        assert!(Generator::from_blob(&b).unwrap_err().to_string().contains("tensors"));
    }

    #[test]
    fn mlp_rejects_a_broken_layer_chain() {
        // layer 1 consumes 5 but layer 0 produces 3
        let b = container_bytes(0, 4, 2, &[1e-8], &[(3, 4), (3, 1), (2, 5), (2, 1)]);
        assert!(Generator::from_blob(&b).unwrap_err().to_string().contains("consumes 5"));
    }

    #[test]
    fn mlp_rejects_a_header_that_disagrees_with_the_tensors() {
        let b = container_bytes(0, 7, 2, &[1e-8], &[(2, 4), (2, 1)]);
        assert!(Generator::from_blob(&b).unwrap_err().to_string().contains("latent"));
        let b = container_bytes(0, 4, 3, &[1e-8], &[(2, 4), (2, 1)]);
        assert!(Generator::from_blob(&b).unwrap_err().to_string().contains("output"));
    }

    #[test]
    fn mlp_rejects_a_bias_of_the_wrong_shape() {
        let b = container_bytes(0, 4, 2, &[1e-8], &[(2, 4), (2, 2)]);
        assert!(Generator::from_blob(&b).unwrap_err().to_string().contains("bias"));
    }

    #[test]
    fn mlp_rejects_the_wrong_scalar_count() {
        let b = container_bytes(0, 4, 2, &[], &[(2, 4), (2, 1)]);
        assert!(Generator::from_blob(&b).unwrap_err().to_string().contains("scalars"));
    }

    #[test]
    fn gate_noise_is_refused_for_an_mlp() {
        let b = container_bytes(0, 4, 2, &[1e-8], &[(2, 4), (2, 1)]);
        let g = Generator::from_blob(&b).unwrap();
        assert!(g.forward_cpu(&[0.0; 4], Some(&[0.0; 2])).is_err());
        assert!(g.forward_cpu(&[0.0; 3], None).is_err(), "wrong latent length must be an error, not a panic");
    }

    const NYT: &[u8] = include_bytes!("../vector_search/weights/nytimes_256_v3.bin");

    #[test]
    fn nytimes_blob_matches_pytorch() {
        assert_matches_golden(NYT, include_str!("../vector_search/weights/nytimes_256_v3.golden.json"), false);
    }

    #[test]
    fn nytimes_blob_declares_its_shape() {
        let g = Generator::from_blob(NYT).unwrap();
        assert_eq!((g.latent_dim(), g.output_dim()), (512, 256));
        let Generator::Spherical(s) = &g else { panic!("expected the spherical variant") };
        assert_eq!(s.trunk[0].in_dim, 256, "trunk consumes latent_dim - skip_dim");
        assert_eq!(s.tangent_in.in_dim, 256, "tangent_in consumes skip_dim");
        assert!((s.cos_r * s.cos_r + s.sin_r * s.sin_r - 1.0).abs() < 1e-6);
        assert!(s.sin_r > 0.19 && s.cos_r > 0.07, "r lies in [0.2, 1.5], so both are positive");
    }

    #[test]
    fn spherical_output_is_unit_norm_and_the_tangent_is_orthogonal() {
        let g = Generator::from_blob(NYT).unwrap();
        let Generator::Spherical(s) = &g else { panic!() };
        let latent: Vec<f32> = (0..512).map(|i| ((i * 37 % 101) as f32) / 50.0 - 1.0).collect();
        let out = g.forward_cpu(&latent, None).unwrap();
        let norm = out.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm was {norm}");
        // out . u == cos_r exactly when t is orthogonal to u and both are unit.
        let u = s.direction_cpu(&latent);
        let dot: f32 = out.iter().zip(&u).map(|(a, b)| a * b).sum();
        assert!((dot - s.cos_r).abs() < 1e-5, "out.u was {dot}, cos_r is {}", s.cos_r);
    }

    /// `projected_tangent_norm_cpu` returns exactly what `forward_cpu` divides
    /// by, checked on the 8 golden latents.
    ///
    /// The assertion is bit-exact, not within a tolerance: the reconstruction
    /// repeats the only two steps `forward_cpu` takes after the shared helper --
    /// divide by `max(norm, eps)`, then `cos_r * u + sin_r * t` -- so both sides
    /// must come out identical bit for bit. Two mutations MEASURED as caught
    /// (2026-09-21): accumulating the sum of squares with `*x * *x + ss` instead
    /// of `mul_add` (the reconstruction then differs in the last bit at row 1
    /// coordinate 0), and iterating the accumulation in reverse index order (row 0
    /// coordinate 0).
    ///
    /// It does NOT catch a dropped projection. `forward_cpu` and this norm read
    /// the same `v` out of one shared helper, so a projection removed there moves
    /// both sides of the reconstruction together. That is what the shared helper
    /// is for; the mutation is caught by `nytimes_blob_matches_pytorch` and
    /// `spherical_output_is_unit_norm_and_the_tangent_is_orthogonal`, both MEASURED
    /// as failing when the projection line is replaced with `t[j] = t[j]`.
    ///
    /// It also pins `direction_cpu` against the helper's `u`: the two reach the
    /// direction by separate code paths and must agree bit for bit, so neither
    /// can be changed alone.
    ///
    /// What it does NOT show: nothing here bounds how small the projected norm
    /// gets on latents drawn from the real N(0, 1) prior, which is what decides
    /// whether the division can amplify a cross-architecture difference. That is
    /// a measurement over many rows, and `measure_nytimes_projected_tangent_norms`
    /// in `vector_search` (gated, `#[ignore]`d) is where it is taken. The eight
    /// golden rows' norms are 1.588, 1.598, 1.609, 1.621, 1.634, 1.647, 1.662 and
    /// 1.677 (MEASURED 2026-09-21 by this test under a temporary print, via
    /// `cargo test -p tig-challenges gan_generator -- --nocapture`), all far above
    /// the 1e-8 floor, so the `max(norm, eps)` branch is not exercised here either.
    /// Those eight latents are the exporter's own fixed sample, not draws from the
    /// prior, so they say nothing about the tail.
    #[test]
    fn nytimes_projected_tangent_norm_is_what_forward_cpu_divides_by() {
        let golden: Golden =
            serde_json::from_str(include_str!("../vector_search/weights/nytimes_256_v3.golden.json")).unwrap();
        let g = Generator::from_blob(NYT).unwrap();
        let Generator::Spherical(s) = &g else { panic!("expected the spherical variant") };
        assert_eq!(golden.latents.len(), 8, "golden file should carry 8 rows");
        for (row, latent) in golden.latents.iter().enumerate() {
            let norm = s.projected_tangent_norm_cpu(latent);
            assert!(norm.is_finite() && norm > 0.0, "row {row}: the projected norm is {norm}");

            let (u, v) = s.direction_and_projected_tangent_cpu(latent);
            let direct = s.direction_cpu(latent);
            assert_eq!(u.len(), direct.len(), "row {row}: the two directions differ in length");
            assert!(
                u.iter().zip(&direct).all(|(a, b)| a.to_bits() == b.to_bits()),
                "row {row}: direction_cpu disagrees with the shared helper's u"
            );

            let divisor = norm.max(s.eps);
            let out = s.forward_cpu(latent);
            assert_eq!(out.len(), u.len());
            for j in 0..out.len() {
                let want = s.cos_r.mul_add(u[j], s.sin_r * (v[j] / divisor));
                assert_eq!(
                    out[j].to_bits(),
                    want.to_bits(),
                    "row {row} coord {j}: forward_cpu gave {} but the reconstruction gives {}",
                    out[j],
                    want
                );
            }
        }
    }

    /// trunk 4->3, direction 2x3, tangent_in 5x4 (+b), gamma 5x3 (+b), beta 5x3 (+b), tangent_out 2x5 (+b)
    fn tiny_spherical(latent: u32, shapes: &[(u32, u32)]) -> Vec<u8> {
        container_bytes(2, latent, 2, &[0.6, 0.8, 1e-8], shapes)
    }
    const TINY_SPH: [(u32, u32); 11] =
        [(3, 4), (3, 1), (2, 3), (5, 4), (5, 1), (5, 3), (5, 1), (5, 3), (5, 1), (2, 5), (2, 1)];

    #[test]
    fn spherical_accepts_a_consistent_tiny_blob() {
        assert!(Generator::from_blob(&tiny_spherical(8, &TINY_SPH)).is_ok());
    }

    #[test]
    fn spherical_rejects_a_latent_that_is_not_trunk_plus_skip() {
        assert!(Generator::from_blob(&tiny_spherical(9, &TINY_SPH)).unwrap_err().to_string().contains("latent"));
    }

    #[test]
    fn spherical_rejects_a_gamma_that_does_not_match_tangent_in() {
        let mut shapes = TINY_SPH;
        shapes[5] = (6, 3);
        shapes[6] = (6, 1);
        assert!(Generator::from_blob(&tiny_spherical(8, &shapes)).unwrap_err().to_string().contains("gamma"));
    }

    #[test]
    fn spherical_rejects_an_even_tensor_count() {
        assert!(Generator::from_blob(&tiny_spherical(8, &TINY_SPH[..10])).unwrap_err().to_string().contains("tensors"));
    }

    const SIFT: &[u8] = include_bytes!("../vector_search/weights/sift_128_v4.bin");

    #[test]
    fn sift_v4_blob_matches_pytorch_with_an_exact_support_pattern() {
        assert_matches_golden(SIFT, include_str!("../vector_search/weights/sift_128_v4.golden.json"), true);
    }

    #[test]
    fn sift_v4_blob_declares_its_shape() {
        let g = Generator::from_blob(SIFT).unwrap();
        assert_eq!((g.latent_dim(), g.output_dim()), (128, 128));
        let Generator::StructuredGate(s) = &g else { panic!("expected the structured_gate variant") };
        assert_eq!(s.logit_clamp, 4.0, "configs/sift/v4.yaml sets logit_clamp: 4.0");
        assert_eq!(s.magnitude_floor, 1.0e-6);
        assert_eq!((s.sparsity_head.out_dim, s.coupling.in_dim, s.smoothing.out_dim), (1, 128, 128));
    }

    #[test]
    fn structured_gate_requires_noise_of_the_right_length() {
        let g = Generator::from_blob(SIFT).unwrap();
        assert!(g.forward_cpu(&[0.0; 128], None).is_err());
        assert!(g.forward_cpu(&[0.0; 128], Some(&[0.0; 127])).is_err());
    }

    #[test]
    fn structured_gate_output_is_non_negative_and_unit_norm() {
        let g = Generator::from_blob(SIFT).unwrap();
        let latent: Vec<f32> = (0..128).map(|i| ((i * 29 % 97) as f32) / 48.0 - 1.0).collect();
        let noise: Vec<f32> = (0..128).map(|i| ((i * 53 % 89) as f32) / 15.0 - 3.0).collect();
        let out = g.forward_cpu(&latent, Some(&noise)).unwrap();
        assert!(out.iter().all(|v| *v >= 0.0));
        assert!((out.iter().map(|x| x * x).sum::<f32>().sqrt() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn a_row_with_every_gate_closed_falls_back_to_the_argmax_logit() {
        let g = Generator::from_blob(SIFT).unwrap();
        let latent: Vec<f32> = (0..128).map(|i| ((i * 29 % 97) as f32) / 48.0 - 1.0).collect();
        let Generator::StructuredGate(s) = &g else { panic!("expected the structured_gate variant") };

        // Zero noise smooths to zero (the smoothing layer has no bias), so
        // gate_margin_cpu returns the logits themselves: compute the
        // expected argmax independently of forward_cpu's own fallback code.
        let margins = s.gate_margin_cpu(&latent, &[0.0; 128]);
        let max = margins.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let expected = margins.iter().position(|m| *m == max).unwrap();
        let ties = margins.iter().filter(|m| **m == max).count();
        assert_eq!(ties, 1, "the maximum logit must be unique in this row, or this test cannot tell first-argmax from last-argmax");

        // -1e6 everywhere closes every gate whatever the logits are (|logit| <= 4).
        // Smoothing is linear with positive weights, so the smoothed noise is
        // large and negative at every coordinate too.
        let out = g.forward_cpu(&latent, Some(&[-1.0e6; 128])).unwrap();
        let open: Vec<usize> = (0..128).filter(|j| out[*j] != 0.0).collect();
        assert_eq!(open.len(), 1, "exactly one coordinate is rescued");
        // This rules out argmin and any other wrong index, but a tie-free
        // row (asserted above) cannot by itself rule out a last-maximum
        // implementation agreeing with a first-maximum one by coincidence;
        // the_argmax_fallback_breaks_ties_toward_the_first_index covers that.
        assert_eq!(open[0], expected, "the rescued coordinate must be the argmax logit");
        assert!((out[open[0]] - 1.0).abs() < 1e-6, "a one-hot row normalises to 1.0");
    }

    #[test]
    fn the_argmax_fallback_breaks_ties_toward_the_first_index() {
        let zero = |in_dim: usize, out_dim: usize| Layer {
            in_dim,
            out_dim,
            weights: vec![0.0; in_dim * out_dim],
            bias: vec![0.0; out_dim],
        };
        let mut smoothing = zero(4, 4);
        for i in 0..4 {
            smoothing.weights[i * 4 + i] = 1.0; // identity: preserves the noise's sign
        }
        let s = StructuredGate {
            trunk: vec![zero(2, 2)],
            magnitude_head: zero(2, 4),
            gate_head: zero(2, 4),
            sparsity_head: zero(2, 1),
            coupling: zero(4, 4),
            smoothing,
            logit_clamp: 4.0,
            magnitude_floor: 1e-6,
            eps: 1e-8,
        };
        // Every weight and bias is zero, so every logit is exactly 0.0: a
        // four-way tie. -1e6 noise through the identity smoothing closes
        // every gate, so the fallback alone decides which coordinate opens.
        // magnitude is softplus(0) = ln 2 at every coordinate, so a one-hot
        // row still normalises to exactly 1.0.
        let out = s.forward_cpu(&[0.0, 0.0], &[-1.0e6; 4]);
        assert_eq!(out, vec![1.0, 0.0, 0.0, 0.0],
            "the fallback must break the tie toward the FIRST index, as torch.argmax does; a last-maximum implementation would open index 3");
    }

    #[test]
    fn softplus_matches_pytorch_including_the_threshold() {
        use crate::gan_generator::structured_gate::softplus;
        assert!((softplus(0.0) - std::f32::consts::LN_2).abs() < 1e-7);
        // In f32, exp(x).ln_1p() already rounds to exactly x for any x from
        // 20 up to about 88.7 (where exp(x) overflows f32 to infinity), so
        // this pins the value at x=25 but does not exercise the x > 20
        // branch: the branch and its absence agree here.
        assert_eq!(softplus(25.0), 25.0, "pins the value at x=25; does not by itself exercise the x > 20 branch");
        // exp(100.0) overflows f32 to infinity, and ln_1p(inf) is inf, so
        // this assertion only passes because the branch returns x directly.
        assert_eq!(softplus(100.0), 100.0, "without the x > 20 branch, exp(100) overflows f32 and ln_1p gives inf");
        assert!(softplus(-100.0) >= 0.0 && softplus(-100.0) < 1e-30);
    }

    #[test]
    fn structured_gate_rejects_a_sparsity_head_wider_than_one() {
        // trunk 3x4, magnitude 2x3, gate 2x3, sparsity 2x3 (WRONG), coupling 2x2, smoothing 2x2
        let shapes = [(3, 4), (3, 1), (2, 3), (2, 1), (2, 3), (2, 1), (2, 3), (2, 1), (2, 2), (2, 2)];
        let b = container_bytes(1, 4, 2, &[4.0, 1e-6, 1e-8], &shapes);
        assert!(Generator::from_blob(&b).unwrap_err().to_string().contains("sparsity"));
    }

    #[test]
    fn structured_gate_rejects_a_non_square_coupling() {
        let shapes = [(3, 4), (3, 1), (2, 3), (2, 1), (2, 3), (2, 1), (1, 3), (1, 1), (2, 3), (2, 2)];
        let b = container_bytes(1, 4, 2, &[4.0, 1e-6, 1e-8], &shapes);
        assert!(Generator::from_blob(&b).unwrap_err().to_string().contains("coupling"));
    }

    /// trunk 4->3, magnitude_head 3->2, gate_head 3->2, sparsity_head 3->1,
    /// coupling 2x2, smoothing 2x2: every check passes with these shapes and
    /// latent_dim 4 / output_dim 2, so each rejection test below changes
    /// exactly one tuple (or the header) to break exactly one check.
    const VALID_STRUCTURED_GATE: [(u32, u32); 10] =
        [(3, 4), (3, 1), (2, 3), (2, 1), (2, 3), (2, 1), (1, 3), (1, 1), (2, 2), (2, 2)];

    #[test]
    fn structured_gate_rejects_a_trunk_that_does_not_consume_latent_dim() {
        // trunk still declares in_dim 4 (shape (3, 4) is unchanged), but the
        // header now claims latent_dim 5: only the trunk-vs-latent_dim check
        // can fail. magnitude_head/gate_head/sparsity_head still match
        // hidden=3, and coupling/smoothing still match output_dim=2.
        let b = container_bytes(1, 5, 2, &[4.0, 1e-6, 1e-8], &VALID_STRUCTURED_GATE);
        assert!(Generator::from_blob(&b).unwrap_err().to_string().contains("consume latent_dim"));
    }

    #[test]
    fn structured_gate_rejects_a_magnitude_head_of_the_wrong_width() {
        // magnitude_head W/b now declare out_dim 5 instead of output_dim 2;
        // in_dim is still 3, matching hidden. trunk, gate_head,
        // sparsity_head, coupling and smoothing are all untouched and still
        // consistent with latent_dim 4 / hidden 3 / output_dim 2.
        let mut shapes = VALID_STRUCTURED_GATE;
        shapes[2] = (5, 3);
        shapes[3] = (5, 1);
        let b = container_bytes(1, 4, 2, &[4.0, 1e-6, 1e-8], &shapes);
        assert!(Generator::from_blob(&b).unwrap_err().to_string().contains("magnitude_head"));
    }

    #[test]
    fn structured_gate_rejects_a_gate_head_of_the_wrong_width() {
        // gate_head W/b now declare out_dim 5 instead of output_dim 2;
        // trunk, magnitude_head, sparsity_head, coupling and smoothing are
        // all untouched and still consistent.
        let mut shapes = VALID_STRUCTURED_GATE;
        shapes[4] = (5, 3);
        shapes[5] = (5, 1);
        let b = container_bytes(1, 4, 2, &[4.0, 1e-6, 1e-8], &shapes);
        assert!(Generator::from_blob(&b).unwrap_err().to_string().contains("gate_head"));
    }

    #[test]
    fn structured_gate_rejects_a_non_square_smoothing() {
        // smoothing is now 3x2 (in_dim 2 still matches output_dim, out_dim 3
        // does not). trunk, magnitude_head, gate_head, sparsity_head and
        // coupling are all untouched and still consistent.
        let mut shapes = VALID_STRUCTURED_GATE;
        shapes[9] = (3, 2);
        let b = container_bytes(1, 4, 2, &[4.0, 1e-6, 1e-8], &shapes);
        assert!(Generator::from_blob(&b).unwrap_err().to_string().contains("smoothing"));
    }

    #[test]
    fn structured_gate_rejects_a_non_positive_logit_clamp() {
        // shapes are the valid fixture throughout; only scalars[0]
        // (logit_clamp) is bad. NaN is already covered by expect_scalars'
        // finiteness check, so it is not exercised here.
        let zero = container_bytes(1, 4, 2, &[0.0, 1e-6, 1e-8], &VALID_STRUCTURED_GATE);
        assert!(Generator::from_blob(&zero).unwrap_err().to_string().contains("logit_clamp must be positive"));
        let negative = container_bytes(1, 4, 2, &[-4.0, 1e-6, 1e-8], &VALID_STRUCTURED_GATE);
        assert!(Generator::from_blob(&negative).unwrap_err().to_string().contains("logit_clamp must be positive"));
    }
}
