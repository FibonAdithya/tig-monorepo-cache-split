//! CPU-only generator code: blob parsing and one reference forward pass per
//! architecture. Deliberately outside the `c004` feature gate, like
//! `audit_sampling`, so it builds and tests on a machine with no CUDA toolkit.
//! The GPU drivers live in `vector_search::generator`.

pub mod blob;
pub mod mlp;
pub mod v1;

use anyhow::{anyhow, Result};
use blob::Tensor;
pub use mlp::Mlp;
pub use v1::Layer;

pub const ARCH_MLP: u32 = 0;

pub enum Generator {
    Mlp(Mlp),
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
            other => Err(anyhow!("weight blob declares unknown arch {}", other)),
        }
    }

    pub fn latent_dim(&self) -> usize {
        match self {
            Generator::Mlp(m) => m.layers[0].in_dim,
        }
    }

    pub fn output_dim(&self) -> usize {
        match self {
            Generator::Mlp(m) => m.layers.last().unwrap().out_dim,
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
///
/// `Mlp` (this task) has no bias-free layer, so nothing in-crate calls this
/// yet; Tasks 4 and 5 add the architectures that do.
#[allow(dead_code)]
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
}
