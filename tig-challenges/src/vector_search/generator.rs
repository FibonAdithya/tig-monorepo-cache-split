use anyhow::{anyhow, Result};

/// Latent dimensionality the generator samples from.
pub const LATENT_DIM: usize = 128;

const MAGIC: &[u8; 8] = b"TIGGAN01";

/// Weights are committed rather than fetched: every verifier regenerates the
/// instance independently and compares a fixed-point quality integer exactly,
/// so a single differing byte would fail verification network-wide.
pub(super) const V1_BLOB: &[u8] = include_bytes!("weights/v1_sift.bin");

pub struct Layer {
    pub in_dim: usize,
    pub out_dim: usize,
    /// Row-major `[out_dim][in_dim]`, matching PyTorch `nn.Linear.weight`, so
    /// the kernel indexes `weight + col * in_dim` without a transpose.
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}

pub struct GeneratorWeights {
    pub layers: Vec<Layer>,
}

fn take<'a>(blob: &'a [u8], at: &mut usize, len: usize) -> Result<&'a [u8]> {
    let end = at
        .checked_add(len)
        .ok_or_else(|| anyhow!("weight blob length overflow"))?;
    if end > blob.len() {
        return Err(anyhow!(
            "weight blob truncated: needed {} bytes at offset {}, have {}",
            len,
            at,
            blob.len()
        ));
    }
    let out = &blob[*at..end];
    *at = end;
    Ok(out)
}

fn read_u32(blob: &[u8], at: &mut usize) -> Result<u32> {
    let b = take(blob, at, 4)?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn read_f32s(blob: &[u8], at: &mut usize, count: usize) -> Result<Vec<f32>> {
    let byte_len = count
        .checked_mul(4)
        .ok_or_else(|| anyhow!("weight blob element count overflow: {} floats", count))?;
    let b = take(blob, at, byte_len)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

pub fn parse_weights(blob: &[u8]) -> Result<GeneratorWeights> {
    let mut at = 0usize;
    if take(blob, &mut at, 8)? != MAGIC {
        return Err(anyhow!("weight blob has wrong magic; expected TIGGAN01"));
    }
    let num_layers = read_u32(blob, &mut at)? as usize;
    if num_layers == 0 {
        return Err(anyhow!("weight blob declares zero layers"));
    }

    // Not pre-reserving `num_layers` capacity: it comes straight from the
    // blob and hasn't been checked against the remaining byte count yet, so
    // a hostile value (e.g. near `u32::MAX`) must not trigger a huge
    // allocation before the per-layer reads below have a chance to fail.
    let mut layers = Vec::new();
    for i in 0..num_layers {
        let in_dim = read_u32(blob, &mut at)? as usize;
        let out_dim = read_u32(blob, &mut at)? as usize;
        if in_dim == 0 || out_dim == 0 {
            return Err(anyhow!("layer {} has a zero dimension", i));
        }
        let weight_count = in_dim.checked_mul(out_dim).ok_or_else(|| {
            anyhow!(
                "layer {} weight count overflow: {} * {}",
                i,
                in_dim,
                out_dim
            )
        })?;
        let weights = read_f32s(blob, &mut at, weight_count)?;
        let bias = read_f32s(blob, &mut at, out_dim)?;
        layers.push(Layer { in_dim, out_dim, weights, bias });
    }

    // Trailing bytes mean the blob and this parser disagree about the format,
    // which is worth failing loudly rather than silently ignoring.
    if at != blob.len() {
        return Err(anyhow!(
            "weight blob has {} trailing bytes",
            blob.len() - at
        ));
    }
    Ok(GeneratorWeights { layers })
}

pub fn v1_weights() -> Result<GeneratorWeights> {
    parse_weights(V1_BLOB)
}

/// Reference forward pass, matching `gan_linear`'s accumulation order exactly.
///
/// Exists to pin the port against PyTorch in tests. `generate_instance` never
/// calls it — instances are always produced on the GPU.
pub fn forward_cpu(weights: &GeneratorWeights, latent: &[f32]) -> Vec<f32> {
    let mut current = latent.to_vec();
    for (i, layer) in weights.layers.iter().enumerate() {
        let apply_activation = i + 1 < weights.layers.len();
        let mut next = Vec::with_capacity(layer.out_dim);
        for col in 0..layer.out_dim {
            let w = &layer.weights[col * layer.in_dim..(col + 1) * layer.in_dim];
            let mut acc = layer.bias[col];
            for k in 0..layer.in_dim {
                acc = current[k].mul_add(w[k], acc);
            }
            next.push(if apply_activation && acc < 0.0 { acc * 0.2 } else { acc });
        }
        current = next;
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blob_with(layers: &[(u32, u32)]) -> Vec<u8> {
        let mut v = Vec::from(*b"TIGGAN01");
        v.extend_from_slice(&(layers.len() as u32).to_le_bytes());
        for &(in_dim, out_dim) in layers {
            v.extend_from_slice(&in_dim.to_le_bytes());
            v.extend_from_slice(&out_dim.to_le_bytes());
            for _ in 0..(in_dim * out_dim + out_dim) {
                v.extend_from_slice(&1.5f32.to_le_bytes());
            }
        }
        v
    }

    #[test]
    fn parses_layer_dims_and_values() {
        let g = parse_weights(&blob_with(&[(2, 3)])).unwrap();
        assert_eq!(g.layers.len(), 1);
        assert_eq!(g.layers[0].in_dim, 2);
        assert_eq!(g.layers[0].out_dim, 3);
        assert_eq!(g.layers[0].weights.len(), 6);
        assert_eq!(g.layers[0].bias.len(), 3);
        assert_eq!(g.layers[0].weights[0], 1.5);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut b = blob_with(&[(2, 3)]);
        b[0] = b'X';
        assert!(parse_weights(&b).is_err());
    }

    #[test]
    fn rejects_truncated_blob() {
        let b = blob_with(&[(2, 3)]);
        assert!(parse_weights(&b[..b.len() - 4]).is_err());
    }

    #[test]
    fn rejects_trailing_bytes() {
        let mut b = blob_with(&[(2, 3)]);
        b.push(0);
        assert!(parse_weights(&b).is_err());
    }

    #[test]
    fn rejects_layer_dims_that_overflow_element_count() {
        // Built by hand rather than via `blob_with`: that helper itself
        // computes `in_dim * out_dim` as a plain u32 multiplication to size
        // the body, which would panic on these dimensions before
        // `parse_weights` ever saw the blob. `in_dim * out_dim * 4` (the
        // byte length of the weight matrix) overflows `usize` even though
        // `in_dim * out_dim` alone does not, so this exercises the
        // `checked_mul` in `read_f32s`.
        let mut b = Vec::from(*b"TIGGAN01");
        b.extend_from_slice(&1u32.to_le_bytes()); // num_layers
        b.extend_from_slice(&u32::MAX.to_le_bytes()); // in_dim
        b.extend_from_slice(&u32::MAX.to_le_bytes()); // out_dim
        assert!(parse_weights(&b).is_err());
    }

    #[test]
    fn rejects_absurd_layer_count() {
        // Declares billions of layers with no data to back them. Must
        // return a clean Err (via the truncation check on the very first
        // layer) rather than pre-allocating a huge Vec for `num_layers`.
        let mut b = Vec::from(*b"TIGGAN01");
        b.extend_from_slice(&u32::MAX.to_le_bytes()); // num_layers
        assert!(parse_weights(&b).is_err());
    }

    #[test]
    fn embedded_v1_blob_has_expected_shape() {
        let g = v1_weights().unwrap();
        let dims: Vec<(usize, usize)> =
            g.layers.iter().map(|l| (l.in_dim, l.out_dim)).collect();
        assert_eq!(dims, vec![(128, 512), (512, 1024), (1024, 1024), (1024, 128)]);
        assert_eq!(g.layers[0].in_dim, LATENT_DIM);
    }

    #[test]
    fn cpu_forward_matches_pytorch_golden_vectors() {
        #[derive(serde::Deserialize)]
        struct Golden {
            latents: Vec<Vec<f32>>,
            outputs: Vec<Vec<f32>>,
        }
        let golden: Golden = serde_json::from_str(include_str!(
            "weights/golden_vectors.json"
        ))
        .unwrap();

        let weights = v1_weights().unwrap();
        for (latent, expected) in golden.latents.iter().zip(&golden.outputs) {
            let got = forward_cpu(&weights, latent);
            assert_eq!(got.len(), expected.len());
            for (i, (g, e)) in got.iter().zip(expected).enumerate() {
                // PyTorch sums a dot product in a different order than our
                // fixed sequential loop, so exact equality is not expected;
                // this bound catches a wrong transpose, a missing bias or a
                // misapplied activation, which is what the test is for.
                //
                // 1e-5 against a largest observed deviation of 4.47e-7: over 20x
                // headroom on reordering error, but tight enough to fail on a
                // subtle arithmetic regression. At the original 1e-4 the margin
                // was ~200x and the test would have passed through real damage.
                assert!(
                    (g - e).abs() < 1e-5,
                    "coordinate {i}: got {g}, expected {e}"
                );
            }
        }
    }
}
