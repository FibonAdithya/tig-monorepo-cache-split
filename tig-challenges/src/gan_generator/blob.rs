use super::v1::{read_f32s, read_u32, take};
use anyhow::{anyhow, Result};

const MAGIC: &[u8; 8] = b"TIGGAN02";

/// Upper bounds checked BEFORE the counts are used as loop bounds. The largest
/// real architecture has 3 scalars and 15 tensors. Without these a hostile
/// count would still fail on truncation, but only after a very long loop of
/// failing reads is ruled out by inspection; an explicit bound needs no such
/// argument.
pub const MAX_SCALARS: usize = 16;
pub const MAX_TENSORS: usize = 64;

pub struct Tensor {
    pub rows: usize,
    pub cols: usize,
    /// Row-major `[rows][cols]`, matching PyTorch `nn.Linear.weight`.
    pub data: Vec<f32>,
}

impl std::fmt::Debug for Tensor {
    /// Hand-written rather than derived, for the same reason `Generator`'s is:
    /// `data` holds up to a million weight floats, so a derived `Debug` would
    /// dump them all when an assertion on a real blob fails. Print the shape
    /// and the element count only. `Container`'s derived `Debug` is bounded by
    /// this one, since its `tensors` field is the only large thing in it.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("data_len", &self.data.len())
            .finish()
    }
}

#[derive(Debug)]
pub struct Container {
    pub arch: u32,
    pub latent_dim: usize,
    pub output_dim: usize,
    pub scalars: Vec<f32>,
    pub tensors: Vec<Tensor>,
}

pub fn parse_container(blob: &[u8]) -> Result<Container> {
    let mut at = 0usize;
    if take(blob, &mut at, 8)? != MAGIC {
        return Err(anyhow!("weight blob has wrong magic; expected TIGGAN02"));
    }
    let arch = read_u32(blob, &mut at)?;
    let latent_dim = read_u32(blob, &mut at)? as usize;
    let output_dim = read_u32(blob, &mut at)? as usize;
    if latent_dim == 0 || output_dim == 0 {
        return Err(anyhow!("weight blob declares a zero latent or output dimension"));
    }

    let n_scalars = read_u32(blob, &mut at)? as usize;
    if n_scalars > MAX_SCALARS {
        return Err(anyhow!("weight blob declares {} scalars; at most {} allowed", n_scalars, MAX_SCALARS));
    }
    let scalars = read_f32s(blob, &mut at, n_scalars)?;

    let n_tensors = read_u32(blob, &mut at)? as usize;
    if n_tensors > MAX_TENSORS {
        return Err(anyhow!("weight blob declares {} tensors; at most {} allowed", n_tensors, MAX_TENSORS));
    }
    let mut tensors = Vec::new();
    for i in 0..n_tensors {
        let rows = read_u32(blob, &mut at)? as usize;
        let cols = read_u32(blob, &mut at)? as usize;
        if rows == 0 || cols == 0 {
            return Err(anyhow!("tensor {} has a zero dimension", i));
        }
        let count = rows
            .checked_mul(cols)
            .ok_or_else(|| anyhow!("tensor {} element count overflow: {} * {}", i, rows, cols))?;
        let data = read_f32s(blob, &mut at, count)?;
        tensors.push(Tensor { rows, cols, data });
    }
    if at != blob.len() {
        return Err(anyhow!("weight blob has {} trailing bytes", blob.len() - at));
    }
    Ok(Container { arch, latent_dim, output_dim, scalars, tensors })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A well-formed container whose tensor values are a small deterministic
    /// ramp, so tests that check values are not comparing constants.
    pub(crate) fn container_bytes(arch: u32, latent: u32, out: u32,
                                  scalars: &[f32], tensors: &[(u32, u32)]) -> Vec<u8> {
        let mut v = Vec::from(*b"TIGGAN02");
        for x in [arch, latent, out, scalars.len() as u32] { v.extend_from_slice(&x.to_le_bytes()); }
        for s in scalars { v.extend_from_slice(&s.to_le_bytes()); }
        v.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
        let mut n = 0u32;
        for &(rows, cols) in tensors {
            v.extend_from_slice(&rows.to_le_bytes());
            v.extend_from_slice(&cols.to_le_bytes());
            for _ in 0..rows * cols {
                v.extend_from_slice(&(((n % 17) as f32 - 8.0) * 0.0625).to_le_bytes());
                n += 1;
            }
        }
        v
    }

    #[test]
    fn parses_header_scalars_and_tensors() {
        let c = parse_container(&container_bytes(2, 24, 8, &[0.5, 0.25], &[(3, 2), (3, 1)])).unwrap();
        assert_eq!((c.arch, c.latent_dim, c.output_dim), (2, 24, 8));
        assert_eq!(c.scalars, vec![0.5, 0.25]);
        assert_eq!((c.tensors[0].rows, c.tensors[0].cols), (3, 2));
        assert_eq!(c.tensors[0].data, vec![-0.5, -0.4375, -0.375, -0.3125, -0.25, -0.1875]);
        assert_eq!(c.tensors[1].data[0], -0.125, "second tensor continues the ramp");
    }

    #[test]
    fn rejects_bad_magic() {
        let mut b = container_bytes(0, 4, 2, &[1e-8], &[(2, 4), (2, 1)]);
        b[7] = b'3';
        assert!(parse_container(&b).unwrap_err().to_string().contains("magic"));
    }

    #[test]
    fn rejects_truncation() {
        let b = container_bytes(0, 4, 2, &[1e-8], &[(2, 4), (2, 1)]);
        assert!(parse_container(&b[..b.len() - 4]).unwrap_err().to_string().contains("truncated"));
    }

    #[test]
    fn rejects_trailing_bytes() {
        let mut b = container_bytes(0, 4, 2, &[1e-8], &[(2, 4), (2, 1)]);
        b.push(0);
        assert!(parse_container(&b).unwrap_err().to_string().contains("trailing"));
    }

    #[test]
    fn rejects_absurd_scalar_count_before_reading_them() {
        let mut b = Vec::from(*b"TIGGAN02");
        for x in [0u32, 4, 2, u32::MAX] { b.extend_from_slice(&x.to_le_bytes()); }
        assert!(parse_container(&b).unwrap_err().to_string().contains("scalars"));
    }

    #[test]
    fn rejects_absurd_tensor_count_before_reading_them() {
        let mut b = Vec::from(*b"TIGGAN02");
        for x in [0u32, 4, 2, 0, u32::MAX] { b.extend_from_slice(&x.to_le_bytes()); }
        assert!(parse_container(&b).unwrap_err().to_string().contains("tensors"));
    }

    #[test]
    fn rejects_tensor_dims_that_overflow_the_element_count() {
        // Which multiplication fires depends on the width of `usize`. On a
        // 64-bit target (2^32-1)^2 = 2^64 - 2^33 + 1 fits in a usize, so the
        // `rows.checked_mul(cols)` above cannot overflow; what fires is
        // `read_f32s`' `count.checked_mul(4)`, the byte length of the tensor.
        // The container-level `checked_mul` is live only where usize is 32
        // bits. Both errors say "overflow", which is all this asserts; it
        // deliberately does not name one of the two messages, because that
        // would pin the test to one pointer width.
        let mut b = Vec::from(*b"TIGGAN02");
        for x in [0u32, 4, 2, 0, 1, u32::MAX, u32::MAX] { b.extend_from_slice(&x.to_le_bytes()); }
        assert!(parse_container(&b).unwrap_err().to_string().contains("overflow"));
    }

    #[test]
    fn rejects_a_zero_dimension() {
        let b = container_bytes(0, 4, 2, &[1e-8], &[(0, 4)]);
        assert!(parse_container(&b).unwrap_err().to_string().contains("zero"));
    }

    #[test]
    fn rejects_zero_latent_or_output_dim() {
        let b = container_bytes(0, 0, 2, &[1e-8], &[(2, 4), (2, 1)]);
        assert!(parse_container(&b).unwrap_err().to_string().contains("zero"));
    }
}
