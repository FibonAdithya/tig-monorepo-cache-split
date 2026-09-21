//! CPU-only generator code: blob parsing and one reference forward pass per
//! architecture. Deliberately outside the `c004` feature gate, like
//! `audit_sampling`, so it builds and tests on a machine with no CUDA toolkit.
//! The GPU drivers live in `vector_search::generator`.

pub mod blob;
pub mod v1;

pub use v1::Layer;
