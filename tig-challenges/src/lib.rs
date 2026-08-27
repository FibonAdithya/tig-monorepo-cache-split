pub const BUILD_TIME_PATH: &str = env!("CARGO_MANIFEST_DIR");

const QUALITY_PRECISION: i32 = 1_000_000;

macro_rules! conditional_pub {
    (fn $name:ident $($rest:tt)*) => {
        #[cfg(not(feature = "hide_verification"))]
        pub fn $name $($rest)*

        // `dead_code` allowed on this arm only. Under `hide_verification` these
        // functions are private with no in-crate caller BY CONSTRUCTION -- that
        // is what the feature is for -- so the lint fires on every one of them
        // in every algorithm build that links this crate, along with everything
        // reachable only through them. Real dead code still warns in the
        // ungated and per-challenge builds, where the arm above applies.
        #[cfg(feature = "hide_verification")]
        #[allow(dead_code)]
        fn $name $($rest)*
    };
}

macro_rules! impl_kv_string_serde {
    ($name:ident { $( $field:ident : $ty:ty ),* $(,)? }) => {
        paste::paste! {
            #[derive(Debug, Clone, PartialEq)]
            pub struct $name {
                $( pub $field : $ty ),*
            }

            impl serde::Serialize for $name {
                fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
                where
                    S: serde::Serializer,
                {
                    let mut parts = Vec::new();
                    $(
                        parts.push(format!("{}={}", stringify!($field), self.$field));
                    )*
                    // optional: sort keys for deterministic output
                    parts.sort();
                    let s = parts.join(",");
                    serializer.serialize_str(&s)
                }
            }

            impl<'de> serde::Deserialize<'de> for $name {
                fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
                where
                    D: serde::Deserializer<'de>
                {
                    use serde::de::{Visitor, Error};
                    use std::fmt;

                    struct VisitorImpl;

                    impl<'de> Visitor<'de> for VisitorImpl {
                        type Value = $name;

                        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            write!(f, "a string of the form 'key=value,key=value'")
                        }

                        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
                        where
                            E: Error,
                        {
                            let mut map = std::collections::HashMap::new();

                            if !v.is_empty() {
                                for part in v.split(',') {
                                    let mut kv = part.splitn(2, '=');
                                    let key = kv.next().ok_or_else(|| E::custom(format!("Missing key in '{}'", part)))?;
                                    let val = kv.next().ok_or_else(|| E::custom(format!("Missing value in '{}'", part)))?;
                                    map.insert(key, val);
                                }
                            }

                            Ok($name {
                                $(
                                    $field: map.get(stringify!($field))
                                        .ok_or_else(|| E::custom(format!("Missing field '{}'", stringify!($field))))?
                                        .parse::<$ty>()
                                        .map_err(E::custom)?,
                                )*
                            })
                        }
                    }

                    deserializer.deserialize_str(VisitorImpl)
                }
            }
        }
    };
}

macro_rules! impl_base64_serde {
    ($name:ident { $( $field:ident : $ty:ty ),* $(,)? }) => {
        paste::paste! {
            #[derive(Debug, Clone)]
            pub struct $name {
                $( pub $field : $ty ),*
            }

            #[derive(serde::Serialize, serde::Deserialize)]
            struct [<$name Data>] {
                $( $field : $ty ),*
            }

            impl serde::Serialize for $name {
                fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
                where
                    S: serde::Serializer,
                {
                    use flate2::{write::GzEncoder, Compression};
                    use base64::engine::general_purpose::STANDARD as BASE64;
                    use base64::Engine;
                    use std::io::Write;

                    let helper = [<$name Data>] {
                        $( $field: self.$field.clone() ),*
                    };

                    let bincode_data = bincode::serialize(&helper)
                        .map_err(|e| serde::ser::Error::custom(format!("Bincode serialization failed: {}", e)))?;

                    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
                    encoder
                        .write_all(&bincode_data)
                        .map_err(|e| serde::ser::Error::custom(format!("Compression failed: {}", e)))?;
                    let compressed_data = encoder
                        .finish()
                        .map_err(|e| serde::ser::Error::custom(format!("Compression finish failed: {}", e)))?;

                    let encoded = BASE64.encode(&compressed_data);
                    serializer.serialize_str(&encoded)
                }
            }

            impl<'de> serde::Deserialize<'de> for $name {
                fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
                where
                    D: serde::Deserializer<'de>,
                {
                    use flate2::read::GzDecoder;
                    use base64::engine::general_purpose::STANDARD as BASE64;
                    use base64::Engine;
                    use std::io::Read;
                    use std::fmt;

                    struct VisitorImpl;

                    impl<'de> serde::de::Visitor<'de> for VisitorImpl {
                        type Value = $name;

                        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            write!(f, "a base64 encoded, compressed, bincode serialized {}", stringify!($name))
                        }

                        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
                        where
                            E: serde::de::Error,
                        {
                            let compressed = BASE64.decode(v)
                                .map_err(|e| E::custom(format!("Base64 decode failed: {}", e)))?;

                            let mut decoder = GzDecoder::new(&compressed[..]);
                            let mut decompressed = Vec::new();
                            decoder
                                .read_to_end(&mut decompressed)
                                .map_err(|e| E::custom(format!("Decompression failed: {}", e)))?;

                            let data: [<$name Data>] = bincode::deserialize(&decompressed)
                                .map_err(|e| E::custom(format!("Bincode deserialization failed: {}", e)))?;

                            Ok($name {
                                $( $field: data.$field ),*
                            })
                        }
                    }

                    deserializer.deserialize_str(VisitorImpl)
                }
            }
        }
    };
}

#[cfg(feature = "c001")]
pub mod satisfiability;
#[cfg(feature = "c001")]
pub use satisfiability as c001;
#[cfg(feature = "c002")]
pub mod vehicle_routing;
#[cfg(feature = "c002")]
pub use vehicle_routing as c002;
#[cfg(feature = "c003")]
pub mod knapsack;
#[cfg(feature = "c003")]
pub use knapsack as c003;
pub mod audit_sampling;
#[cfg(feature = "c004")]
pub mod vector_search;
#[cfg(feature = "c004")]
pub use vector_search as c004;
#[cfg(feature = "c005")]
pub mod hypergraph;
#[cfg(feature = "c005")]
pub use hypergraph as c005;
#[cfg(feature = "c006")]
pub mod neuralnet_optimizer;
#[cfg(feature = "c006")]
pub use neuralnet_optimizer as c006;
#[cfg(feature = "c007")]
pub mod job_scheduling;
#[cfg(feature = "c007")]
pub use job_scheduling as c007;
#[cfg(feature = "c008")]
pub mod energy_arbitrage;
#[cfg(feature = "c008")]
pub use energy_arbitrage as c008;

/// The one seam in c004 that no compiler checks: `AUDIT_BLOCK`, `AUDIT_TQ` and
/// `AUDIT_MAX_DIMS` each exist twice, once as a `#define` in
/// `vector_search/kernels.cu` and once as a Rust `const` in
/// `vector_search/mod.rs`, and the two copies must agree. A mismatch is silent:
/// too small an `AUDIT_BLOCK` leaves tile rows unexamined and reports recall
/// *higher* than reality, and a mismatched `AUDIT_TQ` leaves the tail of the
/// sample list unaudited. Each doc comment on the Rust side says "must equal
/// X in kernels.cu"; this is the check behind that sentence.
///
/// Deliberately ungated, and deliberately reading both files as text. The Rust
/// consts live behind `#[cfg(feature = "c004")]`, and that build pulls in
/// cudarc, whose build script requires nvcc -- so a c004-gated test cannot even
/// compile on a machine without CUDA, which is precisely where an unchecked
/// seam would go unnoticed. Parsing the sources costs the ability to name the
/// consts directly and buys a test that runs everywhere.
///
/// Inside `#[cfg(test)]` so the two sources are not embedded in the shipped
/// library -- `include_str!` in a non-test item would put them in every
/// algorithm `.so` that links this crate.
#[cfg(test)]
mod audit_constant_agreement_tests {
    const KERNELS_CU: &str = include_str!("vector_search/kernels.cu");
    const VECTOR_SEARCH_RS: &str = include_str!("vector_search/mod.rs");

    /// The value of `#define <name> <integer>` in kernels.cu.
    ///
    /// Panics rather than returning an Option: a lookup that quietly found
    /// nothing would turn the assertion below into a comparison of two
    /// absences, which is the shape of a test that passes after the thing it
    /// guards has been deleted.
    fn cu_define(name: &str) -> u32 {
        let prefix = format!("#define {} ", name);
        let matches: Vec<&str> = KERNELS_CU
            .lines()
            .map(str::trim)
            .filter(|l| l.starts_with(&prefix))
            .collect();
        assert_eq!(
            matches.len(),
            1,
            "expected exactly one `#define {}` in kernels.cu, found {}: {:?}",
            name,
            matches.len(),
            matches
        );
        matches[0][prefix.len()..]
            .trim()
            .parse::<u32>()
            .unwrap_or_else(|e| panic!("`{}` is not an integer #define: {}", matches[0], e))
    }

    /// The value of `const <name>: u32 = <integer>;` in vector_search/mod.rs.
    fn rs_const(name: &str) -> u32 {
        let prefix = format!("const {}: u32 = ", name);
        let matches: Vec<&str> = VECTOR_SEARCH_RS
            .lines()
            .map(str::trim)
            .filter(|l| l.starts_with(&prefix) && l.ends_with(';'))
            .collect();
        assert_eq!(
            matches.len(),
            1,
            "expected exactly one `const {}: u32` in vector_search/mod.rs, found {}: {:?}",
            name,
            matches.len(),
            matches
        );
        let body = &matches[0][prefix.len()..matches[0].len() - 1];
        body.trim()
            .replace('_', "")
            .parse::<u32>()
            .unwrap_or_else(|e| panic!("`{}` is not an integer const: {}", matches[0], e))
    }

    #[test]
    fn audit_constants_agree_between_kernels_cu_and_mod_rs() {
        for name in ["AUDIT_BLOCK", "AUDIT_TQ", "AUDIT_MAX_DIMS"] {
            let cu = cu_define(name);
            let rs = rs_const(name);
            assert_eq!(
                cu, rs,
                "{} is {} in kernels.cu but {} in vector_search/mod.rs; the two \
                 are one constant that happens to live in two languages, and a \
                 mismatch changes what the audit examines without any error",
                name, cu, rs
            );
        }
    }
}
