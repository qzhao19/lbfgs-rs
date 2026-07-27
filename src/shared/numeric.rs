// Compile time precision selection
#[cfg(feature = "f32")]
pub type ScalarType = f32;

#[cfg(feature = "f64")]
pub type ScalarType = f64;

#[cfg(all(feature = "f32", feature = "f64"))]
compile_error!(
    "Features `f32` and `f64` are mutually exclusive. \
     Use `cargo build --no-default-features --features f32` or `cargo build --features f64`."
);

#[cfg(not(any(feature = "f32", feature = "f64")))]
compile_error!(
    "lbfgs-rs supports single (f32) and double (f64) precision floating-point formats exclusively. \
     Use `cargo build --features f32` or `cargo build --features f64`."
);

// Features numeric type and label numeric type
pub type FeatureType = ScalarType;
pub type LabelType = ScalarType;
