#![allow(dead_code)]

/// Declare the `(LANES, UNROLL)` const pair for one (precision, simd)
/// backend. 
/// 
/// - `prec`:   `f32` or `f64` — selects `FTYPE_*` vs `DTYPE_*`.
/// - `simd`:   backend feature name as a string literal, e.g. `"neon"`,
///   `"sse"`, `"avx2"`.
/// - `arch`:   `target_arch` literal, e.g. `"aarch64"`, `"x86_64"`.
/// - `lanes`:  vector lanes (128-bit / element size; 256-bit for AVX2).
/// - `unroll`: vectorisation unroll factor.
/// - `macos` (optional): a full cfg predicate such as
///   `not(target_os = "macos")` to disable this backend on macOS.
macro_rules! define_arch_consts {
    (
        prec = f32,
        simd = $simd:literal,
        arch = $arch:literal,
        lanes = $lanes:literal,
        unroll = $unroll:literal
        $(, macos = $($macos_filter:tt)+)?
    ) => {
        #[cfg(all(
            target_arch = $arch,
            feature = "f32",
            feature = $simd
            $(, $($macos_filter)+)?
        ))]
        pub(crate) const FTYPE_LANES: usize = $lanes;

        #[cfg(all(
            target_arch = $arch,
            feature = "f32",
            feature = $simd
            $(, $($macos_filter)+)?
        ))]
        pub(crate) const FTYPE_UNROLL: usize = $unroll;
    };

    (
        prec = f64,
        simd = $simd:literal,
        arch = $arch:literal,
        lanes = $lanes:literal,
        unroll = $unroll:literal
        $(, macos = $($macos_filter:tt)+)?
    ) => {
        #[cfg(all(
            target_arch = $arch,
            feature = "f64",
            feature = $simd
            $(, $($macos_filter)+)?
        ))]
        pub(crate) const DTYPE_LANES: usize = $lanes;

        #[cfg(all(
            target_arch = $arch,
            feature = "f64",
            feature = $simd
            $(, $($macos_filter)+)?
        ))]
        pub(crate) const DTYPE_UNROLL: usize = $unroll;
    };
}

// ── f32 backends (FTYPE_*) ──
// NEON: aarch64, excluded on macOS
define_arch_consts!(
    prec = f32,
    simd = "neon",
    arch = "aarch64",
    lanes = 4,
    unroll = 4,
    macos = not(target_os = "macos")
);
// SSE: x86_64
define_arch_consts!(
    prec = f32,
    simd = "sse",
    arch = "x86_64",
    lanes = 4,
    unroll = 4
);
// AVX2: x86_64
define_arch_consts!(
    prec = f32,
    simd = "avx2",
    arch = "x86_64",
    lanes = 8,
    unroll = 2
);

// ── f64 backends (DTYPE_*) ──
// NEON: aarch64, excluded on macOS
define_arch_consts!(
    prec = f64,
    simd = "neon",
    arch = "aarch64",
    lanes = 2,
    unroll = 2,
    macos = not(target_os = "macos")
);
// SSE: x86_64
define_arch_consts!(
    prec = f64,
    simd = "sse",
    arch = "x86_64",
    lanes = 2,
    unroll = 2
);
// AVX2: x86_64
define_arch_consts!(
    prec = f64,
    simd = "avx2",
    arch = "x86_64",
    lanes = 4,
    unroll = 2
);
