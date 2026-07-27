#![allow(dead_code)]

#[cfg(all(
    target_arch = "aarch64",
    feature = "f32",
    feature = "neon",
    not(target_os = "macos")
))]
pub(crate) const FTYPE_LANES: usize = 4; // 128-bit / 32-bit = 4

#[cfg(all(
    target_arch = "aarch64",
    feature = "f32",
    feature = "neon",
    not(target_os = "macos")
))]
pub(crate) const FTYPE_UNROLL: usize = 4;

#[cfg(all(
    target_arch = "aarch64",
    feature = "f64",
    feature = "neon",
    not(target_os = "macos")
))]
pub(crate) const DTYPE_LANES: usize = 2; // 128-bit / 64-bit = 2

#[cfg(all(
    target_arch = "aarch64",
    feature = "f64",
    feature = "neon",
    not(target_os = "macos")
))]
pub(crate) const DTYPE_UNROLL: usize = 2;
