use crate::shared::types::primitives::ScalarType;

#[cfg(all(target_arch = "aarch64", feature = "neon", not(target_os = "macos")))]
use std::arch::aarch64::*;

#[cfg(all(
    target_arch = "aarch64",
    feature = "neon",
    feature = "f32",
    not(target_os = "macos")
))]
use crate::shared::constants::simd_params::{FTYPE_LANES, FTYPE_UNROLL};

#[cfg(all(
    target_arch = "aarch64",
    feature = "neon",
    feature = "f64",
    not(target_os = "macos")
))]
use crate::shared::constants::simd_params::{DTYPE_LANES, DTYPE_UNROLL};

/// Scalar fused multiply-add: acc[i] += src[i] × scalar — portable fallback.
#[inline]
pub fn vec_scaled_add_inplace_ansi(src: &[ScalarType], scalar: ScalarType, acc: &mut [ScalarType]) {
    let len: usize = src.len();
    for i in 0..len {
        acc[i] = acc[i] + src[i] * scalar;
    }
}

#[cfg(all(
    target_arch = "aarch64",
    feature = "neon",
    feature = "f32",
    not(target_os = "macos")
))]
#[inline]
unsafe fn vec_scaled_add_inplace_float_impl(src: &[f32], scalar: f32, acc: &mut [f32]) {
    let len: usize = src.len();

    // Scalar broadcast to vector register
    let scalar_v = vdupq_n_f32(scalar);

    let acc_ptr = acc.as_mut_ptr();
    let src_ptr = src.as_ptr();

    let step: usize = FTYPE_LANES * FTYPE_UNROLL;
    let chunks: usize = len / step;

    let mut i: usize = 0usize;

    for _ in 0..chunks {
        // Each read-process-write operation is independent,
        // allowing the CPU's load/memory units
        // FMA pipeline to be scheduled in parallel.
        let a0 = vld1q_f32(acc_ptr.add(i));
        let s0 = vld1q_f32(src_ptr.add(i));
        vst1q_f32(acc_ptr.add(i), vfmaq_f32(a0, s0, scalar_v));
        i += FTYPE_LANES;

        let a1 = vld1q_f32(acc_ptr.add(i));
        let s1 = vld1q_f32(src_ptr.add(i));
        vst1q_f32(acc_ptr.add(i), vfmaq_f32(a1, s1, scalar_v));
        i += FTYPE_LANES;

        let a2 = vld1q_f32(acc_ptr.add(i));
        let s2 = vld1q_f32(src_ptr.add(i));
        vst1q_f32(acc_ptr.add(i), vfmaq_f32(a2, s2, scalar_v));
        i += FTYPE_LANES;

        let a3 = vld1q_f32(acc_ptr.add(i));
        let s3 = vld1q_f32(src_ptr.add(i));
        vst1q_f32(acc_ptr.add(i), vfmaq_f32(a3, s3, scalar_v));
        i += FTYPE_LANES;
    }

    // Handling remaining blocks that are less than 4-way
    // but still fill one vector width.
    while i + FTYPE_LANES <= len {
        let av = vld1q_f32(acc_ptr.add(i));
        let sv = vld1q_f32(src_ptr.add(i));
        vst1q_f32(acc_ptr.add(i), vfmaq_f32(av, sv, scalar_v));
        i += FTYPE_LANES;
    }

    while i < len {
        *acc_ptr.add(i) += *src_ptr.add(i) * scalar;
        i += 1;
    }
}

#[cfg(all(
    target_arch = "aarch64",
    feature = "neon",
    feature = "f64",
    not(target_os = "macos")
))]
#[inline]
unsafe fn vec_scaled_add_inplace_double_impl(src: &[f64], scalar: f64, acc: &mut [f64]) {
    let len = src.len();

    let scalar_v = vdupq_n_f64(scalar);

    let acc_ptr = acc.as_mut_ptr();
    let src_ptr = src.as_ptr();

    let step = DTYPE_LANES * DTYPE_UNROLL;
    let chunks = len / step;

    let mut i: usize = 0usize;
    for _ in 0..chunks {
        let a0 = vld1q_f64(acc_ptr.add(i));
        let s0 = vld1q_f64(src_ptr.add(i));
        vst1q_f64(acc_ptr.add(i), vfmaq_f64(a0, s0, scalar_v));
        i += DTYPE_LANES;

        let a1 = vld1q_f64(acc_ptr.add(i));
        let s1 = vld1q_f64(src_ptr.add(i));
        vst1q_f64(acc_ptr.add(i), vfmaq_f64(a1, s1, scalar_v));
        i += DTYPE_LANES;
    }

    while i + DTYPE_LANES <= len {
        let av = vld1q_f64(acc_ptr.add(i));
        let sv = vld1q_f64(src_ptr.add(i));
        vst1q_f64(acc_ptr.add(i), vfmaq_f64(av, sv, scalar_v));
        i += DTYPE_LANES;
    }

    while i < len {
        *acc_ptr.add(i) += *src_ptr.add(i) * scalar;
        i += 1;
    }
}

// ── Dispatch wrappers ──

/// Element-wise fused multiply-add: acc[i] += src[i] × scalar.
pub fn vec_scaled_add_inplace(src: &[ScalarType], scalar: ScalarType, acc: &mut [ScalarType]) {
    debug_assert_eq!(src.len(), acc.len(), "vector length mismatch");

    #[cfg(all(
        target_arch = "aarch64",
        feature = "neon",
        feature = "f64",
        not(target_os = "macos")
    ))]
    unsafe {
        vec_scaled_add_inplace_double_impl(src, scalar, acc);
    }

    #[cfg(all(
        target_arch = "aarch64",
        feature = "neon",
        feature = "f32",
        not(target_os = "macos")
    ))]
    unsafe {
        vec_scaled_add_inplace_float_impl(src, scalar, acc);
    }

    #[cfg(any(
        not(all(target_arch = "aarch64", feature = "neon")),
        target_os = "macos"
    ))]
    vec_scaled_add_inplace_ansi(src, scalar, acc);
}
