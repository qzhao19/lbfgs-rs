use crate::shared::types::primitives::ScalarType;

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
use std::arch::aarch64::*;

#[cfg(all(target_arch = "aarch64", feature = "neon", feature = "f32"))]
use crate::shared::constants::{FTYPE_LANES, FTYPE_UNROLL};

#[cfg(all(target_arch = "aarch64", feature = "neon", feature = "f64"))]
use crate::shared::constants::{DTYPE_LANES, DTYPE_UNROLL};

/// Scalar fused multiply-add: acc[i] += src[i] × scalar — portable fallback.
#[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
#[inline]
pub fn vecadd_ansi(src: &[ScalarType], scalar: ScalarType, acc: &mut [ScalarType]) {
    debug_assert_eq!(src.len(), acc.len(), "vector length mismatch");
    let len: usize = src.len();
    for i in 0..len {
        acc[i] = acc[i] + src[i] * scalar;
    }
}

#[cfg(all(target_arch = "aarch64", feature = "neon", feature = "f32"))]
#[inline]
unsafe fn vecadd_neon_float_impl(src: &[ScalarType], scalar: ScalarType, acc: &mut [ScalarType]) {
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
    while i + FTYPE_LANES < len {
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

#[cfg(all(target_arch = "aarch64", feature = "neon", feature = "f64"))]
#[inline]
unsafe fn vecadd_neon_double_impl(src: &[ScalarType], scalar: ScalarType, acc: &mut [ScalarType]) {
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

    while i + DTYPE_LANES < len {
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
pub fn vecadd(src: &[ScalarType], scalar: ScalarType, acc: &mut [ScalarType]) {
    #[cfg(all(target_arch = "aarch64", feature = "neon", feature = "f64"))]
    unsafe {
        vecadd_neon_double_impl(src, scalar, acc);
    }
    // NEON f32 path
    #[cfg(all(target_arch = "aarch64", feature = "neon", feature = "f32"))]
    unsafe {
        vecadd_neon_float_impl(src, scalar, acc);
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
    vecadd_ansi(src, scalar, acc);
}
