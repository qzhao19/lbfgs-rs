#[cfg(all(
    target_arch = "aarch64",
    feature = "neon",
    feature = "f64",
    not(target_os = "macos")
))]
use std::arch::aarch64::*;

#[cfg(all(
    target_arch = "aarch64",
    feature = "neon",
    feature = "f32",
    not(target_os = "macos")
))]
use crate::shared::arch::{FTYPE_LANES, FTYPE_UNROLL};

#[cfg(all(
    target_arch = "aarch64",
    feature = "neon",
    feature = "f32",
    not(target_os = "macos")
))]
#[inline]
pub(crate) unsafe fn vec_diff_float_impl(x: &[f32], y: &[f32], out: &mut [f32]) {
    let len: usize = x.len();

    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();
    let out_ptr = out.as_mut_ptr();

    let step: usize = FTYPE_LANES * FTYPE_UNROLL;
    let chunks: usize = len / step;

    let mut i: usize = 0usize;

    for _ in 0..chunks {
        // Each read-process-write operation is independent,
        // allowing the CPU's load/memory units
        // FMA pipeline to be scheduled in parallel.
        let x0 = vld1q_f32(x_ptr.add(i));
        let y0 = vld1q_f32(y_ptr.add(i));
        vst1q_f32(out_ptr.add(i), vsubq_f32(x0, y0));
        i += FTYPE_LANES;

        let x1 = vld1q_f32(x_ptr.add(i));
        let y1 = vld1q_f32(y_ptr.add(i));
        vst1q_f32(out_ptr.add(i), vsubq_f32(x1, y1));
        i += FTYPE_LANES;

        let x2 = vld1q_f32(x_ptr.add(i));
        let y2 = vld1q_f32(y_ptr.add(i));
        vst1q_f32(out_ptr.add(i), vsubq_f32(x2, y2));
        i += FTYPE_LANES;

        let x3 = vld1q_f32(x_ptr.add(i));
        let y3 = vld1q_f32(y_ptr.add(i));
        vst1q_f32(out_ptr.add(i), vsubq_f32(x3, y3));
        i += FTYPE_LANES;
    }

    // Handling remaining blocks that are less than 4-way
    // but still fill one vector width.
    while i + FTYPE_LANES <= len {
        let xv = vld1q_f32(x_ptr.add(i));
        let yv = vld1q_f32(y_ptr.add(i));
        vst1q_f32(out_ptr.add(i), vsubq_f32(xv, yv));
        i += FTYPE_LANES;
    }

    // Handle remaining block that are less than one vector width
    while i < len {
        *out_ptr.add(i) = *x_ptr.add(i) - *y_ptr.add(i);
        i += 1;
    }
}

/// NEON-accelerated dot product for f32.
#[cfg(all(
    target_arch = "aarch64",
    feature = "neon",
    feature = "f32",
    not(target_os = "macos")
))]
#[inline]
pub(crate) unsafe fn vec_dot_float_impl(x: &[f32], y: &[f32]) -> f32 {
    let len: usize = x.len();

    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();

    let step: usize = FTYPE_LANES * FTYPE_UNROLL;
    let chunks: usize = len / step;

    let mut i: usize = 0usize;
    for _ in 0..chunks {
        let x0 = vld1q_f32(x_ptr.add(i));
        let y0 = vld1q_f32(y_ptr.add(i));
        acc0 = vfmaq_f32(acc0, x0, y0);
        i += FTYPE_LANES;

        let x1 = vld1q_f32(x_ptr.add(i));
        let y1 = vld1q_f32(y_ptr.add(i));
        acc1 = vfmaq_f32(acc1, x1, y1);
        i += FTYPE_LANES;

        let x2 = vld1q_f32(x_ptr.add(i));
        let y2 = vld1q_f32(y_ptr.add(i));
        acc2 = vfmaq_f32(acc2, x2, y2);
        i += FTYPE_LANES;

        let x3 = vld1q_f32(x_ptr.add(i));
        let y3 = vld1q_f32(y_ptr.add(i));
        acc3 = vfmaq_f32(acc3, x3, y3);
        i += FTYPE_LANES;
    }

    // Handling remaining blocks that are less than 4-way
    // but still fill one vector width.
    while i + FTYPE_LANES <= len {
        let xv = vld1q_f32(x_ptr.add(i));
        let yv = vld1q_f32(y_ptr.add(i));
        acc0 = vfmaq_f32(acc0, xv, yv);
        i += FTYPE_LANES;
    }

    // Merge 4 registers
    let acc01 = vaddq_f32(acc0, acc1);
    let acc23 = vaddq_f32(acc2, acc3);
    let acc = vaddq_f32(acc01, acc23);

    // Horizontal sum via vaddvq_f32
    let mut sum = vaddvq_f32(acc);

    // Handle remaining elements at the end
    // that are less than one vector width
    while i < len {
        sum += *x_ptr.add(i) * *y_ptr.add(i);
        i += 1;
    }

    return sum;
}

#[cfg(all(
    target_arch = "aarch64",
    feature = "neon",
    feature = "f32",
    not(target_os = "macos")
))]
#[inline]
pub(crate) unsafe fn vec_ncpy_float_impl(x: &[f32], out: &mut [f32]) {
    let len: usize = x.len();

    let x_ptr = x.as_ptr();
    let out_ptr = out.as_mut_ptr();

    let step: usize = FTYPE_LANES * FTYPE_UNROLL;
    let chunks: usize = len / step;

    let mut i: usize = 0usize;

    for _ in 0..chunks {
        // Each read-process-write operation is independent
        let x0 = vld1q_f32(x_ptr.add(i));
        vst1q_f32(out_ptr.add(i), vnegq_f32(x0));
        i += FTYPE_LANES;

        let x1 = vld1q_f32(x_ptr.add(i));
        vst1q_f32(out_ptr.add(i), vnegq_f32(x1));
        i += FTYPE_LANES;

        let x2 = vld1q_f32(x_ptr.add(i));
        vst1q_f32(out_ptr.add(i), vnegq_f32(x2));
        i += FTYPE_LANES;

        let x3 = vld1q_f32(x_ptr.add(i));
        vst1q_f32(out_ptr.add(i), vnegq_f32(x3));
        i += FTYPE_LANES;
    }

    // Handling remaining blocks that are less than 4-way
    // but still fill one vector width.
    while i + FTYPE_LANES <= len {
        let xv = vld1q_f32(x_ptr.add(i));
        vst1q_f32(out_ptr.add(i), vnegq_f32(xv));
        i += FTYPE_LANES;
    }

    // Handle remaining block that are less than one vector
    while i < len {
        *out_ptr.add(i) = -*x_ptr.add(i);
        i += 1;
    }
}

#[cfg(all(
    target_arch = "aarch64",
    feature = "neon",
    feature = "f32",
    not(target_os = "macos")
))]
#[inline]
pub(crate) unsafe fn vec_norm2_float_impl(x: &[f32], squared: bool) -> f32 {
    let len: usize = x.len();
    let x_ptr = x.as_ptr();

    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    let step: usize = FTYPE_LANES * FTYPE_UNROLL;
    let chunks: usize = len / step;

    let mut i: usize = 0usize;
    for _ in 0..chunks {
        // Each load-process operation is independent.
        let x0 = vld1q_f32(x_ptr.add(i));
        acc0 = vfmaq_f32(acc0, x0, x0);
        i += FTYPE_LANES;

        let x1 = vld1q_f32(x_ptr.add(i));
        acc1 = vfmaq_f32(acc1, x1, x1);
        i += FTYPE_LANES;

        let x2 = vld1q_f32(x_ptr.add(i));
        acc2 = vfmaq_f32(acc2, x2, x2);
        i += FTYPE_LANES;

        let x3 = vld1q_f32(x_ptr.add(i));
        acc3 = vfmaq_f32(acc3, x3, x3);
        i += FTYPE_LANES;
    }

    // Handling remaining blocks that are less than 4-way
    // but still fill one vector width.
    while i + FTYPE_LANES <= len {
        let xv = vld1q_f32(x_ptr.add(i));
        acc0 = vfmaq_f32(acc0, xv, xv);
        i += FTYPE_LANES;
    }

    // Merge 4 registers
    let acc01 = vaddq_f32(acc0, acc1);
    let acc23 = vaddq_f32(acc2, acc3);
    let acc = vaddq_f32(acc01, acc23);

    // Horizontal sum via vaddvq_f32
    let mut sum = vaddvq_f32(acc);

    // Handle remaining elements that are less than one vector width
    while i < len {
        sum += *x_ptr.add(i) * *x_ptr.add(i);
        i += 1;
    }

    if squared {
        sum
    } else {
        sum.sqrt()
    }
}

#[cfg(all(
    target_arch = "aarch64",
    feature = "neon",
    feature = "f32",
    not(target_os = "macos")
))]
#[inline]
pub(crate) unsafe fn vec_scale_inplace_float_impl(x: &mut [f32], scalar: f32) {
    let len: usize = x.len();

    // Scalar broadcast to vector register
    let scalar_v = vdupq_n_f32(scalar);

    let x_ptr = x.as_mut_ptr();

    let step: usize = FTYPE_LANES * FTYPE_UNROLL;
    let chunks: usize = len / step;

    let mut i: usize = 0usize;

    for _ in 0..chunks {
        // Each read-process-write operation is independent
        let x0 = vld1q_f32(x_ptr.add(i));
        vst1q_f32(x_ptr.add(i), vmulq_f32(x0, scalar_v));
        i += FTYPE_LANES;

        let x1 = vld1q_f32(x_ptr.add(i));
        vst1q_f32(x_ptr.add(i), vmulq_f32(x1, scalar_v));
        i += FTYPE_LANES;

        let x2 = vld1q_f32(x_ptr.add(i));
        vst1q_f32(x_ptr.add(i), vmulq_f32(x2, scalar_v));
        i += FTYPE_LANES;

        let x3 = vld1q_f32(x_ptr.add(i));
        vst1q_f32(x_ptr.add(i), vmulq_f32(x3, scalar_v));
        i += FTYPE_LANES;
    }

    // Handling remaining blocks that are less than 4-way
    // but still fill one vector width.
    while i + FTYPE_LANES <= len {
        let xv = vld1q_f32(x_ptr.add(i));
        vst1q_f32(x_ptr.add(i), vmulq_f32(xv, scalar_v));
        i += FTYPE_LANES;
    }

    // Handle remaining block that are less than one vector width
    while i < len {
        *x_ptr.add(i) *= scalar;
        i += 1;
    }
}

#[cfg(all(
    target_arch = "aarch64",
    feature = "neon",
    feature = "f32",
    not(target_os = "macos")
))]
#[inline]
pub(crate) unsafe fn vec_scaled_add_float_impl(x: &[f32], y: &[f32], scalar: f32, out: &mut [f32]) {
    let len: usize = x.len();

    // Scalar broadcast to vector register
    let scalar_v = vdupq_n_f32(scalar);

    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();
    let out_ptr = out.as_mut_ptr();

    let step: usize = FTYPE_LANES * FTYPE_UNROLL;
    let chunks: usize = len / step;

    let mut i: usize = 0usize;

    for _ in 0..chunks {
        // Each read-process-write operation is independent,
        // allowing the CPU's load/memory units
        // FMA pipeline to be scheduled in parallel.
        let x0 = vld1q_f32(x_ptr.add(i));
        let y0 = vld1q_f32(y_ptr.add(i));
        vst1q_f32(out_ptr.add(i), vfmaq_f32(y0, x0, scalar_v));
        i += FTYPE_LANES;

        let x1 = vld1q_f32(x_ptr.add(i));
        let y1 = vld1q_f32(y_ptr.add(i));
        vst1q_f32(out_ptr.add(i), vfmaq_f32(y1, x1, scalar_v));
        i += FTYPE_LANES;

        let x2 = vld1q_f32(x_ptr.add(i));
        let y2 = vld1q_f32(y_ptr.add(i));
        vst1q_f32(out_ptr.add(i), vfmaq_f32(y2, x2, scalar_v));
        i += FTYPE_LANES;

        let x3 = vld1q_f32(x_ptr.add(i));
        let y3 = vld1q_f32(y_ptr.add(i));
        vst1q_f32(out_ptr.add(i), vfmaq_f32(y3, x3, scalar_v));
        i += FTYPE_LANES;
    }

    // Handling remaining blocks that are less than 4-way
    // but still fill one vector width.
    while i + FTYPE_LANES <= len {
        let xv = vld1q_f32(x_ptr.add(i));
        let yv = vld1q_f32(y_ptr.add(i));
        vst1q_f32(out_ptr.add(i), vfmaq_f32(yv, xv, scalar_v));
        i += FTYPE_LANES;
    }

    while i < len {
        *out_ptr.add(i) = *x_ptr.add(i) * scalar + *y_ptr.add(i);
        i += 1;
    }
}

#[cfg(all(
    target_arch = "aarch64",
    feature = "neon",
    feature = "f32",
    not(target_os = "macos")
))]
#[inline]
pub(crate) unsafe fn vec_scaled_add_inplace_float_impl(src: &[f32], scalar: f32, acc: &mut [f32]) {
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
