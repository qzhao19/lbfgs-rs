#[cfg(all(target_arch = "x86_64", feature = "avx2", feature = "f32"))]
use std::arch::x86_64::*;

#[cfg(all(target_arch = "x86_64", feature = "avx2", feature = "f32"))]
use crate::shared::arch::{FTYPE_LANES, FTYPE_UNROLL};

#[cfg(all(target_arch = "x86_64", feature = "avx2", feature = "f32"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
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
        let x0 = _mm256_loadu_ps(x_ptr.add(i));
        let y0 = _mm256_loadu_ps(y_ptr.add(i));
        _mm256_storeu_ps(out_ptr.add(i), _mm256_sub_ps(x0, y0));
        i += FTYPE_LANES;

        let x1 = _mm256_loadu_ps(x_ptr.add(i));
        let y1 = _mm256_loadu_ps(y_ptr.add(i));
        _mm256_storeu_ps(out_ptr.add(i), _mm256_sub_ps(x1, y1));
        i += FTYPE_LANES;
    }

    // Handling remaining blocks that are less than unroll-way
    // but still fill one vector width.
    while i + FTYPE_LANES <= len {
        let xv = _mm256_loadu_ps(x_ptr.add(i));
        let yv = _mm256_loadu_ps(y_ptr.add(i));
        _mm256_storeu_ps(out_ptr.add(i), _mm256_sub_ps(xv, yv));
        i += FTYPE_LANES;
    }

    // Handle remaining block that are less than one vector width
    while i < len {
        *out_ptr.add(i) = *x_ptr.add(i) - *y_ptr.add(i);
        i += 1;
    }
}

/// AVX2-accelerated dot product for f32.
#[cfg(all(target_arch = "x86_64", feature = "avx2", feature = "f32"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn vec_dot_float_impl(x: &[f32], y: &[f32]) -> f32 {
    let len: usize = x.len();

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();

    let step: usize = FTYPE_LANES * FTYPE_UNROLL;
    let chunks: usize = len / step;

    let mut i: usize = 0usize;
    for _ in 0..chunks {
        // _mm256_fmadd_ps(x, y, z) = x*y + z   (single-round FMA, same as
        // NEON vfmaq_f32(acc, x, y) = acc + x*y).
        let x0 = _mm256_loadu_ps(x_ptr.add(i));
        let y0 = _mm256_loadu_ps(y_ptr.add(i));
        acc0 = _mm256_fmadd_ps(x0, y0, acc0);
        i += FTYPE_LANES;

        let x1 = _mm256_loadu_ps(x_ptr.add(i));
        let y1 = _mm256_loadu_ps(y_ptr.add(i));
        acc1 = _mm256_fmadd_ps(x1, y1, acc1);
        i += FTYPE_LANES;
    }

    // Handling remaining blocks that are less than unroll-way
    // but still fill one vector width.
    while i + FTYPE_LANES <= len {
        let xv = _mm256_loadu_ps(x_ptr.add(i));
        let yv = _mm256_loadu_ps(y_ptr.add(i));
        acc0 = _mm256_fmadd_ps(xv, yv, acc0);
        i += FTYPE_LANES;
    }

    // Merge 2 registers
    let acc = _mm256_add_ps(acc0, acc1);

    // Horizontal sum: 256-bit → 128-bit → scalar.
    // AVX2 has no fused horizontal add for f32; reduce low/high 128-bit halves
    // via extract + the SSE 4-lane fold sequence.
    let hi128 = _mm256_extractf128_ps(acc, 1);
    let lo128 = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo128, hi128);

    let shuf1 = _mm_shuffle_ps(sum128, sum128, 0x4e); // [a2, a3, a0, a1]
    let sum1 = _mm_add_ps(sum128, shuf1); // [a0+a2, a1+a3, ?, ?]
    let shuf2 = _mm_shuffle_ps(sum1, sum1, 0xb1); // [a1+a3, a0+a2, ?, ?]
    let sum2 = _mm_add_ps(sum1, shuf2); // all lanes = a0+a1+a2+a3
    let mut sum: f32 = _mm_cvtss_f32(sum2);

    // Handle remaining elements at the end
    // that are less than one vector width
    while i < len {
        sum += *x_ptr.add(i) * *y_ptr.add(i);
        i += 1;
    }

    return sum;
}

#[cfg(all(target_arch = "x86_64", feature = "avx2", feature = "f32"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn vec_ncpy_float_impl(x: &[f32], out: &mut [f32]) {
    let len: usize = x.len();

    let x_ptr = x.as_ptr();
    let out_ptr = out.as_mut_ptr();

    // Sign-bit mask: XOR-ing flips the sign, equivalent to vnegq_f32.
    let sign_mask = _mm256_set1_ps(-0.0f32);

    let step: usize = FTYPE_LANES * FTYPE_UNROLL;
    let chunks: usize = len / step;

    let mut i: usize = 0usize;

    for _ in 0..chunks {
        // Each read-process-write operation is independent
        let x0 = _mm256_loadu_ps(x_ptr.add(i));
        _mm256_storeu_ps(out_ptr.add(i), _mm256_xor_ps(x0, sign_mask));
        i += FTYPE_LANES;

        let x1 = _mm256_loadu_ps(x_ptr.add(i));
        _mm256_storeu_ps(out_ptr.add(i), _mm256_xor_ps(x1, sign_mask));
        i += FTYPE_LANES;
    }

    // Handling remaining blocks that are less than unroll-way
    // but still fill one vector width.
    while i + FTYPE_LANES <= len {
        let xv = _mm256_loadu_ps(x_ptr.add(i));
        _mm256_storeu_ps(out_ptr.add(i), _mm256_xor_ps(xv, sign_mask));
        i += FTYPE_LANES;
    }

    // Handle remaining block that are less than one vector
    while i < len {
        *out_ptr.add(i) = -*x_ptr.add(i);
        i += 1;
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx2", feature = "f32"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn vec_norm2_float_impl(x: &[f32], squared: bool) -> f32 {
    let len: usize = x.len();
    let x_ptr = x.as_ptr();

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    let step: usize = FTYPE_LANES * FTYPE_UNROLL;
    let chunks: usize = len / step;

    let mut i: usize = 0usize;
    for _ in 0..chunks {
        // Each load-process operation is independent.
        let x0 = _mm256_loadu_ps(x_ptr.add(i));
        acc0 = _mm256_fmadd_ps(x0, x0, acc0);
        i += FTYPE_LANES;

        let x1 = _mm256_loadu_ps(x_ptr.add(i));
        acc1 = _mm256_fmadd_ps(x1, x1, acc1);
        i += FTYPE_LANES;
    }

    // Handling remaining blocks that are less than unroll-way
    // but still fill one vector width.
    while i + FTYPE_LANES <= len {
        let xv = _mm256_loadu_ps(x_ptr.add(i));
        acc0 = _mm256_fmadd_ps(xv, xv, acc0);
        i += FTYPE_LANES;
    }

    // Merge 2 registers
    let acc = _mm256_add_ps(acc0, acc1);

    // Horizontal sum via 256→128 fold + SSE 4-lane fold.
    let hi128 = _mm256_extractf128_ps(acc, 1);
    let lo128 = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo128, hi128);

    let shuf1 = _mm_shuffle_ps(sum128, sum128, 0x4e);
    let sum1 = _mm_add_ps(sum128, shuf1);
    let shuf2 = _mm_shuffle_ps(sum1, sum1, 0xb1);
    let sum2 = _mm_add_ps(sum1, shuf2);
    let mut sum: f32 = _mm_cvtss_f32(sum2);

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

#[cfg(all(target_arch = "x86_64", feature = "avx2", feature = "f32"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn vec_scale_inplace_float_impl(x: &mut [f32], scalar: f32) {
    let len: usize = x.len();

    // Scalar broadcast to vector register
    let scalar_v = _mm256_set1_ps(scalar);

    let x_ptr = x.as_mut_ptr();

    let step: usize = FTYPE_LANES * FTYPE_UNROLL;
    let chunks: usize = len / step;

    let mut i: usize = 0usize;

    for _ in 0..chunks {
        // Each read-process-write operation is independent
        let x0 = _mm256_loadu_ps(x_ptr.add(i));
        _mm256_storeu_ps(x_ptr.add(i), _mm256_mul_ps(x0, scalar_v));
        i += FTYPE_LANES;

        let x1 = _mm256_loadu_ps(x_ptr.add(i));
        _mm256_storeu_ps(x_ptr.add(i), _mm256_mul_ps(x1, scalar_v));
        i += FTYPE_LANES;
    }

    // Handling remaining blocks that are less than unroll-way
    // but still fill one vector width.
    while i + FTYPE_LANES <= len {
        let xv = _mm256_loadu_ps(x_ptr.add(i));
        _mm256_storeu_ps(x_ptr.add(i), _mm256_mul_ps(xv, scalar_v));
        i += FTYPE_LANES;
    }

    // Handle remaining block that are less than one vector width
    while i < len {
        *x_ptr.add(i) *= scalar;
        i += 1;
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx2", feature = "f32"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn vec_scaled_add_float_impl(x: &[f32], y: &[f32], scalar: f32, out: &mut [f32]) {
    let len: usize = x.len();

    // Scalar broadcast to vector register
    let scalar_v = _mm256_set1_ps(scalar);

    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();
    let out_ptr = out.as_mut_ptr();

    let step: usize = FTYPE_LANES * FTYPE_UNROLL;
    let chunks: usize = len / step;

    let mut i: usize = 0usize;

    for _ in 0..chunks {
        // out = x*scalar + y  ←  _mm256_fmadd_ps(x, scalar, y) = x*scalar + y
        let x0 = _mm256_loadu_ps(x_ptr.add(i));
        let y0 = _mm256_loadu_ps(y_ptr.add(i));
        _mm256_storeu_ps(out_ptr.add(i), _mm256_fmadd_ps(x0, scalar_v, y0));
        i += FTYPE_LANES;

        let x1 = _mm256_loadu_ps(x_ptr.add(i));
        let y1 = _mm256_loadu_ps(y_ptr.add(i));
        _mm256_storeu_ps(out_ptr.add(i), _mm256_fmadd_ps(x1, scalar_v, y1));
        i += FTYPE_LANES;
    }

    // Handling remaining blocks that are less than unroll-way
    // but still fill one vector width.
    while i + FTYPE_LANES <= len {
        let xv = _mm256_loadu_ps(x_ptr.add(i));
        let yv = _mm256_loadu_ps(y_ptr.add(i));
        _mm256_storeu_ps(out_ptr.add(i), _mm256_fmadd_ps(xv, scalar_v, yv));
        i += FTYPE_LANES;
    }

    while i < len {
        *out_ptr.add(i) = *x_ptr.add(i) * scalar + *y_ptr.add(i);
        i += 1;
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx2", feature = "f32"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn vec_scaled_add_inplace_float_impl(src: &[f32], scalar: f32, acc: &mut [f32]) {
    let len: usize = src.len();

    // Scalar broadcast to vector register
    let scalar_v = _mm256_set1_ps(scalar);

    let acc_ptr = acc.as_mut_ptr();
    let src_ptr = src.as_ptr();

    let step: usize = FTYPE_LANES * FTYPE_UNROLL;
    let chunks: usize = len / step;

    let mut i: usize = 0usize;

    for _ in 0..chunks {
        // acc += src*scalar  ←  _mm256_fmadd_ps(src, scalar, acc) = src*scalar + acc
        let a0 = _mm256_loadu_ps(acc_ptr.add(i));
        let s0 = _mm256_loadu_ps(src_ptr.add(i));
        _mm256_storeu_ps(acc_ptr.add(i), _mm256_fmadd_ps(s0, scalar_v, a0));
        i += FTYPE_LANES;

        let a1 = _mm256_loadu_ps(acc_ptr.add(i));
        let s1 = _mm256_loadu_ps(src_ptr.add(i));
        _mm256_storeu_ps(acc_ptr.add(i), _mm256_fmadd_ps(s1, scalar_v, a1));
        i += FTYPE_LANES;
    }

    // Handling remaining blocks that are less than unroll-way
    // but still fill one vector width.
    while i + FTYPE_LANES <= len {
        let av = _mm256_loadu_ps(acc_ptr.add(i));
        let sv = _mm256_loadu_ps(src_ptr.add(i));
        _mm256_storeu_ps(acc_ptr.add(i), _mm256_fmadd_ps(sv, scalar_v, av));
        i += FTYPE_LANES;
    }

    while i < len {
        *acc_ptr.add(i) += *src_ptr.add(i) * scalar;
        i += 1;
    }
}
