#[cfg(all(target_arch = "x86_64", feature = "avx2", feature = "f64"))]
use std::arch::x86_64::*;

#[cfg(all(target_arch = "x86_64", feature = "avx2", feature = "f64"))]
use crate::shared::arch::{DTYPE_LANES, DTYPE_UNROLL};

#[cfg(all(target_arch = "x86_64", feature = "avx2", feature = "f64"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn vec_diff_double_impl(x: &[f64], y: &[f64], out: &mut [f64]) {
    let len = x.len();

    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();
    let out_ptr = out.as_mut_ptr();

    let step = DTYPE_LANES * DTYPE_UNROLL;
    let chunks = len / step;

    let mut i: usize = 0usize;
    for _ in 0..chunks {
        let x0 = _mm256_loadu_pd(x_ptr.add(i));
        let y0 = _mm256_loadu_pd(y_ptr.add(i));
        _mm256_storeu_pd(out_ptr.add(i), _mm256_sub_pd(x0, y0));
        i += DTYPE_LANES;

        let x1 = _mm256_loadu_pd(x_ptr.add(i));
        let y1 = _mm256_loadu_pd(y_ptr.add(i));
        _mm256_storeu_pd(out_ptr.add(i), _mm256_sub_pd(x1, y1));
        i += DTYPE_LANES;
    }

    while i + DTYPE_LANES <= len {
        let xv = _mm256_loadu_pd(x_ptr.add(i));
        let yv = _mm256_loadu_pd(y_ptr.add(i));
        _mm256_storeu_pd(out_ptr.add(i), _mm256_sub_pd(xv, yv));
        i += DTYPE_LANES;
    }

    while i < len {
        *out_ptr.add(i) = *x_ptr.add(i) - *y_ptr.add(i);
        i += 1;
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx2", feature = "f64"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn vec_dot_double_impl(x: &[f64], y: &[f64]) -> f64 {
    let len = x.len();

    let mut acc0 = _mm256_setzero_pd();
    let mut acc1 = _mm256_setzero_pd();

    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();

    let step = DTYPE_LANES * DTYPE_UNROLL;
    let chunks = len / step;

    let mut i: usize = 0usize;
    for _ in 0..chunks {
        // _mm256_fmadd_pd(x, y, z) = x*y + z   (single-round FMA, same as
        // NEON vfmaq_f64(acc, x, y) = acc + x*y).
        let x0 = _mm256_loadu_pd(x_ptr.add(i));
        let y0 = _mm256_loadu_pd(y_ptr.add(i));
        acc0 = _mm256_fmadd_pd(x0, y0, acc0);
        i += DTYPE_LANES;

        let x1 = _mm256_loadu_pd(x_ptr.add(i));
        let y1 = _mm256_loadu_pd(y_ptr.add(i));
        acc1 = _mm256_fmadd_pd(x1, y1, acc1);
        i += DTYPE_LANES;
    }

    while i + DTYPE_LANES <= len {
        let xv = _mm256_loadu_pd(x_ptr.add(i));
        let yv = _mm256_loadu_pd(y_ptr.add(i));
        acc0 = _mm256_fmadd_pd(xv, yv, acc0);
        i += DTYPE_LANES;
    }

    // Merge 2 registers
    let acc = _mm256_add_pd(acc0, acc1);

    // Horizontal sum: 256-bit (4 doubles) → scalar.
    // AVX2 has no fused horizontal add for doubles; reduce low/high 128-bit
    // halves via extract + cast + SSE2 2-lane fold.
    let hi128 = _mm256_extractf128_pd(acc, 1);
    let lo128 = _mm256_castpd256_pd128(acc);
    let sum128 = _mm_add_pd(lo128, hi128);

    let hi = _mm_unpackhi_pd(sum128, sum128);
    let mut sum = _mm_cvtsd_f64(sum128) + _mm_cvtsd_f64(hi);

    while i < len {
        sum += *x_ptr.add(i) * *y_ptr.add(i);
        i += 1;
    }

    return sum;
}

#[cfg(all(target_arch = "x86_64", feature = "avx2", feature = "f64"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn vec_ncpy_double_impl(x: &[f64], out: &mut [f64]) {
    let len = x.len();

    let x_ptr = x.as_ptr();
    let out_ptr = out.as_mut_ptr();

    // Sign-bit mask: XOR-ing flips the sign, equivalent to vnegq_f64.
    let sign_mask = _mm256_set1_pd(-0.0f64);

    let step = DTYPE_LANES * DTYPE_UNROLL;
    let chunks = len / step;

    let mut i: usize = 0usize;
    for _ in 0..chunks {
        let x0 = _mm256_loadu_pd(x_ptr.add(i));
        _mm256_storeu_pd(out_ptr.add(i), _mm256_xor_pd(x0, sign_mask));
        i += DTYPE_LANES;

        let x1 = _mm256_loadu_pd(x_ptr.add(i));
        _mm256_storeu_pd(out_ptr.add(i), _mm256_xor_pd(x1, sign_mask));
        i += DTYPE_LANES;
    }

    while i + DTYPE_LANES <= len {
        let xv = _mm256_loadu_pd(x_ptr.add(i));
        _mm256_storeu_pd(out_ptr.add(i), _mm256_xor_pd(xv, sign_mask));
        i += DTYPE_LANES;
    }

    while i < len {
        *out_ptr.add(i) = -*x_ptr.add(i);
        i += 1;
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx2", feature = "f64"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn vec_norm2_double_impl(x: &[f64], squared: bool) -> f64 {
    let len = x.len();
    let x_ptr = x.as_ptr();

    let mut acc0 = _mm256_setzero_pd();
    let mut acc1 = _mm256_setzero_pd();

    let step = DTYPE_LANES * DTYPE_UNROLL;
    let chunks = len / step;

    let mut i: usize = 0usize;
    for _ in 0..chunks {
        let x0 = _mm256_loadu_pd(x_ptr.add(i));
        acc0 = _mm256_fmadd_pd(x0, x0, acc0);
        i += DTYPE_LANES;

        let x1 = _mm256_loadu_pd(x_ptr.add(i));
        acc1 = _mm256_fmadd_pd(x1, x1, acc1);
        i += DTYPE_LANES;
    }

    while i + DTYPE_LANES <= len {
        let xv = _mm256_loadu_pd(x_ptr.add(i));
        acc0 = _mm256_fmadd_pd(xv, xv, acc0);
        i += DTYPE_LANES;
    }

    // Merge 2 registers
    let acc = _mm256_add_pd(acc0, acc1);

    // Horizontal sum via 256→128 fold + SSE2 2-lane fold.
    let hi128 = _mm256_extractf128_pd(acc, 1);
    let lo128 = _mm256_castpd256_pd128(acc);
    let sum128 = _mm_add_pd(lo128, hi128);

    let hi = _mm_unpackhi_pd(sum128, sum128);
    let mut sum = _mm_cvtsd_f64(sum128) + _mm_cvtsd_f64(hi);

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

#[cfg(all(target_arch = "x86_64", feature = "avx2", feature = "f64"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn vec_scale_inplace_double_impl(x: &mut [f64], scalar: f64) {
    let len = x.len();

    let scalar_v = _mm256_set1_pd(scalar);

    let x_ptr = x.as_mut_ptr();

    let step = DTYPE_LANES * DTYPE_UNROLL;
    let chunks = len / step;

    let mut i: usize = 0usize;
    for _ in 0..chunks {
        let x0 = _mm256_loadu_pd(x_ptr.add(i));
        _mm256_storeu_pd(x_ptr.add(i), _mm256_mul_pd(x0, scalar_v));
        i += DTYPE_LANES;

        let x1 = _mm256_loadu_pd(x_ptr.add(i));
        _mm256_storeu_pd(x_ptr.add(i), _mm256_mul_pd(x1, scalar_v));
        i += DTYPE_LANES;
    }

    while i + DTYPE_LANES <= len {
        let xv = _mm256_loadu_pd(x_ptr.add(i));
        _mm256_storeu_pd(x_ptr.add(i), _mm256_mul_pd(xv, scalar_v));
        i += DTYPE_LANES;
    }

    while i < len {
        *x_ptr.add(i) *= scalar;
        i += 1;
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx2", feature = "f64"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn vec_scaled_add_double_impl(
    x: &[f64],
    y: &[f64],
    scalar: f64,
    out: &mut [f64],
) {
    let len = x.len();

    let scalar_v = _mm256_set1_pd(scalar);

    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();
    let out_ptr = out.as_mut_ptr();

    let step = DTYPE_LANES * DTYPE_UNROLL;
    let chunks = len / step;

    let mut i: usize = 0usize;
    for _ in 0..chunks {
        // out = x*scalar + y  ←  _mm256_fmadd_pd(x, scalar, y) = x*scalar + y
        let x0 = _mm256_loadu_pd(x_ptr.add(i));
        let y0 = _mm256_loadu_pd(y_ptr.add(i));
        _mm256_storeu_pd(out_ptr.add(i), _mm256_fmadd_pd(x0, scalar_v, y0));
        i += DTYPE_LANES;

        let x1 = _mm256_loadu_pd(x_ptr.add(i));
        let y1 = _mm256_loadu_pd(y_ptr.add(i));
        _mm256_storeu_pd(out_ptr.add(i), _mm256_fmadd_pd(x1, scalar_v, y1));
        i += DTYPE_LANES;
    }

    while i + DTYPE_LANES <= len {
        let xv = _mm256_loadu_pd(x_ptr.add(i));
        let yv = _mm256_loadu_pd(y_ptr.add(i));
        _mm256_storeu_pd(out_ptr.add(i), _mm256_fmadd_pd(xv, scalar_v, yv));
        i += DTYPE_LANES;
    }

    while i < len {
        *out_ptr.add(i) = *x_ptr.add(i) * scalar + *y_ptr.add(i);
        i += 1;
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx2", feature = "f64"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn vec_scaled_add_inplace_double_impl(src: &[f64], scalar: f64, acc: &mut [f64]) {
    let len = src.len();

    let scalar_v = _mm256_set1_pd(scalar);

    let acc_ptr = acc.as_mut_ptr();
    let src_ptr = src.as_ptr();

    let step = DTYPE_LANES * DTYPE_UNROLL;
    let chunks = len / step;

    let mut i: usize = 0usize;
    for _ in 0..chunks {
        // acc += src*scalar  ←  _mm256_fmadd_pd(src, scalar, acc) = src*scalar + acc
        let a0 = _mm256_loadu_pd(acc_ptr.add(i));
        let s0 = _mm256_loadu_pd(src_ptr.add(i));
        _mm256_storeu_pd(acc_ptr.add(i), _mm256_fmadd_pd(s0, scalar_v, a0));
        i += DTYPE_LANES;

        let a1 = _mm256_loadu_pd(acc_ptr.add(i));
        let s1 = _mm256_loadu_pd(src_ptr.add(i));
        _mm256_storeu_pd(acc_ptr.add(i), _mm256_fmadd_pd(s1, scalar_v, a1));
        i += DTYPE_LANES;
    }

    while i + DTYPE_LANES <= len {
        let av = _mm256_loadu_pd(acc_ptr.add(i));
        let sv = _mm256_loadu_pd(src_ptr.add(i));
        _mm256_storeu_pd(acc_ptr.add(i), _mm256_fmadd_pd(sv, scalar_v, av));
        i += DTYPE_LANES;
    }

    while i < len {
        *acc_ptr.add(i) += *src_ptr.add(i) * scalar;
        i += 1;
    }
}
