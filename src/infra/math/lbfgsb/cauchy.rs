//! Generalized Cauchy Point (GCP) computation — port of Fortran `cauchy`
//! (lbfgsb.f, subroutine at line 1222) with its heapsort helper `hpsolb`
//! (line 2341).
//!
//! The GCP is the first local minimiser of the piecewise quadratic
//! `Q(x + s) = g's + 1/2·s'Bs` along the piecewise-linear projected-gradient
//! path `x+(t) = P(x - t·g, l, u)`. It is computed segment by segment:
//! breakpoints (parameter values where a path component hits a bound) are
//! consumed in increasing order; on each segment the one-dimensional
//! quadratic model is minimised exactly, and the derivative information
//! `f1 = Q'(t)`, `f2 = Q''(t)` is updated incrementally using the implicit
//! BFGS product `bmv`.
//!
//! Index conventions (kept 1-based, mirroring Fortran, so every line below
//! cross-checks against the .f source):
//! - variable indices `i`, `ibp` and the contents of `iorder` are 1-based
//!   (`1..=n`);
//! - `pointr` is the 1-based ring column of the stored pairs (advanced
//!   with `pointr % m + 1`, exactly as in Fortran);
//! - matrix entries are read through the Fortran-named accessors
//!   `mat.wy(i, pointr)` / `mat.ws(i, pointr)` on [`LimitedMemBfgsMat`].

use super::middle_mat::middle_mat_vec_prod;

use crate::infra::math::bfgs_mat::LimitedMemBfgsMat;
use crate::infra::math::vec_ops::{vec_dot, vec_scale_inplace, vec_scaled_add_inplace};
use crate::shared::numeric::ScalarType;

/// Failure modes of the GCP computation (mirrors `info != 0` on exit from
/// Fortran `cauchy`). The caller (the future `mainlb` driver) must refresh
/// the L-BFGS memory (`col = 0`, `head = 1`, `theta = 1`, `iupdat = 0`,
/// `updatd = false`) and restart the iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CauchyError {
    /// A triangular system solved inside `bmv` is singular
    /// (Fortran: "singular triangular system detected").
    SingularSystem,
}

/// Scratch / work buffers for [`cauchy`], allocated once per optimisation
/// run (mirrors `mainlb`'s `wa(8m)`, `t(n)`, `d(n)` and `indx2(n)`).
pub(crate) struct CauchyWorkspace {
    /// Breakpoint order (Fortran `iorder`, aliased to `indx2` in `mainlb`).
    /// Entries hold **1-based** variable indices. Pure scratch within GCP.
    pub iorder: Vec<usize>,
    /// Breakpoints `t` (Fortran `t`): `t[i-1]` is the breakpoint distance
    /// of variable `iorder[i-1]`.
    pub t: Vec<ScalarType>,
    /// Cauchy direction `P(x - t·g) - x` (Fortran `d`), scratch within GCP
    /// (`mainlb` reuses its own `d` array for the search direction later).
    pub d: Vec<ScalarType>,
    /// `p = W'd` (2m): direction-dependent part of the derivative updates.
    pub p: Vec<ScalarType>,
    /// `c = W'(xcp - x)` (2m) — **persistent output**: consumed by `cmprlb`
    /// to assemble the subspace gradient `r = -Z'B(xcp - x) - Z'g`.
    pub c: Vec<ScalarType>,
    /// Row of `W` at the current breakpoint (2m).
    pub wbp: Vec<ScalarType>,
    /// Output buffer of `bmv` (2m).
    pub v: Vec<ScalarType>,
}

impl CauchyWorkspace {
    pub fn new(n: usize, m: usize) -> Self {
        Self {
            iorder: vec![0; n],
            t: vec![0.0; n],
            d: vec![0.0; n],
            p: vec![0.0; 2 * m],
            c: vec![0.0; 2 * m],
            wbp: vec![0.0; 2 * m],
            v: vec![0.0; 2 * m],
        }
    }
}

/// Compute the Generalized Cauchy Point.
///
/// Port of Fortran `cauchy`. Inputs:
/// - `x`, `l`, `u`, `nbd`, `g` describe the current iterate and its bounds
///   (`nbd[i-1]` ∈ {0,1,2,3} as in Fortran; for `nbd = 0` variables the
///   caller must have `iwhere[i-1] == -1`, which is what `active` sets up);
/// - `iwhere` carries the bound-status codes (-3, -1, 0, 1, 2, 3) produced by
///   `active` / a previous `cauchy` call; it is updated **in place**:
///   1 = fixed at lower bound, 2 = fixed at upper bound, 3 = fixed forever,
///   -3 = free with zero gradient, 0 = free, -1 = free without bounds;
/// - `sbgnrm` is the ∞-norm of the projected gradient at `x` (must be > 0
///   for anything but the trivial return; Fortran projects it through
///   `projgr`, not yet ported);
/// - `epsmch` is the machine precision (the 2011 revision of Algorithm 778
///   is precisely about computing it correctly — pass `ScalarType::EPSILON`).
///
/// Outputs on success:
/// - the GCP in `xcp`;
/// - `W'(xcp - x)` in `ws.c` (consumed later by `cmprlb`);
/// - the updated `iwhere` classification;
/// - the number of quadratic segments explored (`nseg`).
///
/// Note: on the two early-exit paths (zero projected gradient, zero Cauchy
/// direction) Fortran leaves the caller's `nseg` counter untouched; here 0
/// is returned instead, and `ws.c` is likewise left untouched.
pub(crate) fn cauchy(
    x: &[ScalarType],
    l: &[ScalarType],
    u: &[ScalarType],
    nbd: &[u8],
    g: &[ScalarType],
    mat: &LimitedMemBfgsMat,
    iwhere: &mut [i32],
    xcp: &mut [ScalarType],
    ws: &mut CauchyWorkspace,
    sbgnrm: ScalarType,
    epsmch: ScalarType,
) -> Result<usize, CauchyError> {
    let n = x.len();
    let m = mat.capacity;
    let col = mat.col();
    let col2 = 2 * col;
    debug_assert_eq!(l.len(), n);
    debug_assert_eq!(u.len(), n);
    debug_assert_eq!(nbd.len(), n);
    debug_assert_eq!(g.len(), n);
    debug_assert_eq!(iwhere.len(), n);
    debug_assert_eq!(xcp.len(), n);
    debug_assert_eq!(mat.dim, n, "matrix dimension must match the problem");
    debug_assert!(ws.iorder.len() >= n && ws.t.len() >= n && ws.d.len() >= n);
    debug_assert!(ws.p.len() >= col2 && ws.c.len() >= col2);
    debug_assert!(ws.wbp.len() >= col2 && ws.v.len() >= col2);

    // Split the workspace into disjoint borrows once.
    let t = &mut ws.t;
    let iorder = &mut ws.iorder;
    let d = &mut ws.d;
    let p = &mut ws.p;
    let c = &mut ws.c;
    let wbp = &mut ws.wbp;
    let v = &mut ws.v;

    // ── Trivial case: the projected gradient is zero, GCP = x.
    if sbgnrm <= 0.0 {
        xcp[..n].copy_from_slice(&x[..n]);
        return Ok(0);
    }

    let theta = mat.theta;
    let head = mat.head(); // 1-based, derived from the ring cursors

    let mut bnded = true;
    let mut nfree = n + 1; // 1-based "next free-slot" pointer (line 1429)
    let mut nbreak = 0;
    let mut ibkmin = 0;
    let mut bkmin: ScalarType = 0.0;
    let mut f1: ScalarType = 0.0;

    // p := 0
    p[..col2].fill(0.0);

    // ── Determine the bound status of each variable, the Cauchy direction
    //    d and the breakpoints; build p = W'd along the way.
    //    (Fortran do-loop 50, lines 1447-1508)
    for i in 1..=n {
        let neggi = -g[i - 1];
        // Fortran keeps tl/tu function-scoped (they stay valid whenever the
        // breakpoint branches below read them, because those branches are
        // only reachable when the status-recomputing block has run).
        let mut tl: ScalarType = 0.0;
        let mut tu: ScalarType = 0.0;

        if iwhere[i - 1] != 3 && iwhere[i - 1] != -1 {
            // x(i) is not a constant and has bounds: compute the difference
            // between x(i) and its bounds.
            if nbd[i - 1] <= 2 {
                tl = x[i - 1] - l[i - 1];
            }
            if nbd[i - 1] >= 2 {
                tu = u[i - 1] - x[i - 1];
            }
            // A variable close enough to a bound is treated as at the bound.
            let xlower = nbd[i - 1] <= 2 && tl <= 0.0;
            let xupper = nbd[i - 1] >= 2 && tu <= 0.0;

            // Reset iwhere(i).
            iwhere[i - 1] = 0;
            if xlower {
                if neggi <= 0.0 {
                    iwhere[i - 1] = 1;
                }
            } else if xupper {
                if neggi >= 0.0 {
                    iwhere[i - 1] = 2;
                }
            } else if neggi.abs() <= 0.0 {
                iwhere[i - 1] = -3;
            }
        }

        if iwhere[i - 1] != 0 && iwhere[i - 1] != -1 {
            d[i - 1] = 0.0;
        } else {
            d[i - 1] = neggi;
            f1 -= neggi * neggi;
            // p := p + W'e_i·(g_i).  (Fortran do-loop 40, lines 1477-1481)
            let mut pointr = head;
            for j in 1..=col {
                p[j - 1] += mat.wy(i, pointr) * neggi;
                p[col + j - 1] += mat.ws(i, pointr) * neggi;
                pointr = pointr % m + 1;
            }
            if nbd[i - 1] <= 2 && nbd[i - 1] != 0 && neggi < 0.0 {
                // x(i) + d(i) is bounded below: compute the breakpoint t(i).
                nbreak += 1;
                iorder[nbreak - 1] = i;
                t[nbreak - 1] = tl / -neggi;
                if nbreak == 1 || t[nbreak - 1] < bkmin {
                    bkmin = t[nbreak - 1];
                    ibkmin = nbreak;
                }
            } else if nbd[i - 1] >= 2 && neggi > 0.0 {
                // x(i) + d(i) is bounded above: compute the breakpoint t(i).
                nbreak += 1;
                iorder[nbreak - 1] = i;
                t[nbreak - 1] = tu / neggi;
                if nbreak == 1 || t[nbreak - 1] < bkmin {
                    bkmin = t[nbreak - 1];
                    ibkmin = nbreak;
                }
            } else {
                // x(i) + d(i) is not bounded.
                nfree -= 1;
                iorder[nfree - 1] = i;
                if neggi.abs() > 0.0 {
                    bnded = false;
                }
            }
        }
    }

    // Complete the initialisation of p for theta != 1.
    // (Fortran lines 1514-1517: dscal(col, theta, p(col+1)) — a no-op when
    // col == 0, hence the `col > 0` guard, which vec_scale_inplace's
    // non-empty debug_assert would otherwise reject.)
    if theta != 1.0 && col > 0 {
        vec_scale_inplace(&mut p[col..col2], theta);
    }

    // Initialise GCP xcp = x.  (Fortran line 1521)
    xcp[..n].copy_from_slice(&x[..n]);

    // d is a zero vector: return with the initial xcp as GCP.
    // (Fortran lines 1523-1527; Fortran leaves c untouched on this path —
    // so does this port, matching its behaviour.)
    if nbreak == 0 && nfree == n + 1 {
        return Ok(0);
    }

    // c = W'(xcp - x) = 0.  (Fortran lines 1531-1533)
    c[..col2].fill(0.0);

    // Initialise the derivative f2 = psi''(0) and its safeguard f2_org.
    // (Fortran lines 1537-1544)
    let mut f2 = -theta * f1;
    let f2_org = f2;
    if col > 0 {
        if !middle_mat_vec_prod(mat, &p[..col2], &mut v[..col2]) {
            return Err(CauchyError::SingularSystem);
        }
        f2 -= vec_dot(&v[..col2], &p[..col2]);
    }
    let mut dtm = -f1 / f2;
    let mut tsum: ScalarType = 0.0;
    let mut nseg = 1;

    // ── Walk the breakpoints in increasing order.
    //    (Fortran loop starting at label 777, lines 1562-1677)
    if nbreak > 0 {
        let mut nleft = nbreak;
        let mut iter = 1;
        let mut tj: ScalarType = 0.0;

        'segments: loop {
            // Find the next smallest breakpoint; dt = t(nleft) - t(nleft+1).
            // (Fortran lines 1567-1588)
            let tj0 = tj;
            let ibp;
            if iter == 1 {
                // The smallest breakpoint is already known — no heapsort.
                tj = bkmin;
                ibp = iorder[ibkmin - 1];
            } else {
                if iter == 2 {
                    // Replace the used smallest-breakpoint slot by the
                    // breakpoint numbered nbreak, before the heapsort call.
                    if ibkmin != nbreak {
                        t[ibkmin - 1] = t[nbreak - 1];
                        iorder[ibkmin - 1] = iorder[nbreak - 1];
                    }
                }
                // Update the heap structure of breakpoints and extract the
                // least element into position nleft.
                hpsolb(nleft, &mut t[..nleft], &mut iorder[..nleft], iter - 2);
                tj = t[nleft - 1];
                ibp = iorder[nleft - 1];
            }

            let dt = tj - tj0;

            // If a minimiser is within this interval, locate the GCP and
            // return.  (goto 888)
            if dtm < dt {
                break 'segments;
            }

            // Otherwise fix one variable and reset the corresponding
            // component of d to zero.  (Fortran lines 1605-1618)
            tsum += dt;
            nleft -= 1;
            iter += 1;
            let dibp = d[ibp - 1];
            d[ibp - 1] = 0.0;
            let zibp = if dibp > 0.0 {
                xcp[ibp - 1] = u[ibp - 1];
                iwhere[ibp - 1] = 2;
                u[ibp - 1] - x[ibp - 1]
            } else {
                xcp[ibp - 1] = l[ibp - 1];
                iwhere[ibp - 1] = 1;
                l[ibp - 1] - x[ibp - 1]
            };

            // All n variables are fixed: return with xcp as GCP.
            // (Fortran lines 1620-1625 — jumps straight to the c update at
            // label 999, skipping the xcp shift because d is all zeros.)
            if nleft == 0 && nbreak == n {
                dtm = dt;
                if col > 0 {
                    vec_scaled_add_inplace(&p[..col2], dtm, &mut c[..col2]);
                }
                return Ok(nseg);
            }

            // ── Update the derivative information.
            //    (Fortran lines 1629-1666)
            nseg += 1;
            let dibp2 = dibp * dibp;
            f1 = f1 + dt * f2 + dibp2 - theta * dibp * zibp;
            f2 = f2 - theta * dibp2;

            if col > 0 {
                // c = c + dt·p.
                vec_scaled_add_inplace(&p[..col2], dt, &mut c[..col2]);
                // wbp = the row of W corresponding to the breakpoint hit.
                let mut pointr = head;
                for j in 1..=col {
                    wbp[j - 1] = mat.wy(ibp, pointr);
                    wbp[col + j - 1] = theta * mat.ws(ibp, pointr);
                    pointr = pointr % m + 1;
                }
                // (wbp)·M·c, (wbp)·M·p and (wbp)·M·(wbp)'.
                if !middle_mat_vec_prod(mat, &wbp[..col2], &mut v[..col2]) {
                    return Err(CauchyError::SingularSystem);
                }
                let wmc = vec_dot(&c[..col2], &v[..col2]);
                let wmp = vec_dot(&p[..col2], &v[..col2]);
                let wmw = vec_dot(&wbp[..col2], &v[..col2]);
                // p = p - dibp·wbp.
                vec_scaled_add_inplace(&wbp[..col2], -dibp, &mut p[..col2]);
                // Complete the f1/f2 updates while col > 0.
                f1 = f1 + dibp * wmc;
                f2 = f2 + 2.0 * dibp * wmp - dibp2 * wmw;
            }

            // Safeguard f2 against rounding: f2 >= epsmch·f2_org.
            // (Fortran line 1666)
            f2 = (epsmch * f2_org).max(f2);

            if nleft > 0 {
                dtm = -f1 / f2;
                continue 'segments;
            } else if bnded {
                // The whole projected path is bounded and fully consumed.
                f1 = 0.0;
                f2 = 0.0;
                dtm = 0.0;
                break 'segments;
            } else {
                dtm = -f1 / f2;
                break 'segments;
            }
        }
    }

    // ── 888: a minimiser of Q lies on the current segment.
    //    (Fortran lines 1681-1694)
    if dtm <= 0.0 {
        dtm = 0.0;
    }
    tsum += dtm;
    // Move the free variables and the variables whose breakpoints have not
    // been reached: xcp = xcp + tsum·d.
    vec_scaled_add_inplace(&d[..n], tsum, &mut xcp[..n]);

    // ── 999: c = c + dtm·p = W'(x^c - x), used later by cmprlb.
    //    (Fortran line 1701)
    if col > 0 {
        vec_scaled_add_inplace(&p[..col2], dtm, &mut c[..col2]);
    }

    Ok(nseg)
}

/// Sort out the least element of `t` and put the remaining elements in heap
/// order — port of Fortran `hpsolb` (lbfgsb.f line 2341; Algorithm 232 of
/// CACM, J. W. J. Williams: HEAPSORT).
///
/// On exit `t[n-1]` holds the least element and `t[0..n-1]` the remaining
/// elements in heap form (Fortran: "t(n) stores the least element, t(1)..t(n-1)
/// store the rest as a heap"); `iorder` is permuted in tandem with `t` and
/// its entries remain the **1-based** variable indices of the breakpoints.
///
/// `iheap == 0` (re)builds the heap from scratch; any other value assumes
/// `t` is already a heap and only performs the extraction step.
pub(crate) fn hpsolb(n: usize, t: &mut [ScalarType], iorder: &mut [usize], iheap: usize) {
    debug_assert_eq!(t.len(), n);
    debug_assert_eq!(iorder.len(), n);

    if iheap == 0 {
        // Rearrange the elements t(1)..t(n) to form a heap.
        // (Fortran lines 2394-2412)
        for k in 2..=n {
            let ddum = t[k - 1];
            let indxin = iorder[k - 1];
            let mut i = k;
            // Add ddum to the heap (sift-up).  (Fortran "10 continue")
            loop {
                if i > 1 {
                    let j = i / 2;
                    if ddum < t[j - 1] {
                        t[i - 1] = t[j - 1];
                        iorder[i - 1] = iorder[j - 1];
                        i = j;
                        continue;
                    }
                }
                break;
            }
            t[i - 1] = ddum;
            iorder[i - 1] = indxin;
        }
    }

    if n > 1 {
        // Assign to `out` the value of t(1), the least member of the heap,
        // and rearrange the remaining members to form a heap in t(1..n-1).
        // (Fortran lines 2419-2445)
        let mut i = 1;
        let out = t[0];
        let indxou = iorder[0];
        let ddum = t[n - 1];
        let indxin = iorder[n - 1];

        // Restore the heap (sift-down).  (Fortran "30 continue")
        loop {
            let mut j = i + i;
            if j <= n - 1 {
                // Pick the smaller child: t(j+1) < t(j)  (1-based).
                if t[j] < t[j - 1] {
                    j += 1;
                }
                if t[j - 1] < ddum {
                    t[i - 1] = t[j - 1];
                    iorder[i - 1] = iorder[j - 1];
                    i = j;
                    continue;
                }
            }
            break;
        }
        t[i - 1] = ddum;
        iorder[i - 1] = indxin;

        // Put the least member in t(n).
        t[n - 1] = out;
        iorder[n - 1] = indxou;
    }
}
