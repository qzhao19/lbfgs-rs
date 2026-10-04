use crate::infra::math::vec_ops::{vec_dot, vec_scale_inplace, vec_scaled_add_inplace};
use crate::shared::numeric::ScalarType;

/// One slot of the limited-memory correction history: a single (s, y) pair
/// - `s = x_{k+1} - x_k`, 
/// - `y = g_{k+1} - g_k` 
/// - `ys = s'y` its curvature scalar 
#[derive(Clone, Debug)]
pub(crate) struct LimitedMemCorrHistory {
    /// Displacement vector
    pub s: Vec<ScalarType>,
    /// Gradient difference vector
    pub y: Vec<ScalarType>,
    /// Curvature `s'y`
    pub ys: ScalarType,
}

impl LimitedMemCorrHistory {
    pub fn initialize(len: usize) -> Self {
        Self {
            s: vec![0.0; len],
            y: vec![0.0; len],
            ys: 0.0,
        }
    }
}

/// The limited-memory BFGS matrix in one place: correction-pair ring buffer
/// plus the compact-representation state required by L-BFGS-B
/// (`B = theta*I - W*M^{-1}*W'` with `W = [Y, theta*S]`
pub(crate) struct LimitedMemBfgsMat {
    /// Ring buffer of correction-pair slots (`ws`/`wy` columns).
    history: Vec<LimitedMemCorrHistory>,
    /// Ring-buffer capacity (the "m" in L-BFGS / L-BFGS-B).
    pub capacity: usize,
    /// Dimension of every stored vector `s` and `y` (= `n_features`).
    pub dim: usize,
    /// Write cursor: 0-based slot where the next [`update`] will store.
    end: usize,
    /// Number of currently stored pairs (`col`).
    bound: usize,

    // ── Compact-representation derived state (L-BFGS-B only) ──
    /// `S'Y`, `m × m` column-major in **logical order** (`sy`):
    /// entry `(j, k)` belongs to the `j`-th oldest s-vector and the `k`-th
    /// oldest y-vector, `j, k ∈ 1..=col`
    pub sy: Vec<ScalarType>,
    /// Upper triangle holds `J'`, the Cholesky factor of
    /// `T = theta*S'S + L*D^{-1}*L'` (eq. 2.26 in Byrd-Lu-Nocedal-Zhu),
    /// `m × m` column-major (`wt`). Produced by `formt` (pending),
    /// consumed by `bmv`
    pub wt: Vec<ScalarType>,
    /// Scaling factor `B_0 = theta*I` (`theta`, initial 1; set by
    /// `matupd` to `y'y / y's`)
    pub theta: ScalarType,
}

impl LimitedMemBfgsMat {
    /// Allocate zeroed storage with the `mainlb` start-up state:
    /// empty ring (`col = 0`, `head = 1`) and `theta = 1`
    pub fn new(capacity: usize, dim: usize) -> Self {
        Self {
            history: std::iter::repeat_with(|| LimitedMemCorrHistory::initialize(dim))
                .take(capacity)
                .collect(),
            capacity,
            dim,
            end: 0,
            bound: 0,
            sy: vec![0.0; capacity * capacity],
            wt: vec![0.0; capacity * capacity],
            theta: 1.0,
        }
    }

    /// Push one correction pair into the ring buffer, storing `ys = s'y`
    ///
    /// The plain L-BFGS driver calls this directly after each accepted
    /// step. The L-BFGS-B driver will reach it through `matupd` (pending),
    /// which additionally maintains `sy`/`ss`/`theta`
    pub fn update(&mut self, s: &[ScalarType], y: &[ScalarType]) {
        debug_assert_eq!(s.len(), self.dim, "s length mismatch");
        debug_assert_eq!(y.len(), self.dim, "y length mismatch");

        let lm_slot: &mut LimitedMemCorrHistory = &mut self.history[self.end];
        lm_slot.s.copy_from_slice(s);
        lm_slot.y.copy_from_slice(y);
        lm_slot.ys = vec_dot(s, y);

        self.end = (self.end + 1) % self.capacity;
        if self.bound < self.capacity {
            self.bound += 1;
        }
    }

    /// Number of correction pairs currently stored (`col`)
    pub fn col(&self) -> usize {
        self.bound
    }

    /// **1-based** ring position of the oldest stored pair
    pub fn head(&self) -> usize {
        if self.bound == self.capacity {
            self.end + 1
        } else {
            1
        }
    }

    /// `ws(i, j)`: component `i` (1-based) of the s-vector stored
    /// at 1-based ring position `j` (must be one of `head .. head + col`)
    #[inline]
    pub fn ws(&self, i: usize, j: usize) -> ScalarType {
        debug_assert!(i >= 1 && i <= self.dim, "ws: variable index out of range");
        debug_assert!(
            j >= 1 && j <= self.capacity,
            "ws: ring position out of range"
        );
        self.history[j - 1].s[i - 1]
    }

    /// `wy(i, j)`: component `i` (1-based) of the y-vector stored
    /// at 1-based ring position `j`
    #[inline]
    pub fn wy(&self, i: usize, j: usize) -> ScalarType {
        debug_assert!(i >= 1 && i <= self.dim, "wy: variable index out of range");
        debug_assert!(
            j >= 1 && j <= self.capacity,
            "wy: ring position out of range"
        );
        self.history[j - 1].y[i - 1]
    }

    /// In-place two-loop recursion: `v <- H * v`
    ///
    /// Traverses the pairs newest → oldest using the write cursor `end`,
    /// with `H_0 = gamma*I`, `gamma = ys/y'y` of the most recent pair
    pub fn apply_hv(&self, d: &mut [ScalarType]) {
        if self.bound == 0 {
            return;
        }
        debug_assert_eq!(d.len(), self.dim, "direction vector length mismatch");

        let m: usize = self.capacity;
        let mut alpha: Vec<ScalarType> = vec![0.0; m];

        // j: index for traversing history information
        let mut j: usize = self.end;

        // Forward pass
        for _ in 0..self.bound {
            // starting with the most recent history message
            j = (j + m - 1) % m;

            // alpha_{j} = s^{T}_{j} @ d_{j} * rho_{j}, rho_{j} = 1/ys
            alpha[j] = vec_dot(&self.history[j].s, d) / self.history[j].ys;

            // d_{i} = d_{i+1} - (alpha_{i} * y_{i})
            vec_scaled_add_inplace(&self.history[j].y, -alpha[j], d);
        }

        // H_0 = gamma * I, gamma = ys / yy
        let latest: usize = (self.end + m - 1) % m;
        let ys: ScalarType = self.history[latest].ys;
        let yy: ScalarType = vec_dot(&self.history[latest].y, &self.history[latest].y);
        let gamma: ScalarType = ys / yy;
        vec_scale_inplace(d, gamma);

        // Backward pass
        for _ in 0..self.bound {
            // beta_j = rho_{j} * y^{T}_{j} @ d_{j}, rho_{j} = 1/ys
            let beta: ScalarType = vec_dot(&self.history[j].y, d) / self.history[j].ys;

            // gamma_{i+1} = gamma_{i} + (alpha_{j} - beta_{j}) * s_{j}
            let coef: ScalarType = alpha[j] - beta;
            vec_scaled_add_inplace(&self.history[j].s, coef, d);

            // starting the earliest history information to traverse backward
            j = (j + 1) % m;
        }
    }
}
