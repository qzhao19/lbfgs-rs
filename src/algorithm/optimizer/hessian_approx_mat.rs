use crate::infra::math::vec_ops::{vec_dot, vec_scale_inplace, vec_scaled_add_inplace};
use crate::shared::numeric::ScalarType;

/// One slot of the limited-memory correction history: a single (s, y) pair
/// s: x_{k+1} - x_k
/// y: g_{k+1} - g_k
/// ys: its curvature scalar
#[derive(Clone, Debug)]
struct LimitedMemCorrHistory {
    s: Vec<ScalarType>,
    y: Vec<ScalarType>,
    ys: ScalarType,
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

/// Limited-memory inverse-Hessian approximation for L-BFGS.
pub(crate) struct LimitedMemHessianApproxMat {
    /// - `history`: ring buffer of correction-pair slots.
    history: Vec<LimitedMemCorrHistory>,
    /// `capacity`: ring-buffer capacity (the "m" in L-BFGS).
    capacity: usize,
    /// `dim`: Dimension of every stored vector `s` and `y`, equal to `n_features`
    dim: usize,
    /// - `end`: write cursor, index of the slot where the *next* `update` will store
    end: usize,
    /// - `bound`: number of currently available correction pairs
    bound: usize,
}

impl LimitedMemHessianApproxMat {
    /// Allocate a ring buffer of `capacity` slots, each sized `dim`.
    pub fn new(capacity: usize, dim: usize) -> Self {
        Self {
            history: std::iter::repeat_with(|| LimitedMemCorrHistory::initialize(dim))
                .take(capacity)
                .collect(),
            capacity,
            dim,
            end: 0,
            bound: 0,
        }
    }

    /// Push one correction pair into ring buffer, store `ys` = `y*s`
    pub fn update(&mut self, s: &[ScalarType], y: &[ScalarType]) {
        debug_assert_eq!(s.len(), self.dim, "s length mismatch");
        debug_assert_eq!(y.len(), self.dim, "y length mismatch");

        // Define a mutable ref to point to end-th slot
        let lm_slot: &mut LimitedMemCorrHistory = &mut self.history[self.end];
        lm_slot.s.copy_from_slice(s);
        lm_slot.y.copy_from_slice(y);
        lm_slot.ys = vec_dot(s, y);

        self.end = (self.end + 1) % self.capacity;
        if self.bound < self.capacity {
            self.bound += 1;
        }
    }

    /// In-place two-loop recursion: v <- H * v
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
            // beta_j = rho_{j} * y_{T}_{j} @ d_{j}, rho_{j} = 1/ys
            let beta: ScalarType = vec_dot(&self.history[j].y, d) / self.history[j].ys;

            // gamma_{i+1} = gamma_{i} + (alpha_{j} - beta_{j}) * s_{j}
            let coef: ScalarType = alpha[j] - beta;
            vec_scaled_add_inplace(&self.history[j].s, coef, d);

            // starting the earliest history information to traverse backward
            j = (j + 1) % m;
        }
    }
}
