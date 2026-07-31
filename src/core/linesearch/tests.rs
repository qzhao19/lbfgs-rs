use crate::core::linesearch::backtracking::BacktrackingLineSearch;
use crate::core::linesearch::bracketing::BracketingLineSearch;
use crate::core::linesearch::linesearch::LineSearch;
use crate::core::loss::loss::LossFunc;
use crate::data::dataset::Dataset;
use crate::data::dense::DenseDataset;
use crate::shared::exception::LbfgsError;
use crate::shared::numeric::{FeatureType, LabelType, ScalarType};
use crate::shared::parameter::{LineSearchCondition, LineSearchParam};

// ── Existing helpers (kept as-is) ──

/// Build the Iris binary classification dataset.
fn build_iris_dataset() -> DenseDataset {
    // Feature 0: sepal_length (150 values, column-major in C++ source)
    let sepal_length: &[FeatureType] = &[
        5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9, 5.4, 4.8, 4.8, 4.3, 5.8, 5.7, 5.4, 5.1,
        5.7, 5.1, 5.4, 5.1, 4.6, 5.1, 4.8, 5.0, 5.0, 5.2, 5.2, 4.7, 4.8, 5.4, 5.2, 5.5, 4.9, 5.0,
        5.5, 4.9, 4.4, 5.1, 5.0, 4.5, 4.4, 5.0, 5.1, 4.8, 5.1, 4.6, 5.3, 5.0, 7.0, 6.4, 6.9, 5.5,
        6.5, 5.7, 6.3, 4.9, 6.6, 5.2, 5.0, 5.9, 6.0, 6.1, 5.6, 6.7, 5.6, 5.8, 6.2, 5.6, 5.9, 6.1,
        6.3, 6.1, 6.4, 6.6, 6.8, 6.7, 6.0, 5.7, 5.5, 5.5, 5.8, 6.0, 5.4, 6.0, 6.7, 6.3, 5.6, 5.5,
        5.5, 6.1, 5.8, 5.0, 5.6, 5.7, 5.7, 6.2, 5.1, 5.7, 6.3, 5.8, 7.1, 6.3, 6.5, 7.6, 4.9, 7.3,
        6.7, 7.2, 6.5, 6.4, 6.8, 5.7, 5.8, 6.4, 6.5, 7.7, 7.7, 6.0, 6.9, 5.6, 7.7, 6.3, 6.7, 7.2,
        6.2, 6.1, 6.4, 7.2, 7.4, 7.9, 6.4, 6.3, 6.1, 7.7, 6.3, 6.4, 6.0, 6.9, 6.7, 6.9, 5.8, 6.8,
        6.7, 6.7, 6.3, 6.5, 6.2, 5.9,
    ];
    // Feature 1: sepal_width
    let sepal_width: &[FeatureType] = &[
        3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1, 3.7, 3.4, 3.0, 3.0, 4.0, 4.4, 3.9, 3.5,
        3.8, 3.8, 3.4, 3.7, 3.6, 3.3, 3.4, 3.0, 3.4, 3.5, 3.4, 3.2, 3.1, 3.4, 4.1, 4.2, 3.1, 3.2,
        3.5, 3.6, 3.0, 3.4, 3.5, 2.3, 3.2, 3.5, 3.8, 3.0, 3.8, 3.2, 3.7, 3.3, 3.2, 3.2, 3.1, 2.3,
        2.8, 2.8, 3.3, 2.4, 2.9, 2.7, 2.0, 3.0, 2.2, 2.9, 2.9, 3.1, 3.0, 2.7, 2.2, 2.5, 3.2, 2.8,
        2.5, 2.8, 2.9, 3.0, 2.8, 3.0, 2.9, 2.6, 2.4, 2.4, 2.7, 2.7, 3.0, 3.4, 3.1, 2.3, 3.0, 2.5,
        2.6, 3.0, 2.6, 2.3, 2.7, 3.0, 2.9, 2.9, 2.5, 2.8, 3.3, 2.7, 3.0, 2.9, 3.0, 3.0, 2.5, 2.9,
        2.5, 3.6, 3.2, 2.7, 3.0, 2.5, 2.8, 3.2, 3.0, 3.8, 2.6, 2.2, 3.2, 2.8, 2.8, 2.7, 3.3, 3.2,
        2.8, 3.0, 2.8, 3.0, 2.8, 3.8, 2.8, 2.8, 2.6, 3.0, 3.4, 3.1, 3.0, 3.1, 3.1, 3.1, 2.7, 3.2,
        3.3, 3.0, 2.5, 3.0, 3.4, 3.,
    ];
    // Feature 2: petal_length
    let petal_length: &[FeatureType] = &[
        1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5, 1.5, 1.6, 1.4, 1.1, 1.2, 1.5, 1.3, 1.4,
        1.7, 1.5, 1.7, 1.5, 1.0, 1.7, 1.9, 1.6, 1.6, 1.5, 1.4, 1.6, 1.6, 1.5, 1.5, 1.4, 1.5, 1.2,
        1.3, 1.4, 1.3, 1.5, 1.3, 1.3, 1.3, 1.6, 1.9, 1.4, 1.6, 1.4, 1.5, 1.4, 4.7, 4.5, 4.9, 4.0,
        4.6, 4.5, 4.7, 3.3, 4.6, 3.9, 3.5, 4.2, 4.0, 4.7, 3.6, 4.4, 4.5, 4.1, 4.5, 3.9, 4.8, 4.0,
        4.9, 4.7, 4.3, 4.4, 4.8, 5.0, 4.5, 3.5, 3.8, 3.7, 3.9, 5.1, 4.5, 4.5, 4.7, 4.4, 4.1, 4.0,
        4.4, 4.6, 4.0, 3.3, 4.2, 4.2, 4.2, 4.3, 3.0, 4.1, 6.0, 5.1, 5.9, 5.6, 5.8, 6.6, 4.5, 6.3,
        5.8, 6.1, 5.1, 5.3, 5.5, 5.0, 5.1, 5.3, 5.5, 6.7, 6.9, 5.0, 5.7, 4.9, 6.7, 4.9, 5.7, 6.0,
        4.8, 4.9, 5.6, 5.8, 6.1, 6.4, 5.6, 5.1, 5.6, 6.1, 5.6, 5.5, 4.8, 5.4, 5.6, 5.1, 5.1, 5.9,
        5.7, 5.2, 5.0, 5.2, 5.4, 5.1,
    ];
    // Feature 3: petal_width
    let petal_width: &[FeatureType] = &[
        0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.4, 0.4, 0.3,
        0.3, 0.3, 0.2, 0.4, 0.2, 0.5, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.4, 0.1, 0.2, 0.2, 0.2,
        0.2, 0.1, 0.2, 0.2, 0.3, 0.3, 0.2, 0.6, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 1.4, 1.5, 1.5, 1.3,
        1.5, 1.3, 1.6, 1.0, 1.3, 1.4, 1.0, 1.5, 1.0, 1.4, 1.3, 1.4, 1.5, 1.0, 1.5, 1.1, 1.8, 1.3,
        1.5, 1.2, 1.3, 1.4, 1.4, 1.7, 1.5, 1.0, 1.1, 1.0, 1.2, 1.6, 1.5, 1.6, 1.5, 1.3, 1.3, 1.3,
        1.2, 1.4, 1.2, 1.0, 1.3, 1.2, 1.3, 1.3, 1.1, 1.3, 2.5, 1.9, 2.1, 1.8, 2.2, 2.1, 1.7, 1.8,
        1.8, 2.5, 2.0, 1.9, 2.1, 2.0, 2.4, 2.3, 1.8, 2.2, 2.3, 1.5, 2.3, 2.0, 2.0, 1.8, 2.1, 1.8,
        1.8, 1.8, 2.1, 1.6, 1.9, 2.0, 2.2, 1.5, 1.4, 2.3, 2.4, 1.8, 1.8, 2.1, 2.4, 2.3, 1.9, 2.3,
        2.5, 2.3, 1.9, 2.0, 2.3, 1.8,
    ];
    // Labels: 75 × -1, then 75 × +1 (binary Iris: setosa+half-versicolor vs rest)
    let y: Vec<LabelType> = (0..77)
        .map(|_| -1.0 as LabelType)
        .chain((0..73).map(|_| 1.0 as LabelType))
        .collect();

    let n = 150usize;
    let ncols = 4usize;
    // Transpose column-major → row-major
    let mut x_data = Vec::with_capacity(n * ncols);
    for i in 0..n {
        x_data.push(sepal_length[i]);
        x_data.push(sepal_width[i]);
        x_data.push(petal_length[i]);
        x_data.push(petal_width[i]);
    }

    DenseDataset::new(x_data, y, n, ncols, false).unwrap()
}

/// Quadratic loss: f(w) = 0.5 * w·w, ∇f(w) = w. Data-independent.
pub struct QuadraticLoss;

impl QuadraticLoss {
    pub fn new() -> Self {
        Self
    }
}

impl LossFunc for QuadraticLoss {
    fn evaluate_with_gradient(
        &mut self,
        dataset: &dyn Dataset,
        w: &[FeatureType],
        grad: &mut [FeatureType],
    ) -> ScalarType {
        grad.fill(0.0);
        let n: usize = dataset.nrows();
        let inv_n: ScalarType = 1.0 as ScalarType / n as ScalarType;
        let w_dot_w: ScalarType = w.iter().map(|&wi| wi * wi).sum();
        let loss = 0.5 * w_dot_w * inv_n;
        for i in 0..w.len() {
            grad[i] += w[i] * inv_n;
        }
        loss
    }
}

fn dummy_dataset(n_features: usize, n_samples: usize) -> DenseDataset {
    let x_data = vec![0.0 as FeatureType; n_features * n_samples];
    let y_data = vec![0.0 as LabelType; n_samples];
    DenseDataset::new(x_data, y_data, n_samples, n_features, false).unwrap()
}

fn default_params(condition: LineSearchCondition) -> LineSearchParam {
    LineSearchParam {
        dec_factor: 0.5,
        inc_factor: 2.0,
        ftol: 0.1,
        wolfe: 0.1,
        max_stepsize: 1e10,
        min_stepsize: 1e-10,
        max_linesearch_iters: 100,
        max_searches: 100,
        condition,
    }
}

fn epsilon() -> ScalarType {
    if cfg!(feature = "f32") {
        1e-5 as ScalarType
    } else {
        1e-10 as ScalarType
    }
}

fn assert_close(a: ScalarType, b: ScalarType) {
    assert!(
        (a - b).abs() <= epsilon() * (1.0 + a.abs() + b.abs()),
        "assert_close failed: {} vs {} (eps={})",
        a,
        b,
        epsilon()
    );
}

fn assert_close_slice(a: &[ScalarType], b: &[ScalarType]) {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() <= epsilon() * (1.0 + x.abs() + y.abs()),
            "assert_close_slice failed at [{}]: {} vs {}",
            i,
            x,
            y
        );
    }
}

// ── Algorithm factory ──

#[derive(Clone, Copy, Debug)]
enum Algo {
    Backtracking,
    Bracketing,
    // MoreThuente,  // future
}

impl Algo {
    fn make(
        self,
        n_features: usize,
        n_samples: usize,
        params: LineSearchParam,
    ) -> Box<dyn LineSearch> {
        match self {
            Algo::Backtracking => Box::new(BacktrackingLineSearch::new(
                dummy_dataset(n_features, n_samples),
                QuadraticLoss::new(),
                params,
            )),
            Algo::Bracketing => Box::new(BracketingLineSearch::new(
                dummy_dataset(n_features, n_samples),
                QuadraticLoss::new(),
                params,
            )),
        }
    }
}

/// Bundles search inputs/outputs. `run` accepts any `LineSearch` impl.
struct SearchFixture {
    xp: Vec<FeatureType>,
    gp: Vec<FeatureType>,
    d: Vec<FeatureType>,
    x: Vec<FeatureType>,
    g: Vec<FeatureType>,
    fx: ScalarType,
    stepsize: ScalarType,
}

impl SearchFixture {
    fn new(
        xp: Vec<FeatureType>,
        d: Vec<FeatureType>,
        stepsize: ScalarType,
        n_samples: usize,
    ) -> Self {
        let gp: Vec<FeatureType> = xp.iter().map(|&v| v / n_samples as ScalarType).collect();
        let fx = 0.5 * xp.iter().map(|&v| v * v).sum::<ScalarType>() / n_samples as ScalarType;
        let n = xp.len();
        Self {
            xp,
            gp,
            d,
            x: vec![0.0; n],
            g: vec![0.0; n],
            fx,
            stepsize,
        }
    }

    fn run(&mut self, ls: &mut dyn LineSearch) -> Result<usize, LbfgsError> {
        ls.search(
            &self.xp,
            &self.gp,
            &self.d,
            &mut self.x,
            &mut self.g,
            &mut self.fx,
            &mut self.stepsize,
        )
    }
}

// ── Shared case bodies (algo-agnostic) ──
//
// Each takes `&mut dyn LineSearch` so backtracking / bracketing / future
// algos share identical assertions. Cases whose step-update rules diverge
// get looser checks (Ok/Err + qualitative properties) rather than exact
// (count, stepsize) pairs.

mod cases {
    use super::*;

    // ── Contract ──

    pub fn c01_stepsize_zero(ls: &mut dyn LineSearch) {
        let mut fix = SearchFixture::new(vec![10.0, 0.0], vec![-1.0, 0.0], 0.0, 1);
        assert_eq!(fix.run(ls), Err(LbfgsError::InvalidParameters));
    }

    pub fn c02_stepsize_negative(ls: &mut dyn LineSearch) {
        let mut fix = SearchFixture::new(vec![10.0, 0.0], vec![-1.0, 0.0], -1.0, 1);
        assert_eq!(fix.run(ls), Err(LbfgsError::InvalidParameters));
    }

    pub fn c03_stepsize_tiny_negative(ls: &mut dyn LineSearch) {
        let mut fix = SearchFixture::new(vec![10.0, 0.0], vec![-1.0, 0.0], -1e-12, 1);
        assert_eq!(fix.run(ls), Err(LbfgsError::InvalidParameters));
    }

    pub fn c04_dg_init_positive(ls: &mut dyn LineSearch) {
        let mut fix = SearchFixture::new(vec![10.0, 0.0], vec![1.0, 0.0], 1.0, 1);
        assert_eq!(fix.run(ls), Err(LbfgsError::IncreaseGradient));
    }

    /// dg_init == 0: passes gate, Armijo fails until step is tiny.
    /// Both algos shrink (×0.5 vs bisect) so qualitative props hold.
    pub fn c05_dg_init_zero_enters_loop(ls: &mut dyn LineSearch) {
        let mut fix = SearchFixture::new(vec![10.0, 0.0], vec![0.0, 1.0], 1.0, 1);
        let result = fix.run(ls);
        assert_ne!(result, Err(LbfgsError::IncreaseGradient));
        assert!(
            result.is_ok(),
            "expected Ok due to fp rounding, got {:?}",
            result
        );
        let count = result.unwrap();
        assert!(count > 10, "expected many shrinks, got count={}", count);
        assert!(fix.stepsize < 1e-5);
        assert!(fix.fx.is_finite());
    }

    pub fn c06_dg_init_tiny_negative(ls: &mut dyn LineSearch) {
        let mut fix = SearchFixture::new(vec![1e-150, 0.0], vec![-1.0, 0.0], 1.0, 1);
        let result = fix.run(ls);
        assert_ne!(result, Err(LbfgsError::IncreaseGradient));
    }

    // ── Armijo ──
    // xp=[10,0], d=[-1,0] → fx_init=50, dg_init=-10
    // Satisfied when t ∈ (0, 18]

    pub fn a01_one_step_satisfies(ls: &mut dyn LineSearch) {
        let mut fix = SearchFixture::new(vec![10.0, 0.0], vec![-1.0, 0.0], 1.0, 1);
        assert_eq!(fix.run(ls), Ok(1));
        assert!(fix.fx < 50.0);
        assert_eq!(fix.stepsize, 1.0);
        assert_close_slice(&fix.x, &[9.0, 0.0]);
    }

    /// t=20 fails Armijo; both ×0.5 (backtrack) and bisect → t=10 succeeds.
    pub fn a02_shrink_once(ls: &mut dyn LineSearch) {
        let mut fix = SearchFixture::new(vec![10.0, 0.0], vec![-1.0, 0.0], 20.0, 1);
        assert_eq!(fix.run(ls), Ok(2));
        assert_eq!(fix.stepsize, 10.0);
        assert!(fix.fx < 50.0);
    }

    pub fn a03_large_stepsize(ls: &mut dyn LineSearch) {
        let mut fix = SearchFixture::new(vec![10.0, 0.0], vec![-1.0, 0.0], 1e6, 1);
        let result = fix.run(ls);
        assert!(result.is_ok());
        assert!(fix.fx < 50.0);
        assert!(fix.stepsize < 1e6);
    }

    /// Backtracking-only: dec_factor=1.0 freezes step → max iters.
    /// Bracketing ignores dec_factor and bisects → succeeds.
    pub fn a04_no_shrink_diverges_backtracking(ls: &mut dyn LineSearch) {
        let mut fix = SearchFixture::new(vec![10.0, 0.0], vec![-1.0, 0.0], 20.0, 1);
        assert_eq!(fix.run(ls), Err(LbfgsError::MaximumLineSearchIteration));
    }

    pub fn a04_bisect_succeeds_bracketing(ls: &mut dyn LineSearch) {
        let mut fix = SearchFixture::new(vec![10.0, 0.0], vec![-1.0, 0.0], 20.0, 1);
        let result = fix.run(ls);
        assert!(
            result.is_ok(),
            "bracketing should bisect and succeed, got {:?}",
            result
        );
        assert!(fix.fx < 50.0);
        assert!(fix.stepsize < 20.0);
    }

    pub fn a05_lenient_ftol(ls: &mut dyn LineSearch) {
        let mut fix = SearchFixture::new(vec![10.0, 0.0], vec![-1.0, 0.0], 1.0, 1);
        assert_eq!(fix.run(ls), Ok(1));
    }

    pub fn a06_strict_ftol(ls: &mut dyn LineSearch) {
        let mut fix = SearchFixture::new(vec![10.0, 0.0], vec![-1.0, 0.0], 1.0, 1);
        let result = fix.run(ls);
        assert!(
            result.is_ok(),
            "expected Ok due to fp rounding, got {:?}",
            result
        );
        let count = result.unwrap();
        assert!(count > 10, "expected many shrinks, got count={}", count);
        assert!(fix.stepsize < 1e-5);
        assert!(fix.fx.is_finite());
    }

    // ── Wolfe ──
    // Growing while hi=∞: bracketing doubles (= backtracking with inc=2).

    pub fn w01_wolfe_lower_bound_grow(ls: &mut dyn LineSearch) {
        let mut fix = SearchFixture::new(vec![10.0, 0.0], vec![-1.0, 0.0], 1.0, 1);
        assert_eq!(fix.run(ls), Ok(5));
        assert_eq!(fix.stepsize, 16.0);
    }

    pub fn w02_standard_wolfe_not_strong(ls: &mut dyn LineSearch) {
        let mut fix = SearchFixture::new(vec![10.0, 0.0], vec![-1.0, 0.0], 16.0, 1);
        assert_eq!(fix.run(ls), Ok(1));
        let dg_at_step: ScalarType = 16.0 - 10.0;
        let strong_bound: ScalarType = -0.1 * (-10.0);
        assert!(dg_at_step.abs() > strong_bound);
    }

    pub fn w03_wolfe_satisfied_immediately(ls: &mut dyn LineSearch) {
        let mut fix = SearchFixture::new(vec![10.0, 0.0], vec![-1.0, 0.0], 10.0, 1);
        assert_eq!(fix.run(ls), Ok(1));
        assert_eq!(fix.stepsize, 10.0);
    }

    pub fn w04_wolfe_zero_degenerate(ls: &mut dyn LineSearch) {
        let mut fix = SearchFixture::new(vec![10.0, 0.0], vec![-1.0, 0.0], 1.0, 1);
        assert_eq!(fix.run(ls), Ok(5));
    }

    pub fn w05_lenient_wolfe(ls: &mut dyn LineSearch) {
        let mut fix = SearchFixture::new(vec![10.0, 0.0], vec![-1.0, 0.0], 1.0, 1);
        assert_eq!(fix.run(ls), Ok(1));
        assert_eq!(fix.stepsize, 1.0);
    }

    // ── Bounds ──

    pub fn b01_below_min_but_satisfies(ls: &mut dyn LineSearch) {
        let mut fix = SearchFixture::new(vec![10.0, 0.0], vec![-1.0, 0.0], 1e-6, 1);
        assert_eq!(fix.run(ls), Ok(1));
    }

    pub fn b02_below_min_and_fails(ls: &mut dyn LineSearch) {
        let mut fix = SearchFixture::new(vec![0.0, 0.0], vec![1.0, 0.0], 1e-6, 1);
        assert_eq!(fix.run(ls), Err(LbfgsError::MinimumStep));
    }

    pub fn b03_shrink_to_min(ls: &mut dyn LineSearch) {
        let mut fix = SearchFixture::new(vec![0.0, 0.0], vec![0.0, 1.0], 1.0, 1);
        assert_eq!(fix.run(ls), Err(LbfgsError::MinimumStep));
    }

    pub fn b04_grow_past_max(ls: &mut dyn LineSearch) {
        let mut fix = SearchFixture::new(vec![10.0, 0.0], vec![-1.0, 0.0], 1.0, 1);
        assert_eq!(fix.run(ls), Err(LbfgsError::MaximumStep));
    }

    pub fn b05_max_iters_one(ls: &mut dyn LineSearch) {
        let mut fix = SearchFixture::new(vec![10.0, 0.0], vec![-1.0, 0.0], 20.0, 1);
        assert_eq!(fix.run(ls), Err(LbfgsError::MaximumLineSearchIteration));
    }

    pub fn b06a_max_searches_zero_satisfies(ls: &mut dyn LineSearch) {
        let mut fix = SearchFixture::new(vec![10.0, 0.0], vec![-1.0, 0.0], 1.0, 1);
        assert_eq!(fix.run(ls), Ok(1));
    }

    pub fn b06b_max_iters_zero_fails(ls: &mut dyn LineSearch) {
        let mut fix = SearchFixture::new(vec![10.0, 0.0], vec![-1.0, 0.0], 20.0, 1);
        assert_eq!(fix.run(ls), Err(LbfgsError::MaximumLineSearchIteration));
    }

    pub fn b07_large_max_searches(ls: &mut dyn LineSearch) {
        let mut fix = SearchFixture::new(vec![10.0, 0.0], vec![-1.0, 0.0], 1.0, 1);
        assert_eq!(fix.run(ls), Ok(5));
    }

    // ── Output side effects ──

    pub fn o01_x_correct(ls: &mut dyn LineSearch) {
        let mut fix = SearchFixture::new(vec![10.0, 0.0], vec![-1.0, 0.0], 1.0, 1);
        fix.run(ls).unwrap();
        assert_close_slice(&fix.x, &[9.0, 0.0]);
    }

    pub fn o02_g_correct(ls: &mut dyn LineSearch) {
        let mut fix = SearchFixture::new(vec![10.0, 0.0], vec![-1.0, 0.0], 1.0, 1);
        fix.run(ls).unwrap();
        assert_close_slice(&fix.g, &fix.x);
    }

    pub fn o03_fx_correct(ls: &mut dyn LineSearch) {
        let mut fix = SearchFixture::new(vec![10.0, 0.0], vec![-1.0, 0.0], 1.0, 1);
        fix.run(ls).unwrap();
        let expected = 0.5 * fix.x.iter().map(|&v| v * v).sum::<ScalarType>();
        assert_close(fix.fx, expected);
    }

    pub fn o04_stepsize_final(ls: &mut dyn LineSearch) {
        let mut fix = SearchFixture::new(vec![10.0, 0.0], vec![-1.0, 0.0], 20.0, 1);
        fix.run(ls).unwrap();
        assert_eq!(fix.stepsize, 10.0);
    }

    pub fn o05_stepsize_on_error(ls: &mut dyn LineSearch) {
        let mut fix = SearchFixture::new(vec![10.0, 0.0], vec![-1.0, 0.0], 20.0, 1);
        let _ = fix.run(ls);
        assert_eq!(fix.stepsize, 20.0);
    }

    pub fn o06_snapshot_on_error(ls: &mut dyn LineSearch) {
        let mut fix = SearchFixture::new(vec![10.0, 0.0], vec![-1.0, 0.0], 20.0, 1);
        let _ = fix.run(ls);
        assert_close_slice(&fix.x, &[-10.0, 0.0]);
        assert_close_slice(&fix.g, &[-10.0, 0.0]);
        assert_close(fix.fx, 50.0);
    }

    // ── Numeric ──

    pub fn n01_tiny_values(ls: &mut dyn LineSearch) {
        let mut fix = SearchFixture::new(vec![1e-10, 0.0], vec![-1e-10, 0.0], 1e-5, 1);
        let result = fix.run(ls);
        assert!(result.is_ok());
        assert!(fix.fx.is_finite());
    }

    pub fn n02_large_values(ls: &mut dyn LineSearch) {
        let mut fix = SearchFixture::new(vec![1e10, 0.0], vec![-1e10, 0.0], 1e5, 1);
        let _ = fix.run(ls);
        assert!(fix.fx.is_finite());
    }

    pub fn n07_tiny_negative_dg(ls: &mut dyn LineSearch) {
        let mut fix = SearchFixture::new(vec![1e-150, 0.0], vec![-1.0, 0.0], 1.0, 1);
        let result = fix.run(ls);
        assert_ne!(result, Err(LbfgsError::IncreaseGradient));
    }

    // ── Dimensions ──

    pub fn run_dim_test(algo: Algo, n_features: usize, n_samples: usize) {
        let mut ls = algo.make(
            n_features,
            n_samples,
            default_params(LineSearchCondition::Armijo),
        );
        let xp = vec![10.0; n_features];
        let d = vec![-1.0; n_features];
        let mut fix = SearchFixture::new(xp, d, 1.0, n_samples);
        assert_eq!(fix.run(ls.as_mut()), Ok(1));
        let expected_x = vec![9.0; n_features];
        assert_close_slice(&fix.x, &expected_x);
    }
}

// ── Suite macro: generates one identical mod per algorithm ──

macro_rules! linesearch_suite {
    ($mod_name:ident, $algo:expr) => {
        mod $mod_name {
            use super::*;

            fn make(params: LineSearchParam) -> Box<dyn LineSearch> {
                $algo.make(2, 1, params)
            }

            // ── contract ──
            mod contract {
                use super::*;

                #[test]
                fn c01_stepsize_zero() {
                    cases::c01_stepsize_zero(
                        make(default_params(LineSearchCondition::Armijo)).as_mut(),
                    );
                }
                #[test]
                fn c02_stepsize_negative() {
                    cases::c02_stepsize_negative(
                        make(default_params(LineSearchCondition::Armijo)).as_mut(),
                    );
                }
                #[test]
                fn c03_stepsize_tiny_negative() {
                    cases::c03_stepsize_tiny_negative(
                        make(default_params(LineSearchCondition::Armijo)).as_mut(),
                    );
                }
                #[test]
                fn c04_dg_init_positive() {
                    cases::c04_dg_init_positive(
                        make(default_params(LineSearchCondition::Armijo)).as_mut(),
                    );
                }
                #[test]
                fn c05_dg_init_zero_enters_loop() {
                    cases::c05_dg_init_zero_enters_loop(
                        make(default_params(LineSearchCondition::Armijo)).as_mut(),
                    );
                }
                #[test]
                fn c06_dg_init_tiny_negative() {
                    cases::c06_dg_init_tiny_negative(
                        make(default_params(LineSearchCondition::Armijo)).as_mut(),
                    );
                }
            }

            // ── armijo ──
            mod armijo {
                use super::*;

                #[test]
                fn a01_one_step_satisfies() {
                    cases::a01_one_step_satisfies(
                        make(default_params(LineSearchCondition::Armijo)).as_mut(),
                    );
                }
                #[test]
                fn a02_shrink_once() {
                    cases::a02_shrink_once(
                        make(default_params(LineSearchCondition::Armijo)).as_mut(),
                    );
                }
                #[test]
                fn a03_large_stepsize() {
                    cases::a03_large_stepsize(
                        make(default_params(LineSearchCondition::Armijo)).as_mut(),
                    );
                }
                #[test]
                fn a04_dec_factor_behavior() {
                    let mut params = default_params(LineSearchCondition::Armijo);
                    params.dec_factor = 1.0;
                    let mut ls = make(params);
                    // Divergence point: backtracking freezes; bracketing bisects.
                    match $algo {
                        Algo::Backtracking => {
                            cases::a04_no_shrink_diverges_backtracking(ls.as_mut());
                        }
                        Algo::Bracketing => {
                            cases::a04_bisect_succeeds_bracketing(ls.as_mut());
                        }
                    }
                }
                #[test]
                fn a05_lenient_ftol() {
                    let mut params = default_params(LineSearchCondition::Armijo);
                    params.ftol = 1e-12;
                    cases::a05_lenient_ftol(make(params).as_mut());
                }
                #[test]
                fn a06_strict_ftol() {
                    let mut params = default_params(LineSearchCondition::Armijo);
                    params.ftol = 1.0;
                    cases::a06_strict_ftol(make(params).as_mut());
                }
            }

            // ── wolfe ──
            mod wolfe {
                use super::*;

                #[test]
                fn w01_wolfe_lower_bound_grow() {
                    cases::w01_wolfe_lower_bound_grow(
                        make(default_params(LineSearchCondition::Wolfe)).as_mut(),
                    );
                }
                #[test]
                fn w02_standard_wolfe_not_strong() {
                    cases::w02_standard_wolfe_not_strong(
                        make(default_params(LineSearchCondition::Wolfe)).as_mut(),
                    );
                }
                #[test]
                fn w03_wolfe_satisfied_immediately() {
                    let mut params = default_params(LineSearchCondition::Wolfe);
                    params.wolfe = 0.5;
                    cases::w03_wolfe_satisfied_immediately(make(params).as_mut());
                }
                #[test]
                fn w04_wolfe_zero_degenerate() {
                    let mut params = default_params(LineSearchCondition::Wolfe);
                    params.wolfe = 0.0;
                    cases::w04_wolfe_zero_degenerate(make(params).as_mut());
                }
                #[test]
                fn w05_lenient_wolfe() {
                    let mut params = default_params(LineSearchCondition::Wolfe);
                    params.wolfe = 0.9;
                    cases::w05_lenient_wolfe(make(params).as_mut());
                }
            }

            // ── bounds ──
            mod bounds {
                use super::*;

                #[test]
                fn b01_below_min_but_satisfies() {
                    let mut params = default_params(LineSearchCondition::Armijo);
                    params.min_stepsize = 1e-5;
                    cases::b01_below_min_but_satisfies(make(params).as_mut());
                }
                #[test]
                fn b02_below_min_and_fails() {
                    let mut params = default_params(LineSearchCondition::Armijo);
                    params.min_stepsize = 1e-5;
                    cases::b02_below_min_and_fails(make(params).as_mut());
                }
                #[test]
                fn b03_shrink_to_min() {
                    cases::b03_shrink_to_min(
                        make(default_params(LineSearchCondition::Armijo)).as_mut(),
                    );
                }
                #[test]
                fn b04_grow_past_max() {
                    let mut params = default_params(LineSearchCondition::Wolfe);
                    params.max_stepsize = 5.0;
                    cases::b04_grow_past_max(make(params).as_mut());
                }
                #[test]
                fn b05_max_iters_one() {
                    let mut params = default_params(LineSearchCondition::Armijo);
                    params.max_linesearch_iters = 1;
                    cases::b05_max_iters_one(make(params).as_mut());
                }
                #[test]
                fn b06a_max_searches_zero_satisfies() {
                    let mut params = default_params(LineSearchCondition::Armijo);
                    params.max_searches = 0;
                    cases::b06a_max_searches_zero_satisfies(make(params).as_mut());
                }
                #[test]
                fn b06b_max_iters_zero_fails() {
                    let mut params = default_params(LineSearchCondition::Armijo);
                    params.max_linesearch_iters = 0;
                    cases::b06b_max_iters_zero_fails(make(params).as_mut());
                }
                #[test]
                fn b07_large_max_searches() {
                    let mut params = default_params(LineSearchCondition::Wolfe);
                    params.max_searches = 10000;
                    cases::b07_large_max_searches(make(params).as_mut());
                }
            }

            // ── output ──
            mod output {
                use super::*;

                #[test]
                fn o01_x_correct() {
                    cases::o01_x_correct(
                        make(default_params(LineSearchCondition::Armijo)).as_mut(),
                    );
                }
                #[test]
                fn o02_g_correct() {
                    cases::o02_g_correct(
                        make(default_params(LineSearchCondition::Armijo)).as_mut(),
                    );
                }
                #[test]
                fn o03_fx_correct() {
                    cases::o03_fx_correct(
                        make(default_params(LineSearchCondition::Armijo)).as_mut(),
                    );
                }
                #[test]
                fn o04_stepsize_final() {
                    cases::o04_stepsize_final(
                        make(default_params(LineSearchCondition::Armijo)).as_mut(),
                    );
                }
                #[test]
                fn o05_stepsize_on_error() {
                    let mut params = default_params(LineSearchCondition::Armijo);
                    params.max_linesearch_iters = 1;
                    cases::o05_stepsize_on_error(make(params).as_mut());
                }
                #[test]
                fn o06_snapshot_on_error() {
                    let mut params = default_params(LineSearchCondition::Armijo);
                    params.max_linesearch_iters = 1;
                    cases::o06_snapshot_on_error(make(params).as_mut());
                }
            }

            // ── numeric ──
            mod numeric {
                use super::*;

                #[test]
                fn n01_tiny_values() {
                    cases::n01_tiny_values(
                        make(default_params(LineSearchCondition::Armijo)).as_mut(),
                    );
                }
                #[test]
                fn n02_large_values() {
                    cases::n02_large_values(
                        make(default_params(LineSearchCondition::Armijo)).as_mut(),
                    );
                }
                #[test]
                fn n07_tiny_negative_dg() {
                    cases::n07_tiny_negative_dg(
                        make(default_params(LineSearchCondition::Armijo)).as_mut(),
                    );
                }
            }

            // ── dimensions ──
            mod dimensions {
                use super::*;

                #[test]
                fn d01_scalar() {
                    cases::run_dim_test($algo, 1, 1);
                }
                #[test]
                fn d02_scalar_many_samples() {
                    cases::run_dim_test($algo, 1, 100);
                }
                #[test]
                fn d03_iris_features() {
                    cases::run_dim_test($algo, 4, 1);
                }
                #[test]
                fn d04_small() {
                    cases::run_dim_test($algo, 8, 1);
                }
                #[test]
                fn d05_medium() {
                    cases::run_dim_test($algo, 32, 1);
                }
                #[test]
                fn d06_hidden_layer() {
                    cases::run_dim_test($algo, 128, 1);
                }
                #[test]
                fn d07_mnist() {
                    cases::run_dim_test($algo, 784, 1);
                }
                #[test]
                fn d08_medium_large() {
                    cases::run_dim_test($algo, 1024, 1);
                }
                #[test]
                fn d09_large() {
                    cases::run_dim_test($algo, 4096, 1);
                }
                #[test]
                fn d10_many_samples() {
                    cases::run_dim_test($algo, 4, 1000);
                }
            }
        }
    };
}

// Instantiate one suite per algorithm. Adding MoreThuente later:
//   linesearch_suite!(morethuente, Algo::MoreThuente);
linesearch_suite!(backtracking, Algo::Backtracking);
linesearch_suite!(bracketing, Algo::Bracketing);

// ── C++ reference: backtracking-only (golden values from liblbfgs) ──
#[cfg(feature = "f64")]
mod cpp_reference {
    use super::*;
    use crate::core::loss::logistic::LogLoss;

    #[test]
    fn iris_logloss_wolfe_matches_cpp() {
        let dataset = build_iris_dataset();
        let loss_fn = LogLoss::new();

        let params = LineSearchParam {
            dec_factor: 0.5,
            inc_factor: 2.1,
            ftol: 1e-4,
            wolfe: 0.9,
            max_stepsize: 1e+20,
            min_stepsize: 1e-20,
            max_linesearch_iters: 20,
            max_searches: 40,
            condition: LineSearchCondition::Wolfe,
        };

        let mut ls = BacktrackingLineSearch::new(dataset, loss_fn, params);

        let tol: ScalarType = 1e-5;

        // Pre-set inputs for both iterations (from C++ test)
        let xp: [Vec<FeatureType>; 2] = [
            vec![1.0, 1.0, 1.0, 1.0],
            vec![-0.67094175, 0.00470329, 0.23273091, 0.80343884],
        ];
        let mut x: [Vec<FeatureType>; 2] = [
            vec![1.0, 1.0, 1.0, 1.0],
            vec![-0.67094175, 0.00470329, 0.23273091, 0.80343884],
        ];
        let mut g: [Vec<FeatureType>; 2] = [
            vec![2.75991281, 1.64394249, 1.26730677, 0.32466222],
            vec![-2.32159527, -1.03857245, -1.92900394, -0.68107431],
        ];
        let d: [Vec<FeatureType>; 2] = [
            vec![-2.75991281, -1.64394249, -1.26730677, -0.32466222],
            vec![0.7449502, 0.38705061, 0.48413709, 0.154796],
        ];
        let mut fx: [ScalarType; 2] = [5.996018216462658, 0.9177489446446626];
        let mut stepsize: ScalarType = 0.2883013346297236;

        let expect_x: [[FeatureType; 4]; 2] = [
            [-0.67094175, 0.00470329, 0.23273091, 0.80343884],
            [-0.21992446, 0.23903643, 0.52584339, 0.89715742],
        ];
        let expect_g: [[FeatureType; 4]; 2] = [
            [-2.32159528, -1.03857245, -1.92900395, -0.68107431],
            [2.04742451, 1.19381727, 0.98833884, 0.26008274],
        ];
        let expect_fx: [ScalarType; 2] = [0.9177489530218861, 0.9059209513061773];
        let expect_step: [ScalarType; 2] = [0.6054328027224196, 0.6054328027224196];

        for i in 0..2 {
            let gp = g[i].clone();
            ls.search(
                &xp[i],
                &gp,
                &d[i],
                &mut x[i],
                &mut g[i],
                &mut fx[i],
                &mut stepsize,
            )
            .expect("search should succeed");

            assert!(
                (stepsize - expect_step[i]).abs() <= tol,
                "iter {} stepsize: {} vs {}",
                i,
                stepsize,
                expect_step[i]
            );
            assert!(
                (fx[i] - expect_fx[i]).abs() <= tol,
                "iter {} fx: {} vs {}",
                i,
                fx[i],
                expect_fx[i]
            );
            for j in 0..4 {
                assert!(
                    (x[i][j] - expect_x[i][j]).abs() <= tol,
                    "iter {} x[{}]: {} vs {}",
                    i,
                    j,
                    x[i][j],
                    expect_x[i][j]
                );
                assert!(
                    (g[i][j] - expect_g[i][j]).abs() <= tol,
                    "iter {} g[{}]: {} vs {}",
                    i,
                    j,
                    g[i][j],
                    expect_g[i][j]
                );
            }
        }
    }
}
