mod algorithm;
mod data;
mod infra;
mod shared;

// Re-export public types for callers.
pub use crate::shared::numeric::{FeatureType, LabelType, ScalarType};
pub use crate::shared::parameter::OptimizeArgs;

use crate::algorithm::optimizer::lbfgs::LBFGS as InnerLbfgs;
use crate::algorithm::optimizer::optimizer::Optimizer;
use crate::data::dataset::Dataset;
use crate::data::dense::DenseDataset;
use crate::shared::parameter::{LineSearchPolicy, LossType};

/// Public L-BFGS driver.
///
/// Construct with [`LBFGS::new`], run with [`LBFGS::optimize`],
/// then read the solution via [`LBFGS::get_weight`].
///
/// ```ignore
/// let mut opt = lbfgs_rs::LBFGS::new(
///     x0,
///     lbfgs_rs::OptimizeArgs {
///         max_iters: Some(20),
///         condition: Some("wolfe".to_string()),
///         ..Default::default()
///     },
///     Some("lbfgs".to_string()),
///     Some("backtracking".to_string()),
///     Some("logloss".to_string()),
///     None,
///     false,
/// )?;
/// let status = opt.optimize(x_train, y_train)?;
/// let w = opt.get_weight();
/// ```
pub struct LBFGS {
    x0: Vec<ScalarType>,
    args: OptimizeArgs,
    search: Option<String>,
    loss: Option<String>,
    /// One-shot: `Box<dyn Fn>` is not `Clone`, so it is consumed on the first
    /// successful attach inside [`LBFGS::optimize`].
    callback: Option<Box<dyn Fn(&[ScalarType])>>,
    verbose: bool,
    /// Cached solution after a successful (or partially completed) run.
    w_opt: Vec<ScalarType>,
}

impl LBFGS {
    /// Construct a new L-BFGS driver.
    ///
    /// # Arguments
    /// - `x0`: initial parameter vector (weights). Length is `n_features`.
    /// - `args`: optional hyperparameters; `None` fields keep
    ///   `LbfgsParams::default()` values. Pass `OptimizeArgs::default()` for
    ///   all defaults.
    /// - `method`: `Some("lbfgs")` or `None` (defaults to `"lbfgs"`).
    ///   `"lbfgs-b"` is reserved and returns an error.
    /// - `search`: `Some("backtracking")` | `Some("bracketing")` or `None`
    ///   (defaults to `Backtracking`). `"morethuente"` is reserved and
    ///   returns an error.
    /// - `loss`: `Some("logloss")` or `None` (defaults to `LogLoss`).
    /// - `callback`: optional `Fn(&[ScalarType])` invoked once at the end of
    ///   [`LBFGS::optimize`] with the recorded loss history.
    /// - `verbose`: print per-iteration diagnostics when `true`.
    pub fn new(
        x0: Vec<ScalarType>,
        args: OptimizeArgs,
        method: Option<String>,
        search: Option<String>,
        loss: Option<String>,
        callback: Option<Box<dyn Fn(&[ScalarType])>>,
        verbose: bool,
    ) -> Result<Self, String> {
        let method = method.unwrap_or_else(|| "lbfgs".to_string());
        if method != "lbfgs" {
            return Err(format!("method not yet implemented: {}", method));
        }

        Ok(Self {
            x0,
            args,
            search,
            loss,
            callback,
            verbose,
            w_opt: Vec::new(),
        })
    }

    /// Run L-BFGS on the training data `(x, y)`.
    ///
    /// - `x`: row-major flattened feature matrix; length must equal
    ///   `n_features * n_samples`, where `n_features == x0.len()`.
    /// - `y`: per-sample labels; length `n_samples`.
    ///
    /// Returns a termination status message on success, or an error message
    /// on failure. After the call, use [`LBFGS::get_weight`] for the solution.
    pub fn optimize(&mut self, x: Vec<FeatureType>, y: Vec<LabelType>) -> Result<String, String> {
        // Resolve line-search policy.
        let linesearch_policy = match self.search.as_deref() {
            None => LineSearchPolicy::Backtracking,
            Some("backtracking") => LineSearchPolicy::Backtracking,
            Some("bracketing") => LineSearchPolicy::Bracketing,
            Some(s) => return Err(format!("unknown search method: {}", s)),
        };

        // Resolve loss type.
        let loss_type = match self.loss.as_deref() {
            None => LossType::LogLoss,
            Some("logloss") => LossType::LogLoss,
            Some(s) => return Err(format!("unknown loss: {}", s)),
        };

        self.args
            .validate()
            .map_err(|e| e.error_message().to_string())?;

        // Merge args (clone so the driver remains re-runnable on new data).
        let params =
            self.args
                .clone()
                .to_lbfgs_params(loss_type, linesearch_policy, self.verbose)?;

        // Validate data shape and build DenseDataset.
        let n_samples = y.len();
        if n_samples == 0 {
            return Err("empty label vector".to_string());
        }
        if x.len() % n_samples != 0 {
            return Err(format!(
                "feature matrix length {} not divisible by n_samples {}",
                x.len(),
                n_samples
            ));
        }
        let n_features = x.len() / n_samples;
        if n_features != self.x0.len() {
            return Err(format!(
                "n_features ({}) does not match x0 length ({})",
                n_features,
                self.x0.len()
            ));
        }

        let dataset: Box<dyn Dataset> = Box::new(
            DenseDataset::new(x, y, n_samples, n_features, false)
                .map_err(|e| format!("dataset construction failed: {}", e))?,
        );

        // Build inner optimiser, attach one-shot callback, run.
        let mut inner = InnerLbfgs::new(params, self.x0.clone());
        if let Some(cb) = self.callback.take() {
            inner.set_callback(cb);
        }

        let result = inner.optimize(dataset);

        // Always cache the latest weights (even on error the inner path may
        // have a partial / restored solution).
        self.w_opt = inner.get_weight();

        match result {
            Ok(status) => Ok(status.status_message().to_string()),
            Err(error) => Err(error.error_message().to_string()),
        }
    }

    /// Return the optimised weight vector `w_opt` from the last
    /// [`LBFGS::optimize`] call. Empty if `optimize` has not been run yet.
    pub fn get_weight(&self) -> Vec<ScalarType> {
        self.w_opt.clone()
    }
}
