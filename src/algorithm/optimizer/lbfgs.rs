use super::hessian_approx_mat::LimitedMemHessianApproxMat;
use super::optimizer::Optimizer;
use crate::algorithm::linesearch::backtracking::BacktrackingLineSearch;
use crate::algorithm::linesearch::bracketing::BracketingLineSearch;
use crate::algorithm::linesearch::linesearch::LineSearch;
use crate::algorithm::loss::logistic::LogLoss;
use crate::algorithm::loss::loss::LossFunc;
use crate::data::dataset::Dataset;
use crate::infra::math::vec_ops::{vec_diff, vec_ncpy, vec_norm2};
use crate::shared::exception::{LbfgsError, LbfgsStatus};
use crate::shared::numeric::ScalarType;
use crate::shared::parameter::{LbfgsParams, LineSearchPolicy, LossType};

pub(crate) struct LBFGS {
    lbfgs_params: LbfgsParams,
    x0: Vec<ScalarType>,
    w_opt: Vec<ScalarType>,
    loss_history: Vec<ScalarType>,
    callback: Option<Box<dyn Fn(&[ScalarType])>>,
}

impl LBFGS {
    pub fn new(lbfgs_params: LbfgsParams, x0: Vec<ScalarType>) -> Self {
        Self {
            lbfgs_params,
            x0,
            w_opt: Vec::new(),
            loss_history: Vec::new(),
            callback: None,
        }
    }
}

impl Optimizer for LBFGS {
    fn optimize(&mut self, dataset: Box<dyn Dataset>) -> Result<LbfgsStatus, LbfgsError> {
        let n_features: usize = dataset.ncols();
        self.loss_history.clear();

        // Copy params out
        let epsilon = self.lbfgs_params.epsilon;
        let delta = self.lbfgs_params.delta;
        let max_iters = self.lbfgs_params.max_iters;
        let mem_size = self.lbfgs_params.mem_size;
        let past = self.lbfgs_params.past;
        let loss_type = self.lbfgs_params.loss;
        let linesearch_policy = self.lbfgs_params.linesearch_policy;
        let linesearch_params = self.lbfgs_params.linesearch_params;
        let verbose = self.lbfgs_params.verbose;

        // Copy weight vector x from initialize x0
        let mut x: Vec<ScalarType> = self.x0.clone();

        // Limited-memory inverse-Hessian approximation
        let mut hessian = LimitedMemHessianApproxMat::new(mem_size, n_features);

        // Define intermediate variables: previous x, gradient, previous gradient, directions
        let mut xp: Vec<ScalarType> = vec![0.0; n_features];
        let mut g: Vec<ScalarType> = vec![0.0; n_features];
        let mut gp: Vec<ScalarType> = vec![0.0; n_features];
        let mut d: Vec<ScalarType> = vec![0.0; n_features];

        // Build loss function, evaluate loss/gradient at the initial point
        let mut loss_fn: Box<dyn LossFunc> = match loss_type {
            LossType::LogLoss => Box::new(LogLoss::new()),
        };
        let mut fx: ScalarType = loss_fn.evaluate_with_gradient(&*dataset, &x, &mut g);

        // Build line search
        let mut ls: Box<dyn LineSearch> = match linesearch_policy {
            LineSearchPolicy::Backtracking => Box::new(BacktrackingLineSearch {
                dataset,
                loss_fn,
                linesearch_params,
            }),
            LineSearchPolicy::Bracketing => Box::new(BracketingLineSearch {
                dataset,
                loss_fn,
                linesearch_params,
            }),
        };

        // Define a vector for storing past function value
        let mut pfx: Vec<ScalarType> = vec![0.0; std::cmp::max(1, past)];
        pfx[0] = fx;

        // Intialize d = -g, stepsize = 1.0 / ||d||
        vec_ncpy(&g, &mut d);
        let mut stepsize: ScalarType = 1.0 / vec_norm2(&d, false);

        // Compute ||x|| and ||g||
        // Convergence test 0: already at a stationary point
        // ||g(x)|| / max(1, ||x||) < epsilon
        let mut xnorm: ScalarType = vec_norm2(&x, false);
        let mut gnorm: ScalarType = vec_norm2(&g, false);
        if gnorm / xnorm.max(1.0) <= epsilon {
            self.w_opt = x.clone();
            return Ok(LbfgsStatus::AlreadyMinimized);
        }

        let mut k: usize = 1;
        let record_history = self.callback.is_some();
        let result: Result<LbfgsStatus, LbfgsError> = loop {
            // Store current xp = x and gp = g
            xp.copy_from_slice(&x);
            gp.copy_from_slice(&g);

            // Update x, g, fx, stepsize
            let search_status = ls.search(&xp, &gp, &d, &mut x, &mut g, &mut fx, &mut stepsize);
            if let Err(error) = search_status {
                x.copy_from_slice(&xp);
                g.copy_from_slice(&gp);
                break Err(error);
            }

            // Trigger calback to record loss value
            if record_history {
                self.loss_history.push(fx);
            }

            // Convergence test 1: gradient test
            // ||g(x)|| / max(1, ||x||) < epsilon
            xnorm = vec_norm2(&x, false);
            gnorm = vec_norm2(&g, false);

            if verbose {
                println!(
                    "iteration = {}, loss = {}, xnorm value = {}, gnorm value = {}",
                    k, fx, xnorm, gnorm
                );
            }
            if gnorm / xnorm.max(1.0) <= epsilon {
                break Ok(LbfgsStatus::Convergence);
            }

            // Convergence test 2: objective function value
            // |f(past_x) - f(x)| / max(1, |f(x)|) < delta.
            if past > 0 && past <= k {
                let rate = (pfx[k % past] - fx).abs() / fx.abs().max(1.0);
                if rate < delta {
                    break Ok(LbfgsStatus::Stop);
                }
                pfx[k % past] = fx;
            }

            if max_iters != 0 && max_iters < k + 1 {
                break Err(LbfgsError::MaximumIteration);
            }

            // Update correction history: s = x - xp, y = g - gp
            // s_{k+1} = x_{k+1} - x_{k} = step * d_{k}.
            // y_{k+1} = g_{k+1} - g_{k}.
            let mut s: Vec<ScalarType> = vec![0.0; n_features];
            let mut y: Vec<ScalarType> = vec![0.0; n_features];
            vec_diff(&x, &xp, &mut s);
            vec_diff(&g, &gp, &mut y);
            hessian.update(&s, &y);

            // Search direction: d = -g,
            vec_ncpy(&g, &mut d);

            // Compute d <- H * d, two-loop recursion
            hessian.apply_hv(&mut d);

            k += 1;
            stepsize = 1.0;
        };

        self.w_opt = x;

        // Invoke callback function
        if let Some(cb) = &self.callback {
            cb(&self.loss_history);
        }

        return result;
    }

    fn get_weight(&self) -> Vec<ScalarType> {
        return self.w_opt.clone();
    }

    fn set_callback(&mut self, callback: Box<dyn Fn(&[ScalarType])>) {
        self.callback = Some(callback);
    }
}
