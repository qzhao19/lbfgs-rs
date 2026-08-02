use super::optimizer::{LimitedMemoryBuf, Optimizer};
use crate::algorithm::linesearch::backtracking::BacktrackingLineSearch;
use crate::algorithm::linesearch::bracketing::BracketingLineSearch;
use crate::algorithm::linesearch::linesearch::LineSearch;
use crate::algorithm::loss::logistic::LogLoss;
use crate::algorithm::loss::loss::LossFunc;
use crate::data::dataset::Dataset;
use crate::infra::math::vec_ops::{
    vec_diff, vec_dot, vec_ncpy, vec_norm2, vec_scale_inplace, vec_scaled_add_inplace,
};
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

        // Initialize limited-memory correction history
        let mut lm_buf: Vec<LimitedMemoryBuf> =
            std::iter::repeat_with(|| LimitedMemoryBuf::initialize(n_features))
                .take(mem_size)
                .collect();

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
        let mut end: usize = 0;
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

            // Update vector s and y, store at slot `end`
            // s_{k+1} = x_{k+1} - x_{k} = step * d_{k}.
            // y_{k+1} = g_{k+1} - g_{k}.
            vec_diff(&x, &xp, &mut lm_buf[end].mem_s);
            vec_diff(&g, &gp, &mut lm_buf[end].mem_y);

            // Compute scalars ys and yy:
            // ys = y^t @ s, s = 1 / rho.
            // yy = y^t @ y.
            let ys: ScalarType = vec_dot(&lm_buf[end].mem_y, &lm_buf[end].mem_s);
            let yy: ScalarType = vec_dot(&lm_buf[end].mem_y, &lm_buf[end].mem_y);
            lm_buf[end].mem_ys = ys;

            // d = -g
            vec_ncpy(&g, &mut d);

            // bound: number of currently available historical messages
            // k: number of iterations
            // end: indicates the location of the latest history information.
            //      after each iteration, end is updated to the next position
            // j: index for traversing history information
            let bound: usize = if mem_size <= k { mem_size } else { k };
            k += 1;
            end = (end + 1) % mem_size;
            let mut j = end;

            // two-loop recursion — forward pass
            for _ in 0..bound {
                // if (--j == -1) j = m-1 traverse history forward,
                // starting with the most recent history message
                j = (j + mem_size - 1) % mem_size;

                // alpha_{j} = s^{T}_{j} @ d_{j} * rho_{j}, rho_{j} = 1/mem_ys
                let alpha: ScalarType = vec_dot(&lm_buf[j].mem_s, &d) / lm_buf[j].mem_ys;
                lm_buf[j].mem_alpha = alpha;

                // d_{i} = d_{i+1} - (alpha_{i} * y_{i})
                vec_scaled_add_inplace(&lm_buf[j].mem_y, -alpha, &mut d);
            }

            let scale: ScalarType = ys / yy;
            vec_scale_inplace(&mut d, scale);

            // two-loop recursion — backward pass
            for _ in 0..bound {
                // beta_j = rho_{j} * y_{T}_{j} @ d_{J}, rho_{j} = 1/mem_ys
                let beta: ScalarType = vec_dot(&lm_buf[j].mem_y, &d) / lm_buf[j].mem_ys;

                // gamma_{i+1} = gamma_{i} + (alpha_{j} - beta_{j}) * s_{j}
                let coef: ScalarType = lm_buf[j].mem_alpha - beta;
                vec_scaled_add_inplace(&lm_buf[j].mem_s, coef, &mut d);

                // Starting the earliest history information to traverse backward
                j = (j + 1) % mem_size;
            }

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
