use super::super::linesearch::LineSearch;
use crate::algorithm::loss::LossFunc;
use crate::data::dataset::Dataset;
use crate::infra::math::kernel::{vec_dot, vec_scale, vec_scaled_add};
use crate::shared::types::error::LbfgsError;
use crate::shared::types::linesearch::{LineSearchCondition, LineSearchParamType};
use crate::shared::types::primitives::{FeatureType, ScalarType};

pub struct BracketingLineSearch {
    /// Dataset reference owned by linesearch
    pub dataset: Box<dyn Dataset>,

    /// Loss function
    pub loss_fn: Box<dyn LossFunc>,

    /// Linesearch hyperparameters
    pub search_param: LineSearchParamType,
}

impl BracketingLineSearch {
    pub fn new<DatasetType, LossFuncType>(
        dataset: DatasetType,
        loss_fn: LossFuncType,
        search_param: LineSearchParamType,
    ) -> Self
    where
        DatasetType: Dataset + 'static,
        LossFuncType: LossFunc + 'static,
    {
        Self {
            dataset: Box::new(dataset),
            loss_fn: Box::new(loss_fn),
            search_param,
        }
    }
}

impl LineSearch for BracketingLineSearch {
    fn search(
        &mut self,
        xp: &[FeatureType],
        gp: &[FeatureType],
        d: &[FeatureType],
        x: &mut [FeatureType],
        g: &mut [FeatureType],
        fx: &mut ScalarType,
        stepsize: &mut ScalarType,
    ) -> Result<usize, LbfgsError> {
        let n_samples: usize = self.dataset.nrows();
        let n_features: usize = self.dataset.ncols();
        let inv_n_samples: ScalarType = 1.0 as ScalarType / n_samples as ScalarType;

        // step must be positive
        if *stepsize <= 0.0 {
            return Err(LbfgsError::InvalidParameters);
        }

        // fx_init = loss at xp;
        // dg_init = directional derivative along d at xp
        let fx_init = *fx;
        let dg_init = vec_dot(d, gp);

        // search direction must decrease the objective
        if dg_init > 0.0 {
            return Err(LbfgsError::IncreaseGradient);
        }

        // Tolerance threshold: the minimum descent
        let dg_test = self.search_param.ftol * dg_init;

        // Bracketing interval [stepsize_lo, stepsize_hi]
        let mut stepsize_hi: ScalarType = ScalarType::INFINITY;
        let mut stepsize_lo: ScalarType = 0.0;

        // Local buffer to restore accumulated gradient
        let mut acc_grad: Vec<ScalarType> = vec![0.0; n_features];

        let mut count: usize = 0;
        loop {
            // x_{k+1} = x_k + stepsize * d_k
            vec_scaled_add(d, xp, *stepsize, x);

            // Reset gradient buffer
            acc_grad.fill(0.0);

            // Compute acc loss and gradient at the new X
            let mut acc_loss: ScalarType =
                self.loss_fn
                    .evaluate_with_gradient(&*self.dataset, x, &mut acc_grad);

            // Normalize accumulated loss and gradient
            acc_loss *= inv_n_samples;
            vec_scale(&acc_grad, inv_n_samples, g);

            *fx = acc_loss;

            // Increment iteration
            count += 1;

            // Armijo / Wolfe condition logic
            if *fx > fx_init + *stepsize * dg_test {
                // Armijo condition not satisfied: step too large, shrink upper bound
                stepsize_hi = *stepsize
            } else {
                if self.search_param.condition == LineSearchCondition::Armijo {
                    return Ok(count);
                }

                let dg: ScalarType = vec_dot(d, g);
                if dg < self.search_param.wolfe * dg_init {
                    // Wolfe condition not satisfied - step too small, raise lower bound
                    stepsize_lo = *stepsize;
                } else {
                    if self.search_param.condition == LineSearchCondition::Wolfe {
                        return Ok(count);
                    }

                    // Check strong wolfe condition
                    if dg > -self.search_param.wolfe * dg_init {
                        // Strong Wolfe not satisfied - step too large, shrink upper bound
                        stepsize_hi = *stepsize;
                    } else {
                        // Both Armijo and (strong) Wolfe satisfied
                        return Ok(count);
                    }
                }
            }

            // Bounds checks
            if *stepsize < self.search_param.min_stepsize {
                return Err(LbfgsError::MinimumStep);
            }
            if *stepsize > self.search_param.max_stepsize {
                return Err(LbfgsError::MaximumStep);
            }
            if count >= self.search_param.max_searches {
                return Err(LbfgsError::MaximumLineSearch);
            }

            // Update step size: double while the upper bound is still +INF,
            // otherwise bisect the bracketing interval.
            *stepsize = if stepsize_hi.is_infinite() {
                2.0 * *stepsize
            } else {
                (stepsize_lo + stepsize_hi) * 0.5
            };
        }
    }
}
