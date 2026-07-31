#![allow(dead_code)]

use super::linesearch::LineSearch;
use crate::core::loss::loss::LossFunc;
use crate::data::dataset::Dataset;
use crate::infra::math::vec_ops::{vec_dot, vec_scaled_add};
use crate::shared::exception::LbfgsError;
use crate::shared::numeric::{FeatureType, ScalarType};
use crate::shared::parameter::{LineSearchCondition, LineSearchParam};

pub(crate) struct BacktrackingLineSearch {
    /// Dataset reference owned by linesearch
    pub dataset: Box<dyn Dataset>,

    /// Loss function
    pub loss_fn: Box<dyn LossFunc>,

    /// Linesearch hyperparameters
    pub linesearch_params: LineSearchParam,
}

impl BacktrackingLineSearch {
    pub fn new<DatasetType, LossFuncType>(
        dataset: DatasetType,
        loss_fn: LossFuncType,
        linesearch_params: LineSearchParam,
    ) -> Self
    where
        DatasetType: Dataset + 'static,
        LossFuncType: LossFunc + 'static,
    {
        Self {
            dataset: Box::new(dataset),
            loss_fn: Box::new(loss_fn),
            linesearch_params,
        }
    }
}

impl LineSearch for BacktrackingLineSearch {
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
        // step must be positive
        if *stepsize <= 0.0 {
            return Err(LbfgsError::InvalidParameters);
        }

        // fx_init = loss at xp;
        // dg_init = directional derivative along d at xp
        let fx_init = *fx;
        let dg_init = vec_dot(d, gp);

        // Search direction must decrease the objective
        if dg_init > 0.0 {
            return Err(LbfgsError::IncreaseGradient);
        }

        // Tolerance threshold: the minimum descent
        let dg_test = self.linesearch_params.ftol * dg_init;

        let mut count: usize = 0;
        loop {
            // x_{k+1} = x_k + stepsize * d_k
            vec_scaled_add(d, xp, *stepsize, x);

            // Compute loss and gradient at the new x.
            *fx = self.loss_fn.evaluate_with_gradient(&*self.dataset, x, g);

            // Increment iteration
            count += 1;

            // Armijo / (strong) Wolfe condition logic
            let width: ScalarType = if *fx > fx_init + *stepsize * dg_test {
                // If Armijo condition not satisfied
                self.linesearch_params.dec_factor
            } else {
                if self.linesearch_params.condition == LineSearchCondition::Armijo {
                    return Ok(count);
                }

                // Compute derivative at new x
                let dg: ScalarType = vec_dot(d, g);
                if dg < self.linesearch_params.wolfe * dg_init {
                    // Wolfe condition not satisfied - grow step
                    self.linesearch_params.inc_factor
                } else {
                    if self.linesearch_params.condition == LineSearchCondition::Wolfe {
                        return Ok(count);
                    }

                    if dg > -self.linesearch_params.wolfe * dg_init {
                        // Strong Wolfe not satisfied - shrink step
                        self.linesearch_params.dec_factor
                    } else {
                        // Both Armijo and (strong) Wolfe satisfied
                        return Ok(count);
                    }
                }
            };

            // Bounds checks
            if *stepsize < self.linesearch_params.min_stepsize {
                return Err(LbfgsError::MinimumStep);
            }
            if *stepsize > self.linesearch_params.max_stepsize {
                return Err(LbfgsError::MaximumStep);
            }
            if count >= self.linesearch_params.max_linesearch_iters {
                return Err(LbfgsError::MaximumLineSearchIteration);
            }

            // Update step size for next iteration
            *stepsize *= width;
        }
    }
}
