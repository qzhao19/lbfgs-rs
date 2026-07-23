use super::super::loss::LossFunc;
use crate::data::dataset::Dataset;
use crate::infra::math::ops_neon::{vec_dot, vec_scaled_add_inplace};
use crate::shared::types::{FeatureType, LabelType, ScalarType};

pub struct LogLoss;

impl LogLoss {
    pub fn new() -> Self {
        Self {}
    }

    /// Compute loss for a single sample
    #[inline]
    fn evaluate(&self, y_pred: FeatureType, y_true: LabelType) -> ScalarType {
        let z = y_pred * y_true;
        if z > 18.0 {
            return (-z).exp();
        }
        if z < -18.0 {
            return -z;
        }
        return (-z).exp().ln_1p();
    }

    /// Compute gradient of loss
    #[inline]
    fn derivate(&self, y_pred: FeatureType, y_true: LabelType) -> ScalarType {
        let z = y_pred * y_true;
        if z > 18.0 {
            return (-z).exp() * (-y_true);
        }

        if z < -18.0 {
            return -y_true;
        }

        return -y_true / (z.exp() + 1.0);
    }
}

impl LossFunc for LogLoss {
    fn evaluate_with_gradient(
        &mut self,
        dataset: &dyn Dataset,
        w: &[FeatureType],
        grad: &mut [FeatureType],
    ) -> ScalarType {
        let n_samples: usize = dataset.nrows();
        let n_features: usize = dataset.ncols();

        let mut loss: ScalarType = 0.0;
        let mut buf: Vec<FeatureType> = vec![0.0; n_features];

        for i in 0..n_samples {
            // Get i-th row sample
            dataset.fill_x_row(i, &mut buf);

            // Compute h_hat = dot(x_i, w)
            let y_hat: FeatureType = vec_dot(&buf, w);

            // Get relative label
            let y_true = dataset.y_row(i);

            // acc_loss += loss
            loss += self.evaluate(y_hat, y_true);
            let dloss: ScalarType = self.derivate(y_hat, y_true);

            // grad += dloss * x_i
            vec_scaled_add_inplace(&buf, dloss, grad);
        }

        return loss;
    }
}
