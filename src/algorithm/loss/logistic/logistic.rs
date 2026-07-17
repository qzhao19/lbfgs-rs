use super::super::loss::LossFunc;
use crate::data::dataset::Dataset;
use crate::infra::math::kernel::ops_neon::vecadd_neon::vecadd;
use crate::infra::math::kernel::ops_neon::vecdot_neon::vecdot;
use crate::shared::types::primitives::{FeatureType, LabelType, ScalarType};

pub struct LogLoss;

impl LogLoss {
    pub fn new() -> Self {
        Self {}
    }
}

impl LossFunc for LogLoss {
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
            let y_hat: FeatureType = vecdot(&buf, w);

            // Get relative label
            let y_true = dataset.y_row(i);
            loss += self.evaluate(y_hat, y_true);
            let dloss = self.derivate(y_hat, y_true);

            // grad += dloss * x_i
            vecadd(&buf, dloss, grad);
        }

        return loss;
    }
}
