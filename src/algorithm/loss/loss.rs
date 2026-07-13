use crate::shared::types::ScalarType;

pub trait LossFunc {
    // Loss value of a single sample 
    fn evaluate(&self, y_pred: ScalarType, y_true: ScalarType) -> ScalarType;

    // Derivate of a single sample
    fn derivate(&self, y_pred: ScalarType, y_true: ScalarType) -> ScalarType;

    // Calculate the total loss and accumulate the gradients over the entire dataset.
    fn evaluate_with_gradient(
        &mut self, 
        x: &[Vec<Scalar>],
        y: &[Scalar],
        w: &[Scalar],
        grad: &mut [Scalar],
    ) -> ScalarType;

    // Setup hyper-parameters
    fn set_params(&mut self, name: &str, value: &ScalarType);

    // Setup callback function
    fn set_callback(&mut self, callback: Box<dyn Fn(&[ScalarType])>);
}
