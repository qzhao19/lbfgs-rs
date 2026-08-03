use crate::data::dataset::Dataset;
use crate::shared::exception::{LbfgsError, LbfgsStatus};
use crate::shared::numeric::ScalarType;

pub(crate) trait Optimizer {
    fn optimize(&mut self, dataset: Box<dyn Dataset>) -> Result<LbfgsStatus, LbfgsError>;

    fn get_weight(&self) -> Vec<ScalarType>;

    fn set_callback(&mut self, callback: Box<dyn Fn(&[ScalarType])>);
}
