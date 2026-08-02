use crate::data::dataset::Dataset;
use crate::shared::exception::{LbfgsError, LbfgsStatus};
use crate::shared::numeric::ScalarType;

pub(crate) trait Optimizer {
    fn optimize(&mut self, dataset: Box<dyn Dataset>) -> Result<LbfgsStatus, LbfgsError>;

    fn get_weight(&self) -> Vec<ScalarType>;

    fn set_callback(&mut self, callback: Box<dyn Fn(&[ScalarType])>);
}

/// One slot of the limited-memory correction history
pub(crate) struct LimitedMemoryBuf {
    pub mem_ys: ScalarType,
    pub mem_alpha: ScalarType,
    pub mem_y: Vec<ScalarType>,
    pub mem_s: Vec<ScalarType>,
}

impl LimitedMemoryBuf {
    pub fn initialize(size: usize) -> Self {
        Self {
            mem_ys: 0.0,
            mem_alpha: 0.0,
            mem_y: vec![0.0; size],
            mem_s: vec![0.0; size],
        }
    }
}
