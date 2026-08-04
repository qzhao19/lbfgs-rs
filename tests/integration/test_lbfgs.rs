use std::cell::RefCell;
use std::rc::Rc;

use lbfgs_rs::OptimizeArgs;
use lbfgs_rs::LBFGS;
use lbfgs_rs::{FeatureType, LabelType, ScalarType};

/// Iris binary classification dataset
///
/// Returns `(x, y)` where `x` is a row-major flattened feature matrix
/// (`n_samples * n_features`) and `y` is the label vector (`n_samples`).
fn build_iris_xy() -> (Vec<FeatureType>, Vec<LabelType>) {
    let sepal_length: &[FeatureType] = &[
        5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9, 5.4, 4.8, 4.8, 4.3, 5.8, 5.7, 5.4, 5.1,
        5.7, 5.1, 5.4, 5.1, 4.6, 5.1, 4.8, 5.0, 5.0, 5.2, 5.2, 4.7, 4.8, 5.4, 5.2, 5.5, 4.9, 5.0,
        5.5, 4.9, 4.4, 5.1, 5.0, 4.5, 4.4, 5.0, 5.1, 4.8, 5.1, 4.6, 5.3, 5.0, 7.0, 6.4, 6.9, 5.5,
        6.5, 5.7, 6.3, 4.9, 6.6, 5.2, 5.0, 5.9, 6.0, 6.1, 5.6, 6.7, 5.6, 5.8, 6.2, 5.6, 5.9, 6.1,
        6.3, 6.1, 6.4, 6.6, 6.8, 6.7, 6.0, 5.7, 5.5, 5.5, 5.8, 6.0, 5.4, 6.0, 6.7, 6.3, 5.6, 5.5,
        5.5, 6.1, 5.8, 5.0, 5.6, 5.7, 5.7, 6.2, 5.1, 5.7, 6.3, 5.8, 7.1, 6.3, 6.5, 7.6, 4.9, 7.3,
        6.7, 7.2, 6.5, 6.4, 6.8, 5.7, 5.8, 6.4, 6.5, 7.7, 7.7, 6.0, 6.9, 5.6, 7.7, 6.3, 6.7, 7.2,
        6.2, 6.1, 6.4, 7.2, 7.4, 7.9, 6.4, 6.3, 6.1, 7.7, 6.3, 6.4, 6.0, 6.9, 6.7, 6.9, 5.8, 6.8,
        6.7, 6.7, 6.3, 6.5, 6.2, 5.9,
    ];
    let sepal_width: &[FeatureType] = &[
        3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1, 3.7, 3.4, 3.0, 3.0, 4.0, 4.4, 3.9, 3.5,
        3.8, 3.8, 3.4, 3.7, 3.6, 3.3, 3.4, 3.0, 3.4, 3.5, 3.4, 3.2, 3.1, 3.4, 4.1, 4.2, 3.1, 3.2,
        3.5, 3.6, 3.0, 3.4, 3.5, 2.3, 3.2, 3.5, 3.8, 3.0, 3.8, 3.2, 3.7, 3.3, 3.2, 3.2, 3.1, 2.3,
        2.8, 2.8, 3.3, 2.4, 2.9, 2.7, 2.0, 3.0, 2.2, 2.9, 2.9, 3.1, 3.0, 2.7, 2.2, 2.5, 3.2, 2.8,
        2.5, 2.8, 2.9, 3.0, 2.8, 3.0, 2.9, 2.6, 2.4, 2.4, 2.7, 2.7, 3.0, 3.4, 3.1, 2.3, 3.0, 2.5,
        2.6, 3.0, 2.6, 2.3, 2.7, 3.0, 2.9, 2.9, 2.5, 2.8, 3.3, 2.7, 3.0, 2.9, 3.0, 3.0, 2.5, 2.9,
        2.5, 3.6, 3.2, 2.7, 3.0, 2.5, 2.8, 3.2, 3.0, 3.8, 2.6, 2.2, 3.2, 2.8, 2.8, 2.7, 3.3, 3.2,
        2.8, 3.0, 2.8, 3.0, 2.8, 3.8, 2.8, 2.8, 2.6, 3.0, 3.4, 3.1, 3.0, 3.1, 3.1, 3.1, 2.7, 3.2,
        3.3, 3.0, 2.5, 3.0, 3.4, 3.,
    ];
    let petal_length: &[FeatureType] = &[
        1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5, 1.5, 1.6, 1.4, 1.1, 1.2, 1.5, 1.3, 1.4,
        1.7, 1.5, 1.7, 1.5, 1.0, 1.7, 1.9, 1.6, 1.6, 1.5, 1.4, 1.6, 1.6, 1.5, 1.5, 1.4, 1.5, 1.2,
        1.3, 1.4, 1.3, 1.5, 1.3, 1.3, 1.3, 1.6, 1.9, 1.4, 1.6, 1.4, 1.5, 1.4, 4.7, 4.5, 4.9, 4.0,
        4.6, 4.5, 4.7, 3.3, 4.6, 3.9, 3.5, 4.2, 4.0, 4.7, 3.6, 4.4, 4.5, 4.1, 4.5, 3.9, 4.8, 4.0,
        4.9, 4.7, 4.3, 4.4, 4.8, 5.0, 4.5, 3.5, 3.8, 3.7, 3.9, 5.1, 4.5, 4.5, 4.7, 4.4, 4.1, 4.0,
        4.4, 4.6, 4.0, 3.3, 4.2, 4.2, 4.2, 4.3, 3.0, 4.1, 6.0, 5.1, 5.9, 5.6, 5.8, 6.6, 4.5, 6.3,
        5.8, 6.1, 5.1, 5.3, 5.5, 5.0, 5.1, 5.3, 5.5, 6.7, 6.9, 5.0, 5.7, 4.9, 6.7, 4.9, 5.7, 6.0,
        4.8, 4.9, 5.6, 5.8, 6.1, 6.4, 5.6, 5.1, 5.6, 6.1, 5.6, 5.5, 4.8, 5.4, 5.6, 5.1, 5.1, 5.9,
        5.7, 5.2, 5.0, 5.2, 5.4, 5.1,
    ];
    let petal_width: &[FeatureType] = &[
        0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.4, 0.4, 0.3,
        0.3, 0.3, 0.2, 0.4, 0.2, 0.5, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.4, 0.1, 0.2, 0.2, 0.2,
        0.2, 0.1, 0.2, 0.2, 0.3, 0.3, 0.2, 0.6, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 1.4, 1.5, 1.5, 1.3,
        1.5, 1.3, 1.6, 1.0, 1.3, 1.4, 1.0, 1.5, 1.0, 1.4, 1.3, 1.4, 1.5, 1.0, 1.5, 1.1, 1.8, 1.3,
        1.5, 1.2, 1.3, 1.4, 1.4, 1.7, 1.5, 1.0, 1.1, 1.0, 1.2, 1.6, 1.5, 1.6, 1.5, 1.3, 1.3, 1.3,
        1.2, 1.4, 1.2, 1.0, 1.3, 1.2, 1.3, 1.3, 1.1, 1.3, 2.5, 1.9, 2.1, 1.8, 2.2, 2.1, 1.7, 1.8,
        1.8, 2.5, 2.0, 1.9, 2.1, 2.0, 2.4, 2.3, 1.8, 2.2, 2.3, 1.5, 2.3, 2.0, 2.0, 1.8, 2.1, 1.8,
        1.8, 1.8, 2.1, 1.6, 1.9, 2.0, 2.2, 1.5, 1.4, 2.3, 2.4, 1.8, 1.8, 2.1, 2.4, 2.3, 1.9, 2.3,
        2.5, 2.3, 1.9, 2.0, 2.3, 1.8,
    ];
    // Labels: 77 × -1, then 73 × +1 (binary Iris: 150 samples, same y as linesearch test).
    let y: Vec<LabelType> = (0..77)
        .map(|_| -1.0 as LabelType)
        .chain((0..73).map(|_| 1.0 as LabelType))
        .collect();

    let n = 150usize;
    let ncols = 4usize;
    let mut x_data = Vec::with_capacity(n * ncols);
    for i in 0..n {
        x_data.push(sepal_length[i]);
        x_data.push(sepal_width[i]);
        x_data.push(petal_length[i]);
        x_data.push(petal_width[i]);
    }
    return (x_data, y);
}

/// Build a public [`LBFGS`] driver with the same hyperparameters as the old
/// internal-API tests: `delta=1e-6`, `epsilon=1e-5`, `max_iters=0`, `mem_size=8`,
/// `past=3`, default line-search params.
///
/// - `search`: `"backtracking"` or `"bracketing"`.
/// - `callback`: optional loss-history callback (passed at construction).
fn make_optimizer(search: &str, callback: Option<Box<dyn Fn(&[ScalarType])>>) -> LBFGS {
    let x0: Vec<ScalarType> = vec![1.0, 1.0, 1.0, 1.0];
    let args = OptimizeArgs {
        delta: Some(1e-6),
        epsilon: Some(1e-5),
        max_iters: Some(0),
        mem_size: Some(8),
        past: Some(3),
        ..Default::default()
    };
    LBFGS::new(
        x0,
        args,
        Some("lbfgs".to_string()),
        Some(search.to_string()),
        Some("logloss".to_string()),
        callback,
        false,
    )
    .expect("LBFGS::new should succeed")
}

// ── Backtracking line search ──

mod backtracking {
    use super::*;

    #[test]
    fn runs() {
        let (x, y) = build_iris_xy();
        let mut optimizer = make_optimizer("backtracking", None);
        let _ = optimizer.optimize(x, y);
        let coef = optimizer.get_weight();
        eprint!("coefficients = ");
        for c in &coef {
            eprint!("{} ", c);
        }
        eprintln!();
    }

    #[test]
    fn converges() {
        let (x, y) = build_iris_xy();
        let mut optimizer = make_optimizer("backtracking", None);
        optimizer.optimize(x, y).expect("optimize should not throw");
        assert!(optimizer.get_weight().len() > 0);
    }

    #[test]
    fn convergence_speed() {
        let (x, y) = build_iris_xy();
        let all_losses: Rc<RefCell<Vec<ScalarType>>> =
            Rc::new(RefCell::new(Vec::with_capacity(100 * 150)));
        let losses_cb = all_losses.clone();
        let mut optimizer = make_optimizer(
            "backtracking",
            Some(Box::new(move |loss_history: &[ScalarType]| {
                losses_cb.borrow_mut().extend_from_slice(loss_history);
            })),
        );
        optimizer.optimize(x, y).expect("optimize should not throw");

        let losses = all_losses.borrow();
        assert!(!losses.is_empty(), "loss history should not be empty");
        let initial_loss = losses[0];
        let final_loss = *losses.last().unwrap();
        let improvement_ratio = (initial_loss - final_loss) / initial_loss;
        assert!(
            improvement_ratio > 0.3 as ScalarType,
            "insufficient convergence rate (improvement_ratio = {})",
            improvement_ratio
        );
    }
}

// ── Bracketing line search ──

mod bracketing {
    use super::*;

    #[test]
    fn runs() {
        let (x, y) = build_iris_xy();
        let mut optimizer = make_optimizer("bracketing", None);
        let _ = optimizer.optimize(x, y);
        let coef = optimizer.get_weight();
        print!("coefficients = ");
        for c in &coef {
            print!("{} ", c);
        }
        println!();
    }

    #[test]
    fn converges() {
        let (x, y) = build_iris_xy();
        let mut optimizer = make_optimizer("bracketing", None);
        optimizer.optimize(x, y).expect("optimize should not throw");
        assert!(optimizer.get_weight().len() > 0);
    }

    #[test]
    fn convergence_speed() {
        let (x, y) = build_iris_xy();
        let all_losses: Rc<RefCell<Vec<ScalarType>>> =
            Rc::new(RefCell::new(Vec::with_capacity(100 * 150)));
        let losses_cb = all_losses.clone();
        let mut optimizer = make_optimizer(
            "bracketing",
            Some(Box::new(move |loss_history: &[ScalarType]| {
                losses_cb.borrow_mut().extend_from_slice(loss_history);
            })),
        );
        optimizer.optimize(x, y).expect("optimize should not throw");

        let losses = all_losses.borrow();
        assert!(!losses.is_empty(), "loss history should not be empty");
        let initial_loss = losses[0];
        let final_loss = *losses.last().unwrap();
        let improvement_ratio = (initial_loss - final_loss) / initial_loss;
        assert!(
            improvement_ratio > 0.3 as ScalarType,
            "insufficient convergence rate (improvement_ratio = {})",
            improvement_ratio
        );
    }
}
