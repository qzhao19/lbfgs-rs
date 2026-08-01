use super::logistic::LogLoss;
use super::loss::LossFunc;
use crate::data::dense::DenseDataset;
use crate::shared::numeric::{FeatureType, LabelType, ScalarType};

// ── helpers ──

fn evaluate(y_pred: FeatureType, y_true: LabelType) -> ScalarType {
    let z = y_pred * y_true;
    if z > 18.0 {
        return (-z).exp();
    }
    if z < -18.0 {
        return -z;
    }
    return (-z).exp().ln_1p();
}

/// Compute gradient of loss w.r.t. prediction
fn derivate(y_pred: FeatureType, y_true: LabelType) -> ScalarType {
    let z = y_pred * y_true;
    if z > 18.0 {
        return (-z).exp() * (-y_true);
    }

    if z < -18.0 {
        return -y_true;
    }

    return -y_true / (z.exp() + 1.0);
}

/// Tolerance for floating-point comparisons.
fn epsilon() -> ScalarType {
    if cfg!(feature = "f32") {
        1e-5 as ScalarType
    } else {
        1e-10 as ScalarType
    }
}

/// Tolerance for the Iris reference test — matches the C++ reference's 1e-5
/// for f64; loosened for f32 due to accumulation over 150 samples.
fn iris_tolerance() -> ScalarType {
    if cfg!(feature = "f32") {
        1e-3 as ScalarType
    } else {
        1e-5 as ScalarType
    }
}

/// Build a dataset from explicit rows.
fn make_dataset(x_rows: &[&[FeatureType]], y: &[LabelType]) -> DenseDataset {
    let ncols = x_rows[0].len();
    let nrows = x_rows.len();
    let mut x_data = Vec::with_capacity(nrows * ncols);
    for row in x_rows {
        x_data.extend_from_slice(row);
    }
    DenseDataset::new(x_data, y.to_vec(), nrows, ncols, false).unwrap()
}

/// Build the Iris binary classification dataset.
fn build_iris_dataset() -> DenseDataset {
    // Feature 0: sepal_length (150 values, column-major in C++ source)
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
    // Feature 1: sepal_width
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
    // Feature 2: petal_length
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
    // Feature 3: petal_width
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
    // Labels: 75 × -1, then 75 × +1 (binary Iris: setosa+half-versicolor vs rest)
    let y: Vec<LabelType> = (0..77)
        .map(|_| -1.0 as LabelType)
        .chain((0..73).map(|_| 1.0 as LabelType))
        .collect();

    let n = 150usize;
    let ncols = 4usize;
    // Transpose column-major → row-major
    let mut x_data = Vec::with_capacity(n * ncols);
    for i in 0..n {
        x_data.push(sepal_length[i]);
        x_data.push(sepal_width[i]);
        x_data.push(petal_length[i]);
        x_data.push(petal_width[i]);
    }

    DenseDataset::new(x_data, y, n, ncols, false).unwrap()
}

// T4 — evaluate_with_gradient integration

mod with_gradient {
    use super::*;

    #[test]
    fn t4_1_single_sample_matches_manual() {
        // 1 sample, 2 features.0Compare aggregate loss & gradient to
        // manual evaluate + derivate + vecadd.
        let x: &[&[FeatureType]] = &[&[1.0, 2.0]];
        let y: &[LabelType] = &[1.0];
        let ds = make_dataset(x, y);
        let w = vec![0.5, -0.5];

        // Manual computation
        let mut loss = LogLoss::new();
        let y_hat = (1.0 * 0.5 + 2.0 * (-0.5)) as ScalarType; // = -0.5
        let l_manual = evaluate(y_hat, 1.0);
        let d = derivate(y_hat, 1.0);
        let g_manual = vec![d * 1.0, d * 2.0];

        // Via evaluate_with_gradient
        let mut grad = vec![0.0; 2];
        let l_api = loss.evaluate_with_gradient(&ds, &w, &mut grad);

        assert!(
            (l_api - l_manual).abs() < epsilon(),
            "loss: api {l_api} vs manual {l_manual}"
        );
        for j in 0..2 {
            assert!(
                (grad[j] - g_manual[j]).abs() < epsilon(),
                "grad[{j}]: api {} vs manual {}",
                grad[j],
                g_manual[j]
            );
        }
    }

    #[test]
    fn t4_2_multi_sample_accumulation() {
        // 3 samples; evaluate_with_gradient now returns the MEAN loss and
        // MEAN gradient (it applies inv_n_samples internally), so the manual
        // reference must be averaged the same way before comparing.
        let x: &[&[FeatureType]] = &[&[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0]];
        let y: &[LabelType] = &[1.0, 1.0, -1.0];
        let ds = make_dataset(x, y);
        let w = vec![0.1, -0.2];

        // Manual: accumulate per-sample contributions, then divide by n to
        // match the function's averaged output.
        let mut loss = LogLoss::new();
        let inv_n: ScalarType = 1.0 / 3.0;
        let mut l_manual = 0.0;
        let mut g_manual = vec![0.0; 2];
        for i in 0..3 {
            let y_hat = x[i][0] * w[0] + x[i][1] * w[1];
            l_manual += evaluate(y_hat, y[i]);
            let d = derivate(y_hat, y[i]);
            g_manual[0] += d * x[i][0];
            g_manual[1] += d * x[i][1];
        }
        l_manual *= inv_n;
        for j in 0..2 {
            g_manual[j] *= inv_n;
        }

        let mut grad = vec![0.0; 2];
        let l_api = loss.evaluate_with_gradient(&ds, &w, &mut grad);

        assert!(
            (l_api - l_manual).abs() < epsilon(),
            "loss: api {l_api} vs manual {l_manual}"
        );
        for j in 0..2 {
            assert!(
                (grad[j] - g_manual[j]).abs() < epsilon(),
                "grad[{j}]: api {} vs manual {}",
                grad[j],
                g_manual[j]
            );
        }
    }

    #[test]
    fn t4_3_zero_weights_yield_zero_gradient_with_unit_features() {
        // w = 0 → y_hat = 0 → z = 0 → d = -y * 0.5
        // grad = sum_i (-0.5 * y_i) * x_i
        let x: &[&[FeatureType]] = &[&[1.0, 1.0], &[1.0, 1.0]];
        let y: &[LabelType] = &[1.0, -1.0];
        let ds = make_dataset(x, y);
        let w = vec![0.0, 0.0];

        let mut grad = vec![0.0; 2];
        let mut loss = LogLoss::new();
        let _l = loss.evaluate_with_gradient(&ds, &w, &mut grad);

        // d_0 = -1*0.5 = -0.5; d_1 = -(-1)*0.5 = 0.5
        // grad = (-0.5 + 0.5) * [1,1] = [0, 0]
        assert!(
            grad[0].abs() < epsilon(),
            "grad[0] = {} expected 0",
            grad[0]
        );
        assert!(
            grad[1].abs() < epsilon(),
            "grad[1] = {} expected 0",
            grad[1]
        );
    }

    /// Iris dataset reference test.
    ///
    /// Uses the same 150-sample Iris dataset, same w = [1,1,1,1], and
    /// compares the MEAN loss/gradient (averaged inside
    /// `evaluate_with_gradient`) against the reference values, which
    /// are also already averaged over n_samples (=150).
    #[test]
    fn t4_4_iris_matches_cpp_reference() {
        let ds = build_iris_dataset();
        let n_samples = ds.nrows();
        assert_eq!(n_samples, 150);
        assert_eq!(ds.ncols(), 4);

        let w: Vec<FeatureType> = vec![1.0, 1.0, 1.0, 1.0];

        let mut loss = LogLoss::new();
        let mut grad = vec![0.0 as FeatureType; 4];
        // evaluate_with_gradient now applies inv_n_samples internally —
        // its return value and grad are already averaged over n_samples.
        let mean_loss = loss.evaluate_with_gradient(&ds, &w, &mut grad);

        println!("iris t4_5 — mean_loss = {mean_loss:.6}  (expected 5.99602)");
        println!(
            "iris t4_5 — mean_grad = [{:.6}, {:.6}, {:.6}, {:.6}]  (expected [2.75991, 1.64394, 1.26731, 0.324662])",
            grad[0], grad[1], grad[2], grad[3]
        );

        // reference expected values (tolerance 1e-5 for f64).
        let expected_loss = 5.99602 as ScalarType;
        let expected_grad = [
            2.75991 as ScalarType,
            1.64394 as ScalarType,
            1.26731 as ScalarType,
            0.324662 as ScalarType,
        ];
        let tol = iris_tolerance();

        assert!(
            (mean_loss - expected_loss).abs() < tol,
            "mean_loss: got {mean_loss}, expected {expected_loss} (tol {tol})"
        );
        for j in 0..4 {
            assert!(
                (grad[j] - expected_grad[j]).abs() < tol,
                "mean_grad[{j}]: got {}, expected {} (tol {})",
                grad[j],
                expected_grad[j],
                tol
            );
        }
    }
}

// T5 — LBFGS robustness scenarios

mod lbfgs_robustness {
    use super::*;

    #[test]
    fn t5_1_linearly_separable_decreases_loss() {
        // A single gradient step should reduce total loss on a
        // linearly separable dataset.0This is the minimal LBFGS sanity:
        // the gradient direction must be a descent direction.
        let x: &[&[FeatureType]] = &[&[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0], &[2.0, 1.0]];
        let y: &[LabelType] = &[1.0, 1.0, -1.0, -1.0];
        let ds = make_dataset(x, y);

        let mut loss = LogLoss::new();
        let mut w = vec![0.0, 0.0];

        let l0 = {
            let mut g = vec![0.0; 2];
            loss.evaluate_with_gradient(&ds, &w, &mut g)
        };

        // Compute gradient at w=0
        let mut g = vec![0.0; 2];
        loss.evaluate_with_gradient(&ds, &w, &mut g);

        // Take a small step in the negative gradient direction
        let lr = 0.1 as ScalarType;
        for j in 0..2 {
            w[j] -= lr * g[j];
        }

        let l1 = {
            let mut g_tmp = vec![0.0; 2];
            loss.evaluate_with_gradient(&ds, &w, &mut g_tmp)
        };

        assert!(
            l1 < l0,
            "loss did not decrease: l0={l0}, l1={l1} — gradient is not a descent direction"
        );
    }

    #[test]
    fn t5_2_gradient_points_toward_correct_classification() {
        // For a misclassified sample (y_true=1, y_pred<0), the gradient
        // w.r.t.0w should push w toward making y_pred more positive.
        // Concretely: x=[1,0], y=1, w=[-1, 0] → y_hat=-1, misclassified.
        // grad = d * x where d = -y / (1 + exp(z)), z = -1.
        // d = -1 / (1 + e^{-1} * ...0) = -1 / (1 + exp(-1)) → negative.
        // grad[0] = d * 1 < 0 → negative gradient → step -lr*grad > 0 → w increases.
        let x: &[&[FeatureType]] = &[&[1.0, 0.0]];
        let y: &[LabelType] = &[1.0];
        let ds = make_dataset(x, y);
        let w = vec![-1.0, 0.0];

        let mut loss = LogLoss::new();
        let mut grad = vec![0.0; 2];
        loss.evaluate_with_gradient(&ds, &w, &mut grad);

        // grad[0] should be negative (so -lr*grad[0] > 0, increasing w[0])
        assert!(
            grad[0] < 0.0,
            "grad[0] = {} should be < 0 to push w[0] upward",
            grad[0]
        );
    }

    #[test]
    fn t5_3_large_weights_stay_finite() {
        // Simulate a late-stage LBFGS iterate with large weights.
        // The loss/gradient must remain finite (no NaN/Inf) for the
        // optimizer to continue.
        let x: &[&[FeatureType]] = &[&[1.0, 1.0], &[1.0, -1.0]];
        let y: &[LabelType] = &[1.0, -1.0];
        let ds = make_dataset(x, y);
        let w = vec![1e3, -1e3];

        let mut loss = LogLoss::new();
        let mut grad = vec![0.0; 2];
        let l = loss.evaluate_with_gradient(&ds, &w, &mut grad);

        assert!(l.is_finite(), "loss not finite at large weights: {l}");
        for j in 0..2 {
            assert!(grad[j].is_finite(), "grad[{j}] not finite: {}", grad[j]);
        }
    }

    #[test]
    fn t5_4_loss_is_convex_in_w() {
        // Logistic loss is convex.0Pick 3 weight vectors along a line
        // and verify the middle one has loss ≤ max(endpoints).
        let x: &[&[FeatureType]] = &[&[1.0, 2.0], &[2.0, 1.0], &[-1.0, -1.0]];
        let y: &[LabelType] = &[1.0, 1.0, -1.0];
        let ds = make_dataset(x, y);

        let mut loss = LogLoss::new();
        let w_a = vec![-1.0, 0.5];
        let w_b = vec![0.0, 0.0]; // midpoint
        let w_c = vec![1.0, -0.5];

        let l_a = {
            let mut g = vec![0.0; 2];
            loss.evaluate_with_gradient(&ds, &w_a, &mut g)
        };
        let l_b = {
            let mut g = vec![0.0; 2];
            loss.evaluate_with_gradient(&ds, &w_b, &mut g)
        };
        let l_c = {
            let mut g = vec![0.0; 2];
            loss.evaluate_with_gradient(&ds, &w_c, &mut g)
        };

        assert!(
            l_b <= l_a.max(l_c) + epsilon(),
            "convexity violated: l_a={l_a}, l_b={l_b}, l_c={l_c}"
        );
    }
}
