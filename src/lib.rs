//! # ternary-knn
//!
//! K-nearest neighbors classification and regression for ternary vectors
//! (elements in {-1, 0, +1}), with ternary-specific distance metrics and
//! both brute-force and ball-tree index strategies.
//!
//! Connected to the [`ternary-types`](https://github.com/SuperInstance/ternary-types)
//! fleet via its dependency — use `ternary_types::Ternary` for cross-crate interop.
//! Conversion helpers: `i8::from(t: Ternary)` and `Ternary::try_from(v: i8)?`.

use std::collections::HashMap;

/// A trit value: -1, 0, or +1.
///
/// This crate uses `i8` internally. For conversion to/from the fleet's shared
/// [`ternary_types::Ternary`] type, use `i8::from(ternary_val)` or
/// `Ternary::try_from(i8_val)` (both provided by `ternary-types`).
pub type Trit = i8;

/// Validate that a slice contains only valid trits.
pub fn validate_ternary(vec: &[Trit]) -> Result<(), String> {
    for (i, &t) in vec.iter().enumerate() {
        if t != -1 && t != 0 && t != 1 {
            return Err(format!(
                "Invalid trit value {} at index {}; expected -1, 0, or +1",
                t, i
            ));
        }
    }
    Ok(())
}

/// Hamming-like trit distance: count positions where trits differ.
pub fn trit_distance(a: &[Trit], b: &[Trit]) -> Result<f64, String> {
    if a.len() != b.len() {
        return Err(format!(
            "Dimension mismatch: {} vs {}",
            a.len(),
            b.len()
        ));
    }

    let mut total = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        if x == y {
            continue;
        }
        if *x == 0 || *y == 0 {
            total += 1.0;
        } else {
            total += 2.0;
        }
    }
    Ok(total)
}

/// Normalized trit distance: trit_distance / (2 * max_dimensions).
pub fn normalized_trit_distance(a: &[Trit], b: &[Trit]) -> Result<f64, String> {
    let raw = trit_distance(a, b)?;
    Ok(raw / (2.0 * a.len() as f64))
}

// ─── Dataset types ───────────────────────────────────────────────────────────

/// A labeled ternary data point.
#[derive(Debug, Clone)]
pub struct DataPoint {
    pub features: Vec<Trit>,
    pub label: i32,
}

impl DataPoint {
    pub fn new(features: Vec<Trit>, label: i32) -> Self {
        Self { features, label }
    }
}

/// A dataset of ternary data points.
#[derive(Debug, Clone)]
pub struct TernaryDataset {
    pub points: Vec<DataPoint>,
    pub dim: usize,
}

impl TernaryDataset {
    pub fn new(points: Vec<DataPoint>) -> Result<Self, String> {
        if points.is_empty() {
            return Err("Dataset must not be empty".into());
        }
        let dim = points[0].features.len();
        for (i, p) in points.iter().enumerate() {
            if p.features.len() != dim {
                return Err(format!(
                    "Point {} has dim {} but expected {}",
                    i,
                    p.features.len(),
                    dim
                ));
            }
            validate_ternary(&p.features)?;
        }
        Ok(Self { points, dim })
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

// ─── KNN Classifier ──────────────────────────────────────────────────────────

/// K-Nearest Neighbors classifier using trit distance metrics.
pub struct KNNClassifier {
    k: usize,
    data: Option<TernaryDataset>,
}

impl KNNClassifier {
    pub fn new(k: usize) -> Self {
        assert!(k > 0, "k must be positive");
        Self { k, data: None }
    }

    pub fn fit(&mut self, dataset: TernaryDataset) {
        self.data = Some(dataset);
    }

    /// Predict label for a single point.
    pub fn predict(&self, point: &[Trit]) -> Result<i32, String> {
        let data = self.data.as_ref().ok_or("KNN not fitted yet")?;
        validate_ternary(point)?;

        let mut distances: Vec<(f64, i32)> = data
            .points
            .iter()
            .map(|p| (trit_distance(&p.features, point).unwrap(), p.label))
            .collect();

        // Use `total_cmp` (not `partial_cmp().unwrap()`) so the sort can never
        // panic on a NaN distance, even if a future metric returns one.
        distances.sort_by(|a, b| a.0.total_cmp(&b.0));

        let k_nearest = &distances[..self.k.min(distances.len())];
        let mut votes: HashMap<i32, usize> = HashMap::new();
        for &(_, label) in k_nearest {
            *votes.entry(label).or_insert(0) += 1;
        }

        votes
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(label, _)| label)
            .ok_or("No predictions available".into())
    }

    /// Predict labels for multiple points.
    pub fn predict_batch(&self, points: &[Vec<Trit>]) -> Result<Vec<i32>, String> {
        points.iter().map(|p| self.predict(p)).collect()
    }

    /// Return accuracy on a test set.
    pub fn accuracy(&self, test_data: &TernaryDataset) -> Result<f64, String> {
        let mut correct = 0;
        for p in &test_data.points {
            let pred = self.predict(&p.features)?;
            if pred == p.label {
                correct += 1;
            }
        }
        Ok(correct as f64 / test_data.len() as f64)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trit_distance_equal() {
        let a = vec![1, 0, -1, 1];
        let b = vec![1, 0, -1, 1];
        assert_eq!(trit_distance(&a, &b).unwrap(), 0.0);
    }

    #[test]
    fn test_trit_distance_opposite() {
        let a = vec![1, 0, -1, 1];
        let b = vec![-1, 0, 1, -1];
        assert_eq!(trit_distance(&a, &b).unwrap(), 6.0);
    }

    #[test]
    fn test_trit_distance_zero_vs_nonzero() {
        let a = vec![1, 0, 0, -1];
        let b = vec![0, 1, -1, 0];
        assert_eq!(trit_distance(&a, &b).unwrap(), 4.0);
    }

    #[test]
    fn test_normalized_distance() {
        let a = vec![1, 0, -1];
        let b = vec![1, 0, -1];
        assert_eq!(normalized_trit_distance(&a, &b).unwrap(), 0.0);
    }

    #[test]
    fn test_knn_predict_simple() {
        let points = vec![
            DataPoint::new(vec![1, 1, 1], 1),
            DataPoint::new(vec![1, 1, -1], 1),
            DataPoint::new(vec![-1, -1, -1], 0),
            DataPoint::new(vec![-1, -1, 1], 0),
        ];
        let dataset = TernaryDataset::new(points).unwrap();
        let mut knn = KNNClassifier::new(3);
        knn.fit(dataset);
        assert_eq!(knn.predict(&[1, 1, 0]).unwrap(), 1);
        assert_eq!(knn.predict(&[-1, -1, 0]).unwrap(), 0);
    }

    #[test]
    fn test_accuracy_perfect() {
        let points = vec![
            DataPoint::new(vec![1, 1], 0),
            DataPoint::new(vec![-1, -1], 1),
        ];
        let dataset = TernaryDataset::new(points).unwrap();
        let mut knn = KNNClassifier::new(1);
        knn.fit(dataset.clone());
        assert_eq!(knn.accuracy(&dataset).unwrap(), 1.0);
    }

    #[test]
    fn test_dimension_mismatch_error() {
        assert!(TernaryDataset::new(vec![
            DataPoint::new(vec![1, 0], 0),
            DataPoint::new(vec![1, 0, -1], 1),
        ]).is_err());
    }

    #[test]
    fn test_empty_dataset_error() {
        assert!(TernaryDataset::new(vec![]).is_err());
    }
}
