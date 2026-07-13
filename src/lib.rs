//! # ternary-knn
//!
//! K-nearest neighbors **classification** for ternary vectors
//! (elements in {-1, 0, +1}), using a ternary-specific distance metric and
//! brute-force neighbor search.
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
    ///
    /// Returns the majority label among the `k` nearest neighbors (by trit
    /// distance). On a **voting tie** (two labels with equal vote counts), the
    /// **smallest label wins**, making the result deterministic regardless of
    /// map iteration order. Distance ties are broken by dataset order (stable
    /// sort). `k` larger than the dataset is clamped to the dataset size.
    pub fn predict(&self, point: &[Trit]) -> Result<i32, String> {
        let data = self.data.as_ref().ok_or("KNN not fitted yet")?;
        validate_ternary(point)?;

        // Validate the query dimension up front so a bad-length query returns a
        // clean error instead of panicking inside `trit_distance(...).unwrap()`.
        if point.len() != data.dim {
            return Err(format!(
                "Dimension mismatch: query has {} dims but dataset has {}",
                point.len(),
                data.dim
            ));
        }

        // Dimensions are now guaranteed equal, but propagate any error rather
        // than `.unwrap()` so the code is robust to metric changes.
        let mut distances: Vec<(f64, i32)> = data
            .points
            .iter()
            .map(|p| Ok((trit_distance(&p.features, point)?, p.label)))
            .collect::<Result<Vec<_>, String>>()?;

        // Use `total_cmp` (not `partial_cmp().unwrap()`) so the sort can never
        // panic on a NaN distance, even if a future metric returns one.
        distances.sort_by(|a, b| a.0.total_cmp(&b.0));

        let k_nearest = &distances[..self.k.min(distances.len())];
        let mut votes: HashMap<i32, usize> = HashMap::new();
        for &(_, label) in k_nearest {
            *votes.entry(label).or_insert(0) += 1;
        }

        // Majority vote. Pick the highest count, then break ties by choosing
        // the smallest label so the result is deterministic regardless of
        // HashMap iteration order.
        let best_count = votes.values().copied().max().unwrap_or(0);
        let winner = votes
            .into_iter()
            .filter(|&(_, count)| count == best_count)
            .map(|(label, _)| label)
            .min()
            .ok_or_else(|| "No predictions available".to_string())?;

        Ok(winner)
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

    #[test]
    fn test_k_larger_than_dataset() {
        // k=10 but only 3 points; k must clamp to dataset size.
        // query [1,1]: dist to [1,1]=0, [1,0]=1, [-1,-1]=4.
        // k clamped to 3 -> votes label5=2, label9=1 -> predict 5.
        let points = vec![
            DataPoint::new(vec![1, 1], 5),
            DataPoint::new(vec![1, 0], 5),
            DataPoint::new(vec![-1, -1], 9),
        ];
        let dataset = TernaryDataset::new(points).unwrap();
        let mut knn = KNNClassifier::new(10);
        knn.fit(dataset);
        assert_eq!(knn.predict(&[1, 1]).unwrap(), 5);
    }

    #[test]
    fn test_predict_dimension_mismatch_returns_error() {
        // A query whose length differs from the dataset must return a clean
        // error rather than panicking inside trit_distance(...).unwrap().
        let dataset = TernaryDataset::new(vec![
            DataPoint::new(vec![1, 1], 0),
            DataPoint::new(vec![-1, -1], 1),
        ])
        .unwrap();
        let mut knn = KNNClassifier::new(1);
        knn.fit(dataset);
        assert!(knn.predict(&[1, 1, 1]).is_err());
    }

    #[test]
    fn test_predict_not_fitted() {
        let knn = KNNClassifier::new(1);
        assert!(knn.predict(&[1, 0]).is_err());
    }

    #[test]
    fn test_voting_tie_smallest_label_wins() {
        // Two points equidistant from the query with distinct labels 0 and 2.
        // With k=2 the vote is tied (1 each); the smallest label must win.
        let points = vec![
            DataPoint::new(vec![1, 0], 0),
            DataPoint::new(vec![0, 1], 2),
        ];
        let dataset = TernaryDataset::new(points).unwrap();
        let mut knn = KNNClassifier::new(2);
        knn.fit(dataset);
        assert_eq!(knn.predict(&[0, 0]).unwrap(), 0);
    }

    #[test]
    fn test_all_identical_points() {
        // Degenerate case: every point has identical features and label.
        let points = vec![
            DataPoint::new(vec![0, 0, 0], 7),
            DataPoint::new(vec![0, 0, 0], 7),
            DataPoint::new(vec![0, 0, 0], 7),
        ];
        let dataset = TernaryDataset::new(points).unwrap();
        let mut knn = KNNClassifier::new(3);
        knn.fit(dataset);
        assert_eq!(knn.predict(&[0, 0, 0]).unwrap(), 7);
    }

    #[test]
    fn test_normalized_distance_nonzero() {
        // [1,0,-1] vs [-1,0,1]: raw = 2 + 0 + 2 = 4; normalized = 4/(2*3).
        let a = vec![1, 0, -1];
        let b = vec![-1, 0, 1];
        assert_eq!(normalized_trit_distance(&a, &b).unwrap(), 4.0 / 6.0);
    }

    #[test]
    fn test_trit_distance_dimension_mismatch() {
        assert!(trit_distance(&[1, 0], &[1, 0, -1]).is_err());
    }

    #[test]
    fn test_validate_ternary_rejects_invalid() {
        assert!(validate_ternary(&[1, 0, 2, -1]).is_err());
        assert!(validate_ternary(&[1, 0, -1]).is_ok());
    }
}
