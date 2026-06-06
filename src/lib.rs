//! # ternary-knn
//!
//! K-nearest neighbors classification and regression for ternary vectors
//! (elements in {-1, 0, +1}), with ternary-specific distance metrics and
//! both brute-force and ball-tree index strategies.

use std::collections::HashMap;

/// A ternary value: Negative (-1), Zero (0), or Positive (+1).
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
///
/// Two trits are "same" if equal, "different" if not.
/// For two trits a, b: distance is 0 if a==b, else 1.
/// Special case: 1 and -1 are considered "opposite" with distance 2.
///
/// So the per-position cost is:
/// - 0 if a == b
/// - 1 if one is 0 and the other is ±1
/// - 2 if a == -b and a != 0 (i.e., -1 vs +1)
pub fn trit_distance(a: &[Trit], b: &[Trit]) -> Result<f64, String> {
    if a.len() != b.len() {
        return Err(format!(
            "Dimension mismatch: {} vs {}",
            a.len(),
            b.len()
        ));
    }
    let mut dist = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        if x == y {
            // same
        } else if (*x == -1 && *y == 1) || (*x == 1 && *y == -1) {
            dist += 2.0;
        } else {
            // one is 0, the other is ±1
            dist += 1.0;
        }
    }
    Ok(dist)
}

/// Weighted trit distance with per-position weights.
///
/// Same per-position cost as `trit_distance`, multiplied by the weight at that position.
pub fn weighted_trit_distance(a: &[Trit], b: &[Trit], weights: &[f64]) -> Result<f64, String> {
    if a.len() != b.len() {
        return Err(format!(
            "Dimension mismatch: {} vs {}",
            a.len(),
            b.len()
        ));
    }
    if a.len() != weights.len() {
        return Err(format!(
            "Weight dimension mismatch: vector len {} vs weights len {}",
            a.len(),
            weights.len()
        ));
    }
    let mut dist = 0.0;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let base = if x == y {
            0.0
        } else if (*x == -1 && *y == 1) || (*x == 1 && *y == -1) {
            2.0
        } else {
            1.0
        };
        dist += base * weights[i];
    }
    Ok(dist)
}

/// A labeled ternary data point for classification or regression.
#[derive(Debug, Clone)]
pub struct LabeledPoint {
    pub features: Vec<Trit>,
    pub label: f64,
}

/// KNN Classifier: predicts a discrete class label via majority vote.
#[derive(Debug, Clone)]
pub struct KnnClassifier {
    /// Training data.
    pub data: Vec<LabeledPoint>,
    /// Number of neighbors.
    pub k: usize,
    /// Whether to use distance-weighted voting.
    pub weighted: bool,
}

impl KnnClassifier {
    /// Create a new classifier with k neighbors.
    pub fn new(k: usize) -> Self {
        Self {
            data: Vec::new(),
            k,
            weighted: false,
        }
    }

    /// Enable or disable distance-weighted voting.
    pub fn with_weighted(mut self, weighted: bool) -> Self {
        self.weighted = weighted;
        self
    }

    /// Add a training point.
    pub fn add(&mut self, features: Vec<Trit>, label: f64) -> Result<(), String> {
        validate_ternary(&features)?;
        self.data.push(LabeledPoint { features, label });
        Ok(())
    }

    /// Fit the classifier from a slice of (features, label) pairs.
    pub fn fit(&mut self, samples: &[(Vec<Trit>, f64)]) -> Result<(), String> {
        for (features, label) in samples {
            self.add(features.clone(), *label)?;
        }
        Ok(())
    }

    /// Predict the label for a query point using brute-force search.
    pub fn predict(&self, query: &[Trit]) -> Result<f64, String> {
        validate_ternary(query)?;
        if self.data.is_empty() {
            return Err("No training data".into());
        }
        let k = self.k.min(self.data.len());
        let mut dists: Vec<(f64, f64)> = self
            .data
            .iter()
            .map(|p| (trit_distance(query, &p.features).unwrap(), p.label))
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let neighbors = &dists[..k];

        if self.weighted {
            let mut votes: HashMap<i64, f64> = HashMap::new();
            for &(d, label) in neighbors {
                let w = if d == 0.0 { 1e10 } else { 1.0 / d };
                let rounded = label.round() as i64;
                *votes.entry(rounded).or_insert(0.0) += w;
            }
            let best = votes
                .into_iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(label, _)| label as f64);
            Ok(best.unwrap_or(neighbors[0].1))
        } else {
            let mut votes: HashMap<i64, usize> = HashMap::new();
            for &(_, label) in neighbors {
                *votes.entry(label.round() as i64).or_insert(0) += 1;
            }
            let max_count = *votes.values().max().unwrap();
            // Tie-breaking: smallest label value (deterministic)
            let best = votes
                .iter()
                .filter(|&(_, &c)| c == max_count)
                .map(|(&l, _)| l)
                .min()
                .unwrap();
            Ok(best as f64)
        }
    }
}

/// KNN Regressor: predicts a continuous value via weighted average of neighbor labels.
#[derive(Debug, Clone)]
pub struct KnnRegressor {
    pub data: Vec<LabeledPoint>,
    pub k: usize,
}

impl KnnRegressor {
    pub fn new(k: usize) -> Self {
        Self {
            data: Vec::new(),
            k,
        }
    }

    /// Add a training point.
    pub fn add(&mut self, features: Vec<Trit>, label: f64) -> Result<(), String> {
        validate_ternary(&features)?;
        self.data.push(LabeledPoint { features, label });
        Ok(())
    }

    /// Predict using inverse-distance weighting of the k nearest neighbors.
    pub fn predict(&self, query: &[Trit]) -> Result<f64, String> {
        validate_ternary(query)?;
        if self.data.is_empty() {
            return Err("No training data".into());
        }
        let k = self.k.min(self.data.len());
        let mut dists: Vec<(f64, f64)> = self
            .data
            .iter()
            .map(|p| (trit_distance(query, &p.features).unwrap(), p.label))
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let neighbors = &dists[..k];

        let mut total_weight = 0.0;
        let mut weighted_sum = 0.0;
        for &(d, label) in neighbors {
            let w = if d == 0.0 { 1e10 } else { 1.0 / d };
            total_weight += w;
            weighted_sum += w * label;
        }
        Ok(weighted_sum / total_weight)
    }
}

// ─── Ball Tree Index ─────────────────────────────────────────────────────────

/// A node in the ball tree.
#[derive(Debug)]
struct BallTreeNode {
    /// Center of the ball (ternary centroid, rounded to nearest trit).
    center: Vec<Trit>,
    /// Radius of the ball (max distance from center to any child point).
    radius: f64,
    /// Leaf data: indices into the original dataset.
    indices: Vec<usize>,
    /// Children (empty for leaves).
    children: Vec<BallTreeNode>,
}

const BALL_TREE_LEAF_SIZE: usize = 8;

/// Ball-tree spatial index for fast KNN queries on ternary vectors.
#[derive(Debug)]
pub struct BallTree {
    root: BallTreeNode,
    data: Vec<Vec<Trit>>,
}

impl BallTree {
    /// Build a ball tree from a set of ternary vectors.
    pub fn build(data: Vec<Vec<Trit>>) -> Result<Self, String> {
        for (i, v) in data.iter().enumerate() {
            validate_ternary(v).map_err(|e| format!("Point {}: {}", i, e))?;
        }
        if data.is_empty() {
            return Err("Cannot build ball tree from empty data".into());
        }
        let dim = data[0].len();
        let indices: Vec<usize> = (0..data.len()).collect();
        let root = Self::build_recursive(&data, &indices, dim);
        Ok(Self { root, data })
    }

    fn trit_centroid(points: &[Vec<Trit>], dim: usize) -> Vec<Trit> {
        let mut center = vec![0.0f64; dim];
        for p in points {
            for (i, &v) in p.iter().enumerate() {
                center[i] += v as f64;
            }
        }
        let n = points.len() as f64;
        center
            .iter()
            .map(|&c| {
                let avg = c / n;
                if avg > 0.33 {
                    1
                } else if avg < -0.33 {
                    -1
                } else {
                    0
                }
            })
            .collect()
    }

    fn max_radius(center: &[Trit], points: &[Vec<Trit>]) -> f64 {
        points
            .iter()
            .map(|p| trit_distance(center, p).unwrap())
            .fold(0.0f64, f64::max)
    }

    fn build_recursive(data: &[Vec<Trit>], indices: &[usize], dim: usize) -> BallTreeNode {
        let points: Vec<&Vec<Trit>> = indices.iter().map(|&i| &data[i]).collect();
        let center = Self::trit_centroid(&points.iter().map(|&&ref p| p.clone()).collect::<Vec<_>>(), dim);
        let radius = Self::max_radius(&center, &points.iter().map(|&&ref p| p.clone()).collect::<Vec<_>>());

        if indices.len() <= BALL_TREE_LEAF_SIZE {
            return BallTreeNode {
                center,
                radius,
                indices: indices.to_vec(),
                children: Vec::new(),
            };
        }

        // Split along the dimension with greatest spread
        let mut best_dim = 0;
        let mut best_spread = 0.0;
        for d in 0..dim {
            let vals: Vec<i8> = indices.iter().map(|&i| data[i][d]).collect();
            let min = vals.iter().copied().min().unwrap() as f64;
            let max = vals.iter().copied().max().unwrap() as f64;
            let spread = max - min;
            if spread > best_spread {
                best_spread = spread;
                best_dim = d;
            }
        }

        let median_val = {
            let mut vals: Vec<i8> = indices.iter().map(|&i| data[i][best_dim]).collect();
            vals.sort();
            vals[vals.len() / 2]
        };

        let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = indices
            .iter()
            .partition(|&&i| data[i][best_dim] <= median_val);

        // Handle degenerate case where all points go to one side
        if left_indices.is_empty() || right_indices.is_empty() {
            let mid = indices.len() / 2;
            let mut sorted = indices.to_vec();
            sorted.sort_by_key(|&i| data[i][best_dim]);
            let (l, r) = sorted.split_at(mid);
            let left = Self::build_recursive(data, l, dim);
            let right = Self::build_recursive(data, r, dim);
            let all_indices: Vec<usize> = l.iter().chain(r.iter()).copied().collect();
            return BallTreeNode {
                center,
                radius,
                indices: all_indices,
                children: vec![left, right],
            };
        }

        let left = Self::build_recursive(data, &left_indices, dim);
        let right = Self::build_recursive(data, &right_indices, dim);
        let all_indices: Vec<usize> = left_indices
            .iter()
            .chain(right_indices.iter())
            .copied()
            .collect();

        BallTreeNode {
            center,
            radius,
            indices: all_indices,
            children: vec![left, right],
        }
    }

    /// Query the k nearest neighbors. Returns indices and distances sorted by distance.
    pub fn query(&self, query: &[Trit], k: usize) -> Result<Vec<(usize, f64)>, String> {
        validate_ternary(query)?;
        // Max-heap: we keep the k closest, evicting the farthest
        let mut heap: Vec<(f64, usize)> = Vec::with_capacity(k + 1);
        self.search(&self.root, query, k, &mut heap);
        heap.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        Ok(heap.into_iter().map(|(d, i)| (i, d)).collect())
    }

    fn search(
        &self,
        node: &BallTreeNode,
        query: &[Trit],
        k: usize,
        heap: &mut Vec<(f64, usize)>,
    ) {
        let dist_to_center = trit_distance(query, &node.center).unwrap();

        // Prune: if the closest this ball can be is farther than our k-th best, skip
        let lower_bound = (dist_to_center - node.radius).max(0.0);
        if heap.len() >= k && lower_bound >= heap[0].0 {
            return;
        }

        if node.children.is_empty() {
            // Leaf node: check all points
            for &idx in &node.indices {
                let d = trit_distance(query, &self.data[idx]).unwrap();
                if heap.len() < k {
                    heap.push((d, idx));
                    if heap.len() == k {
                        // Convert to max-heap behavior by sorting
                        heap.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                    }
                } else if d < heap[0].0 {
                    heap[0] = (d, idx);
                    // Re-sort to maintain max at top
                    heap.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                }
            }
        } else {
            // Visit children ordered by distance to their centers
            let mut children: Vec<_> = node.children.iter().collect();
            children.sort_by(|a, b| {
                let da = trit_distance(query, &a.center).unwrap();
                let db = trit_distance(query, &b.center).unwrap();
                da.partial_cmp(&db).unwrap()
            });
            for child in children {
                self.search(child, query, k, heap);
            }
        }
    }
}

/// Brute-force KNN baseline: returns (distance, index) pairs sorted ascending.
pub fn brute_force_knn(data: &[Vec<Trit>], query: &[Trit], k: usize) -> Result<Vec<(usize, f64)>, String> {
    validate_ternary(query)?;
    let mut dists: Vec<(f64, usize)> = data
        .iter()
        .enumerate()
        .map(|(i, p)| (trit_distance(query, p).unwrap(), i))
        .collect();
    dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let k = k.min(dists.len());
    Ok(dists[..k].iter().map(|&(d, i)| (i, d)).collect())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_match_distance_zero() {
        let a = vec![1, -1, 0, 1, 0];
        assert_eq!(trit_distance(&a, &a).unwrap(), 0.0);
    }

    #[test]
    fn test_opposite_distance_max() {
        let a = vec![1, 1, 1];
        let b = vec![-1, -1, -1];
        // Each position: opposite => cost 2, total 6
        assert_eq!(trit_distance(&a, &b).unwrap(), 6.0);
    }

    #[test]
    fn test_mixed_distances() {
        let a = vec![1, 0, -1];
        let b = vec![-1, 0, 1];
        // pos 0: opposite (2), pos 1: same (0), pos 2: opposite (2)
        assert_eq!(trit_distance(&a, &b).unwrap(), 4.0);
    }

    #[test]
    fn test_zero_vs_nonzero() {
        let a = vec![0, 0, 0];
        let b = vec![1, -1, 1];
        // Each: 0 vs ±1 => cost 1 each
        assert_eq!(trit_distance(&a, &b).unwrap(), 3.0);
    }

    #[test]
    fn test_weighted_distance() {
        let a = vec![1, -1, 0];
        let b = vec![-1, 0, 1];
        let w = vec![2.0, 1.0, 3.0];
        // pos 0: opposite(2)*2=4, pos 1: 0vs±1(1)*1=1, pos 2: 0vs±1(1)*3=3 => 8
        assert_eq!(weighted_trit_distance(&a, &b, &w).unwrap(), 8.0);
    }

    #[test]
    fn test_validate_ternary() {
        assert!(validate_ternary(&[1, -1, 0]).is_ok());
        assert!(validate_ternary(&[2]).is_err());
        assert!(validate_ternary(&[1, 3, -1]).is_err());
    }

    #[test]
    fn test_knn_classifies_correctly_simple() {
        let mut knn = KnnClassifier::new(1);
        // Two classes: 0 and 1
        knn.add(vec![1, 1, 1], 0.0).unwrap();
        knn.add(vec![-1, -1, -1], 1.0).unwrap();
        assert_eq!(knn.predict(&[1, 1, 1]).unwrap(), 0.0);
        assert_eq!(knn.predict(&[-1, -1, -1]).unwrap(), 1.0);
    }

    #[test]
    fn test_knn_k5() {
        let mut knn = KnnClassifier::new(5);
        // Class 0: positive vectors
        for _ in 0..3 {
            knn.add(vec![1, 1, 0], 0.0).unwrap();
        }
        // Class 1: negative vectors
        for _ in 0..2 {
            knn.add(vec![-1, -1, 0], 1.0).unwrap();
        }
        // Query near positive: should classify as 0
        assert_eq!(knn.predict(&[1, 1, 1]).unwrap(), 0.0);
    }

    #[test]
    fn test_knn_tie_breaking() {
        let mut knn = KnnClassifier::new(2);
        knn.add(vec![1, 0, 0], 0.0).unwrap();
        knn.add(vec![-1, 0, 0], 1.0).unwrap();
        // Query [0,0,0] is equidistant: dist to [1,0,0] = 1, dist to [-1,0,0] = 1
        // Tie-breaking: smallest label wins => 0
        assert_eq!(knn.predict(&[0, 0, 0]).unwrap(), 0.0);
    }

    #[test]
    fn test_knn_weighted_voting() {
        let mut knn = KnnClassifier::new(3).with_weighted(true);
        knn.add(vec![1, 1, 1], 0.0).unwrap(); // dist 0 to query
        knn.add(vec![-1, -1, -1], 1.0).unwrap(); // dist 6 to [1,1,1]
        knn.add(vec![-1, -1, 0], 1.0).unwrap(); // dist 4 to [1,1,1]
        // Query [1,1,1]: class 0 gets weight ~1e10, class 1 gets 1/6 + 1/4
        assert_eq!(knn.predict(&[1, 1, 1]).unwrap(), 0.0);
    }

    #[test]
    fn test_knn_regressor_weighted() {
        let mut knn = KnnRegressor::new(3);
        knn.add(vec![1, 0, 0], 10.0).unwrap();
        knn.add(vec![-1, 0, 0], 20.0).unwrap();
        knn.add(vec![0, 1, 0], 15.0).unwrap();
        let pred = knn.predict(&[1, 0, 0]).unwrap();
        // Exact match gets huge weight, should be close to 10
        assert!((pred - 10.0).abs() < 0.1, "Expected ~10, got {}", pred);
    }

    #[test]
    fn test_ball_tree_matches_brute_force() {
        let data: Vec<Vec<Trit>> = vec![
            vec![1, 1, 1, 0],
            vec![-1, -1, -1, 0],
            vec![1, 0, -1, 1],
            vec![0, 0, 0, 0],
            vec![1, -1, 1, -1],
            vec![-1, 1, -1, 1],
            vec![1, 1, 0, 0],
            vec![0, -1, 1, 1],
            vec![-1, 0, 0, -1],
            vec![0, 1, -1, 0],
        ];
        let tree = BallTree::build(data.clone()).unwrap();
        let query = vec![1, 0, 1, 0];

        let bt_results = tree.query(&query, 3).unwrap();
        let bf_results = brute_force_knn(&data, &query, 3).unwrap();

        // Distances should match (indices may differ for equidistant points)
        let bt_dists: Vec<f64> = bt_results.iter().map(|&(_, d)| d).collect();
        let bf_dists: Vec<f64> = bf_results.iter().map(|&(_, d)| d).collect();
        assert_eq!(bt_dists.len(), bf_dists.len());
        for (bd, fd) in bt_dists.iter().zip(bf_dists.iter()) {
            assert!((bd - fd).abs() < 1e-10, "Distance mismatch: {} vs {}", bd, fd);
        }
    }

    #[test]
    fn test_ball_tree_single_point() {
        let data = vec![vec![1, -1, 0]];
        let tree = BallTree::build(data).unwrap();
        let results = tree.query(&[1, -1, 0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
        assert_eq!(results[0].1, 0.0);
    }

    #[test]
    fn test_ball_tree_k_equals_n() {
        let data: Vec<Vec<Trit>> = vec![
            vec![1, 0],
            vec![-1, 0],
            vec![0, 1],
        ];
        let tree = BallTree::build(data.clone()).unwrap();
        let results = tree.query(&[0, 0], 3).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_brute_force_knn_basic() {
        let data: Vec<Vec<Trit>> = vec![
            vec![1, 1],
            vec![-1, -1],
            vec![0, 0],
        ];
        let results = brute_force_knn(&data, &[1, 1], 2).unwrap();
        assert_eq!(results[0].0, 0); // closest is [1,1]
        assert_eq!(results[0].1, 0.0);
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let a = vec![1, 0];
        let b = vec![1, 0, 1];
        assert!(trit_distance(&a, &b).is_err());
    }
}

/// Helper wrapper for f64 ordering in tests.
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedFloat(f64);

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}
