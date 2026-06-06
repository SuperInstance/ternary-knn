# ternary-knn

**K-Nearest Neighbors for Ternary Vector Spaces**

A high-performance KNN implementation designed for ternary data — vectors whose elements are drawn from {-1, 0, +1}. This arises naturally in balanced ternary computing, ternary neural network quantization, trit-based cryptography, and ternary logic circuits.

---

## Why Ternary KNN?

Standard KNN uses Euclidean or cosine distance on continuous vectors. When your data lives in a ternary space, those generic metrics miss the structure:

- **Trit distance** captures the *degree* of disagreement: same (0), different (1), or opposite (2). This is richer than Hamming distance for binary data.
- **Weighted trit distance** lets domain knowledge express which feature positions matter more.
- **Ball tree indexing** exploits the bounded, discrete nature of ternary vectors for sub-linear query time.

This crate provides both brute-force and ball-tree-based KNN, with classifiers and regressors tailored for ternary feature spaces.

---

## Distance Metrics

### Trit Distance

For two trits `a` and `b`:

| a   | b   | Cost |
|-----|-----|------|
|  1  |  1  |  0   |
|  0  |  0  |  0   |
| -1  | -1  |  0   |
|  1  |  0  |  1   |
| -1  |  0  |  1   |
|  0  |  1  |  1   |
|  0  | -1  |  1   |
|  1  | -1  |  2   |
| -1  |  1  |  2   |

Opposite trits (±1) cost twice as much as a zero-vs-nonzero disagreement. The total distance is the sum of per-position costs.

### Weighted Trit Distance

Each position has a weight `w[i] ≥ 0`. The weighted distance is `Σ w[i] * cost(a[i], b[i])`.

---

## Quick Start

```rust
use ternary_knn::{KnnClassifier, KnnRegressor, BallTree, brute_force_knn, trit_distance};

// --- Classification ---
let mut knn = KnnClassifier::new(3);
knn.add(vec![1, 1, 1], 0.0).unwrap();   // class 0
knn.add(vec![-1, -1, -1], 1.0).unwrap(); // class 1
knn.add(vec![0, 0, 0], 0.0).unwrap();    // class 0

let label = knn.predict(&[1, 1, 0]).unwrap(); // → 0.0

// --- Weighted voting ---
let knn_w = KnnClassifier::new(5).with_weighted(true);

// --- Regression ---
let mut reg = KnnRegressor::new(3);
reg.add(vec![1, 0, 0], 10.0).unwrap();
reg.add(vec![-1, 0, 0], 20.0).unwrap();
let value = reg.predict(&[1, 0, 0]).unwrap(); // ≈ 10.0

// --- Ball Tree for fast queries ---
let data = vec![
    vec![1, 1, 1, 0],
    vec![-1, -1, -1, 0],
    vec![1, 0, -1, 1],
    vec![0, 0, 0, 0],
];
let tree = BallTree::build(data.clone()).unwrap();
let neighbors = tree.query(&[1, 0, 0, 0], 2).unwrap();
// neighbors: [(index, distance), ...]

// --- Brute force baseline ---
let bf = brute_force_knn(&data, &[1, 0, 0, 0], 2).unwrap();
```

---

## Architecture

### KnnClassifier
- Configurable `k` (number of neighbors)
- Majority voting (unweighted) or inverse-distance weighting
- Deterministic tie-breaking (smallest label)

### KnnRegressor
- Inverse-distance weighted average of neighbor labels
- Handles exact matches gracefully (zero distance → effectively infinite weight)

### BallTree
- Spatial index built by recursive median splits along the dimension with greatest spread
- Centroids computed as rounded mean (ternary projection)
- Radius = max distance from centroid to any child point
- Pruning: skips subtrees whose minimum possible distance exceeds the current k-th best
- Default leaf size: 8 points

### Brute Force
- O(n·d) per query, useful as a correctness baseline

---

## Performance

The ball tree provides significant speedups for large datasets:
- **Build time**: O(n · d · log n)
- **Query time**: O(d · log n) average case (worst case O(n·d) if points are poorly distributed)
- **Memory**: O(n · d)

For small datasets (n < 100), brute force may be faster due to lower constant factors.

---

## Research Applications

- **Ternary neural networks**: classify quantized activations
- **Post-quantum cryptography**: analyze trit-based key distributions
- **Balanced ternary computing**: nearest-neighbor search in ternary processor state spaces
- **Genomics**: some encoding schemes map nucleotides to ternary representations
- **Recommender systems**: ternary sentiment (negative, neutral, positive) nearest-neighbor models

---

## API Reference

| Function / Struct | Description |
|---|---|
| `trit_distance(a, b)` | Hamming-like trit distance with opposite penalty |
| `weighted_trit_distance(a, b, w)` | Weighted variant |
| `validate_ternary(v)` | Check all elements are {-1, 0, +1} |
| `KnnClassifier` | KNN classifier with majority/weighted voting |
| `KnnRegressor` | KNN regressor with inverse-distance weighting |
| `BallTree` | Spatial index for fast KNN queries |
| `brute_force_knn` | Brute-force KNN baseline |

---

## License

MIT
