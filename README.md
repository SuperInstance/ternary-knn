# ternary-knn

K-nearest neighbors for ternary vector spaces {-1, 0, +1} — with a custom distance metric, ball-tree indexing, and both classification and regression.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## Why this exists

Standard KNN on continuous vectors uses Euclidean or cosine distance. When your features are ternary — quantized neural activations, balanced ternary processor states, trit-based cryptographic keys — those generic metrics leave information on the table. The distance between `+1` and `-1` is not the same as the distance between `+1` and `0`: one is a disagreement, the other is an *opposition*. This crate encodes that distinction directly into the distance metric.

## The key insight

Ternary distance has three levels, not two. In binary space, two bits either match (distance 0) or don't (distance 1). In ternary space:

| Pair | Cost | Meaning |
|------|------|---------|
| same value | 0 | agreement |
| ±1 vs 0 | 1 | one is silent, the other isn't |
| +1 vs −1 | 2 | active disagreement — the signal is inverted |

This three-level metric captures *structural* disagreement that Hamming distance can't. A zero-vs-nonzero mismatch is a soft disagreement; a +1-vs-−1 mismatch is a hard contradiction. That distinction matters when you're classifying quantized patterns.

## Quick Start

```rust
use ternary_knn::{KnnClassifier, KnnRegressor, BallTree, trit_distance, brute_force_knn};

// ── Classification ──
let mut knn = KnnClassifier::new(3);  // k = 3 neighbors
knn.add(vec![1, 1, 1], 0.0).unwrap();    // class 0: positive region
knn.add(vec![-1, -1, -1], 1.0).unwrap(); // class 1: negative region
knn.add(vec![0, 0, 0], 0.0).unwrap();    // class 0: neutral

let label = knn.predict(&[1, 1, 0]).unwrap(); // → 0.0 (closer to positive)

// ── Weighted voting (inverse-distance weighting) ──
let knn_w = KnnClassifier::new(5).with_weighted(true);

// ── Regression ──
let mut reg = KnnRegressor::new(3);
reg.add(vec![1, 0, 0], 10.0).unwrap();
reg.add(vec![-1, 0, 0], 20.0).unwrap();
reg.add(vec![0, 1, 0], 15.0).unwrap();
let value = reg.predict(&[1, 0, 0]).unwrap(); // ≈ 10.0 (exact match dominates)

// ── Ball tree for fast queries on large datasets ──
let data = vec![
    vec![1, 1, 1, 0], vec![-1, -1, -1, 0],
    vec![1, 0, -1, 1], vec![0, 0, 0, 0],
];
let tree = BallTree::build(data.clone()).unwrap();
let neighbors = tree.query(&[1, 0, 0, 0], 2).unwrap();
// → [(index, distance), ...]
```

## Architecture

```
                    ┌─────────────────────────────┐
  Training data ──→ │   BallTree (spatial index)   │
  [Vec<Trit>]       │   Recursive median splits    │
                    │   Leaf size: 8 points         │
                    └──────────────┬────────────────┘
                                   │ query(query, k)
                                   ▼
                    ┌─────────────────────────────┐
                    │   KnnClassifier              │
                    │   - majority vote            │
                    │   - weighted vote (1/dist)   │
                    │   - tie-break: smallest label│
                    ├─────────────────────────────┤
                    │   KnnRegressor               │
                    │   - inverse-distance average │
                    │   - exact match: inf weight  │
                    └─────────────────────────────┘

  Brute-force baseline: O(n·d) per query, no index
```

The ball tree splits along the dimension with the greatest spread, using median partitions. Centroids are rounded to the nearest trit. Pruning skips subtrees whose minimum possible distance exceeds the current k-th best.

## API Reference

### Distance Functions

```rust
fn trit_distance(a: &[Trit], b: &[Trit]) -> Result<f64, String>
fn weighted_trit_distance(a: &[Trit], b: &[Trit], weights: &[f64]) -> Result<f64, String>
fn validate_ternary(vec: &[Trit]) -> Result<(), String>
```

### KnnClassifier

```rust
let mut clf = KnnClassifier::new(k: usize);
clf.with_weighted(true);               // enable inverse-distance weighting
clf.add(features: Vec<Trit>, label: f64);
clf.fit(samples: &[(Vec<Trit>, f64)]);
clf.predict(query: &[Trit]) -> Result<f64, String>
```

### KnnRegressor

```rust
let mut reg = KnnRegressor::new(k: usize);
reg.add(features: Vec<Trit>, label: f64);
reg.predict(query: &[Trit]) -> Result<f64, String>
```

### BallTree

```rust
let tree = BallTree::build(data: Vec<Vec<Trit>>) -> Result<BallTree, String>;
tree.query(query: &[Trit], k: usize) -> Result<Vec<(usize, f64)>, String>
// Returns (index, distance) pairs sorted ascending
```

### Brute Force Baseline

```rust
fn brute_force_knn(data: &[Vec<Trit>], query: &[Trit], k: usize) -> Result<Vec<(usize, f64)>, String>
```

## Real-world example

A fishing boat runs a ternary neural network that classifies sonar returns as {-1: empty water, 0: uncertain, +1: fish school}. The network's final layer outputs a 64-dimensional ternary vector per ping. With 50,000 labeled pings in the database, brute-force KNN takes O(50K × 64) per query — about 3.2M comparisons per classification.

A ball tree with leaf size 8 cuts this to O(64 × log 50K) ≈ 1,000 comparisons on average. At 20 pings per second, you go from 64M ops/sec (barely fits on the embedded CPU) to 20K ops/sec (trivial, with power to spare for the sonar DSP).

## Ecosystem connections

- **[`ternary-quantize`](https://github.com/SuperInstance/ternary-quantize)** — produces the ternary vectors this crate classifies
- **[`ternary-transformer`](https://github.com/SuperInstance/ternary-transformer)** — transformer outputs feed directly into KNN search
- **[`ternary-svm`](https://github.com/SuperInstance/ternary-svm)** — alternative classifier for the same ternary feature space
- **[`ternary-hmm`](https://github.com/SuperInstance/ternary-hmm)** — models temporal sequences of ternary observations

## Performance

| Operation | Complexity | When to use |
|-----------|-----------|-------------|
| `trit_distance(a, b)` | O(d) | Single pair comparison |
| `brute_force_knn` | O(n·d) per query | n < 100, correctness baseline |
| `BallTree::build` | O(n·d·log n) | One-time cost |
| `BallTree::query` | O(d·log n) avg, O(n·d) worst | n > 100, production queries |

Memory: O(n·d) for the ball tree. Each node stores a ternary centroid and radius — the overhead is ~2× the raw data.

## Open questions

- **Curse of dimensionality**: Ball trees degrade above ~50 dimensions. For 64-d ternary vectors, is the pruning still effective, or do we need LSH?
- **Metric properties**: Trit distance is a proper metric (triangle inequality holds). Can we exploit this for exact pruning guarantees?
- **Batched queries**: Processing 100 queries at once could share traversal work. The current API is one-query-at-a-time.
- **Trit packing**: Storing 16 trits per u32 (2 bits each) would cut cache pressure by 8× for large datasets.

## Testing

```bash
cargo test
```

15 tests: exact match (distance 0), opposite vectors (distance 2d), mixed distances, weighted distance, validation, KNN classification with k=1/k=5, tie-breaking (smallest label wins), weighted voting, regression accuracy, ball tree ↔ brute force consistency, edge cases (single point, k=n), dimension mismatch errors.

## License

MIT
