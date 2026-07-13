# ternary-knn

K-nearest neighbors **classification** for ternary vector spaces {-1, 0, +1}, using a ternary-specific distance metric and brute-force neighbor search.

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

Trit distance is a **proper metric**: it is non-negative, symmetric, identity-of-indiscernibles holds, and the triangle inequality is satisfied (the worst case is `+1 → 0 → -1`, where `d(+1,-1) = 2 = 1 + 1`).

## Quick Start

```rust
use ternary_knn::{validate_ternary, DataPoint, KNNClassifier, TernaryDataset};

// Build a labeled dataset of ternary vectors. Labels are i32.
let dataset = TernaryDataset::new(vec![
    DataPoint::new(vec![1, 1, 1], 1),
    DataPoint::new(vec![1, 1, -1], 1),
    DataPoint::new(vec![-1, -1, -1], 0),
    DataPoint::new(vec![-1, -1, 1], 0),
])
.unwrap();

// k = 3 neighbors, brute-force search.
let mut knn = KNNClassifier::new(3);
knn.fit(dataset);

// Classify query points (must match the dataset dimension).
assert_eq!(knn.predict(&[1, 1, 0]).unwrap(), 1);   // closer to the +1 cluster
assert_eq!(knn.predict(&[-1, -1, 0]).unwrap(), 0); // closer to the -1 cluster

// Batch prediction over several queries.
let labels = knn
    .predict_batch(&[vec![1, 1, 0], vec![-1, -1, 0]])
    .unwrap();
assert_eq!(labels, vec![1, 0]);

// Report accuracy against a labeled test set.
// assert_eq!(knn.accuracy(&test_set).unwrap(), 0.95);
```

## How prediction works

1. **Distance** — for every training point, compute `trit_distance(query, point)`.
2. **Sort** — order points by ascending distance (stable; NaN-safe via `f64::total_cmp`).
3. **Select** — take the `k` nearest. `k` larger than the dataset is clamped to the dataset size.
4. **Vote** — majority label among those `k` neighbors wins.

**Tie-breaking is deterministic:**
- *Distance ties* are broken by dataset order (the sort is stable).
- *Voting ties* (two labels with equal vote counts) are resolved by the **smallest label** winning.

## API Reference

### Distance & validation functions

```rust
type Trit = i8;

fn validate_ternary(vec: &[Trit]) -> Result<(), String>;
fn trit_distance(a: &[Trit], b: &[Trit]) -> Result<f64, String>;
fn normalized_trit_distance(a: &[Trit], b: &[Trit]) -> Result<f64, String>;
```

`normalized_trit_distance` divides the raw trit distance by `2 * dims`, mapping it to `[0, 1]`.

### `DataPoint` / `TernaryDataset`

```rust
let pt = DataPoint::new(vec![1, 0, -1], 7);
let dataset = TernaryDataset::new(vec![pt]).unwrap(); // Err on empty / dim mismatch / invalid trit
dataset.len();       // usize
dataset.is_empty();  // bool
```

### `KNNClassifier`

```rust
let mut knn = KNNClassifier::new(3);          // panics if k == 0
knn.fit(dataset);                              // consumes the dataset
knn.predict(&[1, 0, -1])?            -> Result<i32, String>;
knn.predict_batch(&[vec![1, 0, -1]])? -> Result<Vec<i32>, String>;
knn.accuracy(&test_dataset)?         -> Result<f64, String>;
```

`predict` returns an error when the classifier is unfitted, the query contains an
invalid trit, or the query dimension differs from the dataset.

## Example: distance by hand

```rust
use ternary_knn::{trit_distance, normalized_trit_distance};

// [1,0,-1] vs [-1,0,1]:
//   pos 0: +1 vs -1  -> 2   (opposition)
//   pos 1:  0 vs  0  -> 0   (agreement)
//   pos 2: -1 vs +1  -> 2   (opposition)
// raw = 4
assert_eq!(trit_distance(&[1, 0, -1], &[-1, 0, 1]).unwrap(), 4.0);
// normalized = 4 / (2 * 3) = 0.6666...
assert_eq!(normalized_trit_distance(&[1, 0, -1], &[-1, 0, 1]).unwrap(), 4.0 / 6.0);
```

## Performance

| Operation | Complexity |
|-----------|-----------|
| `trit_distance(a, b)` | O(d) — single pair |
| `validate_ternary(v)` | O(d) |
| `KNNClassifier::predict` | O(n·d + n·log n) brute force per query |

Memory: O(n·d) — the classifier stores the training set as-is (no auxiliary index).

## Ecosystem connections

- **[`ternary-types`](https://github.com/SuperInstance/ternary-types)** — shared trit type this crate depends on
- **[`ternary-quantize`](https://github.com/SuperInstance/ternary-quantize)** — produces the ternary vectors this crate classifies
- **[`ternary-transformer`](https://github.com/SuperInstance/ternary-transformer)** — transformer outputs feed directly into KNN search
- **[`ternary-svm`](https://github.com/SuperInstance/ternary-svm)** — alternative classifier for the same ternary feature space

## Testing

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check
```

16 unit tests covering: exact-match distance (0), fully-opposed vectors (2 per
dimension), mixed zero/nonzero mismatches, non-trivial normalized distance,
invalid-trit rejection, dataset construction errors (empty / dimension mismatch),
k=1 accuracy, k=3 majority vote, `k` larger than the dataset (clamping), a voting
tie resolved by the smallest label, all-identical points, query dimension mismatch
(clean error), and predicting before fitting.

## License

MIT
