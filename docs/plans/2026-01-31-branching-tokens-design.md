# BranchingTokens Design Specification

## Overview

BranchingTokens is a data generator for training and testing LLMs on transformation tracing tasks.

**The Task:** Given an input vector and output vector, the LLM must generate a trace showing the sequence of transformations that were applied.

**Example:**
```
Input:  3 1 4 1
Output: 1 3 4 9
LLM predicts: cumsum:3,3+1=4,4+4=8,8+1=9:[3,4,8,9] sort_asc:[1,3,4,9]
```

## Core Concepts

- **Vectors:** Lists of integers in range [0, k-1], where k is configurable
- **Transformations:** 16 fixed operations that modify vectors without changing length, always applying mod k
- **Transformation Graph:** A directed layered graph where vertices represent transformations. Paths through this graph define valid transformation sequences
- **Traces:** Step-by-step logs of how each transformation modified the vector, with configurable detail level per transformation

---

## The 16 Transformations

All transformations preserve vector length and apply mod k after execution.

| # | Name | Description | Example (k=10) |
|---|------|-------------|----------------|
| 1 | `reverse` | Reverse the vector | [3,1,4] → [4,1,3] |
| 2 | `sort_asc` | Sort ascending | [3,1,4] → [1,3,4] |
| 3 | `sort_desc` | Sort descending | [3,1,4] → [4,3,1] |
| 4 | `cumsum` | Cumulative sum (mod k) | [3,1,4] → [3,4,8] |
| 5 | `cumsum_reverse` | Cumsum from right | [3,1,4] → [8,5,4] |
| 6 | `add_1` | Add 1 to all (mod k) | [3,1,4] → [4,2,5] |
| 7 | `add_2` | Add 2 to all (mod k) | [3,1,4] → [5,3,6] |
| 8 | `add_3` | Add 3 to all (mod k) | [3,1,4] → [6,4,7] |
| 9 | `diff` | Pairwise differences (first element unchanged) | [3,1,4] → [3,8,3] |
| 10 | `swap_pairs` | Swap adjacent pairs | [3,1,4,2] → [1,3,2,4] |
| 11 | `rotate_left` | Rotate left by 1 | [3,1,4] → [1,4,3] |
| 12 | `rotate_right` | Rotate right by 1 | [3,1,4] → [4,3,1] |
| 13 | `negate` | Negate all (mod k): x → k-x | [3,1,4] → [7,9,6] |
| 14 | `double` | Double all (mod k) | [3,1,4] → [6,2,8] |
| 15 | `square` | Square all (mod k) | [3,1,4] → [9,1,6] |
| 16 | `min_prefix` | Running minimum | [3,1,4] → [3,1,1] |

---

## Transformation Graph Structure

### Definition

- `d` layers (depth), `m` vertices per layer
- Constraint: `d × m ≤ 16` (total vertices ≤ available transformations)
- Each vertex maps to a unique transformation via a configurable mapping
- Edges connect every vertex in layer `i` to every vertex in layer `i+1` (fully connected between adjacent layers)

### Path Sampling

1. Sample desired path length `n` from configured range `[min_len, max_len]`
2. A path of length `n` requires `n` consecutive layers
3. Pick starting layer `s` such that `s + n - 1 ≤ d`
4. At each layer, pick one vertex (uniformly random from allowed paths - see split logic)
5. The sequence of transformations = the transformations mapped to those vertices

### Example (d=4, m=4)

```
Layer 1: [reverse, sort_asc, sort_desc, cumsum]
Layer 2: [cumsum_reverse, add_1, add_2, add_3]
Layer 3: [diff, swap_pairs, rotate_left, rotate_right]
Layer 4: [negate, double, square, min_prefix]
```

A path of length 2 starting at layer 2 might be: `add_2 → rotate_left`

---

## Train/Validation Split

### Path Enumeration

- Enumerate all possible paths through the graph for each valid length
- For length `n`, paths start at any layer `s` where `s + n - 1 ≤ d`
- Total paths = sum over all valid (length, start_layer) combinations of `m^n`

### Split Process

1. Enumerate all unique paths (as sequences of vertex IDs)
2. Shuffle using the global seed
3. Assign first 80% to training, remaining 20% to validation
4. Store as two disjoint sets of allowed paths

### Sampling at Runtime

- Training: sample only from training path set
- Validation: sample only from validation path set
- This ensures the LLM never sees validation transformation sequences during training

### Configurability

- Split ratio configurable (default 80:20)
- The split logic is isolated in a single function for easy modification later

---

## Trace Format & Parsing

### Output Format

```
INPUT OUTPUT TRACE
```

Where:
- `INPUT`: space-separated integers (e.g., `3 1 4 1`)
- `OUTPUT`: space-separated integers (e.g., `1 1 3 4`)
- `TRACE`: transformation traces joined by spaces

### Trace Structure per Transformation

Each transformation defines its own trace format. Example formats:

- **Simple** (sort_asc): `sort_asc:[1,1,3,4]` — just the result
- **Medium** (cumsum): `cumsum:3,3+1=4,4+4=8,8+1=9:[3,4,8,9]` — shows operations and result
- **Detailed** (configurable per transformation in code)

### Full Example

```
3 1 4 1 1 1 3 4 cumsum:3,3+1=4,4+4=8,8+1=9:[3,4,8,9] sort_asc:[1,3,4,9]
```

### Parsing Levels

1. **Coarse:** Split by transformation names → check correct sequence
2. **Medium:** Extract final vectors per transformation → check intermediate states
3. **Fine:** Parse arithmetic expressions → check individual operations

### Implementation

- Each transformation receives a trace list and appends its trace string
- Trace format is defined per-transformation, easily modifiable

---

## Hydra Configuration

```yaml
# config.yaml
seed: 42                    # Global seed for reproducibility

vector:
  length: 8                 # Vector length
  k: 10                     # Modulo value (values in [0, k-1])

graph:
  d: 4                      # Number of layers
  m: 4                      # Vertices per layer
  vertex_to_transform:      # Optional: explicit mapping (default: 0-15)
    - reverse
    - sort_asc
    # ... etc

path:
  min_len: 1                # Minimum transformations in sequence
  max_len: 4                # Maximum transformations in sequence

split:
  train_ratio: 0.8          # Training set proportion

dataset:
  train_size: 10000         # Number of training examples
  val_size: 2000            # Number of validation examples
```

### Validation Rules

- `d × m ≤ 16` (cannot exceed available transformations)
- `max_len ≤ d` (path cannot exceed graph depth)
- `min_len ≥ 1`

---

## Project Structure

```
BranchingTokens/
├── config/
│   └── config.yaml              # Default Hydra config
├── src/
│   ├── __init__.py
│   ├── transformations.py       # 16 transformation functions with trace logic
│   ├── graph.py                 # Graph construction & path enumeration
│   ├── split.py                 # Train/val split logic
│   ├── generator.py             # Example generation (vector + path → trace)
│   ├── parser.py                # Trace parsing at different levels
│   └── dataset.py               # Main dataset generation orchestration
├── scripts/
│   └── generate.py              # Hydra entry point: python scripts/generate.py
├── outputs/                     # Generated datasets (gitignored)
│   ├── train.txt
│   └── val.txt
├── tests/
│   └── test_transformations.py  # Unit tests for transformations
├── specs/
│   └── specs.md                 # Original notes
└── requirements.txt
```

### Output Format

- One example per line in `train.txt` and `val.txt`
- Format: `INPUT OUTPUT TRACE`

---

## Summary of Key Decisions

| Aspect | Decision |
|--------|----------|
| Transformations | 16 fixed, parameter-free operations |
| Modulo | Applied after every transformation |
| Graph | `d` layers × `m` vertices, fully connected between adjacent layers |
| Vertex mapping | 1:1 to transformations (smaller graph = subset of 16) |
| Path sampling | Sample length from range, then walk through consecutive layers |
| Train/val split | Enumerate all paths, shuffle, 80/20 split |
| Trace format | Configurable per transformation, space-delimited |
| LLM task | Given `INPUT OUTPUT`, predict `TRACE` |
| Parsing | Three levels: coarse (order), medium (states), fine (operations) |
| Reproducibility | Single global seed |
| Config | Hydra with validation rules |
