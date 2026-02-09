# Experimental Dataset Generation

Config-based system for generating controlled experimental datasets.

## Quick Start

### Experiment 1: Exhaustive Single Operations
```bash
cd BranchingTokens

# Generate with default sizes (140k train, 20k test)
python scripts/generate_experimental.py --config-name=experiment_1

# Or customize sizes (must be ≤ 160,000 total)
python scripts/generate_experimental.py --config-name=experiment_1 \
    dataset.train_size=100000 \
    dataset.val_size=10000
```

### Experiment 2: Mixed Train, Pure 2-op Test
```bash
# Generate with default sizes (100k train, 10k test)
python scripts/generate_experimental.py --config-name=experiment_2

# Or customize sizes (any total you want!)
python scripts/generate_experimental.py --config-name=experiment_2 \
    dataset.train_size=50000 \
    dataset.val_size=5000
```

---

## Experiment Descriptions

### Experiment 1: Exhaustive Single Operations

**Goal**: Test memorization vs. generalization on single operations

**Dataset Properties**:
- **Total possible**: 160,000 pairs (10^4 inputs × 16 operations)
- **Train**: 140k examples (87.5%)
- **Test**: 20k examples (12.5%)
- **Both**: 100% single-operation traces
- **Coverage**: ALL possible (input, operation) pairs exhaustively enumerated

**What's Balanced**:
- NOT balanced - exhaustive enumeration of all pairs
- Operations appear with equal frequency (10k times each)

**Expected Outcome**:
Near-perfect test accuracy since test is drawn from same distribution as training.

**Config**: `config/experiment_1.yaml`

---

### Experiment 2: Mixed Train (1+2 ops), Pure Test (2 ops)

**Goal**: Test compositional generalization - can the model compose operations it learned individually?

**Dataset Properties**:
- **Train**: Configurable size (default 100k)
  - 50% 1-operation traces
  - 50% 2-operation traces
- **Test**: Configurable size (default 10k)
  - 100% 2-operation traces only
- **Input Separation**: Train and test use completely DISJOINT sets of input vectors
  - 67% of input space reserved for train
  - 33% of input space reserved for test
  - Zero overlap guaranteed

**What's Balanced**:
1. **Sequence length**: 50/50 split between 1-op and 2-op in train
2. **Operations**: Within each length, operations sampled uniformly
   - 1-op: All 16 operations sampled equally (~31 times each per 500 examples)
   - 2-op: All 48 operation pairs sampled equally (~10 times each per 500 examples)

**Expected Outcome**:
Tests whether model can:
1. Learn individual operations from 1-op examples
2. Learn compositional patterns from 2-op examples
3. Generalize composition to unseen input vectors in test set

**Config**: `config/experiment_2.yaml`

---

## Configuration System

### Experiment 1 Config (`config/experiment_1.yaml`)

```yaml
seed: 42

vector:
  length: 4
  k: 10

graph:
  d: 4
  m: 4

path:
  min_len: 1
  max_len: 1  # Single operations only

split:
  train_ratio: 0.875  # 140k/160k

dataset:
  mode: exhaustive  # Generate ALL pairs
  train_size: 140000
  val_size: 20000
```

**Easily configurable**: Change `train_size` and `val_size` via command line!

### Experiment 2 Config (`config/experiment_2.yaml`)

```yaml
seed: 42

vector:
  length: 4
  k: 10

graph:
  d: 4
  m: 4

path:
  min_len: 1
  max_len: 2  # 1-op and 2-op

dataset:
  mode: mixed_train_pure_test
  train_size: 100000  # Easily change this!
  val_size: 10000     # Easily change this!
  train_test_split_inputs: 0.67  # 67% inputs for train
```

**Easily configurable**: Change any size via command line!

---

## Verification

After generation, verify the dataset properties:

```python
import json
from collections import Counter

# Load data
with open('train.json', 'r') as f:
    train = json.load(f)
with open('val.json', 'r') as f:
    val = json.load(f)

# Check sequence length distribution
train_lengths = [ex['output'].count(';') + 1 for ex in train]
print("Train:", Counter(train_lengths))

val_lengths = [ex['output'].count(';') + 1 for ex in val]
print("Test:", Counter(val_lengths))

# For Experiment 2: Check input separation
train_inputs = {ex['input'].split('INPUT : [ ')[1].split(' ]')[0] for ex in train}
test_inputs = {ex['input'].split('INPUT : [ ')[1].split(' ]')[0] for ex in val}
print(f"Input overlap: {len(train_inputs & test_inputs)} (should be 0)")
```

---

## Implementation Details

**Script**: `scripts/generate_experimental.py`

**Key Features**:
- Hydra config system for easy parameter control
- Exhaustive enumeration of input space (10^4 possible vectors)
- Controlled input partitioning for train/test separation
- Uniform sampling for balanced operation distribution
- Fixed seed for reproducibility

**Modified Files**:
- `src/generator.py`: Added `fixed_input` parameter
- `scripts/generate_experimental.py`: Main generation script
- `config/experiment_1.yaml`: Experiment 1 config
- `config/experiment_2.yaml`: Experiment 2 config
