# Experimental Datasets

## Overview

Three controlled experiments to study model learning of compositional operations.

## Experiment 1: Exhaustive Single Operations

**Goal**: Test memorization vs. generalization on single operations

**Dataset**:
- All possible (input vector, single operation) pairs
- 10^4 inputs × 16 operations = 160,000 total examples
- Split: 140k train, 20k test

**Properties**:
- 100% single-operation traces
- All 16 operations represented
- Exhaustive coverage of input space

**Generation**:
```bash
cd BranchingTokens
python scripts/generate_experiments.py 1 --output-dir experiment_1_single_ops
```

**Expected outcome**: Model should achieve near-perfect accuracy since test set is drawn from same distribution as training.

---

## Experiment 2: Mixed Train (1+2 ops), Pure Test (2 ops)

**Goal**: Test compositional generalization - can model compose operations it learned individually?

**Dataset**:
- Train: 100k examples (50k 1-op, 50k 2-op) - balanced
- Test: 10k examples (100% 2-op only)

**Properties**:
- Train contains both 1-op (learning individual operations) and 2-op (learning composition)
- Test is pure 2-op (testing composition)
- **Input separation**: No input vectors overlap between train and test
- Within 2-op: Same input can appear with different operation pairs

**Generation**:
```bash
cd BranchingTokens
python scripts/generate_experiments.py 2 --output-dir experiment_2_mixed_train
```

**Expected outcome**: Tests whether model can generalize compositional patterns to unseen input vectors.

---

## Experiment 3: TBD

[Awaiting description]

---

## Verification Results

### Experiment 1 ✓
- Train: 140,000 examples (100% 1-op)
- Test: 20,000 examples (100% 1-op)
- All 16 operations present

### Experiment 2 ✓
- Train: 100,000 examples (50% 1-op, 50% 2-op)
- Test: 10,000 examples (100% 2-op)
- Input separation: ✓ (0 overlap between train/test inputs)
- Unique train inputs: 6,667
- Unique test inputs: 3,168

---

## Implementation Details

**Script**: `scripts/generate_experiments.py`

**Key Features**:
- Exhaustive enumeration of input space (10^4 possible 4-length vectors)
- Controlled input partitioning for train/test separation
- Balanced generation for mixed experiments
- Fixed seed for reproducibility

**Modified Files**:
- `src/generator.py`: Added `fixed_input` parameter to control input vectors
- `scripts/generate_experiments.py`: New script for experimental dataset generation
