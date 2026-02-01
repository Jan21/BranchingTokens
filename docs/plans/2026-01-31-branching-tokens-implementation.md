# BranchingTokens Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a data generator that creates training/validation examples for LLMs to learn transformation tracing.

**Architecture:** Modular Python package with Hydra config. Core modules: transformations (16 ops with traces), graph (path enumeration), split (train/val), generator (example creation), parser (trace parsing). Entry point generates datasets to text files.

**Tech Stack:** Python 3.10+, Hydra, pytest

---

## Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `config/config.yaml`

**Step 1: Create requirements.txt**

```
hydra-core>=1.3.0
omegaconf>=2.3.0
pytest>=7.0.0
```

**Step 2: Create package structure**

```bash
mkdir -p src tests config scripts
touch src/__init__.py tests/__init__.py
```

**Step 3: Create default Hydra config**

`config/config.yaml`:
```yaml
seed: 42

vector:
  length: 8
  k: 10

graph:
  d: 4
  m: 4

path:
  min_len: 1
  max_len: 4

split:
  train_ratio: 0.8

dataset:
  train_size: 10000
  val_size: 2000
```

**Step 4: Commit**

```bash
git add requirements.txt src/ tests/ config/
git commit -m "chore: project setup with Hydra config"
```

---

## Task 2: Transformation Base & First 4 Transformations

**Files:**
- Create: `src/transformations.py`
- Create: `tests/test_transformations.py`

**Step 1: Write failing tests for reverse, sort_asc, sort_desc, cumsum**

`tests/test_transformations.py`:
```python
import pytest
from src.transformations import reverse, sort_asc, sort_desc, cumsum

class TestReverse:
    def test_basic(self):
        vec = [3, 1, 4]
        trace = []
        result = reverse(vec, 10, trace)
        assert result == [4, 1, 3]

    def test_trace_appended(self):
        vec = [3, 1, 4]
        trace = []
        reverse(vec, 10, trace)
        assert len(trace) == 1
        assert "reverse" in trace[0]

class TestSortAsc:
    def test_basic(self):
        vec = [3, 1, 4]
        trace = []
        result = sort_asc(vec, 10, trace)
        assert result == [1, 3, 4]

    def test_trace_appended(self):
        vec = [3, 1, 4]
        trace = []
        sort_asc(vec, 10, trace)
        assert len(trace) == 1
        assert "sort_asc" in trace[0]

class TestSortDesc:
    def test_basic(self):
        vec = [3, 1, 4]
        trace = []
        result = sort_desc(vec, 10, trace)
        assert result == [4, 3, 1]

class TestCumsum:
    def test_basic(self):
        vec = [3, 1, 4]
        trace = []
        result = cumsum(vec, 10, trace)
        assert result == [3, 4, 8]

    def test_modulo(self):
        vec = [3, 5, 4]
        trace = []
        result = cumsum(vec, 10, trace)
        assert result == [3, 8, 2]  # 3, 3+5=8, 8+4=12 mod 10 = 2

    def test_trace_shows_operations(self):
        vec = [3, 1, 4]
        trace = []
        cumsum(vec, 10, trace)
        assert "3+1=4" in trace[0] or "3,3+1=4" in trace[0]
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_transformations.py -v
```
Expected: FAIL with "ModuleNotFoundError: No module named 'src.transformations'"

**Step 3: Implement the 4 transformations**

`src/transformations.py`:
```python
"""Transformation functions for BranchingTokens.

Each transformation:
- Takes a vector (list of ints), modulo k, and a trace list
- Returns a new vector (same length)
- Appends its trace string to the trace list
- All values are mod k after operation
"""

from typing import List


def _format_vec(vec: List[int]) -> str:
    """Format vector as [a,b,c]."""
    return "[" + ",".join(str(x) for x in vec) + "]"


def reverse(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Reverse the vector."""
    result = vec[::-1]
    trace.append(f"reverse:{_format_vec(result)}")
    return result


def sort_asc(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Sort ascending."""
    result = sorted(vec)
    trace.append(f"sort_asc:{_format_vec(result)}")
    return result


def sort_desc(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Sort descending."""
    result = sorted(vec, reverse=True)
    trace.append(f"sort_desc:{_format_vec(result)}")
    return result


def cumsum(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Cumulative sum with mod k."""
    result = []
    running = 0
    ops = []
    for i, x in enumerate(vec):
        if i == 0:
            result.append(x % k)
            ops.append(str(x % k))
        else:
            new_val = (running + x) % k
            ops.append(f"{running}+{x}={new_val}")
            result.append(new_val)
        running = result[-1]
    trace.append(f"cumsum:{','.join(ops)}:{_format_vec(result)}")
    return result
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_transformations.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/transformations.py tests/test_transformations.py
git commit -m "feat: add reverse, sort_asc, sort_desc, cumsum transformations"
```

---

## Task 3: Transformations 5-8 (cumsum_reverse, add_1, add_2, add_3)

**Files:**
- Modify: `src/transformations.py`
- Modify: `tests/test_transformations.py`

**Step 1: Write failing tests**

Append to `tests/test_transformations.py`:
```python
from src.transformations import cumsum_reverse, add_1, add_2, add_3

class TestCumsumReverse:
    def test_basic(self):
        vec = [3, 1, 4]
        trace = []
        result = cumsum_reverse(vec, 10, trace)
        assert result == [8, 5, 4]  # 3+1+4=8, 1+4=5, 4

class TestAdd1:
    def test_basic(self):
        vec = [3, 1, 4]
        trace = []
        result = add_1(vec, 10, trace)
        assert result == [4, 2, 5]

    def test_modulo(self):
        vec = [9, 1, 4]
        trace = []
        result = add_1(vec, 10, trace)
        assert result == [0, 2, 5]

class TestAdd2:
    def test_basic(self):
        vec = [3, 1, 4]
        trace = []
        result = add_2(vec, 10, trace)
        assert result == [5, 3, 6]

class TestAdd3:
    def test_basic(self):
        vec = [3, 1, 4]
        trace = []
        result = add_3(vec, 10, trace)
        assert result == [6, 4, 7]
```

**Step 2: Run tests to verify new tests fail**

```bash
pytest tests/test_transformations.py -v
```
Expected: FAIL with "cannot import name 'cumsum_reverse'"

**Step 3: Implement the 4 transformations**

Append to `src/transformations.py`:
```python
def cumsum_reverse(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Cumulative sum from right with mod k."""
    reversed_vec = vec[::-1]
    result = []
    running = 0
    ops = []
    for i, x in enumerate(reversed_vec):
        if i == 0:
            result.append(x % k)
            ops.append(str(x % k))
        else:
            new_val = (running + x) % k
            ops.append(f"{running}+{x}={new_val}")
            result.append(new_val)
        running = result[-1]
    result = result[::-1]
    trace.append(f"cumsum_reverse:{','.join(ops)}:{_format_vec(result)}")
    return result


def add_1(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Add 1 to all elements mod k."""
    result = [(x + 1) % k for x in vec]
    ops = [f"{x}+1={r}" for x, r in zip(vec, result)]
    trace.append(f"add_1:{','.join(ops)}:{_format_vec(result)}")
    return result


def add_2(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Add 2 to all elements mod k."""
    result = [(x + 2) % k for x in vec]
    ops = [f"{x}+2={r}" for x, r in zip(vec, result)]
    trace.append(f"add_2:{','.join(ops)}:{_format_vec(result)}")
    return result


def add_3(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Add 3 to all elements mod k."""
    result = [(x + 3) % k for x in vec]
    ops = [f"{x}+3={r}" for x, r in zip(vec, result)]
    trace.append(f"add_3:{','.join(ops)}:{_format_vec(result)}")
    return result
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_transformations.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/transformations.py tests/test_transformations.py
git commit -m "feat: add cumsum_reverse, add_1, add_2, add_3 transformations"
```

---

## Task 4: Transformations 9-12 (diff, swap_pairs, rotate_left, rotate_right)

**Files:**
- Modify: `src/transformations.py`
- Modify: `tests/test_transformations.py`

**Step 1: Write failing tests**

Append to `tests/test_transformations.py`:
```python
from src.transformations import diff, swap_pairs, rotate_left, rotate_right

class TestDiff:
    def test_basic(self):
        vec = [3, 1, 4]
        trace = []
        result = diff(vec, 10, trace)
        assert result == [3, 8, 3]  # 3, 1-3=-2 mod 10=8, 4-1=3

    def test_modulo(self):
        vec = [1, 5, 2]
        trace = []
        result = diff(vec, 10, trace)
        assert result == [1, 4, 7]  # 1, 5-1=4, 2-5=-3 mod 10=7

class TestSwapPairs:
    def test_even_length(self):
        vec = [3, 1, 4, 2]
        trace = []
        result = swap_pairs(vec, 10, trace)
        assert result == [1, 3, 2, 4]

    def test_odd_length(self):
        vec = [3, 1, 4]
        trace = []
        result = swap_pairs(vec, 10, trace)
        assert result == [1, 3, 4]  # last element stays

class TestRotateLeft:
    def test_basic(self):
        vec = [3, 1, 4]
        trace = []
        result = rotate_left(vec, 10, trace)
        assert result == [1, 4, 3]

class TestRotateRight:
    def test_basic(self):
        vec = [3, 1, 4]
        trace = []
        result = rotate_right(vec, 10, trace)
        assert result == [4, 3, 1]
```

**Step 2: Run tests to verify new tests fail**

```bash
pytest tests/test_transformations.py -v
```
Expected: FAIL with "cannot import name 'diff'"

**Step 3: Implement the 4 transformations**

Append to `src/transformations.py`:
```python
def diff(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Pairwise differences. First element unchanged, rest are vec[i] - vec[i-1] mod k."""
    result = [vec[0]]
    ops = [str(vec[0])]
    for i in range(1, len(vec)):
        d = (vec[i] - vec[i - 1]) % k
        ops.append(f"{vec[i]}-{vec[i-1]}={d}")
        result.append(d)
    trace.append(f"diff:{','.join(ops)}:{_format_vec(result)}")
    return result


def swap_pairs(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Swap adjacent pairs. Odd-length vectors keep last element in place."""
    result = list(vec)
    swaps = []
    for i in range(0, len(vec) - 1, 2):
        result[i], result[i + 1] = result[i + 1], result[i]
        swaps.append(f"({vec[i]},{vec[i+1]})->({result[i]},{result[i+1]})")
    if len(vec) % 2 == 1:
        swaps.append(f"{vec[-1]}->stays")
    trace.append(f"swap_pairs:{','.join(swaps)}:{_format_vec(result)}")
    return result


def rotate_left(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Rotate left by 1."""
    result = vec[1:] + vec[:1]
    trace.append(f"rotate_left:{_format_vec(result)}")
    return result


def rotate_right(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Rotate right by 1."""
    result = vec[-1:] + vec[:-1]
    trace.append(f"rotate_right:{_format_vec(result)}")
    return result
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_transformations.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/transformations.py tests/test_transformations.py
git commit -m "feat: add diff, swap_pairs, rotate_left, rotate_right transformations"
```

---

## Task 5: Transformations 13-16 (negate, double, square, min_prefix)

**Files:**
- Modify: `src/transformations.py`
- Modify: `tests/test_transformations.py`

**Step 1: Write failing tests**

Append to `tests/test_transformations.py`:
```python
from src.transformations import negate, double, square, min_prefix

class TestNegate:
    def test_basic(self):
        vec = [3, 1, 4]
        trace = []
        result = negate(vec, 10, trace)
        assert result == [7, 9, 6]  # k-x: 10-3=7, 10-1=9, 10-4=6

class TestDouble:
    def test_basic(self):
        vec = [3, 1, 4]
        trace = []
        result = double(vec, 10, trace)
        assert result == [6, 2, 8]

    def test_modulo(self):
        vec = [6, 1, 4]
        trace = []
        result = double(vec, 10, trace)
        assert result == [2, 2, 8]  # 12 mod 10 = 2

class TestSquare:
    def test_basic(self):
        vec = [3, 1, 4]
        trace = []
        result = square(vec, 10, trace)
        assert result == [9, 1, 6]  # 9, 1, 16 mod 10 = 6

class TestMinPrefix:
    def test_basic(self):
        vec = [3, 1, 4]
        trace = []
        result = min_prefix(vec, 10, trace)
        assert result == [3, 1, 1]  # running min: 3, min(3,1)=1, min(1,4)=1

    def test_already_increasing(self):
        vec = [1, 2, 3]
        trace = []
        result = min_prefix(vec, 10, trace)
        assert result == [1, 1, 1]
```

**Step 2: Run tests to verify new tests fail**

```bash
pytest tests/test_transformations.py -v
```
Expected: FAIL with "cannot import name 'negate'"

**Step 3: Implement the 4 transformations**

Append to `src/transformations.py`:
```python
def negate(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Negate: x -> k - x."""
    result = [(k - x) % k for x in vec]
    ops = [f"{k}-{x}={r}" for x, r in zip(vec, result)]
    trace.append(f"negate:{','.join(ops)}:{_format_vec(result)}")
    return result


def double(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Double all elements mod k."""
    result = [(2 * x) % k for x in vec]
    ops = [f"2*{x}={r}" for x, r in zip(vec, result)]
    trace.append(f"double:{','.join(ops)}:{_format_vec(result)}")
    return result


def square(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Square all elements mod k."""
    result = [(x * x) % k for x in vec]
    ops = [f"{x}^2={r}" for x, r in zip(vec, result)]
    trace.append(f"square:{','.join(ops)}:{_format_vec(result)}")
    return result


def min_prefix(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Running minimum."""
    result = []
    running_min = float('inf')
    ops = []
    for x in vec:
        running_min = min(running_min, x)
        result.append(running_min)
        ops.append(f"min={running_min}")
    trace.append(f"min_prefix:{','.join(ops)}:{_format_vec(result)}")
    return result
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_transformations.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/transformations.py tests/test_transformations.py
git commit -m "feat: add negate, double, square, min_prefix transformations"
```

---

## Task 6: Transformation Registry

**Files:**
- Modify: `src/transformations.py`
- Modify: `tests/test_transformations.py`

**Step 1: Write failing test for registry**

Append to `tests/test_transformations.py`:
```python
from src.transformations import TRANSFORMATIONS, get_transformation

class TestRegistry:
    def test_has_16_transformations(self):
        assert len(TRANSFORMATIONS) == 16

    def test_get_by_name(self):
        fn = get_transformation("reverse")
        assert fn is not None
        vec = [1, 2, 3]
        trace = []
        result = fn(vec, 10, trace)
        assert result == [3, 2, 1]

    def test_get_by_index(self):
        fn = get_transformation(0)
        assert fn is not None

    def test_all_names(self):
        expected = [
            "reverse", "sort_asc", "sort_desc", "cumsum",
            "cumsum_reverse", "add_1", "add_2", "add_3",
            "diff", "swap_pairs", "rotate_left", "rotate_right",
            "negate", "double", "square", "min_prefix"
        ]
        assert list(TRANSFORMATIONS.keys()) == expected
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_transformations.py::TestRegistry -v
```
Expected: FAIL with "cannot import name 'TRANSFORMATIONS'"

**Step 3: Add registry to transformations.py**

Append to `src/transformations.py`:
```python
from typing import Callable, Union

TransformFn = Callable[[List[int], int, List[str]], List[int]]

TRANSFORMATIONS: dict[str, TransformFn] = {
    "reverse": reverse,
    "sort_asc": sort_asc,
    "sort_desc": sort_desc,
    "cumsum": cumsum,
    "cumsum_reverse": cumsum_reverse,
    "add_1": add_1,
    "add_2": add_2,
    "add_3": add_3,
    "diff": diff,
    "swap_pairs": swap_pairs,
    "rotate_left": rotate_left,
    "rotate_right": rotate_right,
    "negate": negate,
    "double": double,
    "square": square,
    "min_prefix": min_prefix,
}


def get_transformation(key: Union[str, int]) -> TransformFn:
    """Get transformation by name or index."""
    if isinstance(key, int):
        name = list(TRANSFORMATIONS.keys())[key]
        return TRANSFORMATIONS[name]
    return TRANSFORMATIONS[key]
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_transformations.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/transformations.py tests/test_transformations.py
git commit -m "feat: add transformation registry with name/index lookup"
```

---

## Task 7: Graph Module

**Files:**
- Create: `src/graph.py`
- Create: `tests/test_graph.py`

**Step 1: Write failing tests**

`tests/test_graph.py`:
```python
import pytest
from src.graph import TransformationGraph

class TestGraphConstruction:
    def test_create_graph(self):
        g = TransformationGraph(d=4, m=4)
        assert g.d == 4
        assert g.m == 4

    def test_vertex_count(self):
        g = TransformationGraph(d=4, m=4)
        assert g.num_vertices == 16

    def test_validation_fails_if_too_many_vertices(self):
        with pytest.raises(ValueError, match="exceeds"):
            TransformationGraph(d=5, m=4)  # 20 > 16

class TestPathEnumeration:
    def test_enumerate_paths_length_1(self):
        g = TransformationGraph(d=2, m=2)
        paths = g.enumerate_paths(length=1)
        # Length 1: can start at layer 0 or 1, each has 2 vertices
        # Layer 0: [0], [1]
        # Layer 1: [2], [3]
        assert len(paths) == 4

    def test_enumerate_paths_length_2(self):
        g = TransformationGraph(d=2, m=2)
        paths = g.enumerate_paths(length=2)
        # Length 2: must start at layer 0, go to layer 1
        # 2*2 = 4 paths: (0,2), (0,3), (1,2), (1,3)
        assert len(paths) == 4

    def test_enumerate_all_paths(self):
        g = TransformationGraph(d=2, m=2)
        all_paths = g.enumerate_all_paths(min_len=1, max_len=2)
        # Length 1: 4 paths, Length 2: 4 paths
        assert len(all_paths) == 8

class TestVertexMapping:
    def test_default_mapping(self):
        g = TransformationGraph(d=2, m=2)
        # Default: vertex 0 -> "reverse", vertex 1 -> "sort_asc", etc.
        assert g.get_transformation_name(0) == "reverse"
        assert g.get_transformation_name(1) == "sort_asc"

    def test_path_to_transformations(self):
        g = TransformationGraph(d=2, m=2)
        path = (0, 2)  # vertex 0 in layer 0, vertex 2 in layer 1
        names = g.path_to_transformation_names(path)
        assert names == ["reverse", "sort_desc"]
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_graph.py -v
```
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement graph module**

`src/graph.py`:
```python
"""Transformation graph for path enumeration."""

from itertools import product
from typing import List, Tuple

from src.transformations import TRANSFORMATIONS


class TransformationGraph:
    """A layered directed graph where vertices represent transformations.

    Structure: d layers, m vertices per layer.
    Edges: fully connected between adjacent layers.
    """

    def __init__(self, d: int, m: int, vertex_mapping: List[str] = None):
        """Create transformation graph.

        Args:
            d: Number of layers (depth)
            m: Vertices per layer
            vertex_mapping: Optional list of transformation names for vertices.
                           If None, uses default order from TRANSFORMATIONS.
        """
        if d * m > 16:
            raise ValueError(f"d * m = {d * m} exceeds 16 available transformations")

        self.d = d
        self.m = m

        if vertex_mapping is None:
            self.vertex_mapping = list(TRANSFORMATIONS.keys())[:d * m]
        else:
            self.vertex_mapping = vertex_mapping

    @property
    def num_vertices(self) -> int:
        return self.d * self.m

    def get_transformation_name(self, vertex_id: int) -> str:
        """Get transformation name for a vertex."""
        return self.vertex_mapping[vertex_id]

    def enumerate_paths(self, length: int) -> List[Tuple[int, ...]]:
        """Enumerate all paths of given length.

        A path of length n uses n consecutive layers.
        """
        paths = []
        # Starting layer can be 0 to d-length
        for start_layer in range(self.d - length + 1):
            # Get vertex ranges for each layer in the path
            layer_vertices = []
            for layer in range(start_layer, start_layer + length):
                start_vertex = layer * self.m
                layer_vertices.append(range(start_vertex, start_vertex + self.m))

            # Generate all combinations
            for combo in product(*layer_vertices):
                paths.append(combo)

        return paths

    def enumerate_all_paths(self, min_len: int, max_len: int) -> List[Tuple[int, ...]]:
        """Enumerate all paths with length in [min_len, max_len]."""
        all_paths = []
        for length in range(min_len, max_len + 1):
            all_paths.extend(self.enumerate_paths(length))
        return all_paths

    def path_to_transformation_names(self, path: Tuple[int, ...]) -> List[str]:
        """Convert path of vertex IDs to transformation names."""
        return [self.get_transformation_name(v) for v in path]
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_graph.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/graph.py tests/test_graph.py
git commit -m "feat: add graph module for path enumeration"
```

---

## Task 8: Split Module

**Files:**
- Create: `src/split.py`
- Create: `tests/test_split.py`

**Step 1: Write failing tests**

`tests/test_split.py`:
```python
import pytest
from src.split import train_val_split

class TestSplit:
    def test_split_returns_disjoint_sets(self):
        paths = [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,)]
        train, val = train_val_split(paths, train_ratio=0.8, seed=42)

        train_set = set(train)
        val_set = set(val)

        assert train_set.isdisjoint(val_set)
        assert train_set.union(val_set) == set(paths)

    def test_split_ratio(self):
        paths = [(i,) for i in range(100)]
        train, val = train_val_split(paths, train_ratio=0.8, seed=42)

        assert len(train) == 80
        assert len(val) == 20

    def test_reproducible_with_seed(self):
        paths = [(i,) for i in range(100)]
        train1, val1 = train_val_split(paths, train_ratio=0.8, seed=42)
        train2, val2 = train_val_split(paths, train_ratio=0.8, seed=42)

        assert train1 == train2
        assert val1 == val2

    def test_different_seeds_give_different_splits(self):
        paths = [(i,) for i in range(100)]
        train1, _ = train_val_split(paths, train_ratio=0.8, seed=42)
        train2, _ = train_val_split(paths, train_ratio=0.8, seed=123)

        assert train1 != train2
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_split.py -v
```
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement split module**

`src/split.py`:
```python
"""Train/validation split logic for transformation paths."""

import random
from typing import List, Tuple


def train_val_split(
    paths: List[Tuple[int, ...]],
    train_ratio: float,
    seed: int
) -> Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]]]:
    """Split paths into train and validation sets.

    Args:
        paths: List of all paths to split
        train_ratio: Fraction for training (e.g., 0.8)
        seed: Random seed for reproducibility

    Returns:
        (train_paths, val_paths) tuple of disjoint path lists
    """
    rng = random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train_paths = shuffled[:split_idx]
    val_paths = shuffled[split_idx:]

    return train_paths, val_paths
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_split.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/split.py tests/test_split.py
git commit -m "feat: add train/val split module"
```

---

## Task 9: Generator Module

**Files:**
- Create: `src/generator.py`
- Create: `tests/test_generator.py`

**Step 1: Write failing tests**

`tests/test_generator.py`:
```python
import pytest
from src.generator import generate_example, generate_vector

class TestVectorGeneration:
    def test_vector_length(self):
        vec = generate_vector(length=8, k=10, seed=42)
        assert len(vec) == 8

    def test_values_in_range(self):
        vec = generate_vector(length=100, k=10, seed=42)
        assert all(0 <= v < 10 for v in vec)

    def test_reproducible(self):
        vec1 = generate_vector(length=8, k=10, seed=42)
        vec2 = generate_vector(length=8, k=10, seed=42)
        assert vec1 == vec2

class TestExampleGeneration:
    def test_generates_example(self):
        example = generate_example(
            path=("reverse", "sort_asc"),
            vector_length=4,
            k=10,
            seed=42
        )
        assert "input" in example
        assert "output" in example
        assert "trace" in example

    def test_input_length_preserved(self):
        example = generate_example(
            path=("reverse",),
            vector_length=5,
            k=10,
            seed=42
        )
        assert len(example["input"]) == 5
        assert len(example["output"]) == 5

    def test_trace_has_correct_length(self):
        example = generate_example(
            path=("reverse", "sort_asc", "add_1"),
            vector_length=4,
            k=10,
            seed=42
        )
        assert len(example["trace"]) == 3

    def test_format_example(self):
        example = generate_example(
            path=("reverse",),
            vector_length=3,
            k=10,
            seed=42
        )
        formatted = example["formatted"]
        # Format: "INPUT OUTPUT TRACE"
        parts = formatted.split()
        # Should have: 3 input + 3 output + 1 trace = 7 parts minimum
        assert len(parts) >= 7
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_generator.py -v
```
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement generator module**

`src/generator.py`:
```python
"""Example generation for BranchingTokens."""

import random
from typing import Dict, List, Tuple

from src.transformations import get_transformation


def generate_vector(length: int, k: int, seed: int) -> List[int]:
    """Generate a random vector.

    Args:
        length: Vector length
        k: Modulo value (values in [0, k-1])
        seed: Random seed

    Returns:
        List of random integers in [0, k-1]
    """
    rng = random.Random(seed)
    return [rng.randint(0, k - 1) for _ in range(length)]


def generate_example(
    path: Tuple[str, ...],
    vector_length: int,
    k: int,
    seed: int
) -> Dict:
    """Generate a single training/validation example.

    Args:
        path: Tuple of transformation names to apply
        vector_length: Length of input vector
        k: Modulo value
        seed: Random seed for vector generation

    Returns:
        Dict with keys: input, output, trace, formatted
    """
    input_vec = generate_vector(vector_length, k, seed)

    current = list(input_vec)
    trace = []

    for transform_name in path:
        fn = get_transformation(transform_name)
        current = fn(current, k, trace)

    output_vec = current

    # Format: "INPUT OUTPUT TRACE"
    input_str = " ".join(str(x) for x in input_vec)
    output_str = " ".join(str(x) for x in output_vec)
    trace_str = " ".join(trace)

    formatted = f"{input_str} {output_str} {trace_str}"

    return {
        "input": input_vec,
        "output": output_vec,
        "trace": trace,
        "formatted": formatted,
    }
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_generator.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/generator.py tests/test_generator.py
git commit -m "feat: add example generator module"
```

---

## Task 10: Parser Module

**Files:**
- Create: `src/parser.py`
- Create: `tests/test_parser.py`

**Step 1: Write failing tests**

`tests/test_parser.py`:
```python
import pytest
from src.parser import (
    parse_coarse,
    parse_medium,
    parse_fine,
    parse_example_line
)

class TestParseExampleLine:
    def test_parse_basic(self):
        line = "3 1 4 1 3 4 sort_asc:[1,3,4]"
        result = parse_example_line(line, input_len=4)
        assert result["input"] == [3, 1, 4, 1]
        assert result["output"] == [1, 3, 4]
        assert result["trace"] == "sort_asc:[1,3,4]"

class TestParseCoarse:
    def test_extracts_transformation_names(self):
        trace = "cumsum:3,3+1=4:[3,4] sort_asc:[3,4]"
        names = parse_coarse(trace)
        assert names == ["cumsum", "sort_asc"]

    def test_single_transformation(self):
        trace = "reverse:[4,1,3]"
        names = parse_coarse(trace)
        assert names == ["reverse"]

class TestParseMedium:
    def test_extracts_intermediate_vectors(self):
        trace = "cumsum:3,3+1=4:[3,4] sort_asc:[3,4]"
        vectors = parse_medium(trace)
        assert vectors == [[3, 4], [3, 4]]

    def test_handles_longer_vectors(self):
        trace = "reverse:[4,1,3,2]"
        vectors = parse_medium(trace)
        assert vectors == [[4, 1, 3, 2]]

class TestParseFine:
    def test_extracts_operations(self):
        trace = "cumsum:3,3+1=4,4+4=8:[3,4,8]"
        ops = parse_fine(trace)
        assert len(ops) == 1
        assert ops[0]["name"] == "cumsum"
        assert "3+1=4" in ops[0]["operations"]
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_parser.py -v
```
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement parser module**

`src/parser.py`:
```python
"""Trace parsing at different granularity levels."""

import re
from typing import Dict, List


def parse_example_line(line: str, input_len: int) -> Dict:
    """Parse a formatted example line.

    Format: "INPUT OUTPUT TRACE" where INPUT and OUTPUT are space-separated ints.

    Args:
        line: The formatted line
        input_len: Length of input/output vectors (needed to know where trace starts)

    Returns:
        Dict with input, output, trace keys
    """
    parts = line.strip().split()

    input_vec = [int(x) for x in parts[:input_len]]
    output_vec = [int(x) for x in parts[input_len:2*input_len]]
    trace = " ".join(parts[2*input_len:])

    return {
        "input": input_vec,
        "output": output_vec,
        "trace": trace,
    }


def parse_coarse(trace: str) -> List[str]:
    """Parse trace at coarse level: extract transformation names in order.

    Args:
        trace: The trace string (space-separated transformation traces)

    Returns:
        List of transformation names in order
    """
    # Each transformation trace starts with "name:"
    pattern = r'([a-z_0-9]+):'
    matches = re.findall(pattern, trace)
    return matches


def parse_medium(trace: str) -> List[List[int]]:
    """Parse trace at medium level: extract result vectors for each transformation.

    Args:
        trace: The trace string

    Returns:
        List of result vectors (one per transformation)
    """
    # Find all [...] patterns (result vectors)
    pattern = r'\[([0-9,]+)\]'
    matches = re.findall(pattern, trace)

    vectors = []
    for match in matches:
        vec = [int(x) for x in match.split(',')]
        vectors.append(vec)

    return vectors


def parse_fine(trace: str) -> List[Dict]:
    """Parse trace at fine level: extract operations for each transformation.

    Args:
        trace: The trace string

    Returns:
        List of dicts with name and operations for each transformation
    """
    # Split by transformation (each starts with name:)
    parts = trace.strip().split()

    results = []
    for part in parts:
        if ':' not in part:
            continue

        # Split into name and rest
        colon_idx = part.index(':')
        name = part[:colon_idx]
        rest = part[colon_idx + 1:]

        # Extract operations (everything before the final [...])
        bracket_idx = rest.rfind('[')
        if bracket_idx > 0:
            ops_str = rest[:bracket_idx].rstrip(':')
            operations = ops_str
        else:
            operations = ""

        results.append({
            "name": name,
            "operations": operations,
        })

    return results
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_parser.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/parser.py tests/test_parser.py
git commit -m "feat: add parser module for trace parsing"
```

---

## Task 11: Dataset Module

**Files:**
- Create: `src/dataset.py`
- Create: `tests/test_dataset.py`

**Step 1: Write failing tests**

`tests/test_dataset.py`:
```python
import pytest
from src.dataset import DatasetGenerator

class TestDatasetGenerator:
    def test_create_generator(self):
        gen = DatasetGenerator(
            seed=42,
            vector_length=4,
            k=10,
            d=2,
            m=2,
            min_len=1,
            max_len=2,
            train_ratio=0.8,
        )
        assert gen is not None

    def test_generate_train_examples(self):
        gen = DatasetGenerator(
            seed=42,
            vector_length=4,
            k=10,
            d=2,
            m=2,
            min_len=1,
            max_len=2,
            train_ratio=0.8,
        )
        examples = gen.generate_train(n=10)
        assert len(examples) == 10

    def test_generate_val_examples(self):
        gen = DatasetGenerator(
            seed=42,
            vector_length=4,
            k=10,
            d=2,
            m=2,
            min_len=1,
            max_len=2,
            train_ratio=0.8,
        )
        examples = gen.generate_val(n=5)
        assert len(examples) == 5

    def test_train_val_use_different_paths(self):
        gen = DatasetGenerator(
            seed=42,
            vector_length=4,
            k=10,
            d=2,
            m=2,
            min_len=1,
            max_len=2,
            train_ratio=0.5,  # 50/50 for easier testing
        )
        # This is tested via the split module, but we check here too
        assert len(gen.train_paths) > 0
        assert len(gen.val_paths) > 0
        assert set(gen.train_paths).isdisjoint(set(gen.val_paths))
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_dataset.py -v
```
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement dataset module**

`src/dataset.py`:
```python
"""Dataset generation orchestration."""

import random
from typing import List, Tuple

from src.graph import TransformationGraph
from src.split import train_val_split
from src.generator import generate_example


class DatasetGenerator:
    """Orchestrates dataset generation with train/val splits."""

    def __init__(
        self,
        seed: int,
        vector_length: int,
        k: int,
        d: int,
        m: int,
        min_len: int,
        max_len: int,
        train_ratio: float,
        vertex_mapping: List[str] = None,
    ):
        """Initialize dataset generator.

        Args:
            seed: Global random seed
            vector_length: Length of vectors
            k: Modulo value
            d: Graph depth (layers)
            m: Vertices per layer
            min_len: Minimum path length
            max_len: Maximum path length
            train_ratio: Fraction for training
            vertex_mapping: Optional custom vertex->transformation mapping
        """
        self.seed = seed
        self.vector_length = vector_length
        self.k = k
        self.rng = random.Random(seed)

        # Build graph and enumerate paths
        self.graph = TransformationGraph(d, m, vertex_mapping)
        all_paths = self.graph.enumerate_all_paths(min_len, max_len)

        # Split paths
        train_vertex_paths, val_vertex_paths = train_val_split(
            all_paths, train_ratio, seed
        )

        # Convert to transformation names
        self.train_paths = [
            tuple(self.graph.path_to_transformation_names(p))
            for p in train_vertex_paths
        ]
        self.val_paths = [
            tuple(self.graph.path_to_transformation_names(p))
            for p in val_vertex_paths
        ]

    def _generate_examples(
        self,
        paths: List[Tuple[str, ...]],
        n: int
    ) -> List[str]:
        """Generate n examples using given paths.

        Args:
            paths: Pool of allowed paths
            n: Number of examples to generate

        Returns:
            List of formatted example strings
        """
        examples = []
        for i in range(n):
            path = self.rng.choice(paths)
            example_seed = self.rng.randint(0, 2**31)
            example = generate_example(
                path=path,
                vector_length=self.vector_length,
                k=self.k,
                seed=example_seed,
            )
            examples.append(example["formatted"])
        return examples

    def generate_train(self, n: int) -> List[str]:
        """Generate n training examples."""
        return self._generate_examples(self.train_paths, n)

    def generate_val(self, n: int) -> List[str]:
        """Generate n validation examples."""
        return self._generate_examples(self.val_paths, n)
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_dataset.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/dataset.py tests/test_dataset.py
git commit -m "feat: add dataset module for orchestrating generation"
```

---

## Task 12: Hydra Entry Point

**Files:**
- Create: `scripts/generate.py`
- Modify: `config/config.yaml` (add defaults section if needed)

**Step 1: Write the Hydra script**

`scripts/generate.py`:
```python
#!/usr/bin/env python3
"""Generate training and validation datasets using Hydra config."""

import os
import hydra
from omegaconf import DictConfig, OmegaConf

from src.dataset import DatasetGenerator


def validate_config(cfg: DictConfig) -> None:
    """Validate configuration parameters."""
    d = cfg.graph.d
    m = cfg.graph.m

    if d * m > 16:
        raise ValueError(f"graph.d * graph.m = {d * m} exceeds 16 transformations")

    if cfg.path.max_len > d:
        raise ValueError(f"path.max_len ({cfg.path.max_len}) > graph.d ({d})")

    if cfg.path.min_len < 1:
        raise ValueError("path.min_len must be >= 1")

    if not 0 < cfg.split.train_ratio < 1:
        raise ValueError("split.train_ratio must be between 0 and 1")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Generate datasets based on config."""
    print(OmegaConf.to_yaml(cfg))

    validate_config(cfg)

    # Get vertex mapping if specified
    vertex_mapping = None
    if "vertex_to_transform" in cfg.graph and cfg.graph.vertex_to_transform:
        vertex_mapping = list(cfg.graph.vertex_to_transform)

    generator = DatasetGenerator(
        seed=cfg.seed,
        vector_length=cfg.vector.length,
        k=cfg.vector.k,
        d=cfg.graph.d,
        m=cfg.graph.m,
        min_len=cfg.path.min_len,
        max_len=cfg.path.max_len,
        train_ratio=cfg.split.train_ratio,
        vertex_mapping=vertex_mapping,
    )

    print(f"Generating {cfg.dataset.train_size} training examples...")
    train_examples = generator.generate_train(cfg.dataset.train_size)

    print(f"Generating {cfg.dataset.val_size} validation examples...")
    val_examples = generator.generate_val(cfg.dataset.val_size)

    # Write to files (Hydra changes cwd to outputs/<date>/<time>)
    os.makedirs(".", exist_ok=True)

    with open("train.txt", "w") as f:
        for example in train_examples:
            f.write(example + "\n")

    with open("val.txt", "w") as f:
        for example in val_examples:
            f.write(example + "\n")

    print(f"Written train.txt ({len(train_examples)} examples)")
    print(f"Written val.txt ({len(val_examples)} examples)")
    print(f"Output directory: {os.getcwd()}")


if __name__ == "__main__":
    main()
```

**Step 2: Make script executable and test manually**

```bash
chmod +x scripts/generate.py
cd /path/to/worktree
python scripts/generate.py --help
```
Expected: Hydra help output with config options

**Step 3: Run with small dataset to verify**

```bash
python scripts/generate.py dataset.train_size=10 dataset.val_size=5
```
Expected: Creates outputs/ subdirectory with train.txt and val.txt

**Step 4: Commit**

```bash
git add scripts/generate.py
git commit -m "feat: add Hydra entry point for dataset generation"
```

---

## Task 13: Integration Test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

`tests/test_integration.py`:
```python
"""Integration test for full pipeline."""

import pytest
from src.dataset import DatasetGenerator
from src.parser import parse_example_line, parse_coarse


class TestFullPipeline:
    def test_end_to_end(self):
        """Test full generation and parsing pipeline."""
        gen = DatasetGenerator(
            seed=42,
            vector_length=4,
            k=10,
            d=4,
            m=4,
            min_len=1,
            max_len=3,
            train_ratio=0.8,
        )

        # Generate examples
        train = gen.generate_train(100)
        val = gen.generate_val(20)

        assert len(train) == 100
        assert len(val) == 20

        # Parse and verify each example
        for line in train[:10]:  # Check first 10
            parsed = parse_example_line(line, input_len=4)

            # Input and output should have correct length
            assert len(parsed["input"]) == 4
            assert len(parsed["output"]) == 4

            # Values should be in range [0, k-1]
            assert all(0 <= v < 10 for v in parsed["input"])
            assert all(0 <= v < 10 for v in parsed["output"])

            # Trace should parse correctly
            names = parse_coarse(parsed["trace"])
            assert len(names) >= 1
            assert len(names) <= 3

    def test_train_val_paths_disjoint(self):
        """Verify train and val use different transformation sequences."""
        gen = DatasetGenerator(
            seed=42,
            vector_length=4,
            k=10,
            d=2,
            m=2,
            min_len=1,
            max_len=2,
            train_ratio=0.5,
        )

        train_paths_set = set(gen.train_paths)
        val_paths_set = set(gen.val_paths)

        assert train_paths_set.isdisjoint(val_paths_set)
```

**Step 2: Run integration test**

```bash
pytest tests/test_integration.py -v
```
Expected: PASS

**Step 3: Run all tests**

```bash
pytest tests/ -v
```
Expected: All tests PASS

**Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration test for full pipeline"
```

---

## Task 14: Final Cleanup

**Files:**
- Create/update: `README.md` (optional, only if requested)

**Step 1: Run full test suite**

```bash
pytest tests/ -v --tb=short
```
Expected: All tests pass

**Step 2: Test the CLI**

```bash
python scripts/generate.py dataset.train_size=100 dataset.val_size=20
cat outputs/*/train.txt | head -5
```
Expected: Valid example lines

**Step 3: Final commit if any cleanup needed**

```bash
git status
# If clean, skip. Otherwise:
git add -A
git commit -m "chore: final cleanup"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Project setup | requirements.txt, config/, src/, tests/ |
| 2 | Transformations 1-4 | src/transformations.py |
| 3 | Transformations 5-8 | src/transformations.py |
| 4 | Transformations 9-12 | src/transformations.py |
| 5 | Transformations 13-16 | src/transformations.py |
| 6 | Transformation registry | src/transformations.py |
| 7 | Graph module | src/graph.py |
| 8 | Split module | src/split.py |
| 9 | Generator module | src/generator.py |
| 10 | Parser module | src/parser.py |
| 11 | Dataset module | src/dataset.py |
| 12 | Hydra entry point | scripts/generate.py |
| 13 | Integration test | tests/test_integration.py |
| 14 | Final cleanup | - |

**Total: 14 tasks, ~50 commits expected**
