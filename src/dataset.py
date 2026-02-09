"""Dataset generation orchestration."""

import random
from itertools import product as cartesian_product
from typing import List, Tuple

from src.graph import TransformationGraph
from src.split import train_val_split
from src.transformations import TRANSFORMATIONS
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
        all_train_lengths: List[int] = None,
        mode: str = "graph",
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
            all_train_lengths: Path lengths where ALL paths go to train (no split)
            mode: "graph" for graph-constrained paths, "combinations" for all function combos
        """
        self.seed = seed
        self.vector_length = vector_length
        self.k = k
        self.rng = random.Random(seed)

        if mode == "combinations":
            self._init_combinations(d, m, min_len, max_len, train_ratio,
                                    seed, vertex_mapping, all_train_lengths)
        else:
            self._init_graph(d, m, min_len, max_len, train_ratio,
                             seed, vertex_mapping, all_train_lengths)

    def _init_graph(self, d, m, min_len, max_len, train_ratio,
                    seed, vertex_mapping, all_train_lengths):
        """Original graph-based path enumeration."""
        self.graph = TransformationGraph(d, m, vertex_mapping)

        if all_train_lengths:
            all_train_set = set(all_train_lengths)
            train_vertex_paths = []
            val_vertex_paths = []

            for length in range(min_len, max_len + 1):
                length_paths = self.graph.enumerate_paths(length)
                if length in all_train_set:
                    train_vertex_paths.extend(length_paths)
                else:
                    t, v = train_val_split(length_paths, train_ratio, seed)
                    train_vertex_paths.extend(t)
                    val_vertex_paths.extend(v)
        else:
            all_paths = self.graph.enumerate_all_paths(min_len, max_len)
            train_vertex_paths, val_vertex_paths = train_val_split(
                all_paths, train_ratio, seed
            )

        self.train_paths = [
            tuple(self.graph.path_to_transformation_names(p))
            for p in train_vertex_paths
        ]
        self.val_paths = [
            tuple(self.graph.path_to_transformation_names(p))
            for p in val_vertex_paths
        ]

    def _init_combinations(self, d, m, min_len, max_len, train_ratio,
                           seed, vertex_mapping, all_train_lengths):
        """All function combinations (no graph structure constraint)."""
        if vertex_mapping:
            transforms = list(vertex_mapping)
        else:
            transforms = list(TRANSFORMATIONS.keys())[:d * m]

        all_train_set = set(all_train_lengths) if all_train_lengths else set()
        train_paths = []
        val_paths = []

        for length in range(min_len, max_len + 1):
            combos = [tuple(c) for c in cartesian_product(transforms, repeat=length)]
            if length in all_train_set:
                train_paths.extend(combos)
            else:
                rng_split = random.Random(seed)
                shuffled = list(combos)
                rng_split.shuffle(shuffled)
                split_idx = int(len(shuffled) * train_ratio)
                train_paths.extend(shuffled[:split_idx])
                val_paths.extend(shuffled[split_idx:])

        self.train_paths = train_paths
        self.val_paths = val_paths

        print(f"Combinations mode: {len(transforms)} functions, "
              f"lengths {min_len}-{max_len}")
        for length in range(min_len, max_len + 1):
            total = len(transforms) ** length
            in_train = sum(1 for p in self.train_paths if len(p) == length)
            in_val = sum(1 for p in self.val_paths if len(p) == length)
            print(f"  Length {length}: {total} total combos -> "
                  f"{in_train} train, {in_val} val")

    def _generate_examples(
        self,
        paths: List[Tuple[str, ...]],
        n: int
    ) -> List[dict]:
        """Generate n examples using given paths.

        Args:
            paths: Pool of allowed paths
            n: Number of examples to generate

        Returns:
            List of example dictionaries with 'input' and 'output' keys
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
            # Return dict with input and output for JSON format
            examples.append({
                "input": example["input"],
                "output": example["output"]
            })
        return examples

    def _generate_balanced_examples(
        self,
        paths: List[Tuple[str, ...]],
        n: int,
        per_length: dict = None,
    ) -> List[dict]:
        """Generate n examples with balanced sequence lengths.

        Args:
            paths: Pool of allowed paths
            n: Total number of examples to generate
            per_length: Optional dict mapping path length -> exact count.
                        Unspecified lengths split the remainder equally.

        Returns:
            List of example dictionaries with balanced sequence lengths
        """
        # Group paths by length
        paths_by_length = {}
        for path in paths:
            length = len(path)
            if length not in paths_by_length:
                paths_by_length[length] = []
            paths_by_length[length].append(path)

        if not paths_by_length:
            raise ValueError("No paths available for generation")

        lengths = sorted(paths_by_length.keys())

        # Determine how many examples per length
        counts = {}
        if per_length:
            specified_total = 0
            for length in lengths:
                if length in per_length:
                    counts[length] = per_length[length]
                    specified_total += per_length[length]

            # Distribute remainder equally among unspecified lengths
            remaining = n - specified_total
            unspecified = [l for l in lengths if l not in counts]
            if unspecified:
                per_unspecified = remaining // len(unspecified)
                extra = remaining % len(unspecified)
                for idx, length in enumerate(unspecified):
                    counts[length] = per_unspecified + (1 if idx < extra else 0)
            elif remaining != 0:
                # All lengths specified but don't sum to n - adjust last one
                counts[lengths[-1]] += remaining
        else:
            # Equal split
            examples_per_length = n // len(lengths)
            remainder = n % len(lengths)
            for idx, length in enumerate(lengths):
                counts[length] = examples_per_length + (1 if idx < remainder else 0)

        print(f"Generating balanced dataset with {len(lengths)} sequence lengths:")
        for length in lengths:
            print(f"  Length {length}: {counts[length]} examples ({len(paths_by_length[length])} available paths)")

        # Generate examples for each length
        examples = []
        for length in lengths:
            num_examples = counts[length]
            length_paths = paths_by_length[length]

            print(f"Generating {num_examples} examples with {length} functions...")

            for _ in range(num_examples):
                path = self.rng.choice(length_paths)
                example_seed = self.rng.randint(0, 2**31)
                example = generate_example(
                    path=path,
                    vector_length=self.vector_length,
                    k=self.k,
                    seed=example_seed,
                )
                examples.append({
                    "input": example["input"],
                    "output": example["output"]
                })

        # Shuffle to mix lengths
        self.rng.shuffle(examples)

        return examples

    def generate_train(self, n: int, balanced: bool = False, per_length: dict = None) -> List[dict]:
        """Generate n training examples.

        Args:
            n: Number of examples to generate
            balanced: If True, balance sequence lengths equally
            per_length: Optional dict mapping path length -> exact count

        Returns:
            List of example dictionaries
        """
        if balanced:
            return self._generate_balanced_examples(self.train_paths, n, per_length=per_length)
        return self._generate_examples(self.train_paths, n)

    def generate_val(self, n: int, balanced: bool = False, per_length: dict = None) -> List[dict]:
        """Generate n validation examples.

        Args:
            n: Number of examples to generate
            balanced: If True, balance sequence lengths equally
            per_length: Optional dict mapping path length -> exact count

        Returns:
            List of example dictionaries
        """
        if balanced:
            return self._generate_balanced_examples(self.val_paths, n, per_length=per_length)
        return self._generate_examples(self.val_paths, n)
