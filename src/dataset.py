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
