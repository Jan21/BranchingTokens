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
