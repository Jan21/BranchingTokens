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
