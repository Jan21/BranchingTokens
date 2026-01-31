import pytest
from src.transformations import reverse, sort_asc, sort_desc, cumsum, cumsum_reverse, add_1, add_2, add_3
from src.transformations import diff, swap_pairs, rotate_left, rotate_right
from src.transformations import negate, double, square, min_prefix

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