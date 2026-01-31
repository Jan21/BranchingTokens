import pytest
from src.transformations import reverse, sort_asc, sort_desc, cumsum, cumsum_reverse, add_1, add_2, add_3

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
