import pytest
from src.parser import (
    parse_coarse,
    parse_medium,
    parse_fine,
    parse_example_line
)

class TestParseExampleLine:
    def test_parse_basic(self):
        line = "ST : [ 3 1 4 1 ] ; BEGIN : sort_asc : [ 1 3 4 ] ; RESULT : [ 1 3 4 ]"
        result = parse_example_line(line, input_len=4)
        assert result["input"] == [3, 1, 4, 1]
        assert result["output"] == [1, 3, 4]
        assert result["trace"] == "sort_asc : [ 1 3 4 ]"

class TestParseCoarse:
    def test_extracts_transformation_names(self):
        trace = "cumsum : 3 , 3 + 1 = 4 : [ 3 4 ] sort_asc : [ 3 4 ]"
        names = parse_coarse(trace)
        assert names == ["cumsum", "sort_asc"]

    def test_single_transformation(self):
        trace = "reverse : [ 4 1 3 ]"
        names = parse_coarse(trace)
        assert names == ["reverse"]

class TestParseMedium:
    def test_extracts_intermediate_vectors(self):
        trace = "cumsum : 3 , 3 + 1 = 4 : [ 3 4 ] sort_asc : [ 3 4 ]"
        vectors = parse_medium(trace)
        assert vectors == [[3, 4], [3, 4]]

    def test_handles_longer_vectors(self):
        trace = "reverse : [ 4 1 3 2 ]"
        vectors = parse_medium(trace)
        assert vectors == [[4, 1, 3, 2]]

class TestParseFine:
    def test_extracts_operations(self):
        trace = "cumsum : 3 , 3 + 1 = 4 , 4 + 4 = 8 : [ 3 4 8 ]"
        ops = parse_fine(trace)
        assert len(ops) == 1
        assert ops[0]["name"] == "cumsum"
        assert "3 + 1 = 4" in ops[0]["operations"]
