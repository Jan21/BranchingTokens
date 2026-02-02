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
        # Format: "INPUT: ... OUTPUT: ... TRACE: ... </s>"
        assert "INPUT:" in formatted
        assert "OUTPUT:" in formatted
        assert "TRACE:" in formatted
        assert formatted.endswith("</s>")
        # Check all parts are whitespace tokenizable
        parts = formatted.split()
        assert len(parts) >= 11  # INPUT: + 3 input + OUTPUT: + 3 output + TRACE: + trace tokens + </s>
