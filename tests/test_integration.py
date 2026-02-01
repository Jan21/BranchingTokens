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
