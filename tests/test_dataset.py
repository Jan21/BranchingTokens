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
