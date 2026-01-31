import pytest
from src.split import train_val_split

class TestSplit:
    def test_split_returns_disjoint_sets(self):
        paths = [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,)]
        train, val = train_val_split(paths, train_ratio=0.8, seed=42)

        train_set = set(train)
        val_set = set(val)

        assert train_set.isdisjoint(val_set)
        assert train_set.union(val_set) == set(paths)

    def test_split_ratio(self):
        paths = [(i,) for i in range(100)]
        train, val = train_val_split(paths, train_ratio=0.8, seed=42)

        assert len(train) == 80
        assert len(val) == 20

    def test_reproducible_with_seed(self):
        paths = [(i,) for i in range(100)]
        train1, val1 = train_val_split(paths, train_ratio=0.8, seed=42)
        train2, val2 = train_val_split(paths, train_ratio=0.8, seed=42)

        assert train1 == train2
        assert val1 == val2

    def test_different_seeds_give_different_splits(self):
        paths = [(i,) for i in range(100)]
        train1, _ = train_val_split(paths, train_ratio=0.8, seed=42)
        train2, _ = train_val_split(paths, train_ratio=0.8, seed=123)

        assert train1 != train2
