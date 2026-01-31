"""Train/validation split logic for transformation paths."""

import random
from typing import List, Tuple


def train_val_split(
    paths: List[Tuple[int, ...]],
    train_ratio: float,
    seed: int
) -> Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]]]:
    """Split paths into train and validation sets.

    Args:
        paths: List of all paths to split
        train_ratio: Fraction for training (e.g., 0.8)
        seed: Random seed for reproducibility

    Returns:
        (train_paths, val_paths) tuple of disjoint path lists
    """
    rng = random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train_paths = shuffled[:split_idx]
    val_paths = shuffled[split_idx:]

    return train_paths, val_paths
