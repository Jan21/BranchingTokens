#!/usr/bin/env python3
"""Generate training and validation datasets using Hydra config."""

import json
import os
import sys

# Add project root to path so 'src' module can be found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig, OmegaConf

from src.dataset import DatasetGenerator


def validate_config(cfg: DictConfig) -> None:
    """Validate configuration parameters."""
    d = cfg.graph.d
    m = cfg.graph.m
    mode = cfg.dataset.get("mode", "graph")

    if d * m > 16:
        raise ValueError(f"graph.d * graph.m = {d * m} exceeds 16 transformations")

    if mode != "combinations" and cfg.path.max_len > d:
        raise ValueError(f"path.max_len ({cfg.path.max_len}) > graph.d ({d})")

    if cfg.path.min_len < 1:
        raise ValueError("path.min_len must be >= 1")

    if not 0 < cfg.split.train_ratio < 1:
        raise ValueError("split.train_ratio must be between 0 and 1")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Generate datasets based on config."""
    print(OmegaConf.to_yaml(cfg))

    validate_config(cfg)

    # Get vertex mapping if specified
    vertex_mapping = None
    if "vertex_to_transform" in cfg.graph and cfg.graph.vertex_to_transform:
        vertex_mapping = list(cfg.graph.vertex_to_transform)

    # Parse all_train_lengths config
    all_train_lengths = None
    if "all_train_lengths" in cfg.split and cfg.split.all_train_lengths:
        all_train_lengths = list(cfg.split.all_train_lengths)

    # Dataset mode: "graph" (default) or "combinations"
    mode = cfg.dataset.get("mode", "graph")

    generator = DatasetGenerator(
        seed=cfg.seed,
        vector_length=cfg.vector.length,
        k=cfg.vector.k,
        d=cfg.graph.d,
        m=cfg.graph.m,
        min_len=cfg.path.min_len,
        max_len=cfg.path.max_len,
        train_ratio=cfg.split.train_ratio,
        vertex_mapping=vertex_mapping,
        all_train_lengths=all_train_lengths,
        mode=mode,
    )

    # Check if balanced generation is requested
    balanced = cfg.dataset.get("balanced", False)

    # Parse per_length config (maps path length -> count)
    per_length = None
    if "per_length" in cfg.dataset and cfg.dataset.per_length:
        per_length = {int(k): int(v) for k, v in cfg.dataset.per_length.items()}

    if balanced:
        if per_length:
            print("\n=== Using BALANCED generation with per-length counts ===\n")
            for length, count in sorted(per_length.items()):
                print(f"  Length {length}: {count} specified")
        else:
            print("\n=== Using BALANCED generation (equal proportions per sequence length) ===\n")
    else:
        print("\n=== Using RANDOM generation (natural distribution) ===\n")

    print(f"Generating {cfg.dataset.train_size} training examples...")
    train_examples = generator.generate_train(cfg.dataset.train_size, balanced=balanced, per_length=per_length)

    print(f"\nGenerating {cfg.dataset.val_size} validation examples...")
    val_examples = generator.generate_val(cfg.dataset.val_size, balanced=balanced)

    # Write to JSON files (Hydra changes cwd to outputs/<date>/<time>)
    os.makedirs(".", exist_ok=True)

    with open("train.json", "w") as f:
        json.dump(train_examples, f, indent=2)

    with open("val.json", "w") as f:
        json.dump(val_examples, f, indent=2)

    print(f"Written train.json ({len(train_examples)} examples)")
    print(f"Written val.json ({len(val_examples)} examples)")
    print(f"Output directory: {os.getcwd()}")


if __name__ == "__main__":
    main()
