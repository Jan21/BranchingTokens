#!/usr/bin/env python3
"""Generate training and validation datasets using Hydra config."""

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

    if d * m > 16:
        raise ValueError(f"graph.d * graph.m = {d * m} exceeds 16 transformations")

    if cfg.path.max_len > d:
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
    )

    print(f"Generating {cfg.dataset.train_size} training examples...")
    train_examples = generator.generate_train(cfg.dataset.train_size)

    print(f"Generating {cfg.dataset.val_size} validation examples...")
    val_examples = generator.generate_val(cfg.dataset.val_size)

    # Write to files (Hydra changes cwd to outputs/<date>/<time>)
    os.makedirs(".", exist_ok=True)

    with open("train.txt", "w") as f:
        for example in train_examples:
            f.write(example + "\n")

    with open("val.txt", "w") as f:
        for example in val_examples:
            f.write(example + "\n")

    print(f"Written train.txt ({len(train_examples)} examples)")
    print(f"Written val.txt ({len(val_examples)} examples)")
    print(f"Output directory: {os.getcwd()}")


if __name__ == "__main__":
    main()
