#!/usr/bin/env python3
"""Generate experimental datasets using Hydra config.

Supports multiple experiment modes:
1. exhaustive: Generate ALL possible (input, operation) pairs
2. mixed_train_pure_test: Train with mixed lengths, test with single length
3. mixed_exhaustive: Exhaustive 1-op + sampled 2-op (no input/path splitting)
"""

import json
import os
import random
import sys
from itertools import product
from typing import List, Tuple, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig, OmegaConf

from src.generator import generate_example
from src.graph import TransformationGraph
from src.split import train_val_split


def enumerate_all_inputs(vector_length: int, k: int) -> List[Tuple[int, ...]]:
    """Enumerate all possible input vectors."""
    return list(product(range(k), repeat=vector_length))


def generate_exhaustive(cfg: DictConfig) -> Tuple[List[dict], List[dict]]:
    """Experiment 1: Generate ALL possible (input, operation) pairs.

    Returns:
        (train_examples, val_examples)
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: EXHAUSTIVE SINGLE OPERATIONS")
    print("=" * 80)

    rng = random.Random(cfg.seed)

    # Build graph and get all single-operation paths
    graph = TransformationGraph(
        d=cfg.graph.d,
        m=cfg.graph.m,
        vertex_mapping=None
    )
    all_paths = graph.enumerate_all_paths(
        min_len=cfg.path.min_len,
        max_len=cfg.path.max_len
    )
    operations = [tuple(graph.path_to_transformation_names(p)) for p in all_paths]

    print(f"\nOperations: {len(operations)}")
    for op in operations:
        print(f"  - {op[0]}")

    # Generate all possible inputs
    all_inputs = enumerate_all_inputs(cfg.vector.length, cfg.vector.k)
    print(f"\nAll possible inputs: {len(all_inputs)}")

    # Generate all (input, operation) pairs
    all_pairs = []
    for input_vec in all_inputs:
        for operation in operations:
            all_pairs.append((input_vec, operation))

    total = len(all_pairs)
    print(f"Total (input, operation) pairs: {total}")
    print(f"  = {len(all_inputs)} inputs × {len(operations)} operations")

    # Verify sizes
    requested_total = cfg.dataset.train_size + cfg.dataset.val_size
    if requested_total > total:
        raise ValueError(
            f"Requested {requested_total} examples but only {total} possible pairs exist"
        )

    # Shuffle and split
    rng.shuffle(all_pairs)
    train_pairs = all_pairs[:cfg.dataset.train_size]
    val_pairs = all_pairs[cfg.dataset.train_size:cfg.dataset.train_size + cfg.dataset.val_size]

    print(f"\nSplit:")
    print(f"  Train: {len(train_pairs)} ({100*len(train_pairs)/total:.1f}%)")
    print(f"  Val:   {len(val_pairs)} ({100*len(val_pairs)/total:.1f}%)")
    print(f"  Unused: {total - len(train_pairs) - len(val_pairs)}")

    # Generate examples
    print("\nGenerating train examples...")
    train_examples = []
    for input_vec, operation in train_pairs:
        example_seed = rng.randint(0, 2**31)
        example = generate_example(
            path=operation,
            vector_length=cfg.vector.length,
            k=cfg.vector.k,
            seed=example_seed,
            fixed_input=list(input_vec)
        )
        train_examples.append({
            "input": example["input"],
            "output": example["output"]
        })

    print("Generating val examples...")
    val_examples = []
    for input_vec, operation in val_pairs:
        example_seed = rng.randint(0, 2**31)
        example = generate_example(
            path=operation,
            vector_length=cfg.vector.length,
            k=cfg.vector.k,
            seed=example_seed,
            fixed_input=list(input_vec)
        )
        val_examples.append({
            "input": example["input"],
            "output": example["output"]
        })

    return train_examples, val_examples


def generate_mixed_train_pure_test(cfg: DictConfig) -> Tuple[List[dict], List[dict]]:
    """Experiment 2: Mixed train (1+2 ops), pure test (2 ops only).

    - Train: Uses all available unique 1-op pairs (exhaustive), fills remaining
      slots with 2-op examples. If enough 1-op pairs exist, uses 50/50 split.
    - Test/Val: 2-op + optional 1-op (controlled by val_1op_size)
    - 1-op training uses exhaustive unique (input, operation) pairs (no duplicates)
    - 2-op uses uniform sampling (with replacement)
    - Input separation: disjoint input sets for train/test

    Returns:
        (train_examples, val_examples)
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: MIXED TRAIN (1+2 ops), PURE TEST (2 ops)")
    print("=" * 80)

    rng = random.Random(cfg.seed)

    # Build graph and get operations
    graph = TransformationGraph(
        d=cfg.graph.d,
        m=cfg.graph.m,
        vertex_mapping=None
    )

    # Get 1-op and 2-op paths separately
    paths_1op = graph.enumerate_all_paths(min_len=1, max_len=1)
    paths_2op = graph.enumerate_all_paths(min_len=2, max_len=2)

    operations_1op = [tuple(graph.path_to_transformation_names(p)) for p in paths_1op]
    operations_2op = [tuple(graph.path_to_transformation_names(p)) for p in paths_2op]

    print(f"\n1-operation paths: {len(operations_1op)}")
    print(f"2-operation paths: {len(operations_2op)}")

    # Generate all possible inputs
    all_inputs = enumerate_all_inputs(cfg.vector.length, cfg.vector.k)
    print(f"\nAll possible inputs: {len(all_inputs)}")

    # Partition inputs between train and test
    rng.shuffle(all_inputs)
    split_point = int(len(all_inputs) * cfg.dataset.train_test_split_inputs)
    train_inputs = all_inputs[:split_point]
    test_inputs = all_inputs[split_point:]

    print(f"\nInput partitioning (NO OVERLAP):")
    print(f"  Train input pool: {len(train_inputs)} ({100*len(train_inputs)/len(all_inputs):.1f}%)")
    print(f"  Test input pool:  {len(test_inputs)} ({100*len(test_inputs)/len(all_inputs):.1f}%)")

    # Enumerate ALL unique 1-op (input, operation) pairs for train inputs
    train_1op_pairs = [(inp, op) for inp in train_inputs for op in operations_1op]
    rng.shuffle(train_1op_pairs)
    available_train_1op = len(train_1op_pairs)

    print(f"\nUnique 1-op train pairs: {available_train_1op}")
    print(f"  = {len(train_inputs)} inputs x {len(operations_1op)} operations")

    # Determine actual 1-op / 2-op split for training
    desired_train_1op = cfg.dataset.train_size // 2
    actual_train_1op = min(desired_train_1op, available_train_1op)
    actual_train_2op = cfg.dataset.train_size - actual_train_1op

    if available_train_1op < desired_train_1op:
        shortfall = desired_train_1op - available_train_1op
        print(f"\n  NOTE: Requested {desired_train_1op} 1-op train examples, "
              f"but only {available_train_1op} unique pairs available.")
        print(f"  Using all {actual_train_1op} unique 1-op pairs for training.")
        print(f"  Filling {shortfall} extra slots with 2-op examples.")

    print(f"\nTrain composition:")
    print(f"  1-op: {actual_train_1op} ({100*actual_train_1op/cfg.dataset.train_size:.1f}%)")
    print(f"  2-op: {actual_train_2op} ({100*actual_train_2op/cfg.dataset.train_size:.1f}%)")
    print(f"  Total: {cfg.dataset.train_size}")

    # Val composition
    val_1op_size = cfg.dataset.get("val_1op_size", 0)
    val_2op_size = cfg.dataset.val_size - val_1op_size

    print(f"\nVal/Test composition:")
    if val_1op_size > 0:
        print(f"  1-op: {val_1op_size}")
    print(f"  2-op: {val_2op_size}")
    print(f"  Total: {cfg.dataset.val_size}")

    print(f"\nOperation sampling:")
    print(f"  1-op train: Exhaustive unique pairs (no duplicates)")
    print(f"  2-op: Uniform sampling (balanced by operation)")
    print(f"  Within 2-op: Each of {len(operations_2op)} operation pairs sampled equally")

    # Generate train examples
    train_examples = []

    # Train: 1-op examples (exhaustive unique pairs)
    print(f"\nGenerating {actual_train_1op} train 1-op examples (unique pairs)...")
    for input_vec, operation in train_1op_pairs[:actual_train_1op]:
        example_seed = rng.randint(0, 2**31)
        example = generate_example(
            path=operation,
            vector_length=cfg.vector.length,
            k=cfg.vector.k,
            seed=example_seed,
            fixed_input=list(input_vec)
        )
        train_examples.append({
            "input": example["input"],
            "output": example["output"]
        })

    # Train: 2-op examples (sampled with replacement)
    print(f"Generating {actual_train_2op} train 2-op examples...")
    for _ in range(actual_train_2op):
        input_vec = rng.choice(train_inputs)
        operation = rng.choice(operations_2op)
        example_seed = rng.randint(0, 2**31)

        example = generate_example(
            path=operation,
            vector_length=cfg.vector.length,
            k=cfg.vector.k,
            seed=example_seed,
            fixed_input=list(input_vec)
        )
        train_examples.append({
            "input": example["input"],
            "output": example["output"]
        })

    # Shuffle train to mix 1-op and 2-op
    rng.shuffle(train_examples)

    # Val/Test examples
    val_examples = []

    # Val: 1-op from test inputs (if requested)
    if val_1op_size > 0:
        test_1op_pairs = [(inp, op) for inp in test_inputs for op in operations_1op]
        rng.shuffle(test_1op_pairs)
        actual_val_1op = min(val_1op_size, len(test_1op_pairs))

        if actual_val_1op < val_1op_size:
            print(f"\n  NOTE: Requested {val_1op_size} val 1-op examples, "
                  f"but only {actual_val_1op} unique pairs available from test inputs.")

        print(f"Generating {actual_val_1op} val 1-op examples...")
        for input_vec, operation in test_1op_pairs[:actual_val_1op]:
            example_seed = rng.randint(0, 2**31)
            example = generate_example(
                path=operation,
                vector_length=cfg.vector.length,
                k=cfg.vector.k,
                seed=example_seed,
                fixed_input=list(input_vec)
            )
            val_examples.append({
                "input": example["input"],
                "output": example["output"]
            })

    # Val: 2-op from test inputs
    print(f"Generating {val_2op_size} val 2-op examples...")
    for _ in range(val_2op_size):
        input_vec = rng.choice(test_inputs)
        operation = rng.choice(operations_2op)
        example_seed = rng.randint(0, 2**31)

        example = generate_example(
            path=operation,
            vector_length=cfg.vector.length,
            k=cfg.vector.k,
            seed=example_seed,
            fixed_input=list(input_vec)
        )
        val_examples.append({
            "input": example["input"],
            "output": example["output"]
        })

    rng.shuffle(val_examples)

    return train_examples, val_examples


def generate_mixed_exhaustive(cfg: DictConfig) -> Tuple[List[dict], List[dict]]:
    """Exhaustive 1-op pairs + sampled 2-op examples.

    - 1-op: Enumerates ALL unique (input, operation) pairs, shuffles, splits
      into train/val. All 16 operations used (no path-level split for 1-op).
    - 2-op: Path-level split using train_ratio, then random sampling from
      the train/val path pools respectively.

    Config keys used:
        dataset.train_size: Total training examples
        dataset.val_size: Total validation examples
        dataset.per_length.1: How many 1-op examples in train (rest filled with 2-op)
        split.train_ratio: Fraction of 2-op paths used for training

    Returns:
        (train_examples, val_examples)
    """
    print("\n" + "=" * 80)
    print("MIXED EXHAUSTIVE: Exhaustive 1-op + Sampled 2-op")
    print("=" * 80)

    rng = random.Random(cfg.seed)

    # Build graph
    graph = TransformationGraph(
        d=cfg.graph.d,
        m=cfg.graph.m,
        vertex_mapping=None
    )

    # Get all operation paths (vertex-level)
    paths_1op = graph.enumerate_all_paths(min_len=1, max_len=1)
    paths_2op = graph.enumerate_all_paths(min_len=2, max_len=2)

    # 1-op: use ALL paths (no split)
    operations_1op = [tuple(graph.path_to_transformation_names(p)) for p in paths_1op]

    # 2-op: split paths using train_ratio
    train_2op_paths, val_2op_paths = train_val_split(
        paths_2op, cfg.split.train_ratio, cfg.seed
    )
    train_ops_2op = [tuple(graph.path_to_transformation_names(p)) for p in train_2op_paths]
    val_ops_2op = [tuple(graph.path_to_transformation_names(p)) for p in val_2op_paths]

    print(f"\n1-op operations: {len(operations_1op)} (all used, no split)")
    print(f"2-op operations: {len(paths_2op)} total")
    print(f"  Train 2-op paths: {len(train_ops_2op)} ({100*len(train_ops_2op)/len(paths_2op):.0f}%)")
    print(f"  Val 2-op paths:   {len(val_ops_2op)} ({100*len(val_ops_2op)/len(paths_2op):.0f}%)")

    # Enumerate all inputs
    all_inputs = enumerate_all_inputs(cfg.vector.length, cfg.vector.k)
    print(f"All possible inputs: {len(all_inputs)}")

    # Enumerate ALL unique 1-op (input, operation) pairs
    all_1op_pairs = [(inp, op) for inp in all_inputs for op in operations_1op]
    rng.shuffle(all_1op_pairs)
    total_1op = len(all_1op_pairs)
    print(f"\nTotal unique 1-op pairs: {total_1op}")
    print(f"  = {len(all_inputs)} inputs x {len(operations_1op)} operations")

    # How many 1-op for train?
    train_1op_count = cfg.dataset.per_length.get("1", cfg.dataset.train_size // 2)
    if train_1op_count > total_1op:
        print(f"  NOTE: Requested {train_1op_count} but only {total_1op} exist. Using all.")
        train_1op_count = total_1op

    # Val gets balanced split (equal 1-op and 2-op) from remaining 1-op pairs
    val_1op_count = min(cfg.dataset.val_size // 2, total_1op - train_1op_count)
    train_2op_count = cfg.dataset.train_size - train_1op_count
    val_2op_count = cfg.dataset.val_size - val_1op_count

    # Split 1-op pairs: first chunk for train, next for val
    train_1op_pairs = all_1op_pairs[:train_1op_count]
    val_1op_pairs = all_1op_pairs[train_1op_count:train_1op_count + val_1op_count]
    unused_1op = total_1op - train_1op_count - val_1op_count

    print(f"\n1-op split:")
    print(f"  Train: {train_1op_count}")
    print(f"  Val:   {val_1op_count}")
    print(f"  Unused: {unused_1op}")

    print(f"\nTrain composition:")
    print(f"  1-op: {train_1op_count} ({100*train_1op_count/cfg.dataset.train_size:.1f}%) [exhaustive]")
    print(f"  2-op: {train_2op_count} ({100*train_2op_count/cfg.dataset.train_size:.1f}%) [sampled from {len(train_ops_2op)} paths]")
    print(f"  Total: {cfg.dataset.train_size}")

    print(f"\nVal composition:")
    print(f"  1-op: {val_1op_count} [exhaustive]")
    print(f"  2-op: {val_2op_count} [sampled from {len(val_ops_2op)} paths]")
    print(f"  Total: {cfg.dataset.val_size}")

    # --- Generate train examples ---
    train_examples = []

    print(f"\nGenerating {train_1op_count} train 1-op examples (exhaustive)...")
    for input_vec, operation in train_1op_pairs:
        example_seed = rng.randint(0, 2**31)
        example = generate_example(
            path=operation,
            vector_length=cfg.vector.length,
            k=cfg.vector.k,
            seed=example_seed,
            fixed_input=list(input_vec)
        )
        train_examples.append({
            "input": example["input"],
            "output": example["output"]
        })

    print(f"Generating {train_2op_count} train 2-op examples (sampled)...")
    for _ in range(train_2op_count):
        input_vec = rng.choice(all_inputs)
        operation = rng.choice(train_ops_2op)
        example_seed = rng.randint(0, 2**31)
        example = generate_example(
            path=operation,
            vector_length=cfg.vector.length,
            k=cfg.vector.k,
            seed=example_seed,
            fixed_input=list(input_vec)
        )
        train_examples.append({
            "input": example["input"],
            "output": example["output"]
        })

    rng.shuffle(train_examples)

    # --- Generate val examples ---
    val_examples = []

    if val_1op_count > 0:
        print(f"Generating {val_1op_count} val 1-op examples (exhaustive)...")
        for input_vec, operation in val_1op_pairs:
            example_seed = rng.randint(0, 2**31)
            example = generate_example(
                path=operation,
                vector_length=cfg.vector.length,
                k=cfg.vector.k,
                seed=example_seed,
                fixed_input=list(input_vec)
            )
            val_examples.append({
                "input": example["input"],
                "output": example["output"]
            })

    print(f"Generating {val_2op_count} val 2-op examples (sampled)...")
    for _ in range(val_2op_count):
        input_vec = rng.choice(all_inputs)
        operation = rng.choice(val_ops_2op)
        example_seed = rng.randint(0, 2**31)
        example = generate_example(
            path=operation,
            vector_length=cfg.vector.length,
            k=cfg.vector.k,
            seed=example_seed,
            fixed_input=list(input_vec)
        )
        val_examples.append({
            "input": example["input"],
            "output": example["output"]
        })

    rng.shuffle(val_examples)

    return train_examples, val_examples


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Generate experimental datasets based on config mode."""

    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Check mode
    mode = cfg.dataset.get("mode", "standard")

    if mode == "exhaustive":
        train_examples, val_examples = generate_exhaustive(cfg)
    elif mode == "mixed_train_pure_test":
        train_examples, val_examples = generate_mixed_train_pure_test(cfg)
    elif mode == "mixed_exhaustive":
        train_examples, val_examples = generate_mixed_exhaustive(cfg)
    else:
        raise ValueError(
            f"Unknown dataset mode: {mode}\n"
            f"Supported modes: exhaustive, mixed_train_pure_test, mixed_exhaustive"
        )

    # Save results
    os.makedirs(".", exist_ok=True)

    with open("train.json", "w") as f:
        json.dump(train_examples, f, indent=2)

    with open("val.json", "w") as f:
        json.dump(val_examples, f, indent=2)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"✓ train.json: {len(train_examples)} examples")
    print(f"✓ val.json: {len(val_examples)} examples")
    print(f"✓ Output directory: {os.getcwd()}")


if __name__ == "__main__":
    main()
