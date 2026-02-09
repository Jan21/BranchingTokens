"""Convenience wrapper to run train.py with data overrides.

Usage:
    python run.py --train data/train_1.json --test data/val_1.json
    python run.py --train data/train_2_1.json --test data/val_2_1.json
    python run.py --train data/train_2_1.json --test data/val_2_1.json --epochs 50
    python run.py  # uses defaults from base.yaml
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Run training with data overrides")
    parser.add_argument("--train", type=str, help="Path to train JSON file")
    parser.add_argument("--test", type=str, help="Path to test/val JSON file")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--k", type=int, help="Modulo value for transformations")
    args, extra = parser.parse_known_args()

    cmd = [sys.executable, "train.py"]

    if args.train:
        cmd.append(f"data.train_file={args.train}")
    if args.test:
        cmd.append(f"data.test_file={args.test}")
    if args.epochs:
        cmd.append(f"model.epochs={args.epochs}")
    if args.batch_size:
        cmd.append(f"model.batch_size={args.batch_size}")
    if args.lr:
        cmd.append(f"optim.lr={args.lr}")
    if args.k:
        cmd.append(f"vector.k={args.k}")

    # Pass through any extra Hydra overrides directly
    cmd.extend(extra)

    print(f"Running: {' '.join(cmd)}")
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
