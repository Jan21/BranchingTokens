"""Analyze sequence length distribution in evaluation data."""

import json
import sys
from collections import defaultdict, Counter

sys.path.insert(0, '.')
from utils.metrics import extract_input_output
from utils.metrics.parser import parse_coarse

def analyze_data(filepath, num_examples=512):
    """Analyze the distribution of sequence lengths in data."""

    with open(filepath, 'r') as f:
        data = json.load(f)

    # Limit to num_examples (same as evaluation)
    data = data[:num_examples]

    length_stats = defaultdict(lambda: {
        'count': 0,
        'functions_used': Counter(),
        'examples': []
    })

    for idx, example in enumerate(data):
        try:
            # Parse trace to count functions
            trace_str = example['output']
            functions = parse_coarse(f"TRACE: {trace_str}")
            num_functions = len(functions)

            # Track statistics
            stats = length_stats[num_functions]
            stats['count'] += 1
            for func in functions:
                stats['functions_used'][func] += 1

            # Keep first few examples for inspection
            if len(stats['examples']) < 3:
                stats['examples'].append({
                    'input': example['input'],
                    'output': trace_str,
                    'functions': functions
                })

        except Exception as e:
            print(f"Warning: Failed to parse example {idx}: {e}")
            continue

    return length_stats


def print_analysis(stats):
    """Print analysis results."""
    print("=" * 80)
    print("SEQUENCE LENGTH DISTRIBUTION")
    print("=" * 80)

    total_examples = sum(s['count'] for s in stats.values())
    print(f"\nTotal examples analyzed: {total_examples}")

    for length in sorted(stats.keys()):
        s = stats[length]
        pct = (s['count'] / total_examples) * 100
        print(f"\n{length}-function sequences: {s['count']} examples ({pct:.1f}%)")

        print(f"  Functions used:")
        for func, count in s['functions_used'].most_common():
            avg_per_example = count / s['count']
            print(f"    {func}: {count} times ({avg_per_example:.2f} per example)")

        if s['examples']:
            print(f"  Sample examples:")
            for i, ex in enumerate(s['examples'][:2], 1):
                print(f"    Example {i}:")
                print(f"      Functions: {' -> '.join(ex['functions'])}")
                print(f"      Trace: {ex['output'][:100]}...")


if __name__ == "__main__":
    import sys
    from omegaconf import OmegaConf

    # Load config to get file paths
    config = OmegaConf.load("config/base.yaml")
    train_file = config.data.train_file
    test_file = config.data.test_file

    # Analyze ENTIRE train set
    print("=" * 80)
    print("ANALYZING ENTIRE TRAIN SET")
    print("=" * 80)
    print(f"File: {train_file}\n")

    with open(train_file, 'r') as f:
        train_data = json.load(f)
    train_total = len(train_data)

    print(f"Analyzing all {train_total} examples...\n")
    train_stats = analyze_data(train_file, train_total)
    print_analysis(train_stats)

    # Analyze ENTIRE test set
    print("\n\n")
    print("=" * 80)
    print("ANALYZING ENTIRE TEST SET")
    print("=" * 80)
    print(f"File: {test_file}\n")

    with open(test_file, 'r') as f:
        test_data = json.load(f)
    test_total = len(test_data)

    print(f"Analyzing all {test_total} examples...\n")
    test_stats = analyze_data(test_file, test_total)
    print_analysis(test_stats)

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("""
If 4-function sequences have higher accuracy than 1-function:
- Check if 4-function examples are much rarer (statistical noise)
- Check if 4-function examples use easier functions
- Check if single-function examples include difficult functions
- Consider that the model might have learned compositional patterns better
""")
