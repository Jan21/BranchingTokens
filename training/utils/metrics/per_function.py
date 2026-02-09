"""Per-function metrics computation for transformation traces.

Tracks accuracy metrics at the individual function level:
- Exact match accuracy: Did the function's output vector match exactly?
- Token accuracy: Element-wise correctness of output vectors
- Sequence accuracy: Was the function in the correct position?
- Operations accuracy: Were the intermediate operations correct?
"""

import logging
from typing import Dict, List, Optional
from collections import defaultdict

from .parser import parse_coarse, parse_medium, parse_fine, parse_example_line


logger = logging.getLogger(__name__)


class PerFunctionMetrics:
    """Compute per-function accuracy metrics across a dataset."""

    def __init__(self):
        """Initialize metrics tracking dictionaries."""
        # Track counts and accuracies for each function
        self.function_stats = defaultdict(lambda: {
            'count': 0,
            'exact_match_correct': 0,
            'token_correct': 0,
            'token_total': 0,
            'sequence_correct': 0,
            'operations_correct': 0,
            'operations_total': 0,
        })

        # Track exact match accuracy by sequence length (number of functions)
        self.length_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    def process_example(self, pred_line: str, gt_line: str) -> None:
        """Process one prediction-ground truth pair.

        Args:
            pred_line: Prediction in parser format "INPUT: ... OUTPUT: ... TRACE: ..."
            gt_line: Ground truth in parser format
        """
        try:
            # Parse both lines
            pred_parsed = parse_example_line(pred_line)
            gt_parsed = parse_example_line(gt_line)

            pred_trace = pred_parsed['trace']
            gt_trace = gt_parsed['trace']

            # Extract function names (coarse level)
            try:
                pred_functions = parse_coarse(pred_trace)
                gt_functions = parse_coarse(gt_trace)
            except Exception as e:
                logger.warning(f"Failed to parse coarse functions: {e}")
                return

            # Extract function outputs (medium level)
            try:
                pred_outputs = parse_medium(pred_trace)
                gt_outputs = parse_medium(gt_trace)
            except Exception as e:
                logger.warning(f"Failed to parse medium outputs: {e}")
                pred_outputs = []
                gt_outputs = []

            # Extract operations (fine level) for operation-level accuracy
            try:
                pred_fine = parse_fine(pred_trace)
                gt_fine = parse_fine(gt_trace)
            except Exception as e:
                logger.warning(f"Failed to parse fine operations: {e}")
                pred_fine = []
                gt_fine = []

            # Track exact match by sequence length
            num_functions = len(gt_functions)
            self.length_stats[num_functions]['total'] += 1
            # Check if entire prediction matches ground truth
            if pred_trace == gt_trace:
                self.length_stats[num_functions]['correct'] += 1

            # Process each ground truth function
            for idx, gt_func in enumerate(gt_functions):
                stats = self.function_stats[gt_func]
                stats['count'] += 1

                # Check if function appears in prediction at all
                if gt_func not in pred_functions:
                    # Function missing entirely - all metrics fail
                    if idx < len(gt_outputs):
                        stats['token_total'] += len(gt_outputs[idx])
                    continue

                # Find the function's index in prediction
                pred_idx = pred_functions.index(gt_func)

                # 1. Sequence accuracy: Is function in correct position?
                if pred_idx == idx:
                    stats['sequence_correct'] += 1

                # 2. Exact match accuracy: Does output vector match?
                if (idx < len(gt_outputs) and pred_idx < len(pred_outputs)):
                    gt_output = gt_outputs[idx]
                    pred_output = pred_outputs[pred_idx]

                    if gt_output == pred_output:
                        stats['exact_match_correct'] += 1

                    # 3. Token accuracy: Element-wise correctness
                    min_len = min(len(gt_output), len(pred_output))
                    max_len = max(len(gt_output), len(pred_output))

                    for i in range(min_len):
                        if gt_output[i] == pred_output[i]:
                            stats['token_correct'] += 1

                    stats['token_total'] += max_len

                # 4. Operations accuracy: Do operations match?
                if idx < len(gt_fine) and pred_idx < len(pred_fine):
                    gt_ops = gt_fine[idx].get('operations', '')
                    pred_ops = pred_fine[pred_idx].get('operations', '')

                    # Only track if there are operations to compare
                    if gt_ops:
                        stats['operations_total'] += 1
                        if gt_ops == pred_ops:
                            stats['operations_correct'] += 1

        except Exception as e:
            logger.warning(f"Error processing example: {e}")
            logger.debug(f"pred_line: {pred_line[:100]}...")
            logger.debug(f"gt_line: {gt_line[:100]}...")

    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute final metrics for each function.

        Returns:
            Dictionary mapping function name to metrics dict:
            {
                "function_name": {
                    "exact_match": float,
                    "token_accuracy": float,
                    "sequence_correct": float,
                    "operations_accuracy": float or None,
                    "count": int
                }
            }
        """
        metrics = {}

        for func_name, stats in self.function_stats.items():
            count = stats['count']
            if count == 0:
                continue

            # Compute metrics
            exact_match = stats['exact_match_correct'] / count
            sequence_correct = stats['sequence_correct'] / count

            # Token accuracy: ratio of correct tokens to total tokens
            token_accuracy = (stats['token_correct'] / stats['token_total']
                              if stats['token_total'] > 0 else 0.0)

            # Operations accuracy: only if function has operations
            operations_accuracy = None
            if stats['operations_total'] > 0:
                operations_accuracy = stats['operations_correct'] / stats['operations_total']

            metrics[func_name] = {
                'exact_match': exact_match,
                'token_accuracy': token_accuracy,
                'sequence_correct': sequence_correct,
                'operations_accuracy': operations_accuracy,
                'count': count
            }

        return metrics

    def get_length_metrics(self) -> Dict[str, Dict]:
        """Compute exact match accuracy by sequence length.

        Returns:
            Dictionary mapping sequence length to metrics:
            {
                "1": {"correct": int, "total": int, "accuracy": float},
                "2": {"correct": int, "total": int, "accuracy": float},
                ...
            }
        """
        metrics = {}
        for length, stats in self.length_stats.items():
            total = stats['total']
            correct = stats['correct']
            accuracy = correct / total if total > 0 else 0.0
            metrics[str(length)] = {
                'correct': correct,
                'total': total,
                'accuracy': accuracy
            }
        return metrics
