"""Functional correctness metrics for transformation traces.

Checks whether predicted traces are:
1. Parse-valid: Can they be parsed with valid operation names?
2. Ops-valid: Are all operation names real transformations?
3. Result-correct: Do the predicted operations produce the correct final output?
4. Intermediate-valid: Do intermediate vectors match re-execution?
"""

import sys
import os
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from .parser import parse_coarse, parse_medium, parse_example_line

# Import transformations from BranchingTokens (sibling directory)
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_bt_path = os.path.join(_repo_root, "BranchingTokens")
if _bt_path not in sys.path:
    sys.path.insert(0, _bt_path)

from src.transformations import TRANSFORMATIONS


logger = logging.getLogger(__name__)


class FunctionalCorrectnessMetrics:
    """Compute functional correctness metrics across a dataset.

    Instead of exact string matching, this re-executes predicted operations
    on the input vector and checks whether the result matches ground truth.
    """

    def __init__(self, k: int = 10):
        """Initialize metrics tracking.

        Args:
            k: Modulo value for transformation execution.
        """
        self.k = k

        # Aggregate counters
        self.total = 0
        self.parse_valid = 0
        self.ops_valid = 0
        self.result_correct = 0
        self.intermediate_valid = 0
        self.valid = 0  # combined: intermediate_valid AND result_correct

        # Per-length tracking
        self.length_stats = defaultdict(lambda: {
            'total': 0,
            'parse_valid': 0,
            'ops_valid': 0,
            'result_correct': 0,
            'intermediate_valid': 0,
            'valid': 0,
        })

    def _execute_operations(
        self, op_names: List[str], input_vec: List[int]
    ) -> Tuple[Optional[List[int]], List[List[int]]]:
        """Execute a sequence of operations on an input vector.

        Args:
            op_names: List of operation names to apply sequentially.
            input_vec: The input vector.

        Returns:
            (final_output, intermediate_results) or (None, []) if invalid.
        """
        current = list(input_vec)
        intermediates = []

        for name in op_names:
            if name not in TRANSFORMATIONS:
                return None, []
            fn = TRANSFORMATIONS[name]
            trace_sink = []  # discard trace strings
            current = fn(current, self.k, trace_sink)
            intermediates.append(list(current))

        return current, intermediates

    def process_example(self, pred_line: str, gt_line: str) -> Dict[str, bool]:
        """Process one prediction-ground truth pair.

        Args:
            pred_line: Prediction in parser format "INPUT: ... OUTPUT: ... TRACE: ..."
            gt_line: Ground truth in parser format

        Returns:
            Dict with boolean results for each metric.
        """
        result = {
            'parse_valid': False,
            'ops_valid': False,
            'result_correct': False,
            'intermediate_valid': False,
            'valid': False,
        }

        self.total += 1

        try:
            gt_parsed = parse_example_line(gt_line)
            input_vec = gt_parsed['input']
            expected_output = gt_parsed['output']
            gt_trace = gt_parsed['trace']
            gt_ops = parse_coarse(gt_trace)
            num_functions = len(gt_ops)
        except Exception as e:
            logger.warning(f"Failed to parse ground truth: {e}")
            self.length_stats[0]['total'] += 1
            return result

        self.length_stats[num_functions]['total'] += 1

        try:
            pred_parsed = parse_example_line(pred_line)
            pred_trace = pred_parsed['trace']

            # 1. Parse validity
            pred_ops = parse_coarse(pred_trace)
            if len(pred_ops) == 0:
                return result

            result['parse_valid'] = True
            self.parse_valid += 1
            self.length_stats[num_functions]['parse_valid'] += 1

            # 2. Operation validity
            all_ops_valid = all(name in TRANSFORMATIONS for name in pred_ops)
            if not all_ops_valid:
                return result

            result['ops_valid'] = True
            self.ops_valid += 1
            self.length_stats[num_functions]['ops_valid'] += 1

            # 3. Result correctness
            final_output, intermediates = self._execute_operations(pred_ops, input_vec)

            if final_output is not None and final_output == expected_output:
                result['result_correct'] = True
                self.result_correct += 1
                self.length_stats[num_functions]['result_correct'] += 1

            # 4. Intermediate validity
            pred_vectors = parse_medium(pred_trace)

            if (len(intermediates) == len(pred_vectors) and
                    all(iv == pv for iv, pv in zip(intermediates, pred_vectors))):
                result['intermediate_valid'] = True
                self.intermediate_valid += 1
                self.length_stats[num_functions]['intermediate_valid'] += 1

            # 5. Valid = intermediates correct AND result correct
            if result['intermediate_valid'] and result['result_correct']:
                result['valid'] = True
                self.valid += 1
                self.length_stats[num_functions]['valid'] += 1

        except Exception as e:
            logger.warning(f"Error processing functional correctness: {e}")

        return result

    def get_metrics(self) -> Dict[str, float]:
        """Compute final aggregate metrics.

        Returns:
            Dictionary with metric names and values.
        """
        if self.total == 0:
            return {
                'parse_validity': 0.0,
                'ops_validity': 0.0,
                'result_correctness': 0.0,
                'intermediate_validity': 0.0,
                'valid': 0.0,
                'total': 0,
            }

        return {
            'parse_validity': self.parse_valid / self.total,
            'ops_validity': self.ops_valid / self.total,
            'result_correctness': self.result_correct / self.total,
            'intermediate_validity': self.intermediate_valid / self.total,
            'valid': self.valid / self.total,
            'total': self.total,
        }

    def get_length_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute functional correctness metrics by sequence length.

        Returns:
            Dictionary mapping sequence length to metrics.
        """
        metrics = {}
        for length, stats in self.length_stats.items():
            total = stats['total']
            if total == 0:
                continue
            metrics[str(length)] = {
                'parse_validity': stats['parse_valid'] / total,
                'ops_validity': stats['ops_valid'] / total,
                'result_correctness': stats['result_correct'] / total,
                'intermediate_validity': stats['intermediate_valid'] / total,
                'valid': stats['valid'] / total,
                'total': total,
            }
        return metrics
