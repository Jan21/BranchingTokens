"""Metrics module for fine-grained evaluation.

This module provides:
- Format adapter to convert clean_framework format to parser format
- Parser for extracting transformation functions from traces
- Per-function metrics computation (exact match, token accuracy, sequence, operations)
- Functional correctness metrics (re-execute predicted ops to check result)
"""

from .adapter import clean_to_parser_format, extract_input_output
from .per_function import PerFunctionMetrics
from .functional_correctness import FunctionalCorrectnessMetrics

__all__ = [
    "clean_to_parser_format",
    "extract_input_output",
    "PerFunctionMetrics",
    "FunctionalCorrectnessMetrics",
]
