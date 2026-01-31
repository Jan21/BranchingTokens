"""Trace parsing at different granularity levels."""

import re
from typing import Dict, List


def parse_example_line(line: str, input_len: int) -> Dict:
    """Parse a formatted example line.

    Format: "INPUT OUTPUT TRACE" where INPUT and OUTPUT are space-separated ints.

    Args:
        line: The formatted line
        input_len: Length of input/output vectors (needed to know where trace starts)

    Returns:
        Dict with input, output, trace keys
    """
    parts = line.strip().split()

    input_vec = [int(x) for x in parts[:input_len]]

    # Find where trace starts (first non-integer token after input)
    trace_start = input_len
    for i in range(input_len, len(parts)):
        try:
            int(parts[i])
            trace_start = i + 1
        except ValueError:
            trace_start = i
            break

    output_vec = [int(x) for x in parts[input_len:trace_start]]
    trace = " ".join(parts[trace_start:])

    return {
        "input": input_vec,
        "output": output_vec,
        "trace": trace,
    }


def parse_coarse(trace: str) -> List[str]:
    """Parse trace at coarse level: extract transformation names in order.

    Args:
        trace: The trace string (space-separated transformation traces)

    Returns:
        List of transformation names in order
    """
    # Each transformation trace starts with "name:" where name begins with a letter
    pattern = r'([a-z_][a-z_0-9]*):'
    matches = re.findall(pattern, trace)
    return matches


def parse_medium(trace: str) -> List[List[int]]:
    """Parse trace at medium level: extract result vectors for each transformation.

    Args:
        trace: The trace string

    Returns:
        List of result vectors (one per transformation)
    """
    # Find all [...] patterns (result vectors)
    pattern = r'\[([0-9,]+)\]'
    matches = re.findall(pattern, trace)

    vectors = []
    for match in matches:
        vec = [int(x) for x in match.split(',')]
        vectors.append(vec)

    return vectors


def parse_fine(trace: str) -> List[Dict]:
    """Parse trace at fine level: extract operations for each transformation.

    Args:
        trace: The trace string

    Returns:
        List of dicts with name and operations for each transformation
    """
    # Split by transformation (each starts with name:)
    parts = trace.strip().split()

    results = []
    for part in parts:
        if ':' not in part:
            continue

        # Split into name and rest
        colon_idx = part.index(':')
        name = part[:colon_idx]
        rest = part[colon_idx + 1:]

        # Extract operations (everything before the final [...])
        bracket_idx = rest.rfind('[')
        if bracket_idx > 0:
            ops_str = rest[:bracket_idx].rstrip(':')
            operations = ops_str
        else:
            operations = ""

        results.append({
            "name": name,
            "operations": operations,
        })

    return results
