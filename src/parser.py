"""Trace parsing at different granularity levels."""

import re
from typing import Dict, List


def parse_example_line(line: str, input_len: int = None) -> Dict:
    """Parse a formatted example line.

    Format: "INPUT: ... OUTPUT: ... TRACE: ..."

    Args:
        line: The formatted line
        input_len: Unused, kept for backward compatibility

    Returns:
        Dict with input, output, trace keys
    """
    # Find the keyword positions
    input_match = re.search(r'INPUT:', line)
    output_match = re.search(r'OUTPUT:', line)
    trace_match = re.search(r'TRACE:', line)

    if not all([input_match, output_match, trace_match]):
        raise ValueError(f"Invalid format, missing INPUT:/OUTPUT:/TRACE: keywords in: {line}")

    # Extract content between keywords
    input_start = input_match.end()
    input_end = output_match.start()
    input_str = line[input_start:input_end].strip()

    output_start = output_match.end()
    output_end = trace_match.start()
    output_str = line[output_start:output_end].strip()

    trace_start = trace_match.end()
    trace_str = line[trace_start:].strip()
    if trace_str.endswith('</s>'):
        trace_str = trace_str[:-4].strip()

    # Parse vectors
    input_vec = [int(x) for x in input_str.split()]
    output_vec = [int(x) for x in output_str.split()]

    return {
        "input": input_vec,
        "output": output_vec,
        "trace": trace_str,
    }


def parse_coarse(trace: str) -> List[str]:
    """Parse trace at coarse level: extract transformation names in order.

    Args:
        trace: The trace string (space-separated transformation traces)

    Returns:
        List of transformation names in order
    """
    # Each transformation trace starts with "name :" (name followed by space and colon)
    pattern = r'([a-z_][a-z_0-9]*) :'
    matches = re.findall(pattern, trace)
    return matches


def parse_medium(trace: str) -> List[List[int]]:
    """Parse trace at medium level: extract result vectors for each transformation.

    Args:
        trace: The trace string

    Returns:
        List of result vectors (one per transformation)
    """
    # Find all [ ... ] patterns (result vectors with spaced format)
    # Pattern matches: [ 1 , 2 , 3 ]
    pattern = r'\[ ([\d\s,]+) \]'
    matches = re.findall(pattern, trace)

    vectors = []
    for match in matches:
        # Parse "1 , 2 , 3" format
        parts = match.split(',')
        vec = [int(x.strip()) for x in parts]
        vectors.append(vec)

    return vectors


def parse_fine(trace: str) -> List[Dict]:
    """Parse trace at fine level: extract operations for each transformation.

    Args:
        trace: The trace string

    Returns:
        List of dicts with name and operations for each transformation
    """
    # Split into individual transformation blocks
    # Each block starts with "name : " and ends before the next "name :"
    # Pattern: transformation_name : [ops :] [ result ]

    results = []

    # Find all transformation names and their positions
    name_pattern = r'([a-z_][a-z_0-9]*) :'
    name_matches = list(re.finditer(name_pattern, trace))

    for i, match in enumerate(name_matches):
        name = match.group(1)

        # Get the content after this name until the next name (or end)
        start = match.end()
        if i + 1 < len(name_matches):
            end = name_matches[i + 1].start()
        else:
            end = len(trace)

        content = trace[start:end].strip().rstrip(';').strip()

        # Find the result vector (last [ ... ] in content)
        vec_pattern = r'\[ [\d\s,]+ \]'
        vec_matches = list(re.finditer(vec_pattern, content))

        if vec_matches:
            # Operations are everything before the last vector
            last_vec_start = vec_matches[-1].start()
            ops_str = content[:last_vec_start].rstrip(' :')
        else:
            ops_str = ""

        results.append({
            "name": name,
            "operations": ops_str,
        })

    return results
