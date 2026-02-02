"""Trace parsing at different granularity levels."""

import re
from typing import Dict, List


def parse_example_line(line: str, input_len: int) -> Dict:
    """Parse a formatted example line.

    Format: "ST : [ INPUT ] ; BEGIN : TRACE ; RESULT : [ OUTPUT ]"

    Args:
        line: The formatted line
        input_len: Length of input/output vectors (needed for validation)

    Returns:
        Dict with input, output, trace keys
    """
    line = line.strip()

    # Extract input vector (between "ST : [" and "] ;")
    input_match = re.search(r'ST : \[ (.*?) \] ;', line)
    if not input_match:
        raise ValueError("Could not find input vector in line")
    input_vec = [int(x) for x in input_match.group(1).split()]

    # Extract trace (between "BEGIN :" and "; RESULT")
    trace_match = re.search(r'BEGIN : (.*?) ; RESULT', line)
    if not trace_match:
        raise ValueError("Could not find trace in line")
    trace = trace_match.group(1)

    # Extract output vector (between "RESULT : [" and "]")
    output_match = re.search(r'RESULT : \[ (.*?) \]', line)
    if not output_match:
        raise ValueError("Could not find output vector in line")
    output_vec = [int(x) for x in output_match.group(1).split()]

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
    # Each transformation trace starts with "name :" where name begins with a letter
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
    # Find all [ ... ] patterns (result vectors with space-separated numbers)
    pattern = r'\[ ([\d\s]+) \]'
    matches = re.findall(pattern, trace)

    vectors = []
    for match in matches:
        vec = [int(x) for x in match.split()]
        vectors.append(vec)

    return vectors


def parse_fine(trace: str) -> List[Dict]:
    """Parse trace at fine level: extract operations for each transformation.

    Args:
        trace: The trace string

    Returns:
        List of dicts with name and operations for each transformation
    """
    # Split trace into individual transformations using regex
    # Each transformation starts with "name :" and contains operations and a result vector
    pattern = r'([a-z_][a-z_0-9]*) : (.*?) : \[ ([\d\s]+) \]'
    matches = re.finditer(pattern, trace)

    results = []
    for match in matches:
        name = match.group(1)
        operations = match.group(2).strip()  # Operations string (may be empty)
        result_vec = [int(x) for x in match.group(3).split()]

        results.append({
            "name": name,
            "operations": operations,
            "result": result_vec,
        })

    return results
