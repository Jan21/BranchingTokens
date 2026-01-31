"""Transformation functions for BranchingTokens.

Each transformation:
- Takes a vector (list of ints), modulo k, and a trace list
- Returns a new vector (same length)
- Appends its trace string to the trace list
- All values are mod k after operation
"""

from typing import List


def _format_vec(vec: List[int]) -> str:
    """Format vector as [a,b,c]."""
    return "[" + ",".join(str(x) for x in vec) + "]"


def reverse(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Reverse the vector."""
    result = vec[::-1]
    trace.append(f"reverse:{_format_vec(result)}")
    return result


def sort_asc(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Sort ascending."""
    result = sorted(vec)
    trace.append(f"sort_asc:{_format_vec(result)}")
    return result


def sort_desc(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Sort descending."""
    result = sorted(vec, reverse=True)
    trace.append(f"sort_desc:{_format_vec(result)}")
    return result


def cumsum(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Cumulative sum with mod k."""
    result = []
    running = 0
    ops = []
    for i, x in enumerate(vec):
        if i == 0:
            result.append(x % k)
            ops.append(str(x % k))
        else:
            new_val = (running + x) % k
            ops.append(f"{running}+{x}={new_val}")
            result.append(new_val)
        running = result[-1]
    trace.append(f"cumsum:{','.join(ops)}:{_format_vec(result)}")
    return result


def cumsum_reverse(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Cumulative sum from right with mod k."""
    reversed_vec = vec[::-1]
    result = []
    running = 0
    ops = []
    for i, x in enumerate(reversed_vec):
        if i == 0:
            result.append(x % k)
            ops.append(str(x % k))
        else:
            new_val = (running + x) % k
            ops.append(f"{running}+{x}={new_val}")
            result.append(new_val)
        running = result[-1]
    result = result[::-1]
    trace.append(f"cumsum_reverse:{','.join(ops)}:{_format_vec(result)}")
    return result


def add_1(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Add 1 to all elements mod k."""
    result = [(x + 1) % k for x in vec]
    ops = [f"{x}+1={r}" for x, r in zip(vec, result)]
    trace.append(f"add_1:{','.join(ops)}:{_format_vec(result)}")
    return result


def add_2(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Add 2 to all elements mod k."""
    result = [(x + 2) % k for x in vec]
    ops = [f"{x}+2={r}" for x, r in zip(vec, result)]
    trace.append(f"add_2:{','.join(ops)}:{_format_vec(result)}")
    return result


def add_3(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Add 3 to all elements mod k."""
    result = [(x + 3) % k for x in vec]
    ops = [f"{x}+3={r}" for x, r in zip(vec, result)]
    trace.append(f"add_3:{','.join(ops)}:{_format_vec(result)}")
    return result
