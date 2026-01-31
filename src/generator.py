"""Example generation for BranchingTokens."""

import random
from typing import Dict, List, Tuple

from src.transformations import get_transformation


def generate_vector(length: int, k: int, seed: int) -> List[int]:
    """Generate a random vector.

    Args:
        length: Vector length
        k: Modulo value (values in [0, k-1])
        seed: Random seed

    Returns:
        List of random integers in [0, k-1]
    """
    rng = random.Random(seed)
    return [rng.randint(0, k - 1) for _ in range(length)]


def generate_example(
    path: Tuple[str, ...],
    vector_length: int,
    k: int,
    seed: int
) -> Dict:
    """Generate a single training/validation example.

    Args:
        path: Tuple of transformation names to apply
        vector_length: Length of input vector
        k: Modulo value
        seed: Random seed for vector generation

    Returns:
        Dict with keys: input, output, trace, formatted
    """
    input_vec = generate_vector(vector_length, k, seed)

    current = list(input_vec)
    trace = []

    for transform_name in path:
        fn = get_transformation(transform_name)
        current = fn(current, k, trace)

    output_vec = current

    # Format: "INPUT OUTPUT TRACE"
    input_str = " ".join(str(x) for x in input_vec)
    output_str = " ".join(str(x) for x in output_vec)
    trace_str = " ".join(trace)

    formatted = f"{input_str} {output_str} {trace_str}"

    return {
        "input": input_vec,
        "output": output_vec,
        "trace": trace,
        "formatted": formatted,
    }
