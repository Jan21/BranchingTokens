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
    seed: int,
    fixed_input: List[int] = None
) -> Dict:
    """Generate a single training/validation example.

    Args:
        path: Tuple of transformation names to apply
        vector_length: Length of input vector
        k: Modulo value
        seed: Random seed for vector generation
        fixed_input: Optional fixed input vector (overrides random generation)

    Returns:
        Dict with keys: input, output, trace, formatted
    """
    if fixed_input is not None:
        input_vec = list(fixed_input)
    else:
        input_vec = generate_vector(vector_length, k, seed)

    current = list(input_vec)
    trace = []

    for transform_name in path:
        fn = get_transformation(transform_name)
        current = fn(current, k, trace)

    output_vec = current

    # Format compatible with clean_framework
    # Framework expects: {"input": "...", "output": "..."}
    # Framework will tokenize as: [BOS] input [OUT] output [EOS]
    input_str = " , ".join(str(x) for x in input_vec)
    output_str = " , ".join(str(x) for x in output_vec)
    trace_str = " ; ".join(trace)

    # Input includes: INPUT : <vec> OUTPUT : <vec>
    # Output is just the trace
    # Framework adds [OUT] between them and model predicts only the trace
    formatted_input = f"INPUT : [ {input_str} ] , OUTPUT : [ {output_str} ]"

    return {
        "input": formatted_input,
        "output": trace_str,
        "trace": trace,
    }
