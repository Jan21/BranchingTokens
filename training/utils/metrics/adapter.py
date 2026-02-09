"""Format adapter to convert between clean_framework and parser formats.

The clean_framework uses:
    input:  "INPUT : [ 9 , 6 ] , OUTPUT : [ 0 , 0 ]"
    output: "sort_asc : [ 6 , 9 ] ; add_1 : ..."

The parser expects:
    "INPUT: 9 6 OUTPUT: 0 0 TRACE: sort_asc : [ 6 , 9 ] ; add_1 : ..."
"""

import re
from typing import Tuple


# Patterns to extract vectors from clean_framework format
INPUT_PATTERN = r'INPUT\s*:\s*\[\s*([\d\s,]+)\s*\]'
OUTPUT_PATTERN = r'OUTPUT\s*:\s*\[\s*([\d\s,]+)\s*\]'


def extract_input_output(input_field: str) -> Tuple[str, str]:
    """Extract input and output vectors from clean_framework input field.

    Args:
        input_field: String like "INPUT : [ 9 , 6 ] , OUTPUT : [ 0 , 0 ]"

    Returns:
        Tuple of (input_str, output_str) like ("9 6", "0 0")

    Raises:
        ValueError: If input or output pattern not found
    """
    # Extract input vector
    input_match = re.search(INPUT_PATTERN, input_field)
    if not input_match:
        raise ValueError(f"Could not find INPUT pattern in: {input_field}")

    input_numbers = input_match.group(1)
    # Remove commas and extra spaces
    input_str = ' '.join(input_numbers.replace(',', '').split())

    # Extract output vector
    output_match = re.search(OUTPUT_PATTERN, input_field)
    if not output_match:
        raise ValueError(f"Could not find OUTPUT pattern in: {input_field}")

    output_numbers = output_match.group(1)
    # Remove commas and extra spaces
    output_str = ' '.join(output_numbers.replace(',', '').split())

    return input_str, output_str


def clean_to_parser_format(input_str: str, output_str: str, trace_str: str) -> str:
    """Convert clean_framework format to parser format.

    Args:
        input_str: Input vector like "9 6"
        output_str: Output vector like "0 0"
        trace_str: Trace string like "sort_asc : [ 6 , 9 ] ; add_1 : ..."

    Returns:
        Parser format string: "INPUT: 9 6 OUTPUT: 0 0 TRACE: sort_asc : ..."
    """
    return f"INPUT: {input_str} OUTPUT: {output_str} TRACE: {trace_str}"
