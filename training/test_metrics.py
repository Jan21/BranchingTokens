"""Quick test script for per-function metrics."""

import sys
sys.path.insert(0, '.')

from utils.metrics import clean_to_parser_format, extract_input_output, PerFunctionMetrics

# Test 1: Format adapter
print("=" * 60)
print("Test 1: Format Adapter")
print("=" * 60)

input_field = "INPUT : [ 9 , 6 , 9 , 9 ] , OUTPUT : [ 0 , 0 , 0 , 0 ]"
print(f"Input field: {input_field}")

try:
    input_str, output_str = extract_input_output(input_field)
    print(f"Extracted input: {input_str}")
    print(f"Extracted output: {output_str}")

    trace_str = "sort_asc : [ 6 , 9 , 9 , 9 ] ; add_1 : 6 + 1 = 7 , 9 + 1 = 0 , 9 + 1 = 0 , 9 + 1 = 0 : [ 7 , 0 , 0 , 0 ]"
    parser_format = clean_to_parser_format(input_str, output_str, trace_str)
    print(f"\nParser format:\n{parser_format}")
    print("\n✓ Format adapter works!\n")
except Exception as e:
    print(f"✗ Format adapter failed: {e}\n")
    import traceback
    traceback.print_exc()

# Test 2: Per-function metrics
print("=" * 60)
print("Test 2: Per-Function Metrics")
print("=" * 60)

try:
    metrics = PerFunctionMetrics()

    # Create sample prediction and ground truth
    input_str = "9 6 9 9"
    output_str = "0 0 0 0"

    # Perfect prediction
    gt_trace = "sort_asc : [ 6 , 9 , 9 , 9 ] ; add_1 : 6 + 1 = 7 , 9 + 1 = 0 : [ 7 , 0 , 0 , 0 ]"
    pred_trace = "sort_asc : [ 6 , 9 , 9 , 9 ] ; add_1 : 6 + 1 = 7 , 9 + 1 = 0 : [ 7 , 0 , 0 , 0 ]"

    gt_line = clean_to_parser_format(input_str, output_str, gt_trace)
    pred_line = clean_to_parser_format(input_str, output_str, pred_trace)

    print(f"Ground truth: {gt_line[:80]}...")
    print(f"Prediction:   {pred_line[:80]}...")

    metrics.process_example(pred_line, gt_line)

    # Test with a 3-function example
    gt_trace_3 = "sort_asc : [ 6 , 9 , 9 , 9 ] ; add_1 : 6 + 1 = 7 : [ 7 , 0 , 0 , 0 ] ; reverse : [ 0 , 0 , 0 , 7 ]"
    pred_trace_3 = "sort_asc : [ 6 , 9 , 9 , 9 ] ; add_1 : 6 + 1 = 7 : [ 7 , 0 , 0 , 0 ] ; reverse : [ 0 , 0 , 0 , 7 ]"
    metrics.process_example(
        clean_to_parser_format(input_str, output_str, pred_trace_3),
        clean_to_parser_format(input_str, output_str, gt_trace_3)
    )

    # Get results
    results = metrics.get_metrics()

    print("\nPer-function results:")
    for func_name, func_metrics in results.items():
        print(f"\n{func_name}:")
        print(f"  Exact match: {func_metrics['exact_match']:.2f}")
        print(f"  Token accuracy: {func_metrics['token_accuracy']:.2f}")
        print(f"  Sequence correct: {func_metrics['sequence_correct']:.2f}")
        if func_metrics['operations_accuracy'] is not None:
            print(f"  Operations accuracy: {func_metrics['operations_accuracy']:.2f}")
        print(f"  Count: {func_metrics['count']}")

    # Get length-based metrics
    length_results = metrics.get_length_metrics()
    print("\nExact match by sequence length:")
    for length, length_metrics in sorted(length_results.items(), key=lambda x: int(x[0])):
        label = f"{length} function" if int(length) == 1 else f"{length} functions"
        print(f"  {label}: {length_metrics['accuracy']:.2f} ({length_metrics['correct']}/{length_metrics['total']})")

    print("\n✓ Per-function metrics work!\n")
except Exception as e:
    print(f"✗ Per-function metrics failed: {e}\n")
    import traceback
    traceback.print_exc()

print("=" * 60)
print("All tests completed!")
print("=" * 60)
