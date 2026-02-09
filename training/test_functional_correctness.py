"""Test script for functional correctness metrics."""

import sys
sys.path.insert(0, '.')

from utils.metrics import clean_to_parser_format, extract_input_output, FunctionalCorrectnessMetrics


# Test 1: Exact match prediction (all metrics should pass)
print("=" * 60)
print("Test 1: Exact match prediction")
print("=" * 60)

fc = FunctionalCorrectnessMetrics(k=10)

# sort_asc([9,6,9,9]) = [6,9,9,9], then add_1([6,9,9,9]) = [7,0,0,0]
input_str = "9 6 9 9"
output_str = "7 0 0 0"
gt_trace = "sort_asc : [ 6 , 9 , 9 , 9 ] ; add_1 : 6 + 1 = 7 , 9 + 1 = 0 , 9 + 1 = 0 , 9 + 1 = 0 : [ 7 , 0 , 0 , 0 ]"
pred_trace = gt_trace  # exact match

gt_line = clean_to_parser_format(input_str, output_str, gt_trace)
pred_line = clean_to_parser_format(input_str, output_str, pred_trace)

result = fc.process_example(pred_line, gt_line)
print(f"Result: {result}")
assert result['parse_valid'] == True, f"Expected parse_valid=True, got {result['parse_valid']}"
assert result['ops_valid'] == True, f"Expected ops_valid=True, got {result['ops_valid']}"
assert result['result_correct'] == True, f"Expected result_correct=True, got {result['result_correct']}"
assert result['intermediate_valid'] == True, f"Expected intermediate_valid=True, got {result['intermediate_valid']}"
print("PASS\n")


# Test 2: Functionally equivalent prediction (different ops, same result)
print("=" * 60)
print("Test 2: Functionally equivalent (different ops, same output)")
print("=" * 60)

fc2 = FunctionalCorrectnessMetrics(k=10)

# reverse([1,2,3,4]) = [4,3,2,1], reverse([4,3,2,1]) = [1,2,3,4] -> identity
# rotate_left([1,2,3,4]) = [2,3,4,1], rotate_right([2,3,4,1]) = [1,2,3,4] -> also identity
input_str = "1 2 3 4"
output_str = "1 2 3 4"
gt_trace = "reverse : [ 4 , 3 , 2 , 1 ] ; reverse : [ 1 , 2 , 3 , 4 ]"
pred_trace = "rotate_left : [ 2 , 3 , 4 , 1 ] ; rotate_right : [ 1 , 2 , 3 , 4 ]"

gt_line = clean_to_parser_format(input_str, output_str, gt_trace)
pred_line = clean_to_parser_format(input_str, output_str, pred_trace)

result = fc2.process_example(pred_line, gt_line)
print(f"Result: {result}")
assert result['parse_valid'] == True
assert result['ops_valid'] == True
assert result['result_correct'] == True, f"Expected result_correct=True (functionally equivalent), got {result['result_correct']}"
assert result['intermediate_valid'] == True, "Intermediates should match re-execution"
print("PASS\n")


# Test 3: Invalid operation name
print("=" * 60)
print("Test 3: Invalid operation name")
print("=" * 60)

fc3 = FunctionalCorrectnessMetrics(k=10)

pred_trace = "fake_op : [ 1 , 2 , 3 , 4 ]"
gt_trace = "reverse : [ 4 , 3 , 2 , 1 ]"

gt_line = clean_to_parser_format("1 2 3 4", "4 3 2 1", gt_trace)
pred_line = clean_to_parser_format("1 2 3 4", "4 3 2 1", pred_trace)

result = fc3.process_example(pred_line, gt_line)
print(f"Result: {result}")
assert result['parse_valid'] == True, "Should parse (fake_op matches name pattern)"
assert result['ops_valid'] == False, "fake_op is not a valid transformation"
assert result['result_correct'] == False
print("PASS\n")


# Test 4: Valid ops but wrong result
print("=" * 60)
print("Test 4: Valid ops but wrong result")
print("=" * 60)

fc4 = FunctionalCorrectnessMetrics(k=10)

# GT: sort_asc([4,2,3,1]) = [1,2,3,4]
# Pred: reverse([4,2,3,1]) = [1,3,2,4] != [1,2,3,4]
gt_trace = "sort_asc : [ 1 , 2 , 3 , 4 ]"
pred_trace = "reverse : [ 1 , 3 , 2 , 4 ]"

gt_line = clean_to_parser_format("4 2 3 1", "1 2 3 4", gt_trace)
pred_line = clean_to_parser_format("4 2 3 1", "1 2 3 4", pred_trace)

result = fc4.process_example(pred_line, gt_line)
print(f"Result: {result}")
assert result['parse_valid'] == True
assert result['ops_valid'] == True
assert result['result_correct'] == False, f"reverse([4,2,3,1])=[1,3,2,4] != [1,2,3,4], got {result['result_correct']}"
print("PASS\n")


# Test 5: Unparseable prediction (empty/garbage trace)
print("=" * 60)
print("Test 5: Unparseable prediction")
print("=" * 60)

fc5 = FunctionalCorrectnessMetrics(k=10)

gt_trace = "reverse : [ 4 , 3 , 2 , 1 ]"
pred_trace = "1234 garbage"  # no valid "name :" pattern

gt_line = clean_to_parser_format("1 2 3 4", "4 3 2 1", gt_trace)
pred_line = clean_to_parser_format("1 2 3 4", "4 3 2 1", pred_trace)

result = fc5.process_example(pred_line, gt_line)
print(f"Result: {result}")
assert result['parse_valid'] == False
assert result['ops_valid'] == False
assert result['result_correct'] == False
print("PASS\n")


# Test 6: Aggregate metrics
print("=" * 60)
print("Test 6: Aggregate metrics")
print("=" * 60)

fc_agg = FunctionalCorrectnessMetrics(k=10)

# Example 1: correct
gt_line1 = clean_to_parser_format("1 2 3 4", "4 3 2 1", "reverse : [ 4 , 3 , 2 , 1 ]")
pred_line1 = clean_to_parser_format("1 2 3 4", "4 3 2 1", "reverse : [ 4 , 3 , 2 , 1 ]")
fc_agg.process_example(pred_line1, gt_line1)

# Example 2: wrong result
gt_line2 = clean_to_parser_format("4 2 3 1", "1 2 3 4", "sort_asc : [ 1 , 2 , 3 , 4 ]")
pred_line2 = clean_to_parser_format("4 2 3 1", "1 2 3 4", "reverse : [ 1 , 3 , 2 , 4 ]")
fc_agg.process_example(pred_line2, gt_line2)

metrics = fc_agg.get_metrics()
print(f"Metrics: {metrics}")
assert metrics['total'] == 2
assert metrics['parse_validity'] == 1.0, f"Both are parseable, got {metrics['parse_validity']}"
assert metrics['ops_validity'] == 1.0
assert metrics['result_correctness'] == 0.5, f"1 of 2 correct, got {metrics['result_correctness']}"

length_metrics = fc_agg.get_length_metrics()
print(f"Length metrics: {length_metrics}")
assert "1" in length_metrics, "Both examples have 1 GT function"
print("PASS\n")


print("=" * 60)
print("All functional correctness tests passed!")
print("=" * 60)
