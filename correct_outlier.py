import json
import argparse
from pathlib import Path
import pandas as pd

def correct_specific_outliers(input_file: Path, output_file: Path, value_to_subtract: float, outlier_threshold: float):
    """
    Reads a SHAP results JSON, subtracts a value from results for outlier questions,
    and recalculates aggregated statistics.
    """
    print(f"Loading data from: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)

    per_question_results = data.get('per_question_results', {})
    if not per_question_results:
        print("No per-question results found. Exiting.")
        return

    adjectives = list(data.get('adjective_mapping', {}).keys())
    if not adjectives:
        num_adjectives = len(next(iter(per_question_results.values())))
        adjectives = [f"adj_{i}" for i in range(num_adjectives)]

    modified_per_question_results = {}
    outlier_questions_count = 0

    # Process each question's results
    for q_id, values in per_question_results.items():
        # Identify a question as an outlier if any of its values exceeds the threshold
        is_outlier_question = any(abs(v) > outlier_threshold for v in values)

        if is_outlier_question:
            print("true")
            outlier_questions_count += 1
            # If it's an outlier, subtract the specified value from all its results
            modified_values = [(v / value_to_subtract) for v in values]

            print(values, modified_values)
            modified_per_question_results[q_id] = modified_values
        else:
            # Otherwise, keep the original results
            modified_per_question_results[q_id] = values

    print(f"\nFound and applied correction to {outlier_questions_count} outlier question(s).")

    # Recalculate aggregated results from the modified data
    print("Re-aggregating results with the modified data...")
    df = pd.DataFrame(modified_per_question_results).T
    if not df.empty:
        df.columns = adjectives
        agg_results = pd.DataFrame({
            'mean_shapley': df.mean(),
            'std_shapley': df.std(),
            'mean_abs_shapley': df.abs().mean()
        }).sort_values(by='mean_abs_shapley', ascending=False)
        agg_results_dict = agg_results.to_dict('index')
    else:
        agg_results_dict = {}
        print("Warning: No data remained after processing; aggregated results will be empty.")


    # Create the new, corrected data structure
    corrected_data = {
        'parameters': data.get('parameters', {}),
        'adjective_mapping': data.get('adjective_mapping', {}),
        'aggregated_results': agg_results_dict,
        'per_question_results': modified_per_question_results
    }

    print(f"\nSaving corrected data to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(corrected_data, f, indent=2)

    print("Correction script finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Subtracts a constant value from outlier results in a SHAP JSON file.")
    parser.add_argument("--input_file", type=Path, required=True, help="Path to the original JSON file.")
    parser.add_argument("--output_file", type=Path, required=True, help="Path to save the corrected JSON output.")
    parser.add_argument("--value", type=float, default=1e9, help="The value to subtract from outlier results.")
    parser.add_argument("--threshold", type=float, default=1e3, help="The magnitude a value must exceed to be considered part of an outlier set.")

    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"Error: Input file not found at {args.input_file}")
    else:
        correct_specific_outliers(args.input_file, args.output_file, args.value, args.threshold)