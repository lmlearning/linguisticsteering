import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
from scipy import stats

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def format_p_value(p):
    """Formats a p-value for clear display."""
    return "< .001" if p < 0.001 else f"{p:.4f}"

def run_one_sample_tests(input_path: str, top_k: int):
    """
    Loads results and performs one-sample hypothesis tests on the
    absolute Shapley values of the top_k most impactful adjectives.
    """
    if not os.path.exists(input_path):
        logging.error(f"Input file not found at '{input_path}'")
        return

    logging.info(f"Loading data from {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)

    # --- Extract and Prepare Data ---
    model_name = data.get('parameters', {}).get('model', 'Unknown Model')
    logging.info(f"Analyzing model: {model_name}")

    agg_df = pd.DataFrame(data['aggregated_results']).T.sort_values(
        'mean_abs_shapley', ascending=False
    )
    top_k_adjectives = agg_df.head(top_k).index.tolist()
    logging.info(f"Selected Top-{top_k} adjectives for hypothesis testing: {top_k_adjectives}")

    per_question_results = data.get('per_question_results', {})
    if not per_question_results:
        logging.error("No 'per_question_results' found in the input file. Cannot perform tests.")
        return

    adjective_map = data['adjective_mapping']
    adjectives_ordered = sorted(adjective_map.keys(), key=lambda k: adjective_map[k])
    df_per_question = pd.DataFrame(per_question_results.values(), columns=adjectives_ordered)

    # We will be testing the absolute Shapley values
    df_abs = df_per_question[top_k_adjectives].abs()

    # --- Perform One-Sample Tests (vs. Zero) ---
    print("\n" + "="*80)
    print("One-Sample Tests: Is the Mean Absolute Impact Significantly Greater Than Zero?")
    print("="*80)
    print(f"{'Adjective':<15} | {'Mean Abs Impact':<20} | {'T-test p-value':<20} | {'Wilcoxon p-value':<20}")
    print("-"*80)

    results = []
    for adj in top_k_adjectives:
        values = df_abs[adj].dropna().to_numpy()
        mean_abs = np.mean(values)
        
        # One-sample T-test: tests if the mean is different from a given value (0).
        # We use alternative='greater' because our hypothesis is that the impact > 0.
        ttest_res = stats.ttest_1samp(values, 0, alternative='greater')
        
        # Wilcoxon signed-rank test: non-parametric alternative.
        # Tests if the distribution is centered around a given value (0).
        # We also use alternative='greater' here.
        wilcoxon_res = stats.wilcoxon(values, alternative='greater')
        
        results.append({
            "adjective": adj,
            "mean_abs_impact": mean_abs,
            "ttest_p": ttest_res.pvalue,
            "wilcoxon_p": wilcoxon_res.pvalue
        })

    # Sort results for easier reading, matching the original impact ranking
    for res in results:
        print(f"{res['adjective']:<15} | {res['mean_abs_impact']:<20.4f} | {format_p_value(res['ttest_p']):<20} | {format_p_value(res['wilcoxon_p']):<20}")
    print("="*80 + "\n")
    
    # --- Save results to a file ---
    output_dir = f"{os.path.splitext(os.path.basename(input_path))[0]}_variance_analysis"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    table_path = os.path.join(output_dir, "one_sample_test_results.md")
    df_results = pd.DataFrame(results)
    with open(table_path, 'w') as f:
        f.write(f"# One-Sample Hypothesis Test Results for {model_name}\n\n")
        f.write(f"This table tests whether the mean absolute impact of the top {top_k} adjectives is statistically greater than zero.\n\n")
        f.write(df_results.to_markdown(index=False, floatfmt=".4f"))
    logging.info(f"Test results saved to {table_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform one-sample hypothesis tests on Shapley value results.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "--input_results", 
        type=str, 
        required=True, 
        help="Path to the JSON results file from an experiment run (e.g., o3_corrected.json)."
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=10, 
        help="Number of top adjectives (by absolute impact) to test."
    )
    
    args = parser.parse_args()
    
    run_one_sample_tests(args.input_results, args.top_k)