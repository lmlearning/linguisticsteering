import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def bootstrap_confidence_interval(data: np.ndarray, n_bootstrap: int = 1000, ci_level: float = 0.95):
    """
    Calculates the confidence interval of the mean for a 1D array of data using bootstrap resampling.
    """
    if data.size == 0:
        return np.nan, (np.nan, np.nan)
        
    bootstrap_means = np.zeros(n_bootstrap)
    n_samples = len(data)
    
    for i in range(n_bootstrap):
        # Sample with replacement from the original data
        resample_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        resample = data[resample_indices]
        bootstrap_means[i] = np.mean(resample)
        
    # Calculate the confidence interval from the percentiles of the bootstrap distribution
    lower_bound_percentile = (1 - ci_level) / 2 * 100
    upper_bound_percentile = (1 - (1 - ci_level) / 2) * 100
    
    ci = np.percentile(bootstrap_means, [lower_bound_percentile, upper_bound_percentile])
    original_mean = np.mean(data)
    
    return original_mean, tuple(ci)

def analyze_variance_absolute(input_path: str, top_k: int, n_bootstrap: int):
    """
    Loads results, performs bootstrap analysis on the MEAN ABSOLUTE Shapley values 
    for the top_k adjectives, and generates a corresponding violin plot.
    """
    if not os.path.exists(input_path):
        logging.error(f"Input file not found at '{input_path}'")
        return

    logging.info(f"Loading data from {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)

    # --- Setup Output Directory ---
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = f"{base_name}_variance_analysis"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logging.info(f"Outputs will be saved to '{output_dir}/'")

    # --- Extract Data ---
    agg_df = pd.DataFrame(data['aggregated_results']).T.sort_values(
        'mean_abs_shapley', ascending=False
    )
    top_k_adjectives = agg_df.head(top_k).index.tolist()
    logging.info(f"Selected Top-{top_k} adjectives for variance analysis: {top_k_adjectives}")

    per_question_results = data.get('per_question_results', {})
    if not per_question_results:
        logging.error("No 'per_question_results' found in the input file. Cannot perform variance analysis.")
        return

    # Convert per-question results to a DataFrame for easier slicing
    adjective_map = data['adjective_mapping']
    adjectives_ordered = sorted(adjective_map.keys(), key=lambda k: adjective_map[k])
    df_per_question = pd.DataFrame(per_question_results.values(), columns=adjectives_ordered)

    # --- Perform Bootstrap Analysis on ABSOLUTE values ---
    bootstrap_results = {}
    logging.info(f"Performing bootstrap analysis with {n_bootstrap} replicates on absolute Shapley values...")
    for adj in tqdm(top_k_adjectives, desc="Bootstrapping Adjectives"):
        # The key change is here: we use the absolute values for the analysis
        abs_shap_values_for_adj = np.abs(df_per_question[adj].dropna().to_numpy())
        original_mean, ci = bootstrap_confidence_interval(abs_shap_values_for_adj, n_bootstrap)
        bootstrap_results[adj] = {
            "mean_abs_shapley": original_mean,
            "95_ci_lower": ci[0],
            "95_ci_upper": ci[1]
        }

    # --- Print Summary Table ---
    print("\n" + "="*80)
    print("Bootstrap Analysis of Mean ABSOLUTE Shapley Values (Overall Impact)")
    print("="*80)
    print(f"{'Adjective':<15} | {'Mean Absolute Impact':<20} | {'95% Confidence Interval':<30}")
    print("-"*80)
    for adj, res in bootstrap_results.items():
        ci_str = f"[{res['95_ci_lower']:.4f}, {res['95_ci_upper']:.4f}]"
        print(f"{adj:<15} | {res['mean_abs_shapley']:<20.4f} | {ci_str:<30}")
    print("="*80 + "\n")

    # --- Generate Violin Plot of ABSOLUTE values ---
    logging.info("Generating violin plot of absolute value distributions...")
    
    # Reshape data into a long format suitable for seaborn
    plot_df_long = df_per_question[top_k_adjectives].abs().melt(
        var_name='adjective', value_name='absolute_shapley_value'
    )
    
    # Ensure the order in the plot is the same as our impact ranking
    order = top_k_adjectives

    plt.figure(figsize=(12, 8))
    sns.violinplot(
        data=plot_df_long,
        x='absolute_shapley_value',
        y='adjective',
        order=order,
        palette='viridis',
        inner='quartile',
        cut=0 # Prevents violins from extending past the data range
    )
    
    plt.title(f'Distribution of Per-Question Absolute Shapley Values for Top {top_k} Impactful Adjectives', fontsize=16, pad=20)
    plt.xlabel('Absolute Shapley Value (Magnitude of Impact on a Single Question)', fontsize=12)
    plt.ylabel('Adjective', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"{base_name}_absolute_shapley_distributions.png")
    plt.savefig(plot_path, dpi=300)
    logging.info(f"Violin plot saved to {plot_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform bootstrap variance analysis on absolute Shapley value results.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "--input_results", 
        type=str, 
        required=True, 
        help="Path to the JSON results file from the initial sampling script (e.g., o3_corrected.json)."
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=10, 
        help="Number of top adjectives (by absolute impact) to analyze."
    )
    parser.add_argument(
        "--n_bootstrap", 
        type=int, 
        default=5000, 
        help="Number of bootstrap replicates to generate for confidence intervals."
    )
    
    args = parser.parse_args()
    
    # Set a seed for the bootstrap resampling to be reproducible
    np.random.seed(4221)
    
    analyze_variance_absolute(args.input_results, args.top_k, args.n_bootstrap)