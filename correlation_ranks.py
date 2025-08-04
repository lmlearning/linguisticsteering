import json
import argparse
from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr
from itertools import combinations

def calculate_rank_correlation(files: list[Path], metric: str, output_path: Path | None):
    """
    Loads multiple JSON result files, extracts adjective rankings, and computes
    the Spearman's rank correlation matrix between all pairs of files.
    """
    rankings = {}
    filenames = [f.name for f in files]

    # 1. Load data and extract the ranking for each file
    for file in files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            
            # Create a DataFrame from the aggregated results for easy sorting
            agg_results = data.get('aggregated_results', {})
            if not agg_results:
                print(f"Warning: No 'aggregated_results' in {file.name}. Skipping.")
                continue

            df = pd.DataFrame.from_dict(agg_results, orient='index')
            
            # Sort by the specified metric to get the rank
            if metric not in df.columns:
                print(f"Warning: Metric '{metric}' not found in {file.name}. Skipping.")
                continue

            ranked_adjectives = df.sort_values(by=metric, ascending=False).index.tolist()
            
            # Store the rank of each adjective in a dictionary
            rankings[file.name] = {adj: i for i, adj in enumerate(ranked_adjectives)}

        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            print(f"Could not process file {file.name}. Error: {e}. Skipping.")
            continue
    
    if len(rankings) < 2:
        print("Need at least two valid files to compute correlation. Exiting.")
        return

    # 2. Compute the correlation matrix for all pairs of files
    correlation_matrix = pd.DataFrame(index=filenames, columns=filenames, dtype=float)

    for file1, file2 in combinations(filenames, 2):
        if file1 not in rankings or file2 not in rankings:
            continue
            
        map1 = rankings[file1]
        map2 = rankings[file2]
        
        # Find the common adjectives between the two files to ensure a fair comparison
        common_adjectives = set(map1.keys()) & set(map2.keys())
        
        if len(common_adjectives) < 2:
            print(f"Warning: Not enough common adjectives between {file1} and {file2} to correlate.")
            rho = float('nan')
        else:
            # Create lists of ranks for only the common adjectives
            ranks1 = [map1[adj] for adj in common_adjectives]
            ranks2 = [map2[adj] for adj in common_adjectives]
            
            # Calculate Spearman's rho
            rho, _ = spearmanr(ranks1, ranks2)

        # Populate the matrix
        correlation_matrix.loc[file1, file2] = rho
        correlation_matrix.loc[file2, file1] = rho

    # Fill diagonal with 1s
    for file in filenames:
        if file in correlation_matrix.index:
            correlation_matrix.loc[file, file] = 1.0

    print("\n## Spearman's Rank Correlation (ρ) Matrix:")
    print(correlation_matrix.to_string())

    # 3. Save to CSV if an output path is provided
    if output_path:
        try:
            correlation_matrix.to_csv(output_path)
            print(f"\nCorrelation matrix saved to: {output_path}")
        except Exception as e:
            print(f"\nError saving to {output_path}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Calculate Spearman's rank correlation between SHAP results from multiple JSON files."
    )
    parser.add_argument(
        "files",
        type=Path,
        nargs='+',
        help="One or more paths to the JSON result files."
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="mean_abs_shapley",
        help="The metric in 'aggregated_results' to use for ranking (default: mean_abs_shapley)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional. Path to save the resulting correlation matrix as a CSV file."
    )

    args = parser.parse_args()
    calculate_rank_correlation(args.files, args.metric, args.output)