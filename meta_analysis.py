import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
import argparse
from scipy.stats import spearmanr
from itertools import combinations

def create_subject_groups():
    """Defines broad academic categories for MMLU subjects."""
    return {
        "STEM": ["abstract_algebra", "college_biology", "college_chemistry", "college_computer_science", 
                 "college_mathematics", "college_physics", "computer_security", "electrical_engineering", 
                 "elementary_mathematics", "high_school_biology", "high_school_chemistry", 
                 "high_school_computer_science", "high_school_mathematics", "high_school_physics", 
                 "high_school_statistics", "machine_learning", "virology", "anatomy", "astronomy",
                 "medical_genetics", "nutrition", "conceptual_physics"],
        "Humanities": ["formal_logic", "high_school_european_history", "high_school_us_history", 
                       "high_school_world_history", "philosophy", "prehistory", "world_religions", "moral_disputes",
                       "moral_scenarios", "logical_fallacies"],
        "Social Sciences": ["econometrics", "high_school_geography", "high_school_government_and_politics", 
                            "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology", 
                            "professional_psychology", "public_relations", "security_studies", "sociology", 
                            "us_foreign_policy", "global_facts"],
        "Professional": ["business_ethics", "clinical_knowledge", "college_medicine", "human_aging", "human_sexuality",
                         "international_law", "jurisprudence", "management", "marketing", "professional_accounting", 
                         "professional_law", "professional_medicine"]
    }

def load_summary_data(analysis_dir):
    """
    Loads aggregated results and summary from a single model's analysis directory
    using a robust method to parse the markdown table.
    """
    try:
        summary_path = [f for f in os.listdir(analysis_dir) if f.endswith('_summary.json')][0]
        agg_results_path = [f for f in os.listdir(analysis_dir) if f.endswith('_summary_table.md')][0]
    except IndexError:
        print(f"Error: Missing required summary files in directory '{analysis_dir}'.")
        print("Please ensure both a '_summary.json' and '_summary_table.md' file exist.")
        return None, None

    with open(os.path.join(analysis_dir, summary_path), 'r') as f:
        summary_data = json.load(f)
        
    # --- Robust Markdown Table Parsing Logic ---
    with open(os.path.join(analysis_dir, agg_results_path), 'r') as f:
        lines = f.readlines()

    # Find the header line (first line with |)
    header_line_index = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('|'):
            header_line_index = i
            break
    
    if header_line_index == -1:
        print(f"Error: Could not find markdown table header in {agg_results_path}")
        return None, None

    # Extract headers (skipping the separator line)
    headers = [h.strip() for h in lines[header_line_index].strip().strip('|').split('|')]
    
    # Extract data rows
    data_rows = []
    for line in lines[header_line_index + 2:]: # Skip header and separator line
        if line.strip().startswith('|'):
            cleaned_line = line.strip().strip('|').split('|')
            data_rows.append([d.strip() for d in cleaned_line])
        else:
            break # Stop if we hit the end of the table
            
    df = pd.DataFrame(data_rows, columns=headers)
    # --- End of Robust Parsing Logic ---
    
    for col in ['mean_shapley', 'std_shapley', 'mean_abs_shapley']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df[['adjective', 'mean_shapley', 'std_shapley', 'mean_abs_shapley']].set_index('adjective')
    
    return df, summary_data['analysis_summary']


def meta_analysis(model_inputs):
    """
    Performs a meta-analysis across multiple models' analysis results.
    `model_inputs` is a list of (name, path) tuples.
    """
    output_dir = "meta_analysis_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Outputs will be saved to '{output_dir}/'")

    all_data = {}
    for model_name, directory_path in model_inputs:
        df, summary = load_summary_data(directory_path)
        if df is not None and summary is not None:
            all_data[model_name] = {'df': df, 'summary': summary}
            print(f"Successfully loaded data for model: {model_name}")
            
    if len(all_data) < 2:
        print("Meta-analysis requires at least two valid analysis directories. Exiting.")
        return

    model_names = list(all_data.keys())

    # --- 1. Combine Data for Comparison ---
    mean_abs_shapley_df = pd.DataFrame({
        model: data['df']['mean_abs_shapley'] for model, data in all_data.items()
    }).dropna()
    mean_shapley_df = pd.DataFrame({
        model: data['df']['mean_shapley'] for model, data in all_data.items()
    }).dropna()
    domain_sensitivity_data = []
    for model, data in all_data.items():
        if 'mean_sensitivity_by_group' in data['summary']:
            for group, sensitivity in data['summary']['mean_sensitivity_by_group']:
                domain_sensitivity_data.append({'model': model, 'domain': group, 'sensitivity': sensitivity})
    domain_df = pd.DataFrame(domain_sensitivity_data)


    # --- 2. Generate Markdown Summary ---
    summary_md_path = os.path.join(output_dir, "meta_analysis_summary.md")
    with open(summary_md_path, 'w') as f:
        f.write("# LLM Adjective Sensitivity Meta-Analysis\n\n")
        f.write(f"Comparing models: `{'`, `'.join(model_names)}`\n\n")
        f.write("## Rank Stability of Adjective Impact\n\n")
        f.write("Spearman rank correlation of adjective impact (mean absolute Shapley value). 1.0 = identical ranking.\n\n")
        rank_corr = mean_abs_shapley_df.corr(method='spearman')
        f.write(rank_corr.to_markdown(floatfmt=".3f"))
        f.write("\n\n")
        f.write("## Top 10 Most Impactful Adjectives Across Models\n\n")
        top_10_df = pd.DataFrame({
            model: data['df'].sort_values('mean_abs_shapley', ascending=False).head(10).index.tolist()
            for model, data in all_data.items()
        })
        f.write(top_10_df.to_markdown())
        f.write("\n\n")
        if not domain_df.empty:
            f.write("## Sensitivity to Steering by Academic Domain\n\n")
            f.write(domain_df.pivot(index='domain', columns='model', values='sensitivity').to_markdown(floatfmt=".4f"))
            f.write("\n\n")

    print(f"Markdown summary saved to {summary_md_path}")

    # --- 3. Generate Plots ---
    print("Generating comparative plots...")
    sns.set_theme(style="whitegrid", palette="muted")
    
    # Plot 1: Overall Model Sensitivity Comparison
    if not domain_df.empty:
        plt.figure(figsize=(12, 8))
        sns.barplot(data=domain_df, x='sensitivity', y='model', hue='domain', dodge=True, palette='magma')
        plt.title('Overall Model Sensitivity to Adjectival Steering by Domain', fontsize=16, pad=20)
        plt.xlabel('Average Steering Magnitude (Sum of Abs. Shapley Values)', fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.legend(title='Domain', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plot_1_model_sensitivity.png"), dpi=300)
        plt.close()

    # Plot 2: Scatter plots comparing mean shapley values
    model_pairs = list(combinations(model_names, 2))
    num_pairs = len(model_pairs)
    if num_pairs > 0:
        cols = 2 if num_pairs > 1 else 1
        rows = (num_pairs + cols - 1) // cols # Ceiling division
        fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 7 * rows), squeeze=False)
        axes = axes.flatten()

        for i, (model1, model2) in enumerate(model_pairs):
            ax = axes[i]
            corr = mean_shapley_df[model1].corr(mean_shapley_df[model2])
            sns.regplot(data=mean_shapley_df, x=model1, y=model2, ax=ax,
                        scatter_kws={'alpha': 0.6, 's': 50}, line_kws={'color': 'red', 'linestyle': '--'})
            ax.set_title(f'{model1} vs. {model2}\nPearson Correlation: {corr:.3f}', fontsize=14)
            ax.set_xlabel(f'Mean Shapley Value ({model1})', fontsize=10)
            ax.set_ylabel(f'Mean Shapley Value ({model2})', fontsize=10)
            ax.axhline(0, color='grey', linestyle=':', lw=1)
            ax.axvline(0, color='grey', linestyle=':', lw=1)
        
        for j in range(i + 1, len(axes)): fig.delaxes(axes[j])
        fig.suptitle('Cross-Model Alignment of Adjective Steering Effects', fontsize=20, y=1.02 if rows > 1 else 1.05)
        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, "plot_2_shapley_correlation.png"), dpi=300)
        plt.close()

    # Plot 3: Rank Correlation Heatmap
    if not rank_corr.empty:
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(rank_corr, dtype=bool))
        sns.heatmap(rank_corr, annot=True, cmap='viridis', fmt=".3f", mask=mask, linewidths=0.5)
        plt.title('Spearman Rank Correlation of Adjective Impact', fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plot_3_rank_correlation.png"), dpi=300)
        plt.close()

    print("\nMeta-analysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform a meta-analysis across multiple LLM adjective sensitivity reports.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "model_inputs",
        nargs='+',
        type=str,
        help="One or more model-path pairs in the format 'ModelName:path/to/analysis_dir'.\n"
             "Example:\n"
             "python meta_analysis.py O3:o3_complete_analysis/ GPT4:gpt4_analysis/"
    )
    args = parser.parse_args()

    parsed_inputs = []
    for arg in args.model_inputs:
        if ':' not in arg:
            print(f"Error: Argument '{arg}' is not in the required 'ModelName:path/to/dir' format. Skipping.")
            continue
        parts = arg.split(':', 1)
        name, path = parts[0], parts[1]
        if not os.path.isdir(path):
            print(f"Error: The provided path for model '{name}' does not exist or is not a directory: '{path}'. Skipping.")
            continue
        parsed_inputs.append((name, path))

    if len(parsed_inputs) < 2:
        print("Error: Please provide at least two valid model-path pairs to compare.")
    else:
        meta_analysis(parsed_inputs)