import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

def load_data(filepath):
    """Loads the JSON data from the specified file path."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        exit()
    except json.JSONDecodeError:
        print(f"Error: The file '{filepath}' is not a valid JSON file.")
        exit()

def set_plot_style():
    """Sets a publication-quality style for the plots."""
    sns.set_theme(style="whitegrid", palette="viridis")
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'darkgray',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'text.color': 'black',
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })

def plot_question_impact_pie(data, output_filename="question_impact.png"):
    """
    Generates and saves a pie chart showing the proportion of questions
    where adjectives had an effect versus no effect.
    """
    total_questions = len(data['per_question_results'])
    affected_questions = sum(1 for q in data['per_question_results'].values() if any(abs(val) > 1e-9 for val in q))
    unaffected_questions = total_questions - affected_questions

    labels = 'Affected Questions', 'Unaffected Questions'
    sizes = [affected_questions, unaffected_questions]
    colors = ['#663399', '#D8BFD8'] # RebeccaPurple and Thistle
    explode = (0.1, 0)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=140, textprops={'fontsize': 12, 'color': 'black'})
    ax.axis('equal')
    plt.title('Proportion of Questions with Adjective Influence', fontsize=16, pad=20)
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()
    print(f"Saved chart: {output_filename}")

def plot_category_impact(data, output_filename="category_impact.png"):
    """
    Generates and saves a bar chart showing the number of affected questions
    per category, sorted in descending order.
    """
    category_stats = {}
    for q_key, values in data['per_question_results'].items():
        category = '_'.join(q_key.split('_')[2:])
        if category not in category_stats:
            category_stats[category] = {'affected': 0, 'total': 0}
        
        category_stats[category]['total'] += 1
        if any(abs(v) > 1e-9 for v in values):
            category_stats[category]['affected'] += 1
            
    df = pd.DataFrame.from_dict(category_stats, orient='index').reset_index()
    df.rename(columns={'index': 'category', 'affected': 'affected_count'}, inplace=True)
    df_sorted = df.sort_values('affected_count', ascending=False)

    plt.figure(figsize=(12, 10))
    sns.barplot(x='affected_count', y='category', data=df_sorted, palette='viridis')
    plt.title('Number of Affected Questions by Category', fontsize=16)
    plt.xlabel('Number of Questions with Non-Zero Influence', fontsize=12)
    plt.ylabel('Category', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Saved chart: {output_filename}")

def plot_multi_dimensional_analysis(data, output_filename="multi_dimensional_analysis.png"):
    """
    Generates and saves a bubble chart for multi-dimensional adjective analysis.
    """
    df = pd.DataFrame.from_dict(data['aggregated_results'], orient='index').reset_index()
    df.rename(columns={'index': 'adjective'}, inplace=True)

    colors = ['#2E8B57' if x >= 0 else '#FF6347' for x in df['mean_shapley']] # SeaGreen and Tomato
    sizes = df['mean_abs_shapley'] * 5000 

    plt.figure(figsize=(14, 10))
    plt.scatter(df['mean_shapley'], df['std_shapley'], s=sizes, c=colors, alpha=0.7, edgecolors="w", linewidth=0.5)
    
    plt.title('Multi-dimensional Adjective Analysis', fontsize=18, pad=20)
    plt.xlabel('Mean Shapley Value (Directional Influence: Negative vs. Positive)', fontsize=14)
    plt.ylabel('Standard Deviation (Variability of Influence)', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    p1 = plt.scatter([], [], s=0.1*5000, c='#483D8B', alpha=0.7, label='Low Influence')
    p2 = plt.scatter([], [], s=0.4*5000, c='#483D8B', alpha=0.7, label='Medium Influence')
    p3 = plt.scatter([], [], s=0.8*5000, c='#483D8B', alpha=0.7, label='High Influence')
    plt.legend(handles=[p1, p2, p3], title='Bubble Size (Absolute Influence)', loc='upper right', labelspacing=2)

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Saved chart: {output_filename}")

def plot_influence_variability(data, output_filename="influence_variability.png"):
    """
    Generates a bar chart of adjectives with the most unpredictable effects.
    """
    df = pd.DataFrame.from_dict(data['aggregated_results'], orient='index').reset_index()
    df.rename(columns={'index': 'adjective'}, inplace=True)
    df['variability'] = df['mean_abs_shapley'] - df['mean_shapley'].abs()
    df_sorted = df.sort_values('variability', ascending=False)
    
    plt.figure(figsize=(12, 20))
    sns.barplot(x='variability', y='adjective', data=df_sorted, palette='plasma')
    plt.title('Adjective Influence Variability (Unpredictability)', fontsize=16)
    plt.xlabel('Variability Score (Mean Abs Shapley - |Mean Shapley|)', fontsize=12)
    plt.ylabel('Adjective', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Saved chart: {output_filename}")

def plot_aggregated_influence(data, output_filename="aggregated_influence.png"):
    """
    Generates a sorted bar chart of aggregated adjective influence.
    """
    df = pd.DataFrame.from_dict(data['aggregated_results'], orient='index').reset_index()
    df.rename(columns={'index': 'adjective'}, inplace=True)
    df_sorted = df.sort_values('mean_abs_shapley', ascending=False)
    
    colors = ['#2E8B57' if x >= 0 else '#FF6347' for x in df_sorted['mean_shapley']]

    plt.figure(figsize=(12, 20))
    sns.barplot(x='mean_abs_shapley', y='adjective', data=df_sorted, palette=colors)
    plt.title('Overall Influence of Adjectives (Sorted by Magnitude)', fontsize=16)
    plt.xlabel('Mean Absolute Shapley Value (Overall Impact)', fontsize=12)
    plt.ylabel('Adjective', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Saved chart: {output_filename}")

def plot_per_question_analysis(data, question_key, output_filename="per_question_analysis.png"):
    """
    Generates a bar chart for the Shapley values of a single, specified question.
    """
    reverse_mapping = {v: k for k, v in data['adjective_mapping'].items()}
    shapley_values = data['per_question_results'].get(question_key)

    if not shapley_values:
        print(f"Warning: Question key '{question_key}' not found or has no data. Skipping plot.")
        return

    df_data = [{'adjective': reverse_mapping.get(i, f'Unknown_{i}'), 'shapley_value': v} for i, v in enumerate(shapley_values)]
    df = pd.DataFrame(df_data)
    
    # Filter out zero-value adjectives and sort
    df_filtered = df[df['shapley_value'].abs() > 1e-9].copy()
    df_filtered['abs_shapley'] = df_filtered['shapley_value'].abs()
    df_sorted = df_filtered.sort_values('abs_shapley', ascending=False)

    if df_sorted.empty:
        print(f"Note: No influential adjectives found for question '{question_key}'. Skipping plot.")
        return

    colors = ['#2E8B57' if x >= 0 else '#FF6347' for x in df_sorted['shapley_value']]

    plt.figure(figsize=(12, 10))
    sns.barplot(x='shapley_value', y='adjective', data=df_sorted, palette=colors)
    plt.title(f'Adjective Influence for Question: {question_key}', fontsize=16)
    plt.xlabel('Shapley Value', fontsize=12)
    plt.ylabel('Adjective', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Saved chart: {output_filename}")


def main():
    """Main function to parse arguments and generate all plots."""
    parser = argparse.ArgumentParser(description="Generate publication-quality charts from adjective influence data.")
    parser.add_argument("json_file", type=str, help="Path to the input JSON data file.")
    args = parser.parse_args()

    # Load data and set plot style
    raw_data = load_data(args.json_file)
    set_plot_style()

    # Generate and save all plots
    plot_question_impact_pie(raw_data, "1_question_impact_pie.png")
    plot_category_impact(raw_data, "2_category_impact_barchart.png")
    plot_multi_dimensional_analysis(raw_data, "3_multi_dimensional_bubble_chart.png")
    plot_influence_variability(raw_data, "4_influence_variability_barchart.png")
    plot_aggregated_influence(raw_data, "5_aggregated_influence_barchart.png")
    
    # For the per-question plot, we find the most impacted question to use as an example.
    # A user could loop through all keys in raw_data['per_question_results'] to generate all plots.
    question_impact = {
        key: sum(abs(v) for v in values) 
        for key, values in raw_data['per_question_results'].items()
    }
    most_impactful_question = max(question_impact, key=question_impact.get) if question_impact else None
    
    if most_impactful_question:
        plot_per_question_analysis(raw_data, most_impactful_question, f"6_per_question_{most_impactful_question}.png")
    else:
        print("No questions with influence data found to generate a per-question plot.")


if __name__ == "__main__":
    # Example usage from command line:
    # python your_script_name.py path/to/your/data.json
    main()
