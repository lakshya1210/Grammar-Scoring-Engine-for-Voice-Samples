import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_score_distribution(df, score_column='grammar_score'):
    """Plot the distribution of grammar scores."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df[score_column].dropna(), bins=20, kde=True)
    plt.title('Distribution of Grammar Scores')
    plt.xlabel('Score (0-100)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_error_categories(df, n_categories=10):
    """Plot the most common error categories."""
    # Extract error categories from all samples
    all_categories = {}
    
    for _, row in df.iterrows():
        if pd.isna(row['grammar_features']) or not isinstance(row['grammar_features'], dict):
            continue
            
        error_categories = row['grammar_features'].get('error_categories', {})
        for category, count in error_categories.items():
            if category in all_categories:
                all_categories[category] += count
            else:
                all_categories[category] = count
    
    # Create DataFrame for visualization
    categories_df = pd.DataFrame({
        'Category': list(all_categories.keys()),
        'Count': list(all_categories.values())
    })
    
    # Sort and select top categories
    categories_df = categories_df.sort_values('Count', ascending=False).head(n_categories)
    
    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Count', y='Category', data=categories_df)
    plt.title(f'Top {n_categories} Grammar Error Categories')
    plt.xlabel('Error Count')
    plt.tight_layout()
    plt.show()

def visualize_results(results_df):
    """Generate standard visualizations for grammar analysis results."""
    if len(results_df) == 0:
        print("No data to visualize")
        return
        
    print(f"Analyzing {len(results_df)} samples")
    
    # Score distribution
    plot_score_distribution(results_df)
    
    # Error rate vs score
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['error_rate'], results_df['grammar_score'], alpha=0.6)
    plt.title('Error Rate vs Grammar Score')
    plt.xlabel('Error Rate')
    plt.ylabel('Grammar Score')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Error categories
    plot_error_categories(results_df)
    
    # Summary statistics
    print("\nSummary Statistics:")
    print(results_df[['grammar_score', 'error_count', 'error_rate']].describe())