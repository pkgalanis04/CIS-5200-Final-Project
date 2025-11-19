"""
Evaluation metrics and comparison functions for spaced repetition models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
import os


def compute_recall_rate(results: List[Dict]) -> float:
    """
    Compute average recall rate across episodes.
    
    Args:
        results: List of episode results with 'recall_rate' key
        
    Returns:
        Average recall rate
    """
    if not results:
        return 0.0
    return np.mean([r['recall_rate'] for r in results])


def compute_intervention_efficiency(results: List[Dict]) -> float:
    """
    Compute intervention efficiency: average time between correct recalls.
    Lower is better (more efficient).
    
    Args:
        results: List of episode results
        
    Returns:
        Average intervention efficiency
    """
    if not results:
        return np.inf
    
    efficiencies = []
    for result in results:
        # For simplicity, use inverse of recall rate as proxy
        # Higher recall rate = more efficient
        if result['recall_rate'] > 0:
            efficiency = 1.0 / result['recall_rate']
        else:
            efficiency = np.inf
        efficiencies.append(efficiency)
    
    return np.mean(efficiencies)


def compute_cumulative_reward(results: List[Dict]) -> float:
    """
    Compute average cumulative reward across episodes.
    
    Args:
        results: List of episode results with 'cumulative_reward' key
        
    Returns:
        Average cumulative reward
    """
    if not results:
        return 0.0
    return np.mean([r['cumulative_reward'] for r in results])


def evaluate_model(results: List[Dict], model_name: str) -> Dict:
    """
    Evaluate a model and return all metrics.
    
    Args:
        results: List of episode results
        model_name: Name of the model
        
    Returns:
        Dictionary with evaluation metrics
    """
    return {
        'model': model_name,
        'recall_rate': compute_recall_rate(results),
        'intervention_efficiency': compute_intervention_efficiency(results),
        'cumulative_reward': compute_cumulative_reward(results),
        'num_episodes': len(results)
    }


def compare_models(all_results: Dict[str, List[Dict]], output_dir: str = 'results') -> pd.DataFrame:
    """
    Compare multiple models and create comparison table.
    
    Args:
        all_results: Dictionary mapping model names to their results
        output_dir: Directory to save comparison table
        
    Returns:
        DataFrame with comparison metrics
    """
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    comparison_data = []
    
    for model_name, results in all_results.items():
        metrics = evaluate_model(results, model_name)
        comparison_data.append(metrics)
        
        print(f"\n{model_name}:")
        print(f"  Recall Rate: {metrics['recall_rate']:.4f}")
        print(f"  Intervention Efficiency: {metrics['intervention_efficiency']:.4f}")
        print(f"  Cumulative Reward: {metrics['cumulative_reward']:.2f}")
        print(f"  Episodes: {metrics['num_episodes']}")
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'model_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nComparison table saved to {csv_path}")
    
    return df


def plot_training_curves(all_results: Dict[str, List[Dict]], 
                        metric: str = 'recall_rate',
                        output_dir: str = 'results'):
    """
    Plot training curves for all models.
    
    Args:
        all_results: Dictionary mapping model names to their results
        metric: Metric to plot ('recall_rate', 'cumulative_reward')
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    for model_name, results in all_results.items():
        episodes = [r['episode'] for r in results]
        values = [r[metric] for r in results]
        
        # Smooth with moving average
        window = max(1, len(values) // 20)
        if window > 1:
            values_smooth = pd.Series(values).rolling(window=window, center=True).mean()
            plt.plot(episodes, values_smooth, label=model_name, linewidth=2)
        else:
            plt.plot(episodes, values, label=model_name, linewidth=2)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.title(f'Training Curves: {metric.replace("_", " ").title()}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'training_curves_{metric}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training curve saved to {plot_path}")
    plt.close()


def plot_comparison_bar_chart(comparison_df: pd.DataFrame, 
                              metric: str = 'recall_rate',
                              output_dir: str = 'results'):
    """
    Plot bar chart comparing models on a metric.
    
    Args:
        comparison_df: DataFrame with model comparison
        metric: Metric to plot
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    models = comparison_df['model'].values
    values = comparison_df[metric].values
    
    bars = plt.bar(models, values, alpha=0.7, edgecolor='black')
    
    # Color bars
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, bar in enumerate(bars):
        bar.set_color(colors[i % len(colors)])
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.title(f'Model Comparison: {metric.replace("_", " ").title()}', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'comparison_{metric}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison chart saved to {plot_path}")
    plt.close()


def create_full_evaluation(all_results: Dict[str, List[Dict]], 
                          output_dir: str = 'results'):
    """
    Create full evaluation with comparison table and plots.
    
    Args:
        all_results: Dictionary mapping model names to their results
        output_dir: Directory to save all outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comparison table
    comparison_df = compare_models(all_results, output_dir)
    
    # Plot training curves
    plot_training_curves(all_results, 'recall_rate', output_dir)
    plot_training_curves(all_results, 'cumulative_reward', output_dir)
    
    # Plot comparison bar charts
    plot_comparison_bar_chart(comparison_df, 'recall_rate', output_dir)
    plot_comparison_bar_chart(comparison_df, 'cumulative_reward', output_dir)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    
    return comparison_df

