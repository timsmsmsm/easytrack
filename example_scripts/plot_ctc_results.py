"""
Compare tracking results from benchmark_results.csv with comparison plots.

This script generates multiple plots showing:
1. Overall metric comparison (bar plots with error bars)
2. Per-dataset comparison for each metric
3. Metric distributions (scatter plots)
4. Performance differences between methods
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

from pandas.core.interchange.dataframe_protocol import DataFrame

# Configure plotting
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

# Configuration
COLORS = {
    'easytrack': '#1f77b4',          # Blue
    'btrack': '#ff7f0e',             # Orange
    'train': '#2ca02c',              # Green
    'test': '#d62728',               # Red
    'Random': '#9467bd',             # Purple
    'Trained from Random': '#8c564b', # Brown
}
METRIC_CONFIGS = {
    'TRA': {'label': 'TRA (Tracking)', 'is_higher_better': True},
    'DET': {'label': 'DET (Detection)', 'is_higher_better': True},
    'AOGM': {'label': 'AOGM (Overlap)', 'is_higher_better': False},
}


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load and validate benchmark data."""
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} rows from {csv_path}")
    print(f"  Datasets: {len(df['dataset'].unique())} unique")
    if 'method' in df.columns:
        print(f"  Methods: {list(df['method'].unique())}")
    print(f"  Metrics: {', '.join([m for m in METRIC_CONFIGS.keys() if m in df.columns])}")
    return df


def compute_metric_by_dataset(df: pd.DataFrame, metric: str) -> Dict[str, Dict]:
    """Compute metric values for each dataset and method."""
    datasets = sorted(df['dataset'].unique())
    methods = sorted(df['method'].unique())
    
    results = {method: [] for method in methods}
    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]
        for method in methods:
            value = dataset_df[dataset_df['method'] == method][metric].mean()
            results[method].append(value)
    
    return {
        'datasets': datasets,
        'methods': methods,
        'data': results
    }


def plot_overall_comparison(ax, df: pd.DataFrame, metrics: List[str], methods: List[str]) -> None:
    """Plot overall metric comparison across all datasets."""
    x = np.arange(len(metrics))
    width = 0.35
    
    for i, method in enumerate(methods):
        method_df = df[df['method'] == method]
        means = [method_df[m].mean() for m in metrics]
        ax.bar(x + i * width, means, width, label=method, alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Metric', fontsize=20, fontweight='bold')
    ax.set_ylabel('Mean Score', fontsize=20, fontweight='bold')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)


def plot_metric_by_dataset(ax, metric_data: Dict, metric: str, metric_config: Dict) -> None:
    """Plot per-dataset comparison for a specific metric (publication-ready)."""
    datasets = metric_data['datasets']
    methods = metric_data['methods']
    data = metric_data['data']
    
    x = np.arange(len(datasets))

    # Calculate bar width based on number of methods
    num_methods = len(methods)
    width = max(0.08, 0.7 / num_methods)

    # Plot bars for each method with proper coloring
    for i, method in enumerate(methods):
        # Center the bars around each dataset position
        offset = (i - num_methods / 2 + 0.5) * width
        color = COLORS.get(method, f'C{i}')

        # Prepare bar heights with minimum value for zero values
        bar_heights = []
        for val in data[method]:
            if np.isnan(val):
                bar_heights.append(np.nan)
            elif val == 0.0:
                bar_heights.append(0.02)  # Minimum height for visibility
            else:
                bar_heights.append(val)

        bars = ax.bar(x + offset, bar_heights, width, label=method,
                     alpha=0.85, edgecolor='black', linewidth=1.2, color=color)

        # Add value labels on top of bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            original_value = data[method][j]
            if not np.isnan(original_value):
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                       f'{original_value:.4f}', ha='center', va='bottom', fontsize=8)

    # Styling
    ax.set_xlabel('Dataset', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel(f'{metric} Score', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_yscale('log' if not metric_config['is_higher_better'] else 'linear')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=11)
    ax.tick_params(axis='y', labelsize=11)

    # Legend styling
    ax.legend(loc='lower left', fontsize=11, frameon=True, fancybox=True,
             shadow=True, edgecolor='gray', framealpha=0.95)

    # Grid styling
    ax.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)

    # Set y-axis limits based on metric type
    if metric_config['is_higher_better']:
        ax.set_ylim([0.7, 1.01])

    # Improve layout
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)


def plot_scatter_metrics(ax, df: pd.DataFrame, metric1: str = 'DET', metric2: str = 'TRA') -> None:
    """Plot scatter plot comparing two metrics, colored by method."""
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        ax.scatter(method_df[metric1], method_df[metric2],
                  label=method, alpha=0.6, s=100, color=COLORS.get(method, 'gray'))
    
    ax.set_xlabel(f'{metric1} Score', fontsize=20, fontweight='bold')
    ax.set_ylabel(f'{metric2} Score', fontsize=20, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])


def plot_performance_difference(ax, metric_data_list: Dict) -> None:
    """Plot performance difference between first two methods."""
    METRIC_COLORS = {'TRA': 'Green', 'DET': 'Orange'}
    # Get first two methods (assuming we're comparing two)
    methods = metric_data_list[list(metric_data_list.keys())[0]]['methods']
    if len(methods) != 2:
        ax.text(0.5, 0.5, 'Comparison requires exactly 2 methods', 
               ha='center', va='center', transform=ax.transAxes)
        return
    
    method1, method2 = methods[0], methods[1]
    datasets = metric_data_list[list(metric_data_list.keys())[0]]['datasets']
    metrics = list(metric_data_list.keys())
    
    x = np.arange(len(datasets))
    width = 0.35
    
    # Plot bars with distinct colors for TRA and DET
    for i, metric in enumerate(metrics[:2]):
        data = metric_data_list[metric]['data']
        diff = np.array(data[method1]) - np.array(data[method2])
        offset = (i - len(metrics) / 2 + 0.5) * width
        color = METRIC_COLORS[metric]
        ax.bar(x + offset, diff, width, label=f'{metric}', alpha=0.85, color=color, edgecolor='black')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Dataset', fontsize=20, fontweight='bold')
    ax.set_ylabel(f'Score Difference ({method1} - {method2})', fontsize=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)


def print_summary_statistics(df: pd.DataFrame, metrics: List[str]) -> None:
    """Print comprehensive summary statistics."""
    methods = sorted(df['method'].unique())
    
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    for metric in metrics:
        config = METRIC_CONFIGS.get(metric, {})
        print(f"\n{metric} Metric: {config.get('label', metric)}")
        
        metric_stats = []
        for method in methods:
            method_df = df[df['method'] == method]
            values = method_df[metric].values
            mean = values.mean()
            std = values.std()
            min_val = values.min()
            max_val = values.max()
            
            metric_stats.append((method, mean, std, min_val, max_val, values))
            print(f"  {method:20s}: {mean:8.4f} ± {std:.4f} (min: {min_val:.4f}, max: {max_val:.4f})")
        
        # Compute pairwise differences
        if len(methods) == 2:
            m1, m2 = methods[0], methods[1]
            diff = metric_stats[0][5] - metric_stats[1][5]
            
            is_higher_better = config.get('is_higher_better', True)
            wins = np.sum(diff > 0) if is_higher_better else np.sum(diff < 0)
            
            print(f"  Difference ({m1} - {m2}): {diff.mean():+.4f} ± {diff.std():.4f}")
            print(f"  {m1} wins: {wins}/{len(diff)}")
    
    print("\n" + "="*70)


def save_plot(filename: str, output_dir: Path = None) -> Path:
    """Save current figure as a high-resolution PNG file."""
    if output_dir is None:
        output_dir = Path(__file__).parent
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved {filename}")
    plt.close()
    return output_path


def plot_restart_robustness(df_restart: pd.DataFrame, output_dir: Path) -> None:
    """Plot restart robustness results grouped by dataset.

    Shows:
    - Random baseline metrics (first trial)
    - Optimised train metrics
    - Optimised test metrics
    Averaged across all restarts for each dataset.
    """
    # Filter out error rows (rows with 'error' column filled)
    df_valid = df_restart[df_restart['error'].isna()].copy()

    if df_valid.empty:
        print("  WARNING: No valid restart robustness results found")
        return

    # Group by dataset and calculate mean metrics
    datasets = sorted(df_valid['dataset'].unique())
    metrics = ['TRA', 'DET', 'AOGM']

    # Prepare data for plotting
    restart_data = {}
    for dataset in datasets:
        dataset_df = df_valid[df_valid['dataset'] == dataset]
        restart_data[dataset] = {
            'random_train': {},
            'opt_train': {},
            'opt_test': {}
        }

        # Random baseline (first trial)
        for metric in metrics:
            col = f'random_train_{metric}'
            if col in dataset_df.columns:
                valid_vals = dataset_df[col].copy()

                if len(valid_vals) > 0:
                    restart_data[dataset]['random_train'][metric] = valid_vals.mean()
                else:
                    # No valid random baseline - likely all timed out
                    restart_data[dataset]['random_train'][metric] = np.nan

        # Optimised train metrics
        for metric in metrics:
            col = f'train_{metric}'
            if col in dataset_df.columns:
                valid_vals = dataset_df[col].copy()

                if len(valid_vals) > 0:
                    restart_data[dataset]['opt_train'][metric] = valid_vals.mean()
                else:
                    restart_data[dataset]['opt_train'][metric] = np.nan

        # Optimised test metrics
        for metric in metrics:
            col = f'test_{metric}'
            if col in dataset_df.columns:
                valid_vals = dataset_df[col].copy()

                if len(valid_vals) > 0:
                    restart_data[dataset]['opt_test'][metric] = valid_vals.mean()
                else:
                    restart_data[dataset]['opt_test'][metric] = np.nan

    # Create comparison plots for each metric
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(datasets))
        width = 0.25

        random_vals = [restart_data[d]['random_train'].get(metric, np.nan) for d in datasets]
        train_vals = [restart_data[d]['opt_train'].get(metric, np.nan) for d in datasets]
        test_vals = [restart_data[d]['opt_test'].get(metric, np.nan) for d in datasets]

        # Plot bars
        ax.bar(x - width, random_vals, width, label='Random (first trial)',
               alpha=0.85, edgecolor='black', color='#9467bd')
        ax.bar(x, train_vals, width, label='Optimised (train)',
               alpha=0.85, edgecolor='black', color='#2ca02c')
        ax.bar(x + width, test_vals, width, label='Optimised (test)',
               alpha=0.85, edgecolor='black', color='#d62728')

        # Add value labels on bars
        for i, (rv, tv, tv2) in enumerate(zip(random_vals, train_vals, test_vals)):
            if not np.isnan(rv):
                ax.text(i - width, rv, f'{rv:.2f}', ha='center', va='bottom', fontsize=9)
            if not np.isnan(tv):
                ax.text(i, tv, f'{tv:.2f}', ha='center', va='bottom', fontsize=9)
            if not np.isnan(tv2):
                ax.text(i + width, tv2, f'{tv2:.2f}', ha='center', va='bottom', fontsize=9)

        # Styling
        ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'{metric} Score', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.legend(fontsize=11, loc='best')
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)

        # Set y-axis limits
        metric_config = METRIC_CONFIGS.get(metric, {})
        if metric_config.get('is_higher_better', True):
            ax.set_ylim([0, 1.05])

        fig.tight_layout()
        save_plot(f"10_restart_robustness_{metric.lower()}_by_dataset.png", output_dir)


def main():
    """Main execution function."""
    csv_path = Path(__file__).parent / "ctc_results" / "ctc_benchmark_results.csv"
    optimised_results_path = Path(__file__).parent / "ctc_results" / "optimisation_results.csv"
    restart_results_path = Path(__file__).parent / "ctc_results" / "restart_robustness_results_v1.csv"
    output_dir = csv_path.parent
    
    # Load and validate data
    df = load_data(csv_path)
    df_optimised = load_data(optimised_results_path)
    df_restart = load_data(restart_results_path)

    # Group df_optimised by eval_type for each dataset and average metrics to compare with df
    if 'eval_type' in df_optimised.columns:
        metrics_to_avg = [m for m in METRIC_CONFIGS.keys() if m in df_optimised.columns]

        # Group by eval_type and dataset, then average the metrics
        df_optimised_grouped = df_optimised.groupby(['eval_type', 'dataset']).agg({
            **{metric: 'mean' for metric in metrics_to_avg},
            'method': 'first'  # Keep the method (should be same for each group)
        }).reset_index()

        df_optimised_grouped['method'] = df_optimised_grouped['eval_type']

        # Use the aggregated version
        df_optimised = df_optimised_grouped

    # Concatenate both df_optimised and df into df
    df = pd.concat([df, df_optimised], ignore_index=True)
    # Save df to a CSV file
    df.to_csv(output_dir / "combined_benchmark_results.csv", index=False)

    # Obtaining metrics
    metrics = list(METRIC_CONFIGS.keys())
    methods = sorted(df['method'].unique())

    eval_type_prefix = ""
    print("\n" + "="*70)
    print("GENERATING INDIVIDUAL PLOTS")
    print("="*70)

    # Pre-compute all metric comparisons for this eval_type
    metric_data_list = {metric: compute_metric_by_dataset(df, metric) for metric in metrics}
    # Save csv for each metric_data
    for metric in metrics:
        metric_df = pd.DataFrame({
            'dataset': metric_data_list[metric]['datasets'],
            **{method: metric_data_list[metric]['data'][method] for method in metric_data_list[metric]['methods']}
        })
        metric_df.to_csv(output_dir / f"{eval_type_prefix}01_metric_{metric.lower()}.csv", index=False)

    # 1. Overall comparison
    plt.figure(figsize=(10, 6))
    plot_overall_comparison(plt.gca(), df, metrics, methods)
    save_plot(f"{eval_type_prefix}01_overall_comparison.png", output_dir)

    # 2-4. Per-dataset comparisons for each metric
    for metric in metrics:
        plt.figure(figsize=(14, 6))
        plot_metric_by_dataset(plt.gca(), metric_data_list[metric], metric, METRIC_CONFIGS[metric])
        filename = f"{eval_type_prefix}02_metric_{metric.lower()}_by_dataset.png"
        save_plot(filename, output_dir)

    # 6. Performance difference
    plt.figure(figsize=(14, 6))
    plot_performance_difference(plt.gca(), metric_data_list)
    save_plot(f"{eval_type_prefix}04_performance_difference.png", output_dir)

    # 7. Restart robustness results (grouped by dataset)
    print("\n" + "="*70)
    print("GENERATING RESTART ROBUSTNESS PLOTS")
    print("="*70)
    plot_restart_robustness(df_restart, output_dir)

    # Print statistics
    print_summary_statistics(df, metrics)


if __name__ == '__main__':
    main()


