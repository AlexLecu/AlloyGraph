#!/usr/bin/env python3
"""
AlloyGraph Results Analyzer v2

Comprehensive metrics calculation and visualization for evaluation results.

Metrics calculated:
- R², MAE, RMSE, MAPE per property
- Percentage within tolerance bands (±10%, ±15%, ±20%, ±25%)
- Metrics stratified by confidence level
- Metrics stratified by processing type
- Error distribution analysis

Usage:
    python analyze_results.py --input output/predictions.csv
    python analyze_results.py --input output/predictions.csv --compare output/predictions_ml_only.csv
"""

import os
import json
import argparse
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# Property configurations
PROPERTIES = {
    'yield_strength': {
        'pred_col': 'pred_ys',
        'actual_col': 'actual_ys',
        'label': 'Yield Strength',
        'unit': 'MPa',
        'acceptable_mape': 20,
        'acceptable_r2': 0.7
    },
    'tensile_strength': {
        'pred_col': 'pred_uts',
        'actual_col': 'actual_uts',
        'label': 'Tensile Strength',
        'unit': 'MPa',
        'acceptable_mape': 15,
        'acceptable_r2': 0.7
    },
    'elongation': {
        'pred_col': 'pred_el',
        'actual_col': 'actual_el',
        'label': 'Elongation',
        'unit': '%',
        'acceptable_mape': 30,
        'acceptable_r2': 0.5
    },
    'elastic_modulus': {
        'pred_col': 'pred_em',
        'actual_col': 'actual_em',
        'label': 'Elastic Modulus',
        'unit': 'GPa',
        'acceptable_mape': 10,
        'acceptable_r2': 0.6
    }
}


def calculate_metrics(actual, predicted, property_name=''):
    """Calculate comprehensive metrics for a property."""
    # Filter valid pairs
    mask = ~(np.isnan(actual) | np.isnan(predicted) | (actual == 0))
    actual = actual[mask]
    predicted = predicted[mask]

    if len(actual) < 2:
        return None

    # Basic metrics
    r2 = r2_score(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    # Percentage within tolerance bands
    pct_error = np.abs((actual - predicted) / actual) * 100
    within_10 = (pct_error <= 10).sum() / len(actual) * 100
    within_15 = (pct_error <= 15).sum() / len(actual) * 100
    within_20 = (pct_error <= 20).sum() / len(actual) * 100
    within_25 = (pct_error <= 25).sum() / len(actual) * 100

    # Error statistics
    errors = predicted - actual
    mean_error = np.mean(errors)  # Bias
    std_error = np.std(errors)

    return {
        'property': property_name,
        'n_samples': len(actual),
        'R2': round(r2, 4),
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'MAPE': round(mape, 2),
        'within_10pct': round(within_10, 1),
        'within_15pct': round(within_15, 1),
        'within_20pct': round(within_20, 1),
        'within_25pct': round(within_25, 1),
        'mean_error': round(mean_error, 2),
        'std_error': round(std_error, 2),
        'mean_actual': round(np.mean(actual), 2),
        'mean_predicted': round(np.mean(predicted), 2)
    }


def analyze_by_confidence(df, prop_config):
    """Analyze metrics stratified by confidence level."""
    results = []

    for conf_level in ['HIGH', 'MEDIUM', 'LOW', 'VERY LOW']:
        subset = df[df['confidence_level'] == conf_level]

        if len(subset) < 2:
            continue

        actual = subset[prop_config['actual_col']].values
        predicted = subset[prop_config['pred_col']].values

        metrics = calculate_metrics(actual, predicted, f"{prop_config['label']} ({conf_level})")
        if metrics:
            metrics['confidence_level'] = conf_level
            results.append(metrics)

    return results


def analyze_by_processing(df, prop_config):
    """Analyze metrics stratified by processing type."""
    results = []

    for processing in ['wrought', 'cast', 'forged']:
        subset = df[df['processing'] == processing]

        if len(subset) < 2:
            continue

        actual = subset[prop_config['actual_col']].values
        predicted = subset[prop_config['pred_col']].values

        metrics = calculate_metrics(actual, predicted, f"{prop_config['label']} ({processing})")
        if metrics:
            metrics['processing'] = processing
            results.append(metrics)

    return results


def create_scatter_plots(df, output_dir, suffix=''):
    """Create predicted vs actual scatter plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    confidence_colors = {
        'HIGH': '#2ecc71',
        'MEDIUM': '#3498db',
        'LOW': '#e67e22',
        'VERY LOW': '#e74c3c',
        'UNKNOWN': '#95a5a6'
    }

    for ax, (prop_name, config) in zip(axes.flat, PROPERTIES.items()):
        pred_col = config['pred_col']
        actual_col = config['actual_col']

        if pred_col not in df.columns or actual_col not in df.columns:
            ax.text(0.5, 0.5, f'No data for {config["label"]}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(config['label'])
            continue

        subset = df[[pred_col, actual_col, 'confidence_level']].dropna()

        if len(subset) < 2:
            ax.text(0.5, 0.5, f'Insufficient data for {config["label"]}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(config['label'])
            continue

        # Plot by confidence level
        for conf_level, color in confidence_colors.items():
            conf_data = subset[subset['confidence_level'] == conf_level]
            if len(conf_data) > 0:
                ax.scatter(
                    conf_data[actual_col],
                    conf_data[pred_col],
                    c=color,
                    label=conf_level,
                    alpha=0.7,
                    edgecolors='black',
                    linewidth=0.5,
                    s=60
                )

        # Perfect prediction line
        all_vals = pd.concat([subset[actual_col], subset[pred_col]])
        min_val, max_val = all_vals.min(), all_vals.max()
        margin = (max_val - min_val) * 0.05
        ax.plot([min_val - margin, max_val + margin],
                [min_val - margin, max_val + margin],
                'k--', alpha=0.5, label='Perfect')

        # ±20% bands
        x_line = np.linspace(min_val, max_val, 100)
        ax.fill_between(x_line, x_line * 0.8, x_line * 1.2, alpha=0.1, color='green')

        # Calculate metrics for title
        actual = subset[actual_col].values
        predicted = subset[pred_col].values
        r2 = r2_score(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100

        ax.set_xlabel(f'Actual {config["label"]} ({config["unit"]})')
        ax.set_ylabel(f'Predicted {config["label"]} ({config["unit"]})')
        ax.set_title(f'{config["label"]}\nR² = {r2:.3f}, MAPE = {mape:.1f}%, n = {len(subset)}')
        ax.legend(loc='upper left', fontsize=8)

    plt.suptitle('Predicted vs Actual (colored by confidence, shaded ±20% band)',
                 fontsize=12, y=1.02)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f'scatter_plots{suffix}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


def create_error_distribution(df, output_dir, suffix=''):
    """Create error distribution histograms."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, (prop_name, config) in zip(axes.flat, PROPERTIES.items()):
        pred_col = config['pred_col']
        actual_col = config['actual_col']

        if pred_col not in df.columns or actual_col not in df.columns:
            continue

        subset = df[[pred_col, actual_col]].dropna()
        if len(subset) < 2:
            continue

        # Calculate percentage errors
        pct_errors = (subset[pred_col] - subset[actual_col]) / subset[actual_col] * 100

        ax.hist(pct_errors, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
        ax.axvline(pct_errors.mean(), color='blue', linestyle='-', linewidth=2,
                   label=f'Mean: {pct_errors.mean():.1f}%')

        ax.set_xlabel('Percentage Error (%)')
        ax.set_ylabel('Count')
        ax.set_title(f'{config["label"]} Error Distribution\nStd: {pct_errors.std():.1f}%')
        ax.legend()

    plt.tight_layout()

    output_path = os.path.join(output_dir, f'error_distribution{suffix}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


def worst_predictions(df, n=10):
    """Find worst predictions by percentage error."""
    worst = []

    for prop_name, config in PROPERTIES.items():
        pred_col = config['pred_col']
        actual_col = config['actual_col']

        if pred_col not in df.columns or actual_col not in df.columns:
            continue

        subset = df[['alloy', 'temperature', pred_col, actual_col, 'confidence_level']].dropna()
        if len(subset) == 0:
            continue

        subset = subset.copy()
        subset['pct_error'] = np.abs(subset[pred_col] - subset[actual_col]) / subset[actual_col] * 100
        subset['property'] = prop_name

        top_worst = subset.nlargest(n, 'pct_error')
        worst.append(top_worst)

    if worst:
        return pd.concat(worst).sort_values('pct_error', ascending=False).head(n * 2)
    return pd.DataFrame()


def compare_methods(df1, df2, label1='Method 1', label2='Method 2'):
    """Compare metrics between two prediction methods."""
    comparison = []

    for prop_name, config in PROPERTIES.items():
        pred_col = config['pred_col']
        actual_col = config['actual_col']

        for df, label in [(df1, label1), (df2, label2)]:
            if pred_col not in df.columns or actual_col not in df.columns:
                continue

            subset = df[[pred_col, actual_col]].dropna()
            if len(subset) < 2:
                continue

            metrics = calculate_metrics(
                subset[actual_col].values,
                subset[pred_col].values,
                config['label']
            )
            if metrics:
                metrics['method'] = label
                comparison.append(metrics)

    return pd.DataFrame(comparison)


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze AlloyGraph prediction results')
    parser.add_argument('--input', type=str, required=True,
                        help='Input predictions CSV file')
    parser.add_argument('--compare', type=str, default=None,
                        help='Optional second file for comparison (e.g., ML-only)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: same as input)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load data
    print(f"\nLoading {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} predictions")

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(args.input)
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Overall metrics
    print("\n" + "=" * 80)
    print("OVERALL METRICS")
    print("=" * 80)

    all_metrics = []
    for prop_name, config in PROPERTIES.items():
        pred_col = config['pred_col']
        actual_col = config['actual_col']

        if pred_col not in df.columns or actual_col not in df.columns:
            print(f"\n{config['label']}: No data available")
            continue

        subset = df[[pred_col, actual_col]].dropna()
        if len(subset) < 2:
            print(f"\n{config['label']}: Insufficient data (n={len(subset)})")
            continue

        metrics = calculate_metrics(
            subset[actual_col].values,
            subset[pred_col].values,
            config['label']
        )

        if metrics:
            all_metrics.append(metrics)

            # Check against acceptable thresholds
            r2_status = "PASS" if metrics['R2'] >= config['acceptable_r2'] else "FAIL"
            mape_status = "PASS" if metrics['MAPE'] <= config['acceptable_mape'] else "FAIL"

            print(f"\n{config['label']} ({config['unit']}):")
            print(f"  n = {metrics['n_samples']}")
            print(f"  R² = {metrics['R2']:.4f} [{r2_status}, threshold: {config['acceptable_r2']}]")
            print(f"  MAE = {metrics['MAE']:.2f} {config['unit']}")
            print(f"  RMSE = {metrics['RMSE']:.2f} {config['unit']}")
            print(f"  MAPE = {metrics['MAPE']:.2f}% [{mape_status}, threshold: {config['acceptable_mape']}%]")
            print(f"  Within ±10%: {metrics['within_10pct']:.1f}%")
            print(f"  Within ±20%: {metrics['within_20pct']:.1f}%")
            print(f"  Bias (mean error): {metrics['mean_error']:.2f}")

    # Save metrics
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = os.path.join(output_dir, f'metrics_{timestamp}.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\nSaved: {metrics_path}")

    # Metrics by confidence level
    print("\n" + "=" * 80)
    print("METRICS BY CONFIDENCE LEVEL")
    print("=" * 80)

    conf_metrics = []
    for prop_name, config in PROPERTIES.items():
        results = analyze_by_confidence(df, config)
        conf_metrics.extend(results)

        for m in results:
            print(f"\n{m['property']}:")
            print(f"  n = {m['n_samples']}, R² = {m['R2']:.3f}, MAPE = {m['MAPE']:.1f}%")

    if conf_metrics:
        conf_df = pd.DataFrame(conf_metrics)
        conf_path = os.path.join(output_dir, f'metrics_by_confidence_{timestamp}.csv')
        conf_df.to_csv(conf_path, index=False)

    # Metrics by processing
    print("\n" + "=" * 80)
    print("METRICS BY PROCESSING TYPE")
    print("=" * 80)

    proc_metrics = []
    for prop_name, config in PROPERTIES.items():
        results = analyze_by_processing(df, config)
        proc_metrics.extend(results)

        for m in results:
            print(f"\n{m['property']}:")
            print(f"  n = {m['n_samples']}, R² = {m['R2']:.3f}, MAPE = {m['MAPE']:.1f}%")

    # Create visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    create_scatter_plots(df, output_dir, f'_{timestamp}')
    create_error_distribution(df, output_dir, f'_{timestamp}')

    # Worst predictions
    print("\n" + "=" * 80)
    print("WORST PREDICTIONS (by % error)")
    print("=" * 80)

    worst = worst_predictions(df, n=10)
    if len(worst) > 0:
        print(worst[['alloy', 'property', 'temperature', 'pct_error', 'confidence_level']].to_string(index=False))

    # Comparison with another method
    if args.compare and os.path.exists(args.compare):
        print("\n" + "=" * 80)
        print("METHOD COMPARISON")
        print("=" * 80)

        df2 = pd.read_csv(args.compare)
        comparison = compare_methods(df, df2, 'Full System', 'ML-Only')

        if len(comparison) > 0:
            print(comparison[['property', 'method', 'R2', 'MAPE', 'within_20pct']].to_string(index=False))

            comp_path = os.path.join(output_dir, f'comparison_{timestamp}.csv')
            comparison.to_csv(comp_path, index=False)
            print(f"\nSaved: {comp_path}")

    # Summary report
    report = {
        'timestamp': timestamp,
        'input_file': args.input,
        'total_predictions': len(df),
        'unique_alloys': df['alloy'].nunique() if 'alloy' in df.columns else 0,
        'metrics': all_metrics,
        'by_confidence': conf_metrics,
        'by_processing': proc_metrics
    }

    report_path = os.path.join(output_dir, f'analysis_report_{timestamp}.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved: {report_path}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
