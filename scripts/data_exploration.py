import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
import os
import numpy as np

# Add the path to the DataLoader script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.data_loader import DataLoader

def explore_experiment(experiment_name, data_loader):
    experiment_data = data_loader.load_experiment_data(experiment_name)
    # Create directories for artifacts if they don't exist
    artifacts_dir = os.path.join('artifacts', experiment_name)
    plots_dir = os.path.join(artifacts_dir, 'plots')
    stats_dir = os.path.join(artifacts_dir, 'stats')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    # Save Descriptive Statistics
    stats = experiment_data.describe()
    stats.to_csv(os.path.join(stats_dir, f'{experiment_name}_statistics.csv'))

    # Save Missing Values
    missing_values = experiment_data.isnull().sum()
    missing_values.to_csv(os.path.join(stats_dir, f'{experiment_name}_missing_values.csv'))

    # Time Series Plot
    plt.figure(figsize=(12, 6))
    plt.plot(experiment_data['time'], experiment_data['data'])  # Using 'time' column for x-axis
    plt.title(f'Time Series - {experiment_name}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Signal')
    plt.savefig(os.path.join(plots_dir, f'{experiment_name}_timeseries.png'))
    plt.close()

    # Signal Derivative
    derivative = np.diff(experiment_data['data'].to_numpy())
    plt.figure(figsize=(12, 6))
    plt.plot(derivative, label='Derivative of Signal')
    plt.title(f'Signal Derivative - {experiment_name}')
    plt.xlabel('Time')
    plt.ylabel('Derivative')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f'{experiment_name}_derivative.png'))
    plt.close()

    # Distribution Plot
    plt.figure(figsize=(10, 5))
    sns.histplot(experiment_data['data'], kde=True)
    plt.title(f'Data Distribution - {experiment_name}')
    plt.xlabel('Signal')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(plots_dir, f'{experiment_name}_distribution.png'))
    plt.close()

    # Count rows per measurement
    measurement_counts = experiment_data.groupby('measurement').count()

    # Save the counts to a CSV file
    measurement_counts.to_csv(os.path.join(stats_dir, f'{experiment_name}_measurement_counts.csv'))

    print(f"Row counts for {experiment_name} measurements saved to '{experiment_name}_measurement_counts.csv'.")

    # Count duplicates for each measurement
    measurement_duplicates = experiment_data.groupby('measurement').apply(lambda x: x.duplicated().sum())
    print(f"Duplicates per measurement in {experiment_name}:")
    print(measurement_duplicates)

    # Boxplot
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=experiment_data['data'])
    plt.title(f'Data Spread - {experiment_name}')
    plt.xlabel('Signal')
    plt.savefig(os.path.join(plots_dir, f'{experiment_name}_boxplot.png'))
    plt.close()

    # Correlation Matrix
    numeric_cols = experiment_data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(experiment_data[numeric_cols].corr(), annot=True, fmt=".2f")
        plt.title(f'Correlation Matrix - {experiment_name}')
        plt.savefig(os.path.join(plots_dir, f'{experiment_name}_correlation_matrix.png'))
        plt.close()

    # Create subfolder for measurement time series plots
    measurement_plots_dir = os.path.join(plots_dir, 'measurement_ts')
    os.makedirs(measurement_plots_dir, exist_ok=True)

    # Create time series plot for each measurement
    for measurement in experiment_data['measurement'].unique():
        measurement_data = experiment_data[experiment_data['measurement'] == measurement]
        plt.figure(figsize=(12, 6))
        plt.plot(measurement_data['time'], measurement_data['data'])
        plt.title(f'Time Series - {experiment_name} - Measurement {measurement}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Signal')
        plt.savefig(os.path.join(measurement_plots_dir, f'{experiment_name}_measurement_{measurement}_timeseries.png'))
        plt.close()


def main():
    # Set plot style
    sns.set_style("darkgrid")

    # Initialize DataLoader
    data_loader = DataLoader('config.json')

    # List of experiments to explore
    experiments = data_loader.config['experiments'].keys()

    for experiment_name in experiments:
        explore_experiment(experiment_name, data_loader)

if __name__ == "__main__":
    main()
