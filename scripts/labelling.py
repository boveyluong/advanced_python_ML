import pandas as pd
from pathlib import Path

# Define the start and end times for each experiment's last measurement
time_ranges = {
    'experiment1_measurement_5': {'start': 9.0, 'end': 11.4},
    'experiment2_measurement_6': {'start': 15.0, 'end': 19.6},
    'experiment3_measurement_3': {'start': 6.1, 'end': 7.4},
    'experiment4_measurement_6': {'start': 12.0, 'end': 14.9},
}

# Sampling rate and window length in milliseconds
sampling_rate_hz = 10000  # 10kHz
window_length_ms = 100  # 100ms window

def add_labels_to_features(feature_file: Path, times: dict):
    print(f"Processing file: {feature_file.name}")

    # Read the feature data
    features_data = pd.read_csv(feature_file)
    
    # Calculate the window indices for the start and end times
    start_window_index = int(times['start'] * 1000 / window_length_ms)
    end_window_index = int(times['end'] * 1000 / window_length_ms)
    
    # Initialize the label column to 0
    features_data['label'] = 0
    
    # Label the windows within the start and end times as 1
    features_data.loc[start_window_index:end_window_index, 'label'] = 1
    
    return features_data

def main():
    # Path to the folder containing the extracted features
    features_data_folder = Path('.data/extracted_features')
    
    all_labeled_features = []

    # Iterate over the feature files and label them based on the start and end times
    for experiment, times in time_ranges.items():
        feature_file = features_data_folder / f"{experiment}.csv"
        if feature_file.is_file():
            labeled_data = add_labels_to_features(feature_file, times)
            all_labeled_features.append(labeled_data)
        else:
            print(f"Feature file not found for {experiment}")

    # Union all labeled feature files into one final dataset
    if all_labeled_features:
        final_dataset = pd.concat(all_labeled_features, ignore_index=True)
        final_dataset_path = features_data_folder / 'final_labeled_features_dataset.csv'
        final_dataset.to_csv(final_dataset_path, index=False)
        print(f"Final labeled features dataset created at {final_dataset_path}")
    else:
        print("No labeled features to concatenate.")

if __name__ == "__main__":
    main()
