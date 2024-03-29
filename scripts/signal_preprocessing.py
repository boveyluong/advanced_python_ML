import pandas as pd
import sys
from pathlib import Path
import os
import numpy as np

# Add the path to the DataLoader script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.signal_preprocessor import SignalPreprocessor  # Update the import path as needed

def load_full_data() -> pd.DataFrame:
    """
    Load the full DataFrame from the saved CSV file.
    """
    full_dataframe_path = '.data/full_dataframe.csv'
    return pd.read_csv(full_dataframe_path)

def preprocess_and_save(data: pd.DataFrame, experiment_name: str, measurement_name: str, preprocessor: SignalPreprocessor):
    """
    Preprocess the data for a given experiment and measurement and save the results.
    """
    preprocessed_windows = preprocessor.preprocess(data)
    preprocessor.save_preprocessed_data(preprocessed_windows, experiment_name, measurement_name)

def main():
    # Load the full experiment data
    full_data = load_full_data()

    # Initialize the signal preprocessor
    preprocessor = SignalPreprocessor()

    # Get unique experiment and measurement combinations
    combinations = full_data[['experiment', 'measurement']].drop_duplicates()

    for _, row in combinations.iterrows():
        experiment_name = row['experiment']
        measurement_name = row['measurement']
        
        # Filter data for the current experiment and measurement
        specific_data = full_data[(full_data['experiment'] == experiment_name) & 
                                  (full_data['measurement'] == measurement_name)].copy()

        # Reset the time index for the measurement
        specific_data = specific_data.copy()
        specific_data['time'] = specific_data['time'] - specific_data['time'].iloc[0] 
        
        print(f"Processing {experiment_name} {measurement_name}...")
        
        # Preprocess and save the data
        preprocess_and_save(specific_data, experiment_name, measurement_name, preprocessor)

    print("All measurements have been preprocessed and saved.")

if __name__ == "__main__":
    main()
