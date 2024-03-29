# Import necessary libraries and classes
import pandas as pd
from pathlib import Path
import sys
import os

# Add the path to the DataLoader, SignalPreprocessor, and FeatureExtractor scripts
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.feature_extractor import FeatureExtractor

def process_file(preprocessed_file_path, extracted_features_directory, feature_extractor):
    # Load preprocessed data
    data = pd.read_csv(preprocessed_file_path)
    
    data['id'] = data.index // 1000  # Assign an ID based on 1000 samples per window

    # Extract features
    features = feature_extractor.extract_features(data)

    # Save extracted features with the same naming convention
    features_save_path = extracted_features_directory / preprocessed_file_path.name
    features.to_csv(features_save_path, index=False)
    print(f"Features extracted and saved to {features_save_path}")

def main(file_to_process=None):
    # Define the path to the preprocessed data directory and extracted features directory
    preprocessed_data_directory = Path('.data/preprocessed')
    extracted_features_directory = Path('.data/extracted_features')
    extracted_features_directory.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    # Initialize the feature extractor
    feature_extractor = FeatureExtractor()

    if file_to_process:
        # Process only the specified file
        preprocessed_file_path = preprocessed_data_directory / file_to_process
        if preprocessed_file_path.is_file():
            process_file(preprocessed_file_path, extracted_features_directory, feature_extractor)
        else:
            print(f"File {file_to_process} not found in {preprocessed_data_directory}.")
    else:
        # Process all files in the preprocessed data directory
        for preprocessed_file in preprocessed_data_directory.glob('*.csv'):
            process_file(preprocessed_file, extracted_features_directory, feature_extractor)

if __name__ == "__main__":
    file_to_process = sys.argv[1] if len(sys.argv) > 1 else None
    main(file_to_process)
