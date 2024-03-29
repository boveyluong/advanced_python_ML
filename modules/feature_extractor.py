import json
import logging
from pathlib import Path
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.utilities.dataframe_functions import impute
import pandas as pd

class FeatureExtractor:
    """
    FeatureExtractor class for extracting features from time series data using the tsfresh library.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the FeatureExtractor with configuration from a JSON file.
        """
        self.config = self.load_and_validate_config(config_path)
        self.extraction_settings = self.get_extraction_settings()
        
    def load_and_validate_config(self, config_path: str) -> dict:
        """
        Load and validate configuration settings from a JSON file.
        """
        try:
            with open(config_path, 'r') as config_file:
                config = json.load(config_file)
                
            # Validate the loaded configuration
            self.validate_config(config)
            logging.info("Configuration successfully loaded and validated for FeatureExtractor.")
            return config
        except FileNotFoundError:
            logging.error(f"Configuration file not found at {config_path}.")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON format in configuration: {e}")
            raise
    
    def validate_config(self, config):
        """
        Check the loaded configuration for necessary fields and formats.
        """
        # Adjust the required fields for your feature extraction configuration
        required_fields = ['feature_extraction']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in configuration: {field}")
    
    def get_extraction_settings(self):
        """
        Fetch and return feature extraction settings based on the loaded configuration.
        """
        feature_extraction_config = self.config['feature_extraction']
        
        if feature_extraction_config['default_fc_parameters'] == "ComprehensiveFCParameters":
            return ComprehensiveFCParameters()
        else:
            logging.error("Unsupported feature extraction parameters.")
            raise ValueError("Unsupported feature extraction parameters.")
    
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract time series features from the provided data using the tsfresh library.
        
        :param data: Pandas DataFrame with columns 'id', 'time', and 'value'.
        :return: DataFrame with extracted features.
        """
        extracted_features = extract_features(data,
                                              column_id='id', column_sort='time',
                                              default_fc_parameters=self.extraction_settings,
                                              impute_function=impute)
        return extracted_features
