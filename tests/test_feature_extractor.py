import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "modules"))

from feature_extractor import FeatureExtractor

def create_test_data():
    """Creates a sample DataFrame structured for tsfresh feature extraction."""
    np.random.seed(42)  # For reproducible results
    # Simulate 100 data points split into 10 windows, each with a unique 'id'
    data = pd.DataFrame({
        'id': np.repeat(np.arange(10), 10),
        'time': np.tile(np.arange(10), 10),
        'value': np.random.rand(100),
    })
    return data

def test_extract_features():
    data = create_test_data()
    feature_extractor = FeatureExtractor()
    extracted_features = feature_extractor.extract_features(data)

    # Basic checks
    assert not extracted_features.empty, "Extracted features should not be empty."
    assert isinstance(extracted_features.index, pd.Index), "The index of the extracted features DataFrame should be a Pandas Index."

    
    # More specific checks can include:
    # - Verifying the number of features extracted matches expectations
    # - Checking for the presence of specific expected feature columns
    # - Ensuring no NaN values are present after imputation

# To execute the test, use the command: pytest tests/test_feature_extractor.py
