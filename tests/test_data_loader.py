# test_dataloader.py
import pytest
from unittest.mock import patch, mock_open
import sys
from pathlib import Path
# Add the path to the DataLoader script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.data_loader import DataLoader
import pandas as pd
import os
import logging
from typing import Any, Dict, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Example of correctly formatted mock CSV data
sample_csv_data = "time\tdata\n0.0\t100\n0.1\t110\n"

# Sample config for DataLoader
sample_config = {
    "experiments": {
        "experiment1": [
            {"path": ".data\\Experiment_1\\measurement_1.tsv", "type": "tsv"}
        ]
    }
}

@patch("builtins.open", new_callable=mock_open, read_data=sample_csv_data)
@patch("os.path.exists", return_value=True)
@patch("json.load", return_value=sample_config)
def test_load_experiment_data(mock_json_load, mock_exists, mock_file):
    # Initialize DataLoader with a mock config path
    loader = DataLoader("config.json")

    # Attempt to load data for "experiment1"
    loaded_data = loader.load_experiment_data("experiment1")

    # Check that the DataFrame is not empty
    assert not loaded_data.empty

    # Check that the DataFrame has the correct columns
    assert list(loaded_data.columns) == ["time", "data", "experiment", "measurement"]

    # Check that the 'experiment' and 'measurement' columns are correctly populated
    assert all(loaded_data["experiment"] == "experiment1")
    assert all(loaded_data["measurement"].str.contains("measurement"))

    # Check that both columns are numeric
    assert loaded_data["time"].dtype == "float64"
    assert loaded_data["data"].dtype == "float64"

# to execute the test, run "pytest" in the terminal
