# Import necessary libraries and classes
import pandas as pd
import sys
from pathlib import Path
import os
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.data_loader import DataLoader

def main():
    # Initialize the DataLoader and load all experiment data
    data_loader = DataLoader('config.json')
    full_data = data_loader.load_experiment_data()  # Assuming this method loads all data if no argument is provided

    # Save the combined data to a CSV file in the ".data" folder
    full_data.to_csv('.data/full_dataframe.csv', index=False)
    print("All data has been successfully saved to '.data/full_dataframe.csv'.")

if __name__ == "__main__":
    main()