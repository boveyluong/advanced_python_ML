import pandas as pd

def load_measurement_data(experiment_name: str, measurement_name: str = None) -> pd.DataFrame:
    """
    Loads data for an entire experiment or a specific measurement within an experiment.

    Args:
    - experiment_name (str): The name of the experiment to load.
    - measurement_name (str, optional): The name of the specific measurement to load. If None, loads all measurements from the experiment.

    Returns:
    - pd.DataFrame: Data for the specified experiment or measurement.
    """
    # Define the path to the full dataframe
    full_dataframe_path = '.data/full_dataframe.csv'
    
    # Load the full dataframe
    full_data = pd.read_csv(full_dataframe_path)

    # Filter the dataframe for the specified experiment
    if measurement_name:
        # Filter for a specific measurement within the experiment
        specific_data = full_data[(full_data['experiment'] == experiment_name) & 
                                  (full_data['measurement'] == measurement_name)]
    else:
        # Load all measurements for the specified experiment
        specific_data = full_data[full_data['experiment'] == experiment_name]

    return specific_data

def main():
    # Specify the experiment name you want to load
    experiment_name = 'experiment1'  # Example experiment name

    # Optionally specify the measurement name you want to load
    measurement_name = 'measurement_1'  # Example measurement name, set to None to load all measurements in the experiment

    # Load the specified experiment/measurement data
    data = load_measurement_data(experiment_name, measurement_name)

    if data.empty:
        print(f"No data found for {experiment_name} {measurement_name}.")
    else:
        # Proceed with signal preprocessing or feature engineering on the loaded data
        print(f"Data for {experiment_name} {measurement_name if measurement_name else 'all measurements'} loaded successfully. Number of samples: {len(data)}")

if __name__ == "__main__":
    main()


# test the function
# Load the specified measurement data
measurement_data = load_measurement_data('experiment2', 'measurement_2')
print(measurement_data.head())
# print the shape and length of the data
print(measurement_data.shape)
print(len(measurement_data))