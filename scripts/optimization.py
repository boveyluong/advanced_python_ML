import pandas as pd
from pathlib import Path
import sys
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

# Add the path to the DataLoader script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.optimizer import RandomForestOptimizer
from modules.data_loader import DataLoader

def main():
     # Initialize DataLoader
    data_loader = DataLoader('config.json')
    features = pd.read_csv('.data/extracted_features/final_labeled_features_dataset.csv')

    config = data_loader.config['algorithms']
    
    print(features.head(5))

    # Impute any missing values in the feature set
    impute(features)

    features_df = features.drop(['label'],axis=1)
    target = features.loc[:, 'label']

    # Select only relevant features
    relevant_features = select_features(features_df, target)

    # Instantiate the optimizer
    rf_optimizer  = RandomForestOptimizer(config=config["optimizer"])
    X_train, X_test, y_train, y_test  = rf_optimizer.split_data_set(relevant_features, target)

    # Fit the model using grid search
    rf_optimizer.train(X_train, y_train)

    # Tuning the algorithm (Hyper Parameter Tuning). For time reason the next sentence is commented
    # rf_optimizer.hyper_parameter_tuning(X_train, y_train)

    # Getting the accuracy of the model 
    rf_score = rf_optimizer.get_score(X_test, y_test) 

    # Round the accuracy
    acc_random_forest = round(rf_score * 100, 2)

    # Showing the accuracy
    print("Accuray:", acc_random_forest, "%")
    print("oob score:", rf_optimizer.get_oob_score(), "%")

if __name__ == "__main__":
    main()
