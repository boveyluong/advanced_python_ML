import pandas as pd
from pathlib import Path
import sys
import os
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
import joblib
from numpy import ndarray
import threadpoolctl
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

# Add the path to the DataLoader script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.learner import Learner
from modules.evaluator import Evaluator
from modules.data_loader import DataLoader

def train_and_evaluate(learner: 'Learner', X_train: ndarray, X_test: ndarray, y_train: ndarray, y_test: ndarray, algorithm_ ='random_forest') -> None:
    """Training and evaluating the models
    Args:
        learner (Learner): algorithm to be trained and evaluated
        X_train (ndarray): array containing the features for training data.
        Y_train (ndarray): array containing the target values.
        X_test (ndarray): array containing the features to test
        y_test (ndarray): array containing the test true target values
        plots_dir (str): directory where the plots are to saved
        algorithm_ (str, optional): the name of the algorithm. Defaults to 'random_forest'.
    """
    print(f'*************************************************Training and evaluating {algorithm_}****************************************************')


    artifacts_dir = os.path.join('artifacts', 'results')
    plots_dir = os.path.join(artifacts_dir, 'plots')
    models_dir = os.path.join(artifacts_dir, 'models')

    learner.train(X_train=X_train, Y_train=y_train)

    learner.accuracy(X_test, y_test)

    learner.cross_validation(X_train=X_train, y_train=y_train)

    evaluator = Evaluator(learner.model, X_test, y_test)
    evaluator.evaluate_model()
   

    evaluator.plot_metrics(model = learner.model, X_test=X_test, y_test=y_test, plots_dir=plots_dir, algorithm_=algorithm_)
    cm = evaluator.confusion_matrix()
    evaluator.plot_confusion_matrix(plots_dir=plots_dir, algorithm_= algorithm_, target_names=['Anomalie', 'Keine Anomalie'], conf_matrix=cm)
    #save the trained model
    filename = os.path.join(models_dir, f'{algorithm_}_trained_time_series_model.pkl')

    # if not os.path.isdir(learner.model):
    #     os.makedirs(learner.model)
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filename, "wb") as f:
        # f.write(learner.model)
        joblib.dump(learner.model, filename=filename)

def create_folder ():
    for directory in ['artifacts/results', 'artifacts/results/plots', 'artifacts/results/models']:
        os.makedirs(directory, exist_ok=True) # This will create the directory if it does not exist


def main():
    create_folder()
    # Initialize DataLoader
    data_loader = DataLoader('config.json')
    # List of algorithm to configuration
    config = data_loader.config['algorithms']

    features = pd.read_csv('.data/extracted_features/final_labeled_features_dataset.csv')
    
    print(features.head(5))

    features_df = features.drop(['label'], axis=1)
    target = features['label']
  
    # Impute any missing values in the feature set
    impute(features_df)

    # Select only relevant features
    relevant_features = select_features(features_df, target)

    # Apply Random Over Sampling to account for the imbalanced dataset
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(relevant_features, target)

    # Now split the resampled data
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, random_state=42)

    learner_random_forest = Learner(config=config["random_forest"])
    learner_decision_tree = Learner(config=config["decision_tree"])
    learner_knn = Learner(config=config["k_nearest_neighbors"])

    train_and_evaluate(algorithm_=config["random_forest"]["name"], learner=learner_random_forest, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    train_and_evaluate(algorithm_=config["decision_tree"]["name"], learner=learner_decision_tree, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    train_and_evaluate(algorithm_=config["k_nearest_neighbors"]["name"], learner=learner_knn, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

if __name__ == "__main__":
    main()
