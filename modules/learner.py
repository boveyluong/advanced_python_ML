from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from numpy import ndarray
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from .model import Model

class Learner (Model):
    def __init__(self, config: dict = {"name": 'random_forest', 'n_estimators': 1000, 'random_state': 42}) -> None:
        """Initializes the Learner with the specified algorithm

        Args:
            config (dict, optional): config dictionary of the specified algorithm. Defaults to {"name": 'random_forest', 'n_estimators': 1000, 'random_state': 42}.
        """
        self.config = config
        if config["name"] == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=config["n_estimators"], random_state=config["random_state"])
        elif  config["name"] == 'k_nearest_neighbors':
            self.model = KNeighborsClassifier(n_neighbors = config['n_neighbors']) 
        elif  config["name"] == 'decision_tree':
            self.model = DecisionTreeClassifier(max_depth = config["max_depth"]) 
        super().__init__(self.model)

    def predict(self,  X_test: ndarray) -> ndarray:
        """Makes predictions using the trained Random Forest model.

        Args:
            X_test (ndarray): array containing the test features data to make the prediction

        Returns:
            ndarray: return the predicted class
        """
        return self.model.predict(X_test)
    
    def accuracy(self, X_test: ndarray, y_test: ndarray) -> None:
        """Evaluates the trained model on the test data and prints a classification report.

        Args:
            X_test (ndarray): array containing the test features data to provide the accuracy
            y_test (ndarray): array containing the true labels of the testing set.
        """
        predictions = self.predict(X_test)
        score = accuracy_score(y_test, predictions)
        print(classification_report(y_test, predictions))
        print(f"Accuracy: {round(score * 100, 2)}") # type: ignore

    def cross_validation(self,  X_train: ndarray, y_train: ndarray) -> None:
        """Perform cross-validated scoring for an estimator on the dataset to assess how well the model will generalize on independant dataset

        Args:
            X_train (ndarray): array containing the features for training data for the cross_validation
            y_train (ndarray): array containing the target values for the cross_validation
        """
        scores = cross_val_score(self.model, X_train, y_train, cv=10, scoring = "accuracy")
        print("Scores:", scores) 
        print("Mean:", scores.mean())
        print("Standard Deviation:", scores.std())
