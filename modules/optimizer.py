from sklearn.ensemble import RandomForestClassifier
import sys
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from numpy import ndarray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from .model import Model

class RandomForestOptimizer (Model):
    def __init__(self,  config: dict):
        self.rf_model = RandomForestClassifier(criterion=config['criterion'], min_samples_leaf = config["min_samples_leaf"], min_samples_split = config["min_samples_split"], n_estimators=config["n_estimators"], max_features=config['n_estimators'], oob_score= config['oob_score'], random_state=config['random_state'], n_jobs=config["n_jobs"])
        self.config = config
        super().__init__(self.rf_model)
        
    def hyper_parameter_tuning(self, X_train: ndarray , y_train: ndarray) -> None:
        """
        Fit the RandomForestClasifier model using grid search.

        Best Parameter: {'criterion': 'entropy', 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 100} 
        Best Estimator: RandomForestClassifier(criterion='entropy', max_features=100,
                               min_samples_split=4, n_jobs=-1, oob_score=True,
                               random_state=42)
        Best Score: 0.9142857142857143
        Classes: [0 1]

        Parameters:
        - X_train (ndarray): Training data.
        - y_train (ndarray): Target values.

        """
        self.clf = GridSearchCV(estimator=self.model,
                                 param_grid=self.config["param_grid"],
                                 n_jobs=self.config['tuning']['n_jobs'], 
                                 cv = self.config['tuning']['cv'], 
                                 scoring= self.config['tuning']['scoring'])
        self.clf.fit(X_train, y_train)

        print(f'Best Parameter: {self.clf.best_params_} \n Best Estimator: {self.clf.best_estimator_} \n Best Score: {self.clf.best_score_} \n Classes: {self.clf.classes_} \n Features name {self.clf.feature_names_in_}') 
       

    def get_score(self, X_test: ndarray, y_test: ndarray) -> float:
        """ Get the model accuracy

        Returns:
            float: the model accuracy value
        """
        return self.model.score(X_test, y_test)     

    def get_oob_score(self) -> float:
        """ Get the oob_score
        Description: It stands for "Out-of-Bag" score and provides an estimate of the model's performance on unseen data 

        Returns:
            float: Floating value of oob_score_
        """
        return self.model.oob_score_ 
    