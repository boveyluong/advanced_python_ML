from typing import Tuple
from numpy import ndarray
from sklearn.model_selection import train_test_split 
import pandas as pd 

class Model:
    def __init__(self, model) -> None:
        self.model = model 

    def split_data_set(self, X: pd.DataFrame, y: pd.Series, random_state: int = 42, test_size: float = 0.3, shuffle = True) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        """Split the dataset into training and testing set

        Args:
            X (Dataframe): the dataframe containing the features to make the prediction
            y (Series): the array containg
            random_state (int, optional): ensure a random split with the same values every time the code run. Defaults to 42.
            test_size (float, optional): _description_. Defaults to 0.25.

        Returns:
            Tuple[ndarray, ndarray, ndarray, ndarray]: return the splitted train-test inputs
        """
        X_train, X_test, y_train, y_test = train_test_split(X,y, 
                                   random_state=random_state,  
                                   test_size=test_size,  
                                   shuffle=shuffle) 
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train: ndarray, Y_train: ndarray) -> None:
        """Train the model

        Args:
            X_train (ndarray): Train data
            Y_train (ndarray): Target data 
        """
        self.model.fit(X_train, Y_train)
    
    def get_trained_model(self) -> None:
        """Getting the trained Model
        """
        return self.model