# kostja
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score, matthews_corrcoef
#import scikitplot as skplt
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show
from bokeh.models import HoverTool
import seaborn as sns
import os


class Evaluator:
   """A class to evaluate the performance of a machine learning model on a given test dataset.

    Attributes:
        model: The machine learning model to be evaluated.
        X_test: Features from the test dataset.
        y_test: Labels from the test dataset.
        predictions: Predictions made by the model on X_test.

    Methods:
        evaluate_model(): Prints a summary of various evaluation metrics.
        plot_precision_and_recall(): Generates a plot for precision and recall curves.
        plot_metrics(): Generates ROC and Precision-Recall curves for the model.
        confusion_matrix(): Prints the confusion matrix for model predictions.
        plot_confusion_matrix(): Generates a heatmap for the confusion matrix."""
   def __init__(self, model, X_test, y_test):
      """Initializes the Evaluator with a model and test data.

        Args:
            model: The machine learning model to be evaluated.
            X_test: The features of the test dataset.
            y_test: The labels of the test dataset.
      """
      self.model = model
      self.X_test = X_test
      self.y_test = y_test
      self.predictions = model.predict (X_test)

   def evaluate_model(self):
      """
      Prints evaluation metrics including Accuracy, Precision, Recall, F1-Score, MCC, ROC-AUC, and PR-AUC
      """  
      metrics = {
            "Accuracy": accuracy_score(self.y_test, self.predictions),
            "Precision": precision_score(self.y_test, self.predictions),
            "Recall": recall_score(self.y_test, self.predictions),
            "F1-Score": f1_score(self.y_test, self.predictions),
            "MCC": matthews_corrcoef(self.y_test, self.predictions),
            "ROC-AUC": roc_auc_score(self.y_test, self.model.predict_proba(self.X_test)[:,1]),
            "PR-AUC": average_precision_score(self.y_test, self.model.predict_proba(self.X_test)[:,1])
        }
      print("Evaluation Metrics:")
      for metric, value in metrics.items():
         print(f"{metric}: {value}")
         
   def plot_precision_and_recall(self, precision, recall, threshold, plots_dir, algorithm_):
      """enerates and saves a plot showing the precision and recall curves as a function of the threshold.

        Args:
            precision: Precision values corresponding to different thresholds.
            recall: Recall values corresponding to different thresholds.
            threshold: Threshold values used to calculate precision and recall.
            plots_dir: Directory to save the plot.
            algorithm_: Name of the algorithm used for labeling the plot."""
      plt.plot(threshold, precision[:-1], "darkorange", label="precision", linewidth=5)
      plt.plot(threshold, recall[:-1], "navy", label="recall", linewidth=5)
      plt.xlabel("threshold", fontsize=19)
      plt.legend(loc="upper right", fontsize=19)
      plt.ylim([0, 1])
      plt.figure(figsize=(14, 7)) 
      #plt.savefig(os.path.join(plots_dir, f'{algorithm_}_recall_precision.png'))
   
   def plot_metrics(self, model,X_test, y_test, algorithm_, plots_dir):
      """Generates and saves plots for ROC and Precision-Recall curves.

        Args:
            model: The machine learning model to be evaluated.
            X_test: The features of the test dataset.
            y_test: The labels of the test dataset.
            algorithm_: Name of the algorithm used for labeling the plots.
            plots_dir: Directory to save the plots."""
      
      fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # ROC curve
      fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
      roc_auc = auc(fpr, tpr)
      ax[0].plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
      ax[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
      ax[0].set_xlim([0.0, 1.0])
      ax[0].set_ylim([0.0, 1.05])
      ax[0].set_xlabel('False Positive Rate')
      ax[0].set_ylabel('True Positive Rate')
      ax[0].set_title(f'Receiver Operating Characteristic {algorithm_} times series')
      ax[0].legend(loc='lower right')

    # Precision-Recall curve
      precision, recall, thresholds = precision_recall_curve(y_test, self.model.predict_proba(X_test)[:, 1])
      pr_auc = auc(recall, precision)
      ax[1].plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (area = {:.2f})'.format(pr_auc))
      ax[1].set_xlim([0.0, 1.0])
      ax[1].set_ylim([0.0, 1.05])
      ax[1].set_xlabel('Recall')
      ax[1].set_ylabel('Precision')
      ax[1].set_title(f'Precision-Recall Curve,{algorithm_} times serie')
      ax[1].legend(loc='lower left')
      plt.savefig(os.path.join(plots_dir, f'{algorithm_}_roc_pr_curve.png'))
    
   def confusion_matrix(self):
      """Prints the confusion matrix for the model predictions on the test data."""
      cm = confusion_matrix(self.y_test, self.predictions)
      print("Confusion Matrix:")
      print(cm)
      return cm
      
   def plot_confusion_matrix(self, plots_dir, target_names, algorithm_, conf_matrix = None):
      """Generates and saves a heatmap of the confusion matrix.

        Args:
            plots_dir: Directory to save the heatmap.
            target_names: Names of the target classes.
            algorithm_: Name of the algorithm used for labeling the heatmap.
            conf_matrix: Confusion matrix to be plotted. If None, the matrix is computed."""
      plt.figure(figsize=(8, 6))
      sns.heatmap(conf_matrix, annot=True, fmt='g',
                  cmap=sns.cubehelix_palette(as_cmap=True),
                  xticklabels=target_names,
                  yticklabels=target_names)
      plt.xlabel('Predicted Labels')
      plt.ylabel('True Labels')
      plt.title(f'Confusion Matrix {algorithm_} times series')
      plt.savefig(os.path.join(plots_dir, f'{algorithm_}_confusion_matrix.png'))
