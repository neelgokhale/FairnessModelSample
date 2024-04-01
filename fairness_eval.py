# ../model.py

import os
import numpy as np
import pandas as pd

from typing import Optional

from sklearn.metrics import confusion_matrix

def get_confusion_matrix(y_true: pd.Series | np.ndarray | list,
                         y_pred: pd.Series | np.ndarray | list) -> np.ndarray:
    """
    Calculate the confusion matrix for given true labels and predicted labels.

    Parameters:
    - y_true: True labels.
    - y_pred: Predicted labels.

    Returns:
    - Confusion matrix as a numpy array.
    """
    cm = confusion_matrix(y_true, y_pred)
    return cm

def get_fnr_rate(cm: np.ndarray) -> float:
    """
    Calculate the false negative rate from a confusion matrix.

    Parameters:
    - cm: Confusion matrix.

    Returns:
    - False negative rate as a float.
    """
    tn, _, fn, _ = cm.ravel()
    fnr_rate = fn / (fn + tn)
    return fnr_rate

def get_fpr_rate(cm: np.ndarray) -> float:
    """
    Calculate the false positive rate from a confusion matrix.

    Parameters:
    - cm: Confusion matrix.

    Returns:
    - False positive rate as a float.
    """
    _, tp, _, fp = cm.ravel()
    fpr_rate = fp / (fp + tp)
    return fpr_rate

def get_acc_rate(cm: np.ndarray) -> float:
    """
    Calculate the accuracy rate from a confusion matrix.

    Parameters:
    - cm: Confusion matrix.

    Returns:
    - Accuracy rate as a float.
    """
    tn, fp, fn, tp = cm.ravel()
    acc_rate = (tn + tp) / (tn + tp + fn + fp)
    return acc_rate

def calculate_fairness_score(df: pd.DataFrame, 
                             feature_cols: list[str]=['SEX', 'AGE']) -> float:
    """
    Calculate the fairness score based on given dataframe and protected features.

    Parameters:
    - df: DataFrame containing true and predicted labels along with protected features.
    - feature_cols: List of column names representing protected features. Default is ['SEX', 'AGE'].

    Returns:
    - Fairness score as a float.
    """
    
    # setup trackers for each metric
    fpr_dict = {(group, []) for group in feature_cols}
    fnr_dict = {(group, []) for group in feature_cols}
    acc_dict = {(group, []) for group in feature_cols}

    # group by protected features
    for prot_group in feature_cols:
        for name, group in df.groupby(by=prot_group):
            cm = get_confusion_matrix(group['y_true'], group['y_pred'])
