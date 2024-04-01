# ../model.py

import os
import numpy as np
import pandas as pd
import math

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
                             feature_cols: list[str]=['SEX', 'AGE']) -> dict:
    """
    Calculate the fairness score based on given dataframe and protected features.

    Parameters:
    - df: DataFrame containing true and predicted labels along with protected features.
    - feature_cols: List of column names representing protected features. Default is ['SEX', 'AGE'].

    Returns:
    - Fairness score as a float.
    """
    
    # setup trackers for each metric
    fpr_dict = {group:"" for group in feature_cols}
    fnr_dict = {(group, []) for group in feature_cols}
    acc_dict = {(group, []) for group in feature_cols}

    # group by protected features
    for prot_group in feature_cols:
        for name, group in df.groupby(by=prot_group):
            cm = get_confusion_matrix(group['y_true'], group['y_pred'])
            
            #add group fpr to tracker
            fpr = get_fpr_rate(cm)
            fpr_dict[group] = fpr
            #add group fnr to tracker
            fnr = get_fnr_rate(cm)
            fnr_dict[group] = fnr
            #add group acc to tracker
            acc = get_acc_rate(cm)
            acc_dict[group] = acc
     
    #setup bias tracker
    bias_dict =  {group:"" for group in feature_cols}      
     
    #calculate bias score
    for prot_group in feature_cols:
        bias = (math.sqrt((1/3)*
                          (fpr_dict[prot_group]**2 + 
                           fnr_dict[prot_group]**2 +
                           acc_dict[prot_group]**2)))
        bias_dict[prot_group] = bias
            
    return bias_dict

def get_theil_binary(y_true: pd.Series | np.ndarray | list) -> float:
    """
    Calculate the theil index for a binary outcome model where the outcome model
    where the outcome variable is the calculated benefit
    
    Parameters:
    - df: DataFrame containing true and predicted labels along with protected features.

    Returns:
    - Theil index value as a float
    """
    #calculate the average of the outcome variable
    avg_benefit = sum(y_true) / len(y_true)
    
    #create variable to track sum of each benefit ratio in the data
    theil_sum = 0
    
    #find logged benefit ratio for each row and add value to the tracker
    for value in y_true:
        theil_sum += math.log((value/avg_benefit) ** value)
         
    #multiply the sum of logged benefit ratios by 1 over n, then divide
    #by the average benefit    
    theil = (theil_sum * (1/len(y_true)) ) / avg_benefit
    
    return theil
        