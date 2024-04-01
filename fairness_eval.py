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

def group_bias_score(df: pd.DataFrame, 
                             feature_cols: list[str]=['SEX', 'AGE']) -> pd.DataFrame:
    """
    Calculate the bias score based on given dataframe and protected features.

    Parameters:
    - df: DataFrame containing true and predicted labels along with protected features.
    - feature_cols: List of column names representing protected features. Default is ['SEX', 'AGE'].

    Returns:
    - Fairness score as a float.
    """
    
    #initialize empty dataframe
    df = pd.DataFrame(columns=['Protected Group','Category','FPR', 'FNR', 'ACC'])

    # group by protected features
    for prot_group in feature_cols:
        for name, group in df.groupby(by=prot_group):
            cm = get_confusion_matrix(group['y_true'], group['y_pred'])
            
            #add group fpr to tracker
            fpr = get_fpr_rate(cm)
            #add group fnr to tracker
            fnr = get_fnr_rate(cm)
            #add group acc to tracker
            acc = get_acc_rate(cm)
            
            #add rows to empty dataframe
            df.loc[len(df)] = [prot_group, name, fpr, fnr, acc]
    
    df[['FPR_REF','FNR_REF','ACC_REF', 'FPR_DIFF','FNR_DIFF','ACC_DIFF']] = np.NaN
    for category in df.groupby(by=['Protected Group', 'Category']):
        #find fpr ref and set value in df
        df.loc[df['Category'] == category[0][1], 'FPR_REF'  ] = (min(group[1]['FPR']))
        
        #find fnr ref
        fnr_ref = 1
        fnr = (min(group[1]['FNR']))
        if fnr < fnr_ref:
            fnr_ref = fnr
        df.loc[df['Category'] == category[0][1], 'FNR_REF'  ] = fnr_ref 

        #find acc ref
        acc_ref = 1
        acc = (min(group[1]['ACC']))
        if acc < acc_ref:
            acc_ref = acc
        df.loc[df['Category'] == category[0][1], 'ACC_REF'  ] = acc_ref 
        
        #find the differences between the maximum of each metric and the ref
        df.loc[df['Category'] == category[0][1], 'FPR_DIFF'  ] = max(df[['Category', 'FPR']]['FPR']) - min(df[['Category', 'FPR']]['FPR'])
        df.loc[df['Category'] == category[0][1], 'FNR_DIFF'  ] = max(df[['Category', 'FNR']]['FNR']) - fnr_ref
        df.loc[df['Category'] == category[0][1], 'ACC_DIFF'  ] = max(df[['Category', 'ACC']]['ACC']) - acc_ref
    
    #calculate group bias scores FIX TO FIND SCORE FOR EACH GROUP
    df['Bias'] = np.sqrt((1/3) * (
        df['FPR_DIFF'] ** 2 + df['FNR_DIFF'] ** 2 + df['ACC_DIFF'] ** 2))
            
    return df

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
    
    #convert theil to bias score
    
    return theil


def fairness_score(df: pd.DataFrame, y_true: pd.Series | np.ndarray | list, qs_score: float,
                         feature_cols: list[str]=['SEX', 'AGE']) -> float:
    """
    Calculate the bias score based on given dataframe and protected features.

    Parameters:
    - df: DataFrame containing true and predicted labels along with protected features.
    - feature_cols: List of column names representing protected features. Default is ['SEX', 'AGE'].

    Returns:
    - Fairness score as a float.
    """
    #get individual and group scores
    theil = get_theil_binary(y_true)
    group = group_bias_score(df, feature_cols)
    
    #extract group bias scores from dataframe
    scores = []
    for group in feature_cols:
        group_bias = df['Protected Group']['Bias'][0]
        scores.append(group_bias)
        
    scores.append(theil)
    
    #initialize array with all fairness scores.    
    scores = np.array(scores)
    #square each score
    scores = scores ** 2
    #sum up scores, multiply by 1/(number of scores), then take the square root to find raw fairness score
    raw_fairness = np.sqrt((1/len(scores)) * sum(scores))
    
    #calculate final fairness score by multiplying raw score with qualitative score given as input
    fairness = qs_score * raw_fairness
    
    return fairness
    
    
    
        