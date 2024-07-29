import numpy as np
import pandas as pd

def get_metrics(targets, prediction_df, ):
    metrics = {}
    for target in targets:
        metrics[target] = {}
        for set in ['train','val','test']:
            true = prediction_df[prediction_df['set']==set][f"{target}_true"]
            pred = prediction_df[prediction_df['set']==set][f"{target}_pred"]
            metrics[target][set+'_r2'] = r2_score(true,pred)
            metrics[target][set+'_mae'] = mean_absolute_error(true,pred)
            metrics[target][set+'_rmse'] = np.sqrt(mean_squared_error(true,pred))
            metrics[target][set+'_max_error'] = max_error(true,pred)
    metrics = pd.DataFrame(metrics).T
    metrics = metrics[[f"{set}_{metric}" for set in ['train','val','test'] for metric in ['r2','mae','rmse','max_error']]]
    return metrics  

def r2_score(true,pred):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    return 1 - (ss_res / ss_tot)

def mean_absolute_error(true,pred):
    return np.mean(np.abs(true-pred))

def mean_squared_error(true,pred):
    return np.mean((true-pred)**2)

def max_error(true,pred):
    return np.max(np.abs(true-pred))
