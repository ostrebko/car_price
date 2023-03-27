import numpy as np
import pandas as pd




def mape_func(y_true, y_pred):
    """
    Function to calc MAPE-metric

    return: calculated metric MAPE
    -------
    params:
    
    y_true - there are known values of target y
    y_pred - predicted values of target y
    """
    return np.mean(np.abs((y_pred - y_true)/y_true))

