import pandas as pd
import numpy as np
def calculate_iv(df, feature, target):
    """
    Calculate the information value (IV) of a feature in a dataset.

    Parameters:
    df (pandas.DataFrame): the dataset
    feature (str): the name of the feature to calculate IV for
    target (str): the name of the target variable

    Returns:
    float: the information value (IV) of the feature
    """
    df = df[[feature, target]]
    df = df.dropna()
    n = df.shape[0]
    good = df[target].sum()
    bad = n - good
    unique_values = df[feature].unique()
    iv = 0
    for value in unique_values:
        n1 = df[df[feature] == value].shape[0]
        good1 = df[(df[feature] == value) & (df[target] == 1)].shape[0]
        bad1 = n1 - good1
        if good1 == 0 or bad1 == 0:
            continue
        woe = np.log((good1 / good) / (bad1 / bad))
        iv += (good1 / good - bad1 / bad) * woe
    return iv