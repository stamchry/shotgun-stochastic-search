# main/helpers.py

import numpy as np
import pandas as pd
from scipy.special import gamma as gamma_func



def model_selection_prior(hyperparameter: float, k: int, p: int):
    return ((hyperparameter) ** k) * ((1 - hyperparameter) ** (p - k))

def deletion(gamma: list[int]) -> list[list[int]]:
    return [gamma[:i] + [0] + gamma[i + 1:] for i, k in enumerate(gamma) if k == 1]

def addition(gamma):
    return [gamma[:i] + [1] + gamma[i + 1:] for i, k in enumerate(gamma) if k == 0]

def replacement(gamma):
    gamma_replacement = []

    for i, k in enumerate(gamma):
        for j, l in enumerate(gamma):
            if k == 1 and l == 0 and i != j:
                gamma_i = gamma.copy()
                gamma_i[i] = 0
                gamma_i[j] = 1
                gamma_replacement.append(gamma_i)

    return gamma_replacement

def nbd(gamma):
    return addition(gamma), replacement(gamma), deletion(gamma)


def create_dataframe_with_score(feature_list, score, existing_dataframe):
    # Add 0 to the end of the feature list
    feature_list_with_score = feature_list+[score]

    # Get the column names from the existing DataFrame
    existing_column_names = list(existing_dataframe.columns)

    # Create a DataFrame with one row and column names based on the existing_column_names
    df = pd.DataFrame([feature_list_with_score], columns=existing_column_names + ["Score"])
    return df

