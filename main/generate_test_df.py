#main/generate_test_df_linear_regression.py
from sklearn.datasets import make_classification,make_regression
import numpy as np
import pandas as pd
import random



def generate_test_df_linear_regression(num_of_samples, num_of_features, num_of_informative):
    np.random.seed(42)
    X,y, coeff = make_regression(n_samples=num_of_samples,
                          n_features=num_of_features,
                          n_informative=num_of_informative,
                          shuffle=False,coef=True)
    

    # Create column names indicating whether the feature is informative or redundant
    informative_feature_names = [f'informative_feature_{i+1}' for i in range(num_of_informative)]
    random_feature_names = [f'random_{i+1}' for i in range(num_of_features - num_of_informative)]
    column_names = informative_feature_names + random_feature_names

    # Create a DataFrame from X and y
    df = pd.DataFrame(X, columns=column_names)
    df['target'] = y
    
    return df,X,y, coeff


def generate_test_df_binary_regression(num_of_samples, num_of_features, informative_features, redundant_features):
    np.random.seed(40)
    X, y = make_classification(n_samples=num_of_samples,
                                n_features=num_of_features,
                                n_informative=informative_features,
                                n_redundant=redundant_features,
                                shuffle=False)

    # Create column names indicating whether the feature is informative or redundant
    informative_feature_names = [f'informative_feature_{i+1}' for i in range(informative_features)]
    redundant_feature_names = [f'redundant_feature_{i+1}' for i in range(redundant_features)]
    random_feature_names = [f'random_{i+1}' for i in range(num_of_features - informative_features - redundant_features)]
    column_names = informative_feature_names + redundant_feature_names + random_feature_names

    # Create a DataFrame from X and y
    df = pd.DataFrame(X, columns=column_names)
    df['target'] = y
    
    return df
