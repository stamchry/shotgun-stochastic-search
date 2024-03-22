# Import necessary libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
# Import helper functions and regression modules from the local package
from .helpers import nbd, create_dataframe_with_score, model_selection_prior
from .linear_regression import get_score_linear
from .binary_regression import binary_regression

# Define a class for Stepwise Subset Selection (SSS)
class SSS:
    # Constructor to initialize the SSS object with parameters
    def __init__(self, iterations, hyperparameter, tau, regression_type, delta=None):
        self.iterations = iterations  # Number of iterations for the algorithm
        self.hyperparameter = hyperparameter  # Hyperparameter for model selection
        self.tau = tau  # Tau parameter for regression scoring
        self.regression_type = regression_type  # Type of regression ('linear' or 'binary')
        self.delta = delta if regression_type == 'linear' else None  # Delta parameter for linear regression
        self.G = None  # DataFrame to store selected models and their scores

    # Method to fit the SSS model to the data
    def fit(self, X: np.ndarray, y: np.ndarray, df: pd.DataFrame, num_of_best_scores: int):
        starting_model = [0] * X.shape[1]  # Initialize starting model with zeros
        starting_model[0] = 1  # Starting model will include only the first feature
        
        # Create a DataFrame to store models and their scores
        self.G = create_dataframe_with_score(starting_model, 0, df.iloc[:,:-1])

        # Define scoring function based on regression type
        if self.regression_type == 'linear':
            score_func = lambda model: self.get_linear_regression_score(X, y, model, df)
        elif self.regression_type == 'binary':
            score_func = lambda model: self.get_logistic_regression_score(X, y, model, df)
        else:
            raise ValueError("Invalid regression_type. Supported types are 'linear' and 'binary'.")

        # Iterate through the specified number of iterations
        for _ in tqdm(range(self.iterations)):

            # Generate neighboring models (gamma_plus, gamma_zero, gamma_minus)
            gamma_plus, gamma_zero, gamma_minus = nbd(starting_model)

            # Calculate scores for each neighboring model
            gamma_plus_df = pd.DataFrame(columns=self.G.columns[:-1], data=gamma_plus)
            gamma_plus_df['Score'] = gamma_plus_df.apply(score_func, axis=1)

            gamma_zero_df = pd.DataFrame(columns=self.G.columns[:-1], data=gamma_zero)
            gamma_zero_df['Score'] = gamma_zero_df.apply(score_func, axis=1)

            gamma_minus_df = pd.DataFrame(columns=self.G.columns[:-1], data=gamma_minus)
            gamma_minus_df['Score'] = gamma_minus_df.apply(score_func, axis=1)

            # Concatenate the dataframes of neighboring models
            to_concat = [
                x for x in [self.G, gamma_plus_df, gamma_minus_df, gamma_zero_df]
                if not x.empty
            ]
            self.G = pd.concat(
                to_concat,
                axis=0, ignore_index=True
            )

            # Sample models from neighboring models based on their scores
            samples_of_samples_df = pd.DataFrame(columns=self.G.columns)
            for df in [gamma_plus_df, gamma_zero_df, gamma_minus_df]:
                if not df.empty:
                    sample = df.sample(n=1, weights='Score')
                    if samples_of_samples_df.empty:
                        samples_of_samples_df = sample
                    else:
                        samples_of_samples_df = pd.concat(
                            [samples_of_samples_df, sample],
                            axis=0, ignore_index=True
                        )

            # Select the model with the highest score from the sampled models
            if not samples_of_samples_df.empty:
                model_for_next_iteration = samples_of_samples_df.sample(n=1, weights='Score')
                starting_model = model_for_next_iteration.iloc[0, :-1].tolist()
            else:
                starting_model = [0] * X.shape[1]

            # Select the top models based on the specified number of best scores
            self.G = self.G.sort_values(by='Score', ascending=False, ignore_index=True).head(num_of_best_scores)

        # Calculate relative importance of individual models conditioned on the set of top models
        self.G['Relative Importance'] = self.G['Score'] / self.G['Score'].sum()

        # Return the final DataFrame containing selected models and their scores
        return self.G
    
    # Method to calculate linear regression score for a model
    def get_linear_regression_score(self, X, y, model, df):
        return get_score_linear(X[:, model == 1], y, self.tau, self.delta) * model_selection_prior(self.hyperparameter, k=np.sum(model), p=df.shape[1]-1)

    # Method to calculate logistic regression score for a model
    def get_logistic_regression_score(self, X, y, model, df):
        return binary_regression(X[:, model ==1], y) * model_selection_prior(self.hyperparameter, k=np.sum(model), p=df.shape[1]-1)
