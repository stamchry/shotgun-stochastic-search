"""
Shotgun Stochastic Search algorithm class definition.

This module provides the ShotgunStochasticSearch class structure to perform the Shotgun Stochastic
Search, which rapidly explores 'Large p' Regression model space to find likely models by iteratively evaluating neighborhoods. Originally described by Chris Hans, Adrian Dobra and Mike West.
"""

# Import necessary libraries
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

# Import helper functions and regression modules from the local package
from .helpers import nbd, model_selection_prior
from .linear_regression import get_score_linear
from .binary_regression import binary_regression

# Define a class for Shotgun Stochastic Search
class ShotgunStochasticSearch:
    # Constructor to initialize the ShotgunStochasticSearch object with parameters
    def __init__(self, iterations, hyperparameter, tau, regression_type, delta=None):
        self.iterations = iterations  # Number of iterations for the algorithm
        self.hyperparameter = hyperparameter  # Hyperparameter for model selection
        self.tau = tau  # Tau parameter for regression scoring
        self.regression_type = regression_type  # Type of regression ('linear' or 'binary')
        self.delta = delta if regression_type == 'linear' else None  # Delta parameter for linear regression
        self.cache = {}  # Cache mapping model boolean tuples to scores
        self.G = None

    # Method to fit the ShotgunStochasticSearch model to the data
    def fit(self, X: np.ndarray, y: np.ndarray, df: pd.DataFrame, num_of_best_scores: int):
        starting_model = (1,) + (0,) * (X.shape[1] - 1)
        
        # Calculate isolated score
        def compute_score_pure(model_tuple):
            model_arr = np.array(model_tuple)
            if self.regression_type == 'linear':
                score = self.get_linear_regression_score(X, y, model_arr, df)
            elif self.regression_type == 'binary':
                score = self.get_logistic_regression_score(X, y, model_arr, df)
            else:
                raise ValueError("Invalid regression_type.")
            return score

        # Evaluate and cache neighborhood
        def evaluate_neighbors(neighbors):
            unevaluated = [n for n in neighbors if n not in self.cache]
            if unevaluated:
                scores = Parallel(n_jobs=-1)(delayed(compute_score_pure)(n) for n in unevaluated)
                for n, s in zip(unevaluated, scores):
                    # Replace NaN / None with zeros to safeguard probabilities
                    self.cache[n] = s if (s is not None and not np.isnan(s)) else 0.0
            
            return [(n, self.cache[n]) for n in neighbors if self.cache[n] > 0]

        # Seed initial model into cache
        self.cache[starting_model] = compute_score_pure(starting_model)

        # Iterate through the specified number of iterations
        for _ in tqdm(range(self.iterations)):

            # Generate neighboring models (gamma_plus, gamma_zero, gamma_minus)
            gamma_plus, gamma_zero, gamma_minus = nbd(starting_model)

            # Evaluate neighborhoods cleanly utilizing multiprocessing where available
            plus_scores = evaluate_neighbors(gamma_plus)
            zero_scores = evaluate_neighbors(gamma_zero)
            minus_scores = evaluate_neighbors(gamma_minus)

            # Sample models from neighboring models based on their scores
            samples = []
            for nbr_scores in [plus_scores, zero_scores, minus_scores]:
                if nbr_scores:
                    models = [x[0] for x in nbr_scores]
                    scores = [x[1] for x in nbr_scores]
                    total_score = sum(scores)
                    if total_score > 0:
                        probs = [s / total_score for s in scores]
                        chosen = random.choices(models, weights=probs, k=1)[0]
                        samples.append(chosen)

            # Step rule: Select the master model with the highest relative sampled score
            if samples:
                sample_scores = [self.cache[s] for s in samples]
                total_sample_score = sum(sample_scores)
                if total_sample_score > 0:
                    sample_probs = [s / total_sample_score for s in sample_scores]
                    starting_model = random.choices(samples, weights=sample_probs, k=1)[0]
                else:
                    starting_model = (0,) * X.shape[1]
            else:
                starting_model = (0,) * X.shape[1]

        # Algorithm finished. Extract top `num_of_best_scores` sequences from the cache
        sorted_cache = sorted(self.cache.items(), key=lambda x: x[1], reverse=True)[:num_of_best_scores]
        
        # Build cleanly formatted final DataFrame
        final_list = []
        for model_tuple, score in sorted_cache:
            row = list(model_tuple) + [score]
            final_list.append(row)
            
        columns = list(df.columns[:-1]) + ['Score']
        self.G = pd.DataFrame(final_list, columns=columns)
        
        # Calculate relative importance
        total_top_score = self.G['Score'].sum()
        if total_top_score > 0:
            self.G['Relative Importance'] = self.G['Score'] / total_top_score
        else:
            self.G['Relative Importance'] = 0.0

        return self.G
    
    # Method to calculate linear regression score for a model
    def get_linear_regression_score(self, X, y, model, df):
        return get_score_linear(X[:, model == 1], y, self.tau, self.delta) * model_selection_prior(self.hyperparameter, k=np.sum(model), p=df.shape[1]-1)

    # Method to calculate logistic regression score for a model
    def get_logistic_regression_score(self, X, y, model, df):
        return binary_regression(X[:, model ==1], y) * model_selection_prior(self.hyperparameter, k=np.sum(model), p=df.shape[1]-1)
