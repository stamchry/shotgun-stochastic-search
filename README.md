# Shotgun Stochastic Search for “Large p” Regression

This repository contains a Python package implementation of the algorithm proposed by Chris Hans, Adrian Dobra and Mike West in their paper *Shotgun Stochastic Search for "Large p" Regression* ([DOI: 10.1198/016214507000000121](https://doi.org/10.1198/016214507000000121)).

## Overview

Model search in regression with very large numbers of candidate predictors raises challenges for both model specification and computation. Standard approaches like MCMC and step-wise methods are often infeasible in these high-dimensional spaces. 

**Shotgun Stochastic Search** is an approach that rapidly explores "interesting" regions of high-dimensional model spaces to identify areas of high posterior probability. It evaluates massive neighborhoods of models iteratively, computing marginal likelihoods and posteriors to score variables and find competitive models.

This package provides implementations for:
- **Linear Regression**
- **Binary / Logistic Regression**

## Installation

The project is structured as a standard Python package. To install it locally in development mode:

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install the package
pip install -e .
```

## Basic Usage

Here is a quick example of running the `ShotgunStochasticSearch` class:

```python
import numpy as np
import pandas as pd
from shotgun_stochastic_search import ShotgunStochasticSearch

# Load your features (X), target (y), and full dataframe (df) here
# X, y, df = ...

# Initialize the Shotgun Stochastic Search
search = ShotgunStochasticSearch(
    iterations=1000, 
    hyperparameter=0.1, 
    tau=1, 
    delta=3, 
    regression_type='linear' # Use 'binary' for logistic regression
)

# Fit the ShotgunStochasticSearch model returning a DataFrame with the highest-scored models traversed
result = search.fit(X, y, df, num_of_best_scores=100)

# You can then aggregate and extract the relative importance of variables!
```

**Check out `examples/`** for advanced and full workable scripts spanning simulated classification, linear regression, and testing on the real-world Tecator dataset.

## Project Structure

- `shotgun_stochastic_search/`: The core Python package containing the mathematical modules.
- `examples/`: Example scripts demonstrating linear and binary models.
- `tests/`: Utilities used for data generation modeling.
- `data/`: Local storage for dataset CSVs (`tecator_data.csv`).
