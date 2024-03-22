#main/linear_regression.py

import numpy as np
from .helpers import model_selection_prior, gamma_func

def m_gamma(X: np.ndarray, t: int, k: int):
    return t * np.identity(k) + np.matmul(X.T, X)

def q_gamma(y: np.ndarray, X: np.ndarray, t: int, k: int):
    return np.dot(y, y) - np.linalg.multi_dot([y.T, X, np.linalg.inv(m_gamma(X, t, k)), X.T, y])

def marginal_likelihood(X_gamma: np.ndarray, y: np.ndarray, tau: int, delta: int):
    (n, k) = np.shape(X_gamma)
    M = m_gamma(X_gamma, t=tau, k=k)
    q = q_gamma(y=y, X=X_gamma, t=tau, k=k)
    return (gamma_func((n + delta + k) / 2)) / (((np.pi) ** (n / 2)) * 
                                                (tau ** ((n - k) / 2)) * (np.sqrt(np.linalg.det(M))) * (np.sqrt(1 + (q / tau)) ** ((n + delta + k))) * gamma_func((delta + k) / 2))

def get_score_linear(X_gamma: np.ndarray, y: np.ndarray, tau, delta):
    return marginal_likelihood(X_gamma, y, tau, delta)
