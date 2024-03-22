import numpy as np
from scipy.optimize import minimize

def binary_regression(X,y,tau=1):
    # Vectorized likelihood calculation
    def likelihood(beta):
        p_i = 1 / (1 + np.exp(-(beta[0] + np.dot(X, beta[1:]))))
        return np.prod((p_i**y)*((1-p_i)**(1-y)))

    # Vectorized prior calculation
    def prior(beta):
        return np.exp(-0.5 * np.sum((beta / tau)**2))

    # Vectorized posterior calculation
    def posterior(beta):
        return likelihood(beta) * prior(beta)

    def log_neg_posterior(beta):
        return -np.log(posterior(beta))

    # Initialize beta with zeros
    beta_init = np.zeros(X.shape[1] + 1)

    # Set the desired gtol value
    gtol = 1e-1  # For example, you can set it to 1e-6

    # Use BFGS method for optimization
    result = minimize(log_neg_posterior, beta_init, method='BFGS', options={'gtol': gtol})


    if result.success:
        hess = result.hess_inv
        laplace_approximation = ((2 * np.pi) ** (X.shape[1] / 2)) * np.sqrt(np.linalg.det(hess)) * np.exp(-result.fun)
        return laplace_approximation
    else:
        raise ValueError("Optimization failed.")
