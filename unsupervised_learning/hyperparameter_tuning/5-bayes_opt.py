#!/usr/bin/env python3
"""
5. Bayesian Optimization
"""

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Performs Bayesian optimization on a noiseless 1D Gaussian process
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Class constructor

        Parameters:
        - f: black-box function to be optimized
        - X_init: np.ndarray of shape (t, 1), initial input points
        - Y_init: np.ndarray of shape (t, 1), output values for X_init
        - bounds: tuple (min, max), bounds for search space
        - ac_samples: int, number of samples to check in acquisition
        - l: kernel length parameter
        - sigma_f: kernel standard deviation
        - xsi: float, exploration-exploitation factor
        - minimize: bool, True to minimize, False to maximize
        """
        MIN, MAX = bounds
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(MIN, MAX, num=ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location using Expected Improvement

        Returns:
        - X_next: numpy.ndarray of shape (1,), next best sample
        - EI: numpy.ndarray of shape (ac_samples,), expected improvement values
        """
        mu_sample, _ = self.gp.predict(self.gp.X)
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            mu_opt = np.min(mu_sample)
            imp = mu_opt - mu - self.xsi
        else:
            mu_opt = np.max(mu_sample)
            imp = mu - mu_opt - self.xsi

        Z = imp / sigma
        EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        EI[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function

        Parameters:
        - iterations: int, number of iterations to run

        Returns:
        - X_opt: numpy.ndarray of shape (1,), optimal input point
        - Y_opt: numpy.ndarray of shape (1,), optimal function value
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()

            # Fix: correct way to check if a row already exists in self.gp.X
            if any(np.allclose(X_next, x) for x in self.gp.X):
                break

            Y = self.f(X_next)
            self.gp.update(X_next, Y)

        idx = np.argmin(self.gp.Y) if self.minimize else np.argmax(self.gp.Y)
        X_opt = self.gp.X[idx]
        Y_opt = self.gp.Y[idx]
        return X_opt, Y_opt#!/usr/bin/env python3
"""
5. Bayesian Optimization
"""

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Performs Bayesian optimization on a noiseless 1D Gaussian process
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Class constructor

        Parameters:
        - f: black-box function to be optimized
        - X_init: np.ndarray of shape (t, 1), initial input points
        - Y_init: np.ndarray of shape (t, 1), output values for X_init
        - bounds: tuple (min, max), bounds for search space
        - ac_samples: int, number of samples to check in acquisition
        - l: kernel length parameter
        - sigma_f: kernel standard deviation
        - xsi: float, exploration-exploitation factor
        - minimize: bool, True to minimize, False to maximize
        """
        MIN, MAX = bounds
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(MIN, MAX, num=ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location using Expected Improvement

        Returns:
        - X_next: numpy.ndarray of shape (1,), next best sample
        - EI: numpy.ndarray of shape (ac_samples,), expected improvement values
        """
        mu_sample, _ = self.gp.predict(self.gp.X)
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            mu_opt = np.min(mu_sample)
            imp = mu_opt - mu - self.xsi
        else:
            mu_opt = np.max(mu_sample)
            imp = mu - mu_opt - self.xsi

        Z = imp / sigma
        EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        EI[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function

        Parameters:
        - iterations: int, number of iterations to run

        Returns:
        - X_opt: numpy.ndarray of shape (1,), optimal input point
        - Y_opt: numpy.ndarray of shape (1,), optimal function value
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()

            # Prevent duplicate sampling by checking with tolerance
            if np.any(np.all(np.isclose(self.gp.X, X_next, atol=1e-8), axis=1)):
                break

            Y = self.f(X_next)
            self.gp.update(X_next, Y)

        idx = np.argmin(self.gp.Y) if self.minimize else np.argmax(self.gp.Y)
        X_opt = self.gp.X[idx]
        Y_opt = self.gp.Y[idx]
        return X_opt, Y_opt
