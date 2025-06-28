#!/usr/bin/env python3
"""
4. Bayesian Optimization - Acquisition
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
        - f: callable, the black-box function to be optimized
        - X_init: numpy.ndarray of shape (t, 1), initial input samples
        - Y_init: numpy.ndarray of shape (t, 1), function values for X_init
        - bounds: tuple (min, max), search bounds
        - ac_samples: int, number of acquisition samples
        - l: float, length scale for kernel
        - sigma_f: float, output scale for kernel
        - xsi: float, exploration-exploitation factor
        - minimize: bool, True if minimizing the function, else maximizing
        """
        MIN, MAX = bounds
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(MIN, MAX, ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location using Expected Improvement

        Returns:
        - X_next: numpy.ndarray of shape (1,), the optimal next sample point
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
