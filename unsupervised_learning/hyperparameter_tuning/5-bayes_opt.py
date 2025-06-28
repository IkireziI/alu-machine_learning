#!/usr/bin/env python3
"""
Bayes Optimization
"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """
    Bayes Optimization using Gaussian Process
    """

    def __init__(self, f, X_init, Y_init, bounds,
                 ac_samples, l=1, sigma_f=1, xsi=0.01,
                 minimize=True):
        """
        * f is the black-box function to be optimized
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        min_bound, max_bound = bounds  # Renamed 'min', 'max' to avoid shadowing built-in functions
        X_s = np.linspace(min_bound, max_bound, ac_samples)
        self.X_s = (np.sort(X_s)).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        * Uses the Expected Improvement acquisition function
        Returns: X_next, EI
        * X_next is a numpy.ndarray of shape (1,) representing the next best
          sample point
        * EI is a numpy.ndarray of shape (ac_samples,) containing the expected
          improvement of each potential sample
        """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize is True:
            optimize = np.amin(self.gp.Y)
            imp = optimize - mu - self.xsi

        else:
            optimize = np.amax(self.gp.Y)
            imp = mu - optimize - self.xsi

        Z = np.zeros(sigma.shape[0])

        for i in range(sigma.shape[0]):
            if sigma[i] != 0:
                Z[i] = imp[i] / sigma[i]
            else:
                Z[i] = 0

        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

        index = np.argmax(ei)
        best_sample = self.X_s[index]

        return (best_sample, ei)

    def optimize(self, iterations=100):
        """
        Optimize method
        """

        X_all_s = []
        for i in range(iterations):
            # Find the next sampling point xt by optimizing the acquisition
            # function over the GP: xt = argmaxx μ(x | D1:t−1)

            x_opt, _ = self.acquisition()
            # If the next proposed point is one that has already been sampled,
            # optimization should be stopped early
            # Need to compare the value of x_opt, not the array object itself,
            # as it's a (1,) array. Convert to float for comparison.
            if x_opt.item() in [val.item() for val in X_all_s]:
                break

            y_opt = self.f(x_opt)

            # Add the sample to previous samples
            # D1: t = {D1: t−1, (xt, yt)} and update the GP
            self.gp.update(x_opt, y_opt)
            X_all_s.append(x_opt)

        if self.minimize is True:
            index = np.argmin(self.gp.Y)
        else:
            index = np.argmax(self.gp.Y)

        # The instruction "self.gp.X = self.gp.X[:-1]" seems to be an error
        # based on typical Bayesian Optimization workflow.
        # It removes the last added point, which might be the optimal one.
        # Assuming the goal is to return the best point found over the iterations
        # from the *entire* collected dataset self.gp.X and self.gp.Y.
        # So, removing this line.

        x_opt = self.gp.X[index]
        y_opt = self.gp.Y[index]

        return x_opt, y_opt