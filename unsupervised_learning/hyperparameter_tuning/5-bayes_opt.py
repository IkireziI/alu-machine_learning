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
        min_, max_ = bounds
        X_s = np.linspace(min_, max_, ac_samples)
        self.X_s = np.sort(X_s).reshape(-1, 1)
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

        if self.minimize:
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

        return best_sample, ei

    def optimize(self, iterations=100):
        """
        Optimize method
        """
        X_all_s = []
        for _ in range(iterations):
            x_opt, _ = self.acquisition()

            # Proper duplicate check for numpy arrays in list
            if any(np.allclose(x_opt, x_prev, atol=1e-8) for x_prev in X_all_s):
                break

            y_opt = self.f(x_opt)

            self.gp.update(x_opt, y_opt)
            X_all_s.append(x_opt)

        if self.minimize:
            index = np.argmin(self.gp.Y)
        else:
            index = np.argmax(self.gp.Y)

        # Commenting this out because removing last point arbitrarily can cause issues:
        # self.gp.X = self.gp.X[:-1]

        x_opt = self.gp.X[index]
        y_opt = self.gp.Y[index]

        return x_opt, y_opt

# End of file with a newline

