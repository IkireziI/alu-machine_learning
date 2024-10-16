#!/usr/bin/env python3
"""
This module defines a function to compute the posterior probability
using Bayesian inference with continuous prior probabilities.
"""

from scipy import special


def posterior(x, n, p1, p2):
    """
    Calculate the posterior probability that the true probability of success
    lies within a specific range [p1, p2] given data.

    Args:
        x (int): Number of successes.
        n (int): Total number of trials.
        p1 (float): Lower bound of the probability range.
        p2 (float): Upper bound of the probability range.

    Returns:
        float: Posterior probability.

    Raises:
        ValueError: If any of the input criteria are not met.
    """
    # Input checks
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )
    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(p1, float):
        raise ValueError("p1 must be a float in the range [0, 1]")
    if not isinstance(p2, float):
        raise ValueError("p2 must be a float in the range [0, 1]")

    if not (0 <= p1 <= 1):
        raise ValueError("p1 must be a float in the range [0, 1]")
    if not (0 <= p2 <= 1):
        raise ValueError("p2 must be a float in the range [0, 1]")

    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    # Beta function to calculate the cumulative distribution
    def beta_cdf(p, a, b):
        """Helper function to calculate the beta CDF."""
        return special.betainc(a, b, p)

    # Parameters for the beta distribution
    a = x + 1
    b = n - x + 1

    # Calculate the posterior probability using the Beta CDF
    posterior_p1 = beta_cdf(p1, a, b)
    posterior_p2 = beta_cdf(p2, a, b)

    # Return the probability that the true p lies within [p1, p2]
    return posterior_p2 - posterior_p1

